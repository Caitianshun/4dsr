"""Copy (Claude 4dsr-e9, 2026-10-18) of experiments/dynamic_sr_temporal_20261012/temporal_diag.py (sha256 5a9a15e72f489105...)
placed under the geoprobe directory so that it can be uploaded to the remote hosts through the registered sync prefix. Changes:
ROOT is resolved three levels up (the file sits one directory deeper); the static-std PNGs are written with Pillow when imageio
is missing. Everything else is byte-identical. Used for the pre-registered temporal check of the unified spatial correction
(prereg_unified_correction.txt: "for the chosen default vs the other a cam00 temporal-difference check is run once").

Original docstring:
Zero-training temporal diagnostic on the held-out camera (2026-10-12).

Renders cam00 for all 300 test frames from several existing checkpoints of ONE scene/seed in a single pass
(shared HR reads and shared GT optical flow), and measures temporal behaviour against the HR video.
No parameter is updated and no training image is read. Model loading, post-hoc culling and the render path
are imported unchanged from the 2026-10-09 cull-retrain entry, so per-frame PSNR must equal the frozen
evaluation of the same checkpoint (checked on the first frames; the run aborts on a mismatch).

Per arm (R = render, G = HR frame, dX_t = X_t - X_{t-1}):
  psnr        per-frame PSNR (sanity).
  static      on S = pixels never detected as moving in the HR video (sequence level, eroded, border removed):
              E_t[(R-G)^2] = (Rbar-Gbar)^2 + Var_t R + Var_t G - 2 Cov_t(R,G)
              bias2 = time-constant error; flicker = Var_t R (render temporal variance; ideal 0 on S);
              gt_var = reference sensor/compression noise; also flicker split into low (Gaussian sigma=2 px)
              and high (residual) bands, and mean |dR - dG| on S.
  dynamic     on per-frame motion masks D_t (HR flow > 0.75 px with a photometric change, or a large HR frame
              difference; dilated 8 px):
              tdiff = mean |dR - dG|; warp = mean |R_t - W(R_{t-1})| on D_t and flow-consistent pixels, with
              W the backward warp by the HR flow (DIS, forward-backward check); warp_G is the HR floor;
              tOF = mean ||F(R_{t-1},R_t) - F(G_{t-1},G_t)|| (TecoGAN tOF with DIS flow), on D_t and on all pixels.
  tLP         |LPIPS(R_t,R_{t-1}) - LPIPS(G_t,G_{t-1})| (TecoGAN tLP; AlexNet LPIPS 0.1), whole frame.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import json
import math
import socket
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CULL = ROOT / 'experiments/dynamic_sr_cull_retrain_20261009'
sys.path.insert(0, str(CULL))
import rt  # noqa: E402  (sets sys.path to the pinned runtime)
import cullretrain as cr  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

SCHEMA = 'temporal_diag_v1'


def gaussian_kernel(sigma, device):
    r = int(math.ceil(3 * sigma))
    x = torch.arange(-r, r + 1, dtype=torch.float32, device=device)
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def blur(x, k):
    """Separable Gaussian blur of a (3,H,W) tensor with reflect padding."""
    r = (k.numel() - 1) // 2
    y = F.pad(x[None], (r, r, 0, 0), mode='reflect')
    y = F.conv2d(y, k.view(1, 1, 1, -1).repeat(3, 1, 1, 1), groups=3)
    y = F.pad(y, (0, 0, r, r), mode='reflect')
    return F.conv2d(y, k.view(1, 1, -1, 1).repeat(3, 1, 1, 1), groups=3)[0]


def dilate(mask, r):
    return F.max_pool2d(mask[None, None].float(), 2 * r + 1, stride=1, padding=r)[0, 0] > 0.5


def box(x, r):
    return F.avg_pool2d(x[None, None], 2 * r + 1, stride=1, padding=r, count_include_pad=False)[0, 0]


_local = threading.local()


def dis_flow(a, b):
    """DIS flow from uint8 image a to b: a(p) ~ b(p + F(p)). One DIS object per thread."""
    import cv2
    if not hasattr(_local, 'dis'):
        _local.dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    return _local.dis.calc(a, b, None)


def to_gray_u8(x, scale):
    """(3,H,W) in [0,1] -> HxW uint8 luma at the flow scale (area average)."""
    y = (0.299 * x[0] + 0.587 * x[1] + 0.114 * x[2])[None, None]
    if scale != 1.0:
        y = F.interpolate(y, scale_factor=scale, mode='area')
    return (y[0, 0].clamp(0, 1) * 255 + 0.5).to(torch.uint8).cpu().numpy()


def flow_full(flow, shape, scale, device):
    """Flow at flow scale (h,w,2) -> full-resolution (2,H,W) tensor in full-resolution pixels."""
    f = torch.from_numpy(flow).to(device).permute(2, 0, 1)[None]
    if scale != 1.0:
        f = F.interpolate(f, size=shape, mode='bilinear', align_corners=False) / scale
    return f[0]


class Warper:
    def __init__(self, H, W, device):
        ys, xs = torch.meshgrid(torch.arange(H, device=device, dtype=torch.float32),
                                torch.arange(W, device=device, dtype=torch.float32), indexing='ij')
        self.base = torch.stack((xs, ys))
        self.H, self.W = H, W

    def __call__(self, img, flow):
        """Sample img (C,H,W) at p + flow(p)."""
        p = self.base + flow
        grid = torch.stack((2 * p[0] / (self.W - 1) - 1, 2 * p[1] / (self.H - 1) - 1), -1)[None]
        return F.grid_sample(img[None], grid, mode='bilinear', padding_mode='border', align_corners=True)[0]


def load_arm(arm):
    ckpt = ROOT / arm['checkpoint']
    plan = torch.load(ckpt, map_location='cpu', weights_only=False)['plan']
    g, payload = cr.load_for_render(ckpt, plan)
    del payload
    h, opt, pipe, sh = cr.configuration(plan)
    manifest_path, manifest = cr.plan_scene(plan)
    record = dict(name=arm['name'], checkpoint=cr.entry(ckpt), manifest=cr.entry(manifest_path),
                  points_loaded=int(g.get_xyz.shape[0]))
    if arm.get('cull'):
        record['cull'] = cr.cull(g, cr.Culler(manifest), None, 'posthoc', optimizer_backed=False)
    record['points_evaluated'] = int(g.get_xyz.shape[0])
    background = (torch.ones if cr.white_background(plan) else torch.zeros)(3, dtype=torch.float32, device='cuda')
    test = sorted([o for o in manifest['observations'] if o['split'] == 'test'], key=lambda o: int(o['frame_index']))
    if [int(o['frame_index']) for o in test] != list(range(300)) or {o['camera_id'] for o in test} != {'cam00'}:
        raise ValueError('cam00 all-300 test split required: ' + arm['name'])
    return dict(arm=arm, g=g, pipe=pipe, background=background, manifest=manifest, test=test, record=record)


def render(m, i):
    obs = m['test'][i]
    with torch.no_grad():
        cam = cr.native.hr_camera(cr.ev.camera_item(m['manifest'], obs, torch), i, m['manifest'])
        state = cr.native.native_effective_state(m['g'], float(obs['time']), torch)
        raw = cr.native.native_rgb(cam, m['g'], state, m['pipe'], m['background'])['render'].detach()
    return raw.clamp(0, 1)


def reference_psnr(path, frames):
    out = {}
    with open(path, newline='') as f:
        for r in csv.DictReader(f):
            if r['scope'] == 'test' and int(r['frame']) in frames:
                out[int(r['frame'])] = float(r['psnr'])
    return out


def run(spec, out, frames=None, threads=6):
    import lpips
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'summary.json').exists():
        raise FileExistsError('Completed diagnostic preserved: ' + str(out))
    started = time.time()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    runtime = rt.check_runtime()
    models = [load_arm(a) for a in spec['arms']]
    ref = models[0]
    root = Path(ref['manifest']['_root'])
    hr_paths = [o['hr_path'] for o in ref['test']]
    for m in models[1:]:
        if Path(m['manifest']['_root']) != root or [o['hr_path'] for o in m['test']] != hr_paths:
            raise ValueError('cam00 HR sequence differs between arms: ' + m['arm']['name'])
        if [float(o['time']) for o in m['test']] != [float(o['time']) for o in ref['test']]:
            raise ValueError('cam00 times differ between arms: ' + m['arm']['name'])
    legacy, _, _, _ = cr.ev.metric_helpers()
    metric = lpips.LPIPS(net='alex', version='0.1', spatial=False, pnet_rand=False, pretrained=True,
                         verbose=False).cuda().eval().requires_grad_(False)
    lp = lambda a, b: float(metric(a[None], b[None], normalize=True).item())
    W_hr, H_hr = ref['manifest']['resolutions']['hr']
    scale = float(spec.get('flow_scale', 1.0))
    dev = 'cuda'
    k = gaussian_kernel(2.0, dev)
    warp = Warper(H_hr, W_hr, dev)
    idx = list(range(300)) if frames is None else list(frames)
    if idx != list(range(idx[0], idx[0] + len(idx))):
        raise ValueError('frames must be consecutive')
    pool = cf.ThreadPoolExecutor(threads)
    io = pool.submit(legacy.read_rgb, root / hr_paths[idx[0]])
    z = lambda: torch.zeros((3, H_hr, W_hr), dtype=torch.float64, device=dev)
    acc = {m['arm']['name']: dict(s1=z(), s2=z(), sx=z(), l1=z(), l2=z(), h1=z(), h2=z(),
                                  td=torch.zeros((H_hr, W_hr), dtype=torch.float64, device=dev)) for m in models}
    gacc = dict(s1=z(), s2=z(), l1=z(), l2=z(), h1=z(), h2=z())
    ever = torch.zeros((H_hr, W_hr), dtype=torch.bool, device=dev)
    rows = {m['arm']['name']: [] for m in models}
    prev = {}
    prev_g = prev_gg = None
    checks = {}
    for n, i in enumerate(idx):
        gt = torch.from_numpy(io.result()).to(dev).permute(2, 0, 1).contiguous()
        if n + 1 < len(idx):
            io = pool.submit(legacy.read_rgb, root / hr_paths[idx[n + 1]])
        gt64 = gt.double()
        gl = blur(gt, k)
        gh = gt - gl
        for key, v in (('s1', gt64), ('s2', gt64 * gt64), ('l1', gl.double()), ('l2', gl.double() ** 2),
                       ('h1', gh.double()), ('h2', gh.double() ** 2)):
            gacc[key] += v
        gg = to_gray_u8(gt, scale)
        renders = {m['arm']['name']: render(m, i) for m in models}
        grays = {name: to_gray_u8(r, scale) for name, r in renders.items()}
        if n > 0:
            fwd_job = pool.submit(dis_flow, prev_gg, gg)
            bwd_job = pool.submit(dis_flow, gg, prev_gg)
            r_jobs = {name: pool.submit(dis_flow, prev[name][1], grays[name]) for name in renders}
            f_fwd = flow_full(fwd_job.result(), (H_hr, W_hr), scale, dev)   # frame t-1 coordinates
            f_bwd = flow_full(bwd_job.result(), (H_hr, W_hr), scale, dev)   # frame t coordinates
            f_fwd_at = warp(f_fwd, f_bwd)
            s = (f_bwd + f_fwd_at).pow(2).sum(0)
            valid = s < 0.01 * (f_bwd.pow(2).sum(0) + f_fwd_at.pow(2).sum(0)) + 0.5
            dg = gt - prev_g
            # flow alone flags textureless noisy walls (spurious DIS flow); require a photometric change as well
            change = box(dg.abs().mean(0), 4)
            motion_t = ((f_bwd.pow(2).sum(0).sqrt() > 0.75) & (change > 0.015)) | (change > 0.04)
            motion_prev = ((f_fwd.pow(2).sum(0).sqrt() > 0.75) & (change > 0.015)) | (change > 0.04)
            D = dilate(motion_t, 8)
            Dp = dilate(motion_prev, 8)
            ever |= D | Dp
            Dv = D & valid
            g_warp = (gt - warp(prev_g, f_bwd)).abs().mean(0)
            lp_g = lp(gt, prev_g)
        for m in models:
            name = m['arm']['name']
            r = renders[name]
            a = acc[name]
            r64 = r.double()
            rl = blur(r, k)
            rh = r - rl
            a['s1'] += r64
            a['s2'] += r64 * r64
            a['sx'] += r64 * gt64
            a['l1'] += rl.double()
            a['l2'] += rl.double() ** 2
            a['h1'] += rh.double()
            a['h2'] += rh.double() ** 2
            mse = float((r - gt).pow(2).mean().item())
            row = dict(frame=i, psnr=-10 * math.log10(max(mse, 1e-12)), mse=mse)
            if n > 0:
                pr = prev[name][0]
                tdm = ((r - pr) - dg).abs().mean(0)
                a['td'] += tdm.double()
                f_r = flow_full(r_jobs[name].result(), (H_hr, W_hr), scale, dev)
                e_of = (f_r - f_fwd).pow(2).sum(0).sqrt()
                r_warp = (r - warp(pr, f_bwd)).abs().mean(0)
                lp_r = lp(r, pr)
                row.update(dyn_frac=float(D.float().mean()), tdiff_dyn=float(tdm[D].mean()) if D.any() else None,
                           tdiff_all=float(tdm.mean()),
                           warp_R=float(r_warp[Dv].mean()) if Dv.any() else None,
                           warp_G=float(g_warp[Dv].mean()) if Dv.any() else None,
                           tOF_dyn=float(e_of[Dp].mean()) if Dp.any() else None, tOF_all=float(e_of.mean()),
                           lpips_pair_R=lp_r, lpips_pair_G=lp_g, tLP=abs(lp_r - lp_g))
            rows[name].append(row)
            prev[name] = (r, grays[name])
        prev_g, prev_gg = gt, gg
        if n == 2:   # sanity against the frozen evaluation of the same checkpoint and cull
            for m in models:
                csv_path = m['arm'].get('reference_csv')
                if not csv_path:
                    continue
                refp = reference_psnr(ROOT / csv_path, set(idx[:3]))
                got = {r['frame']: r['psnr'] for r in rows[m['arm']['name']][:3]}
                diff = max(abs(got[f] - refp[f]) for f in refp)
                checks[m['arm']['name']] = dict(max_abs_psnr_diff_first3=diff)
                if diff > 2e-3:
                    raise RuntimeError(f"render differs from frozen eval for {m['arm']['name']}: {diff}")
            rt.write_json(out / 'sanity.json', checks)
    pool.shutdown()
    T = len(idx)
    border = int(spec.get('border', 16))
    S = ~dilate(ever, 4)
    S[:border] = False
    S[-border:] = False
    S[:, :border] = False
    S[:, -border:] = False
    Sm = S[None].expand(3, -1, -1)
    mean_s = lambda x: float(x[Sm].mean().item())
    gbar, gvar = gacc['s1'] / T, gacc['s2'] / T - (gacc['s1'] / T) ** 2
    gvl = gacc['l2'] / T - (gacc['l1'] / T) ** 2
    gvh = gacc['h2'] / T - (gacc['h1'] / T) ** 2
    summary = dict(schema=SCHEMA, scene=spec['scene'], host=socket.gethostname(), gpu=torch.cuda.get_device_name(),
                   runtime=runtime, sources={str(p.relative_to(ROOT)): rt.sha(p) for p in (Path(__file__), CULL / 'cullretrain.py', CULL / 'rt.py')},
                   frames=[idx[0], idx[-1]], flow_scale=scale, hr_size=[W_hr, H_hr], static_fraction=float(S.float().mean()),
                   ever_moving_fraction=float(ever.float().mean()), sanity=checks,
                   gt=dict(static_var=mean_s(gvar), static_var_low=mean_s(gvl), static_var_high=mean_s(gvh)), arms={})
    try:
        import imageio.v2 as imageio
    except ImportError:  # remote venvs may lack imageio; Pillow writes the same 8-bit PNGs
        from PIL import Image as _Image
        imageio = type('imageio', (), {'imwrite': staticmethod(lambda p, a: _Image.fromarray(a).save(p))})
    std_dir = out / 'static_std'
    std_dir.mkdir(exist_ok=True)
    gstd = gvar.clamp_min(0).sqrt().mean(0)
    for m in models:
        name = m['arm']['name']
        a = acc[name]
        rbar = a['s1'] / T
        rvar = a['s2'] / T - rbar ** 2
        cov = a['sx'] / T - rbar * gbar
        total = a['s2'] / T - 2 * a['sx'] / T + gacc['s2'] / T
        bias2 = (rbar - gbar) ** 2
        rvl = a['l2'] / T - (a['l1'] / T) ** 2
        rvh = a['h2'] / T - (a['h1'] / T) ** 2
        rs = [r for r in rows[name] if 'tLP' in r]
        avg = lambda key: (float(np.mean([r[key] for r in rs if r.get(key) is not None])) if rs else None)
        summary['arms'][name] = dict(
            record=m['record'], reference_csv=m['arm'].get('reference_csv'),
            psnr=float(np.mean([r['psnr'] for r in rows[name]])),
            static=dict(total_mse=mean_s(total), bias2=mean_s(bias2), flicker_var=mean_s(rvar), gt_var=mean_s(gvar),
                        cov=mean_s(cov), identity_gap=mean_s(total - (bias2 + rvar + gvar - 2 * cov)),
                        flicker_var_low=mean_s(rvl), flicker_var_high=mean_s(rvh),
                        flicker_rms_8bit=255 * math.sqrt(max(mean_s(rvar), 0)),
                        tdiff=float((a['td'] / max(T - 1, 1))[S].mean().item())),
            dynamic=dict(dyn_frac=avg('dyn_frac'), tdiff_dyn=avg('tdiff_dyn'), warp_R=avg('warp_R'), warp_G=avg('warp_G'),
                         tOF_dyn=avg('tOF_dyn')),
            whole=dict(tdiff_all=avg('tdiff_all'), tOF_all=avg('tOF_all'), tLP=avg('tLP'),
                       lpips_pair_R=avg('lpips_pair_R'), lpips_pair_G=avg('lpips_pair_G')))
        rstd = rvar.clamp_min(0).sqrt().mean(0)
        img = (torch.cat((gstd, rstd), 1) * 10 * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()
        imageio.imwrite(std_dir / f'{name}_static_std_x10_gt_left.png', img)
        with (out / f'per_frame_{name}.csv').open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=sorted({k2 for r in rows[name] for k2 in r}))
            w.writeheader()
            w.writerows(rows[name])
    imageio.imwrite(std_dir / 'static_mask.png', (S.to(torch.uint8) * 255).cpu().numpy())
    summary['seconds'] = time.time() - started
    summary['peak_GB'] = torch.cuda.max_memory_allocated() / 1e9
    rt.write_json(out / 'summary.json', summary)
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--spec', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--frames', type=int, nargs='*')
    p.add_argument('--threads', type=int, default=6)
    a = p.parse_args(argv)
    spec = json.loads(Path(a.spec).read_text())
    s = run(spec, Path(a.out), a.frames, a.threads)
    print(json.dumps({k: s[k] for k in ('scene', 'seconds', 'static_fraction', 'sanity')}, default=str))


if __name__ == '__main__':
    main()
