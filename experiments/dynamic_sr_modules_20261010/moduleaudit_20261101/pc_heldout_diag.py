"""Held-out-camera diagnostic of P+C v2 (Claude 4dsr-4e, 2026-11-01; question of 4dsr-4f on the cam10 LOCO read-outs, where the
whole-image LR PSNR of parent -> v2 drops by up to 1.2 dB in Cook07 / Coffee07 while the dynamic mask does not; design revised
with 4dsr-4f's review at 09:28 PST). Render only: no training, no HR image, no cam00. At the camera C that a LOCO fit excluded
(perframe_fit_loco.py --exclude-cameras C), in C's LR space and with loco_eval.py's observation O(f) = area(camera model of C
applied to the HR render), clamped after pooling.
Weights: for every free Gaussian i and fitted camera j, its alpha-blending weight summed over the fitter's dilated LR dynamic mask
of j (W_ij) and over the whole image of j (A_ij), in LR pixels: the gradient of the area-pooled raw render with respect to the
SH-DC with all SH set to 0 (colour 0.5, no clamp), divided by SH_C0. The fitter's loss is the whole-image L1, so A is what
constrains a residual in the fit; W is what an occlusion-aware version of the fitter's selection rule (projected centre in the
mask of >= 2 cameras) would count. Classes at tau: n_W = #{j: W_ij >= tau}, n_A = #{j: A_ij >= tau}.
Arms (the fit's residuals restricted to a subset of the free set):
  parent | full | P (dxyz only) | C (dsh0 only)
  vis<tau> = n_W >= 2 and hid<tau> = n_W <= 1, for every --taus
  W0 = n_W == 0, W1 = n_W == 1; A0 = n_A == 0, A1 = n_A == 1, A2 = n_A >= 2; hidP / hidC = hid<tau_mid> with dxyz / dsh0 only
Regions at C: M = C's mask by the fitter's rule (loco_eval's dyn mask); rings R2 = dil(M,2)\\M, R6 = dil(M,6)\\dil(M,2),
R16 = dil(M,16)\\dil(M,6) (elliptic, LR px); far = rest.
Writes OUT/diag.json: per arm and region pooled MSE / PSNR, change against the parent and each region's share of the whole-image
change; tdiff of parent / full as loco_eval (reproduction check against loco.json); per-frame whole-image MSE change (full, hid,
A0, A1) with first/last-100-frame means and slope; per class (W0, W1, W2, A0, A1, A2) per-frame medians of opacity, max scale,
max_j A, residual sizes, residual energy shares and C-static / C-dynamic weight; log10 histograms of max_j W and max_j A over the
free set; every 5th frame the cosine between a free Gaussian's position residual and its 8-NN mean (the fitter's kNN graph); a
unit check (sum_i A_ij over all Gaussians = alpha coverage of j, <= LR pixel count). OUT/maps/*.png: mean per-pixel squared-error
change against the parent (red = worse) of full, P, C, hid, vis, A0, A1 and the dynamic-mask frequency; the frame with the largest
change (GT, parent, full, change).
  pc_heldout_diag.py --checkpoint CK --fit LOCO_FIT_DIR --camera cam10 --out OUT [--frames 0-299] [--taus 0.01,0.05,0.2]"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import socket
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ABS = HERE.parent / 'absorb_20261021'
if str(ABS) not in sys.path:
    sys.path.insert(0, str(ABS))

import absorb_bound2 as ab  # noqa: E402
import perframe_fit as pf  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

cr, native, ev = ab.cr, ab.native, ab.ev
ROOT = ab.ROOT
SH_C0 = 0.28209479177387814
REGIONS = ('M', 'R2', 'R6', 'R16', 'far')
CLASSES = ('W0', 'W1', 'W2', 'A0', 'A1', 'A2')
CLASS_STATS = ('count', 'opacity', 'max_scale', 'maxA', 'mu_px', 'dsh0', 'mu2_share', 'dsh02_share', 'cam_static_weight',
               'cam_dyn_weight')
HIST_EDGES = np.linspace(-6.0, 4.0, 101)


def frames_arg(s):
    out = []
    for part in s.split(','):
        a, _, b = part.partition('-')
        out += list(range(int(a), int(b) + 1)) if b else [int(a)]
    return out


def psnr(se, n):
    return -10 * math.log10(max(se / max(n, 1), 1e-12))


def subset(res, sel=None, keep=('dxyz', 'dsh0')):
    r = dict(idx=res['idx'] if sel is None else res['idx'][sel])
    for k in ('dxyz', 'dsh0'):
        v = res.get(k)
        r[k] = None if (v is None or k not in keep) else (v if sel is None else v[sel])
    return r


def med(x):
    return float(x.median()) if x.numel() else None


def slope(y):
    y = np.asarray(y, dtype=np.float64)
    return float(np.polyfit(np.arange(y.size, dtype=np.float64), y, 1)[0]) if y.size >= 2 else None


def hist_quantiles(counts, qs=(0.05, 0.25, 0.5, 0.75, 0.95)):
    c = np.cumsum(counts)
    if not c.size or c[-1] == 0:
        return None
    return {f'{q:g}': float(HIST_EDGES[min(int(np.searchsorted(c, q * c[-1])) + 1, HIST_EDGES.size - 1)]) for q in qs}


def diverging(x, lim):
    """x (H, W) float, red = positive (worse), blue = negative, white = 0; |x| >= lim saturates."""
    t = np.clip(x / max(lim, 1e-12), -1, 1)
    rgb = np.ones(x.shape + (3,), dtype=np.float32)
    pos, neg = t > 0, t < 0
    rgb[pos, 1] -= t[pos]
    rgb[pos, 2] -= t[pos]
    rgb[neg, 0] += t[neg]
    rgb[neg, 1] += t[neg]
    return (rgb * 255).astype(np.uint8)


def to_u8(img):
    return (img.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255 + 0.5).astype(np.uint8)


def save_row(path, tiles, scale):
    h = max(t.shape[0] for t in tiles)
    pad = [np.pad(t if t.ndim == 3 else np.repeat(t[..., None], 3, 2), ((0, h - t.shape[0]), (0, 4), (0, 0)), constant_values=255)
           for t in tiles]
    im = Image.fromarray(np.concatenate(pad, 1))
    if scale > 1:
        im = im.resize((im.width * scale, im.height * scale), Image.NEAREST)
    im.save(path)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--fit', required=True)
    p.add_argument('--camera', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--frames', default='0-299')
    p.add_argument('--taus', default='0.01,0.05,0.2')
    a = p.parse_args()
    out = ROOT / a.out
    if (out / 'diag.json').exists():
        raise FileExistsError(f'completed diagnostic preserved: {out}')
    (out / 'maps').mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dev = 'cuda'
    started = time.time()
    ck = ROOT / a.checkpoint
    payload = torch.load(ck, map_location='cpu', weights_only=False)
    plan, raw_aff = payload['plan'], payload.get('affine')
    del payload
    manifest_path, manifest = cr.plan_scene(plan)
    root = Path(manifest['_root'])
    train = list(manifest['splits']['train'])
    c = a.camera
    if c not in train or c == 'cam00':
        raise SystemExit('--camera must be a training camera (never cam00)')
    model = ab.camera_model(raw_aff, dev)
    g = ab.load_model(ck, plan, dev)
    h, opt, pipe, sh = cr.configuration(plan)
    cull_record = cr.cull(g, cr.Culler(manifest, device=dev), None, 'posthoc', optimizer_backed=False)
    xyz_sha = hashlib.sha256(g._xyz.detach().float().cpu().numpy().tobytes()).hexdigest()
    ck_sha = cr.entry(ck)['sha256']
    fit = ROOT / a.fit
    run = json.loads((fit / 'run.json').read_text())
    if run.get('checkpoint_sha256') != ck_sha or int(run.get('points_after_cull', -1)) != int(g._xyz.shape[0]) \
            or run.get('post_cull_xyz_sha256') != xyz_sha:
        raise SystemExit('the fit does not belong to this checkpoint / cull')
    excluded = list(run.get('excluded_cameras') or [])
    if c not in excluded:
        raise SystemExit(f'the fit did not exclude {c} (excluded={excluded})')
    fitted = [cc for cc in train if cc not in excluded]
    if run.get('fitting') is not None and list(run['fitting']) != fitted:
        raise SystemExit('fitted camera list differs from run.json')
    lr_w, lr_h = manifest['resolutions']['lr']
    hr_w, hr_h = manifest['resolutions']['hr']
    s = hr_w // lr_w
    if hr_w != s * lr_w or hr_h != s * lr_h:
        raise SystemExit('HR is not an integer multiple of LR')
    obs = {(o['camera_id'], int(o['frame_index'])): o for o in manifest['observations']}
    index = {cc: i for i, cc in enumerate(train)}
    background = torch.ones(3, device=dev) if cr.white_background(plan) else torch.zeros(3, device=dev)
    names = list(raw_aff['names'])
    delta = raw_aff['delta'].double() - raw_aff['delta'].double().mean(0)

    def canon_fn(cc):
        Ainv = torch.linalg.inv(torch.eye(3, dtype=torch.float64) + delta[names.index(cc), :, :3]).float().to(dev)
        b = delta[names.index(cc), :, 3].float().to(dev)
        return lambda y: torch.einsum('ij,jhw->ihw', Ainv, y - b[:, None, None])

    cams_all = [c] + fitted
    canon = {cc: canon_fn(cc) for cc in cams_all}
    k3 = torch.ones(3, 3, device=dev)
    medbg = {cc: torch.stack([canon[cc](ab.read_rgb(root / obs[(cc, f)]['lr_path'], dev)) for f in range(0, 300, 5)]).median(0).values
             for cc in cams_all}
    rings = {r: ab.ellipse(r, dev) for r in (2, 6, 16)}
    taus = [float(x) for x in a.taus.split(',')]
    tau_mid = taus[len(taus) // 2]
    tm = f'{tau_mid:g}'
    arms = (['parent', 'full', 'P', 'C'] + [f'vis{t:g}' for t in taus] + [f'hid{t:g}' for t in taus]
            + ['W0', 'W1', 'A0', 'A1', 'A2', 'hidP', 'hidC'])
    acc = {arm: {r: [0.0, 0] for r in REGIONS + ('all',)} for arm in arms}
    tacc = {arm: dict(tse=0.0, tn=0, tse_d=0.0, tn_d=0) for arm in ('parent', 'full')}
    map_arms = ('full', 'P', 'C', f'hid{tm}', f'vis{tm}', 'A0', 'A1')
    mapsum = {arm: torch.zeros(lr_h, lr_w, device=dev, dtype=torch.float64) for arm in map_arms}
    dynfreq = torch.zeros(lr_h, lr_w, device=dev, dtype=torch.float64)
    hist = dict(maxW=np.zeros(HIST_EDGES.size - 1), maxA=np.zeros(HIST_EDGES.size - 1))
    per_frame, fstats, knn_rows = [], [], []
    unit_check = None
    prev = {}
    worst = dict(delta=-1.0, frame=None)
    frames = frames_arg(a.frames)
    for f in frames:
        o = obs[(c, f)]
        with torch.no_grad():
            S = {k: v.detach() for k, v in native.native_effective_state(g, float(o['time']), torch).items()}
        res = pf.load_residual(fit / f'f{f:04d}.npz', dev)
        idx = res['idx']
        n = int(idx.numel())
        with np.load(fit / f'f{f:04d}.npz') as z:
            px_world = float(z['px_world'])
        W = torch.zeros(len(fitted), n, device=dev)
        A = torch.zeros(len(fitted), n, device=dev)

        def weights(cc, mask_lr, check=False):
            cam_ = native.hr_camera(ev.camera_item(manifest, obs[(cc, f)], torch), index[cc], manifest)
            shz = torch.zeros_like(S['sh']).requires_grad_(True)
            img = native.native_rgb(cam_, g, dict(S, sh=shz), pipe, background)['render']
            up = mask_lr.float().repeat_interleave(s, 0).repeat_interleave(s, 1)
            L = (img[0] * up).sum() / s ** 2 + img[1].sum() / s ** 2
            gsh, = torch.autograd.grad(L, shz)
            w_in, w_all = gsh[idx, 0, 0] / SH_C0, gsh[idx, 0, 1] / SH_C0
            if not check:
                return w_in, w_all, None
            bg = float(background[1])
            alpha = (img[1].detach() - bg) / (0.5 - bg)
            return w_in, w_all, dict(sum_weight_all_gaussians_lrpx=float(gsh[:, 0, 1].sum() / SH_C0),
                                     alpha_coverage_lrpx=float(alpha.sum() / s ** 2), lr_pixels=lr_h * lr_w)

        for j, cc in enumerate(fitted):
            yc = ab.read_rgb(root / obs[(cc, f)]['lr_path'], dev)
            mc = ab.dilate((canon[cc](yc) - medbg[cc]).abs().amax(0) > 0.06, k3)
            W[j], A[j], uc = weights(cc, mc, check=unit_check is None)
            if uc is not None:
                unit_check = dict(frame=f, camera=cc, **uc)
        y = ab.read_rgb(root / o['lr_path'], dev)
        M = ab.dilate((canon[c](y) - medbg[c]).abs().amax(0) > 0.06, k3)
        wc_dyn, wc_all, _ = weights(c, M)
        wc_static = (wc_all - wc_dyn).clamp_min(0)
        nW = {t: (W >= t).sum(0) for t in taus}
        nA = (A >= tau_mid).sum(0)
        vis = {t: nW[t] >= 2 for t in taus}
        hid_m = ~vis[tau_mid]
        cls = dict(W0=nW[tau_mid] == 0, W1=nW[tau_mid] == 1, W2=nW[tau_mid] >= 2, A0=nA == 0, A1=nA == 1, A2=nA >= 2)
        if n:
            hist['maxW'] += np.histogram(np.log10(W.max(0).values.clamp_min(1e-6).cpu().numpy()), HIST_EDGES)[0]
            hist['maxA'] += np.histogram(np.log10(A.max(0).values.clamp_min(1e-6).cpu().numpy()), HIST_EDGES)[0]
        D2, D6, D16 = (ab.dilate(M, rings[r]) for r in (2, 6, 16))
        reg = dict(M=M, R2=D2 & ~M, R6=D6 & ~D2, R16=D16 & ~D6, far=~D16)
        dynfreq += M.double()
        cam = native.hr_camera(ev.camera_item(manifest, o, torch), index[c], manifest)
        sel = {'parent': None, 'full': subset(res), 'P': subset(res, keep=('dxyz',)), 'C': subset(res, keep=('dsh0',))}
        for t in taus:
            sel[f'vis{t:g}'] = subset(res, vis[t])
            sel[f'hid{t:g}'] = subset(res, ~vis[t])
        for k in ('W0', 'W1', 'A0', 'A1', 'A2'):
            sel[k] = subset(res, cls[k])
        sel['hidP'] = subset(res, hid_m, keep=('dxyz',))
        sel['hidC'] = subset(res, hid_m, keep=('dsh0',))
        e2s, Os = {}, {}
        with torch.no_grad():
            for arm in arms:
                st = S if sel[arm] is None else pf.apply_residual(S, sel[arm])
                O = ab.area(model.apply(native.native_rgb(cam, g, st, pipe, background)['render'], c), (lr_h, lr_w))
                e2 = ((O - y) ** 2).sum(0)
                e2s[arm] = e2
                for r, m in reg.items():
                    acc[arm][r][0] += float(e2[m].sum())
                    acc[arm][r][1] += int(m.sum()) * 3
                acc[arm]['all'][0] += float(e2.sum())
                acc[arm]['all'][1] += e2.numel() * 3
                if arm in tacc:
                    if arm in prev and prev[arm][0] == f - 1:
                        _, O0, y0, m0 = prev[arm]
                        t2 = (((O - O0) - (y - y0)) ** 2).sum(0)
                        mm = M | m0
                        tacc[arm]['tse'] += float(t2.sum())
                        tacc[arm]['tn'] += t2.numel() * 3
                        tacc[arm]['tse_d'] += float(t2[mm].sum())
                        tacc[arm]['tn_d'] += int(mm.sum()) * 3
                    prev[arm] = (f, O, y, M)
                    Os[arm] = O
        for arm in map_arms:
            mapsum[arm] += (e2s[arm] - e2s['parent']).double()
        npx = 3 * lr_h * lr_w
        dm = {k: float((e2s[k] - e2s['parent']).sum()) / npx for k in ('full', f'hid{tm}', 'A0', 'A1')}
        per_frame.append(dict(frame=f, **{f'd_mse_{k}': v for k, v in dm.items()}))
        if dm['full'] > worst['delta']:
            worst = dict(delta=dm['full'], frame=f, y=y, parent=Os['parent'], full=Os['full'],
                         change=(e2s['full'] - e2s['parent']).cpu().numpy())
        mu_px = res['dxyz'].norm(dim=1) / px_world if res.get('dxyz') is not None else torch.zeros(n, device=dev)
        dc = res['dsh0'].abs().mean(1) if res.get('dsh0') is not None else torch.zeros(n, device=dev)
        op = S['opacity'][idx].reshape(-1)
        msc = S['scales'][idx].max(1).values
        maxA = A.max(0).values if n else torch.zeros(0, device=dev)
        mu2, dc2 = max(float((mu_px ** 2).sum()), 1e-12), max(float((dc ** 2).sum()), 1e-12)
        row = dict(frame=f, free=n, px_world=px_world, vis={f'{t:g}': int(vis[t].sum()) for t in taus})
        for k, mk in cls.items():
            row[k] = dict(count=int(mk.sum()), opacity=med(op[mk]), max_scale=med(msc[mk]), maxA=med(maxA[mk]),
                          mu_px=med(mu_px[mk]), dsh0=med(dc[mk]), mu2_share=float((mu_px[mk] ** 2).sum()) / mu2,
                          dsh02_share=float((dc[mk] ** 2).sum()) / dc2, cam_static_weight=float(wc_static[mk].sum()),
                          cam_dyn_weight=float(wc_dyn[mk].sum()))
        fstats.append(row)
        if f % 5 == 0 and n > 16 and res.get('dxyz') is not None:
            nb = ab.knn(S['xyz'][idx])
            mu = res['dxyz'] / px_world
            mnb = mu[nb].mean(1)
            cos = torch.nn.functional.cosine_similarity(mu, mnb, dim=1, eps=1e-12)
            ratio = mu.norm(dim=1) / mnb.norm(dim=1).clamp_min(1e-12)
            knn_rows.append(dict(frame=f, cos=dict(hid=med(cos[hid_m]), vis=med(cos[~hid_m]), A0=med(cos[cls['A0']]),
                                                   A1=med(cos[cls['A1']]), A2=med(cos[cls['A2']])),
                                 norm_ratio=dict(hid=med(ratio[hid_m]), vis=med(ratio[~hid_m]))))
        if f % 25 == 0:
            print(json.dumps(dict(frame=f, free=n, W=[row[k]['count'] for k in ('W0', 'W1', 'W2')],
                                  A=[row[k]['count'] for k in ('A0', 'A1', 'A2')], d_mse=dm,
                                  seconds=round(time.time() - started, 1))), flush=True)
    nf = len(frames)
    summ = {arm: {r: dict(mse=se / max(nn, 1), psnr=psnr(se, nn)) for r, (se, nn) in d.items()} for arm, d in acc.items()}
    for arm in arms:
        tot = acc[arm]['all'][0] - acc['parent']['all'][0]
        summ[arm]['share_of_change'] = {r: ((acc[arm][r][0] - acc['parent'][r][0]) / tot if abs(tot) > 0 else None) for r in REGIONS}
        summ[arm]['d_psnr_vs_parent'] = {r: summ[arm][r]['psnr'] - summ['parent'][r]['psnr'] for r in REGIONS + ('all',)}
    tdiff = {arm: dict(tdiff_psnr=psnr(t['tse'], t['tn']), tdiff_dyn_psnr=psnr(t['tse_d'], t['tn_d'])) for arm, t in tacc.items()}
    region_fraction = {r: acc['parent'][r][1] / max(acc['parent']['all'][1], 1) for r in REGIONS}
    trend = {}
    for k in ('full', f'hid{tm}', 'A0', 'A1'):
        v = [r[f'd_mse_{k}'] for r in per_frame]
        trend[k] = dict(first100=float(np.mean(v[:100])), last100=float(np.mean(v[-100:])), slope_per_frame=slope(v))
    classes = {}
    for k in CLASSES:
        classes[k] = {}
        for m in CLASS_STATS:
            vals = [r[k][m] for r in fstats if r[k][m] is not None]
            classes[k][m] = float(np.median(vals)) if vals else None
    knn = {}
    for key in ('cos', 'norm_ratio'):
        groups = sorted({gname for r in knn_rows for gname in r[key]})
        knn[key] = {gname: (float(np.median([r[key][gname] for r in knn_rows if r[key][gname] is not None]))
                            if any(r[key][gname] is not None for r in knn_rows) else None) for gname in groups}
    histograms = {k: dict(edges_log10=HIST_EDGES.tolist(), counts=v.astype(int).tolist(), quantiles_log10=hist_quantiles(v))
                  for k, v in hist.items()}
    lim = max(float(torch.quantile(mapsum['full'].abs().flatten().float() / nf, 0.995)), 1e-8)
    tiles = [diverging((mapsum[arm] / nf).cpu().numpy(), lim) for arm in map_arms]
    tiles.append((255 * (1 - (dynfreq / nf).cpu().numpy())).astype(np.uint8))
    scale = 2 if lr_w < 500 else 1
    save_row(out / 'maps' / ('mean_change_' + '_'.join(map_arms) + '_dynfreq.png'), tiles, scale)
    wl = max(float(np.quantile(np.abs(worst['change']), 0.995)), 1e-8)
    save_row(out / 'maps' / f'worst_frame_{worst["frame"]:04d}_gt_parent_full_change.png',
             [to_u8(worst['y']), to_u8(worst['parent']), to_u8(worst['full']), diverging(worst['change'], wl)], scale)
    rec = dict(doc=__doc__, kind='pc_heldout_diag', checkpoint=cr.entry(ck), fit=a.fit, camera=c, fitted=fitted,
               frames=[frames[0], frames[-1], nf], taus=taus, tau_mid=tau_mid, cull_record=cull_record,
               post_cull_xyz_sha256=xyz_sha, points=int(g._xyz.shape[0]), region_pixel_fraction=region_fraction, summary=summ,
               tdiff=tdiff, trend=trend, classes=classes, knn_drag=knn, histograms=histograms, unit_check=unit_check,
               free_set_mean=dict(free=float(np.mean([r['free'] for r in fstats])),
                                  vis={f'{t:g}': float(np.mean([r['vis'][f'{t:g}'] for r in fstats])) for t in taus}),
               per_frame=per_frame, free_set=fstats, knn_rows=knn_rows,
               map_scale_mse=dict(full_mean_change_lim=lim, worst_frame_lim=wl, worst_frame=worst['frame'], map_arms=list(map_arms)),
               host=socket.gethostname(), gpu=torch.cuda.get_device_name(), seconds=time.time() - started)
    (out / 'diag.json').write_text(json.dumps(rec, indent=1))
    print(json.dumps(dict(d_psnr={arm: {r: round(v, 3) for r, v in summ[arm]['d_psnr_vs_parent'].items()} for arm in arms},
                          share_full={r: (round(v, 3) if v is not None else None) for r, v in summ['full']['share_of_change'].items()},
                          tdiff={k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in tdiff.items()},
                          counts={k: classes[k]['count'] for k in CLASSES}, trend=trend, unit_check=unit_check,
                          seconds=round(rec['seconds'], 1))), flush=True)


if __name__ == '__main__':
    main()
