"""SR-stage module kit, part 13: motion supervision from the matched SR teacher video, and motion-appearance decoupling
(Claude, session 4dsr-25, 2026-10-21; design taken over from 4dsr-70, joint review with 4dsr-f3). New file; modkit6.py (hash
pinned below) and everything it imports are imported, never modified.

  modkit13.py [--flow-weight W] [--flow-dir DIR] [--flow-D 2] [--flow-views 0,1,2] [--flow-fb TAU] [--flow-dead DELTA]
              [--flow-start K] [--flow-model {gflow,centre}] [--app-weight A] [--app-sh-weight B] [--teacher-tf DIR]
              <modkit6 / modkit5 / modkit4 / modkit2 / modkit options> -- --mode fix cullretrain sr <SR args>

Premise (tmp/claude_perscene_design_20261021/out/motion_premise_*.json; CPU, training cameras, D = 2 frames): position at a given
time is fitted to 0.3-0.8 HR px, but the Gaussian trajectories between frames are off (moving-region EPE of the parent's projected
Gaussian-centre motion 1.98 / 0.92 HR px on Cook07 / Discussion07), while RAFT on the matched SR teacher video is at 0.75 / 0.39 and
RAFT on the LR video at 2.93 / 1.68 (all against RAFT on HR, diagnostic only).

--flow-weight W : weight of the flow term (0 = off). In every SR step (one same-time batch at frame f, cameras c0 c1 c2) with
    f + D inside the clip, the deformation is queried once more at the time of frame f + D (grad-enabled). For each selected view c
    the projected displacement d_i = pi_c(x_i(f + D)) - pi_c(x_i(f)) (HR px) of every Gaussian is splatted with the frame-f
    geometry (means, covariances and opacities DETACHED, so only the displacement receives gradient) as the precomputed colour
    (d_x, d_y, 1); flow = channels 0-1 / channel 2. The flow image is mapped to the observation frame with the camera's learned warp
    field (modkit2 --warp-field, detached; bilinear) and pooled to the LR grid (alpha-weighted 4x4 mean) = the grid of the cache.
    loss_c = sum over valid LR pixels of max(0, |flow - target| - DELTA) / max(#active, 64) [HR px];
    valid = fb < TAU and alpha_lr > 0.95; active = valid and (|target| > 0.5 or |flow| > 0.5). The term is the mean over views.
--flow-model    : gflow (default) = the Gaussian flow of GaussianFlow (Gao et al. 2024): a pixel x keeps its normalised position in
    each covering Gaussian, flow_i(x) = S2^(1/2) S1^(-1/2) (x - mu1) + mu2 - x with the 2D EWA covariances S1, S2 (+0.3 low-pass, as
    the rasterizer) and 2D means mu1, mu2 at f and f + D, so rotation / scale deformation is included. Because the per-pixel blend
    is linear, it is splatted exactly as sum_i w_i (mu2 - mu1) + (sum_i w_i B_i) x - sum_i w_i B_i mu1 with B_i = A_i - I (3 passes of
    3 precomputed channels). centre = only the centre displacement (the definition of the premise test).
--flow-dir DIR  : flowcache.py output directory of one source (<DIR>/<cam>.flow.npy, .fb.npy, .frames.npy; LR grid, HR px).
--app-weight A  : motion-appearance decoupling, A * (mean_i |alpha_i(f + D) - alpha_i(f)| + B * mean_i |SH_DC_i(f + D) - SH_DC_i(f)|_1)
    over all Gaussians (effective opacity after the sigmoid; DC band of the deformed SH). Wu et al.'s deformation also changes
    opacity and colour over time, so motion can be faked by cross-fading; this term charges such changes along the trajectory.
--teacher-tf DIR: temporally filtered teacher target (4dsr-f3's 'tf' control, no new loss): mean of T(f) and the warp of T(f + D) along
    the cached flow of DIR (a flowcache.py teacher_D<D> directory) where that flow is > 0.5 HR px and fb < 1 HR px; T(f) elsewhere.
    Tells "removing teacher flicker along motion" apart from "supervising motion".
Both terms are added to the registered regulariser (GaussianModel.compute_regulation, called once per SR step after the renders),
so the registered loss, schedule, densification statistics (the flow splat uses its own screen-space tensor without grad) and RNG
streams are otherwise untouched. modkit13.json in --out records the switches, the cache identity, term statistics and source hashes;
modkit13_trace.jsonl logs the terms every 20 steps.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
MOD = HERE.parent
GEO = MOD / 'geoprobe_20261017'
ROOT = MOD.parents[1]
for _p in (MOD, GEO):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import modkit6  # noqa: E402

PINNED = {'modkit6.py': '6eab5d7124643c5b'}


class FlowCache:
    """mmap of one flowcache.py source; target(c, f) -> (flow (2,h,w), fb_quantised (h,w)) as CUDA tensors or None."""

    def __init__(self, path, cameras):
        import numpy as np
        self.dir = Path(path)
        self.rows, self.flow, self.fb, self.bflow, self.bfb = {}, {}, {}, {}, {}
        for c in cameras:
            if not (self.dir / f'{c}.frames.npy').exists():
                continue
            fr = np.load(self.dir / f'{c}.frames.npy')
            self.rows[c] = {int(f): i for i, f in enumerate(fr)}
            self.flow[c] = np.load(self.dir / f'{c}.flow.npy', mmap_mode='r')
            self.fb[c] = np.load(self.dir / f'{c}.fb.npy', mmap_mode='r')
            if (self.dir / f'{c}.bflow.npy').exists():
                self.bflow[c] = np.load(self.dir / f'{c}.bflow.npy', mmap_mode='r')
                self.bfb[c] = np.load(self.dir / f'{c}.bfb.npy', mmap_mode='r')

    def target(self, c, f):
        import numpy as np
        import torch
        i = self.rows.get(c, {}).get(int(f))
        if i is None:
            return None
        fl = torch.from_numpy(np.asarray(self.flow[c][i], dtype=np.float32)).cuda(non_blocking=True)
        fb = torch.from_numpy(np.asarray(self.fb[c][i])).cuda(non_blocking=True)
        return fl, fb

    def target_cpu(self, c, f):
        import numpy as np
        i = self.rows.get(c, {}).get(int(f))
        if i is None:
            return None
        return np.asarray(self.flow[c][i], dtype=np.float32), np.asarray(self.fb[c][i])

    def back_cpu(self, c, f, D):
        """backward flow f -> f - D (stored with the pair f - D -> f) and its fb error, or None"""
        import numpy as np
        i = self.rows.get(c, {}).get(int(f) - D)
        if i is None or c not in self.bflow:
            return None
        return np.asarray(self.bflow[c][i], dtype=np.float32), np.asarray(self.bfb[c][i])


def make_tf_reader(original, cache, D, stats):
    """--teacher-tf: the teacher target of (camera c, frame f) becomes the per-pixel mean of T(f) and its neighbours T(f + D) / T(f - D)
    warped back along the cached teacher flows f -> f + D / f -> f - D, each neighbour only where its flow is > 0.5 HR px and its fb
    < 1 HR px (LR-grid cache, bilinearly upsampled); T(f) where no neighbour qualifies. Runs in the prefetch worker threads (CPU)."""
    import re
    import numpy as np
    import torch
    import torch.nn.functional as F
    pat = re.compile(r'/(cam\d+)/(\d+)\.png$')

    def read_teacher(path):
        t = original(path)
        m = pat.search(str(path).replace('\\', '/'))
        if m is None:
            stats['tf_unparsed'] += 1
            return t
        c, f = m.group(1), int(m.group(2))
        h, w = t.shape[-2:]
        py, px = torch.meshgrid(torch.arange(h, dtype=torch.float32), torch.arange(w, dtype=torch.float32), indexing='ij')
        num, den, used = t.clone(), torch.ones(h, w), 0
        for tg, g in ((cache.target_cpu(c, f), f + D), (cache.back_cpu(c, f, D), f - D)):
            if tg is None:
                continue
            fl, fbq = tg
            flow = F.interpolate(torch.from_numpy(fl)[None], size=(h, w), mode='bilinear', align_corners=False)[0]
            ok = torch.from_numpy(((np.linalg.norm(fl, axis=0) > 0.5) & (fbq < 10)).astype(np.float32))
            ok = F.interpolate(ok[None, None], size=(h, w), mode='nearest')[0, 0]
            nb = original(str(path)[:m.start()] + f'/{c}/{g:04d}.png')
            grid = torch.stack([(px + flow[0]) * (2 / (w - 1)) - 1, (py + flow[1]) * (2 / (h - 1)) - 1], -1)
            warped = F.grid_sample(nb[None], grid[None], mode='bilinear', padding_mode='border', align_corners=True)[0]
            num += ok[None] * warped
            den += ok
            used += 1
        if not used:
            stats['tf_missing'] += 1
            return t
        stats['tf_reads'] += 1
        stats['tf_pixels'] += float((den > 1).float().mean())
        return num / den[None]

    return read_teacher


def install(cr, native, modkit2, s, stats, out):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from scene.gaussian_model import GaussianModel
    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
    ctx = dict(g=None, cursor=0, cams=[], state=None, manifest=None, times=None, cache=None)
    hr_camera, effective, regulation = native.hr_camera, native.native_effective_state, GaussianModel.compute_regulation
    restore, ulr = GaussianModel.restore, GaussianModel.update_learning_rate
    views = [int(v) for v in s['flow_views'].split(',') if v != '']
    trace = (Path(out) / 'modkit13_trace.jsonl').open('a', buffering=1)
    acc = dict(flow=[], epe_active=[], app_alpha=[], app_sh=[])

    def restore_(self, model_args, training_args):
        restore(self, model_args, training_args)
        ctx['g'] = self

    def ulr_(self, iteration):
        if self is ctx['g']:
            ctx['cursor'] = int(iteration) - cr.LR_OFFSET
            ctx['cams'], ctx['state'] = [], None
            if ctx['cursor'] == 20 and stats['calls'] == 0:
                raise RuntimeError('modkit13: compute_regulation hook never fired (time_smoothness_weight == 0?)')
        return ulr(self, iteration)

    def hr_camera_(observation, uid, manifest):
        camera = hr_camera(observation, uid, manifest)
        if ctx['manifest'] is None:
            ctx['manifest'] = manifest
            ctx['times'] = {(o['camera_id'], int(o['frame_index'])): float(o['time']) for o in manifest['observations']}
            if s['flow_weight']:
                ctx['cache'] = FlowCache(ROOT / s['flow_dir'] if not Path(s['flow_dir']).is_absolute() else s['flow_dir'],
                                         list(manifest['splits']['train']))
                stats['cache_cameras'] = len(ctx['cache'].rows)
        if 'split' in observation and observation['split'] != 'train':
            return camera
        ctx['cams'].append((camera, observation['camera_id'], int(observation['frame_index'])))
        return camera

    def effective_(g, time_value, torch_module=None):
        st = effective(g, time_value, torch_module)
        if g is ctx['g'] and torch.is_grad_enabled() and ctx['state'] is None:
            ctx['state'] = st
        return st

    def project(xyz, cam, w, h):
        P = cam.full_proj_transform.to(xyz.device)
        p = torch.cat([xyz, torch.ones_like(xyz[:, :1])], 1) @ P
        ndc = p[:, :2] / (p[:, 3:4] + 1e-7)
        return torch.stack([((ndc[:, 0] + 1) * w - 1) * 0.5, ((ndc[:, 1] + 1) * h - 1) * 0.5], 1)

    def cov2d(st, cam, w, h):
        V = cam.world_view_transform.to(st['xyz'].device)
        pc = torch.cat([st['xyz'], torch.ones_like(st['xyz'][:, :1])], 1) @ V
        tz = pc[:, 2].clamp_min(1e-4)
        tanx, tany = math.tan(cam.FoVx * .5), math.tan(cam.FoVy * .5)
        fx, fy = w / (2 * tanx), h / (2 * tany)
        tx = (pc[:, 0] / tz).clamp(-1.3 * tanx, 1.3 * tanx) * tz
        ty = (pc[:, 1] / tz).clamp(-1.3 * tany, 1.3 * tany) * tz
        J = torch.zeros(tz.shape[0], 2, 3, device=tz.device)
        J[:, 0, 0], J[:, 0, 2] = fx / tz, -fx * tx / tz ** 2
        J[:, 1, 1], J[:, 1, 2] = fy / tz, -fy * ty / tz ** 2
        T = J @ V[:3, :3].T
        S = T @ native.state_covariance(st, torch) @ T.transpose(1, 2)
        return S + 0.3 * torch.eye(2, device=S.device)

    def sqrt2(M):
        s_ = (M[:, 0, 0] * M[:, 1, 1] - M[:, 0, 1] * M[:, 1, 0]).clamp_min(1e-12).sqrt()
        t_ = (M[:, 0, 0] + M[:, 1, 1] + 2 * s_).clamp_min(1e-12).sqrt()
        return (M + s_[:, None, None] * torch.eye(2, device=M.device)) / t_[:, None, None]

    def inv2(M):
        det = (M[:, 0, 0] * M[:, 1, 1] - M[:, 0, 1] * M[:, 1, 0]).clamp_min(1e-12)
        return torch.stack([torch.stack([M[:, 1, 1], -M[:, 0, 1]], -1), torch.stack([-M[:, 1, 0], M[:, 0, 0]], -1)], 1) / det[:, None, None]

    def model_flow(cam, st, st2):
        """(3, H, W): alpha-weighted flow (x, y) sums and the alpha sum; normalise by channel 2"""
        w, h = int(cam.image_width), int(cam.image_height)
        mu1, mu2 = project(st['xyz'], cam, w, h), project(st2['xyz'], cam, w, h)
        d = (mu2 - mu1).clamp(-64, 64)
        one = torch.ones_like(d[:, :1])
        p1 = splat(cam, st, torch.cat([d, one], 1))
        if s['flow_model'] == 'centre':
            return p1
        B = sqrt2(cov2d(st2, cam, w, h)) @ inv2(sqrt2(cov2d(st, cam, w, h))) - torch.eye(2, device=d.device)
        B = B.clamp(-4, 4)
        v = (B @ mu1[:, :, None])[:, :, 0]
        p2 = splat(cam, st, torch.stack([B[:, 0, 0], B[:, 0, 1], B[:, 1, 0]], 1))
        p3 = splat(cam, st, torch.cat([B[:, 1, 1:2], v], 1))
        py, px = torch.meshgrid(torch.arange(h, device=d.device, dtype=torch.float32),
                                torch.arange(w, device=d.device, dtype=torch.float32), indexing='ij')
        fx_ = p1[0] + p2[0] * px + p2[1] * py - p3[1]
        fy_ = p1[1] + p2[2] * px + p3[0] * py - p3[2]
        return torch.stack([fx_, fy_, p1[2]])

    def splat(cam, st, colors):
        w, h = int(cam.image_width), int(cam.image_height)
        settings = GaussianRasterizationSettings(image_height=h, image_width=w, tanfovx=math.tan(cam.FoVx * .5),
                                                 tanfovy=math.tan(cam.FoVy * .5), bg=torch.zeros(3, device=colors.device),
                                                 scale_modifier=1., viewmatrix=cam.world_view_transform.to(colors.device),
                                                 projmatrix=cam.full_proj_transform.to(colors.device), sh_degree=0,
                                                 campos=cam.camera_center.to(colors.device), prefiltered=False, debug=False)
        points = torch.zeros_like(st['xyz'], requires_grad=False)
        out_ = GaussianRasterizer(raster_settings=settings)(means3D=st['xyz'].detach(), means2D=points, shs=None,
                                                            colors_precomp=colors, opacities=st['opacity'].detach(),
                                                            scales=st['scales'].detach(), rotations=st['rotation'].detach(),
                                                            cov3D_precomp=None)
        return out_[0]

    def to_obs(img, cid):
        aff = modkit2.REG.get('affine')
        if aff is None or not aff.ext.get('wf_grid') or cid not in aff.index:
            return img
        h, w = img.shape[-2:]
        d = (aff.wfield[aff.index[cid]] - aff.wfield.mean(0)).detach()
        up = F.interpolate(d[None], size=(h, w), mode='bilinear', align_corners=True)[0]
        py, px = torch.meshgrid(torch.arange(h, device=img.device, dtype=torch.float32) + 0.5,
                                torch.arange(w, device=img.device, dtype=torch.float32) + 0.5, indexing='ij')
        grid = torch.stack([(px + up[0]) * (2 / w) - 1, (py + up[1]) * (2 / h) - 1], -1)
        return F.grid_sample(img[None], grid[None], mode='bilinear', padding_mode='border', align_corners=False)[0]

    def compute_regulation(self, *args, **kwargs):
        value = regulation(self, *args, **kwargs)
        if self is not ctx['g'] or not torch.is_grad_enabled():
            return value
        stats['calls'] += 1
        cur, st, cams = ctx['cursor'], ctx['state'], ctx['cams']
        if st is None or len(cams) < 3 or cur < s['flow_start']:
            return value
        f = cams[0][2]
        t2 = ctx['times'].get((cams[0][1], f + s['flow_D']))
        if t2 is None:
            stats['skipped_end'] += 1
            return value
        st2 = effective(self, t2, torch)
        extra = value.new_zeros(())
        row = dict(cursor=cur, frame=f)
        if s['flow_weight']:
            terms, epes = [], []
            for v in views:
                cam, cid, _ = cams[v]
                tgt = ctx['cache'].target(cid, f)
                if tgt is None:
                    stats['missing_target'] += 1
                    continue
                target, fbq = tgt
                w = int(cam.image_width)
                img = to_obs(model_flow(cam, st, st2), cid)
                k = w // target.shape[-1]
                a_lr = F.avg_pool2d(img[2:3][None], k)[0, 0]
                fl_lr = F.avg_pool2d(img[:2][None], k)[0] / a_lr.clamp_min(1e-4)
                e = (fl_lr - target).norm(dim=0)
                valid = (fbq < s['flow_fb'] * 10) & (a_lr.detach() > 0.95)
                active = valid & ((target.norm(dim=0) > 0.5) | (fl_lr.detach().norm(dim=0) > 0.5))
                n_act = int(active.sum())
                terms.append(torch.relu(e - s['flow_dead'])[valid].sum() / max(n_act, 64))
                epes.append(float(e.detach()[active].mean()) if n_act else float('nan'))
                row.setdefault('views', []).append(dict(camera=cid, valid=int(valid.sum()), active=n_act, epe_active=epes[-1]))
            if terms:
                flow_term = torch.stack(terms).mean()
                extra = extra + s['flow_weight'] * flow_term
                acc['flow'].append(float(flow_term.detach()))
                acc['epe_active'].append(float(np.nanmean(epes)) if not all(math.isnan(x) for x in epes) else float('nan'))
                row['flow'] = acc['flow'][-1]
                stats['flow_steps'] += 1
        if s['app_weight']:
            da = (st2['opacity'] - st['opacity']).abs().mean()
            ds = (st2['sh'][:, 0, :] - st['sh'][:, 0, :]).abs().sum(-1).mean()
            extra = extra + s['app_weight'] * (da + s['app_sh_weight'] * ds)
            acc['app_alpha'].append(float(da.detach()))
            acc['app_sh'].append(float(ds.detach()))
            row.update(app_alpha=acc['app_alpha'][-1], app_sh=acc['app_sh'][-1])
            stats['app_steps'] += 1
        if not bool(torch.isfinite(extra)):
            raise FloatingPointError('modkit13: nonfinite motion term')
        if cur % 20 == 0:
            trace.write(json.dumps(row) + '\n')
        return value + extra

    GaussianModel.restore, GaussianModel.update_learning_rate = restore_, ulr_
    GaussianModel.compute_regulation = compute_regulation
    native.hr_camera, native.native_effective_state = hr_camera_, effective_
    return acc


def main():
    argv = sys.argv[1:]
    if '--' not in argv:
        raise SystemExit(__doc__)
    modkit2 = modkit6.modkit5.modkit4.modkit2
    sha = modkit2.modkit.calibsr_v2.sha
    for name, digest in PINNED.items():
        if not sha(GEO / name).startswith(digest):
            raise SystemExit(f'{name} changed')
    k = argv.index('--')
    own, rest = argv[:k], argv[k + 1:]
    s = dict(flow_weight=0.0, flow_dir=None, flow_D=2, flow_views='0,1,2', flow_fb=1.0, flow_dead=0.25, flow_start=1,
             flow_model='gflow', app_weight=0.0, app_sh_weight=1.0, teacher_tf=None)
    flags = {'--flow-weight': ('flow_weight', float), '--flow-dir': ('flow_dir', str), '--flow-D': ('flow_D', int),
             '--flow-views': ('flow_views', str), '--flow-fb': ('flow_fb', float), '--flow-dead': ('flow_dead', float),
             '--flow-start': ('flow_start', int), '--flow-model': ('flow_model', str), '--app-weight': ('app_weight', float),
             '--app-sh-weight': ('app_sh_weight', float), '--teacher-tf': ('teacher_tf', str)}
    passthrough = []
    i = 0
    while i < len(own):
        if own[i] in flags:
            name, cast = flags[own[i]]
            s[name] = cast(own[i + 1]); i += 2
        else:
            passthrough.append(own[i]); i += 1
    if s['flow_weight'] < 0 or s['app_weight'] < 0 or s['app_sh_weight'] < 0 or not 0 < s['flow_fb'] <= 25.5 or s['flow_D'] < 1:
        raise SystemExit('switch out of range')
    if s['flow_model'] not in ('gflow', 'centre'):
        raise SystemExit('--flow-model must be gflow or centre')
    if s['flow_weight'] and not s['flow_dir']:
        raise SystemExit('--flow-weight needs --flow-dir')
    if s['teacher_tf'] and not str(s['teacher_tf']).rstrip('/').endswith(f"_D{s['flow_D']}"):
        raise SystemExit('--teacher-tf must be a flowcache.py <source>_D<D> directory matching --flow-D')
    if s['flow_dir'] and not str(s['flow_dir']).rstrip('/').endswith(f"_D{s['flow_D']}"):
        raise SystemExit('--flow-dir must be a flowcache.py <source>_D<D> directory matching --flow-D')
    if len(rest) < 4 or rest[0] != '--mode' or rest[2] != 'cullretrain' or rest[3] != 'sr':
        raise SystemExit('expected -- --mode fix cullretrain sr ...')
    sr_args = rest[3:]
    out = Path(sr_args[sr_args.index('--out') + 1])
    out = out if out.is_absolute() else ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    import srfix  # noqa: E402
    cr = srfix.cr
    native = cr.native
    stats = dict(calls=0, flow_steps=0, app_steps=0, skipped_end=0, missing_target=0, tf_reads=0, tf_missing=0, tf_unparsed=0,
                 tf_pixels=0.0)
    holder = {}
    real_main = srfix.main
    started = time.time()

    def motion_main():
        """innermost wrapper (installed before modkit6 / modkit5 / ... install theirs): runs right before the SR loop"""
        if s['flow_weight'] or s['app_weight']:
            holder['acc'] = install(cr, native, modkit2, s, stats, out)
        if s['teacher_tf']:
            p = Path(s['teacher_tf'])
            p = p if p.is_absolute() else ROOT / p
            sr_parent = Path(sr_args[sr_args.index('--parent-checkpoint') + 1])
            import torch
            plan = torch.load(sr_parent if sr_parent.is_absolute() else ROOT / sr_parent, map_location='cpu', weights_only=False)['plan']
            _, manifest = cr.plan_scene(plan)
            cr.read_teacher = make_tf_reader(cr.read_teacher, FlowCache(p, list(manifest['splits']['train'])), s['flow_D'], stats)
        return real_main()

    srfix.main = motion_main
    sys.argv = [str(GEO / 'modkit6.py')] + passthrough + ['--'] + rest
    modkit6.main()
    accs = holder.get('acc') or {}

    def window(xs, a, b):
        v = [x for x in xs[a:b] if x == x]
        return sum(v) / len(v) if v else None

    term = {k: dict(first500=window(v, 0, 500), last500=window(v, -500, None), n=len(v)) for k, v in accs.items()}
    cache_record = None
    if s['flow_dir']:
        p = Path(s['flow_dir'])
        p = p if p.is_absolute() else ROOT / p
        js = sorted(p.parent.glob(f'flowcache_{p.name.rsplit("_D", 1)[0]}_D{s["flow_D"]}.json'))
        cache_record = dict(dir=str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p),
                            record=str(js[0].relative_to(ROOT)) if js else None, record_sha256=sha(js[0]) if js else None)
    record = dict(switches=s, passthrough=passthrough, argv=rest, stats=stats, terms=term, flow_cache=cache_record,
                  seconds=time.time() - started,
                  sources={str(Path(f).resolve().relative_to(ROOT)): sha(f) for f in (__file__, modkit6.__file__, srfix.__file__)})
    (out / 'modkit13.json').write_text(json.dumps(record, indent=1))
    print(json.dumps(dict(modkit13=s, stats=stats, terms=term), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
