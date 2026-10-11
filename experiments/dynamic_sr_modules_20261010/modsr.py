"""Module switches on top of the validated 2026-10-09 cull/affine/D-SSIM entry (cullretrain.py is imported, never modified).

  lr      LR stage as cullretrain.run_lr, optionally with --time-full-range (HexPlane time coordinate 2t-1).
  sr      SR stage as cullretrain.run_sr plus switches (all off = cullretrain.run_sr, first-step loss bit-identical):
            --teacher-pos-detach MODE   'none' | 'all' | 'late'  teacher views b, c render with detached deformed centres
                                        (prior gradients update appearance/shape/opacity but never positions);
                                        'late' applies it for cursor > --detach-start * stop
            --xyz-rewarm ETA            add ETA * spatial_lr_scale * ramp * cosine to the author xyz learning rate
            --sh-rest-lr-mult M         multiply the constant f_rest learning rate
            --holdout-cameras CAM ...   drop every schedule row containing these cameras (SR-stage holdout for selection)
            --time-full-range           parent must have been trained with the same flag (checked via modsr.json)
  eval    cullretrain.run_eval with --time-full-range support.
  extras  legal-signal and diagnostic metrics of one checkpoint: holdout-camera teacher/LR agreement, affine-corrected
          training-camera diagnostics, all-camera LR-closure run check, cam00 region x band budget (diagnostic only).
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import math
import socket
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
CULL = HERE.parent / 'dynamic_sr_cull_retrain_20261009'
sys.path.insert(0, str(CULL))

import cullretrain as cr  # noqa: E402  (sets up runtime paths, imports torch and the author code)
import rt  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

native, prefix, losses, ev = cr.native, cr.prefix, cr.losses, cr.ev
TIME_PATCHED = False


def patch_time_full_range():
    """Map the author's time value t = frame/300 in [0, 1) to 2t - 1 before the HexPlane lookup (align_corners=True),
    so that all time cells are used. Applied once per process; every deformation call goes through get_density."""
    global TIME_PATCHED
    if TIME_PATCHED:
        return
    from scene.hexplane import HexPlaneField
    original = HexPlaneField.get_density

    def get_density(self, pts, timestamps=None):
        return original(self, pts, timestamps * 2.0 - 1.0)

    HexPlaneField.get_density = get_density
    TIME_PATCHED = True


def mod_record(path):
    p = Path(path)
    p = p if p.is_dir() else p.parent
    f = p / 'modsr.json'
    return json.loads(f.read_text()) if f.exists() else {}


def check_time_flag(checkpoint, flag):
    parent = mod_record(checkpoint)
    if bool(parent.get('time_full_range', False)) != bool(flag):
        raise ValueError(f'time mapping mismatch: checkpoint trained with time_full_range={parent.get("time_full_range", False)}, '
                         f'requested {flag}')


def write_mod(out, a, kind):
    rt.write_json(Path(out) / 'modsr.json', dict(kind=kind, time_full_range=bool(getattr(a, 'time_full_range', False)),
                                                  switches={k: v for k, v in vars(a).items() if k not in ('func',)},
                                                  source={rel: rt.sha(p) for rel, p in (('modsr.py', Path(__file__)),)}))


# ----------------------------------------------------------------------------------------------- LR stage
def run_lr(a):
    if a.time_full_range:
        patch_time_full_range()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    write_mod(a.out, a, 'lr')
    return cr.run_lr(a)


# ----------------------------------------------------------------------------------------------- SR stage
def run_sr(a):
    """cullretrain.run_sr with module switches; identical code path when every switch is off."""
    from scene.gaussian_model import GaussianModel
    from utils.loss_utils import ssim
    if a.time_full_range:
        patch_time_full_range()
    check_time_flag(a.parent_checkpoint, a.time_full_range)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'complete.json').exists():
        raise FileExistsError('Completed run preserved: ' + str(out))
    write_mod(out, a, 'sr')
    base = cr.read(Path(a.baseline_sr) / 'config.json')
    definition = base['definition']
    if definition != dict(schedule='rows', lr_weights=[1.0, 0.0, 0.0], sr_weights=[0.0, 0.05, 0.05], X=False, E=False):
        raise ValueError('Only the registered Bsync definition is supported')
    parent_ckpt = Path(a.parent_checkpoint)
    payload = torch.load(parent_ckpt, map_location='cuda', weights_only=False)
    plan = payload['plan']
    if plan['configuration'] != base['configuration']:
        raise ValueError('Parent configuration differs from the baseline SR configuration')
    h, opt, pipe, sh = cr.configuration(plan)
    manifest_path, manifest = cr.plan_scene(plan)
    if cr.rel(manifest_path) != base['data']['manifest']['path']:
        raise ValueError('Manifest differs from the baseline SR plan')
    dataset = cr.N3DVPreparedDataset(manifest, 'train', 'lr', device='cpu', cache=False)
    extent = plan['data']['cameras_extent']
    torch.set_num_threads(a.cpu_threads)
    started = time.time()
    record = dict(kind='sr', arm=a.arm, args={k: v for k, v in vars(a).items() if k != 'func'}, parent_checkpoint=cr.entry(parent_ckpt),
                  baseline_sr=cr.rel(a.baseline_sr), scene=plan['scene'], seed=plan['seed'], lambda_SR=a.lambda_sr, dssim=a.dssim,
                  host=socket.gethostname(), gpu=torch.cuda.get_device_name(), runtime=rt.check_runtime(),
                  sources={cr.rel(p): rt.sha(p) for p in (Path(cr.__file__), Path(rt.__file__), Path(__file__))}, started_unix=started)
    g = GaussianModel(sh, h)
    g._deformation = g._deformation.cuda()
    g.restore(payload['model'], opt)
    g._deformation_accum = payload['deformation_accum']
    prefix.restore_rng(payload['rng'], torch, np)
    affine = None
    if a.affine:
        affine = cr.CameraAffine(manifest['splits']['train'], lr_init=a.affine_sr_lr, lr_final=a.affine_sr_lr, max_steps=cr.SR_STEPS)
        if payload.get('affine') is not None:
            affine.load_state_dict(payload['affine'])
            for group in affine.optimizer.param_groups:
                group['lr'] = a.affine_sr_lr
            record['affine_inherited'] = True
    del payload
    if a.sh_rest_lr_mult != 1.0:
        for group in g.optimizer.param_groups:
            if group['name'] == 'f_rest':
                group['lr'] = group['lr'] * a.sh_rest_lr_mult
                record['f_rest_lr'] = group['lr']
    g.optimizer.zero_grad(set_to_none=True)
    with torch.no_grad():
        for name in ('max_radii2D', 'xyz_gradient_accum', 'denom'):
            getattr(g, name).zero_()
    table = cr.read(cr.ROOT / base['dependencies']['schedule']['path'])
    rows = table[definition['schedule']]
    record_keys = [tuple(k) for k in table['record_keys']]
    holdout = set(a.holdout_cameras or [])
    if holdout:
        unknown = holdout - set(manifest['splits']['train'])
        if unknown:
            raise ValueError(f'holdout cameras not in the training split: {sorted(unknown)}')
        rows = [r for r in rows if not ({record_keys[i][0] for i in r} & holdout)]
        record['holdout'] = dict(cameras=sorted(holdout), rows_kept=len(rows), refilled_to=cr.SR_STEPS if a.holdout_refill else None)
        if a.holdout_refill:
            # keep the optimisation length of the full schedule: cycle the remaining same-time rows in their order
            rows = [rows[i % len(rows)] for i in range(cr.SR_STEPS)]
    teacher_index = cr.read(cr.ROOT / base['dependencies']['teacher']['path'])
    teacher_rows = {(e['camera'], int(e['frame'])): e for e in teacher_index['entries']}
    root = Path(manifest['_root'])
    teacher_paths = {key: native.safe_child(root, e['relative_path']) for key, e in teacher_rows.items()}
    stop = a.stop or min(cr.SR_STEPS, len(rows))
    detach_from = {'none': None, 'all': 0, 'late': int(a.detach_start * stop)}[a.teacher_pos_detach]
    record.update(stop=stop, detach_from=detach_from)

    def fetch(i):
        row = rows[i]
        items = [dataset[index] for index in row]
        keys = [record_keys[index] for index in row]
        teachers = [cr.read_teacher(teacher_paths[keys[j]]) for j in (1, 2)]
        return row, items, keys, teachers

    prefetch = cr.Prefetch(fetch, list(range(stop)))
    culler = cr.Culler(manifest) if a.cull else None
    record.update(culler=culler.describe() if culler else None, prune_gate=a.prune_gate, affine=bool(a.affine),
                  cull_interval=a.cull_interval, schedule=cr.entry(cr.ROOT / base['dependencies']['schedule']['path']),
                  teacher_index=cr.entry(cr.ROOT / base['dependencies']['teacher']['path']))
    rt.write_json(out / 'run.json', record)
    background = torch.ones(3, dtype=torch.float32, device='cuda') if cr.white_background(plan) else torch.zeros(3, dtype=torch.float32, device='cuda')
    counts = dict(densify=0, prune=0, pruned_points=0, grow=0, opacity_reset=0)
    cull_rows = []
    log = (out / 'trajectory.jsonl').open('a', buffering=1)
    torch.cuda.reset_peak_memory_stats()
    try:
        if culler is not None:
            cull_rows.append(dict(cr.cull(g, culler, log, 'sr_start'), cursor=0))
        for cursor in range(1, stop + 1):
            tick = time.time()
            g.update_learning_rate(cr.LR_OFFSET + cursor)
            if a.xyz_rewarm > 0:
                ramp = min(1.0, cursor / max(1, a.rewarm_warmup))
                extra = a.xyz_rewarm * g.spatial_lr_scale * ramp * 0.5 * (1 + math.cos(math.pi * cursor / stop))
                for group in g.optimizer.param_groups:
                    if group['name'] == 'xyz':
                        group['lr'] = group['lr'] + extra
            if (cr.LR_OFFSET + cursor) % 1000 == 0:
                g.oneupSHdegree()
            g.optimizer.zero_grad(set_to_none=True)
            row, items, keys, teacher_cpu = prefetch.get(cursor - 1)
            if len({k[0] for k in keys}) != 3 or len({int(k[1]) for k in keys}) != 1:
                raise ValueError('Registered same-time batch differs')
            cameras = [native.hr_camera(obs, i, manifest) for i, obs in zip(row, items)]
            lrs = [obs['image'].cuda() for obs in items]
            teachers = [t.cuda() for t in teacher_cpu]
            state = native.native_effective_state(g, cameras[0].time, torch)
            detached = detach_from is not None and cursor > detach_from
            teacher_state = dict(state, xyz=state['xyz'].detach()) if detached else state
            packages = [native.native_rgb(cameras[0], g, state, pipe, background)] + \
                       [native.native_rgb(c, g, teacher_state, pipe, background) for c in cameras[1:]]
            rgbs = [p['render'] for p in packages]
            if affine is not None:
                rgbs = [affine.apply(rgb, obs['camera_id']) for rgb, obs in zip(rgbs, items)]
            lr_values = [losses.lr_loss(rgb, lr) for rgb, lr in zip(rgbs, lrs)]
            lr_term = sum(w * v for w, v in zip(definition['lr_weights'], lr_values))
            sr_values = [losses.sr_loss(rgbs[i], t) for i, t in zip((1, 2), teachers)]
            if a.dssim:
                down = losses.degradation(rgbs[0], lrs[0].shape[-2:])
                lr_term = (1 - a.dssim) * lr_term + a.dssim * (1 - ssim(down[None], lrs[0][None]))
                sr_mix = [(1 - a.dssim) * v + a.dssim * (1 - ssim(rgbs[i][None], t[None])) for v, i, t in zip(sr_values, (1, 2), teachers)]
                sr_term = a.lambda_sr * (sr_mix[0] + sr_mix[1])
            else:
                sr_term = a.lambda_sr * (sr_values[0] + sr_values[1])
            regularizer = g.compute_regulation(h.time_smoothness_weight, h.l1_time_planes, h.plane_tv_weight) if h.time_smoothness_weight != 0 else rgbs[0].sum() * 0.
            loss = lr_term + sr_term + regularizer
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError('Nonfinite SR objective')
            loss.backward()
            statistics = prefix.combine_batch_statistics(packages, torch)
            before = int(g.get_xyz.shape[0])
            with torch.no_grad():
                cr.topology(g, opt, 'fine', cursor, extent, out, statistics, a.prune_gate, counts)
                g.optimizer.step()
                g.optimizer.zero_grad(set_to_none=True)
            if affine is not None:
                affine.step(cursor)
            if culler is not None and cursor % a.cull_interval == 0:
                cull_rows.append(dict(cr.cull(g, culler, log, f'sr_{cursor}'), cursor=cursor))
            log.write(json.dumps(dict(cursor=cursor, loss=float(loss.detach()), LR_L1=[float(v.detach()) for v in lr_values],
                                      SR_L1=[float(v.detach()) for v in sr_values], LR_term=float(lr_term.detach()),
                                      SR_term=float(sr_term.detach()), regularizer=float(regularizer.detach()),
                                      points_before=before, points_after=int(g.get_xyz.shape[0]), keys=[list(k) for k in keys],
                                      detached=detached, seconds=time.time() - tick)) + '\n')
            del items, cameras, lrs, teachers, packages, rgbs, state, teacher_state, loss
        prefetch.close()
        if culler is not None and stop % a.cull_interval != 0:
            cull_rows.append(dict(cr.cull(g, culler, log, f'sr_end_{stop}'), cursor=stop))
        path = cr.save(out / f'checkpoint_{stop:05d}.pt', cr.SCHEMA_SR, g, plan,
                       dict(affine=affine.state_dict() if affine else None, metadata=dict(cursor=stop, arm=a.arm, modsr=True)))
        torch.cuda.synchronize()
        result = dict(status='completed', cursor=stop, seconds=time.time() - started, counts=counts, culls=cull_rows,
                      points=int(g.get_xyz.shape[0]), peak_GB=torch.cuda.max_memory_allocated() / 1e9,
                      affine=affine.summary() if affine else None, checkpoint=cr.entry(path))
        rt.write_json(out / 'complete.json', dict(record, **result))
        return result
    except BaseException:
        prefetch.close()
        rt.write_json(out / f'failure_{time.time_ns()}.json', dict(record, error=traceback.format_exc()))
        raise
    finally:
        log.close()


# ----------------------------------------------------------------------------------------------- evaluation
def run_eval(a):
    if a.time_full_range:
        patch_time_full_range()
    check_time_flag(a.checkpoint, a.time_full_range)
    return cr.run_eval(a)


def band_images(err):
    """Exact radial-FFT split of an HxWx3 error image into LR band (r<0.125), mid, high (cycles per HR pixel)."""
    h, w = err.shape[:2]
    r = np.sqrt(np.fft.fftfreq(h)[:, None] ** 2 + np.fft.fftfreq(w)[None] ** 2)
    spec = np.fft.fft2(err.astype(np.float64), axes=(0, 1))
    return [np.real(np.fft.ifft2(spec * m[..., None], axes=(0, 1))) for m in (r < .125, (r >= .125) & (r < .25), r >= .25)]


def region_masks(hr, hr_other):
    from scipy import ndimage as ndi
    dyn = ndi.binary_dilation(ndi.uniform_filter(np.abs(hr - hr_other).mean(-1), 9) > .04, iterations=8)
    gray = hr.mean(-1)
    edge = (np.hypot(ndi.sobel(gray, 0), ndi.sobel(gray, 1)) > .25) & ~dyn
    return dict(dynamic=dyn, static_edge=edge, static_flat=~dyn & ~edge)


def run_extras(a):
    """Legal selection signals (training data only) and development diagnostics (cam00 HR) for one checkpoint."""
    import lpips
    if a.time_full_range:
        patch_time_full_range()
    check_time_flag(a.checkpoint, a.time_full_range)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'extras.json').exists():
        raise FileExistsError('Completed extras preserved: ' + str(out))
    started = time.time()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    checkpoint = Path(a.checkpoint)
    g, payload = cr.load_for_render(checkpoint, torch.load(checkpoint, map_location='cpu', weights_only=False)['plan'])
    plan = payload['plan']
    affine_state = payload.get('affine')
    del payload
    h, opt, pipe, sh = cr.configuration(plan)
    manifest_path, manifest = cr.plan_scene(plan)
    record = dict(kind='extras', checkpoint=cr.entry(checkpoint), manifest=cr.entry(manifest_path), posthoc_cull=bool(a.cull),
                  gpu=torch.cuda.get_device_name(), runtime=rt.check_runtime(), sources={cr.rel(p): rt.sha(p) for p in (Path(cr.__file__), Path(__file__))})
    if a.cull:
        record['cull'] = cr.cull(g, cr.Culler(manifest), None, 'posthoc', optimizer_backed=False)
    affine = None
    if affine_state is not None:
        affine = cr.CameraAffine(manifest['splits']['train'])
        affine.load_state_dict(affine_state)
    legacy, closure, _, fft = ev.metric_helpers()
    metric = lpips.LPIPS(net='alex', version='0.1', spatial=False, pnet_rand=False, pretrained=True, verbose=False).cuda().eval().requires_grad_(False)
    background = torch.ones(3, dtype=torch.float32, device='cuda') if cr.white_background(plan) else torch.zeros(3, dtype=torch.float32, device='cuda')
    root = Path(manifest['_root'])
    obs_by = {(o['camera_id'], int(o['frame_index'])): o for o in manifest['observations']}
    teacher_paths = {}
    if a.baseline_sr:
        base = cr.read(Path(a.baseline_sr) / 'config.json')
        teacher_index = cr.read(cr.ROOT / base['dependencies']['teacher']['path'])
        teacher_paths = {(e['camera'], int(e['frame'])): native.safe_child(root, e['relative_path']) for e in teacher_index['entries']}

    def render(obs, uid):
        with torch.no_grad():
            cam = native.hr_camera(ev.camera_item(manifest, obs, torch), uid, manifest)
            state = native.native_effective_state(g, float(obs['time']), torch)
            return native.native_rgb(cam, g, state, pipe, background)['render'].detach()

    empty = None
    result = dict(record)
    # (1) holdout cameras: agreement with the frozen teacher and the true LR (no affine: unseen colour response)
    hold = {}
    for cam in a.holdout_cameras or []:
        rows = []
        for f in range(5, 300, a.holdout_stride):
            obs = obs_by[(cam, f)]
            raw = render(obs, f)
            pred = legacy.image_array(raw)
            empty = np.zeros(pred.shape[:2], bool)
            row = dict(frame=f, LR_closure_psnr=closure(raw, legacy.read_rgb(root / obs['lr_path']), empty)['full']['psnr'])
            if teacher_paths:
                t_np = cr.read_teacher(teacher_paths[(cam, f)]).permute(1, 2, 0).numpy().astype(np.float32)
                q = legacy.spatial_metrics(pred, t_np, empty, metric, 'cuda')['full']
                row.update(teacher_psnr=q['psnr'], teacher_ssim=q['ssim'], teacher_lpips=q['lpips_alex'])
            rows.append(row)
        hold[cam] = dict(frames=len(rows), **{k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k != 'frame'})
    result['holdout'] = hold
    # (2) all training cameras, LR closure with each camera's affine (legal run-quality check) and without
    qc = {}
    for cam in manifest['splits']['train']:
        vals, vals_raw = [], []
        for f in a.qc_frames:
            obs = obs_by[(cam, f)]
            raw = render(obs, f)
            lr = legacy.read_rgb(root / obs['lr_path'])
            empty = np.zeros((raw.shape[1], raw.shape[2]), bool)
            vals_raw.append(closure(raw, lr, empty)['full']['psnr'])
            if affine is not None:
                with torch.no_grad():
                    corrected = affine.apply(raw, cam).clamp(0, 1)
                vals.append(closure(corrected, lr, empty)['full']['psnr'])
        qc[cam] = dict(raw=float(np.mean(vals_raw)), affine=float(np.mean(vals)) if vals else None)
    result['train_LR_closure'] = qc
    # (3) development diagnostics with HR (never used for selection): affine-corrected training-camera HR metrics and
    #     cam00 region x band budget
    if a.diagnostics:
        diag_cam = manifest['splits']['train'][0]
        drows = []
        for f in range(15, 300, 30):
            obs = obs_by[(diag_cam, f)]
            raw = render(obs, f)
            hr = legacy.read_rgb(root / obs['hr_path'])
            empty = np.zeros(hr.shape[:2], bool)
            q = legacy.spatial_metrics(legacy.image_array(raw), hr, empty, metric, 'cuda')['full']
            row = dict(psnr_raw=q['psnr'], lpips_raw=q['lpips_alex'])
            if affine is not None:
                with torch.no_grad():
                    corrected = affine.apply(raw, diag_cam).clamp(0, 1)
                qa = legacy.spatial_metrics(legacy.image_array(corrected), hr, empty, metric, 'cuda')['full']
                row.update(psnr_affine=qa['psnr'], lpips_affine=qa['lpips_alex'])
            drows.append(row)
        result['train_diagnostic'] = dict(camera=diag_cam, **{k: float(np.mean([r[k] for r in drows])) for k in drows[0]})
        budget = {}
        for f in range(0, 300, a.region_stride):
            obs = obs_by[('cam00', f)]
            pred = legacy.image_array(render(obs, f)).astype(np.float64)
            hr = legacy.read_rgb(root / obs['hr_path']).astype(np.float64)
            other = legacy.read_rgb(root / obs_by[('cam00', f + 30 if f + 30 < 300 else f - 30)]['hr_path']).astype(np.float64)
            masks = region_masks(hr, other)
            bands = band_images(pred - hr)
            n = hr.shape[0] * hr.shape[1] * 3
            for rname, m in masks.items():
                cell = budget.setdefault(rname, dict(area=0.0, low=0.0, mid=0.0, high=0.0))
                cell['area'] += float(m.mean())
                for bname, b in zip(('low', 'mid', 'high'), bands):
                    cell[bname] += float((b[m] ** 2).sum() / n)
        frames = len(range(0, 300, a.region_stride))
        result['cam00_region_band'] = {r: {k: v / frames for k, v in c.items()} for r, c in budget.items()}
    result['seconds'] = time.time() - started
    rt.write_json(out / 'extras.json', result)
    return result


def render_values(camera, g, state, values, pipe, background):
    """Author raster settings with per-Gaussian precomputed colours (values: N x 3)."""
    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
    xyz = state['xyz']
    points = torch.zeros_like(g.get_xyz, dtype=g.get_xyz.dtype, device=xyz.device, requires_grad=True) + 0
    settings = GaussianRasterizationSettings(image_height=int(camera.image_height), image_width=int(camera.image_width),
                                             tanfovx=math.tan(camera.FoVx * .5), tanfovy=math.tan(camera.FoVy * .5), bg=background,
                                             scale_modifier=1., viewmatrix=camera.world_view_transform.to(xyz.device),
                                             projmatrix=camera.full_proj_transform.to(xyz.device), sh_degree=g.active_sh_degree,
                                             campos=camera.camera_center.to(xyz.device), prefiltered=False, debug=pipe.debug)
    image, radii, depth = GaussianRasterizer(raster_settings=settings)(means3D=xyz, means2D=points, shs=None, colors_precomp=values,
                                                                         opacities=state['opacity'], scales=state['scales'],
                                                                         rotations=state['rotation'], cov3D_precomp=None)
    return image, radii, depth


def run_evidence(a):
    """Occlusion-aware multi-view LR evidence per Gaussian and its cam00 map (development diagnostic).

    For every training camera c at LR resolution, the total alpha-blending weight W_c(g) = sum_p w_g(p) of each Gaussian is
    the gradient of the channel-sum of a render with constant precomputed colour 1 (exact, includes occlusion). The
    evidence of g is the number of training cameras with W_c(g) >= tau, and its continuous version sum_c min(1, W_c(g)).
    Maps at cam00 (HR) are alpha-normalised renders of these per-Gaussian values."""
    from n3dv_data import observation_to_4dgs_camera
    if a.time_full_range:
        patch_time_full_range()
    check_time_flag(a.checkpoint, a.time_full_range)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    checkpoint = Path(a.checkpoint)
    g, payload = cr.load_for_render(checkpoint, torch.load(checkpoint, map_location='cpu', weights_only=False)['plan'])
    plan = payload['plan']
    del payload
    h, opt, pipe, sh = cr.configuration(plan)
    manifest_path, manifest = cr.plan_scene(plan)
    record = dict(kind='evidence', checkpoint=cr.entry(checkpoint), tau=a.tau, frames=a.frames, posthoc_cull=bool(a.cull),
                  gpu=torch.cuda.get_device_name(), runtime=rt.check_runtime(), sources={cr.rel(p): rt.sha(p) for p in (Path(cr.__file__), Path(__file__))})
    if a.cull:
        record['cull'] = cr.cull(g, cr.Culler(manifest), None, 'posthoc', optimizer_backed=False)
    obs_by = {(o['camera_id'], int(o['frame_index'])): o for o in manifest['observations']}
    zero = torch.zeros(3, dtype=torch.float32, device='cuda')
    train = manifest['splits']['train']
    n = int(g.get_xyz.shape[0])
    stats = {}
    for f in a.frames:
        with torch.no_grad():
            state = native.native_effective_state(g, float(obs_by[('cam00', f)]['time']), torch)
        count = torch.zeros(n, device='cuda')
        soft = torch.zeros(n, device='cuda')
        for i, cam in enumerate(train):
            camera = observation_to_4dgs_camera(ev.camera_item(manifest, obs_by[(cam, f)], torch), i)
            values = torch.ones(n, 3, device='cuda', requires_grad=True)
            image, _, _ = render_values(camera, g, state, values, pipe, zero)
            image[0].sum().backward()
            w = values.grad[:, 0].detach()
            count += (w >= a.tau).float()
            soft += w.clamp(max=1.0)
        hr_cam = native.hr_camera(ev.camera_item(manifest, obs_by[('cam00', f)], torch), 0, manifest)
        with torch.no_grad():
            alpha = render_values(hr_cam, g, state, torch.ones(n, 3, device='cuda'), pipe, zero)[0][0]
            e_count = render_values(hr_cam, g, state, (count / len(train))[:, None].expand(n, 3).contiguous(), pipe, zero)[0][0]
            e_soft = render_values(hr_cam, g, state, (soft / len(train))[:, None].expand(n, 3).contiguous(), pipe, zero)[0][0]
        np.savez_compressed(out / f'evidence_cam00_{f:04d}.npz', alpha=alpha.cpu().numpy().astype(np.float32),
                            count=e_count.cpu().numpy().astype(np.float32), soft=e_soft.cpu().numpy().astype(np.float32),
                            per_gaussian_count=count.cpu().numpy().astype(np.uint8), per_gaussian_soft=soft.cpu().numpy().astype(np.float32))
        opac = state['opacity'].detach().squeeze(-1)
        stats[f] = dict(cameras=len(train), mean_count_opacity_weighted=float((count * opac).sum() / opac.sum()),
                        count_hist=torch.bincount(count.long(), minlength=len(train) + 1).cpu().tolist())
    record.update(stats=stats, seconds=time.time() - started)
    rt.write_json(out / 'evidence.json', record)
    return record


# ----------------------------------------------------------------------------------------------- CLI
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    q = sub.add_parser('lr')
    q.add_argument('--parent', required=True)
    q.add_argument('--fine0')
    q.add_argument('--coarse-only', action='store_true')
    q.add_argument('--stop', type=int)
    s = sub.add_parser('sr')
    s.add_argument('--parent-checkpoint', required=True)
    s.add_argument('--baseline-sr', required=True)
    s.add_argument('--lambda-sr', type=float, default=0.05)
    s.add_argument('--dssim', type=float, default=0.0)
    s.add_argument('--affine-sr-lr', type=float, default=1e-3)
    s.add_argument('--stop', type=int)
    s.add_argument('--teacher-pos-detach', choices=('none', 'all', 'late'), default='none')
    s.add_argument('--detach-start', type=float, default=0.75)
    s.add_argument('--xyz-rewarm', type=float, default=0.0, help='peak extra xyz lr / spatial_lr_scale (author init is 1.6e-4)')
    s.add_argument('--rewarm-warmup', type=int, default=200)
    s.add_argument('--sh-rest-lr-mult', type=float, default=1.0)
    s.add_argument('--holdout-cameras', nargs='*')
    s.add_argument('--holdout-refill', action='store_true', help='cycle the filtered rows back to the full step count')
    for x in (q, s):
        x.add_argument('--out', required=True)
        x.add_argument('--arm', required=True)
        x.add_argument('--cull', action='store_true')
        x.add_argument('--affine', action='store_true')
        x.add_argument('--prune-gate', choices=('author', 'none'), default='author')
        x.add_argument('--cull-interval', type=int, default=1000)
        x.add_argument('--cpu-threads', type=int, default=4)
        x.add_argument('--time-full-range', action='store_true')
    e = sub.add_parser('eval')
    e.add_argument('--checkpoint', required=True)
    e.add_argument('--plan')
    e.add_argument('--out', required=True)
    e.add_argument('--label', required=True)
    e.add_argument('--cull', action='store_true')
    e.add_argument('--frames', type=int, nargs='*')
    e.add_argument('--cpu-threads', type=int, default=4)
    e.add_argument('--time-full-range', action='store_true')
    x = sub.add_parser('extras')
    x.add_argument('--checkpoint', required=True)
    x.add_argument('--out', required=True)
    x.add_argument('--baseline-sr', help='registered Bsync endpoint (teacher index) for holdout teacher agreement')
    x.add_argument('--cull', action='store_true')
    x.add_argument('--holdout-cameras', nargs='*')
    x.add_argument('--holdout-stride', type=int, default=10)
    x.add_argument('--qc-frames', type=int, nargs='*', default=[15, 150, 285])
    x.add_argument('--diagnostics', action='store_true')
    x.add_argument('--region-stride', type=int, default=10)
    x.add_argument('--time-full-range', action='store_true')
    v = sub.add_parser('evidence')
    v.add_argument('--checkpoint', required=True)
    v.add_argument('--out', required=True)
    v.add_argument('--frames', type=int, nargs='*', default=[0, 150, 299])
    v.add_argument('--tau', type=float, default=0.25, help='minimum total blending weight (LR pixels) for a camera to count')
    v.add_argument('--cull', action='store_true')
    v.add_argument('--time-full-range', action='store_true')
    a = p.parse_args(argv)
    result = dict(lr=run_lr, sr=run_sr, eval=run_eval, extras=run_extras, evidence=run_evidence)[a.cmd](a)
    print(json.dumps({k: v for k, v in result.items() if k in ('status', 'seconds', 'points', 'test', 'holdout', 'train_diagnostic')}, default=str)[:4000])


if __name__ == '__main__':
    main()
