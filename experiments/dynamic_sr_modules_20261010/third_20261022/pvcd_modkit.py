"""PVCD = pseudo-view consensus distillation on the w4 recipe (Claude 4dsr-46, 2026-10-23; joint with 4dsr-e3, pre-registration
tmp/claude_third_f3/prereg_pvcd.md). Port of experiments/dynamic_sr_cull_retrain_20261009/variance_20261014/consistency_distill.py
(4dsr-ea, 10-14/15, S1-③ / S1-③b) from `srfix.py --mode fix <entry> sr` to the modkit4.py chain that run_modkit4_remote.py uses for
the w4 arm (area operator, matched teacher, 4x4 warp field). The consistency term, its defaults and the pseudo-view sampler are the
original ones (PseudoViews / quat / slerp / rot are imported from consistency_distill.py, not copied):
    L_c = w * [(1 - d) * mean|R_s - mean_k R_k| + d * (1 - SSIM(R_s, mean_k R_k))],   w = 0.5, d = 0.2;
    --scale s > 1: student render and consensus are first reduced by the LR term's own degradation (losses.degradation, which
    modkit.py sets to calibsr_v2.area_degradation for --lr-operator area, i.e. A4 for s = 4) to (H/s, W/s).
One pseudo-view per SR iteration, in addition to the registered batch views (it does not replace any): training camera i, one of its
two nearest training cameras j, centre lerp / rotation slerp with alpha ~ U(0.25, 0.75), intrinsics of i, time uniform over the
training frame times, own numpy Generator (--seed). Only training-camera poses are read; no test pose, test image or HR image.

Changes against consistency_distill.py, all forced by the modkit chain:
- The term is attached to GaussianModel.compute_regulation (called once per SR iteration by cullretrain's SR loop; checked at the
  end: calls == SR iterations is recorded) BEFORE modkit4.main() installs its own wrappers, and the pseudo-view renders use the
  ORIGINAL native.hr_camera / native.native_rgb / native.native_effective_state captured at import time. So the pseudo-views pass through none of the modkit
  wrappers: no per-camera time offset (modkit2 tags cameras with mk_camera_id), no DropGaussian, no screen-space point tracking
  (modkit install_tracking), no wrapped deformation state, and no training camera model (warp field / colour affine): student and sources are both rendered
  natively (identity camera model), as cam00 is evaluated. The pseudo-view render does not enter the densification statistics.
- Sources may carry per-frame residuals: --teacher LABEL=CKPT[=FITDIR] applies FITDIR/fNNNN.npz of the pseudo-view's frame with
  perframe_20261021/eval_resid.apply (the evaluator's own code; FITDIR/run.json must name the source checkpoint sha256 and its
  post-cull point count). Without =FITDIR the source is its SR checkpoint (pre-P), as in consistency_distill.py.
Sources are loaded with the evaluator's post-hoc training-camera cull (ens_eval.load_model(..., cull=True)) and frozen.

usage: pvcd_modkit.py --teacher L=ckpt[=fit] [--teacher ...] [--weight 0.5] [--dssim 0.2] [--scale 4] [--seed 20261014]
                      [--log-every 50] -- <modkit4.py arguments exactly as run_modkit4_remote.py passes them, including its own -->
       pvcd_modkit.py --dry-run --manifest rel/manifest.json     (CPU: 8 pseudo-views, as consistency_distill.py --dry-run)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
VAR = ROOT / 'experiments/dynamic_sr_cull_retrain_20261009/variance_20261014'
GEO = ROOT / 'experiments/dynamic_sr_modules_20261010/geoprobe_20261017'
PERF = ROOT / 'experiments/dynamic_sr_modules_20261010/perframe_20261021'
for _p in (VAR, GEO, PERF):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import consistency_distill as cd  # noqa: E402  (PseudoViews, dry_run; sets the registered runtime paths)


def main():
    argv = sys.argv[1:]
    if '--dry-run' in argv:
        p = argparse.ArgumentParser(prog='pvcd_modkit.py')
        p.add_argument('--dry-run', action='store_true')
        p.add_argument('--manifest', required=True)
        return cd.dry_run(p.parse_args(argv).manifest)
    if '--' not in argv:
        raise SystemExit(__doc__)
    cut = argv.index('--')
    own, kit = argv[:cut], argv[cut + 1:]
    p = argparse.ArgumentParser(prog='pvcd_modkit.py')
    p.add_argument('--teacher', action='append', required=True)
    p.add_argument('--weight', type=float, default=0.5)
    p.add_argument('--dssim', type=float, default=0.2)
    p.add_argument('--scale', type=int, default=4, choices=(1, 2, 4))
    p.add_argument('--seed', type=int, default=20261014)
    p.add_argument('--log-every', type=int, default=50)
    a = p.parse_args(own)
    if '--' not in kit:
        raise SystemExit('the modkit4 arguments must contain their own -- before the srfix arguments')
    sr_args = kit[kit.index('--') + 1:]
    if sr_args[:4] != ['--mode', 'fix', 'cullretrain', 'sr']:
        raise SystemExit('expected the w4 chain: ... -- --mode fix cullretrain sr <SR args>')

    import rt  # noqa: F401  (pinned runtime and registered experiment code on sys.path)
    import srfix  # imports cullretrain (unmodified) and torch
    import torch
    import ens_eval
    import full_evaluate_registered_v2 as ev
    import eval_resid
    from scene.gaussian_model import GaussianModel
    from utils.loss_utils import ssim
    cr = srfix.cr
    native = cr.native
    losses = sys.modules['losses']
    hr_camera0, native_rgb0, nes0 = native.hr_camera, native.native_rgb, native.native_effective_state   # captured before modkit installs its wrappers

    out = Path(sr_args[sr_args.index('--out') + 1])
    out = out if out.is_absolute() else ROOT / out
    if (out / 'complete.json').exists():
        raise FileExistsError('Completed run preserved: ' + str(out))
    out.mkdir(parents=True, exist_ok=True)
    parent = ROOT / sr_args[sr_args.index('--parent-checkpoint') + 1]
    plan = torch.load(parent, map_location='cpu', weights_only=False)['plan']
    h, _, pipe, _ = cr.configuration(plan)
    if h.time_smoothness_weight == 0:
        raise SystemExit('compute_regulation is not called when time_smoothness_weight == 0; the hook would never run')
    _, manifest = cr.plan_scene(plan)
    frame_of = {}
    for o in manifest['observations']:
        if o['split'] == 'train':
            frame_of.setdefault(float(o['time']), int(o['frame_index']))

    teachers = []
    for spec in a.teacher:
        parts = spec.split('=')
        if len(parts) not in (2, 3):
            raise SystemExit('bad --teacher ' + spec)
        m = ens_eval.load_model(parts[0], parts[1], True)
        m['g']._deformation.requires_grad_(False)
        m['resid'] = None
        if len(parts) == 3:
            d = ROOT / parts[2]
            meta = json.loads(((d / 'run.json') if (d / 'run.json').exists() else (d / 'meta.json')).read_text())
            sha = eval_resid.meta_value(meta, ('checkpoint_sha256', 'parent_sha256', 'parent.sha256', 'checkpoint.sha256'))
            pts = eval_resid.meta_value(meta, ('points_after_cull', 'cull.points_after'))
            if sha != m['info']['checkpoint']['sha256'] or pts is None or int(pts) != m['info']['points_evaluated']:
                raise SystemExit(f'{parts[0]}: residuals {d} do not belong to {parts[1]} after the cull')
            m['resid'] = dict(dir=d, cache={}, applied=0)
            m['info']['residuals'] = str(d.relative_to(ROOT))
        teachers.append(m)
    student_test = [o['hr_sha256'] for o in sorted((o for o in manifest['observations'] if o['split'] == 'test'), key=lambda o: int(o['frame_index']))]
    for t in teachers:
        if [o['hr_sha256'] for o in t['test']] != student_test:
            raise ValueError('source scene (cam00 HR identity) differs from the student manifest: ' + t['label'])
    background = torch.ones(3, dtype=torch.float32, device='cuda') if cr.white_background(plan) else torch.zeros(3, dtype=torch.float32, device='cuda')

    class Native0:   # the sampler builds its camera with the unwrapped hr_camera (no mk_camera_id tag)
        hr_camera = staticmethod(hr_camera0)

    def source_state(m, t):
        st = nes0(m['g'], t, torch)
        r = m['resid']
        if r is None:
            return st
        f = frame_of[float(t)]
        z = r['cache'].get(f)
        if z is None:
            path = r['dir'] / f'f{f:04d}.npz'
            if not path.exists():
                raise SystemExit(f'{m["label"]}: no residuals for frame {f}')
            with np.load(path) as zz:
                z = {k: zz[k] for k in zz.files}
            r['cache'][f] = z
        r['applied'] += 1
        return eval_resid.apply(st, z)

    views = cd.PseudoViews(manifest, a.seed)
    state = dict(calls=0, values=[], started=time.time())
    log = (out / 'pvcd.jsonl').open('a', buffering=1)
    original = GaussianModel.compute_regulation

    def compute_regulation(self, *args, **kwargs):
        reg = original(self, *args, **kwargs)
        camera, t, info = views.sample(ev, Native0, torch)
        with torch.no_grad():
            target = torch.stack([native_rgb0(camera, m['g'], source_state(m, t), m['pipe'], m['background'])['render']
                                  for m in teachers]).clamp(0, 1).mean(0)
        render = native_rgb0(camera, self, nes0(self, t, torch), pipe, background)['render']
        if a.scale > 1:
            size = (int(render.shape[-2]) // a.scale, int(render.shape[-1]) // a.scale)
            render, target = losses.degradation(render, size), losses.degradation(target, size)
        l1 = (render - target).abs().mean()
        value = (1 - a.dssim) * l1 + a.dssim * (1 - ssim(render[None], target[None])) if a.dssim else l1
        state['calls'] += 1
        state['values'].append(float(value.detach()))
        if state['calls'] % a.log_every == 0:
            log.write(json.dumps(dict(call=state['calls'], consistency=float(np.mean(state['values'][-a.log_every:])), last=info,
                                      seconds=time.time() - state['started'])) + '\n')
        return reg + a.weight * value

    GaussianModel.compute_regulation = compute_regulation
    record = dict(kind='pvcd_modkit', K=len(teachers), weight=a.weight, dssim=a.dssim, scale=a.scale, seed=a.seed, modkit4_argv=kit,
                  teachers=[t['info'] for t in teachers], pseudo_views=dict(cameras=views.cams, neighbours=views.neighbours,
                  alpha=[0.25, 0.75], times=len(views.times)),
                  source={str(Path(f).resolve().relative_to(ROOT)): rt.sha(Path(f)) for f in (__file__, cd.__file__, ens_eval.__file__, eval_resid.__file__)})
    rt.write_json(out / 'pvcd.json', record)
    import modkit4
    sys.argv = [str(GEO / 'modkit4.py')] + kit
    try:
        modkit4.main()
    finally:
        GaussianModel.compute_regulation = original
        log.close()
    done = json.loads((out / 'complete.json').read_text()) if (out / 'complete.json').exists() else {}
    record.update(calls=state['calls'], sr_steps=cr.SR_STEPS, calls_equal_steps=state['calls'] == cr.SR_STEPS,
                  consistency_first_100=float(np.mean(state['values'][:100])) if state['values'] else None,
                  consistency_last_100=float(np.mean(state['values'][-100:])) if state['values'] else None,
                  residual_frames_applied={t['label']: (t['resid'] or {}).get('applied') for t in teachers},
                  complete=bool(done))
    rt.write_json(out / 'pvcd.json', record)


if __name__ == '__main__':
    main()
