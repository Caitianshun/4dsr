"""S1-③ multi-model consistency distillation (Claude 4dsr-ea, 2026-10-14; docs/compressed_deep_research_2026-10-14.md 5.1).

The registered SR stage (srfix.py --mode fix <csc|cullretrain|modsr> sr ...) runs unchanged, plus one term per SR iteration:
the student renders a pseudo-view and is pulled towards the per-pixel mean of K >= 2 frozen, already trained SR models
rendered at the same pseudo-view and time (stop-gradient):
    L_c = w * [(1 - d) * mean|R_s - mean_k R_k| + d * (1 - SSIM(R_s, mean_k R_k))],   w = 0.5, d = 0.2 (fixed in advance).
Pseudo-view: a training camera i and one of its two nearest training cameras j (camera centres); centre lerp and rotation
slerp with alpha ~ U(0.25, 0.75); intrinsics of camera i at HR; time drawn uniformly from the training frame times. The
sampler has its own numpy Generator, so the registered torch/numpy RNG streams of the SR stage are not consumed. No test or
dev camera pose, test image or HR image is read: the pseudo-view is built from training-camera poses only.

--scale s (2026-10-15, S1-③b; README): s = 1 is the form above (the first test, run by 4dsr-78, used it). For s > 1 both the
student render and the teacher mean are first reduced by the LR term's own degradation (bicubic, antialias, losses.degradation)
to (H/s, W/s), so the term only constrains the band below 1/(2s) cycles per HR pixel (s = 4: the LR band; s = 2: LR band plus
the evaluator's mid band) and leaves the finer detail to the registered teacher term.

The term is attached by wrapping GaussianModel.compute_regulation, which every registered SR loop (cullretrain.py, csc.py,
modsr.py) calls exactly once per iteration (time_smoothness_weight = 0.001 in the dynerf configuration; checked at start).
cullretrain.py, csc.py, modsr.py and srfix.py are imported unmodified (pattern of biasfit.py). Teachers are rendered raw
(no camera affine), as the evaluator renders test views, after the evaluator's post-hoc training-camera culling. The
pseudo-view render does not enter the densification statistics (those come from the three registered batch views).

usage: consistency_distill.py --teacher LABEL=rel/ckpt.pt [--teacher LABEL=rel/ckpt.pt ...] [--weight 0.5] [--dssim 0.2] [--scale 1]
           [--seed 20261014] [--log-every 50] -- <csc|cullretrain|modsr> sr <that entry's SR arguments>
       consistency_distill.py --dry-run --manifest rel/manifest.json   (CPU: build and summarise 8 pseudo-views)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
MOD = ROOT / 'experiments/dynamic_sr_modules_20261010'
for _p in (HERE.parent, MOD, HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

ENTRIES = ('csc', 'cullretrain', 'modsr')


def quat(R):
    """Rotation matrix -> unit quaternion (w, x, y, z)."""
    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = np.asarray(q, dtype=np.float64)
    return q / np.linalg.norm(q)


def rot(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def slerp(q0, q1, a):
    d = float(np.dot(q0, q1))
    if d < 0:
        q1, d = -q1, -d
    if d > 0.9995:
        q = q0 + a * (q1 - q0)
        return q / np.linalg.norm(q)
    th = math.acos(d)
    return (math.sin((1 - a) * th) * q0 + math.sin(a * th) * q1) / math.sin(th)


class PseudoViews:
    """Pseudo-views between neighbouring training cameras (training poses only)."""

    def __init__(self, manifest, seed):
        self.manifest = manifest
        self.rng = np.random.default_rng(seed)
        self.cams = list(manifest['splits']['train'])
        train_obs = [o for o in manifest['observations'] if o['split'] == 'train']
        if {o['camera_id'] for o in train_obs} != set(self.cams):
            raise ValueError('training observations and the training split disagree')
        self.first = {}
        for o in train_obs:
            self.first.setdefault(o['camera_id'], o)
        self.times = sorted({float(o['time']) for o in train_obs})
        self.w2c = {c: np.asarray(manifest['cameras'][c]['w2c'], dtype=np.float64) for c in self.cams}
        self.center = {c: -self.w2c[c][:3, :3].T @ self.w2c[c][:3, 3] for c in self.cams}
        self.neighbours = {}
        for c in self.cams:
            d = sorted((float(np.linalg.norm(self.center[c] - self.center[o])), o) for o in self.cams if o != c)
            self.neighbours[c] = [o for _, o in d[:2]]

    def sample(self, ev, native, torch):
        i = self.cams[int(self.rng.integers(len(self.cams)))]
        j = self.neighbours[i][int(self.rng.integers(2))]
        a = float(self.rng.uniform(0.25, 0.75))
        t = self.times[int(self.rng.integers(len(self.times)))]
        R = rot(slerp(quat(self.w2c[i][:3, :3]), quat(self.w2c[j][:3, :3]), a))
        c = (1 - a) * self.center[i] + a * self.center[j]
        w2c = np.eye(4)
        w2c[:3, :3], w2c[:3, 3] = R, -R @ c
        obs = ev.camera_item(self.manifest, self.first[i], torch)  # camera i's intrinsics and LR size (payload only)
        obs.update(w2c=w2c, time=t)
        camera = native.hr_camera(obs, 0, self.manifest)
        return camera, t, dict(i=i, j=j, alpha=a, time=t, center=c.tolist())


def dry_run(manifest_rel):
    import rt  # noqa: F401  (pinned runtime and registered experiment code on sys.path)
    import torch
    import full_evaluate_registered_v2 as ev
    import full_refine_registered_v2 as native
    from n3dv_data import load_manifest
    manifest = load_manifest(ROOT / manifest_rel)
    pv = PseudoViews(manifest, 0)
    rows = []
    for _ in range(8):
        cam, t, info = pv.sample(ev, native, torch)
        ci, cj = pv.center[info['i']], pv.center[info['j']]
        rows.append(dict(info, hr=[int(cam.image_width), int(cam.image_height)], dist_i=float(np.linalg.norm(np.array(info['center']) - ci)),
                         dist_ij=float(np.linalg.norm(ci - cj))))
    print(json.dumps(dict(cameras=pv.cams, neighbours=pv.neighbours, times=[pv.times[0], pv.times[-1], len(pv.times)], samples=rows), indent=1))


def main():
    p = argparse.ArgumentParser(prog='consistency_distill.py')
    p.add_argument('--teacher', action='append', default=[])
    p.add_argument('--weight', type=float, default=0.5)
    p.add_argument('--dssim', type=float, default=0.2)
    p.add_argument('--scale', type=int, default=1, choices=(1, 2, 4))
    p.add_argument('--seed', type=int, default=20261014)
    p.add_argument('--log-every', type=int, default=50)
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--manifest')
    p.add_argument('rest', nargs=argparse.REMAINDER)
    a = p.parse_args()
    if a.dry_run:
        return dry_run(a.manifest)
    rest = a.rest[1:] if a.rest[:1] == ['--'] else a.rest
    if len(rest) < 2 or rest[0] not in ENTRIES or rest[1] != 'sr':
        raise SystemExit('expected: consistency_distill.py --teacher L=ckpt --teacher L=ckpt [...] -- <csc|cullretrain|modsr> sr <args>')
    if len(a.teacher) < 1:
        raise SystemExit('need at least one --teacher LABEL=checkpoint (K = 1 is the single-teacher ablation)')

    import rt  # noqa: F401  (pinned runtime and registered experiment code on sys.path)
    import srfix  # imports cullretrain (unmodified) and torch
    import cullretrain as cr
    import torch
    import ens_eval
    import full_evaluate_registered_v2 as ev
    import full_refine_registered_v2 as native
    import losses
    from scene.gaussian_model import GaussianModel
    from utils.loss_utils import ssim

    sr_args = rest[2:]
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
    teachers = [ens_eval.load_model(lab, path, True) for lab, path in (t.split('=', 1) for t in a.teacher)]
    student_test = [o['hr_sha256'] for o in sorted((o for o in manifest['observations'] if o['split'] == 'test'), key=lambda o: int(o['frame_index']))]
    for t in teachers:
        if [o['hr_sha256'] for o in t['test']] != student_test:
            raise ValueError('teacher scene (cam00 HR identity) differs from the student manifest: ' + t['label'])
        t['g']._deformation.requires_grad_(False)
    background = torch.ones(3, dtype=torch.float32, device='cuda') if cr.white_background(plan) else torch.zeros(3, dtype=torch.float32, device='cuda')
    views = PseudoViews(manifest, a.seed)
    state = dict(calls=0, values=[], started=time.time())
    log = (out / 'consistency.jsonl').open('a', buffering=1)
    original = GaussianModel.compute_regulation

    def compute_regulation(self, *args, **kwargs):
        reg = original(self, *args, **kwargs)
        camera, t, info = views.sample(ev, native, torch)
        with torch.no_grad():
            target = torch.stack([native.native_rgb(camera, m['g'], native.native_effective_state(m['g'], t, torch), m['pipe'], m['background'])['render']
                                  for m in teachers]).clamp(0, 1).mean(0)
        render = native.native_rgb(camera, self, native.native_effective_state(self, t, torch), pipe, background)['render']
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
    record = dict(kind='consistency_distill', entry=rest[0], K=len(teachers), weight=a.weight, dssim=a.dssim, scale=a.scale, seed=a.seed, argv=rest,
                  teachers=[t['info'] for t in teachers], pseudo_views=dict(cameras=views.cams, neighbours=views.neighbours,
                  alpha=[0.25, 0.75], times=len(views.times)),
                  source={str(Path(f).resolve().relative_to(ROOT)): rt.sha(Path(f)) for f in (__file__, ens_eval.__file__, losses.__file__)})
    rt.write_json(out / 'consistency.json', record)
    sys.argv = ['srfix.py', '--mode', 'fix', *rest]
    try:
        srfix.main()
    finally:
        GaussianModel.compute_regulation = original
        log.close()
    record.update(calls=state['calls'], consistency_first_100=float(np.mean(state['values'][:100])) if state['values'] else None,
                  consistency_last_100=float(np.mean(state['values'][-100:])) if state['values'] else None)
    rt.write_json(out / 'consistency.json', record)


if __name__ == '__main__':
    main()
