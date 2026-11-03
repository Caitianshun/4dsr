"""Renders for the descriptive intervention-response matrix A1 (Claude 4dsr-4e, module audit 2026-11-01; statistics by 4dsr-4f's
tmp/claude_indep_moduleaudit_20261101/scripts/a1_decomp.py). Render only: no training, no HR, no cam00 image is read.

Model loading, cull, deformed state, residuals and rendering are those of area4_baselines_20261024/ours/render_ours_png_cam.py:
ens_eval.load_model(..., cull=True) -> native.native_effective_state(t) -> eval_resid.apply(DIR/f####.npz) -> native.hr_camera ->
native.native_rgb at HR (TF32 off). Camera model = the checkpoint's own per-training-camera warp field + colour affine, rebuilt with
modkit2.make_ext_affine over cullretrain.CameraAffine (absorb_bound2.camera_model); a checkpoint without a warp field (the
no-field base) gets its colour affine only. A4 = 4x4 average pooling of the unclamped HR render (the area operator of the LR term).

  --mode train   for every training camera and frame in --frames: A4(M_c(R)) clamped to [0, 1], float16, H/4 x W/4 x 3, written to
                 OUT/<camera>/<frame:04d>.npy, plus OUT/index.json (checkpoint, residuals, camera model summary, frames).
  --mode pair    two or more runs (--run LABEL=CK[=RESID]); at the cam00 pose (identity camera model; the pose only, no image) and at
                 every training camera (each run through its own camera model), frames in --frames: per pair (i, j) the pixel mean of
                 (R_i - R_j)^2 / 2 at HR and after A4, per view and pooled; written to OUT/pairs.json. Nothing else is stored.
usage: render_a1.py --mode train --checkpoint CK [--resid DIR] --out OUT [--frames 0:300:10]
       render_a1.py --mode pair --run A=CK[=RESID] --run B=CK[=RESID] [...] --out OUT [--frames 0:300:10]
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
MOD = HERE.parent
for _p in (MOD / 'geoprobe_20261017', MOD / 'perframe_20261021'):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import eval_full  # noqa: E402  (runtime paths, torch, cullretrain, ens_eval, ev, native)
import eval_resid  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

ROOT = eval_full.ROOT
cr = eval_full.ens_eval.cr if hasattr(eval_full.ens_eval, 'cr') else sys.modules['cullretrain']


def camera_model(raw):
    src = (MOD / 'geoprobe_20261017' / 'modkit2.py').read_text()
    ns = dict(math=math, json=json, Path=Path, REG={})
    exec(src[src.index('def make_ext_affine('):src.index('def manifest_dt(')], ns)
    grid = int(raw['wfield'].shape[-1]) if raw.get('wfield') is not None else 0
    m = ns['make_ext_affine'](cr.CameraAffine, None, grid, 2e-3, 0.0)(list(raw['names']), device='cuda')
    with torch.no_grad():
        if grid:
            m.wfield.copy_(raw['wfield'].to(device='cuda', dtype=torch.float32))
        m.delta.copy_(raw['delta'].to(device='cuda', dtype=torch.float32))
    m.wfield.requires_grad_(False)
    m.delta.requires_grad_(False)
    return m, grid


class Run:
    def __init__(self, label, ck, resid):
        self.label = label
        self.m = eval_full.ens_eval.load_model(label, ck, True)
        raw = torch.load(ROOT / ck, map_location='cpu', weights_only=False).get('affine')
        if raw is None:
            raise SystemExit(f'{label}: checkpoint has no camera model')
        self.cm, self.grid = camera_model(raw)
        if list(self.cm.names) != list(self.m['manifest']['splits']['train']):
            raise SystemExit(f'{label}: camera model cameras differ from the training split')
        self.resid = None
        if resid:
            d = ROOT / resid
            meta = json.loads(((d / 'run.json') if (d / 'run.json').exists() else (d / 'meta.json')).read_text())
            rsha = eval_resid.meta_value(meta, ('checkpoint_sha256', 'parent_sha256', 'parent.sha256', 'checkpoint.sha256'))
            pts = eval_resid.meta_value(meta, ('points_after_cull', 'cull.points_after'))
            if rsha != self.m['info']['checkpoint']['sha256'] or pts is None or int(pts) != self.m['info']['points_evaluated']:
                raise SystemExit(f'{label}: residuals {d} do not belong to {ck} after the cull')
            self.resid = d
        self.info = dict(label=label, checkpoint=self.m['info']['checkpoint'], residuals=resid, warp_grid=self.grid,
                         points_evaluated=self.m['info']['points_evaluated'])

    def render(self, obs, frame, camera_model=True):
        man = self.m['manifest']
        cam = eval_full.native.hr_camera(eval_full.ev.camera_item(man, obs, torch), frame, man)
        st = eval_full.native.native_effective_state(self.m['g'], float(obs['time']), torch)
        if self.resid is not None:
            with np.load(self.resid / f'f{frame:04d}.npz') as z:
                st = eval_resid.apply(st, {k: z[k] for k in z.files})
        r = eval_full.native.native_rgb(cam, self.m['g'], st, self.m['pipe'], self.m['background'])['render']
        if camera_model and obs['split'] == 'train':
            r = self.cm.apply(r, obs['camera_id'])
        return r


def a4(x):
    return F.avg_pool2d(x[None], 4)[0]


def frames_arg(s):
    a, b, c = (int(v) for v in s.split(':'))
    return list(range(a, b, c))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--mode', choices=('train', 'pair'), required=True)
    p.add_argument('--checkpoint')
    p.add_argument('--resid')
    p.add_argument('--run', action='append', default=[])
    p.add_argument('--out', required=True)
    p.add_argument('--frames', default='0:300:10')
    a = p.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    out = ROOT / a.out
    if (out / 'index.json').exists() or (out / 'pairs.json').exists():
        raise FileExistsError('completed output preserved: ' + str(out))
    out.mkdir(parents=True, exist_ok=True)
    frames = frames_arg(a.frames)
    t0 = time.time()
    if a.mode == 'train':
        run = Run('model', a.checkpoint, a.resid)
        man = run.m['manifest']
        by = {(o['camera_id'], int(o['frame_index'])): o for o in man['observations'] if o['split'] == 'train'}
        n = 0
        with torch.no_grad():
            for c in man['splits']['train']:
                (out / c).mkdir(exist_ok=True)
                for f in frames:
                    o = by.get((c, f))
                    if o is None:
                        continue
                    lr = a4(run.render(o, f)).clamp(0, 1).permute(1, 2, 0).cpu().numpy().astype(np.float16)
                    np.save(out / c / f'{f:04d}.npy', lr)
                    n += 1
        (out / 'index.json').write_text(json.dumps(dict(run=run.info, frames=frames, views=n, cameras=list(man['splits']['train']),
                                                        layout='<camera>/<frame:04d>.npy float16 HxWx3 = A4(camera model(render)), clamp',
                                                        seconds=time.time() - t0, gpu=torch.cuda.get_device_name()), indent=1))
        print(json.dumps(dict(views=n, seconds=time.time() - t0)), flush=True)
        return
    runs = []
    for spec in a.run:
        parts = spec.split('=')
        runs.append(Run(parts[0], parts[1], parts[2] if len(parts) == 3 else None))
    if len(runs) < 2:
        raise SystemExit('pair mode needs two or more --run')
    man = runs[0].m['manifest']
    test = {int(o['frame_index']): o for o in man['observations'] if o['split'] == 'test'}
    train = {(o['camera_id'], int(o['frame_index'])): o for o in man['observations'] if o['split'] == 'train'}
    pairs = list(itertools.combinations(range(len(runs)), 2))
    acc = {f'{runs[i].label}-{runs[j].label}': dict(test_hr=[], test_a4=[], train_hr=[], train_a4=[]) for i, j in pairs}
    with torch.no_grad():
        for f in frames:
            views = [('test', test[f])] if f in test else []
            views += [('train', train[(c, f)]) for c in man['splits']['train'] if (c, f) in train]
            for kind, o in views:
                rs = [r.render(o, f).clamp(0, 1) for r in runs]
                for i, j in pairs:
                    d = (rs[i] - rs[j]).pow(2) / 2
                    k = f'{runs[i].label}-{runs[j].label}'
                    acc[k][f'{kind}_hr'].append(float(d.mean()))
                    acc[k][f'{kind}_a4'].append(float((a4(rs[i]) - a4(rs[j])).pow(2).mean() / 2))
    res = {k: {kk: dict(mean=float(np.mean(v)) if v else None, n=len(v)) for kk, v in d.items()} for k, d in acc.items()}
    (out / 'pairs.json').write_text(json.dumps(dict(runs=[r.info for r in runs], frames=frames, pairs=res,
                                                    definition='pixel mean of (R_i - R_j)^2 / 2, clamp [0,1]; test = cam00 pose with identity '
                                                               'camera model (no image read); train = each run through its own camera model',
                                                    seconds=time.time() - t0, gpu=torch.cuda.get_device_name()), indent=1))
    print(json.dumps(res), flush=True)


if __name__ == '__main__':
    main()
