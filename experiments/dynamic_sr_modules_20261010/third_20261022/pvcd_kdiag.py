"""PVCD zero-training diagnostics for the source count K and the pose question (Claude 4dsr-33, 2026-10-26;
pre-registration tmp/claude_pvcd_k45_20261026/prereg_k45.md). Nothing is trained, fitted or selected here.

Loads N >= 3 frozen SR-stage checkpoints of one scene (independent runs of the same recipe) with the evaluator's post-hoc
training-camera cull, exactly as PVCD loads its sources (ens_eval.load_model(..., cull=True)), and renders them natively
(identity camera model), as PVCD's consistency term and the cam00 evaluator do.

(1) K-curve at cam00 (300 frames, or every --cam00-every-th): error of the per-pixel mean of every non-empty subset of the N
    renders against HR, at full resolution and after the area x4 operator A4 (4x4 box mean = the band PVCD s4 constrains;
    A4(HR) is the clean LR of cam00). The error splits exactly into an A4 part and a sub-block detail part.
    Reported per subset size K: mean / min / max over subsets, and along the nested order of the --checkpoint arguments.
(2) Run disagreement by pose type: per-pixel population variance across the N renders, averaged over pixels and channels, at
    full resolution, after A4, and for the sub-block detail (full = A4 part + detail part, exactly), at
    (a) training cameras (every training camera at --train-frames), (b) PVCD pseudo-views (consistency_distill.PseudoViews with
    --pv-seed, --pv-samples draws: the sampler PVCD uses), (c) the cam00 frames of (1).

usage: pvcd_kdiag.py --checkpoint L=ckpt [--checkpoint ...] --out rel/dir [--cam00-every 1] [--train-frames 0 30 ...]
                     [--pv-samples 400] [--pv-seed 20261014]
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import socket
import sys
import time
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
VAR = ROOT / 'experiments/dynamic_sr_cull_retrain_20261009/variance_20261014'
for _p in (VAR,):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import consistency_distill_view as cd  # noqa: E402  (PseudoViews; same code as consistency_distill.py, own ens_eval_view import)
import ens_eval_view as ens_eval  # noqa: E402  (same bytes as ens_eval.py b3c67fa3; sets the registered runtime paths)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', action='append', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--cam00-every', type=int, default=1)
    p.add_argument('--train-frames', type=int, nargs='*', default=list(range(0, 300, 30)))
    p.add_argument('--pv-samples', type=int, default=400)
    p.add_argument('--pv-seed', type=int, default=20261014)
    a = p.parse_args()

    import rt
    import torch
    import torch.nn.functional as F
    import full_evaluate_registered_v2 as ev
    import full_refine_registered_v2 as native

    out = ROOT / a.out
    if (out / 'summary.json').exists():
        raise FileExistsError('Completed diagnostic preserved: ' + str(out))
    out.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    started = time.time()
    specs = [c.split('=', 1) for c in a.checkpoint]
    labels = [s[0] for s in specs]
    if len(set(labels)) != len(labels) or len(labels) < 3:
        raise ValueError('need >= 3 distinct labels')
    record = dict(kind='pvcd_kdiag', args=vars(a), host=socket.gethostname(), gpu=torch.cuda.get_device_name(),
                  runtime=rt.check_runtime(), sources={str(Path(f).resolve().relative_to(ROOT)): rt.sha(Path(f)) for f in
                                                       (__file__, cd.__file__, ens_eval.__file__)}, started_unix=started)
    models = [ens_eval.load_model(lab, path, True) for lab, path in specs]
    record['models'] = [m['info'] for m in models]
    ref = models[0]
    for m in models[1:]:
        if [o['hr_sha256'] for o in m['test']] != [o['hr_sha256'] for o in ref['test']]:
            raise ValueError('cam00 HR identity differs between manifests: ' + m['label'])
    manifest = ref['manifest']
    (out / 'run.json').write_text(json.dumps(record, indent=1))
    legacy = ens_eval.legacy_helpers()
    root = Path(manifest['_root'])

    def render(m, camera, t):
        st = native.native_effective_state(m['g'], float(t), torch)
        return native.native_rgb(camera, m['g'], st, m['pipe'], m['background'])['render'].detach().clamp(0, 1)

    def a4(x):
        return F.avg_pool2d(x, 4)

    def disagreement(X):
        """X: (N, 3, H, W). Population variance across runs, mean over pixels/channels: full, A4 part, detail part."""
        full = float(X.var(dim=0, unbiased=False).mean())
        low = float(a4(X).var(dim=0, unbiased=False).mean())
        return full, low, full - low

    subsets = [c for k in range(1, len(labels) + 1) for c in itertools.combinations(range(len(labels)), k)]
    nested = [tuple(range(k)) for k in range(1, len(labels) + 1)]
    name = lambda c: '+'.join(labels[i] for i in c)
    rows_k, rows_d = [], []

    with torch.no_grad():
        # (1) + (2c): cam00
        frames = [o for o in ref['test'] if int(o['frame_index']) % a.cam00_every == 0]
        for n, obs in enumerate(frames):
            index = int(obs['frame_index'])
            hr = torch.from_numpy(legacy.read_rgb(root / obs['hr_path'])).permute(2, 0, 1).cuda()
            X = torch.stack([render(m, native.hr_camera(ev.camera_item(m['manifest'], m['test'][index], torch), index, m['manifest']),
                                    m['test'][index]['time']) for m in models])
            if X.shape[1:] != hr.shape:
                raise ValueError(f'render {tuple(X.shape[1:])} vs HR {tuple(hr.shape)}')
            hr4 = a4(hr[None])[0]
            for c in subsets:
                e = X[list(c)].mean(0) - hr
                mse = float((e * e).mean())
                mse4 = float((a4(e[None])[0] ** 2).mean())
                rows_k.append(dict(frame=index, combo=name(c), k=len(c), mse=mse, mse_a4=mse4, mse_detail=mse - mse4))
            f, l, d = disagreement(X)
            rows_d.append(dict(kind='cam00', view='cam00', frame=index, i='', alpha='', full=f, a4=l, detail=d))
            if n % 50 == 0:
                print(json.dumps(dict(stage='cam00', frame=index, seconds=round(time.time() - started, 1))), flush=True)
        del hr4
        # (2a): training cameras
        train = {(o['camera_id'], int(o['frame_index'])): o for o in manifest['observations'] if o['split'] == 'train'}
        for cam in manifest['splits']['train']:
            for fr in a.train_frames:
                X = torch.stack([render(m, native.hr_camera(ev.camera_item(m['manifest'], train[(cam, fr)], torch), fr, m['manifest']),
                                        train[(cam, fr)]['time']) for m in models])
                f, l, d = disagreement(X)
                rows_d.append(dict(kind='train', view=cam, frame=fr, i=cam, alpha=0.0, full=f, a4=l, detail=d))
        print(json.dumps(dict(stage='train', views=sum(r['kind'] == 'train' for r in rows_d), seconds=round(time.time() - started, 1))), flush=True)
        # (2b): PVCD pseudo-views (the sampler PVCD uses, unwrapped camera constructor)

        class Native0:
            hr_camera = staticmethod(native.hr_camera)

        views = cd.PseudoViews(manifest, a.pv_seed)
        for n in range(a.pv_samples):
            camera, t, info = views.sample(ev, Native0, torch)
            X = torch.stack([render(m, camera, t) for m in models])
            f, l, d = disagreement(X)
            rows_d.append(dict(kind='pseudo', view=f"{info['i']}-{info['j']}", frame=t, i=info['i'], alpha=info['alpha'], full=f, a4=l, detail=d))
        print(json.dumps(dict(stage='pseudo', views=a.pv_samples, seconds=round(time.time() - started, 1))), flush=True)

    with (out / 'kcurve_per_frame.csv').open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['frame', 'combo', 'k', 'mse', 'mse_a4', 'mse_detail'])
        w.writeheader()
        w.writerows(rows_k)
    with (out / 'disagreement_per_view.csv').open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['kind', 'view', 'frame', 'i', 'alpha', 'full', 'a4', 'detail'])
        w.writeheader()
        w.writerows(rows_d)

    psnr = lambda x: float(-10 * np.log10(max(x, 1e-12)))
    by = {}
    for r in rows_k:
        by.setdefault(r['combo'], []).append(r)
    combo = {}
    for c in subsets:
        rs = by[name(c)]
        combo[name(c)] = dict(k=len(c), frames=len(rs),
                              psnr=float(np.mean([psnr(r['mse']) for r in rs])),
                              psnr_a4=float(np.mean([psnr(r['mse_a4']) for r in rs])),
                              psnr_detail=float(np.mean([psnr(r['mse_detail']) for r in rs])),
                              pooled_psnr=psnr(float(np.mean([r['mse'] for r in rs]))),
                              pooled_psnr_a4=psnr(float(np.mean([r['mse_a4'] for r in rs]))),
                              pooled_psnr_detail=psnr(float(np.mean([r['mse_detail'] for r in rs]))))
    single_mean = {k: float(np.mean([combo[name((i,))][k] for i in range(len(labels))])) for k in ('psnr', 'psnr_a4', 'psnr_detail')}
    per_k = {}
    for k in range(1, len(labels) + 1):
        cs = [combo[name(c)] for c in subsets if len(c) == k]
        per_k[str(k)] = {q: dict(mean=float(np.mean([x[q] for x in cs])), min=float(np.min([x[q] for x in cs])),
                                 max=float(np.max([x[q] for x in cs])), gain_vs_mean_single=float(np.mean([x[q] for x in cs])) - single_mean[q])
                         for q in ('psnr', 'psnr_a4', 'psnr_detail')}
        per_k[str(k)]['subsets'] = len(cs)
    nested_curve = {str(len(c)): dict(combo=name(c), **{q: combo[name(c)][q] for q in ('psnr', 'psnr_a4', 'psnr_detail')}) for c in nested}
    dis = {}
    for kind in ('train', 'pseudo', 'cam00'):
        rs = [r for r in rows_d if r['kind'] == kind]
        dis[kind] = dict(views=len(rs), full=float(np.mean([r['full'] for r in rs])), a4=float(np.mean([r['a4'] for r in rs])),
                         detail=float(np.mean([r['detail'] for r in rs])))
        dis[kind]['a4_share'] = dis[kind]['a4'] / dis[kind]['full'] if dis[kind]['full'] > 0 else None
    for kind in ('pseudo', 'cam00'):
        dis[kind + '_over_train'] = {q: dis[kind][q] / dis['train'][q] if dis['train'][q] > 0 else None for q in ('full', 'a4', 'detail')}
    result = dict(record, status='completed', labels=labels, per_k=per_k, nested=nested_curve, combos=combo, disagreement=dis,
                  seconds=time.time() - started, peak_GB=torch.cuda.max_memory_allocated() / 1e9)
    (out / 'summary.json').write_text(json.dumps(result, indent=1))
    print(json.dumps(dict(per_k={k: (round(v['psnr']['mean'], 4), round(v['psnr_a4']['mean'], 4)) for k, v in per_k.items()},
                          disagreement={k: {q: (round(v[q], 8) if isinstance(v.get(q), float) else v.get(q)) for q in ('full', 'a4', 'detail')}
                                        for k, v in dis.items() if k in ('train', 'pseudo', 'cam00')})), flush=True)


if __name__ == '__main__':
    main()
