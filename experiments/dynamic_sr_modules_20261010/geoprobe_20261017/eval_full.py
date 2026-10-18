"""Full cam00 evaluation for module arms (Claude 4dsr-e9, 2026-10-17): group_eval_bands.py (byte-frozen) plus two diagnostic
metrics; everything else is unchanged (same render path, PSNR / SSIM / LPIPS with the legacy functions, FFT bands).

1. Colour-gauge-aligned PSNR (apsnr): the test camera's colour response is not modelled (identity colour; the training cameras'
   affines have mean identity), and the run-to-run drift of that gauge moves whole-image PSNR by ~0.1 dB on Discussion07. For each
   model a global 3x4 colour affine is fitted (normal equations, float64) between the render and HR on the even-indexed scored
   frames and applied to the odd-indexed ones; apsnr is the equal-frame mean PSNR of the aligned odd frames, psnr_odd the
   unaligned PSNR of the same frames (NeRF-W-style test-time appearance alignment; a metric, never a method).
2. Region PSNR: background B = per-pixel median of the HR frames 0, 5, 10, ... of the view; a pixel of frame t is dynamic when
   max_rgb |HR_t - B| > 0.06, dilated by 5 px (as region_eval.py). dyn_psnr / static_psnr: PSNR of the pooled MSE over the scored
   frames, unaligned; adyn_psnr / astatic_psnr: the same over the aligned odd frames.
The renders of the odd frames are kept on the CPU (float16) for the second pass.

Original docstring (group_eval_bands.py):
group_eval.py plus FFT band PSNR (Claude 4dsr-e9, 2026-10-17): a copy of
experiments/dynamic_sr_cull_retrain_20261009/selfens_20261016/group_eval.py (4dsr-c9, sha256 02ce6d5cc8f6f7ea...)
placed under the geoprobe directory, with one addition: per image, the float32 residual pred - HR is split by a float64 FFT into
radial bands low r < 0.125, mid 0.125 <= r < 0.25, high r >= 0.25 (cycles / HR px; no window, no DC removal), the band MSE is the
band energy / (3 (hw)^2) and band PSNR = -10 log10(band MSE); summaries report the equal-frame mean band PSNR and pooled shares.
This is the definition of experiments/dynamic_sr_same_observation_20260930/frequency.py (decompose), used by the local
`cullretrain eval` band columns, which cannot run on the remote hosts. Everything else is unchanged.

Original docstring:
Group evaluation for the single-run ensemble probes (Claude 4dsr-c9, 2026-10-16; README in this directory).

Same scored quantities as variance_20261014/ens_eval.py (and therefore `cullretrain eval`): all 300 frames of one
non-training camera, post-hoc training-camera culling of every model, TF32 off, PSNR / SSIM (5-pixel border excluded) /
AlexNet LPIPS with the legacy functions. Only the explicitly named groups are scored (ens_eval.py scores every pair), and a
group may use the per-pixel median instead of the mean. The SSIM / MSE work of the groups runs in a thread pool; LPIPS runs
once per image in the main thread (the scalar forward of spatial_metrics; the spatial map is not needed here).

usage: group_eval.py --checkpoint LABEL=rel/ckpt.pt [...] [--group NAME=L1+L2[+...]] [--median NAME=L1+L2+L3[+...]]
                     --view CAMERA --out rel/out/dir [--frames ...] [--verify-hr]
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import json
import socket
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CULL = HERE.parents[1] / 'dynamic_sr_cull_retrain_20261009'
VAR = CULL / 'variance_20261014'
for _p in (CULL, VAR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import rt  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import cullretrain as cr  # noqa: E402
try:  # the A100 keeps the older S1-a ens_eval.py; ens_eval_view.py is byte-identical to the current ens_eval.py
    import ens_eval_view as ens_eval  # noqa: E402
except ImportError:
    import ens_eval  # noqa: E402
if not hasattr(ens_eval, 'view_observations'):
    raise SystemExit('ens_eval without --view support: ' + ens_eval.__file__)
import full_evaluate_registered_v2 as ev  # noqa: E402
import full_refine_registered_v2 as native  # noqa: E402

ROOT = rt.ROOT


BANDS = ('low', 'mid', 'high')
_masks = {}
TAU, DILATE = 0.06, 5


def dyn_mask(hr, bg):
    import torch.nn.functional as F
    d = torch.from_numpy(np.abs(hr - bg).max(axis=2) > TAU)[None, None].float()
    k = 2 * DILATE + 1
    return (F.max_pool2d(d, k, 1, DILATE)[0, 0] > 0).numpy()


def band_mse(pred, hr):
    h, w = pred.shape[:2]
    if (h, w) not in _masks:
        fy = np.fft.fftfreq(h)[:, None]
        fx = np.fft.fftfreq(w)[None, :]
        r = np.sqrt(fy * fy + fx * fx)
        _masks[h, w] = [r < .125, (r >= .125) & (r < .25), r >= .25]
    err = (pred - hr).astype(np.float64)
    spectrum = np.abs(np.fft.fft2(err, axes=(0, 1))) ** 2
    return [float(spectrum[m].sum()) / (3.0 * float(h * w) ** 2) for m in _masks[h, w]]


def parse_groups(items, labels, kind):
    groups = []
    for item in items or []:
        name, members = item.split('=', 1)
        members = members.split('+')
        if len(members) < 2 or len(set(members)) != len(members) or set(members) - set(labels):
            raise ValueError(f'bad {kind} group {item}')
        if kind == 'median' and len(members) < 3:
            raise ValueError('median group needs >= 3 members: ' + item)
        groups.append(dict(name=name, members=members, kind=kind))
    return groups


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', action='append', required=True)
    p.add_argument('--group', action='append', default=[])
    p.add_argument('--median', action='append', default=[])
    p.add_argument('--view', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--frames', type=int, nargs='*')
    p.add_argument('--verify-hr', action='store_true')
    p.add_argument('--threads', type=int, default=12)
    p.add_argument('--cpu-threads', type=int, default=8)
    a = p.parse_args()
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'summary.json').exists():
        raise FileExistsError('Completed evaluation preserved: ' + str(out))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(a.cpu_threads)
    started = time.time()
    specs = [c.split('=', 1) for c in a.checkpoint]
    labels = [s[0] for s in specs]
    if len(set(labels)) != len(labels):
        raise ValueError('duplicate labels')
    groups = parse_groups(a.group, labels, 'mean') + parse_groups(a.median, labels, 'median')
    names = labels + [g['name'] for g in groups]
    if len(set(names)) != len(names):
        raise ValueError('group names must differ from labels and from each other')
    record = dict(kind='eval_full', args=vars(a), host=socket.gethostname(), gpu=torch.cuda.get_device_name(), runtime=rt.check_runtime(),
                  sources={cr.rel(q): rt.sha(q) for q in (Path(__file__), Path(ens_eval.__file__), Path(cr.__file__), Path(rt.__file__), ev.LEGACY)},
                  groups=groups, started_unix=started)
    models = [ens_eval.load_model(lab, path, True) for lab, path in specs]
    for m in models:
        m['test'] = ens_eval.view_observations(m['manifest'], a.view, m['label'])
    record['models'] = [m['info'] for m in models]
    ref = models[0]
    for m in models:
        if [o['hr_sha256'] for o in m['test']] != [o['hr_sha256'] for o in ref['test']]:
            raise ValueError(f'{a.view} HR identity differs between manifests: ' + m['label'])
    (out / 'run.json').write_text(json.dumps(record, indent=1))
    legacy = ens_eval.legacy_helpers()
    import lpips
    metric = lpips.LPIPS(net='alex', version='0.1', spatial=False, pnet_rand=False, pretrained=True, verbose=False).cuda().eval().requires_grad_(False)
    interior = None
    root = Path(ref['manifest']['_root'])
    frames = [o for o in ref['test'] if a.frames is None or int(o['frame_index']) in set(a.frames)]
    pool = cf.ThreadPoolExecutor(max_workers=a.threads)

    def cpu_metrics(pred, hr):
        err = ((pred - hr) ** 2).mean(axis=2)
        ssim = legacy.ssim_map_rgb(pred, hr)
        mse = legacy.masked_mean(err, np.ones(err.shape, bool))
        bands = band_mse(pred, hr)
        return dict(mse=mse, psnr=legacy.metric_psnr(mse), ssim=legacy.masked_mean(ssim, interior),
                    **{f'{b}_mse': v for b, v in zip(BANDS, bands)}, **{f'{b}_psnr': -10 * float(np.log10(max(v, 1e-12))) for b, v in zip(BANDS, bands)})

    rows = []
    bg = np.median(np.stack([legacy.read_rgb(root / ref['test'][f]['hr_path']) for f in range(0, len(ref['test']), 5)]), axis=0)
    gram = {n: [np.zeros((4, 4)), np.zeros((4, 3))] for n in names}
    keep = []  # odd frames: (index, hr path, dynamic mask, {name: float16 render})
    region = {n: dict(dyn=0.0, sta=0.0) for n in names}
    n_dyn = n_sta = 0
    for i, obs in enumerate(frames):
        index = int(obs['frame_index'])
        path = root / obs['hr_path']
        if a.verify_hr and rt.sha(path) != obs['hr_sha256']:
            raise ValueError('HR bytes differ from manifest: ' + str(path))
        hr = legacy.read_rgb(path)
        if interior is None:
            interior = np.zeros(hr.shape[:2], bool)
            interior[5:-5, 5:-5] = True
        preds = {}
        with torch.no_grad():
            for m in models:
                o = m['test'][index]
                cam = native.hr_camera(ev.camera_item(m['manifest'], o, torch), index, m['manifest'])
                st = native.native_effective_state(m['g'], float(o['time']), torch)
                preds[m['label']] = legacy.image_array(native.native_rgb(cam, m['g'], st, m['pipe'], m['background'])['render'].detach())
        for g in groups:
            stack = [preds[k] for k in g['members']]
            preds[g['name']] = (np.median(stack, axis=0) if g['kind'] == 'median' else np.mean(stack, axis=0)).astype(np.float32)
        futures = {n: pool.submit(cpu_metrics, preds[n], hr) for n in names}
        dm = dyn_mask(hr, bg)
        nd = int(dm.sum())
        n_dyn += 3 * nd
        n_sta += 3 * (dm.size - nd)
        for n in names:
            e = ((preds[n].astype(np.float64) - hr) ** 2).sum(axis=2)
            region[n]['dyn'] += float(e[dm].sum())
            region[n]['sta'] += float(e[~dm].sum())
        if i % 2 == 0:
            for n in names:
                x = preds[n].reshape(-1, 3).astype(np.float64)
                X = np.concatenate([x, np.ones((x.shape[0], 1))], 1)
                gram[n][0] += X.T @ X
                gram[n][1] += X.T @ hr.reshape(-1, 3).astype(np.float64)
        else:
            keep.append((index, path, dm, {n: preds[n].astype(np.float16) for n in names}))
        ty = torch.from_numpy(np.ascontiguousarray(hr)).permute(2, 0, 1)[None].cuda() * 2 - 1
        lp = {}
        with torch.inference_mode():
            for n in names:
                tx = torch.from_numpy(np.ascontiguousarray(preds[n])).permute(2, 0, 1)[None].cuda() * 2 - 1
                lp[n] = float(metric(tx, ty).item())
        for n in names:
            q = futures[n].result()
            row = dict(frame=index, name=n, psnr=q['psnr'], ssim=q['ssim'], lpips=lp[n], mse=q['mse'],
                       **{k: q[k] for b in BANDS for k in (f'{b}_mse', f'{b}_psnr')})
            rows.append(row)
        if i % 50 == 0:
            print(json.dumps(dict(frame=index, seconds=round(time.time() - started, 1))), flush=True)
    pool.shutdown()
    aligned = {}
    for n in names:
        sol = np.linalg.solve(gram[n][0] + 1e-9 * np.eye(4), gram[n][1])  # 4 x 3: y = x A^T + b
        aligned[n] = dict(A=sol[:3].T.tolist(), b=sol[3].tolist(), psnr_odd=[], apsnr=[], adyn=0.0, asta=0.0)
    an_dyn = an_sta = 0
    for index, path, dm, rend in keep:
        hr = legacy.read_rgb(path).astype(np.float64)
        nd = int(dm.sum())
        an_dyn += 3 * nd
        an_sta += 3 * (dm.size - nd)
        for n in names:
            x = rend[n].astype(np.float64)
            A, b = np.array(aligned[n]['A']), np.array(aligned[n]['b'])
            y = np.clip(x @ A.T + b, 0, 1)
            aligned[n]['psnr_odd'].append(legacy.metric_psnr(float(((x - hr) ** 2).mean())))
            e = ((y - hr) ** 2).sum(axis=2)
            aligned[n]['apsnr'].append(legacy.metric_psnr(float(e.mean() / 3)))
            aligned[n]['adyn'] += float(e[dm].sum())
            aligned[n]['asta'] += float(e[~dm].sum())
    with (out / 'per_frame.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['frame', 'name', 'psnr', 'ssim', 'lpips', 'mse'] + [f'{b}_{k}' for b in BANDS for k in ('mse', 'psnr')])
        w.writeheader()
        w.writerows(rows)
    by = {n: [r for r in rows if r['name'] == n] for n in names}
    mean = lambda n, k: float(np.mean([r[k] for r in by[n]]))
    summary = {n: dict(frames=len(by[n]), psnr=mean(n, 'psnr'), ssim=mean(n, 'ssim'), lpips=mean(n, 'lpips'),
                       PSNR_of_pooled_MSE=float(-10 * np.log10(np.mean([r['mse'] for r in by[n]]))),
                       **{f'{b}_psnr': mean(n, f'{b}_psnr') for b in BANDS},
                       **{f'{b}_share_pct': 100 * sum(r[f'{b}_mse'] for r in by[n]) / sum(r['mse'] for r in by[n]) for b in BANDS}) for n in names}
    for n in names:
        q = aligned[n]
        summary[n].update(psnr_odd=float(np.mean(q['psnr_odd'])), apsnr=float(np.mean(q['apsnr'])),
                          dyn_psnr=-10 * float(np.log10(max(region[n]['dyn'] / max(n_dyn, 1), 1e-12))),
                          static_psnr=-10 * float(np.log10(max(region[n]['sta'] / max(n_sta, 1), 1e-12))),
                          adyn_psnr=-10 * float(np.log10(max(q['adyn'] / max(an_dyn, 1), 1e-12))),
                          astatic_psnr=-10 * float(np.log10(max(q['asta'] / max(an_sta, 1), 1e-12))),
                          dyn_share=n_dyn / max(n_dyn + n_sta, 1), colour_affine=dict(A=q['A'], b=q['b']))
    for g in groups:
        s, singles = summary[g['name']], [summary[k] for k in g['members']]
        s.update(kind=g['kind'], members=g['members'])
        for k in ('psnr', 'ssim', 'lpips'):
            s[k + '_gain_vs_mean_single'] = s[k] - float(np.mean([x[k] for x in singles]))
        s['psnr_gain_vs_best_single'] = s['psnr'] - max(x['psnr'] for x in singles)
    result = dict(record, status='completed', frames=len(frames), summary=summary, seconds=time.time() - started,
                  peak_GB=torch.cuda.max_memory_allocated() / 1e9)
    (out / 'summary.json').write_text(json.dumps(result, indent=1))
    print(json.dumps({k: (round(v['psnr'], 3), round(v['lpips'], 4), round(v['mid_psnr'], 2), round(v['high_psnr'], 2), round(v['apsnr'], 3),
                          round(v['dyn_psnr'], 3), round(v['adyn_psnr'], 3)) for k, v in summary.items()}), flush=True)


if __name__ == '__main__':
    main()
