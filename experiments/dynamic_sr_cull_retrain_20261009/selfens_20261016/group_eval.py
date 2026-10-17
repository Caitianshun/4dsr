"""Group evaluation for the single-run ensemble probes (Claude 4dsr-c9, 2026-10-16; README in this directory).

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
VAR = HERE.parent / 'variance_20261014'
for _p in (HERE.parent, VAR):
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
    record = dict(kind='group_eval', args=vars(a), host=socket.gethostname(), gpu=torch.cuda.get_device_name(), runtime=rt.check_runtime(),
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
        return dict(mse=mse, psnr=legacy.metric_psnr(mse), ssim=legacy.masked_mean(ssim, interior))

    rows = []
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
        ty = torch.from_numpy(np.ascontiguousarray(hr)).permute(2, 0, 1)[None].cuda() * 2 - 1
        lp = {}
        with torch.inference_mode():
            for n in names:
                tx = torch.from_numpy(np.ascontiguousarray(preds[n])).permute(2, 0, 1)[None].cuda() * 2 - 1
                lp[n] = float(metric(tx, ty).item())
        for n in names:
            q = futures[n].result()
            row = dict(frame=index, name=n, psnr=q['psnr'], ssim=q['ssim'], lpips=lp[n], mse=q['mse'])
            rows.append(row)
        if i % 50 == 0:
            print(json.dumps(dict(frame=index, seconds=round(time.time() - started, 1))), flush=True)
    pool.shutdown()
    with (out / 'per_frame.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['frame', 'name', 'psnr', 'ssim', 'lpips', 'mse'])
        w.writeheader()
        w.writerows(rows)
    by = {n: [r for r in rows if r['name'] == n] for n in names}
    mean = lambda n, k: float(np.mean([r[k] for r in by[n]]))
    summary = {n: dict(frames=len(by[n]), psnr=mean(n, 'psnr'), ssim=mean(n, 'ssim'), lpips=mean(n, 'lpips'),
                       PSNR_of_pooled_MSE=float(-10 * np.log10(np.mean([r['mse'] for r in by[n]])))) for n in names}
    for g in groups:
        s, singles = summary[g['name']], [summary[k] for k in g['members']]
        s.update(kind=g['kind'], members=g['members'])
        for k in ('psnr', 'ssim', 'lpips'):
            s[k + '_gain_vs_mean_single'] = s[k] - float(np.mean([x[k] for x in singles]))
        s['psnr_gain_vs_best_single'] = s['psnr'] - max(x['psnr'] for x in singles)
    result = dict(record, status='completed', frames=len(frames), summary=summary, seconds=time.time() - started,
                  peak_GB=torch.cuda.max_memory_allocated() / 1e9)
    (out / 'summary.json').write_text(json.dumps(result, indent=1))
    print(json.dumps({k: (round(v['psnr'], 3), round(v['lpips'], 4)) for k, v in summary.items()}), flush=True)


if __name__ == '__main__':
    main()
