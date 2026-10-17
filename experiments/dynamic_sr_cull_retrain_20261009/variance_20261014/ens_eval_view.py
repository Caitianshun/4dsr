"""S1 variance study (2026-10-14): cam00 metrics of K trained checkpoints and of their render averages.

Protocol follows `cullretrain.py eval` for the scored quantities: PSNR / SSIM / AlexNet LPIPS from the legacy functions
(extracted verbatim from experiments/dynamic_sr_20260918/evaluate.py with full_evaluate_registered_v2.extract),
post-hoc training-camera culling of every model, TF32 off, all 300 cam00 frames, the same camera construction and
deformation call. The LR-closure, FFT and ROI extras are omitted (their helper files are outside the remote sync
prefixes); they do not enter PSNR / SSIM / LPIPS.

Averages are means of the clamped [0, 1] renders: every pair, and the mean of all K models when K > 2.
Checkpoints may come from different seeds or input variants of the same scene; the cam00 HR identity (hr_sha256 per
frame) must agree across their manifests, and HR pixels are read from the manifest of --hr-from (default: first model), or
from --hr-manifest (e.g. the clean scene whose hr/ files the compressed derivations hard-link; identity checked per frame).
HR is used only for scoring; nothing here trains or selects a parameter.

--view CAMERA (2026-10-15): score another non-training camera of the manifests (e.g. the complete-holdout dev camera cam03)
instead of cam00; all 300 frames of that camera are required, and the camera must not be a training camera of any model.

usage: ens_eval.py --checkpoint LABEL=rel/path/checkpoint.pt [--checkpoint ...] --out rel/out/dir [--frames 0 150 299] [--view cam00]
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import rt  # noqa: E402  (pinned runtime and registered experiment code on sys.path)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import cullretrain as cr  # noqa: E402
import full_evaluate_registered_v2 as ev  # noqa: E402
import full_refine_registered_v2 as native  # noqa: E402

ROOT = rt.ROOT
LEGACY_NAMES = ('read_rgb', 'image_array', 'ssim_map_rgb', 'masked_mean', 'metric_psnr', 'spatial_metrics', 'aggregate')


def legacy_helpers():
    import cv2
    from PIL import Image
    scope = dict(np=np, cv2=cv2, torch=torch, Image=Image, Path=Path)
    return ev.extract(ev.LEGACY, LEGACY_NAMES, scope)


def load_model(label, path, cull):
    path = ROOT / path
    plan = torch.load(path, map_location='cpu', weights_only=False)['plan']
    g, payload = cr.load_for_render(path, plan)
    del payload
    h, opt, pipe, sh = cr.configuration(plan)
    manifest_path, manifest = cr.plan_scene(plan)
    info = dict(label=label, checkpoint=cr.entry(path), manifest=cr.entry(manifest_path), points_loaded=int(g.get_xyz.shape[0]))
    if cull:
        info['cull'] = cr.cull(g, cr.Culler(manifest), None, 'posthoc', optimizer_backed=False)
    info['points_evaluated'] = int(g.get_xyz.shape[0])
    background = torch.ones(3, dtype=torch.float32, device='cuda') if cr.white_background(plan) else torch.zeros(3, dtype=torch.float32, device='cuda')
    test = sorted([o for o in manifest['observations'] if o['split'] == 'test'], key=lambda o: int(o['frame_index']))
    if [int(o['frame_index']) for o in test] != list(range(300)) or {o['camera_id'] for o in test} != {'cam00'}:
        raise ValueError('cam00 all-300 test split required: ' + label)
    return dict(label=label, g=g, pipe=pipe, background=background, manifest=manifest, test=test, info=info)


def view_observations(manifest, view, label):
    if view in manifest['splits']['train']:
        raise ValueError(f'{view} is a training camera of {label}')
    obs = sorted([o for o in manifest['observations'] if o['camera_id'] == view], key=lambda o: int(o['frame_index']))
    if [int(o['frame_index']) for o in obs] != list(range(300)) or {o['split'] for o in obs} - {'dev', 'test'}:
        raise ValueError(f'{view}: all 300 dev/test observations required ({label})')
    return obs


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', action='append', required=True, help='LABEL=relative/path/checkpoint.pt (repeat)')
    p.add_argument('--out', required=True)
    p.add_argument('--frames', type=int, nargs='*')
    p.add_argument('--hr-from')
    p.add_argument('--hr-manifest', help='relative path of a manifest that supplies the cam00 HR files (identity checked)')
    p.add_argument('--no-cull', action='store_true')
    p.add_argument('--verify-hr', action='store_true', help='SHA-check every HR file before scoring')
    p.add_argument('--cpu-threads', type=int, default=8)
    p.add_argument('--view', default='cam00')
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
    if len(set(labels)) != len(labels) or len(labels) < 2:
        raise ValueError('need >= 2 distinct labels')
    record = dict(kind='ens_eval', args=vars(a), host=socket.gethostname(), gpu=torch.cuda.get_device_name(),
                  runtime=rt.check_runtime(), sources={cr.rel(q): rt.sha(q) for q in (Path(__file__), Path(cr.__file__), Path(rt.__file__), ev.LEGACY)},
                  started_unix=started)
    models = [load_model(lab, path, not a.no_cull) for lab, path in specs]
    for m in models:
        m['test'] = view_observations(m['manifest'], a.view, m['label'])
    record['models'] = [m['info'] for m in models]
    ref = next(m for m in models if m['label'] == (a.hr_from or labels[0]))
    for m in models:
        if [o['hr_sha256'] for o in m['test']] != [o['hr_sha256'] for o in ref['test']]:
            raise ValueError(f'{a.view} HR identity differs between manifests: ' + m['label'])
    hr_manifest, hr_test = ref['manifest'], ref['test']
    if a.hr_manifest:
        from n3dv_data import load_manifest
        hr_manifest = load_manifest(ROOT / a.hr_manifest)
        hr_test = sorted([o for o in hr_manifest['observations'] if o['camera_id'] == a.view], key=lambda o: int(o['frame_index']))
        if [o['hr_sha256'] for o in hr_test] != [o['hr_sha256'] for o in ref['test']]:
            raise ValueError(f'--hr-manifest {a.view} HR identity differs from the models')
        record['hr_manifest'] = cr.entry(ROOT / a.hr_manifest)
    (out / 'run.json').write_text(json.dumps(record, indent=1))
    legacy = legacy_helpers()
    import lpips
    metric = lpips.LPIPS(net='alex', version='0.1', spatial=False, pnet_rand=False, pretrained=True, verbose=False).cuda().eval().requires_grad_(False)
    combos = [(lab,) for lab in labels] + list(itertools.combinations(labels, 2))
    if len(labels) > 2:
        combos.append(tuple(labels))
    name = lambda c: '+'.join(c)
    root = Path(hr_manifest['_root'])
    frames = [o for o in hr_test if a.frames is None or int(o['frame_index']) in set(a.frames)]
    rows = []
    for i, obs in enumerate(frames):
        index = int(obs['frame_index'])
        path = root / obs['hr_path']
        if a.verify_hr and rt.sha(path) != obs['hr_sha256']:
            raise ValueError('HR bytes differ from manifest: ' + str(path))
        hr = legacy.read_rgb(path)
        empty = np.zeros(hr.shape[:2], bool)
        preds = {}
        with torch.no_grad():
            for m in models:
                o = m['test'][index]
                cam = native.hr_camera(ev.camera_item(m['manifest'], o, torch), index, m['manifest'])
                state = native.native_effective_state(m['g'], float(o['time']), torch)
                raw = native.native_rgb(cam, m['g'], state, m['pipe'], m['background'])['render'].detach()
                preds[m['label']] = legacy.image_array(raw)
        errors = {k: (v.astype(np.float64) - hr) for k, v in preds.items()}
        for c in combos:
            pred = preds[c[0]] if len(c) == 1 else (np.mean([preds[k] for k in c], axis=0)).astype(np.float32)
            q = legacy.spatial_metrics(pred, hr, empty, metric, 'cuda')['full']
            row = dict(frame=index, combo=name(c), k=len(c), psnr=q['psnr'], ssim=q['ssim'], lpips=q['lpips_alex'], mse=q['mse'])
            if len(c) == 2:
                ea, eb = errors[c[0]], errors[c[1]]
                row['err_corr'] = float((ea * eb).sum() / np.sqrt((ea * ea).sum() * (eb * eb).sum()))
            rows.append(row)
        if i % 50 == 0:
            print(json.dumps(dict(frame=index, seconds=round(time.time() - started, 1))), flush=True)
    with (out / 'per_frame.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['frame', 'combo', 'k', 'psnr', 'ssim', 'lpips', 'mse', 'err_corr'])
        w.writeheader()
        w.writerows(rows)
    mean = lambda rs, k: float(np.mean([r[k] for r in rs]))
    by = {name(c): [r for r in rows if r['combo'] == name(c)] for c in combos}
    summary = {}
    for c in combos:
        rs = by[name(c)]
        s = dict(k=len(c), frames=len(rs), psnr=mean(rs, 'psnr'), ssim=mean(rs, 'ssim'), lpips=mean(rs, 'lpips'),
                 PSNR_of_pooled_MSE=float(-10 * np.log10(np.mean([r['mse'] for r in rs]))))
        if len(c) >= 2:
            singles = [summary[x] for x in c]
            for k in ('psnr', 'ssim', 'lpips'):
                s[k + '_gain_vs_mean_single'] = s[k] - float(np.mean([x[k] for x in singles]))
            s['psnr_gain_vs_best_single'] = s['psnr'] - max(x['psnr'] for x in singles)
        if len(c) == 2:
            da = np.array([r['psnr'] for r in by[c[0]]]) - np.array([r['psnr'] for r in by[c[1]]])
            s.update(err_corr=mean(rs, 'err_corr'), single_delta_psnr=float(da.mean()), single_delta_psnr_frame_sd=float(da.std(ddof=1)) if len(da) > 1 else None,
                     single_delta_lpips=float(np.mean([r['lpips'] for r in by[c[0]]]) - np.mean([r['lpips'] for r in by[c[1]]])))
        summary[name(c)] = s
    result = dict(record, status='completed', frames=len(frames), summary=summary, seconds=time.time() - started,
                  peak_GB=torch.cuda.max_memory_allocated() / 1e9)
    (out / 'summary.json').write_text(json.dumps(result, indent=1))
    print(json.dumps({k: (round(v['psnr'], 3), round(v['lpips'], 4)) for k, v in summary.items()}), flush=True)


if __name__ == '__main__':
    main()
