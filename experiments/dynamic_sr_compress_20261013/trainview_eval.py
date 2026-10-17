"""Training-view HR diagnostic for the compressed-LR track (Claude, 2026-10-13). mml_bw, GPU through the training service.

  model     For one SR checkpoint, render every training camera at --frames (default 0, 15, ..., 285: ten I frames and
            ten P frames of the GOP-30 streams) at HR, with the learned camera colour affine and without, and compare with
            that camera's HR image (PSNR, SSIM, AlexNet LPIPS; the same legacy functions as cullretrain eval).
            With --teacher-index, the same keys also score that run's teacher images against HR (teacher quality) and the
            affine-corrected render against the teacher (fit to the teacher).
  teachers  Teacher images only: each LABEL=INDEX teacher index is scored against HR at the same keys, with the HR taken
            from --manifest. Relative paths in an index are resolved against the index's own directory.

The SR stage never reads training-view HR, so these numbers are a development diagnostic of how close a model or teacher
is to the truth on the observed views. They are not novel-view results. cam00 is never rendered or read here.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CULL = HERE.parent / 'dynamic_sr_cull_retrain_20261009'
sys.path.insert(0, str(CULL))

import cullretrain as cr  # noqa: E402  (sets up runtime paths, imports torch and the author code)
import rt  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

native, ev = cr.native, cr.ev
FRAMES = list(range(0, 300, 15))


def summarize(rows, keys):
    groups = dict(all=rows, I=[r for r in rows if r['iframe']], P=[r for r in rows if not r['iframe']])
    out = {}
    for name, rs in groups.items():
        cell = dict(n=len(rs))
        for k in keys:
            vals = [r[k] for r in rs if r.get(k) is not None]
            if vals:
                cell[k] = float(np.mean(vals))
        if rs and all('mse' in r for r in rs):
            cell['PSNR_of_pooled_MSE'] = float(-10 * np.log10(np.mean([r['mse'] for r in rs])))
        out[name] = cell
    cams = sorted({r['camera'] for r in rows})
    out['per_camera'] = {c: {k: float(np.mean([r[k] for r in rows if r['camera'] == c])) for k in keys
                             if all(r.get(k) is not None for r in rows if r['camera'] == c)} for c in cams}
    return out


def write_rows(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
        writer.writeheader()
        writer.writerows(rows)


def teacher_lookup(index_path):
    index_path = Path(index_path).resolve()
    index = json.loads(index_path.read_text())
    base = index_path.parent
    paths = {}
    for e in index['entries']:
        p = (base / e['relative_path']).resolve()
        if base not in p.parents:
            raise ValueError(f'teacher path escapes its scene root: {p}')
        paths[(e['camera'], int(e['frame']))] = p
    return index, paths


def lpips_metric():
    import lpips
    return lpips.LPIPS(net='alex', version='0.1', spatial=False, pnet_rand=False, pretrained=True, verbose=False).cuda().eval().requires_grad_(False)


def run_model(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'summary.json').exists():
        raise FileExistsError('Completed diagnostic preserved: ' + str(out))
    started = time.time()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(a.cpu_threads)
    checkpoint = Path(a.checkpoint)
    g, payload = cr.load_for_render(checkpoint, torch.load(checkpoint, map_location='cpu', weights_only=False)['plan'])
    plan = payload['plan']
    affine_state = payload.get('affine')
    del payload
    h, opt, pipe, sh = cr.configuration(plan)
    manifest_path, manifest = cr.plan_scene(plan)
    train = list(manifest['splits']['train'])
    if 'cam00' in train:
        raise ValueError('cam00 must not be a training camera')
    record = dict(kind='trainview_model', label=a.label, checkpoint=cr.entry(checkpoint), manifest=cr.entry(manifest_path),
                  posthoc_cull=bool(a.cull), frames=a.frames, cameras=train, gpu=torch.cuda.get_device_name(),
                  runtime=rt.check_runtime(), sources={cr.rel(p): rt.sha(p) for p in (Path(cr.__file__), Path(__file__))},
                  points_loaded=int(g.get_xyz.shape[0]))
    if a.cull:
        record['cull'] = cr.cull(g, cr.Culler(manifest), None, 'posthoc', optimizer_backed=False)
    record['points_evaluated'] = int(g.get_xyz.shape[0])
    if affine_state is None:
        raise ValueError('checkpoint has no camera affine')
    affine = cr.CameraAffine(train)
    affine.load_state_dict(affine_state)
    teacher_paths = {}
    if a.teacher_index:
        index, teacher_paths = teacher_lookup(a.teacher_index)
        record['teacher_index'] = dict(path=cr.rel(Path(a.teacher_index).resolve()), sha256=rt.sha(Path(a.teacher_index)),
                                       input_policy=index.get('input_policy'))
    legacy, closure, _, fft = ev.metric_helpers()
    metric = lpips_metric()
    background = torch.ones(3, dtype=torch.float32, device='cuda') if cr.white_background(plan) else torch.zeros(3, dtype=torch.float32, device='cuda')
    root = Path(manifest['_root'])
    obs_by = {(o['camera_id'], int(o['frame_index'])): o for o in manifest['observations']}
    rows = []
    for uid, (cam, f) in enumerate((c, f) for c in train for f in a.frames):
        obs = obs_by[(cam, f)]
        if obs['split'] != 'train':
            raise ValueError(f'{cam}/{f} is not a training observation')
        with torch.no_grad():
            camera = native.hr_camera(ev.camera_item(manifest, obs, torch), uid, manifest)
            state = native.native_effective_state(g, float(obs['time']), torch)
            raw = native.native_rgb(camera, g, state, pipe, background)['render'].detach()
            corrected = affine.apply(raw, cam).clamp(0, 1)
        hr = legacy.read_rgb(root / obs['hr_path'])
        empty = np.zeros(hr.shape[:2], bool)
        pred = legacy.image_array(corrected)
        q = legacy.spatial_metrics(pred, hr, empty, metric, 'cuda')['full']
        qr = legacy.spatial_metrics(legacy.image_array(raw), hr, empty, metric, 'cuda')['full']
        row = dict(camera=cam, frame=f, iframe=f % 30 == 0, psnr=q['psnr'], ssim=q['ssim'], lpips=q['lpips_alex'], mse=q['mse'],
                   psnr_raw=qr['psnr'], ssim_raw=qr['ssim'], lpips_raw=qr['lpips_alex'])
        if teacher_paths:
            t_np = cr.read_teacher(teacher_paths[(cam, f)]).permute(1, 2, 0).numpy().astype(np.float32)
            qt = legacy.spatial_metrics(t_np, hr, empty, metric, 'cuda')['full']
            qf = legacy.spatial_metrics(pred, t_np, empty, metric, 'cuda')['full']
            row.update(teacher_psnr=qt['psnr'], teacher_ssim=qt['ssim'], teacher_lpips=qt['lpips_alex'],
                       fit_psnr=qf['psnr'], fit_lpips=qf['lpips_alex'])
        rows.append(row)
    write_rows(out / 'metrics_per_observation.csv', rows)
    keys = [k for k in ('psnr', 'ssim', 'lpips', 'psnr_raw', 'ssim_raw', 'lpips_raw', 'teacher_psnr', 'teacher_ssim',
                        'teacher_lpips', 'fit_psnr', 'fit_lpips') if k in rows[0]]
    summary = dict(record, status='completed', observations=len(rows), metrics=summarize(rows, keys),
                   seconds=time.time() - started, peak_GB=torch.cuda.max_memory_allocated() / 1e9)
    rt.write_json(out / 'summary.json', summary)
    print(json.dumps(dict(label=a.label, all=summary['metrics']['all'], seconds=summary['seconds'])))
    return summary


def run_teachers(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'summary.json').exists():
        raise FileExistsError('Completed diagnostic preserved: ' + str(out))
    started = time.time()
    manifest_path = Path(a.manifest).resolve()
    manifest = json.loads(manifest_path.read_text())
    root = manifest_path.parent
    train = sorted(manifest['splits']['train'])
    if 'cam00' in train:
        raise ValueError('cam00 must not be a training camera')
    obs_by = {(o['camera_id'], int(o['frame_index'])): o for o in manifest['observations']}
    runtime = rt.check_runtime()  # before LPIPS: also pins torch.hub to the runtime copy that holds the AlexNet weights
    legacy, closure, _, fft = ev.metric_helpers()
    metric = lpips_metric()
    result = dict(kind='trainview_teachers', manifest=cr.entry(manifest_path), frames=a.frames, cameras=train,
                  gpu=torch.cuda.get_device_name(), runtime=runtime,
                  sources={cr.rel(p): rt.sha(p) for p in (Path(cr.__file__), Path(__file__))}, teachers={})
    for spec in a.teacher:
        label, path = spec.split('=', 1)
        index, paths = teacher_lookup(path)
        rows = []
        for cam in train:
            for f in a.frames:
                obs = obs_by[(cam, f)]
                hr = legacy.read_rgb(root / obs['hr_path'])
                empty = np.zeros(hr.shape[:2], bool)
                t_np = cr.read_teacher(paths[(cam, f)]).permute(1, 2, 0).numpy().astype(np.float32)
                q = legacy.spatial_metrics(t_np, hr, empty, metric, 'cuda')['full']
                rows.append(dict(teacher=label, camera=cam, frame=f, iframe=f % 30 == 0, psnr=q['psnr'], ssim=q['ssim'],
                                 lpips=q['lpips_alex'], mse=q['mse']))
        write_rows(out / f'teacher_{label}.csv', rows)
        result['teachers'][label] = dict(index=dict(path=cr.rel(Path(path).resolve()), sha256=rt.sha(Path(path))),
                                         input_policy=index.get('input_policy'),
                                         metrics=summarize(rows, ('psnr', 'ssim', 'lpips')))
        print(label, json.dumps(result['teachers'][label]['metrics']['all']), flush=True)
    result.update(status='completed', seconds=time.time() - started)
    rt.write_json(out / 'summary.json', result)
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    m = sub.add_parser('model')
    m.add_argument('--checkpoint', required=True)
    m.add_argument('--label', required=True)
    m.add_argument('--teacher-index')
    m.add_argument('--cull', action='store_true')
    m.add_argument('--cpu-threads', type=int, default=4)
    t = sub.add_parser('teachers')
    t.add_argument('--manifest', required=True)
    t.add_argument('--teacher', nargs='+', required=True, help='LABEL=teacher_index.json')
    for x in (m, t):
        x.add_argument('--out', required=True)
        x.add_argument('--frames', type=int, nargs='+', default=FRAMES)
    a = p.parse_args(argv)
    dict(model=run_model, teachers=run_teachers)[a.cmd](a)


if __name__ == '__main__':
    main()
