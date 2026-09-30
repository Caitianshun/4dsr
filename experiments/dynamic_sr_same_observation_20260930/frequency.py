"""Radial FFT residual audit, compatible with mml's freq_band_decomp.py.

This is a frozen-checkpoint diagnostic. HR is never a training target here.
The main 4dsr comparison retains float renders, rather than mml's PNG inputs.
Run register, cached and render (independent jobs), then summarize, in that order.
"""
import argparse
import csv
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import time

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / 'output/dynamic_sr_prior_diagnosis_20260929'
OUT = ROOT / 'output/dynamic_sr_same_observation_20260930/frequency'
ARMS = ['LR-direct-HRrender', 'LR-direct-Bicubic', 'HR-direct-6k', 'U6000',
        'J1', 'Async2', 'Sync2', 'T', 'P1-from-start']
BANDS = ['low', 'mid', 'high']
MML = Path('/home/cai_tianshun/Project/mml')


def read(p): return json.loads(Path(p).read_text())
def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()
def write(p, value):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    temporary = p.with_suffix(p.suffix + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(p)
def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec); sys.modules[name] = obj
    spec.loader.exec_module(obj); return obj
def csvfile(p, rows):
    with Path(p).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


_masks = {}
def masks(h, w):
    if (h, w) not in _masks:
        fy = np.fft.fftfreq(h)[:, None]; fx = np.fft.fftfreq(w)[None, :]
        radius = np.sqrt(fy * fy + fx * fx)
        _masks[h, w] = [radius < .125, (radius >= .125) & (radius < .25), radius >= .25]
        assert np.all(sum(m.astype(np.uint8) for m in _masks[h, w]) == 1)
    return _masks[h, w]


def decompose(pred, gt):
    assert pred.shape == gt.shape and pred.shape[-1] == 3
    assert pred.dtype == gt.dtype == np.float32
    err = pred - gt  # mml subtracts float32 images before the FFT.
    h, w = err.shape[:2]
    spectrum = np.abs(np.fft.fft2(err.astype(np.float64), axes=(0, 1))) ** 2
    energies = [float(spectrum[m].sum()) / (3.0 * float(h * w) ** 2) for m in masks(h, w)]
    mse = float(np.mean(err.astype(np.float64) ** 2))
    gap = abs(sum(energies) - mse)
    assert gap < 1e-11, (gap, mse)
    result = dict(mse=mse, psnr=-10 * float(np.log10(max(mse, 1e-12))), parseval_gap=gap,
                  legacy_float32_psnr=-10 * float(np.log10(max(float(np.mean(err ** 2)), 1e-12))))
    for name, value in zip(BANDS, energies):
        result[name + '_mse'] = value
        result[name + '_psnr'] = -10 * float(np.log10(max(value, 1e-12)))
    return result


def aggregate(rows):
    assert rows
    result = dict(observations=len(rows), repeats=len(set(r['repeat'] for r in rows)),
                  psnr=float(np.mean([r['psnr'] for r in rows])),
                  legacy_float32_psnr=float(np.mean([r['legacy_float32_psnr'] for r in rows])),
                  mse=float(np.mean([r['mse'] for r in rows])),
                  max_parseval_gap=max(r['parseval_gap'] for r in rows))
    for b in BANDS:
        result[b + '_mse'] = float(np.mean([r[b + '_mse'] for r in rows]))
        result[b + '_psnr'] = float(np.mean([r[b + '_psnr'] for r in rows]))
        result[b + '_share_pct'] = 100 * sum(r[b + '_mse'] for r in rows) / sum(r['mse'] for r in rows)
    assert abs(sum(result[b + '_share_pct'] for b in BANDS) - 100) < 1e-8
    result['psnr_of_pooled_mse'] = -10 * float(np.log10(max(result['mse'], 1e-12)))
    return result


def verify():
    # Exact compatibility includes PNG ingestion, pooled shares and mean log-PSNR.
    reference = module('mml_fft_readonly_reference', MML / 'scripts/freq_band_decomp.py')
    rng = np.random.default_rng(20260930); rows = []
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder); (root / 'gt').mkdir(); (root / 'render').mkdir()
        for i, amplitude in enumerate([2, 8, 32]):
            gt = rng.integers(32, 224, (64, 64, 3), dtype=np.uint8)
            pred = np.clip(gt.astype(np.int16) + rng.integers(-amplitude, amplitude + 1, gt.shape), 0, 255).astype(np.uint8)
            Image.fromarray(gt).save(root / 'gt' / f'{i}.png'); Image.fromarray(pred).save(root / 'render' / f'{i}.png')
            rows.append(dict(repeat=1, **decompose(pred.astype(np.float32) / 255, gt.astype(np.float32) / 255)))
        expected = reference.decompose('compatibility_fixture', str(root)); actual = aggregate(rows)
    gaps = [abs(actual['psnr'] - expected['psnr_total_mean'])]
    for b in BANDS:
        gaps += [abs(actual[b + '_psnr'] - expected['psnr_band_mean'][b]),
                 abs(actual[b + '_share_pct'] - expected['energy_share_pct'][b])]
    assert max(gaps) < 1e-10, gaps
    samples = []
    yy, xx = np.indices((64, 64))
    for kx, ky, expected_band in [(0, 0, 'low'), (7, 0, 'low'), (8, 0, 'mid'), (16, 0, 'high'), (7, 7, 'mid')]:
        signal = np.cos(2 * np.pi * (kx * xx + ky * yy) / 64).astype(np.float32)
        a = np.repeat(signal[..., None], 3, 2)
        d = decompose(a, np.zeros_like(a))
        fraction = d[expected_band + '_mse'] / d['mse']; assert fraction > 1 - 1e-12
        samples.append(dict(fx=kx / 64, fy=ky / 64, band=expected_band, fraction=fraction))
    return dict(status='passed', png_reference_max_gap=max(gaps), periodic_sinusoid_cases=samples,
                diagonal_case='fx=fy=7/64 is inside the nominal LR Nyquist square, but outside the radial low disk')


def register():
    assert not (OUT / 'protocol.json').exists(), 'Protocol already registered'
    old = read(ROOT / 'output/dynamic_sr_temporal_prior_20260928/protocol.json')
    refs = read(ROOT / 'output/dynamic_sr_temporal_prior_20260928/baseline_registry.json')['references']
    sync = read(ROOT / 'output/dynamic_sr_sync_multiview_20260928/checkpoint_index.json')['checkpoints']
    inventory = read(OLD / 'checkpoint_inventory.json')['checkpoints']
    new = read(ROOT / 'output/dynamic_sr_same_observation_20260930/train/complete.json')
    entries = []
    for arm, cache in [('LR-direct-HRrender', 'LR6k'), ('LR-direct-Bicubic', None), ('HR-direct-6k', 'HR6k')]:
        entries.append(dict(arm=arm, repeat=1, checkpoint=refs[arm]['checkpoint'], cache=cache))
    entries.append(dict(arm='U6000', repeat=1, checkpoint=old['parent'], cache='U6000'))
    for arm in ['J1', 'Async2', 'Sync2', 'T']:
        for repeat in [1, 2]:
            key = f'r{repeat}_{arm}'
            if arm == 'J1': cp = old['historical_J1'][str(repeat)]['checkpoint']
            elif arm == 'T': cp = next({k: row[k] for k in ['path', 'sha256']} for row in inventory if row['task_id'] == key and row['step'] == 12000)
            else: cp = dict(path=sync[key]['checkpoint'], sha256=sync[key]['sha256'])
            cache = f'r1_{arm}' if repeat == 1 and arm in ['J1', 'Async2'] else None
            entries.append(dict(arm=arm, repeat=repeat, checkpoint=cp, cache=cache))
    entries.append(dict(arm='P1-from-start', repeat=1, cache=None,
                        checkpoint=dict(path=str(Path('output/dynamic_sr_same_observation_20260930/train') / new['final_checkpoint']), sha256=new['final_sha256'])))
    for e in entries: assert sha(ROOT / e['checkpoint']['path']) == e['checkpoint']['sha256']
    write(OUT / 'verification.json', verify())
    files = [MML / 'scripts/freq_band_decomp.py', MML / 'docs/week3_freq_band_diagnostic.md',
             MML / 'docs/hf_lrgate_loss_walkthrough.md', Path(__file__),
             ROOT / 'experiments/dynamic_sr_prior_diagnosis_20260929/spectral_common.py',
             ROOT / 'experiments/dynamic_sr_motion_bound_20260923/motion_model.py',
             ROOT / 'experiments/dynamic_sr_detail_supervision_20260924/evaluate.py']
    protocol = dict(status='registered', schema=1, manifest=old['manifest'], entries=entries,
                    observations='cam00 and cam01: all 60 registered frames each, equal cardinality; 13 endpoints, 1560 pairs',
                    training_diagnostic='cam02/cam06/cam12/cam18 x frames 0/40/80/118: 16 fixed pairs',
                    fft='numpy.fft.fft2(float32 prediction-minus-HR promoted to float64), axes=(0,1), norm=backward, RGB',
                    bands=dict(low='r<0.125 (includes DC)', mid='0.125<=r<0.25', high='r>=0.25'),
                    normalization='sum(abs(FFT(error))^2) / [3*(H*W)^2]; no window, no DC subtraction',
                    aggregation='mean per-frame PSNR; pooled band MSE / pooled total MSE; both cameras equal frame count; two continuation repeats pooled where applicable',
                    adaptation='mml input is quantized render PNG; 4dsr main uses clamp(raw float32 render,0,1) before PNG. Arithmetic compatibility separately verified on PNG fixture.',
                    information_boundary='HR and development-camera LR diagnostic only; no updates, no new masks or loss weights selected from HR',
                    sources={str(f): sha(f) for f in files})
    write(OUT / 'protocol.json', protocol); print('Registered 13 frozen endpoints', flush=True)


def run(kind):
    p = read(OUT / 'protocol.json'); assert sha(__file__) == p['sources'][str(Path(__file__))]
    assert sha(ROOT / p['manifest']['path']) == p['manifest']['sha256']
    manifest_path = ROOT / p['manifest']['path']; manifest = read(manifest_path)
    observations = [o for o in manifest['observations'] if o['camera_id'] in ['cam00', 'cam01']]
    assert len(observations) == 120
    data = manifest_path.parent; begun = time.monotonic(); entries = [e for e in p['entries'] if bool(e['cache']) == (kind == 'cached')]
    renderer = None
    if kind == 'render':
        import torch
        import torch.nn.functional as F
        from types import SimpleNamespace
        assert os.environ['CUDA_VISIBLE_DEVICES'] == '1'
        sys.path.insert(0, str(ROOT / 'experiments/dynamic_sr_20260918'))
        motion = module('frequency_motion_model', ROOT / 'experiments/dynamic_sr_motion_bound_20260923/motion_model.py')
        ev = module('frequency_render_camera', ROOT / 'experiments/dynamic_sr_detail_supervision_20260924/evaluate.py')
        from n3dv_data import load_manifest, observation_to_4dgs_camera
        manifest = load_manifest(manifest_path); torch.set_num_threads(2)
        import diff_gaussian_rasterization._C as ext
        renderer = dict(gpu=torch.cuda.get_device_name(), visible_cuda=os.environ['CUDA_VISIBLE_DEVICES'],
                        torch=torch.__version__, cuda=torch.version.cuda, extension_sha256=sha(ext.__file__))
    renders = 0; all_rows = []
    for e in entries:
        path = OUT / 'endpoints' / f"r{e['repeat']}_{e['arm']}.json"
        assert not path.exists(), path
        cp = ROOT / e['checkpoint']['path']; assert sha(cp) == e['checkpoint']['sha256']
        model = None; before = None; rows = []; sources = []
        if kind == 'render':
            model = motion.load_model(cp, manifest); model.g._deformation.eval()
            watched = [model.g._xyz, model.g._scaling, model.g._rotation, model.g._opacity, model.g._features_dc,
                       model.g._features_rest, *model.g._deformation.parameters()]
            if model.children is not None: watched += list(model.children.parameters())
            before = [x._version for x in watched]
        for i, o in enumerate(observations):
            gt_path = data / o['hr_path']; assert sha(gt_path) == o['hr_sha256']
            gt = np.asarray(Image.open(gt_path).convert('RGB'), dtype=np.float32) / 255
            if e['cache']:
                npz = OLD / 'observables' / e['cache'] / f"{o['camera_id']}_{o['frame_index']:04d}.npz"
                receipt = read(npz.with_suffix('.json'))
                assert receipt['checkpoint']['sha256'] == e['checkpoint']['sha256']
                assert receipt['npz_sha256'] == sha(npz)
                with np.load(npz) as a: pred = np.clip(a['rgb_raw'].transpose(1, 2, 0), 0, 1)
                sources.append(dict(path=str(npz.relative_to(ROOT)), sha256=receipt['npz_sha256']))
            else:
                with torch.no_grad():
                    if e['arm'] == 'LR-direct-Bicubic':
                        w, h = manifest['resolutions']['lr']; cal = manifest['cameras'][o['camera_id']]
                        cam = observation_to_4dgs_camera(dict(image=torch.zeros(3, h, w), camera_id=o['camera_id'],
                            frame_index=o['frame_index'], time=o['time'], width=w, height=h,
                            K=np.asarray(cal['K_lr']), w2c=np.asarray(cal['w2c'])), i)
                        raw = motion.render_model(model, cam)['render'].clamp(0, 1)
                        raw = F.interpolate(raw[None], size=gt.shape[:2], mode='bicubic', align_corners=False, antialias=True)[0]
                    else: raw = motion.render_model(model, ev.render_camera(manifest, o, i))['render']
                    assert torch.isfinite(raw).all()
                    pred = raw.clamp(0, 1).permute(1, 2, 0).cpu().numpy(); renders += 1
            rows.append(dict(arm=e['arm'], repeat=e['repeat'], camera=o['camera_id'], frame=o['frame_index'], **decompose(pred, gt)))
        if model is not None:
            assert before == [x._version for x in watched]
            del model, watched; gc.collect(); torch.cuda.empty_cache()
        write(path, dict(entry=e, rows=rows, cache_sources=sources, renderer=renderer, parameter_updates=0))
        all_rows += rows
        print(f"{e['arm']} r{e['repeat']}: {aggregate(rows)['psnr']:.6f} dB", flush=True)
    if kind == 'render': train_diagnostic(manifest, data, torch, F, motion, ev, renderer)
    write(OUT / f'{kind}_complete.json', dict(status='completed', endpoints=len(entries), pairs=len(all_rows),
         rgb_forwards=renders + (16 if kind == 'render' else 0), parameter_updates=0,
         seconds=time.monotonic() - begun, protocol_sha256=sha(OUT / 'protocol.json'), renderer=renderer))


def train_diagnostic(manifest, data, torch, F, motion, ev, renderer):
    e = next(e for e in read(OUT / 'protocol.json')['entries'] if e['arm'] == 'P1-from-start')
    model = motion.load_model(ROOT / e['checkpoint']['path'], manifest); model.g._deformation.eval()
    rows = []; closures = []; sources = []
    teachers = {(o['camera'], o['frame']): o for o in read(ROOT / 'output/dynamic_sr_same_observation_20260930/protocol.json')['observations']}
    observations = [o for o in manifest['observations'] if o['camera_id'] in ['cam02', 'cam06', 'cam12', 'cam18'] and o['frame_index'] in [0, 40, 80, 118]]
    assert len(observations) == 16
    with torch.no_grad():
        for i, o in enumerate(observations):
            t = teachers[o['camera_id'], o['frame_index']]
            paths = dict(hr=data / o['hr_path'], lr=data / o['lr_path'], sr=data / t['sr_path'])
            for role, path in paths.items():
                expected = o[role + '_sha256'] if role != 'sr' else t['sr_sha256']
                assert sha(path) == expected; sources.append(dict(role=role, path=str(path), sha256=expected))
            images = {role: np.asarray(Image.open(path).convert('RGB'), dtype=np.float32) / 255 for role, path in paths.items()}
            raw = motion.render_model(model, ev.render_camera(manifest, o, i))['render']
            model_image = raw.clamp(0, 1).permute(1, 2, 0).cpu().numpy()
            bicubic = F.interpolate(torch.from_numpy(images['lr'].transpose(2, 0, 1))[None].to('cuda'), size=images['hr'].shape[:2], mode='bicubic', align_corners=False, antialias=True)[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
            for label, pred, gt in [('SwinIR-to-HR', images['sr'], images['hr']), ('LR-bicubic-to-HR', bicubic, images['hr']),
                                    ('P1-to-HR', model_image, images['hr']), ('P1-to-SwinIR', model_image, images['sr'])]:
                rows.append(dict(arm=label, repeat=1, camera=o['camera_id'], frame=o['frame_index'], **decompose(pred, gt)))
            for label, image in [('P1-raw', raw), ('SwinIR', torch.from_numpy(images['sr'].transpose(2, 0, 1)).to('cuda'))]:
                d = F.interpolate(image[None], size=images['lr'].shape[:2], mode='bicubic', align_corners=False, antialias=True)[0].clamp(0, 1)
                error = d.permute(1, 2, 0).cpu().numpy().astype(np.float64) - images['lr'].astype(np.float64)
                closures.append(dict(arm=label, camera=o['camera_id'], frame=o['frame_index'], lr_l1=float(np.abs(error).mean()), lr_mse=float((error ** 2).mean())))
    write(OUT / 'train16.json', dict(status='completed_diagnostic', rows=rows, closures=closures, sources=sources,
                                   checkpoint=e['checkpoint'], renderer=renderer, rgb_forwards=16, parameter_updates=0))


def summarize():
    p = read(OUT / 'protocol.json'); rows = []
    for e in p['entries']:
        d = read(OUT / 'endpoints' / f"r{e['repeat']}_{e['arm']}.json"); assert d['entry'] == e and len(d['rows']) == 120
        rows += d['rows']
    assert len(rows) == 1560
    expected = read(OLD / 'final_summary.json')['averages'] + read(ROOT / 'output/dynamic_sr_same_observation_20260930/comparison_summary.json')['new_run']
    summaries = []
    for arm in ARMS:
        for scope in ['cam00', 'cam01', 'equal_camera_mean']:
            selected = [r for r in rows if r['arm'] == arm and (scope == 'equal_camera_mean' or r['camera'] == scope)]
            summary = dict(arm=arm, scope=scope, **aggregate(selected))
            old = next(v['psnr'] for v in expected if v['arm'] == arm and v['scope'] == scope)
            summary['main_psnr_gap'] = summary['legacy_float32_psnr'] - old
            assert abs(summary['main_psnr_gap']) < 1e-4, summary
            summaries.append(summary)
    gaps = []
    for scope in ['cam00', 'cam01', 'equal_camera_mean']:
        new = next(r for r in summaries if r['arm'] == 'P1-from-start' and r['scope'] == scope)
        for arm in ['U6000', 'J1', 'Async2', 'Sync2', 'T']:
            old = next(r for r in summaries if r['arm'] == arm and r['scope'] == scope)
            d = dict(reference=arm, scope=scope, mse_delta=new['mse'] - old['mse'])
            for b in BANDS:
                d[b + '_mse_delta'] = new[b + '_mse'] - old[b + '_mse']
                d[b + '_mse_relative_pct'] = 100 * (new[b + '_mse'] / old[b + '_mse'] - 1)
                d[b + '_share_of_total_delta_pct'] = 100 * d[b + '_mse_delta'] / d['mse_delta']
            gaps.append(d)
    train = read(OUT / 'train16.json'); ts = [dict(arm=arm, **aggregate([r for r in train['rows'] if r['arm'] == arm])) for arm in sorted(set(r['arm'] for r in train['rows']))]
    closures = [dict(arm=arm, **{k: float(np.mean([r[k] for r in train['closures'] if r['arm'] == arm])) for k in ['lr_l1', 'lr_mse']}) for arm in ['P1-raw', 'SwinIR']]
    csvfile(OUT / 'per_frame.csv', rows); csvfile(OUT / 'model_summary.csv', summaries); csvfile(OUT / 'p1_band_gaps.csv', gaps)
    write(OUT / 'summary.json', dict(status='completed_frequency_audit', protocol_sha256=sha(OUT / 'protocol.json'),
          summaries=summaries, p1_gaps=gaps, train16=ts, train16_closure=closures,
          max_main_psnr_gap=max(abs(r['main_psnr_gap']) for r in summaries),
          max_parseval_gap=max(r['parseval_gap'] for r in rows), rgb_forwards=976, cached_pairs=600, parameter_updates=0,
          costs={kind: read(OUT / f'{kind}_complete.json') for kind in ['cached', 'render']}))
    print(json.dumps(dict(status='completed_frequency_audit', pairs=len(rows), max_main_psnr_gap=max(abs(r['main_psnr_gap']) for r in summaries))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=['register', 'cached', 'render', 'summarize'])
    action = parser.parse_args().action
    if action == 'register': register()
    elif action == 'summarize': summarize()
    else: run(action)
