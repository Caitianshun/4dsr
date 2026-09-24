#!/usr/bin/env python3
"""Bounded CPU-only saved-image diagnostic; no training or model calls.

Fixed frame 40, training camera 02 and held-out camera 00 in three scenes.
Crop selection uses HR temporal variation and HR detail energy only.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch

PROJECT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('old_detail', PROJECT / 'experiments/dynamic_sr_20260918/analyze_sr_details.py')
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
OUT = PROJECT / 'output/dynamic_sr_20260919/final_assessment/detail_audit'
BASE = PROJECT / 'output/dynamic_sr_20260919'
SCENES = {'cook_spinach': 'n3dv_prepared/cook_spinach', 'cut_roasted_beef': 'n3dv_prepared/cut_roasted_beef', 'meetroom_discussion': 'meetroom_prepared/discussion'}
CASES = ['lr_long', 'sr_w10', 'sr_w10_dense', 'hr_oracle_w10']
EPS = (1 / 255) ** 2
BLOCK = 168


def select_crops(hr_high, changing):
    h, w = changing.shape
    candidates = []
    for y in range(0, h - BLOCK + 1, BLOCK // 2):
        for x in range(0, w - BLOCK + 1, BLOCK // 2):
            sl = np.s_[y:y + BLOCK, x:x + BLOCK]
            candidates.append(dict(x0=x, y0=y, x1=x + BLOCK, y1=y + BLOCK,
                changing_fraction=float(changing[sl].mean()),
                hr_high_energy=float(np.mean(hr_high[sl] ** 2))))
    result = {}
    for name, predicate in [('time_changing', lambda r: r['changing_fraction'] >= .5),
                            ('mostly_static', lambda r: r['changing_fraction'] <= .05)]:
        eligible = [r for r in candidates if predicate(r)]
        if not eligible:
            result[name] = None
        else:
            result[name] = max(eligible, key=lambda r: r['hr_high_energy'])
    return result


def make_panel(path, scene, camera, images, crops):
    width, display, pad, rowh = 1900, 224, 12, 514
    panel = Image.new('RGB', (width, 60 + 2 * rowh + 65), 'white')
    d = ImageDraw.Draw(panel)
    f, sm = old.font(17), old.font(13)
    is_test = camera == 'cam00'
    d.text((pad, 8), f'{scene} / {camera} / frame 0040 / step 18000 | crops selected using HR only', font=f, fill='black')
    d.text((pad, 33), 'Held-out LR -> SR is a privileged diagnostic, not a novel-view method.' if is_test else 'Training view: the same SwinIR image supervises SR branches; HR supervises oracle only.', font=sm, fill='black')
    keys = ['context', 'hr', 'bicubic', 'swinir', 'lr_long', 'sr_w10', 'sr_w10_dense', 'hr_oracle_w10']
    labels = ['HR context', 'HR reference', 'Actual LR bicubic', 'Actual LR -> SwinIR', 'LR-only', 'SR weight 1', 'SR weight 1 + dense', 'HR teacher oracle']
    for row, (name, crop) in enumerate(crops.items()):
        top = 60 + row * rowh
        if crop is None:
            d.text((pad, top), f'{name}: no eligible crop', font=f, fill='black')
            continue
        x0, y0, x1, y1 = [crop[k] for k in ['x0', 'y0', 'x1', 'y1']]
        sl = np.s_[y0:y1, x0:x1]
        d.text((pad, top), f'{name}: [{x0},{y0},{x1},{y1}) | changing fraction={crop["changing_fraction"]:.1%}; HR residual energy={crop["hr_high_energy"]:.6f}', font=sm, fill='black')
        for col, (key, label) in enumerate(zip(keys, labels)):
            left = pad + col * (display + pad)
            d.text((left, top + 23), label, font=sm, fill='black')
            if key == 'context':
                h, w = images['hr'].shape[:2]
                ctx = old.rgb_image(images['hr']).resize((display, display * h // w), Image.Resampling.LANCZOS)
                cd = ImageDraw.Draw(ctx)
                scale = display / w
                cd.rectangle((x0 * scale, y0 * scale, (x1 - 1) * scale, (y1 - 1) * scale), outline='red', width=2)
                panel.paste(ctx, (left, top + 46))
                d.text((left, top + 290), 'Same RMSE scale [0, 0.15].\n168 x 168 crop enlarged\nwith nearest display.\nGT-ranked diagnostic crop;\nnot a random example.', font=sm, fill='black')
            else:
                p = old.rgb_image(images[key][sl]).resize((display, display), Image.Resampling.NEAREST)
                panel.paste(p, (left, top + 46))
                e = old.err_image(images[key][sl], images['hr'][sl]).resize((display, display), Image.Resampling.NEAREST)
                panel.paste(e, (left, top + 46 + display))
    bar = np.tile(np.linspace(0, 255, 512).astype(np.uint8), (18, 1))
    bar = Image.fromarray(cv2.cvtColor(cv2.applyColorMap(bar, cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB))
    panel.paste(bar, (pad, panel.height - 55))
    d.text((pad, panel.height - 31), '0          RGB pixel RMSE          0.15 (clipped); lower is closer to HR', font=sm, fill='black')
    panel.save(path)


def measure(images, lr, mask):
    low = {k: old.resize(old.resize(v, lr.shape[:2]), v.shape[:2]) for k, v in images.items()}
    high = {k: images[k] - low[k] for k in images}
    fullerr = {k: old.mse_map(v, images['hr']) for k, v in images.items()}
    higherr = {k: old.mse_map(v, high['hr']) for k, v in high.items()}
    support = (fullerr['bicubic'] - fullerr['swinir'] > EPS) & (higherr['bicubic'] - higherr['swinir'] > EPS)
    regions = {'full': np.ones_like(mask), 'hr_supported_sr_improvement': support,
               'time_changing': mask, 'remaining': ~mask}
    metrics = {}
    for name, m in regions.items():
        methods = {}
        for k in images:
            if k == 'hr':
                continue
            lo, hi = low[k] - low['hr'], high[k] - high['hr']
            vals = dict(mse=old.mean(fullerr[k], m), low_mse=old.mean(np.mean(lo ** 2, axis=-1), m),
                high_mse=old.mean(np.mean(hi ** 2, axis=-1), m),
                cross_term=old.mean(2 * np.mean(lo * hi, axis=-1), m),
                high_correlation=old.corr(old.stats(high[k], high['hr'], m)),
                to_sr_mse=old.mean(old.mse_map(images[k], images['swinir']), m))
            vals['psnr_from_mse'] = -10 * math.log10(vals['mse'])
            if abs(vals['low_mse'] + vals['high_mse'] + vals['cross_term'] - vals['mse']) > 3e-8:
                raise AssertionError('Error decomposition is inconsistent')
            methods[k] = vals
        metrics[name] = dict(pixels=int(m.sum()), fraction=float(m.mean()), methods=methods)
    return metrics, support, high


def audit(scene, datarel):
    data = PROJECT / 'data/dynamic_sr' / datarel
    manifest = json.loads((data / 'manifest.json').read_text())
    records = []
    for camera in ['cam02', 'cam00']:
        obs = [o for o in manifest['observations'] if o['camera_id'] == camera and o['frame_index'] == 40][0]
        files = {'hr': data / obs['hr_path'], 'lr': data / obs['lr_path']}
        if camera == 'cam02':
            files['swinir'] = data / 'sr_swinir_x4' / camera / '0040.png'
        elif scene == 'meetroom_discussion':
            files['swinir'] = BASE / f'{scene}_real_lr_reference/swinir/0040.png'
        else:
            files['swinir'] = PROJECT / 'output/dynamic_sr_20260918' / f'{scene}_pilot_v1_sr_reference_diagnostic/swinir/0040.png'
        for case in CASES:
            files[case] = BASE / (f'{scene}_{case}/train_renders_18000/{camera}/0040.png' if camera == 'cam02' else f'{scene}_{case}_evaluation/predictions/0040.png')
        loaded = {k: old.read(p) for k, p in files.items()}
        lr = loaded.pop('lr')
        loaded['bicubic'] = old.resize(lr, loaded['hr'].shape[:2])
        assert len({v.shape for v in loaded.values()}) == 1
        if camera == 'cam00':
            ev = json.loads((BASE / f'{scene}_sr_w10_evaluation/metrics.json').read_text())
            mask_path = Path(ev['evaluation_cache']) / 'dynamic_mask.png'
            changing = np.asarray(Image.open(mask_path)) > 0
            mask_source = dict(path=str(mask_path), sha256=old.sha(mask_path), definition='Existing 60-frame HR temporal std mask; not semantic motion')
        else:
            selected = [o for o in manifest['observations'] if o['camera_id'] == camera and o['frame_index'] in [0, 40, 80, 118]]
            stack = np.stack([old.read(data / o['hr_path']) for o in selected])
            changing = np.max(np.std(stack, axis=0), axis=-1) > .025
            changing = cv2.morphologyEx(changing.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            changing = cv2.morphologyEx(changing, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
            changing = cv2.dilate(changing, np.ones((5, 5), np.uint8)) > 0
            mask_source = dict(definition='Diagnostic four-frame HR temporal std max RGB > .025, open3 close5 dilate5; not standard evaluator ROI', files=[dict(path=str(data / o['hr_path']), sha256=old.sha(data / o['hr_path'])) for o in selected])
        metrics, support, high = measure(loaded, lr, changing)
        crops = select_crops(high['hr'], changing)
        panel = OUT / f'{scene}_{camera}_0040.png'
        make_panel(panel, scene, camera, loaded, crops)
        Image.fromarray((support.astype(np.uint8) * 255)).save(OUT / f'{scene}_{camera}_0040_hr_supported_mask.png')
        records.append(dict(camera=camera, frame=40, split=obs['split'], files={k: dict(path=str(p), sha256=old.sha(p)) for k, p in files.items()},
            mask_source=mask_source, crops=crops, metrics=metrics, panel=str(panel)))
    return dict(manifest_path=str(data / 'manifest.json'), manifest_sha256=old.sha(data / 'manifest.json'), observations=records)


def main():
    start = time.monotonic()
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'metrics.json').exists():
        raise FileExistsError('Do not overwrite previous diagnostic')
    result = dict(schema='final_detail_audit_v1', source_sha256=old.sha(__file__),
        protocol=dict(frame=40, cameras=['cam02', 'cam00'], endpoint=18000,
            scope='Six fixed observation diagnostics, not full benchmark aggregates or causal proof',
            image_precision='Saved uint8 RGB PNG / 255; bicubic newly computed float from actual LR',
            sr_reference='Training SwinIR teacher for cam02; privileged actual held-out LR SwinIR for cam00',
            low='U(D(X)) with torch bicubic align_corners=False antialias=True, clamp after each resize',
            high='X-U(D(X)); scale-defined residual, not orthogonal frequency projector',
            decomposition='MSE_total=MSE_low+MSE_high+2mean(error_low*error_high)',
            correlation='Pearson over RGB high residuals in the specified mask',
            support=f'Per-pixel RGB MSE bicubic minus SR > {EPS}, both full signal and high residual; HR-referenced evaluation mask only',
            support_limitation='Reference-supported improvement in observed RGB does not establish geometric truth, cross-view consistency or train-time computable confidence',
            crop_selection=dict(block=BLOCK, stride=BLOCK//2, dynamic_min=.5, static_max=.05, ranking='Maximum HR high residual energy; no model-error selection'),
            png_warning='Not interchangeable with evaluator float PSNR/SSIM/LPIPS; no new perceptual model evaluation here'),
        scenes={s: audit(s, p) for s, p in SCENES.items()})
    result['elapsed_seconds'] = time.monotonic() - start
    (OUT / 'metrics.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    (OUT / 'analysis_source.py').write_text(Path(__file__).read_text())
    print(json.dumps({s: [{"camera": r['camera'], "full": r['metrics']['full'], "supported": r['metrics']['hr_supported_sr_improvement'], "crops": r['crops']} for r in x['observations']] for s, x in result['scenes'].items()}, indent=2))


if __name__ == '__main__':
    main()
