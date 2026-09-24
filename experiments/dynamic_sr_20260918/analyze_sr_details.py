#!/usr/bin/env python3
"""CPU-only, evaluation-only error audit of actual-LR SwinIR diagnostics.

This compares saved 8-bit images. The direct-LR SR images see the held-out
camera and are a diagnostic reference, not an admissible novel-view baseline
or an upper bound on multiview reconstruction. No model is trained here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn.functional as F

PROJECT = Path(__file__).resolve().parents[2]
SCENES = ['cook_spinach', 'cut_roasted_beef']
FIXED_FRAMES = [0, 40, 80, 118]
EPS = (1 / 255) ** 2
BLOCK = 168


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return np.asarray(Image.open(path).convert('RGB'), dtype=np.float32) / 255


def resize(x, size):
    # size is (height, width); same floating operator as the prepared protocol.
    tensor = torch.from_numpy(np.ascontiguousarray(x)).permute(2, 0, 1)[None]
    with torch.inference_mode():
        y = F.interpolate(tensor, size=size, mode='bicubic', align_corners=False,
                          antialias=True).clamp(0, 1)
    return y[0].permute(1, 2, 0).numpy()


def mse_map(x, y):
    return np.mean((x - y) ** 2, axis=2)


def mean(x, mask):
    return float(x[mask].mean(dtype=np.float64))


def weighted_mean(x, w):
    return float(np.sum(x * w, dtype=np.float64) / np.sum(w, dtype=np.float64))


def stats(x, y, mask):
    a, b = x[mask].astype(np.float64).ravel(), y[mask].astype(np.float64).ravel()
    return dict(n=len(a), sx=float(a.sum()), sy=float(b.sum()),
                sxx=float(np.dot(a, a)), syy=float(np.dot(b, b)), sxy=float(np.dot(a, b)))


def corr(s):
    c = s['sxy'] - s['sx'] * s['sy'] / s['n']
    a = s['sxx'] - s['sx'] ** 2 / s['n']
    b = s['syy'] - s['sy'] ** 2 / s['n']
    return float(c / math.sqrt(a * b)) if a > 1e-15 and b > 1e-15 else None


def add_stats(a, b):
    for key, value in b.items():
        a[key] = a.get(key, 0) + value


def font(size):
    for p in ['/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
              '/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf']:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def rgb_image(x):
    return Image.fromarray(np.rint(np.clip(x, 0, 1) * 255).astype(np.uint8))


def err_image(pred, gt):
    # Fixed image-wise RGB RMSE scale [0, .15], never normalize each patch.
    e = np.sqrt(mse_map(pred, gt))
    z = np.rint(np.minimum(e / .15, 1) * 255).astype(np.uint8)
    return Image.fromarray(cv2.cvtColor(cv2.applyColorMap(z, cv2.COLORMAP_INFERNO),
                                      cv2.COLOR_BGR2RGB))


def grid_candidates(gt, bic, sr, joint, dynamic, high):
    h, w = gt.shape[:2]
    eb, es, ej = [mse_map(x, gt) for x in [bic, sr, joint]]
    hf_b, hf_s = [mse_map(high[x], high['hr']) for x in ['bicubic', 'swinir']]
    result = []
    assert w % BLOCK == 0 and h % BLOCK == 0
    for y in range(0, h, BLOCK):
        for x in range(0, w, BLOCK):
            sl = np.s_[y:y + BLOCK, x:x + BLOCK]
            result.append(dict(x0=x, y0=y, x1=x + BLOCK, y1=y + BLOCK,
                               sr_mse=float(es[sl].mean()), bicubic_mse=float(eb[sl].mean()),
                               joint_mse=float(ej[sl].mean()),
                               improvement_mse=float((eb - es)[sl].mean()),
                               degradation_mse=float((es - eb)[sl].mean()),
                               time_changing_fraction=float(dynamic[sl].mean()),
                               sr_high_residual_mse=float(hf_s[sl].mean()),
                               bicubic_high_residual_mse=float(hf_b[sl].mean())))
    return result


def make_panel(path, scene, frame, images, rankings, criteria_only=None):
    scale, pad, title_h, row_h = 224, 12, 40, 500
    width = 6 * (scale + pad) + pad
    criteria = [('sr_mse', 'Largest SR residual'), ('improvement_mse', 'Largest SR improvement'),
                ('degradation_mse', 'Worst signed SR-minus-bicubic error')]
    if rankings.get('time_changing_residual'):
        criteria.append(('time_changing_residual', 'Largest SR residual among blocks >=50% time-changing'))
    if criteria_only:
        criteria = [item for item in criteria if item[0] in criteria_only]
    panel = Image.new('RGB', (width, title_h + len(criteria) * row_h + 65), 'white')
    d = ImageDraw.Draw(panel)
    f, small = font(16), font(13)
    d.text((pad, 8), f'{scene} / cam00 / frame {frame:04d} | selected by fixed grid scores; not random examples', font=f, fill='black')
    labels = ['Context / location', 'Reference HR', 'Actual LR nearest', 'Actual LR bicubic', 'Actual LR -> SwinIR', 'Novel-view joint 4DGS']
    gt = images['hr']
    for row, (key, title) in enumerate(criteria):
        item = rankings[key][0]
        x0, y0, x1, y1 = [item[k] for k in ['x0', 'y0', 'x1', 'y1']]
        sl = np.s_[y0:y1, x0:x1]
        top = title_h + row * row_h
        suffix = ' (no net-worse block)' if key == 'degradation_mse' and item['degradation_mse'] <= 0 else ''
        d.text((pad, top), f'{title}{suffix}: [{x0},{y0},{x1},{y1}) | SR MSE={item["sr_mse"]:.7f}; bic-SR={item["improvement_mse"]:+.7f}; changing={item["time_changing_fraction"]:.1%}', font=small, fill='black')
        context = rgb_image(gt).resize((scale, scale * gt.shape[0] // gt.shape[1]), Image.Resampling.LANCZOS)
        cd = ImageDraw.Draw(context)
        s = scale / gt.shape[1]
        cd.rectangle((x0 * s, y0 * s, (x1 - 1) * s, (y1 - 1) * s), outline='red', width=2)
        panel.paste(context, (pad, top + 48))
        for col, (key_image, label) in enumerate(zip(['context', 'hr', 'nearest', 'bicubic', 'swinir', 'joint'], labels)):
            left = pad + col * (scale + pad)
            d.text((left, top + 24), label, font=small, fill='black')
            if key_image == 'context':
                d.text((left, top + 240), 'Same [0, 0.15] RGB RMSE\ncolor scale in all panels.\nRGB shown with nearest\ndisplay enlargement (4/3x).', font=small, fill='black')
                continue
            crop = images[key_image][sl]
            display = rgb_image(crop).resize((scale, scale), Image.Resampling.NEAREST)
            panel.paste(display, (left, top + 48))
            error = err_image(crop, gt[sl]).resize((scale, scale), Image.Resampling.NEAREST)
            panel.paste(error, (left, top + 48 + scale))
    bottom = title_h + len(criteria) * row_h
    gradient = np.tile(np.linspace(0, 255, 500).astype(np.uint8), (18, 1))
    bar = Image.fromarray(cv2.cvtColor(cv2.applyColorMap(gradient, cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB))
    panel.paste(bar, (pad, bottom + 4))
    d.text((pad, bottom + 25), '0                  RGB pixel RMSE                  0.15 (clipped)', font=small, fill='black')
    d.text((540, bottom + 4), 'Each crop is 168x168 HR pixels. Joint does not see held-out LR.\nDirect SR does: useful diagnostic; not a fair novel-view baseline.', font=small, fill='black')
    panel.save(path)


def rebuild_panels(result, out):
    """Refresh labels and add a predeclared time-changing subset; no metrics rerun."""
    for scene, value in result['scenes'].items():
        sources = {row['frame']: row['files'] for row in value['sources']}
        for frame, selection in value['candidates'].items():
            eligible = [item for item in selection['grid'] if item['time_changing_fraction'] >= .5]
            selection['top5']['time_changing_residual'] = sorted(eligible, key=lambda x: x['sr_mse'], reverse=True)[:5]
            images = {key: read(item['path']) for key, item in sources[int(frame)].items()}
            h, w = images['hr'].shape[:2]
            images['nearest'] = cv2.resize(images['lr'], (w, h), interpolation=cv2.INTER_NEAREST)
            make_panel(out / f'{scene}_frame{int(frame):04d}_selected_details.png', scene, int(frame), images, selection['top5'])
            if selection['top5']['time_changing_residual']:
                make_panel(out / f'{scene}_frame{int(frame):04d}_time_changing_detail.png',
                           scene, int(frame), images, selection['top5'], ['time_changing_residual'])
    result['panel_revision'] = dict(version=3, script_sha256=sha(__file__),
        change='Signed worst-block wording; add largest residual among grid blocks with time-changing fraction >= 0.5',
        analysis_unchanged=True, original_analysis_source=str(out / 'analysis_source_v1.py'))
    temp = out / 'details.json.tmp'
    temp.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    temp.replace(out / 'details.json')


def audit_scene(scene, out):
    data = PROJECT / 'data/dynamic_sr/n3dv_prepared' / scene
    manifest_path = data / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    base = PROJECT / 'output/dynamic_sr_20260918'
    direct = base / f'{scene}_pilot_v1_sr_reference_diagnostic'
    joint_dir = base / f'{scene}_pilot_v1_joint/evaluation'
    joint_metrics = json.loads((joint_dir / 'metrics.json').read_text())
    cache = Path(joint_metrics['evaluation_cache'])
    mask = np.asarray(Image.open(cache / 'dynamic_mask.png')) > 0
    regions = {'full': np.ones_like(mask), 'time_changing': mask, 'remaining': ~mask}
    rows, candidates, hashes, source_rows = [], {}, {}, []
    corr_acc = {region: {method: {} for method in ['bicubic', 'swinir', 'joint']} for region in regions}
    observations = sorted([o for o in manifest['observations'] if o['split'] == 'test'],
                          key=lambda x: x['frame_index'])
    assert len(observations) == 60 and {o['camera_id'] for o in observations} == {'cam00'}
    for o in observations:
        frame = o['frame_index']
        paths = {'hr': data / o['hr_path'], 'lr': data / o['lr_path'],
                 'bicubic': direct / 'bicubic' / f'{frame:04d}.png',
                 'swinir': direct / 'swinir' / f'{frame:04d}.png',
                 'joint': joint_dir / 'predictions' / f'{frame:04d}.png'}
        images = {key: read(p) for key, p in paths.items()}
        source_rows.append(dict(frame=frame, files={k: dict(path=str(p), sha256=sha(p)) for k, p in paths.items()}))
        hr, lr, bic, sr, joint = [images[k] for k in ['hr', 'lr', 'bicubic', 'swinir', 'joint']]
        assert hr.shape == bic.shape == sr.shape == joint.shape and mask.shape == hr.shape[:2]
        h, w = hr.shape[:2]
        low_lr = {k: resize(images[k], lr.shape[:2]) for k in ['hr', 'bicubic', 'swinir', 'joint']}
        low_hr = {k: resize(v, hr.shape[:2]) for k, v in low_lr.items()}
        high = {k: images[k] - low_hr[k] for k in low_hr}
        errors = {k: mse_map(images[k], hr) for k in ['bicubic', 'swinir', 'joint']}
        gain = errors['bicubic'] - errors['swinir']
        scene_row = dict(frame=frame, regions={})
        for region, region_mask in regions.items():
            lr_weights = cv2.resize(region_mask.astype(np.float32), (lr.shape[1], lr.shape[0]), interpolation=cv2.INTER_AREA)
            b = dict(pixels=int(region_mask.sum()),
                     improvement_fraction=mean((gain > EPS).astype(np.float32), region_mask),
                     degradation_fraction=mean((gain < -EPS).astype(np.float32), region_mask),
                     negligible_fraction=mean((np.abs(gain) <= EPS).astype(np.float32), region_mask),
                     improvement_fraction_zero_threshold=mean((gain > 0).astype(np.float32), region_mask),
                     degradation_fraction_zero_threshold=mean((gain < 0).astype(np.float32), region_mask),
                     mean_gain_mse=mean(gain, region_mask),
                     sr_residual_mse=mean(errors['swinir'], region_mask),
                     bicubic_residual_mse=mean(errors['bicubic'], region_mask),
                     joint_residual_mse=mean(errors['joint'], region_mask),
                     joint_to_direct_sr_mse=mean(mse_map(joint, sr), region_mask),
                     lr_consistency_mse={}, decomposition={})
            for method in ['hr', 'bicubic', 'swinir', 'joint']:
                b['lr_consistency_mse'][method] = weighted_mean(mse_map(low_lr[method], lr), lr_weights)
            for method in ['bicubic', 'swinir', 'joint']:
                low_error = low_hr[method] - low_hr['hr']
                high_error = high[method] - high['hr']
                low_mse = mean(np.mean(low_error ** 2, axis=2), region_mask)
                high_mse = mean(np.mean(high_error ** 2, axis=2), region_mask)
                cross = mean(2 * np.mean(low_error * high_error, axis=2), region_mask)
                s = stats(high[method], high['hr'], region_mask)
                add_stats(corr_acc[region][method], s)
                b['decomposition'][method] = dict(low_mse=low_mse, high_mse=high_mse,
                    cross_term=cross, reconstructed_total_mse=low_mse + high_mse + cross,
                    high_correlation=corr(s),
                    predicted_high_energy=mean(np.mean(high[method] ** 2, axis=2), region_mask),
                    target_high_energy=mean(np.mean(high['hr'] ** 2, axis=2), region_mask))
                if abs(low_mse + high_mse + cross - mean(errors[method], region_mask)) > 2e-8:
                    raise AssertionError('Decomposition mismatch')
            scene_row['regions'][region] = b
        rows.append(scene_row)
        if frame in FIXED_FRAMES:
            grid = grid_candidates(hr, bic, sr, joint, mask, high)
            rankings = {key: sorted(grid, key=lambda x: x[key], reverse=True)[:5]
                        for key in ['sr_mse', 'improvement_mse', 'degradation_mse']}
            candidates[str(frame)] = dict(grid=grid, top5=rankings)
            images['nearest'] = cv2.resize(lr, (w, h), interpolation=cv2.INTER_NEAREST)
            make_panel(out / f'{scene}_frame{frame:04d}_selected_details.png', scene, frame, images, rankings)

    def recursive_mean(values):
        first = values[0]
        if isinstance(first, dict):
            return {key: recursive_mean([v[key] for v in values]) for key in first}
        return float(np.mean(values)) if first is not None else None
    aggregate = {region: recursive_mean([r['regions'][region] for r in rows]) for region in regions}
    for region in regions:
        a = aggregate[region]
        a['sr_mse_reduction_vs_bicubic_fraction'] = 1 - a['sr_residual_mse'] / a['bicubic_residual_mse']
        a['sr_error_energy_share'] = (a['sr_residual_mse'] * a['pixels'] /
                                     (aggregate['full']['sr_residual_mse'] * aggregate['full']['pixels']))
        for method in ['bicubic', 'swinir', 'joint']:
            a['decomposition'][method]['high_correlation_pooled'] = corr(corr_acc[region][method])
            a['decomposition'][method]['lr_consistency_psnr_from_mse'] = -10 * math.log10(a['lr_consistency_mse'][method])
        a['sr_high_mse_reduction_vs_bicubic_fraction'] = 1 - a['decomposition']['swinir']['high_mse'] / a['decomposition']['bicubic']['high_mse']
    return dict(scene=scene, frame_count=len(rows), camera='cam00', manifest_sha256=sha(manifest_path),
                fixed_mask=str(cache / 'dynamic_mask.png'), fixed_mask_sha256=sha(cache / 'dynamic_mask.png'),
                time_changing_pixel_fraction=float(mask.mean()), aggregate=aggregate,
                rows=rows, candidates=candidates, sources=source_rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenes', nargs='+', default=SCENES, choices=SCENES)
    parser.add_argument('--out', type=Path, default=PROJECT / 'output/dynamic_sr_20260918/sr_detail_audit')
    parser.add_argument('--wait-seconds', type=int, default=600)
    parser.add_argument('--panels-only', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.panels_only:
        rebuild_panels(json.loads((args.out / 'details.json').read_text()), args.out)
        return
    if (args.out / 'details.json').exists():
        raise FileExistsError(args.out / 'details.json')
    (args.out / 'analysis_source_v1.py').write_text(Path(__file__).read_text())
    start = time.monotonic()
    # Local readiness wait consumes no model turns; never trains or touches GPUs.
    while True:
        ready = all((PROJECT / 'output/dynamic_sr_20260918' /
                    f'{scene}_pilot_v1_sr_reference_diagnostic/metrics.json').is_file()
                    for scene in args.scenes)
        if ready:
            break
        if time.monotonic() - start > args.wait_seconds:
            raise TimeoutError('Direct-LR SR diagnostics not complete')
        time.sleep(5)
    result = dict(schema='sr_detail_audit_v1', script_sha256=sha(__file__),
       protocol=dict(rgb_range=[0, 1], image_precision='saved uint8 PNG / 255',
        input='actual held-out LR for bicubic/SwinIR; train-only inputs for joint 4DGS',
        scope='diagnostic only, not a fair novel-view baseline or an upper bound',
        pixel_comparison='mean over RGB squared error; bicubic_error minus SR_error',
        change_threshold_mse=EPS,
        threshold_interpretation='(1/255)^2 in RGB [0,1]; numerical reporting tolerance, not a visibility threshold',
        mask='fixed GT temporal standard-deviation mask from existing evaluator; not semantic motion',
        low_lr_region_weights='area-downsampled HR mask; fractional pixel footprint weights',
        downsample='torch bicubic antialias=True align_corners=False, then clamp [0,1]',
        upsample='torch bicubic antialias=True align_corners=False, then clamp [0,1]',
        high_residual='H(X)=X-U(D(X)); scale-defined residual, not an orthogonal frequency projector',
        decomposition='total MSE=low MSE+high MSE+2mean(low_error*high_error); cross term retained',
        correlation='Pearson across flattened RGB high residuals, pooled across all 60 frames',
        grid=dict(block_size=BLOCK, nonoverlapping=True, fixed_frames=FIXED_FRAMES,
                  selected_by=['largest SR residual MSE','largest bicubic-minus-SR MSE','largest SR-minus-bicubic MSE'],
                  examples='GT-ranked diagnostic crops; intentionally selected, not representative samples'),
        error_colormap=dict(quantity='RGB pixel RMSE', minimum=0, maximum=.15, colormap='inferno', saturation=True),
        warning='Rendered output PNGs and saved bicubic PNGs include 8-bit quantization; main evaluator scores floating predictions where available.'),
       scenes={scene: audit_scene(scene, args.out) for scene in args.scenes})
    result['elapsed_seconds'] = time.monotonic() - start
    (args.out / 'details.json').write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    print(json.dumps({scene: data['aggregate'] for scene, data in result['scenes'].items()}, indent=2))


if __name__ == '__main__':
    main()
