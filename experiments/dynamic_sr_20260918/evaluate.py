#!/usr/bin/env python3
"""Fixed test-camera evaluation for the public N3DV dynamic-SR pilot.

No training HR is opened. GT-derived masks/flows live in an evaluation-only,
checkpoint-independent cache. Use --prepare-cache-only for CPU cache creation,
or --sampling-only for a cheap fixed-checkpoint rasterization diagnosis.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from n3dv_data import N3DVPreparedDataset, load_manifest, observation_to_4dgs_camera


VERSION = 'n3dv_eval_v1'
PROJECT = Path(__file__).resolve().parents[2]


def file_sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    tmp.replace(path)


def read_rgb(path):
    with Image.open(path) as im:
        return np.asarray(im.convert('RGB')).astype(np.float32) / 255.0


def write_rgb(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.rint(np.clip(value, 0, 1) * 255).astype(np.uint8)).save(path)


def image_array(tensor):
    return tensor.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy()


def ssim_map_rgb(x, y):
    """11x11 Gaussian, sigma=1.5, population covariance, RGB channel mean.

    Scores exclude a five-pixel outer border, so no artificial padding enters
    the reported full-image or ROI SSIM. Unlike blacked-out ROI images, context
    near the dynamic-mask boundary remains the real image content.
    """
    filt = lambda z: cv2.GaussianBlur(z, (11, 11), 1.5, borderType=cv2.BORDER_REFLECT_101)
    ux, uy = filt(x), filt(y)
    vx, vy, cov = filt(x * x) - ux * ux, filt(y * y) - uy * uy, filt(x * y) - ux * uy
    val = ((2 * ux * uy + .01 ** 2) * (2 * cov + .03 ** 2)) / (
        (ux * ux + uy * uy + .01 ** 2) * (vx + vy + .03 ** 2))
    return val.mean(axis=2)


def masked_mean(x, mask):
    return float(x[mask].mean()) if np.any(mask) else None


def metric_psnr(mse):
    return float(-10 * np.log10(max(float(mse), 1e-12))) if mse is not None else None


def spatial_metrics(pred, gt, dynamic, lpips_model=None, lpips_device='cuda', bbox=None):
    err = ((pred - gt) ** 2).mean(axis=2)
    ssim = ssim_map_rgb(pred, gt)
    interior = np.zeros(dynamic.shape, bool)
    interior[5:-5, 5:-5] = True
    masks = {'full': np.ones(dynamic.shape, bool), 'dynamic': dynamic, 'static': ~dynamic}
    result = {}
    for name, mask in masks.items():
        mse = masked_mean(err, mask)
        result[name] = {'mse': mse, 'psnr': metric_psnr(mse),
                        'ssim': masked_mean(ssim, mask & interior),
                        'pixel_count': int(mask.sum())}
    if lpips_model is not None:
        tx = torch.from_numpy(np.ascontiguousarray(pred)).permute(2, 0, 1)[None].to(lpips_device) * 2 - 1
        ty = torch.from_numpy(np.ascontiguousarray(gt)).permute(2, 0, 1)[None].to(lpips_device) * 2 - 1
        with torch.inference_mode():
            result['full']['lpips_alex'] = float(lpips_model(tx, ty).item())
            # Official LPIPS spatial mode retains the real full-image context.
            # Masking this upsampled map is a named regional extension, not the
            # standard scalar LPIPS and not blacking pixels outside the ROI.
            lpips_model.spatial = True
            try:
                spatial = lpips_model(tx, ty)[0, 0].cpu().numpy()
            finally:
                lpips_model.spatial = False
        for name, mask in masks.items():
            result[name]['lpips_alex_spatial_mask'] = masked_mean(spatial, mask)
    return result


def bbox_of_mask(mask, minimum=64):
    yy, xx = np.where(mask)
    if len(xx) == 0:
        return None
    h, w = mask.shape
    x0, x1, y0, y1 = int(xx.min()), int(xx.max()) + 1, int(yy.min()), int(yy.max()) + 1
    def expand(lo, hi, limit):
        size = min(limit, max(minimum, hi - lo))
        lo = min(max(0, (lo + hi - size) // 2), limit - size)
        return lo, lo + size
    x0, x1 = expand(x0, x1, w)
    y0, y1 = expand(y0, y1, h)
    return [x0, y0, x1, y1]


def backward_warp(source, target_to_source):
    h, w = target_to_source.shape[:2]
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    return cv2.remap(source, xx + target_to_source[..., 0], yy + target_to_source[..., 1],
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def dis_pair(previous, current, scale=.5, alpha=.01, beta=.5):
    h, w = previous.shape[:2]
    sh, sw = max(16, round(h * scale)), max(16, round(w * scale))
    def gray(x):
        rgb = np.rint(np.clip(x, 0, 1) * 255).astype(np.uint8)
        return cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), (sw, sh), interpolation=cv2.INTER_AREA)
    a, b = gray(previous), gray(current)
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    forward = dis.calc(a, b, None)
    backward = dis.calc(b, a, None)
    warped_forward = backward_warp(forward, backward)
    fb_squared = ((backward + warped_forward) ** 2).sum(axis=2)
    threshold = alpha * ((backward ** 2).sum(axis=2) + (warped_forward ** 2).sum(axis=2)) + beta
    yy, xx = np.mgrid[:sh, :sw]
    sx, sy = xx + backward[..., 0], yy + backward[..., 1]
    valid = (fb_squared <= threshold) & (sx >= 1) & (sx < sw - 2) & (sy >= 1) & (sy < sh - 2)
    valid = cv2.erode(valid.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    return backward.astype(np.float32), valid, fb_squared


def full_flow(flow, valid, height, width):
    fh, fw = flow.shape[:2]
    out = cv2.resize(flow, (width, height), interpolation=cv2.INTER_LINEAR)
    out[..., 0] *= width / fw
    out[..., 1] *= height / fh
    mask = cv2.resize(valid.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)
    yy, xx = np.mgrid[:height, :width]
    mask &= (xx + out[..., 0] >= 1) & (xx + out[..., 0] < width - 2)
    mask &= (yy + out[..., 1] >= 1) & (yy + out[..., 1] < height - 2)
    return out, mask


def temporal_metrics(previous_pred, pred, previous_gt, gt, flow, valid, dynamic):
    """L1[(P_t-W(P_prev))-(G_t-W(G_prev))] on fixed valid correspondences."""
    wp = backward_warp(previous_pred, flow)
    wg = backward_warp(previous_gt, flow)
    pd, gd = pred - wp, gt - wg
    error = np.abs(pd - gd).mean(axis=2)
    raw = np.abs(pd).mean(axis=2)
    actual = np.abs(gd).mean(axis=2)
    out = {}
    for name, roi in [('full', np.ones_like(dynamic)), ('dynamic', dynamic), ('static', ~dynamic)]:
        mask = valid & roi
        out[name] = {'gt_relative_warp_l1': masked_mean(error, mask),
                     'prediction_warp_l1': masked_mean(raw, mask), 'gt_warp_l1': masked_mean(actual, mask),
                     'valid_pixel_count': int(mask.sum()), 'roi_pixel_count': int(roi.sum()),
                     'valid_fraction_of_roi': float(mask.sum() / max(1, roi.sum()))}
    return out


def cache_identity(manifest, observations, args):
    root = Path(manifest['_root'])
    return {'version': VERSION, 'manifest_sha256': file_sha(manifest['_manifest_path']),
            'split': 'test', 'std_threshold': args.dynamic_threshold,
            'flow_scale': args.flow_scale, 'flow_preset': 'DIS_MEDIUM',
            'fb_alpha': .01, 'fb_beta': .5, 'opencv_version': cv2.__version__,
            'gt_files': [{'path': o['hr_path'], 'sha256': file_sha(root / o['hr_path'])}
                         for o in observations]}


def prepare_cache(manifest, observations, args):
    root = Path(manifest['_root'])
    identity = cache_identity(manifest, observations, args)
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    cache = (args.cache_dir or (root / 'evaluation_cache')) / key
    cache.mkdir(parents=True, exist_ok=True)
    info_path = cache / 'cache.json'
    if info_path.exists():
        info = json.loads(info_path.read_text())
        if info['identity'] != identity:
            raise RuntimeError('Evaluation cache identity collision')
        needed = [cache / 'dynamic_mask.png'] + [cache / x['file'] for x in info['flow_pairs']]
        if not all(p.is_file() for p in needed):
            raise RuntimeError('Incomplete fixed evaluation cache; do not silently mix caches')
        return cache, info
    mean = m2 = None
    for i, obs in enumerate(observations):
        x = read_rgb(root / obs['hr_path']).astype(np.float64)
        if mean is None:
            mean, m2 = np.zeros_like(x), np.zeros_like(x)
        delta = x - mean
        mean += delta / (i + 1)
        m2 += delta * (x - mean)
    std = np.sqrt((m2 / max(1, len(observations))).mean(axis=2)).astype(np.float32)
    mask = (std > args.dynamic_threshold).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8)).astype(bool)
    Image.fromarray(mask.astype(np.uint8) * 255).save(cache / 'dynamic_mask.png')
    np.save(cache / 'temporal_std.npy', std)
    flow_pairs = []
    previous = read_rgb(root / observations[0]['hr_path'])
    (cache / 'flows').mkdir(exist_ok=True)
    for i, obs in enumerate(observations[1:], start=1):
        current = read_rgb(root / obs['hr_path'])
        flow, valid, fb = dis_pair(previous, current, args.flow_scale)
        name = f"flows/{observations[i-1]['frame_index']:04d}_{obs['frame_index']:04d}.npz"
        np.savez_compressed(cache / name, backward=flow, valid=valid.astype(np.uint8))
        flow_pairs.append({'previous_frame': observations[i-1]['frame_index'],
                           'current_frame': obs['frame_index'], 'file': name,
                           'flow_grid_hw': list(valid.shape), 'half_grid_valid_fraction': float(valid.mean()),
                           'fb_squared_median': float(np.median(fb))})
        previous = current
        if i % 10 == 0:
            print(f'GT_FLOW_CACHE {i}/{len(observations)-1}', flush=True)
    info = {'identity': identity, 'cache_key': key, 'dynamic_fraction': float(mask.mean()),
            'dynamic_bbox_xyxy': bbox_of_mask(mask),
            'mask_definition': 'sqrt(mean_RGB population temporal variance)>threshold; open3,close5,dilate5; fixed across models; not semantic segmentation',
            'flow_definition': 'GT RGB->gray uint8; half-size INTER_AREA; DIS medium; target-to-previous flow; FB squared error <= .01*(|b|²+|warped_f|²)+.5; in-bounds and 3x3 erosion; flow then scaled to HR',
            'flow_pairs': flow_pairs}
    write_json(info_path, info)
    return cache, info


def aggregate(rows, key):
    """Frame/pair equal averages; PSNR of pooled MSE is separately named."""
    groups = sorted(set(g for r in rows for g in r.get(key, {})))
    out = {}
    for group in groups:
        fields = sorted(set(k for r in rows for k, v in r.get(key, {}).get(group, {}).items()
                            if isinstance(v, (int, float)) or v is None))
        out[group] = {}
        for field in fields:
            values = [r[key][group][field] for r in rows if group in r.get(key, {})
                      and r[key][group].get(field) is not None]
            if values:
                out[group][field + '_mean'] = float(np.mean(values))
                out[group][field + '_std_frames'] = float(np.std(values))
        if 'mse_mean' in out[group]:
            out[group]['psnr_from_mean_mse'] = metric_psnr(out[group]['mse_mean'])
    return out


def run_sampling(g, manifest, args):
    from common import render_image, resized_camera, downsample
    data = N3DVPreparedDataset(manifest, 'test', 'lr')
    data.observations.sort(key=lambda x: x['frame_index'])
    rows = []
    begin = time.monotonic()
    for i in range(len(data)):
        item = data[i]
        cam = observation_to_4dgs_camera(item, i)
        gt = item['image'].cuda()
        h, w = gt.shape[-2:]
        native = render_image(g, cam)['render'].clamp(0, 1)
        measures = {}
        for factor in (1, 2, 4):
            # Match training: filter the raw render, then clamp. Clamping before
            # filtering would change the operator around saturated highlights.
            pred = native if factor == 1 else render_image(g, resized_camera(cam, h * factor, w * factor))['render']
            reduced = pred if factor == 1 else downsample(pred, (h, w))
            mse_gt = float((reduced - gt).square().mean())
            mse_native = float((reduced - native).square().mean())
            measures[f'render_{factor}x_lr_grid'] = {'mse_to_real_lr': mse_gt, 'psnr_to_real_lr': metric_psnr(mse_gt),
                                                    'mse_to_native_lr_render': mse_native,
                                                    'psnr_to_native_lr_render': metric_psnr(mse_native)}
        rows.append({'frame_index': item['frame_index'], 'sampling': measures})
    result = {'mode': 'fixed_checkpoint_sampling_only', 'rows': rows,
              'operator_order': 'raw render -> bicubic antialias reduction -> clamp [0,1]',
              'aggregate': aggregate(rows, 'sampling'), 'elapsed_seconds': time.monotonic() - begin,
              'interpretation': 'Same checkpoint/FoV/time, rasterize LR/2LR/4LR; reduce with training bicubic-AA operator. Measures sampling dependence, not proof of geometry or motion error; native 1x self-comparison is exactly zero.'}
    return result


def encode_video(folder, path, fps):
    env = os.environ.copy()
    explicit = env.get('FOURDSR_FFMPEG')
    if explicit is not None:
        if not explicit.strip():
            raise ValueError('FOURDSR_FFMPEG must name an executable file, not an empty path')
        ffmpeg = Path(explicit).expanduser().resolve()
    else:
        ffmpeg = PROJECT / 'data/tools/ffmpeg/usr/bin/ffmpeg'
        lib = PROJECT / 'data/tools/ffmpeg/usr/lib/x86_64-linux-gnu'
        env['LD_LIBRARY_PATH'] = ':'.join([str(lib), str(lib / 'blas'), str(lib / 'lapack'), env.get('LD_LIBRARY_PATH', '')])
    if not ffmpeg.is_file():
        raise FileNotFoundError(f'FFmpeg executable not found: {ffmpeg}; set FOURDSR_FFMPEG to its installed path')
    if not os.access(ffmpeg, os.X_OK):
        raise PermissionError(f'FFmpeg file is not executable: {ffmpeg}')
    cmd = [str(ffmpeg), '-hide_banner', '-loglevel', 'error', '-y', '-framerate', str(fps),
           '-pattern_type', 'glob', '-i', str(folder / '*.png'), '-c:v', 'libx264',
           '-preset', 'veryfast', '-crf', '18', '-pix_fmt', 'yuv420p', str(path)]
    completed = subprocess.run(cmd, capture_output=True, text=True, env=env)
    return {'path': str(path), 'fps': fps, 'returncode': completed.returncode,
            'stderr': completed.stderr.strip(), 'command': cmd}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint', type=Path)
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--cache-dir', type=Path)
    ap.add_argument('--dynamic-threshold', type=float, default=.025)
    ap.add_argument('--flow-scale', type=float, default=.5)
    ap.add_argument('--sampling-only', action='store_true')
    ap.add_argument('--prepare-cache-only', action='store_true')
    ap.add_argument('--bicubic-baseline', action='store_true')
    ap.add_argument('--skip-lpips', action='store_true', help='Explicit smoke-test omission; recorded as unavailable')
    ap.add_argument('--lpips-device', choices=['cpu', 'cuda'], default='cuda')
    ap.add_argument('--no-video', action='store_true')
    args = ap.parse_args()
    if not 0 < args.flow_scale <= 1 or args.dynamic_threshold <= 0:
        ap.error('flow scale must be (0,1] and dynamic threshold positive')
    if not args.prepare_cache_only and args.checkpoint is None:
        ap.error('--checkpoint required unless --prepare-cache-only')
    if args.prepare_cache_only and args.sampling_only:
        ap.error('cache preparation and sampling-only are different modes')
    args.out.mkdir(parents=True, exist_ok=True)
    result_path = args.out / ('sampling.json' if args.sampling_only else 'metrics.json')
    if result_path.exists() and not args.prepare_cache_only:
        raise FileExistsError(f'Refusing to overwrite finished evaluation: {result_path}')
    cv2.setNumThreads(4)
    cv2.setRNGSeed(20260918)
    torch.set_num_threads(4)
    manifest = load_manifest(args.manifest)
    observations = sorted([o for o in manifest['observations'] if o['split'] == 'test'], key=lambda x: x['frame_index'])
    if not observations or {o['camera_id'] for o in observations} != {'cam00'}:
        raise ValueError('This evaluator requires the fixed cam00 test sequence')
    base = {'version': VERSION, 'scene': manifest['scene'], 'manifest': manifest['_manifest_path'],
            'manifest_sha256': file_sha(manifest['_manifest_path']), 'script_sha256': file_sha(__file__),
            'test_camera': 'cam00', 'frame_indices': [o['frame_index'] for o in observations],
            'versions': {'torch': torch.__version__, 'numpy': np.__version__, 'opencv': cv2.__version__}}
    if not args.sampling_only:
        cache, cache_info = prepare_cache(manifest, observations, args)
        base.update({'evaluation_cache': str(cache), 'cache_key': cache_info['cache_key']})
        if args.prepare_cache_only:
            write_json(args.out / 'cache_preparation.json', {**base, 'cache_info': cache_info})
            print(json.dumps({'cache': str(cache), 'dynamic_fraction': cache_info['dynamic_fraction']}), flush=True)
            return
    from common import load_checkpoint, render_image, resized_camera
    start = time.monotonic()
    g, _, _, checkpoint = load_checkpoint(args.checkpoint)
    g._deformation.eval()
    base.update({'checkpoint': str(args.checkpoint.resolve()), 'checkpoint_sha256': file_sha(args.checkpoint),
                 'checkpoint_metadata': checkpoint['metadata'], 'gaussian_count': len(g._xyz)})
    with torch.inference_mode():
        if args.sampling_only:
            write_json(result_path, {**base, **run_sampling(g, manifest, args)})
            print(f'SAMPLING_DONE {result_path}', flush=True)
            return
        lpips_model = None
        if not args.skip_lpips:
            import lpips
            lpips_model = lpips.LPIPS(net='alex', spatial=False).to(args.lpips_device).eval()
        data = N3DVPreparedDataset(manifest, 'test', 'hr')
        # Preserve exact observation order even if source manifest camera ordering changes.
        data.observations = observations
        dynamic = np.asarray(Image.open(cache / 'dynamic_mask.png')) > 0
        bbox = cache_info['dynamic_bbox_xyxy']
        rows, previous_pred, previous_gt, previous_bicubic = [], None, None, None
        lr_w, lr_h = manifest['resolutions']['lr']
        for i in range(len(data)):
            item = data[i]
            cam = observation_to_4dgs_camera(item, i)
            pred = image_array(render_image(g, cam)['render'])
            gt = image_array(item['image'])
            row = {'frame_index': item['frame_index'], 'time': item['time'],
                   'spatial': spatial_metrics(pred, gt, dynamic, lpips_model, args.lpips_device, bbox)}
            if previous_pred is not None:
                pair = cache_info['flow_pairs'][i-1]
                with np.load(cache / pair['file'], allow_pickle=False) as f:
                    flow, valid = full_flow(f['backward'], f['valid'], gt.shape[0], gt.shape[1])
                row['temporal'] = temporal_metrics(previous_pred, pred, previous_gt, gt, flow, valid, dynamic)
            if args.bicubic_baseline:
                native = render_image(g, resized_camera(cam, lr_h, lr_w))['render'].clamp(0, 1)
                up = F.interpolate(native[None], size=pred.shape[:2], mode='bicubic', align_corners=False, antialias=True)[0].clamp(0, 1)
                bicubic = image_array(up)
                row['bicubic_spatial'] = spatial_metrics(bicubic, gt, dynamic, lpips_model, args.lpips_device, bbox)
                if previous_bicubic is not None:
                    row['bicubic_temporal'] = temporal_metrics(previous_bicubic, bicubic, previous_gt, gt, flow, valid, dynamic)
                previous_bicubic = bicubic
            frame = item['frame_index']
            write_rgb(args.out / 'predictions' / f'{frame:04d}.png', pred)
            if frame in (0, 40, 80, 118):
                write_rgb(args.out / 'visuals' / f'{frame:04d}_gt.png', gt)
                write_rgb(args.out / 'visuals' / f'{frame:04d}_prediction.png', pred)
                error = np.abs(pred - gt).mean(axis=2)
                heat = cv2.applyColorMap(np.rint(np.clip(error / .2, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
                write_rgb(args.out / 'visuals' / f'{frame:04d}_error_mae_scale_0.2.png', heat[..., ::-1] / 255)
                if args.bicubic_baseline:
                    write_rgb(args.out / 'visuals' / f'{frame:04d}_bicubic_lr_render.png', bicubic)
            rows.append(row)
            previous_pred, previous_gt = pred, gt
            if (i + 1) % 10 == 0:
                print(f'EVAL {i+1}/{len(data)} {manifest["scene"]}', flush=True)
        result = {**base, 'rows': rows, 'aggregate': aggregate(rows, 'spatial'),
                  'temporal_aggregate': aggregate(rows, 'temporal'), 'dynamic_fraction': float(dynamic.mean()),
                  'dynamic_bbox_xyxy': bbox, 'dynamic_threshold': args.dynamic_threshold,
                  'lpips_status': 'explicitly_skipped' if args.skip_lpips else 'alex_v0.1_standard_full_and_fixed_mask_spatial_map',
                  'metric_definitions': {
                      'psnr': 'RGB [0,1], no shaving; per-frame PSNR averaged, pooled-MSE PSNR separate',
                      'ssim': 'RGB mean 11x11 Gaussian sigma1.5 population SSIM map; outer5 excluded; dynamic/static mask applied to real full-image map',
                      'lpips': 'lpips_alex is standard full-image AlexNet v0.1 on RGB [-1,1]. lpips_alex_spatial_mask is the mean of the official spatial=True full-image upsampled map over a fixed ROI; a regional extension, not standard scalar LPIPS. Receptive fields retain outside-ROI context; no pixels blacked out.',
                      'temporal': 'Mean abs((P_t-warp(P_prev))-(GT_t-warp(GT_prev))), fixed GT DIS backward flow; valid coverage reported per ROI; not geometry accuracy',
                      'std_frames': 'Descriptive across correlated frames/pairs, not confidence intervals or independent sample uncertainty',
                      'error_visual': 'Mean RGB absolute error, fixed [0,.2] mapped to inferno'},
                  'elapsed_seconds': time.monotonic() - start}
        if args.bicubic_baseline:
            result['bicubic_aggregate'] = aggregate(rows, 'bicubic_spatial')
            result['bicubic_temporal_aggregate'] = aggregate(rows, 'bicubic_temporal')
        if not args.no_video:
            differences = np.diff([o['frame_index'] for o in observations])
            if len(differences) and np.all(differences == differences[0]):
                result['video'] = encode_video(args.out / 'predictions', args.out / 'prediction.mp4', 30 / float(differences[0]))
            else:
                result['video'] = {'status': 'not_encoded_nonuniform_frame_intervals'}
        write_json(result_path, result)
        print(json.dumps({'path': str(result_path), 'aggregate': result['aggregate'],
                          'temporal': result['temporal_aggregate']}), flush=True)


if __name__ == '__main__':
    main()
