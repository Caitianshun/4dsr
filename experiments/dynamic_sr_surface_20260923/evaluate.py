#!/usr/bin/env python3
"""One evaluation entry point for baseline/global/local surface-residual forks.

Reuses the fixed 9/18 PSNR, SSIM, LPIPS and GT-relative temporal definitions.
Official ZJU masks are a disclosed public input, fixed across all branches:
per-frame foreground is primary, per-camera mask union is supplementary.
Evaluation RGB/masks are never passed to the model's rendering function.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from PIL import Image
import torch


ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / 'experiments/dynamic_sr_20260918'
if str(OLD) not in sys.path:
    sys.path.append(str(OLD))
_spec = importlib.util.spec_from_file_location('surface_legacy_metrics', OLD / 'evaluate.py')
legacy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(legacy)
from n3dv_data import load_manifest, observation_to_4dgs_camera

VERSION = 'zju_surface_residual_eval_v1'


def official_mask(manifest, observation, expected_shape):
    path = Path(manifest['_root']) / observation['mask_path']
    digest = legacy.file_sha(path)
    if 'mask_sha256' in observation and digest != observation['mask_sha256']:
        raise ValueError(f'Official-mask hash mismatch: {path}')
    with Image.open(path) as image:
        values = np.asarray(image.convert('L'))
    if values.shape != expected_shape:
        raise ValueError(f'Official mask must already match the prepared HR grid: {path}')
    return values > 0, {'path': str(path.resolve()), 'sha256': digest,
                        'camera_id': observation['camera_id'], 'frame_index': observation['frame_index']}


def spatial_metrics(pred, gt, foreground, union, metric, device):
    """Extend legacy metrics to two fixed external masks without extra LPIPS passes."""
    masks = {'full': np.ones(foreground.shape, dtype=bool), 'foreground': foreground,
             'background': ~foreground, 'foreground_union': union}
    error = ((pred - gt) ** 2).mean(axis=2)
    ssim = legacy.ssim_map_rgb(pred, gt)
    interior = np.zeros(foreground.shape, dtype=bool)
    interior[5:-5, 5:-5] = True
    result = {}
    for name, mask in masks.items():
        mse = legacy.masked_mean(error, mask)
        result[name] = {'mse': mse, 'psnr': legacy.metric_psnr(mse),
                        'ssim': legacy.masked_mean(ssim, mask & interior),
                        'pixel_count': int(mask.sum())}
    if metric is not None:
        tx = torch.from_numpy(np.ascontiguousarray(pred)).permute(2, 0, 1)[None].to(device) * 2 - 1
        ty = torch.from_numpy(np.ascontiguousarray(gt)).permute(2, 0, 1)[None].to(device) * 2 - 1
        with torch.inference_mode():
            result['full']['lpips_alex'] = float(metric(tx, ty).item())
            metric.spatial = True
            try:
                spatial = metric(tx, ty)[0, 0].cpu().numpy()
            finally:
                metric.spatial = False
        for name, mask in masks.items():
            result[name]['lpips_alex_spatial_mask'] = legacy.masked_mean(spatial, mask)
    return result


def temporal_metrics(previous_pred, pred, previous_gt, gt, flow, valid, foreground, union):
    official = legacy.temporal_metrics(previous_pred, pred, previous_gt, gt, flow, valid, foreground)
    fixed = legacy.temporal_metrics(previous_pred, pred, previous_gt, gt, flow, valid, union)
    return {'full': official['full'], 'foreground': official['dynamic'],
            'background': official['static'], 'foreground_union': fixed['dynamic']}


def render_camera(manifest, observation, uid):
    """Camera calibration/time only: no held-out RGB, mask, or LR enters rendering."""
    width, height = manifest['resolutions']['hr']
    calibration = manifest['cameras'][observation['camera_id']]
    item = {'image': torch.zeros((3, height, width), dtype=torch.float32),
            'camera_id': observation['camera_id'], 'frame_index': observation['frame_index'],
            'time': observation['time'], 'width': width, 'height': height,
            'K': np.asarray(calibration['K_hr'], dtype=np.float64),
            'w2c': np.asarray(calibration['w2c'], dtype=np.float64)}
    return observation_to_4dgs_camera(item, uid)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--split', choices=['dev', 'test'], default='test')
    parser.add_argument('--cache-dir', type=Path)
    parser.add_argument('--flow-scale', type=float, default=.5)
    parser.add_argument('--dynamic-threshold', type=float, default=.025,
                        help='Legacy cache identity only; official masks define reported foreground')
    parser.add_argument('--prepare-cache-only', action='store_true')
    parser.add_argument('--skip-lpips', action='store_true', help='Smoke only; omission is recorded explicitly')
    parser.add_argument('--lpips-device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--no-video', action='store_true', help='Compatibility option; no physical FPS is assumed')
    args = parser.parse_args()
    if not 0 < args.flow_scale <= 1 or args.dynamic_threshold <= 0:
        parser.error('flow-scale must be (0,1] and dynamic-threshold positive')
    if not args.prepare_cache_only and args.checkpoint is None:
        parser.error('--checkpoint is required for model evaluation')
    target = args.out / ('cache_preparation.json' if args.prepare_cache_only else 'metrics.json')
    if target.exists():
        raise FileExistsError(f'Refusing to overwrite completed evaluation: {target}')
    args.out.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(4)
    cv2.setRNGSeed(20260918)
    torch.set_num_threads(4)
    started = time.monotonic()
    manifest = load_manifest(args.manifest)
    observations = sorted((o for o in manifest['observations'] if o['split'] == args.split),
                          key=lambda o: (o['camera_id'], o['frame_index']))
    if not observations:
        raise ValueError(f'No observations in split {args.split}')
    cameras = sorted({o['camera_id'] for o in observations})
    train_cameras = {o['camera_id'] for o in manifest['observations'] if o['split'] == 'train'}
    if set(cameras) & train_cameras:
        raise ValueError('This is a held-out-camera evaluator; train/evaluation cameras overlap')
    keys = [(o['camera_id'], o['frame_index']) for o in observations]
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate evaluation observation')
    width, height = manifest['resolutions']['hr']
    grouped = {camera: [o for o in observations if o['camera_id'] == camera] for camera in cameras}
    masks, unions, caches, cache_metadata, mask_inputs = {}, {}, {}, {}, []
    for camera, records in grouped.items():
        union = np.zeros((height, width), dtype=bool)
        for observation in records:
            mask, identity = official_mask(manifest, observation, (height, width))
            masks[(camera, observation['frame_index'])] = mask
            mask_inputs.append(identity)
            union |= mask
        if not union.any():
            raise ValueError(f'No foreground in registered official masks: {camera}')
        unions[camera] = union
        # The legacy helper makes a checkpoint-independent GT flow cache. It
        # labels its own internal identity split="test"; our result records
        # the actual dev/test split and exact observation/image identities.
        cache, info = legacy.prepare_cache(manifest, records, args)
        caches[camera], cache_metadata[camera] = cache, info
        legacy.write_rgb(args.out / 'foreground_masks' / f'{camera}_union.png', union.astype(np.float32))
    foreground_key = hashlib.sha256(json.dumps(mask_inputs, sort_keys=True).encode()).hexdigest()
    source_files = {str(Path(__file__).resolve()): legacy.file_sha(__file__),
                    str(OLD / 'evaluate.py'): legacy.file_sha(OLD / 'evaluate.py'),
                    str(OLD / 'n3dv_data.py'): legacy.file_sha(OLD / 'n3dv_data.py')}
    base = {'version': VERSION, 'scene': manifest['scene'], 'source_dataset': manifest.get('source_dataset'),
            'manifest': manifest['_manifest_path'], 'manifest_sha256': legacy.file_sha(manifest['_manifest_path']),
            'script_sha256': legacy.file_sha(__file__), 'metric_helpers_sha256': legacy.file_sha(OLD / 'evaluate.py'),
            'split': args.split, 'evaluation_cameras': cameras,
            'observation_keys': [{'camera_id': cam, 'frame_index': frame} for cam, frame in keys],
            'frame_indices_by_camera': {cam: [o['frame_index'] for o in records] for cam, records in grouped.items()},
            'evaluation_caches': {cam: {'path': str(caches[cam]), 'cache_key': info['cache_key'],
                                       'identity': info['identity']} for cam, info in cache_metadata.items()},
            'official_mask_inputs': mask_inputs, 'foreground_cache_key': foreground_key,
            'versions': {'torch': str(torch.__version__), 'numpy': np.__version__, 'opencv': cv2.__version__},
            'protocol': 'Nonstandard ZJU-Mocap short-window held-out-camera pilot; no claim of the standard benchmark protocol',
            'information_boundary': 'Only calibration, time and an all-zero image placeholder enter rendering; held-out RGB/masks are evaluation-only. Prepared RGB uses publicly supplied official-mask black-background compositing, shared by every branch.',
            'video': {'status': 'not_encoded', 'reason': 'Normalized time is not physical FPS; no FPS assumed'}}
    if args.prepare_cache_only:
        legacy.write_json(target, {**base, 'cache_info': cache_metadata, 'elapsed_seconds': time.monotonic() - started})
        print(json.dumps({'path': str(target), 'mode': 'CPU cache only', 'cameras': cameras}), flush=True)
        return

    from residual_model import load_model, render_model
    import residual_model
    source_files[str(Path(residual_model.__file__).resolve())] = legacy.file_sha(residual_model.__file__)
    model = load_model(args.checkpoint, manifest=manifest, restore_rng=False)
    model.g._deformation.eval()
    if model.detail is not None:
        model.detail.eval()
    checkpoint_manifest = model.checkpoint['metadata'].get('manifest_sha')
    if checkpoint_manifest is not None and checkpoint_manifest != base['manifest_sha256']:
        raise ValueError('Checkpoint/manifest identity mismatch')
    base.update({'checkpoint': str(args.checkpoint.resolve()), 'checkpoint_sha256': legacy.file_sha(args.checkpoint),
                 'checkpoint_metadata': model.checkpoint['metadata'], 'branch': model.branch,
                 'gaussian_count': len(model.g._xyz), 'source_sha256': source_files})
    metric = None
    if not args.skip_lpips:
        import lpips
        metric = lpips.LPIPS(net='alex', spatial=False).to(args.lpips_device).eval().requires_grad_(False)
    rows = []
    with torch.inference_mode():
        uid = 0
        for camera, records in grouped.items():
            previous_pred = previous_gt = None
            union = unions[camera]
            for index, observation in enumerate(records):
                cam = render_camera(manifest, observation, uid)
                uid += 1
                # Every branch, including the unmodified baseline, uses this
                # identical renderer and final [0,1] metric clipping.
                raw = render_model(model, cam)['render']
                if tuple(raw.shape) != (3, height, width) or not torch.isfinite(raw).all():
                    raise ValueError(f'Nonfinite/wrong-shaped render: {camera}/{observation["frame_index"]}')
                pred = legacy.image_array(raw)
                gt_path = Path(manifest['_root']) / observation['hr_path']
                if 'hr_sha256' in observation and legacy.file_sha(gt_path) != observation['hr_sha256']:
                    raise ValueError(f'HR hash mismatch: {gt_path}')
                gt = legacy.read_rgb(gt_path)
                if gt.shape != pred.shape:
                    raise ValueError(f'HR/render shape mismatch: {gt_path}')
                foreground = masks[(camera, observation['frame_index'])]
                row = {'camera_id': camera, 'frame_index': observation['frame_index'], 'time': observation['time'],
                        'spatial': spatial_metrics(pred, gt, foreground, union, metric, args.lpips_device),
                        'render_clipped_fraction': float(((raw < 0) | (raw > 1)).float().mean()),
                        'foreground_fraction': float(foreground.mean())}
                if previous_pred is not None:
                    pair = cache_metadata[camera]['flow_pairs'][index - 1]
                    if pair['previous_frame'] != records[index - 1]['frame_index'] or pair['current_frame'] != observation['frame_index']:
                        raise ValueError('Flow-cache pair identity mismatch')
                    with np.load(caches[camera] / pair['file'], allow_pickle=False) as values:
                        flow, valid = legacy.full_flow(values['backward'], values['valid'], height, width)
                    row['temporal'] = temporal_metrics(previous_pred, pred, previous_gt, gt, flow, valid, foreground, union)
                legacy.write_rgb(args.out / 'predictions' / camera / f'{observation["frame_index"]:04d}.png', pred)
                rows.append(row)
                previous_pred, previous_gt = pred, gt
                if (index + 1) % 10 == 0:
                    print(f'EVAL {args.split} {camera} {index + 1}/{len(records)}', flush=True)
    result = {**base, 'rows': rows, 'aggregate': legacy.aggregate(rows, 'spatial'),
              'temporal_aggregate': legacy.aggregate(rows, 'temporal'),
              'by_camera': {cam: {'aggregate': legacy.aggregate([r for r in rows if r['camera_id'] == cam], 'spatial'),
                                  'temporal_aggregate': legacy.aggregate([r for r in rows if r['camera_id'] == cam], 'temporal')}
                            for cam in cameras},
              'lpips_status': 'explicitly_skipped' if args.skip_lpips else 'alex_v0.1_standard_full_and_fixed_mask_spatial_map',
              'metric_definitions': {
                  'psnr': 'RGB [0,1], no shaving; equal-observation average of PSNR; pooled MSE PSNR separately named',
                  'ssim': 'Legacy RGB mean 11x11 Gaussian sigma1.5 population SSIM; outer5 excluded; masks applied to real full-image map',
                  'lpips': 'Standard scalar AlexNet v0.1 LPIPS only for full. Foreground/background/union use official spatial=True map averaged within mask; regional extension, not scalar LPIPS; outside-mask context retained.',
                  'foreground': 'Per-frame official ZJU mask >0 at prepared undistorted HR grid; fixed public input across branches, not inferred from predictions',
                  'foreground_union': 'Supplementary fixed spatial union of official masks over every registered evaluation frame of the same camera; may contain background in individual frames',
                  'temporal': 'Legacy mean abs((P_t-warp(P_prev))-(GT_t-warp(GT_prev))) on fixed GT DIS valid correspondences, same camera only; current-frame official mask defines foreground ROI; not geometry accuracy',
                  'std_frames': 'Descriptive over correlated frames/pairs; not independent-sample confidence intervals'},
              'elapsed_seconds': time.monotonic() - started, 'parameter_updates': 0}
    legacy.write_json(target, result)
    legacy.write_json(args.out / 'complete.json', {'status': 'completed_evaluation', 'metrics_sha256': legacy.file_sha(target),
                                                  'checkpoint_sha256': base['checkpoint_sha256'], 'observations': len(rows),
                                                  'split': args.split, 'parameter_updates': 0})
    print(json.dumps({'path': str(target), 'aggregate': result['aggregate'],
                      'temporal': result['temporal_aggregate']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
