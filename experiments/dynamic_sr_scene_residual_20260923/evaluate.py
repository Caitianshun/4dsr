#!/usr/bin/env python3
"""Full-scene RGB evaluation shared by joint/global/local residual models.

Calls the existing 9/18 metric and GT-flow/cache functions directly. The
time-changing/static regions are evaluation-only proxies, not person masks.
No RGB compositing, official masks, or held-out pixel inputs to rendering.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
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
spec = importlib.util.spec_from_file_location('scene_residual_legacy_metrics', OLD / 'evaluate.py')
legacy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(legacy)
from n3dv_data import load_manifest, observation_to_4dgs_camera

VERSION = 'full_scene_residual_eval_v1'


def render_camera(manifest, observation, uid):
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
    parser.add_argument('--dynamic-threshold', type=float, default=.025)
    parser.add_argument('--prepare-cache-only', action='store_true')
    parser.add_argument('--skip-lpips', action='store_true', help='Smoke only; explicitly marked unavailable')
    parser.add_argument('--lpips-device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--no-video', action='store_true', help='Compatibility option; this entry writes PNGs only')
    args = parser.parse_args()
    if not 0 < args.flow_scale <= 1 or args.dynamic_threshold <= 0:
        parser.error('flow-scale must be (0,1] and dynamic-threshold positive')
    if not args.prepare_cache_only and args.checkpoint is None:
        parser.error('--checkpoint is required for model evaluation')
    result_path = args.out / ('cache_preparation.json' if args.prepare_cache_only else 'metrics.json')
    if result_path.exists():
        raise FileExistsError(f'Refusing to overwrite completed evaluation: {result_path}')
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
        raise ValueError('Held-out-camera evaluation cannot overlap training cameras')
    keys = [(o['camera_id'], o['frame_index']) for o in observations]
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate evaluation observation')
    width, height = manifest['resolutions']['hr']
    grouped = {camera: [o for o in observations if o['camera_id'] == camera] for camera in cameras}
    caches, cache_info, dynamic_masks = {}, {}, {}
    for camera, records in grouped.items():
        caches[camera], cache_info[camera] = legacy.prepare_cache(manifest, records, args)
        with Image.open(caches[camera] / 'dynamic_mask.png') as image:
            dynamic_masks[camera] = np.asarray(image) > 0
        if dynamic_masks[camera].shape != (height, width):
            raise ValueError('Fixed evaluation region shape differs from HR resolution')
    base = {'version': VERSION, 'scene': manifest['scene'], 'manifest': manifest['_manifest_path'],
            'manifest_sha256': legacy.file_sha(manifest['_manifest_path']), 'script_sha256': legacy.file_sha(__file__),
            'metric_helpers_sha256': legacy.file_sha(OLD / 'evaluate.py'), 'split': args.split,
            'evaluation_cameras': cameras, 'observation_keys': [{'camera_id': c, 'frame_index': f} for c, f in keys],
            'frame_indices_by_camera': {c: [o['frame_index'] for o in records] for c, records in grouped.items()},
            'evaluation_caches': {c: {'path': str(caches[c]), 'cache_key': info['cache_key'], 'identity': info['identity'],
                                     'dynamic_mask_sha256': legacy.file_sha(caches[c] / 'dynamic_mask.png')}
                                  for c, info in cache_info.items()},
            'dynamic_fraction_by_camera': {c: float(mask.mean()) for c, mask in dynamic_masks.items()},
            'dynamic_bbox_by_camera': {c: info['dynamic_bbox_xyxy'] for c, info in cache_info.items()},
            'versions': {'torch': str(torch.__version__), 'numpy': np.__version__, 'opencv': cv2.__version__},
            'protocol': 'Full-scene RGB short-window held-out-camera pilot; complete original background retained',
            'information_boundary': 'Camera calibration/time and a zero-image placeholder enter rendering. HR-derived time-changing masks and flows are fixed evaluation-only caches, never model or training inputs; no official mask or RGB compositing.',
            'cache_split_note': 'Legacy cache internal split label is test; actual split and exact observation/image identity are recorded here.',
            'video': {'status': 'not_encoded', 'reason': 'No physical FPS is inferred from normalized time'}}
    if args.prepare_cache_only:
        legacy.write_json(result_path, {**base, 'cache_info': cache_info, 'elapsed_seconds': time.monotonic() - started})
        print(json.dumps({'path': str(result_path), 'mode': 'CPU cache only'}), flush=True)
        return

    from residual_model import load_model, render_model
    import residual_model
    model = load_model(args.checkpoint, manifest=manifest, restore_rng=False)
    model.g._deformation.eval()
    if model.detail is not None:
        model.detail.eval()
    if model.checkpoint['metadata'].get('manifest_sha') != base['manifest_sha256']:
        raise ValueError('Checkpoint/manifest identity mismatch')
    base.update({'checkpoint': str(args.checkpoint.resolve()), 'checkpoint_sha256': legacy.file_sha(args.checkpoint),
                 'checkpoint_metadata': model.checkpoint['metadata'], 'branch': model.branch,
                 'gaussian_count': len(model.g._xyz), 'residual_model_sha256': legacy.file_sha(residual_model.__file__),
                 'gpu': torch.cuda.get_device_name(), 'visible_cuda': os.environ.get('CUDA_VISIBLE_DEVICES'),
                 'lpips_device': args.lpips_device})
    metric = None
    if not args.skip_lpips:
        import lpips
        metric = lpips.LPIPS(net='alex', spatial=False).to(args.lpips_device).eval().requires_grad_(False)
    rows, render_seconds = [], 0.0
    with torch.inference_mode():
        uid = 0
        for camera, records in grouped.items():
            previous_pred = previous_gt = None
            dynamic = dynamic_masks[camera]
            for index, observation in enumerate(records):
                cam = render_camera(manifest, observation, uid)
                uid += 1
                torch.cuda.synchronize()
                render_started = time.monotonic()
                raw = render_model(model, cam)['render']
                torch.cuda.synchronize()
                render_s = time.monotonic() - render_started
                render_seconds += render_s
                if tuple(raw.shape) != (3, height, width) or not torch.isfinite(raw).all():
                    raise ValueError(f'Invalid render: {camera}/{observation["frame_index"]}')
                pred = legacy.image_array(raw)
                gt_path = Path(manifest['_root']) / observation['hr_path']
                if legacy.file_sha(gt_path) != observation['hr_sha256']:
                    raise ValueError(f'HR hash mismatch: {gt_path}')
                gt = legacy.read_rgb(gt_path)
                if gt.shape != pred.shape:
                    raise ValueError(f'HR/render shape mismatch: {gt_path}')
                row = {'camera_id': camera, 'frame_index': observation['frame_index'], 'time': observation['time'],
                        'render_seconds': render_s,
                        'spatial': legacy.spatial_metrics(pred, gt, dynamic, metric, args.lpips_device),
                        'render_clipped_fraction': float(((raw < 0) | (raw > 1)).float().mean())}
                if previous_pred is not None:
                    pair = cache_info[camera]['flow_pairs'][index - 1]
                    if pair['previous_frame'] != records[index - 1]['frame_index'] or pair['current_frame'] != observation['frame_index']:
                        raise ValueError('Flow-cache pair identity mismatch')
                    with np.load(caches[camera] / pair['file'], allow_pickle=False) as values:
                        flow, valid = legacy.full_flow(values['backward'], values['valid'], height, width)
                    row['temporal'] = legacy.temporal_metrics(previous_pred, pred, previous_gt, gt, flow, valid, dynamic)
                legacy.write_rgb(args.out / 'predictions' / camera / f'{observation["frame_index"]:04d}.png', pred)
                rows.append(row)
                previous_pred, previous_gt = pred, gt
                if (index + 1) % 10 == 0:
                    print(f'EVAL {args.split} {camera} {index + 1}/{len(records)}', flush=True)
    result = {**base, 'rows': rows, 'aggregate': legacy.aggregate(rows, 'spatial'),
              'temporal_aggregate': legacy.aggregate(rows, 'temporal'),
              'by_camera': {c: {'aggregate': legacy.aggregate([r for r in rows if r['camera_id'] == c], 'spatial'),
                                 'temporal_aggregate': legacy.aggregate([r for r in rows if r['camera_id'] == c], 'temporal')}
                            for c in cameras},
              'lpips_status': 'explicitly_skipped' if args.skip_lpips else 'alex_v0.1_standard_full_and_fixed_mask_spatial_map',
              'metric_definitions': {
                  'psnr': 'Legacy RGB [0,1], no shaving; equal-frame mean PSNR; pooled MSE PSNR separately named',
                  'ssim': 'Legacy RGB mean 11x11 Gaussian sigma1.5 population SSIM; outer5 excluded; real full-image map',
                  'lpips': 'Full uses standard scalar AlexNet v0.1 LPIPS. Dynamic/static use the spatial=True full-image map averaged in a fixed ROI, a regional extension with outside-mask context retained.',
                  'regions': 'Full-scene RGB is primary. Dynamic is fixed HR temporal std>.025 by default, open3/close5/dilate5; static is its complement. Time-changing proxy only, not human/semantic segmentation; evaluation-only.',
                  'temporal': 'Legacy mean abs((P_t-warp(P_prev))-(GT_t-warp(GT_prev))) on fixed GT DIS valid correspondences; same camera only; valid coverage reported; not geometry accuracy',
                  'std_frames': 'Descriptive over correlated frames/pairs; not independent-sample uncertainty'},
              'dynamic_threshold': args.dynamic_threshold, 'flow_scale': args.flow_scale,
              'render_seconds': render_seconds, 'render_seconds_per_frame': render_seconds / len(rows),
              'render_timing': 'CUDA-synchronized render_model only, includes first frame; excludes camera construction, CPU copies, metrics and I/O; not end-to-end FPS',
              'elapsed_seconds': time.monotonic() - started, 'parameter_updates': 0}
    legacy.write_json(result_path, result)
    legacy.write_json(args.out / 'complete.json', {'status': 'completed_evaluation', 'metrics_sha256': legacy.file_sha(result_path),
                                                  'checkpoint_sha256': base['checkpoint_sha256'], 'observations': len(rows),
                                                  'split': args.split, 'parameter_updates': 0})
    print(json.dumps({'path': str(result_path), 'aggregate': result['aggregate'],
                      'temporal': result['temporal_aggregate']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
