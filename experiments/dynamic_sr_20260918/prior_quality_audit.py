#!/usr/bin/env python3
"""Evaluate existing train-view SR targets against evaluator-only train HR.

This is an image prior diagnostic, not reconstruction evaluation or training.
No model is loaded or changed; only the frozen LPIPS evaluation network runs.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from evaluate import aggregate, file_sha, read_rgb, spatial_metrics, write_json
from generate_prior import CHECKPOINT_SHA256
from n3dv_data import load_manifest


PROJECT = Path(__file__).resolve().parents[2]
CAMERAS = ('cam02', 'cam06', 'cam12', 'cam18')
SCENES = ('cook_spinach', 'cut_roasted_beef')


def main():
    import json
    import lpips

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path,
                        default=PROJECT/'output/dynamic_sr_20260918/prior_quality_audit')
    args = parser.parse_args()
    output = args.out.resolve()
    if (output/'metrics.json').exists() or (output/'started.json').exists():
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    identity = {
        'version': 'train_prior_quality_audit_v1',
        'started_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'existing frozen SwinIR train-view targets versus matching evaluator-only HR; no reconstruction training',
        'cameras': list(CAMERAS), 'frame_indices': list(range(0, 120, 2)),
        'checkpoint_sha256_expected': CHECKPOINT_SHA256,
        'script_sha256': file_sha(__file__),
        'evaluate_script_sha256': file_sha(Path(__file__).with_name('evaluate.py')),
        'torch_version': torch.__version__, 'numpy_version': np.__version__,
        'device': torch.cuda.get_device_name(),
        'metric_definitions': {
            'psnr': 'full-image RGB [0,1] MSE, then per-image PSNR; frame-equal mean',
            'ssim': 'RGB 11x11 Gaussian sigma=1.5, population covariance; outer five-pixel border excluded',
            'lpips': 'standard full-image AlexNet LPIPS v0.1; RGB [-1,1]; spatial_metrics implementation reused',
            'roi': 'none: all-false mask passed to existing evaluator and only full result retained',
            'std_frames': 'descriptive image dispersion, not uncertainty or statistical confidence',
        },
        'bicubic': 'matching true LR uint8 PNG, torch bicubic align_corners=False antialias=True to HR dimensions, clamp [0,1], no output requantization',
        'swinir': 'existing sr_swinir_x4 uint8 PNGs, matched receipt input/output and manifest hashes checked; no new SR inference',
        'information_boundary': 'HR opened exclusively for these diagnostic metrics; no weights/optimizer/checkpoints updated',
    }
    write_json(output/'started.json', identity)
    torch.set_num_threads(4)
    cv2.setNumThreads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    metric = lpips.LPIPS(net='alex', spatial=False).cuda().eval()
    metric.requires_grad_(False)
    results = {}
    try:
        for scene in SCENES:
            scene_started = time.monotonic()
            root = PROJECT/'data/dynamic_sr/n3dv_prepared'/scene
            manifest = load_manifest(root/'manifest.json')
            manifest_sha = file_sha(root/'manifest.json')
            config = json.loads((root/'sr_swinir_x4/prior_config.json').read_text())
            assert config['manifest_sha256'] == manifest_sha
            assert config['checkpoint_sha256'] == CHECKPOINT_SHA256
            assert file_sha(config['checkpoint']) == CHECKPOINT_SHA256
            observations = sorted([o for o in manifest['observations']
                                   if o['split'] == 'train' and o['camera_id'] in CAMERAS],
                                  key=lambda o: (o['camera_id'], o['frame_index']))
            assert len(observations) == 240
            for camera in CAMERAS:
                assert [o['frame_index'] for o in observations if o['camera_id'] == camera] == list(range(0, 120, 2))
            rows = []
            for observation in observations:
                lr_path = root/observation['lr_path']
                hr_path = root/observation['hr_path']
                prior_path = root/'sr_swinir_x4'/observation['camera_id']/lr_path.name
                receipt_path = prior_path.with_suffix('.json')
                receipt = json.loads(receipt_path.read_text())
                hashes = {'lr_sha256': file_sha(lr_path), 'hr_sha256': file_sha(hr_path),
                          'prior_sha256': file_sha(prior_path), 'receipt_sha256': file_sha(receipt_path)}
                assert hashes['lr_sha256'] == observation['lr_sha256'] == receipt['input_sha256']
                assert hashes['hr_sha256'] == observation['hr_sha256']
                assert hashes['prior_sha256'] == receipt['output_sha256']
                assert receipt['relative_path'] == f'{observation["camera_id"]}/{lr_path.name}'
                lr, gt, prior = read_rgb(lr_path), read_rgb(hr_path), read_rgb(prior_path)
                assert list(gt.shape[:2][::-1]) == manifest['resolutions']['hr']
                assert prior.shape == gt.shape
                assert list(lr.shape[:2][::-1]) == manifest['resolutions']['lr']
                empty_mask = np.zeros(gt.shape[:2], dtype=bool)
                with torch.inference_mode():
                    tensor = torch.from_numpy(np.ascontiguousarray(lr)).permute(2, 0, 1)[None].cuda()
                    bicubic = F.interpolate(tensor, size=gt.shape[:2], mode='bicubic',
                                            align_corners=False, antialias=True)[0].clamp(0, 1)
                    bicubic = bicubic.permute(1, 2, 0).cpu().numpy()
                    scores = {name: {'full': spatial_metrics(pred, gt, empty_mask, metric)['full']}
                              for name, pred in [('bicubic', bicubic), ('swinir', prior)]}
                rows.append({'camera_id': observation['camera_id'], 'frame_index': observation['frame_index'],
                             'lr_path': observation['lr_path'], 'hr_path': observation['hr_path'],
                             'prior_path': str(prior_path.relative_to(root)), **hashes, **scores})
            per_camera = {
                camera: {mode: aggregate([r for r in rows if r['camera_id'] == camera], mode)['full']
                         for mode in ('bicubic', 'swinir')} for camera in CAMERAS
            }
            results[scene] = {
                'manifest': str(root/'manifest.json'), 'manifest_sha256': manifest_sha,
                'prior_config': config, 'image_count': len(rows), 'all_hash_checks_passed': True,
                'aggregate': {mode: aggregate(rows, mode)['full'] for mode in ('bicubic', 'swinir')},
                'per_camera': per_camera, 'rows': rows,
                'elapsed_seconds': time.monotonic()-scene_started,
            }
            write_json(output/f'{scene}.json', results[scene])
            print(json.dumps({'scene': scene, 'image_count': len(rows),
                              'aggregate': results[scene]['aggregate'],
                              'elapsed_seconds': results[scene]['elapsed_seconds']}), flush=True)
        result = {**identity, 'scenes': results,
                  'image_count': sum(x['image_count'] for x in results.values()),
                  'finished_utc': datetime.now(timezone.utc).isoformat(),
                  'elapsed_seconds': time.monotonic()-started,
                  'max_cuda_memory_allocated_bytes': torch.cuda.max_memory_allocated()}
        write_json(output/'metrics.json', result)
        print(json.dumps({'completed': str(output/'metrics.json'), 'elapsed_seconds': result['elapsed_seconds']}), flush=True)
    except BaseException as exc:
        write_json(output/'failure.json', {'error_type': type(exc).__name__, 'error': str(exc),
                                          'completed_scenes': list(results),
                                          'elapsed_seconds': time.monotonic()-started})
        raise


if __name__ == '__main__':
    main()
