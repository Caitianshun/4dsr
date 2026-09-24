#!/usr/bin/env python3
"""Read-only Joint rendering on a fixed subset of its SR-supervised train views."""
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch

from common import load_checkpoint, render_image, resized_camera
from evaluate import aggregate, file_sha, image_array, read_rgb, spatial_metrics, write_json, write_rgb
from n3dv_data import load_manifest, N3DVPreparedDataset, observation_to_4dgs_camera


PROJECT = Path(__file__).resolve().parents[2]
CAMERAS = ('cam02', 'cam06', 'cam12', 'cam18')
FRAMES = (0, 40, 80, 118)
SCENES = ('cook_spinach', 'cut_roasted_beef')


def main():
    import lpips

    out = PROJECT/'output/dynamic_sr_20260918/prior_quality_audit'
    if (out/'prior_fit.json').exists() or (out/'prior_fit_started.json').exists():
        raise FileExistsError(out/'prior_fit.json')
    prior_audit = json.loads((out/'metrics.json').read_text())
    started = time.monotonic()
    identity = {
        'version': 'joint_train_prior_fit_v1',
        'started_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'fixed subset of train views used for SR supervision; not held-out novel-view benchmark',
        'cameras': list(CAMERAS), 'frame_indices': list(FRAMES),
        'script_sha256': file_sha(__file__),
        'evaluate_script_sha256': file_sha(Path(__file__).with_name('evaluate.py')),
        'prior_audit_sha256': file_sha(out/'metrics.json'),
        'device': torch.cuda.get_device_name(),
        'information_boundary': 'checkpoints only loaded for rendering; no backward pass, optimizer step, training, or checkpoint writes; HR used only for scoring',
        'metric_definitions': prior_audit['metric_definitions'],
        'render_definition': 'float32 raw HR render clamped to [0,1], evaluated before PNG quantization; same as main evaluate.py',
    }
    write_json(out/'prior_fit_started.json', identity)
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
            assert manifest_sha == prior_audit['scenes'][scene]['manifest_sha256']
            checkpoint = PROJECT/'output/dynamic_sr_20260918'/f'{scene}_pilot_v1_joint/checkpoint_final.pt'
            checkpoint_sha = file_sha(checkpoint)
            gaussian, _, _, state = load_checkpoint(checkpoint)
            gaussian._deformation.eval()
            data = N3DVPreparedDataset(manifest, 'train', 'lr')
            data.observations = sorted([o for o in data.observations
                                        if o['camera_id'] in CAMERAS and o['frame_index'] in FRAMES],
                                       key=lambda o: (o['camera_id'], o['frame_index']))
            assert len(data) == 16
            prior_rows = {(r['camera_id'], r['frame_index']): r
                          for r in prior_audit['scenes'][scene]['rows']}
            rows = []
            width, height = manifest['resolutions']['hr']
            mask = np.zeros((height, width), dtype=bool)
            with torch.inference_mode():
                for i in range(len(data)):
                    item, observation = data[i], data.observations[i]
                    camera = resized_camera(observation_to_4dgs_camera(item, i), height, width)
                    prediction = image_array(render_image(gaussian, camera)['render'])
                    known = prior_rows[(item['camera_id'], item['frame_index'])]
                    prior_path, hr_path = root/known['prior_path'], root/observation['hr_path']
                    assert file_sha(prior_path) == known['prior_sha256']
                    assert file_sha(hr_path) == known['hr_sha256']
                    prior, gt = read_rgb(prior_path), read_rgb(hr_path)
                    row = {'camera_id': item['camera_id'], 'frame_index': item['frame_index'],
                           'render_vs_hr': {'full': spatial_metrics(prediction, gt, mask, metric)['full']},
                           'render_vs_prior': {'full': spatial_metrics(prediction, prior, mask, metric)['full']},
                           'prior_vs_hr': known['swinir'],
                           'prior_sha256': known['prior_sha256'], 'hr_sha256': known['hr_sha256']}
                    rows.append(row)
                    write_rgb(out/'train_view_renders'/scene/item['camera_id']/f'{item["frame_index"]:04d}.png', prediction)
            modes = ('render_vs_hr', 'render_vs_prior', 'prior_vs_hr')
            assert file_sha(checkpoint) == checkpoint_sha
            results[scene] = {
                'checkpoint': str(checkpoint), 'checkpoint_sha256': checkpoint_sha,
                'checkpoint_metadata': state['metadata'], 'manifest_sha256': manifest_sha,
                'gaussian_count': len(gaussian._xyz), 'render_count': len(rows),
                'aggregate': {mode: aggregate(rows, mode)['full'] for mode in modes},
                'per_camera': {camera: {mode: aggregate([r for r in rows if r['camera_id'] == camera], mode)['full']
                                       for mode in modes} for camera in CAMERAS},
                'rows': rows, 'elapsed_seconds': time.monotonic()-scene_started,
            }
            write_json(out/f'{scene}_prior_fit.json', results[scene])
            print(json.dumps({'scene': scene, 'aggregate': results[scene]['aggregate'],
                              'elapsed_seconds': results[scene]['elapsed_seconds']}), flush=True)
            del gaussian, state
            torch.cuda.empty_cache()
        result = {**identity, 'scenes': results, 'render_count': 32,
                  'finished_utc': datetime.now(timezone.utc).isoformat(),
                  'elapsed_seconds': time.monotonic()-started}
        write_json(out/'prior_fit.json', result)
        print(json.dumps({'completed': str(out/'prior_fit.json'), 'elapsed_seconds': result['elapsed_seconds']}), flush=True)
    except BaseException as exc:
        write_json(out/'prior_fit_failure.json', {'error_type': type(exc).__name__, 'error': str(exc),
                                                'completed_scenes': list(results),
                                                'elapsed_seconds': time.monotonic()-started})
        raise


if __name__ == '__main__':
    main()
