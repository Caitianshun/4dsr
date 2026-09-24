"""Post-completion pair checks and fixed-frame credible-detail diagnosis (CPU).

Reuses the earlier HR-supported residual mask/crops. Saved PNG diagnostics are
kept separate from float-render PSNR/SSIM/LPIPS evaluation. No model training.
"""
import importlib.util
import json
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'output/dynamic_sr_20260920/continuation'
DEST = BASE / 'assessment'
OLD = ROOT / 'output/dynamic_sr_20260919'
spec = importlib.util.spec_from_file_location('detail', ROOT / 'experiments/dynamic_sr_20260919/final_detail_audit.py')
detail = importlib.util.module_from_spec(spec)
spec.loader.exec_module(detail)
helpers = detail.old


def read(path):
    return json.loads(Path(path).read_text())


def panel(path, scene, camera, images, crops):
    keys = ['hr', 'bicubic', 'swinir', 'early', 'joint', 'appearance_only']
    labels = ['HR', 'Actual LR bicubic', 'Actual LR -> SwinIR', 'Original 6k', 'Joint 18k', 'Appearance only 18k']
    w, pad, rowh = 260, 12, 570
    canvas = Image.new('RGB', (6 * (w + pad) + pad, 2 * rowh + 90), 'white')
    draw = ImageDraw.Draw(canvas)
    font = helpers.font(15)
    draw.text((pad, 5), f'{scene} / {camera} / frame 0040; same pre-existing HR-ranked crops', fill='black', font=font)
    draw.text((pad, 28), 'Cam00 LR/SR are privileged references; all values here are saved PNG diagnostics.' if camera == 'cam00'
              else 'Cam02: SwinIR is the actual training teacher; HR is evaluation-only.', fill='black', font=font)
    for j, (region, crop) in enumerate(crops.items()):
        if crop is None:
            continue
        x0, y0, x1, y1 = [crop[k] for k in ('x0', 'y0', 'x1', 'y1')]
        box = np.s_[y0:y1, x0:x1]
        top = 55 + j * rowh
        draw.text((pad, top), f'{region}: [{x0},{y0},{x1},{y1}); image above, error below', fill='black', font=font)
        for i, (key, label) in enumerate(zip(keys, labels)):
            left = pad + i * (w + pad)
            draw.text((left, top + 22), label, fill='black', font=font)
            canvas.paste(helpers.rgb_image(images[key][box]).resize((w, w), Image.Resampling.NEAREST), (left, top + 44))
            canvas.paste(helpers.err_image(images[key][box], images['hr'][box]).resize((w, w), Image.Resampling.NEAREST), (left, top + 44 + w))
    draw.text((pad, canvas.height - 26), 'Error: fixed per-pixel RGB RMSE range [0, 0.15]; lower is closer to HR.', fill='black', font=font)
    canvas.save(path)


def main():
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    started = time.monotonic()
    DEST.mkdir(exist_ok=False)
    old_detail = read(OLD / 'final_assessment/detail_audit/metrics.json')
    result = dict(protocol='4 fixed observations only; reference-supported residual improvement is not geometry truth',
                  source_sha256=helpers.sha(__file__), scenes={})
    for scene in ('cook_spinach', 'meetroom_discussion'):
        runs = {m: BASE / f'{scene}_{m}' for m in ('joint', 'appearance_only')}
        configs = {m: read(p / 'config.json') for m, p in runs.items()}
        dones = {m: read(p / 'complete.json') for m, p in runs.items()}
        checks = {k: configs['joint'][k] == configs['appearance_only'][k]
                  for k in ('parent_sha256', 'manifest_sha256', 'source_prefix_draw_sha256')}
        checks['actual_draw_sha256'] = dones['joint']['additional_draw_sha256'] == dones['appearance_only']['additional_draw_sha256']
        checks['frozen_support'] = dones['appearance_only']['frozen_verification']['passed'] is True
        checks['exposure'] = all(read(p / 'full_exposure.json') == read(OLD / f'{scene}_sr_w01/exposure.json') for p in runs.values())
        checks['parent_file_sha'] = all(helpers.sha(c['source_checkpoint']) == c['parent_sha256'] for c in configs.values())
        for m, p in runs.items():
            ev = read(BASE / f'{scene}_{m}_evaluation/metrics.json')
            checks[m + '_checkpoint_sha'] = ev['checkpoint_sha256'] == helpers.sha(p / 'checkpoint_final.pt')
            checks[m + '_evaluation_complete'] = len(ev['rows']) == 60 and sum('temporal' in row for row in ev['rows']) == 59
        assert all(checks.values()), checks
        manifest = read(configs['joint']['manifest'])
        data = Path(configs['joint']['manifest']).parent
        observations = []
        for camera in ('cam02', 'cam00'):
            prior = next(x for x in old_detail['scenes'][scene]['observations'] if x['camera'] == camera)
            files = {k: Path(prior['files'][k]['path']) for k in ('hr', 'lr', 'swinir')}
            files['early'] = ROOT / f'output/dynamic_sr_20260920/output_swap/{scene}/G6C6' / (
                'predictions/0040.png' if camera == 'cam00' else 'train_predictions/cam02/0040.png')
            for m, p in runs.items():
                files[m] = BASE / f'{scene}_{m}_evaluation/predictions/0040.png' if camera == 'cam00' else p / 'train_renders_18000/cam02/0040.png'
            images = {k: helpers.read(p) for k, p in files.items()}
            lr = images.pop('lr')
            images['bicubic'] = helpers.resize(lr, images['hr'].shape[:2])
            if camera == 'cam00':
                mask = np.asarray(Image.open(prior['mask_source']['path'])) > 0
            else:
                selected = [x for x in manifest['observations'] if x['camera_id'] == camera and x['frame_index'] in (0, 40, 80, 118)]
                stack = np.stack([helpers.read(data / x['hr_path']) for x in selected])
                mask = np.max(np.std(stack, axis=0), axis=-1) > .025
                mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
                mask = cv2.dilate(mask, np.ones((5, 5), np.uint8)) > 0
            metrics, supported, _ = detail.measure(images, lr, mask)
            earlier_mask = np.asarray(Image.open(OLD / f'final_assessment/detail_audit/{scene}_{camera}_0040_hr_supported_mask.png')) > 0
            assert np.array_equal(supported, earlier_mask), 'The HR-supported region changed'
            path = DEST / f'{scene}_{camera}_0040.png'
            panel(path, scene, camera, images, prior['crops'])
            observations.append(dict(camera=camera, frame=40, metrics=metrics, crops=prior['crops'],
                                     panel=str(path), supported_mask_identical_to_earlier=True,
                                     files={k: dict(path=str(p), sha256=helpers.sha(p)) for k, p in files.items()}))
        result['scenes'][scene] = dict(pair_checks=checks, observations=observations)
    result['elapsed_seconds'] = time.monotonic() - started
    result['detail_definitions'] = old_detail['protocol']
    (DEST / 'metrics.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    (DEST / 'analysis_source.py').write_text(Path(__file__).read_text())
    for scene, value in result['scenes'].items():
        print(scene, value['pair_checks'])
        for obs in value['observations']:
            x = obs['metrics']['hr_supported_sr_improvement']
            print(obs['camera'], 'supported_fraction', x['fraction'],
                  {m: {k: x['methods'][m][k] for k in ('mse', 'low_mse', 'high_mse', 'high_correlation')}
                   for m in ('early', 'joint', 'appearance_only')})


if __name__ == '__main__':
    main()
