#!/usr/bin/env python3
"""Export four fixed training/dev examples from completed old RGB residual arms.

No training, new teacher generation, frame search or endpoint selection. The
four observations and three pixel boxes are fixed in this source. HR is read
only for evaluation/display and never supplied to the model renderer.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback

import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / 'experiments/dynamic_sr_20260918'
SCENE = ROOT / 'experiments/dynamic_sr_scene_residual_20260923'
sys.path.insert(0, str(OLD))


def load_source(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Explicit, unique module names prevent importing the new motion model or a
# sibling evaluate.py. These source paths are recorded with hashes below.
legacy = load_source('motion_examples_legacy_metric_helpers', OLD / 'evaluate.py')
scene = load_source('motion_examples_old_scene_residual_model', SCENE / 'residual_model.py')
from n3dv_data import load_manifest, observation_to_4dgs_camera

BRANCHES = ('joint', 'global_residual', 'local_residual')
FIXED_KEYS = (('cam02', 40), ('cam02', 80), ('cam01', 40), ('cam01', 80))
BOXES = {'clothing_reference': (504, 420, 672, 588),
         'chair_reference': (252, 420, 420, 588),
         'window_reference': (1092, 504, 1260, 672)}
LABELS = {'hr': 'True HR', 'lr_bicubic': 'LR bicubic', 'swinir': 'Cached SwinIR (train)',
          'joint': 'Joint 6000', 'global_residual': 'Global 6000', 'local_residual': 'Local 6000'}


def camera_from_calibration(m, observation, uid):
    width, height = m['resolutions']['hr']
    calibration = m['cameras'][observation['camera_id']]
    return observation_to_4dgs_camera({
        'image': torch.zeros((3, height, width), dtype=torch.float32),
        'camera_id': observation['camera_id'], 'frame_index': observation['frame_index'],
        'time': observation['time'], 'width': width, 'height': height,
        'K': np.asarray(calibration['K_hr'], dtype=np.float64),
        'w2c': np.asarray(calibration['w2c'], dtype=np.float64)}, uid)


def chw(array):
    return torch.from_numpy(np.ascontiguousarray(array)).permute(2, 0, 1)


def spatial_scores(pred, gt, metric, device):
    """Legacy full scalar LPIPS + full-image-map ROI metrics, computed once."""
    error = ((pred - gt) ** 2).mean(2)
    ssim = legacy.ssim_map_rgb(pred, gt)
    tx, ty = chw(pred)[None].to(device) * 2 - 1, chw(gt)[None].to(device) * 2 - 1
    scalar = float(metric(tx, ty).item())
    metric.spatial = True
    try:
        lpips_map = metric(tx, ty)[0, 0].cpu().numpy()
    finally:
        metric.spatial = False
    interior = np.zeros(error.shape, dtype=bool)
    interior[5:-5, 5:-5] = True
    masks = {'full': np.ones(error.shape, dtype=bool)}
    for name, (x0, y0, x1, y1) in BOXES.items():
        mask = np.zeros(error.shape, dtype=bool)
        mask[y0:y1, x0:x1] = True
        masks[name] = mask
    result = {}
    for name, mask in masks.items():
        mse = legacy.masked_mean(error, mask)
        result[name] = {'mse': mse, 'psnr': legacy.metric_psnr(mse),
                        'ssim': legacy.masked_mean(ssim, mask & interior),
                        'lpips_alex_spatial_mask': legacy.masked_mean(lpips_map, mask),
                        'pixels': int(mask.sum())}
        if name == 'full':
            result[name]['lpips_alex'] = scalar
    return result


def lr_projection_scores(raw_chw, real_lr):
    """Match the training closure: D(raw render), clamp afterward, no uint8."""
    projected = F.interpolate(raw_chw[None].float(), size=real_lr.shape[:2],
                              mode='bicubic', align_corners=False, antialias=True)[0].clamp(0, 1)
    error = projected - chw(real_lr)
    mse = float(error.square().mean())
    return {'mse': mse, 'l1': float(error.abs().mean()), 'psnr': legacy.metric_psnr(mse)}


def panels(folder, values, observation):
    images = {name: Image.fromarray(np.rint(np.clip(array, 0, 1) * 255).astype(np.uint8))
              for name, array in values.items()}
    files = {}
    title = f'{observation["split"]} {observation["camera_id"]} frame{observation["frame_index"]} | fixed diagnostic'
    for name, box in BOXES.items():
        panel = Image.new('RGB', (168 * len(images), 168 + 80), 'white')
        draw = ImageDraw.Draw(panel)
        draw.text((5, 5), title, fill='black')
        draw.text((5, 22), 'Fixed pixel boxes; different cameras do not show matched physical points.', fill='black')
        draw.text((5, 39), f'{name} xyxy={box}; semantic name is a reference, not a segmentation.', fill='black')
        for index, (label, image) in enumerate(images.items()):
            draw.text((index * 168 + 4, 62), LABELS[label], fill='black')
            panel.paste(image.crop(box), (index * 168, 80))
        destination = folder / f'roi_{name}.png'
        panel.save(destination)
        files[name] = str(destination)
    overview = Image.new('RGB', (1920, 820), 'white')
    draw = ImageDraw.Draw(overview)
    draw.text((6, 5), title + ' | display resized; native full RGB images saved separately', fill='black')
    for index, (label, image) in enumerate(images.items()):
        x, y = (index % 3) * 640, 35 + (index // 3) * 390
        draw.text((x + 6, y), LABELS[label], fill='black')
        overview.paste(image.resize((640, 360), Image.Resampling.LANCZOS), (x, y + 22))
    overview.save(folder / 'full_comparison.png')
    files['full_comparison'] = str(folder / 'full_comparison.png')
    annotated = images['hr'].copy()
    draw = ImageDraw.Draw(annotated)
    for name, (x0, y0, x1, y1) in BOXES.items():
        draw.rectangle((x0, y0, x1 - 1, y1 - 1), outline='#ff3030', width=2)
        draw.text((x0 + 3, y0 - 16), name, fill='#ff3030', stroke_width=1, stroke_fill='white')
    annotated.save(folder / 'hr_box_locations.png')
    files['hr_box_locations'] = str(folder / 'hr_box_locations.png')
    return files


def run(args):
    started = time.monotonic()
    m = load_manifest(args.manifest)
    if m['scene'] != 'meetroom_discussion' or m['resolutions']['hr'] != [1280, 720]:
        raise ValueError('The fixed boxes require original MeetRoom discussion 1280x720 RGB')
    manifest_hash = legacy.file_sha(args.manifest)
    observations = []
    for camera, frame in FIXED_KEYS:
        matches = [o for o in m['observations'] if (o['camera_id'], o['frame_index']) == (camera, frame)]
        if len(matches) != 1 or matches[0]['split'] != ('train' if camera == 'cam02' else 'dev'):
            raise ValueError(f'Unexpected registered observation: {camera}/{frame}')
        observations.append(matches[0])
    configs, checkpoints = {}, {}
    for branch in BRANCHES:
        folder = args.root / branch
        config = json.loads((folder / 'config.json').read_text())
        done = json.loads((folder / 'complete.json').read_text())
        if config['branch'] != branch or config['manifest_sha256'] != manifest_hash:
            raise ValueError(f'Old branch/manifest identity mismatch: {branch}')
        if done['status'] != 'completed' or done['parameter_updates'] != 6000 or done.get('smoke', False):
            raise ValueError(f'Old fixed 6000-step branch is incomplete: {branch}')
        configs[branch] = config
        path = folder / 'checkpoint_6000.pt'
        checkpoints[branch] = {'path': str(path.resolve()), 'sha256': legacy.file_sha(path)}
    if len({c['parent_sha256'] for c in configs.values()}) != 1:
        raise ValueError('Old branches do not share their parent checkpoint')
    data, input_files = {}, []
    for observation in observations:
        key = (observation['camera_id'], observation['frame_index'])
        entry = {'observation': observation, 'values': {}}
        for kind in ['hr', 'lr']:
            path = Path(m['_root']) / observation[f'{kind}_path']
            digest = legacy.file_sha(path)
            if digest != observation[f'{kind}_sha256']:
                raise ValueError(f'{kind} image identity mismatch: {path}')
            entry[kind] = legacy.read_rgb(path)
            input_files.append({'role': kind, 'path': str(path), 'sha256': digest})
        entry['values']['hr'] = chw(entry['hr'])
        entry['values']['lr_bicubic'] = F.interpolate(chw(entry['lr'])[None],
            size=entry['hr'].shape[:2], mode='bicubic', align_corners=False, antialias=False)[0].clamp(0, 1)
        if observation['split'] == 'train':
            matches = [r for r in configs['joint']['teacher_inputs']
                       if (r['camera'], r['frame']) == key]
            if len(matches) != 1:
                raise ValueError(f'Missing recorded frozen teacher: {key}')
            teacher = Path(matches[0]['path'])
            digest = legacy.file_sha(teacher)
            if digest != matches[0]['sha256']:
                raise ValueError(f'Frozen teacher identity mismatch: {teacher}')
            entry['values']['swinir'] = chw(legacy.read_rgb(teacher))
            input_files.append({'role': 'existing_train_swinir', 'path': str(teacher), 'sha256': digest})
        data[key] = entry
    with torch.inference_mode():
        for branch in BRANCHES:
            model = scene.load_model(checkpoints[branch]['path'], manifest=m, restore_rng=False)
            if model.branch != branch or model.checkpoint['metadata']['intervention_step'] != 6000:
                raise ValueError(f'Wrong loaded branch/endpoint: {branch}')
            if model.checkpoint['metadata']['manifest_sha'] != manifest_hash:
                raise ValueError(f'Loaded checkpoint/manifest mismatch: {branch}')
            model.g._deformation.eval()
            if model.detail is not None:
                model.detail.eval()
            for uid, (key, entry) in enumerate(data.items()):
                camera = camera_from_calibration(m, entry['observation'], uid)
                raw = scene.render_model(model, camera)['render']
                if tuple(raw.shape) != (3, 720, 1280) or not bool(torch.isfinite(raw).all()):
                    raise ValueError(f'Invalid old render: {branch}/{key}')
                entry['values'][branch] = raw.detach().cpu()
                del raw, camera
            del model
            gc.collect()
            torch.cuda.empty_cache()
        import lpips
        metric = lpips.LPIPS(net='alex', spatial=False).to(args.lpips_device).eval().requires_grad_(False)
        rows, displays = [], []
        for key, entry in data.items():
            observation = entry['observation']
            folder = args.out / f'{observation["split"]}_{key[0]}_frame{key[1]:04d}'
            folder.mkdir()
            values = entry['values']
            order = ['hr', 'lr_bicubic', *BRANCHES] + (['swinir'] if 'swinir' in values else [])
            displayed = {}
            for variant in order:
                raw = values[variant]
                pred = legacy.image_array(raw)
                displayed[variant] = pred
                path = folder / f'{variant}.png'
                legacy.write_rgb(path, pred)
                rows.append({'camera_id': key[0], 'frame_index': key[1], 'split': observation['split'],
                             'variant': variant, 'full_image': str(path),
                             'spatial': spatial_scores(pred, entry['hr'], metric, args.lpips_device),
                             'lr_backprojection': lr_projection_scores(raw, entry['lr']),
                             'render_clipped_fraction': float(((raw < 0) | (raw > 1)).float().mean())})
            displays.append({'camera_id': key[0], 'frame_index': key[1], 'split': observation['split'],
                             'files': panels(folder, displayed, observation)})
            print(f'EXPORTED fixed {observation["split"]} {key[0]} frame{key[1]}', flush=True)
    result = {'status': 'completed_fixed_examples', 'parameter_updates': 0, 'observations_per_model': 4,
              'finished_utc': datetime.now(timezone.utc).isoformat(),
              'manifest': str(args.manifest.resolve()), 'manifest_sha256': manifest_hash,
              'fixed_keys': FIXED_KEYS, 'fixed_endpoint': 6000, 'checkpoints': checkpoints,
              'input_files': input_files, 'boxes_xyxy_exclusive': BOXES, 'rows': rows, 'displays': displays,
              'sources': {str(path): legacy.file_sha(path) for path in
                          [Path(__file__), OLD / 'evaluate.py', SCENE / 'residual_model.py', OLD / 'common.py']},
              'gpu': torch.cuda.get_device_name(), 'visible_cuda': os.environ.get('CUDA_VISIBLE_DEVICES'),
              'lpips_device': args.lpips_device, 'elapsed_seconds': time.monotonic() - started,
              'information_boundary': 'Only calibration/time and a zero RGB placeholder enter render. HR for metric/display only. cam02 uses its recorded existing frozen teacher. No dev teacher generated/read, training, gradients or parameter updates.',
              'metric_definitions': {
                  'full': 'Legacy float RGB clipped [0,1], RGB PSNR with MSE floor1e-12 and 11x11 sigma1.5 SSIM excluding outer5; standard scalar AlexNet v0.1 LPIPS.',
                  'roi': 'Same full-image SSIM and LPIPS spatial maps averaged inside fixed pixel boxes. Outside-box context retained. ROI LPIPS is not the scalar LPIPS of a cropped image.',
                  'lr_backprojection': 'Bicubic AA align_corners=False downsample of unclipped raw HR render, then clamp[0,1], compared with real quantized LR PNG; no rounding. MSE/L1/PSNR over complete LR image.',
                  'lr_bicubic': 'Real LR PNG enlarged bicubic align_corners=False antialias=False, then clamp[0,1].',
                  'png': 'Display only: clipped float RGB rounded to uint8; metrics are computed before PNG quantization; ROIs at native168x168 and full contact sheets resized for display.'},
              'limitations': ['Four fixed correlated observations only, two train cam02 and two dev cam01; no full-training average or independent validation claim.',
                             'Same pixel coordinates across cameras are not matched physical points. Clothing/chair/window names are reference labels, not guaranteed semantic content or masks.',
                             'No additional frames, grids, temporal metrics, current B/C outputs or teacher generation.']}
    legacy.write_json(args.out / 'metrics.json', result)
    lines = ['# 固定旧结果图像证据：cam02 训练与 cam01 开发', '',
             '仅帧 40/80、旧三模型追加 6000 步，共四个固定观察；不代表完整训练均分或独立验证。HR 仅用于评价，cam02 另列原缓存 SwinIR；cam01 不生成或读取教师。', '',
             '三个 168×168 框按固定像素坐标取样。跨相机不对应同一物理位置；衣物、椅背、窗边只是参考名称，不保证另一相机该框包含同类对象。', '',
             '指标保留浮点精度，PNG 为量化展示。区域 SSIM/LPIPS 保留全图上下文，区域 LPIPS 与全图标准标量定义不同。LR 回投由未裁剪渲染降采样后裁剪，与真实 LR PNG 比较。', '',
             '| 相机/帧/划分 | 方法 | 全图 PSNR | SSIM | LPIPS | LR MSE | LR L1 | LR PSNR |',
             '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in rows:
        full, lr = row['spatial']['full'], row['lr_backprojection']
        lines.append(f'| {row["camera_id"]}/{row["frame_index"]}/{row["split"]} | {LABELS[row["variant"]]} | {full["psnr"]:.6f} | {full["ssim"]:.6f} | {full["lpips_alex"]:.6f} | {lr["mse"]:.8g} | {lr["l1"]:.8g} | {lr["psnr"]:.6f} |')
    lines += ['', '同位置三个 ROI 的 PSNR/SSIM/LPIPS 空间图均值及完整身份记录见 [metrics.json](metrics.json)。', '']
    for display in displays:
        lines += [f'{display["split"]} {display["camera_id"]}，帧 {display["frame_index"]}：', '',
                  f'![完整 RGB 对照]({display["files"]["full_comparison"]})', '']
        for name in BOXES:
            lines += [f'![{name} 固定原生像素框]({display["files"][name]})', '']
    (args.out / 'README.md').write_text('\n'.join(lines))
    legacy.write_json(args.out / 'complete.json', {'status': 'completed_fixed_examples',
                      'observations_per_model': 4, 'parameter_updates': 0,
                      'metrics_sha256': legacy.file_sha(args.out / 'metrics.json')})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT / 'output/dynamic_sr_scene_residual_20260923/batch_v1')
    parser.add_argument('--manifest', type=Path, default=ROOT / 'data/dynamic_sr/meetroom_prepared/discussion/manifest.json')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--lpips-device', choices=['cpu', 'cuda'], default='cuda')
    args = parser.parse_args()
    args.out = args.out.resolve()
    args.root = args.root.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    cv2.setNumThreads(4)
    try:
        run(args)
    except BaseException:
        legacy.write_json(args.out / 'failed.json', {'status': 'failed', 'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
