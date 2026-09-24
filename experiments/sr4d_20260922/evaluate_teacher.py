#!/usr/bin/env python3
"""Fixed 16 training-observation SwinIR/bicubic references, evaluation only.

Read frozen author float LR and frozen teacher PNGs, never held-out LR. Reuse
the exact comparison metrics; no training, reconstruction or teacher inference.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F

from evaluate_comparison import (ManifestData, choose_rows, read_lr, image_scores,
                                  read_rgb, tensor_state_hash, write_json, sha256)

HERE = Path(__file__).resolve().parent
METRICS = ('mse', 'psnr', 'ssim', 'lpips_alex', 'scale_high_residual_mse',
           'scale_high_residual_pred_energy', 'scale_high_residual_gt_energy',
           'lr_bilinear_mse', 'lr_box_mse')


def fixed_training_rows(data, cameras, frames, test_camera):
    selected, selection = choose_rows(data, cameras, frames, test_camera)
    rows = [row for group, row in selected if group == 'teacher_camera_train']
    if len(cameras) != 4 or len(frames) != 4 or len(set(frames)) != 4 or len(rows) != 16:
        raise ValueError('Teacher reference requires the same four cameras x four frames = 16 observations')
    if any(row['split'] != 'train' for row in rows):
        raise ValueError('Only training observations may enter teacher reference evaluation')
    return rows, selection


def summarize(rows):
    result = {}
    for method in ('bicubic', 'swinir'):
        mean = lambda part: {key: float(np.mean([row[method][key] for row in part])) for key in METRICS}
        result[method] = {'count': len(rows), 'metrics': mean(rows),
                          'per_camera': {cid: mean([r for r in rows if r['camera_id'] == cid])
                                         for cid in sorted({r['camera_id'] for r in rows})}}
    result['swinir_minus_bicubic'] = {
        key: result['swinir']['metrics'][key] - result['bicubic']['metrics'][key] for key in METRICS}
    return result


def write_report(path, result):
    means = result['summary']
    lines = [f"# {result['scene']}：固定训练观察的 SR 教师参照", '',
             '这组评价回答：输入相同低分辨率图像时，冻结 SwinIR 教师相对双三次放大，提供了多少接近 HR 参考的细节。它评价二维先验本身，不是新视角重建。', '',
             '仅使用与重建比较相同的 4 个训练相机、4 个固定时刻，共 16 张图。没有读取留出视角 LR，没有训练或重新生成教师图。', '',
             '| 参照 | PSNR ↑ | SSIM ↑ | LPIPS-Alex ↓ | 尺度细节残差 MSE ↓ | bilinear LR 闭环 MSE ↓ | box LR 闭环 MSE ↓ |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for method, label in [('bicubic', 'float LR 双三次放大'), ('swinir', '冻结 SwinIR PNG')]:
        row = means[method]['metrics']
        lines.append(f"| {label} | {row['psnr']:.6f} | {row['ssim']:.6f} | {row['lpips_alex']:.6f} | {row['scale_high_residual_mse']:.8g} | {row['lr_bilinear_mse']:.8g} | {row['lr_box_mse']:.8g} |")
    lines += ['', 'PSNR 是逐图 dB 的均值；SSIM 与重建评价复用同一项目函数；LPIPS 使用 AlexNet v0.1，输入归一化到 [-1,1]。SwinIR 按实际训练使用的冻结 8-bit PNG 评价，双三次结果保持浮点，先验量化差异予以保留。', '',
              '尺度细节残差定义为原图减去“bilinear 无抗混叠缩小，再双三次放大”的结果，两种参照与 HR 残差的差异取均方误差。它不是正交频带分解，也不能单凭残差能量证明细节真实。', '',
              'LR 闭环分别用作者 bilinear 输入退化和 SR4D 的 4×4 box 平均算子计算。二者含义不同，分列报告；闭环更好不能单独证明 HR 内容真实。', '',
              '**单图 SR 不是新视角重建的严格上界。** 教师直接看到对应训练视角的 LR；新视角方法没有相应留出 LR 输入。多视角重建也可能汇集教师单图没有的信息。应在相同训练观察上比较“先验质量”与“模型保留先验的程度”，不能用这里的均分宣称新视角理论上限。', '',
              '教师模型仍是原冻结 Classical ×4 DF2K SwinIR，其训练退化与当前 bilinear 输入不完全相同；本次不据评价结果调模型或选择样例。', '',
              '| 相机 | 帧索引 | 时间 |', '| --- | ---: | ---: |']
    lines += [f"| {r['camera_id']} | {r['frame_index']} | {r['time']:.9g} |" for r in result['rows']]
    lines += ['', '逐图全部指标、输入与教师哈希、LPIPS 权重状态哈希和源代码哈希见同目录 `metrics.json`；预先固定的行身份见 `selection.json`。', '']
    Path(path).write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--prior-cameras', required=True)
    parser.add_argument('--train-frames', default='0,40,80,118')
    parser.add_argument('--test-camera', default='cam00', help='Metadata selection compatibility only; no held-out LR read')
    parser.add_argument('--out', required=True)
    parser.add_argument('--audit-selection-only', action='store_true', help='CPU-only metadata check; does not require teacher cache')
    args = parser.parse_args()
    started = time.monotonic()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    try:
        torch.set_num_threads(4)
        data = ManifestData(args.manifest)
        if data.document.get('comparison_protocol') != 'sr4d_author_lr_v1':
            raise ValueError('Requires frozen author float-LR comparison manifest')
        frames = [int(frame) for frame in args.train_frames.split(',')]
        rows, selection = fixed_training_rows(data, args.prior_cameras.split(','), frames, args.test_camera)
        identities = [{'camera_id': row['camera_id'], 'frame_index': row['frame_index'],
                       'time': row['time'], 'split': row['split'],
                       'lr_float_path': row['lr_float_path'], 'lr_float_sha256': row['lr_float_sha256'],
                       'hr_path': row['hr_path'], 'hr_sha256': row['hr_sha256']} for row in rows]
        manifest_hash = sha256(data.path)
        selection = {'manifest': str(data.path), 'manifest_sha256': manifest_hash,
                     'comparison_selection': selection, 'selected_rows': identities,
                     'count': len(rows), 'heldout_lr_reads': 0,
                     'rule': 'Exactly evaluate_comparison teacher_camera_train rows; no metric-driven selection'}
        write_json(out / 'selection.json', selection)
        if args.audit_selection_only:
            write_json(out / 'status.json', {'state': 'selection_checked', 'gpu_used': False, 'count': len(rows)})
            return
        prior_root = data.root / 'sr_swinir_x4'
        config_path, receipt_path = prior_root / 'prior_config.json', prior_root / 'complete.json'
        config = json.loads(config_path.read_text())
        receipt = json.loads(receipt_path.read_text())
        if config['manifest_sha256'] != manifest_hash or receipt['config'] != config or receipt['status'] != 'complete':
            raise ValueError('Teacher cache is incomplete or belongs to a different manifest/config')
        if not set(args.prior_cameras.split(',')) <= set(config['cameras']):
            raise ValueError('Requested camera lacks a declared teacher cache')
        cached = {(row['camera_id'], row['frame']): row for row in receipt['rows']}
        if len(cached) != len(receipt['rows']):
            raise ValueError('Duplicate teacher cache identities')
        selected_cache = []
        for row in rows:
            cached_row = cached[(row['camera_id'], row['frame_index'])]
            path = prior_root / row['camera_id'] / Path(row['lr_path']).name
            if cached_row['lr_sha256'] != row['lr_float_sha256'] or sha256(path) != cached_row['prior_sha256']:
                raise ValueError('Teacher image/input identity mismatch')
            selected_cache.append((path, cached_row['prior_sha256']))
        if not torch.cuda.is_available():
            raise RuntimeError('Run perceptual evaluation on a remote GPU; local CPU audit is supported separately')
        import lpips
        torch.cuda.reset_peak_memory_stats()
        model = lpips.LPIPS(net='alex', version='0.1').cuda().eval().requires_grad_(False)
        source_paths = [Path(__file__), HERE / 'evaluate_comparison.py', HERE / 'sr4d_common.py',
                        HERE.parent / 'dynamic_sr_20260918/evaluate.py']
        identity = {'source_hashes': {str(path): sha256(path) for path in source_paths},
                    'prior_config': config, 'prior_config_sha256': sha256(config_path),
                    'prior_complete_sha256': sha256(receipt_path),
                    'lpips_state_sha256': tensor_state_hash(model),
                    'environment': {'host': platform.node(), 'python': sys.version, 'torch': torch.__version__,
                                    'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(),
                                    'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES')}}
        write_json(out / 'identity.json', identity)
        evaluated = []
        with torch.inference_mode():
            for row, (prior_path, prior_hash) in zip(rows, selected_cache):
                if row['split'] != 'train':
                    raise RuntimeError('Refuse held-out LR access')
                lr = read_lr(data, row).cuda()
                hr_path = data.root / row['hr_path']
                if sha256(hr_path) != row['hr_sha256']:
                    raise ValueError('HR reference identity mismatch')
                hr = read_rgb(hr_path)
                teacher = torch.from_numpy(np.ascontiguousarray(read_rgb(prior_path))).permute(2, 0, 1).cuda()
                bicubic = F.interpolate(lr[None], size=(data.hr_wh[1], data.hr_wh[0]),
                                        mode='bicubic', align_corners=False)[0]
                item = {'camera_id': row['camera_id'], 'frame_index': row['frame_index'],
                        'time': row['time'], 'split': row['split'], 'hr_sha256': row['hr_sha256'],
                        'lr_float_sha256': row['lr_float_sha256'], 'prior_path': str(prior_path),
                        'prior_sha256': prior_hash}
                for method, prediction in [('bicubic', bicubic), ('swinir', teacher)]:
                    metric = image_scores(prediction, hr, model, data.lr_wh)
                    # Exactly the model comparison convention: D(raw), then clamp.
                    bilinear = F.interpolate(prediction[None], size=lr.shape[-2:],
                                             mode='bilinear', align_corners=False)[0].clamp(0, 1)
                    box = F.avg_pool2d(prediction[None], kernel_size=4, stride=4)[0].clamp(0, 1)
                    metric['lr_bilinear_mse'] = float((bilinear - lr).square().mean())
                    metric['lr_box_mse'] = float((box - lr).square().mean())
                    item[method] = metric
                evaluated.append(item)
        if sha256(data.path) != manifest_hash or sha256(config_path) != identity['prior_config_sha256'] or sha256(receipt_path) != identity['prior_complete_sha256']:
            raise RuntimeError('Frozen input/cache metadata changed during evaluation')
        result = {'complete': True, 'scene': data.scene, 'selection': selection, 'identity': identity,
                  'rows': evaluated, 'summary': summarize(evaluated), 'elapsed_s': time.monotonic() - started,
                  'peak_GiB': torch.cuda.max_memory_allocated() / 2 ** 30, 'heldout_lr_reads': 0,
                  'limitations': ['Single-image SR is not a strict novel-view reconstruction upper bound',
                                  'Fixed training-view diagnostic, not held-out reconstruction performance',
                                  'Teacher is frozen 8-bit PNG; bicubic is float prediction',
                                  'Scale residual is not an orthogonal frequency band or proof of genuine geometry',
                                  'Classical bicubic-trained SwinIR faces the shared bilinear input degradation']}
        write_json(out / 'metrics.json', result)
        write_report(out / 'report.md', result)
        write_json(out / 'status.json', {'state': 'completed', 'count': len(evaluated),
                                        'metrics_sha256': sha256(out / 'metrics.json'),
                                        'elapsed_s': result['elapsed_s']})
        print(json.dumps({'complete': True, 'scene': data.scene, 'summary': result['summary']}), flush=True)
    except BaseException:
        write_json(out / 'status.json', {'state': 'failed', 'traceback': traceback.format_exc(),
                                        'elapsed_s': time.monotonic() - started})
        raise


if __name__ == '__main__':
    main()
