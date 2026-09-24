#!/usr/bin/env python3
"""One-shot CPU summary of completed N3DV pilot JSONs; never starts/polls jobs.

Missing files/fields remain null and are shown as missing, never as zero.
Frame averages are descriptive; no frame-based confidence intervals are made.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

from PIL import Image, ImageDraw, ImageFont


PROJECT = Path(__file__).resolve().parents[2]
VARIANTS = ['warmup', 'lr_native', 'lr_integrated', 'joint', 'appearance', 'frozen', 'hr_reference']
METRICS = {
    'full_psnr': ('aggregate', 'full', 'psnr_mean'),
    'full_ssim': ('aggregate', 'full', 'ssim_mean'),
    'full_lpips': ('aggregate', 'full', 'lpips_alex_mean'),
    'variation_psnr': ('aggregate', 'dynamic', 'psnr_mean'),
    'variation_ssim': ('aggregate', 'dynamic', 'ssim_mean'),
    'variation_lpips_spatial': ('aggregate', 'dynamic', 'lpips_alex_spatial_mask_mean'),
    'full_temporal_gt_relative': ('temporal_aggregate', 'full', 'gt_relative_warp_l1_mean'),
    'variation_temporal_gt_relative': ('temporal_aggregate', 'dynamic', 'gt_relative_warp_l1_mean'),
    'variation_flow_valid_fraction': ('temporal_aggregate', 'dynamic', 'valid_fraction_of_roi_mean'),
}


def number(value):
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else None


def at(obj, keys):
    for key in keys:
        if not isinstance(obj, dict) or key not in obj:
            return None
        obj = obj[key]
    return obj


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    path = Path(path)
    if not path.is_file():
        return None, {'path': str(path), 'status': 'missing'}
    try:
        data = json.loads(path.read_text())
        if not isinstance(data, dict):
            raise ValueError('Root must be a JSON object')
        return data, {'path': str(path), 'status': 'available', 'sha256': sha(path)}
    except (OSError, ValueError) as exc:
        return None, {'path': str(path), 'status': 'unreadable', 'error': str(exc)}


def mean_present(values):
    values = [number(x) for x in values]
    values = [x for x in values if x is not None]
    return mean(values) if values else None


def peak_memory(log_path):
    if not log_path.is_file():
        return None
    values = []
    for line in log_path.read_text().splitlines():
        try:
            val = number(json.loads(line).get('allocated_gb'))
            if val is not None:
                values.append(val)
        except ValueError:
            pass
    return max(values) if values else None


def read_variant(root, scene, prefix, variant):
    train_dir = root / f'{scene}_{prefix}_{variant}'
    eval_dir = root / f'{scene}_{prefix}_warmup_evaluation' if variant == 'warmup' else train_dir / 'evaluation'
    data, source = read_json(eval_dir / 'metrics.json')
    values = {name: number(at(data, path)) for name, path in METRICS.items()}
    metadata = data.get('checkpoint_metadata', {}) if data else {}
    if variant == 'warmup' and data and data.get('checkpoint'):
        train_dir = Path(data['checkpoint']).parent
    complete, complete_source = read_json(train_dir / 'complete.json')
    if not metadata and complete:
        metadata = complete
    cost = {'training_seconds': number(metadata.get('elapsed_s')),
            'evaluation_seconds': number(data.get('elapsed_seconds')) if data else None,
            'gaussian_count': number(data.get('gaussian_count')) if data else None,
            'peak_logged_training_allocated_gb': peak_memory(train_dir / 'training.jsonl'),
            'cost_scope': 'from-initialization training' if variant in ['warmup', 'hr_reference'] else 'branch continuation only; shared warmup and prior excluded'}
    if source['status'] == 'available':
        status = 'available' if all(x is not None for x in values.values()) else 'available_with_missing_fields'
    else:
        status = source['status']
    return {'variant': variant, 'status': status, 'metrics': values, 'cost': cost,
            'sources': {'metrics': source, 'training_complete': complete_source},
            'evaluation_directory': str(eval_dir),
            'checkpoint_sha256': data.get('checkpoint_sha256') if data else None,
            'manifest_sha256': data.get('manifest_sha256') if data else None,
            'evaluation_cache_key': data.get('cache_key') if data else None,
            'frame_indices': data.get('frame_indices') if data else None,
            'seed': metadata.get('seed'), 'dynamic_fraction': number(data.get('dynamic_fraction')) if data else None}


def summarize_sampling(root, scene, prefix):
    # Corrected operator: D(raw HR render), then clamp, matching training.
    # The legacy suite clamped HR before D; do not silently treat them as equal.
    corrected_path = root / f'{scene}_{prefix}_sampling_warmup_linear' / 'sampling.json'
    data, source = read_json(corrected_path)
    source_kind = 'corrected_linear_before_downsample'
    if source['status'] == 'missing':
        data, source = read_json(root / f'{scene}_{prefix}_sampling_warmup' / 'sampling.json')
        source_kind = 'legacy_clamp_before_downsample'
    rows = []
    native = number(at(data, ('aggregate', 'render_1x_lr_grid', 'psnr_to_real_lr_mean')))
    for factor in (1, 2, 4):
        key = f'render_{factor}x_lr_grid'
        score = number(at(data, ('aggregate', key, 'psnr_to_real_lr_mean')))
        rows.append({'factor': factor, 'psnr_to_real_lr': score,
                     'delta_psnr_vs_native_grid': score - native if score is not None and native is not None else None,
                     'mse_to_real_lr': number(at(data, ('aggregate', key, 'mse_to_real_lr_mean'))),
                     'mse_to_native_lr_render': number(at(data, ('aggregate', key, 'mse_to_native_lr_render_mean')))})
    return {'source': source, 'source_kind': source_kind,
            'operator_order': data.get('operator_order') if data and data.get('operator_order') else
                ('legacy HR-clamp-before-D; not equivalent to corrected D(raw-HR)-then-clamp' if source_kind.startswith('legacy') else 'corrected directory; operator_order field missing'),
            'preferred_corrected_path': str(corrected_path),
            'rows': rows, 'elapsed_seconds': number(data.get('elapsed_seconds')) if data else None}


def summarize_probe(root, scene, prefix):
    candidates = [root / f'{scene}_{prefix}_coupling_probe{suffix}.json'
                  for suffix in ['_paired_integrated', '_paired', '']]
    selected_path = next((p for p in candidates if p.is_file()), candidates[0])
    data, source = read_json(selected_path)
    output = []
    for kind in ['lr', 'lr_null_1', 'lr_null_2', 'sr_joint', 'sr_appearance', 'hr_oracle', 'shifted_sr']:
        selected = [r for r in data.get('rows', []) if r.get('kind') == kind] if data else []
        targets = []
        for row in selected:
            target = row.get('target_frame')
            other = [p for p in row.get('probes', []) if p.get('frame') != target]
            motion = row.get('learned_xyz_increment_vs_lr', {})
            targets.append({'target_frame': target,
                'mean_probe_delta_vs_lr_step': number(row.get('mean_probe_delta_vs_lr_step')),
                'cross_time_lr_delta_vs_lr_step': mean_present(p.get('delta_vs_lr_step') for p in other),
                'cross_time_probe_count': len(other),
                'cross_time_learned_xyz_increment_vs_lr': mean_present(v for t, v in motion.items() if str(t) != str(target)),
                'parameter_increment_vs_lr_l2': number(row.get('parameter_increment_vs_lr_l2')),
                'nonappearance_parameter_increment_max_abs': number(row.get('nonappearance_parameter_increment_max_abs'))})
        output.append({'kind': kind, 'target_count': len(targets), 'targets': targets,
                       'mean_probe_delta_vs_lr_step': mean_present(t['mean_probe_delta_vs_lr_step'] for t in targets),
                       'cross_time_lr_delta_vs_lr_step': mean_present(t['cross_time_lr_delta_vs_lr_step'] for t in targets),
                       'cross_time_learned_xyz_increment_vs_lr': mean_present(t['cross_time_learned_xyz_increment_vs_lr'] for t in targets),
                       'parameter_increment_vs_lr_l2': mean_present(t['parameter_increment_vs_lr_l2'] for t in targets),
                       'nonappearance_parameter_increment_max_abs': max((t['nonappearance_parameter_increment_max_abs'] for t in targets if t['nonappearance_parameter_increment_max_abs'] is not None), default=None)})
    return {'source': source, 'rows': output, 'checkpoint': data.get('checkpoint') if data else None,
            'protocol': data.get('protocol', data.get('optimizer')) if data else None,
            'interpretation': 'Descriptive mean across target interventions. Positive LR delta is worse than the identical-state LR-only step. Learned XYZ change is representation movement, not true motion error or meters. HR oracle uses privileged training HR only for diagnosis; shifted SR is an artificial perturbation.'}


def font(size):
    path = Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
    return ImageFont.truetype(str(path), size) if path.is_file() else ImageFont.load_default()


def make_collage(scene, records, manifest_root, out, roi):
    frame = 40
    paths = {}
    manifest, msource = read_json(manifest_root / scene / 'manifest.json')
    if manifest:
        obs = next((o for o in manifest.get('observations', []) if o.get('split') == 'test'
                    and o.get('camera_id') == 'cam00' and o.get('frame_index') == frame), None)
        if obs:
            paths['GT'] = manifest_root / scene / obs['hr_path']
    for key in ['lr_integrated', 'joint', 'appearance']:
        paths[key] = Path(records[key]['evaluation_directory']) / 'predictions' / '0040.png'
    # Available GT visual copies are exactly the same frame, not guessed inputs.
    if 'GT' not in paths or not paths['GT'].is_file():
        for record in records.values():
            candidate = Path(record['evaluation_directory']) / 'visuals' / '0040_gt.png'
            if candidate.is_file():
                paths['GT'] = candidate
                break
    labels = ['GT', 'lr_integrated', 'joint', 'appearance']
    reference = next((paths[x] for x in labels if x in paths and paths[x].is_file()), None)
    if reference is None:
        return {'status': 'missing_all_images', 'frame': frame, 'panels': {x: str(paths.get(x, '')) for x in labels}}
    with Image.open(reference) as im:
        w, h = im.size
    crop = [round(roi[0] * w), round(roi[1] * h), round(roi[2] * w), round(roi[3] * h)]
    cw, ch = crop[2] - crop[0], crop[3] - crop[1]
    tile_w, tile_h = 420, round(420 * ch / cw)
    margin, title_h, label_h, footer_h = 16, 48, 30, 40
    canvas = Image.new('RGB', (margin * 5 + tile_w * 4, title_h + label_h + tile_h + footer_h), '#f4f4f4')
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 10), f'{scene} | cam00 frame 0040 | fixed central crop {crop}', fill='#111111', font=font(21))
    panels = []
    for i, label in enumerate(labels):
        x = margin + i * (tile_w + margin)
        y = title_h + label_h
        draw.text((x, title_h), label, fill='#111111', font=font(19))
        path = paths.get(label)
        if path and path.is_file():
            with Image.open(path) as im:
                im = im.convert('RGB')
                if im.size != (w, h):
                    raise ValueError(f'Cannot compare mismatched image sizes: {path}: {im.size} vs {(w,h)}')
                panel = im.crop(tuple(crop)).resize((tile_w, tile_h), Image.Resampling.LANCZOS)
            canvas.paste(panel, (x, y))
            panels.append({'label': label, 'status': 'available', 'path': str(path), 'sha256': sha(path)})
        else:
            draw.rectangle((x, y, x + tile_w, y + tile_h), fill='#dddddd')
            draw.text((x + 18, y + 25), 'MISSING: no completed image', fill='#555555', font=font(17))
            panels.append({'label': label, 'status': 'missing', 'path': str(path) if path else None})
    draw.text((margin, title_h + label_h + tile_h + 12),
              'Same crop and display resize for every panel; qualitative view only, no score inferred from this frame.',
              fill='#444444', font=font(15))
    destination = out / f'{scene}_frame0040_central_roi.png'
    canvas.save(destination)
    return {'status': 'complete' if all(p['status'] == 'available' for p in panels) else 'partial',
            'path': str(destination), 'frame': frame, 'crop_xyxy': crop, 'crop_normalized': roi,
            'input_image_size': [w, h], 'panels': panels, 'manifest_source': msource}


def fmt(value, digits=4, sign=False):
    if number(value) is None:
        return '缺失'
    return f'{value:+.{digits}f}' if sign else f'{value:.{digits}f}'


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                     ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])


def markdown(summary):
    lines = ['# 通用动态 SR 先导结果快照', '', f"生成时间：{summary['created_at_utc']}。一次性读取；未等待、轮询或触发训练。", '',
             '缺失文件或字段均明确标识，不填零。下列帧均指标是描述性汇总，连续帧不作为独立统计样本；本表不给出基于帧数的显著性或置信区间。', '',
             '↑：PSNR、SSIM；↓：LPIPS、GT 相对时序残差。变化区 LPIPS 为官方 spatial-map 的固定 mask 均值，区别于标准全图 LPIPS；时间变化 ROI 包含亮度及边缘变化，不等同真实物体运动掩码。', '']
    for scene, scene_result in summary['scenes'].items():
        lines += [f'## {scene}', '', '质量（直接 HR 渲染）：hr_reference 使用额外训练 HR、独立训练和不同增密轨迹，仅作诊断，不是纯 LR 方法或严格上界。', '']
        rows = []
        for key in VARIANTS:
            r = scene_result['variants'][key]; m = r['metrics']
            rows.append([key, r['status'], fmt(m['full_psnr'], 3), fmt(m['full_ssim']), fmt(m['full_lpips']),
                         fmt(m['variation_psnr'], 3), fmt(m['variation_ssim']), fmt(m['variation_lpips_spatial']),
                         fmt(m['full_temporal_gt_relative'], 5), fmt(m['variation_temporal_gt_relative'], 5)])
        lines += [table(['方法', '状态', '全图 PSNR', 'SSIM', 'LPIPS', '变化区 PSNR', 'SSIM', 'LPIPS-map', '全图时间残差', '变化区时间残差'], rows), '',
                  '差值统一为“本方法 − lr_integrated”；负 LPIPS/时间残差表示更低误差。warmup 未经过分支继续优化，不能将其差值单独归因于某个 SR 机制。', '']
        lines += [table(['方法', 'Δ 全图 PSNR', 'Δ 变化区 PSNR', 'Δ 变化区 LPIPS-map', 'Δ 变化区时间残差'],
                         [[key] + [fmt(scene_result['variants'][key]['delta_vs_lr_integrated'][metric], 5, True)
                                   for metric in ['full_psnr', 'variation_psnr', 'variation_lpips_spatial', 'variation_temporal_gt_relative']]
                          for key in VARIANTS]), '', '计算成本：分支训练时间不包含共享 warmup 或 SR 先验生成时间。不同分支的相同步数不代表相同成本。', '']
        lines += [table(['方法', '本阶段训练秒', '评价秒', '高斯数', '日志峰值分配 GB', '变化区流有效比例'],
                         [[key, fmt(r['cost']['training_seconds'], 2), fmt(r['cost']['evaluation_seconds'], 2),
                           fmt(r['cost']['gaussian_count'], 0), fmt(r['cost']['peak_logged_training_allocated_gb'], 3),
                           fmt(r['metrics']['variation_flow_valid_fraction'])]
                          for key, r in scene_result['variants'].items()]), '', '固定 warmup 检查点的采样诊断：', '']
        lines += [f"采样来源：`{scene_result['sampling']['source_kind']}`；算子顺序：`{scene_result['sampling']['operator_order']}`。", '',
                  table(['渲染网格', '缩回 LR 对真实 LR PSNR', 'Δ 对 native 网格', '对 native 渲染 MSE'],
                         [[f"{r['factor']}×LR", fmt(r['psnr_to_real_lr'], 3), fmt(r['delta_psnr_vs_native_grid'], 3, True),
                           fmt(r['mse_to_native_lr_render'], 7)] for r in scene_result['sampling']['rows']]), '',
                  '单步干预：以下是相对同状态 LR-only Adam 更新的额外改变；正 LR 误差差值表示更差。XYZ 是学习表示的变化，不是三维运动真值误差。HR oracle 为特权信息诊断。', '']
        lines += [f"诊断来源：`{scene_result['probe']['source']['path']}`；检查点：`{scene_result['probe']['checkpoint']}`。", '',
                  table(['干预', '已完成目标数', '跨时间 LR 误差差值', '跨时间 XYZ 额外变化', '非外观参数最大额外变化'],
                         [[r['kind'], r['target_count'], fmt(r['cross_time_lr_delta_vs_lr_step'], 7, True),
                           fmt(r['cross_time_learned_xyz_increment_vs_lr'], 7), fmt(r['nonappearance_parameter_increment_max_abs'], 7)]
                          for r in scene_result['probe']['rows']]), '']
        if scene_result['collage'].get('path'):
            lines += [f"![固定第40帧中央区域]({scene_result['collage']['path']})", '']
        if scene_result['protocol_warnings']:
            lines += ['协议检查提示：' + '；'.join(scene_result['protocol_warnings']), '']
    lines += ['## 完整性与解释边界', '',
              f"已存在完整指标文件：{summary['available_metric_files']} / {summary['expected_metric_files']}。详见 summary.json 中每个文件的路径、状态、SHA256 和每个指标的原字段路径。", '',
              '这些结果是两个场景、短窗、当前随机种子的先导。不得据此直接主张完整 benchmark SOTA、真实运动精度提升或 CVPR 贡献已经成立。采样差异、空间保真与时间残差需要联合解释；单帧拼图不替代完整序列评价。']
    return '\n'.join(lines) + '\n'


def collect(args):
    out = args.out.resolve(); out.mkdir(parents=True, exist_ok=True)
    summary = {'schema': 'dynamic_sr_summary_v1', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
               'root': str(args.root.resolve()), 'prefix': args.prefix,
               'metric_field_paths': {k: list(v) for k, v in METRICS.items()},
               'baseline_for_deltas': 'lr_integrated', 'frame_statistics_policy': 'descriptive only; frames are correlated, no independent-frame confidence intervals',
               'expected_metric_files': len(args.scenes) * len(VARIANTS), 'available_metric_files': 0, 'scenes': {}}
    for scene in args.scenes:
        records = {v: read_variant(args.root, scene, args.prefix, v) for v in VARIANTS}
        baseline = records['lr_integrated']['metrics']
        for record in records.values():
            record['delta_vs_lr_integrated'] = {k: value - baseline[k] if value is not None and baseline[k] is not None else None
                                                for k, value in record['metrics'].items()}
            summary['available_metric_files'] += int(record['sources']['metrics']['status'] == 'available')
        warnings = []
        for field in ['manifest_sha256', 'evaluation_cache_key', 'seed', 'frame_indices']:
            available = [r[field] for r in records.values() if r[field] is not None]
            if len({json.dumps(x, sort_keys=True) for x in available}) > 1:
                warnings.append(f'{field} differs between available variants; investigate before causal comparison')
        scene_result = {'variants': records, 'sampling': summarize_sampling(args.root, scene, args.prefix),
                        'probe': summarize_probe(args.root, scene, args.prefix), 'protocol_warnings': warnings,
                        'collage': make_collage(scene, records, args.manifest_root, out, args.roi)}
        summary['scenes'][scene] = scene_result
    # Equal scene weighting only when every requested scene supplies that field.
    summary['scene_equal_means'] = {}
    for variant in VARIANTS:
        values = {}
        for metric in METRICS:
            scores = [summary['scenes'][s]['variants'][variant]['metrics'][metric] for s in args.scenes]
            values[metric] = mean(scores) if all(x is not None for x in scores) else None
        summary['scene_equal_means'][variant] = values
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=PROJECT / 'output/dynamic_sr_20260918')
    p.add_argument('--manifest-root', type=Path, default=PROJECT / 'data/dynamic_sr/n3dv_prepared')
    p.add_argument('--prefix', default='pilot_v1')
    p.add_argument('--scenes', nargs='+', default=['cook_spinach', 'cut_roasted_beef'])
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--roi', nargs=4, type=float, default=[.35, .38, .76, .99], metavar=('X0', 'Y0', 'X1', 'Y1'))
    p.add_argument('--overwrite', action='store_true', help='Explicitly replace a previous summary snapshot, never source results')
    args = p.parse_args()
    if not (0 <= args.roi[0] < args.roi[2] <= 1 and 0 <= args.roi[1] < args.roi[3] <= 1):
        p.error('--roi must be an ordered normalized rectangle within [0,1]')
    if (args.out / 'summary.json').exists() and not args.overwrite:
        raise FileExistsError('Summary exists; use a fresh --out or explicit --overwrite')
    result = collect(args)
    (args.out / 'summary.json').write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    (args.out / 'summary.md').write_text(markdown(result))
    print(json.dumps({'summary': str((args.out / 'summary.json').resolve()),
                      'markdown': str((args.out / 'summary.md').resolve()),
                      'available_metric_files': result['available_metric_files'],
                      'expected_metric_files': result['expected_metric_files']}))


if __name__ == '__main__':
    main()
