#!/usr/bin/env python3
"""Read one CPU-only snapshot of the six controlled-fit branches.

Never loads checkpoints/images, starts jobs, waits, or polls. Missing artifacts
and fields stay null/pending. HR-oracle results are privileged diagnostics.

Usage:
  /usr/bin/python3 experiments/dynamic_sr_20260919/summarize_controls.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[2]
CASES = {
    'lr_long': ('none', 0., False),
    'sr_w01': ('sr', .1, False),
    'sr_w10': ('sr', 1., False),
    'hr_oracle_w10': ('hr', 1., False),
    'sr_w10_dense': ('sr', 1., True),
    'lr_dense': ('none', 0., True),
}
SCENES = ('cook_spinach', 'cut_roasted_beef', 'meetroom_discussion')
STEPS = (1200, 6000, 12000, 18000)
FIT_KEYS = ('psnr', 'ssim', 'lpips_alex', 'mse')
EVAL_PATHS = {
    'full_psnr': ('aggregate', 'full', 'psnr_mean'),
    'full_ssim': ('aggregate', 'full', 'ssim_mean'),
    'full_lpips': ('aggregate', 'full', 'lpips_alex_mean'),
    'full_mse': ('aggregate', 'full', 'mse_mean'),
    'variation_psnr': ('aggregate', 'dynamic', 'psnr_mean'),
    'variation_ssim': ('aggregate', 'dynamic', 'ssim_mean'),
    'variation_lpips_spatial': ('aggregate', 'dynamic', 'lpips_alex_spatial_mask_mean'),
    'variation_mse': ('aggregate', 'dynamic', 'mse_mean'),
    'full_temporal_gt_relative_l1': ('temporal_aggregate', 'full', 'gt_relative_warp_l1_mean'),
    'variation_temporal_gt_relative_l1': ('temporal_aggregate', 'dynamic', 'gt_relative_warp_l1_mean'),
    'full_temporal_prediction_warp_l1': ('temporal_aggregate', 'full', 'prediction_warp_l1_mean'),
    'variation_temporal_prediction_warp_l1': ('temporal_aggregate', 'dynamic', 'prediction_warp_l1_mean'),
    'full_temporal_gt_warp_l1': ('temporal_aggregate', 'full', 'gt_warp_l1_mean'),
    'variation_temporal_gt_warp_l1': ('temporal_aggregate', 'dynamic', 'gt_warp_l1_mean'),
    'full_flow_valid_fraction': ('temporal_aggregate', 'full', 'valid_fraction_of_roi_mean'),
    'variation_flow_valid_fraction': ('temporal_aggregate', 'dynamic', 'valid_fraction_of_roi_mean'),
}
SAMPLING_KEYS = ('psnr_to_real_lr', 'mse_to_real_lr',
                 'mse_to_native_lr_render', 'psnr_to_native_lr_render')


def number(value):
    return float(value) if isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value) else None


def at(obj, *keys):
    for key in keys:
        if not isinstance(obj, dict):
            return None
        obj = obj.get(key)
    return obj


def read_json(path):
    """Hash precisely the bytes parsed, even while the queue is still active."""
    source = {'path': str(path), 'status': 'pending'}
    if not path.is_file():
        return {}, source
    try:
        raw = path.read_bytes()
        obj = json.loads(raw)
        if not isinstance(obj, dict):
            raise ValueError('JSON root is not an object')
        source.update(status='available', sha256=hashlib.sha256(raw).hexdigest())
        return obj, source
    except (OSError, ValueError) as exc:
        source.update(status='unreadable', error=str(exc))
        return {}, source


def artifact_status(source, values):
    if source['status'] != 'available':
        return source['status']
    return 'available' if all(v is not None for v in values.values()) else 'partial'


def aggregation_audit(values, row_values):
    """Check recorded means against observations; never replace recorded data."""
    result = {}
    for key, recorded in values.items():
        vals = [number(x) for x in row_values.get(key, [])]
        valid = [v for v in vals if v is not None]
        recomputed = mean(valid) if valid else None
        difference = recorded - recomputed if recorded is not None and recomputed is not None else None
        status = 'pending' if difference is None else (
            'match' if abs(difference) <= 1e-8 else 'mismatch')
        result[key] = dict(status=status, valid_rows=len(valid), total_rows=len(vals),
                           recomputed_mean=recomputed, recorded_minus_recomputed=difference)
    return result


def read_fit(path, step):
    data, source = read_json(path)
    rows = data.get('rows', [])
    values = {f'{target}_{key}': number(at(data, 'aggregate', original, key))
              for target, original in [('render_hr', 'render_hr'), ('render_sr', 'render_prior')]
              for key in FIT_KEYS}
    values['lr_psnr'] = number(data.get('lr_psnr'))
    row_values = {f'{target}_{key}': [at(r, original, key) for r in rows]
                  for target, original in [('render_hr', 'render_hr'), ('render_sr', 'render_prior')]
                  for key in FIT_KEYS}
    row_values['lr_psnr'] = [r.get('lr_psnr') for r in rows]
    ids = [[r.get('camera_id'), r.get('frame_index')] for r in rows]
    return dict(status=artifact_status(source, values), source=source,
                requested_step=step, recorded_step=data.get('step'), metrics=values,
                observation_count=len(rows) if source['status'] == 'available' else None,
                observations=sorted(ids, key=str) if rows else None,
                aggregation_audit=aggregation_audit(values, row_values))


def read_evaluation(path):
    data, source = read_json(path)
    values = {key: number(at(data, *loc)) for key, loc in EVAL_PATHS.items()}
    rows = data.get('rows', [])
    row_values = {}
    for key, (agg, region, metric) in EVAL_PATHS.items():
        row_key = 'spatial' if agg == 'aggregate' else 'temporal'
        row_values[key] = [at(r, row_key, region, metric.removesuffix('_mean'))
                           for r in rows if row_key in r]
    return dict(status=artifact_status(source, values), source=source, metrics=values,
                manifest_sha256=data.get('manifest_sha256'), checkpoint_sha256=data.get('checkpoint_sha256'),
                frame_indices=data.get('frame_indices'), test_camera=data.get('test_camera'),
                cache_key=data.get('cache_key'), version=data.get('version'),
                dynamic_fraction=number(data.get('dynamic_fraction')),
                gaussian_count=number(data.get('gaussian_count')),
                elapsed_seconds=number(data.get('elapsed_seconds')),
                spatial_frame_count=len(rows) if rows else None,
                temporal_pair_count=sum('temporal' in r for r in rows) if rows else None,
                metric_definitions=data.get('metric_definitions'), lpips_status=data.get('lpips_status'),
                checkpoint_metadata=data.get('checkpoint_metadata'),
                aggregation_audit=aggregation_audit(values, row_values))


def read_sampling(path):
    data, source = read_json(path)
    values = {f'render_{factor}x_{key}': number(at(data, 'aggregate', f'render_{factor}x_lr_grid', f'{key}_mean'))
              for factor in (1, 2, 4) for key in SAMPLING_KEYS}
    rows = data.get('rows', [])
    row_values = {f'render_{factor}x_{key}': [at(r, 'sampling', f'render_{factor}x_lr_grid', key) for r in rows]
                  for factor in (1, 2, 4) for key in SAMPLING_KEYS}
    return dict(status=artifact_status(source, values), source=source, metrics=values,
                operator_order=data.get('operator_order'), manifest_sha256=data.get('manifest_sha256'),
                checkpoint_sha256=data.get('checkpoint_sha256'), frame_indices=data.get('frame_indices'),
                test_camera=data.get('test_camera'), version=data.get('version'),
                elapsed_seconds=number(data.get('elapsed_seconds')),
                aggregation_audit=aggregation_audit(values, row_values))


def log_snapshot(path):
    source = {'path': str(path), 'status': 'pending'}
    if not path.is_file():
        return {'source': source, 'last_step': None, 'last_points': None,
                'peak_allocated_gb': None, 'last_learning_rates': None, 'growth_events': []}
    try:
        raw = path.read_bytes()
    except OSError as exc:
        return {'source': {**source, 'status': 'unreadable', 'error': str(exc)},
                'last_step': None, 'last_points': None, 'peak_allocated_gb': None,
                'last_learning_rates': None, 'growth_events': []}
    source.update(status='available', sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
    rows, invalid = [], 0
    for line in raw.splitlines():
        try:
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError('Non-object log row')
            rows.append(row)
        except ValueError:
            invalid += 1
    last = rows[-1] if rows else {}
    memory = [number(r.get('peak_gb')) for r in rows]
    return dict(source=source, valid_rows=len(rows), incomplete_or_invalid_rows=invalid,
                last_step=number(last.get('step')), last_points=number(last.get('points')),
                peak_allocated_gb=max((x for x in memory if x is not None), default=None),
                last_learning_rates=last.get('learning_rates'),
                growth_events=[{'step': r.get('step'), **r['growth']} for r in rows if isinstance(r.get('growth'), dict)])


def read_case(root, scene, case, steps, queue):
    train = root / f'{scene}_{case}'
    cfg, cfg_source = read_json(train / 'config.json')
    complete, complete_source = read_json(train / 'complete.json')
    teacher, weight, dense = CASES[case]
    actual = {key: cfg.get(key) for key in ('teacher', 'weight', 'dense', 'seed', 'steps', 'prior_cameras')}
    config_errors = [f'{k}: expected {v!r}, got {cfg[k]!r}'
                     for k, v in {'teacher': teacher, 'weight': weight, 'dense': dense}.items()
                     if k in cfg and cfg[k] != v]
    fit = {str(step): read_fit(train / f'fit_{step}.json', step) for step in steps}
    evaluation = read_evaluation(root / f'{scene}_{case}_evaluation' / 'metrics.json')
    sampling = read_sampling(root / f'{scene}_{case}_sampling' / 'sampling.json')
    log = log_snapshot(train / 'training.jsonl')
    cost = dict(initial_points=number(cfg.get('initial_points')), final_points=number(complete.get('points')),
                train_seconds=number(complete.get('train_s')), diagnostic_seconds=number(complete.get('diagnostic_s')),
                wall_seconds=number(complete.get('wall_s')), peak_logged_allocated_gb=log['peak_allocated_gb'],
                evaluation_seconds=evaluation['elapsed_seconds'], sampling_seconds=sampling['elapsed_seconds'])
    available_times = [cost[k] for k in ('wall_seconds', 'evaluation_seconds', 'sampling_seconds')]
    cost['total_branch_and_assessment_seconds'] = sum(available_times) if all(x is not None for x in available_times) else None
    stage_statuses = [x['status'] for x in fit.values()] + [evaluation['status'], sampling['status']]
    status = 'pending'
    if complete_source['status'] == 'available' and all(s == 'available' for s in stage_statuses):
        status = 'complete'
    elif cfg_source['status'] == 'available' or any(s == 'available' for s in stage_statuses):
        status = 'running_or_partial'
    current = str(queue.get('current') or '')
    if queue.get('status') == 'failed' and current in (f'{scene}_{case}', f'{scene}_{case}_eval', f'{scene}_{case}_sampling'):
        status = 'failed'
    if config_errors:
        status = 'configuration_mismatch'
    checks = []
    for label, data in [('evaluation', evaluation), ('sampling', sampling)]:
        if cfg.get('manifest_sha256') and data.get('manifest_sha256') and cfg['manifest_sha256'] != data['manifest_sha256']:
            checks.append(f'{label} manifest differs from training config')
    if evaluation.get('checkpoint_sha256') and sampling.get('checkpoint_sha256') and evaluation['checkpoint_sha256'] != sampling['checkpoint_sha256']:
        checks.append('Evaluation and sampling use different checkpoints')
    for step, result in fit.items():
        if result['recorded_step'] is not None and str(result['recorded_step']) != step:
            checks.append(f'fit_{step} recorded_step mismatch')
        if result['observations'] and len({tuple(x) for x in result['observations']}) != len(result['observations']):
            checks.append(f'fit_{step} duplicate observations')
    for label, art in [('evaluation', evaluation), ('sampling', sampling), *[(f'fit_{k}', f) for k, f in fit.items()]]:
        checks += [f'{label}/{key} aggregate mismatch' for key, audit in art['aggregation_audit'].items()
                   if audit['status'] == 'mismatch']
    if checks:
        status = 'data_inconsistency'
    return dict(case=case, status=status,
                information_condition='privileged_training_HR_diagnostic' if teacher == 'hr' or cfg.get('teacher') == 'hr' else 'training_LR_and_optional_frozen_SR',
                expected_configuration=dict(teacher=teacher, weight=weight, dense=dense), configuration=actual,
                configuration_errors=config_errors, consistency_errors=checks,
                parent_sha256=cfg.get('parent_sha256'), manifest_sha256=cfg.get('manifest_sha256'),
                optimizer_preserved=cfg.get('optimizer_preserved'), device=cfg.get('device'),
                scheduler_offset=cfg.get('scheduler_offset'), scheduler_max_steps=cfg.get('scheduler_max_steps'),
                fit=fit, evaluation=evaluation, sampling=sampling, cost=cost, training_log=log,
                sources=dict(config=cfg_source, complete=complete_source))


def comparison(a, b, checks):
    """Do not subtract scores if an observed protocol mismatch is present."""
    mismatch = [name for name, x, y in checks if x is not None and y is not None and x != y]
    unknown = [name for name, x, y in checks if x is None or y is None]
    if mismatch:
        status = 'incomparable'
    elif unknown:
        status = 'pending_protocol'
    else:
        status = 'available' if all(v is not None for v in [*a.values(), *b.values()]) else 'partial_or_pending'
    delta = {key: a[key] - b[key] if not mismatch and not unknown and a[key] is not None and b.get(key) is not None else None for key in a}
    return dict(status=status, mismatched_fields=mismatch, pending_protocol_fields=unknown, delta=delta)


def compare_cases(a, b):
    common = [(key, a.get(key), b.get(key)) for key in ('parent_sha256', 'manifest_sha256')]
    common += [('seed', a['configuration']['seed'], b['configuration']['seed'])]
    common += [('case_integrity', a['configuration_errors'] + a['consistency_errors'], []),
               ('reference_integrity', b['configuration_errors'] + b['consistency_errors'], [])]
    out = {'interpretation': 'case minus reference; HR-oracle differences are diagnostics, not pure-LR method gains', 'fit': {}}
    for step, af in a['fit'].items():
        bf = b['fit'][step]
        out['fit'][step] = comparison(af['metrics'], bf['metrics'], common + [
            ('fit_observations', af['observations'], bf['observations']),
            ('fit_step', af['recorded_step'], bf['recorded_step'])])
    for name, extra in [('evaluation', ('manifest_sha256', 'frame_indices', 'test_camera', 'cache_key', 'version')),
                        ('sampling', ('manifest_sha256', 'frame_indices', 'test_camera', 'operator_order', 'version'))]:
        av, bv = a[name], b[name]
        out[name] = comparison(av['metrics'], bv['metrics'], common +
                               [(key, av.get(key), bv.get(key)) for key in extra])
    out['cost'] = comparison(a['cost'], b['cost'], common)
    return out


def build_summary(root, scenes, steps):
    summary = dict(version='dynamic_sr_control_summary_v1', created_at_utc=datetime.now(timezone.utc).isoformat(),
                   output_root=str(root), milestones=list(steps), cases=list(CASES), scenes={},
                   scope='One-shot CPU snapshot. No scientific cause/convergence conclusions; null means pending/unavailable, not zero.',
                   metric_notes={
                       'quality': 'PSNR/SSIM higher better; LPIPS lower better. Each uses the same reference within a comparison.',
                       'roi': 'Fixed temporal-variation ROI, not semantic motion ground truth. Regional LPIPS is spatial-map masked mean.',
                       'temporal': 'GT-relative warped L1 is the primary temporal residual. Prediction and GT warp L1 are supplementary, not temporal PSNR/SSIM/LPIPS.',
                       'fit': 'Training-view render vs HR and vs fixed SR; render_sr keeps SR as reference even for HR-oracle training.',
                       'sampling': '1x/2x/4x raster grids reduced to real LR, not HR reconstruction scores.',
                       'cost': 'Continuation branch only; common parent training and prior generation excluded. Wall includes fit diagnostics/checkpoint I/O.',
                       'statistics': 'Means over correlated frames are descriptive, not independent statistical repetitions.'})
    for scene in scenes:
        queue, source = read_json(root / f'{scene}_queue.json')
        records = {case: read_case(root, scene, case, steps, queue) for case in CASES}
        for case, record in records.items():
            record['comparisons'] = {ref: compare_cases(record, records[ref]) for ref in ('lr_long', 'sr_w10')}
        summary['scenes'][scene] = {'queue_source': source, 'queue_status': queue.get('status', 'pending'),
                                    'queue_current': queue.get('current'), 'queue_error': queue.get('error'), 'controls': records}
    return summary


def fmt(value, digits=4, signed=False):
    return 'pending' if number(value) is None else f'{value:+.{digits}f}' if signed else f'{value:.{digits}f}'


def triplet(values, prefix, signed=False):
    keys = [prefix + x for x in ('psnr', 'ssim', 'lpips_alex')]
    return ' / '.join(fmt(values.get(k), d, signed) for k, d in zip(keys, (3, 4, 4)))


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                     ['| ' + ' | '.join(str(v).replace('|', '\\|').replace('\n', ' ') for v in row) + ' |' for row in rows])


def markdown(s):
    lines = ['# 动态 SR 原因控制实验：指标快照', '', f"生成于 {s['created_at_utc']}。只读取一次，不触发训练或轮询。", '',
             '`pending` 表示文件/字段尚未提供；JSON 保存为 null，不补零。本文只汇总数据，不自动给出因果或收敛结论。', '',
             '质量同时列 PSNR↑、SSIM↑、LPIPS↓。变化区 LPIPS 是固定 ROI 内 spatial-map 均值；ROI 是时间变化区，非真实运动掩码。时间评价已有三种 L1：GT 相对残差、预测自身 warp 残差、GT warp 残差，不能命名为时序 PSNR/SSIM/LPIPS。', '',
             '**HR oracle 用真实训练 HR 更新参数，仅作额外信息诊断。** 训练拟合表的 render→SR 在 oracle 分支也始终以冻结 SwinIR 为参照，不能解释为其训练目标损失。', '']
    for scene, result in s['scenes'].items():
        records = result['controls']
        lines += [f'## {scene}', '', f"队列状态：{result['queue_status']}；当前项：{result['queue_current'] or 'pending/none'}。", '']
        if result['queue_error']:
            lines += [f"队列异常：`{result['queue_error']}`", '']
        lines += ['### 最终新视角质量', '']
        for privileged in (False, True):
            rows = []
            for case, r in records.items():
                if ('privileged' in r['information_condition']) != privileged:
                    continue
                m = r['evaluation']['metrics']
                rows.append([case, r['status'], *(fmt(m[k], 3 if k.endswith('psnr') else 4) for k in
                    ('full_psnr', 'full_ssim', 'full_lpips', 'variation_psnr', 'variation_ssim', 'variation_lpips_spatial'))])
            lines += [('额外训练 HR 诊断：' if privileged else '纯训练 LR／冻结 SR 条件：'), '',
                      table(['控制', '整体状态', '全图 PSNR', 'SSIM', 'LPIPS', '变化区 PSNR', 'SSIM', 'LPIPS-map'], rows), '']
        lines += ['### 最终时序诊断', '', table(['控制', '全图 GT 相对 L1', '变化区 GT 相对 L1',
            '全图预测 warp L1', '变化区预测 warp L1', '全图 GT warp L1', '变化区 GT warp L1', '变化区有效覆盖'],
            [[case] + [fmt(r['evaluation']['metrics'][k], 6) for k in
             ('full_temporal_gt_relative_l1', 'variation_temporal_gt_relative_l1',
              'full_temporal_prediction_warp_l1', 'variation_temporal_prediction_warp_l1',
              'full_temporal_gt_warp_l1', 'variation_temporal_gt_warp_l1', 'variation_flow_valid_fraction')]
             for case, r in records.items()]), '',
            '### 固定训练观察的拟合轨迹', '', '三元组为 PSNR / SSIM / LPIPS；HR 与 SR 是不同参照，不能跨列相减并称为真实性增益。', '']
        lines += [table(['控制', '追加步数', '状态', '样本数', 'render→HR', 'render→SR', 'D(render)→LR PSNR'],
             [[case, step, f['status'], f['observation_count'] if f['observation_count'] is not None else 'pending',
               triplet(f['metrics'], 'render_hr_'), triplet(f['metrics'], 'render_sr_'), fmt(f['metrics']['lr_psnr'], 3)]
              for case, r in records.items() for step, f in r['fit'].items()]), '',
            '### 跨渲染网格采样与成本', '',
            '采样三列都缩回真实 LR 评分，不能视为对应倍率 HR 质量。成本只含本次追加训练；公共父模型和 SR 先验生成成本未计入。不同点数分支的同迭代数不是同算力。', '',
            table(['控制', 'LR PSNR：1× / 2× / 4× 网格', '初始→最终点数', '训练秒', '诊断秒', '训练目录总秒', '峰值分配 GB', '评测＋采样秒'],
             [[case, ' / '.join(fmt(r['sampling']['metrics'][f'render_{n}x_psnr_to_real_lr'], 3) for n in (1, 2, 4)),
               f"{fmt(r['cost']['initial_points'], 0)} → {fmt(r['cost']['final_points'], 0)}",
               fmt(r['cost']['train_seconds'], 1), fmt(r['cost']['diagnostic_seconds'], 1), fmt(r['cost']['wall_seconds'], 1),
               fmt(r['cost']['peak_logged_allocated_gb'], 3),
               ' / '.join(fmt(r['cost'][k], 1) for k in ('evaluation_seconds', 'sampling_seconds'))]
              for case, r in records.items()]), '']
        for reference in ('lr_long', 'sr_w10'):
            lines += [f'### 差值：控制 − {reference}', '',
                '只在父模型、数据与评价协议可核对时计算。负 LPIPS/残差表示数值下降；oracle 差值仍仅作诊断。', '',
                table(['控制', '协议状态', 'Δ 全图 PSNR / SSIM / LPIPS', 'Δ 变化区 PSNR / SSIM / LPIPS-map',
                       'Δ 变化区时间 L1', 'Δ 4×→LR PSNR', 'Δ 训练秒', 'Δ 最终点数'],
                 [[case, r['comparisons'][reference]['evaluation']['status'],
                   ' / '.join(fmt(r['comparisons'][reference]['evaluation']['delta'][k], 4, True) for k in ('full_psnr', 'full_ssim', 'full_lpips')),
                   ' / '.join(fmt(r['comparisons'][reference]['evaluation']['delta'][k], 4, True) for k in ('variation_psnr', 'variation_ssim', 'variation_lpips_spatial')),
                   fmt(r['comparisons'][reference]['evaluation']['delta']['variation_temporal_gt_relative_l1'], 6, True),
                   fmt(r['comparisons'][reference]['sampling']['delta']['render_4x_psnr_to_real_lr'], 4, True),
                   fmt(r['comparisons'][reference]['cost']['delta']['train_seconds'], 1, True),
                   fmt(r['comparisons'][reference]['cost']['delta']['final_points'], 0, True)] for case, r in records.items()]), '',
                table(['控制', '追加步数', '拟合协议状态', 'Δ render→HR：PSNR / SSIM / LPIPS', 'Δ render→SR：PSNR / SSIM / LPIPS'],
                 [[case, step, f['status'], triplet(f['delta'], 'render_hr_', True), triplet(f['delta'], 'render_sr_', True)]
                  for case, r in records.items() for step, f in r['comparisons'][reference]['fit'].items()]), '']
        issues = []
        for case, r in records.items():
            issues += [f'{case}: {x}' for x in r['configuration_errors'] + r['consistency_errors']]
            for label, art in [('evaluation', r['evaluation']), ('sampling', r['sampling']), *[(f'fit_{k}', f) for k, f in r['fit'].items()]]:
                if art['source']['status'] == 'unreadable':
                    issues.append(f'{case}/{label}: unreadable JSON; see machine summary')
                issues += [f'{case}/{label}: {k} aggregate mismatch' for k, v in art['aggregation_audit'].items() if v['status'] == 'mismatch']
        lines += ['一致性审计：' + ('；'.join(issues) if issues else '当前可读记录未检出聚合数值/身份冲突；pending 不表示已通过。'), '']
    lines += ['## 来源和解释范围', '',
              '每项 JSON 路径与读取字节的 SHA256、拟合观察列表、评价帧/缓存标识、聚合复算、全部差值、训练日志状态见同目录 `summary.json`。不把连续视频帧当作独立重复实验，不跨不同数据域自动合成一个均分。', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT / 'output/dynamic_sr_20260919')
    parser.add_argument('--out', type=Path, help='Default: ROOT/assessment_controls_v1')
    parser.add_argument('--scenes', nargs='+', default=list(SCENES))
    parser.add_argument('--milestones', default=','.join(map(str, STEPS)))
    args = parser.parse_args()
    root = args.root.resolve()
    out = (args.out or root / 'assessment_controls_v1').resolve()
    steps = tuple(int(x) for x in args.milestones.split(','))
    if not steps or len(set(steps)) != len(steps) or any(x < 0 for x in steps):
        parser.error('milestones must be distinct nonnegative integers')
    summary = build_summary(root, args.scenes, steps)
    summary['summarizer_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    out.mkdir(parents=True, exist_ok=True)
    for name, body in [('summary.json', json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + '\n'),
                       ('summary.md', markdown(summary))]:
        path = out / name
        temp = path.with_suffix(path.suffix + '.tmp')
        temp.write_text(body)
        temp.replace(path)
    print(json.dumps({'summary_json': str(out / 'summary.json'), 'summary_markdown': str(out / 'summary.md'),
                      'control_statuses': {scene: {k: v['status'] for k, v in r['controls'].items()} for scene, r in summary['scenes'].items()}}, ensure_ascii=False))


if __name__ == '__main__':
    main()
