#!/usr/bin/env python3
"""Summarize a completed joint/ordinary_split/bound_split batch; CPU/files only.

No endpoint selection, parameter updates, model loading, or automatic efficacy
decision. JSON is authoritative; PNG panels are explicitly quantized previews.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil


PROJECT = Path(__file__).resolve().parents[2]
BRANCHES = ('joint', 'ordinary_split', 'bound_split')
ENDPOINTS = (('dev', 1200), ('dev', 6000), ('test', 6000))
REGIONS = ('full', 'dynamic', 'static')
LABELS = {'joint': 'A Joint', 'ordinary_split': 'B Ordinary', 'bound_split': 'C Bound'}
CAMERAS = {'dev': 'cam01', 'test': 'cam00'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def number(value, label, nullable=False):
    if value is None and nullable:
        return None
    require(isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value), f'Missing/nonfinite {label}: {value}')
    return float(value)


def local_manifest(recorded, explicit):
    """Resolve a copied remote record without editing its original source path."""
    candidates = [explicit] if explicit is not None else [Path(recorded)]
    if explicit is None:
        parts = Path(recorded).parts
        for index in range(len(parts) - 1):
            if parts[index:index + 2] == ('data', 'dynamic_sr'):
                candidates.append(PROJECT.joinpath(*parts[index:]))
    for path in candidates:
        if path is not None and Path(path).is_file():
            return Path(path).resolve()
    raise FileNotFoundError('Local prepared manifest not found; provide --manifest while preserving its SHA256')


def make_quality_plot(rows, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = ['#536878', '#dc8b26', '#237d73']
    figure, axes = plt.subplots(2, 3, figsize=(10.5, 5.6), layout='constrained')
    for row_index, region in enumerate(('full', 'dynamic')):
        for column, key in enumerate(('psnr', 'ssim', 'lpips')):
            axis = axes[row_index, column]
            values = [next(r for r in rows if r['branch'] == branch and r['split'] == 'test'
                           and r['step'] == 6000 and r['region'] == region)[key] for branch in BRANCHES]
            axis.scatter(range(3), values, color=colors, s=45, zorder=3)
            axis.set_xticks(range(3), [LABELS[b] for b in BRANCHES])
            suffix = 'lower' if key == 'lpips' else 'higher'
            name = 'LPIPS spatial ROI' if key == 'lpips' and region != 'full' else key.upper()
            axis.set_title(f'{region}: {name} ({suffix})')
            axis.grid(axis='y', alpha=.25)
            axis.ticklabel_format(axis='y', style='plain', useOffset=False)
    figure.suptitle('MeetRoom vrheadset confirmation | fixed test cam00 | 6000 added steps\nLocal axis ranges; descriptive single-run comparison')
    figure.savefig(out / 'quality.png', dpi=160)
    plt.close(figure)


def make_auxiliary_plot(rows, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(2, 3, figsize=(11, 5.6), layout='constrained')
    for row_index, region in enumerate(('full', 'dynamic')):
        for column, (key, title, direction) in enumerate((('temporal', 'GT-relative warp L1', 'lower'),
                                                        ('lr_psnr', 'LR reprojection PSNR', 'higher'),
                                                        ('lr_l1', 'LR reprojection L1', 'lower'))):
            axis = axes[row_index, column]
            values = [next(r for r in rows if r['branch'] == b and r['split'] == 'test'
                           and r['step'] == 6000 and r['region'] == region)[key] for b in BRANCHES]
            axis.scatter(range(3), values, color=['#536878', '#dc8b26', '#237d73'], s=45, zorder=3)
            axis.set_xticks(range(3), [LABELS[b] for b in BRANCHES])
            axis.set_title(f'{region}: {title} ({direction})')
            axis.ticklabel_format(axis='y', style='plain', useOffset=False)
            axis.grid(axis='y', alpha=.25)
    figure.suptitle('Fixed test cam00 / 6000 steps | same unquantized renders for all methods')
    figure.savefig(out / 'temporal_lr_reprojection.png', dpi=160)
    plt.close(figure)


def make_roi(batch, manifest_path, manifest, read_identity, out):
    from PIL import Image, ImageDraw
    matches = [o for o in manifest['observations']
               if o['split'] == 'test' and o['camera_id'] == 'cam00' and o['frame_index'] == 40]
    require(len(matches) == 1, 'Registered test cam00/frame40 must exist exactly once')
    observation = matches[0]
    paths = {'HR': manifest_path.parent / observation['hr_path']}
    for branch in BRANCHES:
        paths[LABELS[branch]] = batch / branch / 'eval_test_6000/predictions/cam00/0040.png'
    width, height = manifest['resolutions']['hr']
    require((width, height) == (1280, 720), 'Historical MeetRoom ROI protocol expects 1280x720')
    images, identities = {}, []
    for label, path in paths.items():
        digest = read_identity(path)
        if label == 'HR':
            require(digest == observation['hr_sha256'], 'Fixed HR image identity changed')
        with Image.open(path) as image:
            require(image.size == (width, height), f'Image size mismatch: {path}')
            images[label] = image.convert('RGB')
        identities.append({'label': label, 'path': str(path), 'sha256': digest})
    # Inherit the pixel coordinates only; their discussion semantics do not
    # transfer to this different sequence. Keep keys for artifact compatibility.
    boxes = {'time_changing': (504, 420, 672, 588), 'mostly_static': (1092, 504, 1260, 672)}
    box_labels = {'time_changing': 'fixed pixel box 1', 'mostly_static': 'fixed pixel box 2'}
    files = {}
    for name, box in boxes.items():
        tile_width, tile_height = box[2] - box[0], box[3] - box[1]
        panel = Image.new('RGB', (4 * tile_width, tile_height + 48), 'white')
        draw = ImageDraw.Draw(panel)
        for index, (label, image) in enumerate(images.items()):
            panel.paste(image.crop(box), (index * tile_width, 48))
            draw.text((index * tile_width + 5, 5), label, fill='black')
        draw.text((5, 27), f'cam00 frame40 | {box_labels[name]} | inherited coordinates; no semantic match', fill='black')
        name_on_disk = f'{name}_test_cam00_frame40.png'
        panel.save(out / name_on_disk)
        files[name] = name_on_disk
    # Entire original scene stays visible. Resize only this display panel;
    # saved prediction files and metric inputs remain untouched.
    overview = Image.new('RGB', (1280, 780), 'white')
    draw = ImageDraw.Draw(overview)
    for index, (label, image) in enumerate(images.items()):
        x, y = (index % 2) * 640, (index // 2) * 390
        draw.text((x + 6, y + 5), label, fill='black')
        overview.paste(image.resize((640, 360), Image.Resampling.LANCZOS), (x, y + 30))
    files['full_scene'] = 'full_scene_test_cam00_frame40.png'
    overview.save(out / files['full_scene'])
    return {'camera_id': 'cam00', 'frame_index': 40,
            'regions_xyxy_exclusive': {key: list(value) for key, value in boxes.items()},
            'selection': 'Pixel coordinates inherited from the discussion 2026-09-20 example and applied unchanged to vrheadset, not selected using confirmation differences; no physical or semantic correspondence between sequences; illustrative only',
            'source_report': str(PROJECT / 'docs/dynamic_sr_final_detail_audit_2026-09-20.md'),
            'images': identities, 'files': files,
            'precision': 'Saved uint8 PNGs; ROI panels native resolution, full-scene overview display resized only. Neither replaces float-render metrics.'}


def markdown(summary):
    lines = ['# MeetRoom vrheadset 确认序列：运动绑定细分固定端点汇总', '',
             'A 是本次 vrheadset 确认流程从本序列共同 LR 父模型新训的 Joint；批次阶段复用本轮 A 检查点并统一评价。B 是普通空间细分，C 是共享父运动的绑定细分。保留完整人物与复杂背景，不作人体抠图；所有主指标使用本轮统一脚本的浮点渲染。', '',
             '整图是主指标；时间变化区来自固定 HR 评价缓存，仅用于评价，不是人物语义或训练掩码。PSNR/SSIM 越高越好，LPIPS/参考相对时序 L1 越低越好。区域 LPIPS 是空间图区域均值。', '',
             '| 划分/追加步数 | 分支 | 区域 | PSNR | SSIM | LPIPS | 时序 L1 | LR 回投 PSNR | LR 回投 L1 |',
             '| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    names = {'full': '整图', 'dynamic': '时间变化区', 'static': '其余区域'}
    for row in summary['metrics']:
        values = ['未提供' if row[k] is None else f"{row[k]:.8f}" for k in
                  ('psnr', 'ssim', 'lpips', 'temporal', 'lr_psnr', 'lr_l1')]
        lines.append(f"| {row['split']}/{row['step']} | {LABELS[row['branch']]} | {names[row['region']]} | "
                     + ' | '.join(values) + ' |')
    lines += ['', '固定 test/6000 的 C 差值如下；ΔLPIPS、Δ时序 L1 和 ΔLR L1 为正表示变差。其余固定端点差值及逐帧胜数保存在 JSON。', '',
              '| 比较 | 区域 | ΔPSNR | ΔSSIM | ΔLPIPS | Δ时序 L1 | ΔLR PSNR | ΔLR L1 |',
              '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in summary['deltas']:
        if row['split'] != 'test':
            continue
        values = ['未提供' if row[k] is None else f"{row[k]:+.8f}" for k in
                  ('psnr', 'ssim', 'lpips', 'temporal', 'lr_psnr', 'lr_l1')]
        lines.append(f"| {row['comparison']} | {names[row['region']]} | " + ' | '.join(values) + ' |')
    lines += ['', '真实 LR 回投使用同一份未量化渲染：先 bicubic 抗混叠降采样（align_corners=False），再截断到 [0,1]，最后与实际 8-bit LR PNG 比较。预测没有先保存成 PNG；留出 LR 在渲染之后才读取，不是模型输入。回投误差小只说明观测一致性，不证明高频真实。', '',
              '| 分支 | 活跃高斯 | 存储高斯 | 父控制量 | 可训练标量总数 | 训练秒数 | 峰值 allocated GB | test 渲染秒数 |',
              '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for cost in summary['costs']:
        lines.append(f"| {LABELS[cost['branch']]} | {cost['active_gaussians']} | {cost['stored_gaussians']} | "
                     f"{cost['parent_control_count']} | {cost['trainable_parameters']} | "
                     f"{cost['train_seconds']:.2f} | {cost['peak_allocated_gb']:.3f} | {cost['test_6000_render_seconds']:.3f} |")
    out = Path(summary['output'])
    lines += ['', '预算检查只约束参与渲染的活跃高斯增加不超过 20%。存储高斯、父控制量、可训练标量和固定缓冲另列，不能因此声称总容量或训练预算相同。A 的训练成本来自本次确认流程新训 A 的完整记录；批次阶段复用其检查点，不重复训练。', '',
              '渲染秒数为 CUDA 同步的 render_model 耗时合计，包含首帧，不含相机构造、拷贝、指标或 I/O；不是端到端 FPS。总评价时间见 JSON。', '',
              f'![固定测试质量]({out / "quality.png"})', '',
              f'![时序与 LR 回投]({out / "temporal_lr_reprojection.png"})', '',
              f'![完整场景]({out / summary["roi"]["files"]["full_scene"]})', '',
              f'![固定像素框 1：沿用 discussion 坐标]({out / summary["roi"]["files"]["time_changing"]})', '',
              f'![固定像素框 2：沿用 discussion 坐标]({out / summary["roi"]["files"]["mostly_static"]})', '',
              '图像均为固定 test cam00/frame40。局部像素坐标直接沿用 discussion 先前展示坐标，未按 vrheadset 方法差异选取；不代表同一物理位置，也不保证该框仍属于时间变化或静态对象，仅作固定展示。局部是原生分辨率 8-bit PNG 裁剪，全图仅作显示缩小；两者不能替代浮点主指标。', '',
              '一个新的同数据集确认序列、一个采样种子、60 帧短窗；不等于跨数据域验证。连续帧不独立，逐帧胜数和帧标准差均不构成显著性检验。dev/1200、dev/6000、test/6000 全部保留，不挑最佳端点。本汇总不自动宣布方法有效。', '',
              '来源身份、成本、完整容量字典和差值见 [summary.json](summary.json)。', '']
    return '\n'.join(lines)


def main(args):
    root = args.root.resolve()
    out = (args.out or root / 'summary').resolve()
    require(not out.exists(), f'Refusing to overwrite summary: {out}')
    identities = {}

    def identify(path):
        path = Path(path).resolve()
        digest = sha(path)
        identities[str(path)] = digest
        return digest

    def read(path):
        path = Path(path).resolve()
        digest = identify(path)
        value = json.loads(path.read_text())
        require(sha(path) == digest, f'File changed while reading: {path}')
        return value

    # Complete every status gate before reading any metric or training score.
    batch_complete = read(root / 'complete.json')
    require(batch_complete.get('status') == 'completed_and_evaluated',
            'Batch is not completed_and_evaluated; no metrics read')
    train_done, eval_done = {}, {}
    for branch in BRANCHES:
        done = read(root / branch / 'complete.json')
        require(done.get('status') == 'completed' and done.get('parameter_updates') == 6000
                and not done.get('smoke', False), f'{branch}: formal 6000-step training incomplete')
        train_done[branch] = done
        for split, step in ENDPOINTS:
            receipt = read(root / branch / f'eval_{split}_{step}' / 'complete.json')
            require(receipt.get('status') == 'completed_evaluation' and receipt['split'] == split
                    and receipt['parameter_updates'] == 0, f'{branch}/{split}/{step}: evaluation incomplete')
            eval_done[branch, split, step] = receipt

    configs = {b: read(root / b / 'config.json') for b in BRANCHES}
    require(configs['ordinary_split']['selection_sha256'] == configs['bound_split']['selection_sha256'],
            'B/C must use the identical saved split selection and initialization inputs')
    manifest_path = local_manifest(configs['joint']['manifest'], args.manifest)
    manifest = read(manifest_path)
    require(sha(manifest_path) == configs['joint']['manifest_sha256'], 'Manifest identity differs from training')
    reference, metrics, costs, evaluations, all_values = {}, [], [], [], {}
    for branch in BRANCHES:
        config, done = configs[branch], train_done[branch]
        for field in ('manifest_sha256', 'parent_sha256', 'seed', 'steps', 'sr_weight', 'prior_cameras', 'prior_subdir'):
            require(config[field] == configs['joint'][field], f'Unmatched control config: {branch}/{field}')
        for field in ('teacher_count', 'teacher_cameras', 'scheduler_offset', 'initial_points'):
            require(config[field] == configs['joint'][field], f'Unmatched input/schedule: {branch}/{field}')
        teacher_identity = lambda c: [(r['camera'], r['frame'], r['sha256']) for r in c['teacher_inputs']]
        require(teacher_identity(config) == teacher_identity(configs['joint']), f'{branch}: teacher cache identity differs')
        require(config['branch'] == branch and config['steps'] == 6000, f'{branch}: wrong branch/step count')
        require(done['draw_sha256'] == train_done['joint']['draw_sha256'], f'{branch}: sampling draw hash differs')
        if (root / branch / 'exposure.json').is_file():
            exposure = read(root / branch / 'exposure.json')
            require(exposure == read(root / 'joint/exposure.json'), f'{branch}: exposure counts differ')
        branch_capacity = None
        for split, step in ENDPOINTS:
            folder = root / branch / f'eval_{split}_{step}'
            values = read(folder / 'metrics.json')
            receipt = eval_done[branch, split, step]
            require(sha(folder / 'metrics.json') == receipt['metrics_sha256'], 'Metric/receipt hash mismatch')
            require(values.get('version') == 'motion_bound_eval_v1', 'All branches require this new unified evaluator')
            require(values['checkpoint_sha256'] == receipt['checkpoint_sha256'] and values['branch'] == branch,
                    f'{branch}/{split}/{step}: checkpoint/branch mismatch')
            checkpoint = Path(values['checkpoint'])
            require(identify(checkpoint) == values['checkpoint_sha256'], f'Checkpoint changed: {checkpoint}')
            require(values['split'] == split and values['checkpoint_metadata']['intervention_step'] == step,
                    f'{branch}/{split}/{step}: endpoint mismatch')
            require(values['evaluation_cameras'] == [CAMERAS[split]], f'Unexpected {split} camera')
            require(values['manifest_sha256'] == config['manifest_sha256'], 'Evaluation/training manifest differs')
            require(len(values['rows']) == receipt['observations'] == 60, 'Incomplete fixed sequence')
            require([(r['camera_id'], r['frame_index']) for r in values['rows']]
                    == [(CAMERAS[split], f) for f in range(0, 120, 2)], 'Wrong fixed evaluation sequence')
            require(values['lpips_status'] == 'alex_v0.1_standard_full_and_fixed_mask_spatial_map', 'LPIPS skipped')
            require(values['lr_reprojection_protocol']['source_precision'] == 'float32_raw_render_before_png_quantization',
                    'No 8-bit prediction reprojection may be mixed into the float-render main table')
            identity = {key: values[key] for key in ('script_sha256', 'metric_helpers_sha256', 'motion_model_sha256',
                        'metric_definitions', 'manifest_sha256', 'observation_keys', 'evaluation_caches', 'versions',
                        'lr_reprojection_protocol', 'gpu', 'lpips_device', 'dynamic_threshold', 'flow_scale')}
            identity['observed_lr_sha256'] = [r['observed_lr_sha256'] for r in values['rows']]
            if split in reference:
                require(identity == reference[split], f'{branch}/{split}/{step}: evaluation identities differ')
            else:
                reference[split] = identity
            capacity = values['capacity']
            for field in ('active_gaussians', 'stored_gaussians', 'parent_control_count', 'trainable_parameters'):
                require(isinstance(capacity.get(field), int) and capacity[field] >= 0, f'Missing capacity {field}')
            if branch_capacity is not None:
                require(capacity == branch_capacity, f'{branch}: registered capacity changed across endpoints')
            branch_capacity = capacity
            for region in REGIONS:
                spatial, temporal = values['aggregate'][region], values['temporal_aggregate'][region]
                lr = values['lr_reprojection_aggregate'][region]
                lpips_key = 'lpips_alex_mean' if region == 'full' else 'lpips_alex_spatial_mask_mean'
                metrics.append({'branch': branch, 'split': split, 'step': step, 'region': region,
                                'psnr': number(spatial['psnr_mean'], 'PSNR'), 'ssim': number(spatial['ssim_mean'], 'SSIM'),
                                'lpips': number(spatial[lpips_key], 'LPIPS'),
                                'temporal': number(temporal.get('gt_relative_warp_l1_mean'), 'temporal', True),
                                'temporal_valid_fraction': number(temporal.get('valid_fraction_of_roi_mean'), 'flow coverage', True),
                                'lr_psnr': number(lr['psnr_mean'], 'LR PSNR'), 'lr_l1': number(lr['l1_mean'], 'LR L1'),
                                'lr_mse': number(lr['mse_mean'], 'LR MSE'),
                                'lr_psnr_from_mean_mse': number(lr['psnr_from_mean_mse'], 'LR pooled PSNR')})
            evaluations.append({'branch': branch, 'split': split, 'step': step, 'path': str(folder / 'metrics.json'),
                                'sha256': receipt['metrics_sha256'], 'checkpoint_sha256': receipt['checkpoint_sha256'],
                                'elapsed_seconds': values['elapsed_seconds']})
            all_values[branch, split, step] = values
        final_test = all_values[branch, 'test', 6000]
        costs.append({'branch': branch, **branch_capacity, 'capacity': branch_capacity,
                      'train_seconds': number(done['train_s'], 'train seconds'),
                      'wall_seconds': number(done['wall_s'], 'wall seconds'),
                      'peak_allocated_gb': number(done['peak_gb'], 'peak allocated GB'),
                      'gpu': config.get('gpu'), 'visible_cuda': config.get('visible_cuda'),
                      'confirmation_training_reused_in_batch': branch == 'joint',
                      'test_6000_evaluation_seconds': number(final_test['elapsed_seconds'], 'evaluation seconds'),
                      'test_6000_render_seconds': number(final_test['render_seconds'], 'render seconds')})
    count = {c['branch']: c['active_gaussians'] for c in costs}
    require(count['joint'] > 0, 'Empty parent model')
    require(count['ordinary_split'] == count['bound_split'], 'B/C active-render budgets differ')
    require(count['joint'] <= count['bound_split'] <= math.floor(1.2 * count['joint']),
            'Split active-render budget exceeds +20%')
    budget = {'status': 'passed', 'parent_active': count['joint'], 'split_active': count['bound_split'],
              'active_increase_fraction': count['bound_split'] / count['joint'] - 1,
              'maximum_increase_fraction': .2,
              'scope': 'Active rendered primitives only; storage, parent controls, trainable scalars and fixed buffers separately reported; not equal total capacity or compute'}
    metric_index = {(r['branch'], r['split'], r['step'], r['region']): r for r in metrics}
    deltas, wins = [], []
    for split, step in ENDPOINTS:
        for region in REGIONS:
            proposed = metric_index['bound_split', split, step, region]
            for baseline in ('ordinary_split', 'joint'):
                comparison = metric_index[baseline, split, step, region]
                label = f'C-{LABELS[baseline]}'
                deltas.append({'comparison': label, 'split': split, 'step': step, 'region': region,
                               **{key: None if proposed[key] is None or comparison[key] is None else proposed[key] - comparison[key]
                                  for key in ('psnr', 'ssim', 'lpips', 'temporal', 'lr_psnr', 'lr_l1', 'lr_mse')}})
                lpips_key = 'lpips_alex' if region == 'full' else 'lpips_alex_spatial_mask'
                for group, key, higher in (('spatial', 'psnr', True), ('spatial', 'ssim', True),
                                           ('spatial', lpips_key, False), ('temporal', 'gt_relative_warp_l1', False),
                                           ('lr_reprojection', 'psnr', True), ('lr_reprojection', 'l1', False)):
                    differences = []
                    for c, b in zip(all_values['bound_split', split, step]['rows'], all_values[baseline, split, step]['rows']):
                        cv, bv = c.get(group, {}).get(region, {}).get(key), b.get(group, {}).get(region, {}).get(key)
                        if cv is not None and bv is not None:
                            differences.append((cv - bv) * (1 if higher else -1))
                    wins.append({'comparison': label, 'split': split, 'step': step, 'region': region,
                                 'metric': f'{group}/{key}', 'wins': sum(d > 0 for d in differences),
                                 'ties': sum(d == 0 for d in differences), 'losses': sum(d < 0 for d in differences),
                                 'n': len(differences), 'note': 'Exact floating comparison; correlated frames/pairs, not a significance test'})
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(__file__, out / 'source.py')
    roi = make_roi(root, manifest_path, manifest, identify, out)
    make_quality_plot(metrics, out)
    make_auxiliary_plot(metrics, out)
    result = {'status': 'descriptive_summary', 'root': str(root), 'output': str(out),
              'fixed_endpoints': [list(x) for x in ENDPOINTS], 'metrics': metrics, 'deltas': deltas,
              'frame_wins': wins, 'costs': costs, 'active_budget_check': budget, 'evaluation_files': evaluations,
              'roi': roi, 'local_manifest': str(manifest_path), 'recorded_manifest': configs['joint']['manifest'],
              'protocol_identity': reference, 'input_sha256': identities, 'source_sha256': sha(__file__),
              'information_boundary': 'Completed-result CPU aggregation only. Full-scene RGB retained. GT masks/flows and held-out observed LR are evaluation-only. A is newly trained on this confirmation sequence before the batch reuses its checkpoints; all evaluations use the same float-render protocol.',
              'limitations': ['One new same-dataset confirmation sequence, one explicit sampling seed, nonstandard 60-frame short-window pilot; not cross-domain confirmation.',
                              'No endpoint selection, statistical significance claim, or automatic efficacy verdict.',
                              'LR consistency, temporal agreement and visual sharpness do not alone prove high-frequency or geometry truth.',
                              'Dynamic/static LPIPS are spatial-map extensions; full scalar LPIPS remains primary.',
                              'Active rendered count is not total stored capacity; parent controls and parameter counts are separately reported.',
                              'Render timing includes the first frame and excludes camera/metrics/I/O; not end-to-end FPS.',
                              'Fixed pixel boxes inherited from discussion use saved uint8 predictions; semantic labels do not establish object identity in vrheadset; main metrics use float renders.'],
              'finished_utc': datetime.now(timezone.utc).isoformat()}
    for path, digest in identities.items():
        require(sha(path) == digest, f'Input changed during summarization: {path}')
    write(out / 'summary.json', result)
    (out / 'summary.md').write_text(markdown(result))
    write(out / 'complete.json', {'status': 'completed_summary', 'summary_sha256': sha(out / 'summary.json'),
                                  'source_sha256': sha(__file__), 'parameter_updates': 0, 'cpu_only': True})
    print(json.dumps({'output': str(out), 'metrics': len(metrics), 'deltas': len(deltas), 'gpu_used': False}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True, help='Completed batch with joint/ordinary_split/bound_split')
    parser.add_argument('--out', type=Path, help='New directory; defaults to ROOT/summary')
    parser.add_argument('--manifest', type=Path, help='Local hash-identical copy of prepared manifest')
    main(parser.parse_args())
