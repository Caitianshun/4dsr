#!/usr/bin/env python3
"""Summarize only a fully completed surface-residual batch; CPU/files only.

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
BRANCHES = ('joint', 'global_residual', 'local_residual')
ENDPOINTS = (('dev', 1200), ('dev', 6000), ('test', 6000))
REGIONS = ('full', 'foreground', 'foreground_union')
LABELS = {'joint': 'Joint', 'global_residual': 'Global', 'local_residual': 'Local'}
CAMERAS = {'dev': 'cam07', 'test': 'cam01'}


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
    for row_index, region in enumerate(('full', 'foreground')):
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
    figure.suptitle('CoreView387 pilot | fixed test cam01 | 6000 added steps\nLocal axis ranges; descriptive single-run comparison')
    figure.savefig(out / 'quality.png', dpi=160)
    plt.close(figure)


def make_roi(batch, manifest_path, manifest, read_identity, out):
    import numpy as np
    from PIL import Image, ImageDraw
    matches = [o for o in manifest['observations']
               if o['split'] == 'test' and o['camera_id'] == 'cam01' and o['frame_index'] == 40]
    require(len(matches) == 1, 'The registered test cam01/frame40 must exist exactly once')
    observation = matches[0]
    paths = {'HR': manifest_path.parent / observation['hr_path']}
    mask_path = manifest_path.parent / observation['mask_path']
    mask_sha = read_identity(mask_path)
    if 'mask_sha256' in observation:
        require(mask_sha == observation['mask_sha256'], 'Official mask identity changed')
    with Image.open(mask_path) as image:
        mask = np.asarray(image.convert('L')) > 0
    yy, xx = np.where(mask)
    require(len(xx) > 0, 'Empty official foreground mask for the fixed ROI')
    width, height = manifest['resolutions']['hr']
    require(mask.shape == (height, width), 'Official ROI mask is not on the prepared HR grid')
    box = (max(0, int(xx.min()) - 16), max(0, int(yy.min()) - 16),
           min(width, int(xx.max()) + 17), min(height, int(yy.max()) + 17))
    for branch in BRANCHES:
        paths[LABELS[branch]] = batch / branch / 'eval_test_6000/predictions/cam01/0040.png'
    images = {}
    identities = []
    for label, path in paths.items():
        digest = read_identity(path)
        if label == 'HR' and 'hr_sha256' in observation:
            require(digest == observation['hr_sha256'], 'Fixed HR image identity changed')
        with Image.open(path) as image:
            require(image.size == (width, height), f'Panel image size mismatch: {path}')
            images[label] = image.convert('RGB').crop(box)
        identities.append({'label': label, 'path': str(path), 'sha256': digest})
    tile_width, tile_height = box[2] - box[0], box[3] - box[1]
    panel = Image.new('RGB', (4 * tile_width, tile_height + 48), 'white')
    draw = ImageDraw.Draw(panel)
    for index, (label, crop) in enumerate(images.items()):
        panel.paste(crop, (index * tile_width, 48))
        draw.text((index * tile_width + 6, 6), label, fill='black')
    draw.text((6, 27), 'test cam01 frame40 | official mask bbox + 16 px | native-resolution PNG crops', fill='black')
    panel.save(out / 'test_cam01_frame40_roi.png')
    return {'camera_id': 'cam01', 'frame_index': 40, 'xyxy_exclusive': list(box),
            'selection': 'Entire nonzero official per-frame mask bbox plus fixed 16-pixel margin, clipped to image; no error-based selection',
            'mask_path': str(mask_path), 'mask_sha256': mask_sha, 'images': identities,
            'precision': 'Saved uint8 PNG crops at native resolution, unmodified source predictions; not the float-render metric source'}


def markdown(summary):
    lines = ['# CoreView387 表面残差三分支：固定端点汇总', '',
             '本文件仅描述已完成结果，不自动判定方法有效。数据为使用官方掩码、黑底合成和共同 LR 初始化的非标准短窗先导；不使用 SMPL 模板或旧 HOI 权重。', '',
             'Joint 为原联合训练；Global/Local 各加每点 4×RGB 系数，底座仍接受 LR+SR 更新，仅残差渲染的几何/不透明度支撑停止梯度。Global 与 Local 具有相同附加参数量。', '',
             'PSNR/SSIM 越高越好，LPIPS/参考相对时序 L1 越低越好。前景 LPIPS 为完整图像空间 LPIPS 图在官方区域内的均值，不是标准整图标量 LPIPS。', '',
             '| 划分/追加步数 | 分支 | 区域 | PSNR | SSIM | LPIPS | 时序 L1 |',
             '| --- | --- | --- | ---: | ---: | ---: | ---: |']
    names = {'full': '整图', 'foreground': '逐帧前景', 'foreground_union': '前景并集'}
    for row in summary['metrics']:
        temporal = '未提供' if row['temporal'] is None else f"{row['temporal']:.8f}"
        lines.append(f"| {row['split']}/{row['step']} | {LABELS[row['branch']]} | {names[row['region']]} | "
                     f"{row['psnr']:.6f} | {row['ssim']:.6f} | {row['lpips']:.6f} | {temporal} |")
    lines += ['', '固定 test/6000 的 Local 差值如下；ΔLPIPS 和 Δ时序为正表示变差。完整 dev 差值保存在 JSON。', '',
              '| 比较 | 区域 | ΔPSNR | ΔSSIM | ΔLPIPS | Δ时序 L1 |',
              '| --- | --- | ---: | ---: | ---: | ---: |']
    for row in summary['deltas']:
        if row['split'] != 'test':
            continue
        temporal = '未提供' if row['temporal'] is None else f"{row['temporal']:+.8f}"
        lines.append(f"| {row['comparison']} | {names[row['region']]} | {row['psnr']:+.6f} | "
                     f"{row['ssim']:+.6f} | {row['lpips']:+.6f} | {temporal} |")
    lines += ['', '| 分支 | 高斯数 | 新增系数 | 训练秒数 | 峰值 allocated GB | test/6000 总评价秒数 | 独立渲染秒数 |',
              '| --- | ---: | ---: | ---: | ---: | ---: | --- |']
    for cost in summary['costs']:
        render = '未单独记录' if cost['test_6000_render_seconds'] is None else f"{cost['test_6000_render_seconds']:.3f}"
        lines.append(f"| {LABELS[cost['branch']]} | {cost['points']} | {cost['extra_detail_parameters']} | "
                     f"{cost['train_seconds']:.2f} | {cost['peak_allocated_gb']:.3f} | "
                     f"{cost['test_6000_evaluation_seconds']:.2f} | {render} |")
    lines += ['', '总评价时间包含缓存、指标和文件写入，不能当作独立渲染耗时或 FPS。相同步数不等于相同计算预算。', '',
              f"![固定测试指标]({Path(summary['output']) / 'quality.png'})", '',
              f"![固定人物区域]({Path(summary['output']) / 'test_cam01_frame40_roi.png'})", '',
              '人物框仅由 test cam01/frame40 官方 mask 的包围框加 16 像素确定。显示原始保存的 8-bit PNG 裁剪，不修改预测；图像不替代浮点主指标。', '',
              '只有一个场景、一个显式采样种子；连续帧不独立，未估计训练随机波动。dev 两个端点与固定 test/6000 全部保留，不挑最好端点，不作显著性或普遍有效性声明。前景掩码属于所有分支一致的公开额外信息，前景并集可能包含单帧背景。', '',
              '原始文件身份、各分支成本、数据来源路径和全部差值见 [summary.json](summary.json)。', '']
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

    # Complete all gates before reading any metric values or training scores.
    batch_complete = read(root / 'complete.json')
    require(batch_complete.get('status') == 'completed_and_evaluated', 'Batch is not completed_and_evaluated; no metrics read')
    train_done, eval_done = {}, {}
    for branch in BRANCHES:
        done = read(root / branch / 'complete.json')
        require(done.get('status') == 'completed' and done.get('parameter_updates') == 6000
                and not done.get('smoke', False), f'{branch}: formal 6000-step training incomplete')
        train_done[branch] = done
        for split, step in ENDPOINTS:
            done_eval = read(root / branch / f'eval_{split}_{step}' / 'complete.json')
            require(done_eval.get('status') == 'completed_evaluation' and done_eval['split'] == split
                    and done_eval['parameter_updates'] == 0, f'{branch}/{split}/{step}: evaluation incomplete')
            eval_done[branch, split, step] = done_eval

    configs = {branch: read(root / branch / 'config.json') for branch in BRANCHES}
    manifest_path = local_manifest(configs['joint']['manifest'], args.manifest)
    manifest = read(manifest_path)
    require(sha(manifest_path) == configs['joint']['manifest_sha256'], 'Local manifest differs from the training identity')
    reference, metrics, costs, evaluations = {}, [], [], []
    for branch in BRANCHES:
        config, done = configs[branch], train_done[branch]
        for field in ['manifest_sha256', 'parent_sha256', 'seed', 'steps', 'sr_weight']:
            require(config[field] == configs['joint'][field], f'Unmatched control config: {branch}/{field}')
        require(config['branch'] == branch and config['steps'] == 6000 and not config['topology_changes'],
                f'{branch}: wrong registered branch or topology protocol')
        points = done['points']
        extra = config['detail_parameters']
        require(points == config['initial_points'] == configs['joint']['initial_points']
                and extra == (0 if branch == 'joint' else 12 * points),
                f'{branch}: unexpected point count or extra coefficient capacity')
        final_test = None
        for split, step in ENDPOINTS:
            folder = root / branch / f'eval_{split}_{step}'
            values = read(folder / 'metrics.json')
            receipt = eval_done[branch, split, step]
            require(sha(folder / 'metrics.json') == receipt['metrics_sha256'], 'Evaluation metrics/receipt hash mismatch')
            require(values['checkpoint_sha256'] == receipt['checkpoint_sha256'] and values['branch'] == branch,
                    f'{branch}/{split}/{step}: wrong checkpoint or branch')
            require(values['split'] == split and values['checkpoint_metadata']['intervention_step'] == step,
                    f'{branch}/{split}/{step}: endpoint identity mismatch')
            require(values['evaluation_cameras'] == [CAMERAS[split]], f'Unexpected {split} evaluation camera')
            require(values['manifest_sha256'] == config['manifest_sha256'], 'Evaluation/training manifest differs')
            require(values['gaussian_count'] == points and len(values['rows']) == 60
                    and receipt['observations'] == 60, 'Incomplete or different-topology evaluation')
            require([(r['camera_id'], r['frame_index']) for r in values['rows']]
                    == [(CAMERAS[split], f) for f in range(0, 120, 2)], 'Wrong fixed evaluation sequence')
            identity = {key: values[key] for key in ['script_sha256', 'metric_helpers_sha256', 'metric_definitions',
                        'manifest_sha256', 'observation_keys', 'foreground_cache_key', 'evaluation_caches', 'versions']}
            if split in reference:
                require(identity == reference[split], f'{branch}/{split}/{step}: metric/data identities differ')
            else:
                reference[split] = identity
            for region in REGIONS:
                spatial, temporal = values['aggregate'][region], values['temporal_aggregate'][region]
                lpips_key = 'lpips_alex_mean' if region == 'full' else 'lpips_alex_spatial_mask_mean'
                metrics.append({'branch': branch, 'split': split, 'step': step, 'region': region,
                                'psnr': number(spatial['psnr_mean'], 'PSNR'), 'ssim': number(spatial['ssim_mean'], 'SSIM'),
                                'lpips': number(spatial[lpips_key], 'LPIPS'),
                                'temporal': number(temporal.get('gt_relative_warp_l1_mean'), 'temporal', True),
                                'temporal_valid_fraction': number(temporal.get('valid_fraction_of_roi_mean'), 'flow coverage', True)})
            evaluations.append({'branch': branch, 'split': split, 'step': step, 'path': str(folder / 'metrics.json'),
                                'sha256': receipt['metrics_sha256'], 'checkpoint_sha256': receipt['checkpoint_sha256'],
                                'elapsed_seconds': values['elapsed_seconds']})
            if split == 'test':
                final_test = values
        costs.append({'branch': branch, 'points': points, 'extra_detail_parameters': extra,
                      'train_seconds': number(done['train_s'], 'train seconds'),
                      'wall_seconds': number(done['wall_s'], 'wall seconds'),
                      'peak_allocated_gb': number(done['peak_gb'], 'peak allocated GB'),
                      'gpu': config.get('gpu'), 'visible_cuda': config.get('visible_cuda'),
                      'test_6000_evaluation_seconds': number(final_test['elapsed_seconds'], 'evaluation seconds'),
                      'test_6000_render_seconds': number(final_test.get('render_seconds'), 'render seconds', True)})
    metric_index = {(r['branch'], r['split'], r['step'], r['region']): r for r in metrics}
    deltas = []
    for split, step in ENDPOINTS:
        for region in REGIONS:
            local = metric_index['local_residual', split, step, region]
            for baseline in ('global_residual', 'joint'):
                comparison = metric_index[baseline, split, step, region]
                deltas.append({'comparison': f'Local-{LABELS[baseline]}', 'split': split, 'step': step, 'region': region,
                               **{key: None if local[key] is None or comparison[key] is None else local[key] - comparison[key]
                                  for key in ('psnr', 'ssim', 'lpips', 'temporal')}})
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(__file__, out / 'source.py')
    roi = make_roi(root, manifest_path, manifest, identify, out)
    make_quality_plot(metrics, out)
    result = {'status': 'descriptive_summary', 'root': str(root), 'output': str(out),
              'fixed_endpoints': [list(x) for x in ENDPOINTS],
              'metrics': metrics, 'deltas': deltas, 'costs': costs, 'evaluation_files': evaluations,
              'roi': roi, 'local_manifest': str(manifest_path), 'recorded_manifest': configs['joint']['manifest'],
              'protocol_identity': reference, 'input_sha256': identities, 'source_sha256': sha(__file__),
              'information_boundary': 'Completed-result aggregation only. Official masks are provided public information. No model, target, parameter or input prediction is modified.',
              'limitations': ['One scene, one explicit sampling seed, nonstandard short-window pilot; correlated frames, no significance test.',
                              'No best-dev/test endpoint selection or automatic efficacy verdict.',
                              'Foreground LPIPS is a spatial-map regional extension; full scalar LPIPS is reported separately.',
                              'Evaluation elapsed_seconds includes caches/metrics/I/O and is not rendering-only runtime; missing render_seconds remains null.',
                              'Official-mask bbox panel uses uint8 saved predictions; main metrics use float renders.'],
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
    parser.add_argument('--root', type=Path, required=True, help='Batch with completed joint/global_residual/local_residual')
    parser.add_argument('--out', type=Path, help='New output directory; defaults to ROOT/summary')
    parser.add_argument('--manifest', type=Path, help='Local copy if recorded remote manifest path cannot be resolved; hash must match')
    main(parser.parse_args())
