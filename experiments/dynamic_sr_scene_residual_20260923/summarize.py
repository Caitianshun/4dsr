#!/usr/bin/env python3
"""Summarize only a fully completed full-scene residual batch; CPU/files only.

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
REGIONS = ('full', 'dynamic', 'static')
LABELS = {'joint': 'Joint', 'global_residual': 'Global', 'local_residual': 'Local'}
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
    figure.suptitle('MeetRoom discussion pilot | fixed test cam00 | 6000 added steps\nLocal axis ranges; descriptive single-run comparison')
    figure.savefig(out / 'quality.png', dpi=160)
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
    # Existing 9/20 regions are unchanged, not selected from these methods.
    boxes = {'time_changing': (504, 420, 672, 588), 'mostly_static': (1092, 504, 1260, 672)}
    files = {}
    for name, box in boxes.items():
        tile_width, tile_height = box[2] - box[0], box[3] - box[1]
        panel = Image.new('RGB', (4 * tile_width, tile_height + 48), 'white')
        draw = ImageDraw.Draw(panel)
        for index, (label, image) in enumerate(images.items()):
            panel.paste(image.crop(box), (index * tile_width, 48))
            draw.text((index * tile_width + 5, 5), label, fill='black')
        draw.text((5, 27), f'cam00 frame40 | historical {name} ROI | native PNG crops', fill='black')
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
            'selection': 'Fixed 2026-09-20 cam00 frame40 detail-audit coordinates, originally chosen using HR detail energy; no selection by current method differences; illustrative, not representative sampling',
            'source_report': str(PROJECT / 'docs/dynamic_sr_final_detail_audit_2026-09-20.md'),
            'images': identities, 'files': files,
            'precision': 'Saved uint8 PNGs; ROI panels native resolution, full-scene overview display resized only. Neither replaces float-render metrics.'}


def markdown(summary):
    lines = ['# MeetRoom 完整场景残差三分支：固定端点汇总', '',
             '本文件仅描述已完成结果，不自动判定方法有效。使用 MeetRoom discussion 的完整 RGB 和复杂背景；不作人体抠图或黑底合成，不要求官方掩码。', '',
             'Joint 为原联合训练；Global/Local 各加每点 4×RGB 系数，底座仍接受 LR+SR 更新，仅残差渲染的几何/不透明度支撑停止梯度。Global 与 Local 的附加参数量相同。', '',
             '整图为主指标；动态/静态区辅助。动态区来自固定 HR 时间变化评价缓存，仅用于评价，不是人物语义或训练掩码。PSNR/SSIM 越高越好，LPIPS/参考相对时序 L1 越低越好；区域 LPIPS 是空间图区域均值。', '',
             '| 划分/追加步数 | 分支 | 区域 | PSNR | SSIM | LPIPS | 时序 L1 |',
             '| --- | --- | --- | ---: | ---: | ---: | ---: |']
    names = {'full': '整图', 'dynamic': '时间变化区', 'static': '其余区域'}
    for row in summary['metrics']:
        temporal = '未提供' if row['temporal'] is None else f"{row['temporal']:.8f}"
        lines.append(f"| {row['split']}/{row['step']} | {LABELS[row['branch']]} | {names[row['region']]} | "
                     f"{row['psnr']:.6f} | {row['ssim']:.6f} | {row['lpips']:.6f} | {temporal} |")
    lines += ['', '固定 test/6000 的 Local 差值如下；ΔLPIPS 和 Δ时序为正表示变差。全部 dev 差值保存在 JSON。', '',
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
    out = Path(summary['output'])
    lines += ['', '总评价时间包含缓存、指标和文件写入。独立渲染时间为 CUDA 同步的 render_model 耗时合计，包含首帧，不含相机构造、拷贝、指标或 I/O，不是端到端 FPS。相同步数不等于同计算预算。', '',
              f'![固定测试指标]({out / "quality.png"})', '',
              f'![完整场景]({out / summary["roi"]["files"]["full_scene"]})', '',
              f'![固定时间变化区域]({out / summary["roi"]["files"]["time_changing"]})', '',
              f'![固定静态区域]({out / summary["roi"]["files"]["mostly_static"]})', '',
              '全图与局部均为 test cam00/frame40；局部坐标继承 9 月 20 日审计，原先依据 HR 高频能量选取，未按本轮收益选择，仅作固定展示，不代表随机样本。原预测文件未改；全图仅作显示缩小，局部为原生分辨率 8-bit PNG 裁剪，不能替代浮点主指标。', '',
              '一个已使用的开发场景、一个显式采样种子、60 帧短窗。连续帧不独立，未估计训练随机波动。dev/1200、dev/6000 和 test/6000 均保留，不挑最好端点，不作显著性或普遍有效性声明。', '',
              '来源身份、各分支成本和全部差值见 [summary.json](summary.json)。', '']
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
                        'manifest_sha256', 'observation_keys', 'evaluation_caches', 'versions']}
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
              'information_boundary': 'Completed-result aggregation only. Original full-scene RGB retained; no provided person mask or compositing. GT time-changing masks/flows are evaluation-only. No model, target, parameter or prediction source is modified.',
              'limitations': ['One scene, one explicit sampling seed, nonstandard short-window pilot; correlated frames, no significance test.',
                              'No best-dev/test endpoint selection or automatic efficacy verdict.',
                              'Dynamic/static LPIPS are spatial-map regional extensions; full scalar LPIPS is reported separately.',
                              'Evaluation elapsed_seconds includes caches/metrics/I/O; render_seconds is separately CUDA-synchronized render_model time including the first frame, not end-to-end FPS.',
                              'Historical fixed ROI panels use saved uint8 predictions; whole-scene panel is display resized; main metrics use float renders.'],
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
