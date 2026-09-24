#!/usr/bin/env python3
"""No-training final-output diagnostic of DC versus angular SH increments.

E: DC6/H6; A: DC18/H18 from appearance-only continuation; D: DC18/H6;
H: DC6/H18. All use strictly identical final spatial support at each time.
E/A metrics are reused from verified original evaluation; only D/H are new.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image
import torch

from output_swap import (PROJECT, OLD, ORIGINAL, UPSTREAM, N3DVPreparedDataset,
    load_manifest, observation_to_4dgs_camera, load_checkpoint, final_outputs,
    render_outputs, parity_check, resized_camera, downsample, image_array,
    metric_psnr, read_rgb, spatial_metrics, temporal_metrics, prepare_cache,
    full_flow, aggregate, file_sha, write_json, write_rgb)
from common import named_parameters

CONTINUATION = PROJECT/'output/dynamic_sr_20260920/continuation'
SWAPS = PROJECT/'output/dynamic_sr_20260920/output_swap'
SCENES = ['cook_spinach', 'meetroom_discussion']
DEFINITIONS = {'E': 'DC6/H6, reused early endpoint',
    'A': 'DC18/H18, reused appearance-only endpoint',
    'D': 'DC18/H6, only DC increment retained',
    'H': 'DC6/H18, only higher-order increment retained'}


def read_json(path):
    return json.loads(Path(path).read_text())


def tensor_hash(x):
    return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def split_mix(early, late, name):
    assert early[4].ndim == 3 and early[4].shape[1:] == (16, 3)
    assert early[4].shape == late[4].shape
    dc, hi = (late, early) if name == 'D' else (early, late)
    sh = torch.cat((dc[4][:, :1, :], hi[4][:, 1:, :]), dim=1).contiguous()
    assert torch.equal(sh[:, :1], dc[4][:, :1])
    assert torch.equal(sh[:, 1:], hi[4][:, 1:])
    return (*early[:4], sh)


def conditional_contrasts(values):
    e, a, d, h = [values[k] for k in ['E', 'A', 'D', 'H']]
    return dict(DC_increment_given_early_higher=d-e,
        DC_increment_given_late_higher=a-h,
        higher_increment_given_early_DC=h-e,
        higher_increment_given_late_DC=a-d,
        interaction=a-d-h+e, joint_increment=a-e)


def build_contrasts(results):
    result = dict(interpretation='Metric differences of output ablations. Positive MSE means worse. Interaction is A-D-H+E; it is not a fitted causal model.',
                  test={}, train={})
    for region in ['full', 'dynamic', 'static']:
        result['test'][region] = {}
        for metric in ['mse_mean', 'psnr_mean', 'ssim_mean', 'lpips_alex_spatial_mask_mean']:
            vals = {k: v['aggregate'][region][metric] for k, v in results.items()}
            result['test'][region][metric] = dict(values=vals, **conditional_contrasts(vals))
    for target in ['render_hr', 'render_prior']:
        result['train'][target] = {}
        for metric in ['mse', 'psnr', 'ssim', 'lpips_alex']:
            vals = {k: v['train_aggregate'][target][metric] for k, v in results.items()}
            result['train'][target][metric] = dict(values=vals, **conditional_contrasts(vals))
    return result


def reuse_endpoints(scene, manifest_hash, app_checkpoint_hash, cache_key):
    early_path = SWAPS/scene/'G6C6/metrics.json'
    early_meta = read_json(SWAPS/scene/'metadata.json')
    assert early_meta['manifest_sha256'] == manifest_hash
    assert early_meta['cache_key'] == cache_key
    early = read_json(early_path)
    app_path = CONTINUATION/f'{scene}_appearance_only_evaluation/metrics.json'
    app = read_json(app_path)
    assert app['manifest_sha256'] == manifest_hash
    assert app['checkpoint_sha256'] == app_checkpoint_hash
    assert app['cache_key'] == cache_key
    fit_path = CONTINUATION/f'{scene}_appearance_only/fit_18000.json'
    fit = read_json(fit_path)
    assert len(early['rows']) == len(app['rows']) == 60
    assert len(early['train_rows']) == len(fit['rows']) == 16
    common = ['aggregate', 'temporal_aggregate']
    result = {'E': {k: early[k] for k in common}, 'A': {k: app[k] for k in common}}
    result['E']['lr_closure_aggregate'] = early['lr_closure_aggregate']
    sampling_path = CONTINUATION/f'{scene}_appearance_only_sampling/sampling.json'
    sampling = read_json(sampling_path)
    assert sampling['checkpoint_sha256'] == app_checkpoint_hash and len(sampling['rows']) == 60
    closure_rows = [{'closure': {'full': dict(
        mse=r['sampling']['render_4x_lr_grid']['mse_to_real_lr'],
        psnr=r['sampling']['render_4x_lr_grid']['psnr_to_real_lr'])}} for r in sampling['rows']]
    result['A']['lr_closure_aggregate'] = aggregate(closure_rows, 'closure')
    result['E'].update(train_aggregate=early['train_aggregate'], train_lr_psnr=early['train_lr_psnr'])
    result['A'].update(train_aggregate=fit['aggregate'], train_lr_psnr=fit['lr_psnr'])
    provenance = {'E': dict(test_path=str(early_path), test_sha256=file_sha(early_path)),
        'A': dict(test_path=str(app_path), test_sha256=file_sha(app_path),
                  fit_path=str(fit_path), fit_sha256=file_sha(fit_path),
                  sampling_path=str(sampling_path), sampling_sha256=file_sha(sampling_path))}
    return result, provenance


@torch.no_grad()
def evaluate_scene(scene, root, metric):
    begin = time.monotonic()
    out = root/scene
    out.mkdir(exist_ok=False)
    source = ORIGINAL/f'{scene}_sr_w01'
    app_run = CONTINUATION/f'{scene}_appearance_only'
    config, app_config = read_json(source/'config.json'), read_json(app_run/'config.json')
    assert config['teacher'] == 'sr' and config['weight'] == .1 and not config['dense']
    assert app_config['mode'] == 'appearance_only' and app_config['topology_changes'] is False
    assert app_config['fork_intervention_step'] == 6000 and app_config['endpoint_intervention_step'] == 18000
    manifest = load_manifest(config['manifest'])
    manifest_hash = file_sha(config['manifest'])
    assert manifest_hash == app_config['manifest_sha256'] == config['manifest_sha256']
    prior_cameras = config['prior_cameras'].split(',')
    assert app_config['prior_cameras'] == config['prior_cameras']
    paths = {'E': source/'checkpoint_6000.pt', 'A': app_run/'checkpoint_18000.pt'}
    hashes = {name: file_sha(path) for name, path in paths.items()}
    assert hashes['E'] == app_config['parent_sha256']
    models, checkpoints = {}, {}
    for name, path in paths.items():
        model, _, _, ck = load_checkpoint(path)
        model._deformation.eval()
        for p in named_parameters(model).values():
            p.requires_grad_(False)
        models[name], checkpoints[name] = model, ck
    early, app = models['E'], models['A']
    assert len(early.get_xyz) == len(app.get_xyz) == config['initial_points']
    assert early.active_sh_degree == app.active_sh_degree == config['sh_degree'] == 3
    assert checkpoints['E']['hidden'] == checkpoints['A']['hidden']
    assert checkpoints['A']['metadata']['parent_sha'] == hashes['E']
    frozen = {}
    for name, p in named_parameters(early).items():
        if name not in ['_features_dc', '_features_rest'] and 'shs_deform.' not in name:
            other = named_parameters(app)[name]
            assert torch.equal(p, other), name
            frozen[name] = tensor_hash(p)
    assert len(frozen) == len(app_config['frozen_parameter_hashes'])
    assert torch.equal(early._deformation_table, app._deformation_table)
    parity = parity_check(models, manifest, prior_cameras)
    del checkpoints
    observations = sorted([o for o in manifest['observations'] if o['split'] == 'test'],
                          key=lambda o: o['frame_index'])
    assert len(observations) == 60 and {o['camera_id'] for o in observations} == {'cam00'}
    cache, cache_info = prepare_cache(manifest, observations,
        SimpleNamespace(dynamic_threshold=.025, flow_scale=.5, cache_dir=None))
    dynamic = np.asarray(Image.open(cache/'dynamic_mask.png')) > 0
    bbox = cache_info['dynamic_bbox_xyxy']
    results, reused = reuse_endpoints(scene, manifest_hash, hashes['A'], cache_info['cache_key'])
    old_meta = read_json(SWAPS/scene/'metadata.json')
    assert old_meta['checkpoints']['6000']['sha256'] == hashes['E']
    meta = dict(scene=scene, definitions=DEFINITIONS, checkpoints={k: dict(path=str(paths[k]),
        sha256=hashes[k]) for k in paths}, reused_metrics=reused,
        manifest_sha256=manifest_hash, manifest=config['manifest'], cache_key=cache_info['cache_key'],
        evaluation_cache=str(cache), dynamic_fraction=float(dynamic.mean()), dynamic_bbox_xyxy=bbox,
        renderer_endpoint_parity=parity, frozen_parameter_hashes=frozen,
        deformation_table_hash=tensor_hash(early._deformation_table), point_count=len(early.get_xyz),
        script_sha256=file_sha(__file__), output_swap_helper_sha256=file_sha(Path(__file__).with_name('output_swap.py')),
        common_sha256=file_sha(OLD/'common.py'), evaluator_sha256=file_sha(OLD/'evaluate.py'),
        upstream_renderer_sha256=file_sha(UPSTREAM/'gaussian_renderer/__init__.py'),
        gpu=torch.cuda.get_device_name(), cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        versions=dict(torch=torch.__version__, numpy=np.__version__, opencv=cv2.__version__),
        interpretation='DC is angle-independent at a fixed time; higher-order SH changes with angle. Both are outputs of a time-conditioned SH head. Angular changes may encode legitimate reflectance, not necessarily spurious compensation. Off-trajectory ablations are not deployment methods or independent holdout validation.',
        information_boundary='No optimizer step, parameter update, or HR-driven selection. Exactly two prespecified new output mixes; HR used only for evaluation with existing mask/flow.',
        topology_identity='Common 6k parent; no densification/reorder in either path; canonical xyz and all geometry parameters and deformation table are bitwise identical.')
    write_json(out/'metadata.json', meta)
    w, h = manifest['resolutions']['hr']
    data = N3DVPreparedDataset(manifest, 'test', 'lr')
    data.observations = observations
    rows = {k: [] for k in ['D', 'H']}
    previous_pred, previous_gt, support_checks = {}, None, []
    for i, obs in enumerate(observations):
        item = data[i]
        cam = resized_camera(observation_to_4dgs_camera(item, i), h, w)
        e, a = final_outputs(early, cam.time), final_outputs(app, cam.time)
        checks = {}
        for idx, name in enumerate(['xyz', 'scale', 'rotation', 'opacity']):
            assert torch.equal(e[idx], a[idx]), (scene, obs['frame_index'], name)
            assert torch.isfinite(e[idx]).all()
            checks[name] = dict(equal=True, max_abs_delta=0., sha256=tensor_hash(e[idx]))
        support_checks.append(dict(frame_index=obs['frame_index'], time=obs['time'], outputs=checks))
        gt, lr = read_rgb(Path(manifest['_root'])/obs['hr_path']), item['image'].cuda()
        if i:
            pair = cache_info['flow_pairs'][i-1]
            with np.load(cache/pair['file'], allow_pickle=False) as f:
                flow, valid = full_flow(f['backward'], f['valid'], h, w)
        for name in rows:
            mix = split_mix(e, a, name)
            raw = render_outputs(e, mix, cam, early.active_sh_degree)
            assert torch.isfinite(raw).all()
            pred = image_array(raw)
            lr_mse = float((downsample(raw, lr.shape[-2:])-lr).square().mean())
            row = dict(frame_index=obs['frame_index'], time=obs['time'],
                spatial=spatial_metrics(pred, gt, dynamic, metric, 'cuda', bbox),
                closure={'full': dict(mse=lr_mse, psnr=metric_psnr(lr_mse))})
            if i:
                row['temporal'] = temporal_metrics(previous_pred[name], pred, previous_gt, gt, flow, valid, dynamic)
            write_rgb(out/name/'predictions'/f'{obs["frame_index"]:04d}.png', pred)
            rows[name].append(row)
            previous_pred[name] = pred
        previous_gt = gt
        if (i+1) % 20 == 0:
            print(f'{scene} SH increment test {i+1}/60', flush=True)
    write_json(out/'geometry_equality_all_times.json', dict(all_equal=True, rows=support_checks))
    train = {k: [] for k in rows}
    data = N3DVPreparedDataset(manifest, 'train', 'lr')
    data.observations = [o for o in data.observations if o['camera_id'] in prior_cameras and
                         o['frame_index'] in [0, 40, 80, 118]]
    assert len(data) == 16
    zero_mask = np.zeros((h, w), bool)
    for i, obs in enumerate(data.observations):
        item = data[i]
        cam = resized_camera(observation_to_4dgs_camera(item, i), h, w)
        e, a = final_outputs(early, cam.time), final_outputs(app, cam.time)
        assert all(torch.equal(e[j], a[j]) for j in range(4))
        hr = read_rgb(Path(manifest['_root'])/obs['hr_path'])
        prior = read_rgb(Path(manifest['_root'])/'sr_swinir_x4'/obs['camera_id']/Path(obs['lr_path']).name)
        lr = item['image'].cuda()
        for name in train:
            raw = render_outputs(e, split_mix(e, a, name), cam, early.active_sh_degree)
            pred = image_array(raw)
            lr_mse = float((downsample(raw, lr.shape[-2:])-lr).square().mean())
            train[name].append(dict(camera_id=obs['camera_id'], frame_index=obs['frame_index'],
                lr_psnr=metric_psnr(lr_mse), render_hr=spatial_metrics(pred, hr, zero_mask, metric)['full'],
                render_prior=spatial_metrics(pred, prior, zero_mask, metric)['full']))
            write_rgb(out/name/'train_predictions'/obs['camera_id']/f'{obs["frame_index"]:04d}.png', pred)
    integrity = {}
    for name in rows:
        train_aggregate = {kind: {key: float(np.mean([r[kind][key] for r in train[name]]))
             for key in ['psnr', 'ssim', 'lpips_alex', 'mse']} for kind in ['render_hr', 'render_prior']}
        result = dict(case=name, rows=rows[name], aggregate=aggregate(rows[name], 'spatial'),
            temporal_aggregate=aggregate(rows[name], 'temporal'),
            lr_closure_aggregate=aggregate(rows[name], 'closure'), train_rows=train[name],
            train_aggregate=train_aggregate, train_lr_psnr=float(np.mean([r['lr_psnr'] for r in train[name]])))
        write_json(out/name/'metrics.json', result)
        results[name] = {k: v for k, v in result.items() if k not in ['rows', 'train_rows']}
        integrity[name] = dict(test_rows=len(rows[name]), train_rows=len(train[name]),
            temporal_pairs=sum('temporal' in r for r in rows[name]),
            test_png_count=len(list((out/name/'predictions').glob('*.png'))),
            train_png_count=len(list((out/name/'train_predictions').glob('*/*.png'))),
            metrics_sha256=file_sha(out/name/'metrics.json'))
        assert integrity[name]['test_rows'] == integrity[name]['test_png_count'] == 60
        assert integrity[name]['train_rows'] == integrity[name]['train_png_count'] == 16
        assert integrity[name]['temporal_pairs'] == 59
    contrast = build_contrasts(results)
    write_json(out/'contrasts.json', contrast)
    write_json(out/'metrics.json', dict(**meta, cases=results, contrasts=contrast,
        geometry_equality_all_times=True, elapsed_seconds=time.monotonic()-begin))
    write_json(out/'complete.json', dict(status='complete', no_training=True,
        endpoint_parity=True, geometry_equality_all_times=True, integrity=integrity,
        metrics_sha256=file_sha(out/'metrics.json'), contrasts_sha256=file_sha(out/'contrasts.json'),
        elapsed_seconds=time.monotonic()-begin))
    print(json.dumps(dict(scene=scene, status='complete', elapsed_seconds=time.monotonic()-begin,
        full={k: v['aggregate']['full'] for k, v in results.items()})), flush=True)
    return dict(scene=scene, cases=results, contrasts=contrast, path=str(out/'metrics.json'),
                elapsed_seconds=time.monotonic()-begin)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, default=PROJECT/'output/dynamic_sr_20260920/sh_increment')
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out/'complete.json').exists():
        raise FileExistsError(args.out/'complete.json')
    cv2.setNumThreads(4)
    cv2.setRNGSeed(20260918)
    torch.set_num_threads(4)
    import lpips
    metric = lpips.LPIPS(net='alex', spatial=False).cuda().eval().requires_grad_(False)
    begin = time.monotonic()
    results = []
    for scene in SCENES:
        results.append(evaluate_scene(scene, args.out, metric))
        gc.collect()
        torch.cuda.empty_cache()
    write_json(args.out/'metrics.json', dict(scenes=results, definitions=DEFINITIONS,
        script_sha256=file_sha(__file__), elapsed_seconds=time.monotonic()-begin))
    lines = ['# SH 增量输出诊断', '',
        '没有进行训练。E=6k 原输出；A=18k 只更新外观的完整输出；D=只保留 DC 增量；H=只保留高阶 SH 增量。', '',
        '所有组合在每个时刻使用同一位置、尺度、旋转和透明度。DC 在固定时刻不随方向变化；高阶 SH 表达方向变化，但不等价于错误镜面反射。两个部分均可随时间变化。', '',
        'E/A 复用哈希核对后的历史浮点评测。D/H 新计算 60 帧 cam00 和 16 个固定训练观测；同一 ROI/光流和指标定义。离轨输出消融用于诊断，不能作为训练算法或独立测试结果。', '']
    for scene in results:
        lines += [f'## {scene["scene"]}', '', '| 输出 | 测试 PSNR | 测试 SSIM | 测试 LPIPS | 训练 HR PSNR | 训练先验 PSNR |', '| --- | ---: | ---: | ---: | ---: | ---: |']
        for k in ['E', 'A', 'D', 'H']:
            v = scene['cases'][k]
            a, th, tp = v['aggregate']['full'], v['train_aggregate']['render_hr'], v['train_aggregate']['render_prior']
            lines.append(f'| {k} | {a["psnr_mean"]:.6f} | {a["ssim_mean"]:.6f} | {a["lpips_alex_mean"]:.6f} | {th["psnr"]:.6f} | {tp["psnr"]:.6f} |')
        lines += ['', 'MSE 条件差（正值为更差；不把两个增量视为可独立相加）：', '',
            '| 区域 | DC 增量，高阶早期 | DC 增量，高阶后期 | 高阶增量，DC 早期 | 高阶增量，DC 后期 | 交互项 |',
            '| --- | ---: | ---: | ---: | ---: | ---: |']
        for region in ['full', 'dynamic', 'static']:
            c = scene['contrasts']['test'][region]['mse_mean']
            fields = ['DC_increment_given_early_higher', 'DC_increment_given_late_higher',
                'higher_increment_given_early_DC', 'higher_increment_given_late_DC', 'interaction']
            lines.append('| '+region+' | '+' | '.join(f'{c[k]:+.8f}' for k in fields)+' |')
        lines += ['', '所有逐帧与训练拟合三指标、区域、时序、LR 闭环和条件差均见对应 metrics.json/contrasts.json。', '']
    (args.out/'summary.md').write_text('\n'.join(lines)+'\n')
    write_json(args.out/'complete.json', dict(status='complete', scenes=SCENES,
        no_training=True, metrics_sha256=file_sha(args.out/'metrics.json'),
        summary_sha256=file_sha(args.out/'summary.md'), elapsed_seconds=time.monotonic()-begin))
    print('SH_INCREMENT_COMPLETE '+str(args.out), flush=True)


if __name__ == '__main__':
    main()
