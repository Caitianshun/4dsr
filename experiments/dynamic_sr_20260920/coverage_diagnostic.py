#!/usr/bin/env python3
"""Evaluation-only three-tier SR-supervision-coverage diagnostic.

Prespecified camera IDs and frame indices; no fitting or HR-driven selection.
Original float metrics are reused only after identity and render-probe checks.
"""
from __future__ import annotations

import gc
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from output_swap import (PROJECT, OLD, ORIGINAL, load_manifest,
    N3DVPreparedDataset, observation_to_4dgs_camera, resized_camera,
    load_checkpoint, render_image, image_array, read_rgb, spatial_metrics,
    file_sha, write_json, write_rgb)

ROOT = PROJECT / 'output/dynamic_sr_20260920/coverage_diagnostic'
CONT = PROJECT / 'output/dynamic_sr_20260920/continuation'
SWAP = PROJECT / 'output/dynamic_sr_20260920/output_swap'
FRAMES = [0, 40, 80, 118]
SCENES = ['cook_spinach', 'meetroom_discussion']
KEYS = ['psnr', 'ssim', 'lpips_alex', 'mse']


def read_json(path):
    return json.loads(Path(path).read_text())


def average(rows):
    assert rows
    return {k: float(np.mean([r['metrics'][k] for r in rows])) for k in KEYS}


@torch.no_grad()
def scene_diagnostic(scene, metric):
    start = time.monotonic()
    source = ORIGINAL / f'{scene}_sr_w01'
    config = read_json(source / 'config.json')
    manifest = load_manifest(config['manifest'])
    manifest_hash = file_sha(config['manifest'])
    assert manifest_hash == config['manifest_sha256']
    all_observations = {(o['camera_id'], o['frame_index']): o for o in manifest['observations']}
    train_cameras = sorted({o['camera_id'] for o in manifest['observations'] if o['split'] == 'train'})
    prior_cameras = sorted(config['prior_cameras'].split(','))
    remaining = sorted(set(train_cameras) - set(prior_cameras))
    selected = [remaining[i] for i in [0, (len(remaining)-1)//2, len(remaining)-1]]
    assert len(set(selected)) == 3
    groups = {'sr_supervised_train': prior_cameras,
              'lr_only_train': selected, 'novel_cam00': ['cam00']}
    path_root = Path(manifest['_root'])
    inputs = []
    for group, cameras in groups.items():
        for cam in cameras:
            for frame in FRAMES:
                obs = all_observations[(cam, frame)]
                assert obs['split'] == ('test' if cam == 'cam00' else 'train')
                lr, hr = path_root / obs['lr_path'], path_root / obs['hr_path']
                assert lr.is_file() and hr.is_file()
                row = dict(group=group, camera_id=cam, frame_index=frame,
                    split=obs['split'], time=obs['time'], lr_path=str(lr),
                    lr_sha256=file_sha(lr), hr_path=str(hr), hr_sha256=file_sha(hr),
                    lr_was_training_input=cam in train_cameras,
                    sr_was_training_target=cam in prior_cameras,
                    hr_for_evaluation_only=True)
                inputs.append(row)
    out = ROOT / scene
    out.mkdir()
    selection = dict(scene=scene, frames=FRAMES, groups=groups,
        all_train_cameras=train_cameras, eligible_lr_only_cameras=remaining,
        selection_rule='First, lower-median, last sorted eligible camera ID; fixed before metrics; not based on HR or error.',
        manifest=config['manifest'], manifest_sha256=manifest_hash,
        source_config=str(source/'config.json'), source_config_sha256=file_sha(source/'config.json'),
        inputs=inputs)
    write_json(out/'selection.json', selection)
    old_meta = read_json(SWAP/scene/'metadata.json')
    assert old_meta['manifest_sha256'] == manifest_hash
    early_metrics_path = SWAP/scene/'G6C6/metrics.json'
    early_metrics = read_json(early_metrics_path)
    checkpoints = {'early_6k': source/'checkpoint_6000.pt',
        'joint_18k': CONT/f'{scene}_joint/checkpoint_18000.pt',
        'appearance_18k': CONT/f'{scene}_appearance_only/checkpoint_18000.pt'}
    result = dict(selection=selection, checkpoints={}, branches={})
    w, h = manifest['resolutions']['hr']
    empty_mask = np.zeros((h, w), dtype=bool)
    for branch, checkpoint in checkpoints.items():
        ck_hash = file_sha(checkpoint)
        if branch == 'early_6k':
            assert old_meta['checkpoints']['6000']['sha256'] == ck_hash
            train_rows, test_rows = early_metrics['train_rows'], early_metrics['rows']
            provenance = [early_metrics_path, SWAP/scene/'metadata.json']
        else:
            run = checkpoint.parent
            run_config = read_json(run/'config.json')
            test_path = run.with_name(run.name+'_evaluation')/'metrics.json'
            fit_path = run/'fit_18000.json'
            test, fit = read_json(test_path), read_json(fit_path)
            assert test['checkpoint_sha256'] == ck_hash
            assert test['manifest_sha256'] == manifest_hash == run_config['manifest_sha256']
            assert run_config['prior_cameras'] == config['prior_cameras']
            assert run_config['parent_sha256'] == old_meta['checkpoints']['6000']['sha256']
            train_rows, test_rows = fit['rows'], test['rows']
            provenance = [test_path, fit_path, run/'config.json']
        rows = []
        for r in train_rows:
            if r['frame_index'] in FRAMES:
                assert r['camera_id'] in prior_cameras
                rows.append(dict(group='sr_supervised_train', camera_id=r['camera_id'],
                    frame_index=r['frame_index'], metrics=r['render_hr'], reused=True))
        for r in test_rows:
            if r['frame_index'] in FRAMES:
                rows.append(dict(group='novel_cam00', camera_id='cam00',
                    frame_index=r['frame_index'], metrics=r['spatial']['full'], reused=True))
        assert len(rows) == 20
        model, _, _, ck = load_checkpoint(checkpoint)
        model._deformation.eval()
        assert ck['metadata']['manifest_sha'] == manifest_hash
        data = N3DVPreparedDataset(manifest, 'train', 'lr')
        probe_pairs = [(prior_cameras[0], 40), ('cam00', 40)]
        new_pairs = [(cam, frame) for cam in selected for frame in FRAMES]
        data.observations = [all_observations[p] for p in new_pairs + probe_pairs]
        probes = []
        for i, obs in enumerate(data.observations):
            item = data[i]
            camera = resized_camera(observation_to_4dgs_camera(item, i), h, w)
            pred = image_array(render_image(model, camera)['render'])
            gt = read_rgb(path_root/obs['hr_path'])
            values = spatial_metrics(pred, gt, empty_mask, metric)['full']
            assert all(np.isfinite(values[k]) for k in KEYS)
            if i < len(new_pairs):
                destination = out/branch/'predictions'/obs['camera_id']/f'{obs["frame_index"]:04d}.png'
                write_rgb(destination, pred)
                rows.append(dict(group='lr_only_train', camera_id=obs['camera_id'],
                    frame_index=obs['frame_index'], metrics=values, reused=False,
                    prediction_path=str(destination), prediction_sha256=file_sha(destination)))
            else:
                old = next(r['metrics'] for r in rows if r['camera_id'] == obs['camera_id'] and r['frame_index'] == obs['frame_index'])
                delta = {k: abs(values[k]-old[k]) for k in KEYS}
                assert delta['psnr'] < 2e-5 and delta['ssim'] < 2e-6 and delta['lpips_alex'] < 1e-5, delta
                probes.append(dict(camera_id=obs['camera_id'], frame_index=obs['frame_index'],
                                   metric_abs_delta=delta, passed=True))
        branch_result = dict(rows=rows, groups={}, reuse_probes=probes,
            reused_sources=[dict(path=str(p), sha256=file_sha(p)) for p in provenance])
        for group, cams in groups.items():
            subset = [r for r in rows if r['group'] == group]
            assert len(subset) == len(cams)*4
            branch_result['groups'][group] = dict(count=len(subset), mean=average(subset),
                per_camera={cam: average([r for r in subset if r['camera_id'] == cam]) for cam in cams})
        result['branches'][branch] = branch_result
        result['checkpoints'][branch] = dict(path=str(checkpoint), sha256=ck_hash,
            metadata=ck['metadata'], point_count=len(model.get_xyz))
        write_json(out/f'{branch}.json', branch_result)
        del model, ck, data
        gc.collect()
        torch.cuda.empty_cache()
    result['deltas_from_early'] = {}
    for branch in ['joint_18k', 'appearance_18k']:
        result['deltas_from_early'][branch] = {
            group: {k: result['branches'][branch]['groups'][group]['mean'][k]-
                       result['branches']['early_6k']['groups'][group]['mean'][k] for k in KEYS}
            for group in groups}
    result['elapsed_seconds'] = time.monotonic()-start
    write_json(out/'metrics.json', result)
    return result


def main():
    begin = time.monotonic()
    ROOT.mkdir(exist_ok=False)
    torch.set_num_threads(4)
    import lpips
    metric = lpips.LPIPS(net='alex').cuda().eval()
    result = dict(protocol='Evaluation only, fixed four frames; no HR selection or optimization. Group means average image-level scores; equal frames per camera.',
        frames=FRAMES, script_sha256=file_sha(__file__),
        common_sha256=file_sha(OLD/'common.py'), evaluator_sha256=file_sha(OLD/'evaluate.py'),
        gpu=torch.cuda.get_device_name(), cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        versions=dict(torch=torch.__version__, numpy=np.__version__), scenes={})
    write_json(ROOT/'protocol.json', result)
    for scene in SCENES:
        result['scenes'][scene] = scene_diagnostic(scene, metric)
    result['elapsed_seconds'] = time.monotonic()-begin
    result['completed'] = True
    write_json(ROOT/'metrics.json', result)
    lines = ['# SR监督覆盖：无需重训的三层评价', '',
        '每相机仅固定0/40/80/118四帧。所有组使用相同帧集合，不与60帧均值混用。'
        'PSNR/SSIM越高越好，LPIPS越低越好；结果是开发诊断，不是相机覆盖的因果干预。', '',
        '| 场景 | 观察层 | 状态 | PSNR | SSIM | LPIPS |', '|---|---|---|---:|---:|---:|']
    for scene, s in result['scenes'].items():
        for group in s['selection']['groups']:
            for branch, b in s['branches'].items():
                v = b['groups'][group]['mean']
                lines.append(f'| {scene} | {group} | {branch} | {v["psnr"]:.4f} | {v["ssim"]:.6f} | {v["lpips_alex"]:.6f} |')
    lines += ['', '相机按未提供SR先验的训练相机ID排序，选首/中/末，未按误差选择。'
              '组间绝对差异受相机位置、内容、遮挡影响；同相机前后变化较有解释价值。'
              '少数相关帧及每场景3个LR-only相机不能代替全相机基准或独立重复。', '',
              f'完整输入、来源哈希、逐图指标、复用探针、耗时见metrics.json。本次耗时{result["elapsed_seconds"]:.1f}秒。']
    (ROOT/'summary.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(completed=True, elapsed_seconds=result['elapsed_seconds'],
        output=str(ROOT), deltas={s:v['deltas_from_early'] for s,v in result['scenes'].items()}), ensure_ascii=False))


if __name__ == '__main__':
    main()
