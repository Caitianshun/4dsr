#!/usr/bin/env python3
"""Evaluation-only matched LR / SR0.1 / sparse HR-oracle trajectories.

Fixed frames and camera-ID selection precede metric computation. The LR metric
uses exactly the trained D(Render_HR) path, not native-LR rasterization. Native
LR PSNR is explicitly a separate out-of-objective diagnostic.
"""
from __future__ import annotations

import gc
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[2]
OLD = PROJECT / 'experiments/dynamic_sr_20260918'
sys.path.insert(0, str(OLD))
from common import UPSTREAM, downsample, load_checkpoint, render_image, resized_camera
from n3dv_data import N3DVPreparedDataset, load_manifest, observation_to_4dgs_camera
from evaluate import file_sha, image_array, read_rgb, ssim_map_rgb, metric_psnr, write_json

SOURCE = PROJECT / 'output/dynamic_sr_20260919'
DEST = PROJECT / 'output/dynamic_sr_20260921/resolution_trajectory'
SCENES = ['cook_spinach', 'cut_roasted_beef', 'meetroom_discussion']
CASES = ['lr_long', 'sr_w01', 'hr_oracle_w10']
STEPS = [1200, 6000, 12000, 18000]
FRAMES = [0, 40, 80, 118]
KEYS = ['psnr', 'ssim', 'lpips_alex', 'mse']


def read(path):
    return json.loads(Path(path).read_text())


@torch.no_grad()
def scores(pred, gt, metric):
    assert pred.shape == gt.shape and np.isfinite(pred).all()
    mse = float(((pred - gt) ** 2).mean(axis=2).mean())
    tx = torch.from_numpy(np.ascontiguousarray(pred)).permute(2, 0, 1)[None].cuda() * 2 - 1
    ty = torch.from_numpy(np.ascontiguousarray(gt)).permute(2, 0, 1)[None].cuda() * 2 - 1
    return dict(mse=mse, psnr=metric_psnr(mse),
                ssim=float(ssim_map_rgb(pred, gt)[5:-5, 5:-5].mean()),
                lpips_alex=float(metric(tx, ty).item()))


def group_mean(rows, mode):
    return {k: float(np.mean([r[mode][k] for r in rows])) for k in KEYS}


def selection(scene):
    configs = {case: read(SOURCE/f'{scene}_{case}/config.json') for case in CASES}
    base = configs['sr_w01']
    for case, config in configs.items():
        for k in ['manifest_sha256', 'parent_sha256', 'initial_points', 'sh_degree',
                  'seed', 'prior_cameras', 'scheduler_offset']:
            assert config[k] == base[k], (scene, case, k)
        assert config['dense'] is False
    assert configs['lr_long']['teacher'] == 'none' and configs['lr_long']['weight'] == 0
    assert base['teacher'] == 'sr' and base['weight'] == .1
    assert configs['hr_oracle_w10']['teacher'] == 'hr' and configs['hr_oracle_w10']['weight'] == 1
    manifest = load_manifest(base['manifest'])
    assert file_sha(base['manifest']) == base['manifest_sha256']
    prior = sorted(base['prior_cameras'].split(','))
    train = sorted({o['camera_id'] for o in manifest['observations'] if o['split'] == 'train'})
    remaining = sorted(set(train)-set(prior))
    selected = [remaining[i] for i in [0, (len(remaining)-1)//2, len(remaining)-1]]
    assert len(set(selected)) == 3
    groups = {'teacher_camera_train': prior, 'lr_only_camera_train': selected, 'novel_cam00': ['cam00']}
    all_obs = {(o['camera_id'], o['frame_index']): o for o in manifest['observations']}
    inputs, observations = [], []
    root = Path(manifest['_root'])
    for group, cams in groups.items():
        for cam in cams:
            for frame in FRAMES:
                o = all_obs[(cam, frame)]
                assert o['split'] == ('test' if cam == 'cam00' else 'train')
                inputs.append(dict(group=group, camera_id=cam, frame_index=frame, split=o['split'],
                    lr_path=str(root/o['lr_path']), lr_sha256=file_sha(root/o['lr_path']),
                    hr_path=str(root/o['hr_path']), hr_sha256=file_sha(root/o['hr_path'])))
                observations.append(o)
    record = dict(scene=scene, groups=groups, frames=FRAMES, all_train_cameras=train,
        eligible_lr_only_cameras=remaining, inputs=inputs, manifest=base['manifest'],
        manifest_sha256=base['manifest_sha256'], common_parent_sha256=base['parent_sha256'],
        selection_rule='Prior cameras from original config; first, lower-median, last remaining train camera ID; frames 0/40/80/118; no HR/error selection.',
        source_configs={c: dict(path=str(SOURCE/f'{scene}_{c}/config.json'),
            sha256=file_sha(SOURCE/f'{scene}_{c}/config.json'), config=v) for c,v in configs.items()})
    return record, manifest, observations, configs


def old_references(scene, case, step):
    fit_path = SOURCE/f'{scene}_{case}/fit_{step}.json'
    fit = read(fit_path)
    ref = {(r['camera_id'], r['frame_index']): r['render_hr'] for r in fit['rows']}
    test_path = (SOURCE/f'{scene}_{case}_evaluation/metrics.json' if step == 18000 else
                 SOURCE/f'generalization_trajectory/{scene}_{case}_step{step}/metrics.json')
    test = read(test_path) if test_path.is_file() else None
    if test:
        ref.update({('cam00', r['frame_index']): r['spatial']['full'] for r in test['rows']})
    return ref, fit_path, test_path, test


@torch.no_grad()
def evaluate_checkpoint(scene, case, step, select, manifest, observations, config, metric):
    start = time.monotonic()
    path = SOURCE/f'{scene}_{case}/checkpoint_{step}.pt'
    ck_sha = file_sha(path)
    model, _, _, ck = load_checkpoint(path)
    model._deformation.eval()
    assert ck['metadata']['manifest_sha'] == select['manifest_sha256']
    assert ck['metadata']['parent_sha'] == select['common_parent_sha256']
    assert ck['metadata']['intervention_step'] == step
    assert len(model.get_xyz) == config['initial_points']
    refs, fit_path, test_path, old_test = old_references(scene, case, step)
    if old_test:
        assert old_test['checkpoint_sha256'] == ck_sha
        assert old_test['manifest_sha256'] == select['manifest_sha256']
    data = N3DVPreparedDataset(manifest, 'train', 'lr')
    data.observations = observations
    width, height = manifest['resolutions']['hr']
    root = Path(manifest['_root'])
    rows, checks = [], []
    for i, obs in enumerate(observations):
        item = data[i]
        cam_lr = observation_to_4dgs_camera(item, i)
        cam_hr = resized_camera(cam_lr, height, width)
        raw = render_image(model, cam_hr)['render']
        assert torch.isfinite(raw).all()
        pred_hr = image_array(raw)
        gt_hr, gt_lr = read_rgb(root/obs['hr_path']), read_rgb(root/obs['lr_path'])
        hr = scores(pred_hr, gt_hr, metric)
        # Training downsample includes its own clamp. Do not clamp HR first.
        closure = downsample(raw, item['image'].shape[-2:])
        lr = scores(image_array(closure), gt_lr, metric)
        native = render_image(model, cam_lr)['render']
        native_mse = float((native.clamp(0, 1)-item['image'].cuda()).square().mean())
        identity = (obs['camera_id'], obs['frame_index'])
        if identity in refs:
            delta = {k: abs(hr[k]-refs[identity][k]) for k in KEYS}
            # Different GPUs introduce small LPIPS numeric deviations; PSNR and
            # SSIM must remain tight. Report every deviation rather than hide it.
            passed = delta['psnr'] < 2e-5 and delta['ssim'] < 2e-6 and delta['lpips_alex'] < 1e-4
            checks.append(dict(camera_id=identity[0], frame_index=identity[1], delta=delta, passed=passed))
            assert passed, (scene, case, step, identity, delta)
        rows.append(dict(group=select['inputs'][i]['group'], camera_id=identity[0], frame_index=identity[1],
            hr=hr, lr_integrated=lr, lr_native=dict(mse=native_mse, psnr=metric_psnr(native_mse))))
    groups = {}
    for group, cams in select['groups'].items():
        subset = [r for r in rows if r['group'] == group]
        assert len(subset) == len(cams)*len(FRAMES)
        groups[group] = dict(count=len(subset),
            **{mode: group_mean(subset, mode) for mode in ['hr', 'lr_integrated']},
            lr_native_psnr=float(np.mean([r['lr_native']['psnr'] for r in subset])),
            per_camera={cam: {mode: group_mean([r for r in subset if r['camera_id']==cam], mode)
                              for mode in ['hr', 'lr_integrated']} for cam in cams})
    result = dict(scene=scene, case=case, step=step, checkpoint=str(path), checkpoint_sha256=ck_sha,
        metadata=ck['metadata'], point_count=len(model.get_xyz), rows=rows, groups=groups,
        earlier_metric_checks=checks, earlier_fit=dict(path=str(fit_path), sha256=file_sha(fit_path)),
        earlier_cam00_full60=(dict(path=str(test_path), sha256=file_sha(test_path),
            aggregate=old_test['aggregate']['full'], count=len(old_test['rows'])) if old_test else None),
        elapsed_seconds=time.monotonic()-start)
    del model, ck, data
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    started = time.monotonic()
    DEST.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    selections = {scene: selection(scene) for scene in SCENES}
    protocol = dict(frames=FRAMES, cases=CASES, steps=STEPS,
        statement='Evaluation only; fixed observations; no parameter updates or checkpoint selection.',
        lr_definition='D(Render_HR) with training bicubic antialias downsample; native-LR PSNR separate.',
        hr_definition='HR target evaluation; hr_oracle_w10 is LR + four-camera HR weight1 from LR parent, not full HR-only training.',
        metric_definition='Mean per-image PSNR/SSIM/LPIPS Alex. Float render, clamped [0,1]; SSIM 11x11 Gaussian sigma1.5, 5px boundary excluded. LPIPS normalize [-1,1].',
        source_hashes={str(p): file_sha(p) for p in [Path(__file__),OLD/'common.py',OLD/'evaluate.py',
            OLD/'n3dv_data.py',UPSTREAM/'gaussian_renderer/__init__.py']},
        environment=dict(host=platform.node(), torch=torch.__version__, numpy=np.__version__,
            gpu=torch.cuda.get_device_name(), cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES')),
        selections={scene: v[0] for scene,v in selections.items()})
    write_json(DEST/'protocol.json', protocol)
    import lpips
    metric = lpips.LPIPS(net='alex').cuda().eval().requires_grad_(False)
    result = dict(protocol_path=str(DEST/'protocol.json'), scenes={})
    state = dict(status='running', pid=os.getpid(), completed=[])
    write_json(DEST/'state.json', state)
    try:
        for scene, (select, manifest, observations, configs) in selections.items():
            result['scenes'][scene] = {}
            for case in CASES:
                result['scenes'][scene][case] = {}
                for step in STEPS:
                    value = evaluate_checkpoint(scene, case, step, select, manifest, observations, configs[case], metric)
                    write_json(DEST/scene/case/f'step_{step}.json', value)
                    result['scenes'][scene][case][str(step)] = value
                    state['completed'].append(f'{scene}/{case}/{step}')
                    state['elapsed_seconds'] = time.monotonic()-started
                    write_json(DEST/'state.json', state)
                    print(json.dumps(dict(completed=state['completed'][-1], elapsed_seconds=state['elapsed_seconds'])), flush=True)
        result['elapsed_seconds'] = time.monotonic()-started
        result['peak_gpu_gb'] = torch.cuda.max_memory_allocated()/1e9
        result['completed'] = True
        write_json(DEST/'metrics.json', result)
        state.update(status='complete', elapsed_seconds=result['elapsed_seconds'])
        write_json(DEST/'state.json', state)
        print(json.dumps(dict(completed=True, elapsed_seconds=result['elapsed_seconds'], output=str(DEST))), flush=True)
    except BaseException as e:
        state.update(status='failed', error=repr(e), elapsed_seconds=time.monotonic()-started)
        write_json(DEST/'state.json', state)
        raise


if __name__ == '__main__':
    main()
