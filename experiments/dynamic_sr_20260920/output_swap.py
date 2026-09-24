#!/usr/bin/env python3
"""Evaluation-only swaps of final deformed Gaussian support and SH outputs.

No fitting, optimization, or HR-driven selection. Off-trajectory mixtures are
diagnostics of endpoint output dependence, not deployable method results.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image
import torch

PROJECT = Path(__file__).resolve().parents[2]
OLD = PROJECT / 'experiments/dynamic_sr_20260918'
sys.path.insert(0, str(OLD))
from common import (UPSTREAM, downsample, load_checkpoint, render_image,
                    resized_camera, image_tensor)
from n3dv_data import N3DVPreparedDataset, load_manifest, observation_to_4dgs_camera
from evaluate import (aggregate, file_sha, full_flow, image_array, metric_psnr,
                      prepare_cache, read_rgb, spatial_metrics, temporal_metrics,
                      write_json, write_rgb)
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

CASES = {'G6C6': (6000, 6000), 'G6C18': (6000, 18000),
         'G18C6': (18000, 6000), 'G18C18': (18000, 18000)}
SCENES = ['cook_spinach', 'cut_roasted_beef', 'meetroom_discussion']
ORIGINAL = PROJECT / 'output/dynamic_sr_20260919'


@torch.no_grad()
def final_outputs(model, time_value):
    """Each endpoint supplies its own complete fine deformation forward pass."""
    xyz = model.get_xyz
    times = torch.tensor(time_value).to(xyz.device).repeat(xyz.shape[0], 1)
    pos, scale, rot, opacity, sh = model._deformation(
        xyz, model._scaling, model._rotation, model._opacity,
        model.get_features, times)
    return (pos, model.scaling_activation(scale), model.rotation_activation(rot),
            model.opacity_activation(opacity), sh)


@torch.no_grad()
def render_outputs(geometry, colors, camera, sh_degree):
    """Mirror common.render_image fine path with final-attribute substitution.

    G supplies xyz, activated scale/rotation/opacity; C supplies final SH.
    The rasterizer evaluates these SH coefficients in the direction from the
    camera to G's xyz. Therefore changing G can change RGB through direction,
    in addition to projected footprints, visibility, ordering, and opacity.
    """
    settings = GaussianRasterizationSettings(
        image_height=int(camera.image_height), image_width=int(camera.image_width),
        tanfovx=math.tan(camera.FoVx * .5), tanfovy=math.tan(camera.FoVy * .5),
        bg=torch.zeros(3, device='cuda'), scale_modifier=1.,
        viewmatrix=camera.world_view_transform.cuda(),
        projmatrix=camera.full_proj_transform.cuda(), sh_degree=sh_degree,
        campos=camera.camera_center.cuda(), prefiltered=False, debug=False)
    rasterizer = GaussianRasterizer(raster_settings=settings)
    result, _, _ = rasterizer(
        means3D=geometry[0], means2D=torch.zeros_like(geometry[0]),
        shs=colors[4], colors_precomp=None, opacities=geometry[3],
        scales=geometry[1], rotations=geometry[2], cov3D_precomp=None)
    return result


def max_numeric_delta(a, b, prefix=''):
    """All shared numeric leaves, allowing older reports' extra metadata."""
    rows = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in a.keys() & b.keys():
            rows.extend(max_numeric_delta(a[k], b[k], prefix + '/' + k))
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        rows.append({'field': prefix, 'abs_delta': abs(float(a) - float(b))})
    return rows


@torch.no_grad()
def parity_check(models, manifest, prior_cameras):
    data = N3DVPreparedDataset(manifest, 'train', 'lr')
    all_obs = manifest['observations']
    selections = [('cam00', 0), (prior_cameras[0], 40), (prior_cameras[1], 118)]
    data.observations = [next(o for o in all_obs
                             if (o['camera_id'], o['frame_index']) == s)
                         for s in selections]
    w, h = manifest['resolutions']['hr']
    result = []
    for i in range(len(data)):
        item = data[i]
        camera = resized_camera(observation_to_4dgs_camera(item, i), h, w)
        for step, model in models.items():
            outputs = final_outputs(model, camera.time)
            candidate = render_outputs(outputs, outputs, camera, model.active_sh_degree)
            expected = render_image(model, camera)['render']
            delta = float((candidate - expected).abs().max())
            result.append(dict(step=step, camera_id=item['camera_id'],
                               frame_index=item['frame_index'], max_abs_rgb_delta=delta))
            if not math.isfinite(delta) or delta > 1e-6:
                raise AssertionError(f'Endpoint renderer parity failed: {result[-1]}')
    return result


@torch.no_grad()
def evaluate_scene(scene, root, metric):
    begin = time.monotonic()
    out = root / scene
    out.mkdir(exist_ok=False)
    run = ORIGINAL / f'{scene}_sr_w01'
    config = json.loads((run / 'config.json').read_text())
    assert config['dense'] is False and config['teacher'] == 'sr' and config['weight'] == .1
    manifest = load_manifest(config['manifest'])
    prior_cameras = config['prior_cameras'].split(',')
    manifest_hash = file_sha(config['manifest'])
    assert manifest_hash == config['manifest_sha256']
    models, checkpoints, checkpoint_info = {}, {}, {}
    for step in [6000, 18000]:
        path = run / f'checkpoint_{step}.pt'
        model, hidden, _, ck = load_checkpoint(path)
        model._deformation.eval()
        md = ck['metadata']
        assert md['intervention_step'] == step and not md['args']['dense']
        assert md['manifest_sha'] == manifest_hash
        assert md['parent_sha'] == config['parent_sha256']
        models[step], checkpoints[step] = model, ck
        checkpoint_info[str(step)] = dict(path=str(path), sha256=file_sha(path),
            metadata=md, count=len(model.get_xyz), active_sh_degree=model.active_sh_degree)
    a, b = models.values()
    assert len(a.get_xyz) == len(b.get_xyz) == config['initial_points']
    assert a.active_sh_degree == b.active_sh_degree == config['sh_degree']
    assert a.max_sh_degree == b.max_sh_degree
    assert a.get_features.shape == b.get_features.shape
    assert checkpoints[6000]['hidden'] == checkpoints[18000]['hidden']
    parity = parity_check(models, manifest, prior_cameras)
    # Optimizer state is loaded for faithful checkpoint restoration, never stepped.
    del checkpoints
    observations = sorted([o for o in manifest['observations'] if o['split'] == 'test'],
                          key=lambda o: o['frame_index'])
    assert len(observations) == 60 and {o['camera_id'] for o in observations} == {'cam00'}
    cache, cache_info = prepare_cache(manifest, observations,
        SimpleNamespace(dynamic_threshold=.025, flow_scale=.5, cache_dir=None))
    dynamic = np.asarray(Image.open(cache / 'dynamic_mask.png')) > 0
    bbox = cache_info['dynamic_bbox_xyxy']
    info = dict(scene=scene, case_definitions=CASES, checkpoints=checkpoint_info,
        source_config=str(run/'config.json'), source_config_sha256=file_sha(run/'config.json'),
        manifest=manifest['_manifest_path'], manifest_sha256=manifest_hash,
        script_sha256=file_sha(__file__), upstream_renderer_sha256=file_sha(UPSTREAM/'gaussian_renderer/__init__.py'),
        common_sha256=file_sha(OLD/'common.py'), evaluator_sha256=file_sha(OLD/'evaluate.py'),
        gpu=torch.cuda.get_device_name(), cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        versions=dict(torch=torch.__version__, numpy=np.__version__, opencv=cv2.__version__),
        renderer_endpoint_parity=parity, parity_threshold=1e-6,
        point_identity_basis='Same non-densifying run, identical initial point count and SH shape; controlled_fit has no prune/reorder outside dense branch.',
        information_boundary='Entire script is evaluation-only. HR only enters image metrics and pre-existing GT mask/flow cache; no trainable parameter is updated.',
        geometry_definition='Final deformed xyz plus activated scale, normalized rotation and activated opacity. Includes canonical and shared-network effects; not pure motion.',
        color_definition='Final deformed SH coefficients computed with color checkpoint own xyz/features/trunk. Rasterizer evaluates at geometry checkpoint xyz-camera direction.',
        interpretation='Off-trajectory mixtures probe endpoint dependence. A bad mixture does not prove its component is inaccurate: co-adaptation, SH direction, occlusion and support-color compatibility confound causal attribution.',
        evaluation_cache=str(cache), cache_key=cache_info['cache_key'],
        dynamic_fraction=float(dynamic.mean()), dynamic_bbox_xyxy=bbox,
        roi_definition=cache_info['mask_definition'], prior_cameras=prior_cameras,
        train_frames=[0, 40, 80, 118])
    write_json(out/'metadata.json', info)
    data = N3DVPreparedDataset(manifest, 'test', 'lr')
    data.observations = observations
    w, h = manifest['resolutions']['hr']
    rows = {name: [] for name in CASES}
    previous_pred, previous_gt = {}, None
    for i, obs in enumerate(observations):
        item = data[i]
        cam = resized_camera(observation_to_4dgs_camera(item, i), h, w)
        outputs = {step: final_outputs(model, cam.time) for step, model in models.items()}
        gt = read_rgb(Path(manifest['_root'])/obs['hr_path'])
        lr = item['image'].cuda()
        if i:
            pair = cache_info['flow_pairs'][i-1]
            with np.load(cache/pair['file'], allow_pickle=False) as f:
                flow, valid = full_flow(f['backward'], f['valid'], h, w)
        for name, (gs, cs) in CASES.items():
            raw = render_outputs(outputs[gs], outputs[cs], cam, a.active_sh_degree)
            assert torch.isfinite(raw).all()
            pred = image_array(raw)
            lr_mse = float((downsample(raw, lr.shape[-2:])-lr).square().mean())
            row = dict(frame_index=obs['frame_index'], time=obs['time'],
                spatial=spatial_metrics(pred, gt, dynamic, metric, 'cuda', bbox),
                closure={'full': dict(mse=lr_mse, psnr=metric_psnr(lr_mse))})
            if i:
                row['temporal'] = temporal_metrics(previous_pred[name], pred,
                    previous_gt, gt, flow, valid, dynamic)
            write_rgb(out/name/'predictions'/f'{obs["frame_index"]:04d}.png', pred)
            rows[name].append(row)
            previous_pred[name] = pred
        previous_gt = gt
        if (i+1) % 20 == 0:
            print(f'{scene} TEST {i+1}/60 four output swaps', flush=True)
    data = N3DVPreparedDataset(manifest, 'train', 'lr')
    data.observations = [o for o in data.observations
        if o['camera_id'] in prior_cameras and o['frame_index'] in [0, 40, 80, 118]]
    assert len(data) == 16
    train = {name: [] for name in CASES}
    zero_mask = np.zeros((h, w), bool)
    for i, obs in enumerate(data.observations):
        item = data[i]
        cam = resized_camera(observation_to_4dgs_camera(item, i), h, w)
        outputs = {step: final_outputs(model, cam.time) for step, model in models.items()}
        hr = read_rgb(Path(manifest['_root'])/obs['hr_path'])
        prior = read_rgb(Path(manifest['_root'])/'sr_swinir_x4'/obs['camera_id']/Path(obs['lr_path']).name)
        lr = item['image'].cuda()
        for name, (gs, cs) in CASES.items():
            raw = render_outputs(outputs[gs], outputs[cs], cam, a.active_sh_degree)
            pred = image_array(raw)
            lr_mse = float((downsample(raw, lr.shape[-2:])-lr).square().mean())
            row = dict(camera_id=obs['camera_id'], frame_index=obs['frame_index'],
                lr_psnr=metric_psnr(lr_mse),
                render_hr=spatial_metrics(pred, hr, zero_mask, metric)['full'],
                render_prior=spatial_metrics(pred, prior, zero_mask, metric)['full'])
            train[name].append(row)
            write_rgb(out/name/'train_predictions'/obs['camera_id']/f'{obs["frame_index"]:04d}.png', pred)
    results = {}
    comparisons = {}
    for name, (gs, cs) in CASES.items():
        train_aggregate = {kind: {key: float(np.mean([r[kind][key] for r in train[name]]))
             for key in ['psnr', 'ssim', 'lpips_alex', 'mse']} for kind in ['render_hr', 'render_prior']}
        result = dict(case=name, geometry_step=gs, color_step=cs,
            rows=rows[name], aggregate=aggregate(rows[name], 'spatial'),
            temporal_aggregate=aggregate(rows[name], 'temporal'),
            lr_closure_aggregate=aggregate(rows[name], 'closure'),
            train_rows=train[name], train_aggregate=train_aggregate,
            train_lr_psnr=float(np.mean([r['lr_psnr'] for r in train[name]])))
        if gs == cs:
            old = (ORIGINAL/f'{scene}_sr_w01_evaluation/metrics.json' if gs == 18000
                else ORIGINAL/f'generalization_trajectory/{scene}_sr_w01_step6000/metrics.json')
            comparison = dict(previous_evaluation=str(old), available=old.is_file())
            if old.is_file():
                prior = json.loads(old.read_text())
                deltas = max_numeric_delta(result['aggregate'], prior['aggregate'])
                temporal_deltas = max_numeric_delta(result['temporal_aggregate'], prior['temporal_aggregate'])
                comparison.update(max_aggregate_numeric_delta=max(x['abs_delta'] for x in deltas),
                    max_temporal_numeric_delta=max(x['abs_delta'] for x in temporal_deltas),
                    shared_aggregate_fields=deltas)
                # Metrics should reproduce, allowing floating-point CPU/GPU
                # differences much smaller than a meaningful effect size.
                assert comparison['max_aggregate_numeric_delta'] < 1e-4, comparison
                assert comparison['max_temporal_numeric_delta'] < 1e-6, comparison
            old_fit = json.loads((run/f'fit_{gs}.json').read_text())
            fit_deltas = max_numeric_delta(train_aggregate, old_fit['aggregate'])
            comparison['max_train_fit_numeric_delta'] = max(x['abs_delta'] for x in fit_deltas)
            comparison['train_lr_psnr_delta'] = abs(result['train_lr_psnr']-old_fit['lr_psnr'])
            assert comparison['max_train_fit_numeric_delta'] < 1e-4, comparison
            assert comparison['train_lr_psnr_delta'] < 1e-4, comparison
            comparisons[name] = comparison
        write_json(out/name/'metrics.json', result)
        results[name] = {k: result[k] for k in ['aggregate', 'temporal_aggregate',
            'lr_closure_aggregate', 'train_aggregate', 'train_lr_psnr']}
    write_json(out/'metrics.json', dict(**info, cases=results,
        endpoint_previous_metric_comparison=comparisons, elapsed_seconds=time.monotonic()-begin))
    integrity = {name: dict(test_png_count=len(list((out/name/'predictions').glob('*.png'))),
        train_png_count=len(list((out/name/'train_predictions').glob('*/*.png'))),
        test_rows=len(rows[name]), temporal_pairs=sum('temporal' in r for r in rows[name]),
        train_rows=len(train[name]), metrics_sha256=file_sha(out/name/'metrics.json')) for name in CASES}
    assert all(v['test_png_count']==60 and v['train_png_count']==16 and
        v['temporal_pairs']==59 for v in integrity.values())
    write_json(out/'complete.json', dict(status='complete', metrics_sha256=file_sha(out/'metrics.json'),
        integrity=integrity, renderer_parity_passed=True, previous_metrics_agree=True,
        elapsed_seconds=time.monotonic()-begin))
    print(json.dumps(dict(scene=scene, status='complete', elapsed_seconds=time.monotonic()-begin,
        cases={name: {key: r['aggregate']['full'][key] for key in
            ['psnr_mean','ssim_mean','lpips_alex_mean']} for name, r in results.items()})), flush=True)
    return dict(scene=scene, path=str(out/'metrics.json'), cases=results,
        elapsed_seconds=time.monotonic()-begin)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scenes', nargs='+', choices=SCENES, default=SCENES)
    p.add_argument('--out', type=Path, default=PROJECT/'output/dynamic_sr_20260920/output_swap')
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
    for scene in args.scenes:
        results.append(evaluate_scene(scene, args.out, metric))
        gc.collect()
        torch.cuda.empty_cache()
    write_json(args.out/'metrics.json', dict(scenes=results,
        script_sha256=file_sha(__file__), elapsed_seconds=time.monotonic()-begin))
    write_json(args.out/'complete.json', dict(status='complete', scenes=args.scenes,
        metrics_sha256=file_sha(args.out/'metrics.json'), elapsed_seconds=time.monotonic()-begin))
    print(f'OUTPUT_SWAP_COMPLETE {args.out}', flush=True)


if __name__ == '__main__':
    main()
