#!/usr/bin/env python3
"""Evaluate one native STG 50-frame segment with the registered Wu metrics.

The model is STG only. Multi4D's audited evaluator supplies model-independent
metric, HR-mask, and fixed-ROI helpers. HR enters evaluation after rendering;
the STG camera constructor receives calibration/time and zero RGB placeholders.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import sys
import time
import traceback
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SHARED = ROOT / 'experiments/dynamic_sr_multi4d_area_20261017/evaluate.py'
FIXED_FRAMES = (0, 150, 299)
PREVIEW_FRAMES = (0, 40, 49, 50, 99, 100, 149, 150, 199, 200, 249, 250, 299)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def identity(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': sha(path)}


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def path_value(value):
    value = value['path'] if isinstance(value, dict) else value
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    sys.modules[name] = obj
    spec.loader.exec_module(obj)
    return obj


def segment_bounds(spec):
    start = int(spec.get('segment_start', spec.get('start_frame', 0)))
    duration = int(spec.get('duration', 50))
    if duration != 50 or start not in range(0, 300, 50):
        raise ValueError('Only registered disjoint50 segments within frames0..299 are supported')
    return start, duration


def cached_mask(shared, manifest_path, manifest, camera, helpers, cache_root):
    """Cache one full300 reference-only mask, with provenance and byte hashes."""
    import fcntl
    import numpy as np
    key = sha(manifest_path) + '-' + sha(SHARED)[:16]
    directory = cache_root / key
    directory.mkdir(parents=True, exist_ok=True)
    path, receipt_path = directory / f'{camera}.npz', directory / f'{camera}.json'
    with (directory / f'{camera}.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if path.exists() and receipt_path.exists():
            receipt = read(receipt_path)
            if (receipt['manifest_sha256'] != sha(manifest_path)
                    or receipt['helper_sha256'] != sha(SHARED) or receipt['mask']['sha256'] != sha(path)):
                raise ValueError('Reference-mask cache identity mismatch')
            with np.load(path) as value:
                mask = value['mask'].copy()
            if mask.dtype != np.bool_ or mask.shape != tuple(manifest['resolutions']['hr'][::-1]):
                raise ValueError('Reference-mask cache shape/dtype mismatch')
            return mask, {**identity(receipt_path), 'mask_pixel_sha256': hashlib.sha256(mask.tobytes()).hexdigest()}
        mask = shared.fixed_mask(manifest, manifest_path.parent, camera, helpers)
        temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
        with temporary.open('wb') as stream:
            np.savez_compressed(stream, mask=mask)
        temporary.replace(path)
        write(receipt_path, {'schema': 'stg_reference_mask_v1', 'camera': camera,
                            'manifest_sha256': sha(manifest_path), 'helper_sha256': sha(SHARED),
                            'mask': identity(path), 'full_reference_frames': 300,
                            'source_HR_content_hashes_verified_during_creation': True,
                            'evaluation_only': True})
        return mask, {**identity(receipt_path), 'mask_pixel_sha256': hashlib.sha256(mask.tobytes()).hexdigest()}


def load_native(spec, checkpoint):
    """Load native STG PLY and its same-stem decoder, without optimizer creation."""
    import torch
    upstream = path_value(spec['upstream'])
    if spec.get('adapter'):
        adapter = path_value(spec['adapter'])
        provenance = read(adapter / 'identity.json')
        for name, expected in provenance['original_sha256'].items():
            if sha(upstream / name) != expected:
                raise ValueError(f'Pinned upstream changed before evaluation: {name}')
        for name, expected in provenance['generated_sha256'].items():
            if sha(adapter / name) != expected:
                raise ValueError(f'Generated adapter changed before evaluation: {name}')
    paths = [path_value(spec['adapter'])] if spec.get('adapter') else []
    paths += [upstream, upstream / 'thirdparty/gaussian_splatting']
    for path in reversed(paths):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from helper_train import getmodel, getrenderpip, trbfunction
    model_name = spec.get('model', 'ours_full')
    rgbfunction = spec.get('rgbfunction', 'sandwich')
    rdpip = spec.get('rdpip', 'train_ours_full')
    model = getmodel(model_name)(int(spec.get('sh_degree', 3)), rgbfunction)
    if not checkpoint.with_suffix('.pt').is_file():
        raise FileNotFoundError('STG PLY requires the same-stem .pt RGB decoder')
    model.load_ply(str(checkpoint))
    model.rgbdecoder.cuda().eval().requires_grad_(False)
    render, settings, rasterizer = getrenderpip(rdpip)
    pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
    background = torch.zeros(9, dtype=torch.float32, device='cuda')
    return model, render, settings, rasterizer, trbfunction, pipe, background


def evaluate(args):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    shared = module('stg_shared_evaluation', SHARED)
    spec = read(args.spec)
    start, duration = segment_bounds(spec)
    method = args.method or spec.get('method')
    if method not in ('areaL0', 'areaTm', 'areaOracle'):
        raise ValueError('A registered method arm is required')
    manifest_path = path_value(spec['manifest'])
    manifest, all_tests = shared.validate_manifest(manifest_path)
    if spec.get('manifest_sha256') and spec['manifest_sha256'] != sha(manifest_path):
        raise ValueError('Spec/manifest SHA mismatch')
    tests = [r for r in all_tests if start <= int(r['frame_index']) < start + duration]
    if args.frames is not None:
        if not set(args.frames) <= {r['frame_index'] for r in tests}:
            raise ValueError('Smoke frames must belong to this registered segment')
        tests = [r for r in tests if int(r['frame_index']) in set(args.frames)]
    cameras = manifest['splits']['train']
    training = sorted((r for r in manifest['observations'] if r['split'] == 'train'
                       and int(r['frame_index']) in FIXED_FRAMES
                       and start <= int(r['frame_index']) < start + duration),
                      key=lambda r: (r['camera_id'], r['frame_index']))
    if args.test_only:
        training = []
    helpers = shared.metric_helpers()
    regions, roi_info = shared.roi_protocol(spec, args, manifest_path, manifest)
    fft = module('stg_eval_frequency', shared.FFT)
    active_cameras = sorted({'cam00', *(r['camera_id'] for r in training)})
    masks, mask_identities = {}, {}
    for camera in active_cameras:
        masks[camera], mask_identities[camera] = cached_mask(shared, manifest_path, manifest, camera,
                                                           helpers, args.mask_cache)
    args.out.mkdir(parents=True, exist_ok=True)
    preview = args.out / 'previews'
    preview.mkdir(exist_ok=True)
    torch.set_num_threads(args.cpu_threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    model, render, settings, rasterizer, rbf, pipe, background = load_native(spec, args.checkpoint)
    width, height = manifest['resolutions']['hr']
    model.ts = torch.ones(1, 1, height, width, device='cuda')
    data = module('stg_area_eval_data', HERE / 'data.py')
    metric = None
    if not args.skip_lpips:
        import lpips
        metric = lpips.LPIPS(net='alex', version='0.1', spatial=False, pnet_rand=False,
                            pretrained=True, verbose=False).to(args.lpips_device).eval().requires_grad_(False)
    receipt = read(args.training_receipt) if args.training_receipt else {}
    training_complete = (receipt.get('status') == 'completed'
                         and receipt.get('formal_checkpoint') is True
                         and receipt.get('smoke_only') is False)
    if args.training_receipt:
        # A receipt from another segment/arm must never certify this evaluation.
        for key, expected in (('method', method), ('segment_start', start), ('duration', duration)):
            if receipt.get(key) != expected:
                raise ValueError(f'Training receipt identity mismatch: {key}')
        for key, path in (('checkpoint', args.checkpoint), ('decoder', args.checkpoint.with_suffix('.pt')),
                          ('manifest', manifest_path)):
            if receipt.get(key, {}).get('sha256') != sha(path):
                raise ValueError(f'Training receipt artifact identity mismatch: {key}')
        if receipt.get('metadata', {}).get('spec_sha256') != sha(args.spec):
            raise ValueError('Evaluation spec differs from the training receipt')
    base = dict(schema='stg_area_segment_evaluation_v1', model='STG', method=method,
        status='running', scene=manifest['scene'], spec=identity(args.spec), manifest=identity(manifest_path),
        checkpoint=identity(args.checkpoint), decoder=identity(args.checkpoint.with_suffix('.pt')),
        training_receipt=identity(args.training_receipt) if args.training_receipt else None,
        checkpoint_training_status='completed' if training_complete else 'unverified',
        segment_start=start, duration=duration, posthoc_cull=False, native_segmented_model=True,
        global_time='frame_index/300', local_time='(frame_index-segment_start)/50',
        metric_sources=[identity(shared.METRIC), identity(shared.FFT), identity(SHARED), identity(__file__)],
        data_source=identity(HERE / 'data.py'), mask_receipts=mask_identities, roi_protocol=roi_info,
        spatial_definition='Identical Wu spatial helper; unclipped raw cached, metrics clamp[0,1]; AlexNet-v0.1 LPIPS.',
        temporal_definition='Consecutive prediction frame difference minus reference frame difference, no optical flow.',
        information_boundary='Rendering receives calibration/time/zero placeholders. HR/LR/masks are evaluation only.',
        host=socket.gethostname(), gpu=torch.cuda.get_device_name(),
        capacity_points=int(model.get_xyz.shape[0]),
        decoder_parameters=sum(p.numel() for p in model.rgbdecoder.parameters()),
        checkpoint_bytes=args.checkpoint.stat().st_size + args.checkpoint.with_suffix('.pt').stat().st_size,
        lpips_status='explicitly_skipped_smoke_only' if args.skip_lpips else 'alex_v0.1',
        test_only_engineering_smoke=bool(args.test_only),
        training_cameras=cameras, training_fixed_frames=[f for f in FIXED_FRAMES if start <= f < start + duration])
    write(args.out / 'run.json', base)
    rows, temporal, previous, endpoints = [], [], None, {}
    began = time.monotonic()
    with (args.out / 'per_observation.jsonl').open('w') as log:
        for scope, record in [('test', r) for r in tests] + [('train_diagnostic', r) for r in training]:
            frame, camera_id = int(record['frame_index']), record['camera_id']
            with torch.inference_mode():
                camera = data.make_camera(manifest, record, start, duration)
                package = render(camera, model, pipe, background, scaling_modifier=1.,
                                 basicfunction=rbf, GRsetting=settings, GRzer=rasterizer)
                raw = package['render'].detach()
            if tuple(raw.shape) != (3, height, width) or not bool(torch.isfinite(raw).all()):
                raise ValueError('Invalid native STG render')
            pred = helpers.image_array(raw)
            gt = shared.checked_rgb(manifest_path.parent, record, 'hr', helpers)
            lr = shared.checked_rgb(manifest_path.parent, record, 'lr', helpers)
            spatial = helpers.spatial_metrics(pred, gt, masks[camera_id], metric, args.lpips_device)
            full = spatial['full']
            projected = F.avg_pool2d(raw.float()[None], 4)[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
            row = dict(scope=scope, camera=camera_id, frame=frame,
                       psnr=full['psnr'], ssim=full['ssim'], lpips=full.get('lpips_alex'), mse=full['mse'],
                       LR_diagnostic_psnr=helpers.metric_psnr(float(((projected - lr) ** 2).mean())),
                       regions=spatial, fixed_rois={})
            if scope == 'test':
                bands = fft.decompose(pred, gt)
                row.update({k: bands[k] for k in ('low_mse', 'mid_mse', 'high_mse', 'low_psnr', 'mid_psnr', 'high_psnr')})
                if previous is not None and frame == previous[0] + 1:
                    delta = (pred - previous[1]) - (gt - previous[2])
                    temporal.append(dict(previous_frame=previous[0], frame=frame,
                                         frame_difference_residual_l1=float(np.abs(delta).mean()),
                                         frame_difference_residual_mse=float((delta ** 2).mean()), boundary=False))
                previous = (frame, pred, gt)
            for name, (x0, y0, x1, y1) in regions.get(camera_id, {}).items():
                roi_mask = np.zeros((height, width), bool)
                roi_mask[y0:y1, x0:x1] = True
                row['fixed_rois'][name] = helpers.spatial_metrics(pred, gt, roi_mask, None)['dynamic']
                if frame in PREVIEW_FRAMES:
                    for label, image in (('render', pred), ('HR', gt)):
                        Image.fromarray(np.rint(image[y0:y1, x0:x1].clip(0, 1) * 255).astype(np.uint8)).save(
                            preview / f'{camera_id}_{frame:04d}_{name}_{label}.png')
            if frame in PREVIEW_FRAMES or (scope == 'test' and frame in (start, start + duration - 1)):
                stem = f'{camera_id}_{frame:04d}'
                Image.fromarray(np.rint(pred.clip(0, 1) * 255).astype(np.uint8)).save(preview / f'{stem}.png')
                raw_path = preview / f'{stem}.npz'
                np.savez_compressed(raw_path, rgb_raw=raw.cpu().numpy().astype(np.float32))
                if scope == 'test' and frame in (start, start + duration - 1):
                    endpoints[str(frame)] = {**identity(raw_path), 'frame': frame, 'camera': camera_id,
                                             'reference_hr_sha256': record['hr_sha256']}
            rows.append(row)
            log.write(json.dumps(row, allow_nan=False) + '\n')
            log.flush()
    test_rows = [r for r in rows if r['scope'] == 'test']
    train_rows = [r for r in rows if r['scope'] == 'train_diagnostic']
    write(args.out / 'temporal_per_pair.json', temporal)
    complete = (len(test_rows) == duration and len(temporal) == duration - 1
                and not args.skip_lpips and training_complete and not args.test_only)
    result = dict(base, status='completed' if complete else 'incomplete', frames=len(test_rows),
                  frame_indices=[r['frame'] for r in test_rows], test=shared.aggregate(test_rows),
                  train_diagnostic=shared.aggregate(train_rows),
                  per_observation=identity(args.out / 'per_observation.jsonl'),
                  temporal_per_pair=identity(args.out / 'temporal_per_pair.json'), endpoints=endpoints,
                  temporal=dict(pairs=len(temporal), expected_pairs=duration - 1,
                                frame_difference_residual_l1=shared.mean_metric(temporal, 'frame_difference_residual_l1'),
                                frame_difference_residual_mse=shared.mean_metric(temporal, 'frame_difference_residual_mse')),
                  region_aggregate={name: {key: shared.mean_metric([r['regions'][name] for r in test_rows], key)
                                          for key in ('psnr', 'ssim', 'mse', 'lpips_alex_spatial_mask')}
                                    for name in ('dynamic', 'static')},
                  seconds=time.monotonic() - began, peak_GB=torch.cuda.max_memory_allocated() / 1e9,
                  visual_acceptance='not_performed; fixed frame40 boxes do not guarantee every frame contains a complete face or readable text')
    write(args.out / 'summary.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True, help='Native point_cloud.ply; same-stem .pt decoder required')
    parser.add_argument('--training-receipt', type=Path)
    parser.add_argument('--method', choices=('areaL0', 'areaTm', 'areaOracle'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--frames', type=int, nargs='+')
    parser.add_argument('--skip-lpips', action='store_true')
    parser.add_argument('--test-only', action='store_true',
                        help='Engineering smoke only: skip training-view diagnostics; always incomplete')
    parser.add_argument('--lpips-device', default='cuda')
    parser.add_argument('--cpu-threads', type=int, default=4)
    parser.add_argument('--roi-protocol', type=Path)
    parser.add_argument('--mask-cache', type=Path, default=ROOT / 'output/dynamic_sr_stg_area_20261018/mask_cache')
    args = parser.parse_args()
    if (args.out / 'summary.json').exists():
        raise FileExistsError('Preserve prior evaluation; use a fresh output directory')
    try:
        result = evaluate(args)
        print(json.dumps({'status': result['status'], 'frames': result['frames'], 'summary': str(args.out / 'summary.json')}))
    except BaseException as error:
        write(args.out / 'summary.json', {'schema': 'stg_area_segment_evaluation_v1', 'status': 'failed',
                                         'error': str(error), 'error_type': type(error).__name__,
                                         'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
