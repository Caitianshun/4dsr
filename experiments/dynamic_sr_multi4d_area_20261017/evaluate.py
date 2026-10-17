#!/usr/bin/env python3
"""Native Multi4D evaluation on the registered area4 full-time manifest.

No optimizer steps. The standard spatial metrics are extracted unchanged from
the same helper used by current Wu evaluations. HR/LR references enter only
after rendering; region masks are fixed from HR before checkpoint loading.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import socket
import sys
import time
import traceback
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OLD = ROOT / 'experiments/dynamic_sr_multi4d_20260924'
METRIC = ROOT / 'experiments/dynamic_sr_20260918/evaluate.py'
FFT = ROOT / 'experiments/dynamic_sr_same_observation_20260930/frequency.py'
FIXED_FRAMES = (0, 150, 299)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def path_value(value):
    value = value['path'] if isinstance(value, dict) else value
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def identity(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=sha(path))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temp.replace(path)


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    sys.modules[name] = obj
    spec.loader.exec_module(obj)
    return obj


def metric_helpers():
    """Extract exactly the current spatial implementation without Wu imports."""
    import cv2
    import numpy as np
    import torch
    from PIL import Image
    names = ('read_rgb', 'image_array', 'ssim_map_rgb', 'masked_mean',
             'metric_psnr', 'spatial_metrics')
    nodes = [node for node in ast.parse(METRIC.read_text()).body
             if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in nodes} != set(names):
        raise ValueError('Current metric helper functions changed')
    scope = dict(np=np, cv2=cv2, torch=torch, Image=Image, Path=Path)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(METRIC), 'exec'), scope)
    return SimpleNamespace(**{name: scope[name] for name in names})


def validate_manifest(path):
    manifest = read(path)
    if manifest.get('scale') != 4 or manifest.get('splits', {}).get('test') != ['cam00']:
        raise ValueError('area4 scale4 and cam00-only held-out split required')
    if manifest.get('physical_lr_derivation', {}).get('formation') != 'area4':
        raise ValueError('A registered area4 derived manifest is required')
    rows = manifest['observations']
    keys = [(row['camera_id'], int(row['frame_index'])) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate camera/frame record')
    tests = sorted((row for row in rows if row['split'] == 'test'), key=lambda row: int(row['frame_index']))
    if [(row['camera_id'], int(row['frame_index'])) for row in tests] != [('cam00', i) for i in range(300)]:
        raise ValueError('All 300 cam00 frames are required')
    for row in rows:
        if abs(float(row['time']) - int(row['frame_index']) / 300.) > 1e-7:
            raise ValueError('Time must equal frame_index/300')
        for field in ('hr_path', 'lr_path'):
            image_path = (path.parent / row[field]).resolve()
            if not image_path.is_relative_to(path.parent.resolve()):
                raise ValueError('Image path escapes manifest directory')
    return manifest, tests


def checked_rgb(root, row, kind, helpers):
    path = root / row[kind + '_path']
    if sha(path) != row[kind + '_sha256']:
        raise ValueError(f'{kind} image identity mismatch: {path}')
    return helpers.read_rgb(path)


def fixed_mask(manifest, root, camera, helpers):
    """Model-independent full-window temporal-variation proxy, never semantic."""
    import cv2
    import numpy as np
    rows = sorted((r for r in manifest['observations'] if r['camera_id'] == camera),
                  key=lambda r: int(r['frame_index']))
    mean = m2 = None
    for count, row in enumerate(rows, 1):
        image = checked_rgb(root, row, 'hr', helpers).astype(np.float64)
        if mean is None:
            mean, m2 = np.zeros_like(image), np.zeros_like(image)
        delta = image - mean
        mean += delta / count
        m2 += delta * (image - mean)
    mask = (np.sqrt((m2 / len(rows)).mean(2)) > .025).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return cv2.dilate(mask, np.ones((5, 5), np.uint8)).astype(bool)


def roi_protocol(spec, args, manifest_path, manifest):
    candidate = args.roi_protocol or spec.get('roi_protocol')
    inherited = False
    if not candidate:
        source = ROOT / 'output/dynamic_sr_soft_motion_20260924/roi_registry' / manifest['scene'] / 'roi_protocol.json'
        if not source.exists():
            return {}, dict(status='unavailable', reason='No fixed face/text ROI registered; no semantic claim or blind acceptance')
        candidate, inherited = source, True
    path = path_value(candidate)
    value = read(path)
    manifest_identity = value.get('manifest')
    declared = value.get('manifest_sha256') or (manifest_identity.get('sha256') if isinstance(manifest_identity, dict) else None)
    if value.get('status') != 'fixed_before_prediction_reads' or (declared != sha(manifest_path) and not inherited):
        raise ValueError('ROI protocol must be fixed before prediction reads and match the exact manifest hash')
    lookup = {(row['camera_id'], int(row['frame_index'])): row for row in manifest['observations']}
    for reference in value.get('reference_images', []):
        row = lookup[(reference['camera'], int(reference['frame']))]
        if row['hr_sha256'] != reference['sha256'] or sha(manifest_path.parent / row['hr_path']) != reference['sha256']:
            raise ValueError('Original fixed ROI reference pixels differ from the full-time area4 manifest')
    if inherited and not value.get('reference_images'):
        raise ValueError('Inherited fixed ROIs require byte-identical registered HR references')
    regions = value.get('regions_by_camera_xyxy_exclusive', {})
    width, height = manifest['resolutions']['hr']
    for camera, boxes in regions.items():
        if camera not in manifest['cameras']:
            raise ValueError('Unknown ROI camera')
        for name, box in boxes.items():
            if len(box) != 4 or not all(isinstance(x, int) for x in box):
                raise ValueError(f'Invalid fixed ROI {name}')
            x0, y0, x1, y1 = box
            if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
                raise ValueError('ROI outside HR image')
    return regions, dict(status='fixed', source=identity(path), definition=value,
        current_manifest_sha256=sha(manifest_path), inherited_from_short_window=inherited,
        inheritance_check='Original HR reference frame hashes identical; coordinates copied without prediction inspection' if inherited else None,
        text_status='background_books_reference is a fixed text candidate; readable text presence requires visual review',
        blind_review_status='not_performed; crops are prepared evidence only')


def mean_metric(rows, key):
    import numpy as np
    values = [row.get(key) for row in rows]
    return float(np.mean(values)) if values and all(v is not None for v in values) else None


def aggregate(rows):
    return dict(observations=len(rows), **{key: mean_metric(rows, key) for key in
        ('psnr', 'ssim', 'lpips', 'mse', 'LR_diagnostic_psnr', 'low_psnr', 'mid_psnr', 'high_psnr')},
        PSNR_of_pooled_MSE=(-10 * math.log10(max(mean_metric(rows, 'mse'), 1e-12)) if rows else None))


def evaluate(args):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    spec = read(args.spec)
    manifest_path = path_value(spec['manifest'])
    manifest, tests = validate_manifest(manifest_path)
    root = manifest_path.parent
    if spec.get('manifest_sha256') and spec['manifest_sha256'] != sha(manifest_path):
        raise ValueError('Spec/manifest hash mismatch')
    regions, roi_info = roi_protocol(spec, args, manifest_path, manifest)
    train_cameras = args.train_cameras.split(',') if args.train_cameras else manifest['splits']['train']
    if not set(train_cameras) <= set(manifest['splits']['train']):
        raise ValueError('Training diagnostic cannot contain held-out cameras')
    training = sorted((r for r in manifest['observations'] if r['camera_id'] in train_cameras
                       and r['split'] == 'train' and int(r['frame_index']) in FIXED_FRAMES),
                      key=lambda r: (r['camera_id'], int(r['frame_index'])))
    if len(training) != len(train_cameras) * len(FIXED_FRAMES):
        raise ValueError('Incomplete fixed training diagnostic')
    if args.frames:
        wanted = set(args.frames)
        if not wanted <= set(range(300)):
            raise ValueError('Frame outside 0..299')
        tests = [r for r in tests if int(r['frame_index']) in wanted]
    helpers = metric_helpers()
    fft = module('multi4d_area_frequency', FFT)
    cameras = sorted({'cam00', *train_cameras})
    masks = {camera: fixed_mask(manifest, root, camera, helpers) for camera in cameras}
    args.out.mkdir(parents=True, exist_ok=True)
    preview = args.out / 'previews'
    preview.mkdir(exist_ok=True)
    for camera, mask in masks.items():
        Image.fromarray(mask.astype(np.uint8) * 255).save(preview / f'{camera}_fixed_dynamic_mask.png')
    base = dict(schema='multi4d_area_evaluation_v1', model='Multi4D', posthoc_cull=False,
        render_variant='native_three_branch', scene=manifest['scene'], spec=identity(args.spec),
        manifest=identity(manifest_path), checkpoint=identity(args.checkpoint),
        evaluation_only_HR=True, roi_protocol=roi_info,
        training_fixed_frames=list(FIXED_FRAMES), training_cameras=train_cameras,
        metric_sources=[identity(METRIC), identity(FFT), identity(Path(__file__))],
        spatial_definition='Same Wu helper: RGB float render clamped [0,1], no shaving for PSNR; 11x11 Gaussian sigma1.5 SSIM with outer5 excluded; AlexNet v0.1 LPIPS',
        mask_definition='Fixed full300 HR temporal std RGB mean >.025, open3 close5 dilate5. Time-changing proxy, not semantic segmentation. Evaluation only.',
        temporal_definition='Mean abs((P_t-P_t-1)-(HR_t-HR_t-1)) and squared residual on consecutive cam00 frames; no flow, no geometry claim.',
        information_boundary='Only calibration, frame_index/300 and zero RGB placeholders enter rendering. HR/LR and masks are evaluator references only.',
        host=socket.gethostname(), status='running')
    write(args.out / 'run.json', base)
    upstream = path_value(spec['upstream'])
    os.environ['MULTI4D_UPSTREAM'] = str(upstream)
    for path in (str(OLD), str(upstream)):
        if path in sys.path:
            sys.path.remove(path)
        sys.path.insert(0, path)
    data = module('multi4d_area_eval_data', HERE / 'data.py')
    loader = module('multi4d_area_eval_model', OLD / 'multi_model.py')
    model = loader.load_model(args.checkpoint, manifest=manifest, restore_rng=False)
    metadata = model.checkpoint['metadata']
    if metadata.get('manifest_sha') != sha(manifest_path):
        raise ValueError('Checkpoint/manifest identity mismatch')
    for branch in model.branches:
        for value in vars(branch).values():
            if isinstance(value, torch.nn.Module):
                value.eval()
    torch.set_num_threads(args.cpu_threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    metric = None
    if not args.skip_lpips:
        import lpips
        metric = lpips.LPIPS(net='alex', version='0.1', spatial=False, pnet_rand=False,
            pretrained=True, verbose=False).to(args.lpips_device).eval().requires_grad_(False)
    base.update(method=metadata.get('method', args.method), checkpoint_metadata=metadata,
                capacity=loader.capacity_summary(model), gpu=torch.cuda.get_device_name(),
                lpips_status='explicitly_skipped_smoke_only' if args.skip_lpips else 'alex_v0.1',
                loader_sources=[identity(OLD / 'multi_model.py'), identity(OLD / 'full_state.py'), identity(HERE / 'data.py')])
    training_complete = metadata.get('stage') == 'fine' and metadata.get('completed_updates') == 20000
    base['checkpoint_training_status'] = 'completed' if training_complete else 'incomplete'
    rows, temporal, previous = [], [], None
    started = time.monotonic()
    with (args.out / 'per_observation.jsonl').open('w') as log:
        for index, (scope, record) in enumerate([('test', r) for r in tests] + [('train_diagnostic', r) for r in training]):
            with torch.inference_mode():
                placeholder = torch.zeros((3, manifest['resolutions']['lr'][1], manifest['resolutions']['lr'][0]))
                camera = data.camera(manifest, record, placeholder)
                raw = loader.render_model(model, camera)['render'].detach()
            width, height = manifest['resolutions']['hr']
            if tuple(raw.shape) != (3, height, width) or not bool(torch.isfinite(raw).all()):
                raise ValueError('Invalid native render')
            pred = helpers.image_array(raw)
            gt = checked_rgb(root, record, 'hr', helpers)
            lr = checked_rgb(root, record, 'lr', helpers)
            dynamic = masks[record['camera_id']]
            spatial = helpers.spatial_metrics(pred, gt, dynamic, metric, args.lpips_device)
            full = spatial['full']
            projected = F.avg_pool2d(raw.float()[None], 4)[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
            row = dict(scope=scope, camera=record['camera_id'], frame=int(record['frame_index']),
                psnr=full['psnr'], ssim=full['ssim'], lpips=full.get('lpips_alex'), mse=full['mse'],
                LR_diagnostic_psnr=helpers.metric_psnr(float(((projected - lr) ** 2).mean())),
                regions=spatial, fixed_rois={})
            if scope == 'test':
                bands = fft.decompose(pred, gt)
                row.update({key: bands[key] for key in ('low_mse', 'mid_mse', 'high_mse', 'low_psnr', 'mid_psnr', 'high_psnr')})
                frame = int(record['frame_index'])
                if previous is not None and frame == previous[0] + 1:
                    residual = (pred - previous[1]) - (gt - previous[2])
                    temporal.append(dict(previous_frame=previous[0], frame=frame,
                        frame_difference_residual_l1=float(np.abs(residual).mean()),
                        frame_difference_residual_mse=float((residual ** 2).mean())))
                previous = (frame, pred, gt)
            for name, box in regions.get(record['camera_id'], {}).items():
                x0, y0, x1, y1 = box
                region_mask = np.zeros((height, width), bool)
                region_mask[y0:y1, x0:x1] = True
                roi_scores = helpers.spatial_metrics(pred, gt, region_mask, None)['dynamic']
                row['fixed_rois'][name] = roi_scores
                if int(record['frame_index']) in FIXED_FRAMES:
                    for label, image in [('render', pred), ('HR', gt)]:
                        Image.fromarray(np.rint(image[y0:y1, x0:x1].clip(0, 1) * 255).astype(np.uint8)).save(
                            preview / f'{record["camera_id"]}_{int(record["frame_index"]):04d}_{name}_{label}.png')
            if int(record['frame_index']) in FIXED_FRAMES:
                stem = f'{record["camera_id"]}_{int(record["frame_index"]):04d}'
                Image.fromarray(np.rint(pred.clip(0, 1) * 255).astype(np.uint8)).save(preview / f'{stem}.png')
                np.savez_compressed(preview / f'{stem}.npz', rgb_raw=raw.cpu().numpy().astype(np.float32))
            rows.append(row)
            log.write(json.dumps(row, allow_nan=False) + '\n')
            log.flush()
    test_rows = [r for r in rows if r['scope'] == 'test']
    train_rows = [r for r in rows if r['scope'] == 'train_diagnostic']
    complete = len(test_rows) == 300 and not args.skip_lpips and training_complete
    result = dict(base, status='completed' if complete else 'incomplete', frames=len(test_rows),
        test=aggregate(test_rows), train_diagnostic=aggregate(train_rows),
        train_by_camera={camera: aggregate([r for r in train_rows if r['camera'] == camera]) for camera in train_cameras},
        temporal=dict(pairs=len(temporal), expected_pairs=299,
            status='completed' if len(temporal) == 299 else 'incomplete',
            frame_difference_residual_l1=mean_metric(temporal, 'frame_difference_residual_l1'),
            frame_difference_residual_mse=mean_metric(temporal, 'frame_difference_residual_mse')),
        region_aggregate={name: {key: mean_metric([r['regions'][name] for r in test_rows], key)
            for key in ('psnr', 'ssim', 'mse', 'lpips_alex_spatial_mask')} for name in ('dynamic', 'static')},
        seconds=time.monotonic() - started, peak_GB=torch.cuda.max_memory_allocated() / 1e9,
        visual_acceptance='not_performed; fixed crops are evidence for human blinded review, not an acceptance result')
    write(args.out / 'temporal_per_pair.json', temporal)
    write(args.out / 'summary.json', result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--method')
    parser.add_argument('--frames', type=int, nargs='+', help='Smoke subset, always marked incomplete')
    parser.add_argument('--train-cameras', help='Comma-separated training cameras; default all')
    parser.add_argument('--roi-protocol', type=Path)
    parser.add_argument('--skip-lpips', action='store_true', help='Smoke only; incomplete status')
    parser.add_argument('--lpips-device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--cpu-threads', type=int, default=4)
    args = parser.parse_args(argv)
    if (args.out / 'summary.json').exists():
        raise FileExistsError('Completed/failed evaluation preserved; select a fresh output directory')
    try:
        result = evaluate(args)
        print(json.dumps(dict(status=result['status'], test=result['test'], summary=str(args.out / 'summary.json'))))
    except Exception as error:
        write(args.out / 'summary.json', dict(schema='multi4d_area_evaluation_v1', status='failed',
            method=args.method, spec=str(args.spec), checkpoint=str(args.checkpoint),
            error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc(), test=None))
        raise


if __name__ == '__main__':
    main()
