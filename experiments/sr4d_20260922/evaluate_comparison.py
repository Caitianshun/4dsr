#!/usr/bin/env python3
"""Fixed-endpoint SR4D/Wu comparison; HR is evaluation-only, never selection.

Run one method per fresh process to isolate incompatible upstream module names.
Uses float renders, project PSNR/SSIM and normalized LPIPS-Alex. Explicit input
protocol selects the shared scale-residual/closure operator. Render PNGs are
previews; existing-bicubic observations are the original frozen uint8 PNGs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
OLD = PROJECT / 'experiments/dynamic_sr_20260918'
sys.path.insert(0, str(OLD))
from evaluate import spatial_metrics, read_rgb, write_rgb
from sr4d_common import ManifestData, Camera, load_endpoint, render_endpoint, sha256


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def choose_rows(data, prior_cameras, frames, test_camera):
    available = {(r['camera_id'], r['frame_index']): r for r in data.rows}
    train = sorted(data.document['splits']['train'])
    prior = sorted(set(prior_cameras))
    if len(prior) != len(prior_cameras) or not prior or not set(prior) <= set(train):
        raise ValueError('Teacher cameras must be distinct training cameras')
    remaining = sorted(set(train) - set(prior))
    if len(remaining) < 3:
        raise ValueError('Need three non-teacher training cameras for the fixed diagnostic')
    selected = [remaining[i] for i in [0, (len(remaining) - 1) // 2, len(remaining) - 1]]
    test = sorted([r for r in data.rows if r['camera_id'] == test_camera], key=lambda r: r['frame_index'])
    if not test or any(r['split'] != 'test' for r in test):
        raise ValueError('Expected a nonempty held-out test camera')
    declared_frames = list(data.document['frame_indices'])
    if [r['frame_index'] for r in test] != sorted(declared_frames):
        raise ValueError('Test camera does not cover the whole fixed time window')
    rows = [('novel_cam00', r) for r in test]
    groups = {'novel_cam00': [test_camera], 'teacher_camera_train': prior,
              'lr_only_camera_train': selected}
    for group in ('teacher_camera_train', 'lr_only_camera_train'):
        for cid in groups[group]:
            for frame in frames:
                row = available[(cid, frame)]
                if row['split'] != 'train':
                    raise ValueError('Training diagnostic points into held-out data')
                rows.append((group, row))
    return rows, {'groups': groups, 'train_diagnostic_frames': frames,
                  'novel_frames': [r['frame_index'] for r in test],
                  'all_train_cameras': train, 'nonteacher_candidates': remaining,
                  'rule': 'Teacher cameras from explicit training config; first/lower-middle/last remaining camera ID; fixed frames; every test-camera frame. No metric-based selection.'}


def validate_input_protocol(data, protocol):
    if protocol == 'author-bilinear':
        if data.document.get('comparison_protocol') != 'sr4d_author_lr_v1':
            raise ValueError('Author protocol requires its frozen float-LR manifest')
        if any('lr_float_path' not in row for row in data.rows):
            raise ValueError('Author protocol requires float LR for every observation')
    elif protocol == 'existing-bicubic':
        expected = ('torch F.interpolate bicubic, align_corners=False, antialias=True, '
                    'exact 4x, clamp [0,1], round uint8 PNG')
        if data.document['degradation']['reference_to_lr'] != expected:
            raise ValueError('Existing protocol requires the unchanged bicubic-AA/uint8 manifest')
        if any('lr_float_path' in row for row in data.rows):
            raise ValueError('Existing protocol must not silently consume author float-LR data')
        if any('lr_path' not in row or 'lr_sha256' not in row for row in data.rows):
            raise ValueError('Existing protocol requires original PNG input identities')
    else:
        raise ValueError('Unknown input protocol: ' + protocol)


def input_downsample(image, size, protocol):
    if protocol == 'existing-bicubic':
        # Exactly common.downsample, without importing Wu model/renderer modules
        # into the incompatible SR4D process. CPU parity is checked separately.
        return F.interpolate(image[None], size=size, mode='bicubic',
                             align_corners=False, antialias=True)[0].clamp(0, 1)
    if protocol == 'author-bilinear':
        return F.interpolate(image[None], size=size, mode='bilinear',
                             align_corners=False)[0].clamp(0, 1)
    raise ValueError('Unknown input protocol: ' + protocol)


def read_lr(data, row, input_protocol='author-bilinear'):
    if input_protocol == 'existing-bicubic':
        path = data.root / row['lr_path']
        if sha256(path) != row['lr_sha256']:
            raise ValueError(f'LR identity mismatch: {path}')
        with Image.open(path) as im:
            if im.mode != 'RGB' or im.size != data.lr_wh:
                raise ValueError(f'Expected original RGB LR PNG: {path}')
            a = np.array(im, copy=True)
        if a.dtype != np.uint8:
            raise ValueError('Existing LR must retain its frozen uint8 quantization')
        return torch.from_numpy(a).permute(2, 0, 1).float().div(255)
    if input_protocol != 'author-bilinear':
        raise ValueError('Unknown input protocol: ' + input_protocol)
    if 'lr_float_path' not in row:
        raise ValueError('Strict comparison requires author-protocol float LR, not preview PNG')
    path = data.root / row['lr_float_path']
    if sha256(path) != row['lr_float_sha256']:
        raise ValueError(f'LR identity mismatch: {path}')
    a = np.load(path, allow_pickle=False)
    w, h = data.lr_wh
    if a.dtype != np.float32 or a.shape != (3, h, w):
        raise ValueError(f'Expected CHW float32 LR: {path}')
    if not np.isfinite(a).all() or a.min() < 0 or a.max() > 1:
        raise ValueError('Invalid LR values')
    return torch.from_numpy(a.copy())


def tensor_state_hash(module):
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


@torch.no_grad()
def image_scores(pred_chw, gt_hwc, lpips_model, lr_wh, input_protocol='author-bilinear'):
    pred = pred_chw.clamp(0, 1)
    p = pred.permute(1, 2, 0).cpu().numpy()
    if p.shape != gt_hwc.shape or not np.isfinite(p).all():
        raise ValueError('Prediction and reference mismatch/nonfinite')
    # Reuse exactly the project function; regional metrics are not requested here.
    metric = spatial_metrics(p, gt_hwc, np.zeros(p.shape[:2], bool))['full']
    gt = torch.from_numpy(np.ascontiguousarray(gt_hwc)).permute(2, 0, 1).to(pred.device)
    metric['lpips_alex'] = float(lpips_model(pred[None] * 2 - 1, gt[None] * 2 - 1).item())
    shape = (lr_wh[1], lr_wh[0])
    residual = lambda x: x - F.interpolate(
        input_downsample(x, shape, input_protocol)[None],
        size=x.shape[-2:], mode='bicubic', align_corners=False)[0]
    rp, rg = residual(pred), residual(gt)
    metric['scale_high_residual_mse'] = float((rp - rg).square().mean().item())
    metric['scale_high_residual_pred_energy'] = float(rp.square().mean().item())
    metric['scale_high_residual_gt_energy'] = float(rg.square().mean().item())
    return metric


def endpoint_loader(args, data):
    upstream = Path(args.upstream).resolve()
    if args.method == 'sr4d':
        if args.run is None or args.iteration is None:
            raise ValueError('SR4D requires --run and --iteration')
        run = Path(args.run).resolve()
        config = json.loads((run / 'config.json').read_text())
        if config['manifest_sha256'] != sha256(data.path):
            raise ValueError('SR4D trained manifest differs from evaluator manifest')
        point_dir = 'point_cloud_hr' if args.stage == 'fine' else 'point_cloud'
        files = [run / point_dir / f'iteration_{args.iteration}' / 'point_cloud.ply', run / 'config.json']
        coarse_iteration = args.iteration
        if args.stage == 'fine':
            mp = run / 'deform_hr' / f'iteration_{args.iteration}' / 'stage_metadata.json'
            coarse_iteration = json.loads(mp.read_text())['coarse_iteration']
            files += [mp, run / 'deform_hr' / f'iteration_{args.iteration}' / 'deform_hr.pth']
        files += [run / 'deform_lr' / f'iteration_{coarse_iteration}' / 'deform_lr.pth']
        g, coarse, fine = load_endpoint(upstream, run, args.stage, args.iteration)

        def render(row):
            return render_endpoint(g, coarse, fine, Camera(data, row))['render_hr']

        identity = {'run': str(run), 'stage': args.stage, 'iteration': args.iteration,
                    'coarse_iteration': coarse_iteration, 'point_count': len(g.get_xyz),
                    'training_config': config,
                    'files': {str(p): sha256(p) for p in files}}
    else:
        if args.checkpoint is None:
            raise ValueError('Wu requires --checkpoint')
        # common.py imports Wu-specific arguments/scene/renderer at module import.
        os.environ['FOURDSR_UPSTREAM'] = str(upstream)
        sys.path.insert(0, str(OLD))
        from common import load_checkpoint, render_image
        g, _, _, ck = load_checkpoint(args.checkpoint)
        metadata = ck.get('metadata', {})
        mh = metadata.get('manifest_sha', metadata.get('manifest_sha256'))
        if mh is None or mh != sha256(data.path):
            raise ValueError('Wu checkpoint lacks the exact comparison manifest identity')
        g._deformation.eval()

        def render(row):
            camera = Camera(data, row, device='cuda')
            camera.time = float(row['time'])
            return render_image(g, camera, stage=args.stage)['render']

        identity = {'checkpoint': str(Path(args.checkpoint).resolve()),
                    'checkpoint_sha256': sha256(args.checkpoint), 'stage': args.stage,
                    'iteration_label': args.iteration, 'metadata': metadata,
                    'point_count': len(g.get_xyz)}
    sources = [Path(__file__), HERE / 'sr4d_common.py', OLD / 'evaluate.py', OLD / 'common.py',
               upstream / 'gaussian_renderer/__init__.py', upstream / 'scene/gaussian_model.py']
    identity['evaluation_source_hashes'] = {str(p): sha256(p) for p in sources}
    return render, identity


def summarize(rows):
    result = {}
    metric_keys = ('mse', 'psnr', 'ssim', 'lpips_alex', 'scale_high_residual_mse',
                   'scale_high_residual_pred_energy', 'scale_high_residual_gt_energy')
    for group in sorted({r['group'] for r in rows}):
        subset = [r for r in rows if r['group'] == group]
        means = lambda part: {k: float(np.mean([r['hr'][k] for r in part])) for k in metric_keys}
        result[group] = {'count': len(subset), 'hr': means(subset),
                         'lr_input_mse': float(np.mean([r['lr_input_mse'] for r in subset])),
                         'lr_bicubic_aa_mse': float(np.mean([r['lr_bicubic_aa_mse'] for r in subset])),
                         'lr_bilinear_mse': float(np.mean([r['lr_bilinear_mse'] for r in subset])),
                         'lr_box_mse': float(np.mean([r['lr_box_mse'] for r in subset])),
                         'per_camera': {cid: means([r for r in subset if r['camera_id'] == cid])
                                        for cid in sorted({r['camera_id'] for r in subset})}}
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--method', choices=['sr4d', 'wu'], required=True)
    ap.add_argument('--upstream', required=True)
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--run')
    ap.add_argument('--checkpoint')
    ap.add_argument('--out', required=True)
    ap.add_argument('--stage', choices=['coarse', 'fine'], default='fine')
    ap.add_argument('--iteration', type=int)
    ap.add_argument('--prior-cameras', required=True)
    ap.add_argument('--train-frames', default='0,40,80,118')
    ap.add_argument('--test-camera', default='cam00')
    ap.add_argument('--input-protocol', choices=['author-bilinear', 'existing-bicubic'],
                    default='author-bilinear', help='Explicit observation operator; never rewrites the manifest')
    ap.add_argument('--limit-test', type=int, help='Engineering smoke only: first N test frames')
    ap.add_argument('--skip-train', action='store_true', help='Engineering smoke only: omit train diagnostics')
    ap.add_argument('--audit-selection-only', action='store_true')
    ap.add_argument('--save-all-float', action='store_true')
    args = ap.parse_args()
    started = time.monotonic()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    try:
        torch.set_num_threads(4)
        data = ManifestData(args.manifest)
        validate_input_protocol(data, args.input_protocol)
        frames = [int(s) for s in args.train_frames.split(',')]
        selected, protocol = choose_rows(data, args.prior_cameras.split(','), frames, args.test_camera)
        if args.limit_test is not None:
            if args.limit_test < 1:
                raise ValueError('--limit-test must be positive')
            test_subset = [(g, r) for g, r in selected if g == 'novel_cam00'][:args.limit_test]
            selected = test_subset + [(g, r) for g, r in selected if g != 'novel_cam00']
        if args.skip_train:
            selected = [(g, r) for g, r in selected if g == 'novel_cam00']
        protocol.update({'manifest': str(data.path), 'manifest_sha256': sha256(data.path),
            'evaluation_only': True, 'endpoint_selection': 'Explicit fixed endpoint; no metric feedback to training',
            'engineering_only': args.limit_test is not None or args.skip_train,
            'limit_test': args.limit_test, 'skip_train': args.skip_train,
            'actual_novel_frames': [r['frame_index'] for g, r in selected if g == 'novel_cam00'],
            'input_protocol': args.input_protocol,
            'comparison_protocol': ('existing_bicubic_frozen_v1' if args.input_protocol == 'existing-bicubic'
                                    else data.document['comparison_protocol']),
            'metric_definition': 'Per-image PSNR and project SSIM (11x11 Gaussian sigma1.5, excludes 5px border); LPIPS-Alex v0.1 on [-1,1]. Float render clamped [0,1], no PNG quantization before metrics.',
            'high_residual_definition': ('H(X)=X-U(D(X)); D=' +
                ('bicubic x4 align_corners=False antialias=True clamp[0,1], exactly common.downsample'
                 if args.input_protocol == 'existing-bicubic' else 'bilinear x4 align_corners=False noAA clamp[0,1]') +
                '; U=bicubic align_corners=False noAA; operate on clamped float HR; not orthogonal frequency decomposition.'),
            'closure_definition': ('lr_input_mse uses the selected input-protocol D(raw HR), clamp[0,1], '
                'without rounding rendered LR; compare actual frozen observed LR. Bicubic-AA, bilinear/noAA '
                'and box-average closures are separately named; box is SR4D internal fine-stage operator.'),
            'input_degradation': data.document['degradation'],
            'selection_count': len(selected)})
        write_json(out / 'protocol.json', protocol)
        if args.audit_selection_only:
            write_json(out / 'status.json', {'state': 'selection_checked', 'gpu_used': False,
                'counts': {g: sum(x == g for x, _ in selected) for g in protocol['groups']}})
            return
        if not torch.cuda.is_available():
            raise RuntimeError('Renderer evaluation requires a remote CUDA device')
        torch.cuda.reset_peak_memory_stats()
        import lpips
        perceptual = lpips.LPIPS(net='alex', version='0.1').cuda().eval().requires_grad_(False)
        predictor, identity = endpoint_loader(args, data)
        identity['lpips_state_sha256'] = tensor_state_hash(perceptual)
        identity['environment'] = {'host': platform.node(), 'python': sys.version,
            'torch': torch.__version__, 'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(),
            'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES')}
        write_json(out / 'identity.json', identity)
        rows, saved = [], []
        with torch.inference_mode():
            for index, (group, row) in enumerate(selected):
                hr_path = data.root / row['hr_path']
                if sha256(hr_path) != row['hr_sha256']:
                    raise ValueError(f'HR identity mismatch: {hr_path}')
                reference = read_rgb(hr_path)
                lr = read_lr(data, row, args.input_protocol).cuda()
                raw = predictor(row)
                if raw.shape != (3, data.hr_wh[1], data.hr_wh[0]) or not torch.isfinite(raw).all():
                    raise ValueError('Invalid raw prediction')
                quality = image_scores(raw, reference, perceptual, data.lr_wh, args.input_protocol)
                bilinear = F.interpolate(raw[None], size=lr.shape[-2:], mode='bilinear', align_corners=False)[0]
                bicubic = input_downsample(raw, lr.shape[-2:], 'existing-bicubic')
                box = F.avg_pool2d(raw[None], kernel_size=4, stride=4)[0]
                input_key = 'lr_float' if args.input_protocol == 'author-bilinear' else 'lr'
                rows.append({'group': group, 'camera_id': row['camera_id'], 'frame_index': row['frame_index'],
                    'time': row['time'], 'split': row['split'], 'hr': quality,
                    'hr_path': str(hr_path), 'hr_sha256': row['hr_sha256'],
                    'lr_observation_format': 'CHW_float32_npy' if input_key == 'lr_float' else 'RGB_uint8_PNG',
                    input_key + '_path': str(data.root / row[input_key + '_path']),
                    input_key + '_sha256': row[input_key + '_sha256'],
                    'lr_input_mse': float((input_downsample(raw, lr.shape[-2:], args.input_protocol) - lr).square().mean()),
                    'lr_bicubic_aa_mse': float((bicubic - lr).square().mean()),
                    'lr_bilinear_mse': float((bilinear.clamp(0, 1) - lr).square().mean()),
                    'lr_box_mse': float((box.clamp(0, 1) - lr).square().mean())})
                keep = args.save_all_float or (group == 'novel_cam00' and row['frame_index'] in frames)
                keep |= (group != 'novel_cam00' and row['camera_id'] == protocol['groups'][group][0] and row['frame_index'] == frames[0])
                if keep:
                    stem = f"{row['camera_id']}_{row['frame_index']:04d}"
                    dest = out / 'predictions'
                    dest.mkdir(exist_ok=True)
                    float_path = dest / f'{stem}.npy'
                    np.save(float_path, raw.detach().float().cpu().numpy(), allow_pickle=False)
                    write_rgb(dest / f'{stem}.png', raw.clamp(0, 1).permute(1, 2, 0).cpu().numpy())
                    saved.append({'camera_id': row['camera_id'], 'frame_index': row['frame_index'],
                        'float_raw_chw_path': str(float_path.relative_to(out)), 'sha256': sha256(float_path)})
                if index % 20 == 0:
                    print(json.dumps({'evaluated': index + 1, 'total': len(selected)}), flush=True)
        if sha256(data.path) != protocol['manifest_sha256']:
            raise RuntimeError('Manifest changed during evaluation')
        result = {'complete': True, 'method': args.method, 'scene': data.scene,
            'protocol': protocol, 'identity': identity, 'rows': rows, 'groups': summarize(rows),
            'saved_predictions': saved, 'elapsed_s': time.monotonic() - started,
            'peak_GiB': torch.cuda.max_memory_allocated() / 2 ** 30}
        write_json(out / 'metrics.json', result)
        write_json(out / 'status.json', {'state': 'completed', 'count': len(rows),
            'metrics_sha256': sha256(out / 'metrics.json'), 'elapsed_s': result['elapsed_s']})
        print(json.dumps({'complete': True, 'method': args.method, 'groups': result['groups'],
                          'elapsed_s': result['elapsed_s']}), flush=True)
    except BaseException:
        write_json(out / 'status.json', {'state': 'failed', 'traceback': traceback.format_exc(),
                                       'elapsed_s': time.monotonic() - started})
        raise


if __name__ == '__main__':
    main()
