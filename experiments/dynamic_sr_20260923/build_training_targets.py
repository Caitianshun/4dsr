"""Prepare C/D float32 teacher caches with the frozen P1 LR-only construction.

This command does not decide whether P2 may run. Each invocation creates a new
scene directory and refuses to overwrite. Default selection is the four original
teacher cameras and the 60 even frames. Use --cams and --frames for a small audit.
No HR image is read, CUDA is hidden, and no training is performed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import socket
import sys
import time
import traceback

# Hide CUDA before importing the original target code or torch. All original
# target construction functions operate on CPU numpy / torch tensors.
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import cv2
import numpy as np
import torch
import temporal_sharing as frozen

ROOT = Path(__file__).resolve().parents[2]
SCENES = {
    'cook_spinach': ('n3dv_prepared/cook_spinach', ['cam02', 'cam06', 'cam12', 'cam18']),
    'meetroom_discussion': ('meetroom_prepared/discussion', ['cam02', 'cam04', 'cam08', 'cam12']),
}
ALL_FRAMES = list(range(0, 120, 2))
EDGE_FRAMES = {0, 2, 116, 118}


def stamp():
    return datetime.now(timezone.utc).isoformat()


def array_hash(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def scalar_stats(array, mask=None):
    x = np.asarray(array)
    if mask is not None:
        x = x[mask]
    if not x.size:
        return {'count': 0, 'mean': None, 'min': None, 'max': None}
    return {'count': int(x.size), 'mean': float(x.mean()), 'min': float(x.min()), 'max': float(x.max())}


def atomic_npy(path, array):
    if path.exists():
        raise FileExistsError(path)
    if array.dtype != np.float32 or not np.isfinite(array).all():
        raise ValueError('Teacher must be finite float32 without post-construction clipping')
    tmp = path.with_suffix('.npy.tmp')
    with tmp.open('xb') as f:
        np.save(f, array, allow_pickle=False)
    tmp.replace(path)
    return {'path': str(path), 'sha256': frozen.sha(path), 'bytes': path.stat().st_size,
            'dtype': str(array.dtype), 'shape': list(array.shape),
            'array_sha256': array_hash(array), 'range': [float(array.min()), float(array.max())],
            'outside_0_1_fraction': float(((array < 0) | (array > 1)).mean())}


def construct(lrs, teachers, anchor, observations, camera):
    """Keep operation order identical to P1 build(), through C/D construction.

    Patch selection remains in the training-cache path because it determines the
    exposure-fit exclusion mask. Changing it would change P1's frozen operator.
    Additional scalar diagnostics are computed only after C and D are complete.
    """
    states, eigen = frozen.flow_states(lrs, anchor)
    selected = frozen.patches(states, eigen)
    base = teachers[anchor]
    shape = base.shape[:2]
    lrsize = lrs[anchor].shape[:2]
    if shape != (lrsize[0] * 4, lrsize[1] * 4):
        raise ValueError('Only the frozen x4 geometry is supported')
    highs, w_r, w_p, pair_meta = [], [], [], []
    for s in states:
        gain, bias, exp_info = frozen.exposure_fit(lrs[anchor], s['warped_lr'], s['valid'], selected)
        error = np.mean(np.abs(s['warped_lr'] * gain + bias - lrs[anchor]), axis=2)
        p = np.exp(-(cv2.boxFilter(error, -1, (7, 7), normalize=True) / .03) ** 2).astype(np.float32)
        if not exp_info['known']:
            p.fill(0)
            s['r'].fill(0)
        s['exposure_known'] = exp_info['known']
        s['p'] = p
        flow_hr = frozen.lift(s['flow'], shape) * 4
        aligned = frozen.warp(teachers[s['frame']], flow_hr)
        highs.append(frozen.high(aligned * gain + bias, lrsize))
        w_r.append(frozen.lift(s['r'], shape, True))
        w_p.append(frozen.lift(p, shape, True))
        pair_meta.append({'frame': s['frame'], 'exposure': exp_info,
                          'normalized_dt': observations[(camera, s['frame'])]['time'] - observations[(camera, anchor)]['time'],
                          'applied_gain_float32': gain.tolist(), 'applied_bias_float32': bias.tolist()})
    hs, wr, ps = np.stack(highs), np.stack(w_r), np.stack(w_p)
    support = np.stack([frozen.lift(s['valid'].astype(np.float32), shape, True) * s['exposure_known'] for s in states]).sum(0)
    mpmap = (wr * ps).sum(0) / np.maximum(wr.sum(0), 1e-12)
    eta_c = .5 * (support >= 2).astype(np.float32)
    eta_d = eta_c * mpmap * (mpmap >= .25)
    C = frozen.combine(base, hs, wr, eta_c)
    D = frozen.combine(base, hs, wr * ps, eta_d)

    # Diagnostics below cannot change frozen flow, support, exposure, p or eta.
    for s, meta in zip(states, pair_meta):
        gain = np.asarray(meta['applied_gain_float32'], dtype=np.float32)
        bias = np.asarray(meta['applied_bias_float32'], dtype=np.float32)
        raw_error = np.mean(np.abs(s['warped_lr'] - lrs[anchor]), axis=2)
        corrected_error = np.mean(np.abs(s['warped_lr'] * gain + bias - lrs[anchor]), axis=2)
        den = float(s['r'].sum())
        meta['lr_statistics'] = {
            'valid_pixels': int(s['valid'].sum()), 'valid_fraction': float(s['valid'].mean()),
            'r': scalar_stats(s['r']), 'p': scalar_stats(s['p']),
            'p_weighted_by_r': float((s['r'] * s['p']).sum() / max(den, 1e-12)),
            'raw_rgb_mae_on_valid': scalar_stats(raw_error, s['valid']),
            'exposure_corrected_rgb_mae_on_valid': scalar_stats(corrected_error, s['valid']),
            'forward_backward_flow_error_on_valid': scalar_stats(s['fb'], s['valid']),
            'farneback_dis_disagreement_on_valid': scalar_stats(s['disagreement'], s['valid']),
            'flow_sha256': array_hash(s['flow']), 'valid_sha256': array_hash(s['valid']),
            'r_sha256': array_hash(s['r']), 'p_sha256': array_hash(s['p']),
        }
    maps = {'flow': np.stack([s['flow'] for s in states]),
            'valid': np.stack([s['valid'] for s in states]),
            'r': np.stack([s['r'] for s in states]), 'p': np.stack([s['p'] for s in states]),
            'eta_c_lr': eta_c[::4, ::4], 'eta_d_lr': eta_d[::4, ::4]}
    stats = {'patches_used_for_exposure_exclusion': selected, 'neighbors': pair_meta,
             'C_coverage': float((eta_c > 0).mean()), 'D_coverage': float((eta_d > 0).mean()),
             'mean_eta_C': float(eta_c.mean()), 'mean_eta_D': float(eta_d.mean()),
             'eta_c_lr': scalar_stats(maps['eta_c_lr']), 'eta_d_lr': scalar_stats(maps['eta_d_lr']),
             'lr_support_count': scalar_stats(support[::4, ::4]),
             'weighted_mean_p_lr': scalar_stats(mpmap[::4, ::4])}
    return {'C': C, 'D': D}, stats, maps


def verify_frozen(p1):
    lock = json.loads((p1 / 'targets_locked.json').read_text())
    for filename, key in [('protocol.json', 'protocol_sha256'), ('inputs.json', 'inputs_sha256'),
                          ('observations.json', 'observations_sha256')]:
        if frozen.sha(p1 / filename) != lock[key]:
            raise AssertionError(f'P1 {filename} changed after targets were locked')
    if frozen.sha(frozen.__file__) != lock['source_sha256'] or frozen.sha(p1 / 'source.py') != lock['source_sha256']:
        raise AssertionError('Current temporal_sharing.py differs from the frozen P1 source')
    # These control CPU reduction / optical-flow behavior and are fixed in P1.
    if str(torch.__version__) != lock['torch'] or cv2.__version__ != lock['cv2']:
        raise RuntimeError('Use the same torch/OpenCV runtime as P1 for exact target reconstruction')
    return lock


def build(args):
    started = time.monotonic()
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    if torch.cuda.is_initialized():
        raise RuntimeError('CUDA was initialized before this CPU-only task')
    lock = verify_frozen(args.p1)
    sub, default_cams = SCENES[args.scene]
    cameras = args.cams.split(',') if args.cams else default_cams
    frames = [int(v) for v in args.frames.split(',')] if args.frames else ALL_FRAMES
    if len(cameras) != len(set(cameras)) or not set(cameras) <= set(default_cams):
        raise ValueError('Cameras must be unique members of the four frozen teacher cameras')
    if len(frames) != len(set(frames)) or not set(frames) <= set(ALL_FRAMES):
        raise ValueError('Frames must be unique even original-frame indices 0..118')
    data = ROOT / 'data/dynamic_sr' / sub
    manifest_path = data / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    observations = {(x['camera_id'], x['frame_index']): x for x in manifest['observations']}
    if not set(cameras) <= set(manifest['splits']['train']):
        raise ValueError('Requested camera is not in the training split')
    args.out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(__file__, args.out / 'builder_source.py')
    shutil.copyfile(frozen.__file__, args.out / 'temporal_sharing_source.py')
    source_sha = frozen.sha(__file__)
    h, w = manifest['resolutions']['hr'][1], manifest['resolutions']['hr'][0]
    per_pair_bytes = h * w * 3 * np.dtype(np.float32).itemsize * 2
    protocol = {'schema': 'dynamic_sr_training_targets_v1', 'scene': args.scene,
                'cameras': cameras, 'frames_original': frames, 'full_scene_cameras': default_cams,
                'full_scene_frames': ALL_FRAMES, 'offsets': frozen.OFFSETS,
                'interior_frames': '4..114 inclusive, even; all offsets -4,-2,+2,+4 required',
                'boundary_fallback_frames': sorted(EDGE_FRAMES), 'boundary_rule': 'C=D=A exact float32 decoded teacher',
                'frozen_source_sha256': lock['source_sha256'], 'builder_source_sha256': source_sha,
                'p1_targets_lock_sha256': frozen.sha(args.p1 / 'targets_locked.json'),
                'p1_protocol_sha256': lock['protocol_sha256'], 'p1_directory': str(args.p1),
                'targets': 'Exactly frozen P1 C/D, including LR patch exclusions for exposure fit',
                'array_layout': 'HWC, float32 .npy, original RGB convention, no clipping or quantization',
                'patch_sampling_role': 'LR-only exposure exclusion mask; no training patch crop',
                'no_hr_images_read': True, 'no_gpu': True, 'parameter_updates': 0,
                'p2_authorization': 'Cache availability is not a go/no-go decision or authorization to train',
                'torch_threads': 4, 'opencv_threads': 2, 'torch': str(torch.__version__), 'opencv': cv2.__version__,
                'python': platform.python_version(), 'executable': sys.executable, 'host': socket.gethostname(),
                'predicted_array_payload_bytes_selected': per_pair_bytes * len(cameras) * len(frames),
                'predicted_array_payload_bytes_full_scene': per_pair_bytes * 4 * 60,
                'disk_free_bytes_before': shutil.disk_usage(args.out).free, 'started_utc': stamp()}
    frozen.write(args.out / 'protocol.json', protocol)
    inputs = {str(manifest_path): {'kind': 'manifest_metadata', 'sha256': frozen.sha(manifest_path)}}
    p1_inputs = json.loads((args.p1 / 'inputs.json').read_text())
    p1_rows = json.loads((args.p1 / 'observations.json').read_text())
    completed, equivalences = [], []

    def read_frame(cam, frame):
        o = observations[(cam, frame)]
        if o['split'] != 'train':
            raise AssertionError('Nontraining image requested')
        lp = data / o['lr_path']
        sp = data / 'sr_swinir_x4' / cam / Path(o['lr_path']).name
        for path, kind in [(lp, 'train_lr'), (sp, 'frozen_swinir_teacher')]:
            value = frozen.sha(path)
            if kind == 'train_lr' and value != o['lr_sha256']:
                raise AssertionError(f'LR file identity mismatch: {path}')
            if str(path) in p1_inputs and value != p1_inputs[str(path)]:
                raise AssertionError(f'P1 input file identity changed: {path}')
            inputs[str(path)] = {'kind': kind, 'sha256': value, 'bytes': path.stat().st_size}
        # Never access o['hr_path'] or any HR pixel file.
        return frozen.rgb(lp), frozen.rgb(sp)

    try:
        for cam in cameras:
            for anchor in frames:
                tick = time.monotonic()
                branch = args.out / cam / f'{anchor:04d}'
                branch.mkdir(parents=True)
                if anchor in EDGE_FRAMES:
                    _lr, base = read_frame(cam, anchor)
                    arrays = {'C': base.copy(), 'D': base.copy()}
                    stats = {'boundary_fallback': True, 'fallback_reason': 'full frozen neighborhood unavailable',
                             'neighbors': [], 'C_coverage': 0., 'D_coverage': 0.,
                             'mean_eta_C': 0., 'mean_eta_D': 0., 'exact_A_fallback': True}
                    maps = {}
                else:
                    needed = [anchor] + [anchor + v for v in frozen.OFFSETS]
                    if not all((cam, frame) in observations for frame in needed):
                        raise ValueError('Missing interior neighbor; refusing to silently change the window')
                    lrs, teachers = {}, {}
                    for frame in needed:
                        lrs[frame], teachers[frame] = read_frame(cam, frame)
                    arrays, stats, maps = construct(lrs, teachers, anchor, observations, cam)
                    stats['boundary_fallback'] = False
                built_seconds = time.monotonic() - tick
                files = {k: atomic_npy(branch / f'{k}.npy', x) for k, x in arrays.items()}
                for v in files.values():
                    v['path'] = str(Path(v['path']).relative_to(args.out))
                if args.verify_p1:
                    row = next((r for r in p1_rows if r['scene'] == args.scene and r['camera'] == cam and r['anchor'] == anchor), None)
                    if row is None:
                        raise ValueError('--verify-p1 requires an existing P1 anchor/camera observation')
                    reference = args.p1 / row['data_file']
                    if frozen.sha(reference) != row['sha256']:
                        raise AssertionError('P1 target artifact hash mismatch')
                    checks = {}
                    with np.load(reference, allow_pickle=False) as ref:
                        for k, x in {**arrays, **maps}.items():
                            y = ref[k]
                            same = x.dtype == y.dtype and x.shape == y.shape and np.array_equal(x, y)
                            delta = np.abs(x.astype(np.float64) - y.astype(np.float64))
                            checks[k] = {'exact_equal': bool(same), 'dtype': str(x.dtype),
                                         'shape': list(x.shape), 'max_abs_difference': float(delta.max()),
                                         'nonidentical_values': int(np.count_nonzero(x != y))}
                            if not same:
                                raise AssertionError(f'{args.scene}/{cam}/{anchor}/{k} differs from P1: {checks[k]}')
                    equivalences.append({'camera': cam, 'frame': anchor, 'p1_file': str(reference),
                                          'p1_file_sha256': row['sha256'], 'checks': checks})
                statpath = branch / 'lr_statistics.json'
                frozen.write(statpath, stats)
                completed.append({'scene': args.scene, 'camera': cam, 'frame': anchor,
                                  'time': observations[(cam, anchor)]['time'], 'files': files,
                                  'statistics': {'path': str(statpath.relative_to(args.out)), 'sha256': frozen.sha(statpath)},
                                  'boundary_fallback': stats['boundary_fallback'],
                                  'build_seconds': built_seconds, 'total_seconds': time.monotonic() - tick})
                print(json.dumps({'event': 'target_written', 'scene': args.scene, 'camera': cam, 'frame': anchor,
                                  'total_seconds': completed[-1]['total_seconds'], 'fallback': stats['boundary_fallback']}), flush=True)
                # Free image/flow tensors before the next anchor. No full-scene cache in RAM.
                del arrays, maps
                if anchor not in EDGE_FRAMES:
                    del lrs, teachers
        if torch.cuda.is_initialized():
            raise AssertionError('Unexpected CUDA initialization')
        if frozen.sha(frozen.__file__) != lock['source_sha256'] or frozen.sha(__file__) != source_sha:
            raise AssertionError('Construction source changed during this run')
        frozen.write(args.out / 'input_hashes.json', inputs)
        frozen.write(args.out / 'manifest.json', {'schema': 'dynamic_sr_training_targets_v1', 'scene': args.scene,
                     'protocol_sha256': frozen.sha(args.out / 'protocol.json'),
                     'input_hashes_sha256': frozen.sha(args.out / 'input_hashes.json'), 'targets': completed})
        if args.verify_p1:
            frozen.write(args.out / 'p1_equivalence.json', {'all_exact': True, 'rows': equivalences})
        actual_bytes = sum(v['bytes'] for r in completed for v in r['files'].values())
        interior = [r for r in completed if not r['boundary_fallback']]
        mean_interior_s = float(np.mean([r['total_seconds'] for r in interior])) if interior else None
        frozen.write(args.out / 'complete.json', {'status': 'prepared_cache_subset' if len(completed) < 240 else 'prepared_full_scene_cache',
                     'target_count': len(completed), 'actual_npy_bytes': actual_bytes,
                     'manifest_sha256': frozen.sha(args.out / 'manifest.json'),
                     'elapsed_seconds': time.monotonic() - started, 'max_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                     'mean_measured_interior_seconds': mean_interior_s,
                     'estimated_full_scene_interior_seconds': mean_interior_s * 224 if mean_interior_s is not None else None,
                     'cost_caveat': 'Timing extrapolation only; 224 interior targets and 16 boundary fallbacks per full scene; disk/contention can change cost',
                     'full_scene_array_payload_bytes': protocol['predicted_array_payload_bytes_full_scene'],
                     'p1_equivalence_exact': True if args.verify_p1 else None,
                     'no_hr_images_read': True, 'no_gpu': True, 'parameter_updates': 0, 'finished_utc': stamp()})
    except Exception:
        frozen.write(args.out / 'failed.json', {'status': 'failed', 'traceback': traceback.format_exc(),
                                               'completed_targets': completed, 'no_gpu': True, 'parameter_updates': 0})
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene', required=True, choices=list(SCENES))
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--cams', help='Comma-separated subset of original teacher cameras; defaults to all four')
    p.add_argument('--frames', help='Comma-separated original even frame numbers; defaults to all 60')
    p.add_argument('--p1', type=Path, default=ROOT / 'output/dynamic_sr_20260923/temporal_sharing_v1')
    p.add_argument('--verify-p1', action='store_true', help='Require exact existing P1 C/D and LR-map equivalence for every selected observation')
    build(p.parse_args())
