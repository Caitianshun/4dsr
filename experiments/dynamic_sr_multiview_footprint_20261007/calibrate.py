"""One train76-only frozen X/E scale calibration, with zero Adam updates.

The parent renders each same-time triplet from one effective geometry tensor.
X's effective_xyz derivative includes all source RGB and all target moment
paths. E reuses the 76 anchor RGB renders, with no reference HR search.
"""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import time
import numpy as np
import torch
from fp_common import ROOT, HERE, OUT, read, write, bound, entry, sha, module, setup
from losses import degradation, x_loss, summed_xyz_gradient_rms
from support_cache import SupportCache, parent_camera, CAMERAS, CALIBRATION_FRAMES, legal_records, moment_source_identity

# Numerical guard, not a quality threshold or hyperparameter grid. A vanishing
# X derivative must be diagnosed rather than magnified into a large coefficient.
EPSILON = 1e-12
MIN_X_LR_RMS_RATIO = 1e-6
MAX_MEDIAN_LR_X_RATIO = 1e4


def _rms(gradients):
    active = [value.detach().double() for value in gradients if value is not None]
    if not active: return 0.
    elements = sum(value.numel() for value in active)
    return math.sqrt(sum(float(value.square().sum()) for value in active) / elements)


def _distribution(values):
    values = np.asarray(values, np.float64)
    if not len(values): return dict(count=0, min=None, max=None, q25=None, median=None, q75=None)
    if not np.isfinite(values).all(): raise ValueError('Nonfinite calibration distribution')
    return dict(count=len(values), min=float(values.min()), max=float(values.max()),
                q25=float(np.quantile(values, .25)), median=float(np.median(values)), q75=float(np.quantile(values, .75)))


def calibrate(protocol=None, schedule=None, support_index=None, output=None):
    protocol = Path(protocol or OUT/'protocol.json'); schedule = Path(schedule or OUT/'schedules/schedule_1.json')
    support_index = Path(support_index or OUT/'support/frozen/index.json'); output = Path(output or OUT/'calibration.json')
    if output.exists():
        existing = read(output)
        if existing.get('status') == 'passed':
            for identity in existing['source_files'].values(): bound(identity)
            for name, path in [('protocol', protocol), ('schedule', schedule), ('support_index', support_index)]:
                if existing['identity'][name] != entry(path): raise ValueError('Frozen calibration source changed')
            if moment_source_identity() != existing['identity']['backbone_and_moment_sources']:
                raise ValueError('Frozen calibration backbone or moment source changed')
            return existing
        raise ValueError('Preserve failed calibration; supply a distinct new output path after correcting the implementation')
    shared, policy = setup()
    from n3dv_data import load_manifest
    from common import image_tensor
    from motion_model import load_model, rasterize
    from training_support import digest_state
    from schedules import calibration_triplets
    import footprint
    gm = module('fp_calibration_moments', ROOT/'experiments/dynamic_sr_confidence_geometry_20261006/depth_prior.py')
    registration = read(protocol); table = read(schedule)
    manifest = load_manifest(bound(registration['manifest'])); observations = legal_records(manifest)
    by = {(row['camera_id'], int(row['frame_index'])): row for row in observations}
    keys = [tuple(key) for key in table['record_keys']]
    triplets = calibration_triplets(table)
    if len(triplets) != 76 or {keys[row[0]] for row in triplets} != {(camera, frame) for camera in CAMERAS for frame in CALIBRATION_FRAMES}:
        raise ValueError('Calibration anchors must exactly cover 19 cameras × 4 fixed frames')
    support = SupportCache(support_index)
    model = load_model(bound(registration['parent']), manifest)
    if model.checkpoint['metadata'].get('intervention_step') != 6000:
        raise ValueError('Calibration must use complete registered U6000')
    before = dict(model=digest_state(model.g.capture()), children=digest_state(model.children.state_dict()),
        child_adam=digest_state(model.child_optimizer.state_dict()))
    source_files = {str(path.relative_to(ROOT)): entry(path) for path in [Path(__file__), HERE/'losses.py', HERE/'support_cache.py',
        HERE/'footprint.py', HERE/'schedules.py', Path(gm.__file__), ROOT/'experiments/dynamic_sr_prior_guidance_20260927/gradient_policy.py']}
    identity = dict(protocol=entry(protocol), schedule=entry(schedule), support_index=entry(support_index),
        parent=registration['parent'], manifest=registration['manifest'],
        registered_triplets=[[list(keys[i]) for i in row] for row in triplets],
        shared_effective_xyz=True, xyz_gradients_summed_before_RMS=True,
        X_gradient_paths=['all three source RGB', 'all three target HR moment depths'],
        moment_covariance_opacity_detached=True, source_RGB_attributes_trainable=True,
        epsilon=EPSILON, min_X_to_LR_RMS_signal_ratio=MIN_X_LR_RMS_RATIO,
        max_median_LR_to_X_RMS_ratio=MAX_MEDIAN_LR_X_RATIO,
        HR_reference_images_read=False, development_images_read=False, parameter_updates=0, Adam_calls=0,
        backbone_and_moment_sources=moment_source_identity())
    write(output.with_name(output.stem+'_registration.json'), dict(status='registered_before_forward', identity=identity, source_files=source_files))
    camera_cache, lr_cache, opened = {}, {}, {}
    allowed = {(row['camera_id'], int(row['frame_index'])) for row in observations}
    def camera_lr(key):
        if key not in allowed: raise ValueError(f'Illegal training observation: {key}')
        if key not in camera_cache:
            obs = by[key]; path = Path(manifest['_root'])/obs['lr_path']
            if sha(path) != obs['lr_sha256']: raise ValueError('Calibration LR bytes changed')
            opened[str(path.relative_to(ROOT))] = obs['lr_sha256']
            lr_cache[key] = image_tensor(path)
            camera_cache[key] = parent_camera(manifest, obs, len(camera_cache))
        return camera_cache[key], lr_cache[key].cuda()
    torch.set_num_threads(4); torch.cuda.reset_peak_memory_stats(); started = time.monotonic()
    rows, x_ratios, e_ratios, valid_rows = [], [], [], []
    cost = dict(rgb_forwards=0, moment_forwards=0, effective_xyz_gradient_calls=0,
        RGB_gradient_calls=0, native_RGB_backwards=0, native_moment_backwards=0,
        parameter_updates=0, Adam_calls=0)
    log_path = output.with_name(output.stem+'_gdiag.jsonl')
    attempt_path = output.with_name(output.stem+'_last_attempt.json')
    with log_path.open('x', buffering=1) as log:
        for number, row in enumerate(triplets):
            tick = time.monotonic(); triplet_keys = [keys[i] for i in row]
            if len({key[0] for key in triplet_keys}) != 3 or len({key[1] for key in triplet_keys}) != 1:
                raise ValueError('Calibration triplet is not same-time distinct cameras')
            cameras, lrs = zip(*(camera_lr(key) for key in triplet_keys))
            state = policy.effective_state(model, cameras[0].time)
            write(attempt_path, dict(status='before_registered_batch', batch=number, keys=[list(key) for key in triplet_keys],
                completed_batches=len(rows), confirmed_cost=cost, seconds=time.monotonic()-started,
                incomplete_active_operation_cost_may_be_unknown=True, formal_updates=0))
            rgb, moments = [], []
            for camera in cameras:
                rgb.append(rasterize(model.g, camera, state['xyz'], state['cov'], state['opacity'], state['sh'])['render'])
                cost['rgb_forwards'] += 1
                write(attempt_path, dict(status='RGB_forward_returned', batch=number, completed_batches=len(rows),
                    confirmed_cost=cost, seconds=time.monotonic()-started,
                    incomplete_active_operation_cost_may_be_unknown=True, formal_updates=0))
            for camera in cameras:
                moments.append(gm.render_moments(camera, state['xyz'], state['cov'], state['opacity']))
                cost['moment_forwards'] += 1
                write(attempt_path, dict(status='moment_forward_returned', batch=number, completed_batches=len(rows),
                    confirmed_cost=cost, seconds=time.monotonic()-started,
                    incomplete_active_operation_cost_may_be_unknown=True, formal_updates=0))
            errors = [degradation(value, lr.shape[-2:]) - lr for value, lr in zip(rgb, lrs)]
            ll = sum(error.abs().mean() for error in errors) / 3
            lx, diagnostic = x_loss(rgb, moments, cameras, lrs, triplet_keys, support, footprint)
            if not bool(torch.isfinite(ll) & torch.isfinite(lx)): raise ValueError('Nonfinite calibration loss')
            lr_xyz_rms, _ = summed_xyz_gradient_rms(ll, [state['xyz']], retain_graph=True)
            x_xyz_rms, _ = summed_xyz_gradient_rms(lx, [state['xyz']], retain_graph=True)
            cost['effective_xyz_gradient_calls'] += 2
            cost['native_RGB_backwards'] += 6; cost['native_moment_backwards'] += 3
            lr_rgb = torch.autograd.grad(ll, rgb, retain_graph=True, allow_unused=True)
            x_rgb = torch.autograd.grad(lx, rgb, retain_graph=True, allow_unused=True)
            anchor_l1, anchor_l2 = errors[0].abs().mean(), errors[0].square().mean()
            gl1, = torch.autograd.grad(anchor_l1, rgb[0], retain_graph=True)
            gl2, = torch.autograd.grad(anchor_l2, rgb[0], retain_graph=True)
            cost['RGB_gradient_calls'] += 4
            l1_rgb_rms, l2_rgb_rms = _rms([gl1]), _rms([gl2])
            e_ratio = l1_rgb_rms/(l2_rgb_rms + EPSILON)
            if not math.isfinite(e_ratio) or l2_rgb_rms <= EPSILON:
                raise ValueError('E MSE gradient has no usable signal; do not amplify a zero denominator')
            e_ratios.append(e_ratio)
            empty = diagnostic['effective_edges'] == 0
            negligible = x_xyz_rms <= max(EPSILON, MIN_X_LR_RMS_RATIO * lr_xyz_rms)
            ratio = None if empty or negligible else lr_xyz_rms/(x_xyz_rms + EPSILON)
            if ratio is not None: x_ratios.append(ratio); valid_rows.append(number)
            record = dict(batch=number, keys=[list(key) for key in triplet_keys],
                lr_loss=float(ll.detach()), x_loss=float(lx.detach()),
                LR_effective_xyz_RMS=lr_xyz_rms, X_effective_xyz_RMS=x_xyz_rms,
                LR_RGB_RMS=_rms(lr_rgb), X_RGB_RMS=_rms(x_rgb),
                E_anchor_L1_RGB_RMS=l1_rgb_rms, E_anchor_MSE_RGB_RMS=l2_rgb_rms,
                E_L1_to_MSE_RGB_RMS_ratio=e_ratio, LR_to_X_xyz_RMS_ratio=ratio,
                excluded_empty_support=empty, excluded_negligible_X_gradient=negligible,
                x_diagnostic=diagnostic, rgb_forwards=3, moment_forwards=3,
                formal_updates=0, Adam_calls=0, seconds=time.monotonic()-tick)
            rows.append(record); log.write(json.dumps(record, ensure_ascii=False, allow_nan=False)+'\n')
            write(attempt_path, dict(status='registered_batch_complete', batch=number, completed_batches=len(rows),
                confirmed_cost=cost, seconds=time.monotonic()-started, formal_updates=0))
            if (number+1)%4 == 0:
                print(json.dumps(dict(stage='train76_X_E_calibration', batches=number+1,
                    X_supported_batches=len(x_ratios), wall_s=time.monotonic()-started)), flush=True)
            # Release every retained six-edge graph before rendering the next
            # triplet. Calibration values and coefficients are unchanged.
            del rgb, moments, state, errors, ll, lx, anchor_l1, anchor_l2, lr_rgb, x_rgb, gl1, gl2, _
    if not x_ratios: raise ValueError('X has no usable geometric signal in registered train76 support')
    median_ratio = float(np.median(x_ratios))
    if median_ratio > MAX_MEDIAN_LR_X_RATIO:
        raise ValueError('X geometry signal too weak; refuse a large compensating coefficient')
    coefficient = .1 * median_ratio; kappa = float(np.median(e_ratios))
    after = dict(model=digest_state(model.g.capture()), children=digest_state(model.children.state_dict()),
        child_adam=digest_state(model.child_optimizer.state_dict()))
    if before != after: raise ValueError('Calibration changed parent model or optimizer state')
    for item in source_files.values(): bound(item)
    if moment_source_identity() != identity['backbone_and_moment_sources']:
        raise ValueError('Parent calibration backbone source changed')
    report = dict(status='passed', calibration_stage='frozen_train76_X_E', lambda_X=coefficient, kappa=kappa,
        tau_z=support.index['tau_z'], identity=identity, source_files=source_files,
        registered_batches=76, supported_batches=len(x_ratios),
        excluded_empty_batches=sum(row['excluded_empty_support'] for row in rows),
        excluded_negligible_X_gradient_batches=sum(row['excluded_negligible_X_gradient'] and not row['excluded_empty_support'] for row in rows),
        rows=rows, LR_to_X_RMS_ratio_distribution=_distribution(x_ratios),
        final_scaled_X_to_LR_xyz_RMS_ratio_distribution=_distribution([
            coefficient*rows[i]['X_effective_xyz_RMS']/(rows[i]['LR_effective_xyz_RMS']+EPSILON) for i in valid_rows]),
        E_L1_to_MSE_RGB_RMS_ratio_distribution=_distribution(e_ratios),
        gdiag=entry(log_path), actual_LR_image_reads=opened, parent_state_unchanged=True,
        source_RGB_and_target_moment_paths_included=True, shared_effective_xyz=True,
        one_same_coefficient_X_and_MX=True, cost=dict(cost, seconds=time.monotonic()-started,
            peak_gb=torch.cuda.max_memory_allocated()/1e9, gpu=torch.cuda.get_device_name(),
            physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'), torch=str(torch.__version__)))
    write(output, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=OUT/'protocol.json')
    parser.add_argument('--schedule', type=Path, default=OUT/'schedules/schedule_1.json')
    parser.add_argument('--support-index', type=Path, default=OUT/'support/frozen/index.json')
    parser.add_argument('--output', type=Path, default=OUT/'calibration.json')
    args = parser.parse_args()
    try:
        report = calibrate(args.protocol, args.schedule, args.support_index, args.output)
    except Exception as exc:
        failure = args.output.with_name(args.output.stem+'_failure.json')
        attempt = args.output.with_name(args.output.stem+'_last_attempt.json')
        write(failure, dict(status='failed_calibration_no_formal_updates', error=repr(exc),
            parameter_updates=0, Adam_calls=0, source_sha256=sha(__file__),
            last_attempt=entry(attempt) if attempt.exists() else None,
            partial_confirmed_cost=read(attempt).get('confirmed_cost') if attempt.exists() else None,
            incomplete_active_operation_cost_unknown=True))
        raise
    print(json.dumps(dict(status=report['status'], lambda_X=report['lambda_X'], kappa=report['kappa'], cost=report['cost'])), flush=True)


if __name__ == '__main__': main()
