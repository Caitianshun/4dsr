"""Fixed train76 endpoint gradients and HR-moment structure diagnostics.

Every endpoint is inspected with the same schedule-1 triplets and the same
frozen X support, even when its training arm did not use X. This is a read-only
diagnostic; it consumes legal train LR/full SR and never HR/dev reference images.
Moment dispersion describes the single-surface approximation, not true geometry.
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
from fp_common import ROOT, HERE, OUT, read, write, sha, bound, entry, module, setup
from losses import degradation, sr_loss, x_loss, summed_xyz_gradient_rms
from support_cache import (CAMERAS, CALIBRATION_FRAMES, SupportCache, parent_camera,
                           legal_records, normalize_hr_moments, moment_source_identity)


def _distribution(values):
    values = np.asarray([v for v in values if v is not None], dtype=np.float64)
    if not len(values): return dict(count=0, mean=None, min=None, q25=None, median=None, q75=None, max=None)
    if not np.isfinite(values).all(): raise ValueError('Nonfinite diagnostic scalar')
    return dict(count=len(values), mean=float(values.mean()), min=float(values.min()),
        q25=float(np.quantile(values, .25)), median=float(np.median(values)),
        q75=float(np.quantile(values, .75)), max=float(values.max()))


def _norm(gradients):
    elements = sum(value.numel() for value in gradients)
    square = sum(float(value.detach().double().square().sum()) for value in gradients)
    return dict(l2=math.sqrt(square), rms=math.sqrt(square/max(elements, 1)), elements=elements)


def tensor_versions(value, path='state'):
    """Capture tensor mutation counters without retaining tensor references."""
    if isinstance(value, torch.Tensor):
        return {path: dict(version=int(value._version), shape=list(value.shape),
                           dtype=str(value.dtype), device=str(value.device))}
    result = {}
    if isinstance(value, dict):
        for key in sorted(value, key=lambda x: (type(x).__name__, str(x))):
            result.update(tensor_versions(value[key], path+'/'+str(key)))
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value): result.update(tensor_versions(item, path+'/'+str(index)))
    return result


def _cosine(first, second):
    if len(first) != len(second) or any(a.shape != b.shape for a, b in zip(first, second)):
        raise ValueError('Cosine compares only identical gradient coordinates')
    aa = sum(float(value.detach().double().square().sum()) for value in first)
    bb = sum(float(value.detach().double().square().sum()) for value in second)
    if aa == 0 or bb == 0: return None
    dot = sum(float((a.detach().double()*b.detach().double()).sum()) for a, b in zip(first, second))
    return max(-1., min(1., dot/math.sqrt(aa*bb)))


def _gradient(loss, xyz, rgbs):
    rms, xyz_gradient = summed_xyz_gradient_rms(loss, [xyz], retain_graph=True)
    rgb_gradient = torch.autograd.grad(loss, rgbs, retain_graph=True, allow_unused=True)
    # A is not SR-supervised; its zero gradient still belongs to the fixed
    # three-render RGB coordinates when computing a comparable RGB RMS/cosine.
    rgb_gradient = [torch.zeros_like(rgb) if grad is None else grad for rgb, grad in zip(rgbs, rgb_gradient)]
    if not all(bool(torch.isfinite(grad).all()) for grad in rgb_gradient):
        raise ValueError('Nonfinite RGB diagnostic gradient')
    return dict(xyz=[xyz_gradient], RGB=rgb_gradient,
                summary=dict(effective_xyz=_norm([xyz_gradient]), RGB=_norm(rgb_gradient)))


def moment_statistics(raw):
    """All statistics use native HR moments; unknown variance is separate."""
    result = normalize_hr_moments(raw)
    valid = result['valid_hr']; known = valid & result['known_variance']
    variance = result['variance'][known].detach()
    relative = result['cz'][known].detach()
    sigma = variance.sqrt()
    sigma_q75 = float(torch.quantile(sigma.float(), .75)) if sigma.numel() else None
    relative_q75 = float(torch.quantile(relative.float(), .75)) if relative.numel() else None
    return dict(valid_fraction=float(valid.float().mean()),
        empty_or_invalid_ray_fraction=float((~valid).float().mean()),
        unknown_variance_fraction=float((~result['known_variance']).float().mean()),
        known_supported_variance_pixels=int(known.sum()),
        nonfinite_moment_element_fraction=float((~torch.isfinite(raw.detach())).float().mean()),
        negative_raw_variance_fraction=float((torch.isfinite(result['variance_raw'].detach()) &
                                              (result['variance_raw'].detach() < 0)).float().mean()),
        mean_variance_z=float(variance.double().mean()) if variance.numel() else None,
        mean_sigma_z=float(sigma.double().mean()) if sigma.numel() else None,
        sigma_z_q75=sigma_q75,
        mean_c_z=float(relative.double().mean()) if relative.numel() else None,
        c_z_q75=relative_q75,
        mean_relative_sigma=float(relative.sqrt().double().mean()) if relative.numel() else None,
        epsilon=1e-6, raw_grid=list(raw.shape[-2:]),
        thickness_definition='sigma_z and c_z on finite supported HR rays; not a geometric accuracy measurement')


def _macro(statistics):
    fields = ('valid_fraction', 'empty_or_invalid_ray_fraction', 'unknown_variance_fraction',
        'nonfinite_moment_element_fraction', 'negative_raw_variance_fraction',
        'mean_variance_z', 'mean_sigma_z', 'sigma_z_q75', 'mean_c_z', 'c_z_q75', 'mean_relative_sigma')
    return dict(observations=len(statistics),
        mean_of_observation_statistics={field: _distribution([row[field] for row in statistics]) for field in fields},
        aggregation='Each observation receives equal weight. Means of pixel quantiles are not pooled-pixel quantiles.')


def _triplets(table):
    from schedules import calibration_triplets
    keys = [tuple(key) for key in table['record_keys']]
    rows = calibration_triplets(table)
    if len(rows) != 76 or {keys[row[0]] for row in rows} != {(camera, frame) for camera in CAMERAS for frame in CALIBRATION_FRAMES}:
        raise ValueError('Fixed 19-camera × 4-frame balanced anchors are required')
    triplets = [[keys[i] for i in row] for row in rows]
    if any(len({key[0] for key in row}) != 3 or len({int(key[1]) for key in row}) != 1 for row in triplets):
        raise ValueError('Every diagnostic triplet must have three same-time distinct training cameras')
    return triplets


def _source_identity():
    identity = moment_source_identity()
    for path in (Path(__file__), HERE/'losses.py', HERE/'footprint.py', HERE/'schedules.py',
                 ROOT/'experiments/dynamic_sr_prior_guidance_20260927/shared.py',
                 ROOT/'experiments/dynamic_sr_detail_supervision_20260924/training_support.py'):
        identity[str(path.relative_to(ROOT))] = sha(path)
    return identity


def _exposure(triplets):
    counts = Counter(key for row in triplets for key in row)
    return dict(total_triplet_render_observations=228, unique_observations=len(counts),
        balanced_anchor_observations=76,
        repeated_observation_count=sum(value-1 for value in counts.values()),
        rows=[dict(camera=key[0], frame=int(key[1]), triplet_frequency=value,
                   triplet_mean_weight=value/228, balanced_anchor_mean_weight=1/76)
              for key, value in sorted(counts.items())],
        interpretation='Triplet means include repeated b/c projections. Anchor-only means uniformly cover 19×4.')


def reuse_parent(protocol=None, schedule=None, support_index=None, out=None):
    """CPU-only parent baseline from existing exact HR raw cache/calibration.

    Calibration stored LR/X gradient norms but not SR gradients or pairwise
    cosines. Missing fields remain unknown; no parent render is fabricated.
    """
    protocol = Path(protocol or OUT/'protocol.json'); schedule = Path(schedule or OUT/'schedules/schedule_1.json')
    support_index = Path(support_index or OUT/'support/frozen/index.json')
    out = Path(out or OUT/'diagnostics/U6000'); out.mkdir(parents=True, exist_ok=True)
    output = out/'diagnostics.json'
    if output.exists(): raise ValueError('Diagnostic output exists; preserve it and select a new output directory')
    registration = read(protocol); table = read(schedule); triplets = _triplets(table)
    cal_path = OUT/'calibration.json'; cal = read(cal_path)
    if cal['status'] != 'passed' or cal['identity']['schedule'] != entry(schedule):
        raise ValueError('Parent calibration is incomplete or not the fixed schedule')
    if cal['identity']['parent'] != registration['parent'] or cal['identity']['support_index'] != entry(support_index):
        raise ValueError('Parent calibration and frozen support identity differ')
    parent_index = OUT/'support/parent_moments_hr/index.json'; parent = read(parent_index)
    if parent['parent_sha256'] != registration['parent']['sha256']:
        raise ValueError('Parent raw moment identity mismatch')
    rows = {(row['camera'], int(row['frame'])): row for row in parent['entries']}
    start = time.monotonic(); stats = {}; source = _source_identity()
    for key in sorted({key for triplet in triplets for key in triplet}):
        row = rows[key]; path = parent_index.parent/row['path']
        if sha(path) != row['sha256']: raise ValueError('Parent HR raw moment bytes changed')
        with np.load(path, allow_pickle=False) as data:
            if 'hr_moments' not in data: raise ValueError('Raw native HR moments are required')
            stats[key] = moment_statistics(torch.from_numpy(data['hr_moments']))
    if _source_identity() != source: raise ValueError('Parent diagnostic source changed during run')
    parent_anchor = [stats[row[0]] for row in triplets]
    parent_triplet = [stats[key] for row in triplets for key in row]
    result = dict(status='passed_read_only_parent_reuse', label='U6000', source_identity=source,
        checkpoint=registration['parent'], protocol=entry(protocol), schedule=entry(schedule),
        calibration=entry(cal_path), support_index=entry(support_index), parent_moment_index=entry(parent_index),
        moment_rows=[dict(camera=key[0], frame=int(key[1]), statistics=value) for key, value in sorted(stats.items())],
        balanced_anchor76_structure=_macro(parent_anchor), triplet228_structure=_macro(parent_triplet),
        projection_exposure=_exposure(triplets),
        reused_calibration_rows=cal['rows'],
        reused_calibration_gradient_fields=['LR_effective_xyz_RMS', 'X_effective_xyz_RMS', 'LR_RGB_RMS', 'X_RGB_RMS'],
        missing_not_recomputed=['SR gradient RMS', 'LR/SR/X pairwise gradient cosines'],
        parent_state_unchanged=True, actual_parent_RGB_forwards=0, actual_parent_moment_forwards=0,
        existing_parent_cache_moment_cost_counted_again=False, calibration_gradient_cost_counted_again=False,
        cost=dict(rgb_forwards=0, moment_forwards=0, formal_updates=0, Adam_calls=0,
                  CPU_seconds=time.monotonic()-start, torch=str(torch.__version__)),
        reference_HR_images_read=False, development_images_read=False,
        conclusion_boundary='Dispersion and fixed X support are model diagnostics, not true geometry.')
    write(output, result)
    return result


def diagnose(checkpoint, label, out=None, protocol=None, schedule=None, support_index=None):
    protocol = Path(protocol or OUT/'protocol.json'); schedule = Path(schedule or OUT/'schedules/schedule_1.json')
    support_index = Path(support_index or OUT/'support/frozen/index.json')
    checkpoint = Path(checkpoint); out = Path(out or OUT/'diagnostics'/label)
    output = out/'diagnostics.json'
    if output.exists(): raise ValueError('Diagnostic output exists; preserve it and select a new output directory')
    out.mkdir(parents=True, exist_ok=True)
    if (out/'gdiag.jsonl').exists(): raise ValueError('Partial diagnostics exist; preserve them before a retry')
    shared, policy = setup()
    from common import image_tensor
    from n3dv_data import load_manifest
    from motion_model import load_model, rasterize
    from training_support import digest_state
    import footprint
    gm = module('fp_diagnostic_HR_moments', ROOT/'experiments/dynamic_sr_confidence_geometry_20261006/depth_prior.py')
    registration = read(protocol); table = read(schedule); triplets = _triplets(table)
    manifest = load_manifest(bound(registration['manifest'])); observations = legal_records(manifest)
    obs_by = {(row['camera_id'], int(row['frame_index'])): row for row in observations}
    teachers = read(bound(registration['teacher']))
    if teachers.get('manifest_sha256') != registration['manifest']['sha256']:
        raise ValueError('Frozen full SR teacher manifest mismatch')
    teacher_by = {(row['camera'], int(row['frame'])): row for row in teachers['entries']}
    if set(teacher_by) != set(obs_by): raise ValueError('Teacher must cover only legal training observations')
    cal_path = OUT/'calibration.json'; cal = read(cal_path)
    if cal['status'] != 'passed': raise ValueError('Frozen calibration is not accepted')
    support = SupportCache(support_index)
    model = load_model(checkpoint, manifest)
    if model.branch != 'ordinary_split': raise ValueError('Unexpected representation for registered diagnostics')
    sources = _source_identity()
    identity = dict(checkpoint=entry(checkpoint), checkpoint_metadata=model.checkpoint['metadata'],
        protocol=entry(protocol), schedule=entry(schedule), calibration=entry(cal_path),
        support_index=entry(support_index), frozen_parent=registration['parent'],
        manifest=registration['manifest'], teacher=registration['teacher'], source_identity=sources,
        label=label, actual_training_method=model.checkpoint['metadata'].get('method'),
        diagnostic_objectives=dict(LR='mean of all three true LR L1, regardless of training arm',
            SR='.05 full teacher L1(b) + .05 full teacher L1(c)',
            X='unscaled six-edge complete source RGB plus target moment projection, regardless of training arm'),
        formal_updates=0, Adam_calls=0, balanced_anchor_observations=76,
        no_HR_reference_or_dev_inputs=True, shared_same_time_effective_xyz=True,
        xyz_coordinates='same fixed Gaussian indices, summed before RMS',
        RGB_coordinates='all three CHW RGB renders; absent SR(a) is zero in the same-dimensional RMS/cosine')
    write(out/'registration.json', dict(status='registered_before_diagnostic_forward', **identity))
    before = dict(model=digest_state(model.g.capture()), children=digest_state(model.children.state_dict()),
        child_adam=digest_state(model.child_optimizer.state_dict()),
        parameter_gradients=digest_state({name: param.grad for name, param in shared.all_named(model).items()}))
    versions_before = tensor_versions(dict(model=model.g.capture(), children=model.children.state_dict(),
                                          child_adam=model.child_optimizer.state_dict()))
    rng_before = digest_state(dict(torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all(),
        numpy=np.random.get_state(), python=random.getstate()))
    camera_cache, image_cache, reads = {}, {}, {}
    def pixels(key, teacher=False):
        if key not in obs_by: raise ValueError('Illegal diagnostic observation')
        token = (key, teacher)
        if token not in image_cache:
            row = teacher_by[key] if teacher else obs_by[key]
            relative = row['relative_path'] if teacher else row['lr_path']
            expected = row['sha256'] if teacher else row['lr_sha256']
            if teacher and row['lr_sha256'] != obs_by[key]['lr_sha256']:
                raise ValueError('Full SR teacher does not derive from the registered LR')
            path = Path(manifest['_root'])/relative
            if sha(path) != expected: raise ValueError('Diagnostic input image bytes changed')
            reads[str(path.relative_to(ROOT))] = expected
            image_cache[token] = image_tensor(path)
        return image_cache[token].cuda()
    def camera(key):
        if key not in camera_cache:
            camera_cache[key] = parent_camera(manifest, obs_by[key], len(camera_cache))
        return camera_cache[key]
    torch.set_num_threads(4); torch.cuda.reset_peak_memory_stats(); started = time.monotonic()
    rows = []; anchor_moments = []; triplet_moments = []
    cost = dict(rgb_forwards=0, moment_forwards=0, effective_xyz_gradient_calls=0,
        RGB_gradient_calls=0, native_RGB_backwards=0, native_moment_backwards=0,
        formal_updates=0, Adam_calls=0)
    with (out/'gdiag.jsonl').open('x', buffering=1) as log:
        for number, keys in enumerate(triplets):
            tick = time.monotonic(); cameras = [camera(key) for key in keys]
            lrs = [pixels(key) for key in keys]
            teacher_images = [pixels(keys[j], True) for j in (1, 2)]
            write(out/'last_attempt.json', dict(status='before_diagnostic_batch', batch=number,
                completed_batches=len(rows), confirmed_cost=cost, formal_updates=0,
                incomplete_active_operation_cost_may_be_unknown=True))
            state = policy.effective_state(model, cameras[0].time)
            rgbs, moments = [], []
            for current in cameras:
                rgbs.append(rasterize(model.g, current, state['xyz'], state['cov'], state['opacity'], state['sh'])['render'])
                cost['rgb_forwards'] += 1
                write(out/'last_attempt.json', dict(status='RGB_forward_returned', batch=number,
                    completed_batches=len(rows), confirmed_cost=cost, formal_updates=0,
                    incomplete_active_operation_cost_may_be_unknown=True))
            for current in cameras:
                moments.append(gm.render_moments(current, state['xyz'], state['cov'], state['opacity']))
                cost['moment_forwards'] += 1
                write(out/'last_attempt.json', dict(status='moment_forward_returned', batch=number,
                    completed_batches=len(rows), confirmed_cost=cost, formal_updates=0,
                    incomplete_active_operation_cost_may_be_unknown=True))
            errors = [degradation(rgb, lr.shape[-2:])-lr for rgb, lr in zip(rgbs, lrs)]
            ll = sum(error.abs().mean() for error in errors)/3
            ls = .05*sr_loss(rgbs[1], teacher_images[0]) + .05*sr_loss(rgbs[2], teacher_images[1])
            lx, xdiag = x_loss(rgbs, moments, cameras, lrs, keys, support, footprint)
            if not all(bool(torch.isfinite(value)) for value in (ll, ls, lx)):
                raise ValueError('Nonfinite fixed diagnostic objective')
            gradients = {name: _gradient(value, state['xyz'], rgbs) for name, value in [('LR', ll), ('SR', ls), ('X', lx)]}
            cost['effective_xyz_gradient_calls'] += 3; cost['RGB_gradient_calls'] += 3
            cost['native_RGB_backwards'] += 8; cost['native_moment_backwards'] += 3
            cosines = {f'{a}_vs_{b}': {route: _cosine(gradients[a][route], gradients[b][route]) for route in ('xyz', 'RGB')}
                       for a, b in [('LR', 'SR'), ('LR', 'X'), ('SR', 'X')]}
            structures = [moment_statistics(value['hr_moments']) for value in moments]
            anchor_moments.append(structures[0]); triplet_moments.extend(structures)
            lr_norm = gradients['LR']['summary']['effective_xyz']['rms']
            x_norm = gradients['X']['summary']['effective_xyz']['rms']
            record = dict(batch=number, keys=[list(key) for key in keys],
                lr_l1=float(ll.detach()), lr_mse=float(sum(error.square().mean() for error in errors)/3),
                weighted_full_sr_l1=float(ls.detach()), x_loss=float(lx.detach()),
                gradients={name: value['summary'] for name, value in gradients.items()},
                gradient_cosines=cosines, frozen_lambda_X=float(cal['lambda_X']),
                scaled_X_to_LR_xyz_RMS_ratio=float(cal['lambda_X'])*x_norm/(lr_norm+1e-12),
                x_diagnostic=xdiag, moment_statistics=structures,
                rgb_forwards=3, moment_forwards=3, formal_updates=0, Adam_calls=0,
                seconds=time.monotonic()-tick)
            rows.append(record); log.write(json.dumps(record, ensure_ascii=False, allow_nan=False)+'\n')
            write(out/'last_attempt.json', dict(status='diagnostic_batch_complete', batch=number,
                completed_batches=len(rows), confirmed_cost=cost, formal_updates=0))
            # No old retain_graph objectives remain while the next triplet is
            # rendered. Keeping lx across iterations would retain all six
            # warps and unnecessarily double the live GPU graph footprint.
            del gradients, rgbs, moments, state, ll, ls, lx, errors, teacher_images, lrs
            if (number+1)%4 == 0:
                print(json.dumps(dict(stage='fixed_train76_endpoint_diagnostics', label=label,
                    completed_batches=number+1, wall_s=time.monotonic()-started)), flush=True)
    after = dict(model=digest_state(model.g.capture()), children=digest_state(model.children.state_dict()),
        child_adam=digest_state(model.child_optimizer.state_dict()),
        parameter_gradients=digest_state({name: param.grad for name, param in shared.all_named(model).items()}))
    versions_after = tensor_versions(dict(model=model.g.capture(), children=model.children.state_dict(),
                                         child_adam=model.child_optimizer.state_dict()))
    rng_after = digest_state(dict(torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all(),
        numpy=np.random.get_state(), python=random.getstate()))
    if before != after: raise ValueError('Read-only diagnostics changed model or Adam state')
    if versions_before != versions_after: raise ValueError('Read-only diagnostics changed tensor mutation versions')
    if rng_before != rng_after: raise ValueError('Read-only diagnostics changed global RNG state')
    if _source_identity() != sources: raise ValueError('Diagnostic sources changed during execution')
    if sha(checkpoint) != identity['checkpoint']['sha256']: raise ValueError('Input checkpoint bytes changed')
    for name in ('protocol', 'schedule', 'calibration', 'support_index'): bound(identity[name])
    write(out/'immutability_audit.json', dict(status='passed_model_two_Adam_RNG_and_tensor_versions_unchanged',
        digest_before=before, digest_after=after, tensor_versions_before=versions_before,
        tensor_versions_after=versions_after, tensor_version_rows=len(versions_before),
        RNG_before=rng_before, RNG_after=rng_after, formal_updates=0, Adam_calls=0))
    norms = {name: {route: _distribution([row['gradients'][name][route]['rms'] for row in rows])
                    for route in ('effective_xyz', 'RGB')} for name in ('LR', 'SR', 'X')}
    cosines = {pair: {route: _distribution([row['gradient_cosines'][pair][route] for row in rows])
                      for route in ('xyz', 'RGB')} for pair in ('LR_vs_SR', 'LR_vs_X', 'SR_vs_X')}
    result = dict(status='passed_read_only_fixed_train76_diagnostics', **identity, rows=rows,
        balanced_anchor76_structure=_macro(anchor_moments), triplet228_structure=_macro(triplet_moments),
        projection_exposure=_exposure(triplets), gradient_RMS_distributions=norms,
        gradient_cosine_distributions=cosines,
        X_diagnostic_distribution={name: _distribution([row['x_diagnostic'][name] for row in rows])
            for name in ('effective_edges', 'empty_edges', 'effective_area_fraction', 'support_decrease_fraction', 'loss')},
        parent_state_unchanged=True, two_Adam_states_unchanged=True, global_RNG_unchanged=True,
        tensor_mutation_versions_unchanged=True, parameter_gradients_unchanged=True,
        immutability_audit=entry(out/'immutability_audit.json'),
        actual_legal_image_reads=reads, actual_support_cache_reads=support.opened,
        gdiag=entry(out/'gdiag.jsonl'),
        cost=dict(cost, seconds=time.monotonic()-started,
            peak_gb=torch.cuda.max_memory_allocated()/1e9, gpu=torch.cuda.get_device_name(),
            physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'), torch=str(torch.__version__)),
        reference_HR_images_read=False, development_images_read=False,
        conclusion_boundary='X residual and model depth dispersion are diagnostics. They do not prove true geometry or replace held-out HR quality.')
    write(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--label', required=True)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--protocol', type=Path, default=OUT/'protocol.json')
    parser.add_argument('--schedule', type=Path, default=OUT/'schedules/schedule_1.json')
    parser.add_argument('--support-index', type=Path, default=OUT/'support/frozen/index.json')
    parser.add_argument('--reuse-parent', action='store_true')
    args = parser.parse_args()
    output = args.out or OUT/'diagnostics'/args.label
    try:
        if args.reuse_parent:
            result = reuse_parent(args.protocol, args.schedule, args.support_index, output)
        else:
            if args.checkpoint is None: parser.error('--checkpoint is required for endpoint diagnostics')
            result = diagnose(args.checkpoint, args.label, output, args.protocol, args.schedule, args.support_index)
    except Exception as exc:
        attempt = output/'last_attempt.json'
        write(output/'failure.json', dict(status='failed_read_only_diagnostics', error=repr(exc),
            formal_updates=0, Adam_calls=0, last_attempt=entry(attempt) if attempt.exists() else None,
            partial_confirmed_cost=read(attempt).get('confirmed_cost') if attempt.exists() else None,
            incomplete_active_operation_cost_unknown=True, source_sha256=sha(__file__)))
        raise
    print(json.dumps(dict(status=result['status'], label=args.label, cost=result['cost']), ensure_ascii=False), flush=True)


if __name__ == '__main__': main()
