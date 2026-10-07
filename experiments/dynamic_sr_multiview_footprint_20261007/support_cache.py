"""Frozen U6000 HR moments, physical support and one dispersion weight.

Old parent exports retain only LR normalized statistics. They cannot be
interpolated into HR physical moments. This module exports fresh HR raw
moments once, then freezes all edge weights and parent support on that grid.
No RGB, held-out images, teacher residuals or HR reference images are read.
"""
from __future__ import annotations
import argparse
from collections import OrderedDict
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn.functional as F
from fp_common import ROOT, HERE, OUT, read, write, sha, bound, entry, module, setup

CAMERAS = tuple(f'cam{i:02d}' for i in range(2, 21))
FRAMES = tuple(range(0, 120, 2))
CALIBRATION_FRAMES = (0, 40, 80, 118)
HR_SIZE = (1008, 1344)
LR_SIZE = (252, 336)
RULES = dict(epsilon=1e-6, alpha_empty_floor=1e-3, tau_quantile=.75,
    tau_floor=1e-6, soft_weight_floor=.25, unknown_variance_weight=1.,
    concentrated_rule='finite known variance and c_z <= frozen train76 tau_z on both parent rays',
    occlusion='projected target source-axis z > parent source Z + max(.02 parent source Z, 3 source sigma)',
    occlusion_relative=.02, occlusion_sigma=3., mask_format='numpy packbits little, full target HR grid',
    edge_soft_weight='min(source parent w sampled at frozen parent projection, target parent w); nonnegative area to LR',
    source_occlusion_statistics='nonnegative area-mipmap/bilinear aggregate raw [A,M1,M2] before normalizing source Z and variance',
    source_unknown_variance='neutral w=1; aggregate known-variance indicator must be 1 before any hard occlusion',
    weight_changes_with_current_depth=False, photometric_mask=False, low_texture_mask=False,
    visibility_is_truth=False)


def moment_source_identity():
    """Bind only actual preparation dependencies, never an unfinished trainer."""
    paths = [HERE/'fp_common.py', Path(__file__),
        ROOT/'experiments/dynamic_sr_prior_guidance_20260927/gradient_policy.py',
        ROOT/'experiments/dynamic_sr_motion_bound_20260923/motion_model.py',
        ROOT/'experiments/dynamic_sr_20260918/common.py',
        ROOT/'experiments/dynamic_sr_20260918/n3dv_data.py',
        ROOT/'experiments/dynamic_sr_confidence_geometry_20261006/depth_prior.py']
    result = {str(path.relative_to(ROOT)): sha(path) for path in paths}
    upstream = Path(os.environ.get('FOURDSR_UPSTREAM', '/home/cai_tianshun/Project/4dgs'))
    for name in ('scene/gaussian_model.py', 'scene/deformation.py', 'scene/hexplane.py'):
        result['upstream/'+name] = sha(upstream/name)
    return result


def normalize_hr_moments(moments, epsilon=1e-6):
    """HR raw [A,M1,M2], axial depth; no LR interpolation or signed filters."""
    if moments.ndim != 3 or moments.shape[0] != 3:
        raise ValueError('Expected HR raw moments (3,H,W)')
    a, m1, m2 = moments.unbind(0)
    denominator = torch.where(torch.isfinite(a) & (a >= 0), a, torch.zeros_like(a)) + epsilon
    z_raw = m1 / denominator
    variance_raw = m2 / denominator - z_raw.square()
    valid = (torch.isfinite(a) & torch.isfinite(m1) & (a > RULES['alpha_empty_floor']) &
             torch.isfinite(z_raw) & (z_raw > 0)).detach()
    known_variance = (torch.isfinite(m2) & torch.isfinite(variance_raw)).detach()
    # Safe substitutes happen before projection and sampling. The detached
    # invalid flags still ensure the complete sampling support is excluded.
    z = torch.where(torch.isfinite(z_raw) & (z_raw > 0), z_raw, torch.ones_like(z_raw))
    variance = torch.where(known_variance, variance_raw.clamp_min(0), torch.zeros_like(variance_raw))
    cz = variance / (z.square() + epsilon)
    return dict(alpha=a, z=z, variance=variance, variance_raw=variance_raw, raw_moments=moments,
                cz=cz, valid_hr=valid, known_variance=known_variance)


def dispersion_weight(result, tau):
    weight = .25 + .75 / (1 + result['cz'] / float(tau))
    return torch.where(result['known_variance'], weight, torch.ones_like(weight)).detach()


def pack_mask(value):
    array = value.detach().cpu().numpy().astype(np.uint8)
    return np.packbits(array.reshape(-1), bitorder='little')


def unpack_mask(packed, shape):
    return np.unpackbits(packed, bitorder='little', count=int(np.prod(shape))).reshape(tuple(shape)).astype(bool)


def atomic_npz(path, **arrays):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with tmp.open('wb') as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush(); os.fsync(handle.fileno())
    tmp.replace(path)


def legal_records(manifest):
    rows = sorted((row for row in manifest['observations'] if row['split'] == 'train'),
                  key=lambda row: (row['camera_id'], int(row['frame_index'])))
    expected = {(camera, frame) for camera in CAMERAS for frame in FRAMES}
    if len(rows) != 1140 or {(row['camera_id'], int(row['frame_index'])) for row in rows} != expected:
        raise ValueError('Expected all and only cam02–20 / even frames 0–118')
    return rows


def parent_camera(manifest, obs, uid=0):
    """Construct a camera without reading any pixel file."""
    from n3dv_data import observation_to_4dgs_camera
    calibration = manifest['cameras'][obs['camera_id']]
    item = dict(image=torch.zeros(3, *LR_SIZE), time=float(obs['time']),
        camera_id=obs['camera_id'], frame_index=int(obs['frame_index']),
        width=LR_SIZE[1], height=LR_SIZE[0], K=np.asarray(calibration['K_lr'], np.float64),
        w2c=np.asarray(calibration['w2c'], np.float64))
    from common import resized_camera
    return resized_camera(observation_to_4dgs_camera(item, uid), *HR_SIZE)


def export_parent(protocol=None, out=None):
    """GPU entry: exactly 1140 auxiliary moment forwards, zero updates/RGB."""
    protocol = Path(protocol or OUT/'protocol.json'); out = Path(out or OUT/'support/parent_moments_hr')
    out.mkdir(parents=True, exist_ok=True)
    shared, policy = setup()
    from n3dv_data import load_manifest
    from motion_model import load_model
    gm = module('fp_support_depth', ROOT/'experiments/dynamic_sr_confidence_geometry_20261006/depth_prior.py')
    registration = read(protocol); manifest = load_manifest(bound(registration['manifest']))
    observations = legal_records(manifest)
    identity = dict(parent=registration['parent'], manifest=registration['manifest'], hr_size=list(HR_SIZE),
        quantity='raw unclamped HR [A,M1,M2]; no reconstructed HR from normalized LR',
        source_sha256=sha(__file__), renderer_source_sha256=sha(gm.__file__),
        training_images_opened=0, privileged_train_hr=False, sources=moment_source_identity())
    config = out/'config.json'
    if config.exists() and read(config) != identity:
        raise ValueError('HR parent cache identity changed; use a new output directory')
    write(config, identity)
    model = load_model(bound(registration['parent']), manifest)
    if model.checkpoint['metadata'].get('intervention_step') != 6000:
        raise ValueError('Expected complete U6000 parent')
    torch.set_num_threads(4); torch.cuda.reset_peak_memory_stats()
    started = time.monotonic(); rows = []; rendered = reused = 0
    with torch.no_grad():
        for n, obs in enumerate(observations):
            path = out/f"{obs['camera_id']}_{int(obs['frame_index']):04d}.npz"
            receipt = path.with_suffix('.json')
            if path.exists() and receipt.exists():
                row = read(receipt)
                if sha(path) != row['sha256']:
                    raise ValueError(f'HR parent cache corruption: {path}')
                reused += 1
            else:
                tick = time.monotonic(); camera = parent_camera(manifest, obs, n)
                state = policy.effective_state(model, camera.time)
                result = gm.render_moments(camera, state['xyz'], state['cov'], state['opacity'])
                raw = result['hr_moments']
                if tuple(raw.shape) != (3, *HR_SIZE):
                    raise ValueError('Native renderer did not return HR moments')
                atomic_npz(path, hr_moments=raw.detach().cpu().numpy().astype(np.float32))
                torch.cuda.synchronize(); rendered += 1
                row = dict(camera=obs['camera_id'], frame=int(obs['frame_index']), path=path.name,
                    sha256=sha(path), time=float(camera.time), hr_size=list(HR_SIZE),
                    moment_forwards=1, rgb_forwards=0, parameter_updates=0,
                    seconds=time.monotonic()-tick,
                    nonfinite_moment_elements=int((~torch.isfinite(raw)).sum()))
                write(receipt, row)
            rows.append(row)
            if (n+1)%60 == 0:
                print(json.dumps(dict(stage='export_HR_parent_moments', entries=n+1, rendered=rendered,
                    reused=reused, wall_s=time.monotonic()-started)), flush=True)
    if moment_source_identity() != identity['sources']:
        raise ValueError('Parent HR preparation source changed during export')
    final = dict(status='completed_HR_parent_moments', identity=identity, entries=rows,
        manifest_sha256=registration['manifest']['sha256'], parent_sha256=registration['parent']['sha256'],
        total_moment_forwards=len(rows), new_moment_forwards=rendered, reused_entries=reused,
        rgb_forwards=0, parameter_updates=0, seconds=time.monotonic()-started,
        per_entry_seconds_sum=sum(row['seconds'] for row in rows),
        peak_gb=torch.cuda.max_memory_allocated()/1e9, gpu=torch.cuda.get_device_name(),
        physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'), torch=str(torch.__version__))
    write(out/'index.json', final)
    return out/'index.json'


def camera_dict(manifest, camera):
    calibration = manifest['cameras'][camera]
    return dict(K=np.asarray(calibration['K_hr'], np.float64),
                w2c=np.asarray(calibration['w2c'], np.float64),
                c2w=np.asarray(calibration['c2w'], np.float64))


def required_edges(tables):
    edges = set()
    for table in tables:
        keys = table['record_keys']
        rows = table.get('sync_rows', table.get('rows_sync'))
        if rows is None:
            rows = table['rows']
        rows = list(rows)
        from schedules import calibration_triplets
        rows.extend(calibration_triplets(table))
        for row in rows:
            if isinstance(row, dict): row = row['indices']
            triplet = [tuple(keys[int(i)]) for i in row]
            if len({k[0] for k in triplet}) != 3 or len({int(k[1]) for k in triplet}) != 1:
                raise ValueError('Frozen X cache requires registered same-time camera triplets')
            for i in range(3):
                for j in range(3):
                    if i != j: edges.add((triplet[i][0], triplet[j][0], int(triplet[j][1])))
    return sorted(edges, key=lambda key: (key[2], key[0], key[1]))


def prepare(parent_index=None, schedules=None, out=None, device='cpu'):
    """Prepare frozen support; GPU optional, with no renderer or model updates."""
    import footprint
    parent_index = Path(parent_index or OUT/'support/parent_moments_hr/index.json')
    schedules = [Path(path) for path in (schedules or [OUT/'schedules/schedule_1.json', OUT/'schedules/schedule_2.json'])]
    out = Path(out or OUT/'support/frozen'); out.mkdir(parents=True, exist_ok=True)
    parent = read(parent_index)
    if parent['status'] != 'completed_HR_parent_moments':
        raise ValueError('Full HR parent export is not complete')
    protocol = read(OUT/'protocol.json')
    if parent['parent_sha256'] != protocol['parent']['sha256'] or parent['manifest_sha256'] != protocol['manifest']['sha256']:
        raise ValueError('HR parent source changed')
    manifest = read(bound(protocol['manifest'])); legal_records(manifest)
    rows = {(row['camera'], int(row['frame'])): row for row in parent['entries']}
    if set(rows) != {(camera, frame) for camera in CAMERAS for frame in FRAMES}:
        raise ValueError('HR parent cache does not cover all legal observations')
    tables = [read(path) for path in schedules]; edges = required_edges(tables)
    identity = dict(parent_index=entry(parent_index), schedules=[entry(path) for path in schedules],
        rules=RULES, source_sha256=sha(__file__), footprint_source_sha256=sha(footprint.__file__),
        train76_keys=[[camera, frame] for camera in CAMERAS for frame in CALIBRATION_FRAMES],
        edge_count=len(edges), training_images_opened=0, privileged_train_hr=False)
    if (out/'config.json').exists() and read(out/'config.json') != identity:
        raise ValueError('Frozen support registration changed; use a new output directory')
    write(out/'config.json', identity)
    started = time.monotonic(); verified = set()
    def load(key, selected_device='cpu'):
        row = rows[key]; path = parent_index.parent/row['path']
        if key not in verified:
            if sha(path) != row['sha256']: raise ValueError(f'HR parent bytes changed: {path}')
            verified.add(key)
        with np.load(path, allow_pickle=False) as data:
            if 'hr_moments' not in data or data['hr_moments'].shape != (3, *HR_SIZE):
                raise ValueError('LR normalized statistics cannot substitute for HR raw moments')
            return normalize_hr_moments(torch.from_numpy(data['hr_moments']).to(selected_device))
    tau_path = out/'tau_registration.json'
    if tau_path.exists():
        tau_record = read(tau_path)
        if tau_record['config_sha256'] != sha(out/'config.json'): raise ValueError('tau registration identity changed')
        tau = float(tau_record['tau_z'])
    else:
        positive = []; counts = []
        for camera in CAMERAS:
            for frame in CALIBRATION_FRAMES:
                result = load((camera, frame))
                valid = result['valid_hr'] & result['known_variance'] & torch.isfinite(result['cz']) & (result['cz'] > 0)
                values = result['cz'][valid].numpy(); positive.append(values)
                counts.append(dict(camera=camera, frame=frame, positive_finite_count=len(values),
                    unknown_variance_count=int((~result['known_variance']).sum())))
        total = sum(len(array) for array in positive)
        quantile = float(np.quantile(np.concatenate(positive), .75)) if total else None
        tau = max(quantile or 0., 1e-6)
        tau_record = dict(status='frozen_train76_quantile', config_sha256=sha(out/'config.json'),
            tau_z=tau, raw_positive_cz_q75=quantile, positive_finite_count=total,
            fallback_no_positive_finite=total == 0, rows=counts, parameter_updates=0,
            rgb_forwards=0, moment_forwards=0, seconds=time.monotonic()-started)
        write(tau_path, tau_record)
        del positive
    entries = []; observations = []; obs_seen = set(); regenerated = 0
    # Time-major preparation plus an eight-observation LRU bounds live raw
    # HR storage. It never retains 1140 full-frame parent tensors in memory.
    for frame in FRAMES:
        frame_edges = [edge for edge in edges if edge[2] == frame]
        needed = sorted({camera for source, target, _ in frame_edges for camera in (source, target)})
        frame_cache = OrderedDict()
        def frame_value(camera):
            if camera not in frame_cache:
                value = load((camera, frame), device)
                frame_cache[camera] = (value, dispersion_weight(value, tau))
                if len(frame_cache) > 8: frame_cache.popitem(last=False)
            value = frame_cache.pop(camera); frame_cache[camera] = value
            return value
        for camera in needed:
            path = out/'observations'/f'{camera}_{frame:04d}.npz'; receipt = path.with_suffix('.json')
            if path.exists() and receipt.exists():
                row = read(receipt)
                if sha(path) != row['sha256']: raise ValueError('Observation support cache corruption')
            else:
                value, _ = frame_value(camera)
                atomic_npz(path, valid_hr_packed=pack_mask(value['valid_hr']), shape=np.array(HR_SIZE, np.int32))
                row = dict(camera=camera, frame=frame, path=str(path.relative_to(out)), sha256=sha(path),
                    valid_fraction=float(value['valid_hr'].float().mean()),
                    unknown_variance_fraction=float((~value['known_variance']).float().mean()))
                write(receipt, row)
            observations.append(row); obs_seen.add((camera, frame))
        for source, target, _ in frame_edges:
            path = out/'edges'/f'{source}_to_{target}_{frame:04d}.npz'; receipt = path.with_suffix('.json')
            if path.exists() and receipt.exists():
                row = read(receipt)
                if sha(path) != row['sha256']: raise ValueError('Frozen edge cache corruption')
            else:
                tick = time.monotonic()
                src, source_weight = frame_value(source)
                tgt, target_weight = frame_value(target)
                projected = footprint.project(camera_dict(manifest, source), camera_dict(manifest, target), tgt['z'])
                # These statistics and their coordinates are frozen. Current
                # depth never re-samples source w or source visibility.
                # Nonnegative area and bilinear weights act on raw physical
                # moments first. Averaging normalized per-ray sigma would
                # omit variance between distinct surface means in one source
                # footprint and could falsely declare that footprint thin.
                raw_source = torch.where(torch.isfinite(src['raw_moments']), src['raw_moments'],
                                         torch.zeros_like(src['raw_moments']))
                # w stays an independently mapped fixed soft field. Unknown
                # M2 is made finite for the sampler but its separate indicator
                # prevents the substitute from becoming occlusion evidence.
                mapped = footprint.mip_sample(torch.cat((raw_source, source_weight[None],
                    src['known_variance'].float()[None]), 0), projected['xy'], source_valid=src['valid_hr'])
                physical_source = normalize_hr_moments(mapped['rgb'][:3])
                sz, ssigma, scz = physical_source['z'], physical_source['variance'].sqrt(), physical_source['cz']
                sw, known = mapped['rgb'][3], mapped['rgb'][4]
                concentrated = (known > 1 - 1e-6) & physical_source['valid_hr'] & (scz <= tau) & tgt['known_variance'] & (tgt['cz'] <= tau)
                tolerance = torch.maximum(.02 * sz, 3 * ssigma)
                occluded = concentrated & (projected['source_z'] > sz + tolerance)
                mask = (projected['valid'] & tgt['valid_hr'] & mapped['valid'] & ~occluded).detach()
                weight_hr = torch.minimum(sw.clamp(.25, 1.), target_weight)
                weight_lr = F.interpolate(weight_hr[None, None], size=LR_SIZE, mode='area')[0, 0]
                valid_lr = footprint.degradation_support(mask, LR_SIZE)
                atomic_npz(path, mask_hr_packed=pack_mask(mask), weight_lr=weight_lr.cpu().numpy().astype(np.float32),
                    shape=np.array(HR_SIZE, np.int32))
                row = dict(source=source, target=target, frame=frame, path=str(path.relative_to(out)),
                    sha256=sha(path), frozen_valid_lr_fraction=float(valid_lr.float().mean()),
                    valid_hr_fraction=float(mask.float().mean()), parent_occlusion_fraction=float(occluded.float().mean()),
                    projection_invalid_fraction=float((~projected['valid']).float().mean()),
                    unknown_variance_fraction=float((~tgt['known_variance']).float().mean()),
                    weight_mean=float(weight_lr.mean()),
                    weight_quantiles=[float(x) for x in torch.quantile(weight_lr.float(), weight_lr.new_tensor([.25, .5, .75]).float())],
                    seconds=time.monotonic()-tick, parameter_updates=0, rgb_forwards=0, moment_forwards=0)
                write(receipt, row); regenerated += 1
            entries.append(row)
        del frame_cache
        print(json.dumps(dict(stage='frozen_X_support', frame=frame, completed_edges=len(entries),
            total_edges=len(edges), new_edges=regenerated, wall_s=time.monotonic()-started)), flush=True)
    # Every legal source observation is required by the full schedules.
    if obs_seen != set(rows): raise ValueError('Schedule did not cover all legal source observations')
    result = dict(status='completed_frozen_X_support', identity=identity, tau_registration=entry(tau_path),
        tau_z=tau, observations=observations, entries=entries,
        all_parent_raw_bytes_verified=len(verified), new_edges=regenerated,
        parameter_updates=0, rgb_forwards=0, moment_forwards=0, seconds=time.monotonic()-started,
        compute_device=str(device), actual_hr_depth=True, soft_weights_frozen=True,
        support_truth_claim=False, parent_HR_LRU_capacity=8)
    write(out/'index.json', result)
    return out/'index.json'


class SupportCache:
    """Lazy byte-verified edge cache; CPU LRU avoids unbounded GPU ownership."""
    def __init__(self, index=None, edge_capacity=12, observation_capacity=6):
        self.path = Path(index or OUT/'support/frozen/index.json'); self.index = read(self.path)
        if self.index['status'] != 'completed_frozen_X_support' or not self.index['soft_weights_frozen']:
            raise ValueError('Support cache not complete and frozen')
        self.edges = {(row['source'], row['target'], int(row['frame'])): row for row in self.index['entries']}
        self.observations = {(row['camera'], int(row['frame'])): row for row in self.index['observations']}
        self.edge_capacity = edge_capacity; self.observation_capacity = observation_capacity
        self.edge_lru = OrderedDict(); self.obs_lru = OrderedDict(); self.verified = set(); self.opened = {}

    def training_files(self):
        return [self.path.parent/row['path'] for row in self.index['observations'] + self.index['entries']]

    def training_entries(self):
        """Guard identities without rehashing every 4 GB cache for every arm.

        Cache reads still verify each exact NPZ lazily before its first use.
        """
        return [dict(path=str((self.path.parent/row['path']).relative_to(ROOT)), sha256=row['sha256'])
                for row in self.index['observations'] + self.index['entries']]

    def _load(self, row, lru, key, capacity):
        if key not in lru:
            path = self.path.parent/row['path']
            if row['path'] not in self.verified:
                if sha(path) != row['sha256']: raise ValueError(f'Frozen support bytes changed: {path}')
                self.verified.add(row['path'])
            with np.load(path, allow_pickle=False) as data:
                result = {name: torch.from_numpy(data[name].copy()) for name in data.files if name not in ('shape', 'mask_hr_packed', 'valid_hr_packed')}
                for packed, plain in [('mask_hr_packed', 'mask_hr'), ('valid_hr_packed', 'valid_hr')]:
                    if packed in data: result[plain] = torch.from_numpy(unpack_mask(data[packed], data['shape']))
            lru[key] = result; self.opened[str(path.resolve())] = row['sha256']
            if len(lru) > capacity: lru.popitem(last=False)
        result = lru.pop(key); lru[key] = result
        return result

    def observation(self, camera, frame, device=None):
        key = (camera, int(frame))
        if key not in self.observations: raise ValueError(f'Illegal source observation: {key}')
        result = self._load(self.observations[key], self.obs_lru, key, self.observation_capacity)
        return result if device is None else {name: value.to(device) for name, value in result.items()}

    def edge(self, source_key, target_key, device=None):
        if int(source_key[1]) != int(target_key[1]): raise ValueError('X edge must be same-time')
        key = (source_key[0], target_key[0], int(target_key[1]))
        if key not in self.edges: raise ValueError(f'Unregistered X edge: {key}')
        row = self.edges[key]; result = self._load(row, self.edge_lru, key, self.edge_capacity)
        if device is not None: result = {name: value.to(device) for name, value in result.items()}
        return dict(result, frozen_valid_lr_fraction=row['frozen_valid_lr_fraction'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', required=True, choices=['export', 'prepare', 'all'])
    parser.add_argument('--protocol', type=Path, default=OUT/'protocol.json')
    parser.add_argument('--parent-index', type=Path, default=OUT/'support/parent_moments_hr/index.json')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    if args.mode in ('export', 'all'): export_parent(args.protocol, args.parent_index.parent)
    if args.mode in ('prepare', 'all'): prepare(args.parent_index, out=args.out, device=args.device)


if __name__ == '__main__': main()
