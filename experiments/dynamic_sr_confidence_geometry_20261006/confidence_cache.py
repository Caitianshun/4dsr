"""Frozen train-only LR closure and same-time two-neighbour confidence.

The cache is indexed by (camera, original_frame), never by a persistent Gaussian.
HR files and their derivatives are not opened. Teacher PNGs are legal frozen LR
SwinIR products. The residual H(T) below is a diagnostic descriptor, not Q.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
import csv
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / 'data/dynamic_sr/n3dv_prepared/cook_spinach/manifest.json'
DEFAULT_TEACHERS = ROOT / 'output/dynamic_sr_detail_supervision_20260924/teacher_inventory_v1/cook_spinach/teacher_index.json'
DEFAULT_OUT = ROOT / 'output/dynamic_sr_confidence_geometry_20261006/cache/confidence'
CAMERAS = tuple(f'cam{i:02d}' for i in range(2, 21))
FRAMES = tuple(range(0, 120, 2))
TRAIN76_FRAMES = (0, 40, 80, 118)
POLICY = dict(schema=1, coarse_factor=2, maximum_neighbours=2,
              alpha_minimum=0.1, depth_relative_tolerance=0.05,
              forward_backward_maximum_coarse_pixels=1.0,
              census_radius=2, census_maximum_fraction=0.25,
              photometric_maximum_rgb_mae=0.05,
              texture_minimum_local_std=1/255,
              unknown_confidence=0.5, numeric_scale_floor=1/255,
              closure_patch=3, detail_error_patch=3,
              dynamic_proxy_rgb_mae=0.02,
              footprint='finite-difference projection Jacobian; anti-aliased mip sampling with max singular value',
              scale_calibration='pixel median on fixed train76 only; valid neighbour errors averaged before median',
              neighbour_selection='two nearest legal training camera centres, camera_id breaks ties',
              interpretation='heuristic evidence, not calibrated probability; low texture is unknown')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    tmp.replace(path)


def resize_chw(tensor, shape, mode='bilinear', antialias=False):
    args = {} if mode in ('area', 'nearest') else dict(align_corners=False)
    if antialias:
        args['antialias'] = True
    return F.interpolate(tensor[None], size=shape, mode=mode, **args)[0]


def pool3(value):
    # Replicated image boundaries, fixed for every observation.
    return F.avg_pool2d(F.pad(value[None, None], (1, 1, 1, 1), mode='replicate'), 3, stride=1)[0, 0]


def image_tensor(path):
    with Image.open(path) as im:
        a = np.array(im.convert('RGB'), copy=True)
    return torch.from_numpy(a).permute(2, 0, 1).float().div(255)


class LegalInputs:
    """Validate the precise training boundary before opening any pixel file."""
    def __init__(self, manifest, teacher_index):
        self.manifest_path, self.teacher_path = Path(manifest).resolve(), Path(teacher_index).resolve()
        self.manifest = read(self.manifest_path)
        self.root = self.manifest_path.parent
        self.shape_lr = tuple(reversed(self.manifest['resolutions']['lr']))
        self.shape_hr = tuple(reversed(self.manifest['resolutions']['hr']))
        if self.shape_lr != (252, 336) or self.shape_hr != (1008, 1344):
            raise ValueError('This preregistered short-window cache requires 252x336 LR and x4 teacher')
        if tuple(self.manifest['splits']['train']) != CAMERAS:
            raise ValueError('Training camera protocol changed')
        records = [r for r in self.manifest['observations'] if r['split'] == 'train']
        self.records = {(r['camera_id'], int(r['frame_index'])): r for r in records}
        self.keys = tuple((c, f) for c in CAMERAS for f in FRAMES)
        if len(records) != 1140 or set(self.records) != set(self.keys):
            raise ValueError('Exactly 1140 legal training observations required')
        teachers = read(self.teacher_path)
        if teachers['manifest_sha256'] != sha(self.manifest_path) or not teachers.get('status', '').startswith('completed'):
            raise ValueError('Frozen teacher manifest/status mismatch')
        self.teachers = {(r['camera'], int(r['frame'])): r for r in teachers['entries']}
        if len(teachers['entries']) != 1140 or set(self.teachers) != set(self.keys):
            raise ValueError('Teacher must cover precisely the legal training boundary')
        self.verified = set()
        self.opened = {}

    def pixels(self, key, teacher=False):
        if key not in self.records:
            raise ValueError(f'Illegal observation: {key}')
        row = self.teachers[key] if teacher else self.records[key]
        rel = row['relative_path'] if teacher else row['lr_path']
        path = (self.root / rel).resolve()
        expected = row['sha256'] if teacher else row['lr_sha256']
        # Permit only the exact registry path: no HR image or held-out camera.
        if teacher and row['lr_sha256'] != self.records[key]['lr_sha256']:
            raise ValueError('Teacher LR identity mismatch')
        token = (path, expected)
        if token not in self.verified:
            if sha(path) != expected:
                raise ValueError(f'Input hash mismatch: {path}')
            self.verified.add(token)
        self.opened[str(path.relative_to(self.root))] = expected
        return image_tensor(path)

    def identity(self):
        return dict(manifest_sha256=sha(self.manifest_path), teacher_index_sha256=sha(self.teacher_path),
                    train_cameras=list(CAMERAS), frames=list(FRAMES), entries=1140,
                    hr_images_read=False, hr_depth_read=False, hr_flow_read=False,
                    hr_error_or_masks_used=False, current_training_error_used=False,
                    old_T_weight_used=False)


def teacher_features(teacher, lr, shape_coarse):
    down = resize_chw(teacher, lr.shape[-2:], mode='bicubic', antialias=True).clamp(0, 1)
    e_lr = pool3((down - lr).abs().mean(0))
    # H(T) uses the actual unquantized RGB degradation; it is not Q.
    h = teacher - resize_chw(down, teacher.shape[-2:], mode='bicubic', antialias=True)
    detail = resize_chw(h, shape_coarse, mode='bicubic', antialias=True)
    coarse_lr = resize_chw(lr, shape_coarse, mode='bicubic', antialias=True).clamp(0, 1)
    return e_lr.numpy(), detail.permute(1, 2, 0).numpy(), coarse_lr.permute(1, 2, 0).numpy()


def feature_cache(inputs, out):
    """CPU preparation can precede frozen U6000 geometry export."""
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    identity = dict(**inputs.identity(), policy=POLICY, source_sha256=sha(__file__), torch=torch.__version__)
    config = out / 'config.json'
    if config.exists() and read(config) != identity:
        raise ValueError('Feature cache identity changed; use a new output directory')
    write(config, identity)
    rows = []; t0 = time.monotonic()
    coarse = tuple(x // POLICY['coarse_factor'] for x in inputs.shape_lr)
    for n, key in enumerate(inputs.keys):
        camera, frame = key; path = out / f'{camera}_{frame:04d}.npz'; receipt = path.with_suffix('.json')
        if path.exists() and receipt.exists():
            row = read(receipt)
            if row['sha256'] != sha(path):
                raise ValueError(f'Feature cache corruption: {path}')
        else:
            e_lr, detail, lr = teacher_features(inputs.pixels(key, True), inputs.pixels(key), coarse)
            np.savez_compressed(path, e_lr=e_lr, detail=detail, lr=lr)
            row = dict(camera=camera, frame=frame, path=path.name, sha256=sha(path),
                       lr_sha256=inputs.records[key]['lr_sha256'], teacher_sha256=inputs.teachers[key]['sha256'])
            write(receipt, row)
        rows.append(row)
        if (n + 1) % 60 == 0:
            print(json.dumps(dict(stage='legal_teacher_features', entries=n+1, seconds=time.monotonic()-t0)), flush=True)
    write(out / 'index.json', dict(status='completed', entries=rows, identity=identity,
                                 seconds=time.monotonic()-t0, parameter_updates=0))
    return {(r['camera'], r['frame']): out / r['path'] for r in rows}


def coarse_calibration(cal, factor):
    k = np.asarray(cal['K_lr'], np.float64).copy()
    # Pixel-centre indices follow half-pixel resizing, including principal point.
    scale = np.array([[1/factor, 0, (1/factor-1)/2], [0, 1/factor, (1/factor-1)/2], [0, 0, 1]])
    return dict(K=scale @ k, w2c=np.asarray(cal['w2c'], np.float64), c2w=np.asarray(cal['c2w'], np.float64))


def world_from_depth(depth, cal, xy=None):
    h, w = depth.shape
    if xy is None:
        yy, xx = np.indices((h, w), dtype=np.float64); xy = np.stack((xx, yy), -1)
    rays = np.concatenate((xy, np.ones((*xy.shape[:-1], 1))), -1) @ np.linalg.inv(cal['K']).T
    xyz = rays * depth[..., None]
    return np.concatenate((xyz, np.ones((*xyz.shape[:-1], 1))), -1) @ cal['c2w'].T


def project_world(world, cal):
    camera = world @ cal['w2c'].T
    screen = camera[..., :3] @ cal['K'].T
    z = camera[..., 2]
    # Invalid z remains invalid downstream; avoid infinities in cv2.remap.
    den = np.where(np.abs(screen[..., 2]) > 1e-12, screen[..., 2], np.nan)
    return screen[..., :2] / den[..., None], z


def sample(array, xy, interpolation=cv2.INTER_LINEAR):
    safe = np.nan_to_num(xy, nan=-1e6, posinf=-1e6, neginf=-1e6).astype(np.float32)
    return cv2.remap(array.astype(np.float32), safe[..., 0], safe[..., 1], interpolation,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def projection_footprint(xy):
    """Largest singular value of d(source_pixel)/d(target_pixel)."""
    dy = np.stack([np.gradient(xy[..., i], axis=0) for i in range(2)], -1)
    dx = np.stack([np.gradient(xy[..., i], axis=1) for i in range(2)], -1)
    a = np.sum(dx*dx, -1); d = np.sum(dy*dy, -1); b = np.sum(dx*dy, -1)
    largest = .5*(a+d+np.sqrt(np.maximum((a-d)**2 + 4*b*b, 0)))
    return np.nan_to_num(np.sqrt(np.maximum(largest, 0)), nan=1, posinf=32).clip(1, 32).astype(np.float32)


def mip_sample(array, xy, footprint):
    """Linear blend between area-prefiltered levels; half-pixel grid conversion."""
    lod = np.log2(footprint.clip(1, 32)); result = np.zeros((*xy.shape[:-1], *array.shape[2:]), np.float32)
    total = np.zeros(xy.shape[:-1], np.float32)
    for level in range(6):
        weight = np.maximum(1-np.abs(lod-level), 0).astype(np.float32)
        if not np.any(weight > 0):
            continue
        scale = 2**level
        h, w = array.shape[:2]
        shape = (max(1, int(np.ceil(w/scale))), max(1, int(np.ceil(h/scale))))
        small = cv2.resize(array, shape, interpolation=cv2.INTER_AREA) if level else array
        # Last odd mip dimensions use their actual scale, not a rounded power.
        ratio = np.array([small.shape[1]/w, small.shape[0]/h], np.float64)
        mapped = (xy+.5)*ratio-.5
        warped = sample(small, mapped)
        result += warped*weight[..., None] if array.ndim == 3 else warped*weight
        total += weight
    return result / np.maximum(total[..., None] if array.ndim == 3 else total, 1e-8)


def local_std(gray):
    mean = cv2.boxFilter(gray, -1, (5, 5), borderType=cv2.BORDER_REFLECT_101)
    mean2 = cv2.boxFilter(gray*gray, -1, (5, 5), borderType=cv2.BORDER_REFLECT_101)
    return np.sqrt(np.maximum(mean2-mean*mean, 0))


def census_error(gray, warped):
    """Compare actual aligned LR intensities, no SR descriptor validity test."""
    radius = POLICY['census_radius']; h, w = gray.shape
    a = np.pad(gray, radius, mode='reflect'); b = np.pad(warped, radius, mode='reflect')
    error = np.zeros_like(gray)
    count = 0
    for y in range(-radius, radius+1):
        for x in range(-radius, radius+1):
            if x == 0 and y == 0:
                continue
            aa = a[radius+y:radius+y+h, radius+x:radius+x+w] > gray
            bb = b[radius+y:radius+y+h, radius+x:radius+x+w] > warped
            error += aa != bb; count += 1
    return error/count


def aggregate_parent(depth, alpha):
    # Re-form first moment before area aggregation; do not area-average Z itself.
    factor = POLICY['coarse_factor']; h, w = depth.shape
    size = (w//factor, h//factor)
    a = cv2.resize(alpha.astype(np.float32), size, interpolation=cv2.INTER_AREA)
    m1 = cv2.resize(np.where(np.isfinite(depth), depth, 0).astype(np.float32)*alpha, size, interpolation=cv2.INTER_AREA)
    z = m1 / np.maximum(a, 1e-6)
    return z, a


def neighbour_evidence(target, source, target_cal, source_cal):
    """Return detail error and *known* support, with explicit rejected causes."""
    zt, at = target['depth'], target['alpha']; zs, ass = source['depth'], source['alpha']
    h, w = zt.shape; yy, xx = np.indices((h, w)); pixel = np.stack((xx, yy), -1)
    world = world_from_depth(zt, target_cal); xy, projected_z = project_world(world, source_cal)
    source_z = sample(zs, xy); source_alpha = sample(ass, xy)
    positive = (zt > 0) & (projected_z > 0) & (source_z > 0)
    finite = np.isfinite(xy).all(-1) & np.isfinite(projected_z) & np.isfinite(zt)
    inside = (xy[..., 0] >= 2) & (xy[..., 0] <= w-3) & (xy[..., 1] >= 2) & (xy[..., 1] <= h-3)
    supported = (at >= POLICY['alpha_minimum']) & (source_alpha >= POLICY['alpha_minimum'])
    tolerance = POLICY['depth_relative_tolerance'] * np.maximum(np.abs(projected_z), np.abs(source_z))
    occlusion = np.abs(projected_z-source_z) <= np.maximum(tolerance, 1e-6)
    back_world = world_from_depth(source_z, source_cal, xy)
    back_xy, back_z = project_world(back_world, target_cal)
    fb = np.linalg.norm(back_xy-pixel, axis=-1) <= POLICY['forward_backward_maximum_coarse_pixels']
    target_depth_agrees = np.abs(back_z-zt) <= POLICY['depth_relative_tolerance']*np.maximum(np.abs(back_z), np.abs(zt))
    geometric = positive & finite & inside & supported & occlusion & fb & target_depth_agrees
    footprint = projection_footprint(xy)
    aligned_lr = mip_sample(source['lr'], xy, footprint)
    gray = target['lr'].mean(-1); aligned_gray = aligned_lr.mean(-1)
    textured = (local_std(gray) >= POLICY['texture_minimum_local_std']) & (local_std(aligned_gray) >= POLICY['texture_minimum_local_std'])
    photo = np.abs(target['lr']-aligned_lr).mean(-1)
    census = census_error(gray, aligned_gray)
    observed = (photo <= POLICY['photometric_maximum_rgb_mae']) & (census <= POLICY['census_maximum_fraction'])
    # Require a full valid descriptor footprint, so padded/occluded pixels cannot
    # create apparent Census or detail agreement at their immediate boundaries.
    r = POLICY['census_radius']; kernel = np.ones((2*r+1, 2*r+1), np.uint8)
    geometric_patch = cv2.erode(geometric.astype(np.uint8), kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    valid = geometric_patch & textured & observed
    aligned_detail = mip_sample(source['detail'], xy, footprint)
    error = np.abs(target['detail']-aligned_detail).mean(-1)
    error = cv2.boxFilter(error.astype(np.float32), -1, (3, 3), borderType=cv2.BORDER_REFLECT_101)
    valid &= np.isfinite(error)
    stats = dict(positive_fraction=float(positive.mean()), fov_fraction=float(inside.mean()),
                 alpha_fraction=float(supported.mean()), geometric_fraction=float(geometric_patch.mean()),
                 low_texture_unknown_fraction=float((geometric_patch & ~textured).mean()),
                 photometric_supported_fraction=float((geometric_patch & textured & observed).mean()),
                 valid_fraction=float(valid.mean()), mean_footprint=float(footprint[geometric].mean()) if geometric.any() else None)
    return error, valid, stats


def nearest_cameras(manifest):
    centres = {c: np.asarray(manifest['cameras'][c]['c2w'])[:3, 3] for c in CAMERAS}
    return {c: sorted((d for d in CAMERAS if d != c), key=lambda d: (float(np.linalg.norm(centres[c]-centres[d])), d))[:2]
            for c in CAMERAS}


def statistics(weight):
    w = np.asarray(weight, np.float64).ravel(); sq = float(np.square(w).sum())
    return dict(mean_weight=float(w.mean()), kish_effective_fraction=float(w.sum()**2/(w.size*sq)) if sq else 0,
                minimum_weight=float(w.min()), maximum_weight=float(w.max()),
                fraction_below_0_1=float((w < .1).mean()), fraction_below_0_25=float((w < .25).mean()))


def confidence_from_errors(e_lr, e_view, valid_count, scales, shape_hr):
    cl = np.exp(-e_lr/scales['s_lr']).astype(np.float32)
    cv = np.where(valid_count > 0, np.exp(-e_view/scales['s_view']), POLICY['unknown_confidence']).astype(np.float32)
    l = resize_chw(torch.from_numpy(cl)[None], shape_hr)[0]
    v = resize_chw(torch.from_numpy(cv)[None], shape_hr)[0]
    return cl, cv, (l*v).numpy()


def build_cache(inputs, parent_index, out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    features = feature_cache(inputs, out / 'features')
    parent_path = Path(parent_index).resolve(); parent = read(parent_path)
    rows = parent.get('entries', parent.get('rows', []))
    parents = {(r['camera'], int(r['frame'])): r for r in rows}
    if len(rows) != 1140 or set(parents) != set(inputs.keys):
        raise ValueError('Frozen parent normalized-depth cache needs exactly all 1140 legal observations')
    if not parent.get('status', '').startswith('completed'):
        raise ValueError('Parent geometry export is incomplete')
    if parent.get('manifest_sha256', parent.get('manifest', {}).get('sha256')) != sha(inputs.manifest_path):
        raise ValueError('Parent geometry manifest identity mismatch')
    identity = dict(**inputs.identity(), parent_index_sha256=sha(parent_path), parent=parent.get('parent'),
                    parent_sha256=parent.get('parent_sha256'),
                    policy=POLICY, source_sha256=sha(__file__), torch=torch.__version__)
    config = out / 'config.json'
    if config.exists() and read(config) != identity:
        raise ValueError('Frozen confidence identity changed; use a new cache directory')
    write(config, identity)
    cals = {c: coarse_calibration(inputs.manifest['cameras'][c], POLICY['coarse_factor']) for c in CAMERAS}
    neighbours = nearest_cameras(inputs.manifest); write(out / 'neighbours.json', neighbours)
    store = OrderedDict()
    def get(key):
        if key in store:
            value = store.pop(key)
        else:
            with np.load(features[key], allow_pickle=False) as data:
                value = {name: data[name] for name in ('e_lr', 'detail', 'lr')}
            row = parents[key]; path = Path(row['path'])
            if not path.is_absolute():
                path = parent_path.parent / path
            if sha(path) != row['sha256']:
                raise ValueError(f'Frozen geometry hash mismatch: {path}')
            with np.load(path, allow_pickle=False) as z:
                depth, alpha = z['depth_lr'], z['alpha_lr']
                if depth.shape != inputs.shape_lr or alpha.shape != inputs.shape_lr:
                    raise ValueError('Parent geometry must use normalized camera-z on the native LR grid')
                if not np.isfinite(alpha).all() or (alpha < -1e-5).any():
                    raise ValueError('Parent alpha violates nonnegative moment construction')
                value['depth'], value['alpha'] = aggregate_parent(depth, alpha)
        store[key] = value
        while len(store) > 32:
            store.popitem(last=False)
        return value
    errors = out / 'correspondence'; errors.mkdir(exist_ok=True)
    stats_rows = []; t0 = time.monotonic()
    # Same-time frame-major order permits CPU LRU reuse, with fixed neighbours.
    for frame in FRAMES:
        if len({inputs.records[c, frame]['time'] for c in CAMERAS}) != 1:
            raise ValueError('Same-time camera observations disagree on timestamp')
        for camera in CAMERAS:
            path = errors / f'{camera}_{frame:04d}.npz'; receipt = path.with_suffix('.json')
            if path.exists() and receipt.exists():
                row = read(receipt)
                if sha(path) != row['sha256']:
                    raise ValueError('Correspondence error cache corruption')
            else:
                target = get((camera, frame)); accum = np.zeros(target['depth'].shape, np.float32)
                count = np.zeros(target['depth'].shape, np.uint8); edge_stats = []
                for other in neighbours[camera]:
                    error, valid, edge = neighbour_evidence(target, get((other, frame)), cals[camera], cals[other])
                    accum += np.where(valid, error, 0); count += valid.astype(np.uint8)
                    edge_stats.append(dict(neighbour=other, **edge))
                mean_error = accum/np.maximum(count, 1)
                np.savez_compressed(path, e_view=mean_error, valid_count=count)
                row = dict(camera=camera, frame=frame, path=path.name, sha256=sha(path), neighbours=neighbours[camera],
                           edges=edge_stats, valid_fraction=float((count > 0).mean()), unknown_fraction=float((count == 0).mean()),
                           two_valid_neighbours_fraction=float((count == 2).mean()))
                write(receipt, row)
            stats_rows.append(row)
        print(json.dumps(dict(stage='same_time_correspondence', frame=frame, entries=len(stats_rows), seconds=time.monotonic()-t0)), flush=True)
    train76 = [(c, f) for c in CAMERAS for f in TRAIN76_FRAMES]
    lr_errors, view_errors = [], []
    for key in train76:
        with np.load(features[key]) as z:
            lr_errors.append(z['e_lr'].ravel())
        with np.load(errors / f'{key[0]}_{key[1]:04d}.npz') as z:
            view_errors.append(z['e_view'][z['valid_count'] > 0])
    lr_median = float(np.median(np.concatenate(lr_errors)))
    valid_view_errors = np.concatenate(view_errors)
    view_median = float(np.median(valid_view_errors)) if valid_view_errors.size else None
    scales = dict(s_lr=max(lr_median, 1/255), s_view=max(view_median or 0, 1/255),
                  lr_median=lr_median, valid_view_median=view_median,
                  valid_view_pixels_train76=int(valid_view_errors.size), floor=1/255,
                  observations=[dict(camera=c, frame=f) for c, f in train76], calibration='fixed training LR only')
    write(out / 'scales.json', scales)
    entries = []; coverage = []; sums = np.zeros(3, np.float64)
    for n, key in enumerate(inputs.keys):
        camera, frame = key
        with np.load(features[key]) as z:
            e_lr, coarse_lr = z['e_lr'], z['lr']
        with np.load(errors / f'{camera}_{frame:04d}.npz') as z:
            e_view, count = z['e_view'], z['valid_count']
        cl, cv, confidence = confidence_from_errors(e_lr, e_view, count, scales, inputs.shape_hr)
        path = out / f'{camera}_{frame:04d}.npz'
        np.savez_compressed(path, c_lr=cl, c_view=cv, valid_neighbours=count)
        row = dict(camera=camera, frame=frame, path=path.name, sha256=sha(path),
                   c_lr_shape=list(cl.shape), c_view_shape=list(cv.shape), frozen=True)
        entries.append(row)
        report = dict(camera=camera, frame=frame, **statistics(confidence), mean_c_lr=float(cl.mean()), mean_c_view=float(cv.mean()),
                      valid_fraction=float((count > 0).mean()), unknown_fraction=float((count == 0).mean()),
                      two_valid_neighbours_fraction=float((count == 2).mean()))
        # Pure LR temporal-change proxy for stratified descriptive statistics.
        adjacent = frame+2 if frame < 118 else frame-2
        with np.load(features[camera, adjacent]) as z:
            changed = np.abs(coarse_lr-z['lr']).mean(-1) >= POLICY['dynamic_proxy_rgb_mae']
        coarse_weight = cv2.resize(confidence, (cv.shape[1], cv.shape[0]), interpolation=cv2.INTER_AREA)
        report.update(dynamic_proxy_fraction=float(changed.mean()),
                      mean_weight_dynamic_proxy=float(coarse_weight[changed].mean()) if changed.any() else None,
                      mean_weight_static_proxy=float(coarse_weight[~changed].mean()) if (~changed).any() else None)
        coverage.append(report)
        sums += [float(confidence.astype(np.float64).sum()), float(np.square(confidence.astype(np.float64)).sum()), confidence.size]
    fields = list(coverage[0])
    with (out / 'confidence_coverage.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(coverage)
    summary = dict(mean_weight=float(sums[0]/sums[2]), kish_effective_fraction=float(sums[0]**2/(sums[2]*sums[1])) if sums[1] else 0,
                   mean_valid_fraction=float(np.mean([r['valid_fraction'] for r in coverage])),
                   mean_unknown_fraction=float(np.mean([r['unknown_fraction'] for r in coverage])),
                   static_dynamic_role='LR adjacent-frame colour-change proxy, not semantic ground truth or supervised mask')
    write(out / 'cache_manifest.json', dict(schema=1, status='completed_frozen_train_confidence', entries=entries,
                                  manifest_sha256=sha(inputs.manifest_path), identity=identity, scales=scales,
                                  summary=summary, output='get(): bilinear resize c_lr and c_view separately to HR then multiply; full-image mean',
                                  hr_derived_training_items=False, parameter_updates=0, frozen=True,
                                  input_read_log=inputs.opened, seconds=time.monotonic()-t0,
                                  source_index_sha256={'features': sha(out/'features/index.json'), 'parent': sha(parent_path)}))
    return out / 'cache_manifest.json'


class ConfidenceCache:
    """Bounded CPU LRU; weight is immutable and detached on the requested device."""
    def __init__(self, index, records=None, shape=(1008, 1344), maximum=32):
        self.index_path = Path(index).resolve(); value = read(self.index_path)
        if value.get('status') != 'completed_frozen_train_confidence' or value.get('hr_derived_training_items') is not False:
            raise ValueError('Only completed legal frozen confidence is accepted')
        self.rows = {(r['camera'], int(r['frame'])): r for r in value['entries']}
        expected = {(c, f) for c in CAMERAS for f in FRAMES}
        if set(self.rows) != expected or len(value['entries']) != 1140:
            raise ValueError('Confidence coverage differs from preregistered training observations')
        self.record_keys = [(r['camera_id'], int(r['frame_index'])) for r in records] if records is not None else None
        if self.record_keys is not None and set(self.record_keys) != expected:
            raise ValueError('Training record boundary differs from confidence')
        self.shape, self.maximum = tuple(shape[-2:]), int(maximum)
        if self.shape != (1008, 1344):
            raise ValueError('Confidence HR output shape differs from preregistration')
        self.cache = OrderedDict(); self.hits = self.misses = 0

    def get(self, key, frame=None, device='cuda'):
        if frame is not None:
            key = (key, int(frame))
        if isinstance(key, int):
            if self.record_keys is None:
                raise ValueError('Integer confidence index requires training records')
            key = self.record_keys[key]
        key = (key[0], int(key[1]))
        if key not in self.rows:
            raise ValueError('Held-out camera or unregistered time requested')
        if key in self.cache:
            self.hits += 1; value = self.cache.pop(key)
        else:
            self.misses += 1; row = self.rows[key]; path = self.index_path.parent / row['path']
            if sha(path) != row['sha256']:
                raise ValueError('Frozen confidence hash mismatch')
            with np.load(path, allow_pickle=False) as z:
                cl, cv = torch.from_numpy(z['c_lr'].copy())[None], torch.from_numpy(z['c_view'].copy())[None]
            value = (resize_chw(cl, self.shape)*resize_chw(cv, self.shape)).detach()
            if not torch.isfinite(value).all() or float(value.min()) < 0 or float(value.max()) > 1:
                raise ValueError('Confidence must be finite and within [0,1]')
        self.cache[key] = value
        while len(self.cache) > self.maximum:
            self.cache.popitem(last=False)
        return value.to(device=device, non_blocking=True).detach()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument('--teacher-index', type=Path, default=DEFAULT_TEACHERS)
    ap.add_argument('--parent-cache', type=Path)
    ap.add_argument('--out', type=Path, default=DEFAULT_OUT)
    ap.add_argument('--features-only', action='store_true')
    args = ap.parse_args(); torch.set_num_threads(4); cv2.setNumThreads(4)
    inputs = LegalInputs(args.manifest, args.teacher_index)
    if args.features_only:
        feature_cache(inputs, args.out/'features')
    elif args.parent_cache is None:
        ap.error('--parent-cache is required unless --features-only')
    else:
        print(build_cache(inputs, args.parent_cache, args.out), flush=True)


if __name__ == '__main__':
    main()
