"""Coordinate, missing-evidence and train-only confidence engineering checks."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from confidence_cache import (POLICY, coarse_calibration, world_from_depth, project_world,
                              neighbour_evidence, confidence_from_errors, mip_sample,
                              aggregate_parent, projection_footprint, statistics, write,
                              ConfidenceCache, LegalInputs, sha)


def synthetic_checks():
    h, w = 24, 32; y, x = np.indices((h, w), dtype=np.float32)
    cal = dict(K=np.array([[24., 0, 15.5], [0, 24., 11.5], [0, 0, 1]]), w2c=np.eye(4), c2w=np.eye(4))
    z = np.ones((h, w), np.float32)*3; alpha = np.ones_like(z)
    xy = np.stack((x, y), -1)
    world = world_from_depth(z, cal); projected, zp = project_world(world, cal)
    coord_err = float(np.max(np.abs(projected-xy)))
    assert coord_err < 1e-10 and np.max(np.abs(zp-z)) < 1e-10
    footprint = projection_footprint(projected)
    assert np.max(np.abs(footprint-1)) < 1e-6
    # Linear ramps ensure repeatable texture; matching identical depth/cameras
    # must retain interior correspondence and zero descriptor error.
    image = np.stack((x/(w-1), y/(h-1), (x+y)/(w+h-2)), -1).astype(np.float32)
    detail = np.sin(x[..., None]*1.7)*np.array([.02, .01, .03], np.float32)
    data = dict(depth=z, alpha=alpha, lr=image, detail=detail)
    error, valid, stats = neighbour_evidence(data, data, cal, cal)
    assert valid.sum() > 100 and float(error[valid].max()) < 1e-7
    assert np.max(np.abs(mip_sample(image, xy, footprint)-image)) < 1e-7
    # Twice-minified alternating source pixels average to 0.5 in a mip level,
    # whereas a direct bilinear sample at integer centres aliases to one parity.
    alternating = np.tile((np.arange(64) % 2).astype(np.float32), (48, 1))
    yy, xx = np.indices((12, 16), dtype=np.float32)
    minified_xy = np.stack((xx*2+8, yy*2+8), -1)
    minified = mip_sample(alternating, minified_xy, np.ones((12, 16), np.float32)*2)
    assert np.max(np.abs(minified-.5)) < 1e-7
    flat = dict(data, lr=np.ones_like(image)*.5)
    _, flat_valid, _ = neighbour_evidence(flat, flat, cal, cal)
    assert not flat_valid.any(), 'Low texture must remain unknown'
    zero = np.zeros((h, w), np.uint8)
    cl, cv, weight = confidence_from_errors(np.zeros((2*h, 2*w), np.float32), np.zeros((h, w), np.float32), zero,
                                           dict(s_lr=1/255, s_view=1/255), (4*h, 4*w))
    assert np.array_equal(cv, np.full_like(cv, .5)) and np.allclose(weight, .5)
    known = zero.copy(); known[8:12, 8:12] = 1
    _, cv2, _ = confidence_from_errors(np.zeros((2*h, 2*w), np.float32), np.zeros((h, w), np.float32), known,
                                     dict(s_lr=1/255, s_view=1/255), (4*h, 4*w))
    assert np.all(cv2[known > 0] == 1), 'Missing second neighbour cannot halve known confidence'
    hidden = dict(data, depth=z*.5)
    _, hidden_valid, _ = neighbour_evidence(data, hidden, cal, cal)
    assert not hidden_valid.any(), 'Foreground depth mismatch must reject occluded correspondence'
    # Coarse moments: alpha-weighted depths differ from ordinary mean depth.
    small_z = np.array([[2., 4.], [2., 4.]], np.float32)
    small_a = np.array([[1., .1], [1., .1]], np.float32)
    agg_z, agg_a = aggregate_parent(small_z, small_a)
    assert abs(float(agg_z[0, 0])-2.4/1.1) < 1e-5
    source_cal = dict(K_lr=[[100., 0, 20.5], [0, 100., 10.5], [0, 0, 1]], w2c=np.eye(4), c2w=np.eye(4))
    coarse = coarse_calibration(source_cal, 2)
    assert coarse['K'][0, 2] == 10 and coarse['K'][1, 2] == 5
    return dict(status='passed', dtype='float32 descriptors; float64 calibrated coordinates', tolerance_coordinates=1e-10,
                roundtrip_max_abs=coord_err, identical_support_fraction=float(valid.mean()),
                identical_detail_max_abs=float(error[valid].max()), low_texture='unknown',
                missing_neighbour='0.5 iff no valid evidence; no double penalty',
                occlusion='rejected', depth_aggregation='area moments before normalization',
                minified_checkerboard='area mip recovers mean 0.5 instead of aliasing to one parity',
                pixel_coordinate_rule='half-pixel centre conversion', weight_stats=statistics(weight),
                policy=POLICY, source_sha256=sha(__file__))


def actual_checks(index, manifest, teacher_index):
    inputs = LegalInputs(manifest, teacher_index)
    cache = ConfidenceCache(index, shape=inputs.shape_hr)
    for row in cache.rows.values():
        path = Path(index).resolve().parent / row['path']
        assert sha(path) == row['sha256']
        with np.load(path, allow_pickle=False) as z:
            assert z['c_lr'].shape == (252, 336) and z['c_view'].shape == (126, 168)
            for field in ('c_lr', 'c_view'):
                assert np.isfinite(z[field]).all() and (z[field] >= 0).all() and (z[field] <= 1).all()
            count = z['valid_neighbours']
            assert count.shape == (126, 168) and (count <= 2).all()
            assert np.all(z['c_view'][count == 0] == .5), 'Unknown fallback must be exact before HR resize'
    inspected = []
    for key in [('cam02', 0), ('cam10', 40), ('cam20', 118)]:
        value = cache.get(key, device='cpu')
        assert value.shape == (1, 1008, 1344) and not value.requires_grad
        inspected.append(dict(camera=key[0], frame=key[1], **statistics(value.numpy())))
    for illegal in [('cam00', 0), ('cam01', 0), ('cam02', 1), ('cam20', 120)]:
        try:
            cache.get(illegal, device='cpu')
        except ValueError:
            continue
        raise AssertionError('Illegal confidence observation accepted')
    return dict(status='passed', inspected=inspected, illegal_observations_rejected=True,
                entries=len(cache.rows), index_sha256=sha(index), hr_pixels_opened=False)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--index', type=Path); ap.add_argument('--manifest', type=Path); ap.add_argument('--teacher-index', type=Path)
    args = ap.parse_args(); torch.set_num_threads(4)
    result = dict(synthetic=synthetic_checks())
    if args.index:
        if args.manifest is None or args.teacher_index is None:
            ap.error('Actual cache verification needs manifest and teacher-index')
        result['actual'] = actual_checks(args.index, args.manifest, args.teacher_index)
    write(args.out, result); print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
