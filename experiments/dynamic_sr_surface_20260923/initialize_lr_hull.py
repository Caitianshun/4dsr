"""Fixed CPU LR-mask visual-hull initialization shared by every control branch.

Only frame0 of the twelve training LR masks/images is projected. No HR, held-out
view, body template, or threshold search. Preserve original manifest and SIFT.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['OMP_NUM_THREADS'] = '4'
import cv2
import numpy as np
from scipy.ndimage import binary_erosion

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT / 'data/dynamic_sr/zju_prepared/coreview387_pilot_v1/manifest.json'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def write(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def project(points, camera):
    w = np.asarray(camera['w2c'], dtype=np.float64)
    k = np.asarray(camera['K_lr'], dtype=np.float64)
    p = points @ w[:3, :3].T + w[:3, 3]
    v = p @ k.T
    valid = p[:, 2] > 1e-8
    xy = np.rint(v[:, :2] / np.where(valid, v[:, 2], 1)[:, None]).astype(np.int64)
    valid &= (xy[:, 0] >= 0) & (xy[:, 0] < 256) & (xy[:, 1] >= 0) & (xy[:, 1] < 256)
    return xy, valid


def write_ply(path, points, colors):
    rgb = np.rint(np.clip(colors, 0, 1) * 255).astype(np.uint8)
    with path.open('w') as f:
        f.write(f'ply\nformat ascii 1.0\nelement vertex {len(points)}\n')
        f.write('property float x\nproperty float y\nproperty float z\n')
        f.write('property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n')
        for p, c in zip(points, rgb):
            f.write('%.8g %.8g %.8g %d %d %d\n' % (*p, *c))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=DEFAULT)
    args = parser.parse_args()
    manifest_path = args.manifest.resolve(); root = manifest_path.parent
    destination = root / 'initialization_hull'; new_manifest = root / 'manifest_hull.json'
    if destination.exists() or new_manifest.exists():
        raise FileExistsError('Preserve existing hull artifacts; no overwrite or threshold retry.')
    destination.mkdir(); shutil.copyfile(__file__, destination / 'source.py')
    started = time.monotonic()
    report = dict(status='running', started_utc=datetime.now(timezone.utc).isoformat(), cpu_only=True,
                  source_manifest=str(manifest_path), source_manifest_sha256=sha(manifest_path),
                  script_sha256=sha(__file__), used_hr=False, used_dev_or_test=False, used_smpl=False,
                  method_role='Common initialization for every baseline/module branch, not a proposed SR module',
                  extra_input='Provided segmentation, inherited from shared preprocessing; hull uses only its LR mask derivative',
                  protocol=dict(frame=0, threshold=.25, dilation_lr_pixels=1, dilation_kernel=[3,3],
                                grid_shape=[160,160,160], center='median of original268 training-LR SIFT points',
                                half_extent_meters=[2,2,2], intersection='Every training camera must have positive depth, valid pixel, and dilated foreground',
                                surface='occupancy minus 3x3x3 binary erosion; outside-grid background',
                                max_hull_points=20000, sampling_seed=20260923, add_original_sift=True,
                                projection='Nearest zero-based LR pixel center; no HR tuning',
                                color='Mean RGB across original thresholded, undilated training LR foreground views; drop points with no such view'))
    try:
        m = json.loads(manifest_path.read_text())
        train = m['splits']['train']
        assert train == [f'cam{i:02d}' for i in range(0,23,2)]
        assert m['resolutions']['lr'] == [256,256]
        init_path = root / m['initialization']['npz_path']
        with np.load(init_path, allow_pickle=False) as saved:
            sift_xyz, sift_rgb = saved['points'].copy(), saved['colors'].copy()
        assert sift_xyz.shape == (268,3) and np.isfinite(sift_xyz).all()
        center = np.median(sift_xyz.astype(np.float64), axis=0)
        axes = [np.linspace(c-2, c+2, 160, dtype=np.float64) for c in center]
        grid = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1).reshape(-1,3)
        occupied = np.ones(len(grid), bool)
        observations = {(o['camera_id'],o['frame_index']):o for o in m['observations']}
        masks, images, inputs, intersection_counts = {}, {}, [], []
        cv2.setNumThreads(4)
        for name in train:
            o = observations[(name,0)]; assert o['split']=='train'
            mp, ip = root/o['mask_lr_path'], root/o['lr_path']
            rawmask = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
            bgr = cv2.imread(str(ip), cv2.IMREAD_COLOR)
            assert rawmask is not None and rawmask.shape==(256,256)
            assert bgr is not None and bgr.shape==(256,256,3)
            mask = (rawmask.astype(np.float32)/255 >= .25)
            expanded = cv2.dilate(mask.astype(np.uint8), np.ones((3,3), np.uint8)) > 0
            masks[name] = mask
            images[name] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float64)/255
            # Project only surviving voxels; this is exactly all-camera intersection.
            candidates = np.flatnonzero(occupied)
            for begin in range(0,len(candidates),262144):
                idx = candidates[begin:begin+262144]
                xy, valid = project(grid[idx], m['cameras'][name])
                keep = np.zeros(len(idx), bool)
                keep[valid] = expanded[xy[valid,1],xy[valid,0]]
                occupied[idx] &= keep
            intersection_counts.append(dict(camera=name, remaining_voxels=int(occupied.sum())))
            inputs.append(dict(camera=name, frame=0, mask_lr_path=str(mp), mask_lr_sha256=sha(mp),
                               lr_path=str(ip), lr_sha256=sha(ip)))
        count = int(occupied.sum())
        report.update(sources=inputs, intersection_counts=intersection_counts, occupied_voxels=count,
                      sift_npz_path=str(init_path), sift_npz_sha256=sha(init_path),
                      center_meters=center.tolist(), grid_min_meters=(center-2).tolist(),
                      grid_max_meters=(center+2).tolist(), grid_spacing_meters=4/159)
        if count < 128:
            raise RuntimeError(f'Fixed LR visual hull has {count} occupied voxels (<128); do not search thresholds.')
        volume = occupied.reshape(160,160,160)
        eroded = binary_erosion(volume, structure=np.ones((3,3,3),bool), border_value=0)
        surface = volume & ~eroded
        xyz = grid[surface.ravel()]
        report['surface_points_before_color'] = len(xyz)
        total_rgb = np.zeros((len(xyz),3),np.float64); seen = np.zeros(len(xyz),np.int32)
        for name in train:
            xy, valid = project(xyz,m['cameras'][name])
            idx = np.flatnonzero(valid)
            idx = idx[masks[name][xy[idx,1],xy[idx,0]]]
            total_rgb[idx] += images[name][xy[idx,1],xy[idx,0]]; seen[idx] += 1
        colored = seen > 0
        report['surface_without_original_foreground_color'] = int((~colored).sum())
        xyz, colors, views = xyz[colored], total_rgb[colored]/seen[colored,None], seen[colored]
        if len(xyz)<128:
            raise RuntimeError(f'Only {len(xyz)} colored hull surface points (<128); no threshold retry.')
        if len(xyz)>20000:
            selected = np.random.default_rng(20260923).choice(len(xyz),20000,replace=False)
            xyz, colors, views = xyz[selected], colors[selected], views[selected]
        hull_count = len(xyz)
        all_xyz = np.concatenate([xyz,sift_xyz]).astype(np.float32)
        all_rgb = np.concatenate([colors,sift_rgb]).astype(np.float32)
        np.savez_compressed(destination/'points_lr.npz', points=all_xyz, colors=all_rgb,
                            normals=np.zeros_like(all_xyz),
                            point_source=np.concatenate([np.zeros(hull_count,np.uint8),np.ones(len(sift_xyz),np.uint8)]))
        write_ply(destination/'points_lr.ply',all_xyz,all_rgb)
        report.update(status='prepared', hull_point_count=hull_count, added_sift_point_count=len(sift_xyz),
                      point_count=len(all_xyz), hull_min_meters=xyz.min(0).tolist(), hull_max_meters=xyz.max(0).tolist(),
                      color_support_views_min=int(views.min()), color_support_views_max=int(views.max()),
                      elapsed_seconds=time.monotonic()-started)
        write(destination/'report.json',report)
        new = dict(m)
        new['initialization'] = dict(npz_path='initialization_hull/points_lr.npz',
            ply_path='initialization_hull/points_lr.ply',report_path='initialization_hull/report.json',
            point_count=len(all_xyz), hull_point_count=hull_count, sift_point_count=len(sift_xyz),
            npz_sha256=sha(destination/'points_lr.npz'),train_only_lr=True,provided_lr_mask_visual_hull=True)
        new['original_manifest'] = dict(path='manifest.json',sha256=sha(manifest_path),preserved=True)
        new['initialization_scope'] = 'Common fixed LR-mask visual hull plus original LR SIFT, frame0 hull, same init for every control; not the proposed module.'
        write(new_manifest,new)
        print(json.dumps(dict(status='prepared',manifest=str(new_manifest),manifest_sha256=sha(new_manifest),
                              occupied_voxels=count,hull_points=hull_count,sift_points=len(sift_xyz),total_points=len(all_xyz),
                              elapsed_seconds=report['elapsed_seconds'])),flush=True)
    except BaseException:
        report.update(status='failed',traceback=traceback.format_exc(),elapsed_seconds=time.monotonic()-started)
        write(destination/'report.json',report)
        raise


if __name__=='__main__':main()
