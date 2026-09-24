"""Prepare the fixed CoreView387 RGB+official-mask short-window SR pilot on CPU.

No SMPL, HR geometry, prior model, held-out matching, or GPU is used. Initialization
reuses the verified known-camera LR SIFT triangulator with a 256x256 feature loop.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import itertools
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

os.environ['CUDA_VISIBLE_DEVICES'] = ''
ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / 'experiments/dynamic_sr_20260918'
sys.path.insert(0, str(OLD))
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from prepare_n3dv import atomic_json, sha256, triangulate_pair, write_ply

FRAMES = list(range(0, 120, 2))
TRAIN = [f'cam{i:02d}' for i in range(0, 23, 2)]
DEV, TEST = ['cam07'], ['cam01']
SELECTED = sorted(TRAIN + DEV + TEST)


def save_png(path, array):
    if not cv2.imwrite(str(path), array, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
        raise RuntimeError(f'Cannot write {path}')
    return sha256(path)


def prepare_camera(raw, destination, annots, index, input_identities):
    name = f'cam{index:02d}'
    split = 'train' if name in TRAIN else 'dev' if name in DEV else 'test'
    cams = annots['cams']
    K = np.asarray(cams['K'][index], dtype=np.float64)
    D = np.asarray(cams['D'][index], dtype=np.float64)
    R = np.asarray(cams['R'][index], dtype=np.float64)
    T = np.asarray(cams['T'][index], dtype=np.float64).reshape(3) / 1000.0
    assert np.isfinite(K).all() and np.isfinite(D).all() and np.isfinite(R).all() and np.isfinite(T).all()
    assert np.allclose(R.T @ R, np.eye(3), atol=1e-6) and np.isclose(np.linalg.det(R), 1, atol=1e-6)
    w2c = np.eye(4); w2c[:3, :3] = R; w2c[:3, 3] = T
    # The unchanged original-resolution pinhole K is the target of undistortion.
    mx, my = cv2.initUndistortRectifyMap(K, D, None, K, (1024, 1024), cv2.CV_32FC1)
    resize = np.array([[.25, 0, -.375], [0, .25, -.375], [0, 0, 1]])
    camera = dict(camera_id=name, source_camera_id=f'Camera_B{index+1}', source_camera_index=index,
                  raw_width=1024, raw_height=1024, K_raw=K.tolist(), D_raw=D.tolist(),
                  K_hr=K.tolist(), K_lr=(resize @ K).tolist(), w2c=w2c.tolist(), c2w=np.linalg.inv(w2c).tolist(),
                  extrinsic_convention='OpenCV world-to-camera; source T millimeters converted to meters')
    for sub in ['hr', 'lr', 'mask', 'mask_lr']:
        (destination / sub / name).mkdir(parents=True)
    observations = []
    for frame in FRAMES:
        rel = Path(str(annots['ims'][frame]['ims'][index]))
        assert rel.parts[0] == camera['source_camera_id'] and rel.stem == f'{frame:06d}'
        paths = [raw / rel, raw / 'mask' / rel.with_suffix('.png'), raw / 'mask_cihp' / rel.with_suffix('.png')]
        bgr = cv2.imread(str(paths[0]), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(paths[1]), cv2.IMREAD_GRAYSCALE)
        cihp = cv2.imread(str(paths[2]), cv2.IMREAD_GRAYSCALE)
        assert bgr is not None and bgr.shape == (1024, 1024, 3)
        assert mask is not None and cihp is not None and mask.shape == cihp.shape == (1024, 1024)
        union = ((mask > 0) | (cihp > 0)).astype(np.uint8)
        hr = cv2.remap(bgr, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        foreground = cv2.remap(union, mx, my, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        hr[foreground == 0] = 0
        tensor = torch.from_numpy(hr.copy()).permute(2, 0, 1).float().div_(255).unsqueeze(0)
        low = F.interpolate(tensor, size=(256, 256), mode='bicubic', align_corners=False, antialias=True).clamp_(0, 1)
        lr = low[0].permute(1, 2, 0).mul(255).round().byte().numpy()
        soft = F.interpolate(torch.from_numpy(foreground.astype(np.float32))[None, None], size=(256, 256), mode='area')
        lr_mask = soft[0, 0].mul(255).round().byte().numpy()
        output = {}
        for sub, array, field in [('hr', hr, 'hr'), ('lr', lr, 'lr'), ('mask', foreground * 255, 'mask'), ('mask_lr', lr_mask, 'mask_lr')]:
            path = destination / sub / name / f'{frame:04d}.png'
            output[field + '_path'] = str(path.relative_to(destination))
            output[field + '_sha256'] = save_png(path, array)
        observations.append(dict(camera_id=name, source_camera_id=camera['source_camera_id'],
                                 frame_index=frame, time=frame / 118.0, split=split, **output))
        input_identities.append(dict(camera_id=name, frame_index=frame,
            rgb=dict(path=str(paths[0]), sha256=sha256(paths[0])),
            mask=dict(path=str(paths[1]), sha256=sha256(paths[1])),
            mask_cihp=dict(path=str(paths[2]), sha256=sha256(paths[2]))))
    return camera, observations


def initialize(root, cameras):
    selected_frames = [0, 60, 118]
    sift = cv2.SIFT_create(nfeatures=2500, contrastThreshold=.012, edgeThreshold=12)
    points_all, colors_all, errors_all, frames_all, records, sources = [], [], [], [], [], []
    for frame in selected_frames:
        features = {}
        for name in TRAIN:
            path = root / 'lr' / name / f'{frame:04d}.png'
            bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            assert bgr is not None and bgr.shape == (256, 256, 3)
            # Input is already black-background official-mask RGB. No HR image,
            # depth, mesh, template, or held-out view enters feature extraction.
            keys, descriptors = sift.detectAndCompute(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), None)
            features[name] = (np.array([k.pt for k in keys], dtype=np.float64).reshape(-1, 2),
                              descriptors, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            sources.append(dict(path=str(path.relative_to(root)), sha256=sha256(path), camera_id=name,
                                frame_index=frame, features=len(keys)))
        for a, b in itertools.combinations(TRAIN, 2):
            xyz, rgb, error, report = triangulate_pair(cameras[a], cameras[b], features[a], features[b])
            report['frame_index'] = frame; records.append(report)
            if len(xyz):
                points_all.append(xyz); colors_all.append(rgb); errors_all.append(error)
                frames_all.append(np.full(len(xyz), frame, np.int32))
        print(json.dumps(dict(event='triangulated', frame=frame, points=sum(len(x) for x in points_all))), flush=True)
    diagnostics = dict(train_camera_ids=TRAIN, source_frames=selected_frames, source_images=sources,
                       pair_statistics=records, used_hr_geometry=False, used_smpl=False, used_heldout_images=False,
                       stage='before_spatial_filtering')
    atomic_json(root / 'triangulation_diagnostics.json', diagnostics)
    if not points_all:
        raise RuntimeError('No reliable training-LR SIFT points; no automatic SMPL/HR fallback.')
    xyz, rgb, error, frames = map(np.concatenate, [points_all, colors_all, errors_all, frames_all])
    raw_count = len(xyz)
    radius = np.linalg.norm(xyz - np.median(xyz, axis=0), axis=1)
    good = np.isfinite(radius) & (radius <= max(float(np.quantile(radius, .98)) * 2, 1e-6))
    xyz, rgb, error, frames = xyz[good], rgb[good], error[good], frames[good]
    extent = max(float(np.quantile(np.linalg.norm(xyz - np.median(xyz, axis=0), axis=1), .9)), 1e-4)
    voxel = extent * .002
    order = np.argsort(error, kind='stable')
    _, unique = np.unique(np.floor(xyz[order] / voxel).astype(np.int64), axis=0, return_index=True)
    indices = order[unique]
    if len(indices) > 20000:
        indices = np.random.default_rng(20260923).choice(indices, 20000, replace=False)
    xyz, rgb, error, frames = xyz[indices], rgb[indices], error[indices], frames[indices]
    if len(xyz) < 128:
        atomic_json(root / 'initialization_failed.json', dict(reliable_points=len(xyz), minimum=128, raw_points=raw_count))
        raise RuntimeError(f'Only {len(xyz)} reliable training-LR SIFT points (<128); no automatic geometry fallback.')
    folder = root / 'initialization'; folder.mkdir()
    np.savez_compressed(folder / 'points_lr.npz', points=xyz.astype(np.float32), colors=rgb.astype(np.float32),
                        normals=np.zeros_like(xyz, dtype=np.float32), reprojection_error_lr_px=error.astype(np.float32),
                        source_frame_index=frames)
    write_ply(folder / 'points_lr.ply', xyz, rgb)
    atomic_json(folder / 'report.json', dict(**diagnostics, method='known_calibration_SIFT_mutual_ratio_RANSAC_epipolar_DLT_v1',
        official_colmap_initialization=False, used_hr_images=False, used_development_or_test_images=False,
        raw_point_count=raw_count, point_count=len(xyz), voxel_size_world=voxel,
        reprojection_error_lr_px_median=float(np.median(error)),
        points_per_source_frame=dict(Counter(str(int(f)) for f in frames)),
        limitation='Sparse points from three times mix deforming surfaces in canonical initialization; all branches share this initialization.'))
    return dict(npz_path='initialization/points_lr.npz', ply_path='initialization/points_lr.ply',
                report_path='initialization/report.json', point_count=len(xyz),
                npz_sha256=sha256(folder / 'points_lr.npz'), train_only_lr=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, default=Path('/home/cai_tianshun/Project/mml/CoreView_387'))
    parser.add_argument('--out', type=Path, default=ROOT / 'data/dynamic_sr/zju_prepared/coreview387_pilot_v1')
    args = parser.parse_args(); raw, out = args.raw.resolve(), args.out.resolve()
    if out.exists():
        raise FileExistsError(f'Preserve existing preparation: {out}')
    cv2.setNumThreads(4); cv2.setRNGSeed(20260923); torch.set_num_threads(4)
    assert not torch.cuda.is_initialized()
    out.mkdir(parents=True); (out / 'sources').mkdir()
    tick = time.monotonic()
    state = dict(status='preparing', started_utc=datetime.now(timezone.utc).isoformat(), cpu_only=True,
                 source=str(raw), output=str(out), fps=None)
    atomic_json(out / 'status.json', state)
    try:
        source_files = [Path(__file__), OLD / 'prepare_n3dv.py', OLD / 'n3dv_data.py', raw / 'annots.npy',
                        raw / 'intri.yml', raw / 'extri.yml', raw / 'get_annots.py']
        identities = []
        for i, p in enumerate(source_files):
            copy = out / 'sources' / f'{i}_{p.name}'; shutil.copyfile(p, copy)
            identities.append(dict(path=str(p), sha256=sha256(p), copy=str(copy.relative_to(out))))
        annots = np.load(raw / 'annots.npy', allow_pickle=True).item()
        assert len(annots['ims']) == 654 and len(annots['cams']['K']) == 23
        cameras, observations, inputs = {}, [], []
        for name in SELECTED:
            camera, rows = prepare_camera(raw, out, annots, int(name[3:]), inputs)
            cameras[name] = camera; observations.extend(rows)
            print(json.dumps(dict(event='prepared_camera', camera=name, observations=len(rows))), flush=True)
        atomic_json(out / 'source_image_identities.json', dict(root=str(raw), observations=inputs))
        atomic_json(out / 'sources.json', dict(files=identities,
            image_identities='source_image_identities.json', image_identities_sha256=sha256(out / 'source_image_identities.json'),
            segmentation_version='Existing source dataset mask and mask_cihp files; binary nonzero union, exact per-file hashes recorded. No model rerun or manual editing.',
            used_smpl=False, used_existing_model_weights=False))
        initialization = initialize(out, cameras)
        manifest = dict(schema='n3dv_dynamic_sr_pilot_v1', scene='zju_coreview387_pilot_v1', source_dataset='ZJU-Mocap',
            dataset_family='ZJU-Mocap / NeuralBody raw RGB + supplied masks', source='https://zju3dv.github.io/neuralbody/',
            scale=4, raw_directory=str(raw), source_annots_sha256=sha256(raw / 'annots.npy'),
            source_manifest_path='sources.json', source_manifest_sha256=sha256(out / 'sources.json'),
            frame_indices=FRAMES, time_convention='original_frame_index / 118.0; normalized sequence coordinate, not seconds',
            capture=dict(fps=None, fps_note='No verified original capture frame rate; do not infer from evaluation stride.',
                         original_frame_count=654, original_camera_count=23, synchronized=True),
            resolutions=dict(hr=[1024, 1024], lr=[256, 256]),
            degradation=dict(raw_to_reference='RGB linear remap undistortion with original K and D to same 1024x1024 K; no crop; nearest-neighbor remap of binary union mask; masked black background',
                reference_to_lr='torch F.interpolate bicubic align_corners=False antialias=True, exact x4, clamp[0,1], round uint8 PNG',
                mask_to_lr='area downsample of binary HR foreground to soft coverage, round uint8 PNG; divide by255 when reading',
                quantization='uint8 reference and LR; differentiable render-to-LR operator omits rounding', extra_noise=False),
            extra_inputs=dict(official_masks=True, mask_rule='(mask>0) OR (mask_cihp>0)', masks_from_raw_resolution=True,
                purpose='Shared foreground/background conditioning and evaluation masks; this is RGB+calibration+provided-masks, not pure unmasked RGB',
                smpl=False, hr_geometry=False, existing_avatar_weights=False),
            pixel_coordinate_convention="OpenCV zero-based pixel centers; HR K unchanged after remap; LR u'=(u+0.5)/4-0.5",
            splits=dict(train=TRAIN, dev=DEV, test=TEST),
            split_note='Custom SR development short-window split, not standard NeuralBody benchmark; even indices train, cam07 dev, cam01 test; other cameras omitted.',
            camera_mapping='cam00=Camera_B1 through cam22=Camera_B23; only declared split cameras processed',
            cameras=cameras, observations=observations, initialization=initialization,
            preparation_elapsed_seconds=time.monotonic()-tick,
            versions=dict(opencv=cv2.__version__, numpy=np.__version__, torch=str(torch.__version__)),
            script_sha256={p.name: sha256(p) for p in source_files[:3]})
        assert Counter(o['split'] for o in observations) == {'train':720, 'dev':60, 'test':60}
        assert len(observations)==840 and not torch.cuda.is_initialized()
        atomic_json(out / 'manifest.json', manifest)
        state.update(status='prepared', elapsed_seconds=time.monotonic()-tick, observations=840,
                     initial_points=initialization['point_count'], manifest_sha256=sha256(out / 'manifest.json'))
        atomic_json(out / 'status.json', state); atomic_json(out / 'complete.json', state)
        print(json.dumps(state), flush=True)
    except BaseException:
        state.update(status='failed', elapsed_seconds=time.monotonic()-tick, traceback=traceback.format_exc())
        atomic_json(out / 'status.json', state)
        raise


if __name__ == '__main__':
    main()
