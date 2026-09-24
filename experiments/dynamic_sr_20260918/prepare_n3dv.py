"""Prepare isolated N3DV LR inputs and training-only SIFT triangulation.

This is a documented sparse initialization, not the official COLMAP pipeline.
It uses supplied calibration and matching pixels exclusively from training LR
images. Development/test images are decoded for evaluation but never matched.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from n3dv_data import parse_n3dv_cameras


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def extract_camera(camera: dict, destination: Path, frame_indices: list[int], split: str) -> list[dict]:
    cap = cv2.VideoCapture(camera["video"])
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {camera['video']}")
    target = set(frame_indices)
    observations = []
    camera_id = camera["camera_id"]
    for resolution in ("hr", "lr"):
        (destination / resolution / camera_id).mkdir(parents=True, exist_ok=True)
    try:
        for frame_index in range(max(frame_indices) + 1):
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"Video ended before frame {frame_index}: {camera['video']}")
            if frame_index not in target:
                continue
            if frame.shape[:2] != (camera["raw_height"], camera["raw_width"]):
                raise ValueError(f"Video dimensions disagree with supplied calibration: {frame.shape}")
            half = cv2.resize(frame, (1352, 1014), interpolation=cv2.INTER_AREA)
            hr = half[3:1011, 4:1348].copy()
            # Match the declared training observation operator (apart from PNG
            # quantization), rather than combining an area LR input with a
            # bicubic-trained classical SR prior.
            image_tensor = torch.from_numpy(hr.copy()).permute(2, 0, 1).float().div_(255).unsqueeze(0)
            low_tensor = F.interpolate(image_tensor, size=(252, 336), mode="bicubic",
                                       align_corners=False, antialias=True).clamp_(0, 1)
            lr = low_tensor[0].permute(1, 2, 0).mul(255).round().byte().numpy()
            paths = {}
            for resolution, array in (("hr", hr), ("lr", lr)):
                path = destination / resolution / camera_id / f"{frame_index:04d}.png"
                if not cv2.imwrite(str(path), array, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
                    raise RuntimeError(f"Cannot save {path}")
                paths[resolution + "_path"] = str(path.relative_to(destination))
                paths[resolution + "_sha256"] = sha256(path)
            observations.append({"camera_id": camera_id, "frame_index": frame_index,
                                 "time": frame_index / 300.0, "split": split, **paths})
    finally:
        cap.release()
    return observations


def skew(v: np.ndarray) -> np.ndarray:
    x, y, z = v
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=np.float64)


def known_fundamental(a: dict, b: dict) -> np.ndarray:
    wa, wb = np.asarray(a["w2c"]), np.asarray(b["w2c"])
    relative = wb @ np.linalg.inv(wa)
    essential = skew(relative[:3, 3]) @ relative[:3, :3]
    ka, kb = np.asarray(a["K_lr"]), np.asarray(b["K_lr"])
    return np.linalg.inv(kb).T @ essential @ np.linalg.inv(ka)


def sampson_distance(f: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ha = np.concatenate((a, np.ones((len(a), 1))), axis=1)
    hb = np.concatenate((b, np.ones((len(b), 1))), axis=1)
    fa, ftb = ha @ f.T, hb @ f
    numerator = np.sum(hb * fa, axis=1) ** 2
    denominator = fa[:, 0] ** 2 + fa[:, 1] ** 2 + ftb[:, 0] ** 2 + ftb[:, 1] ** 2
    return numerator / np.maximum(denominator, 1e-20)


def triangulate_pair(a: dict, b: dict, fa: tuple, fb: tuple, ratio: float = 0.8) -> tuple:
    pa, da, rgb_a = fa
    pb, db, rgb_b = fb
    record = {"camera_a": a["camera_id"], "camera_b": b["camera_id"],
              "features_a": len(pa), "features_b": len(pb), "matches": 0,
              "ransac_and_known_pose_inliers": 0, "accepted_points": 0}
    empty = (np.empty((0, 3)), np.empty((0, 3)), np.empty((0,)), record)
    if da is None or db is None or len(da) < 8 or len(db) < 8:
        return empty
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    forward = matcher.knnMatch(da, db, k=2)
    reverse = matcher.knnMatch(db, da, k=2)
    reverse_good = {p[0].queryIdx: p[0].trainIdx for p in reverse
                    if len(p) == 2 and p[0].distance < ratio * p[1].distance}
    matches = [p[0] for p in forward if len(p) == 2 and p[0].distance < ratio * p[1].distance
               and reverse_good.get(p[0].trainIdx) == p[0].queryIdx]
    record["matches"] = len(matches)
    if len(matches) < 8:
        return empty
    xa = np.array([pa[m.queryIdx] for m in matches], dtype=np.float64)
    xb = np.array([pb[m.trainIdx] for m in matches], dtype=np.float64)
    _, mask = cv2.findFundamentalMat(xa, xb, cv2.FM_RANSAC, 1.5, 0.999)
    if mask is None or len(mask) != len(xa):
        return empty
    keep = mask.ravel().astype(bool) & (sampson_distance(known_fundamental(a, b), xa, xb) < 1.5 ** 2)
    xa, xb = xa[keep], xb[keep]
    record["ransac_and_known_pose_inliers"] = len(xa)
    if len(xa) == 0:
        return empty
    wa, wb = np.asarray(a["w2c"]), np.asarray(b["w2c"])
    ka, kb = np.asarray(a["K_lr"]), np.asarray(b["K_lr"])
    proj_a, proj_b = ka @ wa[:3], kb @ wb[:3]
    homogeneous = cv2.triangulatePoints(proj_a, proj_b, xa.T, xb.T).T
    valid_w = np.abs(homogeneous[:, 3]) > 1e-10
    points = homogeneous[:, :3] / np.where(valid_w, homogeneous[:, 3], 1)[:, None]
    points_h = np.c_[points, np.ones(len(points))]
    ca, cb = points_h @ wa[:3].T, points_h @ wb[:3].T
    uv_a, uv_b = points_h @ proj_a.T, points_h @ proj_b.T
    ea = np.linalg.norm(uv_a[:, :2] / np.maximum(uv_a[:, 2:], 1e-10) - xa, axis=1)
    eb = np.linalg.norm(uv_b[:, :2] / np.maximum(uv_b[:, 2:], 1e-10) - xb, axis=1)
    center_a, center_b = np.asarray(a["c2w"])[:3, 3], np.asarray(b["c2w"])[:3, 3]
    ra, rb = points - center_a, points - center_b
    cosine = np.sum(ra * rb, axis=1) / np.maximum(np.linalg.norm(ra, axis=1) * np.linalg.norm(rb, axis=1), 1e-20)
    angle = np.degrees(np.arccos(np.clip(cosine, -1, 1)))
    keep = valid_w & np.all(np.isfinite(points), axis=1) & (ca[:, 2] > 0) & (cb[:, 2] > 0)
    keep &= (ea < 1.5) & (eb < 1.5) & (angle >= 0.75)
    points, xa, xb = points[keep], xa[keep], xb[keep]
    if not len(points):
        return empty
    ia = np.rint(xa).astype(int)
    ib = np.rint(xb).astype(int)
    ia[:, 0] = np.clip(ia[:, 0], 0, rgb_a.shape[1] - 1)
    ia[:, 1] = np.clip(ia[:, 1], 0, rgb_a.shape[0] - 1)
    ib[:, 0] = np.clip(ib[:, 0], 0, rgb_b.shape[1] - 1)
    ib[:, 1] = np.clip(ib[:, 1], 0, rgb_b.shape[0] - 1)
    colors = (rgb_a[ia[:, 1], ia[:, 0]].astype(float) + rgb_b[ib[:, 1], ib[:, 0]].astype(float)) / 510.0
    errors = np.maximum(ea, eb)[keep]
    record["accepted_points"] = len(points)
    record["reprojection_error_median_lr_px"] = float(np.median(errors))
    record["triangulation_angle_median_deg"] = float(np.median(angle[keep]))
    return points, colors, errors, record


def write_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    rgb = np.clip(np.rint(colors * 255), 0, 255).astype(np.uint8)
    with path.open("w") as stream:
        stream.write("ply\nformat ascii 1.0\nelement vertex " + str(len(points)) + "\n")
        stream.write("property float x\nproperty float y\nproperty float z\n")
        stream.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for xyz, color in zip(points, rgb):
            stream.write("%.8g %.8g %.8g %d %d %d\n" % (*xyz, *color))


def make_initialization(root: Path, cameras: dict, train_ids: list[str], frame_indices: list[int],
                        min_points: int = 256, max_points: int = 60000) -> dict:
    assert not set(train_ids) & {"cam00", "cam01"}
    selected_frames = sorted({frame_indices[0], frame_indices[len(frame_indices) // 2], frame_indices[-1]})
    sift = cv2.SIFT_create(nfeatures=2500, contrastThreshold=0.012, edgeThreshold=12)
    all_points, all_colors, all_errors, point_frames, records, sources = [], [], [], [], [], []
    for frame in selected_frames:
        features = {}
        for camera_id in train_ids:
            path = root / "lr" / camera_id / f"{frame:04d}.png"
            bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if bgr is None or bgr.shape[:2] != (252, 336):
                raise ValueError(f"Invalid training LR image: {path}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            keys, descriptors = sift.detectAndCompute(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), None)
            features[camera_id] = (np.array([k.pt for k in keys], dtype=np.float64).reshape(-1, 2), descriptors, rgb)
            sources.append({"path": str(path.relative_to(root)), "sha256": sha256(path),
                            "camera_id": camera_id, "frame_index": frame, "features": len(keys)})
        for camera_a, camera_b in itertools.combinations(train_ids, 2):
            points, colors, errors, record = triangulate_pair(cameras[camera_a], cameras[camera_b],
                                                             features[camera_a], features[camera_b])
            record["frame_index"] = frame
            records.append(record)
            if len(points):
                all_points.append(points); all_colors.append(colors); all_errors.append(errors)
                point_frames.append(np.full(len(points), frame, dtype=np.int32))
        print(f"Triangulated frame {frame}: cumulative {sum(len(p) for p in all_points)} points", flush=True)
    atomic_json(root / "triangulation_diagnostics.json", {"train_camera_ids": train_ids,
                "source_frames": selected_frames, "source_images": sources,
                "pair_statistics": records, "stage": "before_spatial_filtering"})
    if not all_points:
        raise RuntimeError("No reliable pure-LR triangulation points; do not substitute held-out or HR geometry")
    xyz, rgb, error, frames = np.concatenate(all_points), np.concatenate(all_colors), np.concatenate(all_errors), np.concatenate(point_frames)
    raw_count = len(xyz)
    center = np.median(xyz, axis=0)
    radii = np.linalg.norm(xyz - center, axis=1)
    # A robust outlier rule addresses unstable very-far triangulation without HR geometry.
    clip_radius = max(float(np.quantile(radii, 0.98)) * 2.0, 1e-6)
    good = np.isfinite(radii) & (radii <= clip_radius)
    xyz, rgb, error, frames = xyz[good], rgb[good], error[good], frames[good]
    robust_extent = max(float(np.quantile(np.linalg.norm(xyz - np.median(xyz, axis=0), axis=1), 0.9)), 1e-4)
    voxel = robust_extent * 0.002
    # Retain the lowest reprojection-error point in each spatial cell.
    order = np.argsort(error, kind="stable")
    _, unique = np.unique(np.floor(xyz[order] / voxel).astype(np.int64), axis=0, return_index=True)
    indices = order[unique]
    if len(indices) > max_points:
        indices = np.random.default_rng(20260918).choice(indices, max_points, replace=False)
    xyz, rgb, error, frames = xyz[indices], rgb[indices], error[indices], frames[indices]
    if len(xyz) < min_points:
        raise RuntimeError(f"Only {len(xyz)} reliable LR initialization points (<{min_points}); inspect pair diagnostics")
    init = root / "initialization"
    init.mkdir(exist_ok=True)
    np.savez_compressed(init / "points_lr.npz", points=xyz.astype(np.float32), colors=rgb.astype(np.float32),
                        normals=np.zeros_like(xyz, dtype=np.float32), reprojection_error_lr_px=error.astype(np.float32),
                        source_frame_index=frames)
    write_ply(init / "points_lr.ply", xyz, rgb)
    report = {"method": "known_calibration_SIFT_mutual_ratio_RANSAC_epipolar_DLT_v1",
              "official_colmap_initialization": False, "used_hr_images": False,
              "used_development_or_test_images": False,
              "known_camera_calibration": True, "train_camera_ids": train_ids,
              "source_frames": selected_frames, "source_images": sources,
              "raw_point_count": raw_count, "point_count": len(xyz),
              "voxel_size_world": voxel, "reprojection_error_lr_px_median": float(np.median(error)),
              "points_per_source_frame": dict(Counter(str(int(f)) for f in frames)),
              "pair_statistics": records,
              "limitation": "Sparse multi-time points can mix deforming surfaces in the canonical initialization; all experiment branches must share this exact initialization."}
    atomic_json(init / "report.json", report)
    return {"npz_path": "initialization/points_lr.npz", "ply_path": "initialization/points_lr.ply",
            "report_path": "initialization/report.json", "point_count": len(xyz),
            "npz_sha256": sha256(init / "points_lr.npz"), "train_only_lr": True}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--num-frames", type=int, default=60)
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--min-init-points", type=int, default=256)
    parser.add_argument("--max-init-points", type=int, default=60000)
    parser.add_argument("--resume", action="store_true", help="Resume preparation with exactly matching options")
    args = parser.parse_args()
    if args.num_frames < 2 or args.frame_stride < 1 or args.start_frame < 0:
        raise ValueError("Invalid temporal window")
    frame_indices = list(range(args.start_frame, args.start_frame + args.num_frames * args.frame_stride, args.frame_stride))
    if frame_indices[-1] >= 300:
        raise ValueError("This pilot uses the official first 300 frames")
    root = args.output.resolve()
    if (root / "manifest.json").exists():
        raise FileExistsError(f"Completed dataset already exists: {root}")
    spec = {"raw_dir": str(args.raw_dir.resolve()), "frame_indices": frame_indices,
            "min_init_points": args.min_init_points, "max_init_points": args.max_init_points}
    if root.exists() and any(root.iterdir()):
        if not args.resume or not (root / "preparation_spec.json").exists():
            raise FileExistsError(f"Nonempty preparation output: {root}; inspect or explicitly resume")
        if json.loads((root / "preparation_spec.json").read_text()) != spec:
            raise ValueError("Resume options differ from the recorded preparation")
    root.mkdir(parents=True, exist_ok=True)
    atomic_json(root / "preparation_spec.json", spec)
    cv2.setNumThreads(4)
    cv2.setRNGSeed(20260918)
    torch.set_num_threads(4)
    start = time.time()
    cameras = parse_n3dv_cameras(args.raw_dir)
    if not {"cam00", "cam01"}.issubset(cameras):
        raise ValueError("Pilot requires cam00 official test and cam01 development holdout")
    train_ids = sorted(set(cameras) - {"cam00", "cam01"})
    if len(train_ids) < 3:
        raise ValueError("Insufficient training cameras")
    observations = []
    for camera_id, camera in cameras.items():
        split = "test" if camera_id == "cam00" else "dev" if camera_id == "cam01" else "train"
        observations.extend(extract_camera(camera, root, frame_indices, split))
        print(f"Prepared {camera_id} ({split}), {len(frame_indices)} frames", flush=True)
    initialization = make_initialization(root, cameras, train_ids, frame_indices,
                                         args.min_init_points, args.max_init_points)
    manifest = {"schema": "n3dv_dynamic_sr_pilot_v1", "scene": args.raw_dir.name,
                "source": "https://github.com/facebookresearch/Neural_3D_Video", "scale": 4,
                "raw_directory": str(args.raw_dir.resolve()),
                "poses_bounds_sha256": sha256(args.raw_dir / "poses_bounds.npy"),
                "frame_indices": frame_indices, "time_convention": "original_frame_index / 300.0",
                "resolutions": {"hr": [1344, 1008], "lr": [336, 252]},
                "degradation": {"raw_to_reference": "cv2.INTER_AREA resize 1352x1014, center crop x=4,y=3,w=1344,h=1008",
                                "reference_to_lr": "torch F.interpolate bicubic, align_corners=False, antialias=True, exact 4x, clamp [0,1], round uint8 PNG",
                                "quantization": "round(value*255)/255; differentiable training degradation omits this rounding", "extra_noise": False},
                "pixel_coordinate_convention": "OpenCV zero-based pixel centers; raw principal point (W/2,H/2); resize u'=(u+0.5)*s-0.5; crop subtracts x/y offset; raster projection explicitly matches these K matrices",
                "splits": {"train": train_ids, "dev": ["cam01"], "test": ["cam00"]},
                "split_note": "cam00 is the official N3DV test camera; cam01 is additionally held out for development, so pilot scores are not the official full-training-camera benchmark.",
                "cameras": cameras, "observations": observations, "initialization": initialization,
                "preparation_elapsed_seconds": time.time() - start,
                "versions": {"opencv": cv2.__version__, "numpy": np.__version__, "torch": torch.__version__},
                "script_sha256": {"prepare_n3dv.py": sha256(Path(__file__)),
                                   "n3dv_data.py": sha256(Path(__file__).with_name("n3dv_data.py"))}}
    atomic_json(root / "manifest.json", manifest)
    print(json.dumps({"manifest": str(root / "manifest.json"), "observations": len(observations),
                      "initial_points": initialization["point_count"]}), flush=True)


if __name__ == "__main__":
    main()
