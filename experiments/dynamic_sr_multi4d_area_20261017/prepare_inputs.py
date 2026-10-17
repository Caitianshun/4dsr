"""CPU-only frame-zero LR triangulation and native Multi4D sampling inventory.

SIFT, mutual ratio matches, fundamental-matrix RANSAC, known-calibration
epipolar filtering, DLT triangulation and spatial filtering are minimally
adapted from experiments/dynamic_sr_20260918/prepare_n3dv.py. The changes are:
registered training splits (cam01 is legal), arbitrary registered dimensions,
exactly frame zero, explicit OpenCV seed, and source/hash receipts. No HR or
heldout image is decoded; no torch or CUDA package is imported.
"""
from __future__ import annotations

import argparse
from collections import Counter
import itertools
import json
from pathlib import Path
import random
import time

import cv2
import numpy as np

from data import checked_path, inventory, load_manifest, sha

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "experiments/dynamic_sr_20260918/prepare_n3dv.py"


def write(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def skew(vector):
    x, y, z = vector
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=np.float64)


def known_fundamental(a, b):
    wa, wb = np.asarray(a["w2c"]), np.asarray(b["w2c"])
    relative = wb @ np.linalg.inv(wa)
    essential = skew(relative[:3, 3]) @ relative[:3, :3]
    ka, kb = np.asarray(a["K_lr"]), np.asarray(b["K_lr"])
    return np.linalg.inv(kb).T @ essential @ np.linalg.inv(ka)


def sampson_distance(fundamental, a, b):
    ha, hb = np.c_[a, np.ones(len(a))], np.c_[b, np.ones(len(b))]
    fa, ftb = ha @ fundamental.T, hb @ fundamental
    numerator = np.sum(hb * fa, axis=1) ** 2
    denominator = fa[:, 0] ** 2 + fa[:, 1] ** 2 + ftb[:, 0] ** 2 + ftb[:, 1] ** 2
    return numerator / np.maximum(denominator, 1e-20)


def triangulate_pair(a, b, fa, fb, ratio=.8):
    pa, da, rgb_a = fa
    pb, db, rgb_b = fb
    record = dict(camera_a=a["camera_id"], camera_b=b["camera_id"], frame_index=0,
                  features_a=len(pa), features_b=len(pb), matches=0,
                  ransac_and_known_pose_inliers=0, accepted_points=0)
    empty = (np.empty((0, 3)), np.empty((0, 3)), np.empty((0,)), record)
    if da is None or db is None or len(da) < 8 or len(db) < 8:
        return empty
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    forward, reverse = matcher.knnMatch(da, db, k=2), matcher.knnMatch(db, da, k=2)
    reverse_good = {p[0].queryIdx: p[0].trainIdx for p in reverse
                    if len(p) == 2 and p[0].distance < ratio * p[1].distance}
    matches = [p[0] for p in forward if len(p) == 2 and p[0].distance < ratio * p[1].distance
               and reverse_good.get(p[0].trainIdx) == p[0].queryIdx]
    record["matches"] = len(matches)
    if len(matches) < 8:
        return empty
    xa = np.array([pa[m.queryIdx] for m in matches], dtype=np.float64)
    xb = np.array([pb[m.trainIdx] for m in matches], dtype=np.float64)
    _, mask = cv2.findFundamentalMat(xa, xb, cv2.FM_RANSAC, 1.5, .999)
    if mask is None or len(mask) != len(xa):
        return empty
    keep = mask.ravel().astype(bool) & (sampson_distance(known_fundamental(a, b), xa, xb) < 1.5 ** 2)
    xa, xb = xa[keep], xb[keep]
    record["ransac_and_known_pose_inliers"] = len(xa)
    if not len(xa):
        return empty
    wa, wb = np.asarray(a["w2c"]), np.asarray(b["w2c"])
    proj_a, proj_b = np.asarray(a["K_lr"]) @ wa[:3], np.asarray(b["K_lr"]) @ wb[:3]
    homogeneous = cv2.triangulatePoints(proj_a, proj_b, xa.T, xb.T).T
    valid_w = np.abs(homogeneous[:, 3]) > 1e-10
    points = homogeneous[:, :3] / np.where(valid_w, homogeneous[:, 3], 1)[:, None]
    points_h = np.c_[points, np.ones(len(points))]
    ca, cb = points_h @ wa[:3].T, points_h @ wb[:3].T
    uv_a, uv_b = points_h @ proj_a.T, points_h @ proj_b.T
    ea = np.linalg.norm(uv_a[:, :2] / np.maximum(uv_a[:, 2:], 1e-10) - xa, axis=1)
    eb = np.linalg.norm(uv_b[:, :2] / np.maximum(uv_b[:, 2:], 1e-10) - xb, axis=1)
    ra = points - np.asarray(a["c2w"])[:3, 3]
    rb = points - np.asarray(b["c2w"])[:3, 3]
    cosine = np.sum(ra * rb, axis=1) / np.maximum(np.linalg.norm(ra, axis=1) * np.linalg.norm(rb, axis=1), 1e-20)
    angle = np.degrees(np.arccos(np.clip(cosine, -1, 1)))
    keep = valid_w & np.all(np.isfinite(points), axis=1) & (ca[:, 2] > 0) & (cb[:, 2] > 0)
    keep &= (ea < 1.5) & (eb < 1.5) & (angle >= .75)
    points, xa, xb = points[keep], xa[keep], xb[keep]
    if not len(points):
        return empty
    ia, ib = np.rint(xa).astype(int), np.rint(xb).astype(int)
    for coords, rgb in ((ia, rgb_a), (ib, rgb_b)):
        coords[:, 0] = np.clip(coords[:, 0], 0, rgb.shape[1] - 1)
        coords[:, 1] = np.clip(coords[:, 1], 0, rgb.shape[0] - 1)
    colors = (rgb_a[ia[:, 1], ia[:, 0]].astype(float) + rgb_b[ib[:, 1], ib[:, 0]].astype(float)) / 510.
    errors = np.maximum(ea, eb)[keep]
    record.update(accepted_points=len(points), reprojection_error_median_lr_px=float(np.median(errors)),
                  triangulation_angle_median_deg=float(np.median(angle[keep])))
    return points, colors, errors, record


def initialize(manifest_path, destination, seed, min_points=256, max_points=60000):
    started = time.monotonic()
    manifest_path = Path(manifest_path)
    manifest, root = load_manifest(manifest_path), manifest_path.parent
    cv2.setRNGSeed(seed)
    cv2.setNumThreads(4)
    sift = cv2.SIFT_create(nfeatures=2500, contrastThreshold=.012, edgeThreshold=12)
    by_key = {(r["camera_id"], int(r["frame_index"])): r for r in manifest["observations"]}
    features, sources = {}, []
    for camera_id in manifest["splits"]["train"]:
        row = by_key[camera_id, 0]
        path = checked_path(root, row["lr_path"])
        if sha(path) != row["lr_sha256"]:
            raise ValueError(f"Frame-zero LR identity differs: {path}")
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None or tuple(bgr.shape[:2][::-1]) != tuple(manifest["resolutions"]["lr"]):
            raise ValueError(f"Invalid initialization image: {path}")
        keys, descriptors = sift.detectAndCompute(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), None)
        features[camera_id] = (np.array([k.pt for k in keys], dtype=np.float64).reshape(-1, 2),
                               descriptors, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        sources.append(dict(path=row["lr_path"], sha256=row["lr_sha256"], camera_id=camera_id,
                            frame_index=0, features=len(keys)))
    points, colors, errors, pairs = [], [], [], []
    for a, b in itertools.combinations(manifest["splits"]["train"], 2):
        xyz, rgb, error, record = triangulate_pair(manifest["cameras"][a], manifest["cameras"][b],
                                                  features[a], features[b])
        pairs.append(record)
        if len(xyz):
            points.append(xyz)
            colors.append(rgb)
            errors.append(error)
    if not points:
        raise RuntimeError("No reliable frame-zero training-LR triangulation; do not substitute HR/heldout geometry")
    xyz, rgb, error = np.concatenate(points), np.concatenate(colors), np.concatenate(errors)
    raw_count = len(xyz)
    center = np.median(xyz, axis=0)
    radii = np.linalg.norm(xyz - center, axis=1)
    clip_radius = max(float(np.quantile(radii, .98)) * 2., 1e-6)
    good = np.isfinite(radii) & (radii <= clip_radius)
    xyz, rgb, error = xyz[good], rgb[good], error[good]
    extent = max(float(np.quantile(np.linalg.norm(xyz - np.median(xyz, axis=0), axis=1), .9)), 1e-4)
    voxel = extent * .002
    order = np.argsort(error, kind="stable")
    _, unique = np.unique(np.floor(xyz[order] / voxel).astype(np.int64), axis=0, return_index=True)
    indices = order[unique]
    if len(indices) > max_points:
        indices = np.random.default_rng(seed).choice(indices, max_points, replace=False)
    xyz, rgb, error = xyz[indices], rgb[indices], error[indices]
    if len(xyz) < min_points:
        raise RuntimeError(f"Only {len(xyz)} frame-zero LR points; minimum is {min_points}")
    destination.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(destination / "points_lr.npz", points=xyz.astype(np.float32), colors=rgb.astype(np.float32),
                        normals=np.zeros_like(xyz, dtype=np.float32), reprojection_error_lr_px=error.astype(np.float32),
                        source_frame_index=np.zeros(len(xyz), dtype=np.int32))
    with (destination / "points_lr.ply").open("w") as stream:
        stream.write("ply\nformat ascii 1.0\nelement vertex " + str(len(xyz)) + "\n")
        stream.write("property float x\nproperty float y\nproperty float z\n")
        stream.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for point, color in zip(xyz, np.clip(np.rint(rgb * 255), 0, 255).astype(np.uint8)):
            stream.write("%.8g %.8g %.8g %d %d %d\n" % (*point, *color))
    report = dict(method="known_calibration_SIFT_mutual_ratio_RANSAC_epipolar_DLT_v1_frame0",
                  source_implementation=str(SOURCE.relative_to(ROOT)), source_implementation_sha256=sha(SOURCE),
                  preparation_source_sha256=sha(__file__), manifest_sha256=sha(manifest_path), seed=seed,
                  used_hr_images=False, used_development_or_test_images=False, known_camera_calibration=True,
                  official_colmap_initialization=False, train_camera_ids=manifest["splits"]["train"],
                  source_frames=[0], source_images=sources, raw_point_count=raw_count, point_count=len(xyz),
                  voxel_size_world=voxel, clip_radius_world=clip_radius,
                  reprojection_error_lr_px_median=float(np.median(error)), points_per_source_frame={"0": len(xyz)},
                  pair_statistics=pairs, CPU_seconds=time.monotonic() - started, CUDA_calls=0,
                  npz_sha256=sha(destination / "points_lr.npz"), ply_sha256=sha(destination / "points_lr.ply"),
                  limitation="Pure-LR sparse initialization using supplied calibration; not official COLMAP; static BG uses frame zero only")
    write(destination / "report.json", report)
    return dict(npz_path="initialization/points_lr.npz", ply_path="initialization/points_lr.ply",
                report_path="initialization/report.json", point_count=len(xyz),
                npz_sha256=report["npz_sha256"], report_sha256=sha(destination / "report.json"))


def schedule(records, seed, coarse_steps=2000, fine_steps=20000, batch_size=2):
    rng = random.Random(seed)
    result = dict(seed=seed, record_keys=[[r["camera_id"], int(r["frame_index"])] for r in records],
                  batch_size=batch_size, rule="native uniform pop without replacement; independent frozen RNG")
    for stage, steps in (("coarse", coarse_steps), ("fine", fine_steps)):
        bag, batches = list(range(len(records))), []
        for _ in range(steps):
            ids = []
            for _ in range(batch_size):
                ids.append(bag.pop(rng.randint(0, len(bag) - 1)))
                if not bag:
                    bag = list(range(len(records)))
            batches.append(ids)
        result[stage] = batches
        result[stage + "_counts"] = dict(Counter(str(i) for row in batches for i in row))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--teacher-index", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261007)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        checked = inventory(args.manifest, args.teacher_index)
        write(args.out / "inventory.json", checked)
        initialization = initialize(args.manifest, args.out / "initialization", args.seed)
        manifest = load_manifest(args.manifest)
        records = [r for r in manifest["observations"] if r["split"] == "train"]
        sampling = schedule(records, args.seed)
        write(args.out / "multi_schedule.json", sampling)
        receipt = dict(schema="multi4d_area_prepared_v1", status="prepared_CPU_only", scene=manifest["scene"],
                       manifest=str(args.manifest.resolve()), manifest_sha256=sha(args.manifest),
                       teacher_index=str(args.teacher_index.resolve()), teacher_index_sha256=sha(args.teacher_index),
                       initialization=initialization, schedule_sha256=sha(args.out / "multi_schedule.json"),
                       inventory_sha256=sha(args.out / "inventory.json"), seed=args.seed,
                       sources={str(Path(p).relative_to(ROOT)): sha(p) for p in (__file__, Path(__file__).with_name("data.py"), SOURCE)},
                       CPU_seconds=time.monotonic() - started, CUDA_calls=0,
                       used_HR_pixels=False, used_heldout_pixels=False)
        write(args.out / "input_identity.json", receipt)
        write(args.out / "complete.json", receipt)
        print(json.dumps(dict(status=receipt["status"], scene=manifest["scene"], initialization=initialization,
                              training_observations=len(records), out=str(args.out), CPU_seconds=receipt["CPU_seconds"])))
    except BaseException as error:
        write(args.out / "failed.json", dict(status="failed", type=type(error).__name__, error=str(error)))
        raise


if __name__ == "__main__":
    main()
