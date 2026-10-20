#!/usr/bin/env python3
"""Stage registered LR-only observations for the official native 4DGS readers.

CPU only. It copies/links training LR pixels, writes camera metadata and native
frame-zero COLMAP inputs. It never opens HR pixels or held-out pixels. The
coordinate normalization is adapted from Yang's scripts/n3v2blender.py. Source
and initialization provenance must remain alongside each resulting dataset.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess

import numpy as np
from PIL import Image

YANG_COMMIT = "3fe460f1f2342526bccf6c01b63da125faf0fa94"
C4_COMMIT = "ed6a3cb69782c4147151b3898944bc38132bae00"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def rotation(a, b):
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v, c = np.cross(a, b), float(a @ b)
    s = np.linalg.norm(v)
    if s < 1e-10:
        if c > 0:
            return np.eye(3)
        axis = np.cross(a, [1., 0., 0.])
        if np.linalg.norm(axis) < 1e-8:
            axis = np.cross(a, [0., 1., 0.])
        axis /= np.linalg.norm(axis)
        return 2 * np.outer(axis, axis) - np.eye(3)
    k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + k + k @ k * ((1 - c) / (s * s + 1e-10))


def closest(oa, da, ob, db):
    da, db = da / np.linalg.norm(da), db / np.linalg.norm(db)
    c = np.cross(da, db)
    den = np.linalg.norm(c) ** 2
    t = ob - oa
    ta = min(float(np.linalg.det([t, db, c]) / (den + 1e-10)), 0.)
    tb = min(float(np.linalg.det([t, da, c]) / (den + 1e-10)), 0.)
    return (oa + ta * da + ob + tb * db) * .5, den


def normalized_poses(manifest):
    """Author's up-vector, look-at-center and average-radius=4 convention."""
    ids = sorted(manifest["cameras"])
    cv_to_gl = np.diag([1., -1., -1., 1.])
    world_basis = np.eye(4)[[1, 0, 2, 3]]
    world_basis[2] *= -1
    poses = np.stack([world_basis @ np.asarray(manifest["cameras"][c]["c2w"]) @ cv_to_gl for c in ids])
    up = poses[:, :3, 1].sum(0)
    align = np.eye(4)
    align[:3, :3] = rotation(up, np.array([0., 0., 1.]))
    poses = align @ poses
    total, weight = np.zeros(3), 0.
    for i in range(len(poses)):
        for j in range(i + 1, len(poses)):
            p, w = closest(poses[i, :3, 3], poses[i, :3, 2], poses[j, :3, 3], poses[j, :3, 2])
            if w > .01:
                total += p * w
                weight += w
    if weight <= 0:
        raise ValueError("Degenerate calibrated camera layout")
    center = total / weight
    poses[:, :3, 3] -= center
    scale = 4. / np.linalg.norm(poses[:, :3, 3], axis=-1).mean()
    poses[:, :3, 3] *= scale
    # A world-space similarity for transforming initialization points.
    world = np.eye(4)
    world[:3, :3] = scale * (align @ world_basis)[:3, :3]
    world[:3, 3] = -scale * center
    return dict(zip(ids, poses)), world


def quaternion(r):
    # Stable COLMAP convention [qw,qx,qy,qz], adapted from rotmat2qvec.
    rxx, ryx, rzx, rxy, ryy, rzy, rxz, ryz, rzz = r.flat
    k = np.array([[rxx-ryy-rzz, 0, 0, 0], [ryx+rxy, ryy-rxx-rzz, 0, 0],
                  [rzx+rxz, rzy+ryz, rzz-rxx-ryy, 0],
                  [ryz-rzy, rzx-rxz, rxy-ryx, rxx+ryy+rzz]]) / 3.
    _, v = np.linalg.eigh(k)
    q = v[[3, 0, 1, 2], -1]
    return -q if q[0] < 0 else q


def write_colmap(folder, manifest, poses):
    """Only frame zero of registered training cameras appears in this model."""
    sparse = folder / "sparse/0"
    sparse.mkdir(parents=True)
    width, height = manifest["resolutions"]["lr"]
    ids = manifest["splits"]["train"]
    cameras, images = [], []
    gl_to_cv = np.diag([1., -1., -1., 1.])
    with (sparse / "cameras.bin").open("wb") as cb, (sparse / "images.bin").open("wb") as ib:
        cb.write(struct.pack("<Q", len(ids)))
        ib.write(struct.pack("<Q", len(ids)))
        for idx, camera in enumerate(ids, 1):
            k = np.asarray(manifest["cameras"][camera]["K_lr"])
            # COLMAP/author camera uses corner-origin coordinates; manifest K
            # uses integer pixel centers. +0.5 preserves the registered rays.
            params = [k[0, 0], k[1, 1], k[0, 2] + .5, k[1, 2] + .5]
            cb.write(struct.pack("<iiQQdddd", idx, 1, width, height, *params))
            cameras.append(" ".join(map(str, [idx, "PINHOLE", width, height, *params])))
            w2c = np.linalg.inv(poses[camera] @ gl_to_cv)
            q, t = quaternion(w2c[:3, :3]), w2c[:3, 3]
            name = f"{camera}_0000.png"
            ib.write(struct.pack("<idddddddi", idx, *q, *t, idx))
            ib.write(name.encode() + b"\0")
            ib.write(struct.pack("<Q", 0))
            images.append(" ".join(map(str, [idx, *q, *t, idx, name])) + "\n")
    (sparse / "cameras.txt").write_text("\n".join(cameras) + "\n")
    (sparse / "images.txt").write_text("\n".join(images) + "\n")
    (sparse / "points3D.txt").write_text("")
    (sparse / "points3D.bin").write_bytes(struct.pack("<Q", 0))


def projection_audit(manifest, poses):
    errors = []
    for c, pose in poses.items():
        k = np.asarray(manifest["cameras"][c]["K_lr"])
        w, h = manifest["resolutions"]["lr"]
        xyz_cv = np.array([[-1., -.5, 5.], [0., 0., 10.], [2., 1., 15.]])
        # Equivalent native clip projection then rasterizer ndc2Pix.
        cx, cy = k[0, 2] + .5, k[1, 2] + .5
        ndc = xyz_cv[:, :2] / xyz_cv[:, 2:] * [2*k[0, 0]/w, 2*k[1, 1]/h]
        ndc += [(2*cx-w)/w, (2*cy-h)/h]
        actual = ((ndc + 1) * [w, h] - 1) / 2
        expected = xyz_cv @ k.T
        expected = expected[:, :2] / expected[:, 2:]
        errors.append(float(np.max(np.abs(actual - expected))))
        # Exercise the cv -> native gl -> reader cv conversion.
        recovered = pose @ np.diag([1., -1., -1., 1.])
        if not np.allclose(recovered[:3, :3].T @ recovered[:3, :3], np.eye(3), atol=1e-7):
            raise ValueError("Non-orthogonal native pose")
    if max(errors) >= 1e-9:
        raise ValueError(errors)
    return {"max_error_lr_pixels": max(errors), "camera_count": len(errors),
            "principal_point_conversion": "native cx,cy = manifest K_lr cx,cy + 0.5"}


def stage(args):
    manifest_path = args.manifest.resolve()
    m = json.loads(manifest_path.read_text())
    if m["splits"]["test"] != ["cam00"] or m["splits"].get("dev") or "cam00" in m["splits"]["train"]:
        raise ValueError("Requires registered cam00-only holdout")
    if m["scale"] != 4 or m["frame_indices"] != list(range(300)):
        raise ValueError("Requires registered 300-frame x4 data")
    rows = [r for r in m["observations"] if r["split"] == "train"]
    expected = {(c, f) for c in m["splits"]["train"] for f in range(300)}
    if {(r["camera_id"], r["frame_index"]) for r in rows} != expected or len(rows) != len(expected):
        raise ValueError("Incomplete or duplicate training observations")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / "images").mkdir()
    poses, similarity = normalized_poses(m)
    records, frames = [], []
    for row in rows:
        source = (manifest_path.parent / row["lr_path"]).resolve()
        if not source.is_relative_to(manifest_path.parent.resolve()) or sha(source) != row["lr_sha256"]:
            raise ValueError(f"Illegal/changed training LR {source}")
        with Image.open(source) as image:
            if list(image.size) != m["resolutions"]["lr"] or image.mode != "RGB":
                raise ValueError(f"Invalid LR image {source}")
        name = f"{row['camera_id']}_{row['frame_index']:04d}.png"
        try:
            os.link(source, out / "images" / name)
        except OSError:
            shutil.copy2(source, out / "images" / name)
        k = np.asarray(m["cameras"][row["camera_id"]]["K_lr"])
        frames.append({"file_path": f"images/{name[:-4]}", "transform_matrix": poses[row["camera_id"]].tolist(),
                       "time": row["frame_index"] / 30., "fl_x": k[0, 0], "fl_y": k[1, 1],
                       "cx": k[0, 2] + .5, "cy": k[1, 2] + .5})
        records.append({"name": name, "manifest_lr_path": row["lr_path"], "sha256": row["lr_sha256"],
                        "camera": row["camera_id"], "frame": row["frame_index"]})
    w, h = m["resolutions"]["lr"]
    common = {"w": w, "h": h}
    write(out / "transforms_train.json", {**common, "frames": frames})
    # Empty test data prevents the native training reporter from decoding any
    # held-out pixels. HR evaluation is a separate post-checkpoint operation.
    write(out / "transforms_test.json", {**common, "frames": []})
    frame0 = out / "frame0_init"
    (frame0 / "images").mkdir(parents=True)
    for c in m["splits"]["train"]:
        os.link(out / "images" / f"{c}_0000.png", frame0 / "images" / f"{c}_0000.png")
    write_colmap(frame0, m, poses)
    receipt = {"schema": "strict4d_native_lr_stage_v1", "scene": m["scene"],
               "manifest_path": str(manifest_path), "manifest_sha256": sha(manifest_path),
               "training_observations": len(rows), "training_cameras": m["splits"]["train"],
               "test_cameras": ["cam00"], "frame_indices": list(range(300)), "resolutions": m["resolutions"],
               "training_image_records": records, "used_HR_pixels": False, "used_heldout_pixels": False,
               "native_time": "frame_index/30; [0,10] native duration; all frames 0..299",
               "native_world_from_manifest_world": similarity.tolist(),
               "normalization": "Yang n3v2blender up, look-at center, average radius 4; known camera metadata only",
               "projection_audit": projection_audit(m, poses),
               "initialization_status": "pending; no random-point/SIFT fallback permitted",
               "preparation_source_sha256": sha(__file__)}
    write(out / "stage_receipt.json", receipt)
    print(json.dumps({k: receipt[k] for k in ["scene", "training_observations", "manifest_sha256", "projection_audit", "initialization_status"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    stage(parser.parse_args())
