#!/usr/bin/env python3
"""Prepare original-world, training-LR-only COLMAP inputs, without running COLMAP.

LocalDyGS native timestamps: 10,20,...,290,299. Frame 0 optionally supports CEM.
The source manifest supplies known calibration only; no project point initializer,
SR teacher, compensation field, HR image or held-out image is read.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
from PIL import Image


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rotation_quaternion(rotation):
    # COLMAP scalar-first quaternion conversion, same eigenvector convention as
    # each upstream llff2colmap/pre_n3d implementation.
    rxx, ryx, rzx, rxy, ryy, rzy, rxz, ryz, rzz = rotation.flat
    k = np.array([
        [rxx - ryy - rzz, 0, 0, 0],
        [ryx + rxy, ryy - rxx - rzz, 0, 0],
        [rzx + rxz, rzy + ryz, rzz - rxx - ryy, 0],
        [ryz - rzy, rzx - rxz, rxy - ryx, rxx + ryy + rzz],
    ]) / 3
    values, vectors = np.linalg.eigh(k)
    q = vectors[[3, 0, 1, 2], values.argmax()]
    return q if q[0] >= 0 else -q


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--include-frame0", action="store_true")
    a = p.parse_args()
    manifest = json.loads(a.manifest.read_text())
    if a.output.exists():
        raise SystemExit(f"Refusing existing output {a.output}")
    if "_area4" in str(a.manifest):
        raise SystemExit("This run explicitly uses the original LR manifest, not area4")
    if manifest["frame_indices"] != list(range(300)):
        raise SystemExit("Native N3DV protocol requires all 300 frames")
    train = sorted(manifest["splits"]["train"])
    if "cam00" in train or manifest["splits"]["test"] != ["cam00"]:
        raise SystemExit("Unexpected held-out camera protocol")
    lookup = {(x["camera_id"], x["frame_index"]): x for x in manifest["observations"]}
    frames = list(range(10, 300, 10)) + [299]
    if a.include_frame0:
        frames.insert(0, 0)
    width, height = manifest["resolutions"]["lr"]
    receipt = {
        "manifest_sha256": digest(a.manifest), "scene": manifest["scene"],
        "train_cameras": train, "frames": frames, "resolution_lr": [width, height],
        "world_coordinates": "original manifest world; no similarity normalization",
        "COLMAP_principal_point_convention": "corner origin: cx_colmap=cx_manifest+0.5; cy_colmap=cy_manifest+0.5",
        "HR_images_read": False, "heldout_images_read": False,
        "initial_points_created": False, "COLMAP_executed": False,
        "files": [],
    }
    for frame in frames:
        frame_root = a.output / f"frame_{frame:04d}"
        image_root = frame_root / "images"
        manual = frame_root / "sparse_custom"
        image_root.mkdir(parents=True)
        manual.mkdir()
        cameras = ["# CAMERA_ID MODEL WIDTH HEIGHT PARAMS\n"]
        images = ["# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME\n"]
        for index, camera in enumerate(train, 1):
            item = lookup[camera, frame]
            if item["split"] != "train":
                raise RuntimeError("Unexpected observation split")
            source = a.manifest.parent / item["lr_path"]
            identity = digest(source)
            if identity != item["lr_sha256"]:
                raise RuntimeError(f"LR hash mismatch {source}")
            with Image.open(source) as image:
                if image.size != (width, height):
                    raise RuntimeError(f"LR size mismatch {source}: {image.size}")
            output = image_root / f"{camera}.png"
            shutil.copy2(source, output)
            spec = manifest["cameras"][camera]
            k = np.asarray(spec["K_lr"])
            w2c = np.asarray(spec["w2c"])
            q = rotation_quaternion(w2c[:3, :3])
            translation = w2c[:3, 3]
            # COLMAP measures feature locations from the pixel corner; the
            # manifest uses integer pixel centers. Preserve the same rays.
            params = " ".join(format(float(v), ".17g") for v in (k[0, 0], k[1, 1], k[0, 2] + 0.5, k[1, 2] + 0.5))
            cameras.append(f"{index} PINHOLE {width} {height} {params}\n")
            extrinsic = " ".join(format(float(v), ".17g") for v in np.r_[q, translation])
            images.append(f"{index} {extrinsic} {index} {camera}.png\n\n")
            receipt["files"].append({"relative": str(output.relative_to(a.output)), "sha256": identity, "source_relative": item["lr_path"]})
        (manual / "cameras.txt").write_text("".join(cameras))
        (manual / "images.txt").write_text("".join(images))
        (manual / "points3D.txt").write_text("")
        for path in sorted(manual.iterdir()):
            receipt["files"].append({"relative": str(path.relative_to(a.output)), "sha256": digest(path), "source": "fixed manifest calibration; no pose/intrinsic optimization"})
    (a.output / "input_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"scene": manifest["scene"], "frame_count": len(frames), "camera_count": len(train), "file_count": len(receipt["files"]), "output": str(a.output), "COLMAP_executed": False}, indent=2))


if __name__ == "__main__":
    main()
