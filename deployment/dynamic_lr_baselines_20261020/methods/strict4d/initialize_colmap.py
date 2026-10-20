#!/usr/bin/env python3
"""Author-style known-pose frame-zero COLMAP initialization.

--phase sparse is CPU-only. --phase dense invokes COLMAP PatchMatch CUDA and
must be dispatched by the root GPU broker. Neither phase imports training code.
This program only consumes the registered stage's frame0_init training pixels.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess
import time

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def execute(argv, records):
    started = time.time()
    result = subprocess.run([str(v) for v in argv], check=False)
    records.append({"argv": [str(v) for v in argv], "returncode": result.returncode,
                    "elapsed_seconds": time.time() - started})
    if result.returncode:
        raise subprocess.CalledProcessError(result.returncode, argv)


def main(a):
    dataset = a.dataset.resolve()
    record = json.loads((dataset / "stage_receipt.json").read_text())
    folder = dataset / "frame0_init"
    registered = {r["name"]: r for r in record["training_image_records"] if r["frame"] == 0}
    if set(p.name for p in (folder / "images").iterdir()) != set(registered):
        raise ValueError("Frame-zero directory contains unregistered inputs")
    for name, row in registered.items():
        if sha(folder / "images" / name) != row["sha256"]:
            raise ValueError("Frame-zero LR bytes differ")
    audit_path = folder / (a.phase + "_initialization_receipt.json")
    if audit_path.exists():
        raise FileExistsError(audit_path)
    commands = []
    status = {"phase": a.phase, "status": "running", "training_pixel_source": "registered training frame0 LR only",
              "manifest_sha256": record["manifest_sha256"], "used_HR_pixels": False, "used_heldout_pixels": False,
              "commands": commands}
    try:
        if a.phase == "sparse":
            database = folder / "database.db"
            if database.exists():
                raise FileExistsError(database)
            execute([a.colmap, "feature_extractor", "--database_path", database, "--image_path", folder/"images",
                     "--ImageReader.camera_model", "PINHOLE", "--ImageReader.single_camera", "0",
                     "--SiftExtraction.use_gpu", "0"], commands)
            cameras = {}
            for line in (folder / "sparse/0/cameras.txt").read_text().splitlines():
                cols = line.split()
                if cols and not line.startswith("#"):
                    cameras[int(cols[0])] = cols
            originals = {}
            for line in (folder / "sparse/0/images.txt").read_text().splitlines():
                cols = line.split()
                if cols and not line.startswith("#"):
                    originals[cols[-1]] = cols
            connection = sqlite3.connect(database)
            db_images = connection.execute("SELECT image_id,camera_id,name FROM images ORDER BY image_id").fetchall()
            if {r[2] for r in db_images} != set(registered):
                raise ValueError("COLMAP database input identity mismatch")
            generated_cameras, generated_images = {}, []
            for image_id, camera_id, name in db_images:
                cols = originals[name]
                calibration = cameras[int(cols[8])]
                params = np.asarray(list(map(float, calibration[4:])), dtype=np.float64)
                connection.execute("UPDATE cameras SET model=1,width=?,height=?,params=?,prior_focal_length=1 WHERE camera_id=?",
                                   (int(calibration[2]), int(calibration[3]), params.tobytes(), camera_id))
                generated_cameras[camera_id] = " ".join([str(camera_id), *calibration[1:]])
                generated_images.append(" ".join([str(image_id), *cols[1:8], str(camera_id), name]) + "\n")
            connection.commit()
            connection.close()
            created = folder / "created/sparse"
            created.mkdir(parents=True)
            (created/"cameras.txt").write_text("\n".join(generated_cameras.values()) + "\n")
            (created/"images.txt").write_text("\n".join(generated_images) + "\n")
            (created/"points3D.txt").write_text("")
            execute([a.colmap, "exhaustive_matcher", "--database_path", database, "--SiftMatching.use_gpu", "0"], commands)
            triangulated = folder / "triangulated/sparse"
            triangulated.mkdir(parents=True)
            execute([a.colmap, "point_triangulator", "--database_path", database, "--image_path", folder/"images",
                     "--input_path", created, "--output_path", triangulated], commands)
            execute([a.colmap, "model_converter", "--input_path", triangulated, "--output_path", triangulated,
                     "--output_type", "TXT"], commands)
            point_lines = [l for l in (triangulated/"points3D.txt").read_text().splitlines() if l and not l.startswith("#")]
            if not point_lines:
                raise ValueError("No sparse LR geometry; do not fall back to random/SIFT/HR")
            status.update(sparse_point_count=len(point_lines), GPU_calls=0)
        else:
            sparse_receipt = json.loads((folder / "sparse_initialization_receipt.json").read_text())
            if sparse_receipt["status"] != "complete":
                raise ValueError("Sparse initialization did not complete")
            dense = folder / "dense"
            dense.mkdir()
            execute([a.colmap, "image_undistorter", "--image_path", folder/"images", "--input_path", folder/"triangulated/sparse",
                     "--output_path", dense], commands)
            execute([a.colmap, "patch_match_stereo", "--workspace_path", dense, "--PatchMatchStereo.gpu_index", "0"], commands)
            destination = dataset / "points3d.ply"
            if destination.exists():
                raise FileExistsError(destination)
            execute([a.colmap, "stereo_fusion", "--workspace_path", dense, "--output_path", destination], commands)
            if not destination.is_file() or destination.stat().st_size < 1024:
                raise ValueError("Missing/empty COLMAP dense initialization")
            status.update(point_cloud=str(destination), point_cloud_sha256=sha(destination), GPU_calls="COLMAP PatchMatch only")
        status["status"] = "complete"
    except BaseException as error:
        status.update(status="failed", error=repr(error))
        raise
    finally:
        audit_path.write_text(json.dumps(status, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--phase", choices=["sparse", "dense"], required=True)
    p.add_argument("--colmap", default="colmap")
    main(p.parse_args())
