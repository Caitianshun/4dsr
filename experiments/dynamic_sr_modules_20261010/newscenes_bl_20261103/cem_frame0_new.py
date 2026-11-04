"""CEM-only frame-0 COLMAP preparation, execution and acceptance.

plan reads metadata only. prepare/run require --execute and are for the owner,
after input sync/CPU-executor acceptance. No dense reconstruction or LocalDyGS
completion receipt is produced. All substantive COLMAP flags match frame 0 of
the historical run_native_colmap.py.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import resource
import shutil
import sqlite3
import subprocess
import sys
import time
import traceback

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_bl import *


def paths(scene):
    base = OUT / "cem" / scene
    return base / "native_inputs", base / "native_init"


def commands(inputs, output, colmap="/usr/local/bin/colmap"):
    native, folder = Path(inputs) / "frame_0000", Path(output) / "frame_0000"
    sparse = folder / "triangulated"
    final = Path(output) / "cem_data/colmap_0/sparse/0"
    return [
        ("frame0000_features", [colmap, "feature_extractor", "--database_path", str(folder / "database.db"),
                               "--image_path", str(native / "images"), "--ImageReader.camera_model", "PINHOLE",
                               "--ImageReader.single_camera", "0", "--SiftExtraction.use_gpu", "0"]),
        ("frame0000_matching", [colmap, "exhaustive_matcher", "--database_path", str(folder / "database.db"),
                               "--SiftMatching.use_gpu", "0", "--SiftMatching.num_threads", "1"]),
        ("frame0000_triangulation", [colmap, "point_triangulator", "--database_path", str(folder / "database.db"),
                                    "--image_path", str(native / "images"), "--input_path", str(folder / "known_poses"),
                                    "--output_path", str(sparse),
                                    "--Mapper.ba_global_function_tolerance", "0.000001"]),
        ("frame0000_model_txt", [colmap, "model_converter", "--input_path", str(sparse),
                                "--output_path", str(sparse), "--output_type", "TXT"]),
        ("cem_sparse_ply", [colmap, "model_converter", "--input_path", str(final),
                            "--output_path", str(final / "points3D.ply"), "--output_type", "PLY"])]


def plan(scene):
    mp, m, _, _, geo = validate_metadata(scene)
    inputs, out = paths(scene)
    lookup = {(o["camera_id"], o["frame_index"]): o for o in m["observations"]}
    return dict(schema="cem_frame0_plan_v1", status="CPU_PLAN_ONLY_NOT_INITIALIZED",
                scene=scene, manifest=identity(mp), frames=[0], train_cameras=geo["train_cameras"],
                geometry=geo, resolution_lr=m["resolutions"]["lr"],
                inputs=[dict(camera=c, frame=0, path=rel(mp.parent / lookup[c, 0]["lr_path"]),
                             sha256=lookup[c, 0]["lr_sha256"]) for c in geo["train_cameras"]],
                commands=[dict(name=n, argv=a) for n, a in commands(inputs, out)],
                export=rel(out / "cem_data/colmap_0/sparse/0/points3D.ply"),
                HR_read=False, heldout_pixels_read=False, COLMAP_executed=False,
                omissions=["all nonzero frames", "image_undistorter (unused by CEM sparse export)",
                           "PatchMatch", "stereo_fusion", "LocalDyGS dense union", "poses_bounds copy"],
                source_functions=identity(OLD / "localdygs_cem/prepare_native_colmap_area4.py"))


def prepare(scene):
    from PIL import Image
    mp, m, _, _, geo = validate_metadata(scene)
    inputs, _ = paths(scene)
    require(not inputs.exists(), "Native inputs must be fresh")
    # Reuse the original quaternion implementation; no source modifications.
    source = load("_cem_original_prepare", OLD / "localdygs_cem/prepare_native_colmap_area4.py")
    lookup = {(o["camera_id"], o["frame_index"]): o for o in m["observations"]}
    image_root, manual = inputs / "frame_0000/images", inputs / "frame_0000/sparse_custom"
    image_root.mkdir(parents=True)
    manual.mkdir()
    width, height = m["resolutions"]["lr"]
    cameras = ["# CAMERA_ID MODEL WIDTH HEIGHT PARAMS\n"]
    images = ["# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME\n"]
    files = []
    for index, camera in enumerate(geo["train_cameras"], 1):
        item = lookup[camera, 0]
        require(item["split"] == "train", "Held-out initialization forbidden")
        src = mp.parent / item["lr_path"]
        require(sha(src) == item["lr_sha256"], "Frame0 training LR SHA mismatch")
        with Image.open(src) as im:
            require(im.format == "PNG" and im.mode == "RGB" and im.size == (width, height),
                    "Frame0 PNG contract mismatch")
            im.verify()
        dst = image_root / (camera + ".png")
        shutil.copy2(src, dst)
        spec = m["cameras"][camera]
        k, pose = np.asarray(spec["K_lr"]), np.asarray(spec["w2c"])
        q = source.rotation_quaternion(pose[:3, :3])
        params = " ".join(format(float(x), ".17g") for x in (k[0, 0], k[1, 1], k[0, 2] + .5, k[1, 2] + .5))
        extrinsic = " ".join(format(float(x), ".17g") for x in np.r_[q, pose[:3, 3]])
        cameras.append(f"{index} PINHOLE {width} {height} {params}\n")
        images.append(f"{index} {extrinsic} {index} {camera}.png\n\n")
        files.append(dict(relative=dst.relative_to(inputs).as_posix(), sha256=sha(dst),
                          source_relative=item["lr_path"], camera=camera, frame=0))
    (manual / "cameras.txt").write_text("".join(cameras))
    (manual / "images.txt").write_text("".join(images))
    (manual / "points3D.txt").write_text("")
    for p in sorted(manual.iterdir()):
        files.append(dict(relative=p.relative_to(inputs).as_posix(), sha256=sha(p),
                          source="this scene's fixed manifest calibration"))
    receipt = dict(schema="cem_frame0_inputs_v1", scene=scene, manifest=identity(mp),
                   train_cameras=geo["train_cameras"], frames=[0], resolution_lr=[width, height],
                   world_coordinates="original manifest world; no similarity normalization",
                   COLMAP_principal_point_convention="cx_colmap=cx_manifest+0.5; cy_colmap=cy_manifest+0.5",
                   HR_images_read=False, heldout_images_read=False, files=files,
                   initializer_executed=False)
    write_new(inputs / "frame0_input_receipt.json", receipt)
    return receipt


def remap_database(database, manual, destination):
    """Original known-pose camera/image remap, including arbitrary COLMAP IDs."""
    camera_rows = {int(x[0]): x for line in (manual / "cameras.txt").read_text().splitlines()
                   if (x := line.split()) and not line.startswith("#")}
    image_rows = {x[-1]: x for line in (manual / "images.txt").read_text().splitlines()
                  if (x := line.split()) and not line.startswith("#")}
    with sqlite3.connect(database) as connection:
        db_images = connection.execute("SELECT image_id,camera_id,name FROM images ORDER BY image_id").fetchall()
        require(len(db_images) == len(image_rows) and {x[2] for x in db_images} == set(image_rows),
                "COLMAP image identity mismatch")
        remapped_cameras, remapped_images = {}, []
        for image_id, camera_id, name in db_images:
            row = image_rows[name]
            calibration = camera_rows[int(row[8])]
            params = np.asarray(list(map(float, calibration[4:])), np.float64)
            connection.execute("UPDATE cameras SET model=1,width=?,height=?,params=?,prior_focal_length=1 WHERE camera_id=?",
                               (int(calibration[2]), int(calibration[3]), params.tobytes(), camera_id))
            remapped_cameras[camera_id] = " ".join([str(camera_id), *calibration[1:]])
            remapped_images.append(" ".join([str(image_id), *row[1:8], str(camera_id), name]) + "\n")
    destination.mkdir()
    (destination / "cameras.txt").write_text("\n".join(remapped_cameras.values()) + "\n")
    (destination / "images.txt").write_text("\n".join(remapped_images) + "\n")
    (destination / "points3D.txt").write_text("")


def validate_sparse(folder, known):
    def records(path, images=False):
        lines = path.read_text().splitlines()
        result = []
        i = 0
        while i < len(lines):
            line = lines[i]
            i += 1
            if not line.strip() or line.startswith("#"):
                continue
            result.append(line.split())
            if images:
                i += 1  # the next line contains 2D tracks, not another camera
        return result
    cameras = {r[0]: r for r in records(folder / "cameras.txt")}
    reference_cameras = {r[0]: r for r in records(known / "cameras.txt")}
    require(set(cameras) == set(reference_cameras), "Triangulation camera set changed")
    for cid, row in cameras.items():
        ref = reference_cameras[cid]
        require(row[1:4] == ref[1:4] and np.allclose(list(map(float, row[4:])),
                                                  list(map(float, ref[4:])), rtol=0, atol=1e-7),
                "Triangulation changed calibrated intrinsics")
    images = {r[-1]: r for r in records(folder / "images.txt", True)}
    refs = {r[-1]: r for r in records(known / "images.txt", True)}
    require(set(images) == set(refs), "Triangulation image set changed")
    for name, row in images.items():
        ref = refs[name]
        q, qr = np.asarray(list(map(float, row[1:5]))), np.asarray(list(map(float, ref[1:5])))
        require(min(np.linalg.norm(q - qr), np.linalg.norm(q + qr)) < 1e-6
                and np.allclose(list(map(float, row[5:8])), list(map(float, ref[5:8])),
                                rtol=0, atol=1e-6), "Triangulation changed known poses")
    points = records(folder / "points3D.txt")
    require(points and all(len(r) >= 8 and np.isfinite(list(map(float, r[1:4]))).all()
                           for r in points), "No finite triangulated points")
    return len(points)


def run(scene, colmap):
    inputs, out = paths(scene)
    mp, _, _, _, _ = validate_metadata(scene)
    receipt = read(inputs / "frame0_input_receipt.json")
    require(receipt["schema"] == "cem_frame0_inputs_v1" and receipt["frames"] == [0]
            and receipt["manifest"] == identity(mp) and receipt["scene"] == scene
            and receipt["HR_images_read"] is False and receipt["heldout_images_read"] is False,
            "Frame0 input receipt mismatch")
    for entry in receipt["files"]:
        require(sha(inputs / entry["relative"]) == entry["sha256"], "Frame0 input changed")
    require("af2b8659" in subprocess.check_output([colmap, "-h"], text=True),
            "COLMAP must match historical af2b8659 deployment")
    require(not out.exists(), "Fresh native initialization output required")
    out.mkdir(parents=True)
    folder = out / "frame_0000"
    folder.mkdir()
    old_soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    soft = old_soft if old_soft == resource.RLIM_INFINITY else max(
        old_soft, 65536 if hard == resource.RLIM_INFINITY else min(65536, hard))
    resource.setrlimit(resource.RLIMIT_NOFILE, (soft, hard))
    state = dict(schema="cem_frame0_native_init_v1", status="running", scene=scene, frames=[0],
                 manifest=identity(mp), inputs=identity(inputs / "frame0_input_receipt.json"),
                 train_cameras=receipt["train_cameras"], used_HR=False, used_test_pixels=False,
                 dense_reconstruction=False, world_coordinates=receipt["world_coordinates"],
                 colmap_binary=dict(path=colmap, sha256=sha(colmap)), colmap_commit="af2b8659",
                 matcher_threads=1, commands=[], began_unix=time.time(),
                 own_nofile_before=[old_soft, hard], own_nofile_after=list(resource.getrlimit(resource.RLIMIT_NOFILE)))

    def status():
        # Only this new execution's running status is updated; frozen inputs/receipts are never overwritten.
        (out / "frame0_status.json").write_text(__import__("json").dumps(state, indent=2) + "\n")

    try:
        for name, argv in commands(inputs, out, colmap):
            if name == "frame0000_matching":
                remap_database(folder / "database.db", inputs / "frame_0000/sparse_custom",
                               folder / "known_poses")
            if name == "frame0000_triangulation":
                (folder / "triangulated").mkdir()
            if name == "cem_sparse_ply":
                state["points"] = validate_sparse(folder / "triangulated", folder / "known_poses")
                final = out / "cem_data/colmap_0/sparse/0"
                final.mkdir(parents=True)
                for p in sorted((folder / "triangulated").iterdir()):
                    if p.is_file() and p.suffix in (".txt", ".bin"):
                        shutil.copy2(p, final / p.name)
            row = dict(name=name, argv=argv, began_unix=time.time())
            state["commands"].append(row)
            status()
            with (out / (name + ".log")).open("x") as stream:
                code = subprocess.run(argv, stdout=stream, stderr=subprocess.STDOUT).returncode
            row.update(returncode=code, seconds=time.time() - row["began_unix"])
            status()
            require(code == 0, f"COLMAP failed: {name}: {code}")
        final = out / "cem_data/colmap_0/sparse/0"
        require((final / "points3D.ply").stat().st_size > 100, "Empty sparse PLY")
        state.update(status="completed", elapsed_seconds=time.time() - state["began_unix"],
                     sparse_files=[dict(path=rel(p), sha256=sha(p), bytes=p.stat().st_size)
                                   for p in sorted(final.iterdir()) if p.is_file()],
                     calibrated_poses_and_intrinsics_unchanged=True)
        write_new(out / "frame0_complete.json", state)
        write_new(out / "cem_data/frame0_sparse_receipt.json", state)
    except BaseException:
        state.update(status="failed", error=traceback.format_exc())
        raise
    finally:
        status()
    return state


def validate_completion(r, scene, manifest_identity):
    require(r["schema"] == "cem_frame0_native_init_v1" and r["status"] == "completed"
            and r["scene"] == scene and r["frames"] == [0] and not r["dense_reconstruction"]
            and r["manifest"] == manifest_identity
            and r["calibrated_poses_and_intrinsics_unchanged"] is True
            and r["used_HR"] is False and r["used_test_pixels"] is False, "CEM sparse acceptance failed")
    return r


def accepted(scene):
    _, out = paths(scene)
    r = read(out / "frame0_complete.json")
    validate_completion(r, scene, identity(manifest_path(scene)))
    require(r["inputs"] == identity(paths(scene)[0] / "frame0_input_receipt.json"), "Initializer source changed")
    for entry in r["sparse_files"]:
        require(sha(entry["path"]) == entry["sha256"], "CEM sparse export changed")
    required = {"points3D.ply", "points3D.bin", "images.bin", "cameras.bin"}
    require(required <= {Path(e["path"]).name for e in r["sparse_files"]}, "CEM binary sparse assets incomplete")
    return r


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("plan", "prepare", "run", "check"))
    p.add_argument("--scene", choices=SCENES, required=True)
    p.add_argument("--colmap", default="/usr/local/bin/colmap")
    p.add_argument("--execute", action="store_true")
    a = p.parse_args()
    if a.mode == "plan":
        result = plan(a.scene)
        write_new(OUT / "cem" / a.scene / "frame0_plan.json", result)
    elif a.mode == "check":
        result = accepted(a.scene)
    else:
        require(a.execute, "prepare/run require explicit owner --execute")
        result = prepare(a.scene) if a.mode == "prepare" else run(a.scene, a.colmap)
    print(__import__("json").dumps(dict(mode=a.mode, scene=a.scene, status=result.get("status", "prepared"))))


if __name__ == "__main__":
    main()
