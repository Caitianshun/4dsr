"""Training-area4-LR-only, fixed-calibration native CEM frame-zero COLMAP.

The frame-zero feature, matching and triangulation commands follow the audited
20261020 CEM baseline initializer. LocalDyGS multi-time dense fusion is omitted
because CEM never consumes it. No Torch/CUDA, teacher or HR image is imported.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import shutil
import sqlite3
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "deployment/dynamic_lr_baselines_20261020/methods/localdygs_cem"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def module(name):
    spec = importlib.util.spec_from_file_location("cem_initialization_" + name, BASE / (name + ".py"))
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def manifest(path):
    value = json.loads(path.read_text())
    assert path.parent.name in {"cook_spinach_area4", "cut_roasted_beef_area4"}
    assert value["frame_indices"] == list(range(300))
    assert value["splits"]["test"] == ["cam00"]
    assert "cam00" not in value["splits"]["train"]
    assert value["resolutions"]["lr"] == [336, 252]
    assert value["resolutions"]["hr"] == [1344, 1008]
    return value


def prepare(path, destination):
    import numpy as np
    from PIL import Image
    value = manifest(path)
    assert not destination.exists(), "Preserve previous inputs; select a new path"
    quaternion = module("prepare_native_colmap").rotation_quaternion
    images, sparse = destination / "images", destination / "sparse_custom"
    images.mkdir(parents=True)
    sparse.mkdir()
    lookup = {(x["camera_id"], x["frame_index"]): x for x in value["observations"]}
    cameras = ["# CAMERA_ID MODEL WIDTH HEIGHT PARAMS\n"]
    poses = ["# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME\n"]
    files = []
    for index, camera in enumerate(sorted(value["splits"]["train"]), 1):
        row = lookup[camera, 0]
        assert row["split"] == "train"
        source = path.parent / row["lr_path"]
        assert sha(source) == row["lr_sha256"]
        with Image.open(source) as image:
            assert image.size == (336, 252) and image.mode == "RGB"
        target = images / (camera + ".png")
        shutil.copy2(source, target)
        files.append(dict(relative=str(target.relative_to(destination)), sha256=sha(target),
                          camera=camera, frame=0, source_relative=row["lr_path"]))
        k = np.asarray(value["cameras"][camera]["K_lr"])
        w2c = np.asarray(value["cameras"][camera]["w2c"])
        params = " ".join(format(float(x), ".17g") for x in [k[0, 0], k[1, 1], k[0, 2]+.5, k[1, 2]+.5])
        cameras.append(f"{index} PINHOLE 336 252 {params}\n")
        extrinsics = " ".join(format(float(x), ".17g") for x in np.r_[quaternion(w2c[:3, :3]), w2c[:3, 3]])
        poses.append(f"{index} {extrinsics} {index} {camera}.png\n\n")
    for name, text in [("cameras.txt", "".join(cameras)), ("images.txt", "".join(poses)), ("points3D.txt", "")]:
        output = sparse / name
        output.write_text(text)
        files.append(dict(relative=str(output.relative_to(destination)), sha256=sha(output)))
    receipt = dict(status="prepared_CPU_only", scene=value["scene"], manifest_sha256=sha(path),
                   train_cameras=sorted(value["splits"]["train"]), native_frames=[0],
                   source_script_sha256=sha(__file__), quaternion_source_sha256=sha(BASE / "prepare_native_colmap.py"),
                   used_HR=False, used_test_pixels=False, used_SR_teacher=False, CUDA_calls=0,
                   COLMAP_principal_point_convention="corner_origin_manifest_center_plus_half_pixel",
                   files=files)
    write(destination / "input_receipt.json", receipt)
    return receipt


def run(path, inputs, output, colmap, expected_sha):
    # Create a terminal-state channel before importing optional CPU packages or
    # resolving provenance. Never rewrite any prior attempt, including failure.
    assert not output.exists(), "Preserve the previous initialization, including failures"
    output.mkdir(parents=True)
    began = time.monotonic()
    state = dict(status="running", phase="bootstrap",
                 started_UTC=datetime.now(timezone.utc).isoformat(),
                 manifest_path=str(path), inputs_path=str(inputs), native_frames=[0],
                 used_HR=False, used_test_pixels=False, used_SR_teacher=False,
                 CUDA_calls=0, CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES"),
                 commands=[], python_executable=sys.executable)
    write(output / "status.json", state)

    def execute(argv, name):
        start = time.monotonic()
        argv = [str(x) for x in argv]
        row = dict(name=name, argv=argv)
        state["commands"].append(row)
        state["phase"] = name
        write(output / "status.json", state)
        with (output / (name + ".log")).open("x") as stream:
            result = subprocess.run(argv, stdout=stream, stderr=subprocess.STDOUT,
                                    env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
        row.update(returncode=result.returncode, seconds=time.monotonic()-start)
        write(output / "status.json", state)
        assert result.returncode == 0, f"COLMAP {name} failed: {result.returncode}"

    try:
        assert os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU initialization requires explicit empty CUDA_VISIBLE_DEVICES"
        import numpy as np
        from PIL import Image
        value = manifest(path)
        source = json.loads((inputs / "input_receipt.json").read_text())
        assert source["manifest_sha256"] == sha(path)
        assert not source["used_HR"] and not source["used_test_pixels"] and not source["used_SR_teacher"]
        assert source["native_frames"] == [0] and source["CUDA_calls"] == 0
        assert source["train_cameras"] == sorted(value["splits"]["train"])
        helpers = {name: sha(BASE / (name + ".py")) for name in
                   ("prepare_native_colmap", "audit_native_initialization", "run_native_colmap")}
        assert source["quaternion_source_sha256"] == helpers["prepare_native_colmap"]
        lookup = {(x["camera_id"], x["frame_index"]): x for x in value["observations"]}
        for row in source["files"]:
            relative = Path(row["relative"])
            assert not relative.is_absolute() and ".." not in relative.parts
            assert sha(inputs / relative) == row["sha256"]
            if "camera" in row:
                observation = lookup[row["camera"], 0]
                assert row["frame"] == 0 and observation["split"] == "train"
                assert row["source_relative"] == observation["lr_path"]
                assert row["sha256"] == observation["lr_sha256"]
                with Image.open(inputs / relative) as image:
                    assert image.mode == "RGB" and image.size == (336, 252)
        assert sha(colmap) == expected_sha, "COLMAP binary identity differs"
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        target = 65536 if hard == resource.RLIM_INFINITY else min(65536, hard)
        if soft != resource.RLIM_INFINITY:
            resource.setrlimit(resource.RLIMIT_NOFILE, (max(soft, target), hard))
        state.update(manifest_sha256=sha(path), input_receipt_sha256=sha(inputs / "input_receipt.json"),
                     COLMAP_binary_sha256=sha(colmap),
                     COLMAP_help_header=subprocess.check_output([str(colmap), "-h"], text=True,
                         env={**os.environ, "CUDA_VISIBLE_DEVICES": ""}).splitlines()[:5],
                     source_script_sha256=sha(__file__),
                     native_command_source_sha256=helpers["run_native_colmap"],
                     helper_source_sha256=helpers,
                     input_preparer_source_sha256=source["source_script_sha256"],
                     input_reuse_policy="Original input receipt retained; hashes, train-frame0 membership, LR bytes and quaternion helper reverified. Current runner need not equal original preparer.",
                     NOFILE_before=[soft, hard], NOFILE_after=list(resource.getrlimit(resource.RLIMIT_NOFILE)))
        write(output / "status.json", state)
        database = output / "database.db"
        execute([colmap, "feature_extractor", "--database_path", database, "--image_path", inputs / "images",
                 "--ImageReader.camera_model", "PINHOLE", "--ImageReader.single_camera", "0",
                 "--SiftExtraction.use_gpu", "0"], "features")
        rows = module("audit_native_initialization").rows
        camera_rows = {int(r[0]): r for r in rows(inputs / "sparse_custom/cameras.txt")}
        image_rows = {r[-1]: r for r in rows(inputs / "sparse_custom/images.txt")}
        connection = sqlite3.connect(database)
        db_images = connection.execute("SELECT image_id,camera_id,name FROM images ORDER BY image_id").fetchall()
        assert {r[2] for r in db_images} == set(image_rows)
        remapped_cameras, remapped_images = {}, []
        for image_id, camera_id, name in db_images:
            row = image_rows[name]
            calibration = camera_rows[int(row[8])]
            params = np.asarray(list(map(float, calibration[4:])), dtype=np.float64)
            connection.execute("UPDATE cameras SET model=1,width=?,height=?,params=?,prior_focal_length=1 WHERE camera_id=?",
                               (int(calibration[2]), int(calibration[3]), params.tobytes(), camera_id))
            remapped_cameras[camera_id] = " ".join([str(camera_id), *calibration[1:]])
            remapped_images.append(" ".join([str(image_id), *row[1:8], str(camera_id), name]) + "\n")
        connection.commit()
        connection.close()
        manual = output / "known_poses"
        manual.mkdir()
        (manual / "cameras.txt").write_text("\n".join(remapped_cameras.values()) + "\n")
        (manual / "images.txt").write_text("\n".join(remapped_images) + "\n")
        (manual / "points3D.txt").write_text("")
        execute([colmap, "exhaustive_matcher", "--database_path", database,
                 "--SiftMatching.use_gpu", "0", "--SiftMatching.num_threads", "1"], "matching")
        sparse = output / "cem_data/colmap_0/sparse/0"
        sparse.mkdir(parents=True)
        execute([colmap, "point_triangulator", "--database_path", database, "--image_path", inputs / "images",
                 "--input_path", manual, "--output_path", sparse,
                 "--Mapper.ba_global_function_tolerance", "0.000001"], "triangulation")
        execute([colmap, "model_converter", "--input_path", sparse, "--output_path", sparse,
                 "--output_type", "TXT"], "model_txt")
        count = len(rows(sparse / "points3D.txt"))
        assert count > 0, "No native sparse points; no alternative initializer allowed"
        rotation = module("audit_native_initialization").rotation
        cameras = {int(r[0]): r for r in rows(sparse / "cameras.txt")}
        images = [r for r in rows(sparse / "images.txt") if len(r) == 10 and r[-1].endswith(".png")]
        assert sorted(Path(r[-1]).stem for r in images) == sorted(value["splits"]["train"])
        errors = []
        for image in images:
            camera = Path(image[-1]).stem
            k = np.asarray(value["cameras"][camera]["K_lr"])
            w2c = np.asarray(value["cameras"][camera]["w2c"])
            intrinsics = cameras[int(image[8])]
            assert intrinsics[1:4] == ["PINHOLE", "336", "252"]
            k_error = float(np.abs(np.asarray(intrinsics[4:], float) - [k[0, 0], k[1, 1], k[0, 2]+.5, k[1, 2]+.5]).max())
            pose_error = max(float(np.abs(rotation(np.asarray(image[1:5], float))-w2c[:3, :3]).max()),
                             float(np.abs(np.asarray(image[5:8], float)-w2c[:3, 3]).max()))
            assert k_error < 1e-9 and pose_error < 1e-9
            errors.append(dict(camera=camera, K_error=k_error, pose_error=pose_error))
        execute([colmap, "model_converter", "--input_path", sparse, "--output_path", sparse / "points3D.ply",
                 "--output_type", "PLY"], "sparse_ply")
        pointcloud = sparse / "points3D.ply"
        state.update(status="completed", COLMAP_poses_intrinsics_unchanged=True,
                     camera_optimization_used_to_fit_data=False, camera_records=errors,
                     cem_initialization=dict(path=str(pointcloud), sha256=sha(pointcloud), points=count),
                     elapsed_seconds=time.monotonic()-began)
        calibration = dict(status="passed", manifest_sha256=sha(path), native_frames=[0],
                           COLMAP_poses_intrinsics_unchanged=True,
                           camera_optimization_used_to_fit_data=False,
                           max_K_parameter_error=max(x["K_error"] for x in errors),
                           max_pose_matrix_error=max(x["pose_error"] for x in errors),
                           native_camera_records=len(errors), camera_records=errors)
        write(output / "camera_initialization_acceptance.json", calibration)
        write(output / "cem_data/camera_initialization_acceptance.json", calibration)
        state["camera_acceptance_sha256"] = sha(output / "camera_initialization_acceptance.json")
        write(output / "cem_data/native_initialization_receipt.json", state)
        write(output / "complete.json", state)
    except BaseException:
        state.update(status="failed", error=traceback.format_exc(), elapsed_seconds=time.monotonic()-began)
        raise
    finally:
        write(output / "status.json", state)
    return state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["prepare", "run"])
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--colmap", type=Path, default=Path("/usr/local/bin/colmap"))
    parser.add_argument("--colmap-sha256")
    args = parser.parse_args()
    result = prepare(args.manifest, args.output) if args.mode == "prepare" else run(
        args.manifest, args.inputs, args.output, args.colmap, args.colmap_sha256)
    print(json.dumps({k: result[k] for k in ["status", "used_HR", "used_test_pixels"]}))


if __name__ == "__main__":
    main()
