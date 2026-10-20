"""Run author-style native initialization inside one admitted GPU queue.

Uses COLMAP CPU SIFT/matching with fixed calibrated cameras. The LocalDyGS
multi-time PatchMatch stage uses the one broker-visible GPU. Failed outputs
are retained and the script does not replay or substitute another initializer.
"""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import shutil
import sqlite3
import subprocess
import time
import traceback
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--colmap", default="/usr/local/bin/colmap")
    p.add_argument("--poses-bounds", type=Path, required=True)
    p.add_argument("--resume-failed", action="store_true", help="Explicit continuation of the archived failed matcher; never automatic")
    p.add_argument("--matcher-threads", type=int, default=1)
    p.add_argument("--colmap-commit", default="af2b8659")
    a = p.parse_args()
    if a.matcher_threads != 1:
        raise SystemExit("Deployed COLMAP camera-cache race requires exactly one CPU verifier thread")
    colmap_help = subprocess.check_output([a.colmap, "-h"], text=True)
    if a.colmap_commit not in colmap_help:
        raise SystemExit("COLMAP version differs from the audited deployment")
    old_limits = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (max(old_limits[0], min(65536, old_limits[1])) if old_limits[1] != resource.RLIM_INFINITY else max(65536, old_limits[0]), old_limits[1]))
    if a.output.exists() and not a.resume_failed:
        raise SystemExit(f"Refusing existing result directory: {a.output}")
    source = json.loads((a.inputs / "input_receipt.json").read_text())
    if "COLMAP_principal_point_convention" not in source:
        raise SystemExit("Requires checked COLMAP corner-origin calibration")
    for entry in source["files"]:
        if sha(a.inputs / entry["relative"]) != entry["sha256"]:
            raise SystemExit("Frozen initialization input mismatch")
    a.output.mkdir(parents=True, exist_ok=a.resume_failed)
    state = dict(status="running", source_receipt_sha256=sha(a.inputs / "input_receipt.json"),
                 world_coordinates=source["world_coordinates"], used_HR=False, used_test_pixels=False,
                 native_frames=source["frames"], commands=[], began_unix=time.time())
    status_path = a.output / "status.json"
    reused = {}
    logs = a.output
    if a.resume_failed:
        previous = json.loads(status_path.read_text())
        if previous["status"] != "failed" or previous["source_receipt_sha256"] != state["source_receipt_sha256"]:
            raise SystemExit("Explicit continuation requires a matching terminal failed initializer")
        failed = [x for x in previous["commands"] if x.get("returncode", 0)]
        if len(failed) != 1 or not failed[0]["name"].endswith("_matching") or failed[0]["returncode"] != -6:
            raise SystemExit("This continuation is restricted to the diagnosed COLMAP matcher cache abort")
        archive = a.output / "failed_matcher_attempt_20261020"
        archive.mkdir()
        shutil.copy2(status_path, archive / "status.json")
        failed_frame = failed[0]["name"][5:9]
        db = a.output / f"frame_{failed_frame}/database.db"
        shutil.copy2(db, archive / "database.db")
        shutil.copy2(a.output / (failed[0]["name"] + ".log"), archive / "matcher.log")
        preserved = []
        for path in sorted(a.output.glob("frame_*/fused_*.ply")):
            frame = int(path.parent.name.split("_")[-1])
            stages = [x for x in previous["commands"] if x["name"].startswith(f"frame{frame:04d}_")]
            assert len(stages) == 7 and all(x.get("returncode") == 0 for x in stages)
            preserved.append(dict(path=str(path.relative_to(a.output)), sha256=sha(path), frame=frame))
        write(archive / "preserved_dense_points.json", dict(points=preserved, failed_database_sha256=sha(archive / "database.db"),
              prior_status_sha256=sha(archive / "status.json"), completed_dense_frames=len(preserved)))
        state["commands"] = previous["commands"]
        reused = {x["name"]: x for x in previous["commands"] if x.get("returncode") == 0}
        state["previous_failed_attempt"] = dict(path=str(archive), status_sha256=sha(archive / "status.json"),
              preserved_points_receipt_sha256=sha(archive / "preserved_dense_points.json"))
        state["original_began_unix"] = previous["began_unix"]
        logs = a.output / "continuation_01"
        logs.mkdir()
    state["runtime_adaptation"] = dict(colmap_commit=a.colmap_commit, colmap_binary_sha256=sha(Path(a.colmap)),
          cpu_matcher_threads=1, reason="af2b8659 MaybeLoadCameras exposes partially populated map to concurrent verifiers",
          source_url="https://github.com/colmap/colmap/blob/af2b8659/src/colmap/feature/matcher.cc#L186-L200",
          feature_and_geometry_parameters_unchanged=True, own_nofile_before=list(old_limits), own_nofile_after=list(resource.getrlimit(resource.RLIMIT_NOFILE)))
    write(status_path, state)

    def execute(args, name):
        start = time.time()
        command = [str(x) for x in args]
        if name in reused:
            old_command = reused[name]["argv"]
            normalized = command[:-2] if command[-2:] == ["--SiftMatching.num_threads", "1"] else command
            assert old_command in (command, normalized), f"Completed stage identity changed: {name}"
            return
        row = dict(name=name, argv=command, started_unix=start)
        state["commands"].append(row)
        write(status_path, state)
        with (logs / f"{name}.log").open("x") as stream:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
        row.update(returncode=result.returncode, elapsed_seconds=time.time() - start)
        write(status_path, state)
        if result.returncode:
            raise RuntimeError(f"COLMAP stage failed {name}: {result.returncode}")

    try:
        for frame in source["frames"]:
            native = a.inputs / f"frame_{frame:04d}"
            folder = a.output / f"frame_{frame:04d}"
            folder.mkdir(exist_ok=a.resume_failed)
            fused_existing = folder / f"fused_{frame}.ply"
            if a.resume_failed and frame and f"frame{frame:04d}_fusion" in reused:
                points_identity = next(x for x in preserved if x["frame"] == frame)
                assert sha(fused_existing) == points_identity["sha256"]
                continue
            database = folder / "database.db"
            # Match each author's feature settings. CEM frame0 uses defaults;
            # LocalDyGS adds affine/domain-size pooling and 16k features.
            feature = [a.colmap, "feature_extractor", "--database_path", database,
                       "--image_path", native / "images", "--ImageReader.camera_model", "PINHOLE",
                       "--ImageReader.single_camera", "0", "--SiftExtraction.use_gpu", "0"]
            if frame != 0:
                feature += ["--SiftExtraction.max_image_size", "4096", "--SiftExtraction.max_num_features", "16384",
                            "--SiftExtraction.estimate_affine_shape", "1", "--SiftExtraction.domain_size_pooling", "1"]
            execute(feature, f"frame{frame:04d}_features")
            camera_rows = {}
            for line in (native / "sparse_custom/cameras.txt").read_text().splitlines():
                cols = line.split()
                if cols and not line.startswith("#"):
                    camera_rows[int(cols[0])] = cols
            image_rows = {}
            for line in (native / "sparse_custom/images.txt").read_text().splitlines():
                cols = line.split()
                if cols and not line.startswith("#"):
                    image_rows[cols[-1]] = cols
            connection = sqlite3.connect(database)
            db_images = connection.execute("SELECT image_id,camera_id,name FROM images ORDER BY image_id").fetchall()
            if set(x[2] for x in db_images) != set(image_rows):
                raise RuntimeError("COLMAP image identity mismatch")
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
            manual = folder / "known_poses"
            manual.mkdir(exist_ok=a.resume_failed)
            (manual / "cameras.txt").write_text("\n".join(remapped_cameras.values()) + "\n")
            (manual / "images.txt").write_text("\n".join(remapped_images) + "\n")
            (manual / "points3D.txt").write_text("")
            execute([a.colmap, "exhaustive_matcher", "--database_path", database, "--SiftMatching.use_gpu", "0", "--SiftMatching.num_threads", "1"], f"frame{frame:04d}_matching")
            sparse = folder / "triangulated"
            sparse.mkdir(exist_ok=a.resume_failed)
            triangulate = [a.colmap, "point_triangulator", "--database_path", database, "--image_path", native / "images",
                           "--input_path", manual, "--output_path", sparse]
            if frame == 0:
                triangulate += ["--Mapper.ba_global_function_tolerance", "0.000001"]
            else:
                triangulate += ["--clear_points", "1"]
            execute(triangulate, f"frame{frame:04d}_triangulation")
            execute([a.colmap, "model_converter", "--input_path", sparse, "--output_path", sparse, "--output_type", "TXT"], f"frame{frame:04d}_model_txt")
            points = [x for x in (sparse / "points3D.txt").read_text().splitlines() if x and not x.startswith("#")]
            if not points:
                raise RuntimeError(f"No native LR triangulated points at frame {frame}")
            dense = folder / "dense"
            dense.mkdir(exist_ok=a.resume_failed)
            execute([a.colmap, "image_undistorter", "--image_path", native / "images", "--input_path", sparse, "--output_path", dense], f"frame{frame:04d}_undistortion")
            if frame != 0:
                execute([a.colmap, "patch_match_stereo", "--workspace_path", dense, "--PatchMatchStereo.gpu_index", "0"], f"frame{frame:04d}_patchmatch")
                fused = folder / f"fused_{frame}.ply"
                execute([a.colmap, "stereo_fusion", "--workspace_path", dense, "--output_path", fused], f"frame{frame:04d}_fusion")
                if fused.stat().st_size < 1024:
                    raise RuntimeError(f"Empty dense fusion at frame {frame}")
        # CEM keeps the author's frame0 sparse initialization, in its expected
        # STG directory structure. It never consumes the LocalDyGS dense union.
        cem = a.output / "cem_data/colmap_0/sparse/0"
        cem.mkdir(parents=True)
        for item in (a.output / "frame_0000/triangulated").iterdir():
            if item.is_file():
                shutil.copy2(item, cem / item.name)
        execute([a.colmap, "model_converter", "--input_path", cem, "--output_path", cem / "points3D.ply", "--output_type", "PLY"], "cem_sparse_ply")
        local = a.output / "localdygs_data"
        local.mkdir()
        shutil.copy2(a.poses_bounds, local / "poses_bounds.npy")
        import open3d as o3d
        merged = o3d.geometry.PointCloud()
        for frame in [x for x in source["frames"] if x != 0]:
            merged += o3d.io.read_point_cloud(str(a.output / f"frame_{frame:04d}/fused_{frame}.ply"))
        before_points = len(merged.points)
        if not before_points:
            raise RuntimeError("No dense LocalDyGS points; no fallback allowed")
        voxel = 0.02
        while len(merged.points) > 90000:
            merged = merged.voxel_down_sample(voxel)
            voxel += 0.01
        destination = local / "downsampled_points_50.ply"
        o3d.io.write_point_cloud(str(destination), merged)
        state.update(status="completed", elapsed_seconds=time.time() - state["began_unix"],
                     localdygs_initialization=dict(path=str(destination), sha256=sha(destination), before_points=before_points, points=len(merged.points)),
                     cem_initialization=dict(path=str(cem / "points3D.ply"), sha256=sha(cem / "points3D.ply")))
        write(a.output / "complete.json", state)
        write(local / "native_initialization_receipt.json", state)
        write(a.output / "cem_data/native_initialization_receipt.json", state)
    except BaseException:
        state.update(status="failed", error=traceback.format_exc())
        raise
    finally:
        write(status_path, state)


if __name__ == "__main__":
    main()
