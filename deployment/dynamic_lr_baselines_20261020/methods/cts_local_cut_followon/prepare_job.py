"""Freeze the authorized CTS LocalDyGS Cut job after native A100 initialization."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
SHARED = HERE.parent / "localdygs_cem"
INITIALIZATION = Path("output/dynamic_lr_baselines_20261020/a100_native_localdygs_cem/cut_roasted_beef/native_init")


def identity(path):
    path = Path(path)
    return dict(path=str(path.relative_to(ROOT)), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Refusing an existing frozen job identity; no replay")
    initial = ROOT / INITIALIZATION
    receipt = json.loads((initial / "localdygs_data/native_initialization_receipt.json").read_text())
    calibration = json.loads((initial / "localdygs_data/camera_initialization_acceptance.json").read_text())
    assert receipt["status"] == "completed" and calibration["status"] == "passed"
    assert receipt["native_frames"] == [0, *range(10, 300, 10), 299]
    assert not receipt["used_HR"] and not receipt["used_test_pixels"]
    assert identity(initial / "localdygs_data/downsampled_points_50.ply")["sha256"] == receipt["localdygs_initialization"]["sha256"]
    assert identity(initial / "cem_data/colmap_0/sparse/0/points3D.ply")["sha256"] == receipt["cem_initialization"]["sha256"]
    assert len(calibration["camera_records"]) == 31 * 19
    assert {row["camera"] for row in calibration["camera_records"]} == {f"cam{camera:02d}" for camera in range(1, 21) if camera != 4}
    subprocess.run([sys.executable, str(SHARED / "build_job.py"), "--method", "localdygs", "--scene", "cut_roasted_beef",
        "--skip-initialization", "--native-initialization-root", str(INITIALIZATION),
        "--input-receipt", "deployment/dynamic_lr_baselines_20261020/input_verified.json",
        "--output", str(args.output)], cwd=ROOT, check=True)
    job = json.loads(args.output.read_text())
    author = next(row["argv"] for row in job["commands"] if row["name"].endswith("_author_train"))
    assert author[author.index("--frames_start_end") + 1:author.index("--frames_start_end") + 3] == ["0", "300"]
    assert author[author.index("--save_iterations") + 1] == "30000"
    assert author[author.index("--configs") + 1] == "arguments/dynerf/cut_roasted_beef.py"
    assert all(job["env"][name] == "/tmp" for name in ["TMPDIR", "TMP", "TEMP"])
    # Reuse the same isolated leaf and native operators already accepted by
    # Local Cook, while binding this new job to their current exact identities.
    source = ROOT / "output/dynamic_lr_baselines_20261020/localdygs_cem/localdygs_environment/build_identity.json"
    build = json.loads(source.read_text())
    assert build["status"] == "CPU_build_and_import_passed" and not build["author_kernel_bodies_modified"]
    remote_root = Path("/home/cts/Project/4DSR")
    extra = [dict(path="envs/dynamic_lr_baselines_20261020/localdygs/build_identity.json", sha256=hashlib.sha256(source.read_bytes()).hexdigest())]
    extra.append(dict(path="envs/dynamic_lr_baselines_20261020/localdygs/upstream_commit.txt",
                      sha256=hashlib.sha256(source.with_name("upstream_commit.txt").read_bytes()).hexdigest()))
    for row in [build[key] for key in ["compiler", "installed_packages", "header_compatibility", "header_patch"]] + build["build_helpers"] + build["extension_binaries"]:
        extra.append(dict(path=str(Path(row["path"]).relative_to(remote_root)), sha256=row["sha256"]))
    extra.extend([identity(__file__), identity(HERE / "after_cem.py"), identity(initial / "transferred_initialization_acceptance.json")])
    combined = {row["path"]: row for row in job["required_files"]}
    for row in extra:
        if row["path"] in combined:
            assert row == combined[row["path"]]
        combined[row["path"]] = row
    job["required_files"] = sorted(combined.values(), key=lambda row: row["path"])
    job["isolated_CTS_LocalDyGS_build_identity_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    job["native_initialization_origin"] = dict(host="a100-train", COLMAP="3.9.1 Commit Unknown CUDA build",
        source_paths_preserved=True, COLMAP_rerun=False, frame_count=31, camera_records=589,
        original_receipt_sha256=identity(initial / "localdygs_data/native_initialization_receipt.json")["sha256"])
    job["finite_queue_launcher"] = identity(HERE / "after_cem.py")
    args.output.write_text(json.dumps(job, indent=2) + "\n")
    print(json.dumps(dict(job_id=job["job_id"], commands=len(job["commands"]), required_files=len(combined),
        job_sha256=hashlib.sha256(args.output.read_bytes()).hexdigest(), submitted=False), indent=2))


if __name__ == "__main__":
    main()
