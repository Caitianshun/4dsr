"""Verify transferred native point data without rewriting its source paths.

The original host already checked every COLMAP pose/intrinsic record. This
contract binds that acceptance to the current manifest and copied point SHA,
so the receiver does not need the sender's absolute filesystem paths.
"""
import argparse
import hashlib
import json
from pathlib import Path


def identity(path):
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--initialization", required=True, type=Path)
    a = p.parse_args()
    local, cem = a.initialization / "localdygs_data", a.initialization / "cem_data"
    authority = local / "native_initialization_receipt.json"
    calibration = local / "camera_initialization_acceptance.json"
    source = json.loads(authority.read_text())
    audit = json.loads(calibration.read_text())
    assert authority.read_bytes() == (cem / authority.name).read_bytes()
    assert calibration.read_bytes() == (cem / calibration.name).read_bytes()
    assert source["status"] == "completed" and audit["status"] == "passed"
    assert not source["used_HR"] and not source["used_test_pixels"]
    assert audit["initialization_receipt_sha256"] == identity(authority)["sha256"]
    assert audit["manifest_sha256"] == identity(a.manifest)["sha256"]
    assert source["native_frames"] == [0, *range(10, 300, 10), 299]
    manifest = json.loads(a.manifest.read_text())
    expected_cameras = set(manifest["splits"]["train"])
    rows = audit["camera_records"]
    assert {(x["frame"], x["camera"]) for x in rows} == {(f, c) for f in source["native_frames"] for c in expected_cameras}
    assert audit["COLMAP_poses_intrinsics_unchanged"] and not audit["camera_optimization_used_to_fit_data"]
    points = []
    for key, path in [("localdygs_initialization", local / "downsampled_points_50.ply"),
                      ("cem_initialization", cem / "colmap_0/sparse/0/points3D.ply")]:
        current = identity(path)
        assert current["sha256"] == source[key]["sha256"]
        current["original_source_path"] = source[key]["path"]
        points.append(current)
    receipt = dict(status="passed", original_initialization=identity(authority), original_camera_acceptance=identity(calibration),
                   manifest=identity(a.manifest), transferred_points=points,
                   original_source_paths_rewritten=False, COLMAP_rerun=False, camera_optimization=False,
                   native_frame_count=31, camera_records=len(rows))
    destination = a.initialization / "transferred_initialization_acceptance.json"
    destination.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
