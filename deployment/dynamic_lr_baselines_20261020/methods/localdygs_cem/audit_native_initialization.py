"""Verify every native COLMAP time retains the fixed manifest calibration."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rotation(q):
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                     [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])


def rows(path):
    return [line.split() for line in path.read_text().splitlines() if line and not line.startswith("#")]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--initialization", type=Path, required=True)
    p.add_argument("--colmap", type=Path, default=Path("/usr/local/bin/colmap"))
    a = p.parse_args()
    value = json.loads(a.manifest.read_text())
    source = json.loads((a.inputs / "input_receipt.json").read_text())
    result = json.loads((a.initialization / "complete.json").read_text())
    assert source["manifest_sha256"] == digest(a.manifest)
    assert result["status"] == "completed"
    assert result["native_frames"] == source["frames"] == [0, *range(10, 300, 10), 299]
    errors = []
    for frame in source["frames"]:
        folder = a.initialization / f"frame_{frame:04d}/triangulated"
        cameras = {int(x[0]): x for x in rows(folder / "cameras.txt")}
        images = [x for x in rows(folder / "images.txt") if len(x) == 10 and x[-1].endswith(".png")]
        assert sorted(Path(x[-1]).stem for x in images) == sorted(value["splits"]["train"])
        for image in images:
            camera = Path(image[-1]).stem
            spec = value["cameras"][camera]
            k = np.asarray(spec["K_lr"])
            expected = np.asarray(spec["w2c"])
            actual_rotation = rotation(np.array(image[1:5], dtype=float))
            translation = np.array(image[5:8], dtype=float)
            intrinsics = cameras[int(image[8])]
            assert intrinsics[1] == "PINHOLE" and list(map(int, intrinsics[2:4])) == value["resolutions"]["lr"]
            params = np.array(intrinsics[4:], dtype=float)
            k_error = float(np.abs(params - [k[0, 0], k[1, 1], k[0, 2]+.5, k[1, 2]+.5]).max())
            pose_error = max(float(np.abs(actual_rotation-expected[:3, :3]).max()), float(np.abs(translation-expected[:3, 3]).max()))
            assert k_error < 1e-9 and pose_error < 1e-9, (frame, camera, k_error, pose_error)
            errors.append(dict(frame=frame, camera=camera, max_K_parameter_error=k_error, max_pose_matrix_error=pose_error))
    for name in ["localdygs_initialization", "cem_initialization"]:
        info = result[name]
        assert digest(Path(info["path"])) == info["sha256"]
    out = a.initialization / "camera_initialization_acceptance.json"
    receipt = dict(status="passed", manifest_sha256=digest(a.manifest), frames=len(source["frames"]),
                   COLMAP_binary=dict(path=str(a.colmap), resolved_path=str(a.colmap.resolve()), sha256=digest(a.colmap),
                                      help_header=subprocess.run([str(a.colmap), "-h"], capture_output=True, text=True, check=True).stdout.splitlines()[:4]),
                   native_camera_records=len(errors), max_K_parameter_error=max(x["max_K_parameter_error"] for x in errors),
                   max_pose_matrix_error=max(x["max_pose_matrix_error"] for x in errors),
                   COLMAP_poses_intrinsics_unchanged=True, camera_optimization_used_to_fit_data=False,
                   initialization_receipt_sha256=digest(a.initialization / "complete.json"), camera_records=errors)
    payload = json.dumps(receipt, indent=2) + "\n"
    for destination in [out, a.initialization / "localdygs_data/camera_initialization_acceptance.json", a.initialization / "cem_data/camera_initialization_acceptance.json"]:
        destination.write_text(payload)
    print(json.dumps({k:v for k,v in receipt.items() if k != "camera_records"}, indent=2))


if __name__ == "__main__":
    main()
