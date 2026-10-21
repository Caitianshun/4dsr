"""CPU metadata contract for the actual CEM reader/filter/projection functions.

AST loads only pure Python/NumPy classes/functions from the frozen runtime.
COLMAP point-cloud I/O is stubbed because initialization is a separate job.
No model, CUDA library, training image pixels, or held-out pixels are loaded.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_nodes(path, names, namespace):
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    assert {node.name for node in nodes} == set(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), "Refusing to overwrite a CPU contract"
    runtime = args.runtime.resolve()
    spec = importlib.util.spec_from_file_location("protocol_interface", runtime / "protocol_interface.py")
    interface = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(interface)
    sys.modules["protocol_interface"] = interface
    graphics = runtime / "utils/graphics_utils.py"
    readers = runtime / "scene/dataset_readers.py"
    namespace = dict(np=np, os=os, math=math, SceneInfo=SimpleNamespace,
                     torch=SimpleNamespace(zeros=lambda *a, **k: np.zeros(a, **k)),
                     read_extrinsics_binary=lambda _: {}, read_intrinsics_binary=lambda _: {},
                     read_points3D_binary=lambda _: (np.zeros((1, 3)), np.zeros((1, 3)), None),
                     fetchPly=lambda _: None, storePly=lambda *a: None)
    load_nodes(graphics, ["getWorld2View2", "getProjectionMatrixCV", "getProjectionMatrix"], namespace)
    load_nodes(readers, ["CameraInfo2", "getNerfppNorm", "readColmapSceneInfoNeural3DVideo"], namespace)
    source_receipt = json.loads((runtime / "protocol_interface_receipt.json").read_text())
    rows = []
    for scene in ["cook_spinach", "cut_roasted_beef"]:
        path = ROOT / f"data/dynamic_sr/full_time_20261007/prepared/{scene}_area4/manifest_train_ready_seed20261007.json"
        value = json.loads(path.read_text())
        os.environ["DYNAMIC_LR_PROTOCOL_MANIFEST"] = str(path)
        for train_level, render_hr in [("lr", False), ("hr", False), ("lr", True), ("hr", True)]:
            os.environ["CEM_PROTOCOL_TRAIN_LEVEL"] = train_level
            os.environ["DYNAMIC_LR_RENDER_HR"] = "1" if render_hr else "0"
            info = namespace["readColmapSceneInfoNeural3DVideo"]("/CPU-contract-no-initialization", "images", True,
                                                                  SimpleNamespace(start_timestamp=0, end_timestamp=-1))
            assert len(info.train_cameras) == len(value["splits"]["train"]) * 300
            assert len(info.test_cameras) == 300
            assert {cam.timestamp for cam in info.test_cameras} == set(range(300))
            assert all(cam.image_path.startswith("protocol-zero://cam00/") for cam in info.test_cameras)
            all_ids = {i + 1: camera for i, camera in enumerate(sorted(value["cameras"]))}
            max_error = 0.0
            observed = set()
            for split, cameras in [("train", info.train_cameras), ("test", info.test_cameras)]:
                level = train_level if split == "train" else ("hr" if render_hr else "lr")
                width, height = value["resolutions"][level]
                for cam in cameras:
                    camera = all_ids[cam.uid]
                    observed.add((split, camera, cam.timestamp))
                    assert (cam.width, cam.height) == (width, height)
                    expected = value["cameras"][camera]
                    assert np.array_equal(cam.R.T, np.asarray(expected["w2c"])[:3, :3])
                    assert np.array_equal(cam.T, np.asarray(expected["w2c"])[:3, 3])
                    if split == "train":
                        assert Path(cam.image_path).parent.name == camera
                        assert Path(cam.image_path).parents[1].name == level
                        assert Path(cam.image_path).is_file()
                    k = np.asarray(expected[f"K_{level}"])
                    # Execute native projection algebra, including its cx/cy branch.
                    projection = namespace["getProjectionMatrixCV"](cam.near, cam.far, cam.FovX, cam.FovY, cam.cxr, cam.cyr) if cam.cyr != 0 else namespace["getProjectionMatrix"](cam.near, cam.far, cam.FovX, cam.FovY)
                    camera_points = np.array([[0.0, 0.0, 8.0], [0.4, -0.2, 12.0], [-0.5, 0.3, 20.0]])
                    # Map exact world points into each native camera and back.
                    w2c = np.asarray(expected["w2c"])
                    world_h = np.c_[camera_points, np.ones(3)] @ np.linalg.inv(w2c).T
                    native_w2c = np.eye(4)
                    native_w2c[:3, :3], native_w2c[:3, 3] = cam.R.T, cam.T
                    clip = world_h @ (projection @ native_w2c).T
                    native_uv = ((clip[:, :2] / clip[:, 3:4] + 1) * [width, height] - 1) / 2
                    manifest_camera = world_h @ w2c.T
                    direct = manifest_camera[:, :3] @ k.T
                    direct_uv = direct[:, :2] / direct[:, 2:3]
                    max_error = max(max_error, float(np.abs(native_uv - direct_uv).max()))
            expected_rows = {(split, camera, frame) for split in ["train", "test"] for camera in value["splits"][split] for frame in range(300)}
            assert observed == expected_rows and max_error < 1e-9
            rows.append(dict(scene=scene, manifest=str(path), manifest_sha256=sha(path), train_level=train_level,
                             test_level="hr" if render_hr else "lr", train_rows=len(info.train_cameras),
                             heldout_metadata_rows=len(info.test_cameras), max_pixel_error=max_error,
                             actual_native_reader_filter_executed=True, absent_cam04_preserved="cam04" not in value["cameras"],
                             image_file_existence_checked=True, image_pixels_opened=False))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(status="passed_CPU_metadata_contract", CUDA_imported=False,
        native_model_executed=False, native_camera_CUDA_constructor_executed=False,
        scope="actual native reader class/filter and projection algebra; COLMAP point I/O stubbed; no pixels opened",
        runtime=str(runtime), runtime_receipt_sha256=sha(runtime / "protocol_interface_receipt.json"),
        source_interface_sha256=source_receipt["interface_sha256"], verifier_sha256=sha(__file__), cases=rows), indent=2) + "\n")
    print(json.dumps(dict(status="passed", cases=len(rows), receipt=str(args.output), sha256=sha(args.output),
                         max_pixel_error=max(row["max_pixel_error"] for row in rows))))


if __name__ == "__main__":
    main()
