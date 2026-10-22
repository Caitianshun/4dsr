#!/usr/bin/env python3
"""Audit strict4d manifest/camera contracts on NumPy CPU; stage no data by default.

Author projection arithmetic is AST-extracted without importing native modules.
Only the torch.zeros allocation is mapped to NumPy float32. This is not actual
native Camera/DataLoader or CUDA acceptance. Optional --stage-data verifies LR
bytes/shapes and writes metadata/hardlinks into a fresh isolated directory;
copying image bytes as a fallback is explicitly disabled.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RUNTIMES = {
    "yang": "output/dynamic_lr_baselines_20261020/strict4d/runtime/yang_pillowfix",
    "4c4d": "output/dynamic_lr_baselines_20261020/strict4d/runtime/4c4d",
}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def identity(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": sha(path)}


def numpy_native_functions(source: Path) -> tuple[dict, dict]:
    names = {"getWorld2View2", "getProjectionMatrixCenterShift"}
    functions = [node for node in ast.parse(source.read_text()).body
                 if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in functions} != names:
        raise ValueError("Pinned author projection helper functions changed")
    original_ast = ast.dump(ast.Module(body=functions, type_ignores=[]), include_attributes=False)

    class AllocationBackend(ast.NodeTransformer):
        changes = 0

        def visit_Call(self, node):
            self.generic_visit(node)
            if (isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "torch"):
                if (node.func.attr != "zeros" or node.keywords or len(node.args) != 2
                        or [ast.literal_eval(value) for value in node.args] != [4, 4]):
                    raise ValueError("Unexpected Torch operation in extracted native arithmetic")
                self.changes += 1
                return ast.copy_location(ast.Call(
                    func=ast.Attribute(value=ast.Name(id="np", ctx=ast.Load()), attr="zeros", ctx=ast.Load()),
                    args=[ast.Tuple(elts=node.args, ctx=ast.Load())],
                    keywords=[ast.keyword(arg="dtype", value=ast.Attribute(
                        value=ast.Name(id="np", ctx=ast.Load()), attr="float32", ctx=ast.Load()))]), node)
            return node

    transform = AllocationBackend()
    module = transform.visit(ast.Module(body=functions, type_ignores=[]))
    ast.fix_missing_locations(module)
    if transform.changes != 1 or any(isinstance(node, ast.Name) and node.id == "torch" for node in ast.walk(module)):
        raise ValueError("Native projection arithmetic is not the expected CPU-allocation-only AST")
    scope = {"np": np, "math": math, "__builtins__": {}}
    exec(compile(module, str(source), "exec"), scope)
    return scope, {
        "source": identity(source), "functions": sorted(names),
        "native_function_AST_sha256": hashlib.sha256(original_ast.encode()).hexdigest(),
        "allocation_backend_changes": transform.changes,
        "only_transformation": "torch.zeros(4,4) -> np.zeros((4,4),dtype=np.float32); all author arithmetic unchanged",
        "actual_native_Camera_or_Torch_executed": False,
    }


def native_stage_helper(root: Path):
    path = root / "deployment/dynamic_lr_baselines_20261020/methods/strict4d/prepare_native.py"
    spec = importlib.util.spec_from_file_location("strict4d_CPU_stage_helper", path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    return helper, path


def audit(args: argparse.Namespace) -> dict:
    root, manifest_path = args.project_root.resolve(), args.manifest.resolve()
    m = json.loads(manifest_path.read_text())
    runtime = args.runtime.resolve() if args.runtime else root / RUNTIMES[args.method]
    if json.loads((runtime / "adapter_identity.json").read_text())["method"] != args.method:
        raise ValueError("Pinned runtime method identity differs")
    if (m.get("physical_lr_derivation") or m["scale"] != 4 or m["frame_indices"] != list(range(300))
            or m["splits"]["test"] != ["cam00"] or m["splits"].get("dev")
            or "cam00" in m["splits"]["train"]):
        raise ValueError("Requires bicubic x4 full300 with cam00-only holdout")
    train_cameras = m["splits"]["train"]
    if not train_cameras or len(train_cameras) != len(set(train_cameras)):
        raise ValueError("Invalid/duplicate training cameras")
    expected = {(camera, frame) for camera in [*train_cameras, "cam00"] for frame in range(300)}
    rows = m["observations"]
    if len(rows) != len(expected) or {(row["camera_id"], row["frame_index"]) for row in rows} != expected:
        raise ValueError("Missing or duplicate registered camera/frame rows")
    max_time_error = 0.0
    for row in rows:
        split = "test" if row["camera_id"] == "cam00" else "train"
        if row["split"] != split:
            raise ValueError("Camera/split mismatch")
        registered = float(row["time"])
        if not math.isfinite(registered) or abs(registered - row["frame_index"] / 300.0) > 1e-12:
            raise ValueError("Manifest time differs from frame/300")
        error = abs(10.0 * registered - row["frame_index"] / 30.0)
        max_time_error = max(max_time_error, error)
        if error > 1e-12:
            raise ValueError("Native time mapping would differ between training and evaluation")
        for level in ("lr", "hr"):
            if not isinstance(row[level + "_sha256"], str) or len(row[level + "_sha256"]) != 64:
                raise ValueError("Incomplete registered image identity metadata")
    widths = m["resolutions"]
    if (len(widths["lr"]) != 2 or len(widths["hr"]) != 2
            or any(not isinstance(v, int) or v <= 0 for level in widths.values() for v in level)
            or widths["hr"] != [4 * v for v in widths["lr"]]):
        raise ValueError("Expected exact registered LR/HR dimensions at x4")
    helper, helper_path = native_stage_helper(root)
    poses, similarity = helper.normalized_poses(m)
    if not np.isfinite(similarity).all():
        raise ValueError("Nonfinite native world normalization")
    functions, ast_receipt = numpy_native_functions(runtime / "utils/graphics_utils.py")
    point_camera = np.array([[-0.25, -0.1, 2.0], [0.0, 0.0, 5.0],
                             [0.7, 0.2, 8.0], [-0.4, 0.3, 10.0]])
    checks = []
    for camera_id in [*train_cameras, "cam00"]:
        calibration = m["cameras"][camera_id]
        w2c = np.asarray(calibration["w2c"], dtype=np.float64)
        c2w = np.asarray(calibration["c2w"], dtype=np.float64)
        if (w2c.shape != (4, 4) or c2w.shape != (4, 4) or not np.isfinite(w2c).all()
                or not np.isfinite(c2w).all() or not np.allclose(w2c @ c2w, np.eye(4), rtol=0, atol=1e-6)):
            raise ValueError("Invalid registered camera extrinsics")
        native_c2w_cv = poses[camera_id] @ np.diag([1.0, -1.0, -1.0, 1.0])
        native_w2c = np.linalg.inv(native_c2w_cv)
        world_view = functions["getWorld2View2"](native_w2c[:3, :3].T, native_w2c[:3, 3]).T
        world = np.c_[point_camera, np.ones(len(point_camera))] @ np.linalg.inv(w2c).T
        normalized_world = world @ similarity.T
        levels = ["lr", "hr"] if camera_id != "cam00" else ["hr"]
        for level in levels:
            width, height = widths[level]
            k = np.asarray(calibration["K_" + level], dtype=np.float64)
            if (k.shape != (3, 3) or not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0
                    or k[0, 2] + 0.5 <= 0 or k[1, 2] + 0.5 <= 0):
                raise ValueError("Invalid registered intrinsic matrix / native principal-point branch")
            projection = functions["getProjectionMatrixCenterShift"](
                0.01, 100.0, k[0, 2] + 0.5, k[1, 2] + 0.5,
                k[0, 0], k[1, 1], width, height).T
            # Native Camera stores transposed matrices and multiplies them in
            # float32. Evaluate points in float64 after reproducing this product.
            full = world_view @ projection
            clip = normalized_world @ full.astype(np.float64)
            actual = ((clip[:, :2] / clip[:, 3:] + 1) * [width, height] - 1) / 2
            expected_pixel = point_camera @ k.T
            expected_pixel = expected_pixel[:, :2] / expected_pixel[:, 2:]
            error = float(np.max(np.abs(actual - expected_pixel)))
            if not np.isfinite(actual).all() or error >= 0.002:
                raise ValueError(f"Native arithmetic projection differs: {camera_id}/{level}: {error}")
            checks.append({"camera_id": camera_id, "scope": "test" if camera_id == "cam00" else "train",
                           "grid": level, "resolution": [width, height],
                           "expected_image_shape_CHW": [3, height, width],
                           "native_projection_max_error_pixels": error,
                           "applicable_frames": "all 0..299; registered cameras are static",
                           "metadata_observations_checked": 300,
                           "native_time_range": [0.0, 299 / 30.0],
                           "FoVx": 2 * math.atan(width / (2 * k[0, 0])),
                           "FoVy": 2 * math.atan(height / (2 * k[1, 1]))})
    dataset = args.dataset.resolve() if args.dataset else (
        root / "output/dynamic_lr_expansion_20261022/strict4d/inputs" / args.method / m["scene"])
    stage_status = {"requested": bool(args.stage_data), "dataset": str(dataset),
                    "RGB_files_opened_this_run": False, "pixel_byte_copy_fallback_allowed": False,
                    "native_DataLoader_executed": False}
    if args.stage_data:
        def no_copy(*_args, **_kwargs):
            raise OSError("Hardlink unavailable; this CPU staging contract forbids copying image bytes")
        helper.shutil = SimpleNamespace(copy2=no_copy)
        helper.stage(SimpleNamespace(manifest=manifest_path, output=dataset))
        stage_status.update(RGB_files_opened_this_run=True,
                            training_LR_bytes_and_shapes_verified=True,
                            training_image_hardlinks_created=True,
                            camera_metadata_written=True,
                            author_initialization_executed=False)
    receipt_path = dataset / "stage_receipt.json"
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text())
        if (receipt["manifest_sha256"] != sha(manifest_path) or receipt["used_HR_pixels"]
                or receipt["used_heldout_pixels"] or not np.allclose(
                    similarity, receipt["native_world_from_manifest_world"])):
            raise ValueError("Existing native stage receipt disagrees with manifest or camera normalization")
        stage_status.update(stage_receipt=identity(receipt_path), metadata_binding="passed")
    else:
        stage_status["metadata_binding"] = "pending; no stage receipt exists"
    if "torch" in sys.modules:
        raise RuntimeError("NumPy-only contract unexpectedly imported Torch")
    return {
        "schema": "strict4d_numpy_camera_metadata_contract_v1",
        "status": "passed_numpy_projection_and_metadata_native_loader_pending",
        "method": args.method, "scene": m["scene"], "input_protocol": "bicubic",
        "manifest": identity(manifest_path), "runtime_identity": identity(runtime / "adapter_identity.json"),
        "source": identity(Path(__file__)), "normalization_source": identity(helper_path),
        "native_projection_AST": ast_receipt,
        "camera_calibrations_checked": len(checks), "checks": checks,
        "max_projection_error_pixels": max(row["native_projection_max_error_pixels"] for row in checks),
        "projection_error_limit_pixels": 0.002,
        "metadata_frame_time_contract": {"unique_registered_camera_frame_rows": len(rows),
            "all_train_LR_rows": 300 * len(train_cameras), "all_train_HR_rows": 300 * len(train_cameras),
            "all_test_HR_rows": 300, "all_frames": list(range(300)),
            "train_metric_frames": list(range(0, 300, 10)), "test_metric_frames": list(range(300)),
            "registered_time": "frame_index/300", "native_time": "frame_index/30",
            "max_native_time_mapping_error": max_time_error},
        "expected_native_loader_shapes": {"train_LR": [3, *widths["lr"][::-1]],
            "train_HR_evaluation": [3, *widths["hr"][::-1]], "test_HR_evaluation": [3, *widths["hr"][::-1]],
            "source": "manifest.resolutions; no hardcoded N3DV width/height"},
        "stage": stage_status,
        "HR_pixels_opened": False, "heldout_pixels_opened": False,
        "Torch_imported": False, "GPU_calls": 0, "remote_calls": 0, "parameter_updates": 0,
        "pending_acceptance": ["actual native Camera and full native DataLoader in target environment",
            "actual pixel-pool loader shapes/tensors, 12 workers and native batch4",
            "author initialization, CPU deployment gate and per-host runtime/source identities",
            "fresh resource-admitted CUDA train/render acceptance"],
        "not_a_complete_native_deployment_contract": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--method", choices=["yang", "4c4d"], required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--runtime", type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stage-data", action="store_true", help="Explicit CPU LR byte/shape checks and hardlink staging only; no copies, initialization, or GPU")
    args = parser.parse_args()
    result = audit(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "method": args.method, "scene": result["scene"],
                      "camera_calibrations_checked": result["camera_calibrations_checked"],
                      "max_projection_error_pixels": result["max_projection_error_pixels"], "GPU_calls": 0}))


if __name__ == "__main__":
    main()
