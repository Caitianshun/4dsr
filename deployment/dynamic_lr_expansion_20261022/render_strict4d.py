#!/usr/bin/env python3
"""Plan Yang/4C4D HR train/test views; CUDA rendering needs explicit --execute.

The default writes only expected_views.json. It imports neither Torch nor NumPy,
opens no RGB pixels/checkpoint, and performs no native rendering or scoring.
GPU execution must be admitted separately after the current HOLD is lifted.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import yaml


TRAIN_FRAMES = list(range(0, 300, 10))


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, value: dict, *, immutable: bool = False) -> None:
    data = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if immutable and path.exists() and path.read_text() != data:
        raise FileExistsError(f"A different immutable plan already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(data)


def make_plan(args: argparse.Namespace) -> tuple[dict, dict, dict]:
    manifest_path = args.manifest.resolve()
    m = json.loads(manifest_path.read_text())
    config = yaml.safe_load(args.config.read_text())
    if (m["splits"]["test"] != ["cam00"] or m["splits"].get("dev")
            or "cam00" in m["splits"]["train"] or m["frame_indices"] != list(range(300))
            or m["scale"] != 4 or m.get("physical_lr_derivation")):
        raise ValueError("Requires registered bicubic x4 full300 with cam00 held out")
    frames = TRAIN_FRAMES if args.scope == "train" else list(range(300))
    cameras = m["splits"]["train"] if args.scope == "train" else ["cam00"]
    observations = [r for r in m["observations"] if r["split"] == args.scope and r["frame_index"] in frames]
    expected = {(camera, frame) for camera in cameras for frame in frames}
    if len(observations) != len(expected) or {(r["camera_id"], r["frame_index"]) for r in observations} != expected:
        raise ValueError("Missing/duplicate/illegal observations in fixed evaluation scope")
    if config["gaussian_dim"] != 4 or config["time_duration"] != [0.0, 10.0]:
        raise ValueError("Unexpected native model/time configuration")
    if config["OptimizationParams"]["iterations"] != 30000:
        raise ValueError("Only the author terminal iteration is supported")
    if config["ModelParams"]["frame_ratio"] != 1:
        raise ValueError("The registered full300 evaluation requires native frame_ratio=1")
    if args.checkpoint.name != "chkpnt30000.pth":
        raise ValueError("Use the fixed terminal chkpnt30000.pth, never a selected best checkpoint")
    width, height = m["resolutions"]["hr"]
    records = []
    for row in sorted(observations, key=lambda r: (r["camera_id"], r["frame_index"])):
        camera = row["camera_id"]
        k = m["cameras"][camera]["K_hr"]
        if len(k) != 3 or any(len(v) != 3 for v in k) or not all(math.isfinite(float(x)) for v in k for x in v):
            raise ValueError("Invalid per-camera HR intrinsic matrix")
        if k[0][0] <= 0 or k[1][1] <= 0:
            raise ValueError("Focal lengths must be positive")
        records.append({"camera_id": camera, "frame": row["frame_index"],
                        "path": f"{camera}/{row['frame_index']:04d}.png",
                        "native_time": row["frame_index"] / 30.0})
    runtime = args.runtime.resolve()
    helper = args.project_root.resolve() / "deployment/dynamic_lr_baselines_20261020/methods/strict4d/prepare_native.py"
    source_files = [Path(__file__).resolve(), helper, runtime / "adapter_identity.json",
                    runtime / "scene/cameras.py", runtime / "scene/gaussian_model.py",
                    runtime / "gaussian_renderer/__init__.py"]
    if json.loads((runtime / "adapter_identity.json").read_text())["method"] != args.method:
        raise ValueError("Runtime method identity differs from requested renderer")
    source_identities = [{"path": str(p), "sha256": sha(p)} for p in source_files]
    pipe = config["PipelineParams"]
    plan = {
        "schema": "strict4d_fixed_evaluation_expected_views_v1",
        "status": "plan_only_not_rendered_not_GPU_validated",
        "method": args.method, "scene": m["scene"], "input_protocol": "bicubic",
        "manifest": str(manifest_path), "manifest_sha256": sha(manifest_path),
        "scope": args.scope, "frames": frames, "cameras": cameras,
        "resolution": [width, height], "observations": len(records),
        "input_representation": "uint8_RGB_PNG", "render_camera_transform": "identity",
        "runtime": str(runtime), "dataset": str(args.dataset.resolve()),
        "configuration": str(args.config.resolve()), "configuration_sha256": sha(args.config),
        "checkpoint": str(args.checkpoint.resolve()), "expected_iteration": 30000,
        "expected_checkpoint_sha256": args.checkpoint_sha256,
        "checkpoint_selection": "fixed terminal iteration, no heldout selection",
        "source_files": source_identities,
        "pipeline_params": pipe, "white_background": config["ModelParams"]["white_background"],
        "env_map_check": {"expected_resolution": int(pipe["env_map_res"]),
                          "restore_from_terminal_checkpoint": True},
        "native_time_mapping": "native_time = frame_index/30 = 10 * manifest_time",
        "principal_point_conversion": "manifest K_hr cx,cy +0.5 to author corner convention",
        "HR_pixels_opened": False, "GPU_calls": 0, "parameter_updates": 0,
        "no_metric_or_parameter_optimization": True,
        "records": records,
    }
    return plan, m, config


def execute(args: argparse.Namespace, plan: dict, m: dict, config: dict) -> None:
    selected_gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not selected_gpu or "," in selected_gpu:
        raise ValueError("External admitted caller must explicitly bind one CUDA_VISIBLE_DEVICES")
    output = args.output.resolve()
    if (output / "render_index.json").exists():
        raise FileExistsError("Already finalized; do not overwrite/replay a completed endpoint")
    for row in plan["records"]:
        if (output / row["path"]).exists():
            raise FileExistsError("Existing render pixels require a new output directory; no silent resume")
    stage = json.loads((args.dataset / "stage_receipt.json").read_text())
    if (stage["manifest_sha256"] != plan["manifest_sha256"] or stage["used_HR_pixels"]
            or stage["used_heldout_pixels"]):
        raise ValueError("Staged native LR dataset does not match the evaluation manifest")
    # Bind the checkpoint bytes before the first CUDA import or RGB prediction.
    checkpoint_sha = sha(args.checkpoint)
    if args.checkpoint_sha256 and checkpoint_sha != args.checkpoint_sha256:
        raise ValueError("Fixed terminal checkpoint hash differs")
    began = time.monotonic()
    # Everything below is deliberately unreachable without explicit --execute.
    import numpy as np
    import torch
    from PIL import Image

    sys.path.insert(0, str(args.runtime.resolve()))
    from gaussian_renderer import render
    from scene.gaussian_model import GaussianModel
    from scene.cameras import Camera

    helper_path = args.project_root.resolve() / "deployment/dynamic_lr_baselines_20261020/methods/strict4d/prepare_native.py"
    spec = importlib.util.spec_from_file_location("strict4d_frozen_normalization", helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    poses, similarity = helper.normalized_poses(m)
    if not np.allclose(similarity, stage["native_world_from_manifest_world"]):
        raise ValueError("Native world normalization differs from initialization")
    pipe = SimpleNamespace(**config["PipelineParams"])
    model = GaussianModel(int(config["ModelParams"]["sh_degree"]),
                          gaussian_dim=int(config["gaussian_dim"]),
                          time_duration=list(config["time_duration"]),
                          rot_4d=bool(config["rot_4d"]), force_sh_3d=bool(config["force_sh_3d"]),
                          sh_degree_t=2 if pipe.eval_shfs_4d else 0)
    checkpoint, iteration = torch.load(args.checkpoint, map_location="cuda", weights_only=False)
    if int(iteration) != 30000:
        raise ValueError(f"Expected fixed terminal iteration 30000; got {iteration}")
    model.restore(checkpoint, None)
    del checkpoint
    # The coefficient network in 4C4D is used for opacity updates during
    # training. Native inference reads already updated opacities; restore(None)
    # correctly skips optimizer/coefficient training state, as old native render.
    env_resolution = int(pipe.env_map_res)
    if env_resolution:
        if model.env_map is None or tuple(model.env_map.shape) != (3, env_resolution, env_resolution):
            raise ValueError("Checkpoint environment map does not match frozen pipeline configuration")
        if not torch.isfinite(model.env_map).all().item():
            raise ValueError("Checkpoint environment map contains nonfinite values")
    elif model.env_map is not None and model.env_map.numel():
        raise ValueError("Checkpoint has an environment map disabled by frozen configuration")
    white = bool(config["ModelParams"]["white_background"])
    background = torch.tensor([1.0 if white else 0.0] * 3, dtype=torch.float32, device="cuda")
    width, height = plan["resolution"]
    transforms = {}
    for camera_id in plan["cameras"]:
        k = np.asarray(m["cameras"][camera_id]["K_hr"], dtype=np.float64)
        pose_cv = poses[camera_id] @ np.diag([1.0, -1.0, -1.0, 1.0])
        w2c = np.linalg.inv(pose_cv)
        transforms[camera_id] = (k, w2c)
    records = []
    with torch.no_grad():
        for uid, row in enumerate(plan["records"]):
            camera_id, frame = row["camera_id"], row["frame"]
            k, w2c = transforms[camera_id]
            camera = Camera(colmap_id=uid, R=w2c[:3, :3].T, T=w2c[:3, 3],
                            FoVx=2 * math.atan(width / (2 * k[0, 0])),
                            FoVy=2 * math.atan(height / (2 * k[1, 1])),
                            image=np.empty(0), gt_alpha_mask=None,
                            image_name=f"{camera_id}_{frame:04d}", uid=uid, data_device="cuda",
                            timestamp=frame / 30.0, cx=k[0, 2] + 0.5, cy=k[1, 2] + 0.5,
                            fl_x=k[0, 0], fl_y=k[1, 1], resolution=(width, height), meta_only=True).cuda()
            prediction = render(camera, model, pipe, background)["render"]
            if tuple(prediction.shape) != (3, height, width) or not torch.isfinite(prediction).all().item():
                raise ValueError(f"Nonfinite or incorrect HR tensor before clamp/quantization: {camera_id}/{frame}")
            rgb = prediction.clamp(0, 1).permute(1, 2, 0).cpu().numpy()
            destination = output / row["path"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.rint(rgb * 255).clip(0, 255).astype(np.uint8)).save(destination)
            records.append({"camera_id": camera_id, "frame": frame, "path": row["path"],
                            "render_sha256": sha(destination)})
            if len(records) % 100 == 0:
                print(json.dumps({"rendered": len(records), "expected": len(plan["records"])}), flush=True)
    result = {
        "schema": "dynamic_lr_multiview_render_v1", "manifest_sha256": plan["manifest_sha256"],
        "scope": args.scope, "resolution": [width, height],
        "input_representation": "uint8_RGB_PNG", "render_camera_transform": "identity",
        "records": records,
    }
    write_json(output / "render_index.json", result, immutable=True)
    write_json(output / "render_receipt.json", {
        "schema": "strict4d_expansion_native_render_receipt_v1", "status": "rendered_not_scored",
        "method": args.method, "scene": m["scene"], "scope": args.scope,
        "plan_sha256": sha(output / "expected_views.json"),
        "render_index_sha256": sha(output / "render_index.json"),
        "manifest_sha256": plan["manifest_sha256"], "checkpoint_sha256": checkpoint_sha,
        "checkpoint_iteration": iteration, "configuration_sha256": plan["configuration_sha256"],
        "pipeline_params": config["PipelineParams"], "native_envmap_restored": bool(env_resolution),
        "finite_guard_before_clamp_and_quantization": {"checked": len(records), "shape": [3, height, width]},
        "HR_pixels_opened": False, "parameter_updates": 0,
        "elapsed_seconds": time.monotonic() - began, "source_files": plan["source_files"],
        "CUDA_VISIBLE_DEVICES": selected_gpu,
        "render_camera_transform": "identity", "scoring": "separate shared scorer reads HR only after fixed endpoint",
    }, immutable=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--method", choices=["yang", "4c4d"], required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", help="Optional pre-registered fixed endpoint hash")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--scope", choices=["train", "test"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plan-only", action="store_true", help="Default: metadata only, no Torch/CUDA imports")
    mode.add_argument("--execute", action="store_true", help="Actual CUDA render; requires external GPU admission")
    args = parser.parse_args()
    plan, m, config = make_plan(args)
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "expected_views.json", plan, immutable=True)
    if args.execute:
        execute(args, plan, m, config)
    else:
        print(json.dumps({"status": plan["status"], "scope": args.scope,
                          "observations": plan["observations"], "GPU_calls": 0}))


if __name__ == "__main__":
    main()
