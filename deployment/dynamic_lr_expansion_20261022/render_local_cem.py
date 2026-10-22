#!/usr/bin/env python3
"""Plan or stream author-native LocalDyGS/CEM HR renders for train/test views.

Default is CPU plan only: no Torch import and no GPU access. --execute is an
explicit GPU action and is intended only for a later admitted, released GPU.
It is not called by the CPU preparation workflow. This script loads a fixed
trained endpoint; it never updates parameters or reads HR reference pixels.
"""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

from prepare_local_cem import TRAIN_FRAMES, config_recipe, identity, sha


def validate_output_directory(output, plan):
    """Allow plan -> execution in place, while rejecting any result/foreign file."""
    output = Path(output)
    if not output.exists():
        return
    if not output.is_dir() or {p.name for p in output.iterdir()} != {"expected_views.json"}:
        raise ValueError("Render output must be absent or contain only its matching expected_views.json")
    expected = json.loads((output / "expected_views.json").read_text())
    keys = ("manifest_sha256", "method", "scope", "resolution", "iteration", "records")
    if expected.get("status") != "CPU_PLAN_ONLY_NOT_RENDERED" or any(expected.get(k) != plan.get(k) for k in keys):
        raise ValueError("Existing CPU plan differs from the requested rendering identity")


def expected_views(manifest, scope):
    cameras = sorted(manifest["splits"][scope])
    frames = TRAIN_FRAMES if scope == "train" else list(range(300))
    lookup = {(o["camera_id"], o["frame_index"]): o for o in manifest["observations"] if o["split"] == scope}
    records = []
    for camera in cameras:
        for frame in frames:
            if (camera, frame) not in lookup:
                raise RuntimeError(f"Missing manifest observation {(camera, frame)}")
            records.append(dict(camera_id=camera, frame=frame, path=f"{camera}/{frame:04d}.png"))
    return records


def checkpoint_assets(method, model, endpoint):
    directory = model / f"point_cloud/iteration_{endpoint}"
    required = [directory / "point_cloud.ply"]
    if method == "localdygs":
        required += [directory / name for name in ("mlp_opacity.pth", "mlp_cov.pth", "mlp_color.pth", "mlp_offset.pth", "time_embedding.pth")]
        dynamic = [directory / name for name in ("FDHash.pth", "hexplane.pth") if (directory / name).is_file()]
        if not dynamic:
            required.append(directory / "FDHash.pth")
        else:
            required += dynamic
    else:
        required += [directory / name for name in ("dynamic_point_cloud.ply", "shared_point_cloud.ply", "duration.ply")]
        required += [model / f"chkpnt{endpoint}.pth"]
    return [identity(path) for path in required]


def load_native(args, endpoint):
    # All CUDA-capable imports are deliberately below the --execute branch.
    import torch
    from argparse import ArgumentParser
    from arguments import ModelParams, OptimizationParams, PipelineParams, get_combined_args
    from scene import Scene
    from utils.general_utils import safe_state
    parser = ArgumentParser()
    model = ModelParams(parser, sentinel=True)
    optimizer = OptimizationParams(parser)
    pipeline = PipelineParams(parser)
    sys.argv = ["render_local_cem", "--model_path", str(args.model), "--source_path", str(args.source)]
    if args.method == "localdygs":
        import mmcv
        from arguments import ModelHiddenParams
        from utils.general_utils import merge_hparams
        from gaussian_renderer import GaussianModel
        hidden = ModelHiddenParams(parser)
        arguments = merge_hparams(get_combined_args(parser), mmcv.Config.fromfile(str(args.config)))
    else:
        arguments = get_combined_args(parser)
    dataset, opt, pipe = model.extract(arguments), optimizer.extract(arguments), pipeline.extract(arguments)
    safe_state(False)
    if args.method == "localdygs":
        gaussians = GaussianModel(hidden.extract(arguments), opt, dataset.feat_dim, dataset.n_offsets,
            dataset.voxel_size, dataset.update_depth, dataset.update_init_factor, dataset.update_hierachy_factor,
            dataset.use_feat_bank, dataset.appearance_dim, dataset.ratio, dataset.add_opacity_dist,
            dataset.add_cov_dist, dataset.add_color_dist)
        scene = Scene(dataset, gaussians, frames_start_end=[0, 300], load_iteration=endpoint, shuffle=False)
    else:
        from scene import init_model
        gaussians = init_model(dataset.model, dataset)
        scene = Scene(dataset, gaussians, load_iteration=endpoint, shuffle=False, opt=opt)
    background = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0], dtype=torch.float32, device="cuda")
    return dataset, opt, pipe, gaussians, scene, background


def native_camera(args, manifest, row, index):
    import numpy as np
    import torch
    spec = manifest["cameras"][row["camera_id"]]
    k, w2c = np.asarray(spec["K_hr"], dtype=np.float64), np.asarray(spec["w2c"], dtype=np.float64)
    width, height = manifest["resolutions"]["hr"]
    fx, fy = 2 * math.atan(width / (2 * k[0, 0])), 2 * math.atan(height / (2 * k[1, 1]))
    if args.method == "localdygs":
        from scene.cameras import Camera
        from protocol_interface import apply_localdygs_projection
        view = Camera(colmap_id=index, R=w2c[:3, :3].T.copy(), T=w2c[:3, 3].copy(), FoVx=fx, FoVy=fy,
                      image=torch.zeros((3, height, width)), gt_alpha_mask=None,
                      image_name=row["path"], uid=index, time=row["frame"] / 300.0)
        view = apply_localdygs_projection(view, k)
    else:
        from scene.cameras import Cameravideo
        view = Cameravideo(colmap_id=sorted(manifest["cameras"]).index(row["camera_id"]) + 1,
                          R=w2c[:3, :3].T.copy(), T=w2c[:3, 3].copy(), FoVx=fx, FoVy=fy,
                          image=None, gt_alpha_mask=None, image_name=row["path"], image_path="protocol-zero://evaluation-metadata",
                          uid=index, near=0.01, far=300.0, timestamp=row["frame"],
                          cxr=(float(k[0, 2]) + 0.5) / width - 0.5,
                          cyr=(float(k[1, 2]) + 0.5) / height - 0.5, resolution=(width, height))
    # Validate the native camera's pixel projection, including the half-pixel
    # convention. A mismatch fails; it is never repaired with an image warp.
    projection = view.projection_matrix.detach().cpu().numpy().T
    wanted = np.asarray([2 * k[0, 0] / width, 2 * k[1, 1] / height,
                         2 * (k[0, 2] + 0.5) / width - 1, 2 * (k[1, 2] + 0.5) / height - 1])
    actual = np.asarray([projection[0, 0], projection[1, 1], projection[0, 2], projection[1, 2]])
    if not np.allclose(actual, wanted, rtol=0, atol=2e-6):
        raise RuntimeError(f"Native HR projection does not match manifest K: {row}, {actual}, {wanted}")
    return view


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["localdygs", "cem"], required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--iteration", type=int)
    parser.add_argument("--scope", choices=["train", "test"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-index", type=Path, help="Optional preserved result index for exact asset SHA verification")
    parser.add_argument("--runtime-acceptance", type=Path, help="For MeetRoom: receipt with status=passed, matching method/scene, camera_and_initialization_contract_passed=true")
    branch = parser.add_mutually_exclusive_group()
    branch.add_argument("--plan-only", action="store_true", help="Default; emit metadata without importing Torch")
    branch.add_argument("--execute", action="store_true", help="Explicit later GPU action, only after resource admission")
    args = parser.parse_args()
    for key in ("manifest", "runtime", "model", "source", "output", "config", "checkpoint_index", "runtime_acceptance"):
        if getattr(args, key) is not None:
            setattr(args, key, getattr(args, key).resolve())
    value = json.loads(args.manifest.read_text())
    assert "physical_lr_derivation" not in value, "This expansion uses original bicubic LR"
    assert value["frame_indices"] == list(range(300)) and value["splits"]["test"] == ["cam00"]
    assert "cam00" not in value["splits"]["train"]
    recipe = config_recipe(args.method, value["scene"])
    endpoint = args.iteration if args.iteration is not None else recipe["endpoint"]
    if endpoint != recipe["endpoint"]:
        raise SystemExit("Only the frozen author final endpoint is accepted; no test-score checkpoint selection")
    if args.config is None and args.method == "localdygs":
        args.config = args.runtime / recipe["relative"]
    records = expected_views(value, args.scope)
    assets = checkpoint_assets(args.method, args.model, endpoint)
    plan = dict(schema="dynamic_lr_expected_views_v1", status="CPU_PLAN_ONLY_NOT_RENDERED",
                manifest_sha256=sha(args.manifest), method=args.method, scene=value["scene"], scope=args.scope,
                resolution=value["resolutions"]["hr"], iteration=endpoint, input_representation="uint8_RGB_PNG",
                render_camera_transform="identity", records=records, checkpoint_assets=assets,
                model=str(args.model), runtime=str(args.runtime), source=str(args.source),
                config=identity(args.config) if args.config else None,
                missing_assets=[x["path"] for x in assets if not x["exists"]],
                native_CUDA_model_load_and_render_verified=False, HR_reference_pixels_read=False,
                trainable_parameters_updated=False)
    if not args.execute:
        validate_output_directory(args.output, plan)
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "expected_views.json").write_text(json.dumps(plan, indent=2) + "\n")
        print(json.dumps(dict(status=plan["status"], views=len(records), missing_assets=plan["missing_assets"], output=str(args.output))))
        return
    if plan["missing_assets"]:
        raise SystemExit("Missing final model assets: " + ", ".join(plan["missing_assets"]))
    if not (args.model / "cfg_args").is_file() or not args.source.is_dir() or not args.runtime.is_dir():
        raise SystemExit("Missing native model cfg_args/source/runtime; no CUDA import attempted")
    if args.method == "localdygs" and not args.config.is_file():
        raise SystemExit("Missing exact LocalDyGS author config")
    if args.checkpoint_index:
        old = json.loads(args.checkpoint_index.read_text())
        expected = {x["path"].split("/model/", 1)[1]: x["sha256"] for x in old["formal_checkpoint"]}
        for asset in assets:
            suffix = str(Path(asset["path"]).resolve().relative_to(args.model)) if Path(asset["path"]).is_absolute() else None
            if suffix is None:
                from prepare_local_cem import ROOT
                suffix = str((ROOT / asset["path"]).resolve().relative_to(args.model))
            if expected.get(suffix) != asset["sha256"]:
                raise SystemExit("Checkpoint identity differs from preserved endpoint: " + suffix)
    if value["scene"].startswith("meetroom_"):
        if args.runtime_acceptance is None:
            raise SystemExit("MeetRoom requires camera/initialization runtime acceptance; CPU recipe is not readiness")
        accepted = json.loads(args.runtime_acceptance.read_text())
        if not (accepted.get("status") == "passed" and accepted.get("method") == args.method and
                accepted.get("scene") == value["scene"] and accepted.get("camera_and_initialization_contract_passed") is True and
                accepted.get("manifest_sha256") == plan["manifest_sha256"]):
            raise SystemExit("MeetRoom runtime acceptance does not match this exact method/scene/manifest")
    selected = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not selected or "," in selected:
        raise SystemExit("Explicit single-GPU CUDA_VISIBLE_DEVICES from resource admission is required")
    validate_output_directory(args.output, plan)
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ["DYNAMIC_LR_PROTOCOL_MANIFEST"] = str(args.manifest)
    os.environ.pop("DYNAMIC_LR_RENDER_HR", None)
    os.chdir(args.runtime)
    sys.path.insert(0, str(args.runtime))
    import numpy as np
    import torch
    from PIL import Image
    from gaussian_renderer import render
    began = time.monotonic()
    with torch.no_grad():
        dataset, opt, pipe, gaussians, scene, background = load_native(args, endpoint)
        width, height = value["resolutions"]["hr"]
        for index, row in enumerate(records):
            view = native_camera(args, value, row, index)
            if args.method == "localdygs":
                from gaussian_renderer import prefilter_voxel
                visible = prefilter_voxel(view, gaussians, pipe, background)
                prediction = render(view, gaussians, pipe, background, stage="fine", visible_mask=visible, retain_grad=False)["render"]
            else:
                prediction = render(view, gaussians, pipe, background, mode=0, near=dataset.near, far=dataset.far)["render"]
            if tuple(prediction.shape) != (3, height, width) or not bool(torch.isfinite(prediction).all().item()):
                raise RuntimeError(f"Invalid native HR prediction for {row}")
            destination = args.output / row["path"]
            destination.parent.mkdir(exist_ok=True)
            rgb = prediction.detach().float().cpu().permute(1, 2, 0).numpy()
            Image.fromarray(np.rint(np.clip(rgb, 0, 1) * 255).astype(np.uint8), "RGB").save(destination)
            row["render_sha256"] = sha(destination)
    result = dict(schema="dynamic_lr_multiview_render_v1", manifest_sha256=plan["manifest_sha256"], scope=args.scope,
                  resolution=value["resolutions"]["hr"], input_representation="uint8_RGB_PNG", render_camera_transform="identity", records=records)
    (args.output / "render_index.json").write_text(json.dumps(result, indent=2) + "\n")
    plan.update(status="completed", records=records, elapsed_seconds=time.monotonic() - began,
                native_CUDA_model_load_and_render_verified=True,
                finite_before_quantization_verified_count=len(records),
                quantization="np.rint(np.clip(CHW_to_HWC_float32,0,1)*255).astype(np.uint8); PIL RGB PNG",
                renderer_source=identity(Path(__file__)), GPU_visible_mask=selected,
                native_prefilter="LocalDyGS author prefilter_voxel retained; no proposed culling module")
    (args.output / "render_receipt.json").write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps(dict(status="completed", views=len(records), output=str(args.output))))


if __name__ == "__main__":
    main()
