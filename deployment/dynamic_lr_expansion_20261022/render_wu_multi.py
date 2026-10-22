"""Plan fixed-view native Wu/Multi4D HR renderings; CPU plan is the default.

--execute is required before importing torch or a model. GPU binding/admission
belongs to the external caller. Only camera calibration/time enter rendering;
no HR reference, LR image, teacher image, warp, affine fit or culling is used.
The shared scorer separately reads HR after rendering is complete.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / "deployment/dynamic_lr_baselines_20261020/methods/wu_multi"


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def local(path):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def identity(path):
    path = local(path)
    return dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size)


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def frames_for(scope, value):
    if value is None:
        return list(range(0, 300, 10)) if scope == "train" else list(range(300))
    frames = list(range(300)) if value == "all" else [int(item) for item in value.split(",")]
    if len(frames) != len(set(frames)) or not frames or any(not 0 <= frame < 300 for frame in frames):
        raise ValueError("Unique frames in [0,299] required")
    return sorted(frames)


def plan(args):
    spec_path = args.spec.resolve()
    spec = json.loads(spec_path.read_text())
    manifest_path = local(spec["manifest"])
    if sha(manifest_path) != spec["manifest_identity"]["sha256"]:
        raise ValueError("Manifest changed after preparation")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("physical_lr_derivation") or manifest["scene"] != spec["scene"]:
        raise ValueError("The original bicubic scene protocol is required")
    cameras = manifest["splits"][args.scope]
    frames = frames_for(args.scope, args.frames)
    rows = sorted((row for row in manifest["observations"]
                   if row["split"] == args.scope and row["frame_index"] in frames),
                  key=lambda row: (row["camera_id"], row["frame_index"]))
    if [(row["camera_id"], row["frame_index"]) for row in rows] != [(camera, frame) for camera in sorted(cameras) for frame in frames]:
        raise ValueError("Missing or duplicate selected camera/frame")
    checkpoint_entry = spec.get("checkpoint")
    checkpoint = local(args.checkpoint) if args.checkpoint else local(checkpoint_entry["path"]) if checkpoint_entry else None
    checkpoint_exists = bool(checkpoint and checkpoint.is_file())
    if checkpoint_exists and checkpoint_entry and sha(checkpoint) != checkpoint_entry["sha256"]:
        raise ValueError("Checkpoint differs from the reused completed endpoint")
    # Preserve only numerical camera inputs at the renderer boundary.
    geometry = {key: manifest[key] for key in ("resolutions", "cameras")}
    render_rows = [dict(camera_id=row["camera_id"], frame_index=row["frame_index"], time=row["time"]) for row in rows]
    record = dict(schema="dynamic_lr_multiview_expected_v1", status="CPU_plan_only_not_rendered",
                  method=spec["method"], scene=spec["scene"], spec=identity(spec_path),
                  manifest=identity(manifest_path), manifest_sha256=sha(manifest_path),
                  scope=args.scope, cameras=cameras, frames=frames, observations=len(rows),
                  resolution=manifest["resolutions"]["hr"], input_representation="uint8_RGB_PNG",
                  render_camera_transform="identity", no_HR_or_LR_pixels_read=True,
                  no_teacher=True, posthoc_cull=False, colour_alignment=False,
                  checkpoint=identity(checkpoint) if checkpoint_exists else dict(path=str(checkpoint) if checkpoint else None, status="pending_training"),
                  source=identity(__file__), GPU_calls=0,
                  records=[dict(camera_id=row["camera_id"], frame=row["frame_index"],
                                path=f"{row['camera_id']}/{row['frame_index']:04d}.png") for row in render_rows],
                  remaining_TODO=["GPU execution and output verification under future caller resource admission", "HR scoring in the separate shared scorer"])
    return spec, geometry, render_rows, checkpoint, record


def execute(args, spec, geometry, rows, checkpoint, expected):
    if not os.environ.get("CUDA_VISIBLE_DEVICES") or "," in os.environ["CUDA_VISIBLE_DEVICES"]:
        raise ValueError("External admitted caller must explicitly bind a single CUDA_VISIBLE_DEVICES")
    if checkpoint is None or not checkpoint.is_file():
        raise ValueError("Complete checkpoint required; this renderer does not train")
    out = args.out.resolve()
    # Reject stale output before any GPU/model import.
    if (out / "render_index.json").exists() or any(out.glob("cam*/*.png")):
        raise FileExistsError("Preserve existing render outputs; choose a fresh --out")
    for entry in spec["required_files"]:
        if sha(local(entry["path"])) != entry["sha256"]:
            raise ValueError("Prepared input/source changed: " + entry["path"])
    import numpy as np
    from PIL import Image
    import torch
    torch.set_num_threads(args.cpu_threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    if spec["method"] == "wu":
        sys.path.insert(0, str(ROOT / "experiments/dynamic_sr_cull_retrain_20261009"))
        import cullretrain as cr
        training_plan = json.loads((local(spec["parent_plan_directory"]) / "config.json").read_text())
        model, payload = cr.load_for_render(checkpoint, training_plan)
        metadata = payload["metadata"]
        _, _, pipeline, _ = cr.configuration(training_plan)
        background = torch.ones(3, device="cuda") if cr.white_background(training_plan) else torch.zeros(3, device="cuda")

        def render(row):
            dummy = cr.ev.camera_item(geometry, row, torch)
            camera = cr.native.hr_camera(dummy, row["frame_index"], geometry)
            state = cr.native.native_effective_state(model, float(row["time"]), torch)
            return cr.native.native_rgb(camera, model, state, pipeline, background)["render"]
    elif spec["method"] == "multi4d":
        upstream = local(spec["upstream"])
        os.environ["MULTI4D_UPSTREAM"] = str(upstream)
        for directory in (ROOT / "experiments/dynamic_sr_multi4d_20260924", upstream):
            sys.path.insert(0, str(directory))
        loader = module("expansion_multi_model", ROOT / "experiments/dynamic_sr_multi4d_20260924/multi_model.py")
        camera_data = module("expansion_multi_hr_camera", OLD / "multi4d_native/eval_hr_data.py")
        model = loader.load_model(checkpoint, manifest=geometry, restore_rng=False)
        metadata = model.checkpoint["metadata"]
        if (metadata["manifest_sha"] != expected["manifest_sha256"] or metadata["teacher_weight"] != 0
                or metadata["stage"] != "fine" or metadata["completed_updates"] != 20000):
            raise ValueError("Only the complete same-scene no-teacher native Multi4D endpoint is accepted")
        for branch in model.branches:
            for value in vars(branch).values():
                if isinstance(value, torch.nn.Module):
                    value.eval()
        for branch in model.branches:
            for value in vars(branch).values():
                if isinstance(value, torch.nn.Parameter):
                    value.requires_grad_(False)

        def render(row):
            width, height = geometry["resolutions"]["lr"]
            camera = camera_data.camera(geometry, row, torch.zeros((3, height, width)))
            return loader.render_model(model, camera)["render"]
    else:
        raise ValueError("Unsupported method")
    start = time.monotonic()
    records = []
    width, height = geometry["resolutions"]["hr"]
    for index, row in enumerate(rows):
        with torch.no_grad():
            raw = render(row).detach()
        if tuple(raw.shape) != (3, height, width) or not bool(torch.isfinite(raw).all()):
            raise ValueError("Invalid native HR render")
        array = raw.cpu().permute(1, 2, 0).numpy()
        png = np.rint(np.clip(array, 0, 1) * 255).astype(np.uint8)
        name = f"{row['camera_id']}/{row['frame_index']:04d}.png"
        path = out / name
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(png, mode="RGB").save(path)
        records.append(dict(camera_id=row["camera_id"], frame=row["frame_index"], path=name, render_sha256=sha(path)))
        if (index + 1) % 100 == 0:
            print(json.dumps(dict(rendered=index + 1, total=len(rows))), flush=True)
    result = dict(schema="dynamic_lr_multiview_render_v1", manifest_sha256=expected["manifest_sha256"],
                  scope=args.scope, resolution=[width, height], input_representation="uint8_RGB_PNG",
                  render_camera_transform="identity", records=records, status="completed",
                  method=spec["method"], scene=spec["scene"], checkpoint=identity(checkpoint),
                  evaluator_source=identity(__file__), no_HR_or_LR_pixels_read=True,
                  no_teacher=True, colour_alignment=False, posthoc_cull=False,
                  checkpoint_metadata=metadata, seconds=time.monotonic() - start,
                  CUDA_VISIBLE_DEVICES=os.environ["CUDA_VISIBLE_DEVICES"], torch=str(torch.__version__))
    write(out / "render_index.json", result)
    print(json.dumps(dict(status="render_completed_HR_scoring_pending", observations=len(records), out=str(out))), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--scope", required=True, choices=("train", "test"))
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path, help="Completed newly trained endpoint; omitted for reused endpoint")
    parser.add_argument("--frames", help="Comma-separated explicit frames or all; defaults train0,10,...290/test0..299")
    parser.add_argument("--execute", action="store_true", help="Explicit GPU execution; caller supplies resource admission and binding")
    parser.add_argument("--plan-only", action="store_true", help="Explicit spelling of default; incompatible with --execute")
    parser.add_argument("--cpu-threads", type=int, choices=(1, 2, 4), default=1)
    args = parser.parse_args()
    if args.execute and args.plan_only:
        parser.error("--execute and --plan-only are mutually exclusive")
    spec, geometry, rows, checkpoint, expected = plan(args)
    args.out.mkdir(parents=True, exist_ok=True)
    write(args.out / "expected_views.json", expected)
    if not args.execute:
        if "torch" in sys.modules:
            raise RuntimeError("Plan-only unexpectedly imported torch")
        print(json.dumps(dict(status="CPU_plan_only_not_rendered", method=spec["method"],
                              scene=spec["scene"], scope=args.scope, observations=len(rows),
                              expected_views=str((args.out / "expected_views.json").resolve()), GPU_calls=0)))
        return
    execute(args, spec, geometry, rows, checkpoint, expected)


if __name__ == "__main__":
    main()
