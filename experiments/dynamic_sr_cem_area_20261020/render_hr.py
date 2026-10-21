"""Stream CEM native/SR endpoints to the common 300-frame 8-bit PNG protocol.

Only cam00 metadata is used. Its colour and w4 correction are exact identity.
Held-out HR pixels are read later by the independent common evaluator.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

from sr_protocol import CameraCorrection, sha
from sr_stage import native_camera


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-receipt", type=Path, required=True)
    parser.add_argument("--kind", choices=["native", "sr"], required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--camera", choices=["cam00", "cam01"], default="cam00",
                        help="cam00: all300 heldout; cam01: training diagnostic at fixed0/150/299")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert args.gpu.isdecimal() and os.environ.get("CUDA_VISIBLE_DEVICES") == args.gpu
    assert not args.output.exists()
    runtime = args.runtime.resolve()
    manifest = json.loads(args.manifest.read_text())
    assert manifest["splits"]["test"] == ["cam00"] and manifest["frame_indices"] == list(range(300))
    assert manifest["physical_lr_derivation"]["formation"] == "area4"
    training = json.loads(args.training_receipt.read_text())
    assert training["status"] == "completed" and training["manifest"]["sha256"] == sha(args.manifest)
    checkpoint_key = "final_checkpoint" if args.kind == "native" else "checkpoint"
    assert training[checkpoint_key]["sha256"] == sha(args.checkpoint)
    if args.kind == "native":
        assert training["stage"] == 2 and training["endpoint"] == 50000
    else:
        assert training["result_scope"] == "controlled_fixed_topology_SR_continuation"
        assert training["parameter_updates"] == 6000 and training["final_iteration"] == 56000
    protocol = json.loads((runtime / "protocol_interface_receipt.json").read_text())
    assert protocol["version"] == "native_area4_v1"
    assert sha(runtime / "protocol_interface.py") == protocol["interface_sha256"]
    for item in protocol["core_source_files"] + protocol["configurations"]:
        assert sha(runtime / item["path"]) == item["sha256"]
    sys.path.insert(0, str(runtime))
    import numpy as np
    import torch
    from PIL import Image
    from arguments import ModelParams, OptimizationParams, PipelineParams
    from gaussian_renderer import render
    from scene import init_model
    from scene.cameras import Cameravideo
    native_parser = argparse.ArgumentParser()
    model_group, optimizer_group, pipeline_group = ModelParams(native_parser), OptimizationParams(native_parser), PipelineParams(native_parser)
    native_args = native_parser.parse_args([])
    native_args.__dict__.update(json.loads((runtime / f"configs/N3V_s2/{manifest['scene']}_prepared_lr.json").read_text()))
    native_args.duration = 300
    correction = None
    if args.kind == "native":
        payload, iteration = torch.load(args.checkpoint, weights_only=False)
        assert iteration == 50000
    else:
        checkpoint = torch.load(args.checkpoint, weights_only=False)
        assert checkpoint["schema"] == "CEM_area4_SR_fixed_topology_v1" and checkpoint["iteration"] == 56000
        native_args.__dict__.update(checkpoint["native_arguments"])
        payload, iteration = checkpoint["model"], checkpoint["iteration"]
        # No training-camera correction is applied to this held-out view.
        assert "cam00" not in checkpoint["correction"]["names"]
        if args.camera == "cam01":
            correction = CameraCorrection(checkpoint["correction"]["names"], checkpoint["correction"]["w4"])
            correction.load_for_render(checkpoint["correction"])
    dataset, opt, pipe = model_group.extract(native_args), optimizer_group.extract(native_args), pipeline_group.extract(native_args)
    gaussians = init_model(dataset.model, dataset)
    gaussians.restore(payload, opt)
    del payload
    output = args.output.resolve()
    output.mkdir(parents=True)
    background = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0], dtype=torch.float32, device="cuda")
    records = []
    began = time.monotonic()
    frames = list(range(300)) if args.camera == "cam00" else [0, 150, 299]
    with torch.no_grad():
        for frame in frames:
            camera = native_camera(manifest, args.camera, frame, Cameravideo)
            prediction = render(camera, gaussians, pipe, background, near=dataset.near, far=dataset.far)["render"]
            assert tuple(prediction.shape) == (3, 1008, 1344)
            assert bool(torch.isfinite(prediction).all()), f"Nonfinite native output before clipping: frame {frame}"
            raw = None
            if correction is not None:
                raw = output / f"{frame:04d}_raw.png"
                raw_rgb = prediction.detach().float().cpu().permute(1, 2, 0).numpy()
                Image.fromarray(np.rint(np.clip(raw_rgb, 0, 1) * 255).astype(np.uint8), "RGB").save(raw)
                prediction = correction.apply(prediction, args.camera)
                assert bool(torch.isfinite(prediction).all()), f"Nonfinite corrected training diagnostic: frame {frame}"
            rgb = prediction.detach().float().cpu().permute(1, 2, 0).numpy()
            destination = output / f"{frame:04d}.png"
            Image.fromarray(np.rint(np.clip(rgb, 0, 1) * 255).astype(np.uint8), "RGB").save(destination)
            records.append(dict(frame=frame, path=destination.name, sha256=sha(destination),
                                raw_path=raw.name if raw else None, raw_sha256=sha(raw) if raw else None))
            del camera, prediction
    receipt = dict(status="completed", kind=args.kind, iteration=iteration, scene=manifest["scene"], frames=len(frames), frame_indices=frames,
        camera=args.camera, scope="heldout_300" if args.camera == "cam00" else "training_view_diagnostic_only_fixed_three_frames",
        pattern="{frame:04d}.png", training_receipt=dict(path=str(args.training_receipt.resolve()), sha256=sha(args.training_receipt)),
        checkpoint=dict(path=str(args.checkpoint.resolve()), sha256=sha(args.checkpoint)),
        manifest=dict(path=str(args.manifest.resolve()), sha256=sha(args.manifest)),
        runtime_receipt=dict(path=str(runtime / "protocol_interface_receipt.json"), sha256=sha(runtime / "protocol_interface_receipt.json")),
        renderer_source_sha256=sha(__file__), camera_helper_sha256=sha(Path(__file__).parent / "sr_stage.py"),
        raw_prediction_shape=[3, 1008, 1344], finite_before_clip_quantization_passed_frames=len(frames),
        quantization="np.rint(np.clip(CHW_to_HWC_float32,0,1)*255).astype(np.uint8); PIL RGB PNG",
        heldout_pixels_read=False, training_pixels_read=False,
        colour_affine="learned training-camera affine" if correction else "identity",
        warp="learned training-camera w4" if correction and correction.use_w4 else "identity", native_rasterizer_unchanged=True,
        GPU_physical_mask=args.gpu, GPU_name=torch.cuda.get_device_name(0), elapsed_seconds=time.monotonic() - began,
        finished_utc=datetime.now(timezone.utc).isoformat(), records=records)
    (output / "render_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
