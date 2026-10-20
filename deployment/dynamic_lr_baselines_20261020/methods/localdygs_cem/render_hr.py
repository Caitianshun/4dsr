"""Stream native method HR renders to 300 canonical 8-bit RGB PNGs.

Uses each author's final model loader and rasterizer. It does not compute
metrics against LR placeholders or introduce any SR postprocessor.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["localdygs", "cem"], required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--scene", choices=["cook_spinach", "cut_roasted_beef"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    a = parser.parse_args()
    if a.output.exists():
        raise SystemExit(f"Refusing existing render output: {a.output}")
    a.output.mkdir(parents=True)
    a.output = a.output.resolve()
    a.model = a.model.resolve()
    a.source = a.source.resolve()
    runtime = a.runtime.resolve()
    os.chdir(runtime)
    sys.path.insert(0, str(runtime))
    os.environ["DYNAMIC_LR_RENDER_HR"] = "1"
    import torch
    import numpy as np
    from PIL import Image
    from argparse import ArgumentParser
    from arguments import ModelParams, OptimizationParams, PipelineParams, get_combined_args
    from scene import Scene
    from gaussian_renderer import render
    from utils.general_utils import safe_state
    native_parser = ArgumentParser()
    model = ModelParams(native_parser, sentinel=True)
    optimizer = OptimizationParams(native_parser)
    pipeline = PipelineParams(native_parser)
    sys.argv = ["render_hr", "--model_path", str(a.model.resolve()), "--source_path", str(a.source.resolve())]
    if a.method == "localdygs":
        from arguments import ModelHiddenParams
        from utils.general_utils import merge_hparams
        import mmcv
        hidden = ModelHiddenParams(native_parser)
        arguments = get_combined_args(native_parser)
        config = mmcv.Config.fromfile(str(runtime / f"arguments/dynerf/{a.scene}.py"))
        arguments = merge_hparams(arguments, config)
    else:
        arguments = get_combined_args(native_parser)
    dataset = model.extract(arguments)
    opt = optimizer.extract(arguments)
    pipe = pipeline.extract(arguments)
    safe_state(False)
    final_iteration = 30000 if a.method == "localdygs" else 50000
    records = []
    began = time.monotonic()
    with torch.no_grad():
        if a.method == "localdygs":
            from gaussian_renderer import GaussianModel, prefilter_voxel
            gaussians = GaussianModel(hidden.extract(arguments), opt, dataset.feat_dim, dataset.n_offsets,
                dataset.voxel_size, dataset.update_depth, dataset.update_init_factor, dataset.update_hierachy_factor,
                dataset.use_feat_bank, dataset.appearance_dim, dataset.ratio, dataset.add_opacity_dist,
                dataset.add_cov_dist, dataset.add_color_dist)
            scene = Scene(dataset, gaussians, frames_start_end=[0, 300], load_iteration=final_iteration, shuffle=False)
            views = scene.getTestCameras()
        else:
            from scene import init_model
            gaussians = init_model(dataset.model, dataset)
            scene = Scene(dataset, gaussians, load_iteration=final_iteration, shuffle=False, opt=opt)
            views, unused_images = scene.getTestCameras(return_as="generator", shuffle=False, get_img=False)
        if len(views) != 300:
            raise RuntimeError(f"Expected all 300 held-out frames: {len(views)}")
        background = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0], dtype=torch.float32, device="cuda")
        for index, view in enumerate(views):
            if (view.image_width, view.image_height) != (1344, 1008):
                raise RuntimeError("HR camera size mismatch")
            if a.method == "localdygs":
                frame = round(float(view.time) * 300)
                visible = prefilter_voxel(view, gaussians, pipe, background)
                prediction = render(view, gaussians, pipe, background, stage="fine", visible_mask=visible, retain_grad=False)["render"]
            else:
                frame = int(view.timestamp)
                prediction = render(view, gaussians, pipe, background, mode=0, near=dataset.near, far=dataset.far)["render"]
            if frame != index:
                raise RuntimeError(("Held-out frame ordering mismatch", index, frame))
            if tuple(prediction.shape) != (3, 1008, 1344):
                raise RuntimeError(f"Unexpected native HR prediction shape: {tuple(prediction.shape)}")
            if not bool(torch.isfinite(prediction).all().item()):
                raise RuntimeError(f"Native prediction has non-finite values before quantization: frame {frame}")
            destination = a.output / f"{frame:04d}.png"
            rgb = prediction.detach().float().cpu().permute(1, 2, 0).numpy()
            Image.fromarray(np.rint(np.clip(rgb, 0, 1) * 255).astype(np.uint8), "RGB").save(destination)
            records.append(dict(frame=frame, path=destination.name, sha256=digest(destination)))
    receipt = dict(status="completed", method=a.method, scene=a.scene, iteration=final_iteration,
                   elapsed_seconds=time.monotonic() - began, frames=300, pattern="{frame:04d}.png",
                   protocol="8-bit RGB PNG; native rasterizer at HR, no SR",
                   raw_prediction_shape=[3, 1008, 1344], finite_before_clip_quantization_passed_frames=300,
                   quantization="np.rint(np.clip(CHW_to_HWC_float32,0,1)*255).astype(np.uint8); PIL RGB PNG", records=records)
    (a.output / "render_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
