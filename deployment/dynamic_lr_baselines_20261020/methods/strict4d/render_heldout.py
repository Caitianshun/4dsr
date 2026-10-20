#!/usr/bin/env python3
"""Render native 4DGS directly at held-out HR, then compute shared metrics.

No parameter updates, teacher, area downsampling loss, calibration learning or
proposed module. The checkpoint is fixed before the first held-out HR read.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def evaluate(a):
    import numpy as np
    import torch
    import cv2
    from PIL import Image
    sys.path.insert(0, str(a.runtime.resolve()))
    from gaussian_renderer import render
    from scene.gaussian_model import GaussianModel
    from scene.cameras import Camera
    from prepare_native import normalized_poses
    manifest_path = a.manifest.resolve()
    m = json.loads(manifest_path.read_text())
    receipt = json.loads((a.dataset / "stage_receipt.json").read_text())
    if receipt["manifest_sha256"] != sha(manifest_path) or receipt["used_HR_pixels"] or receipt["used_heldout_pixels"]:
        raise ValueError("Stage is not bound to the LR-only manifest")
    if m["splits"]["test"] != ["cam00"]:
        raise ValueError("Requires cam00 holdout")
    rows = sorted([r for r in m["observations"] if r["split"] == "test"], key=lambda r: r["frame_index"])
    if [(r["camera_id"], r["frame_index"]) for r in rows] != [("cam00", i) for i in range(300)]:
        raise ValueError("Requires all 300 heldout frames")
    a.output.mkdir(parents=True, exist_ok=False)
    (a.output / "renders_hr").mkdir()
    poses, similarity = normalized_poses(m)
    if not np.allclose(similarity, receipt["native_world_from_manifest_world"]):
        raise ValueError("Native camera normalization differs from initialization")
    # Only model-independent metric functions are reused. No reconstruction or
    # method modules from this project are imported.
    source = a.metric_source.read_text()
    names = {"ssim_map_rgb", "metric_psnr"}
    nodes = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name in names]
    if {n.name for n in nodes} != names:
        raise ValueError("Shared metric definitions changed")
    scope = {"np": np, "cv2": cv2, "math": math}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(a.metric_source), "exec"), scope)
    model = GaussianModel(3, gaussian_dim=4, time_duration=[0., 10.], rot_4d=True, force_sh_3d=False, sh_degree_t=2)
    checkpoint, iteration = torch.load(a.checkpoint, map_location="cuda", weights_only=False)
    if iteration != a.expected_iteration:
        raise ValueError(f"Expected iteration {a.expected_iteration}; got {iteration}")
    model.restore(checkpoint, None)
    pipe = SimpleNamespace(env_map_res=0, compute_cov3D_python=False, convert_SHs_python=False, debug=False)
    background = torch.zeros(3, dtype=torch.float32, device="cuda")
    if not a.render_only:
        import lpips
        metric = lpips.LPIPS(net="alex", version="0.1", spatial=False, pnet_rand=False).to("cuda").eval()
    width, height = m["resolutions"]["hr"]
    calibration = m["cameras"]["cam00"]
    k = np.asarray(calibration["K_hr"])
    pose_cv = poses["cam00"] @ np.diag([1., -1., -1., 1.])
    w2c = np.linalg.inv(pose_cv)
    results, began = [], time.monotonic()
    with torch.no_grad():
        for row in rows:
            f = row["frame_index"]
            camera = Camera(colmap_id=0, R=w2c[:3, :3].T, T=w2c[:3, 3],
                            FoVx=2*math.atan(width/(2*k[0, 0])), FoVy=2*math.atan(height/(2*k[1, 1])),
                            image=np.empty(0), gt_alpha_mask=None, image_name=f"cam00_{f:04d}", uid=f,
                            data_device="cuda", timestamp=f/30., cx=k[0, 2]+.5, cy=k[1, 2]+.5,
                            fl_x=k[0, 0], fl_y=k[1, 1], resolution=(width, height), meta_only=True).cuda()
            raw_prediction = render(camera, model, pipe, background)["render"]
            if tuple(raw_prediction.shape)!=(3,height,width) or not torch.isfinite(raw_prediction).all().item():
                raise ValueError(f"Invalid native HR render before clamp/quantization at frame {f}: shape={tuple(raw_prediction.shape)}")
            prediction = raw_prediction.clamp(0, 1)
            pred = prediction.permute(1, 2, 0).cpu().numpy()
            prediction_path = a.output / "renders_hr" / f"{f:04d}.png"
            Image.fromarray(np.rint(pred*255).clip(0, 255).astype(np.uint8)).save(prediction_path)
            if a.render_only:
                score = {"camera": "cam00", "frame": f, "path": str(prediction_path), "sha256": sha(prediction_path)}
                results.append(score)
                if f % 50 == 0:
                    print(json.dumps(score), flush=True)
                continue
            path = manifest_path.parent / row["hr_path"]
            if sha(path) != row["hr_sha256"]:
                raise ValueError("Heldout HR bytes changed")
            with Image.open(path) as image:
                gt = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.
            if gt.shape != pred.shape:
                raise ValueError("Prediction/HR shape mismatch")
            mse = float(((pred-gt)**2).mean(axis=2).mean())
            smap = scope["ssim_map_rgb"](pred, gt)
            ssim = float(smap[5:-5, 5:-5].mean())
            target = torch.from_numpy(np.ascontiguousarray(gt)).permute(2, 0, 1).to("cuda")
            lp = float(metric(prediction[None]*2-1, target[None]*2-1).item())
            score = {"camera": "cam00", "frame": f, "mse": mse, "psnr": scope["metric_psnr"](mse), "ssim": ssim, "lpips": lp}
            results.append(score)
            if f % 50 == 0:
                print(json.dumps(score), flush=True)
    result = {"schema": "strict4d_native_hr_heldout_metrics_v1", "status": "complete", "method": a.method,
              "scene": m["scene"], "checkpoint": str(a.checkpoint), "checkpoint_sha256": sha(a.checkpoint),
              "iteration": iteration, "manifest": str(manifest_path), "manifest_sha256": sha(manifest_path),
              "original_LR": True, "no_proposed_modules": True, "training_saw_heldout_pixels": False,
              "metric_source_sha256": sha(a.metric_source),
              "metric_definition": "Float render clamped[0,1]; full RGB MSE/PSNR; same 11x11 sigma1.5 RGB SSIM excluding outer5; AlexNet LPIPS v0.1; no color correction or crop",
              "observations": len(results), "metrics": None if a.render_only else {n: float(np.mean([r[n] for r in results])) for n in ["psnr", "ssim", "lpips", "mse"]},
              "heldout_HR_pixels_read": not a.render_only,
              "finite_and_shape_guard_before_clamp_and_quantization": {"status":"passed","frames":len(results),"shape":[3,height,width]},
              "per_frame": results, "evaluation_seconds": time.monotonic()-began, "parameter_updates": 0}
    (a.output/"summary.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    if a.render_only:
        (a.output/"render_index.json").write_text(json.dumps({"frames": results, "checkpoint_sha256": result["checkpoint_sha256"]}, indent=2)+"\n")
    print(json.dumps(result["metrics"]), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=["yang", "4c4d"], required=True)
    p.add_argument("--runtime", type=Path, required=True)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--metric-source", type=Path, required=True)
    p.add_argument("--expected-iteration", type=int, default=30000)
    p.add_argument("--render-only", action="store_true", help="Save all HR PNGs; common evaluator will read heldout GT separately")
    evaluate(p.parse_args())
