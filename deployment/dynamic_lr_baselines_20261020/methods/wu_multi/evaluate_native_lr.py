"""HR-only held-out evaluation shared by original-LR Wu and Multi4D.

Only cam00 HR references are opened. No test LR, training HR, teacher, culling,
colour fit or image correction enters rendering or scoring.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def metrics():
    import cv2
    import numpy as np
    import torch
    from PIL import Image
    source = ROOT / "experiments/dynamic_sr_20260918/evaluate.py"
    names = {"read_rgb", "write_rgb", "image_array", "ssim_map_rgb", "masked_mean", "metric_psnr", "spatial_metrics"}
    nodes = [n for n in ast.parse(source.read_text()).body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == names
    scope = dict(cv2=cv2, np=np, torch=torch, Image=Image, Path=Path)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), scope)
    return SimpleNamespace(**{name: scope[name] for name in names})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("wu", "multi4d"), required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    import numpy as np
    import torch
    import lpips
    from PIL import Image
    spec = json.loads(args.spec.read_text())
    if args.method == "wu":
        plan = json.loads((ROOT / spec["parent_plan_directory"] / "config.json").read_text())
        manifest_path = ROOT / plan["data"]["manifest"]["path"]
        assert sha(manifest_path) == plan["data"]["manifest"]["sha256"]
    else:
        manifest_path = ROOT / spec["manifest"]
    manifest = json.loads(manifest_path.read_text())
    assert not manifest.get("physical_lr_derivation") and manifest["splits"]["test"] == ["cam00"]
    tests = sorted((r for r in manifest["observations"] if r["split"] == "test"), key=lambda r: r["frame_index"])
    assert [(r["camera_id"], r["frame_index"]) for r in tests] == [("cam00", i) for i in range(300)]
    torch.set_num_threads(args.cpu_threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    if args.method == "wu":
        sys.path.insert(0, str(ROOT / "experiments/dynamic_sr_cull_retrain_20261009"))
        import cullretrain as cr
        model, payload = cr.load_for_render(args.checkpoint, plan)
        metadata = payload["metadata"]
        _, _, pipeline, _ = cr.configuration(plan)
        background = torch.ones(3, device="cuda") if cr.white_background(plan) else torch.zeros(3, device="cuda")
        def render(row):
            camera = cr.native.hr_camera(cr.ev.camera_item(manifest, row, torch), row["frame_index"], manifest)
            state = cr.native.native_effective_state(model, float(row["time"]), torch)
            return cr.native.native_rgb(camera, model, state, pipeline, background)["render"]
    else:
        upstream = ROOT / spec["upstream"]
        os.environ["MULTI4D_UPSTREAM"] = str(upstream)
        for path in (ROOT / "experiments/dynamic_sr_multi4d_20260924", upstream):
            sys.path.insert(0, str(path))
        loader = module("native_lr_multi_model", ROOT / "experiments/dynamic_sr_multi4d_20260924/multi_model.py")
        data = module("native_lr_hr_camera", HERE / "multi4d_native/eval_hr_data.py")
        model = loader.load_model(args.checkpoint, manifest=manifest, restore_rng=False)
        metadata = model.checkpoint["metadata"]
        assert metadata["teacher_weight"] == 0 and metadata["manifest_sha"] == sha(manifest_path)
        assert metadata["stage"] == "fine" and metadata["completed_updates"] == 20000
        for branch in model.branches:
            for value in vars(branch).values():
                if isinstance(value, torch.nn.Module):
                    value.eval()
        def render(row):
            width, height = manifest["resolutions"]["lr"]
            camera = data.camera(manifest, row, torch.zeros((3, height, width)))
            return loader.render_model(model, camera)["render"]
    metric = lpips.LPIPS(net="alex", version="0.1", spatial=False, pnet_rand=False,
                         pretrained=True, verbose=False).cuda().eval().requires_grad_(False)
    helpers = metrics()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / "previews").mkdir()
    (out / "renders_hr").mkdir()
    record = dict(schema="native_LR_heldout_HR_only_v1", method=args.method, scene=manifest["scene"],
                  spec=dict(path=str(args.spec), sha256=sha(args.spec)),
                  manifest=dict(path=str(manifest_path), sha256=sha(manifest_path)),
                  checkpoint=dict(path=str(args.checkpoint), sha256=sha(args.checkpoint)),
                  checkpoint_metadata=metadata, posthoc_cull=False, colour_alignment=False,
                  no_test_LR_opened=True, no_training_HR_opened=True, no_teacher_opened=True,
                  host=socket.gethostname(), gpu=torch.cuda.get_device_name(),
                  CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES"), torch=str(torch.__version__),
                  metric_source_sha256=sha(ROOT / "experiments/dynamic_sr_20260918/evaluate.py"),
                  evaluator_sha256=sha(__file__),
                  spatial_definition="Unchanged legacy spatial_metrics: full RGB PSNR, SSIM outer5 excluded, AlexNet LPIPS v0.1; equal-frame average")
    (out / "run.json").write_text(json.dumps(record, indent=2, default=str) + "\n")
    rows = []
    started = time.monotonic()
    for row in tests:
        with torch.inference_mode():
            raw = render(row).detach()
        width, height = manifest["resolutions"]["hr"]
        assert tuple(raw.shape) == (3, height, width) and bool(torch.isfinite(raw).all())
        # References enter after the native rendering call.
        hr_path = manifest_path.parent / row["hr_path"]
        assert sha(hr_path) == row["hr_sha256"]
        pred, gt = helpers.image_array(raw), helpers.read_rgb(hr_path)
        helpers.write_rgb(out / "renders_hr" / f"{row['frame_index']:04d}.png", pred)
        score = helpers.spatial_metrics(pred, gt, np.zeros((height, width), bool), metric, "cuda")["full"]
        rows.append(dict(camera="cam00", frame=row["frame_index"], psnr=score["psnr"],
                         ssim=score["ssim"], lpips=score["lpips_alex"], mse=score["mse"]))
        if row["frame_index"] in (0, 150, 299):
            stem = f"cam00_{row['frame_index']:04d}"
            Image.fromarray(np.rint(pred.clip(0, 1) * 255).astype(np.uint8)).save(out / "previews" / (stem + ".png"))
            np.savez_compressed(out / "previews" / (stem + ".npz"), rgb_raw=raw.cpu().numpy().astype(np.float32))
        if row["frame_index"] % 50 == 0:
            print(json.dumps(dict(frame=row["frame_index"], seconds=time.monotonic() - started)), flush=True)
    with (out / "metrics_per_observation.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = dict(record, status="completed", frames=300,
                  test={key: float(np.mean([r[key] for r in rows])) for key in ("psnr", "ssim", "lpips", "mse")},
                  metric_protocol="raw_float_diagnostic_only; main table must rescore all renders_hr PNG through shared evaluate_images.py",
                  PNG_protocol="legacy.write_rgb: np.rint(clip(rgb,0,1)*255).astype(uint8)",
                  seconds=time.monotonic() - started, peak_GB=torch.cuda.max_memory_allocated() / 1e9)
    (out / "summary.json").write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(json.dumps(dict(status="completed", test=result["test"])), flush=True)


if __name__ == "__main__":
    main()
