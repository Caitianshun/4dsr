"""Native STG-full 50-frame area4 validation runner.

GPU ownership/admission is external. This entry never chooses a GPU or launches
other workers. PLY+decoder are render checkpoints, not restartable training state.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time
import traceback

import numpy as np

HERE = Path(__file__).resolve().parent
COMMIT = "427abfc58309a4a5213843dd673fb22c4529306c"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + "\n")
    temporary.replace(path)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class Bridge:
    def __init__(self, spec, args, data, native_args):
        import torch
        self.spec, self.a, self.data, self.native_args = spec, args, data, native_args
        self.out = Path(args.out)
        self.start = time.monotonic()
        self.training_start = None
        self.cache = OrderedDict()
        self.teacher_records = {}
        self.weight = float(spec.get("teacher_weight", .2)) if spec["method"] != "areaL0" else 0.
        self.dssim = float(spec.get("teacher_dssim", .2))
        if not 0 <= self.dssim <= 1 or self.weight < 0:
            raise ValueError("Invalid teacher weights")
        self.sample_digest = hashlib.sha256()
        self.meta = dict(schema="stg_area4_run_v1", upstream_commit=COMMIT, method=spec["method"],
                         manifest=dict(path=spec["manifest"], sha256=sha(spec["manifest"])),
                         initialization=dict(path=spec["init"], sha256=sha(spec["init"])),
                         spec_sha256=sha(args.spec), segment_start=int(spec["segment_start"]), duration=50,
                         global_frames=list(range(int(spec["segment_start"]), int(spec["segment_start"]) + 50)),
                         local_time="(global_frame-segment_start)/50", seed=int(spec.get("seed", 0)),
                         native_planned_iterations=30000, native_evaluation_checkpoint=25000,
                         selected_checkpoint=int(native_args.iterations),
                         truncated_after_selected_checkpoint=int(native_args.iterations) < 30000,
                         position_lr_max_steps=int(native_args.position_lr_max_steps),
                         forward_backward_batches=int(native_args.iterations), optimizer_updates=int(native_args.iterations)-1,
                         checkpoint_semantics="native save before controller and optimizer; endpoint has no optimizer update",
                         native_batch=2, batch_semantics="same local timestamp, two sampled distinct training cameras; average gradients",
                         observation="HR render -> non-overlapping avg_pool2d(4,4).clamp(0,1); compare to registered area4 LR",
                         ems="LR SSIM decisions; LR RGB residual/color repeated 4x4 to HR; native HR depth/projection/16x16 patch controller",
                         teacher_weight=self.weight, teacher_dssim=self.dssim,
                         teacher_formula="lambda*((1-dssim)*L1(HR render,target)+dssim*(1-SSIM(HR render,target)))",
                         oracle_diagnostic_only=spec["method"] == "areaOracle", resumable=False,
                         gpu=torch.cuda.get_device_name(), torch=str(torch.__version__),
                         visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                         max_wall_seconds=spec.get("max_wall_seconds"))
        self.log = (self.out / "training.jsonl").open("w", buffering=1)
        self.samples = (self.out / "sampling.jsonl").open("w", buffering=1)

    def scene_factory(self, dataset, gaussians, duration, loader):
        if duration != 50 or loader != "colmap":
            raise ValueError("Only the declared native N3DV 50-frame interface is supported")
        self.scene = self.data.ManifestScene(self.spec["manifest"], self.spec["init"], self.spec["segment_start"],
                                            gaussians, self.out, self.spec.get("seed", 0))
        self.meta["initialization_adapter"] = self.scene.initialization
        self.meta["projection"] = self.scene.projection
        if self.weight:
            self.load_teachers()
        write(self.out / "input_audit.json", dict(metadata=self.meta, lr_records=self.scene.input_identity))
        write(self.out / "run_metadata.json", self.meta)
        return self.scene

    def load_teachers(self):
        rows = {(r["camera_id"], int(r["frame_index"])): r for r in self.scene.records}
        root = Path(self.spec["manifest"]).parent
        index_path = self.spec.get("teacher_index") if self.spec["method"] == "areaTm" else self.spec.get("oracle_teacher_index")
        if self.spec["method"] == "areaOracle" and not index_path:
            entries = [dict(camera=k[0], frame=k[1], relative_path=r["hr_path"], sha256=r["hr_sha256"],
                            lr_sha256=r["lr_sha256"]) for k, r in rows.items()]
            self.meta["teacher_index"] = "registered training-view HR in manifest, diagnostic oracle only"
        else:
            if not index_path:
                raise ValueError("areaTm requires the frozen matching teacher index")
            index = json.loads(Path(index_path).read_text())
            if index["manifest_sha256"] != self.meta["manifest"]["sha256"]:
                raise ValueError("Teacher is bound to another manifest")
            if bool(index.get("oracle", False)) != (self.spec["method"] == "areaOracle"):
                raise ValueError("Teacher/oracle declaration differs from method")
            entries = [entry for entry in index["entries"] if (entry["camera"], int(entry["frame"])) in rows]
            self.meta["teacher_index"] = dict(path=str(index_path), sha256=sha(index_path))
        if len(entries) != len(rows) or {(e["camera"], int(e["frame"])) for e in entries} != set(rows):
            raise ValueError("Teacher must cover the exact training camera/frame segment")
        for entry in entries:
            key = entry["camera"], int(entry["frame"])
            row = rows[key]
            if entry["lr_sha256"] != row["lr_sha256"]:
                raise ValueError("Teacher input LR mismatch")
            path = self.data.checked_path(root, entry["relative_path"])
            if self.spec["method"] == "areaOracle" and (path != self.data.checked_path(root, row["hr_path"]) or entry["sha256"] != row["hr_sha256"]):
                raise ValueError("Oracle must use exactly the registered training HR")
            if sha(path) != entry["sha256"]:
                raise ValueError(f"Teacher bytes changed: {path}")
            self.teacher_records[key] = path
        self.meta["teacher_segment_entries"] = len(entries)

    def begin_iteration(self, iteration):
        import torch
        if self.training_start is None:
            torch.cuda.synchronize()
            self.training_start = time.monotonic()
            torch.cuda.reset_peak_memory_stats()
        self.iteration = iteration
        self.batch_records = []

    def observe(self, image):
        import torch.nn.functional as F
        if image.ndim != 3 or image.shape[-1] % 4 or image.shape[-2] % 4:
            raise ValueError("HR render must be CHW with dimensions divisible by four")
        return F.avg_pool2d(image, 4, 4).clamp(0, 1)

    def add_teacher_loss(self, native_loss, image_hr, camera, ssim):
        import torch
        from PIL import Image
        key = tuple(camera.record_key)
        teacher_loss = image_hr.new_zeros(())
        if self.weight:
            if key not in self.teacher_records:
                raise ValueError("Attempt to supervise a non-training camera/frame")
            if key not in self.cache:
                with Image.open(self.teacher_records[key]) as image:
                    if list(image.size) != self.scene.m["resolutions"]["hr"] or image.mode != "RGB":
                        raise ValueError("Teacher shape/mode mismatch")
                    array = np.asarray(image).copy()
                self.cache[key] = torch.from_numpy(array).permute(2, 0, 1).contiguous()
                while len(self.cache) > int(self.spec.get("teacher_cache_images", 32)):
                    self.cache.popitem(last=False)
            self.cache.move_to_end(key)
            target = self.cache[key].to(image_hr.device).float() / 255.
            teacher_loss = self.weight * ((1-self.dssim)*(image_hr-target).abs().mean() + self.dssim*(1-ssim(image_hr, target)))
        loss = native_loss + teacher_loss
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Nonfinite STG loss")
        self.batch_records.append(dict(camera=key[0], frame=key[1], native_loss=float(native_loss.detach()),
                                       teacher_loss=float(teacher_loss.detach()), loss=float(loss.detach())))
        return loss

    def ems_images(self, image_lr, gt_lr):
        return (image_lr.repeat_interleave(4, -2).repeat_interleave(4, -1),
                gt_lr.repeat_interleave(4, -2).repeat_interleave(4, -1))

    def end_iteration(self, iteration, gaussians, loss, opt):
        import torch
        if len(self.batch_records) != 2 or len({r["frame"] for r in self.batch_records}) != 1 or len({r["camera"] for r in self.batch_records}) != 2:
            raise ValueError("Native batch2/same-timestamp sampling invariant failed")
        sample = dict(iteration=iteration, camera_frames=[[r["camera"], r["frame"]] for r in self.batch_records])
        payload = json.dumps(sample, sort_keys=True, separators=(",", ":")) + "\n"
        self.sample_digest.update(payload.encode())
        self.samples.write(payload)
        elapsed = time.monotonic() - self.training_start
        if iteration <= 40 or iteration % 100 == 0 or iteration == opt.iterations:
            event = dict(iteration=iteration, optimizer_updates=min(iteration, opt.iterations-1),
                         elapsed_training_seconds=elapsed, seconds_per_batch=elapsed / iteration,
                         native_loss=float(np.mean([r["native_loss"] for r in self.batch_records])),
                         teacher_loss=float(np.mean([r["teacher_loss"] for r in self.batch_records])),
                         points=len(gaussians._xyz), gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                         gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(), sampling_sha256=self.sample_digest.hexdigest())
            self.log.write(json.dumps(event, allow_nan=False) + "\n")
            write(self.out / "status.json", dict(status="running", **event))
        if self.spec.get("max_wall_seconds") and elapsed > float(self.spec["max_wall_seconds"]):
            raise TimeoutError("Explicit per-segment wall-time budget exhausted; no complete quality result")

    def finish(self):
        import torch
        torch.cuda.synchronize()
        iteration = self.native_args.iterations
        checkpoint = self.out / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"
        decoder = checkpoint.with_suffix(".pt")
        if not checkpoint.is_file() or not decoder.is_file():
            raise RuntimeError("Missing native PLY/decoder pair")
        reload_check = dict(status="not_run", reason="Only the topology-free 40-step smoke checks in-memory/save parity")
        if iteration == 40:
            from types import SimpleNamespace
            from helper_train import getmodel, getrenderpip, trbfunction
            render, settings, rasterizer = getrenderpip("train_ours_full")
            camera = self.scene.train_camera[0]
            self.scene.gaussians.rgbdecoder.eval()
            kwargs = dict(override_color=None, basicfunction=trbfunction, GRsetting=settings, GRzer=rasterizer)
            background = torch.zeros(9, device="cuda")
            with torch.no_grad():
                before = render(camera, self.scene.gaussians, SimpleNamespace(debug=False), background, **kwargs)["render"]
                restored = getmodel("ours_full")(3, "sandwich")
                restored.load_ply(str(checkpoint))
                restored.rgbdecoder.cuda().eval()
                after = render(camera, restored, SimpleNamespace(debug=False), background, **kwargs)["render"]
                error = float((before-after).abs().max())
            reload_check = dict(status="passed" if error < 1e-6 else "failed", max_abs_error=error,
                                threshold=1e-6, camera=camera.record_key[0], frame=camera.record_key[1],
                                actual_saved_updates=39, training_resume_checked=False)
            write(self.out / "checkpoint_reload_check.json", reload_check)
            if reload_check["status"] != "passed":
                raise RuntimeError(f"Native checkpoint render mismatch: {error}")
        complete = dict(status="completed", method=self.spec["method"], segment_start=self.spec["segment_start"], duration=50,
                        iterations=iteration, optimizer_updates=iteration-1,
                        formal_checkpoint=iteration == 25000, smoke_only=iteration != 25000,
                        checkpoint=dict(path=str(checkpoint), sha256=sha(checkpoint)),
                        decoder=dict(path=str(decoder), sha256=sha(decoder)), manifest=self.meta["manifest"],
                        metadata=self.meta, sampling_sha256=self.sample_digest.hexdigest(),
                        total_seconds=time.monotonic()-self.start, training_seconds=time.monotonic()-self.training_start,
                        peak_allocated_bytes=torch.cuda.max_memory_allocated(), resumable=False,
                        checkpoint_reload_check=reload_check)
        self.log.close()
        self.samples.close()
        write(self.out / "complete.json", complete)
        write(self.out / "status.json", complete)
        return complete


def run(args):
    import torch
    spec = json.loads(Path(args.spec).read_text())
    spec["init"] = spec.get("init", spec.get("initialization"))
    if isinstance(spec["init"], dict):
        spec["init"] = spec["init"]["path"]
    if spec.get("method") not in {"areaL0", "areaTm", "areaOracle"} or int(spec.get("duration", 50)) != 50:
        raise ValueError("Expected one declared three-arm method and native 50-frame duration")
    if int(spec["segment_start"]) not in range(0, 300, 50):
        raise ValueError("Invalid segment start")
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise ValueError("External dispatcher must explicitly set CUDA_VISIBLE_DEVICES after admission")
    out = Path(args.out).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Output must be new/empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    args.out = str(out)
    upstream, adapter = Path(spec["upstream"]).resolve(), Path(spec["adapter"]).resolve()
    identity = json.loads((adapter / "identity.json").read_text())
    if identity["upstream_commit"] != COMMIT:
        raise ValueError("Wrong STG version")
    for name, value in identity["original_sha256"].items():
        if sha(upstream / name) != value:
            raise ValueError(f"Pinned upstream changed: {name}")
    for name, value in identity["generated_sha256"].items():
        if sha(adapter / name) != value:
            raise ValueError(f"Adapter identity changed: {name}")
    dependency_path = Path(spec["python_dependencies"]).resolve() if spec.get("python_dependencies") else None
    sys.path[:0] = [str(adapter), str(upstream), str(upstream / "thirdparty/gaussian_splatting")] + ([str(dependency_path)] if dependency_path else [])
    import diff_gaussian_rasterization_ch9
    import diff_gaussian_rasterization_ch9._C as raster_extension
    import simple_knn._C as knn_extension
    extension_files = {"ch9_python": diff_gaussian_rasterization_ch9.__file__,
                       "ch9_cuda": raster_extension.__file__, "simple_knn_cuda": knn_extension.__file__}
    if dependency_path and not all(Path(p).resolve().is_relative_to(dependency_path) for p in extension_files.values()):
        raise ValueError(f"STG extension was loaded outside the isolated dependencies: {extension_files}")
    from arguments import ModelParams, OptimizationParams, PipelineParams
    from helper_train import setgtisint8
    parser = argparse.ArgumentParser()
    lp, op, pp = ModelParams(parser), OptimizationParams(parser), PipelineParams(parser)
    native_args = parser.parse_args([])
    native_config = Path(spec.get("native_config", upstream / "configs/n3d_full/cut_roasted_beef.json"))
    for key, value in json.loads(native_config.read_text()).items():
        if hasattr(native_args, key):
            setattr(native_args, key, value)
    native_args.iterations = int(args.iterations or spec.get("iterations", 25000))
    if native_args.iterations < 2 or native_args.iterations > 30000:
        raise ValueError("iterations must lie between 2 and the native planned 30000")
    native_args.model_path, native_args.source_path = str(out), str(Path(spec["manifest"]).parent)
    native_args.data_device, native_args.resolution, native_args.eval = "cpu", 1, True
    if native_args.model != "ours_full" or native_args.preprocesspoints != 0 or native_args.batch != 2:
        raise ValueError("Only the audited full/preprocesspoints0/batch2 configuration is supported")
    native_args.gtisint8 = 1
    setgtisint8(1)
    seed = int(spec.get("seed", 0))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    data = load_module("stg_area_data", HERE / "data.py")
    native = load_module("stg_area_native_train", adapter / "adapted_train.py")
    bridge = Bridge(spec, args, data, native_args)
    bridge.meta["extension_modules"] = {name: dict(path=str(Path(path).resolve()), sha256=sha(path)) for name, path in extension_files.items()}
    bridge.meta["native_config"] = dict(path=str(native_config), sha256=sha(native_config))
    bridge.meta["adapter"] = dict(path=str(adapter), identity_sha256=sha(adapter / "identity.json"))
    source_paths = [HERE / "train.py", HERE / "data.py", HERE / "build_adapter.py", adapter / "adapted_train.py",
                    adapter / "helper_model.py", upstream / "helper_train.py", upstream / "thirdparty/gaussian_splatting/scene/oursfull.py",
                    upstream / "thirdparty/gaussian_splatting/renderer/__init__.py"]
    bridge.meta["sources"] = {str(p): sha(p) for p in source_paths}
    (out / "sources").mkdir()
    for i, path in enumerate(source_paths):
        shutil.copyfile(path, out / "sources" / (f"{i:02d}_" + path.name))
    shutil.copyfile(args.spec, out / "spec.json")
    shutil.copyfile(adapter / "identity.json", out / "adapter_identity.json")
    shutil.copyfile(adapter / "upstream.patch", out / "upstream.patch")
    native.Scene, native.args = bridge.scene_factory, native_args
    write(out / "status.json", dict(status="initializing", method=spec["method"], segment_start=spec["segment_start"]))
    native.train(lp.extract(native_args), op.extract(native_args), pp.extract(native_args),
                 [native_args.iterations], -1, densify=1, duration=50,
                 rgbfunction="sandwich", rdpip="train_ours_full", bridge=bridge)
    return bridge.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--iterations", type=int)
    args = parser.parse_args()
    try:
        result = run(args)
        print(json.dumps({k: result[k] for k in ("status", "method", "segment_start", "iterations", "training_seconds", "formal_checkpoint")}, indent=2))
    except Exception:
        out = Path(args.out)
        if out.is_dir() and not (out / "complete.json").exists():
            write(out / "status.json", dict(status="failed", error=traceback.format_exc()))
        raise
