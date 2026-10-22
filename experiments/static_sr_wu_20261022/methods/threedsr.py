"""3DSR diffusion/scene alternation with a shared time-conditioned Wu4DGS.

This is an explicitly adapted and corrected implementation, not untouched static
3DSR. The author diffusion sampler remains unchanged. No model is loaded on import.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import sys

try:
    from .sequence_matters import sha256, source_objects, validate_records, verify_upstream
except ImportError:
    from sequence_matters import sha256, source_objects, validate_records, verify_upstream

UPSTREAM_COMMIT = "8c70a9f64301b4de255672e4aa7da42acc0001a2"
UPSTREAM_URL = "https://github.com/Consistent3DSR/3DSR"


def prepare_runtime_assets(asset_file, upstream, task_root):
    """Reuse only generic cache/workdir helpers; keep 3DSR's source and sampler."""
    try:
        from .splatsure_stablesr import validate_runtime_assets, prepare_workdir, verify_workdir
    except ImportError:
        from splatsure_stablesr import validate_runtime_assets, prepare_workdir, verify_workdir
    asset_file, upstream, task_root = (Path(path).resolve() for path in (asset_file, upstream, task_root))
    assets = json.loads(asset_file.read_text())
    # Cache registrations must remain unambiguous after the worker changes cwd.
    for item in assets.get("files", []) + assets.get("cache_mappings", []):
        if not Path(item["path"]).is_absolute():
            raise ValueError("Runtime model/cache paths must be absolute")
    validate_runtime_assets(assets)
    task_root.mkdir(parents=True, exist_ok=True)
    plan = {"runtime_assets": assets, "source_root": str(upstream), "output_root": str(task_root),
            "execution_workdir": str(task_root / "execution_workdir"),
            "runtime_assets_identity": {"path": str(asset_file), "sha256": sha256(asset_file)}}
    if Path(plan["execution_workdir"]).exists():
        verify_workdir(plan)
    else:
        prepare_workdir(plan)
        verify_workdir(plan)
    environment = os.environ.copy()
    environment.update(assets["cache_environment"])
    environment.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="1")
    environment["PYTHONPATH"] = str(upstream) + os.pathsep + str(upstream / "third_parties") + os.pathsep + environment.get("PYTHONPATH", "")
    return plan, environment


def threedsr_loss(rendered_hr, teacher_hr, observed_lr, ssim_fn, lambda_lr=1.0, lambda_dssim=0.2, lr_operator="bicubic"):
    import torch.nn.functional as F
    if rendered_hr.shape != teacher_hr.shape:
        raise ValueError("Teacher and render size differ")
    if tuple(rendered_hr.shape[-2:]) != tuple(4 * x for x in observed_lr.shape[-2:]):
        raise ValueError("Pinned 3DSR protocol requires x4")
    if lr_operator not in ("bicubic", "area"):
        raise ValueError("LR operator must be explicit: native bicubic or area protocol adaptation")
    kwargs = {"align_corners": False, "antialias": True} if lr_operator == "bicubic" else {}
    reduced = F.interpolate(rendered_hr.unsqueeze(0), size=observed_lr.shape[-2:], mode=lr_operator, **kwargs).squeeze(0)
    def photo(left, right):
        return (1 - lambda_dssim) * (left - right).abs().mean() + lambda_dssim * (1 - ssim_fn(left, right))
    hr = photo(rendered_hr, teacher_hr.detach())
    lr = photo(reduced, observed_lr.detach())
    return hr + lambda_lr * lr, {"hr": hr.detach(), "lr": lr.detach()}


def run_alternating(keys, denoise_all, optimize_shared, render_training, iterations=5000, rounds=4):
    """Run serial diffusion -> shared Wu update -> train-view feedback barriers.

    denoise_all(round_index=int, rendered_paths=dict|None) -> {key: PNG}
    optimize_shared(targets=dict, iterations=int, round_index=int) -> checkpoint
    render_training(keys=list, round_index=int) -> {key: rendered PNG}

    optimize_shared MUST retain one Wu model and optimizer across times and rounds.
    Its observations are (camera,time), not camera-only static poses. The caller
    is responsible for resource admission and fixed evaluation-camera exclusion.
    """
    if rounds != 4 or iterations <= 0 or not keys or len(set(keys)) != len(keys):
        raise ValueError("Pinned 3DSR uses four rounds and unique observation keys")
    expected = set(keys)
    rendered = None
    receipts = []
    for round_index in range(rounds):
        targets = denoise_all(round_index=round_index, rendered_paths=rendered)
        if set(targets) != expected or not all(Path(p).is_file() for p in targets.values()):
            raise RuntimeError("Incomplete diffusion targets; shared optimization was not started")
        checkpoint = optimize_shared(targets=targets, iterations=iterations, round_index=round_index)
        if checkpoint is None:
            raise RuntimeError("Shared Wu optimization returned no checkpoint")
        receipts.append({"round_index": round_index, "checkpoint": checkpoint if isinstance(checkpoint, dict) else str(checkpoint),
                         "target_count": len(targets)})
        if round_index < rounds - 1:
            rendered = render_training(keys=list(keys), round_index=round_index)
            if set(rendered) != expected or not all(Path(p).is_file() for p in rendered.values()):
                raise RuntimeError("Missing Wu feedback render; independent diffusion cannot substitute for 3DSR")
    return receipts


def require_round_state(round_index, previous_state, observation_key, input_hash):
    """Validate separate noisy latent state for each observation and each round."""
    if round_index not in range(4):
        raise ValueError("Round must be 0, 1, 2, or 3")
    if round_index == 0:
        if previous_state is not None:
            raise ValueError("Round zero cannot overwrite prior latent state")
        return
    if previous_state is None:
        raise ValueError("Missing previous latent state")
    if previous_state["key"] != observation_key or previous_state["lr_sha256"] != input_hash:
        raise ValueError("Cross-observation or changed-input latent state rejected")
    if previous_state["round_index"] != round_index - 1:
        raise ValueError("Diffusion rounds must execute exactly once in sequence")


class StableSRTurboBridge:
    """Actual pinned StableSR-Turbo sampler; run in a teacher-only process.

    State is indexed by scene/camera/time key AND patch. This repairs the author's
    shared local x_T variable and missing x_T_1 write-back, following paper Alg.1.
    A second repair applies identical preconditioning to Wu feedback, including
    minimum-size enlargement and padding, then crops before x4 output resizing.
    """
    def __init__(self, upstream, stable_checkpoint, vqgan_checkpoint, state_root, device="cuda", seed=42):
        for checkpoint in (stable_checkpoint, vqgan_checkpoint):
            if not Path(checkpoint).is_file():
                raise FileNotFoundError(f"Required author weight missing: {checkpoint}")
        self.upstream = verify_upstream(upstream, UPSTREAM_COMMIT)
        self.state_root = Path(state_root).resolve()
        self.state_root.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.seed = seed
        import numpy as np
        import torch
        import torch.nn.functional as F
        from omegaconf import OmegaConf
        if not device.startswith("cuda"):
            raise ValueError("Author sampler contains CUDA-specific operations; CPU teacher inference is not supported")
        sys.path.insert(0, str(self.upstream / "third_parties"))
        sys.path.insert(0, str(self.upstream))
        def instantiate(config):
            module, class_name = config.target.rsplit(".", 1)
            return getattr(importlib.import_module(module), class_name)(**config.get("params", {}))
        def load(config_path, checkpoint):
            config = OmegaConf.load(config_path)
            model = instantiate(config.model)
            state = torch.load(checkpoint, map_location="cpu", weights_only=False)["state_dict"]
            missing, unexpected = model.load_state_dict(state, strict=False)
            self.weight_loads.append({"checkpoint": str(Path(checkpoint).resolve()), "sha256": sha256(checkpoint),
                                      "missing_keys": list(missing), "unexpected_keys": list(unexpected)})
            parameter_names = set(dict(model.named_parameters()))
            missing_parameters = parameter_names.intersection(missing)
            if missing_parameters:
                raise ValueError(f"Checkpoint leaves parameters uninitialized: {sorted(missing_parameters)[:12]}")
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            return model.eval().to(device)
        self.weight_loads = []
        self.model = load(self.upstream / "configs/stableSRNew/v2-finetune_text_T_512.yaml", stable_checkpoint)
        self.vq_model = load(self.upstream / "configs/autoencoder/autoencoder_kl_64x64x4_resi.yaml", vqgan_checkpoint)
        self.vq_model.decoder.fusion_w = 0.5
        self.model.register_schedule(given_betas=None, beta_schedule="linear", timesteps=1000,
                                     linear_start=0.00085, linear_end=0.0120, cosine_s=8e-3)
        self.original_sqrt_alpha = self.model.sqrt_alphas_cumprod.clone()
        self.original_sqrt_one_minus_alpha = self.model.sqrt_one_minus_alphas_cumprod.clone()
        # Exact four-step spacing from upstream space_timesteps(1000, [4]).
        self.timesteps = [0, 333, 666, 999]
        new_betas = []
        last_alpha = 1.0
        for index in self.timesteps:
            alpha = self.model.alphas_cumprod[index]
            new_betas.append(float(1 - alpha / last_alpha))
            last_alpha = alpha
        self.model.register_schedule(given_betas=np.asarray(new_betas), timesteps=4)
        self.model.num_timesteps = 1000
        self.model.ori_timesteps = self.timesteps
        self.model.to(device)
        namespace = source_objects(self.upstream / "utils/util_image.py", ["ImageSpliterTh"], {"torch": torch})
        self.splitter = namespace["ImageSpliterTh"]
        colors = source_objects(self.upstream / "utils/wavelet_color_fix.py",
                                ["wavelet_blur", "wavelet_decomposition", "wavelet_reconstruction"],
                                {"torch": torch, "F": F, "Tensor": torch.Tensor})
        self.colorfix = colors["wavelet_reconstruction"]

    def generate_targets(self, manifest, round_index, output, rendered_paths=None):
        import hashlib
        import numpy as np
        import torch
        import torch.nn.functional as F
        from PIL import Image
        records = manifest["records"]
        validate_records(records, manifest["heldout_cameras"])
        if round_index not in range(4):
            raise ValueError("Expected one of four diffusion rounds")
        keys = {record["key"] for record in records}
        if round_index and (rendered_paths is None or set(rendered_paths) != keys):
            raise ValueError("Later diffusion rounds require all shared-Wu training renders")
        if not round_index and rendered_paths is not None:
            raise ValueError("Round zero starts with original LR; render feedback starts at round one")
        output = Path(output).resolve()
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"Use a fresh round directory: {output}")
        output.mkdir(parents=True, exist_ok=True)
        targets = {}
        receipt = {"status": "running", "round_index": round_index, "upstream_commit": UPSTREAM_COMMIT,
                   "weight_loads": self.weight_loads, "targets": targets,
                   "variant": "Wu4DGS adapted; per-observation-patch latent correction; shape correction",
                   "timesteps": self.timesteps, "records": []}
        if getattr(self, "runtime_assets_context", None) is not None:
            receipt["runtime_assets"] = self.runtime_assets_context
        def read(path):
            array = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255
            return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(self.device) * 2 - 1
        try:
            with torch.inference_mode(), self.model.ema_scope(), torch.autocast("cuda"):
                for record in records:
                    key = record["key"]
                    state_path = self.state_root / f"{key}.pt"
                    state_path.parent.mkdir(parents=True, exist_ok=True)
                    input_hash = sha256(record["lr_path"])
                    if "lr_sha256" in record and input_hash != record["lr_sha256"]:
                        raise ValueError(f"Observed LR identity changed: {key}")
                    previous = torch.load(state_path, map_location="cpu", weights_only=False) if state_path.exists() else None
                    require_round_state(round_index, previous, key, input_hash)
                    model_identity = [entry["sha256"] for entry in self.weight_loads]
                    if previous is not None and (previous.get("model_identity") != model_identity
                                                or previous.get("upstream_commit") != UPSTREAM_COMMIT
                                                or previous.get("scene") != record["scene"]):
                        raise ValueError("Scene, diffusion source, or checkpoint identity changed between rounds")
                    # Deterministic stream independent of the camera ordering; differs from author's global RNG.
                    per_key_seed = (self.seed + int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)) % (2**31)
                    torch.manual_seed(per_key_seed + round_index)
                    torch.cuda.manual_seed_all(per_key_seed + round_index)
                    lr = read(record["lr_path"])
                    target_shape = (lr.shape[-2] * 4, lr.shape[-1] * 4)
                    ratio = max(512 / min(lr.shape[-2:]), 4)
                    working_shape = (int(lr.shape[-2] * ratio), int(lr.shape[-1] * ratio))
                    conditioned = F.interpolate(lr, size=working_shape, mode="bicubic", align_corners=False).clamp(-1, 1)
                    pad = (0, (-working_shape[1]) % 32, 0, (-working_shape[0]) % 32)
                    conditioned = F.pad(conditioned, pad, mode="reflect")
                    feedback = None
                    if round_index:
                        feedback = read(rendered_paths[key])
                        if tuple(feedback.shape[-2:]) != target_shape:
                            raise ValueError(f"Wrong Wu feedback render size for {key}")
                        feedback = F.interpolate(feedback, size=working_shape, mode="bicubic", align_corners=False).clamp(-1, 1)
                        feedback = F.pad(feedback, pad, mode="reflect")
                    tiled = conditioned.shape[-2] > 1280 or conditioned.shape[-1] > 1280
                    splitter = self.splitter(conditioned, 1280, 1000, sf=1) if tiled else None
                    patches = splitter if tiled else [(conditioned, (0, conditioned.shape[-2], 0, conditioned.shape[-1]))]
                    patch_states = {}
                    iteration = 3 - round_index
                    for patch_index, (patch, region) in enumerate(patches):
                        patch_key = str(patch_index)
                        init_latent = self.model.get_first_stage_encoding(self.model.encode_first_stage(patch))
                        semantic = self.model.cond_stage_model([""] * 2)
                        common = dict(cond=semantic, struct_cond=init_latent, batch_size=1, timesteps=4, time_replace=4,
                                      tile_size=64, tile_overlap=32, batch_size_sample=2)
                        if round_index == 0:
                            noisy = self.model.q_sample_respace(x_start=init_latent,
                                t=torch.full((1,), 999, device=self.device, dtype=torch.long),
                                sqrt_alphas_cumprod=self.original_sqrt_alpha,
                                sqrt_one_minus_alphas_cumprod=self.original_sqrt_one_minus_alpha,
                                noise=torch.randn_like(init_latent))
                        else:
                            old_patch = previous["patches"][patch_key]
                            if tuple(old_patch["region"]) != tuple(region):
                                raise ValueError("Patch identity changed between diffusion rounds")
                            top, bottom, left, right = region
                            clean_render = self.model.get_first_stage_encoding(
                                self.model.encode_first_stage(feedback[:, :, top:bottom, left:right]))
                            noisy = self.model.sample_canvas_one_iter(iteration=iteration + 1,
                                x_T=old_patch["x_t"].to(self.device), return_x0=False, x0_input=clean_render, **common)
                        _, predicted_clean = self.model.sample_canvas_one_iter(iteration=iteration, x_T=noisy,
                                                                                return_x0=True, **common)
                        _, low_quality_features = self.vq_model.encode(patch)
                        decoded = self.vq_model.decode(predicted_clean / self.model.scale_factor, low_quality_features)
                        decoded = self.colorfix(decoded, patch)
                        patch_states[patch_key] = {"region": list(region), "x_t": noisy.detach().cpu()}
                        if tiled:
                            splitter.update_gaussian(decoded, region)
                        else:
                            assembled = decoded
                    if tiled:
                        assembled = splitter.gather()
                    # Remove padding BEFORE mapping the minimum-side enlargement to the actual x4 resolution.
                    assembled = ((assembled[:, :, :working_shape[0], :working_shape[1]] + 1) / 2).clamp(0, 1)
                    if working_shape != target_shape:
                        assembled = F.interpolate(assembled, size=target_shape, mode="bicubic", align_corners=False).clamp(0, 1)
                    target = output / f"{key}.png"
                    target.parent.mkdir(parents=True, exist_ok=True)
                    rgb = (assembled[0].float().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    Image.fromarray(rgb).save(target)
                    state = {"key": key, "lr_sha256": input_hash, "round_index": round_index,
                             "patches": patch_states, "upstream_commit": UPSTREAM_COMMIT,
                             "model_identity": model_identity, "scene": record["scene"]}
                    temp_state = state_path.with_suffix(".partial")
                    torch.save(state, temp_state)
                    temp_state.replace(state_path)
                    targets[key] = str(target)
                    receipt["records"].append({"key": key, "scene": record["scene"], "camera": record["camera"],
                                                "time": record["time"], "lr_sha256": input_hash,
                                                "render_sha256": sha256(rendered_paths[key]) if round_index else None,
                                                "target_sha256": sha256(target), "patch_count": len(patch_states),
                                                "seed": per_key_seed + round_index})
            receipt["status"] = "diffusion_round_complete"
        except Exception as error:
            receipt["status"] = "failed"
            receipt["error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            (output / "receipt.json").write_text(json.dumps(receipt, indent=2))
        return targets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--stable-checkpoint", required=True)
    parser.add_argument("--vqgan-checkpoint", required=True)
    parser.add_argument("--state-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--round-index", type=int, required=True)
    parser.add_argument("--rendered-paths", help="JSON key -> preceding-round shared-Wu render path")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--runtime-assets", help="Verified OpenCLIP/VGG16/LPIPS cache registration JSON")
    args = parser.parse_args()
    # Resolve every input before entering the isolated teacher working directory.
    for name in ("manifest", "upstream", "stable_checkpoint", "vqgan_checkpoint", "state_root", "output",
                 "rendered_paths", "runtime_assets"):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, str(Path(value).resolve()))
    manifest = json.loads(Path(args.manifest).read_text())
    validate_records(manifest["records"], manifest["heldout_cameras"])
    for record in manifest["records"]:
        record["lr_path"] = str(Path(record["lr_path"]).resolve())
    rendered = json.loads(Path(args.rendered_paths).read_text()) if args.rendered_paths else None
    if rendered is not None:
        rendered = {key: str(Path(path).resolve()) for key, path in rendered.items()}
    runtime = None
    if args.runtime_assets:
        output = Path(args.output)
        runtime, environment = prepare_runtime_assets(args.runtime_assets, args.upstream,
            output.parent / (output.name + "_runtime"))
        os.environ.update(environment)
        os.chdir(runtime["execution_workdir"])
    bridge = StableSRTurboBridge(args.upstream, args.stable_checkpoint, args.vqgan_checkpoint, args.state_root, args.device)
    if runtime is not None:
        bridge.runtime_assets_context = {"registration": runtime["runtime_assets_identity"],
            "execution_workdir": runtime["execution_workdir"], "offline_cache": True}
    targets = bridge.generate_targets(manifest, args.round_index, args.output, rendered)
    print(json.dumps({"status": "diffusion_round_complete", "targets": len(targets)}))


if __name__ == "__main__":
    main()
