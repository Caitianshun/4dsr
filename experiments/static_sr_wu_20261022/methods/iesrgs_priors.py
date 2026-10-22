"""Train-only IE-SRGS prior producers; explicit pinned models, no fallback.

Public functions prepare_external(config, ...), prepare_internal(backend, ...),
and merge_priors(...) write the common hashed prior bundle format. They do not
start training or choose a GPU. Call only after resource admission.
"""
from __future__ import annotations

import importlib.util
import hashlib
import json
import subprocess
import sys
from contextlib import contextmanager
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .iesrgs import UPSTREAM_COMMIT, IESRGSPriors
from .iesrgs_renderer import compute_3d_filter, render_state

try:
    from ..common import checked, identity, load_prior_array, read, resolve, sha, training_manifest, write_new
except ImportError:
    from common import checked, identity, load_prior_array, read, resolve, sha, training_manifest, write_new

CACHE_FORMATS = ('npy', 'npz_deflate')


def _upstream(root):
    root = resolve(root)
    head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if head != UPSTREAM_COMMIT:
        raise ValueError(f"IE-SRGS source commit changed: {head}")
    if subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"], text=True).strip():
        raise ValueError("IE-SRGS source checkout must be clean")
    return root


@contextmanager
def _no_bytecode():
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        yield
    finally:
        sys.dont_write_bytecode = previous


def _validate_cache_format(cache_format):
    if cache_format not in CACHE_FORMATS:
        raise ValueError(f'Unsupported IE cache format: {cache_format}')


def _save_array(root, name, value, cache_format='npy'):
    """Store the existing float32 payload exactly, one image/field per file."""
    _validate_cache_format(cache_format)
    p = (Path(root) / name).with_suffix('.npy' if cache_format == 'npy' else '.npz')
    array = np.ascontiguousarray(value.detach().cpu().numpy().astype(np.float32, copy=False))
    if not np.isfinite(array).all():
        raise ValueError('Nonfinite IE prior array')
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("xb") as handle:
        if cache_format == 'npy':
            np.save(handle, array, allow_pickle=False)
        else:
            np.savez_compressed(handle, array=array)
    item = identity(p)
    item['storage'] = dict(format=cache_format, dtype=str(array.dtype), shape=list(array.shape),
                           array_order='C', array_sha256=hashlib.sha256(array.tobytes(order='C')).hexdigest())
    return item


def _fresh_output(path):
    root = resolve(path)
    root.mkdir(parents=True, exist_ok=False)
    return root


def _base_index(config, manifest, model_identities, kind, cache_format='npy'):
    return dict(schema="static_sr_wu_prior_bundle_v1", manifest=config["manifest"],
                scene=manifest["scene"], training_only=True,
                producer=[identity(__file__), identity(Path(__file__).with_name("iesrgs.py"))],
                model_identities=model_identities, kind=kind, cache_format=cache_format,
                upstream_commit=UPSTREAM_COMMIT, entries=[])


def _entry(row):
    return dict(camera=row["camera_id"], frame=row["frame_index"], time=row["time"],
                lr_sha256=row["lr_sha256"])


def load_swinir_model(upstream_root, swinir_checkpoint, device, dependency_root=None):
    """Strict official SwinIR-M loading with an explicit local timm dependency."""
    root = _upstream(upstream_root)
    swin_path = checked(swinir_checkpoint)
    if swin_path.name != "001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth":
        raise ValueError("IE-SRGS requires named classical DF2K SwinIR-M x4 weights")
    vendor = resolve(dependency_root) if dependency_root else Path(__file__).resolve().parents[2] / "dynamic_sr_20260918/vendor"
    if not (vendor / "timm/__init__.py").is_file():
        raise FileNotFoundError(f"explicit timm dependency missing: {vendor}")
    sys.path.insert(0, str(vendor))
    try:
        with _no_bytecode():
            import timm
        if not Path(timm.__file__).resolve().is_relative_to(vendor):
            raise RuntimeError("another timm dependency is already imported; use a clean producer process")
    finally:
        sys.path.remove(str(vendor))
    spec = importlib.util.spec_from_file_location("iesrgs_source_swinir", root / "submodules/swinir/network_swinir.py")
    module = importlib.util.module_from_spec(spec)
    with _no_bytecode():
        spec.loader.exec_module(module)
    swinir = module.SwinIR(upscale=4, in_chans=3, img_size=64, window_size=8,
                          img_range=1., depths=[6] * 6, embed_dim=180, num_heads=[6] * 6,
                          mlp_ratio=2, upsampler="pixelshuffle", resi_connection="1conv")
    payload = torch.load(swin_path, map_location="cpu", weights_only=True)
    swinir.load_state_dict(payload.get("params", payload), strict=True)
    return swinir.eval().to(device)


def load_external_models(upstream_root, swinir_checkpoint, depth_checkpoint, device,
                         dependency_root=None):
    """Exact released SwinIR-M DF2K x4 and Depth Anything V2-Large architectures."""
    root = _upstream(upstream_root)
    depth_path = checked(depth_checkpoint)
    if depth_path.name != "depth_anything_v2_vitl.pth":
        raise ValueError("IE-SRGS requires Depth Anything V2-Large (vitl) weights")
    swinir = load_swinir_model(root, swinir_checkpoint, device, dependency_root)
    source = str(root / "submodules/Depth-Anything-V2")
    sys.path.insert(0, source)
    try:
        with _no_bytecode():
            from depth_anything_v2.dpt import DepthAnythingV2
        imported = Path(sys.modules["depth_anything_v2.dpt"].__file__).resolve()
        if not imported.is_relative_to(Path(source).resolve()):
            raise RuntimeError("another Depth Anything source is already imported; use a clean producer process")
    finally:
        sys.path.remove(source)
    depth = DepthAnythingV2(encoder="vitl", features=256, out_channels=[256, 512, 1024, 1024])
    depth.load_state_dict(torch.load(depth_path, map_location="cpu", weights_only=True), strict=True)
    return swinir.eval().to(device), depth.eval().to(device)


@torch.no_grad()
def external_tensors(lr, swinir, depth_model):
    """Released mirror-padding, float SR, and quantized monocular depth path.

    The source passes the truncated uint8 SR PNG to Depth Anything, normalizes
    its prediction to uint8, and reads it /65536 for real scenes. This function
    reproduces those transformations in memory, using input_size=518.
    """
    if lr.ndim != 3 or lr.shape[0] != 3 or min(lr.shape[-2:]) < 8:
        raise ValueError("SwinIR source producer requires RGB LR with H,W >= 8")
    if not torch.isfinite(lr).all() or lr.min() < 0 or lr.max() > 1:
        raise ValueError("LR RGB must be finite and in [0,1]")
    h, w = lr.shape[-2:]
    x = lr[None]
    hp, wp = (h // 8 + 1) * 8 - h, (w // 8 + 1) * 8 - w
    x = torch.cat([x, x.flip(2)], 2)[:, :, :h + hp]
    x = torch.cat([x, x.flip(3)], 3)[:, :, :, :w + wp]
    sr = swinir(x)[0, :, :h * 4, :w * 4].clamp(0, 1)
    if tuple(sr.shape) != (3, h * 4, w * 4):
        raise ValueError("SwinIR x4 output shape mismatch")
    rgb_u8 = (sr.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    inverse = np.asarray(depth_model.infer_image(rgb_u8[:, :, ::-1].copy(), 518))
    if inverse.shape != (h * 4, w * 4) or not np.isfinite(inverse).all():
        raise ValueError("invalid Depth Anything output")
    low, high = float(inverse.min()), float(inverse.max())
    if high <= low:
        raise ValueError("constant monocular depth cannot provide Pearson guidance")
    inverse_u8 = ((inverse - low) / (high - low) * 255).astype(np.uint8)
    return sr.cpu(), torch.from_numpy(inverse_u8.astype(np.float32) / 65536.)[None]


def prepare_external(config, upstream_root, swinir_checkpoint, depth_checkpoint,
                     output_dir, device, dependency_root=None, cache_format='npy'):
    """Generate only the exact manifest's legal training-camera priors."""
    _validate_cache_format(cache_format)
    manifest, rows = training_manifest(checked(config["manifest"]), verify_pixels=True)
    source = _upstream(upstream_root)
    swinir, depth = load_external_models(source, swinir_checkpoint, depth_checkpoint, device, dependency_root)
    root = _fresh_output(output_dir)
    index = _base_index(config, manifest, [swinir_checkpoint, depth_checkpoint], "iesrgs_external", cache_format)
    index["producer"] += [identity(source / "resolution.py"), identity(source / "submodules/Depth-Anything-V2/run.py")]
    index["producer"].append(identity(sys.modules["timm"].__file__))
    index["external_models"] = dict(swinir="classical_DF2K_SwinIR-M_x4", depth="DepthAnythingV2_vitl",
                                    depth_input="uint8_truncated_SR_RGB", depth_representation="inverse_relative_uint8_div65536")
    for row in rows:
        path = Path(manifest["_root"]) / row["lr_path"]
        lr = torch.from_numpy(np.array(Image.open(path).convert("RGB"), copy=True)).permute(2, 0, 1).float() / 255
        sr, inverse = external_tensors(lr.to(device), swinir, depth)
        key = f"{row['camera_id']}/{row['frame_index']:04d}"
        item = _entry(row)
        item["external_rgb"] = _save_array(root, key + "_external_rgb.npy", sr, cache_format)
        item["external_inverse_depth"] = _save_array(root, key + "_external_inverse_depth.npy", inverse, cache_format)
        index["entries"].append(item)
    write_new(root / "index.json", index)
    return root / "index.json"


@torch.no_grad()
def prepare_internal(backend, checkpoint_identity, output_dir, require_stage="iesrgs_internal", cache_format='npy'):
    """Freeze a trained internal Wu+Mip teacher and render HR at training views.

    LR cameras determine the filter; HR cameras determine image rasterization.
    Nothing is rendered or loaded from the held-out camera, even if its poses
    are available in the surrounding dataset manifest.
    """
    _validate_cache_format(cache_format)
    checkpoint = checked(checkpoint_identity)
    if getattr(backend, "loaded_checkpoint_identity", None) != identity(checkpoint):
        raise ValueError("backend must be freshly loaded from the indexed internal checkpoint")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("metadata", {}).get("stage") != require_stage:
        raise ValueError("internal priors require an IE-SRGS internal-stage checkpoint")
    if (payload["metadata"].get("iterations") != 30000 or
            payload["metadata"].get("optimizer_updates") != 29999):
        raise ValueError("internal priors require the completed 30000-iteration/29999-update teacher; wiring fixtures are not teachers")
    if payload.get("adaptation_config", {}).get("manifest") != backend.config["manifest"]:
        raise ValueError("internal teacher checkpoint scene/protocol mismatch")
    del payload
    manifest, rows = training_manifest(checked(backend.config["manifest"]), verify_pixels=True)
    root = _fresh_output(output_dir)
    index = _base_index(backend.config, manifest, [checkpoint_identity], "iesrgs_internal", cache_format)
    index["producer"].append(identity(Path(__file__).with_name("iesrgs_renderer.py")))
    index["internal_teacher"] = dict(representation="Wu_4DGS_with_Mip_filter", checkpoint=checkpoint_identity,
                                    depth="RADE_median_depth", filter_reference_resolution="lr",
                                    frozen=True, same_time_training_views_only=True)
    groups = defaultdict(list)
    for row in rows:
        groups[row["frame_index"]].append(row)
    for frame in sorted(groups):
        frame_rows = groups[frame]
        state = backend.effective_state(frame_rows[0]["time"])
        cameras_lr = [backend.camera(row, "lr") for row in frame_rows]
        filt = compute_3d_filter(state["xyz"], cameras_lr)
        for row in frame_rows:
            packet = render_state(backend.camera(row, "hr"), state, backend.background,
                                  backend.g.active_sh_degree, cameras_lr, filter_3d=filt)
            key = f"{row['camera_id']}/{row['frame_index']:04d}"
            item = _entry(row)
            item["internal_rgb"] = _save_array(root, key + "_internal_rgb.npy", packet["render"], cache_format)
            item["internal_depth"] = _save_array(root, key + "_internal_depth.npy", packet["median_depth"], cache_format)
            index["entries"].append(item)
    write_new(root / "index.json", index)
    return root / "index.json"


def merge_priors(external_index, internal_index, output_index):
    """Join by exact camera/frame and LR identity, never positional list order."""
    external, internal = read(external_index), read(internal_index)
    if external.get("kind") != "iesrgs_external" or internal.get("kind") != "iesrgs_internal":
        raise ValueError("expected distinct external and internal IE-SRGS bundles")
    for field in ("manifest", "scene", "training_only", "upstream_commit"):
        if external.get(field) != internal.get(field):
            raise ValueError(f"IE-SRGS prior sources disagree on {field}")
    if external["training_only"] is not True:
        raise ValueError("only training-camera priors are legal")
    key = lambda row: (row["camera"], row["frame"])
    left = {key(row): row for row in external["entries"]}
    right = {key(row): row for row in internal["entries"]}
    if set(left) != set(right) or len(left) != len(external["entries"]) or len(right) != len(internal["entries"]):
        raise ValueError("prior keys are missing or duplicated")
    merged = dict(external)
    merged["kind"] = "iesrgs_complete"
    merged['cache_formats'] = {'external': external.get('cache_format', 'npy'),
                               'internal': internal.get('cache_format', 'npy')}
    merged.pop('cache_format', None)  # Fields can legitimately use different lossless containers.
    merged["internal_teacher"] = internal["internal_teacher"]
    merged["producer"] = external["producer"] + internal["producer"]
    merged["model_identities"] = external["model_identities"] + internal["model_identities"]
    merged["entries"] = []
    for k in sorted(left):
        a, b = left[k], right[k]
        if a["lr_sha256"] != b["lr_sha256"] or a["time"] != b["time"]:
            raise ValueError("internal/external guidance refers to different observations")
        merged["entries"].append({**a, "internal_rgb": b["internal_rgb"], "internal_depth": b["internal_depth"]})
    for item in merged["producer"] + merged["model_identities"]:
        checked(item)
    write_new(output_index, merged)
    return resolve(output_index)


def load_priors(entry, device):
    fields = ("internal_rgb", "internal_depth", "external_rgb", "external_inverse_depth")
    return IESRGSPriors(**{key: torch.from_numpy(load_prior_array(entry[key], require_float32=True)).to(device)
                           for key in fields})
