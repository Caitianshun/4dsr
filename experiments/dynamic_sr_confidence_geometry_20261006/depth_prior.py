"""Legal LR depth priors and position-only auxiliary Gaussian moments.

The historical bicubic moment renderer is deliberately left unchanged. This
adapter renders unclamped [1,z,z²] at HR and averages the *moments* with area
weights before normalization. Its covariance and opacity inputs are detached;
the very same effective_xyz tensor used by RGB must be supplied by the caller.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
DAV2_WEIGHT_SHA256 = "715fade13be8f229f8a70cc02066f656f2423a59effd0579197bbf57860e1378"
NUMERICS = dict(epsilon=1e-6, alpha_support_min=1e-3,
                robust_scale_relative_floor=1e-3, robust_scale_absolute_floor=1e-6,
                tie_band=.05, hinge_margin=.1, max_pairs=4096,
                min_distance_lr=4., max_distance_lr=80., pair_seed=2026100603)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.replace(path)


def normalize_moments(moments, lr_size=None, epsilon=1e-6):
    """Reduce raw HR moments with nonnegative area weights, then divide.

    Raw variance is retained to detect numerical cancellation. Its clipped copy
    is only a display diagnostic and never enters the geometry ranking loss.
    """
    if moments.ndim != 3 or moments.shape[0] != 3:
        raise ValueError("Expected unclamped moment tensor of shape (3,H,W)")
    if lr_size is not None:
        if lr_size[0] > moments.shape[-2] or lr_size[1] > moments.shape[-1]:
            raise ValueError("Moment aggregation cannot upsample")
        moments = F.interpolate(moments[None], size=lr_size, mode="area")[0]
    alpha = moments[0]
    denominator = alpha.clamp_min(epsilon)
    z = moments[1] / denominator
    variance = moments[2] / denominator - z.square()
    # Current alpha is not a loss validity mask; frozen parent support is.
    inverse_depth = 1. / (z.clamp_min(0.) + epsilon)
    return dict(moments=moments, alpha=alpha, expected_z=z,
                inverse_depth=inverse_depth, variance_z_raw=variance,
                variance_z=variance.clamp_min(0.))


def packed_covariance(covariance):
    if covariance.ndim == 2 and covariance.shape[-1] == 6:
        return covariance.contiguous()
    if covariance.ndim != 3 or covariance.shape[-2:] != (3, 3):
        raise ValueError("Expected covariance (N,3,3) or packed (N,6)")
    return torch.stack((covariance[:, 0, 0], covariance[:, 0, 1],
                        covariance[:, 0, 2], covariance[:, 1, 1],
                        covariance[:, 1, 2], covariance[:, 2, 2]), -1).contiguous()


def render_moments(camera, effective_xyz, cov, opacity, lr_size=None):
    """Rasterize a separate G branch, retaining only effective_xyz gradients.

    covariance is detached after all scale/rotation/deformation computations.
    Camera-z and the rasterizer's projection receive live effective_xyz; this
    also keeps the direct color derivative of z and z² in the autograd graph.
    """
    from diff_gaussian_rasterization import (GaussianRasterizationSettings,
                                             GaussianRasterizer)
    xyz = effective_xyz
    view = camera.world_view_transform.to(device=xyz.device, dtype=xyz.dtype)
    z = (torch.cat((xyz, torch.ones_like(xyz[:, :1])), -1) @ view)[:, 2]
    colors = torch.stack((torch.ones_like(z), z, z.square()), -1).contiguous()
    settings = GaussianRasterizationSettings(
        image_height=int(camera.image_height), image_width=int(camera.image_width),
        tanfovx=math.tan(camera.FoVx * .5), tanfovy=math.tan(camera.FoVy * .5),
        bg=torch.zeros(3, device=xyz.device, dtype=xyz.dtype), scale_modifier=1.,
        viewmatrix=view, projmatrix=camera.full_proj_transform.to(xyz.device),
        sh_degree=0, campos=camera.camera_center.to(xyz.device),
        prefiltered=False, debug=False)
    image, radii, native = GaussianRasterizer(raster_settings=settings)(
        means3D=xyz, means2D=torch.zeros_like(xyz), shs=None,
        colors_precomp=colors, opacities=opacity.detach().contiguous(),
        scales=None, rotations=None, cov3D_precomp=packed_covariance(cov.detach()))
    result = normalize_moments(image, lr_size)
    result.update(hr_moments=image, radii=radii, native_depth=native,
                  axial_z=z, effective_xyz=xyz)
    return result


def robust_normalize(values, support=None):
    """Per-frame median/IQR normalization with *detached* statistics."""
    finite = torch.isfinite(values.detach())
    if support is not None:
        finite = finite & support.detach().bool()
    valid_values = values.detach()[finite]
    if not valid_values.numel():
        raise ValueError("No finite supported relative-depth values")
    # Match the registered median definition for both teacher and rendering.
    quartiles = torch.quantile(valid_values, torch.tensor(
        [.25, .5, .75], dtype=values.dtype, device=values.device))
    median = quartiles[1]
    iqr = quartiles[2] - quartiles[0]
    magnitude = torch.quantile(valid_values.abs(), .5)
    scale = torch.maximum(iqr, torch.maximum(1e-3 * magnitude,
                           magnitude.new_tensor(1e-6)))
    return (values - median) / scale, dict(median=median, iqr=iqr, scale=scale,
                                          supported_count=finite.sum())


def observation_seed(camera, frame, base_seed=2026100603):
    digest = hashlib.sha256(f"{base_seed}/{camera}/{int(frame)}".encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2 ** 63 - 1)


def sample_pairs(shape, seed, max_pairs=4096, min_distance=4., max_distance=80.):
    """Local generator; no consumption of training Python/NumPy/torch RNG.

    Half of candidate radii are 4–20 and half 20–80 LR pixels. Integer offsets
    are checked against the exact Euclidean distance after rounding. Finite
    rejection rounds only remove out-of-frame coordinates, never use quality.
    """
    height, width = map(int, shape)
    if height * width < 2 or min(height, width) < 2:
        raise ValueError("Too few pixels for depth point pairs")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    accepted = []
    count = 0
    for _ in range(50):
        remaining = max_pairs - count
        if remaining <= 0:
            break
        n = max(remaining * 2, 128)
        px = torch.randint(width, (n,), generator=generator)
        py = torch.randint(height, (n,), generator=generator)
        angle = torch.rand(n, generator=generator) * (2 * math.pi)
        choose_local = torch.rand(n, generator=generator) < .5
        split = min(20., max_distance)
        lo = torch.where(choose_local, min_distance, split)
        hi = torch.where(choose_local, split, max_distance)
        radius = lo + torch.rand(n, generator=generator) * (hi - lo)
        dx = (radius * angle.cos()).round().long()
        dy = (radius * angle.sin()).round().long()
        qx, qy = px + dx, py + dy
        distance = (dx.square() + dy.square()).float().sqrt()
        keep = ((qx >= 0) & (qx < width) & (qy >= 0) & (qy < height) &
                (distance >= min_distance) & (distance <= max_distance))
        selected = torch.stack((py[keep] * width + px[keep],
                                qy[keep] * width + qx[keep]), -1)[:remaining]
        accepted.append(selected)
        count += len(selected)
    if not count:
        raise ValueError("No legal 4–80 pixel depth pairs fit this image")
    return torch.cat(accepted, 0).contiguous()


@torch.no_grad()
def prepare_geometry_target(depth, variants, frozen_alpha, pairs=None, seed=2026100603,
                            sparse_support=None, tie_band=.05, alpha_min=1e-3):
    """Freeze teacher ordering, LR augmentation agreement and parent support.

    Pair stability is the fraction of available variants (including original)
    whose non-tied order agrees. Missing crop-border values do not vote. Parent
    alpha supplies a binary numerical-support mask, never current alpha. Sparse
    triangulation is optional and must be independent LR-only evidence; without
    it untextured surfaces retain the mild monocular ranking prior.
    """
    depth = torch.as_tensor(depth, dtype=torch.float32).detach().cpu()
    frozen_alpha = torch.as_tensor(frozen_alpha, dtype=torch.float32).detach().cpu()
    if depth.shape != frozen_alpha.shape or depth.ndim != 2:
        raise ValueError("Depth/parent alpha must share an LR (H,W) grid")
    support = torch.isfinite(depth) & torch.isfinite(frozen_alpha) & (frozen_alpha > alpha_min)
    teacher, stats = robust_normalize(depth, support)
    if pairs is None:
        pairs = sample_pairs(depth.shape, seed)
    pairs = torch.as_tensor(pairs, dtype=torch.long).detach().cpu()
    if pairs.ndim != 2 or pairs.shape[1] != 2 or len(pairs) > 4096:
        raise ValueError("Expected at most 4096 fixed depth index pairs")
    if pairs.numel() and (pairs.min() < 0 or pairs.max() >= depth.numel()):
        raise ValueError("Point-pair index outside LR grid")
    first, second = pairs.unbind(1)
    delta = teacher.flatten()[first] - teacher.flatten()[second]
    sign = delta.sign()
    valid_pair = support.flatten()[first] & support.flatten()[second] & (delta.abs() >= tie_band)
    agreements = torch.ones(len(pairs))
    available = torch.ones(len(pairs))
    for value in variants:
        value = torch.as_tensor(value, dtype=torch.float32).detach().cpu()
        if value.shape != depth.shape:
            raise ValueError("Augmented relative depth must restore original LR coordinates")
        if not torch.isfinite(value).any():
            continue
        normalized, _ = robust_normalize(value, support & torch.isfinite(value))
        vp, vq = normalized.flatten()[first], normalized.flatten()[second]
        has_vote = torch.isfinite(vp) & torch.isfinite(vq)
        vdelta = vp - vq
        agrees = (vdelta.sign() == sign) & (vdelta.abs() >= tie_band)
        agreements += (has_vote & agrees).float()
        available += has_vote.float()
    stability = agreements / available
    # Unsupported/tied pairs remain in the full-table denominator with weight 0.
    weights = valid_pair.float() * stability
    if sparse_support is not None:
        sparse_support = torch.as_tensor(sparse_support, dtype=torch.float32).cpu().clamp(0, 1)
        if sparse_support.shape != depth.shape:
            raise ValueError("Sparse support must share the LR grid")
        pair_sparse = torch.minimum(sparse_support.flatten()[first], sparse_support.flatten()[second])
        weights = weights * (.5 + .5 * pair_sparse)
    return dict(teacher=teacher, pairs=pairs, signs=sign, weights=weights,
                frozen_support=support, frozen_alpha=frozen_alpha, stability=stability,
                teacher_stats=stats, pair_seed=int(seed),
                registration=dict(tie_band=tie_band, alpha_support_min=alpha_min,
                    stability="agreeing non-tied available LR variants / available variants incl original",
                    sparse_support=sparse_support is not None, pairs_denominator="all registered pairs"))


def geometry_loss(moment_result, target):
    q = moment_result["inverse_depth"]
    support = target["frozen_support"].to(q.device)
    normalized, stats = robust_normalize(q, support)
    pairs = target["pairs"].to(q.device)
    signs = target["signs"].to(q.device)
    weights = target["weights"].to(q.device)
    first, second = pairs.unbind(1)
    difference = normalized.flatten()[first] - normalized.flatten()[second]
    # Teacher/sampling quantities cannot acquire gradients from current output.
    loss = (weights.detach() * F.relu(.1 - signs.detach() * difference)).mean()
    return loss, dict(registered_pairs=len(pairs), effective_pairs=(weights > 0).sum(),
                     mean_weight=weights.mean(), normalized_stats=stats,
                     active_pair_fraction=((.1 - signs * difference) > 0).float().mean())


class GeometryCache:
    """Hash-checked frozen targets with a bounded CPU cache."""
    def __init__(self, index, capacity=64):
        self.path = Path(index)
        self.index = json.loads(self.path.read_text())
        if self.index.get("privileged_train_hr", False):
            raise ValueError("HR-derived geometry target forbidden")
        self.entries = {(e["camera"], int(e["frame"])): e for e in self.index["entries"]}
        self.capacity = int(capacity)
        self.cache = OrderedDict()
        self.misses = 0

    def get(self, camera, frame, device=None):
        key = (str(camera), int(frame))
        if key not in self.cache:
            entry = self.entries[key]
            path = self.path.parent / entry["path"]
            if sha256(path) != entry["sha256"]:
                raise ValueError(f"Geometry cache checksum mismatch: {path}")
            with np.load(path, allow_pickle=False) as raw:
                self.cache[key] = {name: torch.from_numpy(raw[name].copy()) for name in
                                  ["teacher", "pairs", "signs", "weights", "frozen_support", "frozen_alpha", "stability"]}
            self.misses += 1
            while len(self.cache) > self.capacity:
                self.cache.popitem(last=False)
        value = self.cache.pop(key)
        self.cache[key] = value
        return value if device is None else {k: v.to(device) for k, v in value.items()}


def legal_observations(manifest):
    observations = sorted((o for o in manifest["observations"] if o["split"] == "train"),
                          key=lambda o: (o["camera_id"], o["frame_index"]))
    if any(o["camera_id"] not in manifest["splits"]["train"] for o in observations):
        raise ValueError("Non-training camera in legal observations")
    return observations


def legal_image_audit():
    def guard(event, args):
        if event == "open" and isinstance(args[0], (str, bytes, os.PathLike)):
            path = os.fsdecode(args[0]).replace("\\", "/")
            if path.endswith((".png", ".jpg", ".jpeg")) and "/hr/" in path:
                raise RuntimeError(f"HR image read forbidden in legal depth preparation: {path}")
    sys.addaudithook(guard)


def export_parent(args):
    legal_image_audit()
    sys.path.insert(0, str(ROOT / "experiments/dynamic_sr_prior_guidance_20260927"))
    from gradient_policy import effective_state
    from shared import evaluator, load_manifest, load_model
    torch.set_num_threads(4)
    manifest = load_manifest(args.manifest)
    observations = legal_observations(manifest)
    evaluator_module = evaluator()
    model = load_model(args.resume, manifest)
    if model.checkpoint["metadata"].get("intervention_step") != 6000:
        raise ValueError("Frozen support must come from the registered U6000 parent")
    h, w = reversed(manifest["resolutions"]["lr"])
    entries = []
    start = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for i, obs in enumerate(observations):
            camera = evaluator_module.render_camera(manifest, obs, i)
            state = effective_state(model, camera.time)
            result = render_moments(camera, state["xyz"], state["cov"], state["opacity"], (h, w))
            cpu = {key: value.detach().cpu().numpy().astype(np.float32) for key, value in
                   [("depth_lr", result["expected_z"]), ("alpha_lr", result["alpha"]),
                    ("raw_variance_lr", result["variance_z_raw"])]}
            path = args.out / f'{obs["camera_id"]}_{obs["frame_index"]:04d}.npz'
            np.savez_compressed(path, **cpu)
            calibration = manifest["cameras"][obs["camera_id"]]
            entry = dict(camera=obs["camera_id"], frame=obs["frame_index"], path=path.name,
                         sha256=sha256(path), K_lr=calibration["K_lr"], w2c=calibration["w2c"],
                         time=float(camera.time),
                         alpha_positive_fraction=float((result["alpha"] > 1e-3).float().mean()),
                         minimum_raw_variance=float(result["variance_z_raw"].min()))
            entries.append(entry)
            if (i + 1) % 100 == 0 or i + 1 == len(observations):
                print(json.dumps(dict(mode="parent_moments", done=i + 1, total=len(observations))), flush=True)
    torch.cuda.synchronize()
    write_json(args.out / "index.json", dict(status="completed", entries=entries,
        manifest_sha256=sha256(args.manifest), parent_sha256=sha256(args.resume),
        source_sha256=sha256(__file__), privileged_train_hr=False,
        definition="HR unclamped [A,M1,M2], nonnegative area to LR, Z=M1/max(A,1e-6)",
        depth_is_normalized_camera_z=True, moments_clamped=False, lr_size=[h, w],
        gpu=torch.cuda.get_device_name(), torch=str(torch.__version__),
        visible_cuda=os.environ.get("CUDA_VISIBLE_DEVICES"),
        seconds=time.monotonic() - start, peak_gb=torch.cuda.max_memory_allocated() / 1e9))


def prepare_depth(args):
    import cv2
    legal_image_audit()
    manifest = json.loads(args.manifest.read_text())
    root = args.manifest.parent
    observations = legal_observations(manifest)
    if sha256(args.weight) != DAV2_WEIGHT_SHA256:
        raise ValueError("Depth Anything V2 Small registered weight hash mismatch")
    sys.path.insert(0, str(args.model_source))
    from depth_anything_v2.dpt import DepthAnythingV2
    torch.set_num_threads(4)
    cv2.setNumThreads(4)
    model = DepthAnythingV2(encoder="vits", features=64, out_channels=[48, 96, 192, 384])
    model.load_state_dict(torch.load(args.weight, map_location="cpu", weights_only=True))
    model.cuda().eval()
    torch.cuda.reset_peak_memory_stats()
    reuse = {}
    if args.reuse is not None:
        reuse_index = json.loads((args.reuse / "complete.json").read_text())
        if reuse_index.get("kind") != "legal" or reuse_index.get("model_sha256") != DAV2_WEIGHT_SHA256:
            raise ValueError("Only registered LR-only diagnostic depth can be reused")
        if reuse_index["manifest_sha256"] != sha256(args.manifest):
            raise ValueError("Reused depth belongs to another manifest")
        reuse = {(e["camera"], int(e["frame"])): e for e in reuse_index["rows"]}
    h, w = reversed(manifest["resolutions"]["lr"])
    hh, ww = reversed(manifest["resolutions"]["hr"])
    grids = set()

    @torch.inference_mode()
    def infer(image, shape=(h, w)):
        tensor, _ = model.image2tensor(image, 518)
        grids.add(tuple(tensor.shape[-2:]))
        return F.interpolate(model(tensor)[:, None], size=shape, mode="bilinear",
                             align_corners=True)[0, 0].cpu().numpy().astype(np.float32)

    start = time.monotonic()
    entries = []
    for i, obs in enumerate(observations):
        source = root / obs["lr_path"]
        if sha256(source) != obs["lr_sha256"]:
            raise ValueError(f"Legal LR checksum mismatch: {source}")
        key = (obs["camera_id"], int(obs["frame_index"]))
        reused = key in reuse
        if reused:
            old = args.reuse / reuse[key]["path"]
            if sha256(old) != reuse[key]["sha256"] or obs["lr_sha256"] not in reuse[key]["inputs"].values():
                raise ValueError("Reused LR depth checksum/input mismatch")
            with np.load(old, allow_pickle=False) as raw:
                prediction = {name: raw[name].copy() for name in ["lr", "scale", "crop"]}
        else:
            image = cv2.imread(str(source))
            if image is None or image.shape[:2] != (h, w):
                raise ValueError("Invalid legal LR image shape")
            # Precisely retain the registered 2026-09-27 DAV2 preprocessor.
            prediction = dict(lr=infer(cv2.resize(image, (ww, hh), interpolation=cv2.INTER_CUBIC)))
            small = cv2.resize(image, (int(round(w * 323 / 336)), int(round(h * 242 / 252))), interpolation=cv2.INTER_AREA)
            scale = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
            prediction["scale"] = infer(cv2.resize(scale, (ww, hh), interpolation=cv2.INTER_CUBIC))
            crop = image[3:-3, 4:-4]
            prediction["crop"] = np.full((h, w), np.nan, np.float32)
            prediction["crop"][3:-3, 4:-4] = infer(cv2.resize(crop, (ww, hh), interpolation=cv2.INTER_CUBIC), crop.shape[:2])
        path = args.out / f'{key[0]}_{key[1]:04d}.npz'
        np.savez_compressed(path, **prediction)
        entries.append(dict(camera=key[0], frame=key[1], path=path.name, sha256=sha256(path),
                            inputs={str(source): obs["lr_sha256"]}, reused=reused,
                            original_source_sha256=reuse[key]["sha256"] if reused else None))
        if (i + 1) % 25 == 0 or i + 1 == len(observations):
            print(json.dumps(dict(mode="legal_depth", done=i + 1, total=len(observations))), flush=True)
    write_json(args.out / "index.json", dict(status="completed", kind="legal", entries=entries,
        manifest_sha256=sha256(args.manifest), model_sha256=sha256(args.weight),
        model_source=str(args.model_source), source_sha256=sha256(__file__),
        model_commit=subprocess.check_output(["git", "-C", str(args.model_source), "rev-parse", "HEAD"], text=True).strip(),
        model_sources=[dict(path=str(path), sha256=sha256(path)) for path in
                       [args.model_source / "depth_anything_v2/dpt.py", args.model_source / "depth_anything_v2/util/transform.py"]],
        privileged_train_hr=False, used_for_geometry_training=True,
        actual_network_grids=sorted(grids), raw_output_grid=[h, w], input_size=518,
        preprocess="Registered 2026-09-27 official DAV2 image2tensor; all LR variants first bicubic to HR canvas; raw depth bilinear to LR align_corners=True",
        augmentation="scale336x252->323x242 area->336x252 cubic; crop[3:-3,4:-4] restored with NaN border",
        output="relative inverse-depth-like score; no metric depth/HR scale fitting",
        reused_entries=sum(e["reused"] for e in entries),
        gpu=torch.cuda.get_device_name(), torch=str(torch.__version__),
        visible_cuda=os.environ.get("CUDA_VISIBLE_DEVICES"),
        peak_gb=torch.cuda.max_memory_allocated() / 1e9, seconds=time.monotonic() - start))


def prepare_targets(args):
    depth_index = json.loads(args.depth_index.read_text())
    parent_index = json.loads(args.parent_index.read_text())
    if depth_index.get("privileged_train_hr", False) or parent_index.get("privileged_train_hr", False):
        raise ValueError("HR-derived targets forbidden")
    if depth_index["manifest_sha256"] != parent_index["manifest_sha256"]:
        raise ValueError("Depth and parent moment manifests differ")
    parents = {(e["camera"], int(e["frame"])): e for e in parent_index["entries"]}
    if set(parents) != {(e["camera"], int(e["frame"])) for e in depth_index["entries"]}:
        raise ValueError("Depth and parent caches do not cover exactly the same observations")
    entries = []
    torch.set_num_threads(4)
    for e in depth_index["entries"]:
        key = (e["camera"], int(e["frame"]))
        pe = parents[key]
        dp = args.depth_index.parent / e["path"]
        pp = args.parent_index.parent / pe["path"]
        if sha256(dp) != e["sha256"] or sha256(pp) != pe["sha256"]:
            raise ValueError("Source cache checksum mismatch")
        with np.load(dp, allow_pickle=False) as d, np.load(pp, allow_pickle=False) as p:
            seed = observation_seed(*key)
            target = prepare_geometry_target(d["lr"], [d["scale"], d["crop"]], p["alpha_lr"], seed=seed)
        path = args.out / f"{key[0]}_{key[1]:04d}.npz"
        names = ["teacher", "pairs", "signs", "weights", "frozen_support", "frozen_alpha", "stability"]
        np.savez_compressed(path, **{name: target[name].numpy() for name in names})
        entries.append(dict(camera=key[0], frame=key[1], path=path.name, sha256=sha256(path),
            source_depth_sha256=e["sha256"], source_parent_sha256=pe["sha256"], pair_seed=seed,
            pairs=len(target["pairs"]), positive_weight_pairs=int((target["weights"] > 0).sum()),
            mean_weight=float(target["weights"].mean()),
            teacher_stats={k: float(v) for k, v in target["teacher_stats"].items()}))
    write_json(args.out / "index.json", dict(status="completed", entries=entries,
        privileged_train_hr=False, manifest_sha256=depth_index["manifest_sha256"],
        parent_sha256=parent_index["parent_sha256"], depth_index_sha256=sha256(args.depth_index),
        parent_index_sha256=sha256(args.parent_index), source_sha256=sha256(__file__),
        numerics=NUMERICS, registration=target["registration"],
        info="Only legal LR teacher depth, LR augmentations and frozen U6000 alpha support. No HR supervision."))


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--export-parent", action="store_true")
    mode.add_argument("--prepare-depth", action="store_true")
    mode.add_argument("--prepare-targets", action="store_true")
    for option in ["manifest", "resume", "model-source", "weight", "reuse", "depth-index", "parent-index"]:
        parser.add_argument("--" + option, type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    required = (["manifest", "resume"] if args.export_parent else
                ["manifest", "model_source", "weight"] if args.prepare_depth else
                ["depth_index", "parent_index"])
    if any(getattr(args, field) is None for field in required):
        parser.error("Missing required mode options: " + ", ".join(required))
    args.out.mkdir(parents=True, exist_ok=False)
    try:
        (export_parent if args.export_parent else prepare_depth if args.prepare_depth else prepare_targets)(args)
    except BaseException:
        write_json(args.out / "failed.json", dict(status="failed", traceback=traceback.format_exc()))
        raise


if __name__ == "__main__":
    main()
