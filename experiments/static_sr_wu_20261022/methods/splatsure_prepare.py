"""Frozen-LR SplatSuRe cache producer using the pinned official rasterizer.

No CUDA work runs on import. The calling runner owns resource admission,
checkpoint validation, and construction of the WuBackend. This producer does not open image files itself or update parameters. A backend
may load training LR while constructing cameras; held-out and HR image reads are
forbidden by its contract. Compilation and real render equivalence are separate acceptance gates.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Mapping, Optional, Sequence

import torch

from .splatsure import (
    FidelityAccumulator, RASTERIZER_COMMIT, SplatSuReConfig, UPSTREAM_COMMIT,
    selection_weight_map,
)

RASTERIZER_SETTINGS = {"antialiasing": False, "num_contributions": 10}
RASTERIZER_RETURN_FIELDS = (
    "render", "radii", "true_radii", "inverse_depth", "is_contributing", "contributions"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_official_rasterizer(source_root: str | Path, binary_root: Optional[str | Path] = None):
    """Load an already built official extension under a unique package name.

    Does not install packages, compile CUDA, change sys.path, or replace Wu's
    diff_gaussian_rasterization package. Verify the exact source revision first.
    Required preparation: checkout the pinned commit, initialize its GLM submodule,
    then build_ext into an isolated binary_root (or in place) using the admitted
    environment and target CUDA arch. An isolated wrapper must exactly match source.
    """
    root = Path(source_root).resolve()
    result = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], check=True,
                            capture_output=True, text=True)
    if result.stdout.strip() != RASTERIZER_COMMIT:
        raise RuntimeError(f"SplatSuRe rasterizer must be pinned to {RASTERIZER_COMMIT}")
    modified = subprocess.run(["git", "-C", str(root), "diff", "--quiet", "HEAD", "--"], check=False)
    if modified.returncode:
        raise RuntimeError("SplatSuRe rasterizer has tracked source modifications; audit before use")
    source_package = root / "diff_gaussian_rasterization"
    package_dir = (Path(binary_root).resolve() / "diff_gaussian_rasterization") if binary_root else source_package
    if not (package_dir / "__init__.py").is_file() or sha256_file(package_dir / "__init__.py") != sha256_file(source_package / "__init__.py"):
        raise RuntimeError("compiled-package Python wrapper differs from the pinned source")
    if not list(package_dir.glob("_C*.so")):
        raise RuntimeError("SplatSuRe rasterizer binary is absent. Build to a separate binary_root; do not overwrite the Wu rasterizer installation.")
    name = "_splatsure_rasterizer_" + RASTERIZER_COMMIT[:12]
    if name in sys.modules:
        old = sys.modules[name]
        if Path(old.__file__).resolve() != package_dir / "__init__.py":
            raise RuntimeError("the pinned rasterizer alias is already loaded from another directory")
        return old
    spec = importlib.util.spec_from_file_location(name, package_dir / "__init__.py",
                                                 submodule_search_locations=[str(package_dir)])
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the SplatSuRe rasterizer package")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        sys.modules.pop(name + "._C", None)
        raise
    module.static_sr_extra_settings = dict(RASTERIZER_SETTINGS)
    module.static_sr_return_fields = RASTERIZER_RETURN_FIELDS
    module.static_sr_source_commit = RASTERIZER_COMMIT
    module.static_sr_binary_identities = [dict(path=str(p), sha256=sha256_file(p)) for p in sorted(package_dir.glob("_C*.so"))]
    return module


def unpack_rasterizer_result(result) -> dict:
    """Backend bridge: tuple #4 is inverse depth, never metric Wu depth."""
    if len(result) != 6:
        raise ValueError("SplatSuRe renderer must return six outputs")
    return dict(zip(RASTERIZER_RETURN_FIELDS, result))


def same_time_training_groups(manifest: Mapping, frame_indices: Optional[Sequence[int]] = None):
    """Return deterministic train-only groups and reject conflicting split labels."""
    train_ids = set(manifest["splits"]["train"])
    forbidden = set(manifest["splits"].get("test", [])) | set(manifest["splits"].get("dev", []))
    if train_ids & forbidden:
        raise ValueError("train and held-out camera IDs overlap")
    chosen = None if frame_indices is None else set(int(f) for f in frame_indices)
    groups = {}
    times = {}
    seen = set()
    for obs in manifest["observations"]:
        camera_id = obs["camera_id"]
        if obs["split"] != "train":
            if camera_id in train_ids:
                raise ValueError("manifest train camera has a non-train observation")
            continue
        if camera_id not in train_ids or camera_id in forbidden:
            raise ValueError("training observation is not in the declared train-only camera split")
        frame_index = int(obs["frame_index"])
        if chosen is not None and frame_index not in chosen:
            continue
        key = (camera_id, frame_index)
        if key in seen:
            raise ValueError("duplicate camera/frame training observation")
        seen.add(key)
        timestamp = float(obs["time"])
        if not math.isfinite(timestamp):
            raise ValueError("non-finite observation time")
        if frame_index in times and timestamp != times[frame_index]:
            raise ValueError("same frame_index contains different times across cameras")
        times[frame_index] = timestamp
        groups.setdefault(frame_index, []).append(dict(obs))
    if not groups or (chosen is not None and chosen != set(groups)):
        raise ValueError("requested training frames are missing")
    return [(frame, times[frame], sorted(group, key=lambda x: x["camera_id"]))
            for frame, group in sorted(groups.items())]


def prepare_splatsure_cache(
    backend,
    manifest: Mapping | str | Path,
    output_dir: str | Path,
    raster_api,
    provenance: Mapping,
    config: Optional[SplatSuReConfig] = None,
    *,
    frame_indices: Optional[Sequence[int]] = None,
) -> dict:
    """Generate fixed LR weight maps, independently for each synchronized time.

    Backend contract:
      effective_state(time) -> frozen Wu state;
      render(obs, state=state, resolution='lr', colors=None, raster_api=api)
        -> dict with render [3,H,W], true_radii [N], is_contributing [N].
    When ``colors`` is supplied, backend must set shs=None and black background.
    All three forward passes use the same state/camera/rasterization settings.

    ``provenance`` must identify the scene's LR parent checkpoint SHA256. A cache
    can be generated only after source installation and GPU admission. This call
    records preparation, never claims training or GPU smoke acceptance.
    """
    config = config or SplatSuReConfig()
    if not re.fullmatch(r"[0-9a-f]{64}", str(provenance.get("parent_checkpoint_sha256", ""))):
        raise ValueError("provenance requires the frozen scene-local LR parent checkpoint SHA256")
    if getattr(raster_api, "static_sr_source_commit", None) != RASTERIZER_COMMIT:
        raise ValueError("raster_api must identify the pinned official SplatSuRe rasterizer")
    if isinstance(manifest, (str, Path)):
        path = Path(manifest).resolve()
        source_hash = sha256_file(path)
        manifest = json.loads(path.read_text())
        manifest_source = str(path)
    else:
        source_hash = hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        manifest_source = "provided_mapping_canonical_json"
    groups = same_time_training_groups(manifest, frame_indices)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    (output / "maps").mkdir()
    start = time.monotonic()
    receipt = dict(schema="wu_splatsure_weights_v1", status="preparing", scene=manifest["scene"],
                   manifest_source=manifest_source, manifest_sha256=source_hash,
                   provenance=dict(provenance), upstream_commit=UPSTREAM_COMMIT,
                   rasterizer_commit=RASTERIZER_COMMIT, config=asdict(config),
                   rasterizer_settings=RASTERIZER_SETTINGS,
                   rasterizer_binary_identities=getattr(raster_api, "static_sr_binary_identities", []),
                   producer_files=[dict(path=str(p), sha256=sha256_file(p)) for p in
                       (Path(__file__).resolve(), Path(__file__).with_name("splatsure.py").resolve())],
                   requested_frame_indices=None if frame_indices is None else sorted(set(frame_indices)),
                   training_only=True, time_policy="same_time_train_cameras_only",
                   tie_policy="first_camera_in_lexical_camera_id_order",
                   parameter_updates=0, producer_image_reads=0, backend_LR_image_reads="backend_camera_implementation_dependent",
                   HR_image_reads=0,
                   records=[], frame_diagnostics=[])
    status_path = output / "receipt.json"
    status_path.write_text(json.dumps(receipt, indent=2) + "\n")
    try:
        for frame, timestamp, observations in groups:
            with torch.no_grad():
                state = backend.effective_state(timestamp)
                acc = None
                for index, obs in enumerate(observations):
                    rendered = backend.render(obs, state=state, resolution="lr", raster_api=raster_api)
                    if "true_radii" not in rendered or "is_contributing" not in rendered:
                        raise RuntimeError("missing official contribution outputs; frustum radii are not an acceptable substitute")
                    radii, contrib = rendered["true_radii"], rendered["is_contributing"]
                    if acc is None:
                        acc = FidelityAccumulator.create(radii.numel(), device="cpu", dtype=radii.dtype)
                    acc.update(index, radii.cpu(), contrib.cpu())
                    device = rendered["render"].device
                scores = acc.score(config)
                for index, obs in enumerate(observations):
                    colors = scores[:, None].expand(-1, 3).contiguous().to(device)
                    fidelity = backend.render(obs, state=state, resolution="lr", colors=colors, raster_api=raster_api)["render"]
                    closest = (acc.max_camera_index == index).to(scores.dtype)
                    colors = closest[:, None].expand(-1, 3).contiguous().to(device)
                    closest_render = backend.render(obs, state=state, resolution="lr", colors=colors, raster_api=raster_api)["render"]
                    weights = selection_weight_map(fidelity, closest_render).cpu()
                    camera_id = str(obs["camera_id"])
                    if not re.fullmatch(r"[A-Za-z0-9_-]+", camera_id):
                        raise ValueError("camera ID is unsafe for a cache filename")
                    relative = f"maps/{camera_id}_{frame:06d}.pt"
                    torch.save(weights, output / relative)
                    receipt["records"].append(dict(camera_id=camera_id, frame_index=frame, time=timestamp,
                                                    path=relative, sha256=sha256_file(output / relative),
                                                    shape=list(weights.shape), minimum=float(weights.min()),
                                                    maximum=float(weights.max()), mean=float(weights.mean()),
                                                    zero_fraction=float((weights == 0).float().mean())))
                receipt["frame_diagnostics"].append(dict(frame_index=frame, time=timestamp,
                     cameras=len(observations), gaussians=len(scores),
                     fewer_than_three_views_fraction=float((acc.seen_count < 3).float().mean()),
                     never_contributing_fraction=float((acc.seen_count == 0).float().mean()),
                     mean_fidelity=float(scores.mean())))
        receipt.update(status="prepared", seconds=time.monotonic() - start, render_calls=3 * len(receipt["records"]))
    except BaseException as exc:
        receipt.update(status="failed", seconds=time.monotonic() - start,
                       error_type=type(exc).__name__, error=str(exc))
        status_path.write_text(json.dumps(receipt, indent=2) + "\n")
        raise
    status_path.write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt
