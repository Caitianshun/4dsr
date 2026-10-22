"""Official IE-SRGS RADE rasterization applied to Wu's effective state at t.

No copy of the CUDA implementation is shipped. Build the pinned author's
submodules/diff-gaussian-rasterization-rade in an isolated runtime. This adapter
retains its median-depth backward and the Mip 2D kernel. Source references are
listed in docs/static_sr_wu_20261022_sources/iesrgs.json.
"""
from __future__ import annotations

import importlib
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Mapping, Sequence

import torch

_RASTER_API = None
_WRAPPER_SHA256 = "c1d57f2c1f3c64a31d8d9f289c7e6e965cb4639dfec370474a7fb18d1996c8ec"
_SOURCE_COMMIT = "74d60d5bb0f49974ac0efa3497dbbd9e246480f4"


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for part in iter(lambda: file.read(1 << 20), b""):
            digest.update(part)
    return digest.hexdigest()


def raster_api(binary_root=None):
    """Load the isolated build with recorded binary and pinned wrapper hashes.

    IESRGS_RASTERIZER_ROOT overrides the default output runtime directory. Its
    build_receipt.json and lib/diff_gaussian_rasterization_rade are mandatory.
    Validation is cached after the first import to avoid hashing each render.
    """
    global _RASTER_API
    if _RASTER_API is not None:
        return _RASTER_API
    default = Path(__file__).resolve().parents[3] / "output/static_sr_wu_20261022/runtime/iesrgs"
    root = Path(binary_root or os.environ.get("IESRGS_RASTERIZER_ROOT", default)).resolve()
    receipt_path = root / "build_receipt.json"
    if not receipt_path.is_file():
        raise RuntimeError(f"missing isolated IE-SRGS build receipt: {receipt_path}")
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("source_commit") != _SOURCE_COMMIT or receipt.get("build_status") != "success":
        raise RuntimeError("IE-SRGS rasterizer receipt does not describe the pinned successful build")
    if receipt.get("torch_version") != torch.__version__ or receipt.get("python_abi") != sys.implementation.cache_tag:
        raise RuntimeError("IE-SRGS rasterizer was built for a different Python/PyTorch runtime")
    package = root / "lib/diff_gaussian_rasterization_rade"
    if _hash(package / "__init__.py") != _WRAPPER_SHA256:
        raise RuntimeError("IE-SRGS rasterizer Python wrapper differs from pinned author source")
    binary = package / receipt["binary"]["name"]
    if _hash(binary) != receipt["binary"]["sha256"]:
        raise RuntimeError("IE-SRGS rasterizer binary identity changed")
    sys.path.insert(0, str(root / "lib"))
    try:
        module = importlib.import_module("diff_gaussian_rasterization_rade")
        if not Path(module.__file__).resolve().is_relative_to(package):
            raise RuntimeError("a different RADE rasterizer was already imported; use a clean process")
        _RASTER_API = module
        return module
    except ImportError as exc:
        raise RuntimeError(
            "IE-SRGS requires the pinned diff_gaussian_rasterization_rade "
            "extension with differentiable median_depth. Build it in the "
            "isolated IE-SRGS runtime; Wu accumulated depth is not a substitute."
        ) from exc
    finally:
        sys.path.remove(str(root / "lib"))


@torch.no_grad()
def compute_3d_filter(xyz: torch.Tensor, reference_cameras: Sequence):
    """Author Mip 3D filter evaluated on *deformed* xyz at the given time.

    Exact source criterion: z>0.2; an image rectangle expanded by 15%; minimum
    visible camera z divided by maximum camera focal_x, multiplied by sqrt(.2).
    Never-seen points receive the maximum observed minimum depth. An entirely
    invisible point set is an error rather than an arbitrary smoothing value.
    Reference cameras must all use the stage's training image resolution.
    """
    if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) == 0:
        raise ValueError("xyz must contain N nonempty 3D Gaussian centers")
    if not reference_cameras:
        raise ValueError("Mip filtering requires training cameras")
    distance = xyz.new_full((len(xyz),), 100000.)
    valid_points = torch.zeros(len(xyz), dtype=torch.bool, device=xyz.device)
    maximum_fx = 0.0
    for camera in reference_cameras:
        # Wu's world_view_transform uses row-vector convention, including any
        # scene normalization; this also equals author xyz @ R + T in its frame.
        transform = camera.world_view_transform.to(device=xyz.device, dtype=xyz.dtype)
        camera_xyz = xyz @ transform[:3, :3] + transform[3, :3]
        width, height = int(camera.image_width), int(camera.image_height)
        fx = width / (2 * math.tan(float(camera.FoVx) / 2))
        fy = height / (2 * math.tan(float(camera.FoVy) / 2))
        valid_depth = camera_xyz[:, 2] > .2
        z = camera_xyz[:, 2].clamp_min(.001)
        x = camera_xyz[:, 0] / z * fx + width / 2
        y = camera_xyz[:, 1] / z * fy + height / 2
        visible = (valid_depth & (x >= -.15 * width) & (x <= 1.15 * width) &
                   (y >= -.15 * height) & (y <= 1.15 * height))
        distance[visible] = torch.minimum(distance[visible], z[visible])
        valid_points |= visible
        maximum_fx = max(maximum_fx, fx)
    if not valid_points.any():
        raise ValueError("no Gaussian visible to IE-SRGS training cameras")
    distance[~valid_points] = distance[valid_points].max()
    return (distance / maximum_fx * math.sqrt(.2))[:, None]


def apply_3d_filter(scales, opacity, filter_3d):
    """Covariance dilation plus determinant-based opacity mass compensation."""
    if scales.ndim != 2 or scales.shape[1] != 3:
        raise ValueError("activated scales must have shape [N,3]")
    if tuple(opacity.shape) != (len(scales), 1) or tuple(filter_3d.shape) != (len(scales), 1):
        raise ValueError("opacity and filter_3d must have shape [N,1]")
    before = scales.square()
    after = before + filter_3d.detach().square()
    mass_correction = (before.prod(-1) / after.prod(-1)).sqrt()[:, None]
    return after.sqrt(), opacity * mass_correction


def render_state(camera, state: Mapping[str, torch.Tensor], background,
                 active_sh_degree: int, reference_cameras: Sequence,
                 filter_3d=None, require_depth=True, kernel_size=.1,
                 debug=False, api=None):
    """Rasterize activated Wu state {xyz,scales,rotation,opacity,sh}.

    The caller obtains state from the shared Wu deformation network at camera.t.
    Gradients through state return to canonical Gaussians AND the deformation
    network. Recomputing a detached filter at each timestamp is the explicit
    dynamic adaptation; a supplied filter must belong to that timestamp and
    point topology. Both internal-LR and final-HR stages enable this filter.
    """
    missing = {"xyz", "scales", "rotation", "opacity", "sh"} - state.keys()
    if missing:
        raise ValueError(f"missing activated Wu state fields: {sorted(missing)}")
    xyz = state["xyz"]
    if hasattr(camera, "time"):
        for reference in reference_cameras:
            if hasattr(reference, "time") and not math.isclose(
                    float(reference.time), float(camera.time), abs_tol=1e-8):
                raise ValueError("IE-SRGS filter cameras must share render timestamp")
    if filter_3d is None:
        filter_3d = compute_3d_filter(xyz.detach(), reference_cameras)
    scales, opacity = apply_3d_filter(state["scales"], state["opacity"], filter_3d)
    api = api or raster_api()
    screen = torch.zeros_like(xyz, requires_grad=True) + 0
    if torch.is_grad_enabled():
        screen.retain_grad()
    settings = api.GaussianRasterizationSettings(
        image_height=int(camera.image_height), image_width=int(camera.image_width),
        tanfovx=math.tan(float(camera.FoVx) / 2),
        tanfovy=math.tan(float(camera.FoVy) / 2), kernel_size=kernel_size,
        bg=background, scale_modifier=1.0,
        viewmatrix=camera.world_view_transform.to(xyz),
        projmatrix=camera.full_proj_transform.to(xyz),
        sh_degree=active_sh_degree, campos=camera.camera_center.to(xyz),
        prefiltered=False, require_coord=False, require_depth=require_depth, debug=debug)
    values = api.GaussianRasterizer(raster_settings=settings)(
        means3D=xyz, means2D=screen, shs=state["sh"], colors_precomp=None,
        opacities=opacity, scales=scales, rotations=state["rotation"],
        cov3D_precomp=None)
    if len(values) != 8:
        raise RuntimeError("wrong RADE ABI: IE-SRGS renderer must return eight tensors")
    rgb, radii, expected_coord, median_coord, expected_depth, median_depth, alpha, normal = values
    return dict(render=rgb, viewspace_points=screen, visibility_filter=radii > 0,
                radii=radii, expected_coord=expected_coord, median_coord=median_coord,
                expected_depth=expected_depth, median_depth=median_depth,
                mask=alpha, normal=normal, filter_3d=filter_3d)
