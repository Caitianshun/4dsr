"""Surface-attached RGB residuals on an existing Wu 4DGaussians model.

This module never edits the upstream renderer. The auxiliary rasterization uses
the current deformed support without a gradient to that support, signed RGB
colors, and a black background. The final image is base + H(auxiliary image).
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import sys
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / "experiments/dynamic_sr_20260918"
if str(OLD) not in sys.path:
    sys.path.insert(0, str(OLD))

BRANCHES = ("joint", "global_residual", "local_residual")


def highpass(image: torch.Tensor, lr_size: tuple[int, int]) -> torch.Tensor:
    """H=x-U(D_raw(x)); deliberately no clipping inside either linear resize."""
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError(f"Expected CHW RGB, got {tuple(image.shape)}")
    low = F.interpolate(image[None], size=lr_size, mode="bicubic",
                        align_corners=False, antialias=True)
    up = F.interpolate(low, size=image.shape[-2:], mode="bicubic",
                       align_corners=False, antialias=False)[0]
    return image - up


class TemporalResidual(nn.Module):
    """Four bounded RGB coefficients per canonical Gaussian; same capacity."""

    def __init__(self, points: int, basis: str, time_range: tuple[float, float],
                 device: torch.device | str = "cpu", amplitude: float = 0.2):
        super().__init__()
        if basis not in ("global", "local") or points <= 0:
            raise ValueError((points, basis))
        lo, hi = map(float, time_range)
        if not math.isfinite(lo) or not math.isfinite(hi) or hi < lo:
            raise ValueError(time_range)
        if amplitude <= 0 or not math.isfinite(amplitude):
            raise ValueError(amplitude)
        self.basis = basis
        self.time_range = (lo, hi)
        self.amplitude = float(amplitude)
        self.coefficients = nn.Parameter(torch.zeros(points, 4, 3, device=device))

    def basis_weights(self, time: float | torch.Tensor) -> torch.Tensor:
        t = torch.as_tensor(time, dtype=self.coefficients.dtype,
                            device=self.coefficients.device)
        if t.numel() != 1:
            raise ValueError("A render needs one scalar camera timestamp")
        t = t.reshape(())
        lo, hi = self.time_range
        # A single-time manifest is legal; use its first knot in either branch.
        u = torch.zeros_like(t) if hi == lo else ((t - lo) / (hi - lo)).clamp(0, 1)
        if self.basis == "global":
            v = 1 - u
            return torch.stack((v ** 3, 3 * u * v ** 2, 3 * u ** 2 * v, u ** 3))
        knots = torch.arange(4, device=u.device, dtype=u.dtype)
        return (1 - (3 * u - knots).abs()).clamp_min(0)

    def forward(self, time: float | torch.Tensor) -> torch.Tensor:
        bounded = self.amplitude * self.coefficients.tanh()
        return torch.einsum("k,nkc->nc", self.basis_weights(time), bounded)


@dataclass
class SurfaceModel:
    g: Any
    h: Any
    o: Any
    checkpoint: dict[str, Any]
    detail: TemporalResidual | None
    detail_optimizer: torch.optim.Optimizer | None
    lr_size: tuple[int, int]
    time_range: tuple[float, float]
    branch: str


def manifest_geometry(manifest) -> tuple[dict, tuple[int, int], tuple[float, float]]:
    from n3dv_data import load_manifest
    m = load_manifest(manifest) if not isinstance(manifest, dict) else manifest
    times = [float(r["time"]) for r in m["observations"] if r["split"] == "train"]
    if not times or not all(math.isfinite(t) for t in times):
        raise ValueError("Manifest needs finite training timestamps")
    width, height = map(int, m["resolutions"]["lr"])
    return m, (height, width), (min(times), max(times))


def make_model(g, h, o, checkpoint, manifest, branch: str,
               detail_lr: float = 0.0025) -> SurfaceModel:
    if branch not in BRANCHES or detail_lr <= 0 or not math.isfinite(detail_lr):
        raise ValueError((branch, detail_lr))
    _, lr_size, time_range = manifest_geometry(manifest)
    detail = optimizer = None
    if branch != "joint":
        detail = TemporalResidual(len(g._xyz), branch.removesuffix("_residual"),
                                  time_range, g._xyz.device)
        optimizer = torch.optim.Adam(detail.parameters(), lr=detail_lr,
                                     betas=(0.9, 0.999), eps=1e-15)
    return SurfaceModel(g, h, o, checkpoint, detail, optimizer,
                        lr_size, time_range, branch)


def detail_state(model: SurfaceModel) -> dict:
    """Extra top-level checkpoint field; the original model tuple stays intact."""
    d = model.detail
    return {"schema": 1, "branch": model.branch,
            "basis": d.basis if d is not None else "none",
            "time_range": list(model.time_range), "lr_size": list(model.lr_size),
            "amplitude": d.amplitude if d is not None else 0.2,
            "coefficients": d.coefficients.detach() if d is not None else None,
            "optimizer": model.detail_optimizer.state_dict() if d is not None else None,
            "detail_lr": model.detail_optimizer.param_groups[0]["lr"] if d is not None else None}


def load_model(checkpoint_path, manifest=None, restore_rng: bool = False) -> SurfaceModel:
    """Unified GPU loader for original baseline and augmented checkpoints.

    Evaluation uses restore_rng=False. This function restores a residual Adam
    when present, but does not choose/advance training observation samplers.
    """
    from common import load_checkpoint
    g, h, o, ck = load_checkpoint(checkpoint_path)
    saved = ck.get("surface_residual")
    if saved is None:
        source = manifest if manifest is not None else ck["metadata"].get("manifest")
        if source is None:
            raise ValueError("An original checkpoint requires its prepared manifest")
        _, lr_size, time_range = manifest_geometry(source)
        result = SurfaceModel(g, h, o, ck, None, None, lr_size, time_range, "joint")
    else:
        if saved["schema"] != 1 or saved["branch"] not in BRANCHES:
            raise ValueError("Unsupported surface-residual checkpoint")
        lr_size = tuple(map(int, saved["lr_size"]))
        time_range = tuple(map(float, saved["time_range"]))
        if manifest is not None:
            _, check_lr, check_time = manifest_geometry(manifest)
            if check_lr != lr_size or check_time != time_range:
                raise ValueError("Checkpoint/manifest resolution or time window mismatch")
        d = opt = None
        branch = saved["branch"]
        if branch != "joint":
            if saved["basis"] != branch.removesuffix("_residual"):
                raise ValueError("Checkpoint basis/branch mismatch")
            d = TemporalResidual(len(g._xyz), saved["basis"], time_range,
                                 g._xyz.device, saved["amplitude"])
            coefficients = saved["coefficients"]
            if tuple(coefficients.shape) != tuple(d.coefficients.shape):
                raise ValueError("Point count/detail coefficients mismatch")
            with torch.no_grad():
                d.coefficients.copy_(coefficients)
            opt = torch.optim.Adam(d.parameters(), lr=float(saved["detail_lr"]),
                                   betas=(0.9, 0.999), eps=1e-15)
            opt.load_state_dict(saved["optimizer"])
        result = SurfaceModel(g, h, o, ck, d, opt, lr_size, time_range, branch)
    if restore_rng:
        path = ROOT / "experiments/dynamic_sr_20260920"
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
        from resume_control import restore_global_rng
        restore_global_rng(ck["rng"])
    return result


def render_detail(model: SurfaceModel, camera) -> torch.Tensor:
    """Only the RGB coefficient path differentiates; support follows the base."""
    if model.detail is None:
        raise ValueError("joint has no auxiliary residual rasterization")
    # Importing common first installs the configured upstream extension path.
    import common  # noqa: F401
    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
    g, detail = model.g, model.detail
    with torch.no_grad():
        t = torch.full((len(g._xyz), 1), float(camera.time),
                       dtype=g._xyz.dtype, device=g._xyz.device)
        xyz, scale, rotation, opacity, _ = g._deformation(
            g._xyz, g._scaling, g._rotation, g._opacity, g.get_features, t)
        xyz = xyz.detach().contiguous()
        scale = g.scaling_activation(scale).detach().contiguous()
        rotation = g.rotation_activation(rotation).detach().contiguous()
        opacity = g.opacity_activation(opacity).detach().contiguous()
    settings = GaussianRasterizationSettings(
        image_height=int(camera.image_height), image_width=int(camera.image_width),
        tanfovx=math.tan(camera.FoVx * 0.5), tanfovy=math.tan(camera.FoVy * 0.5),
        bg=torch.zeros(3, dtype=xyz.dtype, device=xyz.device), scale_modifier=1.0,
        viewmatrix=camera.world_view_transform.to(xyz.device),
        projmatrix=camera.full_proj_transform.to(xyz.device), sh_degree=0,
        campos=camera.camera_center.to(xyz.device), prefiltered=False, debug=False)
    # Both positive and negative colors are intentional. Passing SH as well
    # would violate the extension API; no upstream override_color path is used.
    image, _, _ = GaussianRasterizer(raster_settings=settings)(
        means3D=xyz, means2D=torch.zeros_like(xyz), shs=None,
        colors_precomp=detail(camera.time).contiguous(), opacities=opacity,
        scales=scale, rotations=rotation, cov3D_precomp=None)
    return image


def render_model(model: SurfaceModel, camera, detach_base: bool = False) -> dict:
    """Return CHW float RGB under ``render``; never clamp the final image.

    The current joint/global/local training uses the default detach_base=False
    for both LR and SR. The optional low-level flag can suppress the base graph
    for separate diagnostics; it is not this protocol's SR route. Auxiliary
    residual support is always detached, including in the LR loss.
    """
    from common import render_image
    if detach_base:
        with torch.no_grad():
            package = render_image(model.g, camera)
        base = package["render"].detach()
    else:
        package = render_image(model.g, camera)
        base = package["render"]
    if model.detail is None:
        return {**package, "render": base, "base_render": base,
                "detail_render": None, "high_detail": None}
    auxiliary = render_detail(model, camera)
    residual = highpass(auxiliary, model.lr_size)
    return {**package, "render": base + residual, "base_render": base,
            "detail_render": auxiliary, "high_detail": residual}
