"""IE-SRGS supervision, following author code 74d60d5 (AAAI 2026).

This is the IE-SRGS-on-Wu adaptation, not a reproduction of static IE-SRGS.
The authoritative implementation masks L1 only, uses two unmasked DSSIM
losses, and correlates *negative* inverse depth with rendered median depth.
See docs/static_sr_wu_20261022_sources/iesrgs.md for paper/code differences.
No CUDA dependency is imported by this module.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from collections import defaultdict
from typing import Any, Mapping, Sequence
import random

import torch
import torch.nn.functional as F

UPSTREAM_URL = "https://github.com/ZheXues/IE-SRGS"
UPSTREAM_COMMIT = "74d60d5bb0f49974ac0efa3497dbbd9e246480f4"


@dataclass(frozen=True)
class IESRGSConfig:
    """Released source defaults; same-time sampling is the dynamic adaptation."""
    mask_threshold: float = 0.9
    lambda_dssim: float = 0.2
    internal_depth_weight: float = 0.001
    external_depth_weight: float = 0.0001
    internal_iterations: int = 30_000
    final_iterations: int = 30_000
    internal_views_per_step: int = 3
    internal_view_reduction: str = "sum"
    final_initialization: str = "fresh"
    internal_and_final_mip_filter: bool = True
    kernel_size: float = 0.1
    densify_from_iter: int = 500
    densify_until_iter: int = 15_000
    densification_interval: int = 100
    densify_grad_threshold: float = 0.0002
    opacity_reset_interval: int = 3000
    min_opacity: float = 0.005
    sh_increment_interval: int = 1000
    filter_refresh_interval_after_densification: int = 100
    # Preserve source finite-input values. Zero denominators are interpreted
    # explicitly: 0/0 is no disagreement; nonzero/0 selects the internal prior.
    zero_denominator_policy: str = "source_limit"

    def __post_init__(self):
        if not 0 <= self.lambda_dssim <= 1 or self.mask_threshold <= 0:
            raise ValueError("invalid IE-SRGS mask/loss coefficients")
        if self.internal_views_per_step != 3 or self.internal_view_reduction != "sum":
            raise ValueError("released IE-SRGS accumulates three view losses by summation")
        if self.final_initialization != "fresh":
            raise ValueError("released final IE-SRGS model is independently initialized")
        if self.zero_denominator_policy != "source_limit":
            raise ValueError("unsupported zero-denominator convention")

    def as_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class IESRGSPriors:
    internal_rgb: torch.Tensor
    internal_depth: torch.Tensor
    external_rgb: torch.Tensor
    external_inverse_depth: torch.Tensor

    def to(self, device=None, dtype=None):
        return IESRGSPriors(**{
            key: value.to(device=device, dtype=dtype).detach()
            for key, value in vars(self).items()
        })


def _image(x: torch.Tensor, name: str) -> torch.Tensor:
    if x.ndim == 4 and x.shape[0] == 1:
        x = x[0]
    if x.ndim != 3 or x.shape[0] != 3:
        raise ValueError(f"{name} must be RGB [3,H,W] (or [1,3,H,W])")
    if not torch.is_floating_point(x) or not torch.isfinite(x).all():
        raise ValueError(f"{name} must contain finite floating-point RGB")
    return x


def _depth(x: torch.Tensor, hw, name: str) -> torch.Tensor:
    if x.ndim == 3 and x.shape[0] == 1:
        x = x[0]
    if x.ndim != 2 or tuple(x.shape) != tuple(hw):
        raise ValueError(f"{name} must match RGB spatial resolution {tuple(hw)}")
    if not torch.is_floating_point(x) or not torch.isfinite(x).all():
        raise ValueError(f"{name} must contain finite floating-point depth")
    return x


def ssim(image: torch.Tensor, target: torch.Tensor, window_size: int = 11):
    """Gaussian-window SSIM, mathematically matching the source loss_utils.py."""
    image, target = _image(image, "render"), _image(target, "target")
    if image.shape != target.shape:
        raise ValueError("SSIM input shapes differ")
    coord = torch.arange(window_size, dtype=image.dtype, device=image.device)
    weight = torch.exp(-((coord - window_size // 2) ** 2) / (2 * 1.5 ** 2))
    weight = weight / weight.sum()
    kernel = (weight[:, None] * weight[None, :])[None, None].expand(3, 1, -1, -1)
    conv = lambda value: F.conv2d(value[None], kernel, padding=window_size // 2, groups=3)
    mu_x, mu_y = conv(image), conv(target)
    var_x = conv(image.square()) - mu_x.square()
    var_y = conv(target.square()) - mu_y.square()
    covariance = conv(image * target) - mu_x * mu_y
    return (((2 * mu_x * mu_y + 0.01 ** 2) * (2 * covariance + 0.03 ** 2)) /
            ((mu_x.square() + mu_y.square() + 0.01 ** 2) *
             (var_x + var_y + 0.03 ** 2))).mean()


@torch.no_grad()
def discrepancy_mask(internal_rgb, external_rgb, threshold=0.9):
    """Channel-wise mask from author train.py; 1 chooses internal RGB.

    Division by a positive internal value is unchanged from the source. At
    zero, this matches source comparison semantics without retaining NaN/Inf.
    The paper's epsilon and scalar-per-pixel notation are not silently applied.
    """
    internal_rgb = _image(internal_rgb, "internal_rgb")
    external_rgb = _image(external_rgb, "external_rgb")
    if internal_rgb.shape != external_rgb.shape:
        raise ValueError("internal/external RGB shapes differ")
    if (internal_rgb < 0).any():
        raise ValueError("IE-SRGS internal RGB must be nonnegative")
    delta = (external_rgb - internal_rgb).abs()
    # For nonnegative I, abs((E-I)/I)>=T is equivalent to abs(E-I)>=T*I,
    # except the source 0/0 comparison is False. Explicitly preserve that case.
    return ((delta >= threshold * internal_rgb) &
            ((internal_rgb != 0) | (delta != 0))).to(internal_rgb.dtype)


def pearson_loss(rendered_depth, negative_inverse_depth):
    """1-Pearson; fail on constant input instead of fabricating a depth loss."""
    a = rendered_depth.flatten() - rendered_depth.mean()
    b = negative_inverse_depth.flatten() - negative_inverse_depth.mean()
    aa, bb = a.square().sum(), b.square().sum()
    if not bool(aa.detach() > 0) or not bool(bb.detach() > 0):
        raise ValueError("Pearson depth guidance undefined for constant depth")
    return 1 - (a * b).sum() / torch.sqrt(aa * bb)


def fusion_loss(render_rgb, render_median_depth, priors: IESRGSPriors,
                config: IESRGSConfig | None = None):
    """Compute the complete released mode-2 objective; teachers are detached."""
    cfg = config or IESRGSConfig()
    rgb = _image(render_rgb, "render_rgb")
    priors = priors.to(device=rgb.device, dtype=rgb.dtype)
    internal = _image(priors.internal_rgb, "internal_rgb")
    external = _image(priors.external_rgb, "external_rgb")
    if rgb.shape != internal.shape or rgb.shape != external.shape:
        raise ValueError("RGB priors must exactly match rendered camera and size")
    depth = _depth(render_median_depth, rgb.shape[-2:], "render_median_depth")
    internal_depth = _depth(priors.internal_depth, rgb.shape[-2:], "internal_depth")
    inverse_depth = _depth(priors.external_inverse_depth, rgb.shape[-2:], "external_inverse_depth")
    mask = discrepancy_mask(internal, external, cfg.mask_threshold)
    l1_internal = ((rgb - internal).abs() * mask).mean()
    l1_external = ((rgb - external).abs() * (1 - mask)).mean()
    # Deliberately unmasked: that is what the released training code executes.
    dssim_internal = 1 - ssim(rgb, internal)
    dssim_external = 1 - ssim(rgb, external)
    texture = (1 - cfg.lambda_dssim) * (l1_internal + l1_external)
    texture = texture + cfg.lambda_dssim * (dssim_internal + dssim_external)
    geometry_internal = (depth - internal_depth).abs().mean()
    geometry_external = pearson_loss(depth, -inverse_depth)
    total = texture + cfg.internal_depth_weight * geometry_internal
    total = total + cfg.external_depth_weight * geometry_external
    return dict(total=total, texture=texture, internal_depth=geometry_internal,
                external_depth=geometry_external, l1_internal=l1_internal,
                l1_external=l1_external, dssim_internal=dssim_internal,
                dssim_external=dssim_external, mask_internal_fraction=mask.mean())


def internal_multiview_loss(rendered_images: Sequence[torch.Tensor],
                            lr_images: Sequence[torch.Tensor], lambda_dssim=0.2):
    """The three-view source objective is a sum, not an averaged view loss."""
    if len(rendered_images) != 3 or len(lr_images) != 3:
        raise ValueError("IE-SRGS internal training requires three synchronized views")
    terms = []
    for rendered, target in zip(rendered_images, lr_images):
        rendered = _image(rendered, "render")
        target = _image(target, "lr").detach().to(rendered)
        if rendered.shape != target.shape:
            raise ValueError("internal stage must render at the LR training resolution")
        terms.append((1 - lambda_dssim) * (rendered - target).abs().mean() +
                     lambda_dssim * (1 - ssim(rendered, target)))
    return torch.stack(terms).sum()


class SameTimeViewSampler:
    """Dynamic migration: three distinct training cameras at one timestamp.

    A static three-view consistency objective must not mix different object
    states. Observations use keys scene_id, frame_id, camera_id, split.
    Within each frame a camera pool is consumed without replacement. At a pool
    boundary, exclude cameras already drawn in this batch. This preserves three
    distinct synchronized views while carrying the remaining pool to next call.
    Frame selection itself uses the supplied deterministic random.Random.
    """
    def __init__(self, observations: Sequence[Mapping[str, Any]], seed=50494):
        self.rng = random.Random(seed)
        self.groups = defaultdict(list)
        scenes = set()
        for row in observations:
            if row.get("split") != "train":
                raise ValueError("IE-SRGS sampler accepts training observations only")
            key = (str(row["scene_id"]), str(row["frame_id"]))
            scenes.add(key[0])
            self.groups[key].append(row)
        if len(scenes) != 1:
            raise ValueError("scene-specific IE-SRGS training must contain one scene")
        for key, rows in self.groups.items():
            if len({str(row["camera_id"]) for row in rows}) != len(rows):
                raise ValueError(f"duplicate camera in {key}")
            if len(rows) < 3:
                raise ValueError(f"fewer than three synchronized views in {key}")
        self.keys = sorted(self.groups)
        self.pools = {key: list(rows) for key, rows in self.groups.items()}

    def sample(self):
        key = self.rng.choice(self.keys)
        selected = []
        for _ in range(3):
            used = {str(row["camera_id"]) for row in selected}
            eligible = [i for i, row in enumerate(self.pools[key]) if str(row["camera_id"]) not in used]
            if not eligible:
                self.pools[key] = list(self.groups[key])
                eligible = [i for i, row in enumerate(self.pools[key]) if str(row["camera_id"]) not in used]
            selected.append(self.pools[key].pop(self.rng.choice(eligible)))
        return selected


def filter_refresh_required(iteration: int, cfg: IESRGSConfig,
                            topology_changed: bool = False):
    """Source refresh events; dynamic renderer may additionally refresh at t."""
    return (iteration == 1 or topology_changed or
            (iteration > cfg.densify_until_iter and
             iteration < cfg.final_iterations - 100 and
             iteration % cfg.filter_refresh_interval_after_densification == 0))
