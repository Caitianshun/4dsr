"""SplatSuRe supervision for a Wu-4DGS dynamic adaptation.

Numerical behavior follows pranav-asthana/SplatSuRe at
 df40abb8d92a02656268cb0a0b97acc2495b9307 and its rasterizer at
 83814a5d942991a6a201dfd84f81606bc0382579.

The caller must use *frozen LR geometry at one time* and actual contributing
Gaussians, not just positive rasterizer radii. This module imports no renderer,
loads no weights, accesses no CUDA context, and starts no work on import.
See splatsure_UPSTREAM_LICENSE.txt and the source audit for provenance.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Optional

import torch
import torch.nn.functional as F

UPSTREAM_COMMIT = "df40abb8d92a02656268cb0a0b97acc2495b9307"
RASTERIZER_COMMIT = "83814a5d942991a6a201dfd84f81606bc0382579"


@dataclass(frozen=True)
class SplatSuReConfig:
    ratio_threshold: float = 1.1
    transition_width: float = 0.05
    minimum_views: int = 3
    gamma: float = 0.4
    lambda_dssim: float = 0.2
    lr_operator: str = "bicubic"

    def __post_init__(self):
        if not math.isfinite(self.ratio_threshold):
            raise ValueError("ratio_threshold must be finite")
        if self.transition_width <= 0 or self.minimum_views < 1:
            raise ValueError("transition_width/minimum_views must be positive")
        if not 0 <= self.gamma <= 1 or not 0 <= self.lambda_dssim <= 1:
            raise ValueError("loss coefficients must lie in [0,1]")
        if self.lr_operator not in ("bicubic", "area"):
            raise ValueError("lr_operator must be bicubic (upstream) or area (explicit protocol adaptation)")


def project_covariance(
    means3d: torch.Tensor,
    covariance3d: torch.Tensor,
    w2c: torch.Tensor,
    focal_x: float,
    focal_y: float,
    tan_fovx: float,
    tan_fovy: float,
) -> torch.Tensor:
    """Return [N,2,2] EWA covariance *before* the rasterizer's 0.3 dilation.

    ``w2c`` is conventional column-vector world-to-camera, not the transposed
    ``camera.world_view_transform`` stored by Graphdeco. Corresponds to
    J R Sigma R^T J^T with x/z,y/z clamped to +/- 1.3*tan(FoV/2), as in
    the official CUDA computeCov2D. Principal point does not affect J.
    This is only a covariance calculation, not a visibility test.
    """
    if means3d.ndim != 2 or means3d.shape[-1] != 3:
        raise ValueError("means3d must have shape [N,3]")
    if covariance3d.shape != (len(means3d), 3, 3) or w2c.shape != (4, 4):
        raise ValueError("invalid covariance3d or w2c shape")
    if min(focal_x, focal_y, tan_fovx, tan_fovy) <= 0:
        raise ValueError("camera focal lengths and FoV tangents must be positive")
    rotation = w2c[:3, :3].to(means3d)
    xyz_camera = means3d @ rotation.T + w2c[:3, 3].to(means3d)
    z = xyz_camera[:, 2]
    if torch.any(z == 0):
        raise ValueError("project_covariance cannot project a Gaussian at camera z=0")
    x = (xyz_camera[:, 0] / z).clamp(-1.3 * tan_fovx, 1.3 * tan_fovx) * z
    y = (xyz_camera[:, 1] / z).clamp(-1.3 * tan_fovy, 1.3 * tan_fovy) * z
    jacobian = means3d.new_zeros((len(means3d), 2, 3))
    jacobian[:, 0, 0] = focal_x / z
    jacobian[:, 1, 1] = focal_y / z
    jacobian[:, 0, 2] = -focal_x * x / z.square()
    jacobian[:, 1, 2] = -focal_y * y / z.square()
    transform = jacobian @ rotation
    return transform @ covariance3d @ transform.transpose(-1, -2)


def true_radii_from_covariance(covariance2d: torch.Tensor) -> torch.Tensor:
    """Match the upstream true-radius formula, including its 0.1 floor.

    Unlike mathematical eigvalsh this intentionally preserves the CUDA
    discriminant floor. Do not add 0.3, ceil, or substitute padded tile radii.
    The caller must separately apply the official contribution mask.
    """
    if covariance2d.shape[-2:] != (2, 2):
        raise ValueError("covariance2d must end in [2,2]")
    a, b, c = covariance2d[..., 0, 0], covariance2d[..., 0, 1], covariance2d[..., 1, 1]
    mid = 0.5 * (a + c)
    det = a * c - b.square()
    largest = mid + (mid.square() - det).clamp_min(0.1).sqrt()
    return 3 * largest.sqrt()


@dataclass
class FidelityAccumulator:
    """Streaming per-time statistics; avoids retaining [cameras,Gaussians]."""
    minimum: torch.Tensor
    maximum: torch.Tensor
    max_camera_index: torch.Tensor
    seen_count: torch.Tensor

    @classmethod
    def create(cls, num_gaussians: int, *, device=None, dtype=torch.float32):
        if num_gaussians < 1:
            raise ValueError("empty Gaussian model")
        # Preserve the official sentinel values and strict update comparisons.
        return cls(
            torch.full((num_gaussians,), 1e6, device=device, dtype=dtype),
            torch.full((num_gaussians,), -1.0, device=device, dtype=dtype),
            torch.full((num_gaussians,), -1, device=device, dtype=torch.int64),
            torch.zeros((num_gaussians,), device=device, dtype=torch.int64),
        )

    @torch.no_grad()
    def update(self, camera_index: int, true_radii: torch.Tensor, is_contributing: torch.Tensor):
        radii = true_radii.detach().to(self.minimum).reshape(-1)
        contributions = is_contributing.detach().to(device=self.minimum.device).reshape(-1)
        if radii.shape != self.minimum.shape or contributions.shape != radii.shape:
            raise ValueError("radius/contribution length changed within a frozen frame")
        if not torch.isfinite(radii).all() or not torch.isfinite(contributions).all():
            raise ValueError("non-finite rasterizer radius or contribution count")
        visible = (radii > 0) & (contributions > 0)
        self.seen_count += visible.to(self.seen_count.dtype)
        lower = visible & (radii < self.minimum)
        upper = visible & (radii > self.maximum)
        self.minimum[lower] = radii[lower]
        self.maximum[upper] = radii[upper]
        self.max_camera_index[upper] = camera_index

    @torch.no_grad()
    def score(self, config: Optional[SplatSuReConfig] = None) -> torch.Tensor:
        config = config or SplatSuReConfig()
        ratio = (self.maximum / self.minimum).clamp_min(1)
        scores = torch.sigmoid((ratio - config.ratio_threshold) / config.transition_width)
        scores[self.seen_count < config.minimum_views] = 0
        return scores


def fidelity_from_views(
    true_radii: torch.Tensor, is_contributing: torch.Tensor,
    config: Optional[SplatSuReConfig] = None,
) -> tuple[torch.Tensor, FidelityAccumulator]:
    """Convenience interface for [V,N] statistics, for one time only."""
    if true_radii.ndim != 2 or true_radii.shape != is_contributing.shape:
        raise ValueError("expected equal [views,gaussians] radii and contribution tensors")
    acc = FidelityAccumulator.create(true_radii.shape[1], device=true_radii.device, dtype=true_radii.dtype)
    for i in range(len(true_radii)):
        acc.update(i, true_radii[i], is_contributing[i])
    return acc.score(config), acc


def _map2d(image: torch.Tensor) -> torch.Tensor:
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[0] in (1, 3):
        return image[0]
    raise ValueError("expected a scalar [H,W] or replicated [1|3,H,W] map")


@torch.no_grad()
def selection_weight_map(rendered_fidelity: torch.Tensor, rendered_closest: torch.Tensor) -> torch.Tensor:
    """Exact weight_maps.py postprocessing, not generic max/mean normalization.

    Both input images must be alpha composites with black background. Upstream
    clamps render outputs before combining them, normalizes only when max<0.9,
    and finally clips to [0,1]. A constant raw map below 0.9 becomes zero; this
    behavior is deliberately retained and should be reported in cache QA.
    """
    a, b = _map2d(rendered_fidelity), _map2d(rendered_closest)
    if a.shape != b.shape or not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError("invalid scalar renders")
    raw = 1 - a.clamp(0, 1) + b.clamp(0, 1)
    if raw.max() < 0.9:
        raw = (raw - raw.min()) / (raw.max() - raw.min() + 1e-6)
    return raw.clamp(0, 1)


def ssim_map(image: torch.Tensor, target: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    """Unreduced Graphdeco-style SSIM with sigma=1.5 and zero padding.

    Reimplemented from the stated formula, with the float32 kernel construction
    used by the official weighted_ssim. Returns [C,H,W] (or [B,C,H,W]).
    """
    if image.shape != target.shape or image.ndim not in (3, 4):
        raise ValueError("SSIM inputs must have identical [C,H,W] or [B,C,H,W] shapes")
    c = image.shape[-3]
    coordinates = torch.arange(window_size, dtype=torch.float64) - window_size // 2
    kernel1d = torch.exp(-coordinates.square() / (2 * 1.5**2)).float()
    kernel1d /= kernel1d.sum()
    kernel = (kernel1d[:, None] * kernel1d[None, :])[None, None].expand(c, 1, -1, -1).contiguous().to(image)
    mean_x = F.conv2d(image, kernel, padding=window_size // 2, groups=c)
    mean_y = F.conv2d(target, kernel, padding=window_size // 2, groups=c)
    variance_x = F.conv2d(image.square(), kernel, padding=window_size // 2, groups=c) - mean_x.square()
    variance_y = F.conv2d(target.square(), kernel, padding=window_size // 2, groups=c) - mean_y.square()
    covariance = F.conv2d(image * target, kernel, padding=window_size // 2, groups=c) - mean_x * mean_y
    return ((2 * mean_x * mean_y + 0.01**2) * (2 * covariance + 0.03**2)) / (
        (mean_x.square() + mean_y.square() + 0.01**2) * (variance_x + variance_y + 0.03**2)
    )


def splatsure_loss(
    render_hr: torch.Tensor,
    target_lr: torch.Tensor,
    target_sr: torch.Tensor,
    weight_map: torch.Tensor,
    config: Optional[SplatSuReConfig] = None,
    *,
    downsample: Optional[Callable[[torch.Tensor, tuple[int, int]], torch.Tensor]] = None,
) -> dict[str, torch.Tensor]:
    """Official loss values and gradients; all ground truths/maps are detached.

    ``downsample`` is an explicit protocol adaptation hook. Default is upstream
    bicubic-AA. Using area or a custom operator must be named in the run record.
    There is no weight-sum normalization. Upstream uses 1-mean(W*SSIM), not
    mean(W*(1-SSIM)); their gradients agree but scalar losses differ.
    """
    config = config or SplatSuReConfig()
    if render_hr.ndim != 3 or render_hr.shape[0] != 3 or target_sr.shape != render_hr.shape:
        raise ValueError("render_hr and target_sr must have identical [3,H,W] shape")
    if target_lr.ndim != 3 or target_lr.shape[0] != 3:
        raise ValueError("target_lr must have shape [3,h,w]")
    weight = _map2d(weight_map).detach().to(render_hr)
    if not torch.isfinite(weight).all() or torch.any(weight < 0) or torch.any(weight > 1):
        raise ValueError("SplatSuRe weight map must be finite and lie in [0,1]")
    if weight.shape != render_hr.shape[-2:]:
        weight = F.interpolate(weight[None, None], render_hr.shape[-2:], mode="nearest")[0, 0]
    lr_shape = tuple(target_lr.shape[-2:])
    if downsample is not None:
        render_lr = downsample(render_hr, lr_shape)
    elif config.lr_operator == "area":
        render_lr = F.interpolate(render_hr[None], size=lr_shape, mode="area")[0]
    else:
        render_lr = F.interpolate(render_hr[None], size=lr_shape, mode="bicubic", antialias=True, align_corners=False)[0]
    if render_lr.shape != target_lr.shape:
        raise ValueError("downsample output must exactly match the LR target shape")
    target_lr, target_sr = target_lr.detach().to(render_hr), target_sr.detach().to(render_hr)
    lr_l1 = (render_lr - target_lr).abs().mean()
    sr_l1 = ((render_hr - target_sr).abs() * weight[None]).mean()
    lr_ssim = ssim_map(render_lr, target_lr).mean()
    sr_ssim = (ssim_map(render_hr, target_sr) * weight[None]).mean()
    lr_loss = (1 - config.lambda_dssim) * lr_l1 + config.lambda_dssim * (1 - lr_ssim)
    sr_loss = (1 - config.lambda_dssim) * sr_l1 + config.lambda_dssim * (1 - sr_ssim)
    loss = (1 - config.gamma) * lr_loss + config.gamma * sr_loss
    return dict(loss=loss, lr_loss=lr_loss, sr_loss=sr_loss, lr_l1=lr_l1, sr_l1=sr_l1,
                lr_ssim=lr_ssim, sr_ssim=sr_ssim, mean_sr_weight=weight.mean())
