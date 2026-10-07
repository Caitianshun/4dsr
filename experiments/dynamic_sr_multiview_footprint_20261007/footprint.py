"""Differentiable HR reprojection followed by the registered LR degradation.

Pixel coordinates are zero-based pixel-centre indices, not NDC. Camera objects
use the existing 4DGS ``intrinsics`` and transposed ``world_view_transform``;
dictionary cameras use the manifest's K/K_hr, w2c and c2w convention. Depth is
camera-axis z. The mip filter is an area-prefiltered approximation, not an exact
sensor integral. Every validity decision and LOD decision is detached.

Only source RGB and the current target axial depth acquire gradients. Frozen
source support, frozen edge support, and frozen LR weights remain separate.
"""
from __future__ import annotations

from functools import lru_cache
import math

import torch
import torch.nn.functional as F


def _hw(value, name):
    if value.ndim == 3 and value.shape[0] == 1:
        value = value[0]
    if value.ndim != 2:
        raise ValueError(f"{name} must have shape (H,W) or (1,H,W)")
    return value


def calibration(camera, *, device, dtype):
    """Read the project's existing camera convention without inferring a K."""
    if isinstance(camera, dict):
        k = camera.get('K', camera.get('K_hr'))
        if k is None:
            raise ValueError('Dictionary camera requires an HR K or K_hr')
        w2c = camera.get('w2c')
        c2w = camera.get('c2w')
        if w2c is None and c2w is None:
            raise ValueError('Dictionary camera requires w2c or c2w')
    else:
        if not hasattr(camera, 'intrinsics'):
            raise ValueError('Project camera must carry explicitly resized intrinsics')
        k = camera.intrinsics
        w2c = camera.world_view_transform.transpose(0, 1)
        c2w = None
    k = torch.as_tensor(k, device=device, dtype=dtype).detach()
    if k.shape != (3, 3) or not bool(torch.isfinite(k).all()):
        raise ValueError('Non-finite or incorrectly shaped intrinsics')
    w2c = None if w2c is None else torch.as_tensor(w2c, device=device, dtype=dtype).detach()
    c2w = None if c2w is None else torch.as_tensor(c2w, device=device, dtype=dtype).detach()
    if w2c is None:
        w2c = torch.linalg.inv(c2w)
    if c2w is None:
        c2w = torch.linalg.inv(w2c)
    if (w2c.shape != (4, 4) or c2w.shape != (4, 4) or
            not bool(torch.isfinite(w2c).all() & torch.isfinite(c2w).all())):
        raise ValueError('Non-finite or incorrectly shaped camera transforms')
    return dict(K=k, w2c=w2c, c2w=c2w)


@lru_cache(maxsize=16)
def _pixels(height, width, device_string, dtype):
    yy, xx = torch.meshgrid(torch.arange(height, device=device_string, dtype=dtype),
                            torch.arange(width, device=device_string, dtype=dtype), indexing='ij')
    return torch.stack((xx, yy, torch.ones_like(xx)), -1)


def project(camera_src, camera_tgt, z_tgt, *, epsilon=1e-8):
    """Backproject target HR axial depth and project into the source HR grid.

    Invalid values are replaced *before* multiplication/division. Returned xy
    is finite, but its detached validity flag must accompany every sampler.
    Current depth retains the source-coordinate derivative even when cameras
    and all masks are fixed. ``source_z`` is the source camera's axial depth.
    """
    z_tgt = _hw(z_tgt, 'z_tgt')
    if not z_tgt.is_floating_point():
        raise TypeError('Depth must be floating point')
    source = calibration(camera_src, device=z_tgt.device, dtype=z_tgt.dtype)
    target = calibration(camera_tgt, device=z_tgt.device, dtype=z_tgt.dtype)
    valid_z = (torch.isfinite(z_tgt.detach()) & (z_tgt.detach() > epsilon))
    safe_z = torch.where(valid_z, z_tgt, torch.ones_like(z_tgt))
    pixels = _pixels(*z_tgt.shape, str(z_tgt.device), z_tgt.dtype)
    rays = pixels @ torch.linalg.inv(target['K']).T
    xyz_tgt = rays * safe_z[..., None]
    relative = source['w2c'] @ target['c2w']
    xyz_src = xyz_tgt @ relative[:3, :3].T + relative[:3, 3]
    xyz_finite = torch.isfinite(xyz_src.detach()).all(-1)
    safe_xyz = torch.where(xyz_finite[..., None], xyz_src, torch.zeros_like(xyz_src))
    raw_screen = safe_xyz @ source['K'].T
    # A finite XYZ can still overflow in K multiplication. Replacing Inf only
    # after division leaves 0*Inf in DivBackward, even when the final mask is 0.
    # Guard the whole screen before numerator/denominator enter any division.
    screen_finite = torch.isfinite(raw_screen.detach()).all(-1)
    screen = torch.where(screen_finite[..., None], raw_screen, torch.zeros_like(raw_screen))
    source_z = safe_xyz[..., 2]
    denom = screen[..., 2]
    denominator_valid = torch.isfinite(denom.detach()) & (denom.detach().abs() > epsilon)
    # Finite numerator and denominator can themselves yield an overflowing
    # quotient. This is a representability guard, not a visibility tolerance.
    with torch.no_grad():
        ratio_limit = (torch.finfo(screen.dtype).max / 4) * denom.detach().abs().clamp_max(1.)
        quotient_finite = (screen.detach()[..., :2].abs() <= ratio_limit[..., None]).all(-1)
    denominator_valid = denominator_valid & quotient_finite
    screen = torch.where(denominator_valid[..., None], screen, torch.zeros_like(screen))
    safe_denom = torch.where(denominator_valid, denom, torch.ones_like(denom))
    xy_raw = screen[..., :2] / safe_denom[..., None]
    xy_finite = torch.isfinite(xy_raw.detach()).all(-1)
    valid = (valid_z & xyz_finite & screen_finite & denominator_valid & xy_finite &
             torch.isfinite(source_z.detach()) & (source_z.detach() > epsilon)).detach()
    xy = torch.where(xy_finite[..., None], xy_raw, torch.zeros_like(xy_raw))
    return dict(xy=xy, source_z=source_z, projected_z=source_z, valid=valid,
                target_depth_valid=valid_z.detach())


def _difference(value, axis):
    """Centred finite difference, one-sided at the full-frame boundary."""
    if value.shape[axis] == 1:
        return torch.zeros_like(value)
    front = value.select(axis, 1) - value.select(axis, 0)
    back = value.select(axis, value.shape[axis] - 1) - value.select(axis, value.shape[axis] - 2)
    if value.shape[axis] == 2:
        return torch.stack((front, back), axis)
    middle = (value.narrow(axis, 2, value.shape[axis] - 2) -
              value.narrow(axis, 0, value.shape[axis] - 2)) * .5
    return torch.cat((front.unsqueeze(axis), middle, back.unsqueeze(axis)), axis)


@torch.no_grad()
def projection_footprint(xy, valid=None):
    """Largest singular value of d(source_pixel)/d(target_pixel), detached.

    Finite differences include the full frame. Invalid projected neighbours
    never enter division or SVD; their finite safe coordinates may affect a
    nearby LOD choice, which is a conservative approximation near discontinuity.
    """
    safe = torch.nan_to_num(xy.detach(), nan=0., posinf=0., neginf=0.)
    dx, dy = _difference(safe, 1), _difference(safe, 0)
    a, d, b = dx.square().sum(-1), dy.square().sum(-1), (dx * dy).sum(-1)
    discriminant = torch.clamp((a-d).square() + 4*b.square(), min=0)
    sigma = torch.sqrt(torch.clamp(.5 * (a+d+torch.sqrt(discriminant)), min=0))
    sigma = torch.nan_to_num(sigma, nan=1., posinf=1e6, neginf=1.).clamp_min(1.)
    if valid is not None:
        sigma = torch.where(valid.detach().bool(), sigma, torch.ones_like(sigma))
    return sigma.detach()


def build_area_mipmap(source_rgb, source_valid=None):
    """Rebuild a differentiable area pyramid from the *current* source RGB.

    Each level is an area reduction of the original HR grid, with ceil-sized
    odd dimensions and exact half-pixel coordinate conversion at sampling time.
    Area support uses the identical adaptive pooling bins with a maximum of
    invalid indicators, rather than any signed-kernel mask average.
    """
    if source_rgb.ndim != 3 or not source_rgb.is_floating_point():
        raise ValueError('source_rgb must be floating CHW')
    height, width = source_rgb.shape[-2:]
    finite = torch.isfinite(source_rgb.detach()).all(0)
    valid = finite if source_valid is None else (finite & _hw(source_valid, 'source_valid').detach().bool())
    if valid.shape != (height, width):
        raise ValueError('Source validity shape does not match RGB')
    safe = torch.where(torch.isfinite(source_rgb.detach()), source_rgb, torch.zeros_like(source_rgb))
    levels, masks = [safe], [valid.detach()]
    level = 1
    while levels[-1].shape[-2:] != (1, 1):
        size = (max(1, math.ceil(height / 2**level)), max(1, math.ceil(width / 2**level)))
        levels.append(F.interpolate(safe[None], size=size, mode='area')[0])
        # F.interpolate(mode=area) is adaptive_avg_pool2d; max pooling uses the
        # same floor/ceil bins and therefore rejects any contributing invalid.
        bad = F.adaptive_max_pool2d((~valid)[None, None].float(), size)[0, 0] > 0
        masks.append((~bad).detach())
        level += 1
    return dict(levels=levels, valid=masks, source_size=(height, width),
                source_tensor_id=id(source_rgb),
                source_finite=finite.detach())


@torch.no_grad()
def bilinear_support(valid, xy):
    """Require every bilinear neighbour with a strictly nonzero coefficient.

    A zero coefficient at an exact integer location does not require the next
    pixel. Padding outside the source is invalid whenever it has nonzero weight.
    """
    valid = _hw(valid, 'valid').detach().bool()
    height, width = valid.shape
    finite = torch.isfinite(xy.detach()).all(-1)
    safe = torch.nan_to_num(xy.detach(), nan=0., posinf=0., neginf=0.)
    # Replacing huge/outside coordinates before conversion avoids int overflow.
    bounded = ((safe[..., 0] >= -1) & (safe[..., 0] <= width) &
               (safe[..., 1] >= -1) & (safe[..., 1] <= height))
    safe = torch.where(bounded[..., None], safe, torch.zeros_like(safe))
    x0, y0 = safe[..., 0].floor().long(), safe[..., 1].floor().long()
    fx, fy = safe[..., 0] - x0, safe[..., 1] - y0
    support = finite & bounded
    flat = valid.reshape(-1)
    for ox, oy, coefficient in ((0, 0, (1-fx)*(1-fy)), (1, 0, fx*(1-fy)),
                                 (0, 1, (1-fx)*fy), (1, 1, fx*fy)):
        xx, yy = x0 + ox, y0 + oy
        inside = (xx >= 0) & (xx < width) & (yy >= 0) & (yy < height)
        indices = yy.clamp(0, height-1) * width + xx.clamp(0, width-1)
        supported = inside & flat[indices]
        support &= (coefficient == 0) | supported
    return support.detach()


def mip_sample(source_rgb, xy, footprint=None, source_valid=None, pyramid=None):
    """Differentiable bilinear/linear-between-level sampling with full support."""
    if xy.ndim != 3 or xy.shape[-1] != 2:
        raise ValueError('xy must have shape (H_target,W_target,2)')
    if pyramid is None:
        pyramid = build_area_mipmap(source_rgb, source_valid)
    if pyramid.get('source_tensor_id') != id(source_rgb):
        raise ValueError('Pyramid must be rebuilt from this same current source RGB tensor')
    height, width = pyramid['source_size']
    if tuple(source_rgb.shape[-2:]) != (height, width):
        raise ValueError('Pyramid/source shape mismatch')
    finite_xy = torch.isfinite(xy.detach()).all(-1)
    # grid_sample is never given NaN/Inf or astronomically large coordinates.
    inside = (finite_xy & (xy.detach()[..., 0] >= 0) & (xy.detach()[..., 0] <= width-1) &
              (xy.detach()[..., 1] >= 0) & (xy.detach()[..., 1] <= height-1))
    safe_xy = torch.where(inside[..., None], xy, torch.zeros_like(xy))
    if footprint is None:
        footprint = projection_footprint(xy, finite_xy)
    with torch.no_grad():
        scale = torch.nan_to_num(footprint.detach(), nan=1., posinf=1e6, neginf=1.).clamp_min(1.)
        lod = torch.log2(scale).clamp(0, len(pyramid['levels'])-1).detach()
        minimum, maximum = int(lod.min().floor()), int(lod.max().ceil())
    output = source_rgb.new_zeros((source_rgb.shape[0], *xy.shape[:2]))
    support = inside.detach().clone()
    for level in range(minimum, maximum+1):
        alpha = (1 - (lod-level).abs()).clamp_min(0).detach()
        image = pyramid['levels'][level]
        h, w = image.shape[-2:]
        ratio = xy.new_tensor((w/width, h/height))
        mapped = (safe_xy+.5) * ratio - .5
        # align_corners=False: pixel centres 0..W-1 map to (2u+1)/W - 1.
        grid = (mapped+.5) * mapped.new_tensor((2/w, 2/h)) - 1
        sampled = F.grid_sample(image[None], grid[None], mode='bilinear',
                                padding_mode='zeros', align_corners=False)[0]
        output = output + sampled * alpha[None]
        # Check the actual grid_sample unnormalisation, including floating-point
        # roundoff at integer locations, instead of assuming mapped == sampled.
        sampled_xy = ((grid.detach()+1) * grid.new_tensor((w, h)) - 1) * .5
        level_support = bilinear_support(pyramid['valid'][level], sampled_xy)
        support &= (alpha == 0) | level_support
    return dict(rgb=output, valid=support.detach(), lod=lod,
                diagnostics=dict(projected_inside_fraction=inside.float().mean().detach(),
                                 warp_support_fraction=support.float().mean().detach(),
                                 source_finite_fraction=pyramid['source_finite'].float().mean().detach(),
                                 min_lod=lod.min(), max_lod=lod.max(), mean_lod=lod.mean()))


@lru_cache(maxsize=16)
def _degradation_axis_support(n, m):
    """Reuse the old D0 impulse-axis construction; retain only nonzero indices.

    This is not Q/pseudoinverse construction: no Gram matrix or solve is made.
    PyTorch CPU float64 records the exact nonzero bicubic-AA positions including
    border renormalisation and negative lobes, as in the existing D0 operator.
    """
    impulses = torch.eye(n, dtype=torch.float64).reshape(1, n, 1, n)
    matrix = F.interpolate(impulses, size=(1, m), mode='bicubic',
                           align_corners=False, antialias=True)[0, :, 0].T.contiguous()
    nonzero = matrix != 0
    indices = torch.zeros((m, int(nonzero.sum(1).max())), dtype=torch.long)
    used = torch.zeros_like(indices, dtype=torch.bool)
    for q in range(m):
        row = torch.nonzero(nonzero[q], as_tuple=False).flatten()
        indices[q, :len(row)] = row
        used[q, :len(row)] = True
    return indices, used


@lru_cache(maxsize=32)
def _runtime_axis_support(n, m, device_string):
    indices, used = _degradation_axis_support(n, m)
    return indices.to(device_string), used.to(device_string)


@torch.no_grad()
def degradation_support(valid_hr, lr_size):
    """LR support is the logical AND over every nonzero separable D0 tap."""
    valid_hr = _hw(valid_hr, 'valid_hr').detach().bool()
    height, width = valid_hr.shape
    lh, lw = map(int, lr_size)
    if min(lh, lw) <= 0 or lh > height or lw > width:
        raise ValueError('LR dimensions must be positive and no larger than HR')
    xindices, xused = _runtime_axis_support(width, lw, str(valid_hr.device))
    yindices, yused = _runtime_axis_support(height, lh, str(valid_hr.device))
    invalid = ~valid_hr
    xbad = invalid[:, xindices] & xused[None]
    reduced = xbad.any(-1)
    ybad = reduced[yindices, :] & yused[:, :, None]
    return (~ybad.any(1)).detach()


def downsample(x, lr_size):
    """The unchanged full-frame bicubic-AA D0 followed by the mother clamp."""
    if x.ndim != 3:
        raise ValueError('Expected CHW')
    safe = torch.where(torch.isfinite(x.detach()), x, torch.zeros_like(x))
    raw = F.interpolate(safe[None], size=tuple(lr_size), mode='bicubic',
                        align_corners=False, antialias=True)[0]
    return raw.clamp(0, 1), ((raw.detach() < 0) | (raw.detach() > 1)).float().mean().detach()


def edge_prediction(source_rgb, z_target, camera_src, camera_tgt, lr_size,
                    frozen_mask_hr=None, source_valid_hr=None, pyramid=None):
    """Build one directed X edge; loss normalization belongs to losses.py.

    A prebuilt pyramid is reusable for both edges leaving a source view. It must
    come from this same current RGB tensor and its frozen source support. The
    returned LR mask excludes any invalid HR sample touched by actual D0.
    """
    projection = project(camera_src, camera_tgt, z_target)
    footprint = projection_footprint(projection['xy'], projection['valid'])
    warped = mip_sample(source_rgb, projection['xy'], footprint, source_valid_hr, pyramid)
    valid_hr = projection['valid'] & warped['valid']
    frozen_fraction = source_rgb.new_tensor(1.)
    if frozen_mask_hr is not None:
        frozen = _hw(frozen_mask_hr, 'frozen_mask_hr').detach().bool()
        if frozen.shape != valid_hr.shape:
            raise ValueError('Frozen edge mask/target HR shape mismatch')
        valid_hr = valid_hr & frozen
        frozen_fraction = frozen.float().mean()
    valid_hr = valid_hr.detach()
    prediction, clamp_fraction = downsample(warped['rgb'], lr_size)
    valid_lr = degradation_support(valid_hr, lr_size)
    diagnostics = dict(warped['diagnostics'],
        current_depth_valid_fraction=projection['target_depth_valid'].float().mean().detach(),
        current_projection_valid_fraction=projection['valid'].float().mean().detach(),
        frozen_hr_support_fraction=frozen_fraction.detach(),
        combined_hr_support_fraction=valid_hr.float().mean().detach(),
        valid_lr_fraction=valid_lr.float().mean().detach(),
        clamp_fraction=clamp_fraction)
    return dict(prediction=prediction, valid_lr=valid_lr, valid_hr=valid_hr,
                warped_hr=warped['rgb'], diagnostics=diagnostics)
