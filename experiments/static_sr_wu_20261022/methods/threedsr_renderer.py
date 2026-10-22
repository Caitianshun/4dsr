"""Pinned 3DSR Mip rasterizer applied to the deformed Wu state at time t.

Build the AUTHOR submodules/diff-gaussian-rasterization locally in place or in
an isolated package directory. Import under a unique alias to coexist with Wu.
No substitution with Wu's renderer or IE-SRGS's RADE renderer is permitted.
"""
from __future__ import annotations
import importlib.util
import math
from pathlib import Path
import sys


def raster_api(package_dir=None, upstream=None, binary_root=None):
    from .sequence_matters import sha256, verify_upstream
    from .threedsr import UPSTREAM_COMMIT
    root = Path(__file__).resolve().parents[3]
    upstream = verify_upstream(upstream or root / "tmp/static_sr_wu_20261022/upstream/threedsr", UPSTREAM_COMMIT)
    original = upstream / "submodules/diff-gaussian-rasterization/diff_gaussian_rasterization"
    if package_dir and binary_root:
        raise ValueError("Pass package_dir or binary_root, not both")
    default_build = root / "output/static_sr_wu_20261022/runtime/threedsr/lib/diff_gaussian_rasterization"
    if binary_root:
        package = Path(binary_root).resolve() / "diff_gaussian_rasterization"
    elif package_dir:
        package = Path(package_dir).resolve()
    else:
        package = default_build if default_build.is_dir() else original
    if not (package / "__init__.py").is_file() or sha256(package / "__init__.py") != sha256(original / "__init__.py"):
        raise RuntimeError("3DSR requires its pinned author Mip rasterizer Python API")
    if not list(package.glob("_C*.so")):
        raise RuntimeError("3DSR Mip CUDA extension is not compiled in its isolated package directory")
    name = "_static_sr_wu_threedsr_mip"
    if name in sys.modules:
        module = sys.modules[name]
        if Path(module.__file__).resolve().parent != package:
            raise RuntimeError("3DSR Mip loader already bound to a different build")
        return module
    spec = importlib.util.spec_from_file_location(name, package / "__init__.py", submodule_search_locations=[str(package)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    fields = set(module.GaussianRasterizationSettings._fields)
    if not {"kernel_size", "subpixel_offset"} <= fields or "require_depth" in fields:
        raise RuntimeError("Wrong 3DSR rasterizer ABI")
    return module


def compute_3d_filter(xyz, reference_cameras):
    """Author filter: minimum visible depth / maximum focal_x * sqrt(.2)."""
    import torch
    if not reference_cameras or xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) == 0:
        raise ValueError("3DSR filtering requires points and same-time training cameras")
    with torch.no_grad():
        distance = xyz.new_full((len(xyz),), 100000.)
        visible_any = torch.zeros(len(xyz), dtype=torch.bool, device=xyz.device)
        max_focal = 0.
        for camera in reference_cameras:
            transform = camera.world_view_transform.to(xyz)
            projected = xyz @ transform[:3, :3] + transform[3, :3]
            width, height = int(camera.image_width), int(camera.image_height)
            fx = width / (2 * math.tan(float(camera.FoVx) / 2))
            fy = height / (2 * math.tan(float(camera.FoVy) / 2))
            depth_valid = projected[:, 2] > .2
            depth = projected[:, 2].clamp_min(.001)
            x = projected[:, 0] / depth * fx + width / 2
            y = projected[:, 1] / depth * fy + height / 2
            visible = depth_valid & (x >= -.15*width) & (x <= 1.15*width) & (y >= -.15*height) & (y <= 1.15*height)
            distance[visible] = torch.minimum(distance[visible], depth[visible])
            visible_any |= visible
            max_focal = max(max_focal, fx)
        if not visible_any.any():
            raise ValueError("No Gaussian visible to 3DSR reference cameras")
        distance[~visible_any] = distance[visible_any].max()
        return (distance / max_focal * math.sqrt(.2))[:, None]


def apply_3d_filter(scales, opacity, filter_3d):
    if scales.ndim != 2 or scales.shape[1] != 3 or opacity.shape != (len(scales), 1) or filter_3d.shape != (len(scales), 1):
        raise ValueError("Invalid activated Mip state shape")
    before = scales.square()
    after = before + filter_3d.detach().square()
    return after.sqrt(), opacity * (before.prod(-1) / after.prod(-1)).sqrt()[:, None]


def render_state(camera, state, background, active_sh_degree, reference_cameras,
                 filter_3d=None, kernel_size=.1, debug=False, api=None):
    import torch
    if {"xyz", "scales", "rotation", "opacity", "sh"} - state.keys():
        raise ValueError("Missing activated Wu state")
    for reference in reference_cameras:
        if hasattr(camera, "time") and hasattr(reference, "time") and not math.isclose(float(camera.time), float(reference.time), abs_tol=1e-8):
            raise ValueError("3DSR filtering must use only same-time training cameras")
    xyz = state["xyz"]
    if filter_3d is None:
        filter_3d = compute_3d_filter(xyz.detach(), reference_cameras)
    scales, opacity = apply_3d_filter(state["scales"], state["opacity"], filter_3d)
    api = api or raster_api()
    screen = torch.zeros_like(xyz, requires_grad=True) + 0
    if torch.is_grad_enabled():
        screen.retain_grad()
    settings = api.GaussianRasterizationSettings(
        image_height=int(camera.image_height), image_width=int(camera.image_width),
        tanfovx=math.tan(float(camera.FoVx)/2), tanfovy=math.tan(float(camera.FoVy)/2),
        kernel_size=kernel_size,
        subpixel_offset=torch.zeros((int(camera.image_height), int(camera.image_width), 2), dtype=xyz.dtype, device=xyz.device),
        bg=background, scale_modifier=1., viewmatrix=camera.world_view_transform.to(xyz),
        projmatrix=camera.full_proj_transform.to(xyz), sh_degree=active_sh_degree,
        campos=camera.camera_center.to(xyz), prefiltered=False, debug=debug)
    values = api.GaussianRasterizer(raster_settings=settings)(
        means3D=xyz, means2D=screen, shs=state["sh"], colors_precomp=None,
        opacities=opacity, scales=scales, rotations=state["rotation"], cov3D_precomp=None)
    if len(values) != 2:
        raise RuntimeError("3DSR official Mip renderer must return RGB and radii only")
    rgb, radii = values
    return dict(render=rgb, radii=radii, visibility_filter=radii > 0,
                viewspace_points=screen, filter_3d=filter_3d)
