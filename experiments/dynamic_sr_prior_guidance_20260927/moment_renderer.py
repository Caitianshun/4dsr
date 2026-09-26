"""Unclamped [alpha, axial-z moment, squared-z moment] auxiliary rendering.

Never called by the current RGB training loop. Reduce moments before division.
"""
import math
import torch
from motion_model import packed_covariance


def render_moments(camera,xyz,cov,opacity,lr_size=None):
    from diff_gaussian_rasterization import GaussianRasterizationSettings,GaussianRasterizer
    view=camera.world_view_transform.to(xyz.device)
    z=(torch.cat((xyz,torch.ones_like(xyz[:,:1])),dim=-1)@view)[:,2]
    colors=torch.stack((torch.ones_like(z),z,z.square()),dim=-1).contiguous()
    settings=GaussianRasterizationSettings(image_height=int(camera.image_height),image_width=int(camera.image_width),
        tanfovx=math.tan(camera.FoVx*.5),tanfovy=math.tan(camera.FoVy*.5),bg=torch.zeros(3,device=xyz.device),
        scale_modifier=1.,viewmatrix=view,projmatrix=camera.full_proj_transform.to(xyz.device),sh_degree=0,
        campos=camera.camera_center.to(xyz.device),prefiltered=False,debug=False)
    image,radii,native=GaussianRasterizer(raster_settings=settings)(means3D=xyz,means2D=torch.zeros_like(xyz),
        shs=None,colors_precomp=colors,opacities=opacity.contiguous(),scales=None,rotations=None,cov3D_precomp=packed_covariance(cov))
    moments=image
    if lr_size is not None:
        # Same interpolation as D, but no RGB-range clamp for physical moments.
        moments=torch.nn.functional.interpolate(image[None],size=lr_size,mode='bicubic',align_corners=False,antialias=True)[0]
    alpha=moments[0];depth=moments[1]/alpha.clamp_min(1e-6)
    variance=(moments[2]/alpha.clamp_min(1e-6)-depth.square()).clamp_min(0)
    return dict(moments=moments,alpha=alpha,expected_z=depth,variance_z=variance,native_depth=native,radii=radii)
