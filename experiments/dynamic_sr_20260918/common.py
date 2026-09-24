"""Isolated, auditable adapters for the N3DV dynamic-SR pilot.

The installed upstream 4DGaussians is imported, never edited. All images and
initial geometry are taken from the new prepared public dataset manifest.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

UPSTREAM = Path(os.environ.get('FOURDSR_UPSTREAM', '/home/cai_tianshun/Project/4dgs')).expanduser()
sys.path.insert(0, str(UPSTREAM))
from arguments import ModelHiddenParams, OptimizationParams
from gaussian_renderer import render
from scene.gaussian_model import GaussianModel
from utils.graphics_utils import BasicPointCloud


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    tmp.replace(path)


def defaults():
    parser = argparse.ArgumentParser()
    hp = ModelHiddenParams(parser)
    op = OptimizationParams(parser)
    a = parser.parse_args([])
    h, o = hp.extract(a), op.extract(a)
    # Match the released N3DV representation, retaining all attribute heads.
    h.kplanes_config = dict(grid_dimensions=2, input_coordinate_dim=4,
                           output_coordinate_dim=16, resolution=[64, 64, 64, 150])
    h.multires = [1, 2]
    h.defor_depth, h.net_width = 0, 128
    h.no_do = h.no_dshs = h.no_ds = False
    h.time_smoothness_weight, h.l1_time_planes, h.plane_tv_weight = .001, .0001, .0002
    o.opacity_reset_interval = 60000
    o.lambda_dssim = 0.0
    return h, o


def new_model(points, colors, camera_centers, hidden=None, optim=None):
    h, o = defaults()
    if hidden:
        h.__dict__.update(hidden)
    if optim:
        o.__dict__.update(optim)
    g = GaussianModel(3, h)
    lo, hi = np.min(points, axis=0), np.max(points, axis=0)
    margin = np.maximum((hi - lo) * .05, .01)
    g._deformation.deformation_net.set_aabb(hi + margin, lo - margin)
    center = camera_centers.mean(0)
    extent = float(np.linalg.norm(camera_centers - center, axis=1).max() * 1.1)
    g.create_from_pcd(BasicPointCloud(points, colors, np.zeros_like(points)), extent, 300)
    g.training_setup(o)
    return g, h, o, extent


def save_checkpoint(path, g, h, o, metadata):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = str(path) + '.tmp'
    torch.save(dict(model=g.capture(), hidden=vars(h), optim=vars(o), metadata=metadata,
                    rng=dict(torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all(),
                             numpy=np.random.get_state(), python=random.getstate())), tmp)
    Path(tmp).replace(path)


def load_checkpoint(path):
    ck = torch.load(path, map_location='cuda', weights_only=False)
    h, o = defaults()
    h.__dict__.update(ck['hidden'])
    o.__dict__.update(ck['optim'])
    g = GaussianModel(3, h)
    g._deformation = g._deformation.cuda()
    g.restore(ck['model'], o)
    return g, h, o, ck


def appearance_parameters(g):
    # Shared HexPlanes/features, opacity, shape, and canonical xyz are excluded.
    return [g._features_dc, g._features_rest] + [
        p for name, p in g._deformation.named_parameters()
        if 'deformation_net.shs_deform.' in name]


def named_parameters(g):
    out = {name: getattr(g, name) for name in
           ['_xyz', '_features_dc', '_features_rest', '_scaling', '_rotation', '_opacity']}
    out.update({'deformation.' + n: p for n, p in g._deformation.named_parameters()})
    return out


def image_tensor(path, device='cuda'):
    a = np.array(Image.open(path).convert('RGB'), copy=True)
    return torch.from_numpy(a).permute(2, 0, 1).to(device=device, dtype=torch.float32) / 255


def save_image(path, x):
    a = x.detach().clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).cpu().numpy()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(a).save(path)


def downsample(x, size):
    # Exactly the floating-point operator used to construct the quantized LR.
    return F.interpolate(x[None], size=size, mode='bicubic',
                         align_corners=False, antialias=True)[0].clamp(0, 1)


PIPE = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)


def render_image(g, camera, stage='fine'):
    return render(camera, g, PIPE, torch.zeros(3, device='cuda'), stage=stage)


def resized_camera(camera, height, width):
    result = copy.copy(camera)
    old_h, old_w = camera.image_height, camera.image_width
    result.image_height, result.image_width = int(height), int(width)
    if hasattr(camera, 'intrinsics'):
        k = np.asarray(camera.intrinsics).copy()
        sx, sy = width / old_w, height / old_h
        k[0, :2] *= sx
        k[1, :2] *= sy
        k[0, 2] = (k[0, 2] + .5) * sx - .5
        k[1, 2] = (k[1, 2] + .5) * sy - .5
        result.intrinsics = k
        result.projection_matrix = camera.projection_matrix.clone()
        result.projection_matrix[2, 0] = (2 * k[0, 2] + 1 - width) / width
        result.projection_matrix[2, 1] = (2 * k[1, 2] + 1 - height) / height
        result.full_proj_transform = result.world_view_transform @ result.projection_matrix
    return result


@torch.no_grad()
def dynamic_xyz(g, time, indices=None):
    xyz = g._xyz if indices is None else g._xyz[indices]
    scales = g._scaling if indices is None else g._scaling[indices]
    rot = g._rotation if indices is None else g._rotation[indices]
    opacity = g._opacity if indices is None else g._opacity[indices]
    sh = g.get_features if indices is None else g.get_features[indices]
    t = torch.full((len(xyz), 1), float(time), device='cuda')
    return g._deformation(xyz, scales, rot, opacity, sh, t)[0]


def mse_psnr(x, y, mask=None):
    err = (x - y).square().mean(0)
    if mask is not None:
        err = err[mask]
    mse = float(err.mean())
    return mse, float(-10 * np.log10(max(mse, 1e-12)))
