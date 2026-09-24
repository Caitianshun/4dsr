"""Generic frozen-LR data interface for the deployed author's SR4D training.

No HOI code, semantic masks, learned weights, or training HR decoding. This
module imports no upstream SR4D modules until ``configure_upstream`` is called.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)


def configure_upstream(path):
    root = Path(path).resolve()
    if not (root / 'utils/tasa_utils.py').is_file():
        raise ValueError('Require audited deployed SR4D, including tasa_utils.py')
    for name in ('scene', 'gaussian_renderer', 'arguments', 'utils'):
        module = sys.modules.get(name)
        if module is not None and getattr(module, '__file__', None):
            if root not in Path(module.__file__).resolve().parents:
                raise RuntimeError(f'Conflicting upstream module already imported: {name}')
    sys.path.insert(0, str(root))
    return root


class ManifestData:
    def __init__(self, manifest):
        self.path = Path(manifest).resolve()
        self.root = self.path.parent
        self.document = json.loads(self.path.read_text())
        d = self.document
        if d['schema'] != 'n3dv_dynamic_sr_pilot_v1' or d['scale'] != 4:
            raise ValueError('This interface is fixed to the audited generic x4 manifest')
        if not d['initialization']['train_only_lr']:
            raise ValueError('Initialization must be the shared training-LR-only point cloud')
        self.scene = d['scene']
        self.hr_wh = tuple(d['resolutions']['hr'])
        self.lr_wh = tuple(d['resolutions']['lr'])
        if tuple(x // 4 for x in self.hr_wh) != self.lr_wh:
            raise ValueError('Resolution mismatch')
        self.rows = [dict(row) for row in d['observations']]
        self.train_rows = [row for row in self.rows if row['split'] == 'train']
        if not self.train_rows:
            raise ValueError('Empty training split')
        if len({(r['camera_id'], r['frame_index']) for r in self.rows}) != len(self.rows):
            raise ValueError('Duplicate observations')
        for r in self.rows:
            if r['camera_id'] not in d['splits'][r['split']]:
                raise ValueError('Observation split conflicts with frozen camera split')
        self.cameras = d['cameras']
        centers = []
        for cid in d['splits']['train']:
            c = self.cameras[cid]
            w2c = np.asarray(c['w2c'], dtype=np.float64)
            if w2c.shape != (4, 4) or not np.isfinite(w2c).all():
                raise ValueError('Invalid camera pose')
            centers.append(np.linalg.inv(w2c)[:3, 3])
            for key in ('K_hr', 'K_lr'):
                k = np.asarray(c[key])
                if k.shape != (3, 3) or k[0, 1] != 0 or k[1, 0] != 0:
                    raise ValueError('SR4D rasterizer requires zero-skew intrinsics')
        centers = np.asarray(centers)
        self.extent = float(np.linalg.norm(centers - centers.mean(0), axis=1).max() * 1.1)
        self.init_path = self.root / d['initialization']['npz_path']
        if sha256(self.init_path) != d['initialization']['npz_sha256']:
            raise ValueError('Initialization hash mismatch')
        with np.load(self.init_path) as initial:
            self.points = np.asarray(initial['points'], np.float32)
            self.colors = np.asarray(initial['colors'], np.float32)
        if self.points.shape != self.colors.shape or self.points.shape[1] != 3:
            raise ValueError('Invalid shared initial cloud')
        if not np.isfinite(self.points).all() or not np.isfinite(self.colors).all():
            raise ValueError('Non-finite initialization')

    def verify_training_inputs(self):
        """Hash/check only training LR, never inspect held-out or HR image files."""
        for row in self.train_rows:
            use_float = 'lr_float_path' in row
            key = 'lr_float' if use_float else 'lr'
            p = self.root / row[key + '_path']
            if sha256(p) != row[key + '_sha256']:
                raise ValueError(f'LR hash mismatch: {p}')
            if use_float:
                a = np.load(p, mmap_mode='r')
                w, h = self.lr_wh
                if a.dtype != np.float32 or a.shape not in ((h, w, 3), (3, h, w)):
                    raise ValueError(f'Float LR format mismatch: {p}')
                if not np.isfinite(a).all() or a.min() < 0 or a.max() > 1:
                    raise ValueError(f'Invalid LR values: {p}')
                continue
            with Image.open(p) as im:
                if im.size != self.lr_wh or im.mode != 'RGB':
                    raise ValueError(f'LR shape/mode mismatch: {p}')
                im.verify()
        return {'manifest_sha256': sha256(self.path), 'training_lr_files': len(self.train_rows),
                'initialization_sha256': sha256(self.init_path), 'training_hr_decodes': 0,
                'heldout_image_reads': 0, 'points': len(self.points)}


def projection(k, width, height, device='cpu', near=.01, far=100.):
    """OpenCV zero-based centers -> SR4D ndc2Pix=(ndc+1)*size/2-.5."""
    p = torch.zeros((4, 4), device=device, dtype=torch.float32)
    p[0, 0], p[1, 1] = 2 * k[0][0] / width, 2 * k[1][1] / height
    p[0, 2], p[1, 2] = (2 * k[0][2] + 1) / width - 1, (2 * k[1][2] + 1) / height - 1
    p[2, 2], p[2, 3], p[3, 2] = far / (far - near), -far * near / (far - near), 1
    return p.T.contiguous()


class Camera:
    def __init__(self, data, row, device='cpu'):
        self.data, self.row = data, dict(row)
        self.calibration = data.cameras[row['camera_id']]
        self.image_width, self.image_height = data.hr_wh
        self.image_width_down, self.image_height_down = data.lr_wh
        k = self.calibration['K_hr']
        self.FoVx = 2 * math.atan(self.image_width / (2 * k[0][0]))
        self.FoVy = 2 * math.atan(self.image_height / (2 * k[1][1]))
        self.image_name = f"{row['camera_id']}_{row['frame_index']:04d}"
        self.fid = torch.tensor([row['time']], dtype=torch.float32, device=device)
        self.world_view_transform = torch.tensor(self.calibration['w2c'], dtype=torch.float32, device=device).T.contiguous()
        self.full_proj_transform = self.world_view_transform @ projection(k, *data.hr_wh, device=device)
        self.full_proj_transform_lr = self.world_view_transform @ projection(self.calibration['K_lr'], *data.lr_wh, device=device)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        self._lr = self._edge = None

    @property
    def original_image(self):
        raise RuntimeError('Training HR is unavailable; use separate endpoint evaluator')

    def _load_lr(self):
        if self._lr is not None:
            return
        if self.row['split'] != 'train':
            raise RuntimeError('Held-out LR must not be decoded by the training camera')
        from kornia.filters import spatial_gradient
        use_float = 'lr_float_path' in self.row
        key = 'lr_float' if use_float else 'lr'
        path = self.data.root / self.row[key + '_path']
        if sha256(path) != self.row[key + '_sha256']:
            raise ValueError('Frozen LR changed after preflight')
        if use_float:
            a = np.load(path)
            self._lr = torch.from_numpy(np.array(a, copy=True))
            if self._lr.shape[-1] == 3:
                self._lr = self._lr.permute(2, 0, 1)
            self._lr = self._lr.contiguous()
        else:
            with Image.open(path) as im:
                a = np.array(im.convert('RGB'), copy=True)
            self._lr = torch.from_numpy(a).permute(2, 0, 1).float() / 255
        edge = spatial_gradient(self._lr[None], order=1)[0].abs().sum(1).mean(0)
        edge = (edge - edge.min()) / (edge.max() - edge.min()).clamp_min(1e-8)
        self._edge = edge[None].repeat(3, 1, 1)

    @property
    def downsampled_image(self):
        self._load_lr()
        return self._lr

    @property
    def edgemap(self):
        self._load_lr()
        return self._edge

    def load2device(self, data_device='cuda'):
        self._load_lr()
        self._lr, self._edge = self._lr.to(data_device), self._edge.to(data_device)
        for name in ('fid', 'world_view_transform', 'full_proj_transform', 'full_proj_transform_lr', 'camera_center'):
            setattr(self, name, getattr(self, name).to(data_device))


def camera_geometry_test(data, device='cpu'):
    errors = []
    seen = set()
    for row in data.train_rows:
        if row['camera_id'] in seen:
            continue
        seen.add(row['camera_id'])
        cam = Camera(data, row, device)
        w = torch.tensor(cam.calibration['w2c'], device=device, dtype=torch.float32)
        z = torch.tensor([[.13, .21, 2., 1.], [-.3, .2, 3., 1.]], device=device)
        world = z @ torch.linalg.inv(w).T
        for key, wh, full in [('K_hr', data.hr_wh, cam.full_proj_transform), ('K_lr', data.lr_wh, cam.full_proj_transform_lr)]:
            clip = world @ full
            pixels = ((clip[:, :2] / clip[:, 3:] + 1) * torch.tensor(wh, device=device) - 1) * .5
            ref = z[:, :3] @ torch.tensor(cam.calibration[key], device=device, dtype=torch.float32).T
            errors.append(float((pixels - ref[:, :2] / ref[:, 2:]).abs().max()))
    if max(errors) >= .001:
        raise ValueError(f'Camera projection error: {max(errors)} pixels')
    return max(errors)


class GenericScene:
    """Only the Scene data/save interface; author training body stays intact."""
    def __init__(self, args, gaussians, load_iteration=None, shuffle=True, resolution_scales=(1.,), stage='lr', data=None):
        from utils.graphics_utils import BasicPointCloud
        self.model_path, self.gaussians = args.model_path, gaussians
        self.data = data
        self.cameras_extent = self.data.extent
        self.loaded_iter = load_iteration
        if load_iteration == -1:
            candidates = (Path(self.model_path) / 'point_cloud').glob('iteration_*')
            self.loaded_iter = max(int(p.name.split('_')[-1]) for p in candidates)
        rows = list(self.data.train_rows)
        if shuffle:
            random.shuffle(rows)
        self.train_cameras = {1.: [Camera(self.data, row) for row in rows]}
        self.test_cameras = {1.: []}
        if self.loaded_iter:
            path = Path(self.model_path) / 'point_cloud' / f'iteration_{self.loaded_iter}' / 'point_cloud.ply'
            gaussians.load_ply(str(path), og_number_points=len(self.data.points))
        else:
            cloud = BasicPointCloud(self.data.points, self.data.colors, np.zeros_like(self.data.points))
            gaussians.create_from_pcd(cloud, self.cameras_extent)

    def getTrainCameras(self, scale=1.):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.):
        return []

    def save(self, iteration, stage='lr'):
        if stage not in ('lr', 'hr'):
            raise ValueError(stage)
        name = 'point_cloud_hr' if stage == 'hr' else 'point_cloud'
        self.gaussians.save_ply(str(Path(self.model_path) / name / f'iteration_{iteration}' / 'point_cloud.ply'))


def load_endpoint(upstream, run, stage, iteration, coarse_iteration=None):
    """Load original SR4D PLY + deformation files for external evaluation."""
    configure_upstream(upstream)
    from scene import GaussianModel, DeformModel
    run = Path(run)
    gaussian = GaussianModel(3)
    point_dir = 'point_cloud_hr' if stage == 'fine' else 'point_cloud'
    gaussian.load_ply(str(run / point_dir / f'iteration_{iteration}' / 'point_cloud.ply'))
    coarse = DeformModel(False, False)
    fine = None
    if stage == 'fine':
        metadata = json.loads((run / 'deform_hr' / f'iteration_{iteration}' / 'stage_metadata.json').read_text())
        coarse_iteration = metadata['coarse_iteration']
        fine = DeformModel(False, False)
        fine.load_weights_hr(str(run), iteration)
        fine.deform.requires_grad_(False).eval()
    else:
        coarse_iteration = iteration
    coarse.load_weights_lr(str(run), coarse_iteration)
    coarse.deform.requires_grad_(False).eval()
    return gaussian, coarse, fine


@torch.no_grad()
def render_endpoint(gaussian, coarse, fine, camera):
    from argparse import Namespace
    from gaussian_renderer import render
    # Evaluation does not need to decode the camera's input image.
    for name in ('fid', 'world_view_transform', 'full_proj_transform', 'full_proj_transform_lr', 'camera_center'):
        setattr(camera, name, getattr(camera, name).cuda())
    time = camera.fid[None].expand(len(gaussian.get_xyz), -1)
    values = coarse.step(gaussian.get_xyz.detach(), time)
    if fine is not None:
        values = tuple(a + b for a, b in zip(values, fine.step(gaussian.get_xyz.detach(), time)))
    pipe = Namespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
    return render(camera, gaussian, pipe, torch.zeros(3, device='cuda'), *values, False)
