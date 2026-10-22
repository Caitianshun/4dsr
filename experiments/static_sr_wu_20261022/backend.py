"""Wu 843d5ac shared deformation backend. Imports CUDA only when constructed.

This replaces the STATIC representation of each cited method. It is an explicit
adaptation, not an execution of the unmodified paper. No affine, culling, w4 or
project-specific learned network is installed here.
"""
from __future__ import annotations
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from common import ROOT, checked, identity, read, resolve, sha, training_manifest


class WuBackend:
    def __init__(self, config, checkpoint=None, fresh=False):
        import numpy as np
        import torch
        self.torch = torch
        for p in (ROOT / 'experiments/dynamic_sr_cull_retrain_20261009',):
            sys.path.insert(0, str(p))
        import rt
        import cullretrain as cr
        from scene.gaussian_model import GaussianModel
        from n3dv_data import load_initial_points, N3DVPreparedDataset
        self.runtime = rt.check_runtime()
        self.cr, self.native, self.prefix = cr, cr.native, cr.prefix
        self.config = config
        self.manifest, self.observations = training_manifest(checked(config['manifest']))
        self.dataset = N3DVPreparedDataset(self.manifest, 'train', 'lr', device='cpu')
        self.indices = {(o['camera_id'], o['frame_index']): i for i, o in enumerate(self.dataset.observations)}
        cfg = config['wu_configuration']
        self.h = SimpleNamespace(**cfg['ModelHiddenParams'])
        self.opt = SimpleNamespace(**cfg['OptimizationParams'])
        # Shared Gaussian LR schedule follows the source SR papers' 30k clock.
        # Wu-specific deformation/grid schedules use the same explicit clock.
        self.opt.position_lr_max_steps = 30000
        self.pipe = SimpleNamespace(**cfg['PipelineParams'])
        self.g = GaussianModel(cfg['ModelParams']['sh_degree'], self.h)
        self.loaded_checkpoint_identity = None
        if fresh == bool(checkpoint):
            raise ValueError('Specify exactly one of fresh or checkpoint')
        if checkpoint:
            p = checked(checkpoint) if isinstance(checkpoint, dict) else resolve(checkpoint)
            payload = torch.load(p, map_location='cuda', weights_only=False)
            if payload.get('schema') not in ('static_sr_wu_checkpoint_v1', 'full_author_LR_prefix_checkpoint_v1'):
                raise ValueError('Unsupported or project-module checkpoint schema')
            previous_config = payload.get('adaptation_config')
            if previous_config and (previous_config['method'] != config['method'] or
                                    previous_config['wu_configuration'] != config['wu_configuration']):
                raise ValueError('Checkpoint method or Wu representation differs')
            source_manifest = payload.get('adaptation_config', {}).get('manifest', payload.get('plan', {}).get('data', {}).get('manifest'))
            if source_manifest != config['manifest']:
                raise ValueError('Checkpoint belongs to a different scene or input protocol')
            if payload.get('affine') is not None:
                raise ValueError('Colour-corrected project parent is not a plain Wu parent')
            self.loaded_checkpoint_identity = identity(p)
            self.g._deformation = self.g._deformation.cuda()
            self.g.restore(payload['model'], self.opt)
            self.g._deformation_accum = payload['deformation_accum']
        else:
            from utils.graphics_utils import BasicPointCloud
            init = load_initial_points(self.manifest)
            xyz, colors = init['points'], init['colors']
            self.g._deformation.deformation_net.set_aabb(xyz.max(0), xyz.min(0))
            self.g.create_from_pcd(BasicPointCloud(xyz, colors, np.zeros_like(xyz)), config['cameras_extent'], 300)
            self.g.training_setup(self.opt)
        self.g.optimizer.zero_grad(set_to_none=True)
        self.background = torch.tensor([1., 1., 1.] if cfg['ModelParams']['_white_background'] else [0., 0., 0.], device='cuda')

    def camera(self, observation, resolution='hr', allow_heldout=False):
        import numpy as np
        key = (observation['camera_id'], observation['frame_index'])
        if key not in self.indices and not allow_heldout:
            raise ValueError('Held-out camera is available only to the evaluation path')
        index = self.indices.get(key, -1)
        width, height = self.manifest['resolutions'][resolution]
        cam = self.manifest['cameras'][observation['camera_id']]
        item = dict(camera_id=observation['camera_id'], frame_index=observation['frame_index'],
                    time=observation['time'], width=width, height=height,
                    K=np.asarray(cam['K_' + resolution]), w2c=np.asarray(cam['w2c']),
                    image=self.torch.zeros(3, 1, 1))
        from n3dv_data import observation_to_4dgs_camera
        camera = observation_to_4dgs_camera(item, index)
        camera.image_width, camera.image_height = width, height
        return camera

    def lr(self, observation):
        return self.dataset[self.indices[(observation['camera_id'], observation['frame_index'])]]['image'].cuda()

    def effective_state(self, time):
        return self.native.native_effective_state(self.g, time, self.torch)

    def covariance(self, state):
        return self.native.state_covariance(state, self.torch)

    def render(self, observation, state=None, resolution='hr', colors=None, raster_api=None):
        camera = self.camera(observation, resolution)
        state = state if state is not None else self.effective_state(observation['time'])
        if colors is None and raster_api is None:
            return self.native.native_rgb(camera, self.g, state, self.pipe, self.background)
        if raster_api is None:
            from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
        else:
            if hasattr(raster_api, 'GaussianRasterizationSettings'):
                GaussianRasterizationSettings = raster_api.GaussianRasterizationSettings
                GaussianRasterizer = raster_api.GaussianRasterizer
            else:
                GaussianRasterizationSettings, GaussianRasterizer = raster_api
        torch = self.torch
        points = torch.zeros_like(state['xyz'], requires_grad=True) + 0
        if points.requires_grad:
            points.retain_grad()
        kw = dict(image_height=int(camera.image_height), image_width=int(camera.image_width),
                  tanfovx=math.tan(camera.FoVx / 2), tanfovy=math.tan(camera.FoVy / 2),
                  bg=self.background if colors is None else torch.zeros_like(self.background), scale_modifier=1.,
                  viewmatrix=camera.world_view_transform.to(state['xyz'].device),
                  projmatrix=camera.full_proj_transform.to(state['xyz'].device), sh_degree=self.g.active_sh_degree,
                  campos=camera.camera_center.to(state['xyz'].device), prefiltered=False, debug=self.pipe.debug)
        fields = getattr(GaussianRasterizationSettings, '_fields', ())
        if 'antialiasing' in fields:
            kw.update(antialiasing=False, num_contributions=10)
        result = GaussianRasterizer(raster_settings=GaussianRasterizationSettings(**kw))(
            means3D=state['xyz'], means2D=points, shs=state['sh'] if colors is None else None,
            colors_precomp=colors, opacities=state['opacity'], scales=state['scales'],
            rotations=state['rotation'], cov3D_precomp=None)
        packet = dict(render=result[0], radii=result[1], visibility_filter=result[1] > 0, viewspace_points=points)
        if len(result) == 6:
            packet.update(true_radii=result[2], invdepth=result[3], is_contributing=result[4], contributions=result[5])
        elif len(result) == 3:
            packet['depth'] = result[2]
        else:
            raise ValueError('Unknown rasterizer return contract')
        return packet

    def regularizer(self):
        return self.g.compute_regulation(self.h.time_smoothness_weight, self.h.l1_time_planes, self.h.plane_tv_weight)

    def save(self, path, metadata):
        import numpy as np
        p = resolve(path)
        if p.exists():
            raise FileExistsError(p)
        self.torch.save(dict(schema='static_sr_wu_checkpoint_v1', model=self.g.capture(),
            deformation_accum=self.g._deformation_accum, adaptation_config=self.config,
            effective_optimization=vars(self.opt),
            rng=self.prefix.rng_state(self.torch, np), metadata=metadata), p)
        return {'path': str(p), 'sha256': sha(p)}
