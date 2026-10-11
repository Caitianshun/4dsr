"""Lightweight retrain entry for training-camera visibility culling (M1) and per-camera colour affine (M8).

Subcommands
  lr    LR stage. Restores a registered parent's coarse-end state (checkpoint_fine_00000.pt: model, Adam,
        sampler, RNG) and replays the author fine stage (Wu 4DGaussians 843d5ac, as in full_prefix_registered.py)
        with optional (i) culling of Gaussians no training camera sees, (ii) removal of the 200k-point gate on the
        author low-opacity prune in the fine stage, (iii) a per-training-camera 3x4 colour affine on the render.
        --coarse-only replays the unchanged author coarse stage from the seed instead (used when the parent's
        coarse-end checkpoint is not available locally).
  sr    SR stage. Same-time three-view batches (Bsync) of the registered schedule: LR closure on view a (weight 1)
        + frozen SwinIR teacher L1 on views b, c (weight lambda_SR each), author topology with the SR clock,
        optional culling every --cull-interval steps and at the start, optional colour affine.
  eval  cam00 all 300 frames at HR (PSNR, SSIM, AlexNet LPIPS with the official legacy functions, LR-closure
        diagnostic, FFT bands) plus training-camera diagnostics; optional post-hoc culling of the loaded model.

Culling criterion (10-09 zero-training check, unchanged): a Gaussian is culled iff no training camera c has,
at any of 10 sampled times, a 3-sigma screen footprint intersecting the image at depth > 0.25 * near_c, with
near_c the 5th percentile depth of the LR-triangulated initial points visible in c. cam00 never enters.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import json
import math
import os
import random
import socket
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import rt

import numpy as np
import torch

import full_prefix_registered as prefix
import full_refine_registered_v2 as native
import full_evaluate_registered_v2 as ev
import losses
from n3dv_data import load_manifest, load_initial_points, N3DVPreparedDataset, observation_to_4dgs_camera

ROOT = rt.ROOT
SCHEMA_LR = 'cullretrain_lr_v1'
SCHEMA_SR = 'cullretrain_sr_v1'
CULL_TIMES = [f / 300 for f in range(0, 300, 33)]
SR_STEPS = 6000
LR_OFFSET = 14000


def read(path):
    return json.loads(Path(path).read_text())


def rel(path):
    path = Path(path).resolve()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def entry(path):
    return dict(path=rel(path), sha256=rt.sha(path))


def plan_scene(plan):
    manifest_path = ROOT / plan['data']['manifest']['path']
    manifest = load_manifest(manifest_path)
    return manifest_path, manifest


def configuration(plan):
    c = plan['configuration']
    return (SimpleNamespace(**c['ModelHiddenParams']), SimpleNamespace(**c['OptimizationParams']),
            SimpleNamespace(**c['PipelineParams']), c['ModelParams']['sh_degree'])


def white_background(plan):
    return plan.get('white_background', plan['configuration']['ModelParams']['_white_background'])


# ----------------------------------------------------------------------------------------------- culling
class Culler:
    """Training-camera visibility criterion; geometry from the manifest and the LR initialization only."""

    def __init__(self, manifest, rho=0.25, sigma=3.0, times=CULL_TIMES, device='cuda'):
        init = load_initial_points(manifest)['points'].astype(np.float64)
        cams = manifest['cameras']
        self.train = list(manifest['splits']['train'])
        if 'cam00' in self.train:
            raise ValueError('cam00 must not enter the culling criterion')
        self.W, self.H = manifest['resolutions']['hr']
        self.rho, self.sigma, self.times = rho, sigma, list(times)
        R, T, K, near = [], [], [], []
        for c in self.train:
            w2c = np.array(cams[c]['w2c'], dtype=np.float64)
            k = np.array(cams[c]['K_hr'], dtype=np.float64)
            pc = init @ w2c[:3, :3].T + w2c[:3, 3]
            z = pc[:, 2]
            uv = (pc[:, :2] / np.maximum(z[:, None], 1e-6)) @ k[:2, :2].T + k[:2, 2]
            ok = (z > 0.2) & (uv[:, 0] >= 0) & (uv[:, 0] < self.W) & (uv[:, 1] >= 0) & (uv[:, 1] < self.H)
            near.append(float(np.percentile(z[ok], 5)))
            R.append(w2c[:3, :3]); T.append(w2c[:3, 3]); K.append(k)
        self.near = dict(zip(self.train, near))
        f32 = dict(dtype=torch.float32, device=device)
        self.R = [torch.tensor(r, **f32) for r in R]
        self.T = [torch.tensor(t, **f32) for t in T]
        self.K = [torch.tensor(k, **f32) for k in K]
        self.near_t = near

    @torch.no_grad()
    def mask(self, g):
        n = g.get_xyz.shape[0]
        seen = torch.zeros(n, dtype=torch.bool, device=g.get_xyz.device)
        for t in self.times:
            st = native.native_effective_state(g, float(t), torch)
            xyz, smax = st['xyz'], st['scales'].max(1).values
            for R, T, K, near in zip(self.R, self.T, self.K, self.near_t):
                pc = xyz @ R.T + T
                z = pc[:, 2]
                uv = (pc[:, :2] / z.clamp_min(1e-6)[:, None]) @ K[:2, :2].T + K[:2, 2]
                r = self.sigma * smax * K[0, 0] / z.clamp_min(1e-6)
                infp = (z > 0.2) & (uv[:, 0] > -r) & (uv[:, 0] < self.W + r) & (uv[:, 1] > -r) & (uv[:, 1] < self.H + r)
                seen |= infp & (z > self.rho * near)
        return ~seen

    def describe(self):
        return dict(rho=self.rho, sigma=self.sigma, sampled_times=self.times, near_bounds=self.near,
                    footprint='3-sigma of max activated scale, HR intrinsics', cameras=self.train)


def cull(g, culler, log, where, optimizer_backed=True):
    mask = culler.mask(g)
    before = int(mask.numel())
    removed = int(mask.sum())
    opaque = int((mask & (g.get_opacity.detach()[:, 0] > 0.1)).sum())
    if removed:
        if optimizer_backed:
            g.prune_points(mask)
        else:
            keep = ~mask
            for name in ('_xyz', '_features_dc', '_features_rest', '_scaling', '_rotation', '_opacity'):
                setattr(g, name, torch.nn.Parameter(getattr(g, name).detach()[keep], requires_grad=False))
            for name in ('_deformation_table', 'max_radii2D', 'xyz_gradient_accum', 'denom', '_deformation_accum'):
                value = getattr(g, name, None)
                if torch.is_tensor(value) and value.shape[0] == before:
                    setattr(g, name, value[keep])
        torch.cuda.empty_cache()
    row = dict(event='cull', where=where, points_before=before, culled=removed, culled_opacity_gt_0p1=opaque,
               points_after=int(g.get_xyz.shape[0]))
    if log is not None:
        log.write(json.dumps(row) + '\n')
    return row


# ----------------------------------------------------------------------------------------------- colour affine
class CameraAffine:
    """Per-training-camera 3x4 colour affine on the rendered image; mean over cameras fixed to identity.

    Parameterisation and Adam schedule follow the exposure compensation of the official 3DGS code
    (identity init, lr 1e-2 -> 1e-3 exponential); the mean-identity constraint is applied by re-centring.
    Test views (cam00) use the identity.
    """

    def __init__(self, cameras, device='cuda', lr_init=1e-2, lr_final=1e-3, max_steps=14000):
        self.names = list(cameras)
        self.index = {c: i for i, c in enumerate(self.names)}
        self.delta = torch.zeros(len(self.names), 3, 4, device=device, requires_grad=True)
        self.optimizer = torch.optim.Adam([self.delta], lr=lr_init)
        self.lr_init, self.lr_final, self.max_steps = lr_init, lr_final, max_steps
        self.eye = torch.eye(3, device=device)

    def apply(self, image, camera_id):
        d = self.delta[self.index[camera_id]] - self.delta.mean(0)
        return torch.einsum('ij,jhw->ihw', self.eye + d[:, :3], image) + d[:, 3, None, None]

    def step(self, clock):
        t = min(max(clock / self.max_steps, 0.), 1.)
        lr = math.exp((1 - t) * math.log(self.lr_init) + t * math.log(self.lr_final))
        for group in self.optimizer.param_groups:
            group['lr'] = lr
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

    def state_dict(self):
        return dict(names=self.names, delta=self.delta.detach().clone(), optimizer=self.optimizer.state_dict(),
                    lr_init=self.lr_init, lr_final=self.lr_final, max_steps=self.max_steps)

    def load_state_dict(self, value):
        if value['names'] != self.names:
            raise ValueError('Affine camera list differs')
        with torch.no_grad():
            self.delta.copy_(value['delta'])
        self.optimizer.load_state_dict(value['optimizer'])

    def summary(self):
        d = (self.delta - self.delta.mean(0)).detach().cpu()
        return {c: dict(matrix=(torch.eye(3) + d[i, :, :3]).tolist(), offset=d[i, :, 3].tolist(),
                        deviation_fro=float(d[i].norm())) for i, c in enumerate(self.names)}


# ----------------------------------------------------------------------------------------------- topology
def topology(g, opt, stage, iteration, extent, out, statistics, prune_gate, counts):
    """prefix.author_topology with an optional fine-stage removal of the 200k-point prune gate."""
    radii, visible, summed = statistics
    if iteration >= opt.densify_until_iter:
        return
    g.max_radii2D[visible] = g.max_radii2D[visible].maximum(radii[visible])
    g.add_densification_stats(summed, visible)
    if stage == 'coarse':
        opacity_threshold = opt.opacity_threshold_coarse
        threshold = opt.densify_grad_threshold_coarse
    else:
        opacity_threshold = opt.opacity_threshold_fine_init - iteration * (opt.opacity_threshold_fine_init - opt.opacity_threshold_fine_after) / opt.densify_until_iter
        threshold = opt.densify_grad_threshold_fine_init - iteration * (opt.densify_grad_threshold_fine_init - opt.densify_grad_threshold_after) / opt.densify_until_iter
    size = 20 if iteration > opt.opacity_reset_interval else None
    if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0 and g.get_xyz.shape[0] < 360000:
        g.densify(threshold, opacity_threshold, extent, size, 5, 5, str(out), iteration, stage)
        counts['densify'] += 1
    gate = g.get_xyz.shape[0] > 200000 or (prune_gate == 'none' and stage == 'fine')
    if iteration > opt.pruning_from_iter and iteration % opt.pruning_interval == 0 and gate:
        before = g.get_xyz.shape[0]
        g.prune(threshold, opacity_threshold, extent, size)
        counts['prune'] += 1
        counts['pruned_points'] += before - g.get_xyz.shape[0]
    if iteration % opt.densification_interval == 0 and g.get_xyz.shape[0] < 360000 and opt.add_point:
        g.grow(5, 5, str(out), iteration, stage)
        counts['grow'] += 1
    if iteration % opt.opacity_reset_interval == 0:
        g.reset_opacity()
        counts['opacity_reset'] += 1


# ----------------------------------------------------------------------------------------------- checkpoints
def save(path, schema, g, plan, extra):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    payload = dict(schema=schema, plan=plan, model=g.capture(), deformation_accum=g._deformation_accum,
                   parameter_gradients={name: p.grad for name, p in prefix.parameter_map(g).items()},
                   rng=prefix.rng_state(torch, np), **extra)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    torch.save(payload, tmp)
    os.replace(tmp, path)
    rt.write_json(path.with_suffix('.json'), dict(**entry(path), schema=schema, points=int(g.get_xyz.shape[0]),
                                                 **{k: v for k, v in extra.items() if k in ('metadata',)}))
    return path


def restore_training(path, g, opt):
    """Restore model + Adam + saved gradients + RNG from a registered parent or a cullretrain LR checkpoint."""
    path = Path(path)
    side = path.with_suffix('.json')
    if side.exists():
        expected = read(side).get('sha256')
        if expected and rt.sha(path) != expected:
            raise ValueError('Checkpoint bytes differ from sidecar: ' + str(path))
    payload = torch.load(path, map_location='cuda', weights_only=False)
    if payload['schema'] not in ('full_author_LR_prefix_checkpoint_v1', SCHEMA_LR):
        raise ValueError('Unsupported checkpoint schema ' + payload['schema'])
    g.restore(payload['model'], opt)
    g._deformation_accum = payload['deformation_accum']
    params = prefix.parameter_map(g)
    for name, p in params.items():
        p.grad = payload['parameter_gradients'][name]
    prefix.restore_rng(payload['rng'], torch, np)
    return payload


def load_for_render(path, plan):
    """Parameters only (no optimizer); works for registered SR/parent checkpoints and cullretrain ones."""
    from scene.gaussian_model import GaussianModel
    h, opt, pipe, sh = configuration(plan)
    payload = torch.load(path, map_location='cuda', weights_only=False)
    g = GaussianModel(sh, h)
    (g.active_sh_degree, xyz, deform_state, g._deformation_table, f_dc, f_rest, scaling, rotation, opacity,
     g.max_radii2D, g.xyz_gradient_accum, g.denom, _opt, g.spatial_lr_scale) = payload['model']
    g._deformation.load_state_dict(deform_state)
    g._deformation = g._deformation.cuda().eval()
    for name, t in (('_xyz', xyz), ('_features_dc', f_dc), ('_features_rest', f_rest), ('_scaling', scaling),
                    ('_rotation', rotation), ('_opacity', opacity)):
        setattr(g, name, torch.nn.Parameter(t.detach().clone().float().cuda(), requires_grad=False))
    for p in g._deformation.parameters():
        p.requires_grad_(False)
    g._deformation_accum = payload.get('deformation_accum')
    return g, payload


# ----------------------------------------------------------------------------------------------- LR stage
def lr_iteration(g, h, opt, pipe, background, cameras, camera_ids, stage, iteration, extent, out, render,
                 affine, prune_gate, counts):
    """prefix.author_iteration with optional colour affine and prune-gate change; same order of operations."""
    g.update_learning_rate(iteration)
    if iteration % 1000 == 0:
        g.oneupSHdegree()
    packages, images, targets = [], [], []
    for camera, cid in zip(cameras, camera_ids):
        packet = render(camera, g, pipe, background, stage=stage, cam_type='dynerf')
        packages.append(packet)
        image = packet['render'] if affine is None else affine.apply(packet['render'], cid)
        images.append(image.unsqueeze(0))
        targets.append(camera.original_image.to(background.device).unsqueeze(0))
    prediction = torch.cat(images, dim=0)
    target = torch.cat(targets, dim=0)
    image_loss = (prediction - target[:, :3]).abs().mean()
    loss = image_loss
    if stage == 'fine' and h.time_smoothness_weight != 0:
        loss = loss + g.compute_regulation(h.time_smoothness_weight, h.l1_time_planes, h.plane_tv_weight)
    if opt.lambda_dssim != 0:
        from utils.loss_utils import ssim
        loss = loss + opt.lambda_dssim * (1 - ssim(prediction, target))
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError('Nonfinite LR loss')
    loss.backward()
    statistics = prefix.combine_batch_statistics(packages, torch)
    applied = iteration < opt.iterations
    with torch.no_grad():
        topology(g, opt, stage, iteration, extent, out, statistics, prune_gate, counts)
        if applied:
            g.optimizer.step()
            g.optimizer.zero_grad(set_to_none=True)
    if affine is not None and applied:
        affine.step(iteration)
    return dict(loss=float(loss.detach()), L1=float(image_loss.detach()), stage=stage, iteration=iteration,
                Adam_applied=applied, points=int(g.get_xyz.shape[0]), batch_size=len(cameras))


def run_lr(a):
    from scene.gaussian_model import GaussianModel
    from gaussian_renderer import render
    from utils.graphics_utils import BasicPointCloud
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'complete.json').exists():
        raise FileExistsError('Completed run preserved: ' + str(out))
    parent = Path(a.parent)
    plan = read(parent / 'config.json')
    h, opt, pipe, sh = configuration(plan)
    manifest_path, manifest = plan_scene(plan)
    dataset = N3DVPreparedDataset(manifest, 'train', 'lr', device='cpu', cache=False)
    seed = int(plan['seed'])
    extent = plan['data']['cameras_extent']
    torch.set_num_threads(a.cpu_threads)
    started = time.time()
    record = dict(kind='lr', arm=a.arm, args=vars(a), parent=rel(parent), scene=plan['scene'], seed=seed,
                  host=socket.gethostname(), gpu=torch.cuda.get_device_name(), runtime=rt.check_runtime(),
                  sources={rel(p): rt.sha(p) for p in (Path(__file__), Path(rt.__file__))}, started_unix=started)
    rt.write_json(out / 'run.json', record)
    counts = dict(densify=0, prune=0, pruned_points=0, grow=0, opacity_reset=0)
    log = (out / 'training.jsonl').open('a', buffering=1)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    g = GaussianModel(sh, h)
    if a.coarse_only:
        initial = load_initial_points(manifest)
        xyz, colors = initial['points'], initial['colors']
        g._deformation.deformation_net.set_aabb(xyz.max(axis=0), xyz.min(axis=0))
        g.create_from_pcd(BasicPointCloud(xyz, colors, np.zeros_like(xyz)), extent, 300)
        g.training_setup(opt)
        sampler = prefix.AuthorShuffle(len(dataset), opt.batch_size, torch)
        state = dict(stage='coarse', coarse_iteration=0, fine_iteration=0, optimizer_resets=1)
        start_stage = 'coarse'
    else:
        g._deformation = g._deformation.cuda()
        fine0 = Path(a.fine0) if a.fine0 else parent / 'checkpoint_fine_00000.pt'
        payload = restore_training(fine0, g, opt)
        sampler = prefix.AuthorShuffle(payload['sampler']['length'], payload['sampler']['batch_size'], torch, payload['sampler'])
        state = dict(payload['metadata']['state'])
        if state['stage'] != 'fine' or state['fine_iteration'] != 0:
            raise ValueError('Expected the coarse-end / fine-0 state')
        record['fine0'] = entry(fine0)
        start_stage = 'fine'
        del payload
    culler = Culler(manifest) if a.cull else None
    affine = None
    if a.affine:
        affine = CameraAffine(manifest['splits']['train'], max_steps=opt.iterations)
    record.update(culler=culler.describe() if culler else None, prune_gate=a.prune_gate, affine=bool(a.affine),
                  cull_interval=a.cull_interval)
    rt.write_json(out / 'run.json', record)
    background = torch.ones(3, dtype=torch.float32, device='cuda') if white_background(plan) else torch.zeros(3, dtype=torch.float32, device='cuda')
    cull_rows = []
    stop = a.stop or opt.iterations
    try:
        for stage in ('coarse', 'fine'):
            if stage == 'coarse' and start_stage == 'fine':
                continue
            if stage != state['stage']:
                g.training_setup(opt)
                sampler = prefix.AuthorShuffle(len(dataset), opt.batch_size, torch)
                state.update(stage=stage, optimizer_resets=state['optimizer_resets'] + 1)
                save(out / 'checkpoint_fine_00000.pt', SCHEMA_LR, g, plan,
                     dict(sampler=sampler.state_dict(), metadata=dict(state=dict(state), arm=a.arm, coarse_replayed=True)))
                if a.coarse_only:
                    break
            maximum = opt.coarse_iterations if stage == 'coarse' else min(opt.iterations, stop)
            for iteration in range(state[stage + '_iteration'] + 1, maximum + 1):
                tick = time.time()
                batch = sampler.next()
                cameras, ids = [], []
                for index in batch:
                    item = dataset[index]
                    cameras.append(observation_to_4dgs_camera(item, index))
                    ids.append(item['camera_id'])
                row = lr_iteration(g, h, opt, pipe, background, cameras, ids, stage, iteration, extent, out, render,
                                   affine if stage == 'fine' else None, a.prune_gate, counts)
                state[stage + '_iteration'] = iteration
                row.update(batch_indices=batch, seconds=time.time() - tick)
                log.write(json.dumps(row) + '\n')
                if stage == 'fine' and culler is not None:
                    final = iteration == opt.iterations
                    if (iteration % a.cull_interval == 0 and iteration <= opt.densify_until_iter) or final:
                        if final:
                            g.optimizer.zero_grad(set_to_none=True)
                        cull_rows.append(dict(cull(g, culler, log, f'fine_{iteration}'), iteration=iteration))
        if not a.coarse_only:
            if state['fine_iteration'] == opt.iterations:
                g.optimizer.zero_grad(set_to_none=True)
            name = f"checkpoint_fine_{state['fine_iteration']:05d}.pt"
            save(out / name, SCHEMA_LR, g, plan,
                 dict(sampler=sampler.state_dict(), affine=affine.state_dict() if affine else None,
                      metadata=dict(state=dict(state), arm=a.arm)))
        torch.cuda.synchronize()
        result = dict(status='completed', state=state, seconds=time.time() - started, counts=counts, culls=cull_rows,
                      points=int(g.get_xyz.shape[0]), peak_GB=torch.cuda.max_memory_allocated() / 1e9,
                      affine=affine.summary() if affine else None,
                      checkpoint=entry(out / ('checkpoint_fine_00000.pt' if a.coarse_only else f"checkpoint_fine_{state['fine_iteration']:05d}.pt")))
        rt.write_json(out / 'complete.json', dict(record, **result))
        return result
    except BaseException:
        rt.write_json(out / f'failure_{time.time_ns()}.json', dict(record, error=traceback.format_exc(), state=state))
        raise
    finally:
        log.close()


# ----------------------------------------------------------------------------------------------- SR stage
class Prefetch:
    """Reads the next step's LR/teacher pixels on worker threads; values are identical to serial reads."""

    def __init__(self, fn, keys, workers=3):
        self.fn, self.keys = fn, keys
        self.pool = cf.ThreadPoolExecutor(workers)
        self.pending = {}

    def get(self, i):
        for j in (i, i + 1):
            if 0 <= j < len(self.keys) and j not in self.pending:
                self.pending[j] = self.pool.submit(self.fn, j)
        return self.pending.pop(i).result()

    def close(self):
        self.pool.shutdown(wait=False, cancel_futures=True)


def read_teacher(path):
    from PIL import Image
    with Image.open(path) as image:
        return torch.from_numpy(np.asarray(image.convert('RGB')).copy()).permute(2, 0, 1).float() / 255


def run_sr(a):
    from scene.gaussian_model import GaussianModel
    from utils.loss_utils import ssim
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'complete.json').exists():
        raise FileExistsError('Completed run preserved: ' + str(out))
    base = read(Path(a.baseline_sr) / 'config.json')
    definition = base['definition']
    if definition != dict(schedule='rows', lr_weights=[1.0, 0.0, 0.0], sr_weights=[0.0, 0.05, 0.05], X=False, E=False):
        raise ValueError('Only the registered Bsync definition is supported')
    parent_ckpt = Path(a.parent_checkpoint)
    payload = torch.load(parent_ckpt, map_location='cuda', weights_only=False)
    plan = payload['plan']
    if plan['configuration'] != base['configuration']:
        raise ValueError('Parent configuration differs from the baseline SR configuration')
    h, opt, pipe, sh = configuration(plan)
    manifest_path, manifest = plan_scene(plan)
    if rel(manifest_path) != base['data']['manifest']['path']:
        raise ValueError('Manifest differs from the baseline SR plan')
    dataset = N3DVPreparedDataset(manifest, 'train', 'lr', device='cpu', cache=False)
    extent = plan['data']['cameras_extent']
    torch.set_num_threads(a.cpu_threads)
    started = time.time()
    record = dict(kind='sr', arm=a.arm, args=vars(a), parent_checkpoint=entry(parent_ckpt), baseline_sr=rel(a.baseline_sr),
                  scene=plan['scene'], seed=plan['seed'], lambda_SR=a.lambda_sr, dssim=a.dssim, host=socket.gethostname(),
                  gpu=torch.cuda.get_device_name(), runtime=rt.check_runtime(),
                  sources={rel(p): rt.sha(p) for p in (Path(__file__), Path(rt.__file__))}, started_unix=started)
    g = GaussianModel(sh, h)
    g._deformation = g._deformation.cuda()
    g.restore(payload['model'], opt)
    g._deformation_accum = payload['deformation_accum']
    prefix.restore_rng(payload['rng'], torch, np)
    affine = None
    if a.affine:
        affine = CameraAffine(manifest['splits']['train'], lr_init=a.affine_sr_lr, lr_final=a.affine_sr_lr, max_steps=SR_STEPS)
        if payload.get('affine') is not None:
            affine.load_state_dict(payload['affine'])
            for group in affine.optimizer.param_groups:
                group['lr'] = a.affine_sr_lr
            record['affine_inherited'] = True
    del payload
    g.optimizer.zero_grad(set_to_none=True)
    with torch.no_grad():
        for name in ('max_radii2D', 'xyz_gradient_accum', 'denom'):
            getattr(g, name).zero_()
    table = read(ROOT / base['dependencies']['schedule']['path'])
    rows = table[definition['schedule']]
    record_keys = [tuple(k) for k in table['record_keys']]
    teacher_index = read(ROOT / base['dependencies']['teacher']['path'])
    teacher_rows = {(e['camera'], int(e['frame'])): e for e in teacher_index['entries']}
    root = Path(manifest['_root'])
    teacher_paths = {key: native.safe_child(root, e['relative_path']) for key, e in teacher_rows.items()}
    stop = a.stop or SR_STEPS

    def fetch(i):
        row = rows[i]
        items = [dataset[index] for index in row]
        keys = [record_keys[index] for index in row]
        teachers = [read_teacher(teacher_paths[keys[j]]) for j in (1, 2)]
        return row, items, keys, teachers

    prefetch = Prefetch(fetch, list(range(stop)))
    culler = Culler(manifest) if a.cull else None
    record.update(culler=culler.describe() if culler else None, prune_gate=a.prune_gate, affine=bool(a.affine),
                  cull_interval=a.cull_interval, schedule=entry(ROOT / base['dependencies']['schedule']['path']),
                  teacher_index=entry(ROOT / base['dependencies']['teacher']['path']))
    rt.write_json(out / 'run.json', record)
    background = torch.ones(3, dtype=torch.float32, device='cuda') if white_background(plan) else torch.zeros(3, dtype=torch.float32, device='cuda')
    counts = dict(densify=0, prune=0, pruned_points=0, grow=0, opacity_reset=0)
    cull_rows = []
    log = (out / 'trajectory.jsonl').open('a', buffering=1)
    torch.cuda.reset_peak_memory_stats()
    try:
        if culler is not None:
            cull_rows.append(dict(cull(g, culler, log, 'sr_start'), cursor=0))
        for cursor in range(1, stop + 1):
            tick = time.time()
            g.update_learning_rate(LR_OFFSET + cursor)
            if (LR_OFFSET + cursor) % 1000 == 0:
                g.oneupSHdegree()
            g.optimizer.zero_grad(set_to_none=True)
            row, items, keys, teacher_cpu = prefetch.get(cursor - 1)
            if len({k[0] for k in keys}) != 3 or len({int(k[1]) for k in keys}) != 1:
                raise ValueError('Registered same-time batch differs')
            cameras = [native.hr_camera(obs, i, manifest) for i, obs in zip(row, items)]
            lrs = [obs['image'].cuda() for obs in items]
            teachers = [t.cuda() for t in teacher_cpu]
            state = native.native_effective_state(g, cameras[0].time, torch)
            packages = [native.native_rgb(c, g, state, pipe, background) for c in cameras]
            rgbs = [p['render'] for p in packages]
            if affine is not None:
                rgbs = [affine.apply(rgb, obs['camera_id']) for rgb, obs in zip(rgbs, items)]
            lr_values = [losses.lr_loss(rgb, lr) for rgb, lr in zip(rgbs, lrs)]
            lr_term = sum(w * v for w, v in zip(definition['lr_weights'], lr_values))
            sr_values = [losses.sr_loss(rgbs[i], t) for i, t in zip((1, 2), teachers)]
            if a.dssim:
                down = losses.degradation(rgbs[0], lrs[0].shape[-2:])
                lr_term = (1 - a.dssim) * lr_term + a.dssim * (1 - ssim(down[None], lrs[0][None]))
                sr_mix = [(1 - a.dssim) * v + a.dssim * (1 - ssim(rgbs[i][None], t[None])) for v, i, t in zip(sr_values, (1, 2), teachers)]
                sr_term = a.lambda_sr * (sr_mix[0] + sr_mix[1])
            else:
                sr_term = a.lambda_sr * (sr_values[0] + sr_values[1])
            regularizer = g.compute_regulation(h.time_smoothness_weight, h.l1_time_planes, h.plane_tv_weight) if h.time_smoothness_weight != 0 else rgbs[0].sum() * 0.
            loss = lr_term + sr_term + regularizer
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError('Nonfinite SR objective')
            loss.backward()
            statistics = prefix.combine_batch_statistics(packages, torch)
            before = int(g.get_xyz.shape[0])
            with torch.no_grad():
                topology(g, opt, 'fine', cursor, extent, out, statistics, a.prune_gate, counts)
                g.optimizer.step()
                g.optimizer.zero_grad(set_to_none=True)
            if affine is not None:
                affine.step(cursor)
            if culler is not None and cursor % a.cull_interval == 0:
                cull_rows.append(dict(cull(g, culler, log, f'sr_{cursor}'), cursor=cursor))
            log.write(json.dumps(dict(cursor=cursor, loss=float(loss.detach()), LR_L1=[float(v.detach()) for v in lr_values],
                                      SR_L1=[float(v.detach()) for v in sr_values], LR_term=float(lr_term.detach()),
                                      SR_term=float(sr_term.detach()), regularizer=float(regularizer.detach()),
                                      points_before=before, points_after=int(g.get_xyz.shape[0]), keys=[list(k) for k in keys],
                                      seconds=time.time() - tick)) + '\n')
            del items, cameras, lrs, teachers, packages, rgbs, state, loss
        prefetch.close()
        if culler is not None and stop % a.cull_interval != 0:
            cull_rows.append(dict(cull(g, culler, log, f'sr_end_{stop}'), cursor=stop))
        path = save(out / f'checkpoint_{stop:05d}.pt', SCHEMA_SR, g, plan,
                    dict(affine=affine.state_dict() if affine else None, metadata=dict(cursor=stop, arm=a.arm)))
        torch.cuda.synchronize()
        result = dict(status='completed', cursor=stop, seconds=time.time() - started, counts=counts, culls=cull_rows,
                      points=int(g.get_xyz.shape[0]), peak_GB=torch.cuda.max_memory_allocated() / 1e9,
                      affine=affine.summary() if affine else None, checkpoint=entry(path))
        rt.write_json(out / 'complete.json', dict(record, **result))
        return result
    except BaseException:
        prefetch.close()
        rt.write_json(out / f'failure_{time.time_ns()}.json', dict(record, error=traceback.format_exc()))
        raise
    finally:
        log.close()


# ----------------------------------------------------------------------------------------------- evaluation
def near_camera_count(g, manifest, camera='cam00'):
    """Diagnostic only (never used for training/selection): cam00-view Gaussians nearer than 0.3 x median depth."""
    cam = manifest['cameras'][camera]
    w2c = torch.tensor(cam['w2c'], dtype=torch.float32, device='cuda')
    K = torch.tensor(cam['K_hr'], dtype=torch.float32, device='cuda')
    W, H = manifest['resolutions']['hr']
    with torch.no_grad():
        xyz = g.get_xyz
        pc = xyz @ w2c[:3, :3].T + w2c[:3, 3]
        z = pc[:, 2]
        uv = (pc[:, :2] / z.clamp_min(1e-6)[:, None]) @ K[:2, :2].T + K[:2, 2]
        inimg = (z > 0.2) & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
        if not bool(inimg.any()):
            return 0
        med = z[inimg].median()
        return int((inimg & (z < 0.3 * med) & (g.get_opacity[:, 0] > 0.1)).sum())


def run_eval(a):
    import lpips
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'summary.json').exists():
        raise FileExistsError('Completed evaluation preserved: ' + str(out))
    checkpoint = Path(a.checkpoint)
    plan_source = read(Path(a.plan)) if a.plan else None
    started = time.time()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(a.cpu_threads)
    g, payload = load_for_render(checkpoint, plan_source or torch.load(checkpoint, map_location='cpu', weights_only=False)['plan'])
    plan = plan_source or payload['plan']
    del payload
    h, opt, pipe, sh = configuration(plan)
    manifest_path, manifest = plan_scene(plan)
    record = dict(kind='eval', label=a.label, checkpoint=entry(checkpoint), plan=rel(a.plan) if a.plan else 'embedded',
                  manifest=entry(manifest_path), posthoc_cull=bool(a.cull), host=socket.gethostname(),
                  gpu=torch.cuda.get_device_name(), runtime=rt.check_runtime(),
                  sources={rel(p): rt.sha(p) for p in (Path(__file__), Path(rt.__file__))}, started_unix=started,
                  points_loaded=int(g.get_xyz.shape[0]), cam00_near_count_loaded=near_camera_count(g, manifest))
    if a.cull:
        culler = Culler(manifest)
        record['cull'] = cull(g, culler, None, 'posthoc', optimizer_backed=False)
        record['culler'] = culler.describe()
    record['points_evaluated'] = int(g.get_xyz.shape[0])
    record['cam00_near_count_evaluated'] = near_camera_count(g, manifest)
    rt.write_json(out / 'run.json', record)
    legacy, closure, _, fft = ev.metric_helpers()
    metric = lpips.LPIPS(net='alex', version='0.1', spatial=False, pnet_rand=False, pretrained=True, verbose=False).cuda().eval().requires_grad_(False)
    background = torch.ones(3, dtype=torch.float32, device='cuda') if white_background(plan) else torch.zeros(3, dtype=torch.float32, device='cuda')
    root = Path(manifest['_root'])
    test = sorted([o for o in manifest['observations'] if o['split'] == 'test'], key=lambda o: int(o['frame_index']))
    if [int(o['frame_index']) for o in test] != list(range(300)) or {o['camera_id'] for o in test} != {'cam00'}:
        raise ValueError('cam00 all-300 test split required')
    frames = test if a.frames is None else [o for o in test if int(o['frame_index']) in set(a.frames)]
    diag_cam = manifest['splits']['train'][0]
    diag = sorted([o for o in manifest['observations'] if o['split'] == 'train' and o['camera_id'] == diag_cam
                   and int(o['frame_index']) in range(15, 300, 30)], key=lambda o: int(o['frame_index']))
    pool = cf.ThreadPoolExecutor(4)
    io = lambda o: (legacy.read_rgb(root / o['hr_path']), legacy.read_rgb(root / o['lr_path']))
    rows = []
    previews = {0, 150, 299}
    (out / 'previews').mkdir(exist_ok=True)
    observations = [('test', o) for o in frames] + [('train_diagnostic', o) for o in diag]
    futures = {i: pool.submit(io, o) for i, (_, o) in enumerate(observations[:4])}
    for i, (scope, obs) in enumerate(observations):
        if i + 4 < len(observations):
            futures[i + 4] = pool.submit(io, observations[i + 4][1])
        with torch.no_grad():
            cam = native.hr_camera(ev.camera_item(manifest, obs, torch), i, manifest)
            state = native.native_effective_state(g, float(obs['time']), torch)
            raw = native.native_rgb(cam, g, state, pipe, background)['render'].detach()
        hr, lr = futures.pop(i).result()
        pred = legacy.image_array(raw)
        empty = np.zeros(pred.shape[:2], bool)
        q = legacy.spatial_metrics(pred, hr, empty, metric, 'cuda')['full']
        c = closure(raw, lr, empty)['full']
        row = dict(scope=scope, camera=obs['camera_id'], frame=int(obs['frame_index']), psnr=q['psnr'], ssim=q['ssim'],
                   lpips=q['lpips_alex'], mse=q['mse'], LR_diagnostic_psnr=c['psnr'], LR_diagnostic_l1=c['l1'])
        if scope == 'test':
            band = fft.decompose(pred, hr)
            row.update({k: band[k] for k in ('low_mse', 'mid_mse', 'high_mse', 'low_psnr', 'mid_psnr', 'high_psnr') if k in band})
            if int(obs['frame_index']) in previews:
                from PIL import Image
                Image.fromarray((np.clip(pred, 0, 1) * 255 + 0.5).astype(np.uint8)).save(out / 'previews' / f"cam00_{int(obs['frame_index']):04d}.png")
                np.savez_compressed(out / 'previews' / f"cam00_{int(obs['frame_index']):04d}.npz", rgb_raw=raw.cpu().numpy().astype(np.float32))
        rows.append(row)
    pool.shutdown()
    with (out / 'metrics_per_observation.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
        writer.writeheader()
        writer.writerows(rows)
    mean = lambda rs, k: float(np.mean([r[k] for r in rs])) if rs and all(k in r for r in rs) else None
    test_rows = [r for r in rows if r['scope'] == 'test']
    diag_rows = [r for r in rows if r['scope'] == 'train_diagnostic']
    summary = dict(record, status='completed', frames=len(test_rows),
                   test=dict(psnr=mean(test_rows, 'psnr'), ssim=mean(test_rows, 'ssim'), lpips=mean(test_rows, 'lpips'),
                             LR_diagnostic_psnr=mean(test_rows, 'LR_diagnostic_psnr'),
                             PSNR_of_pooled_MSE=float(-10 * np.log10(np.mean([r['mse'] for r in test_rows]))),
                             low_psnr=mean(test_rows, 'low_psnr'), mid_psnr=mean(test_rows, 'mid_psnr'), high_psnr=mean(test_rows, 'high_psnr')),
                   train_diagnostic=dict(camera=diag_cam, frames=len(diag_rows), psnr=mean(diag_rows, 'psnr'), ssim=mean(diag_rows, 'ssim'),
                                         lpips=mean(diag_rows, 'lpips'), LR_diagnostic_psnr=mean(diag_rows, 'LR_diagnostic_psnr')),
                   seconds=time.time() - started, peak_GB=torch.cuda.max_memory_allocated() / 1e9)
    rt.write_json(out / 'summary.json', summary)
    return summary


# ----------------------------------------------------------------------------------------------- CLI
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    q = sub.add_parser('lr')
    q.add_argument('--parent', required=True, help='registered LR parent directory (config.json, checkpoint_fine_00000.pt)')
    q.add_argument('--fine0', help='alternative coarse-end checkpoint (cullretrain schema) when the parent has none')
    q.add_argument('--coarse-only', action='store_true')
    q.add_argument('--stop', type=int)
    sub_sr = sub.add_parser('sr')
    sub_sr.add_argument('--parent-checkpoint', required=True)
    sub_sr.add_argument('--baseline-sr', required=True, help='registered Bsync endpoint directory (schedule, teacher, definition)')
    sub_sr.add_argument('--lambda-sr', type=float, default=0.05)
    sub_sr.add_argument('--dssim', type=float, default=0.0)
    sub_sr.add_argument('--affine-sr-lr', type=float, default=1e-3)
    sub_sr.add_argument('--stop', type=int)
    for x in (q, sub_sr):
        x.add_argument('--out', required=True)
        x.add_argument('--arm', required=True)
        x.add_argument('--cull', action='store_true')
        x.add_argument('--affine', action='store_true')
        x.add_argument('--prune-gate', choices=('author', 'none'), default='author')
        x.add_argument('--cull-interval', type=int, default=1000)
        x.add_argument('--cpu-threads', type=int, default=4)
    e = sub.add_parser('eval')
    e.add_argument('--checkpoint', required=True)
    e.add_argument('--plan', help='config.json of the registered run (needed for registered checkpoints)')
    e.add_argument('--out', required=True)
    e.add_argument('--label', required=True)
    e.add_argument('--cull', action='store_true', help='post-hoc culling before rendering')
    e.add_argument('--frames', type=int, nargs='*')
    e.add_argument('--cpu-threads', type=int, default=4)
    a = p.parse_args(argv)
    result = dict(lr=run_lr, sr=run_sr, eval=run_eval)[a.cmd](a)
    print(json.dumps({k: v for k, v in result.items() if k in ('status', 'seconds', 'points', 'test', 'train_diagnostic', 'culls', 'counts')}, default=str)[:4000])


if __name__ == '__main__':
    main()
