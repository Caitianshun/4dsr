"""Explicit Wu adaptation stages. Runs only when called; never schedules GPUs.

Use within the existing resource admission/lock service. Each command owns one
visible GPU and one new output directory. A wiring fixture is never a benchmark.
"""
from __future__ import annotations
import argparse
import json
import os
import random
import socket
import time
import traceback
from pathlib import Path
from common import METHODS, SCHEMA, checked, identity, load_prior_array, read, resolve, training_manifest, verify_prior_index, write_new

FIELDS = {'splatsure': ('sr', 'weight'), 'iesrgs': ('internal_rgb', 'internal_depth', 'external_rgb', 'external_inverse_depth'),
          'sequence_matters': ('sr',), 'threedsr': ('sr',)}


def load_tensor(item, device='cuda'):
    import numpy as np
    import torch
    from PIL import Image
    p = resolve(item['path'])
    if p.suffix in ('.npy', '.npz'):
        value = torch.from_numpy(load_prior_array(item).copy()).float()
    else:
        p = checked(item)
        if p.suffix in ('.pt', '.pty'):
            value = torch.load(p, map_location='cpu', weights_only=True)
            if not torch.is_tensor(value):
                raise ValueError('Expected tensor cache')
        else:
            value = torch.from_numpy(np.asarray(Image.open(p).convert('RGB')).copy()).permute(2, 0, 1).float() / 255
    if not torch.isfinite(value).all():
        raise ValueError('Nonfinite prior')
    return value.to(device).detach()


def preflight(config, stage, prior_bundle=None):
    if config.get('schema') != SCHEMA or config.get('method') not in METHODS:
        raise ValueError('Not a registered Wu adaptation')
    m, rows = training_manifest(checked(config['manifest']), verify_pixels=True)
    source = read(checked(config['source_configuration']))
    if source['configuration'] != config['wu_configuration']:
        raise ValueError('Wu representation/configuration changed')
    if config['scene'] != m['scene']:
        raise ValueError('Scene mismatch')
    if stage == 'lr' and config['method'] not in ('splatsure', 'threedsr'):
        raise ValueError('The plain LR teacher stage is specific to SplatSuRe / 3DSR')
    if stage == 'internal' and config['method'] != 'iesrgs':
        raise ValueError('Only IE-SRGS has this internal stage')
    priors = None
    if stage == 'sr':
        if not prior_bundle:
            raise ValueError('A complete source-specific prior bundle is required before SR training')
        priors = verify_prior_index(prior_bundle, config['manifest'], m, rows, FIELDS[config['method']])
        bundle = read(prior_bundle)
        actual_method = bundle.get('method', 'iesrgs' if bundle.get('kind') == 'iesrgs_complete' else None)
        if actual_method != config['method']:
            raise ValueError('Prior bundle was produced for a different method')
        if config['method']=='splatsure' and bundle.get('teacher_variant')!=config.get('teacher_variant','stablesr'):
            raise ValueError('SplatSuRe StableSR main and SwinIR variant must be labeled separately')
    return m, rows, priors


def topology(backend, packets, iteration, maximum, out, stage='sr'):
    """Source 3DGS growth schedule applied to Wu canonical Gaussian tensors.

    No project-specific 360k cap or >200k prune gate; the clone/split functions
    are Wu's geometry-compatible equivalents of source 3DGS clone/split.
    """
    g, torch = backend.g, backend.torch
    until = 2500 if backend.config['method'] == 'threedsr' and stage == 'sr' else 15000
    if iteration == maximum or iteration >= until:
        return
    if backend.config['method'] == 'iesrgs' and stage == 'internal':
        packets = packets[-1:]  # Released mode1 uses final sampled view for screen statistics.
    radii, visible, summed = backend.prefix.combine_batch_statistics(packets, torch)
    g.max_radii2D[visible] = g.max_radii2D[visible].maximum(radii[visible])
    g.add_densification_stats(summed, visible)
    if iteration > 500 and iteration % 100 == 0:
        size = 20 if iteration > 3000 else None
        g.densify(.0002, .005, backend.config['cameras_extent'], size, 5, 5, str(out), iteration, 'fine')
        g.prune(.0002, .005, backend.config['cameras_extent'], size)
    if iteration % 3000 == 0 or (backend.config['wu_configuration']['ModelParams']['_white_background'] and iteration == 500):
        g.reset_opacity()


def batch_loss(backend, batch, stage, prior_rows, cfg):
    from methods.iesrgs import IESRGSConfig, IESRGSPriors, fusion_loss, internal_multiview_loss, ssim
    from methods.splatsure import SplatSuReConfig, splatsure_loss
    from methods.sequence_matters import sequence_matters_loss
    method = cfg['method']
    packets, terms = [], []
    state = backend.effective_state(batch[0]['time'])
    if any(abs(o['time'] - batch[0]['time']) > 1e-8 for o in batch):
        raise ValueError('Batch mixes times')
    for obs in batch:
        if method == 'iesrgs':
            from methods.iesrgs_renderer import render_state
            res = 'lr' if stage == 'internal' else 'hr'
            references = [backend.camera(o, res) for o in backend.observations if o['frame_index'] == obs['frame_index']]
            packet = render_state(backend.camera(obs, res), state, backend.background,
                backend.g.active_sh_degree, references, require_depth=stage != 'internal')
        elif method == 'threedsr':
            from methods.threedsr_renderer import render_state
            res = 'lr' if stage == 'lr' else 'hr'
            references = [backend.camera(o, res) for o in backend.observations if o['frame_index'] == obs['frame_index']]
            packet = render_state(backend.camera(obs, res), state, backend.background,
                backend.g.active_sh_degree, references)
        else:
            packet = backend.render(obs, state=state, resolution='lr' if stage == 'lr' else 'hr')
        packets.append(packet)
        rgb, lr = packet['render'], backend.lr(obs)
        if method == 'splatsure':
            # The released SplatSuRe renderer clamps RGB before both stage losses.
            rgb = rgb.clamp(0, 1)
        if stage in ('lr', 'internal'):
            terms.append(.8 * (rgb - lr).abs().mean() + .2 * (1 - ssim(rgb, lr)))
            continue
        e = prior_rows[(obs['camera_id'], obs['frame_index'])]
        if method == 'splatsure':
            loss = splatsure_loss(rgb, lr, load_tensor(e['sr']), load_tensor(e['weight']),
                SplatSuReConfig(lr_operator='area' if cfg['degradation'] == 'area4' else 'bicubic'))['loss']
        elif method == 'iesrgs':
            priors = IESRGSPriors(**{f: load_tensor(e[f]) for f in FIELDS[method]})
            loss = fusion_loss(rgb, packet['median_depth'], priors, IESRGSConfig())['total']
        elif method == 'sequence_matters':
            loss, _ = sequence_matters_loss(rgb, load_tensor(e['sr']), lr, ssim,
                lr_operator='area' if cfg['degradation'] == 'area4' else 'bicubic')
        elif method == 'threedsr':
            from methods.threedsr import threedsr_loss
            loss, _ = threedsr_loss(rgb, load_tensor(e['sr']), lr, ssim,
                lr_operator='area' if cfg['degradation'] == 'area4' else 'bicubic')
        terms.append(loss)
    return backend.torch.stack(terms).sum() + backend.regularizer(), packets


def optimize(backend, config, rows, stage, priors, steps, out, clock_offset=0, sampling_rng=None):
    """Also called by 3DSR after each diffusion step; backend persists across rounds."""
    import torch
    rng = sampling_rng if sampling_rng is not None else random.Random(config['seed'] + clock_offset)
    source_sampling = sampling_rng is not None and config['method'] == 'threedsr' and stage == 'sr'
    sampling_start = rng.getstate() if source_sampling else None
    groups = {}
    for row in rows:
        groups.setdefault(row['frame_index'], []).append(row)
    pool = []
    updates = 0
    if config['method'] == 'iesrgs' and stage == 'internal':
        from methods.iesrgs import SameTimeViewSampler
        sampler = SameTimeViewSampler([dict(o, scene_id=config['scene'], frame_id=o['frame_index']) for o in rows], config['seed'])
    started = time.time()
    with (out / 'trajectory.jsonl').open('x', buffering=1) as log:
        for local_step in range(1, steps + 1):
            step = clock_offset + local_step
            if config['method'] == 'iesrgs' and stage == 'internal':
                batch = sampler.sample()
            else:
                if not pool:
                    pool = list(rows)
                    if not source_sampling:
                        rng.shuffle(pool)
                if source_sampling:
                    batch = [pool.pop(rng.randint(0, len(pool) - 1))]
                    # Released training_with_iters always evaluates this draw
                    # before its disabled-by-default sample_more_highres flag.
                    rng.random()
                else:
                    batch = [pool.pop()]
            backend.g.update_learning_rate(step)
            if step % 1000 == 0:
                backend.g.oneupSHdegree()
            backend.g.optimizer.zero_grad(set_to_none=True)
            loss, packets = batch_loss(backend, batch, stage, priors, config)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite objective')
            loss.backward()
            grads = [p.grad for group in backend.g.optimizer.param_groups for p in group['params'] if p.grad is not None]
            if not grads or not all(bool(torch.isfinite(g).all()) for g in grads):
                raise FloatingPointError('Missing/nonfinite gradients')
            with torch.no_grad():
                topology(backend, packets, step, clock_offset + steps, out, stage)
                endpoint = 5000 if config['method'] == 'threedsr' and stage == 'sr' else 30000
                if step < endpoint:
                    backend.g.optimizer.step()
                    updates += 1
                backend.g.optimizer.zero_grad(set_to_none=True)
            log.write(json.dumps(dict(iteration=step, local_step=local_step, loss=float(loss.detach()),
                keys=[(o['camera_id'], o['frame_index']) for o in batch],
                gaussians=len(backend.g.get_xyz), seconds=time.time() - started)) + '\n')
    metadata = dict(stage='iesrgs_internal' if stage == 'internal' else stage,
        iterations=clock_offset + steps, optimizer_updates=updates)
    if source_sampling:
        metadata['sampling'] = dict(algorithm='author randint-pop plus disabled highres gate draw',
            python_rng_state_start=sampling_start, python_rng_state_end=rng.getstate(),
            pool_reset_at_stage_start=True, rng_shared_across_rounds=True, sample_more_highres=False)
    return backend.save(out / 'checkpoint.pt', metadata)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--stage', choices=['lr', 'internal', 'sr'], required=True)
    p.add_argument('--prior-bundle')
    p.add_argument('--out', required=True)
    p.add_argument('--check-only', action='store_true')
    p.add_argument('--wiring-fixture-steps', type=int)
    a = p.parse_args()
    config = read(a.config)
    if config['method'] == 'threedsr' and a.stage == 'sr' and not a.check_only:
        raise ValueError('3DSR must use its alternating driver; standalone SR would discard 3D feedback')
    m, rows, priors = preflight(config, a.stage, a.prior_bundle)
    out = resolve(a.out)
    out.mkdir(parents=True, exist_ok=False)
    record = dict(config=identity(a.config), stage=a.stage, host=socket.gethostname(),
        training_keys=len(rows), prior_bundle=identity(a.prior_bundle) if a.prior_bundle else None,
        training_HR_read=False, heldout_pixels_read=False)
    if a.check_only:
        write_new(out / 'preflight.json', dict(record, status='CPU_input_contract_passed', GPU_verified=False))
        return
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if not visible or ',' in visible or visible == '-1':
        raise ValueError('Run through resource admission with exactly one visible physical GPU')
    maximum = config['specification']['sr_steps'] if a.stage == 'sr' else config['specification']['lr_steps']
    steps = maximum if a.wiring_fixture_steps is None else a.wiring_fixture_steps
    if not 1 <= steps <= maximum:
        raise ValueError('Invalid step count')
    try:
        import numpy as np
        import torch
        from backend import WuBackend
        torch.set_num_threads(4)
        random.seed(config['seed']); np.random.seed(config['seed']); torch.manual_seed(config['seed'])
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        backend = WuBackend(config, fresh=True)
        torch.cuda.reset_peak_memory_stats()
        record.update(device=visible, gpu_name=torch.cuda.get_device_name(), gpu_uuid=str(torch.cuda.get_device_properties(0).uuid),
            runtime=backend.runtime, status='started', wiring_fixture=bool(a.wiring_fixture_steps), steps=steps,
            effective_optimization=vars(backend.opt),
            sources={str(p):identity(p) for p in Path(__file__).parent.rglob('*.py')})
        write_new(out / 'run.json', record)
        start = time.time()
        ck = optimize(backend, config, rows, a.stage, priors, steps, out)
        torch.cuda.synchronize()
        if a.stage == 'sr' and not a.wiring_fixture_steps:
            from render_eval import evaluate
            evaluate(backend, ck, out / 'evaluation')
        write_new(out / 'complete.json', dict(record, status='wiring_fixture_completed' if a.wiring_fixture_steps else ('completed_with_evaluation' if a.stage == 'sr' else 'stage_completed'),
            checkpoint=ck, seconds=time.time() - start, peak_bytes=torch.cuda.max_memory_allocated(),
            GPU_acceptance_not_implied_by_CPU_tests=True))
    except BaseException:
        write_new(out / 'failure.json', dict(record, status='failed', error=traceback.format_exc()))
        raise

if __name__ == '__main__':
    main()
