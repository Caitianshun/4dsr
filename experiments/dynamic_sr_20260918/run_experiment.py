"""Controlled training branches on public N3DV, independent of HOI experiments."""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

from common import (UPSTREAM, appearance_parameters, defaults, downsample, image_tensor,
                    load_checkpoint, mse_psnr, named_parameters, new_model,
                    render_image, resized_camera, save_checkpoint, save_image,
                    seed_all, sha256, write_json)
from n3dv_data import (N3DVPreparedDataset, load_initial_points, load_manifest,
                       observation_to_4dgs_camera)


def load_training(manifest):
    data = N3DVPreparedDataset(manifest, 'train', 'lr', cache=True)
    records, cams = [], []
    for idx in range(len(data)):
        item = data[idx]
        records.append(item)
        cams.append(observation_to_4dgs_camera(item, idx))
    return records, cams


@torch.no_grad()
def quick_development(g, manifest, out, tag):
    data = N3DVPreparedDataset(manifest, 'dev', 'lr')
    rows = []
    hr_w, hr_h = manifest['resolutions']['hr']
    for idx in [0, len(data)//3, 2*len(data)//3, len(data)-1]:
        record = data[idx]
        cam = observation_to_4dgs_camera(record, idx)
        pred = render_image(g, cam)['render'].clamp(0, 1)
        _, psnr = mse_psnr(pred, record['image'].cuda())
        pred_hr = render_image(g, resized_camera(cam,hr_h,hr_w))['render']
        integrated = downsample(pred_hr, record['image'].shape[-2:])
        _, integrated_psnr = mse_psnr(integrated, record['image'].cuda())
        rows.append(dict(frame=record['frame_index'], psnr=psnr, integrated_lr_psnr=integrated_psnr))
        if idx in [0, len(data)//3]:
            save_image(out / f'dev_{tag}_{record["frame_index"]:04d}.png', pred)
    result = dict(tag=tag, psnr=float(np.mean([r['psnr'] for r in rows])),
                  integrated_lr_psnr=float(np.mean([r['integrated_lr_psnr'] for r in rows])), rows=rows)
    write_json(out / f'dev_{tag}.json', result)
    print(json.dumps(dict(event='dev', **result)), flush=True)
    return result


def warmup(args, manifest, out):
    records, cameras = load_training(manifest)
    initial = load_initial_points(manifest)
    centers = np.stack([np.asarray(c['c2w'])[:3, 3] for key, c in manifest['cameras'].items()
                        if key not in ['cam00', 'cam01']])
    g, h, o, extent = new_model(initial['points'], initial['colors'], centers)
    start = time.monotonic()
    run_step = 0
    lr_rng = random.Random(args.seed)
    logs = open(out / 'training.jsonl', 'a', buffering=1)
    hr_w, hr_h = manifest['resolutions']['hr']
    hr_cache = {}
    for stage, steps in [('coarse', args.coarse_steps), ('fine', args.fine_steps)]:
        # Official training resets the optimizer between coarse and fine.
        g.training_setup(o)
        for step in range(1, steps + 1):
            run_step += 1
            g.update_learning_rate(step)
            if step % 1000 == 0:
                g.oneupSHdegree()
            idx = lr_rng.randrange(len(records))
            rec, cam = records[idx], cameras[idx]
            target = rec['image'].cuda()
            if args.observation == 'native_lr':
                pkg = render_image(g, cam, stage)
                pred = pkg['render']
            elif args.observation == 'integrated_lr':
                pkg = render_image(g, resized_camera(cam, hr_h, hr_w), stage)
                pred = downsample(pkg['render'], target.shape[-2:])
            elif args.observation == 'hr_reference':
                # observations is not necessarily train-only: explicit lookup.
                observation = next(x for x in manifest['observations']
                                   if x['camera_id'] == rec['camera_id'] and x['frame_index'] == rec['frame_index'])
                if idx not in hr_cache:
                    hr_cache[idx] = image_tensor(Path(manifest['_root']) / observation['hr_path'], device='cpu')
                target = hr_cache[idx].cuda()
                pkg = render_image(g, resized_camera(cam, hr_h, hr_w), stage)
                pred = pkg['render']
            loss_image = (pred - target).abs().mean()
            loss = loss_image
            if stage == 'fine':
                loss = loss + g.compute_regulation(h.time_smoothness_weight,
                                                  h.l1_time_planes, h.plane_tv_weight)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite loss at {stage}:{step}')
            loss.backward()
            with torch.no_grad():
                if 300 < step < min(4000, steps - 500):
                    visible = pkg['visibility_filter']
                    g.max_radii2D[visible] = torch.maximum(g.max_radii2D[visible], pkg['radii'][visible])
                    g.add_densification_stats(pkg['viewspace_points'].grad, visible)
                    if step % 100 == 0 and len(g._xyz) < args.max_points:
                        g.densify(.0002, .005, extent, None, 5, 5, str(out), step, stage)
                g.optimizer.step()
                g.optimizer.zero_grad(set_to_none=True)
            if step % 100 == 0 or step == 1:
                row = dict(stage=stage, step=step, total_step=run_step, l1=float(loss_image),
                           psnr=mse_psnr(pred.detach().clamp(0,1), target)[1],
                           points=len(g._xyz), elapsed_s=time.monotonic()-start,
                           allocated_gb=torch.cuda.max_memory_allocated()/1e9)
                logs.write(json.dumps(row)+'\n')
                print(json.dumps(row), flush=True)
            if stage == 'fine' and (step % 2000 == 0 or step == steps):
                metadata = dict(scene=manifest['scene'], stage=stage, step=step,
                                manifest=manifest['_manifest_path'], manifest_sha=sha256(manifest['_manifest_path']),
                                seed=args.seed, extent=extent, observation=args.observation,
                                elapsed_s=time.monotonic()-start, args=vars(args))
                save_checkpoint(out / f'checkpoint_{step}.pt', g, h, o, metadata)
                quick_development(g, manifest, out, str(step))
    save_checkpoint(out / 'checkpoint_final.pt', g, h, o, metadata)
    write_json(out / 'complete.json', metadata)


def branch(args, manifest, out):
    g, h, o, checkpoint = load_checkpoint(args.checkpoint)
    records, cameras = load_training(manifest)
    hr_w, hr_h = manifest['resolutions']['hr']
    lr_rng = random.Random(args.seed + 177)
    sr_rng = random.Random(args.seed + 211)
    prior_cams = set(args.prior_cameras.split(','))
    sr_ids = [i for i, r in enumerate(records) if r['camera_id'] in prior_cams]
    root = Path(manifest['_root'])
    paths = {}
    for idx in sr_ids:
        obs = next(x for x in manifest['observations'] if x['camera_id'] == records[idx]['camera_id']
                   and x['frame_index'] == records[idx]['frame_index'])
        paths[idx] = root / 'sr_swinir_x4' / obs['camera_id'] / Path(obs['lr_path']).name
    if args.mode not in ['lr_native', 'lr_integrated']:
        missing = [str(p) for p in paths.values() if not p.exists()]
        if missing:
            raise FileNotFoundError(f'{len(missing)} missing prior files: {missing[:2]}')
    named = named_parameters(g)
    app = appearance_parameters(g)
    app_ids = {id(p) for p in app}
    frozen_snapshot = None
    if args.mode == 'frozen':
        for name, p in named.items():
            if id(p) not in app_ids:
                p.requires_grad_(False)
        frozen_snapshot = {n: p.detach().clone() for n,p in named.items() if id(p) not in app_ids}
    # Preserve the checkpoint optimizer state in all branches; topology and SH
    # degree remain fixed. Using grad=None prevents Adam momentum-only updates.
    audit = dict(appearance=[n for n,p in named.items() if id(p) in app_ids],
                 other=[n for n,p in named.items() if id(p) not in app_ids],
                 topology_fixed=True, sh_degree=g.active_sh_degree)
    write_json(out / 'parameter_groups.json', audit)
    log = open(out / 'training.jsonl', 'a', buffering=1)
    started = time.monotonic()
    gradient_audit = {}
    prior_cache = {}
    for step in range(1, args.steps + 1):
        g.update_learning_rate(checkpoint['metadata']['step'] + step)
        lr_idx, sr_idx = lr_rng.randrange(len(records)), sr_rng.choice(sr_ids)
        rec, cam = records[lr_idx], cameras[lr_idx]
        lr_gt = rec['image'].cuda()
        g.optimizer.zero_grad(set_to_none=True)
        if args.mode == 'lr_native':
            lr_pred = render_image(g, cam)['render']
        else:
            lr_pred = downsample(render_image(g, resized_camera(cam, hr_h, hr_w))['render'], lr_gt.shape[-2:])
        lr_loss = (lr_pred - lr_gt).abs().mean()
        reg = g.compute_regulation(h.time_smoothness_weight, h.l1_time_planes, h.plane_tv_weight)
        (lr_loss + reg).backward()
        sr_loss = torch.tensor(0., device='cuda')
        if args.mode not in ['lr_native', 'lr_integrated']:
            if sr_idx not in prior_cache:
                prior_cache[sr_idx] = image_tensor(paths[sr_idx], device='cpu')
            sr_gt = prior_cache[sr_idx].cuda()
            sr_pred = render_image(g, resized_camera(cameras[sr_idx], hr_h, hr_w))['render']
            sr_loss = (sr_pred - sr_gt).abs().mean() * args.sr_weight
            if args.mode in ['appearance', 'frozen']:
                gradients = torch.autograd.grad(sr_loss, app, allow_unused=True)
                for p, grad in zip(app, gradients):
                    if grad is not None:
                        if p.grad is None: p.grad = grad
                        else: p.grad.add_(grad)
                if step == 1:
                    gradient_audit['sr_grad_targets'] = audit['appearance']
            else:
                sr_loss.backward()
        if not torch.isfinite(lr_loss + sr_loss + reg):
            raise FloatingPointError(f'Nonfinite branch loss at {step}')
        g.optimizer.step()
        if step == 1 or step % 100 == 0:
            row = dict(step=step, mode=args.mode, lr_l1=float(lr_loss),
                       sr_weighted_l1=float(sr_loss), elapsed_s=time.monotonic()-started,
                       lr_psnr=mse_psnr(lr_pred.detach().clamp(0,1), lr_gt)[1], points=len(g._xyz),
                       allocated_gb=torch.cuda.max_memory_allocated()/1e9)
            print(json.dumps(row), flush=True)
            log.write(json.dumps(row)+'\n')
    if frozen_snapshot is not None:
        changes = {n: float((named[n].detach()-v).abs().max()) for n,v in frozen_snapshot.items()}
        gradient_audit['frozen_max_changes'] = changes
        if max(changes.values()) != 0:
            raise AssertionError('Frozen parameters changed')
    write_json(out / 'gradient_audit.json', gradient_audit)
    metadata = dict(scene=manifest['scene'], stage='branch', mode=args.mode,
                    step=checkpoint['metadata']['step']+args.steps, args=vars(args),
                    parent_checkpoint=str(Path(args.checkpoint).resolve()),
                    parent_sha=sha256(args.checkpoint), manifest=manifest['_manifest_path'],
                    manifest_sha=sha256(manifest['_manifest_path']), seed=args.seed,
                    elapsed_s=time.monotonic()-started, prior_cameras=sorted(prior_cams),
                    extent=checkpoint['metadata']['extent'])
    save_checkpoint(out / 'checkpoint_final.pt', g, h, o, metadata)
    quick_development(g, manifest, out, 'final')
    write_json(out / 'complete.json', metadata)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--task', choices=['warmup','branch'], required=True)
    p.add_argument('--checkpoint')
    p.add_argument('--seed', type=int, default=20260918)
    p.add_argument('--coarse-steps', type=int, default=1000)
    p.add_argument('--fine-steps', type=int, default=6000)
    p.add_argument('--max-points', type=int, default=120000)
    p.add_argument('--observation', choices=['native_lr','integrated_lr','hr_reference'], default='native_lr')
    p.add_argument('--mode', choices=['lr_native','lr_integrated','joint','appearance','frozen'], default='joint')
    p.add_argument('--steps', type=int, default=1200)
    p.add_argument('--sr-weight', type=float, default=.1)
    p.add_argument('--prior-cameras', default='cam02,cam06,cam12,cam18')
    args = p.parse_args()
    torch.set_num_threads(4)
    seed_all(args.seed)
    out = Path(args.out)
    if (out / 'complete.json').exists() or (out / 'training.jsonl').exists():
        raise FileExistsError(f'Refusing to overwrite existing run: {out}')
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / 'config.json', vars(args))
    source_files=list(Path(__file__).parent.glob('*.py'))
    upstream=UPSTREAM
    source_files += [upstream/'scene/gaussian_model.py',upstream/'scene/deformation.py',
                     upstream/'gaussian_renderer/__init__.py',upstream/'scene/hexplane.py']
    snapshot=out/'source_snapshot'
    snapshot.mkdir(exist_ok=True)
    source_rows=[]
    for i,path in enumerate(source_files):
        dest=snapshot/f'{i:02d}_{path.name}'
        shutil.copy2(path,dest)
        source_rows.append(dict(path=str(path.resolve()),snapshot=str(dest),sha256=sha256(path)))
    write_json(out/'runtime.json',dict(torch=torch.__version__,cuda=torch.version.cuda,
               device=torch.cuda.get_device_name(),python=sys.version,sources=source_rows,
               note='Evaluation helpers may evolve; the train/common/data source snapshots identify this run.'))
    m = load_manifest(args.manifest)
    if args.task == 'warmup': warmup(args, m, out)
    else: branch(args, m, out)


if __name__ == '__main__':
    main()
