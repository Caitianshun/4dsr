"""Paired one-step SR intervention with one exact LR gradient and null repeats.

The original independent-backward probe is retained. This diagnostic isolates
the SR contribution from nondeterministic repeated CUDA backward reductions.
Learned xyz changes are representation changes, not ground-truth motion errors.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from common import (appearance_parameters, downsample, dynamic_xyz, image_tensor,
                    load_checkpoint, named_parameters, render_image,
                    resized_camera, seed_all, sha256, write_json)
from n3dv_data import N3DVPreparedDataset, load_manifest, observation_to_4dgs_camera


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--target-frames', default='20,60,100')
    parser.add_argument('--target-camera', default='cam06')
    args = parser.parse_args()
    if Path(args.out).exists():
        raise FileExistsError(args.out)
    torch.set_num_threads(4)
    seed_all(20260918)
    started = time.monotonic()
    manifest = load_manifest(args.manifest)
    root = Path(manifest['_root'])
    dataset = N3DVPreparedDataset(manifest, 'train', 'lr', cache=True)
    hr_w, hr_h = manifest['resolutions']['hr']
    probe_frames = [0, 20, 40, 60, 80, 100, 118]
    probes = []
    for index, obs in enumerate(dataset.observations):
        if obs['camera_id'] in ['cam02', 'cam12', 'cam18'] and obs['frame_index'] in probe_frames:
            record = dataset[index]
            probes.append((record, observation_to_4dgs_camera(record, index)))
    rows = []
    for frame in map(int, args.target_frames.split(',')):
        index = next(i for i, obs in enumerate(dataset.observations)
                     if obs['camera_id'] == args.target_camera and obs['frame_index'] == frame)
        record, obs = dataset[index], dataset.observations[index]
        camera = observation_to_4dgs_camera(record, index)
        camera_hr = resized_camera(camera, hr_h, hr_w)
        lr_target = record['image'].cuda()
        sr = image_tensor(root / 'sr_swinir_x4' / record['camera_id'] / Path(obs['lr_path']).name)
        hr = image_tensor(root / obs['hr_path'])
        cached_lr_grads = None
        baseline_parameters = baseline_motion = baseline_probe_errors = None
        for kind in ['lr', 'lr_null_1', 'lr_null_2', 'sr_joint', 'sr_appearance', 'hr_oracle', 'shifted_sr']:
            g, hidden, _, checkpoint = load_checkpoint(args.checkpoint)
            g.optimizer.zero_grad(set_to_none=True)
            g.update_learning_rate(checkpoint['metadata']['step'] + 1)
            names = named_parameters(g)
            app = appearance_parameters(g)
            app_ids = {id(p) for p in app}
            app_names = {n for n, p in names.items() if id(p) in app_ids}
            categories = {n: ('appearance' if n in app_names else
                             'shared_deformation' if n.startswith('deformation.') else
                             'canonical_geometry') for n in names}
            before_parameters = {n: p.detach().clone() for n, p in names.items()}
            samples = torch.linspace(0, len(g._xyz) - 1, min(4096, len(g._xyz)), device='cuda').long()
            with torch.no_grad():
                before_motion = {t: dynamic_xyz(g, t / 300, samples).clone() for t in probe_frames}
                before_errors = []
                for rec, cam in probes:
                    image = downsample(render_image(g, resized_camera(cam, hr_h, hr_w))['render'], rec['image'].shape[-2:])
                    before_errors.append(float((image - rec['image'].cuda()).abs().mean()))

            is_null = kind.startswith('lr')
            if is_null:
                pred = render_image(g, camera_hr)['render']
                lr_loss = (downsample(pred, lr_target.shape[-2:]) - lr_target).abs().mean()
                regulation = g.compute_regulation(hidden.time_smoothness_weight, hidden.l1_time_planes, hidden.plane_tv_weight)
                (lr_loss + regulation).backward()
                initial_lr_loss = float(lr_loss)
                if kind == 'lr':
                    cached_lr_grads = {n: None if p.grad is None else p.grad.detach().clone() for n, p in names.items()}
                    cached_lr_loss = initial_lr_loss
                del pred, lr_loss, regulation
            else:
                for n, parameter in names.items():
                    grad = cached_lr_grads[n]
                    parameter.grad = None if grad is None else grad.clone()
                initial_lr_loss = cached_lr_loss

            sr_loss_value = 0.0
            if not is_null:
                target = hr if kind == 'hr_oracle' else sr
                if kind == 'shifted_sr':
                    target = torch.roll(sr, shifts=4, dims=-1)
                pred_sr = render_image(g, camera_hr)['render']
                sr_loss = 0.1 * (pred_sr - target).abs().mean()
                sr_loss_value = float(sr_loss)
                if kind == 'sr_appearance':
                    gradients = torch.autograd.grad(sr_loss, app, allow_unused=True)
                    for parameter, gradient in zip(app, gradients):
                        if gradient is not None:
                            if parameter.grad is None:
                                parameter.grad = gradient
                            else:
                                parameter.grad.add_(gradient)
                else:
                    sr_loss.backward()
                del pred_sr, sr_loss

            gradient_sq = {}
            gradient_increment_sq = {}
            other_gradient_increment_max = 0.0
            for n, parameter in names.items():
                group = categories[n]
                current = parameter.grad
                cached = cached_lr_grads[n]
                gradient_sq[group] = gradient_sq.get(group, 0.) + (float(current.square().sum()) if current is not None else 0.)
                if current is None and cached is None:
                    increment = None
                elif current is None:
                    increment = -cached
                elif cached is None:
                    increment = current
                else:
                    increment = current - cached
                gradient_increment_sq[group] = gradient_increment_sq.get(group, 0.) + (float(increment.square().sum()) if increment is not None else 0.)
                if n not in app_names and increment is not None:
                    other_gradient_increment_max = max(other_gradient_increment_max, float(increment.abs().max()))
            g.optimizer.step()
            with torch.no_grad():
                after_motion = {t: dynamic_xyz(g, t / 300, samples).clone() for t in probe_frames}
                if kind == 'lr':
                    baseline_motion = {t: x.clone() for t, x in after_motion.items()}
                    baseline_parameters = {n: p.detach().clone() for n, p in names.items()}
                displacement = {t: float((after_motion[t] - before_motion[t]).norm(dim=-1).mean()) for t in probe_frames}
                extra_motion = {t: float((after_motion[t] - baseline_motion[t]).norm(dim=-1).mean()) for t in probe_frames}
                temporal_offset = {t: float(((after_motion[t] - after_motion[0]) - (baseline_motion[t] - baseline_motion[0])).norm(dim=-1).mean()) for t in probe_frames}
                parameter_changes = {n: float((p.detach() - baseline_parameters[n]).abs().max()) for n, p in names.items()}
                other_parameter_max = max(value for n, value in parameter_changes.items() if n not in app_names)
                if kind == 'sr_appearance':
                    assert other_gradient_increment_max == 0., other_gradient_increment_max
                    assert other_parameter_max == 0., other_parameter_max
                probe_rows = []
                for (rec, cam), old in zip(probes, before_errors):
                    image = downsample(render_image(g, resized_camera(cam, hr_h, hr_w))['render'], rec['image'].shape[-2:])
                    new = float((image - rec['image'].cuda()).abs().mean())
                    probe_rows.append(dict(camera=rec['camera_id'], frame=rec['frame_index'], before=old, after=new, delta=new-old))
                if kind == 'lr':
                    baseline_probe_errors = [r['after'] for r in probe_rows]
                for row, reference in zip(probe_rows, baseline_probe_errors):
                    row['delta_vs_lr_step'] = row['after'] - reference
                cross_time = [r['delta_vs_lr_step'] for r in probe_rows if r['frame'] != frame]
                row = dict(target_frame=frame, kind=kind, initial_lr_loss=initial_lr_loss,
                           sr_weighted_loss=sr_loss_value,
                           gradient_norms={k: float(np.sqrt(v)) for k, v in gradient_sq.items()},
                           gradient_increment_vs_cached_lr_norms={k: float(np.sqrt(v)) for k, v in gradient_increment_sq.items()},
                           learned_xyz_displacement=displacement,
                           learned_xyz_increment_vs_lr=extra_motion,
                           learned_temporal_offset_increment_vs_lr=temporal_offset,
                           nonappearance_gradient_increment_max_abs=other_gradient_increment_max,
                           nonappearance_parameter_increment_max_abs=other_parameter_max,
                           parameter_increment_vs_lr_max_abs=parameter_changes,
                           probes=probe_rows,
                           mean_probe_delta_vs_lr_step=float(np.mean([r['delta_vs_lr_step'] for r in probe_rows])),
                           mean_cross_time_probe_delta_vs_lr_step=float(np.mean(cross_time)))
            rows.append(row)
            print(json.dumps(dict(frame=frame, kind=kind, xyz_extra=float(np.mean(list(extra_motion.values()))),
                                  cross_time_delta=row['mean_cross_time_probe_delta_vs_lr_step'],
                                  other_parameter_max=other_parameter_max)), flush=True)
            del g, names, before_parameters, before_motion, after_motion, app
            torch.cuda.empty_cache()
    write_json(args.out, dict(scene=manifest['scene'], checkpoint=str(Path(args.checkpoint).resolve()),
                             checkpoint_sha=sha256(args.checkpoint), manifest_sha=sha256(manifest['_manifest_path']),
                             source_sha=sha256(__file__), device=torch.cuda.get_device_name(), torch=torch.__version__,
                             elapsed_s=time.monotonic()-started, rows=rows,
                             appearance_names=sorted(app_names),
                             protocol='One cached LR+regularization gradient reused exactly for all SR branches; identical restored Adam state; two fresh LR-only backward repeats estimate numerical null noise; one update at checkpoint step+1; fixed topology.',
                             interpretation='Learned xyz and temporal offsets are representation changes, not ground-truth motion errors. HR oracle is diagnostic extra supervision. Shifted SR is a correspondence-error stress test with periodic boundary, not a texture-only perturbation.'))


if __name__ == '__main__':
    main()
