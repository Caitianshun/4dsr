"""Single-step causal intervention; uses training observations only.

This does not estimate ground-truth 3D motion. Displacements are changes to the
learned representation, and cross-time LR changes are observable proxies.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from common import (appearance_parameters, downsample, dynamic_xyz, image_tensor,
                    load_checkpoint, named_parameters, render_image, resized_camera,
                    seed_all, sha256, write_json)
from n3dv_data import N3DVPreparedDataset, load_manifest, observation_to_4dgs_camera


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--target-frames', default='20,60,100')
    p.add_argument('--target-camera', default='cam06')
    args = p.parse_args()
    torch.set_num_threads(4)
    seed_all(20260918)
    manifest = load_manifest(args.manifest)
    root = Path(manifest['_root'])
    ds = N3DVPreparedDataset(manifest, 'train', 'lr', cache=True)
    h_w, h_h = manifest['resolutions']['hr']
    target_frames = [int(x) for x in args.target_frames.split(',')]
    probe_cams = ['cam02', 'cam12', 'cam18']
    probe_frames = [0, 20, 40, 60, 80, 100, 118]
    probes = []
    for i, obs in enumerate(ds.observations):
        if obs['camera_id'] in probe_cams and obs['frame_index'] in probe_frames:
            rec = ds[i]
            probes.append((rec, observation_to_4dgs_camera(rec, i)))
    rows = []
    for frame in target_frames:
        idx = next(i for i,x in enumerate(ds.observations)
                   if x['camera_id'] == args.target_camera and x['frame_index'] == frame)
        rec, obs = ds[idx], ds.observations[idx]
        cam = observation_to_4dgs_camera(rec, idx)
        sr_path = root / 'sr_swinir_x4' / rec['camera_id'] / Path(obs['lr_path']).name
        sr = image_tensor(sr_path)
        gt_hr = image_tensor(root / obs['hr_path'])
        baseline_after_motion = None
        baseline_after_parameters = None
        baseline_probe_errors = None
        for kind in ['lr', 'sr_joint', 'sr_appearance', 'hr_oracle', 'shifted_sr']:
            # Same checkpoint and Adam history. LR-only is the counterfactual
            # control for both the gradient and the existing optimizer momentum.
            g, h, o, ck = load_checkpoint(args.checkpoint)
            g.optimizer.zero_grad(set_to_none=True)
            g.update_learning_rate(ck['metadata']['step']+1)
            names = named_parameters(g)
            before_parameters = {n:p.detach().clone() for n,p in names.items()}
            samples = torch.linspace(0,len(g._xyz)-1,min(4096,len(g._xyz)),device='cuda').long()
            with torch.no_grad():
                before_motion = {t: dynamic_xyz(g, t / 300, samples).clone() for t in probe_frames}
                before_lr = []
                for pr, pc in probes:
                    im = downsample(render_image(g, resized_camera(pc,h_h,h_w))['render'], pr['image'].shape[-2:])
                    before_lr.append(float((im-pr['image'].cuda()).abs().mean()))
            pred_hr = render_image(g, resized_camera(cam,h_h,h_w))['render']
            lr_loss = (downsample(pred_hr, rec['image'].shape[-2:])-rec['image'].cuda()).abs().mean()
            reg=g.compute_regulation(h.time_smoothness_weight,h.l1_time_planes,h.plane_tv_weight)
            (lr_loss+reg).backward()
            loss = lr_loss
            if kind != 'lr':
                target = gt_hr if kind == 'hr_oracle' else sr
                if kind == 'shifted_sr':
                    target = torch.roll(sr, shifts=4, dims=-1)
                pred_sr = render_image(g,resized_camera(cam,h_h,h_w))['render']
                loss_sr = .1*(pred_sr-target).abs().mean()
                if kind == 'sr_appearance':
                    app = appearance_parameters(g)
                    gradients = torch.autograd.grad(loss_sr, app, allow_unused=True)
                    for param, grad in zip(app, gradients):
                        if grad is not None:
                            if param.grad is None: param.grad=grad
                            else: param.grad.add_(grad)
                else:
                    loss_sr.backward()
                loss=lr_loss+loss_sr
            norms = {}
            for n,param in names.items():
                category = 'appearance' if ('features' in n or 'shs_deform' in n) else ('shared_deformation' if n.startswith('deformation.') else 'canonical_geometry')
                norms[category] = norms.get(category,0.) + (float(param.grad.square().sum()) if param.grad is not None else 0.)
            g.optimizer.step()
            with torch.no_grad():
                after_motion={t:dynamic_xyz(g,t/300,samples).clone() for t in probe_frames}
                if kind=='lr':
                    baseline_after_motion={t:v.clone() for t,v in after_motion.items()}
                    baseline_after_parameters={n:p.detach().clone() for n,p in names.items()}
                deltas = {t: float((after_motion[t]-before_motion[t]).norm(dim=-1).mean()) for t in probe_frames}
                extra_motion={t:float((after_motion[t]-baseline_after_motion[t]).norm(dim=-1).mean()) for t in probe_frames}
                step_norm=float(torch.sqrt(sum((p.detach()-before_parameters[n]).square().sum() for n,p in names.items())))
                extra_step_norm=float(torch.sqrt(sum((p.detach()-baseline_after_parameters[n]).square().sum() for n,p in names.items())))
                probe_rows=[]
                for (pr,pc), old in zip(probes,before_lr):
                    im=downsample(render_image(g,resized_camera(pc,h_h,h_w))['render'],pr['image'].shape[-2:])
                    new=float((im-pr['image'].cuda()).abs().mean())
                    probe_rows.append(dict(camera=pr['camera_id'],frame=pr['frame_index'],before=old,after=new,delta=new-old))
                if kind=='lr': baseline_probe_errors=[x['after'] for x in probe_rows]
                for row,baseline in zip(probe_rows,baseline_probe_errors):
                    row['delta_vs_lr_step']=row['after']-baseline
            rows.append(dict(target_frame=frame, kind=kind, initial_target_loss=float(loss),
                             gradient_norms={k:float(np.sqrt(v)) for k,v in norms.items()},
                             learned_xyz_displacement=deltas, learned_xyz_increment_vs_lr=extra_motion,
                             parameter_step_l2=step_norm, parameter_increment_vs_lr_l2=extra_step_norm,
                             probes=probe_rows, mean_probe_lr_delta=float(np.mean([x['delta'] for x in probe_rows])),
                             mean_probe_delta_vs_lr_step=float(np.mean([x['delta_vs_lr_step'] for x in probe_rows]))))
            print(json.dumps({k:v for k,v in rows[-1].items() if k not in ['probes']}),flush=True)
            del g, pred_hr, loss
            torch.cuda.empty_cache()
    write_json(args.out, dict(checkpoint=args.checkpoint, checkpoint_sha=sha256(args.checkpoint),
                             manifest_sha=sha256(manifest['_manifest_path']), rows=rows,
                             interpretation='Changes in learned xyz are not ground-truth motion errors; HR oracle is diagnostic-only.',
                             optimizer='Restored identical Adam state and LR at checkpoint step+1; LR+regularization versus LR+regularization+0.1*SR; one step; fixed topology'))


if __name__ == '__main__': main()
