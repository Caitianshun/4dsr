#!/usr/bin/env python3
"""Separate from-initialization native LR versus all-train-camera HR audit.

These 2k/4k/6k fine-stage checkpoints cannot be concatenated with the later
matched-parent interventions. Same seed/sampling/densification rules do not
imply matched effective capacity because resolution changes gradient scale.
"""
from __future__ import annotations

import gc
import os
from pathlib import Path
import platform
import time
import json

import numpy as np
import torch

from trajectory_eval import (PROJECT, OLD, read, selection, scores, group_mean,
    file_sha, write_json, load_checkpoint, N3DVPreparedDataset,
    observation_to_4dgs_camera, resized_camera, render_image, image_array,
    downsample, read_rgb)

DEST = PROJECT/'output/dynamic_sr_20260921/native_resolution_trajectory'
SOURCE = PROJECT/'output/dynamic_sr_20260918'
RUNS = {
    'cook_spinach': {'native_lr':'cook_warmup_s20260918', 'full_hr':'cook_spinach_pilot_v1_hr_reference'},
    'cut_roasted_beef': {'native_lr':'cut_warmup_s20260918', 'full_hr':'cut_roasted_beef_pilot_v1_hr_reference'},
}
STEPS = [2000,4000,6000]


@torch.no_grad()
def evaluate(scene, branch, run, step, select, manifest, observations, metric):
    start = time.monotonic()
    path = run/f'checkpoint_{step}.pt'
    config = read(run/'config.json')
    g,_,_,ck = load_checkpoint(path)
    g._deformation.eval()
    assert ck['metadata']['manifest_sha'] == select['manifest_sha256']
    assert ck['metadata']['step'] == step and ck['metadata']['stage'] == 'fine'
    assert ck['metadata']['observation'] == ('native_lr' if branch == 'native_lr' else 'hr_reference')
    assert config['task'] == 'warmup' and config['checkpoint'] is None
    assert config['seed'] == 20260918 and config['coarse_steps'] == 1000 and config['fine_steps'] == 6000
    data = N3DVPreparedDataset(manifest,'train','lr')
    data.observations = observations
    w,h = manifest['resolutions']['hr']
    root = Path(manifest['_root'])
    rows = []
    for i,obs in enumerate(observations):
        rec = data[i]
        cam_lr = observation_to_4dgs_camera(rec,i)
        cam_hr = resized_camera(cam_lr,h,w)
        pred_hr = render_image(g,cam_hr)['render']
        pred_lr = render_image(g,cam_lr)['render']
        assert torch.isfinite(pred_hr).all() and torch.isfinite(pred_lr).all()
        gt_lr = read_rgb(root/obs['lr_path'])
        row = dict(group=select['inputs'][i]['group'],camera_id=obs['camera_id'],frame_index=obs['frame_index'],
            hr=scores(image_array(pred_hr),read_rgb(root/obs['hr_path']),metric),
            lr_integrated=scores(image_array(downsample(pred_hr,rec['image'].shape[-2:])),gt_lr,metric),
            lr_native=scores(image_array(pred_lr),gt_lr,metric))
        rows.append(row)
    groups = {}
    for group,cams in select['groups'].items():
        subset = [r for r in rows if r['group']==group]
        groups[group] = dict(count=len(subset),
            **{mode:group_mean(subset,mode) for mode in ['hr','lr_integrated','lr_native']},
            per_camera={cam:{mode:group_mean([r for r in subset if r['camera_id']==cam],mode)
                             for mode in ['hr','lr_integrated','lr_native']} for cam in cams})
    result = dict(scene=scene,branch=branch,fine_step=step,total_updates=config['coarse_steps']+step,
        checkpoint=str(path),checkpoint_sha256=file_sha(path),config=config,
        config_sha256=file_sha(run/'config.json'),metadata=ck['metadata'],point_count=len(g.get_xyz),
        active_sh_degree=g.active_sh_degree,groups=groups,rows=rows,elapsed_seconds=time.monotonic()-start)
    del g,ck,data
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    begin = time.monotonic()
    DEST.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    selections = {s:selection(s) for s in RUNS}
    protocol = dict(statement='Separate from-init warmup; fine-stage steps 2000/4000/6000 after 1000 coarse updates. Evaluation only.',
        main_comparison='Native LR trained/evaluated at native LR; full HR trained/evaluated at HR, all training cameras.',
        group_meaning='teacher_camera_train and lr_only_camera_train are camera strata borrowed from prior diagnostics, NOT distinct supervision groups in this from-init experiment. Both groups receive the respective native LR or full HR target.',
        capacity_caution='Identical max_points and densification policy, but effective point count and optimization differ with image resolution. This diagnoses native behavior, not a pure resolution causal effect.',
        inputs={s:v[0] for s,v in selections.items()},script_sha256=file_sha(__file__),
        shared_evaluator_sha256=file_sha(Path(__file__).with_name('trajectory_eval.py')),
        environment=dict(host=platform.node(),gpu=torch.cuda.get_device_name(),torch=torch.__version__,
                         cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES')))
    write_json(DEST/'protocol.json',protocol)
    import lpips
    metric = lpips.LPIPS(net='alex').cuda().eval().requires_grad_(False)
    result = dict(protocol=str(DEST/'protocol.json'),scenes={})
    state = dict(status='running',pid=os.getpid(),completed=[])
    write_json(DEST/'state.json',state)
    try:
        for scene,runs in RUNS.items():
            result['scenes'][scene] = {}
            select,manifest,obs,_ = selections[scene]
            for branch,name in runs.items():
                result['scenes'][scene][branch] = {}
                for step in STEPS:
                    value = evaluate(scene,branch,SOURCE/name,step,select,manifest,obs,metric)
                    result['scenes'][scene][branch][str(step)] = value
                    write_json(DEST/scene/branch/f'step_{step}.json',value)
                    state['completed'].append(f'{scene}/{branch}/{step}')
                    write_json(DEST/'state.json',state)
                    print(json.dumps(dict(completed=state['completed'][-1],elapsed_seconds=time.monotonic()-begin)),flush=True)
        result.update(completed=True,elapsed_seconds=time.monotonic()-begin,peak_gpu_gb=torch.cuda.max_memory_allocated()/1e9)
        write_json(DEST/'metrics.json',result)
        state.update(status='complete',elapsed_seconds=result['elapsed_seconds'])
        write_json(DEST/'state.json',state)
    except BaseException as e:
        state.update(status='failed',error=repr(e));write_json(DEST/'state.json',state);raise


if __name__=='__main__':
    main()
