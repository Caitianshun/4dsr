#!/usr/bin/env python3
"""Expand native 4k/6k unseen-camera diagnosis to every existing test time."""
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch

from native_trajectory_eval import SOURCE, RUNS, DEST as NATIVE
from trajectory_eval import (selection, read, scores, load_checkpoint, N3DVPreparedDataset,
    observation_to_4dgs_camera,resized_camera,render_image,image_array,read_rgb,file_sha,write_json,KEYS)

DEST=NATIVE/'cam00_full60'


@torch.no_grad()
def main():
    begin=time.monotonic();DEST.mkdir(exist_ok=False)
    torch.set_num_threads(4)
    import lpips
    metric=lpips.LPIPS(net='alex').cuda().eval().requires_grad_(False)
    result=dict(script_sha256=file_sha(__file__),protocol='All fixed 60 cam00 frames; each model evaluated at native training resolution. No checkpoint selection.',scenes={})
    for scene,runs in RUNS.items():
        select,m,_,_=selection(scene)
        data=N3DVPreparedDataset(m,'test','lr')
        data.observations=sorted(data.observations,key=lambda o:o['frame_index'])
        assert len(data)==60 and {o['camera_id'] for o in data.observations}=={'cam00'}
        root=Path(m['_root']);w,h=m['resolutions']['hr']
        result['scenes'][scene]={}
        for branch,name in runs.items():
            result['scenes'][scene][branch]={}
            mode='lr_native' if branch=='native_lr' else 'hr'
            for step in [4000,6000]:
                path=SOURCE/name/f'checkpoint_{step}.pt'
                old=read(NATIVE/scene/branch/f'step_{step}.json')
                assert file_sha(path)==old['checkpoint_sha256']
                g,_,_,ck=load_checkpoint(path);g._deformation.eval()
                assert ck['metadata']['manifest_sha']==select['manifest_sha256']
                rows=[];checks=[]
                for i,obs in enumerate(data.observations):
                    cam=observation_to_4dgs_camera(data[i],i)
                    if branch=='full_hr':cam=resized_camera(cam,h,w)
                    raw=render_image(g,cam)['render']
                    gt_path=root/obs['lr_path' if branch=='native_lr' else 'hr_path']
                    value=scores(image_array(raw),read_rgb(gt_path),metric)
                    previous=[r for r in old['rows'] if r['camera_id']=='cam00' and r['frame_index']==obs['frame_index']]
                    if previous:
                        delta={k:abs(value[k]-previous[0][mode][k]) for k in KEYS}
                        assert max(delta.values())<1e-6,delta
                        checks.append(dict(frame_index=obs['frame_index'],max_abs_delta=max(delta.values())))
                    rows.append(dict(frame_index=obs['frame_index'],metrics=value,target=str(gt_path),target_sha256=file_sha(gt_path)))
                val=dict(scene=scene,branch=branch,step=step,mode=mode,checkpoint=str(path),checkpoint_sha256=old['checkpoint_sha256'],
                    manifest_sha256=select['manifest_sha256'],mean={k:float(np.mean([r['metrics'][k] for r in rows])) for k in KEYS},
                    rows=rows,overlap_checks=checks,count=len(rows))
                write_json(DEST/scene/branch/f'step_{step}.json',val)
                result['scenes'][scene][branch][str(step)]=val
                del g,ck;gc.collect();torch.cuda.empty_cache()
                print(json.dumps(dict(completed=f'{scene}/{branch}/{step}',elapsed_seconds=time.monotonic()-begin)),flush=True)
    result.update(completed=True,elapsed_seconds=time.monotonic()-begin)
    write_json(DEST/'metrics.json',result)


if __name__=='__main__':main()
