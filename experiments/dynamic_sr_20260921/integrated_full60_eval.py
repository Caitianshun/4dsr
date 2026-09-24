#!/usr/bin/env python3
"""Confirm LR-only objective-space trajectory on all 60 fixed unseen times."""
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch

from trajectory_eval import (DEST as BASE, SOURCE, SCENES, selection, read, scores,
    load_checkpoint,N3DVPreparedDataset,observation_to_4dgs_camera,resized_camera,
    render_image,image_array,read_rgb,file_sha,write_json,KEYS,downsample)

DEST=BASE/'lr_only_cam00_full60'


@torch.no_grad()
def main():
    begin=time.monotonic();DEST.mkdir(exist_ok=False)
    torch.set_num_threads(4)
    import lpips
    metric=lpips.LPIPS(net='alex').cuda().eval().requires_grad_(False)
    result=dict(script_sha256=file_sha(__file__),protocol='All fixed 60 cam00 times; LR-only models evaluated with trained D(Render_HR) against real LR. Evaluation only.',scenes={})
    for scene in SCENES:
        select,m,_,_=selection(scene)
        data=N3DVPreparedDataset(m,'test','lr')
        data.observations=sorted(data.observations,key=lambda o:o['frame_index'])
        assert len(data)==60 and {o['camera_id'] for o in data.observations}=={'cam00'}
        root=Path(m['_root']);w,h=m['resolutions']['hr']
        result['scenes'][scene]={}
        for step in [6000,18000]:
            path=SOURCE/f'{scene}_lr_long/checkpoint_{step}.pt'
            old=read(BASE/scene/'lr_long'/f'step_{step}.json')
            assert file_sha(path)==old['checkpoint_sha256']
            g,_,_,ck=load_checkpoint(path);g._deformation.eval()
            assert ck['metadata']['manifest_sha']==select['manifest_sha256']
            rows=[];checks=[]
            for i,obs in enumerate(data.observations):
                item=data[i]
                cam=resized_camera(observation_to_4dgs_camera(item,i),h,w)
                raw=render_image(g,cam)['render']
                pred=downsample(raw,item['image'].shape[-2:])
                gt_path=root/obs['lr_path']
                value=scores(image_array(pred),read_rgb(gt_path),metric)
                previous=[r for r in old['rows'] if r['camera_id']=='cam00' and r['frame_index']==obs['frame_index']]
                if previous:
                    delta={k:abs(value[k]-previous[0]['lr_integrated'][k]) for k in KEYS}
                    assert max(delta.values())<1e-6,delta
                    checks.append(dict(frame_index=obs['frame_index'],max_abs_delta=max(delta.values())))
                rows.append(dict(frame_index=obs['frame_index'],metrics=value,target=str(gt_path),target_sha256=file_sha(gt_path)))
            val=dict(scene=scene,case='lr_long',step=step,mode='lr_integrated',checkpoint=str(path),checkpoint_sha256=old['checkpoint_sha256'],
                manifest_sha256=select['manifest_sha256'],mean={k:float(np.mean([r['metrics'][k] for r in rows])) for k in KEYS},
                rows=rows,overlap_checks=checks,count=len(rows))
            write_json(DEST/scene/f'step_{step}.json',val)
            result['scenes'][scene][str(step)]=val
            del g,ck;gc.collect();torch.cuda.empty_cache()
            print(json.dumps(dict(completed=f'{scene}/{step}',elapsed_seconds=time.monotonic()-begin)),flush=True)
    result.update(completed=True,elapsed_seconds=time.monotonic()-begin)
    write_json(DEST/'metrics.json',result)


if __name__=='__main__':main()
