"""One 32-training-observation screen-gradient ranking; zero parameter updates."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import time
import torch
from motion_model import OLD
sys.path.insert(0,str(OLD))
from common import load_checkpoint,downsample,image_tensor,render_image,resized_camera,sha256,write_json
from n3dv_data import load_manifest
from run_experiment import load_training


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--out',required=True)
    args=p.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);start=time.monotonic()
    m=load_manifest(args.manifest);g,h,o,ck=load_checkpoint(args.checkpoint)
    assert ck['metadata']['manifest_sha']==sha256(args.manifest)
    records,cameras=load_training(m);lookup={(r['camera_id'],int(r['frame_index'])):i for i,r in enumerate(records)}
    n=len(g._xyz);score=torch.zeros(n,device='cuda');counts=torch.zeros(n,device='cuda');rows=[]
    for camera in ['cam02','cam04','cam08','cam12']:
        for frame in [0,16,32,48,64,80,96,118]:
            i=lookup[(camera,frame)];obs=records[i]
            cam=resized_camera(cameras[i],m['resolutions']['hr'][1],m['resolutions']['hr'][0])
            teacher=Path(m['_root'])/'sr_swinir_x4'/camera/f'{frame:04d}.png'
            g.optimizer.zero_grad(set_to_none=True)
            package=render_image(g,cam);im=package['render'];lr=obs['image'].cuda();sr=image_tensor(teacher)
            loss=(downsample(im,lr.shape[-2:])-lr).abs().mean()+.1*(im-sr).abs().mean()
            loss.backward()
            grad=package['viewspace_points'].grad[:,:2].norm(dim=-1)
            visible=package['visibility_filter']
            assert torch.isfinite(grad).all()
            score[visible]+=grad[visible];counts[visible]+=1
            rows.append(dict(camera=camera,frame=frame,time=float(cam.time),teacher_sha256=sha256(teacher),loss=float(loss)))
    score=score/counts.clamp_min(1)
    eligible=torch.where((counts>0)&(score>0)&torch.isfinite(score))[0]
    budget=min(int(n*.2),len(eligible))
    assert budget>0
    # Stable argsort gives a deterministic point-index tie break.
    ranked=eligible[torch.argsort(score[eligible],descending=True,stable=True)]
    chosen=ranked[:budget].sort().values
    payload=dict(selected_ids=chosen.cpu(),score=score.cpu(),counts=counts.cpu(),original_count=n,
                 parent_sha256=sha256(args.checkpoint),manifest_sha256=sha256(args.manifest),
                 rule='top screen-space mean gradient of same-view full LR L1 + 0.1 frozen SR L1; visible positive finite; no HR; no size threshold; max floor(.2N) parents; two children replace each',
                 source_sha256=sha256(__file__),observations=rows)
    torch.save(payload,out/'selection.pt')
    write_json(out/'complete.json',dict(status='completed_selection',parameter_updates=0,observations=len(rows),
        selected_parents=budget,original_count=n,active_count=n+budget,max_budget_fraction=.2,
        selection_sha256=sha256(out/'selection.pt'),parent_sha256=payload['parent_sha256'],manifest_sha256=payload['manifest_sha256'],
        selected_id_sha256=hashlib.sha256(chosen.cpu().numpy().tobytes()).hexdigest(),elapsed_s=time.monotonic()-start,
        rule=payload['rule'],gpu=torch.cuda.get_device_name(),script_sha256=sha256(__file__)))
    print((out/'complete.json').read_text(),flush=True)


if __name__=='__main__':main()
