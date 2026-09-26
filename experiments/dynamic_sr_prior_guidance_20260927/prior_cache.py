"""Fixed Depth Anything V2 Small diagnostic cache; separate legal/HR processes.

Raw relative inverse-depth-like predictions are kept as float32, without color
maps or per-image normalization. No outputs of this script enter LPL training.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import cv2
import numpy as np
import torch
import torch.nn.functional as F


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,indent=2))


def main():
    p=argparse.ArgumentParser()
    for k in ['manifest','teacher-index','model-source','weight','out']:p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--kind',choices=['legal','privileged'],required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4);cv2.setNumThreads(4)
    if a.kind=='legal':
        def guard(event,args):
            if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
                s=os.fsdecode(args[0]);assert not (s.endswith('.png') and '/hr/' in s),'HR read in legal cache'
        sys.addaudithook(guard)
    sys.path.insert(0,str(a.model_source))
    from depth_anything_v2.dpt import DepthAnythingV2
    assert sha(a.weight)=='715fade13be8f229f8a70cc02066f656f2423a59effd0579197bbf57860e1378'
    model=DepthAnythingV2(encoder='vits',features=64,out_channels=[48,96,192,384])
    model.load_state_dict(torch.load(a.weight,map_location='cpu',weights_only=True));model.cuda().eval()
    m=json.loads(a.manifest.read_text());root=a.manifest.parent
    index=json.loads(a.teacher_index.read_text());teachers={(e['camera'],e['frame']):e for e in index['entries']}
    observations=sorted([o for o in m['observations'] if o['split']=='train' and (o['frame_index'] in [0,40,80,118] or (o['camera_id'] in ['cam02','cam06','cam12','cam18'] and o['frame_index'] in [38,42,78,82]))],key=lambda o:(o['camera_id'],o['frame_index']))
    assert len(observations)==92
    rows=[];started=time.time();grids=set()
    def read_image(path,digest):
        assert sha(path)==digest,path
        im=cv2.imread(str(path));assert im is not None
        return im
    @torch.inference_mode()
    def infer(im,shape=(252,336)):
        tensor,_=model.image2tensor(im,518);grids.add(tuple(tensor.shape[-2:]))
        return F.interpolate(model(tensor)[:,None],size=shape,mode='bilinear',align_corners=True)[0,0].cpu().numpy().astype(np.float32)
    for o in observations:
        c,f=o['camera_id'],o['frame_index'];inputs={};pred={}
        if a.kind=='legal':
            path=root/o['lr_path'];lr=read_image(path,o['lr_sha256']);inputs[str(path)]=o['lr_sha256']
            pred['lr']=infer(cv2.resize(lr,(1344,1008),interpolation=cv2.INTER_CUBIC))
            scale=cv2.resize(cv2.resize(lr,(323,242),interpolation=cv2.INTER_AREA),(336,252),interpolation=cv2.INTER_CUBIC)
            pred['scale']=infer(cv2.resize(scale,(1344,1008),interpolation=cv2.INTER_CUBIC))
            crop=lr[3:-3,4:-4]
            pred['crop']=np.full((252,336),np.nan,np.float32)
            pred['crop'][3:-3,4:-4]=infer(cv2.resize(crop,(1344,1008),interpolation=cv2.INTER_CUBIC),crop.shape[:2])
        else:
            path=root/o['hr_path'];inputs[str(path)]=o['hr_sha256'];pred['hr']=infer(read_image(path,o['hr_sha256']))
            e=teachers[c,f];path=root/e['relative_path'];inputs[str(path)]=e['sha256'];pred['sr']=infer(read_image(path,e['sha256']))
        path=a.out/f'{c}_{f:04d}.npz';np.savez_compressed(path,**pred)
        rows.append(dict(camera=c,frame=f,path=path.name,sha256=sha(path),inputs=inputs))
        print(json.dumps(dict(camera=c,frame=f,kind=a.kind,done=len(rows))),flush=True)
    write(a.out/'complete.json',dict(status='completed',kind=a.kind,rows=rows,manifest_sha256=sha(a.manifest),model_sha256=sha(a.weight),model_commit=subprocess.check_output(['git','-C',str(a.model_source),'rev-parse','HEAD'],text=True).strip(),source_sha256=sha(__file__),actual_network_grids=sorted(grids),raw_output_grid=[252,336],input_size=518,preprocess='Official BGR to RGB /255; cubic lower-bound aspect resize to multiples of14; ImageNet mean/std. All variants first brought to1344x1008. Bilinear raw output to LR, align_corners=True.',perturbation='scale: LR336x252 ->323x242 area ->336x252 cubic; crop: LR[3:-3,4:-4], restore interior grid; NaN missing border',output='relative inverse-depth-like score, no metric scale; direction checked independently by triangulation',gpu=torch.cuda.get_device_name(),visible_cuda=os.environ.get('CUDA_VISIBLE_DEVICES'),torch=str(torch.__version__),seconds=time.time()-started,used_for_lpl_training=False))


if __name__=='__main__':main()
