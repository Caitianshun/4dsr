"""Pinned existing mathematics and data identities for the bounded SH probe."""
import copy
import csv
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
OUT=ROOT/'output/dynamic_sr_conflict_probe_20260927'
OLD=ROOT/'experiments/dynamic_sr_evidence_repair_20260927'
sys.path.insert(0,str(OLD))
legacy=importlib.import_module('context')
assert Path(legacy.__file__).resolve()==OLD/'context.py'
fixed=importlib.import_module('fixed_model')
assert Path(fixed.__file__).resolve()==OLD/'fixed_model.py'
motion=legacy.motion;detail=legacy.detail;evalmod=legacy.evalmod
sha=legacy.sha256;cpu=legacy.cpu;digest=legacy.digest
downsample=legacy.downsample;image_tensor=legacy.image_tensor
read=lambda p:json.loads(Path(p).read_text())
VERSION='conflict_raw_signed_residual_v1'
COUNTS={'rgb_forwards':0,'auxiliary_forwards':0}

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False));temp.replace(path)

def csvwrite(path,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

def render(model,camera):
    COUNTS['rgb_forwards']+=1
    return fixed.render_baked(model,camera)['render']

def guard_images(protocol,privileged=False):
    allowed={str(Path(x['path']).resolve()) for x in protocol['training_files']};reads={};blocked=[]
    def hook(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        p=os.fsdecode(args[0])
        if not p.lower().endswith(('.png','.jpg','.jpeg')):return
        full=str(Path(p).resolve())
        if not privileged and full not in allowed:
            blocked.append(full);raise AssertionError('Image outside legal training whitelist: '+full)
        reads[full]=reads.get(full,0)+1
    sys.addaudithook(hook)
    return reads,blocked

def load_data(protocol,include_dev=False,hr=False):
    m=legacy.load_manifest(protocol['manifest']['path']);root=Path(m['_root']);inventory=read(protocol['teacher_index']['path'])
    teachers={(e['camera'],e['frame']):e for e in inventory['entries']};data={}
    for o in sorted(m['observations'],key=lambda x:x['camera_id']):
        if o['frame_index']!=40 or (not include_dev and o['split']!='train'):continue
        c=o['camera_id'];lrpath=root/o['lr_path'];assert sha(lrpath)==o['lr_sha256']
        row=dict(observation=o,camera=evalmod.render_camera(m,o,len(data)),lr=image_tensor(lrpath))
        if o['split']=='train':
            t=teachers[c,40];tp=root/t['relative_path'];assert sha(tp)==t['sha256'];row['teacher']=image_tensor(tp)
            with torch.no_grad():row['teacher_H']=detail.highpass(row['teacher'],row['lr'].shape[-2:])
        if hr:
            assert sha(root/o['hr_path'])==o['hr_sha256'];row['hr']=image_tensor(root/o['hr_path'])
        data[c]=row
    assert len(data)==(21 if include_dev else 19)
    masks={}
    for c,item in protocol['masks'].items():
        assert sha(item['path'])==item['sha256']
        masks[c]=torch.from_numpy(np.load(item['path'])['M_hr']).cuda()
    return m,data,masks

def load_initial(protocol):
    initial=read(Path(protocol['_root'])/'initial_state.json')
    assert sha(initial['path'])==initial['sha256']
    return fixed.load_baked(initial['path'])

def load_protocol(path):
    p=read(path);p['_root']=str(Path(path).resolve().parent);p['_protocol_path']=str(Path(path).resolve());return p

def model_hash(model):return digest(fixed.state(model))

def metrics_from_render(pred,row,mask=None):
    lrsize=row['lr'].shape[-2:];low=downsample(pred,lrsize);h=detail.highpass(pred,lrsize)
    d=dict(LR=float((low-row['lr']).abs().double().mean()))
    if 'teacher' in row:
        dr=(pred-row['teacher']).abs();dh=(h-row['teacher_H']).abs()
        d.update(RGB=float(dr.double().mean()),H=float(dh.double().mean()))
        if mask is not None:d.update(M_RGB=float(dr[:,mask].double().mean()),M_H=float(dh[:,mask].double().mean()),M_LR=float((low-row['lr']).abs()[:,mask[::4,::4]].double().mean()))
    return d

def observe(model,data,masks):
    with torch.no_grad():return {c:metrics_from_render(render(model,r['camera']),r,masks.get(c)) for c,r in data.items()}

def sources():
    paths=list(HERE.glob('*.py'))+[OLD/'context.py',OLD/'fixed_model.py',ROOT/'experiments/dynamic_sr_motion_bound_20260923/motion_model.py',ROOT/'experiments/dynamic_sr_detail_supervision_20260924/detail_loss.py',ROOT/'experiments/dynamic_sr_20260918/common.py',ROOT/'experiments/dynamic_sr_detail_supervision_20260924/evaluate.py']
    return {str(p):sha(p) for p in paths}
