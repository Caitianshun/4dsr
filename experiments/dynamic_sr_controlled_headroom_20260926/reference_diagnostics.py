"""Read-only fixed train16 target degradation, using the original D operator."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
import numpy as np
from PIL import Image
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/dynamic_sr_20260918'))
from common import downsample


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def image(p):
    with Image.open(p) as im:return torch.from_numpy(np.array(im.convert('RGB'),dtype=np.float32)/255.).permute(2,0,1)


def main():
    p=argparse.ArgumentParser()
    for k in ['manifest','teacher-index','out']:p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();assert not a.out.exists();m=read(a.manifest);root=a.manifest.parent;idx=read(a.teacher_index)
    assert idx['manifest_sha256']==sha(a.manifest)
    teachers={(x['camera'],x['frame']):x for x in idx['entries']};torch.set_num_threads(4);rows=[]
    with torch.inference_mode():
        for o in m['observations']:
            if o['camera_id'] not in ['cam02','cam06','cam12','cam18'] or o['frame_index'] not in [0,40,80,118]:continue
            assert o['split']=='train';paths={k:root/o[k+'_path'] for k in ['lr','hr']}
            for k in paths:assert sha(paths[k])==o[k+'_sha256']
            t=teachers[o['camera_id'],o['frame_index']];paths['teacher']=root/t['relative_path'];assert sha(paths['teacher'])==t['sha256']
            lr=image(paths['lr']);row=dict(camera=o['camera_id'],frame=o['frame_index'])
            for k in ['teacher','hr']:
                delta=(downsample(image(paths[k]),lr.shape[-2:])-lr).double();mse=float(delta.square().mean())
                row[k]=dict(l1=float(delta.abs().mean()),mse=mse,psnr=float(-10*np.log10(max(mse,1e-12))),max_abs=float(delta.abs().max()))
            rows.append(row)
    assert len(rows)==16
    value=dict(status='completed_read_only_target_diagnostic',parameter_updates=0,manifest_sha256=sha(a.manifest),teacher_index_sha256=sha(a.teacher_index),
        operator='original common.downsample; float bicubic antialias clamp, then compare observed uint8/255 LR',rows=rows,
        aggregate={k:{metric:statistics.mean(r[k][metric] for r in rows) for metric in ['l1','mse','psnr']} for k in ['teacher','hr']})
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(value,indent=2)+'\n')


if __name__=='__main__':main()
