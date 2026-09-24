"""Post-diagnostic, LR/SR-only matched-detail-energy control; never HR fitted.

All targets are locked before HR is read. Uses the already exposed development
observations, so this is explanatory analysis, not independent confirmation.
"""
from pathlib import Path
import json
import time
import shutil
import numpy as np
import torch
import cv2
import lpips
from temporal_sharing import ROOT, rgb, high, measure, sha, write


def main():
    torch.set_num_threads(4);cv2.setNumThreads(2)
    src=ROOT/'output/dynamic_sr_20260923/temporal_sharing_v1'
    out=src/'smoothing_control';out.mkdir(exist_ok=False)
    started=time.monotonic();shutil.copyfile(__file__,out/'source.py')
    rows=json.loads((src/'observations.json').read_text())
    protocol=dict(role='posthoc explanation on development set; no HR fit',
        target='M=A-alpha*H(A); alpha in [0,1] solves mean(H(M)^2)=mean(H(D)^2) analytically using linear H; choose smallest root',
        alpha='derived separately per observation solely from frozen A/D targets; alpha=0 if D energy>=A, nearest endpoint if no root',
        evaluation='same PSNR SSIM LPIPS and detail/edge/LR measures; numerical match is verified before/after clipping',
        source_sha256=sha(__file__),original_targets_lock_sha256=sha(src/'targets_locked.json'))
    write(out/'protocol.json',protocol)
    locked=[]
    for row in rows:
        root=ROOT/'data/dynamic_sr'/row['sub'];m=json.loads((root/'manifest.json').read_text())
        ob=next(o for o in m['observations'] if o['camera_id']==row['camera'] and o['frame_index']==row['anchor'])
        sp=root/'sr_swinir_x4'/row['camera']/Path(ob['lr_path']).name
        A=rgb(sp);D=np.load(src/row['data_file'])['D'];lrsize=(A.shape[0]//4,A.shape[1]//4)
        a=high(A,lrsize);b=high(a,lrsize);hd=high(D,lrsize)
        target=float(np.mean(hd.astype(np.float64)**2));aa=float(np.mean(b.astype(np.float64)**2))
        bb=-2*float(np.mean(a.astype(np.float64)*b));cc=float(np.mean(a.astype(np.float64)**2))-target
        disc=bb*bb-4*aa*cc
        roots=[] if aa<1e-20 or disc<0 else [(-bb-np.sqrt(disc))/(2*aa),(-bb+np.sqrt(disc))/(2*aa)]
        roots=[float(v) for v in roots if 0<=v<=1]
        if cc<=0: alpha=0.;status='D_energy_not_lower'
        elif roots:alpha=min(roots);status='root'
        else:alpha=min([0.,1.],key=lambda v:abs(aa*v*v+bb*v+cc));status='nearest_endpoint'
        M=A-alpha*a
        name=row['data_file'].replace('.npz','_M.npy');np.save(out/name,M)
        actual=float(np.mean(high(M,lrsize).astype(np.float64)**2))
        locked.append(dict(scene=row['scene'],camera=row['camera'],anchor=row['anchor'],split=row['split'],sub=row['sub'],
            file=name,sha256=sha(out/name),teacher_sha256=sha(sp),alpha=alpha,match_status=status,
            raw_high_energy_target=target,raw_high_energy_achieved=actual,relative_energy_mismatch=abs(actual-target)/max(target,1e-20)))
    write(out/'targets_locked.json',dict(targets=locked,source_sha256=sha(__file__),protocol_sha256=sha(out/'protocol.json'),
        all_targets_before_hr=True,build_seconds=time.monotonic()-started))
    metric=lpips.LPIPS(net='alex').eval().cpu();results=[]
    for row in locked:
        root=ROOT/'data/dynamic_sr'/row['sub'];m=json.loads((root/'manifest.json').read_text())
        ob=next(o for o in m['observations'] if o['camera_id']==row['camera'] and o['frame_index']==row['anchor'])
        hp=root/ob['hr_path'];assert sha(hp)==ob['hr_sha256'];assert sha(out/row['file'])==row['sha256']
        M=np.load(out/row['file']);gt=rgb(hp);lr=rgb(root/ob['lr_path'])
        results.append(dict(**row,metrics=measure(M,gt,lr,metric)))
    write(out/'evaluation_rows.json',results)
    summary={}
    original=json.loads((src/'summary.json').read_text())
    for scene in original:
        summary[scene]={}
        for split in ['calibration','check']:
            rr=[r for r in results if r['scene']==scene and r['split']==split]
            avg={k:float(np.mean([r['metrics'][k] for r in rr])) for k in rr[0]['metrics']}
            d=original[scene][split]['means']['D']
            summary[scene][split]=dict(M=avg,D_minus_M={k:d[k]-avg[k] for k in avg},
                alpha_mean=float(np.mean([r['alpha'] for r in rr])),alpha_max=max(r['alpha'] for r in rr),
                max_relative_energy_mismatch=max(r['relative_energy_mismatch'] for r in rr))
    write(out/'summary.json',summary)
    write(out/'complete.json',dict(elapsed_seconds=time.monotonic()-started,summary_sha256=sha(out/'summary.json'),
        targets_lock_sha256=sha(out/'targets_locked.json'),source_sha256=sha(__file__),parameter_updates=0))


if __name__=='__main__':main()
