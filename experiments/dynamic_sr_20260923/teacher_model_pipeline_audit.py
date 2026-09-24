"""Posthoc fixed train-view LR/teacher/model/HR comparison, CPU evaluation only."""
from pathlib import Path
import json, time, shutil
import numpy as np
import torch
import lpips
from PIL import Image, ImageDraw
from temporal_sharing import ROOT, rgb, resize, measure, write, sha


def main():
    base=ROOT/'output/dynamic_sr_20260923/meeting_controls_v1'
    confirm=ROOT/'output/dynamic_sr_20260923/meeting_D_gpu0_confirmation_v1'
    assert json.loads((confirm/'complete.json').read_text())['status']=='completed_and_evaluated'
    out=confirm/'teacher_pipeline_audit';out.mkdir(exist_ok=False)
    shutil.copyfile(__file__,out/'source.py');start=time.monotonic();torch.set_num_threads(4)
    metric=lpips.LPIPS(net='alex').cpu().eval().requires_grad_(False)
    data=ROOT/'data/dynamic_sr/meetroom_prepared/discussion'
    cache=ROOT/'output/dynamic_sr_20260923/meeting_training_targets_v1'
    manifest=json.loads((cache/'manifest.json').read_text());rows=[];inputs=[]
    selected=[r for r in manifest['targets'] if r['frame'] in [0,40,80,118]]
    assert len(selected)==16
    crops=json.loads((ROOT/'output/dynamic_sr_20260919/final_assessment/detail_audit/metrics.json').read_text())['scenes']['meetroom_discussion']['observations'][0]['crops']
    for row in selected:
        cam,frame=row['camera'],row['frame'];name=f'{frame:04d}.png'
        paths=dict(HR=data/'hr'/cam/name,LR=data/'lr'/cam/name,A=data/'sr_swinir_x4'/cam/name,
                   C=cache/row['files']['C']['path'],D=cache/row['files']['D']['path'])
        for key,path in paths.items():
            digest=sha(path);inputs.append(dict(camera=cam,frame=frame,kind=key,path=str(path),sha256=digest))
            if key in ['C','D']:assert digest==row['files'][key]['sha256']
        gt,lr=rgb(paths['HR']),rgb(paths['LR'])
        targets=dict(bicubic=resize(lr,gt.shape[:2]),A=rgb(paths['A']),C=np.load(paths['C']),D=np.load(paths['D']))
        for method,pred in targets.items():
            rows.append(dict(camera=cam,frame=frame,group='boundary' if frame in [0,118] else 'interior',method=method,metrics=measure(pred,gt,lr,metric)))
        if cam=='cam02' and frame==40:
            images={'HR':gt,'LR bicubic':targets['bicubic'],'teacher A':targets['A'],'teacher C':targets['C'],'teacher D':targets['D']}
            for method,folder in [('model A',base/'A'),('model C',base/'C'),('model D GPU0',confirm/'D')]:
                path=folder/'train_renders_6000'/cam/name;inputs.append(dict(path=str(path),sha256=sha(path)));images[method]=rgb(path)
            for region,crop in crops.items():
                x0,y0,x1,y1=[crop[k] for k in ['x0','y0','x1','y1']]
                panel=Image.new('RGB',(8*210,242),'white');draw=ImageDraw.Draw(panel)
                for i,(label,im) in enumerate(images.items()):
                    patch=np.uint8(np.round(np.clip(im[y0:y1,x0:x1],0,1)*255))
                    panel.paste(Image.fromarray(patch).resize((210,210),Image.Resampling.NEAREST),(i*210,32))
                    draw.text((i*210+4,4),label,fill='black')
                draw.text((4,18),f'cam02 frame40 {region}; historical fixed ROI; diagnostic training view',fill='black')
                panel.save(out/f'cam02_frame40_{region}.png')
    means=[]
    for group in ['all','interior','boundary']:
        for method in ['bicubic','A','C','D']:
            subset=[r for r in rows if r['method']==method and (group=='all' or r['group']==group)]
            means.append(dict(group=group,method=method,count=len(subset),metrics={k:float(np.mean([r['metrics'][k] for r in subset])) for k in subset[0]['metrics']}))
    write(out/'summary.json',dict(rows=rows,means=means,inputs=inputs,source_sha256=sha(__file__),elapsed_seconds=time.monotonic()-start,
        scope='Posthoc 16 fixed training observations; teacher quality is not a strict new-view upper bound. No parameter/target updates; CPU only.',
        crops='Historical cam02 frame40 crops selected by HR high energy in prior audit; frozen before current method differences, illustrative only.',
        precision='Targets float32 clipped for metrics, model panels saved uint8; direct model float metrics remain in fit JSON'))
    write(out/'complete.json',dict(status='completed',source_sha256=sha(__file__),summary_sha256=sha(out/'summary.json'),parameter_updates=0))


if __name__=='__main__':main()
