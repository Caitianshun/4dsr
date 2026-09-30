"""Fixed six-column final visuals; CPU only, no scores computed from figures.

T is never substituted or synthesized. Its completed evaluation PNG is used
only for display. Every comparison/error display uses the same uint8 rounding
as the evaluator; official metrics continue to come from pre-PNG floats.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT=Path(__file__).resolve().parents[2]
DEFAULT_ROOT=ROOT/'output/dynamic_sr_prior_diagnosis_20260929'
MANIFEST=ROOT/'data/dynamic_sr/n3dv_prepared/cook_spinach/manifest.json'
TEACHER_INDEX=ROOT/'output/dynamic_sr_detail_supervision_20260924/teacher_inventory_v1/cook_spinach/teacher_index.json'
FIXED_KEYS=[(c,f) for c in ['cam00','cam01'] for f in [40,80]]
ROI_NAMES={'cam00':['face_reference','clothing_reference'],'cam01':['lamp_wall_corner_reference'],'cam02':['face_reference','clothing_reference']}
ERROR_MAX=.20
COLORS=['#00e5d0','#ffe14a']

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()

def read(p):return json.loads(Path(p).read_text())
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+f'.{os.getpid()}.tmp');t.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False));t.replace(p)
def locate(p):return Path(p) if Path(p).is_absolute() else ROOT/p
def asset(p,**extra):return dict(path=str(Path(p).resolve()),sha256=sha(p),**extra)
def rgb(path):
    with Image.open(path) as im:return np.asarray(im.convert('RGB')).astype(np.float32)/255

def display_quantization(im):return np.rint(np.clip(im,0,1)*255).astype(np.uint8).astype(np.float32)/255

def layout(im):
    assert im.ndim==3
    if im.shape[0]==3:im=im.transpose(1,2,0)
    assert im.shape==(1008,1344,3) and np.isfinite(im).all(),im.shape
    return im

def load_float(diagnostic_root,model,camera,frame):
    stem=f'{camera}_{frame:04d}';path=diagnostic_root/'observables'/model/(stem+'.npz');record=read(path.with_suffix('.json'));assert sha(path)==record['npz_sha256'];a=np.load(path)['rgb_raw'];assert a.dtype==np.float32
    return layout(a),asset(path,source_precision='original_float32_before_display_quantization',metadata=asset(path.with_suffix('.json')),checkpoint=record['checkpoint'])

def load_lr_render_bicubic(diagnostic_root,camera,frame):
    stem=f'{camera}_{frame:04d}';directory=diagnostic_root/'structure/LR6k';record=read(directory/(stem+'.json'));p=directory/(stem+'_sampling.npz');assert sha(p)==record['artifacts']['sampling']['sha256'];a=np.load(p)['lr_raw'];assert a.shape==(3,252,336) and a.dtype==np.float32
    result=F.interpolate(torch.from_numpy(a).clamp(0,1)[None],size=(1008,1344),mode='bicubic',align_corners=False,antialias=True)[0].clamp(0,1).numpy().transpose(1,2,0)
    return result,asset(p,source_precision='original_float32_LR_render',operation='clamp LR; torch bicubic align_corners=False antialias=True to HR; clamp; then shared display uint8 rounding',metadata=asset(directory/(stem+'.json')),checkpoint=record['checkpoint'])

def observation_rgb(observations,c,f,kind):
    o=observations[c,f];p=MANIFEST.parent/o[kind+'_path'];assert sha(p)==o[kind+'_sha256'];return rgb(p),asset(p,source_precision='registered_observation_PNG',kind=kind,camera=c,frame=f)

def t_evaluation(run_root,label):
    directory=run_root/'evaluation'/label;receipt_path=directory/'complete.json'
    if not receipt_path.exists():raise FileNotFoundError(f'Waiting for completed T evaluation: {receipt_path}. No T substitute is allowed.')
    rec=read(receipt_path);assert rec['status']=='completed_evaluation';assert sha(directory/'endpoint.json')==rec['endpoint_sha256'];attempt=locate(rec['attempt_directory']);assert attempt.is_relative_to(directory.resolve()),attempt
    return attempt,dict(receipt=asset(receipt_path),endpoint=asset(directory/'endpoint.json'),checkpoint_sha256=rec['checkpoint_sha256'],attempt_directory=str(attempt))

def t_png(attempt,c,f):
    path=attempt/c/'predictions'/c/f'{f:04d}.png';assert path.exists(),f'T prediction missing: {path}';im=layout(rgb(path));return im,asset(path,source_precision='completed_evaluation_uint8_PNG_display_only',formal_metrics=asset(attempt/c/'metrics.json'))

def panel(images,labels,gt,boxes,title,path,roi=None):
    n=len(images);height=6.8 if roi is None else (7.2 if (roi[2]-roi[0])/(roi[3]-roi[1])<1.1 else 6.8)
    fig,axes=plt.subplots(2,n,figsize=(3.1*n,height),squeeze=False)
    fig.subplots_adjust(left=.009,right=.949,bottom=.13,top=.865,wspace=.025,hspace=.11)
    if roi:
        x0,y0,x1,y1=roi;region=np.s_[y0:y1,x0:x1];images=[im[region] for im in images];gt=gt[region]
    for i,(im,label) in enumerate(zip(images,labels)):
        # display_quantization is already common across every image, including T PNG.
        axes[0,i].imshow(im,interpolation='nearest');axes[0,i].set_title(label,fontsize=10)
        if roi is None:
            for j,(name,box) in enumerate(boxes):
                x0,y0,x1,y1=box;axes[0,i].add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,edgecolor=COLORS[j%2],linewidth=1.0))
                if i==0:axes[0,i].text(x0+5,max(15,y0+20),str(j+1),color=COLORS[j%2],fontsize=9,weight='bold')
        error=np.abs(im-gt).mean(2);heat=axes[1,i].imshow(error,cmap='magma',vmin=0,vmax=ERROR_MAX,interpolation='nearest')
        axes[1,i].set_title('Display absolute RGB error',fontsize=8)
        for ax in axes[:,i]:ax.axis('off')
    ca=fig.add_axes([.961,.215,.012,.50]);fig.colorbar(heat,cax=ca,label='Mean absolute RGB error; fixed 0–0.20')
    fig.suptitle(title,fontsize=13,y=.978)
    suffix=('Fixed full frame. '+('; '.join(f'{i+1}: {name.replace("_reference", "").replace("_", " ")}' for i,(name,_) in enumerate(boxes)))) if roi is None else f'Fixed HR-pixel crop xyxy = {list(roi)}; no registration or sharpening.'
    fig.text(.012,.072,suffix,fontsize=9)
    footer='Display only: all columns share evaluator uint8 rounding; T comes from its saved evaluation PNG. Official metrics use pre-PNG floats.' if n==6 else 'Training-view diagnostic only: original LR, frozen SwinIR training PNG, and genuine HR at the identical registered image coordinates.'
    boundary='HR-direct is a trained quality reference, not geometry truth.' if n==6 else 'Genuine HR is diagnostic only; these are image-prior comparisons, not novel-view reconstruction results.'
    fig.text(.012,.042,footer,fontsize=9)
    fig.text(.012,.014,boundary+' Error colors clip at 0.20; no numerical scores are derived from these figures.',fontsize=9)
    fig.savefig(path,dpi=190,facecolor='white',bbox_inches='tight',pad_inches=.10);plt.close(fig)
    return asset(path,full_image=roi is None,roi_xyxy_exclusive=list(roi) if roi else None,display_error_vmin=0,display_error_vmax=ERROR_MAX)

def teacher_figures(out,observations,rois):
    teacher={(r['camera'],r['frame']):r for r in read(TEACHER_INDEX)['entries']};items=[];sources=[];c='cam02'
    for f in [40,80]:
        g,gs=observation_rgb(observations,c,f,'hr');lr,ls=observation_rgb(observations,c,f,'lr');e=teacher[c,f];path=MANIFEST.parent/e['relative_path'];assert sha(path)==e['sha256'];sr=rgb(path);nearest=np.repeat(np.repeat(lr,4,0),4,1);bic=F.interpolate(torch.from_numpy(lr.transpose(2,0,1).copy())[None],size=(1008,1344),mode='bicubic',align_corners=False,antialias=True)[0].numpy().transpose(1,2,0);ims=[display_quantization(x) for x in [nearest,bic,sr,g]];labels=['Observed LR (pixel-repeat display)','Observed LR → Bicubic','Frozen SwinIR from observed LR','Genuine HR (diagnostic only)'];boxes=[(n,rois[c][n]) for n in ROI_NAMES[c]]
        # Separate teacher figures avoid confusing held-out diagnostic LR with legal training input.
        for name,box in [('full',None)]+boxes:
            pathout=out/f'teacher_{c}_{f:04d}_{name}.png';entry=panel(ims,labels,ims[-1],boxes,f'Frozen teacher input comparison / training {c} / original frame {f}',pathout,box);entry.update(camera=c,frame=f,group='teacher_training_view',region=name);items.append(entry)
        sources.append(dict(camera=c,frame=f,HR=gs,LR=ls,SwinIR=asset(path,source_precision='actual_frozen_training_target_PNG',teacher_index=asset(TEACHER_INDEX))))
    return items,sources

def main():
    p=argparse.ArgumentParser();p.add_argument('--diagnostic-root',type=Path,default=DEFAULT_ROOT);p.add_argument('--run-root',type=Path,default=DEFAULT_ROOT);p.add_argument('--t-label',default='r1_T');p.add_argument('--out',type=Path);p.add_argument('--prepare-only',action='store_true');a=p.parse_args();torch.set_num_threads(4);start=time.time();a.diagnostic_root=a.diagnostic_root.resolve();a.run_root=a.run_root.resolve();out=(a.out or a.run_root/'final_figures').resolve();out.mkdir(parents=True,exist_ok=True)
    m=read(MANIFEST);observations={(o['camera_id'],o['frame_index']):o for o in m['observations']};roi_path=a.diagnostic_root/'spectrum/roi_protocol.json';registration=read(roi_path);assert registration['manifest_sha256']==sha(MANIFEST);rois=registration['regions_by_camera_xyxy_exclusive'];assert registration['fixed_still_frames']==[40,80]
    base={};base_sources=[]
    for c,f in FIXED_KEYS:
        gt,gs=observation_rgb(observations,c,f,'hr');lr,ls=load_float(a.diagnostic_root,'LR6k',c,f);bic,bs=load_lr_render_bicubic(a.diagnostic_root,c,f);hr,hs=load_float(a.diagnostic_root,'HR6k',c,f);b,bmeta=load_float(a.diagnostic_root,'r1_J1',c,f);base[c,f]=[display_quantization(x) for x in [gt,lr,bic,hr,b]];base_sources.append(dict(camera=c,frame=f,GT=gs,LR_direct_HR=ls,LR_render_bicubic=bs,HR_direct6k=hs,r1_J1_B=bmeta))
    manifest=dict(fixed_observations=[list(k) for k in FIXED_KEYS],source_manifest=asset(MANIFEST),roi_protocol=asset(roi_path),roi_regions={c:{n:rois[c][n] for n in ROI_NAMES[c]} for c in ROI_NAMES},columns=['Genuine HR','LR6k direct HR render','LR6k native LR render → Bicubic','HR-direct6k','r1_J1 (B)','r1_T'],source_sha256=sha(__file__),torch_version=torch.__version__,GPU_used=False)
    if a.prepare_only:
        items,sources=teacher_figures(out,observations,rois);write(out/'preparation.json',dict(status='prepared_waiting_for_completed_T',**manifest,base_sources=base_sources,teacher_sources=sources,teacher_figures=items,comparison_figures_generated=False,no_T_substitute=True,seconds=time.time()-start));print('Prepared base assets and teacher figures; final six-column panels await completed T.',flush=True);return
    attempt,receipt=t_evaluation(a.run_root,a.t_label);items=[];tsources=[]
    for c,f in FIXED_KEYS:
        t,tm=t_png(attempt,c,f);images=base[c,f]+[display_quantization(t)];assert len(images)==6;labels=['Genuine HR','LR6k / direct HR render','LR6k / LR render → Bicubic','HR-direct6k','r1_J1 / baseline B',f'{a.t_label} / final T'];boxes=[(n,rois[c][n]) for n in ROI_NAMES[c]]
        for name,box in [('full',None)]+boxes:
            path=out/f'comparison_{c}_{f:04d}_{name}.png';entry=panel(images,labels,images[0],boxes,f'cook_spinach / {c} / original frame {f} / fixed paired comparison',path,box);entry.update(camera=c,frame=f,group='six_column_model_comparison',region=name);items.append(entry)
        tsources.append(dict(camera=c,frame=f,T=tm))
    teacher_items,teacher_sources=teacher_figures(out,observations,rois);items+=teacher_items
    write(out/'index.json',dict(status='generated_complete_fixed_comparisons',**manifest,base_sources=base_sources,T_sources=tsources,T_evaluation=receipt,teacher_sources=teacher_sources,figures=items,num_required_model_comparisons=10,num_teacher_panels=6,seconds=time.time()-start,visual_inspection='pending',display_arithmetic='All source float images clamp then np.rint(*255)/255, same as evaluator; T PNG already has this quantization. Heatmaps are mean channel absolute display-image differences, fixed0..0.20. No report metrics are computed from them.',information_boundary='Teacher panels use legal training cam02; dev cam00/01 contain no claimed teacher supervision. Genuine HR is diagnostic only.'))
    print('Final figures complete:',len(items),str(out/'index.json'),flush=True)
if __name__=='__main__':main()
