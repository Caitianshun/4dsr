"""Four preselected float-render/ROI checks and corresponding display figures."""
import ast
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'output/dynamic_sr_same_observation_20260930'
OLD=ROOT/'output/dynamic_sr_prior_diagnosis_20260929'
sys.path.insert(0,str(ROOT/'experiments/dynamic_sr_20260918'))
import numpy as np
import torch
from common import load_checkpoint,render_image,sha256
from n3dv_data import load_manifest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image


def read(p):return json.loads(Path(p).read_text())


def main():
    assert os.environ['CUDA_VISIBLE_DEVICES']=='1'
    complete=read(OUT/'train/complete.json')
    checkpoint=OUT/'train'/complete['final_checkpoint']
    assert sha256(checkpoint)==complete['final_sha256']
    directory=OUT/'figures';directory.mkdir(exist_ok=True)
    assert not (directory/'index.json').exists(), 'Completed visual comparison already exists'
    source=ROOT/'experiments/dynamic_sr_detail_supervision_20260924/evaluate.py'
    spec=importlib.util.spec_from_file_location('same_observation_fixed_evaluator',source)
    ev=importlib.util.module_from_spec(spec);spec.loader.exec_module(ev)
    helper=ROOT/'experiments/dynamic_sr_prior_diagnosis_20260929/evaluation_adapter.py'
    function=next(n for n in ast.parse(helper.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='roi_metrics')
    scope={};exec(compile(ast.Module(body=[function],type_ignores=[]),str(helper),'exec'),scope)
    roi_metrics=scope['roi_metrics']
    registration=read(OLD/'spectrum/roi_protocol.json')
    manifest_path=ROOT/'data/dynamic_sr/n3dv_prepared/cook_spinach/manifest.json'
    assert registration['manifest_sha256']==sha256(manifest_path)
    m=load_manifest(manifest_path)
    observations={(o['camera_id'],o['frame_index']):o for o in m['observations']}
    g,_,_,_=load_checkpoint(checkpoint);g._deformation.eval();torch.set_num_threads(4)
    import lpips
    metric=lpips.LPIPS(net='alex',spatial=False).to('cuda').eval().requires_grad_(False)
    rows=[];figures=[];sources=[];started=time.monotonic()
    with torch.no_grad():
        for camera in ['cam00','cam01']:
            evaluator=read(OUT/'eval'/('test' if camera=='cam00' else 'dev')/'metrics.json')
            for frame in [40,80]:
                observation=observations[camera,frame]
                raw=render_image(g,ev.render_camera(m,observation,frame))['render']
                pred=ev.legacy.image_array(raw)
                expected=next(r['spatial']['full']['psnr'] for r in evaluator['rows'] if r['frame_index']==frame)
                gt_path=manifest_path.parent/observation['hr_path']
                assert sha256(gt_path)==observation['hr_sha256']
                gt=ev.legacy.read_rgb(gt_path)
                psnr=ev.legacy.metric_psnr(float(((pred-gt)**2).mean()))
                assert abs(psnr-expected)<1e-4,(camera,frame,psnr,expected)
                raw_path=directory/f'{camera}_{frame:04d}_float.npz'
                np.savez_compressed(raw_path,rgb_raw=raw.detach().cpu().numpy())
                images=[gt];labels=['Genuine HR'];metrics={}
                regions=registration['regions_by_camera_xyxy_exclusive'][camera]
                for name in ['r1_J1','r1_Async2','P1-from-start']:
                    if name=='P1-from-start':
                        image=pred;path=raw_path
                    else:
                        path=OLD/'observables'/name/f'{camera}_{frame:04d}.npz'
                        assert sha256(path)==read(path.with_suffix('.json'))['npz_sha256']
                        with np.load(path) as values:image=np.clip(values['rgb_raw'].transpose(1,2,0),0,1)
                    q=roi_metrics(ev.legacy,image,gt,regions,metric,'cuda');metrics[name]=q
                    for region,v in q.items():
                        rows.append(dict(model=name,camera=camera,frame=frame,region=region,
                                         psnr=v['psnr'],ssim=v['ssim'],lpips_spatial=v['lpips_alex_spatial_mask']))
                    images.append(image);labels.append(name)
                    sources.append(dict(camera=camera,frame=frame,model=name,path=str(path),sha256=sha256(path)))
                region='face_reference' if camera=='cam00' else 'lamp_wall_corner_reference'
                x0,y0,x1,y1=regions[region]
                fig,axes=plt.subplots(2,4,figsize=(15,6.0),constrained_layout=True)
                for column,(image,label) in enumerate(zip(images,labels)):
                    display=np.rint(np.clip(image,0,1)*255).astype(np.uint8)
                    axes[0,column].imshow(display);axes[0,column].set_title(label,fontsize=12)
                    axes[0,column].add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,edgecolor='#00c8ba',linewidth=1.2))
                    axes[1,column].imshow(display[y0:y1,x0:x1])
                    if column:
                        v=metrics[label][region]
                        axes[1,column].set_title(f"PSNR {v['psnr']:.2f}  SSIM {v['ssim']:.3f}\nRegional LPIPS {v['lpips_alex_spatial_mask']:.3f}",fontsize=10)
                    else:axes[1,column].set_title('Same fixed image coordinates',fontsize=10)
                    for row in [0,1]:axes[row,column].axis('off')
                path=directory/f'{camera}_{frame:04d}_comparison.png'
                fig.savefig(path,dpi=150);plt.close(fig)
                figures.append(dict(camera=camera,frame=frame,region=region,path=str(path),sha256=sha256(path)))
    with (OUT/'regional_quality.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    result=dict(status='completed_fixed_visual_comparison',figures=figures,sources=sources,
                new_render_calls=4,parameter_updates=0,seconds=time.monotonic()-started,
                roi_protocol_sha256=sha256(OLD/'spectrum/roi_protocol.json'),
                metric_helper_sha256=sha256(helper),evaluation_script_sha256=sha256(source),
                policy='Metrics computed from original clamped float32; common uint8 rounding only for display. Historical regional comparisons use r1 only, not two-suffix means.',
                visual_inspection='pending')
    (directory/'index.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps(dict(status=result['status'],new_render_calls=4,figures=len(figures))))


if __name__=='__main__':main()
