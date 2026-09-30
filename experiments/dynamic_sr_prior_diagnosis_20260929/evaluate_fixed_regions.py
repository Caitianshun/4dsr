"""Paired fixed ROI and alpha checks on pre-PNG floating renders, no updates."""
from dv_common import *
shared=setup()
import numpy as np
import torch
import csv
from evaluation_adapter import roi_metrics
from prior_modules import render_sampling
from prepare_texture_maps import scalar_render
from motion_model import load_model
from n3dv_data import load_manifest
from runtime_identity import identity as runtime_identity

def main():
    torch.set_num_threads(4);p=require_run_root(OUT)
    assert os.environ['CUDA_VISIBLE_DEVICES']==p['evaluation']['physical_gpu']
    assert runtime_identity()==p['runtime']['evaluation']
    roi=read(OUT/'spectrum/roi_protocol.json');m=load_manifest(bound(p['manifest']));ev=shared.evaluator()
    observations={(o['camera_id'],o['frame_index']):o for o in m['observations']}
    import lpips
    metric=lpips.LPIPS(net='alex',spatial=False).cuda().eval().requires_grad_(False)
    result=[];assets={};start=time.time()
    for rep in ['1','2']:
        for arm in ['J1','T']:
            label=f'r{rep}_{arm}'
            cp=bound(p['historical_J1'][rep]['checkpoint']) if arm=='J1' else OUT/'runs'/label/'attempt_01/train/checkpoint_12000.pt'
            assets[label]=entry(cp);model=load_model(cp,m);identity=shared.initial_identity(model)
            for c in ['cam00','cam01']:
                for f in [40,80]:
                    o=observations[c,f];camera=ev.render_camera(m,o,f)
                    with torch.no_grad():
                        raw=render_sampling(model,camera,False)['render'];pred=ev.legacy.image_array(raw)
                        gt=ev.legacy.read_rgb(Path(m['_root'])/o['hr_path'])
                        values=roi_metrics(ev.legacy,pred,gt,roi['regions_by_camera_xyxy_exclusive'][c],metric)
                        state=shared.effective_state(model,camera.time)
                        image,_,_=scalar_render(camera,state,torch.ones((len(state['xyz']),3),device='cuda'))
                        alpha=image[0].cpu().numpy()
                    mse=((pred.astype(np.float64)-gt.astype(np.float64))**2).mean(2)
                    for name,rect in roi['regions_by_camera_xyxy_exclusive'][c].items():
                        x0,y0,x1,y1=rect;part=np.s_[y0:y1,x0:x1]
                        result.append(dict(endpoint=label,repeat=rep,arm=arm,camera=c,frame=f,region=name,
                            **values[name],mse=float(mse[part].mean()),
                            full_frame_mse_contribution=float(mse[part].sum()/mse.size),
                            alpha_mean=float(alpha[part].mean()),alpha_lt_half_fraction=float((alpha[part]<.5).mean())))
            assert shared.initial_identity(model)==identity
            del model;torch.cuda.empty_cache()
    pairs=[]
    for a in result:
        if a['arm']!='T':continue
        b=next(b for b in result if b['arm']=='J1' and all(b[k]==a[k] for k in ['repeat','camera','frame','region']))
        pairs.append({**{k:a[k] for k in ['repeat','camera','frame','region']},
            **{'delta_'+k:a[k]-b[k] for k in ['psnr','ssim','lpips_alex_spatial_mask','mse','full_frame_mse_contribution','alpha_mean','alpha_lt_half_fraction']}})
    for name,rr in [('fixed_regions.csv',result),('fixed_region_paired_deltas.csv',pairs)]:
        with (OUT/name).open('w',newline='') as file:
            writer=csv.DictWriter(file,fieldnames=list(dict.fromkeys(k for r in rr for k in r)));writer.writeheader();writer.writerows(rr)
    write(OUT/'fixed_regions.json',dict(status='completed',rows=result,paired_deltas=pairs,
        checkpoints=assets,roi_protocol=entry(OUT/'spectrum/roi_protocol.json'),script=entry(Path(__file__)),
        observations_per_model=4,models=4,extra_rgb_forwards=16,alpha_forwards=16,parameter_updates=0,
        source_precision='clamped_float32_before_PNG',runtime=runtime_identity(),
        physical_gpu=os.environ['CUDA_VISIBLE_DEVICES'],seconds=time.time()-start,
        interpretation='fixed image coordinates, not tracked semantic regions; regional LPIPS is full-context spatial feature-map mean'))

if __name__=='__main__':main()
