"""Assess completed fixed endpoints; no training or target/hyperparameter edits."""
from pathlib import Path
import json
import shutil
import numpy as np
import torch
from PIL import Image,ImageDraw
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from temporal_sharing import ROOT,high,rgb,write,sha


def main():
    torch.set_num_threads(4)
    run=ROOT/'output/dynamic_sr_20260923/meeting_controls_v1'
    assert json.loads((run/'complete.json').read_text())['status']=='completed_and_evaluated'
    out=run/'assessment';out.mkdir(exist_ok=False);shutil.copyfile(__file__,out/'source.py')
    summaries=[];deltas=[];fits=[]
    for step in [1200,3000,6000]:
        scores={b:json.loads((run/b/f'eval_{step}/metrics.json').read_text()) for b in 'ABCD'}
        for branch,m in scores.items():
            row=dict(branch=branch,step=step,psnr=m['aggregate']['full']['psnr_mean'],ssim=m['aggregate']['full']['ssim_mean'],
                lpips=m['aggregate']['full']['lpips_alex_mean'],temporal=m['temporal_aggregate']['full']['gt_relative_warp_l1_mean'],
                dynamic_lpips=m['aggregate']['dynamic']['lpips_alex_spatial_mask_mean'],
                static_lpips=m['aggregate']['static']['lpips_alex_spatial_mask_mean'])
            summaries.append(row)
            fit=json.loads((run/branch/f'fit_{step}.json').read_text())
            own=json.loads((run/branch/f'own_target_fit_{step}.json').read_text())
            fits.append(dict(branch=branch,step=step,train_HR=fit['aggregate']['render_hr'],train_original_teacher=fit['aggregate']['render_prior'],own_target=own['aggregate'],
                train_LR_psnr_mean=float(np.mean([r['lr_psnr'] for r in fit['rows']]))))
        for base in 'ABC':
            d=next(r for r in summaries if r['step']==step and r['branch']=='D')
            a=next(r for r in summaries if r['step']==step and r['branch']==base)
            pair=[]
            for dr,ar in zip(scores['D']['rows'],scores[base]['rows']):
                assert dr['frame_index']==ar['frame_index']
                pair.append(dict(frame=dr['frame_index'],**{k:dr['spatial']['full'][k]-ar['spatial']['full'][k] for k in ['psnr','ssim','lpips_alex']}))
            deltas.append(dict(step=step,comparison=f'D-{base}',mean={k:d[k]-a[k] for k in ['psnr','ssim','lpips','temporal','dynamic_lpips','static_lpips']},
                frames_improved={k:sum(p[k]>0 if k!='lpips_alex' else p[k]<0 for p in pair) for k in ['psnr','ssim','lpips_alex']},
                worst_psnr_delta=min(p['psnr'] for p in pair),best_psnr_delta=max(p['psnr'] for p in pair),frame_deltas=pair))
    parent_path=ROOT/'output/dynamic_sr_20260919/meetroom_discussion_integrated_parent_evaluation/metrics.json'
    parent=json.loads(parent_path.read_text())
    reference=scores['A']
    identity_matches={key:parent[key]==reference[key] for key in ['manifest_sha256','script_sha256','test_camera','frame_indices','cache_key','metric_definitions']}
    assert parent['checkpoint_sha256']==json.loads((run/'A/config.json').read_text())['parent_sha256']
    write(out/'comparison.json',dict(rows=summaries,deltas=deltas,fits=fits,
        historical_parent_reference=dict(path=str(parent_path),sha256=sha(parent_path),identity_matches=identity_matches,
            used_for_comparison=False,scope='Historical evaluator identity differs; excluded from current quantitative comparisons. Exact parent checkpoint identity verified.'),
        scope='one development scene, one seed; fixed endpoints, all correlated 60 frames; no significance claim'))
    fig,axes=plt.subplots(1,4,figsize=(15,3.5),layout='constrained')
    for ax,key,title in zip(axes,['psnr','ssim','lpips','temporal'],['PSNR (higher)','SSIM (higher)','LPIPS (lower)','GT-relative temporal L1 (lower)']):
        for b in 'ABCD':
            rows=[r for r in summaries if r['branch']==b]
            ax.plot([r['step'] for r in rows],[r[key] for r in rows],marker='o',label=b)
        ax.set_title(title);ax.set_xlabel('Additional training steps');ax.grid(alpha=.25)
    axes[0].legend();fig.suptitle('MeetRoom discussion, fixed cam00 development evaluation')
    fig.savefig(out/'quality_trajectory.png',dpi=160);plt.close(fig)
    # Fixed 9/20 audit locations, reused independently of the current differences.
    rois={'time_changing_proxy':(504,420,672,588),'mostly_static':(1092,504,1260,672)}
    gt=rgb(ROOT/'data/dynamic_sr/meetroom_prepared/discussion/hr/cam00/0040.png')
    images={'HR':gt,**{b:rgb(run/b/'eval_6000/predictions/0040.png') for b in 'ABCD'}}
    highs={k:high(im,(180,320)) for k,im in images.items()};roi_rows=[]
    for name,(x0,y0,x1,y1) in rois.items():
        sl=np.s_[y0:y1,x0:x1];panel=Image.new('RGB',(5*252,290),'white');draw=ImageDraw.Draw(panel)
        for i,(b,im) in enumerate(images.items()):
            panel.paste(Image.fromarray(np.round(np.clip(im[sl],0,1)*255).astype(np.uint8)).resize((252,252),Image.Resampling.NEAREST),(252*i,38))
            draw.text((252*i+5,6),b,fill='black')
            if b!='HR':
                ph,gh=highs[b][sl],highs['HR'][sl]
                roi_rows.append(dict(region=name,branch=b,xyxy=[x0,y0,x1,y1],frame=40,
                    mse=float(np.mean((im[sl]-gt[sl])**2)),high_mse=float(np.mean((ph-gh)**2)),
                    high_energy_ratio=float(np.mean(ph**2)/max(np.mean(gh**2),1e-20)),
                    high_correlation=float(np.corrcoef(ph.ravel(),gh.ravel())[0,1])))
        draw.text((5,21),f'{name}, fixed frame40, ROI={x0,y0,x1,y1}; PNG diagnostic',fill='black')
        panel.save(out/f'{name}_frame40.png')
    write(out/'fixed_roi.json',dict(rows=roi_rows,quantization='saved uint8 PNG, distinct from full float-render main metrics',
        selection='fixed regions copied from 2026-09-20 detail audit, not selected from current method gains',
        high_operator='linear H=x-U(D_raw(x)); not an orthogonal decomposition or geometry truth'))
    write(out/'complete.json',dict(source_sha256=sha(__file__),comparison_sha256=sha(out/'comparison.json'),
        models_or_targets_updated=False,main_endpoint=6000))


if __name__=='__main__':main()
