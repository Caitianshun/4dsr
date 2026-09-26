"""Immutable endpoint gates, evaluated separately for each development camera."""
import json
from pathlib import Path

def read(p):return json.loads(Path(p).read_text())

def values(directory):
    j=read(Path(directory)/'metrics.json');a=j['aggregate']
    temporal=j['temporal_aggregate']['dynamic']
    key='gt_relative_warp_l1_mean'
    if key not in temporal:
        raise KeyError(f'Check the accepted temporal metric name: {list(temporal)}')
    return dict(psnr=a['full']['psnr_mean'],ssim=a['full']['ssim_mean'],lpips=a['full']['lpips_alex_mean'],
        dynamic_lpips=a['dynamic']['lpips_alex_spatial_mask_mean'],static_lpips=a['static']['lpips_alex_spatial_mask_mean'],
        temporal=temporal[key],lr_psnr=j['lr_reprojection_aggregate']['full']['psnr_mean'],lr_l1=j['lr_reprojection_aggregate']['full']['l1_mean'])

def differences(f,c):
    return dict(psnr=f['psnr']-c['psnr'],lpips=f['lpips']-c['lpips'],
        dynamic_lpips=f['dynamic_lpips']/c['dynamic_lpips']-1,temporal=f['temporal']/c['temporal']-1)

def decide(protocol,control,frozen,early,step):
    d={c:differences(frozen[c],control[c]) for c in control}
    baseline={c:differences(frozen[c],early[c]) for c in early}
    tolerance=dict(psnr=.001,lpips=.00005,dynamic_lpips=.001,temporal=.001)
    near=[]
    # Conservative pending near any decision boundary; do not silently relax it.
    thresholds=dict(psnr=[-.20,-.05,.10],lpips=[.003],dynamic_lpips=[.05,.03],temporal=[.05,.03])
    for c,row in d.items():
        for k,levels in thresholds.items():
            for v in levels:
                if abs(row[k]-v)<=tolerance[k]:near.append(dict(camera=c,metric=k,value=row[k],boundary=v))
    harm=any(v['psnr']<-.20 or v['dynamic_lpips']>.05 or v['temporal']>.05 for v in d.values())
    safe=all(v['lpips']<=.003 and v['dynamic_lpips']<=.03 and v['temporal']<=.03 for v in d.values())
    recovered=d['cam00']['psnr']>=.10 and d['cam01']['psnr']>=-.05 and safe
    no_control_decline=all(control[c]['psnr']-early[c]['psnr']>-.10 for c in early)
    late=no_control_decline and all(v['psnr']>=-.05 for v in d.values()) and safe
    if step==18000:
        if harm:rule,extend=1,False
        elif recovered:rule,extend=2,True
        elif late:rule,extend=3,True
        else:rule,extend=4,False
        status='stage1_extend' if extend else 'completed_negative'
        if near and not harm:status,extend='pending_threshold_uncertainty',False
    else:
        rule,extend=None,False
        candidate=recovered and all(v['psnr']>=-.10 and v['dynamic_lpips']<=-.03 and v['temporal']<=.03 for v in baseline.values())
        fidelity=all(v['psnr']>=-.10 for v in baseline.values())
        status='candidate_needs_confirmation' if candidate else ('engineering_recovery' if recovered and fidelity and not harm else 'completed_negative')
    return dict(status=status,step=step,matched_rule=rule,extend=extend,control=control,frozen=frozen,early=early,
        frozen_minus_control=d,frozen_minus_early=baseline,near_threshold=near,
        direction='PSNR higher better; LPIPS and temporal errors lower better; relative denominators are C_joint except explicitly early comparisons',
        statistical_claim='single paired paths, correlated60 frames, no independent-seed significance',protocol_sha_required=True)
