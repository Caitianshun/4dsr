"""Explicit new/legacy checkpoint identity dispatch, retaining the original renderer."""
from dv_common import *


def checkpoint_family(checkpoint):
    m=checkpoint['metadata'];batch=checkpoint.get('sr_view_batch')
    if batch is not None:
        assert m['stage']=='sync_multiview' and m['run_id']==RUN_ID
        assert m['method'] in ['Async2','Sync2'] and m['repeat'] in ['1','2']
        assert m['task_id']==f"r{m['repeat']}_{m['method']}"
        assert batch['mode']=={'Async2':'async2','Sync2':'sync2'}[m['method']]
        assert batch['slots']==2 and batch['coefficients']==[.05,.05]
        assert batch['cursor']==m['intervention_step'] and batch['pair_sha256']==m['pair_sha256']
        return 'sync_multiview'
    if m['method']=='J_joint' and 'attribute_routing' in checkpoint:return 'historical_J1'
    if m['method']=='U' and m['intervention_step']==6000:return 'historical_U6000'
    raise ValueError('Unregistered checkpoint family')


def validate_model(model):
    family=checkpoint_family(model.checkpoint);p=require_run_root(OUT)
    assert model.branch=='ordinary_split'
    if family=='sync_multiview':
        m=model.checkpoint['metadata'];batch=model.checkpoint['sr_view_batch']
        assert m['protocol_sha256']==sha(OUT/'protocol.json')
        assert batch['pair_sha256']==p['pair_schedules'][m['repeat']]['sha256']
        assert m['schedule_sha256']==p['schedules'][m['repeat']]['sha256']
        assert batch['sources']==p['sources']
        assert m['intervention_step']==12000
    return family


def roi_metrics(legacy,pred,gt,regions,metric,device='cuda'):
    """Same existing full-context metric helpers, on float predictions before PNG."""
    if not regions:return {}
    import numpy as np
    import torch
    error=((pred-gt)**2).mean(axis=2);ssim=legacy.ssim_map_rgb(pred,gt)
    interior=np.zeros(pred.shape[:2],bool);interior[5:-5,5:-5]=True
    tx=torch.from_numpy(np.ascontiguousarray(pred)).permute(2,0,1)[None].to(device)*2-1
    ty=torch.from_numpy(np.ascontiguousarray(gt)).permute(2,0,1)[None].to(device)*2-1
    metric.spatial=True
    try:
        with torch.inference_mode():spatial=metric(tx,ty)[0,0].cpu().numpy()
    finally:metric.spatial=False
    result={}
    for name,(x0,y0,x1,y1) in regions.items():
        mask=np.zeros(pred.shape[:2],bool);mask[y0:y1,x0:x1]=True
        mse=legacy.masked_mean(error,mask)
        result[name]=dict(psnr=legacy.metric_psnr(mse),ssim=legacy.masked_mean(ssim,mask&interior),
            lpips_alex_spatial_mask=legacy.masked_mean(spatial,mask),pixel_count=int(mask.sum()),
            source_precision='clamped_float32_before_PNG',lpips_role='regional_full_context_spatial_map')
    return result
