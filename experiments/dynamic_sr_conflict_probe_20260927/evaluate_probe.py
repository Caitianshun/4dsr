"""Train-only teacher metrics, then separately authorized frozen-decision HR audit."""
import argparse
from probe_common import *
from parameter_groups import configure,geometry_hash

def main():
    pa=argparse.ArgumentParser();pa.add_argument('--protocol',type=Path,required=True);pa.add_argument('--out',type=Path,required=True);pa.add_argument('--checkpoint',type=Path,required=True);pa.add_argument('--arm',required=True);pa.add_argument('--repeat-id',type=int,required=True);pa.add_argument('--step',type=int,required=True);pa.add_argument('--privileged',action='store_true');a=pa.parse_args();p=load_protocol(a.protocol);a.out.mkdir(parents=True,exist_ok=False);tick=time.time();torch.set_num_threads(4)
    if a.privileged:assert read(Path(p['_root'])/'decision.json')['status']=='frozen'
    reads,blocked=guard_images(p,privileged=a.privileged);_,data,masks=load_data(p,include_dev=a.privileged,hr=a.privileged);model=fixed.load_baked(a.checkpoint);configure(model,p);before=model_hash(model);rows=[];values={};roi={}
    if a.privileged:
        import lpips
        metric=lpips.LPIPS(net='alex',spatial=False).cuda().eval().requires_grad_(False)
        roi=read(legacy.paths()['roi'])['regions_by_camera_xyxy_exclusive']
    from evaluate_probes import scalar_error
    assert Path(sys.modules['evaluate_probes'].__file__).resolve()==OLD/'evaluate_probes.py'
    with torch.no_grad():
        for c,r in data.items():
            pred=render(model,r['camera']);values[c]=metrics_from_render(pred,r,masks.get(c));h=detail.highpass(pred,(252,336));low=downsample(pred,(252,336))
            regions={'full':None}
            if c in masks:regions['common_tracks']=masks[c]
            if a.privileged:
                for n,(x0,y0,x1,y1) in roi.get(c,{}).items():
                    mask=torch.zeros_like(pred[0],dtype=torch.bool);mask[y0:y1,x0:x1]=True;regions['historical_'+n]=mask
            identity=dict(run_id=f'r{a.repeat_id}_{a.arm}',repeat_id=a.repeat_id,arm=a.arm,step=a.step,checkpoint_sha=sha(a.checkpoint),camera=c,frame=40,split=r['observation']['split'],operator_version=VERSION,privileged_eval=a.privileged)
            def record(region,reference,name,value,count,reason=None):rows.append(dict(**identity,region=region,reference=reference,metric=name,value=value,count=count,missing_reason=reason))
            for reg,mask in regions.items():
                pixels=pred.shape[1]*pred.shape[2] if mask is None else int(mask.sum())
                for ref in ['teacher','HR']:
                    target=r.get('teacher' if ref=='teacher' else 'hr')
                    if target is None:
                        if ref=='teacher':record(reg,ref,'H_L1',None,0,'No heldout teacher allowed')
                        continue
                    th=detail.highpass(target,(252,336));rp,tp=(pred,target) if mask is None else (pred[:,mask],target[:,mask]);hp,ht=(h,th) if mask is None else (h[:,mask],th[:,mask])
                    for k,v in scalar_error(rp,tp).items():record(reg,ref,'RGB_'+k.upper(),v,pixels)
                    for k,v in scalar_error(hp,ht).items():record(reg,ref,'H_'+k.upper(),v,pixels)
                if reg=='full':record(reg,'LR','RGB_L1',values[c]['LR'],252*336)
            if a.privileged:
                gt=r['hr'];arr=evalmod.legacy.image_array(pred);truth=evalmod.legacy.image_array(gt)
                quality=evalmod.legacy.spatial_metrics(arr,truth,np.zeros(truth.shape[:2],bool),metric)['full']
                for k,v in quality.items():record('full','HR',k,float(v),1344*1008)
                from PIL import Image
                Image.fromarray((pred.clamp(0,1).permute(1,2,0).cpu().numpy()*255).round().astype(np.uint8)).save(a.out/f'{c}.png')
    assert before==model_hash(model);csvwrite(a.out/'metrics.csv',rows);write(a.out/'complete.json',dict(status='completed',arm=a.arm,repeat_id=a.repeat_id,step=a.step,checkpoint=str(a.checkpoint),checkpoint_sha256=sha(a.checkpoint),values=values,rows=rows,state_unchanged=True,geometry_hash=geometry_hash(model),seconds=time.time()-tick,counts=COUNTS,image_reads=reads,blocked=blocked,privileged=a.privileged,source_sha256=sources()))
if __name__=='__main__':main()
