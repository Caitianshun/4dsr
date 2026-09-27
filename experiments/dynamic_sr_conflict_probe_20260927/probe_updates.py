"""Four disposable actual Adam candidates and conditional attribute transfers."""
import argparse
import torch.nn.functional as F
from probe_common import *
from parameter_groups import *

def clamp_state(model,camera,pred):
    from utils.sh_utils import eval_sh
    with torch.no_grad():
        dirs=model.xyz-camera.camera_center.to(model.xyz.device);dirs=dirs/dirs.norm(dim=1,keepdim=True)
        sh=torch.cat((model.sh_dc,model.sh_rest),1).transpose(1,2);pre=eval_sh(model.degree,sh,dirs)+.5
        low=F.interpolate(pred[None],size=(252,336),mode='bicubic',align_corners=False,antialias=True)[0]
        return dict(SH_preclamp_nonpositive_fraction_all_point_channels=float((pre<=0).float().mean()),RGB_zero_fraction=float((pred==0).float().mean()),RGB_above_one_fraction=float((pred>1).float().mean()),D_preclamp_outside_fraction=float(((low<0)|(low>1)).float().mean()),scope='SH fraction includes all point channels, not visibility-weighted')

def observe_full(model,data,masks):
    vals={};clamps={};anchors={}
    with torch.no_grad():
        for c,r in data.items():
            pred=render(model,r['camera']);vals[c]=metrics_from_render(pred,r,masks.get(c));clamps[c]=clamp_state(model,r['camera'],pred)
            if c in masks:anchors[c]=pred.detach().clone()
    return vals,clamps,anchors

def apply_values(model,theta,base,target,names,epsilon):
    with torch.no_grad():
        for n,v in model.named_parameters():
            if n not in names:value=theta[n]
            elif epsilon==0:value=base[n]
            elif epsilon==1:value=target[n]
            else:value=base[n].double()+epsilon*(target[n].double()-base[n].double())
            v.copy_(value.to(device=v.device,dtype=v.dtype));v.grad=None;v.requires_grad_(n in names)

def projected_centers(x,camera):
    x=x.double();P=camera.full_proj_transform.to(device=x.device,dtype=torch.float64);clip=torch.cat((x,torch.ones_like(x[:,:1])),1)@P
    return (clip[:,:2]/clip[:,3:4]+1)*torch.tensor([camera.image_width,camera.image_height],device=x.device,dtype=torch.float64)/2-.5

def main():
    pa=argparse.ArgumentParser();pa.add_argument('--protocol',type=Path,required=True);pa.add_argument('--out',type=Path,required=True);pa.add_argument('--checkpoint',type=Path);pa.add_argument('--state-id',required=True);pa.add_argument('--resume',action='store_true');a=pa.parse_args();p=load_protocol(a.protocol);a.out.mkdir(parents=True,exist_ok=a.resume);tick=time.time();torch.set_num_threads(4);reads,blocked=guard_images(p);_,data,masks=load_data(p)
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False) if a.checkpoint else None
    source=fixed.load_baked(a.checkpoint) if ck else load_initial(p);theta=fixed.state(source)['values'];shstate=ck['optimizer'] if ck else None;source_hash=model_hash(source);optimizer_hash=digest(shstate);candidates={};candidate_rows=[];steps=0
    existing={r['candidate']:r for r in read(a.out/'candidates.json')} if a.resume else {}
    for key,(wa,wb) in {'0':(0.,0.),'A':(.05,0.),'B':(0.,.05),'AB':(.05,.05)}.items():
        if key in existing:
            entry=existing[key];assert sha(entry['path'])==entry['sha256'];saved=torch.load(entry['path'],map_location='cpu',weights_only=False);candidates[key]=saved['values'];candidate_rows.append(entry);steps+=1;continue
        model=fixed.Baked({n:v.cuda() for n,v in theta.items()},source.base_count,source.degree);opt=configure(model,p,all_attributes=True,sh_state=shstate);initial_optimizer_hash=digest(cpu(opt.state_dict()));opt.zero_grad(set_to_none=True);l0=0.
        for c,r in data.items():
            pred=render(model,r['camera']);loss=(downsample(pred,(252,336))-r['lr']).abs().mean()/19;loss.backward();l0+=float(loss)
        for c,w in [(p['a'],wa),(p['b'],wb)]:
            pred=render(model,data[c]['camera']);loss=w*(pred-data[c]['teacher']).abs().mean();loss.backward()
        gradients={n:cpu(v.grad) if v.grad is not None else None for n,v in model.named_parameters()};opt.step();steps+=1;write(a.out/'update_ledger.json',dict(actual_adam_candidates=steps,state_id=a.state_id))
        values=fixed.state(model)['values'];delta={n:values[n].double()-theta[n].double() for n in theta};candidates[key]=values
        path=a.out/f'candidate_{key}.pt';torch.save(dict(values=values,delta_float64=delta,gradients=gradients,optimizer=cpu(opt.state_dict()),initial_optimizer_hash=initial_optimizer_hash),path)
        candidate_rows.append(dict(candidate=key,path=str(path),sha256=sha(path),L0=l0,initial_optimizer_hash=initial_optimizer_hash,gradient_none={n:v is None for n,v in gradients.items()},gradient_statistics=named_statistics(gradients,source.base_count),update_statistics=named_statistics(delta,source.base_count)))
        del model,opt,gradients;torch.cuda.empty_cache()
    assert len({x['initial_optimizer_hash'] for x in candidate_rows})==1;write(a.out/'candidates.json',candidate_rows)
    model=fixed.Baked({n:v.cuda() for n,v in theta.items()},source.base_count,source.degree);transfer=[];perturbations=[];contracts=[];base_receipts=[]
    if a.resume and (a.out/'bases.json').exists():
        base_receipts=read(a.out/'bases.json');perturbations=read(a.out/'perturbations.json');contracts=read(a.out/'contracts.json')
        with (a.out/'transfer.csv').open() as f:
            for row in csv.DictReader(f):
                for key in ['epsilon','parameter_update_RMS','H_first_order_prediction','H_repeat_zero_envelope','H_readable_threshold','RGB_before','RGB_after','RGB_delta','H_before','H_after','H_delta','LR_before','LR_after','LR_delta']:row[key]=float(row[key])
                for key in ['nonfinite','count']:row[key]=int(row[key])
                row['conditional_on']=row['conditional_on'] or None;transfer.append(row)
    for group,names in p['groups'].items():
        for basekey,target_specs in [('0',[('A','a',None),('B','b',None)]),('A',[('AB','b','a')]),('B',[('AB','a','b')])]:
            if any(r['group']==group and r['base']==basekey for r in base_receipts):continue
            base=candidates[basekey];apply_values(model,theta,base,base,names,0);base_hash=model_hash(model);vals,clamps,base_images=observe_full(model,data,masks)
            # Exactly one extra zero-amplitude exact-copy rendering serves both
            # the same-base repeat and zero-perturbation numerical observation.
            apply_values(model,theta,base,base,names,0);assert model_hash(model)==base_hash;repeat,_,_=observe_full(model,data,masks)
            envelope={c:{reg:abs(repeat[c][prefix+'H']-vals[c][prefix+'H']) for reg,prefix in [('full',''),('common_tracks','M_')] if prefix+'H' in vals[c]} for c in data}
            prediction={};grad_meta=[]
            ds={target:{n:(candidates[target][n].double()-base[n].double()).cuda() for n in names} for target,_,_ in target_specs}
            for c,r in data.items():
                pred=render(model,r['camera']);err=(detail.highpass(pred,(252,336))-r['teacher_H']).abs()
                for region,mask in [('full',None)]+([('common_tracks',masks[c])] if c in masks else []):
                    loss=err.mean() if mask is None else err[:,mask].mean();grads=torch.autograd.grad(loss,[getattr(model,n) for n in names],retain_graph=(mask is None and c in masks),allow_unused=True)
                    for target,_,_ in target_specs:
                        prediction[c,region,target]=sum(float((g.double()*ds[target][n]).sum(dtype=torch.float64)) for n,g in zip(names,grads) if g is not None)
                    grad_meta.append(dict(camera=c,region=region,none={n:g is None for n,g in zip(names,grads)},l2={n:None if g is None else float(g.detach().double().norm()) for n,g in zip(names,grads)}))
                del pred,err
            del ds
            base_receipts.append(dict(group=group,base=basekey,parameter_hash=base_hash,metrics=vals,repeat_metrics=repeat,envelope=envelope,H_gradient_statistics=grad_meta))
            for target,source_teacher,conditional in target_specs:
                increment={n:candidates[target][n].double()-base[n].double() for n in names};stats=named_statistics(increment,source.base_count)
                for epsilon in p['epsilons']:
                    apply_values(model,theta,base,candidates[target],names,epsilon);actual_hash=model_hash(model);after,after_clamps,images=observe_full(model,data,masks)
                    effects={};changes=torch.cat([epsilon*increment[n].flatten() for n in names]);rms=float(changes.square().mean().sqrt());nonfinite=int((~torch.isfinite(changes)).sum());assert nonfinite==0
                    if group in ['xyz','all']:
                        with torch.no_grad():
                            bx=base['xyz'].cuda() if 'xyz' in names else theta['xyz'].cuda()
                            for c in masks:
                                pix=(projected_centers(model.xyz,data[c]['camera'])-projected_centers(bx,data[c]['camera'])).norm(dim=1)
                                effects[c+'_center_displacement_hr_pixels']=dict(p50=float(pix.median()),p95=float(torch.quantile(pix,.95)),max=float(pix.max()))
                    if group in ['shape','all']:
                        with torch.no_grad():
                            old=motion.covariance(base['logscale'].cuda(),base['quaternion'].cuda()).double();new=motion.covariance(model.logscale,model.quaternion).double();rel=(new-old).norm(dim=(1,2))/old.norm(dim=(1,2)).clamp_min(1e-30)
                            effects['covariance_relative_change']=dict(p50=float(rel.median()),p95=float(torch.quantile(rel,.95)),max=float(rel.max()))
                    if group in ['opacity','all']:
                        d=(model.opacity.detach().double().sigmoid().cpu()-base['opacity'].double().sigmoid()).abs();effects['activated_opacity_change']=dict(mean=float(d.mean()),max=float(d.max()))
                    effects['anchor_RGB_change']={c:dict(mean=float((images[c]-base_images[c]).abs().double().mean()),max=float((images[c]-base_images[c]).abs().max())) for c in masks}
                    if epsilon==1:
                        expected={n:(candidates[target][n] if n in names else theta[n]) for n in theta};reference=fixed.Baked({n:v.cuda() for n,v in expected.items()},source.base_count,source.degree);assert model_hash(reference)==actual_hash
                        with torch.no_grad():
                            differences={c:float((render(reference,data[c]['camera'])-images[c]).abs().max()) for c in masks}
                        assert max(differences.values())==0;contracts.append(dict(group=group,base=basekey,target=target,epsilon=1,parameter_exact=True,anchor_render_maxabs=differences));del reference
                    perturbations.append(dict(group=group,base=basekey,target=target,source_teacher=source_teacher,conditional_on=conditional,epsilon=epsilon,statistics=stats,effects=effects,parameter_hash=actual_hash))
                    for c in data:
                        for region,prefix in [('full','')]+([('common_tracks','M_')] if c in masks else []):
                            row=dict(state_id=a.state_id,optimizer_state_hash=optimizer_hash,source_teacher=p[source_teacher],target_camera=c,group=group,epsilon=epsilon,region=region,conditional_on=p[conditional] if conditional else None,base_candidate=basekey,target_candidate=target,base_parameter_hash=base_hash,perturbed_parameter_hash=actual_hash,parameter_update_RMS=rms,base_point_range=f'0:{source.base_count}',child_point_range=f'{source.base_count}:{len(source.xyz)}',nonfinite=nonfinite,clamp_before=json.dumps(clamps[c]),clamp_after=json.dumps(after_clamps[c]),H_first_order_prediction=epsilon*prediction[c,region,target],H_repeat_zero_envelope=envelope[c][region],count=int(masks[c].sum()) if prefix else 1344*1008)
                            for measure in ['RGB','H','LR']:
                                v=vals[c][prefix+measure];w=after[c][prefix+measure];row.update({measure+'_before':v,measure+'_after':w,measure+'_delta':w-v})
                            row['H_readable_threshold']=max(p['thresholds']['transfer_absolute'],p['thresholds']['transfer_relative']*row['H_before'],row['H_repeat_zero_envelope']);transfer.append(row)
                    del images
            del base_images
            csvwrite(a.out/'transfer.csv',transfer);write(a.out/'bases.json',base_receipts);write(a.out/'perturbations.json',perturbations);write(a.out/'contracts.json',contracts)
    assert model_hash(source)==source_hash
    # Gate only the specified conditional SH response: adding b when a is present.
    checks=[]
    for epsilon in p['epsilons']:
        rr={r['target_camera']:r for r in transfer if r['group']=='SH' and r['base_candidate']=='A' and r['source_teacher']==p['b'] and r['epsilon']==epsilon and r['region']=='common_tracks'}
        va,vb=rr[p['a']],rr[p['b']];checks.append(dict(epsilon=epsilon,a_delta=va['H_delta'],a_threshold=va['H_readable_threshold'],b_delta=vb['H_delta'],b_threshold=vb['H_readable_threshold'],passed=va['H_delta']>va['H_readable_threshold'] and -vb['H_delta']>vb['H_readable_threshold']))
    write(a.out/'complete.json',dict(status='completed',state_id=a.state_id,source_state_hash=source_hash,optimizer_state_hash=optimizer_hash,source_unchanged=True,actual_adam_candidates=steps,conditional_SH_checks=checks,conditional_SH_signal=all(r['passed'] for r in checks),seconds=time.time()-tick,counts=COUNTS,rows=len(transfer),contracts_passed=len(contracts),HR_reads=0,image_reads=reads,blocked=blocked,source_sha256=sources(),repeat_note='One zero-amplitude exact-copy rerender per base is also the same-base repeat; all additional H-gradient renders are for derivatives, not envelope selection'))
    print(json.dumps(dict(state=a.state_id,candidates=steps,conditional_SH_checks=checks,seconds=time.time()-tick)))
if __name__=='__main__':main()
