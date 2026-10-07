"""Six controlled endpoints and finite LR-only probes from complete U6000."""
import argparse,os,sys,time,math,random,traceback,shutil
from collections import Counter
from cg_common import *
shared,policy=setup()
import numpy as np
import torch
from motion_model import load_model,refinement_state,capacity_summary,rasterize
from common import downsample,resized_camera
from n3dv_data import load_manifest
from training_support import TeacherCache,initial_identity,digest_state
from schedules import validate_parent,slots,plain
adapter=module('cg_training_adapter',ROOT/'experiments/dynamic_sr_20260918/run_experiment.py')
resume_support=module('cg_resume_support',ROOT/'experiments/dynamic_sr_20260920/resume_control.py')
def rng():return dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all(),numpy=np.random.get_state(),python=random.getstate())
def checkpoint(model,a,p,table,cursor,source,start,train_s,started):
    step=6000+cursor;ck=model.checkpoint
    meta=dict(scene='cook_spinach',stage='confidence_geometry',run_id=HERE.name,branch='ordinary_split',
      method=a.method,repeat=str(a.repeat),step=7200+step,intervention_step=step,suffix_step=cursor,
      manifest_sha=p['manifest']['sha256'],protocol_sha256=sha(OUT/'protocol.json'),schedule_sha256=sha(OUT/f'schedule_{a.repeat}.json'),
      extent=ck['metadata']['extent'],points=132972,segment_start=start,train_s=train_s,elapsed_s=time.monotonic()-started)
    payload=dict(model=model.g.capture(),hidden=vars(model.h),optim=vars(model.o),metadata=meta,rng=rng(),
      samplers=dict(kind='registered_fork',cursor=cursor,schedule_sha256=meta['schedule_sha256']),
      motion_refinement=refinement_state(model),confidence_geometry=dict(sources=source,method=a.method,repeat=a.repeat))
    path=a.out/f'checkpoint_{step}.pt';tmp=path.with_suffix('.tmp')
    with tmp.open('wb') as f:torch.save(payload,f);f.flush();os.fsync(f.fileno())
    tmp.replace(path);write(path.with_suffix('.json'),dict(path=str(path.relative_to(ROOT)),sha256=sha(path),metadata=meta))
    return path,meta
def main(a):
    p=read(OUT/'protocol.json');torch.set_num_threads(4)
    assert a.out.resolve().is_relative_to((OUT/'runs').resolve())
    a.out.mkdir(parents=True,exist_ok=False)
    source=source_identity();allowed={str(local(e['path']).resolve()):e for e in p['training_files']};reads=Counter()
    use_r=a.method in ['R','RG','RG_time','RG_ST','RG_mass','G_Q_uniform']
    use_g=a.method in ['G','RG','RG_time','RG_ST','RG_mass','G_FullSR_ST','G_Q_uniform']
    confidence=None;geometry=None;operator=None;cal={}
    if use_r or use_g or a.method=='G_FullSR_ST':
        cal=read(OUT/'calibration.json');calibration_sha=sha(OUT/'calibration.json');assert cal['status']=='passed'
        if use_r:
            opmod=module('cg_actual_degradation',HERE/'degradation_operator.py');operator=opmod.ActualDegradation((1008,1344),(252,336),device='cuda')
            source[str((HERE/'degradation_operator.py').relative_to(ROOT))]=sha(HERE/'degradation_operator.py')
        if use_r and a.method!='G_Q_uniform' or a.method=='G_FullSR_ST':
            cm=module('cg_confidence_cache',HERE/'confidence_cache.py')
            mode={'RG_time':'time','RG_ST':'ST','RG_mass':'mass','G_FullSR_ST':'ST'}.get(a.method,'view')
            index=OUT/'cache'/f'confidence_{mode}'/'cache_manifest.json'
            confidence=cm.ConfidenceCache(index)
            for e in read(index)['entries']:allowed[str((index.parent/e['path']).resolve())]=e
            source[str((HERE/'confidence_cache.py').relative_to(ROOT))]=sha(HERE/'confidence_cache.py')
        if use_g:
            gm=module('cg_depth_prior',HERE/'depth_prior.py');geometry=gm.GeometryCache(OUT/'cache/geometry/index.json')
            for e in read(OUT/'cache/geometry/index.json')['entries']:allowed[str((OUT/'cache/geometry'/e['path']).resolve())]=e
            source[str((HERE/'depth_prior.py').relative_to(ROOT))]=sha(HERE/'depth_prior.py')
    def guard(event,arg):
        if event!='open' or not isinstance(arg[0],(str,bytes,os.PathLike)):return
        path=Path(os.fsdecode(arg[0]))
        if path.suffix.lower() not in ['.png','.jpg','.jpeg','.npy','.npz']:return
        name=str(path.resolve());assert name in allowed,('Training read outside legal whitelist',name);reads[name]+=1
    sys.addaudithook(guard)
    for name,e in allowed.items():assert sha(name)==e['sha256'],name
    m=load_manifest(bound(p['manifest']));records,cameras=adapter.load_training(m)
    assert len(records)==1140;cameras=[resized_camera(c,1008,1344) for c in cameras]
    table=read(OUT/f'schedule_{a.repeat}.json');assert table['record_keys']==[[r['camera_id'],r['frame_index']] for r in records]
    model=load_model(a.resume or bound(p['parent']),m);g=model.g;ck=model.checkpoint
    assert model.branch=='ordinary_split' and capacity_summary(model)['active_gaussians']==132972
    resume_support.assert_state_equal(ck['model'][12],g.optimizer.state_dict())
    resume_support.assert_state_equal(ck['motion_refinement']['child_optimizer'],model.child_optimizer.state_dict())
    if a.resume:
        start=ck['metadata']['suffix_step'];assert ck['metadata']['method']==a.method and ck['metadata']['repeat']==str(a.repeat)
        assert ck['metadata']['protocol_sha256']==sha(OUT/'protocol.json')
        assert ck['samplers']==dict(kind='registered_fork',cursor=start,schedule_sha256=sha(OUT/f'schedule_{a.repeat}.json'))
        assert ck['confidence_geometry']['sources']==source
        parent_ck=torch.load(bound(p['parent']),map_location='cpu',weights_only=False)
        prefix=validate_parent(read(bound(p['old_schedule'])),parent_ck);del parent_ck
    else:
        assert ck['metadata']['method']=='U' and ck['metadata']['intervention_step']==6000
        prefix=validate_parent(read(bound(p['old_schedule'])),ck);start=0
    assert start<a.stop<=6000
    if a.method.startswith('P_'):assert a.stop==500 and str(a.repeat)=='1'
    names=shared.all_named(model);topology=shared.topology(model);initial=initial_identity(model)
    if a.method in ['P_xyz','P_SH']:
        chosen=['_xyz','children.offset_raw'] if a.method=='P_xyz' else ['_features_dc','_features_rest','children.sh_dc','children.sh_rest']
        for name,param in names.items():param.requires_grad_(name in chosen)
    idx=read(bound(p['teacher']));by={(e['camera'],e['frame']):e for e in idx['entries']}
    paths={i:Path(m['_root'])/by[r['camera_id'],r['frame_index']]['relative_path'] for i,r in enumerate(records)}
    cache=TeacherCache(paths,(3,1008,1344),16);curve=read(bound(p['lr_curve']))['rows']
    write(a.out/'config.json',dict(method=a.method,repeat=a.repeat,parent=p['parent'],resume=entry(a.resume) if a.resume else None,
      source_identity=source,protocol_sha256=sha(OUT/'protocol.json'),schedule_sha256=sha(OUT/f'schedule_{a.repeat}.json'),
      initial_identity=initial,calibration=cal,original_HR_training=False,training_files=list(allowed.values()),
      torch=str(torch.__version__),cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
      frozen_parameter_adam_state_does_not_advance=a.method in ['P_xyz','P_SH']))
    for rel in source:
        path=Path(os.environ.get('FOURDSR_UPSTREAM','/home/cai_tianshun/Project/4dgs'))/rel[9:] if rel.startswith('upstream/') else ROOT/rel
        dest=a.out/'sources'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dest)
    resume_support.restore_global_rng(ck['rng']);assert digest_state(ck['rng'])==digest_state(rng())
    write(a.out/'reload_audit.json',dict(status='passed',prefix=prefix,global_rng_exact=True,two_adam_exact=True,
      identity=initial,topology=topology,fork_cursor=start,new_suffix_is_native_continuation=False))
    started=time.monotonic();train_s=0.;torch.cuda.reset_peak_memory_stats();last=None
    with (a.out/'training.jsonl').open('w',buffering=1) as log:
        for cursor in range(start+1,a.stop+1):
            tick=time.monotonic();step=6000+cursor;g.update_learning_rate(7200+step)
            rates=curve[step-1];assert rates[0]==step
            for opt,rr in [(g.optimizer,rates[1]),(model.child_optimizer,rates[2])]:
                for group in opt.param_groups:
                    assert math.isclose(group['lr'],rr[group['name']],rel_tol=1e-14,abs_tol=0)
                    group['lr']=rr[group['name']]
            g.optimizer.zero_grad(set_to_none=True);model.child_optimizer.zero_grad(set_to_none=True)
            row=table['rows'][cursor-1];li=row[0];srslots=slots(a.method,row)
            packet=policy.render_model(model,cameras[li],trace=use_g);raw=packet['render'];lr=records[li]['image'].cuda()
            ll=(downsample(raw,lr.shape[-2:])-lr).abs().mean();reg=g.compute_regulation(model.h.time_smoothness_weight,model.h.l1_time_planes,model.h.plane_tv_weight)
            total=ll+reg;srvals=[];rgb=1;moments=0;lg=raw.sum()*0
            same= a.method not in ['Jperm','B2perm','P_joint','P_xyz','P_SH']
            if same:
                target=cache.get(li)
                if use_r:
                    weight=torch.ones_like(raw[:1]) if confidence is None else confidence.get(records[li]['camera_id'],records[li]['frame_index'],device=raw.device)
                    sl=operator.detail_loss(raw,target,weight,k_r=cal['k_R'])
                else:
                    weight=1 if confidence is None else confidence.get(records[li]['camera_id'],records[li]['frame_index'],device=raw.device)
                    sl=(weight*(raw-target).abs()).mean()
                total=total+.1*sl;srvals=[float(sl.detach())]
            if use_g:
                state=packet['effective_state'];mp=gm.render_moments(cameras[li],state['xyz'],state['cov'],state['opacity'],lr_size=(252,336));moments=1
                gt=geometry.get(records[li]['camera_id'],records[li]['frame_index'],device=raw.device)
                lg,gdiag=gm.geometry_loss(mp,gt);ramp=min(cursor/500,1.)
                total=total+cal['lambda_G']*ramp*lg
            assert torch.isfinite(total)
            total.backward()
            if not same:
                for index,coefficient in srslots:
                    pred=policy.render_model(model,cameras[index])['render'];rgb+=1
                    sl=(pred-cache.get(index)).abs().mean();assert torch.isfinite(sl)
                    (coefficient*sl).backward();srvals.append(float(sl.detach()))
            if cursor==start+1:
                write(a.out/'first_update_audit.json',dict(suffix_step=cursor,lr_observation=table['record_keys'][li],
                   sr_observations=[dict(key=table['record_keys'][i],coefficient=c) for i,c in srslots],
                   same_render_joint_backward=same,rgb_forwards=rgb,moment_forwards=moments,adam_per_round=2,
                   LR_full_path=True,G_only_effective_xyz=use_g,calibration=cal))
            check=cursor==start+1 or cursor in [100,500,3000,6000]
            if check:assert all(torch.isfinite(x.grad).all() for x in names.values() if x.grad is not None)
            # Write attempt BEFORE the first Adam; partial failures remain countable.
            write(a.out/'last_attempt.json',dict(suffix_step=cursor,rgb=rgb,moments=moments,status='before_adam'))
            g.optimizer.step();model.child_optimizer.step();torch.cuda.synchronize();train_s+=time.monotonic()-tick
            if check:assert shared.topology(model)==topology and all(torch.isfinite(x).all() for x in names.values())
            rec=dict(suffix_step=cursor,intervention_step=step,lr_l1=float(ll.detach()),reg=float(reg.detach()),
              sr_l1=srvals,g_loss=float(lg.detach()),rgb_forwards=rgb,moment_forwards=moments,adam_calls=2,
              train_s=train_s,wall_s=time.monotonic()-started,peak_gb=torch.cuda.max_memory_allocated()/1e9)
            log.write(json.dumps(rec)+'\n')
            if check or cursor%100==0:print(json.dumps(rec),flush=True)
            if cursor in [500,3000,6000] or cursor==a.stop or (a.method.startswith('P_') and cursor==100):
                last,meta=checkpoint(model,a,p,table,cursor,source,start,train_s,started)
    for rel,identity_hash in source.items():
        path=Path(os.environ.get('FOURDSR_UPSTREAM','/home/cai_tianshun/Project/4dgs'))/rel[9:] if rel.startswith('upstream/') else ROOT/rel
        assert sha(path)==identity_hash,('Training source changed',rel)
    if cal:assert sha(OUT/'calibration.json')==calibration_sha,'Frozen coefficient file changed'
    write(a.out/'image_reads.json',dict(actual_reads=dict(reads),all_legal=True,HR_derived_training=False))
    weighted_exposure={}
    for row in table['rows'][start:a.stop]:
        for index,coefficient in slots(a.method,row):
            key=str(table['record_keys'][index]);weighted_exposure[key]=weighted_exposure.get(key,0)+coefficient
    write(a.out/'exposure.json',dict(schedule_sha256=sha(OUT/f'schedule_{a.repeat}.json'),start=start,stop=a.stop,
      actual_LR=Counter(str(table['record_keys'][r[0]]) for r in table['rows'][start:a.stop]),
      coefficient_weighted_SR=weighted_exposure,audit=table['audit'],definition=p['arms'][a.method]))
    write(a.out/'complete.json',dict(status='completed_training',method=a.method,repeat=a.repeat,updates=a.stop-start,
      suffix_endpoint=a.stop,training_rgb_forwards=(3 if a.method=='B2perm' else 2 if a.method=='Jperm' else 1)*(a.stop-start),
      moment_forwards=int(use_g)*(a.stop-start),adam_calls=2*(a.stop-start),train_s=train_s,wall_s=time.monotonic()-started,
      peak_gb=torch.cuda.max_memory_allocated()/1e9,checkpoint=entry(last),gpu=torch.cuda.get_device_name(),
      physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),topology_unchanged=True,source_identity=source))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--method',required=True);ap.add_argument('--repeat',default='1');ap.add_argument('--out',type=Path,required=True);ap.add_argument('--stop',type=int,default=6000);ap.add_argument('--resume',type=Path);a=ap.parse_args()
    try:main(a)
    except BaseException:
        write(a.out/'failed.json',dict(status='failed',traceback=traceback.format_exc()));raise
