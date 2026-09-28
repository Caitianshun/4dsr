"""Complete U6000 continuation, two Adams, registered sampling and SR microbatch accumulation."""
import argparse
import os
import sys
import time
import random
import math
import shutil
import traceback
from collections import Counter
from dv_common import *
shared=setup()
import numpy as np
import torch
import importlib.util
_router_spec=importlib.util.spec_from_file_location('inherited_sr_router',ROOT/'experiments/dynamic_sr_attribute_routing_20260928/sr_attribute_router.py')
_router=importlib.util.module_from_spec(_router_spec);_router_spec.loader.exec_module(_router)
render_model,SRAttributeRouter=_router.render_model,_router.SRAttributeRouter
from sr_view_batch import SRViewBatch, ARM_TO_MODE
from runtime_identity import identity as runtime_identity
from readiness import validate
from schedule import Schedule,plain
from training_support import TeacherCache,initial_identity,digest_state
from motion_model import load_model,refinement_state,capacity_summary
from common import downsample,resized_camera
from n3dv_data import load_manifest
# common inserted this original data adapter path before local files.
import importlib.util
_spec=importlib.util.spec_from_file_location('dv_original_data_adapter',ROOT/'experiments/dynamic_sr_20260918/run_experiment.py')
_adapter=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_adapter)
load_training=_adapter.load_training
sys.path.insert(0,str(ROOT/'experiments/dynamic_sr_20260920'))
from resume_control import restore_global_rng,assert_state_equal

def rng():return dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all(),numpy=np.random.get_state(),python=random.getstate())
def args():
    a=argparse.ArgumentParser();a.add_argument('--protocol',type=Path,default=OUT/'protocol.json');a.add_argument('--method',choices=['Async2','Sync2'],required=True);a.add_argument('--repeat',choices=['1','2'],required=True);a.add_argument('--resume',type=Path);a.add_argument('--out',type=Path,required=True);a.add_argument('--stop',type=int,default=12000);a.add_argument('--run-root',type=Path,default=OUT);return a.parse_args()

def run(a):
    p=require_run_root(a.run_root);assert a.protocol.resolve()==(OUT/'protocol.json').resolve();assert a.out.resolve().is_relative_to(OUT/'runs');a.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4)
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==p['training']['physical_gpu']
    source=sources();parent=bound(p['parent']);resume=a.resume or parent
    runtime=runtime_identity();assert runtime==p['runtime']['training']
    validate(p,source);assert a.stop==p['training']['stop']
    task_id=f'r{a.repeat}_{a.method}'
    assert task_id in {t['task_id'] for t in p['task_plan']}
    allowed={str(local(x['path']).resolve()):x for x in p['training_files']};reads=Counter()
    # Explicit complete image whitelist, activated before data loading.
    def hook(event,arg):
        if event!='open' or not isinstance(arg[0],(str,bytes,os.PathLike)):return
        path=Path(os.fsdecode(arg[0]))
        if path.suffix.lower() not in ['.png','.jpg','.jpeg','.npy','.npz']:return
        name=str(path.resolve());assert name in allowed,('Unregistered training image/data read',name)
        reads[name]+=1
    sys.addaudithook(hook)
    for path,item in allowed.items():assert sha(path)==item['sha256'],path
    m=load_manifest(bound(p['manifest']));records,cameras=load_training(m)
    assert len(records)==1140;w,h=m['resolutions']['hr'];cameras=[resized_camera(c,h,w) for c in cameras]
    schedulepath=bound(p['schedules'][a.repeat]);schedule=read(schedulepath);old=read(bound(p['old_schedule']))
    assert old['record_keys']==[[r['camera_id'],r['frame_index']] for r in records]
    model=load_model(resume,m);g=model.g;ck=model.checkpoint;start=ck['metadata']['intervention_step']
    assert 6000<=start<a.stop and model.branch=='ordinary_split'
    assert ck['metadata']['manifest_sha']==p['manifest']['sha256']
    assert_state_equal(ck['model'][12],g.optimizer.state_dict());assert_state_equal(ck['motion_refinement']['child_optimizer'],model.child_optimizer.state_dict())
    identity=initial_identity(model);assert identity['model_and_optimizer_sha256']==digest_state((ck['model'],ck['motion_refinement']['children'],ck['motion_refinement']['child_optimizer']))
    sampler=Schedule(old,schedule,schedule['mode'])
    for _ in range(6000):sampler.advance()
    if start==6000:
        assert sha(resume)==p['parent']['sha256'] and ck['metadata']['method']=='U';sampler.validate_parent(ck)
    else:
        # Always validate the native parent prefix, even for independent suffix resume.
        parent_ck=torch.load(parent,map_location='cpu',weights_only=False);sampler.validate_parent(parent_ck);del parent_ck
        for _ in range(6000,start):sampler.advance()
        assert ck['metadata']['method']==a.method and ck['metadata']['repeat']==a.repeat
        assert ck['metadata']['protocol_sha256']==sha(a.protocol) and ck['metadata']['schedule_sha256']==sha(schedulepath)
        assert ck['metadata']['schedule_mode']==schedule['mode'];assert plain(ck['samplers'])==plain(sampler.state())
        assert ck['sr_view_batch']['sources']==source and ck['metadata']['task_id']==task_id
    assert capacity_summary(model)['active_gaussians']==p['training']['points']
    pairpath=bound(p['pair_schedules'][a.repeat]);pairtable=read(pairpath)
    batch=SRViewBatch(ARM_TO_MODE[a.method],pairtable,sha(pairpath),cursor=start)
    if start>6000:batch.validate_resume(ck['sr_view_batch'])
    router=SRAttributeRouter('joint',model);policy=router.describe();topology=shared.topology(model);named=shared.all_named(model)
    idx=read(bound(p['teacher']));by={(e['camera'],e['frame']):e for e in idx['entries']}
    paths={i:Path(m['_root'])/by[r['camera_id'],r['frame_index']]['relative_path'] for i,r in enumerate(records)}
    cache=TeacherCache(paths,(3,h,w),64);curve=read(bound(p['lr_curve']))['rows'];assert len(curve)==40000
    (a.out/'sources').mkdir();snapshots=[]
    for rel,hash0 in source.items():
        path=(Path(os.environ['FOURDSR_UPSTREAM'])/rel.removeprefix('upstream/')) if rel.startswith('upstream/') else ROOT/rel
        dest=a.out/'sources'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dest);snapshots.append(dict(path=rel,sha256=hash0))
    write(a.out/'config.json',dict(method=a.method,repeat=a.repeat,protocol_sha256=sha(a.protocol),sources=source,manifest_sha256=p['manifest']['sha256'],parent=p['parent'],resume_sha256=sha(resume),segment_start=start,stop=a.stop,schedule_sha256=sha(schedulepath),schedule_mode=schedule['mode'],policy=policy,sr_view_batch=batch.state(),target_inputs=idx['entries'],training_input_whitelist=list(allowed.values()),gpu=torch.cuda.get_device_name(),physical_gpu=os.environ['CUDA_VISIBLE_DEVICES'],torch=str(torch.__version__),python=sys.version,runtime_flags=dict(cuda=torch.version.cuda,matmul_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_tf32=torch.backends.cudnn.allow_tf32,cudnn_benchmark=torch.backends.cudnn.benchmark,deterministic_algorithms=torch.are_deterministic_algorithms_enabled()),initial_identity=identity,original_HR_training=False))
    restore_global_rng(ck['rng']);assert digest_state(ck['rng'])==digest_state(rng())
    write(a.out/'reload_audit.json',dict(status='passed',identity=identity,runtime=runtime,rng_sha256=digest_state(rng()),sampler_sha256=digest_state(sampler.state()),prefix_verified=True,global_rng_exact=True,two_adam_exact=True,topology=topology))
    kind='formal';label=str(a.out.relative_to(OUT));started=time.monotonic();train_s=0;torch.cuda.reset_peak_memory_stats()
    checks=[]
    with (a.out/'training.jsonl').open('w',buffering=1) as log:
        for step in range(start+1,a.stop+1):
            tick=time.monotonic();g.update_learning_rate(p['training']['lr_offset']+step)
            expected=curve[step-1];assert expected[0]==step
            for opt,rates in [(g.optimizer,expected[1]),(model.child_optimizer,expected[2])]:
                assert {x['name'] for x in opt.param_groups}==set(rates)
                for group in opt.param_groups:
                    # libm/NumPy host roundoff was ~1e-20; execute the exact registered curve.
                    assert math.isclose(group['lr'],rates[group['name']],rel_tol=1e-14,abs_tol=0.),(step,group['name'],group['lr'],rates[group['name']])
                    group['lr']=rates[group['name']]
            li,si=sampler.advance();g.optimizer.zero_grad(set_to_none=True);model.child_optimizer.zero_grad(set_to_none=True)
            debit(kind,label,rgb=1);raw=render_model(model,cameras[li])['render'];lr=records[li]['image'].cuda()
            loss_lr=(downsample(raw,lr.shape[-2:])-lr).abs().mean();reg=g.compute_regulation(model.h.time_smoothness_weight,model.h.l1_time_planes,model.h.plane_tv_weight)
            (loss_lr+reg).backward()
            first=step==start+1
            slots=batch.observations(step,li,si);sr_terms=[];weighted_terms=[]
            # Each render has its own graph, all use the same parameter state.
            # No optimizer/zero_grad call occurs between the two backwards.
            for index,coefficient in slots:
                debit(kind,label,rgb=1)
                prediction=render_model(model,cameras[index])['render']
                sr_unweighted=(prediction-cache.get(index)).abs().mean()
                sr=coefficient*sr_unweighted
                assert torch.isfinite(loss_lr+reg+sr)
                router.backward_sr(sr)
                sr_terms.append(float(sr_unweighted.detach()));weighted_terms.append(float(sr.detach()))
                del prediction,sr,sr_unweighted
            if first:
                write(a.out/'first_update_audit.json',dict(step=step,slots=slots,
                    observations=[dict(camera=records[i]['camera_id'],frame=records[i]['frame_index'],
                                       time=float(cameras[i].time),coefficient=w) for i,w in slots],
                    policy=policy,sr_view_batch=batch.state(),LR_backward_calls=1,regularizer_calls=1,
                    SR_backward_calls=len(slots),optimizer_calls_before_both_SR=0,
                    SR_position_detach=False,first_step_in_formal_budget=True))
            check=first or step in p['training']['save'] or step==a.stop
            if check:assert all(bool(torch.isfinite(x.grad).all()) for x in named.values() if x.grad is not None)
            # Debit a round before its first Adam. A partial failed round still consumes budget.
            debit(kind,label,updates=1);g.optimizer.step();debit(kind,label,adam=1);model.child_optimizer.step();debit(kind,label,adam=1,completed=1)
            torch.cuda.synchronize();train_s+=time.monotonic()-tick
            if check:
                assert shared.topology(model)==topology;assert all(bool(torch.isfinite(x).all()) for x in named.values())
                checks.append(dict(step=step,finite=True,topology_unchanged=True,lr_full_path=True,position_detach=False));write(a.out/'routing_audit.json',dict(policy=policy,checks=checks))
            if True:
                row=dict(step=step,lr_l1=float(loss_lr),reg=float(reg),sr_a_l1=sr_terms[0],sr_b_l1=sr_terms[1],sr_weighted_total=sum(weighted_terms),total_loss=float(loss_lr+reg)+sum(weighted_terms),rgb_forwards=3,adam_calls=2,train_s=train_s,wall_s=time.monotonic()-started,peak_gb=torch.cuda.max_memory_allocated()/1e9,draw_sha256=sampler.draw.hexdigest());log.write(json.dumps(row)+'\n')
                if check or step%100==0:print(json.dumps(row),flush=True)
            if step in p['training']['save'] or step==a.stop:
                metadata=dict(scene=m['scene'],stage='sync_multiview',task_id=task_id,run_id=RUN_ID,branch='ordinary_split',method=a.method,repeat=a.repeat,step=p['training']['lr_offset']+step,intervention_step=step,manifest_sha=p['manifest']['sha256'],protocol_sha256=sha(a.protocol),schedule_sha256=sha(schedulepath),schedule_mode=schedule['mode'],pair_sha256=sha(pairpath),sr_mode=batch.mode,sr_slots=len(slots),extent=ck['metadata']['extent'],draw_sha256=sampler.draw.hexdigest(),lr_draw_sha256=sampler.lr_draw.hexdigest(),sr_frame_sha256=sampler.sr_times.hexdigest(),segment_start=start,segment_updates=step-start,train_s=train_s,elapsed_s=time.monotonic()-started,points=p['training']['points'])
                payload=dict(model=g.capture(),hidden=vars(model.h),optim=vars(model.o),metadata=metadata,rng=rng(),samplers=sampler.state(),motion_refinement=refinement_state(model),sr_view_batch=dict(**batch.state(),policy=policy,topology=topology,sources=source))
                path=a.out/f'checkpoint_{step}.pt';temp=path.with_suffix('.tmp')
                with temp.open('wb') as f:torch.save(payload,f);f.flush();os.fsync(f.fileno())
                temp.replace(path);write(path.with_suffix('.json'),dict(path=str(path.relative_to(ROOT)),sha256=sha(path),metadata=metadata));commit_progress(task_id,step)
        log.flush();os.fsync(log.fileno())
    assert sources()==source
    write(a.out/'image_reads.json',dict(whitelist_sha256=digest_state(sorted(allowed)),actual_reads=dict(reads),all_allowed=all(k in allowed for k in reads)))
    write(a.out/'exposure.json',dict(schedule_sha256=sha(schedulepath),sampler=plain(sampler.state()),segment=schedule['exposure'],sr_view_batch=batch.state(),pair_audit=pairtable['audit']))
    write(a.out/'complete.json',dict(status='completed',actual_updates=a.stop-start,total_updates=a.stop,adam_calls=2*(a.stop-start),training_rgb_forwards=3*(a.stop-start),train_s=train_s,wall_s=time.monotonic()-started,peak_gb=torch.cuda.max_memory_allocated()/1e9,metadata=metadata,source_unchanged=True,engineering=False,task_id=task_id,run_id=RUN_ID))

if __name__=='__main__':
    a=args()
    try:run(a)
    except BaseException:
        write(a.out/'failed.json',dict(status='failed',traceback=traceback.format_exc()));raise
