"""Paired U6000 continuation: unchanged joint vs exact appearance whitelist.

Preserves original loss, sampler suffix, schedule, topology and both Adam states.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time
import traceback
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'experiments/dynamic_sr_detail_supervision_20260924'))
from training_support import initial_identity,TeacherCache,digest_state
from motion_model import load_model,render_model,refinement_state,capacity_summary
from common import UPSTREAM,downsample,resized_camera,named_parameters,sha256,write_json
from n3dv_data import load_manifest
from run_experiment import load_training
sys.path.insert(0,str(ROOT/'experiments/dynamic_sr_20260920'))
from resume_control import restore_global_rng,assert_state_equal
from shared import all_named, gamma_digest, topology
from freeze_policy import apply_policy, frozen_state, assert_no_frozen_grad, color_digest


def read(p):return json.loads(Path(p).read_text())
def rng():return dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all(),numpy=np.random.get_state(),python=random.getstate())
def norm(params):return sum(float(p.grad.double().square().sum()) for p in params if p.grad is not None)**.5


def run(a):
    out=a.out;out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4)
    reads=Counter();allowed=set()
    def io_audit(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        path=Path(os.fsdecode(args[0]))
        if path.suffix.lower()!='.png':return
        s=str(path.resolve())
        if '/hr/' in s or '/sr_swinir_x4/' in s:
            assert s in allowed,f'Illegal training image read for {a.method}: {s}'
            reads[s]+=1
    sys.addaudithook(io_audit)
    m=load_manifest(a.manifest);records,cameras=load_training(m);assert len(records)==1140
    w,h=m['resolutions']['hr'];cameras=[resized_camera(c,h,w) for c in cameras]
    schedule=read(a.schedule);assert schedule['steps']==40000 and schedule['manifest_sha256']==sha256(a.manifest)
    assert schedule['record_keys']==[[r['camera_id'],r['frame_index']] for r in records]
    model=load_model(a.resume,m);g=model.g;ck=model.checkpoint;start=ck['metadata']['intervention_step']
    assert start in ([6000,6002] if a.smoke else [6000,18000]) and start<a.stop
    protocol=read(a.protocol)
    assert sha256(a.manifest)==protocol['manifest']['sha256']
    assert sha256(a.schedule)==protocol['schedule']['sha256']
    assert sha256(a.lr_curve)==protocol['lr_curve']['sha256']
    if not a.smoke: assert read(a.protocol.parent/'p0_decision.json')['allow_p1'] is True
    assert ck['metadata']['manifest_sha']==sha256(a.manifest)
    identity=initial_identity(model)
    assert_state_equal(ck['model'][12],g.optimizer.state_dict())
    assert_state_equal(ck['motion_refinement']['child_optimizer'],model.child_optimizer.state_dict())
    if start==6000:
        assert ck['metadata']['method']=='U' and sha256(a.resume)==protocol['start_sha256']
    else:
        assert ck['metadata']['method']==a.method and ck['metadata']['schedule_sha256']==sha256(a.schedule)
        assert ck['view_recovery']['policy']['arm']==a.method
    assert identity['model_and_optimizer_sha256']==digest_state((ck['model'],ck['motion_refinement']['children'],ck['motion_refinement']['child_optimizer']))
    policy=apply_policy(model,a.method)
    frozen_reference=frozen_state(model);gamma_reference=gamma_digest(model)
    if start>6000 and a.method=='F_app':
        assert frozen_reference==ck['view_recovery']['frozen_reference']
        assert gamma_reference==ck['view_recovery']['gamma_reference']
    initial_color=color_digest(model)
    paths={};inputs=[]
    if True:
        idx=read(a.target_index);assert idx['manifest_sha256']==sha256(a.manifest)
        assert not idx.get('privileged_train_hr',False)
        assert sha256(a.target_index)==protocol['teacher']['sha256']
        by={(e['camera'],e['frame']):e for e in idx['entries']};assert set(by)=={(r['camera_id'],r['frame_index']) for r in records}
        for i,r in enumerate(records):
            e=by[r['camera_id'],r['frame_index']];path=(Path(m['_root'])/e['relative_path']).resolve();allowed.add(str(path))
            assert sha256(path)==e['sha256']
            paths[i]=path;inputs.append(e)
    cache=TeacherCache(paths,(3,h,w),64) if paths else None
    old_ids=[i for i,r in enumerate(records) if r['camera_id'] in schedule['original_teacher_cameras']]
    lr_rng=random.Random(20260923+177);sr_rng=random.Random(20260923+211)
    draw=hashlib.sha256();lr_draw=hashlib.sha256();sr_times=hashlib.sha256();exposure=Counter();lr_exposure=Counter()
    def consume(step):
        li,si,old=schedule['rows'][step-1]
        assert lr_rng.randrange(len(records))==li and sr_rng.choice(old_ids)==old
        assert records[si]['frame_index']==records[old]['frame_index']
        draw.update(f'{li},{si}\n'.encode());lr_draw.update(f'{li}\n'.encode());sr_times.update(f"{records[si]['frame_index']}\n".encode())
        exposure[f"{records[si]['camera_id']}/{records[si]['frame_index']}"]+=1
        lr_exposure[f"{records[li]['camera_id']}/{records[li]['frame_index']}"]+=1
        return li,si
    for step in range(1,start+1):consume(step)
    assert lr_rng.getstate()==ck['samplers']['lr'] and sr_rng.getstate()==ck['samplers']['sr']
    assert dict(exposure)==ck['samplers']['additional_exposure']
    for key,hasher in [('draw_sha256',draw),('lr_draw_sha256',lr_draw),('sr_frame_sha256',sr_times)]: assert ck['metadata'][key]==hasher.hexdigest()
    curve=read(a.lr_curve)['rows'];assert len(curve)==40000
    sources=[HERE/'train.py',HERE/'freeze_policy.py',HERE/'shared.py',ROOT/'experiments/dynamic_sr_motion_bound_20260923/motion_model.py',
        ROOT/'experiments/dynamic_sr_detail_supervision_20260924/training_support.py',ROOT/'experiments/dynamic_sr_detail_supervision_20260924/detail_loss.py',
        *[ROOT/'experiments/dynamic_sr_20260918'/p for p in ['common.py','n3dv_data.py','run_experiment.py']],
        ROOT/'experiments/dynamic_sr_20260920/resume_control.py',
        *[UPSTREAM/p for p in ['scene/gaussian_model.py','scene/deformation.py','scene/hexplane.py','gaussian_renderer/__init__.py']]]
    (out/'sources').mkdir();source_rows=[]
    for i,path in enumerate(sources):
        dest=out/'sources'/f'{i}_{path.name}';shutil.copyfile(path,dest);source_rows.append(dict(path=str(path),sha256=sha256(path),snapshot=str(dest)))
    config=dict(method=a.method,privileged_train_hr=a.method=='O',manifest_sha256=sha256(a.manifest),schedule_sha256=sha256(a.schedule),
        common_start_identity={'checkpoint_sha256':protocol['start_sha256']},restored_identity=identity,segment_start=start,segment_stop=a.stop,
        target_inputs=inputs,target_index_sha256=sha256(a.target_index) if a.target_index else None,
        sources=source_rows,gpu=torch.cuda.get_device_name(),visible_cuda=os.environ.get('CUDA_VISIBLE_DEVICES'),torch=str(torch.__version__),
        sr_weight=.1,renderer='unchanged ordinary_split Wu',topology='fixed 132972',
        second_backward='0.1 * mean(abs(prediction - target))',policy=policy,protocol_sha256=sha256(a.protocol),resume_sha256=sha256(a.resume),topology_identity=topology(model),
        information_boundary='Only original legal train LR and frozen SwinIR. No HR, heldout images, masks or flow in training.')
    write_json(out/'config.json',config)
    restore_global_rng(ck['rng']);write_json(out/'reload_audit.json',dict(status='passed',intervention_step=start,identity=identity,rng_digest=digest_state(rng())))
    named=list(named_parameters(g).items())+[('children.'+n,p) for n,p in model.children.named_parameters()]
    write_json(out/'freeze_audit.json',dict(policy=policy,frozen_reference=frozen_reference,gamma_reference=gamma_reference,checks=[]))
    params=[p for n,p in named];assert capacity_summary(model)['active_gaussians']==132972
    initial_degree=g.active_sh_degree;degree=g.active_sh_degree;started=time.monotonic();train_s=0.;torch.cuda.reset_peak_memory_stats()
    metadata=None
    with (out/'training.jsonl').open('w',buffering=1) as log:
        for step in range(start+1,a.stop+1):
            tick=time.monotonic();g.update_learning_rate(7200+step)
            expected=curve[step-1];assert expected[0]==step
            assert {q['name']:q['lr'] for q in g.optimizer.param_groups}==expected[1]
            assert {q['name']:q['lr'] for q in model.child_optimizer.param_groups}==expected[2]
            li,si=consume(step);g.optimizer.zero_grad(set_to_none=True);model.child_optimizer.zero_grad(set_to_none=True)
            target_lr=records[li]['image'].cuda();rendered_lr=render_model(model,cameras[li])['render']
            lr_loss=(downsample(rendered_lr,target_lr.shape[-2:])-target_lr).abs().mean()
            reg=g.compute_regulation(model.h.time_smoothness_weight,model.h.l1_time_planes,model.h.plane_tv_weight)
            (lr_loss+reg).backward()
            first=step==start+1;before=[None if p.grad is None else p.grad.detach().clone() for p in params] if first else None
            target=cache.get(si) if cache else None
            prediction=render_model(model,cameras[si])['render']
            second=.1*(prediction-target).abs().mean()
            assert bool(torch.isfinite(lr_loss+reg+second));second.backward()
            assert_no_frozen_grad(model)
            check=first or step in [18000,40000] or step==a.stop
            if check:assert all(bool(torch.isfinite(p.grad).all()) for p in params if p.grad is not None)
            if first:
                delta=[float((p.grad-b).abs().max()) if b is not None and p.grad is not None else None for p,b in zip(params,before)]
                if a.method=='Z':assert all(v is None or v==0 for v in delta)
                write_json(out/'first_update_audit.json',dict(step=step,method=a.method,privileged_train_hr=a.method=='O',
                    parameter_names=[n for n,p in named],
                    second_loss=float(second),before_none=[b is None for b in before],after_none=[p.grad is None for p in params],second_max_gradient_delta=delta,finite=True))
            g.optimizer.step();model.child_optimizer.step();torch.cuda.synchronize();train_s+=time.monotonic()-tick
            if check:
                frozen_now=frozen_state(model);gamma_now=gamma_digest(model)
                if a.method=='F_app':
                    assert frozen_now==frozen_reference
                    assert gamma_now==gamma_reference
                changed=color_digest(model)!=initial_color
                assert changed and any(p.requires_grad and p.grad is not None and bool(p.grad.abs().max()>0) for p in params)
                audit=read(out/'freeze_audit.json');audit['checks'].append(dict(step=step,frozen_state=frozen_now,gamma=gamma_now,color_changed=changed,all_frozen_grad_none=True));write_json(out/'freeze_audit.json',audit)
            if check or step%100==0:
                row=dict(step=step,lr_l1=float(lr_loss),reg=float(reg),second_l1=None if target is None else float(second/.1),second_weighted=float(second),
                    train_s=train_s,wall_s=time.monotonic()-started,peak_gb=torch.cuda.max_memory_allocated()/1e9,draw_sha256=draw.hexdigest(),
                    points=capacity_summary(model)['active_gaussians']);log.write(json.dumps(row)+'\n');print(json.dumps(row),flush=True)
            if step in [6000,18000,40000] or step==a.stop:
                assert capacity_summary(model)['active_gaussians']==132972 and g.active_sh_degree==initial_degree
                assert all(bool(torch.isfinite(p).all()) for p in params)
                metadata=dict(scene=m['scene'],stage='view_recovery',branch='ordinary_split',method=a.method,privileged_train_hr=a.method=='O',
                    step=7200+step,intervention_step=step,manifest_sha=sha256(a.manifest),schedule_sha256=sha256(a.schedule),
                    extent=ck['metadata']['extent'],initial_identity=identity,
                    draw_sha256=draw.hexdigest(),lr_draw_sha256=lr_draw.hexdigest(),sr_frame_sha256=sr_times.hexdigest(),
                    elapsed_s=time.monotonic()-started,train_s=train_s,segment_start=start,segment_updates=step-start,points=132972)
                payload=dict(model=g.capture(),hidden=vars(model.h),optim=vars(model.o),metadata=metadata,rng=rng(),
                    samplers=dict(lr=lr_rng.getstate(),sr=sr_rng.getstate(),source_prefix_exposure={},additional_exposure=dict(exposure)),motion_refinement=refinement_state(model),view_recovery=dict(policy=policy,frozen_reference=frozen_reference,gamma_reference=gamma_reference))
                path=out/f'checkpoint_{step}.pt';tmp=path.with_suffix('.tmp')
                with tmp.open('wb') as f:torch.save(payload,f);f.flush();os.fsync(f.fileno())
                tmp.replace(path);write_json(out/f'checkpoint_{step}.json',dict(path=str(path.resolve()),sha256=sha256(path),metadata=metadata))
                if a.smoke:
                    before_id=initial_identity(model);reloaded=load_model(path,m);assert initial_identity(reloaded)==before_id
                    assert apply_policy(reloaded,a.method)==policy
                    assert frozen_state(reloaded)==frozen_state(model)
                    assert gamma_digest(reloaded)==gamma_now
                    write_json(out/'short_reload_audit.json',dict(status='passed',intervention_step=step,identity=before_id))
        log.flush();os.fsync(log.fileno())
    if a.stop==40000:
        for key,v in [('draw_sha256',draw),('lr_draw_sha256',lr_draw),('sr_frame_sha256',sr_times)]:assert v.hexdigest()==schedule[key]
    assert all(sha256(v['path'])==v['sha256'] for v in source_rows)
    write_json(out/'image_reads.json',dict(method=a.method,privileged_train_hr=a.method=='O',actual_high_resolution_opens=dict(reads)))
    write_json(out/'exposure.json',dict(lr=dict(lr_exposure),second_camera=dict(exposure),supervised_second_samples=a.stop,actual_segment_supervised_second_samples=a.stop-start,
        render_calls=2*a.stop,actual_segment_render_calls=2*(a.stop-start)))
    write_json(out/'complete.json',dict(status='completed',metadata=metadata,actual_updates=a.stop-start,total_updates=a.stop,
        elapsed_s=time.monotonic()-started,train_s=train_s,peak_gb=torch.cuda.max_memory_allocated()/1e9,
        cache_misses=cache.misses if cache else 0,smoke=a.smoke,source_unchanged=True,privileged_train_hr=a.method=='O'))


def main():
    p=argparse.ArgumentParser()
    for k in ['manifest','resume','schedule','protocol','lr-curve','out']:p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--target-index',type=Path,required=True);p.add_argument('--method',choices=['C_joint','F_app'],required=True);p.add_argument('--stop',type=int,required=True);p.add_argument('--smoke',action='store_true')
    a=p.parse_args();assert a.stop in ([6002,6004] if a.smoke else [18000,40000])
    try:run(a)
    except BaseException:
        a.out.mkdir(parents=True,exist_ok=True);write_json(a.out/'failed.json',dict(status='failed',traceback=traceback.format_exc()));raise


if __name__=='__main__':main()
