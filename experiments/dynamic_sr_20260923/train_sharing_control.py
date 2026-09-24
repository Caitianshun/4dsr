"""Bounded fixed-topology A/B/C/D forks; only train LR and frozen SR targets.

Every fork restores the same Adam and global RNG and starts explicitly new,
identical independent camera samplers. This is not a legacy sampler continuation.
"""
from __future__ import annotations
import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
OLD=ROOT/'experiments/dynamic_sr_20260918'
sys.path.insert(0,str(OLD))
sys.path.insert(0,str(ROOT/'experiments/dynamic_sr_20260919'))
sys.path.insert(0,str(ROOT/'experiments/dynamic_sr_20260920'))
import numpy as np
import torch
from common import (UPSTREAM,downsample,image_tensor,load_checkpoint,named_parameters,render_image,resized_camera,sha256)
from n3dv_data import load_manifest
from run_experiment import load_training
from controlled_fit import write_json,persist
from resume_control import assert_state_equal,restore_global_rng,checkpoint_with_samplers


def cached_target(cache,cam,frame,method):
    # Constructor has one auditable manifest; layout resolved by its records.
    records=cache['targets']
    r=next(r for r in records if r['camera']==cam and r['frame']==frame)
    entry=r['files'][method]
    path=Path(cache['_root'])/entry['path']
    assert sha256(path)==entry['sha256'],path
    a=np.load(path,allow_pickle=False)
    assert a.dtype==np.float32 and tuple(a.shape)==cache['_shape'] and np.isfinite(a).all()
    return torch.from_numpy(np.ascontiguousarray(a.transpose(2,0,1)))


def setup(args):
    m=load_manifest(args.manifest)
    g,h,o,ck=load_checkpoint(args.checkpoint)
    assert_state_equal(ck['model'][-2],g.optimizer.state_dict())
    assert ck['metadata']['manifest_sha']==sha256(args.manifest)
    records,cameras=load_training(m)
    w,hh=m['resolutions']['hr'];cams=[resized_camera(c,hh,w) for c in cameras]
    prior_cams=set(args.prior_cameras.split(','));ids=[i for i,r in enumerate(records) if r['camera_id'] in prior_cams]
    assert len(ids)==240
    observations={(v['camera_id'],v['frame_index']):v for v in m['observations']}
    paths={i:Path(m['_root'])/'sr_swinir_x4'/records[i]['camera_id']/Path(observations[(records[i]['camera_id'],records[i]['frame_index'])]['lr_path']).name for i in ids}
    cache=json.loads(Path(args.cache_manifest).read_text());cache['_root']=str(Path(args.cache_manifest).parent)
    cache['_shape']=(hh,w,3)
    cache_root=Path(cache['_root'])
    verification=json.loads((cache_root/'verification.json').read_text())
    completion=json.loads((cache_root/'complete.json').read_text())
    assert completion['status']=='prepared_full_scene_cache' and completion['manifest_sha256']==sha256(args.cache_manifest)
    assert verification['status']=='passed' and verification['manifest_sha256']==sha256(args.cache_manifest)
    assert cache['protocol_sha256']==sha256(cache_root/'protocol.json')
    assert cache['input_hashes_sha256']==sha256(cache_root/'input_hashes.json')
    for path,entry in json.loads((cache_root/'input_hashes.json').read_text()).items():
        assert sha256(path)==entry['sha256'],path
    assert cache['schema']=='dynamic_sr_training_targets_v1' and cache['scene']==m['scene']
    assert len(cache['targets'])==len(ids)
    assert {(r['camera'],r['frame']) for r in cache['targets']}=={(records[i]['camera_id'],records[i]['frame_index']) for i in ids}
    memo={}
    def target(i,method):
        key=(i,method)
        if key not in memo:
            memo[key]=image_tensor(paths[i],device='cpu') if method in ['A','B'] else cached_target(cache,records[i]['camera_id'],records[i]['frame_index'],method)
        return memo[key]
    restore_global_rng(ck['rng'])
    return m,g,h,o,ck,records,cams,ids,target


def norms(g):
    return {group['name']:float(sum((p.grad.detach().double().square().sum() for p in group['params'] if p.grad is not None),start=torch.zeros((),device='cuda')).sqrt()) for group in g.optimizer.param_groups}


def calibration(args):
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(__file__,out/'source.py')
    m,g,h,o,ck,records,cams,ids,target=setup(args)
    rows=[]
    selected=[i for i in ids if records[i]['frame_index'] in [32,80]]
    assert len(selected)==8
    for i in selected:
        row=dict(camera_id=records[i]['camera_id'],frame_index=records[i]['frame_index'],norms={})
        for method in ['A','D']:
            g.optimizer.zero_grad(set_to_none=True)
            loss=(render_image(g,cams[i])['render']-target(i,method).cuda()).abs().mean()
            loss.backward();row['norms'][method]=norms(g)
        rows.append(row)
    total=lambda method:float(np.sqrt(np.mean([sum(v*v for v in r['norms'][method].values()) for r in rows])))
    ratio=total('D')/max(total('A'),1e-20)
    assert np.isfinite(ratio) and ratio>0
    # Prove calibration does not alter parameters, Adam, or stored model buffers.
    # Fresh CPU load is independent: g.restore may alias the CUDA ck tensors.
    independent=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    assert_state_equal(independent['model'],g.capture(),label='calibration_no_update_independent_cpu')
    write_json(out/'calibration.json',dict(rows=rows,weight_A=.1,weight_B=.1*ratio,ratio=ratio,
        norm_A=total('A'),norm_D=total('D'),rule='RMS raw SR gradient norm over 8 fixed train observations, no HR; not Adam update matching',
        checkpoint_sha256=sha256(args.checkpoint),cache_manifest_sha256=sha256(args.cache_manifest),
        source_sha256=sha256(__file__),optimizer_steps=0,independent_cpu_state_equal=True))


def train(args):
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False);(out/'sources').mkdir()
    m,g,h,o,ck,records,cams,ids,target=setup(args)
    calibration=json.loads(Path(args.calibration).read_text())
    assert calibration['checkpoint_sha256']==sha256(args.checkpoint)
    assert calibration['cache_manifest_sha256']==sha256(args.cache_manifest)
    weight=calibration['weight_B'] if args.branch=='B' else .1
    config=dict(**vars(args),weight=weight,parent_sha256=sha256(args.checkpoint),manifest_sha256=sha256(args.manifest),
        cache_manifest_sha256=sha256(args.cache_manifest),initial_points=len(g._xyz),sh_degree=g.active_sh_degree,
        topology_changes=False,new_parameters=0,scheduler_offset=ck['metadata']['step'],samplers='new identical fork streams, not resumed legacy streams',
        global_rng='explicit restored parent after initialization',gpu=torch.cuda.get_device_name(),visible_cuda=os.environ.get('CUDA_VISIBLE_DEVICES'),
        information_boundary='no HR image reads in this process; raw float32 C/D not clipped or quantized',sources=[])
    files=[Path(__file__),OLD/'common.py',OLD/'run_experiment.py',OLD/'n3dv_data.py',
        ROOT/'experiments/dynamic_sr_20260919/controlled_fit.py',ROOT/'experiments/dynamic_sr_20260920/resume_control.py',
        UPSTREAM/'scene/gaussian_model.py',UPSTREAM/'scene/deformation.py',UPSTREAM/'scene/hexplane.py',UPSTREAM/'gaussian_renderer/__init__.py']
    for n,f in enumerate(files):
        dst=out/'sources'/f'{n}_{f.name}';shutil.copyfile(f,dst);config['sources'].append(dict(path=str(f),sha256=sha256(f)))
    write_json(out/'config.json',config)
    lr_rng=random.Random(args.seed+177);sr_rng=random.Random(args.seed+211)
    exposure=collections.Counter();digest=hashlib.sha256();started=time.monotonic();train_s=0.
    milestones={int(v) for v in args.milestones.split(',') if int(v)<=args.steps}|{args.steps}
    initial_points=len(g._xyz);initial_sh=g.active_sh_degree;params=named_parameters(g)
    audit_steps={1,1200,3000,6000};torch.cuda.reset_peak_memory_stats()
    with (out/'training.jsonl').open('w',buffering=1) as log:
        for step in range(1,args.steps+1):
            tick=time.monotonic();g.update_learning_rate(ck['metadata']['step']+step)
            li,si=lr_rng.randrange(len(records)),sr_rng.choice(ids);digest.update(f'{li},{si}\n'.encode())
            truth=records[li]['image'].cuda();g.optimizer.zero_grad(set_to_none=True)
            pred=render_image(g,cams[li])['render'];loss_lr=(downsample(pred,truth.shape[-2:])-truth).abs().mean()
            reg=g.compute_regulation(h.time_smoothness_weight,h.l1_time_planes,h.plane_tv_weight)
            (loss_lr+reg).backward()
            if step in audit_steps:
                lr_reg_norm=norms(g)
                lr_grads={id(p):p.grad.detach().clone() for p in params.values() if p.grad is not None}
            loss_sr=(render_image(g,cams[si])['render']-target(si,args.branch).cuda()).abs().mean()
            (weight*loss_sr).backward()
            if not torch.isfinite(loss_lr+reg+loss_sr):raise FloatingPointError(step)
            if step in audit_steps:
                before={k:p.detach().clone() for k,p in params.items()};grad_norm=norms(g)
                sr_norm={group['name']:float(sum(((p.grad.detach()-lr_grads.get(id(p),0)).double().square().sum() for p in group['params'] if p.grad is not None),start=torch.zeros((),device='cuda')).sqrt()) for group in g.optimizer.param_groups}
                del lr_grads
            g.optimizer.step();torch.cuda.synchronize();train_s+=time.monotonic()-tick
            exposure[f"{records[si]['camera_id']}/{records[si]['frame_index']}"]+=1
            if step==1 or step%100==0 or step==args.steps:
                row=dict(step=step,lr_l1=float(loss_lr),teacher_l1=float(loss_sr),reg=float(reg),weight=weight,
                    points=len(g._xyz),train_s=train_s,wall_s=time.monotonic()-started,peak_gb=torch.cuda.max_memory_allocated()/1e9,
                    draw_sha256=digest.hexdigest())
                if step in audit_steps:
                    row['gradient_norms']=grad_norm
                    row['lr_plus_reg_gradient_norms']=lr_reg_norm
                    row['weighted_sr_gradient_norms']=sr_norm
                    row['actual_parameter_delta_norms']={k:float((p.detach()-before[k]).double().norm()) for k,p in params.items()}
                    del before
                log.write(json.dumps(row)+'\n');print(json.dumps(row),flush=True)
            if step in milestones:
                assert len(g._xyz)==initial_points and g.active_sh_degree==initial_sh
                assert all(torch.isfinite(p).all() for p in params.values())
                meta=dict(scene=m['scene'],stage='temporal_sharing_control',step=ck['metadata']['step']+step,
                    intervention_step=step,extent=ck['metadata']['extent'],args=vars(args),parent_sha=config['parent_sha256'],
                    manifest_sha=config['manifest_sha256'],elapsed_s=time.monotonic()-started,train_s=train_s,
                    points=len(g._xyz),draw_sha256=digest.hexdigest())
                checkpoint_with_samplers(out/f'checkpoint_{step}.pt',g,h,o,meta,lr_rng,sr_rng,{},exposure)
        log.flush();os.fsync(log.fileno())
    for record in config['sources']:assert sha256(record['path'])==record['sha256']
    (out/'checkpoint_final.pt').symlink_to(f'checkpoint_{args.steps}.pt')
    write_json(out/'exposure.json',dict(exposure))
    write_json(out/'complete.json',dict(**meta,wall_s=time.monotonic()-started,peak_gb=torch.cuda.max_memory_allocated()/1e9,
        parameter_updates=args.steps,source_unchanged=True,full_sampler_states_saved=True))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['calibrate','train'])
    p.add_argument('--manifest',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--cache-manifest',required=True)
    p.add_argument('--out',required=True);p.add_argument('--calibration');p.add_argument('--branch',choices=list('ABCD'),default='D')
    p.add_argument('--prior-cameras',default='cam02,cam04,cam08,cam12');p.add_argument('--seed',type=int,default=20260923)
    p.add_argument('--steps',type=int,default=6000);p.add_argument('--milestones',default='1200,3000,6000')
    args=p.parse_args();torch.set_num_threads(4)
    if args.mode=='calibrate':calibration(args)
    else:train(args)
