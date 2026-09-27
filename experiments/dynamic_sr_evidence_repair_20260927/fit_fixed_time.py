"""Two fixed 1200-update parameterizations; no HR/dev reads or regularization."""
import argparse
from collections import Counter
import random
import time
from context import *
from renderer import render_model,RENDERER_ID
from fixed_model import bake,render_baked,state

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--arm',choices=['Shared40','Baked40'],required=True);a=parser.parse_args()
    dest=OUT/'fixed_time'/a.arm;dest.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4);guard_training_images()
    p=paths();protocol=read(OUT/'protocol.json');cfg=protocol['fixed_time'];m=load_manifest(p['manifest']);sharedmodel=motion.load_model(p['start'],m,restore_rng=True)
    observations=sorted([o for o in m['observations'] if o['split']=='train' and o['frame_index']==cfg['frame']],key=lambda o:o['camera_id']);assert len(observations)==19
    teachers={(r['camera'],r['frame']):r for r in read(p['teacher'])['entries']};data=[]
    for i,o in enumerate(observations):
        e=teachers[o['camera_id'],o['frame_index']];lp=Path(m['_root'])/o['lr_path'];tp=Path(m['_root'])/e['relative_path'];assert sha256(lp)==o['lr_sha256'] and sha256(tp)==e['sha256']
        data.append((evalmod.render_camera(m,o,i),image_tensor(lp),image_tensor(tp)))
    baked=bake(sharedmodel,cfg['frame']/300);parity=[]
    with torch.no_grad():
        for o,(camera,lr,target) in zip(observations,data):
            original=render_model(sharedmodel,camera)['render'];other=render_baked(baked,camera)['render'];delta=(original-other).abs()
            parity.append(dict(camera=o['camera_id'],maxabs=float(delta.max()),meanabs=float(delta.mean())))
    write_json(dest/'initial_parity.json',dict(rows=parity,passed=all(r['maxabs']==0 for r in parity)))
    assert all(r['maxabs']==0 for r in parity),'Baking must preserve exact initial forward on all19 train cameras'
    rates={q['name']:dict(lr=q['lr'],eps=q['eps'],betas=q['betas']) for q in sharedmodel.g.optimizer.param_groups}
    if a.arm=='Shared40':
        model=sharedmodel;render_fn=render_model;named=all_named(model);opts=[]
        for oldopt in [model.g.optimizer,model.child_optimizer]:
            groups=[{k:v for k,v in g.items()} for g in oldopt.param_groups]
            opt=torch.optim.Adam(groups,**oldopt.defaults);assert not opt.state;opts.append(opt)
        model.g.optimizer,model.child_optimizer=opts
    else:
        model=baked;render_fn=render_baked;named=dict(model.named_parameters());mapping={'xyz':'xyz','logscale':'scaling','quaternion':'rotation','opacity':'opacity','sh_dc':'f_dc','sh_rest':'f_rest'}
        opts=[torch.optim.Adam([dict(params=[named[n]],name=n,**rates[group]) for n,group in mapping.items()],lr=0.,eps=1e-15)]
    schedule_rng=random.Random(cfg['seed']);sequence=[]
    while len(sequence)<cfg['steps']:
        ids=list(range(19));schedule_rng.shuffle(ids);sequence.extend(ids)
    sequence=sequence[:cfg['steps']];write_json(dest/'sequence.json',dict(camera_indices=sequence,cameras=[o['camera_id'] for o in observations]))
    optnames={id(v):n for n,v in named.items()};groups=[]
    for i,opt in enumerate(opts):
        for g in opt.param_groups:groups.append(dict(optimizer=i,group=g['name'],parameters=[optnames[id(v)] for v in g['params']],lr=g['lr'],betas=g['betas'],eps=g['eps']))
    config=dict(arm=a.arm,checkpoint_sha256=sha256(p['start']),renderer_id=RENDERER_ID,point_count=132972,
        trainable_scalars=sum(v.numel() for v in named.values() if v.requires_grad),groups=groups,initial_adam_states=0,regularization=False,
        fixed_time=cfg['frame']/300,source_sha256={str(f):sha256(f) for f in [Path(__file__),HERE/'fixed_model.py',HERE/'renderer.py',HERE/'context.py']},
        protocol_sha256=sha256(OUT/'protocol.json'),sequence_sha256=sha256(dest/'sequence.json'),gpu=torch.cuda.get_device_name(),visible_cuda=os.environ.get('CUDA_VISIBLE_DEVICES'),
        information_boundary='Only frame40 train19 LR and frozen teacher. Same point count and effective state, not equal freedom/capacity/optimization.')
    write_json(dest/'config.json',config);torch.cuda.reset_peak_memory_stats();started=time.time();train_s=0.
    with (dest/'training.jsonl').open('w',buffering=1) as log:
        for step,index in enumerate(sequence,1):
            tick=time.perf_counter();camera,lr,target=data[index]
            for opt in opts:opt.zero_grad(set_to_none=True)
            pred=render_fn(model,camera)['render'];lr_loss=(downsample(pred,lr.shape[-2:])-lr).abs().mean();sr_loss=(pred-target).abs().mean();loss=lr_loss+.1*sr_loss
            assert torch.isfinite(loss);loss.backward()
            for opt in opts:opt.step()
            torch.cuda.synchronize();train_s+=time.perf_counter()-tick
            if step==1 or step%100==0:log.write(json.dumps(dict(step=step,camera=observations[index]['camera_id'],lr_l1=float(lr_loss),teacher_l1=float(sr_loss),loss=float(loss),train_s=train_s))+'\n')
            if step in cfg['save_steps']:
                meta=dict(method=a.arm,frame=cfg['frame'],step=step,manifest_sha=sha256(p['manifest']),renderer_id=RENDERER_ID,intervention_step=step,privileged_train_hr=False)
                if a.arm=='Shared40':
                    meta.update(branch='ordinary_split',extent=sharedmodel.checkpoint['metadata']['extent'])
                    payload=dict(model=model.g.capture(),hidden=vars(model.h),optim=vars(model.o),metadata=meta,motion_refinement=motion.refinement_state(model),rng=rng(),fixed_time=config)
                else:payload=dict(baked=state(model),optimizer=cpu(opts[0].state_dict()),metadata=meta,fixed_time=config,rng=rng())
                ck=dest/f'checkpoint_{step}.pt';torch.save(payload,ck);write_json(dest/f'checkpoint_{step}.json',dict(path=str(ck),sha256=sha256(ck),metadata=meta))
    write_json(dest/'complete.json',dict(status='completed',actual_updates=cfg['steps'],train_s=train_s,seconds=time.time()-started,peak_gb=torch.cuda.max_memory_allocated()/1e9,config=config,source_unchanged=all(sha256(f)==v for f,v in config['source_sha256'].items())))

if __name__=='__main__':main()
