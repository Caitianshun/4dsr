"""Run pinned Multi4D with LR-only native losses and optional fixed teacher."""
import argparse
from collections import Counter,OrderedDict
import importlib.util
import json
import os
from pathlib import Path
import random
import runpy
import shutil
import sys
import time
import traceback
import numpy as np
import torch
from PIL import Image
from multi_data import sha,ManifestScene
import full_state as fs


def write(p,v):
    temp=p.with_suffix('.json.tmp');temp.write_text(json.dumps(v,indent=2,allow_nan=False,default=str)+'\n');temp.replace(p)


class Prepared(Exception):pass


class Context:
    def __init__(self,args,scene,hyper,schedule):
        self.a,self.scene,self.hyper,self.schedule=args,scene,hyper,schedule
        self.out=Path(args.out);self.started=time.monotonic();self.cache=OrderedDict()
        self.exposure=Counter();self.render_calls=0;self.teacher_samples=0
        self.sources={str(Path(__file__).with_name(name)):sha(Path(__file__).with_name(name)) for name in ['multi_train.py','multi_data.py','full_state.py','build_adapter.py']}
        self.sources[str(Path(args.adapter)/'adapted_train.py')]=sha(Path(args.adapter)/'adapted_train.py')
        for name in ['train.py','arguments/__init__.py','arguments/dynerf.py','scene/gaussian_model.py','scene/deformation.py','scene/hexplane.py','gaussian_renderer/__init__.py','utils/loss_utils.py']:
            path=Path(args.upstream)/name;self.sources[str(path)]=sha(path)
        for path in (Path(args.upstream)/'diff_gaussian_rasterization_hybrid').rglob('*'):
            if path.suffix in ['.cu','.h','.cpp'] and 'third_party' not in path.parts:self.sources[str(path)]=sha(path)
        (self.out/'sources').mkdir()
        for i,path in enumerate(self.sources):shutil.copyfile(path,self.out/'sources'/f'{i}_{Path(path).name}')
        self.meta=dict(manifest_sha=sha(args.manifest),init_sha=sha(args.init),method=args.method,
            schedule_sha=sha(args.schedule),upstream_commit='c483e82cd8daa1b4fdc460ed45882fc2fe7a22b0',
            sources=self.sources,gpu=torch.cuda.get_device_name(),torch=str(torch.__version__),
            fine0_sha=sha(args.resume) if args.resume else None,teacher_weight=.4 if args.method=='M1' else 0.)
        self.meta['singleton_scale_fix']=scene.singleton_scale_fix
        self.teachers={}
        if args.method=='M1':
            index=json.loads(Path(args.teacher_index).read_text());assert index['manifest_sha256']==sha(args.manifest)
            assert {(e['camera'],e['frame']) for e in index['entries']}=={c.record_key for c in scene.train_camera}
            for e in index['entries']:
                p=Path(args.manifest).parent/e['relative_path'];assert sha(p)==e['sha256']
                self.teachers[(e['camera'],e['frame'])]=p
            self.meta['teacher_index_sha']=sha(args.teacher_index)
        write(self.out/'input_audit.json',dict(records=scene.input_identity,projection=scene.projection,metadata=self.meta))
        self.log=(self.out/'training.jsonl').open('w',buffering=1)
        torch.cuda.reset_peak_memory_stats()

    def sampler_state(self,stage,step):
        return dict(stage=stage,completed_batches=step,schedule_sha=sha(self.a.schedule),
            exposure=dict(self.exposure),render_calls=self.render_calls,teacher_samples=self.teacher_samples)

    def stage_start(self,stage,*values):
        *branches,opt=values
        step=0
        if stage=='fine' and self.a.resume:
            state=torch.load(self.a.resume,map_location='cpu',weights_only=False)
            assert state['stage']=='fine' and state['metadata']['manifest_sha']==self.meta['manifest_sha']
            assert state['metadata']['schedule_sha']==self.meta['schedule_sha']
            assert state['metadata']['init_sha']==self.meta['init_sha']
            fs.restore(branches,state,opt);step=state['completed_updates']
            assert step==0 or self.a.smoke,'Formal M0/M1 start at common fine0'
            self.meta['restored_branch_identity']=fs.digest([fs.branch_state(g) for g in branches])
            write(self.out/'reload_audit.json',dict(status='passed',completed_updates=step,identity=self.meta['restored_branch_identity'],rng_identity=fs.digest(fs.rng())))
        if stage=='fine' and self.a.method=='coarse':
            state=fs.snapshot(branches,opt,self.hyper,stage,0,self.meta,self.sampler_state(stage,0))
            fs.save(self.out/'fine0.pt',state)
            fs.restore(branches,torch.load(self.out/'fine0.pt',map_location='cpu',weights_only=False),opt)
            write(self.out/'fine0_reload_audit.json',dict(status='passed',identity=fs.digest(state['branches']),rng=fs.digest(state['rng'])))
            raise Prepared()
        return step

    def batch(self,stage,iteration):
        ids=self.schedule[stage][iteration-1]
        for i in ids:self.exposure[f'{stage}/{self.scene.train_camera[i].image_name}']+=1
        self.render_calls+=len(ids)*(1 if iteration>=10000 else 4)
        return ids

    def teacher(self,stage,cameras,prediction):
        if stage!='fine' or self.a.method!='M1':return prediction.new_zeros(())
        targets=[]
        for c in cameras:
            key=c.record_key
            if key in self.cache:im=self.cache.pop(key)
            else:
                with Image.open(self.teachers[key]) as f:im=torch.from_numpy(np.asarray(f.convert('RGB')).copy()).permute(2,0,1).float()/255.
            self.cache[key]=im
            if len(self.cache)>64:self.cache.popitem(last=False)
            targets.append(im.cuda())
        target=torch.stack(targets);assert target.shape==prediction.shape
        self.teacher_samples+=len(cameras)
        return .4*(prediction-target).abs().mean()

    def after_backward(self,stage,step,*values):
        *branches,teacher=values
        if step in {1,2,20,50,3000,3001,9999,10000,10001,16000,20000}:
            norms={}
            for i,g in enumerate(branches):
                grads=[p.grad for group in g.optimizer.param_groups for p in group['params'] if p.grad is not None]
                assert all(bool(torch.isfinite(v).all()) for v in grads),f'Nonfinite gradient {stage} {step} {i}'
                norms[str(i)]=sum(float(v.square().sum()) for v in grads)**.5
            with (self.out/'gradient_audit.jsonl').open('a') as f:f.write(json.dumps(dict(stage=stage,step=step,gradient_norms=norms,teacher=float(teacher)))+'\n')

    def after_update(self,stage,step,loss,ll1,teacher,*values):
        fg,bg,tr,opt,phase3=values;branches=[fg,bg,tr]
        milestones={2000} if stage=='coarse' else {6000,10000,16000,20000}
        if self.a.smoke:milestones|={self.a.stop}
        row=dict(stage=stage,completed_updates=step,loss=float(loss),lr_l1=float(ll1),teacher=float(teacher),
            gaussian_counts=[len(g._xyz) for g in branches],peak_gb=torch.cuda.max_memory_allocated()/1e9,
            elapsed_s=time.monotonic()-self.started,render_calls=self.render_calls,teacher_samples=self.teacher_samples)
        if step==1 or step%100==0 or step in milestones:self.log.write(json.dumps(row)+'\n')
        if step in milestones:
            state=fs.snapshot(branches,opt,self.hyper,stage,step,self.meta,self.sampler_state(stage,step))
            path=self.out/f'{stage}_{step}.pt';fs.save(path,state)
            write(self.out/f'{stage}_{step}.json',dict(**row,checkpoint=str(path),sha256=sha(path),sampler=state['sampler']))
            if self.a.smoke:
                # Restoring all state after a real short update checks controller and Adam tensors.
                before=fs.digest(state['branches']);fs.restore(branches,torch.load(path,map_location='cpu',weights_only=False),opt)
                assert before==fs.digest([fs.branch_state(g) for g in branches])
                write(self.out/'trained_reload_audit.json',dict(status='passed',stage=stage,step=step,identity=before))
        if self.a.smoke and step>=self.a.stop:raise Prepared()


def config(upstream):
    from arguments import ModelParams,PipelineParams,OptimizationParams,ModelHiddenParams
    from utils.params_utils import merge_hparams
    import mmcv
    p=argparse.ArgumentParser();lp=ModelParams(p);op=OptimizationParams(p);pp=PipelineParams(p);hp=ModelHiddenParams(p)
    a=p.parse_args([]);a=merge_hparams(a,mmcv.Config.fromfile(str(Path(upstream)/'arguments/dynerf.py')))
    return lp.extract(a),hp.extract(a),op.extract(a),pp.extract(a)


def main():
    p=argparse.ArgumentParser()
    for k in ['upstream','adapter','manifest','init','schedule','out']:p.add_argument('--'+k,required=True)
    p.add_argument('--method',choices=['coarse','M0','M1'],required=True);p.add_argument('--resume');p.add_argument('--teacher-index')
    p.add_argument('--smoke',action='store_true');p.add_argument('--stop',type=int,default=20)
    a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,a.upstream);torch.set_num_threads(4)
    try:
        spec=importlib.util.spec_from_file_location('multi4d_adapted',Path(a.adapter)/'adapted_train.py');native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)
        dataset,hyper,opt,pipe=config(a.upstream)
        # Match effective official safe_state seed (0), isolated sampler uses a different RNG.
        random.seed(0);np.random.seed(0);torch.manual_seed(0);torch.cuda.manual_seed_all(0);torch.backends.cudnn.deterministic=True
        fg=native.GaussianModel_dynamic(3,hyper);bg=native.GaussianModel(3)
        tr=native.GaussianModelTransient(3,gaussian_dim=4,time_duration=[0.,10.],rot_4d=True,sh_degree_t=2)
        scene=ManifestScene(a.manifest,a.init,fg,bg,tr,out)
        schedule=json.loads(Path(a.schedule).read_text());assert schedule['record_keys']==[list(c.record_key) for c in scene.train_camera]
        ctx=Context(a,scene,hyper,schedule);native.adapter_context=ctx
        write(out/'config.json',dict(args=vars(a),native_optimizer=vars(opt),native_hidden=vars(hyper),metadata=ctx.meta))
        try:
            if a.method=='coarse':
                native.scene_reconstruction(opt,hyper,pipe,[],[],None,fg,scene,'coarse',2000,bg,tr)
                fg.max_radii2D=torch.zeros_like(fg.max_radii2D).cuda();fg.mean_radii2D=torch.zeros_like(fg.mean_radii2D).cuda();bg.max_radii2D=torch.zeros_like(bg.max_radii2D).cuda()
            else:assert a.resume
            native.scene_reconstruction(opt,hyper,pipe,[],[],None,fg,scene,'fine',20000,bg,tr)
        except Prepared:pass
        for path,h in ctx.sources.items():assert sha(path)==h,('Source changed',path)
        write(out/'complete.json',dict(status='smoke_completed' if a.smoke else 'fine0_prepared' if a.method=='coarse' else 'completed',metadata=ctx.meta,
            elapsed_s=time.monotonic()-ctx.started,peak_gb=torch.cuda.max_memory_allocated()/1e9,render_calls=ctx.render_calls,teacher_samples=ctx.teacher_samples))
    except BaseException:
        write(out/'failed.json',dict(status='failed',traceback=traceback.format_exc()));raise


if __name__=='__main__':main()
