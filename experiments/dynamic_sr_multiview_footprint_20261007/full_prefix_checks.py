"""CPU properties for the full author LR-prefix adapter; never train a GPU."""
from __future__ import annotations
import ast
import copy
import json
from pathlib import Path
import random
import sys
import time
import traceback
from types import SimpleNamespace,ModuleType

import numpy as np
import torch
import full_prefix as fp
from fp_common import ROOT,HERE,OUT,read,write,entry,sha


def author_fragments():
    sources=read(fp.READINESS)['original_Wu']['sources']
    path=ROOT/next(e['path'] for e in sources if e['git_path']=='train.py')
    function=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='scene_reconstruction')
    loop=next(n for n in function.body if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='iteration')
    sampler=next(n for n in loop.body if isinstance(n,ast.If) and ast.unparse(n.test)=='opt.dataloader and (not load_in_memory)')
    block=next(n for n in loop.body if isinstance(n,ast.With))
    topology=next(n for n in block.body if isinstance(n,ast.If) and ast.unparse(n.test)=='iteration < opt.densify_until_iter')
    adam=next(n for n in block.body if isinstance(n,ast.If) and ast.unparse(n.test)=='iteration < opt.iterations')
    def compiled(nodes):return compile(ast.fix_missing_locations(ast.Module(body=copy.deepcopy(nodes),type_ignores=[])),str(path),'exec')
    wrapped=ast.For(target=ast.Name(id='_one_iteration',ctx=ast.Store()),iter=ast.Tuple(elts=[ast.Constant(0)],ctx=ast.Load()),body=[sampler],orelse=[])
    return compiled([wrapped]),compiled([topology,adam]),entry(path)


def check_sampler(code):
    from torch.utils.data import DataLoader
    results=[]
    for length in (5,6,20):
        seed=4281
        torch.manual_seed(seed)
        dataset=list(range(length));loader=DataLoader(dataset,batch_size=2,shuffle=True,num_workers=0,collate_fn=list)
        scope=dict(opt=SimpleNamespace(dataloader=True,batch_size=2),load_in_memory=False,loader=iter(loader),
            random_loader=True,viewpoint_stack_loader=loader,viewpoint_stack=dataset,DataLoader=DataLoader)
        expected=[];states=[]
        for _ in range(35):
            exec(code,scope)
            expected.append(list(scope['viewpoint_cams']));torch.randn(4);states.append(torch.get_rng_state().clone())
        torch.manual_seed(seed);sampler=fp.AuthorShuffle(length,2,torch)
        for index in range(35):
            assert sampler.next()==expected[index],(length,index,sampler.state_dict(),expected[index])
            torch.randn(4)
            assert torch.equal(torch.get_rng_state(),states[index]),('CPU DataLoader RNG draw drift',length,index)
            if index in (0,3,12):
                state=sampler.state_dict();before=torch.get_rng_state().clone()
                sampler=fp.AuthorShuffle(length,2,torch,copy.deepcopy(state))
                assert torch.equal(torch.get_rng_state(),before),'Restore consumed new shuffle RNG'
        results.append(dict(length=length,iterations=35,author_epoch_repeats=sampler.repeated,exact_batch_trace_and_CPU_RNG=True))
    return results


class TopologyFake:
    def __init__(self,n):
        self.n=n;self.calls=[];self.max_radii2D=torch.zeros(4)
        self.optimizer=SimpleNamespace(step=lambda:self.calls.append(('Adam',)),zero_grad=lambda **kw:self.calls.append(('zero_grad',kw)))
    @property
    def get_xyz(self):return SimpleNamespace(shape=(self.n,3))
    def add_densification_stats(self,*args):self.calls.append(('stats',tuple(tuple(v.shape) for v in args)))
    def densify(self,*args):self.calls.append(('densify',args));self.n+=7
    def prune(self,*args):self.calls.append(('prune',args));self.n-=3
    def grow(self,*args):self.calls.append(('grow',args))
    def reset_opacity(self):self.calls.append(('opacity_reset',))


def check_topology(code,opt,directory):
    results=[];radii=torch.tensor([1.,5.,3.,7.]);visible=torch.tensor([True,False,True,True]);grad=torch.ones(4,3)
    for stage in ('coarse','fine'):
        for iteration in (500,501,600,9999,10000,14000):
            for n in (199999,200001,359999,360000):
                reference=TopologyFake(n);observed=TopologyFake(n)
                scope=dict(gaussians=reference,opt=opt,stage=stage,iteration=iteration,scene=SimpleNamespace(cameras_extent=2.5,model_path=str(directory)),
                    torch=torch,radii=radii,visibility_filter=visible,viewspace_point_tensor_grad=grad)
                exec(code,scope)
                fp.author_topology(observed,opt,stage,iteration,2.5,directory,(radii,visible,grad))
                if iteration<opt.iterations:observed.optimizer.step();observed.optimizer.zero_grad(set_to_none=True)
                assert reference.calls==observed.calls,(stage,iteration,n,reference.calls,observed.calls)
                assert torch.equal(reference.max_radii2D,observed.max_radii2D)
                results.append(dict(stage=stage,iteration=iteration,points=n,events=[c[0] for c in observed.calls]))
    return dict(cases=len(results),against='Executable original author densification/pruning/Adam AST',boundary_cases=results)


class TinyGaussian:
    def __init__(self):
        self.active_sh_degree=0;self.spatial_lr_scale=2.5
        for name in ('_xyz','_features_dc','_features_rest','_scaling','_rotation','_opacity'):
            setattr(self,name,torch.nn.Parameter(torch.randn(2,3)*.02+.13))
        self._deformation=torch.nn.Linear(3,3,bias=False)
        self._deformation_table=torch.ones(2,dtype=torch.bool);self.max_radii2D=torch.zeros(2)
        self.resets=0;self.regulations=0;self.learning_rate_iterations=[]
    @property
    def get_xyz(self):return self._xyz
    def training_setup(self,opt):
        self.resets+=1;self.xyz_gradient_accum=torch.zeros(2,1);self.denom=torch.zeros(2,1);self._deformation_accum=torch.zeros(2,3)
        values=fp.parameter_map(self)
        self.optimizer=torch.optim.Adam([dict(params=[p],name=name,lr=.02) for name,p in values.items()],eps=1e-15)
    def update_learning_rate(self,iteration):
        self.learning_rate_iterations.append(iteration)
        for group in self.optimizer.param_groups:group['lr']=.02/(iteration+1)
    def oneupSHdegree(self):self.active_sh_degree=min(3,self.active_sh_degree+1)
    def compute_regulation(self,*args):
        self.regulations+=1
        return self._deformation.weight.square().mean()*.001


def bind_native_capture_restore():
    source=ROOT/next(e['path'] for e in read(fp.READINESS)['original_Wu']['sources'] if e['git_path']=='scene/gaussian_model.py')
    cls=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='GaussianModel')
    nodes=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in ('capture','restore')]
    scope={};exec(compile(ast.fix_missing_locations(ast.Module(body=copy.deepcopy(nodes),type_ignores=[])),str(source),'exec'),scope)
    TinyGaussian.capture=scope['capture'];TinyGaussian.restore=scope['restore']


def tiny_render(camera,g,pipe,background,stage,cam_type):
    assert cam_type=='dynerf'
    view=g._xyz*0;view.retain_grad()
    canonical=sum(p.mean() for name,p in fp.parameter_map(g).items() if not name.startswith('deformation.'))
    value=canonical*.12+view[:,0].sum()*(1+camera.index*.2)
    if stage=='fine':value=value+g._deformation(g._xyz).mean()
    # Exercise independent RNG paths across a resumable checkpoint.
    value=value+torch.rand(())*.005+float(np.random.rand()+random.random())*.001
    image=value.expand(3,2,2)
    return dict(render=image,viewspace_points=view,visibility_filter=torch.tensor([True,camera.index%2==0]),radii=torch.tensor([1.+camera.index,2.]))


def equal_tree(a,b):
    if torch.is_tensor(a):return torch.is_tensor(b) and torch.equal(a,b)
    if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and np.array_equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal_tree(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return type(a)==type(b) and len(a)==len(b) and all(equal_tree(x,y) for x,y in zip(a,b))
    return a==b


def check_rng_cpu_transport(directory):
    """Simulate CUDA map_location without initializing or allocating CUDA."""
    from unittest.mock import patch
    original=fp.rng_state(torch,np);cpu_state=original['torch'].clone()
    # Real CPU generator validation rejects non-Byte inputs. The CUDA setter
    # is injected, since executing a CUDA generator would initialize CUDA.
    torch.Generator(device='cpu').set_state(cpu_state)
    try:torch.Generator(device='cpu').set_state(cpu_state.float())
    except TypeError:pass
    else:raise AssertionError('Real generator accepted non-Byte RNG state')
    class MappedCUDAState:
        def __init__(self,value):self.value=value;self.cpu_calls=0;self.device='cuda:SIMULATED_CPU_ONLY'
        def cpu(self):self.cpu_calls+=1;return self.value.clone()
    mapped_host=MappedCUDAState(cpu_state);device_values=[torch.arange(16,dtype=torch.uint8),torch.arange(16,dtype=torch.uint8)+1]
    mapped_device=[MappedCUDAState(value) for value in device_values];received=[]
    def cuda_set(states):
        assert all(isinstance(state,torch.Tensor) and state.device.type=='cpu' and state.dtype==torch.uint8 for state in states)
        assert all(torch.equal(a,b) for a,b in zip(states,device_values));received.extend(states)
    payload=dict(original,torch=mapped_host,cuda=mapped_device)
    with patch.object(torch.cuda,'set_rng_state_all',cuda_set):fp.restore_rng(payload,torch,np)
    assert torch.equal(torch.get_rng_state(),cpu_state) and len(received)==2
    assert mapped_host.cpu_calls==1 and all(state.cpu_calls==1 for state in mapped_device)
    assert all(torch.equal(state.value,value) for state,value in zip(mapped_device,device_values))
    assert not torch.cuda.is_initialized()
    return dict(CUDA_transport_simulated_without_CUDA=True,actual_CPU_generator_set_state_ByteTensor_validation=True,
        CPU_and_all_CUDA_rng_inputs_on_CPU=True,exact_saved_bytes_preserved=True,model_Adam_device_mapping_not_touched=True)


def check_precise_resume_amendment(directory):
    """Use the archived real coarse100 identity; no model forward or mutation."""
    archives=sorted((OUT/'source_preparation_archive').glob('full_prefix_CUDA_RNG_transport_before_fix_*'))
    if not archives:return dict(status='not_applicable_without_this_real_incident_archive')
    archive=archives[-1];manifest=read(archive/'manifest.json')
    old_source=ROOT/next(item['archive']['path'] for item in manifest['copies'] if item['original']['path'].endswith('/full_prefix.py'))
    old_plan=ROOT/next(item['archive']['path'] for item in manifest['copies'] if item['original']['path'].endswith('/seed_20261007/config.json'))
    parent=ROOT/manifest['unchanged_checkpoint']['path'];out=parent.parent
    requested,_=fp.build_plan(ROOT/'data/dynamic_sr/full_time_20261007/prepared/cook_spinach/manifest_train_ready_seed20261007.json',20261007)
    requested={key:value for key,value in requested.items() if key!='CPU_seconds'}
    registered=read(old_plan);sidecar=parent.with_suffix('.json')
    # Build a temporary amendment bound to the real immutable identity. It
    # remains below CPU-check artifacts, not the production run registration.
    value=dict(status='registered_precise_RNG_CPU_transport_compatible_resume',kind=fp.RNG_TRANSPORT_AMENDMENT,
        original_registered_plan=entry(out/'config.json'),old_plan_source=entry(old_plan),
        registered_plan_sha256=fp.digest(registered),requested_plan_sha256=fp.digest(requested),
        old_source=entry(old_source),execution_source=entry(HERE/'full_prefix.py'),parent_checkpoint=entry(parent),
        parent_sidecar=entry(sidecar),parent_state=read(sidecar)['metadata']['state'],source_contract=fp.rng_transport_source_contract(old_source),
        incident_evidence=[entry(archive/'manifest.json')])
    amendment=directory/'precise_rng_amendment_CPU_fixture.json';write(amendment,value)
    assert fp.validate_resume_amendment(amendment,registered,requested,out)==entry(amendment)
    for modification in ('learning_rate','data','other_source'):
        bad=copy.deepcopy(requested)
        if modification=='learning_rate':bad['configuration']['OptimizationParams']['position_lr_init']*=2
        elif modification=='data':bad['data']['heldout_camera']='cam01'
        else:bad['project_sources'][str(HERE.joinpath('fp_common.py').relative_to(ROOT))]='0'*64
        bad_value=dict(value,requested_plan_sha256=fp.digest(bad));bad_path=directory/('rejected_'+modification+'.json');write(bad_path,bad_value)
        try:fp.validate_resume_amendment(bad_path,registered,bad,out)
        except ValueError:pass
        else:raise AssertionError('General identity bypass accepted '+modification)
    # CPU map_location opens the real accepted checkpoint without altering it.
    before=sha(parent);payload=torch.load(parent,map_location='cpu',weights_only=False)
    assert payload['plan']==registered and payload['metadata']==read(sidecar)['metadata']
    assert payload['rng']['torch'].device.type=='cpu' and payload['rng']['torch'].dtype==torch.uint8
    assert all(state.device.type=='cpu' and state.dtype==torch.uint8 for state in payload['rng']['cuda'])
    assert sha(parent)==before==manifest['unchanged_checkpoint']['sha256']
    assert entry(out/'config.json')==manifest['original_plan']
    state=payload['metadata']['state']
    assert state['coarse_iteration']==100 and state['accepted_Adam']==100 and state['optimizer_resets']==1
    return dict(precise_single_source_leaf_amendment_accepted=True,changed_learning_rule_data_other_source_rejected=True,
        checkpoint=entry(parent),original_config=entry(out/'config.json'),checkpoint_bytes_unchanged=True,
        real_CPU_loaded_rng_dtype='torch.uint8',saved_CUDA_rng_states=len(payload['rng']['cuda']),
        accepted_coarse_iteration=100,accepted_Adam=100,source_contract=value['source_contract'],GPU_calls=0)


def tiny_run(directory,cut=None):
    random.seed(431);np.random.seed(431);torch.manual_seed(431)
    opt=SimpleNamespace(iterations=7,coarse_iterations=3,batch_size=2,densify_until_iter=0,lambda_dssim=0)
    h=SimpleNamespace(time_smoothness_weight=.001,l1_time_planes=.0001,plane_tv_weight=.0002)
    model=TinyGaussian();model.training_setup(opt);sampler=fp.AuthorShuffle(6,2,torch)
    journal=fp.Journal(directory,'CPU_toy_author_order','coarse',0)
    state=dict(stage='coarse',coarse_iteration=0,fine_iteration=0,optimizer_resets=1,accepted_RGB=0,accepted_backward=0,accepted_Adam=0)
    plan=dict(scope='CPU tiny test; no official training or native GPU claim')
    grad_saved=False;resumed=False;checkpoints=[]
    for stage,maximum in (('coarse',3),('fine',7)):
        if stage!=state['stage']:
            model.training_setup(opt);sampler=fp.AuthorShuffle(6,2,torch);state.update(stage=stage,optimizer_resets=2)
        if cut==(stage,0):
            path=directory/f'checkpoint_{stage}_00000.pt'
            fp.save_checkpoint(path,model,h,opt,state,plan,sampler,torch,np,journal)
            # Use actual atomic-payload orphan-sidecar recovery.
            path.with_suffix('.json').unlink();assert fp.latest_checkpoint(directory,torch,plan)==path
            replacement=TinyGaussian();replacement.training_setup(opt)
            state,sampler,_=fp.restore_checkpoint(path,replacement,opt,plan,torch,np);model=replacement;resumed=True
        for iteration in range(1,maximum+1):
            indices=sampler.next();cameras=[SimpleNamespace(index=i,original_image=torch.full((3,2,2),.23+.01*i)) for i in indices]
            fp.author_iteration(model,h,opt,SimpleNamespace(),torch.ones(3),cameras,tiny_render,stage,iteration,2.5,directory,torch,journal)
            state[stage+'_iteration']=iteration;state['accepted_RGB']+=len(cameras);state['accepted_backward']+=1;state['accepted_Adam']+=int(iteration<opt.iterations)
            if cut==(stage,iteration):
                path=directory/f'checkpoint_{stage}_{iteration:05d}.pt'
                fp.save_checkpoint(path,model,h,opt,state,plan,sampler,torch,np,journal);checkpoints.append(entry(path))
                replacement=TinyGaussian();replacement.training_setup(opt)
                state,sampler,payload=fp.restore_checkpoint(path,replacement,opt,plan,torch,np)
                grad_saved=any(p.grad is not None for p in fp.parameter_map(replacement).values());model=replacement;resumed=True
    journal.finish('completed_CPU_fixture')
    result=dict(model=model.capture(),deformation_accum=model._deformation_accum,gradients={n:p.grad for n,p in fp.parameter_map(model).items()},
        rng=fp.rng_state(torch,np),sampler=sampler.state_dict(),state=state)
    assert state['accepted_RGB']==20 and state['accepted_backward']==10 and state['accepted_Adam']==9 and state['optimizer_resets']==2
    return result,dict(cut=cut,resumed=resumed,terminal_gradients_restored=grad_saved if cut==('fine',7) else None,checkpoints=checkpoints,
        actual_CPU_Adam_calls=9,actual_CPU_RGB_fake_calls=20,actual_CPU_backward_calls=10)


def check_batch_statistics():
    packets=[]
    for radii,visible,gradient in (([1.,6.],[True,False],[[1.,2.,0.],[4.,0.,0.]]),([5.,2.],[False,True],[[-1.,1.,0.],[0.,3.,0.]])):
        view=torch.zeros(2,3,requires_grad=True);view.grad=torch.tensor(gradient)
        packets.append(dict(radii=torch.tensor(radii),visibility_filter=torch.tensor(visible),viewspace_points=view))
    r,v,g=fp.combine_batch_statistics(packets,torch)
    assert torch.equal(r,torch.tensor([5.,6.])) and bool(v.all())
    assert torch.equal(g,torch.tensor([[0.,3.,0.],[4.,3.,0.]]))
    return dict(max_radii=True,any_visibility=True,sum_gradients_before_norm=True,opposed_X_gradients_cancel=True)


def native_deformation_seed_check(config):
    upstream=Path('/home/cai_tianshun/Project/4dgs');sys.path.insert(0,str(upstream))
    from scene.gaussian_model import GaussianModel
    values=[]
    for seed in (20261007,20261008):
        torch.manual_seed(seed)
        model=GaussianModel(3,SimpleNamespace(**config['ModelHiddenParams']))
        values.append(next(model._deformation.parameters()).detach().clone())
        del model
    assert not torch.equal(values[0],values[1]) and not torch.cuda.is_initialized()
    return dict(fresh_author_deformation_parameters_differ=True,seeds=[20261007,20261008],CUDA_initialized=False,
        initial_Gaussian_geometry_creation_executed=False)


def check_real_train_control(directory):
    """Run the actual train() control flow with CPU model/camera substitutes.

    Uses the legal full manifest and real LR hashes, but creates no rasterizer
    and never opens the pixels. This catches registration, stop and resume
    defects that checking only the iteration helper cannot expose.
    """
    manifest=ROOT/'data/dynamic_sr/full_time_20261007/prepared/cook_spinach/manifest_train_ready_seed20261007.json'
    plan,inputs=fp.build_plan(manifest,20261007)
    plan['status']='CPU_toy_actual_train_control_only_not_a_real_prefix'
    plan['configuration']['OptimizationParams'].update(coarse_iterations=3,iterations=7)
    out=directory/'actual_train_control';out.mkdir(parents=True)
    write(out/'hardware.json',dict(host='CPU_fixture',gpu_uuid=None,GPU_execution=False))
    class Model(TinyGaussian):
        def __init__(self,*args):
            super().__init__()
            self._deformation.cuda=lambda:self._deformation
            self._deformation.deformation_net=SimpleNamespace(set_aabb=lambda *x:None)
        def create_from_pcd(self,*args):pass
        def add_densification_stats(self,gradient,visible):
            self.xyz_gradient_accum[visible]+=gradient[visible,:2].norm(dim=-1,keepdim=True)
            self.denom[visible]+=1
    class Dataset:
        def __init__(self,m,*args,**kwargs):
            self.observations=[r for r in m['observations'] if r['split']=='train']
        def __len__(self):return len(self.observations)
        def __getitem__(self,index):return dict(index=index,image=torch.zeros(3,252,336))
    def manifest_loader(path):
        m=read(path);m['_root']=str(Path(path).parent);return m
    def render(camera,*args,**kwargs):
        packet=tiny_render(camera,*args,**kwargs)
        packet['render']=packet['render'][:,0:1,0:1].expand(3,252,336)
        return packet
    fake_model=ModuleType('scene.gaussian_model');fake_model.GaussianModel=Model
    fake_renderer=ModuleType('gaussian_renderer');fake_renderer.render=render
    fake_data=ModuleType('n3dv_data');fake_data.load_manifest=manifest_loader
    fake_data.load_initial_points=lambda m:dict(points=np.array([[0.,0.,0.],[1.,1.,1.]],dtype=np.float32),colors=np.ones((2,3),dtype=np.float32)*.2)
    fake_data.N3DVPreparedDataset=Dataset
    fake_data.observation_to_4dgs_camera=lambda item,index:SimpleNamespace(index=index,original_image=item['image'])
    replacements={'scene.gaussian_model':fake_model,'gaussian_renderer':fake_renderer,'n3dv_data':fake_data}
    previous={name:sys.modules.get(name) for name in replacements}
    original_ones=torch.ones;original_seed=torch.cuda.manual_seed_all
    def cpu_ones(*args,**kwargs):
        if kwargs.get('device')=='cuda':kwargs['device']='cpu'
        return original_ones(*args,**kwargs)
    records=[]
    try:
        sys.modules.update(replacements);torch.ones=cpu_ones;torch.cuda.manual_seed_all=lambda seed:None
        for index,(stage,stop) in enumerate((('coarse',2),('coarse',3),('fine',2),('fine',3))):
            args=SimpleNamespace(out=out,resume=None if index==0 else 'latest',upstream=None,manifest=manifest,
                seed=20261007,cpu_threads=1,stop_stage=stage,stop_iteration=stop,checkpoint_interval=100)
            fp.train(args,copy.deepcopy(plan),inputs)
            status=read(out/'status.json');records.append(status['state'])
            assert status['state'][stage+'_iteration']==stop and status['status']=='paused_prefix_incomplete'
            assert not torch.cuda.is_initialized()
        assert records[1]['stage']=='coarse' and records[1]['fine_iteration']==0 and records[1]['optimizer_resets']==1
        assert records[2]['optimizer_resets']==2 and records[3]['accepted_Adam']==6
        assert len(list(out.glob('checkpoint_fine_00000.pt')))==1
        # A second fresh execution must reject the existing run before any model
        # construction/operation, rather than silently cold-restarting it.
        args.resume=None
        try:fp.train(args,copy.deepcopy(plan),inputs)
        except ValueError as error:assert 'explicit --resume' in str(error)
        else:raise AssertionError('Existing prefix cold-restarted')
    finally:
        torch.ones=original_ones;torch.cuda.manual_seed_all=original_seed
        for name,value in previous.items():
            if value is None:sys.modules.pop(name,None)
            else:sys.modules[name]=value
    return dict(actual_train_control_executed=True,states=records,coarse_endpoint_does_not_enter_fine=True,
        same_identity_resume_accepted=True,cold_restart_rejected=True,CPU_toy_Adam_calls=6,
        CPU_toy_RGB_fake_calls=12,CPU_toy_backward_calls=6,CUDA_initialized=False)


def main():
    started=time.monotonic();torch.set_num_threads(1);assert not torch.cuda.is_initialized()
    directory=OUT/'operator_checks/full_prefix'/str(time.time_ns());directory.mkdir(parents=True)
    for name in ('full_prefix.py','full_prefix_checks.py'):(directory/name).write_bytes((HERE/name).read_bytes())
    report=dict(status='running_CPU_full_prefix_properties',source={name:entry(HERE/name) for name in ('full_prefix.py','full_prefix_checks.py')},
        GPU_RGB_forwards=0,GPU_moment_forwards=0,GPU_Adam_calls=0,formal_prefix_models_trained=0)
    try:
        sampler_code,topology_code,author=author_fragments();report['author_train']=author
        report['rng_cpu_transport']=check_rng_cpu_transport(directory)
        report['precise_resume_amendment']=check_precise_resume_amendment(directory)
        report['sampler']=check_sampler(sampler_code)
        report['batch_statistics']=check_batch_statistics()
        configuration,_=fp.author_configuration('cook_spinach');report['planned_author_counts']=fp.planned_counts(configuration)
        assert report['planned_author_counts']['total_Adam_calls']==16999 and report['planned_author_counts']['RGB_forwards']==34000
        report['topology']=check_topology(topology_code,SimpleNamespace(**configuration['OptimizationParams']),directory)
        bind_native_capture_restore();baseline,base_info=tiny_run(directory/'uninterrupted')
        trials=[]
        for index,cut in enumerate((('coarse',2),('coarse',3),('fine',0),('fine',3),('fine',7))):
            result,info=tiny_run(directory/f'resume_{index}',cut)
            assert equal_tree(baseline,result),('Interrupted native-capture/restore path differs',cut)
            trials.append(info)
        report['checkpoint_resume']=dict(uninterrupted=base_info,trials=trials,exact_model_Adam_RNG_sampler_and_final_gradients=True)
        # Failure journal must retain the last dispatched operation before completion.
        failed=fp.Journal(directory/'fault_cost','CPU_fault','fine',19);failed.begin('RGB');failed.end('RGB');failed.begin('Adam');failed.finish('injected_failure')
        cost=fp.historical_cost(directory/'fault_cost')
        assert cost['completed']['RGB']==1 and cost['completed']['Adam']==0 and cost['dispatched']['Adam']==1
        assert cost['unresolved_active_operations'][0]['operation']=='Adam'
        report['failure_cost']=cost
        plans=[]
        for scene in ('cook_spinach','cut_roasted_beef'):
            for seed in (20261007,20261008):
                manifest=ROOT/f'data/dynamic_sr/full_time_20261007/prepared/{scene}/manifest_train_ready_seed{seed}.json'
                plan,_=fp.build_plan(manifest,seed);plans.append(dict(scene=scene,seed=seed,data=plan['data'],planned=plan['planned'],configuration_sha256=fp.digest(plan['configuration'])))
                assert plan['data']['extent_cam01_included'] and plan['data']['heldout_camera']=='cam00'
                expected=6000 if scene=='cook_spinach' else 5700
                assert plan['data']['training_observations']==expected
        report['full_input_protocols']=plans
        report['native_CPU_deformation_seed']=native_deformation_seed_check(configuration)
        report['real_train_control']=check_real_train_control(directory)
        assert not torch.cuda.is_initialized()
        report.update(status='passed_CPU_full_prefix_source_protocol_sampler_topology_checkpoint_semantics',CPU_seconds=time.monotonic()-started,
            actual_toy_CPU_Adam_calls=60,actual_toy_CPU_backward_calls=66,actual_toy_CPU_RGB_fake_calls=132,
            limitations=['No native GPU rasterization or real LR prefix training executed.','Full SR suffix and teacher/support are separate adapters.'])
    except BaseException:
        report.update(status='failed_CPU_full_prefix_check_preserved',CPU_seconds=time.monotonic()-started,error=traceback.format_exc())
        write(directory/'receipt.json',report);print(json.dumps(dict(status=report['status'],receipt=str(directory/'receipt.json'),error=report['error'])));raise
    write(directory/'receipt.json',report)
    print(json.dumps(dict(status=report['status'],receipt=str(directory/'receipt.json'),CPU_seconds=report['CPU_seconds'],CUDA_initialized=False)),flush=True)


if __name__=='__main__':main()
