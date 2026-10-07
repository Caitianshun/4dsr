"""CPU-only native SR interface, gradients, topology and checkpoint checks."""
from __future__ import annotations

import ast
import copy
import json
from pathlib import Path
import random
import os
import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
import full_refine as sr
import full_prefix as fp
import full_prefix_checks as old_checks
from fp_common import ROOT,HERE,OUT,entry,read,write,sha


class Deformation(torch.nn.Module):
    def __init__(self):
        super().__init__();self.mlp=torch.nn.Linear(3,3,bias=False);self.grid=torch.nn.Parameter(torch.tensor(.006));self.calls=0
    def forward(self,xyz,scale,rotation,opacity,sh,time):
        self.calls+=1;delta=self.mlp(xyz)*.01+time*self.grid
        return xyz+delta,scale+delta*.01,rotation,opacity+delta[:,:1]*.02,sh+delta[:,None,:]*.03


class Gaussian:
    def __init__(self):
        self._xyz=torch.nn.Parameter(torch.tensor([[-.3,-.2,3.],[.3,.2,4.]]))
        self._features_dc=torch.nn.Parameter(torch.randn(2,1,3)*.03)
        self._features_rest=torch.nn.Parameter(torch.randn(2,15,3)*.02)
        self._scaling=torch.nn.Parameter(torch.full((2,3),-2.))
        self._rotation=torch.nn.Parameter(torch.tensor([[1.,.01,.02,.03],[1.,.02,.03,.01]]))
        self._opacity=torch.nn.Parameter(torch.full((2,1),-.6))
        self._deformation=Deformation();self._deformation_table=torch.ones(2,dtype=torch.bool)
        self.active_sh_degree=2;self.max_sh_degree=3;self.spatial_lr_scale=2.5;self.max_radii2D=torch.zeros(2)
        self.scaling_activation=torch.exp;self.rotation_activation=torch.nn.functional.normalize;self.opacity_activation=torch.sigmoid
        self.events=[];self.regulations=0;self.setup_calls=0
    @property
    def get_xyz(self):return self._xyz
    @property
    def get_features(self):return torch.cat((self._features_dc,self._features_rest),dim=1)
    def training_setup(self,opt):
        self.setup_calls+=1;self.xyz_gradient_accum=torch.zeros(2,1);self.denom=torch.zeros(2,1);self._deformation_accum=torch.zeros(2,3)
        self.optimizer=torch.optim.Adam([
            dict(name='xyz',params=[self._xyz],lr=.002),dict(name='deformation',params=list(self._deformation.mlp.parameters()),lr=.003),
            dict(name='grid',params=[self._deformation.grid],lr=.004),dict(name='f_dc',params=[self._features_dc],lr=.005),
            dict(name='f_rest',params=[self._features_rest],lr=.006),dict(name='scaling',params=[self._scaling],lr=.007),
            dict(name='rotation',params=[self._rotation],lr=.008),dict(name='opacity',params=[self._opacity],lr=.009)],eps=1e-15)
    def update_learning_rate(self,iteration):self.events.append(('lr',iteration))
    def oneupSHdegree(self):self.active_sh_degree=min(3,self.active_sh_degree+1)
    def add_densification_stats(self,grad,visible):
        self.events.append(('stats',grad.clone(),visible.clone()));self.xyz_gradient_accum[visible]+=grad[visible,:2].norm(dim=-1,keepdim=True);self.denom[visible]+=1
    def densify(self,*args):self.events.append(('densify',args))
    def prune(self,*args):self.events.append(('prune',args))
    def grow(self,*args):self.events.append(('grow',args))
    def reset_opacity(self):self.events.append(('opacity_reset',))
    def compute_regulation(self,*args):self.regulations+=1;return self._deformation.grid.square()*.001


def bind_author_capture_restore():
    source=ROOT/next(e['path'] for e in read(fp.READINESS)['original_Wu']['sources'] if e['git_path']=='scene/gaussian_model.py')
    cls=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='GaussianModel')
    nodes=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in ('capture','restore')]
    scope={};exec(compile(ast.fix_missing_locations(ast.Module(body=copy.deepcopy(nodes),type_ignores=[])),str(source),'exec'),scope)
    Gaussian.capture=scope['capture'];Gaussian.restore=scope['restore']


class Raster:
    def __init__(self,raster_settings):self.settings=raster_settings
    def __call__(self,**kw):
        n=kw['means3D'].shape[0];yy,xx=torch.meshgrid(torch.linspace(0,1,self.settings.image_height),torch.linspace(0,1,self.settings.image_width),indexing='ij')
        xyz=kw['means3D'];screen=kw['means2D'];offset=float(self.settings.viewmatrix[3,0])
        v=xyz.mean()*.012+screen.sum()*.017+kw['opacities'].mean()*.02+kw['scales'].mean()*.01+kw['rotations'].mean()*.01+kw['shs'].mean()*.025
        image=torch.stack([.3+v+xx*(.11+xyz[:,2].mean()*.006)+yy*.03+i*.07+offset*.08 for i in range(3)])
        return image,torch.arange(n,dtype=torch.float32)+1,torch.ones_like(xx)*xyz[:,2].mean()


def camera(index,frame=100,size=32):
    w2c=torch.eye(4);w2c[0,3]=index*.12
    return SimpleNamespace(time=frame/300,image_width=size,image_height=size,FoVx=.9,FoVy=.9,
        world_view_transform=w2c.T.clone(),full_proj_transform=torch.eye(4),camera_center=torch.linalg.inv(w2c)[:3,3],
        intrinsics=np.array([[28.,0,(size-1)/2],[0,28.,(size-1)/2],[0,0,1]]),index=index)


def native_render(c,g,state,pipe,bg):return sr.native_rgb(c,g,state,pipe,bg,(SimpleNamespace,Raster))


def moment_render(c,xyz,cov,opacity):
    # Position-only auxiliary fixture. Tests below compare full six-edge X
    # against finite differences and explicitly retain target-depth gradients.
    z=xyz[:,2].mean()+.03*c.index
    alpha=torch.ones(c.image_height,c.image_width)*.9
    return dict(hr_moments=torch.stack((alpha,alpha*z,alpha*(z*z+.03))))


class Support:
    def observation(self,*key,device=None):return dict(valid_hr=torch.ones(32,32,dtype=torch.bool,device=device))
    def edge(self,*keys,device=None):return dict(mask_hr=torch.ones(32,32,dtype=torch.bool,device=device),weight_lr=torch.ones(8,8,device=device),frozen_valid_lr_fraction=1.)


def check_native_renderer():
    path=ROOT/next(e['path'] for e in read(fp.READINESS)['original_Wu']['sources'] if e['git_path']=='gaussian_renderer/__init__.py')
    render_node=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='render')
    # Original device literal must also be redirected; intercepting .cuda()
    # alone misses zeros_like(...device='cuda') in the author render body.
    for part in ast.walk(render_node):
        if isinstance(part,ast.Constant) and part.value=='cuda':part.value='cpu'
    import math
    scope=dict(torch=torch,math=math,GaussianRasterizationSettings=SimpleNamespace,GaussianRasterizer=Raster)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(render_node)],type_ignores=[])),str(path),'exec'),scope)
    author=scope['render'];pipe=SimpleNamespace(convert_SHs_python=False,compute_cov3D_python=False,debug=False);bg=torch.ones(3)
    torch.manual_seed(2026);g=Gaussian();g.training_setup(SimpleNamespace())
    torch.manual_seed(2026);reference=Gaussian();reference.training_setup(SimpleNamespace())
    state=sr.native_effective_state(g,1/3);actual=[native_render(camera(i),g,state,pipe,bg) for i in range(3)]
    with patch.object(torch.Tensor,'cuda',lambda value,*a,**k:value):expected=[author(camera(i),reference,pipe,bg,stage='fine',cam_type='dynerf') for i in range(3)]
    for a,b in zip(actual,expected):
        for key in ('render','radii','visibility_filter','depth'):assert torch.equal(a[key],b[key]),key
    sum(p['render'].square().mean() for p in actual).backward();sum(p['render'].square().mean() for p in expected).backward()
    for name,p in fp.parameter_map(g).items():assert torch.allclose(p.grad,fp.parameter_map(reference)[name].grad,atol=1e-8,rtol=2e-6),(name,p.grad,fp.parameter_map(reference)[name].grad)
    assert g._deformation.calls==1 and reference._deformation.calls==3
    with torch.no_grad():
        evaluation_state=sr.native_effective_state(g,1/3);evaluation=native_render(camera(0),g,evaluation_state,pipe,bg)
    assert not evaluation['render'].requires_grad and torch.equal(evaluation['render'],actual[0]['render'].detach())
    return dict(executable_original_author_render=entry(path),same_RGB_and_attributes=True,summed_parameter_gradients_match=True,
        shared_deformation_calls=1,author_separate_calls=3,no_grad_native_evaluation_supported=True,native_CUDA_acceptance=False)


def check_covariance():
    path=Path(os.environ.get('FOURDSR_UPSTREAM','/home/cai_tianshun/Project/4dgs'))/'utils/general_utils.py'
    nodes=[n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in ('build_rotation','build_scaling_rotation')]
    # Device literal substitution solely permits original math to run on CPU.
    for node in nodes:
        for part in ast.walk(node):
            if isinstance(part,ast.Constant) and part.value=='cuda':part.value='cpu'
    scope=dict(torch=torch);exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(path),'exec'),scope)
    g=Gaussian();state=sr.native_effective_state(g,.3);cov=sr.state_covariance(state);transform=scope['build_scaling_rotation'](state['scales'],state['rotation'])
    assert torch.allclose(cov,transform@transform.transpose(1,2),atol=1e-8,rtol=1e-6)
    assert torch.allclose(cov,cov.transpose(1,2));assert bool((torch.linalg.eigvalsh(cov)>0).all())
    return dict(bound_native_runtime_math_source=dict(path=str(path),sha256=sha(path)),same_positive_definite_covariance=True)


def check_x_path():
    import losses
    g=Gaussian();state=sr.native_effective_state(g,1/3);cams=[camera(i) for i in range(3)];pipe=SimpleNamespace(convert_SHs_python=False,compute_cov3D_python=False,debug=False)
    rgbs=[native_render(c,g,state,pipe,torch.ones(3))['render'] for c in cams];cov=sr.state_covariance(state);moments=[moment_render(c,state['xyz'],cov,state['opacity']) for c in cams]
    lrs=[torch.full((3,8,8),.35+i*.03) for i in range(3)];keys=[(f'cam{i+1:02d}',100) for i in range(3)]
    x,diag=losses.x_loss(rgbs,moments,cams,lrs,keys,Support())
    gradients=torch.autograd.grad(x,[state['xyz']]+rgbs,retain_graph=True,allow_unused=False)
    assert diag['registered_edges']==6 and diag['effective_edges']>0;assert all(bool(torch.isfinite(v).all()) for v in gradients)
    assert float(gradients[0].norm())>0 and all(float(v.norm())>0 for v in gradients[1:])
    # Separate target depth from RGB, proving both branches contribute.
    detached_source=[rgb.detach().requires_grad_(True) for rgb in rgbs]
    z=state['xyz'].detach().clone().requires_grad_(True);detached_mom=[moment_render(c,z,cov.detach(),state['opacity'].detach()) for c in cams]
    depth_x,_=losses.x_loss(detached_source,detached_mom,cams,lrs,keys,Support());analytic,=torch.autograd.grad(depth_x,z)
    direction=torch.zeros_like(z);direction[:,2]=1.;epsilon=1e-3
    def value(v):return losses.x_loss([r.detach() for r in rgbs],[moment_render(c,v,cov.detach(),state['opacity'].detach()) for c in cams],cams,lrs,keys,Support())[0]
    numerical=float((value(z.detach()+epsilon*direction)-value(z.detach()-epsilon*direction))/(2*epsilon));predicted=float((analytic*direction).sum())
    assert abs(numerical-predicted)<max(3e-6,abs(predicted)*.02),(numerical,predicted)
    assert abs(predicted)>1e-8
    return dict(six_edges=True,source_RGB_gradients_all_present=True,target_depth_gradient_present=True,FD=numerical,autograd=predicted)


def check_updates(directory):
    config,_=fp.author_configuration('cook_spinach');opt=SimpleNamespace(**config['OptimizationParams']);h=SimpleNamespace(**config['ModelHiddenParams']);pipe=SimpleNamespace(**config['PipelineParams'])
    results=[]
    for method in sr.METHODS:
        torch.manual_seed(77);g=Gaussian();g.training_setup(opt);g.optimizer.zero_grad(set_to_none=True)
        # Real Adam prefix establishes nonzero moments/step counters, which
        # must survive the SR boundary without a training_setup reset.
        sum(p.square().mean() for p in fp.parameter_map(g).values()).backward();g.optimizer.step()
        original_steps={name:float(g.optimizer.state[p]['step']) for name,p in fp.parameter_map(g).items()}
        parent_optimizer_id=id(g.optimizer);calls=[];original_step=g.optimizer.step
        def step():calls.append('Adam');original_step()
        g.optimizer.step=step
        def density(*args,**kwargs):calls.append('author_topology');return actual_density(*args,**kwargs)
        actual_density=fp.author_topology
        cams=[camera(i,100 if sr.ARMS[method]['schedule']=='rows' else 80+i*20) for i in range(3)]
        keys=[(f'cam{i+1:02d}',100 if sr.ARMS[method]['schedule']=='rows' else 80+i*20) for i in range(3)]
        with patch.object(fp,'author_topology',density):
            result=sr.refinement_iteration(g,h,opt,pipe,torch.ones(3),cams,[torch.full((3,8,8),.38+i*.03) for i in range(3)],
                [torch.full((3,32,32),.44),torch.full((3,32,32),.46)],keys,sr.ARMS[method],1,2.5,directory,torch,
                dict(lambda_X=.1,kappa=2.),Support(),native_render,moment_render,diagnostics=True)
        assert calls==['author_topology','Adam'];assert result['RGB']==3 and result['moments']==(3 if sr.ARMS[method]['X'] else 0)
        assert g.setup_calls==1 and id(g.optimizer)==parent_optimizer_id and g.regulations==1
        assert g._deformation.calls==(1 if sr.ARMS[method]['schedule']=='rows' else 3)
        assert result['LR_clock']==14001 and result['topology_clock']==1
        for name,p in fp.parameter_map(g).items():assert float(g.optimizer.state[p]['step'])==original_steps[name]+1
        assert all(p.grad is None for p in fp.parameter_map(g).values())
        assert result['gdiag'] is not None and len(result['native_audit']['optimizer_groups'])==8
        results.append(dict(method=method,RGB=3,moments=result['moments'],one_Adam=True,parent_moments_inherited=True,regularizer_once=True,topology_before_Adam=True))
    return results


def check_roundtrip(directory):
    config,_=fp.author_configuration('cook_spinach');opt=SimpleNamespace(**config['OptimizationParams']);h=SimpleNamespace(**config['ModelHiddenParams'])
    torch.manual_seed(928);np.random.seed(82);random.seed(77);g=Gaussian();g.training_setup(opt)
    loss=sum(p.square().mean() for p in fp.parameter_map(g).values());loss.backward();g.optimizer.step();g.optimizer.zero_grad(set_to_none=True)
    g._deformation_accum.fill_(.27);g.max_radii2D.fill_(7);g.xyz_gradient_accum.fill_(.5);g.denom.fill_(3)
    boundary=sr.initialize_SR_statistics(g,0,torch,np)
    assert all(bool((getattr(g,name)==0).all()) for name in ('max_radii2D','xyz_gradient_accum','denom'))
    for cursor in (0,100):
        try:sr.initialize_SR_statistics(g,cursor,torch,np)
        except ValueError:pass
        else:raise AssertionError('Density boundary applied twice/on mid-SR state')
    # Subsequent SR statistics are saved and must be inherited on mid-resume.
    g.max_radii2D.fill_(13);g.xyz_gradient_accum.fill_(.76);g.denom.fill_(4)
    plan=dict(parent={'a':1},dependencies=dict(schedule={'path':'fixture','sha256':'abc'}));hardware={'CPU_fixture':True}
    journal=sr.Journal(directory,'fixture',100);path=directory/'checkpoint_00100.pt'
    sr.save_checkpoint(path,g,h,opt,plan,100,{'old_LR':'sampler'},torch,np,journal,hardware)
    payload=sr.checkpoint_payload(path,torch,plan,sr.SCHEMA,'cpu');torch.manual_seed(928);restored=Gaussian()
    audit=sr.restore_native_model(restored,payload,opt,torch,np,plan,sr.SCHEMA)
    assert sr.tree_digest(g.capture())==sr.tree_digest(restored.capture()) and torch.equal(g._deformation_accum,restored._deformation_accum)
    assert payload['parent_LR_sampler']=={'old_LR':'sampler'} and payload['sampler']['cursor']==100
    before=sr.tree_digest(restored.capture());rng_before=sr.tree_digest(fp.rng_state(torch,np))
    # A diagnostic autograd call must not update params/Adam, density buffers,
    # saved parameter gradients or RNG.
    state=sr.native_effective_state(restored,.3);v=state['xyz'].square().mean();torch.autograd.grad(v,state['xyz'])
    assert sr.tree_digest(restored.capture())==before and sr.tree_digest(fp.rng_state(torch,np))==rng_before
    wrong=dict(payload,schema='short_window_custom_checkpoint')
    try:sr.restore_native_model(Gaussian(),wrong,opt,torch,np,plan,sr.SCHEMA)
    except ValueError:pass
    else:raise AssertionError('Short-window checkpoint accepted')
    try:sr.save_checkpoint(path,g,h,opt,plan,100,{},torch,np,journal,hardware)
    except FileExistsError:pass
    else:raise AssertionError('Committed checkpoint overwritten')
    assert float(restored.max_radii2D[0])==13 and float(restored.denom[0])==4
    try:sr.initialize_SR_statistics(restored,0,torch,np)
    except ValueError:pass
    else:raise AssertionError('Restored SR checkpoint boundary applied again')
    return dict(**audit,density_boundary=boundary,mid_resume_statistics_retained=True,boundary_applied_once=True,
        duplicate_checkpoint_preserved=True,wrong_custom_schema_rejected=True,diagnostics_no_updates_or_RNG_draws=True,checkpoint=entry(path))


def check_diagnostic_statistics(directory):
    config,_=fp.author_configuration('cook_spinach');opt=SimpleNamespace(**config['OptimizationParams']);h=SimpleNamespace(**config['ModelHiddenParams']);pipe=SimpleNamespace(**config['PipelineParams'])
    torch.manual_seed(742);left=Gaussian();left.training_setup(opt);right=copy.deepcopy(left)
    cameras=[camera(i) for i in range(3)];keys=[(f'cam{i+1:02d}',100) for i in range(3)]
    results=[]
    for g,diag in ((left,False),(right,True)):
        result=sr.refinement_iteration(g,h,opt,pipe,torch.ones(3),cameras,[torch.full((3,8,8),.35+i*.02) for i in range(3)],
            [torch.full((3,32,32),.4),torch.full((3,32,32),.45)],keys,sr.ARMS['MX'],1,2.5,directory,torch,
            dict(lambda_X=.1,kappa=2.),Support(),native_render,moment_render,diagnostics=diag)
        results.append(result)
    assert sr.tree_digest(left.capture())==sr.tree_digest(right.capture()),'Diagnostic autograd contaminated formal Adam or author density statistics'
    assert torch.equal(left.xyz_gradient_accum,right.xyz_gradient_accum) and torch.equal(left.max_radii2D,right.max_radii2D)
    assert results[0]['summed_viewspace_gradient_RMS']==results[1]['summed_viewspace_gradient_RMS']
    # Opposing view gradients cancel by Gaussian index before the norm. Max
    # radii and visibility are independent unions, not the final view values.
    packets=[]
    for radii,visible,grad in (([2.,1.],[True,False],[[2.,0.,0.],[0.,1.,0.]]),([1.,7.],[False,True],[[-2.,0.,0.],[0.,-1.,0.]]),([4.,3.],[True,False],[[0.,0.,0.],[0.,0.,0.]])):
        value=torch.zeros(2,3,requires_grad=True);value.grad=torch.tensor(grad)
        packets.append(dict(radii=torch.tensor(radii),visibility_filter=torch.tensor(visible),viewspace_points=value))
    radius,visible,gradient=fp.combine_batch_statistics(packets,torch)
    assert torch.equal(radius,torch.tensor([4.,7.])) and bool(visible.all()) and bool((gradient==0).all())
    return dict(gdiag_does_not_change_formal_density_or_Adam=True,same_index_SUM_before_norm=True,max_radii=True,OR_visibility=True)


def check_calibration_rules():
    rows=[dict(E_anchor_L1_RGB_RMS=2.,E_anchor_MSE_RGB_RMS=1.,LR_effective_xyz_RMS=4.,X_effective_xyz_RMS=1.,effective_edges=6) for _ in range(80)]
    result=sr.calibration_coefficients(rows,True)
    assert abs(result['kappa']-2)<1e-10 and abs(result['lambda_X']-.4)<1e-10
    empty=dict(rows[0],effective_edges=0);weak=dict(rows[0],X_effective_xyz_RMS=1e-8)
    selected=sr.calibration_coefficients(rows+[empty,weak],True)
    assert selected['supported_X_batches']==80 and selected['excluded_X_batches']==2
    for bad in ([empty], [dict(rows[0],LR_effective_xyz_RMS=1.,X_effective_xyz_RMS=1e-5)], [dict(rows[0],E_anchor_MSE_RGB_RMS=0.)]):
        try:sr.calibration_coefficients(bad,True)
        except ValueError:pass
        else:raise AssertionError('Empty/negligible/giant calibration signal accepted')
    e=sr.calibration_coefficients([dict(rows[0],LR_effective_xyz_RMS=None,X_effective_xyz_RMS=None,effective_edges=0)],False)
    assert e['lambda_X'] is None and e['kappa']>0
    return dict(training_gradient_medians_only=True,balanced_fixture_rows=80,no_HR_or_score_input=True,empty_or_weak_excluded=True,
                giant_coefficient_rejected=True,E_only_needs_no_Xcache=True,expected_lambda_X=.4)


def check_schedule_and_pending():
    manifest=ROOT/'data/dynamic_sr/full_time_20261007/prepared/cook_spinach/manifest_train_ready_seed20261007.json';m=read(manifest)
    keys=sr.keys_of(m);by={key:i for i,key in enumerate(keys)};cams=m['splits']['train'];rows=[]
    for cam,frame in keys:
        ci=cams.index(cam);rows.append([by[cams[(ci+j)%len(cams)],frame] for j in range(3)])
    triplets=[[by[cams[(ci+j)%len(cams)],frame] for j in range(3)] for ci in range(len(cams)) for frame in (0,100,200,299)]
    context={'manifest':entry(manifest),'parent':{'path':'CPUfixture','sha256':'not_actual_result'}}
    table=dict(schema=sr.SCHEDULE_SCHEMA,identity=context,seed=20261007,updates=6000,record_keys=[list(k) for k in keys],
               audit={'passed':True},rows=rows,random_rows=copy.deepcopy(rows),calibration_triplets=triplets)
    sr.validate_schedule(table,m,context,20261007)
    bad=copy.deepcopy(table);bad['rows'][0]=[0,0,0]
    try:sr.validate_schedule(bad,m,context,20261007)
    except ValueError:pass
    else:raise AssertionError('Invalid same-time triple passed')
    bad=copy.deepcopy(table);bad['random_rows'][0][1]=bad['random_rows'][1][1]
    try:sr.validate_schedule(bad,m,context,20261007)
    except ValueError:pass
    else:raise AssertionError('Unequal SR exposure passed')
    args=SimpleNamespace(manifest=manifest,seed=20261007,method='MX',steps=6000,topology_clock='refinement',
        parent=ROOT/'missing-full-native-final-checkpoint.pt',parent_complete=ROOT/'missing-full-native-complete.json',
        teacher=ROOT/'missing-full-teacher.json',schedule=ROOT/'missing-full-schedule.json',selection=None,support_index=None,calibration=None,upstream=None)
    plan=sr.build_plan(args)
    assert plan['status']=='pending_full_native_refinement_dependencies' and plan['planned_cost']['moment_forwards']==18000
    try:sr.train(SimpleNamespace(),plan)
    except ValueError as exc:assert 'Pending' in str(exc)
    else:raise AssertionError('Pending plan tried training')
    try:sr.validate_support(dict(status='completed_frozen_X_support',soft_weights_frozen=True),context,table,m)
    except ValueError:pass
    else:raise AssertionError('Short U6000 support accepted')
    return dict(full_manifest=entry(manifest),synthetic_schedule_test_only=True,anchors=len(triplets),rows=6000,
        unequal_exposure_rejected=True,duplicate_camera_rejected=True,short_support_rejected=True,
        pending_dependencies=plan['missing'],pending_train_rejected_before_CUDA=True)


def main():
    os.environ['CUDA_VISIBLE_DEVICES']=''
    assert not torch.cuda.is_initialized(),'CPU checks must not initialize CUDA'
    directory=OUT/'operator_checks/full_refine'/str(time.time_ns());directory.mkdir(parents=True)
    started=time.monotonic();bind_author_capture_restore();torch.set_num_threads(2)
    result=dict(status='running_CPU_native_full_refinement_contracts',source=entry(HERE/'full_refine.py'),checker=entry(Path(__file__)),
        GPU_forwards=0,CUDA_initialized=torch.cuda.is_initialized(),formal_updates=0,planned_fixture_Adam_calls=15,
        full_native_CUDA_acceptance=False,full_data_training_executed=False,complete_benchmark_claimed=False,CPU_seconds=time.monotonic()-started)
    write(directory/'receipt.json',result)
    tests=[('native_renderer',check_native_renderer),('covariance',check_covariance),('full_X',check_x_path),
           ('updates',lambda:check_updates(directory)),('exact_native_roundtrip',lambda:check_roundtrip(directory)),
           ('gdiag_formal_statistics',lambda:check_diagnostic_statistics(directory)),('calibration_rules',check_calibration_rules),
           ('full_schedule_and_pending',check_schedule_and_pending),
           ('original_topology',lambda:old_checks.check_topology(old_checks.author_fragments()[1],SimpleNamespace(**fp.author_configuration('cook_spinach')[0]['OptimizationParams']),directory))]
    try:
        for name,call in tests:
            result['active_check']=name;write(directory/'receipt.json',result);result[name]=call()
            assert not torch.cuda.is_initialized(),'CPU fixture initialized CUDA'
            result.update(CUDA_initialized=False,CPU_seconds=time.monotonic()-started);write(directory/'receipt.json',result)
        result.update(status='passed_CPU_native_full_refinement_contracts',active_check=None,fixture_Adam_calls=15)
    except BaseException as exc:
        result.update(status='failed_CPU_contract_preserved',error=repr(exc),CPU_seconds=time.monotonic()-started,CUDA_initialized=torch.cuda.is_initialized())
        write(directory/'receipt.json',result);raise
    write(directory/'receipt.json',result);print(json.dumps(dict(status=result['status'],receipt=entry(directory/'receipt.json'),CPU_seconds=result['CPU_seconds']),ensure_ascii=False))


if __name__=='__main__':main()
