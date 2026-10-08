"""Private real-parent native CUDA compatibility acceptance; never formal SR.

CPU plan/contracts import no tensor or model. CUDA execution must be dispatched
by root under a UUID lock. A passed receipt covers only its accepted real grids.
No selected evidence, teacher, HR GT, held-out pixel or Adam update is produced.
"""
from __future__ import annotations
import argparse, ast, copy, fcntl, hashlib, json, math, os, socket, sys, time, traceback
from pathlib import Path
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[2]
HERE=ROOT/'experiments/dynamic_sr_multiview_footprint_20261007'
BASE=ROOT/'output/dynamic_sr_multiview_footprint_20261007/operator_checks/held_confirmation_native_CUDA_acceptance'
sys.path.insert(0,str(HERE))
from fp_common import local,read,write,sha,entry,bound,module
import full_refine_registered_v2 as native
import full_prefix_registered as prefix
import full_native_protocol as domain
import full_held_confirmation_contract_v2 as held

SOURCE_SHA='c278928f56b0515adedf15eef9a30850e8bd2a4c41e6f6e0f828068ca774b75d'
PIN='843d5ac636c37e4b611242287754f3d4ed150144'
PARENT_SCHEMA='full_author_LR_prefix_checkpoint_v1'
PASSED='passed_native_full_refinement_CUDA_acceptance'
PLAN_SCHEMA='private_real_parent_native_CUDA_acceptance_plan_v1'
FRAME=100
GROUPS=('xyz','deformation','grid','f_dc','f_rest','opacity','scaling','rotation')
GRIDS=((336,252),(320,180))
CHECKS=('native14_oneAdam8_restore','parent_RNG_inherited','SR_boundary_only_density_buffers',
 'RGB_finite_trainable_attributes','raw_HR_moment_finite_position_only_gradient','raw_moments_area_before_normalize',
 'full_D0_parity','six_actual_projection_mipmap_edges_finite','current_source_RGB_and_target_z_gradients',
 'RGB_screen_SUM_max_radii_OR_visibility','same_topology_author_density','SR_capture_sidecar_restore',
 'mid_resume_density_boundary_refused','parameter_versions_no_update','diagnostic_parameters_Adam_gradients_RNG_unchanged',
 'parent_model_Adam_buffers_gradients_RNG_restored')

def require(ok,message):
    if not ok:raise ValueError(message)

def case_metadata(manifest,parent,plan):
    scene=manifest['scene'];descriptor=domain.descriptor(scene)
    held.held_parent_scope(manifest,parent,plan['seed'])
    require(scene in held.HELD and plan['seed']==20261007,'Only the exact two original held seed07 parents are in this scope')
    require(scene==plan['scene'] and manifest['initialization']['seed']==plan['seed'],'Parent and grid must use the same scene/seed')
    require(parent['registered_domain_contract']['scene']==scene,'Cross-domain parent refused')
    require(manifest['resolutions']['lr']==descriptor['native_LR'] and manifest['resolutions']['hr']==descriptor['native_HR'],'Registered native grid differs')
    require(tuple(descriptor['native_LR']) in GRIDS,'Unregistered CUDA acceptance grid')
    require(tuple(x['name'] for x in parent['audit']['optimizer_groups'])==GROUPS,'Exactly one native eight-group Adam required')
    domain.validate_parent_state(parent['state'],plan['configuration'])
    cams=sorted(manifest['splits']['train'])[:3]
    require(len(cams)==3 and 'cam00' not in cams and not manifest['splits']['dev'],'Three legal full training cameras required')
    rows={(r['camera_id'],int(r['frame_index'])):r for r in manifest['observations'] if r['split']=='train'}
    keys=[(c,FRAME) for c in cams]
    require(all(k in rows for k in keys),'Fixed train frame100 missing')
    return dict(scene=scene,seed=plan['seed'],LR=descriptor['native_LR'],HR=descriptor['native_HR'],
        keys=[list(k) for k in keys],expected_parent_state=domain.expected_parent_state(plan['configuration']))

def validate_result(result):
    require(result.get('status')=='accepted_actual_native_CUDA_same_domain_parent','Actual CUDA case status required')
    require(result.get('execution')=='actual_native_CUDA' and result.get('synthetic') is False,'Synthetic or CPU result cannot enter production acceptance')
    require(result.get('parent_is_same_domain') is True and result.get('require_real_grid_parents') is True,'A real same-domain parent is required')
    require(tuple(result['LR']) in GRIDS and result['HR']==[4*x for x in result['LR']],'Unregistered accepted grid')
    require(result['scene']==result['parent_scene'] and result['seed']==result['parent_seed'],'Cross-domain/seed acceptance refused')
    require(result.get('Adam_calls')==0 and result.get('formal_updates')==0 and result.get('HR_GT_or_heldout_pixel_reads')==0,'Acceptance information/update boundary violated')
    require(result.get('RGB_forwards')==3 and result.get('moment_forwards')==3,'Actual three RGB/raw moment calls required')
    require(set(result.get('checks',{}))==set(CHECKS) and all(result['checks'][k] is True for k in CHECKS),'All real CUDA checks must return true')
    require(result['restore_audit'].get('one_Adam_exact') and result['restore_audit'].get('global_RNG_exact'),'Exact native restore proof missing')
    require(result['parent_after']==result['parent_before'],'Real parent fork did not restore exactly')
    require(result['hardware'].get('gpu_uuid','').startswith('GPU-') and result['hardware'].get('CUDA_initialized') is True,'Actual CUDA hardware proof missing')
    return True

def validate_receipt(value):
    require(value.get('status')==PASSED and value.get('source_sha256')==SOURCE_SHA and value.get('author_commit')==PIN,'Foreign CUDA acceptance source/pin/status')
    require(value.get('helper_source')==entry(Path(__file__)),'Acceptance helper source differs')
    require(value.get('observer_RNG_restored') is True and value.get('Adam_calls')==0 and value.get('formal_updates')==0,'Outer RNG/update acceptance missing')
    require(value.get('HR_GT_or_heldout_pixel_reads')==0 and value.get('selection_created') is False,'Acceptance must not synthesize selected evidence or consume held-out pixels')
    results=value.get('results',[]);require(bool(results),'Empty acceptance refused')
    for result in results:
        validate_result(result)
        require(result['scene'] in held.HELD and result['seed']==20261007 and result['LR']==[336,252],'Only exact held seed07 native cases can enter this receipt')
        expected=next(r for r in held.original_inputs()['entries'] if r['scene']==result['scene'])
        require(held.same(result['parent']['checkpoint'],expected['checkpoint']),'Cross-held-parent native receipt refused')
    expected=sorted({tuple(r['LR']) for r in results})
    require(value['accepted_grids']==[list(k) for k in expected],'Accepted grids differ from actual case evidence')
    require(value['parents']==[r['parent'] for r in results],'Accepted real parent bindings differ')
    return True

def build_plan(a):
    confirmation_authorization=entry(local(a.confirmation_authorization));held.validate_authorization(confirmation_authorization)
    require(sha(HERE/'full_refine_registered_v2.py')==SOURCE_SHA and prefix.PIN==PIN,'Frozen registered native source/pin differs')
    upstream=local(a.upstream);requests=[];missing=[]
    if a.manifest and a.parent and a.parent_complete:
        requests.append(dict(manifest=str(a.manifest),parent=str(a.parent),parent_complete=str(a.parent_complete)))
    elif any((a.manifest,a.parent,a.parent_complete)):
        missing.append(dict(dependency='primary manifest + real parent + complete triplet'))
    for path in a.case:
        request=read(local(path));require(request.get('schema')=='private_real_parent_native_CUDA_acceptance_case_v1','Foreign case request')
        requests.append({k:bound(request[k]) if isinstance(request[k],dict) else local(request[k]) for k in ('manifest','parent','parent_complete')})
    require(requests or missing,'At least one explicit real-parent case required')
    cases=[]
    for request in requests:
        absent=[k for k,p in request.items() if not local(p).is_file()]
        if absent:missing.append(dict(dependency='complete same-domain real parent case',request={k:str(v) for k,v in request.items()},absent=absent));continue
        manifest_path=local(request['manifest']);m=read(manifest_path);seed=m['initialization']['seed']
        manifest,data,_=domain.validate_manifest(manifest_path,seed,local(a.readiness))
        parent,p=native.full_parent(request['parent'],request['parent_complete'],data['manifest'],seed)
        require(native.source_files(upstream)['upstream']==p['runtime_source_sha256'],'Actual selected runtime differs from the real parent pinned source tree')
        metadata=case_metadata(manifest,parent,p)
        rows={(r['camera_id'],int(r['frame_index'])):r for r in manifest['observations'] if r['split']=='train'}
        inputs=[]
        for c,f in metadata['keys']:
            row=rows[(c,f)];path=manifest_path.parent/row['lr_path']
            require(not path.is_symlink() and path.resolve().is_relative_to(manifest_path.parent.resolve()),'Legal LR path escaped manifest')
            require(sha(path)==row['lr_sha256'],'Fixed legal LR bytes differ')
            inputs.append(entry(path))
        cases.append(dict(**metadata,manifest=data['manifest'],parent=parent,parent_plan=p,LR_inputs=inputs))
    require(len({(c['scene'],c['seed']) for c in cases})==len(cases),'Duplicate real parent case refused')
    prior=[]
    for path in a.include_receipt:
        identity=entry(local(path));value=read(bound(identity));validate_receipt(value)
        require(value['source_files']==native.source_files(upstream),'Prior accepted dependency tree differs')
        for r in value['results']:
            for key in ('checkpoint','complete','plan','sidecar'):bound(r['parent'][key])
        prior.append(dict(identity=identity,value=value))
    prior_keys={(r['scene'],r['seed']) for v in prior for r in v['value']['results']}
    require(not prior_keys.intersection((c['scene'],c['seed']) for c in cases),'Already accepted same parent must not be rendered again; omit the case and include receipt')
    return dict(schema=PLAN_SCHEMA,status='pending_real_parent_CUDA_acceptance_dependencies' if missing else 'registered_real_parent_CUDA_acceptance_ready',
        missing=missing,cases=cases,prior_receipts=prior,helper_source=entry(Path(__file__)),source_sha256=SOURCE_SHA,
        source_files=native.source_files(upstream),author_commit=PIN,require_real_grid_parents=True,
        synthetic_results_allowed_in_passed_receipt=False,planned_RGB=3*len(cases),planned_moments=3*len(cases),
        Adam_calls=0,formal_updates=0,selection_created=False,HR_GT_or_heldout_pixels_used=False,
        parent_checkpoint_files_modified=False,production_training_gate_bypassed=False,
        confirmation_authorization=confirmation_authorization,scope='Only exact registered held seed07 parents after source-bound method freeze and both actual teacher acceptances; no teacher pixels, formal6000 meter, quality or full benchmark.')

def snapshot(g,torch,np):
    return dict(model=native.tree_digest(g.capture()),Adam=native.tree_digest(g.optimizer.state_dict()),
        deformation_accum=native.tree_digest(g._deformation_accum),
        gradients=native.tree_digest({k:p.grad for k,p in prefix.parameter_map(g).items()}),
        RNG=native.tree_digest(prefix.rng_state(torch,np)))

def finite_summary(tensor,torch):
    finite=bool(torch.isfinite(tensor).all())
    return dict(shape=list(tensor.shape),finite=finite,norm=float(tensor.detach().double().norm()) if finite else None,
        nonfinite=int((~torch.isfinite(tensor.detach())).sum()))

def tensor_proof(tensor,torch,label,nonzero=False):
    result=finite_summary(tensor,torch);require(result['finite'],label+' is nonfinite')
    if nonzero:require(result['norm']>0,label+' is zero; empty visibility is not acceptance')
    return result

def actual_case(case,a,hardware,counters):
    """Only this function may return an accepted actual CUDA case."""
    import numpy as np
    import torch
    import torch.nn.functional as F
    sys.path.insert(0,str(local(a.upstream)));sys.path.insert(1,str(prefix.LEGACY.parent))
    from scene.gaussian_model import GaussianModel
    from n3dv_data import load_manifest,N3DVPreparedDataset
    import footprint,losses
    gm=module('registered_native_acceptance_moments_'+str(time.time_ns()),ROOT/'experiments/dynamic_sr_confidence_geometry_20261006/depth_prior.py')
    manifest=load_manifest(bound(case['manifest']));plan=case['parent_plan'];config=plan['configuration']
    h=SimpleNamespace(**config['ModelHiddenParams']);opt=SimpleNamespace(**config['OptimizationParams']);pipe=SimpleNamespace(**config['PipelineParams'])
    directory=a.out/(case['scene']+'_seed'+str(case['seed']));directory.mkdir()
    started=time.monotonic();counter_start=dict(counters);g=None;payload=None;packets=None;moments=None;state=None
    checks={k:False for k in CHECKS};detail={};parent_before=None;parent_after=None
    def fresh():
        value=GaussianModel(config['ModelParams']['sh_degree'],h);value._deformation=value._deformation.cuda();return value
    try:
        payload=native.checkpoint_payload(bound(case['parent']['checkpoint']),torch,plan,PARENT_SCHEMA,'cuda')
        require(len(payload['model'])==14,'Parent is not native14tuple')
        payload_digest=native.tree_digest(payload)
        g=fresh();audit=native.restore_native_model(g,copy.deepcopy(payload),opt,torch,np,plan,PARENT_SCHEMA)
        parent_before=snapshot(g,torch,np);versions={k:p._version for k,p in prefix.parameter_map(g).items()}
        invariant_before=dict(parameters=native.tree_digest(prefix.parameter_map(g)),Adam=parent_before['Adam'],
            gradients=parent_before['gradients'],RNG=parent_before['RNG'],deformation_accum=parent_before['deformation_accum'])
        require(len(g.capture())==14 and tuple(gr['name'] for gr in g.optimizer.param_groups)==GROUPS,'Native14tuple/oneAdam8 mapping differs')
        checks['native14_oneAdam8_restore']=True;checks['parent_RNG_inherited']=parent_before['RNG']==native.tree_digest(payload['rng'])
        require(checks['parent_RNG_inherited'],'Mapped CUDA parent RNG not inherited')
        boundary=native.initialize_SR_statistics(g,0,torch,np);detail['density_boundary']=boundary
        checks['SR_boundary_only_density_buffers']=True
        dataset=N3DVPreparedDataset(manifest,'train','lr',device='cpu',cache=False)
        indices={(r['camera_id'],int(r['frame_index'])):i for i,r in enumerate(dataset.observations)}
        obs=[dataset[indices[tuple(key)]] for key in case['keys']]
        cameras=[native.hr_camera(o,i,manifest) for i,o in enumerate(obs)]
        lrs=[o['image'].cuda() for o in obs];lr_hw=tuple(reversed(case['LR']));hr_hw=tuple(reversed(case['HR']))
        require(all(tuple(lr.shape)==(3,*lr_hw) for lr in lrs),'Actual LR loader grid differs')
        require(all((c.image_width,c.image_height)==tuple(case['HR']) for c in cameras),'Native HR camera grid differs')
        background=torch.tensor([1.,1.,1.],device='cuda')
        state=native.native_effective_state(g,cameras[0].time,torch)
        packets=[]
        for camera in cameras:
            counters['RGB_attempted']+=1;packet=native.native_rgb(camera,g,state,pipe,background);packets.append(packet);counters['RGB_completed']+=1
        rgbs=[p['render'] for p in packets]
        for rgb in rgbs:require(tuple(rgb.shape)==(3,*hr_hw),'Native RGB is not complete registered HR frame');tensor_proof(rgb,torch,'native RGB')
        require(all(int(p['visibility_filter'].sum())>0 for p in packets),'A real parent camera is empty')
        cov=native.state_covariance(state,torch);moments=[]
        for camera in cameras:
            counters['moment_attempted']+=1;m=gm.render_moments(camera,state['xyz'],cov,state['opacity']);moments.append(m);counters['moment_completed']+=1
        for m in moments:require(tuple(m['hr_moments'].shape)==(3,*hr_hw),'Raw moment HR grid differs');tensor_proof(m['hr_moments'],torch,'raw moment')
        targets=[gm.normalize_moments(m['hr_moments']) for m in moments]
        rgb_loss=sum(losses.lr_loss(rgb,lr) for rgb,lr in zip(rgbs,lrs))/3
        attr_keys=('xyz','scales','rotation','opacity','sh')
        counters['diagnostic_autograd_attempted']+=1
        grads=torch.autograd.grad(rgb_loss,[state[k] for k in attr_keys],retain_graph=True,allow_unused=False)
        counters['diagnostic_autograd_completed']+=1
        detail['RGB_attribute_gradients']={k:tensor_proof(grad,torch,'RGB '+k,nonzero=k=='xyz') for k,grad in zip(attr_keys,grads)}
        checks['RGB_finite_trainable_attributes']=True
        moment_loss=sum(m['hr_moments'][1].mean() for m in moments)/3
        counters['diagnostic_autograd_attempted']+=1
        mgrads=torch.autograd.grad(moment_loss,[state[k] for k in attr_keys],retain_graph=True,allow_unused=True)
        counters['diagnostic_autograd_completed']+=1
        detail['raw_moment_attribute_gradients']={k:None if grad is None else tensor_proof(grad,torch,'moment '+k,nonzero=k=='xyz') for k,grad in zip(attr_keys,mgrads)}
        require(mgrads[0] is not None and all(v is None for v in mgrads[1:]),'Moment covariance/opacity/SH gradient was not detached')
        checks['raw_HR_moment_finite_position_only_gradient']=True
        pooled=[]
        for m in moments:
            z=gm.normalize_moments(m['hr_moments'],lr_hw);raw=F.interpolate(m['hr_moments'][None],size=lr_hw,mode='area')[0]
            require(torch.equal(z['moments'],raw) and torch.equal(z['expected_z'],raw[1]/raw[0].clamp_min(1e-6)),'Raw moments must pool before normalize')
            pooled.append(finite_summary(z['moments'],torch))
        detail['pooled_raw_moments']=pooled;checks['raw_moments_area_before_normalize']=True
        for rgb in rgbs:
            prediction,_=footprint.downsample(rgb,lr_hw);require(torch.equal(prediction,losses.degradation(rgb,lr_hw)),'Current full-frame D0 differs from unchanged loss degradation')
        checks['full_D0_parity']=True
        edge_details=[];live_edges=0
        for s,t in losses.DIRECTED_EDGES:
            z=targets[t]['expected_z'];valid=(targets[s]['alpha'].detach()>=1e-3)
            prediction=footprint.edge_prediction(rgbs[s],z,cameras[s],cameras[t],lr_hw,source_valid_hr=valid)
            tensor_proof(prediction['prediction'],torch,'edge full D0 prediction')
            require(tuple(prediction['prediction'].shape)==(3,*lr_hw),'Projection/mipmap/full D0 output grid differs')
            count=int(prediction['valid_lr'].sum());row=dict(source=s,target=t,valid_lr=count,diagnostics=losses.plain(prediction['diagnostics']))
            if count:
                term=((prediction['prediction']-lrs[t]).square()*prediction['valid_lr'][None]).sum()/(3*count)
                counters['diagnostic_autograd_attempted']+=1
                dg=torch.autograd.grad(term,[rgbs[s],z],retain_graph=True,allow_unused=False)
                counters['diagnostic_autograd_completed']+=1
                row['source_RGB_gradient']=tensor_proof(dg[0],torch,'edge current RGB')
                row['target_z_gradient']=tensor_proof(dg[1],torch,'edge current target z')
                if row['source_RGB_gradient']['norm']>0 and row['target_z_gradient']['norm']>0:live_edges+=1
            edge_details.append(row)
        require(len(edge_details)==6,'Six directed projection edges missing')
        require(live_edges>0,'No current source RGB/current target z gradient: empty geometry cannot pass')
        detail['projection_mipmap_fullD0_edges']=edge_details;detail['live_gradient_edges']=live_edges
        checks['six_actual_projection_mipmap_edges_finite']=True;checks['current_source_RGB_and_target_z_gradients']=True
        # All retained intermediate RGB screen grads are explicitly cleared.
        # This autograd call alone supplies diagnostic density; no Adam/backward.
        for packet in packets:packet['viewspace_points'].grad=None
        counters['diagnostic_autograd_attempted']+=1
        screen_grads=torch.autograd.grad(rgb_loss,[p['viewspace_points'] for p in packets],retain_graph=True,allow_unused=False)
        counters['diagnostic_autograd_completed']+=1
        for packet,grad in zip(packets,screen_grads):packet['viewspace_points'].grad=grad.detach().clone();tensor_proof(grad,torch,'native RGB screen grad')
        radii,visible,summed=prefix.combine_batch_statistics(packets,torch)
        require(torch.equal(radii,torch.stack([p['radii'] for p in packets]).max(0).values),'Radii max differs')
        require(torch.equal(visible,torch.stack([p['visibility_filter'] for p in packets]).any(0)),'Visibility OR differs')
        require(torch.equal(summed,torch.stack(screen_grads).sum(0)),'Same-index RGB screen SUM differs')
        checks['RGB_screen_SUM_max_radii_OR_visibility']=True
        before_density=dict(radii=g.max_radii2D.clone(),accum=g.xyz_gradient_accum.clone(),denom=g.denom.clone())
        # Cursor1 cannot densify/prune/grow/reset with the exact registered recipe.
        require(opt.densify_from_iter>1 and opt.pruning_from_iter>1 and opt.densification_interval>1 and opt.opacity_reset_interval>1,'Unsafe diagnostic topology cursor')
        with torch.no_grad():prefix.author_topology(g,opt,'fine',1,plan['data']['cameras_extent'],directory,(radii,visible,summed))
        expected_radii=before_density['radii'].clone();expected_radii[visible]=torch.maximum(expected_radii[visible],radii[visible])
        expected_accum=before_density['accum'].clone();expected_accum[visible]+=summed[visible,:2].norm(dim=-1,keepdim=True)
        expected_denom=before_density['denom'].clone();expected_denom[visible]+=1
        require(torch.equal(g.max_radii2D,expected_radii) and torch.equal(g.xyz_gradient_accum,expected_accum) and torch.equal(g.denom,expected_denom),'Actual author same-topology density differs')
        checks['same_topology_author_density']=True
        require({k:p._version for k,p in prefix.parameter_map(g).items()}==versions,'Diagnostic changed native parameter versions')
        checks['parameter_versions_no_update']=True
        current=snapshot(g,torch,np)
        invariant_after=dict(parameters=native.tree_digest(prefix.parameter_map(g)),Adam=current['Adam'],
            gradients=current['gradients'],RNG=current['RNG'],deformation_accum=current['deformation_accum'])
        require(invariant_after==invariant_before,'Forward/autograd/density diagnostic changed parameters/Adam/saved gradients/RNG/deformation buffers')
        detail['diagnostic_unchanged_before_restore']=dict(before=invariant_before,after=invariant_after)
        checks['diagnostic_parameters_Adam_gradients_RNG_unchanged']=True
        fixture_plan=dict(schema='isolated_native_CUDA_checkpoint_roundtrip_fixture_v1',status='never_formal_not_selected',
            parent=case['parent'],dependencies=dict(schedule=entry(a.out/'request_plan.json')),source_sha256=SOURCE_SHA,
            synthetic_operation='zero-Adam real-parent clone density/checkpoint interface only; no formal meter/selection/teacher')
        fake_journal=SimpleNamespace(path=a.out/'diagnostic_operations.json',completed=dict(diagnostic_RGB=3,diagnostic_moments=3,
            diagnostic_autograd=counters['diagnostic_autograd_completed']-counter_start['diagnostic_autograd_completed'],
            formal_RGB=0,formal_moments=0,backward=0,Adam=0,iterations=0))
        clone_path=directory/'diagnostic_clone_00000.pt';clone_before=snapshot(g,torch,np)
        native.save_checkpoint(clone_path,g,h,opt,fixture_plan,0,payload['sampler'],torch,np,fake_journal,
            dict(**hardware,synthetic_checkpoint_plan=True,formal_updates=0))
        clone=native.checkpoint_payload(clone_path,torch,fixture_plan,native.SCHEMA,'cuda')
        native.restore_native_model(g,copy.deepcopy(clone),opt,torch,np,fixture_plan,native.SCHEMA)
        require(snapshot(g,torch,np)==clone_before,'Native SR capture/sidecar/restore is not exact')
        try:native.initialize_SR_statistics(g,0,torch,np)
        except ValueError:checks['mid_resume_density_boundary_refused']=True
        else:raise ValueError('Mid-resume density boundary was applied twice')
        checks['SR_capture_sidecar_restore']=True;detail['diagnostic_clone_checkpoint']=entry(clone_path)
        detail['diagnostic_clone_sidecar']=entry(clone_path.with_suffix('.json'))
        detail['diagnostic_clone_is_formal']=False
        require(native.tree_digest(payload)==payload_digest,'Independent reference payload was mutated by clone diagnostics')
        del clone,packets,moments,state,rgbs,cov,grads,mgrads,screen_grads,summed,radii,visible
        packets=None;moments=None;state=None
        native.restore_native_model(g,copy.deepcopy(payload),opt,torch,np,plan,PARENT_SCHEMA)
        if hasattr(g,'_full_SR_statistics_initialized'):del g._full_SR_statistics_initialized
        parent_after=snapshot(g,torch,np);require(parent_after==parent_before,'No-update parent model/Adam/buffers/grads/RNG restore differs')
        checks['parent_model_Adam_buffers_gradients_RNG_restored']=True
        torch.cuda.synchronize()
        result=dict(status='accepted_actual_native_CUDA_same_domain_parent',execution='actual_native_CUDA',synthetic=False,
            require_real_grid_parents=True,parent_is_same_domain=True,scene=case['scene'],seed=case['seed'],
            parent_scene=plan['scene'],parent_seed=plan['seed'],LR=case['LR'],HR=case['HR'],keys=case['keys'],
            manifest=case['manifest'],parent=case['parent'],LR_inputs=case['LR_inputs'],expected_parent_state=case['expected_parent_state'],
            restore_audit=audit,checks=checks,details=detail,parent_before=parent_before,parent_after=parent_after,
            hardware=dict(**hardware,CUDA_initialized=torch.cuda.is_initialized()),RGB_forwards=3,moment_forwards=3,
            Adam_calls=0,formal_updates=0,HR_GT_or_heldout_pixel_reads=0,seconds=time.monotonic()-started,
            diagnostic_operations={k:counters[k]-counter_start[k] for k in counters},
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            limitation='Three fixed training cameras at frame100; current finite/gradient/native-source compatibility, no teacher fitting/quality/full-support-calibration or benchmark claim.')
        validate_result(result);return result
    except BaseException as exc:
        write(directory/'case_failure.json',dict(status='failed_actual_native_CUDA_case_no_success_claim',
            scene=case['scene'],seed=case['seed'],parent=case['parent'],manifest=case['manifest'],source_sha256=SOURCE_SHA,
            checks=checks,details=detail,parent_before=parent_before,parent_after=parent_after,
            error=repr(exc),traceback=traceback.format_exc(),Adam_calls=0,formal_updates=0,
            diagnostic_operations={k:counters[k]-counter_start[k] for k in counters},seconds=time.monotonic()-started))
        raise
    finally:
        # The outer observer restores its own two-level RNG even on failure.
        g=None;payload=None;packets=None;moments=None;state=None

def run(a,plan):
    require(plan['status']=='registered_real_parent_CUDA_acceptance_ready' and plan['cases'],'Pending/empty real parent plan cannot call CUDA')
    require(a.operator_resource_resolved and a.gpu_uuid and a.lock,'Root UUID/shared lock/operator resource resolution required')
    require(a.out is not None and a.out.resolve().is_relative_to((BASE/'runs').resolve()),'Separate private acceptance runs output required')
    require(not a.out.exists(),'Existing acceptance evidence must be preserved')
    a.lock=local(a.lock);a.lock.parent.mkdir(parents=True,exist_ok=True)
    with a.lock.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        inventory=prefix.check_gpu(a.gpu_uuid)
        require(native.source_files(local(a.upstream))==plan['source_files'],'Frozen runtime source tree differs')
        a.out.mkdir(parents=True);write(a.out/'request_plan.json',plan)
        started=time.monotonic();counters={k:0 for k in ('RGB_attempted','RGB_completed','moment_attempted','moment_completed','diagnostic_autograd_attempted','diagnostic_autograd_completed')}
        a._started_native_attempt=True
        import numpy as np
        import torch
        import diff_gaussian_rasterization as raster
        torch.set_num_threads(a.cpu_threads);torch.cuda.init();torch.cuda.reset_peak_memory_stats()
        observer=prefix.rng_state(torch,np);observer_digest=native.tree_digest(observer);results=[];error=None;observer_restored=False
        hardware=dict(host=socket.gethostname(),gpu_uuid=a.gpu_uuid,inventory=inventory,torch=str(torch.__version__),
            GPU_name=torch.cuda.get_device_name(),physical_GPU=os.environ.get('CUDA_VISIBLE_DEVICES'),pid=os.getpid(),
            CUDA_version=str(torch.version.cuda),CUDA_capability=list(torch.cuda.get_device_capability()),
            rasterizer_python=dict(path=str(Path(raster.__file__).resolve()),sha256=sha(Path(raster.__file__))),
            rasterizer_extension=dict(path=str(Path(raster._C.__file__).resolve()),sha256=sha(Path(raster._C.__file__))))
        allowed={bound(i).resolve() for c in plan['cases'] for i in c['LR_inputs']};reads=dict(legal_LR_opens=0,HR_GT_or_heldout_pixel_reads=0,unique_LR=set())
        def guard(event,args):
            if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
            p=Path(os.fsdecode(args[0]));suffix=p.suffix.lower()
            if suffix not in ('.png','.jpg','.jpeg','.npy','.npz','.exr','.tiff','.tif'):return
            p=p.resolve()
            if p not in allowed:raise ValueError('Acceptance pixel/array read outside the fixed legal LR keys: '+str(p))
            reads['legal_LR_opens']+=1;reads['unique_LR'].add(str(p))
        sys.addaudithook(guard)
        try:
            for case in plan['cases']:
                prefix.check_gpu(a.gpu_uuid)
                result=actual_case(case,a,hardware,counters);results.append(result)
                write(a.out/(case['scene']+'_seed'+str(case['seed']))/'case_complete.json',result)
            for item in plan['prior_receipts']:
                value=read(bound(item['identity']));validate_receipt(value)
                for result in value['results']:
                    if not any((r['scene'],r['seed'])==(result['scene'],result['seed']) for r in results):results.append(result)
            require(native.source_files(local(a.upstream))==plan['source_files'],'Source changed during native acceptance')
            for case in plan['cases']:
                for key in ('checkpoint','complete','plan','sidecar'):bound(case['parent'][key])
        except BaseException as exc:
            error=exc
        finally:
            prefix.restore_rng(observer,torch,np)
            observer_restored=native.tree_digest(prefix.rng_state(torch,np))==observer_digest
            write(a.out/'diagnostic_operations.json',dict(counters=counters,Adam=0,formal_updates=0,wall_seconds=time.monotonic()-started,
                observer_RNG_before=observer_digest,observer_RNG_after=native.tree_digest(prefix.rng_state(torch,np)),observer_RNG_restored=observer_restored))
        reads['unique_LR']=sorted(reads['unique_LR'])
        if error is not None or not observer_restored:
            value=dict(status='failed_real_parent_native_CUDA_acceptance',source_sha256=SOURCE_SHA,author_commit=PIN,helper_source=entry(Path(__file__)),
                error=repr(error) if error else 'Observer RNG restore mismatch',traceback=''.join(traceback.format_exception(error)) if error else None,
                completed_cases=results,counters=counters,read_boundary=reads,observer_RNG_restored=observer_restored,
                GPU_calls_attempted=True,Adam_calls=0,formal_updates=0,wall_seconds=time.monotonic()-started)
            write(a.out/'failure.json',value)
            if error is not None:raise error
            raise ValueError(value['error'])
        value=dict(status=PASSED,source_sha256=SOURCE_SHA,author_commit=PIN,helper_source=entry(Path(__file__)),source_files=plan['source_files'],
            accepted_grids=[list(k) for k in sorted({tuple(r['LR']) for r in results})],parents=[r['parent'] for r in results],results=results,
            prior_receipts=[item['identity'] for item in plan['prior_receipts']],observer_RNG_restored=observer_restored,observer_RNG_before=observer_digest,
            observer_RNG_after=native.tree_digest(prefix.rng_state(torch,np)),Adam_calls=0,formal_updates=0,selection_created=False,
            HR_GT_or_heldout_pixel_reads=0,read_boundary=reads,diagnostic_operations=counters,wall_seconds=time.monotonic()-started,
            gate_limitation='This receipt covers accepted_grids only. Frozen production source status/SHA gate does not itself enforce grid; root must enforce the formal method grid is accepted before dispatch.',
            quality_evaluation=False,all_registered_grids_accepted=set(tuple(r['LR']) for r in results)==set(GRIDS))
        validate_receipt(value);write(a.out/'acceptance.json',value);os.chmod(a.out/'acceptance.json',0o444)
        return value

def cpu_contract(a):
    from full_held_confirmation_checks import run_checks
    directory=BASE/'CPU_contracts'/str(time.time_ns())
    return run_checks(directory)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('plan','run','contract'),default='plan')
    p.add_argument('--confirmation-authorization')
    p.add_argument('--manifest',type=Path);p.add_argument('--parent',type=Path);p.add_argument('--parent-complete',type=Path)
    p.add_argument('--case',type=Path,action='append',default=[]);p.add_argument('--include-receipt',type=Path,action='append',default=[])
    p.add_argument('--upstream',type=Path,default=Path(os.environ.get('FOURDSR_UPSTREAM','/home/cai_tianshun/Project/4dgs')));p.add_argument('--readiness',type=Path,default=domain.READINESS)
    p.add_argument('--require-real-grid-parents',action='store_true',default=True)
    p.add_argument('--out',type=Path);p.add_argument('--gpu-uuid');p.add_argument('--lock',type=Path);p.add_argument('--operator-resource-resolved',action='store_true')
    p.add_argument('--cpu-threads',type=int,default=2)
    a=p.parse_args()
    if a.out:a.out=local(a.out)
    started=time.monotonic()
    try:
        if a.mode=='contract':value=cpu_contract(a)
        else:
            require(a.confirmation_authorization,'Actual final freeze/two-teacher held authorization required')
            plan=build_plan(a)
            value=plan if a.mode=='plan' else run(a,plan)
    except BaseException as exc:
        # Includes environment/import/resource failures before the inner RNG
        # observer exists. Preserve independent error evidence, never a gate.
        directory=BASE/'failures'/str(time.time_ns());directory.mkdir(parents=True)
        write(directory/'failure.json',dict(status='failed_or_pending_before_native_CUDA_acceptance',mode=a.mode,
            helper_source=entry(Path(__file__)),source_sha256=SOURCE_SHA,error=repr(exc),traceback=traceback.format_exc(),
            native_attempt_started=getattr(a,'_started_native_attempt',False),GPU_accepted=False,Adam_calls=0,formal_updates=0,
            wall_seconds=time.monotonic()-started,requested_out=str(a.out) if a.out else None))
        raise
    print(json.dumps(value,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
