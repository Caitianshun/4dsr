"""CPU contracts for registered native SR/evaluation; no actual GPU dispatch."""
from __future__ import annotations
import ast, copy, json, os, sys, time, traceback
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
    raise RuntimeError("Use a new process with CUDA_VISIBLE_DEVICES=''")
import torch
import numpy as np
from PIL import Image
if torch.cuda.is_initialized():raise RuntimeError('CPU contracts require uninitialized CUDA')
torch.set_num_threads(1)
import full_native_protocol as domain
import full_prefix_registered as prefix
import full_refine_registered as sr
import full_evaluate_registered as ev
import full_refine_checks as fixture
from fp_common import ROOT,HERE,OUT,entry,read,write,sha,local


def rejects(fn):
    try:fn()
    except (ValueError, AssertionError, FileNotFoundError, KeyError):return
    raise AssertionError('Invalid registered contract accepted')


def main():
    target=OUT/'operator_checks/full_registered'/str(time.time_ns());target.mkdir(parents=True)
    started=time.monotonic();checks=[];sources={str(p.relative_to(ROOT)):sha(p) for p in
        (Path(__file__),HERE/'full_refine_registered.py',HERE/'full_evaluate_registered.py',HERE/'full_native_protocol.py',HERE/'full_prefix_registered.py',HERE/'full_refine.py',HERE/'full_evaluate.py',HERE/'full_prefix.py')}
    def check(name,call):
        tick=time.monotonic();call();assert not torch.cuda.is_initialized()
        checks.append(dict(name=name,passed=True,seconds=time.monotonic()-tick))
    try:
        old_entries=read(OUT/'full_LR_prefixes/checkpoint_index.json')['entries'];accepted=[]
        for row in old_entries:
            cp=local(row['checkpoint']['path']);p=read(cp.parent/'config.json')
            def old_parent(row=row,cp=cp,p=p):
                identity,plan=sr.full_parent(cp,local(row['complete']['path']),p['data']['manifest'],row['seed'])
                e,ep=ev.validate_checkpoint(cp,p['data']['manifest'])
                assert plan==p==ep and e['checkpoint']==row['checkpoint']
                assert identity['registered_domain_contract']['batch_size']==2 and identity['state']['accepted_RGB']==34000
                assert identity['registered_domain_contract']['accepted_exit']==row['exit_acceptance']
                accepted.append(identity)
            check(f"actual_accepted_legacy_{row['scene']}_{row['seed']}_original_source_plan",old_parent)
        ref=read(local(old_entries[0]['checkpoint']['path']).parent/'config.json')
        def mutable_index_safe():
            complete=read(local(old_entries[0]['complete']['path']));before=sr.validate_parent_plan(ref,complete)
            current=domain.completed_prefix(ref['scene'],ref['seed']);changed=copy.deepcopy(current);changed['index']['sha256']='0'*64
            with patch.object(domain,'completed_prefix',lambda *a:changed):after=sr.validate_parent_plan(ref,complete)
            assert before==after and before['accepted_entry']==current['entry']
        check('additional_completed_scenes_do_not_change_existing_parent_plan_identity',mutable_index_safe)
        wrong=copy.deepcopy(ref);wrong['planned']['RGB_forwards']=68000
        check('legacy_batch_count_change_refused',lambda:rejects(lambda:sr.validate_parent_plan(wrong,read(local(old_entries[0]['complete']['path'])))))
        wrong=copy.deepcopy(ref);wrong['project_sources'][str((HERE/'full_prefix.py').relative_to(ROOT))]='0'*64
        check('legacy_unaccepted_source_refused',lambda:rejects(lambda:sr.validate_parent_plan(wrong,read(local(old_entries[0]['complete']['path'])))))
        new_plans={};manifests={};manifest_reads=0
        for scene in ('meetroom_discussion','meetroom_vrheadset','coffee_martini'):
            manifest=ROOT/'data/dynamic_sr/full_time_20261007/prepared'/scene/'manifest_train_ready_seed20261007.json'
            p,inputs=prefix.build_plan(manifest,20261007);new_plans[scene]=p;manifests[scene]=read(manifest);manifest_reads+=len(inputs)
            def new_parent(p=p,scene=scene,inputs=inputs):
                c=sr.validate_parent_plan(p)
                assert c['batch_size']==4 and c['expected_state']['accepted_RGB']==68000 and p['planned']['RGB_forwards']==68000
                assert c['native_HR']==manifests[scene]['resolutions']['hr'] and len(inputs)==p['data']['training_observations']
            check(f'actual_{scene}_registered_batch4_manifest_source_grid',new_parent)
        p=new_plans['meetroom_discussion'];m=manifests['meetroom_discussion']
        fake=target/'synthetic_new_parent';fake.mkdir();cp=fake/'checkpoint_fine_14000.pt';cp.write_bytes(b'CPU metadata fixture, not a model')
        fakeplan=fake/'config.json';write(fakeplan,p)
        audit=accepted[0]['audit'];state=domain.expected_parent_state(p['configuration'])
        meta=dict(state=state,completed_iterations=17000,plan_sha256=prefix.digest(p),representation=sr.REPRESENTATION,audit=audit)
        side=dict(**entry(cp),schema=ev.PARENT_SCHEMA,metadata=meta);write(cp.with_suffix('.json'),side)
        done=dict(status='completed_independent_full_author_LR_prefix',checkpoint=entry(cp),plan=entry(fakeplan),seed=p['seed'],scene=p['scene'],state=state,representation=sr.REPRESENTATION,HR_used=False,heldout_pixels_used=False,teacher_used=False,execution_source=entry(HERE/'full_prefix_registered.py'))
        write(fake/'complete.json',done)
        check('new_batch4_parent_sidecar_refine_evaluate_interface',lambda: (sr.full_parent(cp,fake/'complete.json',p['data']['manifest'],p['seed']),ev.validate_checkpoint(cp,p['data']['manifest'])))
        badside=copy.deepcopy(side);badside['metadata']['state']['accepted_RGB']=34000;write(cp.with_suffix('.json'),badside)
        check('new_batch4_counter34000_refused_before_tensor_load',lambda:rejects(lambda:sr.full_parent(cp,fake/'complete.json',p['data']['manifest'],p['seed'])))
        write(cp.with_suffix('.json'),side)
        for field,value in (('execution_source',entry(HERE/'full_prefix.py')),('HR_used',True),('heldout_pixels_used',True),('teacher_used',True)):
            bad=copy.deepcopy(done);bad[field]=value;write(fake/'complete.json',bad)
            check('new_parent_complete_'+field+'_boundary_refused',lambda:rejects(lambda:sr.full_parent(cp,fake/'complete.json',p['data']['manifest'],p['seed'])))
        write(fake/'complete.json',done)
        for name,change in (
            ('source',lambda x:x['project_sources'].__setitem__(str((HERE/'full_prefix_registered.py').relative_to(ROOT)),'0'*64)),
            ('RGB_count',lambda x:x['planned'].__setitem__('RGB_forwards',34000)),
            ('HR_grid',lambda x:x['data'].__setitem__('resolution_HR',[1344,1008]))):
            bad=copy.deepcopy(p);change(bad)
            check(f'new_registered_{name}_mutation_refused',lambda bad=bad:rejects(lambda:sr.validate_parent_plan(bad)))
        badm=copy.deepcopy(m);badm['resolutions']['lr']=[336,252]
        check('MeetRoom_N3DV_grid_substitution_refused',lambda:rejects(lambda:domain.validate_grid(badm,m['scene'])))
        badm=copy.deepcopy(m);badm['cameras']['cam01']['K_lr'][0][2]+=.5
        check('pixel_centre_intrinsics_error_refused',lambda:rejects(lambda:domain.validate_grid(badm,m['scene'])))
        legal=next(r for r in m['observations'] if r['split']=='train')
        for label,row in (('HR_path',dict(legal,lr_path='hr/cam01/0000.png')),('test',dict(legal,camera_id='cam00',split='test',lr_path='lr/cam00/0000.png'))):
            check(f'illegal_{label}_training_input_refused',lambda row=row:rejects(lambda:domain.legal_LR_path(ROOT,row)))
        evidence=target/'short_evidence.json';write(evidence,dict(synthetic_CPU_selection_boundary_fixture=True))
        selection=dict(status='registered_full_development_selection',baseline_method='B0',selected_candidates=['M','X'],necessary_ablation_methods=[],uses_confirmation_for_selection=False,reason='CPU gate fixture, no actual selection',short_window_evidence=[entry(evidence)])
        check('two_selected_candidates_baseline_accepted',lambda:sr.validate_selection(selection,'M'))
        check('unselected_MX_refused',lambda:rejects(lambda:sr.validate_selection(selection,'MX')))
        too_many=dict(selection,selected_candidates=['M','X','E'])
        check('third_candidate_grid_refused',lambda:rejects(lambda:sr.validate_selection(too_many,'M')))
        held=dict(selection,uses_confirmation_for_selection=True)
        check('confirmation_used_for_selection_refused',lambda:rejects(lambda:sr.validate_selection(held,'M')))
        args=SimpleNamespace(manifest=local(p['data']['manifest']['path']),seed=p['seed'],method='MX',steps=6000,topology_clock='refinement',parent=ROOT/'missing-native-full-parent.pt',parent_complete=ROOT/'missing-native-full-complete.json',teacher=ROOT/'missing-full-teacher.json',schedule=ROOT/'missing-full-schedule.json',selection=None,support_index=None,calibration=None,upstream=None)
        plan=sr.build_plan(args)
        def pending():
            assert plan['planned_cost']['prefix_per_seed']['RGB']==68000 and plan['planned_cost']['RGB_forwards']==18000
            assert plan['planned_cost']['moment_forwards']==18000 and plan['data']['resolution_HR']==[1280,720]
            rejects(lambda:sr.train(SimpleNamespace(),plan))
        check('batch4_registered_pending_keeps_6000_threeRGB_and_no_CUDA',pending)
        oldrow=old_entries[0];oldcp=local(oldrow['checkpoint']['path']);oldplan=read(oldcp.parent/'config.json')
        oldargs=copy.copy(args);oldargs.manifest=local(oldplan['data']['manifest']['path']);oldargs.seed=oldrow['seed'];oldargs.parent=oldcp;oldargs.parent_complete=local(oldrow['complete']['path']);oldargs.method='M'
        oldregistration=sr.build_plan(oldargs)
        check('accepted_Cook_parent_metadata_plan_allowed_without_retraining',lambda: assert_old_registration(oldregistration,oldrow))
        protocol=ev.register(args.manifest,target/'Meet_evaluation_protocol.json')
        check('Meet_evaluation_registration_metadata_only_dynamic_grid',lambda:ev.validate_protocol(protocol,m,p['data']))
        bad=copy.deepcopy(protocol);bad['resolutions']['hr']=[1344,1008]
        check('evaluation_protocol_grid_mutation_refused',lambda:rejects(lambda:ev.validate_protocol(bad,m,p['data'])))
        bad=copy.deepcopy(protocol);bad['test_input_SHA256']='0'*64
        check('evaluation_test_reference_metadata_mutation_refused',lambda:rejects(lambda:ev.validate_protocol(bad,m,p['data'])))
        heldm=copy.deepcopy(m);heldm['role']='held_confirmation';bad=copy.deepcopy(protocol);bad['role']='held_confirmation'
        check('held_evaluation_without_selection_refused',lambda:rejects(lambda:ev.validate_protocol(bad,heldm,p['data'])))
        item=ev.camera_item(m,next(r for r in m['observations'] if r['split']=='test'),torch)
        def camera_payload():
            assert item['image'].shape==(3,180,320) and not item['image'].any()
            assert not {'hr_path','lr_path','hr_image','lr_image'} & set(item)
            fakecam=SimpleNamespace(image_width=320,image_height=180,projection_matrix=torch.eye(4),world_view_transform=torch.eye(4))
            with patch.dict(sys.modules,{'n3dv_data':SimpleNamespace(observation_to_4dgs_camera=lambda *a:fakecam)}):cam=sr.hr_camera(item,0,m)
            k=m['cameras']['cam00']['K_hr'];assert (cam.image_width,cam.image_height)==(1280,720)
            assert abs(float(cam.projection_matrix[2,0])-(2*k[0][2]+1-1280)/1280)<1e-7
        check('Meet_camera_synthetic_LR_payload_and_HR_principal_point',camera_payload)
        for size in ((1344,1008),(1280,720)):
            path=target/f'synthetic_teacher_{size[0]}x{size[1]}.png';Image.new('RGB',size,(40,70,90)).save(path)
            key=('cam01',0);cache=sr.TeacherCache({key:path},{key:sha(path)},'cpu',size)
            check(f'synthetic_teacher_{size[0]}x{size[1]}_accepted',lambda cache=cache,size=size:assert_shape(cache.get(key),(3,size[1],size[0])))
            other=(1280,720) if size==(1344,1008) else (1344,1008)
            bad=sr.TeacherCache({key:path},{key:sha(path)},'cpu',other)
            check(f'synthetic_teacher_wrong_registered_{size[0]}_grid_refused',lambda bad=bad:rejects(lambda:bad.get(key)))
        # Mathematical/rendering paths must remain exact AST copies. Only
        # metadata/grid/source interfaces changed. A function renamed/imported
        # into a new module is not an excuse for a changed scientific formula.
        old=ast.parse((HERE/'full_refine.py').read_text());new=ast.parse((HERE/'full_refine_registered.py').read_text())
        names=('native_effective_state','state_covariance','native_rgb','hr_camera','initialize_SR_statistics','gradient_diagnostics','refinement_iteration','calibration_coefficients','validate_schedule','validate_support','validate_calibration','calibrate')
        def parity():
            f=lambda tree:{n.name:ast.dump(n,include_attributes=False) for n in tree.body if isinstance(n,ast.FunctionDef)}
            a,b=f(old),f(new)
            assert all(a[n]==b[n] for n in names)
            a=f(ast.parse((HERE/'full_prefix.py').read_text()));b=f(ast.parse((HERE/'full_prefix_registered.py').read_text()))
            assert all(a[n]==b[n] for n in ('combine_batch_statistics','author_topology','parameter_map','rng_state','restore_rng','topology_audit'))
        check('native_formulas_six_edges_calibration_density_boundary_clock_AST_exact',parity)
        fixture.bind_author_capture_restore()
        def executable_updates():
            for scene in ('cook_spinach','meetroom_discussion'):
                config,_=domain.configuration(scene);opt=SimpleNamespace(**config['OptimizationParams']);h=SimpleNamespace(**config['ModelHiddenParams']);pipe=SimpleNamespace(**config['PipelineParams'])
                g=fixture.Gaussian();g.training_setup(opt);events=[];step=g.optimizer.step
                def actual_step():events.append('Adam');return step()
                g.optimizer.step=actual_step;cams=[fixture.camera(i) for i in range(3)];keys=[(f'cam{i+1:02d}',100) for i in range(3)]
                def render(c,g,state,pipe,bg):events.append('RGB');return sr.native_rgb(c,g,state,pipe,bg,(SimpleNamespace,fixture.Raster))
                boundary=sr.initialize_SR_statistics(g,0,torch,np)
                r=sr.refinement_iteration(g,h,opt,pipe,torch.ones(3),cams,[torch.full((3,8,8),.35) for _ in cams],[torch.full((3,32,32),.4) for _ in range(2)],keys,sr.ARMS['M'],6000,2.5,target,torch,render=render)
                assert (r['RGB'],r['backward'],r['Adam'],r['LR_clock'],r['topology_clock'])==(3,1,1,20000,6000)
                assert events==['RGB']*3+['Adam'] and g._deformation.calls==1
                assert boundary['cleared_buffers']==['max_radii2D','xyz_gradient_accum','denom']
                rejects(lambda:sr.initialize_SR_statistics(g,0,torch,np))
                assert len(g.capture())==14 and len(g.optimizer.param_groups)==8
            packets=[]
            for radius,vis,grad in (([2.,1.],[True,False],[[2.,0.,0.],[0.,1.,0.]]),([1.,7.],[False,True],[[-2.,0.,0.],[0.,-1.,0.]]),([4.,3.],[True,False],[[0.,0.,0.],[0.,0.,0.]])):
                v=torch.zeros(2,3,requires_grad=True);v.grad=torch.tensor(grad);packets.append(dict(radii=torch.tensor(radius),visibility_filter=torch.tensor(vis),viewspace_points=v))
            r,v,g=prefix.combine_batch_statistics(packets,torch)
            assert torch.equal(r,torch.tensor([4.,7.])) and bool(v.all()) and not bool(g.any())
        check('batch2_and4_parent_recipes_executable_SR_oneAdam8_3RGB_shared_state_density_SUMmaxOR',executable_updates)
        def readonly_eval():
            fn=next(n for n in ast.parse(Path(ev.__file__).read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='evaluate')
            calls=[n.func.attr for n in ast.walk(fn) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
            assert not set(calls)&{'step','zero_grad','backward','densify','prune','grow','reset_opacity','initialize_SR_statistics'}
            assert 'no_grad' in calls
        check('registered_evaluation_no_update_no_density_boundary',readonly_eval)
        assert all(sha(ROOT/name)==value for name,value in sources.items()),'Source changed during CPU checks; rerun frozen revision'
        snapshot=target/'source_snapshot';snapshot.mkdir()
        for name in sources:(snapshot/Path(name).name).write_bytes((ROOT/name).read_bytes())
        receipt=dict(status='passed_registered_full_native_SR_evaluation_CPU_contracts',checks=checks,count=len(checks),sources=sources,actual_legacy_parents=accepted,new_scene_LR_SHA_reads=manifest_reads,
            CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES'],CUDA_initialized=torch.cuda.is_initialized(),GPU_forwards=0,actual_HR_or_test_images_read=0,
            CPU_synthetic_SR_Adam_calls=2,CPU_synthetic_RGB_calls=6,formal_updates=0,wall_seconds=time.monotonic()-started,
            scope='CPU metadata/actual accepted sidecars/all legal LR SHA/AST parity/synthetic teacher and executable native-model fixtures; new parent sidecar is synthetic, no new trained parent, no CUDA acceptance or scientific quality result')
        write(target/'receipt.json',receipt);print(json.dumps(dict(receipt=entry(target/'receipt.json'),count=receipt['count'],seconds=receipt['wall_seconds'],sources=sources),indent=2));return receipt
    except BaseException as exc:
        write(target/'failure.json',dict(status='failed_CPU_contract_preserved',sources=sources,checks_completed=checks,seconds=time.monotonic()-started,error=repr(exc),traceback=traceback.format_exc(),GPU_forwards=0,formal_updates=0));raise


def assert_old_registration(plan,row):
    assert plan['parent']['checkpoint']==row['checkpoint'] and plan['planned_cost']['prefix_per_seed']['RGB']==34000
    assert plan['status']=='pending_full_native_refinement_dependencies' and plan['data']['resolution_HR']==[1344,1008]


def assert_shape(tensor,shape):assert tuple(tensor.shape)==shape
if __name__=='__main__':main()
