"""Small numerical/metadata contracts; no real PNG, NPZ, PT or model access."""
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
import full_temporal_diagnostics as d


def fake_reference(name):
    return dict(path=name, sha256=hashlib.sha256(name.encode()).hexdigest())


def bundle(scene='cook_spinach', seed=20261007):
    role = 'already_used_development' if scene in d.DEVELOPMENT else 'held_confirmation'
    hr = [1280, 720] if scene.startswith('meetroom_') else [1344, 1008]
    manifest = dict(scene=scene, role=role, official_split=True, additional_cam01_dev_holdout=False, full_time_decode_verified=True,
                    frame_indices=list(range(300)), splits=dict(test=['cam00'],dev=[],train=['cam01','cam02','cam03']), initialization=dict(seed=seed),
                    resolutions=dict(hr=hr,lr=[v//4 for v in hr]),cameras={'cam00':dict(K_hr=[[1,0,0],[0,1,0],[0,0,1]])},
                    observations=[dict(split='test',camera_id='cam00',frame_index=f,time=f/300,hr_path=f'hr/cam00/{f:04d}.png',hr_sha256='h'+str(f),lr_sha256='l'+str(f)) for f in range(300)])
    data=dict(manifest=fake_reference(scene+'/manifest.json'))
    plan=dict(schema=d.SR_SCHEMA,status='registered_ready_full_native_SR_refinement',scene=scene,seed=seed,method='M',data=data)
    meta=dict(cursor=6000,density_statistical_boundary_applied=True,terminal_zero_grad=True,plan_sha256=d.digest(plan),audit=dict(optimizer_groups=[dict(name=n) for n in d.GROUPS]))
    cp=dict(schema=d.SR_SCHEMA,metadata=meta,checkpoint=fake_reference('CP06000.pt'),plan=fake_reference('config.json'),sidecar=fake_reference('CP06000.json'))
    identity=dict(checkpoint=cp,manifest=data['manifest'],GPU='GPU-fake',host='fake-CPU-fixture',sources={'eval.py':'frozen'},protocol=fake_reference('protocol.json'))
    complete=dict(status=d.EVAL_STATUS,test_observations=300,formal_updates=0,Adam_calls=0,backward_calls=0,model_RNG_unchanged=True,identity=identity)
    summary=dict(status=d.EVAL_STATUS,identity=identity,model_immutability=dict(equal=True),hardware=dict(physical_GPU='GPU-fake',host='fake-CPU-fixture',torch='mock',cuda='mock'),
                 cost=dict(Adam_calls=0,backward_calls=0,formal_updates=0),native_restore=dict(one_Adam_exact=True,parameters_exact=True,topology_buffers_exact=True,saved_gradients_exact=True,global_RNG_exact=True,no_children=True,optimizer_conversion=False))
    index=dict(status='completed_full_native_float_observations',identity=identity,test_observations=300,training_cache=False,
               entries=[dict(camera='cam00',frame=f,scope='full_test',path=f'floats/cam00_{f:04d}.npz',sha256='f'+str(f),raw_dtype='float32',training_cache=False,checkpoint_sha256=cp['checkpoint']['sha256']) for f in range(300)])
    protocol=dict(schema=d.PROTOCOL_SCHEMA,status='registered_full_native_evaluation_before_prediction_reads',role=role,scene=scene,seed=seed,manifest=data['manifest'],data=data,sources=identity['sources'],camera_interface_SHA256=d.digest(manifest['cameras']),resolutions=manifest['resolutions'],test_keys=[['cam00',f] for f in range(300)],measurement_definitions={'clamp':'official float32'},selection=fake_reference('selection.json'))
    protocol['measurement_definition_SHA256']=d.digest(protocol['measurement_definitions'])
    protocol['test_input_SHA256']=d.digest([(r['camera_id'],r['frame_index'],r['hr_sha256'],r['lr_sha256']) for r in manifest['observations']])
    sidecar=dict(**cp['checkpoint'],schema=d.SR_SCHEMA,metadata=meta)
    return [complete,summary,index,manifest,protocol,plan,sidecar]


def final_freeze():
    development={}
    for scene in d.DEVELOPMENT:
        for seed in d.SEEDS:
            v=bundle(scene,seed);development[scene+'/'+str(seed)]=(v[0],v[5],v[1])
    selection=dict(status='registered_full_development_selection',selected_candidates=['M','X'],baseline_method='Bsync',uses_confirmation_for_selection=False)
    freeze=dict(status=d.MAIN_FREEZE_STATUS,selection=fake_reference('selection.json'),main_method='M',configuration_frozen_before_confirmation_HR=True,confirmation_HR_used_for_selection=False,frozen_configuration=fake_reference('main_config.json'),development_results_complete=True,confirmation_scenes=list(d.CONFIRMATION),full_development_evidence={k:fake_reference(k+'/complete.json') for k in development})
    return freeze,selection,development


def run():
    started=time.monotonic();passed=[]
    def check(name,fn):fn();passed.append(name)
    def reject(name,fn):
        try:fn()
        except (ValueError,KeyError,TypeError):passed.append(name);return
        raise AssertionError(name+' should reject')
    gt=np.full((2,3,3),.5,dtype=np.float32)
    check('perfect changing reconstruction has zero temporal residual',lambda:np.testing.assert_equal(d.adjacent_errors(gt,gt*.5,gt,gt*.5)['clamped_temporal_residual_change_mse'],0.))
    check('constant bias zero temporal but positive spatial',lambda:(np.testing.assert_equal(d.adjacent_errors(gt+.25,gt+.25,gt,gt)['clamped_temporal_residual_change_mse'],0.),np.testing.assert_equal(d.frame_errors(gt+.25,gt)['clamped_spatial_residual_mse'],.0625)))
    check('alternating signed error expected temporal MSE .25',lambda:np.testing.assert_equal(d.adjacent_errors(gt-.25,gt+.25,gt,gt)['clamped_temporal_residual_change_mse'],.25))
    one_channel=gt.copy();one_channel[:,:,0]+=.25
    check('RGB average includes all3 channels',lambda:np.testing.assert_allclose(d.adjacent_errors(gt,one_channel,gt,gt)['clamped_temporal_residual_change_mse'],.0625/3,rtol=0,atol=1e-15))
    check('raw outside range differs explicitly from clamped',lambda:(np.testing.assert_equal(d.adjacent_errors(gt+1.,gt+2.,gt,gt)['clamped_temporal_residual_change_mse'],0.),np.testing.assert_equal(d.adjacent_errors(gt+1.,gt+2.,gt,gt)['raw_temporal_residual_change_mse'],1.)))
    reject('NaN render rejected',lambda:d.frame_errors(np.full_like(gt,np.nan),gt))
    reject('Inf GT rejected',lambda:d.frame_errors(gt,np.full_like(gt,np.inf)))
    reject('different RGB shape rejected',lambda:d.adjacent_errors(gt,gt[:1],gt,gt[:1]))
    reject('wrong float precision rejected',lambda:d.frame_errors(gt.astype(np.float64),gt))
    reject('wrong RGB channel count rejected',lambda:d.frame_errors(gt[:,:,:2],gt[:,:,:2]))
    check('complete development metadata accepted',lambda:d.validate_metadata(*bundle()))
    def mutated(index,fn):
        v=bundle();fn(v[index]);return lambda:d.validate_metadata(*v)
    reject('partial completed count rejected',mutated(0,lambda x:x.update(test_observations=299)))
    reject('missing float frame rejected',mutated(2,lambda x:x['entries'].pop(40)))
    reject('duplicate float frame rejected',mutated(2,lambda x:x['entries'].__setitem__(40,copy.deepcopy(x['entries'][39]))))
    reject('out of order missing GT frame rejected',mutated(3,lambda x:x['observations'].pop(200)))
    reject('wrong role rejected',mutated(3,lambda x:x.update(role='held_confirmation')))
    reject('cam01 cannot enter NVS float set',mutated(2,lambda x:x['entries'][100].update(camera='cam01')))
    reject('native SR cursor5999 rejected',mutated(6,lambda x:x['metadata'].update(cursor=5999)))
    reject('render CP identity changed rejected',mutated(2,lambda x:x['entries'][0].update(checkpoint_sha256='other')))
    reject('HR path escape rejected',mutated(3,lambda x:x['observations'][10].update(hr_path='../outside.png')))
    reject('evaluation model RNG changed rejected',mutated(0,lambda x:x.update(model_RNG_unchanged=False)))
    reject('nonzero diagnostic Adam rejected',mutated(0,lambda x:x.update(Adam_calls=1)))
    held=bundle('coffee_martini');reject('held without final freeze rejected',lambda:d.validate_metadata(*held))
    f,s,dev=final_freeze();check('held single frozen main plus all8 full dev accepted',lambda:d.validate_metadata(*held,freeze=f,selection=s,development=dev))
    bad=copy.deepcopy(f);bad['status']='registered_full_development_selection';reject('development two candidate gate is not final freeze',lambda:d.validate_metadata(*held,freeze=bad,selection=s,development=dev))
    bad=copy.deepcopy(f);bad['confirmation_HR_used_for_selection']=True;reject('confirmation selected from HR rejected',lambda:d.validate_metadata(*held,freeze=bad,selection=s,development=dev))
    bad=copy.deepcopy(dev);bad.pop(next(iter(bad)));reject('incomplete8 developmental evidence rejected',lambda:d.validate_metadata(*held,freeze=f,selection=s,development=bad))
    bad=copy.deepcopy(dev);next(iter(bad.values()))[2]['hardware']['physical_GPU']='other';reject('mixed development evaluation hardware rejected',lambda:d.validate_metadata(*held,freeze=f,selection=s,development=bad))
    bad=copy.deepcopy(dev);next(iter(bad.values()))[0]['identity']['checkpoint']['schema']='full_native_SR_refinement_v1';reject('old unregistered SR schema rejected',lambda:d.validate_metadata(*held,freeze=f,selection=s,development=bad))
    identity=copy.deepcopy(held[0]['identity']);source_key=str((d.HERE/'full_evaluate_registered.py').relative_to(d.ROOT));identity['sources'][source_key]='0'*64
    auth_ref=fake_reference('preauthorization.json');freeze_ref=fake_reference('final_single_main.json');complete_ref=fake_reference('confirmation_complete.json')
    manager=dict(unit='fake-owned.service',invocation_id='a'*32,pid=100,start_ticks=200,start_monotonic=300,host='fake-CPU-fixture',boot_id='fakeboot')
    auth=dict(status='root_authorized_frozen_main_before_confirmation_evaluation',final_main_freeze=freeze_ref,expected_evaluation_registration=identity,protocol=identity['protocol'],checkpoint=identity['checkpoint']['checkpoint'],HR_reads_before_authorization=0,authorized_before_prediction_and_HR=True,backend_source=fake_reference('backend.py'),manager_spec=fake_reference('spec.json'),manager_identity=manager,authorization_written_monotonic_ns=1000,evaluation_command=['python','eval.py','--mode','evaluate'],evaluator_source=dict(path=source_key,sha256='0'*64))
    launch=dict(status='completed_authorized_confirmation_evaluation_child_Exit0',preauthorization=auth_ref,final_main_freeze=freeze_ref,evaluation_complete=complete_ref,backend_source=auth['backend_source'],manager_spec=auth['manager_spec'],manager_identity=manager,authorization_written_monotonic_ns=1000,child_spawn_monotonic_ns=1001,child_exit_monotonic_ns=1002,wait_authority='own_subprocess_wait_return',child_exit_code=0,child_pid=101,child_start_ticks=201,child_argv=auth['evaluation_command'],child_host=manager['host'],child_boot_id=manager['boot_id'])
    check('held before-spawn authorization and exact actual return binding',lambda:d.confirmation_authorization_gate(auth,launch,auth_ref,freeze_ref,complete_ref,identity))
    bad=copy.deepcopy(auth);bad['authorization_written_monotonic_ns']=1003;reject('post-hoc authorization cannot precede already spawned child',lambda:d.confirmation_authorization_gate(bad,launch,auth_ref,freeze_ref,complete_ref,identity))
    bad=copy.deepcopy(launch);bad['child_exit_code']=1;reject('failed evaluator child cannot pass confirmation',lambda:d.confirmation_authorization_gate(auth,bad,auth_ref,freeze_ref,complete_ref,identity))
    bad=copy.deepcopy(launch);bad['manager_identity']['invocation_id']='b'*32;reject('different manager invocation rejected',lambda:d.confirmation_authorization_gate(auth,bad,auth_ref,freeze_ref,complete_ref,identity))
    bad=copy.deepcopy(launch);bad['child_argv']=['python','other.py','--mode','evaluate'];reject('different native child argv rejected',lambda:d.confirmation_authorization_gate(auth,bad,auth_ref,freeze_ref,complete_ref,identity))
    reject('held missing launch proof rejected',lambda:d.confirmation_authorization_gate(auth,None,auth_ref,freeze_ref,complete_ref,identity))
    bad=copy.deepcopy(auth);bad['expected_evaluation_registration']['checkpoint']['checkpoint']['sha256']='wrong';reject('different confirmation evaluation CP rejected',lambda:d.confirmation_authorization_gate(bad,launch,auth_ref,freeze_ref,complete_ref,identity))
    bad=copy.deepcopy(launch);bad['child_host']='other-host';reject('cross host monotonic proof rejected',lambda:d.confirmation_authorization_gate(auth,bad,auth_ref,freeze_ref,complete_ref,identity))
    bad=copy.deepcopy(launch);bad['child_boot_id']='other-boot';reject('cross boot monotonic proof rejected',lambda:d.confirmation_authorization_gate(auth,bad,auth_ref,freeze_ref,complete_ref,identity))
    with tempfile.TemporaryDirectory() as folder:
        root=Path(folder);p=root/'metadata.json';p.write_text('{"fake":true}');ref=dict(path='metadata.json',sha256=d.sha(p));check('exact fake file SHA accepted',lambda:d.read_bound(root,ref));p.write_text('{"fake":false}');reject('changed file SHA rejected before data',lambda:d.read_bound(root,ref));reject('path escape rejected',lambda:d.safe_path(root,'../outside.json'));link=root/'link.json';link.symlink_to(p);reject('symlink input rejected',lambda:d.safe_path(root,'link.json'))
        from types import SimpleNamespace
        original_root=d.ROOT
        try:
            d.ROOT=root;existing=root/'output'/d.HERE.name/'full_temporal_diagnostics'/'existing';existing.mkdir(parents=True)
            reject('existing output rejected before any evaluation input read',lambda:d.diagnose(SimpleNamespace(out=existing)))
            reject('other output namespace rejected before inputs',lambda:d.diagnose(SimpleNamespace(out=root/'other_out')))
        finally:d.ROOT=original_root
    import sys
    assert 'torch' not in sys.modules and 'scene.gaussian_model' not in sys.modules
    return dict(status='passed_small_CPU_numerical_and_input_boundary_contracts',checks=len(passed),cases=passed,CPU_wall_seconds=time.monotonic()-started,source=d.entry(Path(d.__file__)),checks_source=d.entry(Path(__file__)),numpy=np.__version__,real_PNG_NPZ_PT_reads=0,model_imports=0,GPU_calls=0,SSH_or_services=0,formal_updates=0,Adam_calls=0,production_temporal_results_generated=False,production_confirmation_preauthorization_proved=False)


if __name__=='__main__':print(json.dumps(run(),ensure_ascii=False,allow_nan=False))
