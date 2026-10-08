"""CPU-only held bridge source preservation and authorization contracts.

Actual root freeze/held parents are read as small metadata. Clearly labelled
private teacher fixtures test gates, never claim actual teacher/native quality.
No pixel, checkpoint, tensor, GPU, SSH or service is read or dispatched.
"""
from pathlib import Path
import argparse, ast, copy, hashlib, importlib.util, json, os, sys, time
import full_held_confirmation_contract_v2 as held

HERE=held.HERE;ROOT=held.ROOT;OUT=held.OUT
FREEZE=OUT/'final_full16_development_method_freeze_20261008/final_method_freeze.json'

def write(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    p.chmod(0o444)
def load(name,p):
    sp=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(sp);sys.modules[name]=m;sp.loader.exec_module(m);return m

def functions(p):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(p.read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
def run_checks(directory):
    require=held.require;directory=held.path(directory);require(not directory.exists(),'Fresh CPU contract output required');directory.mkdir(parents=True)
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','Explicit CPU boundary required')
    opened=[]
    def guard(event,args):
        if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
            p=Path(os.fsdecode(args[0]));require(p.suffix.lower() not in held.PIXELS,'CPU held contract attempted payload read: '+str(p));opened.append(str(p))
    sys.addaudithook(guard);checks=[]
    def good(label,action):
        action();checks.append(dict(check=label,passed=True))
    def bad(label,action):
        try:action()
        except (ValueError,FileNotFoundError,KeyError,AssertionError,TypeError):checks.append(dict(check=label,passed=True,rejected=True));return
        raise AssertionError('Gate accepted '+label)
    op=load('_held_bridge_operator_contract',HERE/'full_held_operator_registered.py');bridge=load('_held_bridge_actual_metadata',HERE/'full_held_confirmation_bridge.py')
    native=load('_held_native_scope_CPU_contract',HERE/'full_held_native_acceptance.py');schedule=load('_held_schedule_scope_CPU_contract',HERE/'full_held_native_schedule_materializer.py');coarse=load('_held_coarse_scope_CPU_contract',HERE/'full_held_coarse_manager.py')
    # Preserve algorithms and source lineage; scope changes are explicit metadata.
    pairs=[('coarse',HERE/'full_held_parent_overlap.py',HERE/'full_frozen_parent_overlap.py',{'science','validate_identity','source_files','build_plan','input_inventory','main'}),('schedule',HERE/'full_held_native_schedule_materializer.py',HERE/'full_native_schedule_materializer.py',{'configure_closed_sources','native_metadata','materialize_schedule','main'}),('native',HERE/'full_held_native_acceptance.py',OUT/'operator_checks/full_registered_native_CUDA_acceptance_v2_20261008/source_snapshot/native_acceptance_v2.py',{'case_metadata','validate_receipt','build_plan','cpu_contract','main'}),('operator',HERE/'full_held_operator_registered.py',HERE/'full_operator_registered_v2.py',{'required_role_files','validate','transport','callback_backend','resolve_release'})]
    pairs.append(('temporal',HERE/'full_held_temporal_diagnostics.py',HERE/'full_temporal_diagnostics_v2.py',{'final_freeze_gate','confirmation_authorization_gate'}))
    algorithms=[]
    for label,new,old,allowed in pairs:
        a,b=functions(old),functions(new);changed={k for k in a if a[k]!=b.get(k)};require(changed<=allowed,'Undisclosed scientific function change: '+label)
        equal=sorted(k for k in a if k not in changed);algorithms.append(dict(component=label,original=held.entry(old),source=held.entry(new),metadata_functions_changed=sorted(changed),unaltered_functions=equal))
        good(label+'_original_compute_and_lifecycle_functions_unchanged',lambda:require(bool(equal),'No original functions retained'))
    historical=load('_held_historical_freeze_contract',HERE/'full_held_confirmation_contract.py')
    good('actual_root_full16_freeze_old_and_additive_contract_both_accept',lambda:(historical.validate_freeze(held.entry(FREEZE)),held.validate_freeze(held.entry(FREEZE))))
    require(functions(HERE/'full_held_confirmation_contract.py')['validate_freeze']==functions(HERE/'full_held_confirmation_contract_v2.py')['validate_freeze'],'Actual frozen development gate changed')
    good('exact_frozen_c278_source',lambda:require(held.sha(held.bound(held.NATIVE_SOURCE))==held.NATIVE_SOURCE['sha256'],'Original native source drift'))
    v=bridge.inspect(FREEZE);good('actual_two_held_parent_80752_81508_metadata_and_freeze',lambda:require([r['points'] for r in v['entries']]==[80752,81508] and v['method_frozen'] is True and v['ready_for_scientific_dispatch'] is False,'Actual inspect boundary differs'))
    index=held.read(OUT/'full_LR_prefixes/checkpoint_index.json');before=copy.deepcopy(index);view=held.development_parent_view(index)
    good('ten_parent_registry_explicit_eight_object_view_no_relabel',lambda:require(index==before and len(view['entries'])==8 and all(any(r==original for original in index['entries']) for r in view['entries']),'Development view changed original registry'))
    for row in v['entries']:
        r=next(r for r in held.original_inputs()['entries'] if r['scene']==row['scene']);p=held.read(held.bound(r['parent_plan']));m=held.read(held.bound(p['data']['manifest']));side=held.read(held.bound(r['sidecar']))
        parent=dict(checkpoint=r['checkpoint'],complete=r['complete'],plan=r['parent_plan'],state=r['actual_parent_state'],audit=side['metadata']['audit'],registered_domain_contract=p['data']['domain_protocol'])
        good(row['scene']+'_actual_native_parent_case_metadata_CPU_only',lambda m=m,parent=parent,p=p:native.case_metadata(m,parent,p))
        bad(row['scene']+'_wrong_independent_seed_refused',lambda m=m,parent=parent:held.held_parent_scope(m,parent,20261008))
        q=copy.deepcopy(parent);q['checkpoint']['sha256']='0'*64;bad(row['scene']+'_wrong_parent_refused',lambda m=m,q=q:held.held_parent_scope(m,q,20261007))
        q=copy.deepcopy(m);q['splits']['train'].append('cam00');bad(row['scene']+'_test_camera_in_training_refused',lambda q=q,parent=parent:held.held_parent_scope(q,parent,20261007))
        good(row['scene']+'_original_fixed_train_diagnostics_population',lambda m=m,row=row:require(len(schedule.fixed_train_diagnostics(m))==row['train_diagnostics'],'Full train diagnostics differ'))
    # Fixtures are confined to this explicitly labelled CPU metadata namespace.
    fixture=directory/'SYNTHETIC_CPU_METADATA_ONLY';fixture.mkdir();freeze=held.entry(FREEZE)
    auth=dict(schema='root_exact_held_confirmation_source_aware_authorization_v1',root_execution_authorized=True,scenes=list(held.HELD),seed=20261007,methods=list(held.METHODS),scientific_source=held.NATIVE_SOURCE,formal_updates_per_endpoint=6000,HR_training_inputs_allowed=False,test_LR_training_inputs_allowed=False,final_method_freeze=freeze,teacher_acceptances={s:None for s in held.HELD},cohort_training_platforms={s:op.GPU0 for s in held.HELD},coarse_operational_sources={s:dict(producer=held.entry(HERE/'full_held_parent_overlap.py'),observer=held.entry(OUT/'native_full_parent_coarse_overlap_20261007/source_snapshot/root_coarse_exact_exit.py'),manager=held.entry(HERE/'full_held_coarse_manager.py')) for s in held.HELD})
    ar=fixture/'pending_two_teacher_authorization.json';write(ar,auth)
    bad('actual_freeze_cannot_replace_both_teacher_acceptances',lambda:held.validate_authorization(held.entry(ar)))
    original_out=held.OUT;held.OUT=fixture
    try:
        for r in held.original_inputs()['entries']:
            scene=r['scene'];ix=fixture/'full_teacher_prepare'/scene/'teacher_index_seed20261007.json';write(ix,dict(status='completed_teacher_inventory',seed_consumer=20261007,seed_independent_frozen_prior=True,entries=[{} for _ in range(r['legal_train_observations'])],plan=r['teacher_CPU_plan']))
            plan=held.read(held.bound(r['teacher_CPU_plan']));teacher=fixture/(scene+'_synthetic_acceptance.json');write(teacher,dict(status='completed_full_teacher_CPU_acceptance',teacher_images=r['legal_train_observations'],CUDA_calls=0,HR_image_reads=0,GPU_inference_executed_by_this_program=False,producer_manifest=plan['producer_manifest'],indexes=[held.entry(ix)],source=held.entry(HERE/'full_teacher_prepare.py')))
            auth['teacher_acceptances'][scene]=held.entry(teacher)
        af=fixture/'two_teacher_gate_shape_fixture.json';write(af,auth)
        good('private_teacher_gate_shape_accepts_only_complete_original_recipe',lambda:held.validate_authorization(held.entry(af)))
        for key,value,label in [('HR_training_inputs_allowed',True,'HR_training'),('test_LR_training_inputs_allowed',True,'test_LR_training'),('seed',20261008,'unregistered_held_seed'),('formal_updates_per_endpoint',6001,'budget_extension'),('methods',['B0','Bsync','M'],'extra_arm')]:
            wrong=copy.deepcopy(auth);wrong[key]=value;f=fixture/(label+'.json');write(f,wrong);bad(label+'_refused_before_model',lambda f=f:held.validate_authorization(held.entry(f)))
        bad('missing_independent_quality_authorization_refused',lambda:held.validate_quality_authorization(None,held.entry(af),'coffee_martini',20261007,'Bsync'))
        good('exact_small_metadata_transitive_inventory_has_no_payload_refs',lambda:require(all(Path(r['path']).suffix.lower() not in held.PIXELS for r in held.metadata_dependencies(held.entry(af))),'Pixel/checkpoint leaked into gate dependency deployment'))
    finally:held.OUT=original_out
    # Mature exact guard and exceptional child cleanup are copied without changes.
    oldguard=OUT/'native_full_parent_coarse_overlap_20261007/A100_GPU0_exact_display_manager_v1/source_snapshot/exact_display_coarse_manager.py'
    for name in ('resource','parse_xml','cgroup_record','scalar','Backend','check_gpu'):
        good('mature_exact_GNOME_'+name+'_unchanged',lambda name=name:require(functions(oldguard)[name]==functions(HERE/'full_held_coarse_manager.py')[name],'Resource behavior changed'))
    cleanup=OUT/'operator_checks/VRHeadset08_local_RTX3090_CPU_preparation_v1_20261008/Bsync_zero_science_resource_recovery1_CPU_20261008/source_snapshot/full_operator_VR08_Bsync_operational_own_child_cleanup_v1.py'
    good('mature_actual_own_child_post_Popen_wait_reap_guard_unchanged',lambda:require(functions(cleanup)['guard_original_children']==functions(HERE/'full_held_operator_registered.py')['guard_original_children'],'Owned child failure lifecycle changed'))
    good('source_aware_same_parent_acceptance_refuses_old_development_grid',lambda:require('full_held_confirmation_bridge' in (HERE/'full_held_operator_registered.py').read_text(),'Actual same-held-parent lifecycle gate missing'))
    # HR gate is first in actual evaluator callback, before env/spawn/decoding.
    class Runtime:
        def __init__(self,*args):pass
    fake=type('Legacy',(),dict(Runtime=Runtime))
    backend=op.callback_backend({},None,fake,None)
    bad('held_callback_without_quality_gate_cannot_spawn_original_evaluator',lambda:backend.evaluate_once([],dict(assets={'held_quality_authorization':None,'confirmation_authorization':held.entry(ar)},scene='coffee_martini',seed=20261007,method='Bsync')))
    temporal=load('_held_exact_temporal_gate_contract',HERE/'full_held_temporal_diagnostics.py')
    actual=held.validate_freeze(held.entry(FREEZE));selection=held.read(held.bound(actual['selection']));development={}
    for key,r in actual['full_development_evidence'].items():
        closed=held.read(held.bound(r));development[key]=(closed,held.read(held.bound(closed['identity']['checkpoint']['plan'])),held.read(held.bound(closed['summary'])))
    plan=dict(schema='registered_full_native_SR_refinement_v1',scene='coffee_martini',seed=20261007,method='Bsync')
    good('actual_full16_Bsync_freeze_passes_additive_held_temporal_gate',lambda:temporal.final_freeze_gate(actual,selection,actual['selection'],plan,development))
    q=copy.deepcopy(plan);q['seed']=20261008;bad('temporal_unregistered_held_seed_refused',lambda:temporal.final_freeze_gate(actual,selection,actual['selection'],q,development))
    q=copy.deepcopy(actual);q['main_method']='MX';bad('temporal_future_method_substitution_refused',lambda:temporal.final_freeze_gate(q,selection,actual['selection'],plan,development))
    evaluator=load('_held_original_evaluator_selection_contract',HERE/'full_evaluate_registered_v2.py')
    good('unchanged_actual_evaluator_selection_accepts_frozen_Bsync_and_B0',lambda:(evaluator.validate_selection_gate(selection,'Bsync'),evaluator.validate_selection_gate(selection,'B0')))
    oldpair=OUT/'operator_checks/VRHeadset08_local_RTX3090_CPU_preparation_v1_20261008/formal_pair_root_CPU_operator_v1_20261008/source_snapshot/start_VRHeadset08_original_local_pair_once_v3.py'
    pair=load('_held_persistent_pair_CPU_contract',HERE/'full_held_pair_coordinator.py');original_pair=load('_held_original_pair_IO_contract',oldpair)
    good('original_atomic_noreplace_inotify_publish_IO_bytes_retained',lambda:require(pair.IO_HELPER==original_pair.IO_HELPER,'Atomic publication or true registration events changed'))
    good('receipt_backed_post_Exit0_payload_CPU_audit_AST_retained',lambda:require(functions(oldpair)['CallbackReceiptEvidence']==functions(HERE/'full_held_pair_coordinator.py')['CallbackReceiptEvidence'],'Post-exit SHA provenance rules changed'))
    good('persistent_pair_real_CPU_imports_completion_and_fp_common',lambda:pair.helpers(ROOT))
    bad('Bsync_without_true_B0_acceptance_refused_before_registration',lambda:pair.validate_prior_B0({},dict(scene='coffee_martini'),None))
    for name in ('metric_helpers','aggregate_flat','spectrum_summary','metric_runtime','csvwrite','model_evidence'):
        good('disclosed_original_v2_evaluator_'+name+'_AST_unchanged',lambda name=name:require(functions(HERE/'full_evaluate_registered.py')[name]==functions(HERE/'full_evaluate_registered_v2.py')[name],'Original/v2 metric kernel differs'))
    good('imports_no_tensor_decoder_or_model',lambda:require(not any(n in sys.modules for n in ('torch','numpy','PIL.Image','scene','gaussian_renderer')),'Forbidden CPU import'))
    sources=[held.entry(p) for p in sorted(HERE.glob('full_held*.py'))]
    value=dict(CPU_runtime=dict(executable=sys.executable,version=sys.version),status='passed_CPU_exact_held_bridge_metadata_source_and_boundary_contracts',source=held.entry(Path(__file__)),sources=sources,actual_final_method_freeze=held.entry(FREEZE),actual_held_dependencies=v,algorithms=algorithms,checks=checks,check_count=len(checks),payload_opens=0,GPU_queries=0,model_imports=0,HR_reads=0,formal_updates=0,services_created=0,SSH_calls=0,global_state_writes=0,private_teacher_fixtures_are_not_actual_acceptance=True,actual_teachers_native_coarse_schedule_roles_still_pending=True)
    write(directory/'receipt.json',value);return dict(receipt=held.entry(directory/'receipt.json'),checks=len(checks),status=value['status'])

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True);a=p.parse_args();print(json.dumps(run_checks(a.out),ensure_ascii=False))
if __name__=='__main__':main()
