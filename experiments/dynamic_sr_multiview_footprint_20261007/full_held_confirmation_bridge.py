"""CPU materialization around the unchanged c278 native scientific plan.

No training, GPU queries, services, HR reads, method selection or freeze occur.
The root later closes operational roles and dispatches the original science
only after exact source/parent/native/assignment/teacher gates are satisfied.
"""
from pathlib import Path
import argparse, hashlib, importlib.util, json, os, sys, time
import full_held_confirmation_contract_v2 as held

ROOT=held.ROOT;HERE=held.HERE;OUT=held.OUT
def require(ok,message):held.require(ok,message)
def load(name,p):
    if str(HERE) not in sys.path:sys.path.insert(0,str(HERE))
    return held.load(name,p)
def write(p,v):
    p=held.path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(v,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    p.chmod(0o444)
def inputs():
    rows=[]
    for r in held.original_inputs()['entries']:
        p=held.read(held.bound(r['parent_plan']));m=held.read(held.bound(p['data']['manifest']));side=held.read(held.bound(r['sidecar']))
        parent=dict(checkpoint=r['checkpoint'],complete=r['complete'],plan=r['parent_plan'],state=r['actual_parent_state'],audit=side['metadata']['audit'])
        held.held_parent_scope(m,parent,20261007)
        rows.append(dict(scene=r['scene'],seed=20261007,manifest=p['data']['manifest'],parent=r['checkpoint'],parent_complete=r['complete'],parent_acceptance=r['actual_parent_acceptance'],points=r['final_parent_points'],legal_train=r['legal_train_observations'],test_keys=[['cam00',f] for f in range(300)],train_diagnostics=300+4*(len(m['splits']['train'])-1),teacher_plan=r['teacher_CPU_plan'],teacher_index_exists=(OUT/'full_teacher_prepare'/r['scene']/'teacher_index_seed20261007.json').exists(),native_same_parent_acceptance=None,coarse_actual_acceptance=None,schedule=None,B0_scientific_plan=None,Bsync_scientific_plan=None))
    return rows
def inspect(freeze=None):
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU explicit empty CUDA visibility required')
    f=held.entry(freeze) if freeze else None
    if f:held.validate_freeze(f)
    return dict(schema='exact_held_confirmation_CPU_dependencies_v1',status='actual_held_parents_verified_final_freeze_and_teacher_native_assignment_gates_pending',source=held.entry(__file__),scientific_source=held.NATIVE_SOURCE,entries=inputs(),final_method_freeze=f,confirmation_authorization=None,formal_budget_if_root_freezes=dict(endpoints=4,updates=24000,RGB=72000,moments=0,native_Adam=24000,first100_included=True),method_frozen=bool(f),method_selected=False,ready_for_scientific_dispatch=False,HR_reads=0,GPU_queries=0,model_imports=0)
def validate_native(reference,parent,authorization,terminal_reference):
    a=held.validate_authorization(authorization);value=held.read(held.bound(reference));helper=load('_held_original_native_acceptance',HERE/'full_held_native_acceptance.py');helper.validate_receipt(value)
    require(value['source_sha256']==held.NATIVE_SOURCE['sha256'],'Native scientific source differs')
    results=[r for r in value['results'] if r['scene']==parent['registered_domain_contract']['scene'] and r['seed']==20261007 and held.same(r['parent']['checkpoint'],parent['checkpoint'])]
    require(len(results)==1 and results[0]['parent']==parent,'A real same-held-parent native acceptance is required, not generic grid coverage')
    t=held.read(held.bound(terminal_reference));require(t['schema']=='root_actual_held_native_parent_and_private_child_Exit0_v1' and t['native_acceptance']==reference and t['helper_source']==held.entry(HERE/'full_held_native_acceptance.py') and t['owned_cgroup_empty'] is True and t['private_child_Exit0_reaped'] is True,'True actual native parent/child owned lifecycle proof missing')
    terminal=t['terminal'];s=terminal['systemd'];r=held.read(held.bound(t['registration']))
    corepath=OUT/'operator_checks/native_full_SR_production_runtime_20261007/source_snapshot/production_core_v3.py';require(held.sha(corepath)=='590f38a55da141de8a9617c66484d7cad1dbc36ba5ae5411bb11030972f73d2d','Original typed terminal core differs');load('_held_typed_native_parent_terminal',corepath).exact_success(terminal,r)
    require(terminal['kind']=='terminal' and terminal['exit_proved'] is True and terminal['exit_kind']==1 and terminal['exit_code']==0 and s['MainPID']=='0' and s['ExecMainCode']=='1' and s['ExecMainStatus']=='0' and s['Result']=='success','Native parent true successful exit absent')
    require(terminal['unit']==r['unit'] and terminal['invocation_id']==r['invocation_id'] and int(s['ExecMainPID'])==r['pid'] and str(s['ExecMainStartTimestampMonotonic'])==str(r['start_monotonic']),'Native exact owned unit/invocation/PID/start differs')
    require(results[0]['hardware']['gpu_uuid']==a['cohort_training_platforms'][parent['registered_domain_contract']['scene']],'Native acceptance is not on this exact paired training platform')
    return True
def prepare(a):
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU explicit empty CUDA visibility required')
    ar=held.entry(a.confirmation_authorization);authorization=held.validate_authorization(ar,a.scene,20261007)
    require(a.method in held.METHODS and a.scene in held.HELD,'Only B0/Bsync slots for the exact two held scene seed07 cohorts')
    source=held.bound(held.NATIVE_SOURCE);native=load('_held_unchanged_c278_science',source)
    require(not any(k in sys.modules for k in ('torch','numpy','PIL.Image','scene','gaussian_renderer')),'CPU bridge must not import model/tensor/decoder')
    row=next(r for r in inputs() if r['scene']==a.scene)
    upstream=Path(a.upstream).resolve();require(os.environ.get('FOURDSR_UPSTREAM')==str(upstream),'Explicit original pinned upstream path required')
    na=native.parser().parse_args(['--mode','plan','--manifest',str(held.path(row['manifest']['path'])),'--parent',str(held.path(row['parent']['path'])),'--parent-complete',str(held.path(row['parent_complete']['path'])),'--selection',str(held.bound(held.read(held.bound(authorization['final_method_freeze']))['selection'])),'--teacher',str(OUT/'full_teacher_prepare'/a.scene/'teacher_index_seed20261007.json'),'--schedule',str(held.path(a.schedule)),'--seed','20261007','--method',a.method,'--upstream',str(upstream),'--steps','6000','--topology-clock','refinement','--out',str(held.path(a.out)/'science')])
    plan=native.build_plan(na);require(plan['status']=='registered_ready_full_native_SR_refinement' and not plan['missing'],'All original scientific dependencies must actually close')
    expected=held.held_parent_scope(held.read(held.bound(row['manifest'])),plan['parent'],20261007)
    require(plan['planned_cost']['SR_updates']==6000 and plan['planned_cost']['RGB_forwards']==18000 and plan['planned_cost']['moment_forwards']==0 and plan['planned_cost']['Adam']==6000,'Unchanged original single native Adam budget required')
    require(len(plan['training_files']['LR'])==len(plan['training_files']['teacher'])==expected['legal_train_observations'],'Whole LR/teacher population required')
    nref=held.entry(a.native_acceptance);tref=held.entry(a.native_parent_terminal);validate_native(nref,plan['parent'],ar,tref)
    schedref=held.entry(a.schedule_complete);sched=held.read(held.bound(schedref));require(sched['source']==held.entry(HERE/'full_held_native_schedule_materializer.py') and sched['confirmation_authorization']==ar and sched['full_native_context']==plan['full_native_context'],'Actual held schedule/source/context authorization differs')
    assignment=sched['assignment_acceptance'];require(assignment is not None,'Root actual overlap/storage/cost assignment acceptance remains pending');value=held.read(held.bound(assignment));require(value['status']=='root_accepted_actual_frozen_parent_coarse_overlap_assignment' and value['source_registration_accepted'] is True and value['storage_and_cost_accepted'] is True and value['full_native_context']==plan['full_native_context'],'Root actual same-parent full assignment acceptance required')
    closure=dict(schema='exact_held_confirmation_c278_CPU_scientific_plan_v1',status='actual_original_c278_scientific_plan_registered_operational_role_closure_pending',source=held.entry(__file__),scientific_source=held.NATIVE_SOURCE,scene=a.scene,seed=20261007,method=a.method,confirmation_authorization=ar,native_acceptance=nref,native_parent_terminal=tref,schedule_complete=schedref,assignment_acceptance=assignment,scientific_plan=None,training_platform=authorization['cohort_training_platforms'][a.scene],same_suffix_both_arms_same_platform_required=True,training_budget=dict(updates=6000,RGB=18000,moments=0,Adam=6000,first100_included=True),HR_training_reads=False,training_dispatch_allowed=False,evaluation_HR_authorization_required=True,operational_roles_closed=False,CPU_only=True)
    out=held.path(a.out);require(out.is_relative_to(OUT/'held_confirmation_materialized') and not out.exists(),'Fresh isolated held materialization output required')
    out.mkdir(parents=True);write(out/'scientific_plan.json',plan);closure['scientific_plan']=held.entry(out/'scientific_plan.json');write(out/'complete.json',closure)
    return closure
def register_evaluation(a):
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU explicit empty CUDA visibility required')
    ar=held.entry(a.confirmation_authorization);held.validate_authorization(ar,a.scene,20261007)
    held.validate_quality_authorization(held.entry(a.held_quality_authorization),ar,a.scene,20261007,a.method)
    row=next(r for r in inputs() if r['scene']==a.scene)
    evaluator=load('_held_unchanged_ffdf_evaluator',HERE/'full_evaluate_registered_v2.py')
    require(held.sha(evaluator.__file__)=='ffdfd66ea15e1bac4c5f5e285568d087baac912a06e6d4d7f15845cd37e30199','Original evaluator source differs')
    manifest=held.read(held.bound(row['manifest']));schedule=load('_held_registered_train_diagnostics',HERE/'full_held_native_schedule_materializer.py')
    selection=held.read(held.bound(held.read(held.bound(ar))['final_method_freeze']))['selection']
    out=held.path(a.out);require(out.is_relative_to(OUT/'held_confirmation_materialized') and not out.exists(),'Fresh held protocol namespace required')
    return evaluator.register(held.bound(row['manifest']),out,roi=None,teacher_index=OUT/'full_teacher_prepare'/a.scene/'teacher_index_seed20261007.json',train_keys=schedule.fixed_train_diagnostics(manifest),selection=held.bound(selection))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=('inspect','inventory','plan','register-evaluation'),required=True);p.add_argument('--final-method-freeze');p.add_argument('--held-quality-authorization');p.add_argument('--scene',choices=held.HELD);p.add_argument('--method',choices=held.METHODS);p.add_argument('--confirmation-authorization');p.add_argument('--schedule');p.add_argument('--schedule-complete');p.add_argument('--native-acceptance');p.add_argument('--native-parent-terminal');p.add_argument('--upstream');p.add_argument('--out');a=p.parse_args()
    if a.mode=='inspect':v=inspect(a.final_method_freeze)
    elif a.mode=='inventory':
        require(a.confirmation_authorization is not None,'Actual both-teacher root authorization required')
        v=dict(status='actual_small_transitive_held_gate_inventory',source=held.entry(__file__),entries=held.metadata_dependencies(held.entry(a.confirmation_authorization)),payload_opens=0,dispatch_allowed=False)
    elif a.mode=='register-evaluation':
        for key in ('scene','method','confirmation_authorization','held_quality_authorization','out'):require(getattr(a,key) is not None,'Missing exact quality prerequisite: '+key)
        v=register_evaluation(a)
    else:
        for key in ('scene','method','confirmation_authorization','schedule','schedule_complete','native_acceptance','native_parent_terminal','upstream','out'):require(getattr(a,key) is not None,'Missing exact actual prerequisite: '+key)
        v=prepare(a)
    print(json.dumps(v,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
