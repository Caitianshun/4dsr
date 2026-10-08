"""CPU contracts for the source-aware operator; fixtures are never GPU evidence."""
import copy, hashlib, importlib.util, json, sys, tempfile
from pathlib import Path
from types import SimpleNamespace

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sp=importlib.util.spec_from_file_location('operator_contract_subject',HERE/'full_operator_registered_v2.py')
op=importlib.util.module_from_spec(sp);sp.loader.exec_module(op)
passed=[]
def good(label,fn):
    fn();passed.append(label)
def bad(label,fn):
    try:fn()
    except (ValueError,KeyError,AssertionError):passed.append(label);return
    raise AssertionError('Invalid CPU fixture accepted: '+label)

assert not any(v in sys.modules for v in ('torch','numpy','scene','gaussian_renderer'))
assert op.sha(HERE/'full_refine_registered_v2.py')==op.NATIVE_SHA
assert op.sha(HERE/'full_evaluate_registered_v2.py')==op.EVALUATOR_SHA
assert op.sha(HERE/'full_refine_registered.py')=='9662c17f32212d6fdbd8c0f1a0cf405274b5239eb1a1a3b48acd6d60a11b6d09'
assert op.sha(HERE/'full_evaluate_registered.py')=='87fd5f4d29453be980704ce7030d8cf1d7d3d35205ea978d1a81e1b7d0d4ab05'
passed.append('four_old_and_v2_scientific_sources_bound_unmodified_no_tensor_import')

s=dict(scene='meetroom_discussion',seed=20261007,method='B0',no_automatic_scientific_retry=True)
index=dict(entries=[dict(task_key=op.task_key(s),status='planned_only',actual_units_started=False)])
good('fresh_exact_endpoint_allowed',lambda:op.validate_no_repeat(s,index))
for field,value,label in [('status','completed','completed_refused'),('actual_units_started',True,'dispatched_refused')]:
    wrong=copy.deepcopy(index);wrong['entries'][0][field]=value
    bad(label,lambda wrong=wrong:op.validate_no_repeat(s,wrong))
wrong=copy.deepcopy(index);wrong['entries']*=2
bad('duplicate_endpoint_refused',lambda:op.validate_no_repeat(s,wrong))

parent=dict(checkpoint={'path':'cpu_fixture.pt','sha256':'a'*64})
plan=dict(author_commit='843d5ac636c37e4b611242287754f3d4ed150144',data={'resolution_LR':[320,180]},parent=parent)
case=dict(LR=[320,180],scene=s['scene'],parent_scene=s['scene'],seed=s['seed'],parent_seed=s['seed'],
          execution='actual_native_CUDA',synthetic=False,parent=parent,checks={'fixture_check':True})
accept=dict(status='passed_native_full_refinement_CUDA_acceptance',source_sha256=op.NATIVE_SHA,
            author_commit=plan['author_commit'],Adam_calls=0,formal_updates=0,observer_RNG_restored=True,
            accepted_grids=[[320,180]],results=[case])
good('synthetic_CPU_acceptance_shape_only_v2_match',lambda:op.validate_acceptance(s,plan,accept,['fixture_check']))
for field,value,label in [('source_sha256','9662c17f32212d6fdbd8c0f1a0cf405274b5239eb1a1a3b48acd6d60a11b6d09','old_science_acceptance_refused'),
                          ('accepted_grids',[[336,252]],'wrong_actual_grid_refused'),('formal_updates',1,'acceptance_updates_refused')]:
    wrong=copy.deepcopy(accept);wrong[field]=value
    bad(label,lambda wrong=wrong:op.validate_acceptance(s,plan,wrong,['fixture_check']))
for field,value,label in [('parent',{},'foreign_parent_refused'),('synthetic',True,'synthetic_GPU_claim_refused'),
                          ('checks',{'fixture_check':False},'failed_native_check_refused')]:
    wrong=copy.deepcopy(accept);wrong['results'][0][field]=value
    bad(label,lambda wrong=wrong:op.validate_acceptance(s,plan,wrong,['fixture_check']))

core_path=ROOT/'output/dynamic_sr_multiview_footprint_20261007/operator_checks/native_full_SR_production_runtime_20261007/source_snapshot/production_core_v3.py'
assert op.sha(core_path)==op.CORE_SHA
core=op.load('frozen_exact_exit_contract_core',core_path)
r=dict(unit='4dsr-footprint-cpu-fixture.service',invocation_id='f'*32,pid=42,start_ticks='100',start_monotonic='100',
       owned_cgroup='/user.slice/4dsr-footprint-cpu-fixture.service',host='fixture',boot_id='fixture-boot',
       empty_cgroup_helper_source={'path':'fixture.py','sha256':'b'*64})
proof=dict(status='verified_owned_cgroup_absent_or_recursively_empty',authority='CPU_proc_cgroup_after_exact_manager_terminal',
           unit=r['unit'],invocation_id=r['invocation_id'],pid=42,start_ticks='100',start_monotonic='100',
           cgroup=r['owned_cgroup'],host=r['host'],boot_id=r['boot_id'],source=r['empty_cgroup_helper_source'],
           observed_monotonic_ns=300000,path_absent=True,observed_cgroup_files=[],all_pids=[],own_PID_absent=True)
t=dict(kind='terminal',successful=True,exit_proved=True,exit_kind=1,exit_code=0,unit=r['unit'],invocation_id=r['invocation_id'],
       authority='persistent_pidfd_and_RefUnit_retained_exact_systemd_ExecMainStatus',pidfd_exit_proof=True,
       initial_main_pid=42,owned_cgroup_exit_proof=proof,
       systemd=dict(Id=r['unit'],LoadState='loaded',MainPID='0',ExecMainPID='42',ExecMainStartTimestampMonotonic='100',
                    ExecMainExitTimestampMonotonic='200',ExecMainCode='1',ExecMainStatus='0',Result='success',
                    InvocationID=r['invocation_id'],TasksCurrent='0'))
good('synthetic_exact_true_exit_shape_unchanged_core',lambda:core.exact_success(t,r))
for label,change in [('nonzero_exit_refused',lambda v:v.update(exit_code=1)),
                     ('wrong_invocation_refused',lambda v:v.update(invocation_id='e'*32)),
                     ('missing_pidfd_proof_refused',lambda v:v.update(pidfd_exit_proof=False)),
                     ('live_descendant_refused',lambda v:v['owned_cgroup_exit_proof'].update(all_pids=[99]))]:
    wrong=copy.deepcopy(t);change(wrong)
    bad(label,lambda wrong=wrong:core.exact_success(wrong,r))

with tempfile.TemporaryDirectory() as temp:
    e=core.Evidence(temp);ref=e.new('cpu_fixture.json',{'fixture':True});reg=e.new('registration.json',r)
    x=dict(s,assets={'training_registration':reg,'plan':ref},sources={'fixture':ref},production_core_source=ref,
           paths={'callback_out':'callback'})
    events=[]
    class Backend:
        fail=None
        def stage(self,name):
            events.append(name)
            if self.fail==name:raise ValueError('Preserved simulated '+name+' failure')
        def wait_training(self,r):self.stage('exact_exit');return t
        def inspect_training(self,s,t):self.stage('all_SHA_return');return ref,[],{}
        def prepare_evaluation(self,command,s):self.stage('evaluation_resource');return ref
        def evaluate_once(self,command,s):self.stage('evaluate');return dict(exit_code=0,private_process_group_reaped=True,complete=ref)
        def return_closed(self,s,refs):
            self.stage('final_return');entries={v['path']:dict(v,bytes=e.path(v['path']).stat().st_size) for v in refs}
            return dict(status='completed_native_full_training_and_evaluation_all_file_SHA_return',entries=list(entries.values()))
    def training(s,e,plan,*args):events.append('integrity');return {'checkpoint':ref,'required_refs':[ref]}
    def evaluation(s,e,cp,*args):events.append('evaluation_integrity');return {'complete':ref,'required_refs':[ref]}
    c=SimpleNamespace(exact_success=core.exact_success,validate_training=training,validate_evaluation=evaluation,
                      evaluator_argv=lambda s,cp:['cpu-fixture-never-executed'],digest=core.digest)
    good('ordered_exit_return_integrity_uniform_eval_and_final_return',lambda:op.completion_flow(x,e,Backend(),c))
    assert events==['exact_exit','all_SHA_return','integrity','evaluation_resource','evaluate','evaluation_integrity','final_return']
    for stage in ('exact_exit','all_SHA_return','evaluation_resource','evaluate','final_return'):
        y=copy.deepcopy(x);y['paths']['callback_out']='failure_'+stage;events.clear();b=Backend();b.fail=stage
        bad('failed_'+stage+'_cannot_close',lambda y=y,b=b:op.completion_flow(y,e,b,c))
        if stage in ('exact_exit','all_SHA_return','evaluation_resource'):assert 'evaluate' not in events

guard_path=ROOT/'output/dynamic_sr_multiview_footprint_20261007/native_full_parent_coarse_overlap_20261007/A100_GPU0_exact_display_manager_v1/source_snapshot/exact_display_coarse_manager.py'
assert op.sha(guard_path)==op.DISPLAY_SHA
guard=op.load('unchanged_display_guard_cpu_contract',guard_path)
own=dict(pid=99,starttime='9',exe='cpu-fixture',exe_sha256='f'*64,cgroup='0::/fixture',cmdline=['fixture'])
v=dict(uuid=guard.GPU,name='NVIDIA A100-SXM4-40GB',driver='fixture',memory_total='40960 MiB',memory_used='442 MiB',
       utilization='0 %',processes=[dict(pid=4147,type='C+G',name=guard.GNOME['exe'],used_memory='420 MiB')])
good('exact_audited_GNOME_allowed_CPU_shape_only',lambda:guard.resource(v,lambda pid:guard.GNOME,own,idle=True))
wrong=copy.deepcopy(v);wrong['processes'].append(dict(pid=123,type='C',name='openpi',used_memory='1 MiB'))
bad('openpi_foreign_compute_protected',lambda:guard.resource(wrong,lambda pid:{},own,idle=True))
wrong=copy.deepcopy(guard.GNOME);wrong['starttime']='changed'
bad('reused_GNOME_PID_refused',lambda:guard.resource(v,lambda pid:wrong,own,idle=True))
assert not any(v in sys.modules for v in ('torch','numpy','scene','gaussian_renderer'))
print(json.dumps(dict(status='passed_CPU_source_aware_operator_v2_contracts',checks=passed,
                     GPU_calls=0,model_imports=0,actual_training_or_acceptance=False),ensure_ascii=False,indent=2))
