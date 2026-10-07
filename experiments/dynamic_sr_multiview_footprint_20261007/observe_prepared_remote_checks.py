"""CPU structure contracts; fake transport performs no SSH/service/GPU calls."""
import ast,copy,json,os,time
from pathlib import Path
if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('Use a fresh CUDA_VISIBLE_DEVICES empty process')
import torch
assert not torch.cuda.is_initialized()
torch.set_num_threads(1)
import observe_prepared_remote as observer
from fp_common import HERE,OUT,read,write,entry


def main():
    started=time.monotonic();target=OUT/'operator_checks/observe_prepared_remote'/str(time.time_ns());target.mkdir(parents=True)
    cfg,spec,identity=observer.validate_inputs(OUT/'advance/root_registration_v1.json',OUT/'completion/prepared_remote_early_dispatch.json')
    passed=['actual_dispatch_root_registration_source_owner_GPU_WS_protocol_SHA_bindings'];instances=[]
    def terminal(code=0,load='loaded'):
        return dict(kind='terminal',unit=spec['name'],invocation_id=spec['invocation_id'],exit_proved=True,pidfd_exit_proof=True,exit_kind=1,exit_code=code,successful=code==0,systemd=dict(LoadState=load,MainPID='0',ExecMainPID=str(spec['main_pid']),ExecMainExitTimestampMonotonic='100',InvocationID=spec['invocation_id'],ExecMainCode='1',ExecMainStatus=str(code)))
    def factory(value=None,wait_error=False,bundle_error=False,network_interruptions=0):
        class Fake:
            def __init__(self,*args):self.events=[];self.remaining=network_interruptions;instances.append(self)
            def wait_unit(self,received):
                assert received==spec;self.events.append('wait_exact_unit')
                if self.remaining:self.remaining-=1;raise ConnectionError('fake CPU transient transport interruption')
                if wait_error:raise ValueError('fake CPU observer identity error')
                return value
            def current_task(self):self.events.append('current_task');return 'r2_B0',dict(task='r2_B0',status='failed_CPU_fixture')
            def bundle(self,purpose,task,terminal=None):
                assert (purpose,task)==('task','r2_B0');self.events.append(('bundle',terminal))
                if bundle_error:raise IOError('fake CPU partial return error')
                return dict(status='fake_CPU_partial_return_receipt',purpose=purpose,task=task)
            def close(self):self.events.append('close')
        return Fake
    v=observer.observe(cfg,spec,identity,factory(terminal()),target/'success.json')
    assert v['exact_normal_Exit0'] and instances[-1].events==['wait_exact_unit','close'];passed.append('normal_Exit0_preserves_exact_terminal_and_closes_transport')
    t=terminal(2);v=observer.observe(cfg,spec,identity,factory(t),target/'worker_failure.json')
    assert not v['exact_normal_Exit0'] and v['terminal']==t and instances[-1].events==['wait_exact_unit','current_task',('bundle',t),'close'];passed.append('true_failure_returns_current_task_partial_bundle_with_real_terminal')
    v=observer.observe(cfg,spec,identity,factory(wait_error=True),target/'observer_failure.json')
    assert v['terminal'] is None and 'partial_return' not in v and instances[-1].events==['wait_exact_unit','close'];passed.append('nonnetwork_observer_error_keeps_exit_unknown_no_worker_failure_or_partial_claim')
    delays=[];v=observer.observe(cfg,spec,identity,factory(terminal(),network_interruptions=2),target/'network_recovery.json',retry_backoff=lambda attempt:delays.append(attempt))
    network=read(target/'network_recovery.network.json')
    assert v['exact_normal_Exit0'] and v['network_interruptions']==2 and delays==[0,1] and [e['attempt'] for e in network['events']]==[0,1]
    assert all((e['unit'],e['invocation_id'],e['main_pid'])==(spec['name'],spec['invocation_id'],spec['main_pid']) and not e['worker_exit_observed'] for e in network['events'])
    assert instances[-1].events==['wait_exact_unit']*3+['close'];passed.append('network_interruptions_backoff_same_exact_identity_then_true_Exit0_not_model_failure')
    v=observer.observe(cfg,spec,identity,factory(terminal(1),bundle_error=True),target/'bundle_failure.json')
    assert 'partial_return_error' in v and not v['exact_normal_Exit0'] and instances[-1].events[-1]=='close';passed.append('partial_return_failure_retains_error_then_finally_close')
    v=observer.observe(cfg,spec,identity,factory(terminal(load='not-found')),target/'GC_unknown.json')
    assert not v['exact_normal_Exit0'] and v['status']=='failed_prepared_remote_worker_observer_saved';passed.append('GC_manager_default_success_cannot_become_Exit0')
    t=terminal();t['invocation_id']='f'*32;v=observer.observe(cfg,spec,identity,factory(t),target/'wrong_invocation.json')
    assert not v['exact_normal_Exit0'];passed.append('foreign_invocation_refused')
    old=(target/'success.json').read_bytes();count=len(instances)
    try:observer.observe(cfg,spec,identity,factory(terminal()),target/'success.json')
    except FileExistsError:pass
    else:raise AssertionError('Prior observer receipt overwritten')
    assert (target/'success.json').read_bytes()==old and len(instances)==count;passed.append('existing_observer_result_not_overwritten_or_redispatched')
    tree=ast.parse(Path(observer.__file__).read_text());calls=[n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
    assert not set(calls)&{'dispatch','properties','idle_gpu','run','Popen','sleep','poll','step','backward'};passed.append('source_has_no_dispatch_GPU_query_service_shell_timed_poll_or_Adam_calls')
    parser=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main');flags=[n.args[0].value for n in ast.walk(parser) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='add_argument']
    assert flags==['--registration','--dispatch'];passed.append('CLI_only_explicit_registration_and_dispatch')
    assert not torch.cuda.is_initialized()
    frozen=target/'source_snapshot';frozen.mkdir()
    for p in (Path(observer.__file__),Path(__file__)):(frozen/p.name).write_bytes(p.read_bytes())
    receipt=target/'receipt.json';write(receipt,dict(status='passed_prepared_remote_observer_CPU_structure_contracts',checks=passed,count=len(passed),source=entry(Path(observer.__file__)),checks_source=entry(Path(__file__)),actual_identity=identity,CUDA_VISIBLE_DEVICES='',cuda_initialized_before=False,cuda_initialized_after=False,fake_transport=True,actual_SSH_calls=0,service_calls=0,GPU_calls=0,Adam_calls=0,actual_worker_exit_observed=False,wall_seconds=time.monotonic()-started));print(json.dumps(dict(receipt=entry(receipt),source=entry(Path(observer.__file__)),checks_source=entry(Path(__file__)),count=len(passed)),indent=2))

if __name__=='__main__':main()
