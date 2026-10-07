"""CPU contracts for the event controller; never starts SSH/services/GPU work."""
from __future__ import annotations
import copy, json, socket, tempfile, threading, time
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
import advance_after_preparation as a
from fp_common import ROOT, HERE, OUT, entry, read, write, sha


def config():
    source=entry(HERE/'completion.py')
    units={label:dict(name='fixture-'+label+'.service',invocation_id=str(i)*32,main_pid=100+i,source=source) for i,label in enumerate(('prepare_return','overlap','teacher','evaluation'),1)}
    units['overlap']['source']=entry(HERE/'overlap_baseline.py')
    return dict(schema=1,workspace=a.WORKSPACE,remote_host='a100-train',units=units,sources=[entry(HERE/name) for name in ('advance_after_preparation.py','completion.py','run_suite.py','summarize.py')],hardware={'r1':dict(GPU='GPU-pro',model='NVIDIA RTX PRO 6000 Blackwell Workstation Edition',host=socket.gethostname()),'r2':dict(GPU='GPU-a100',model='NVIDIA A100-SXM4-40GB',host='fixture-remote'),'evaluation':dict(GPU='GPU-3090',model='NVIDIA GeForce RTX 3090',host=socket.gethostname())},local_cpu_python='/usr/bin/python3',remote_cpu_python='/usr/bin/python3',local_train_python='/home/cai_tianshun/Project/4dgs/.venv/bin/python',remote_train_python='/home/ubuntu/3DGS/4dsr/.venv/bin/python',remote_activate='/home/ubuntu/3DGS/4dsr/activate_a100.sh')


def terminal(spec,success=True,proved=True):
    kind=1 if success else 2;code=0 if success else 15
    return dict(kind='terminal',unit=a.unit_name(spec['name']),invocation_id=spec['invocation_id'],exit_proved=proved,exit_kind=kind if proved else 0,exit_code=code,successful=success and proved,retained_ExecMainPID_and_start_match=True,systemd=dict(LoadState='loaded',MainPID='0',InvocationID=spec['invocation_id'],ExecMainPID=str(spec.get('main_pid',101)),ExecMainCode=str(kind if proved else 0),ExecMainStatus=str(code),ExecMainStartTimestampMonotonic='10',ExecMainExitTimestampMonotonic='20' if proved else '0',Result='success' if success else 'signal'))


class Backend:
    def __init__(self,cfg,repeat,events=None):self.cfg=cfg;self.repeat=repeat;self.events=events if events is not None else [];self.table={};self.dispatches=[];self.waits=[];self.failed=set();self.unknown=set();self.partial=[];self.busy=False;self.closed=False
    def properties(self,name):return dict(self.table.get(a.unit_name(name),{'LoadState':'not-found','Result':'success','ExecMainStatus':'0'}))
    def idle_gpu(self,gpu):
        self.events.append(('idle',self.repeat))
        if self.busy:raise ValueError('foreign compute')
        return a.parse_idle_gpu(gpu,[gpu['GPU']+', '+gpu['model']+', 0, 10000, 0\n',''])
    def dispatch(self,command):
        self.events.append(('dispatch',self.repeat));self.dispatches.append(command)
        name=next(w.split('=',1)[1] for w in command if w.startswith('--unit='));owner=next(w.split('=',1)[1] for w in command if w.startswith('--setenv='))
        self.table[name]=dict(LoadState='loaded',MainPID='201',ExecMainPID='201',InvocationID=('a' if self.repeat=='1' else 'b')*32,Type='exec',RemainAfterExit='yes',Description=next(w[len('--property=Description='):] for w in command if w.startswith('--property=Description=')),Environment=owner)
        return dict(returncode=0,stdout='fixture dispatch',stderr='')
    def wait_unit(self,spec):
        name=a.unit_name(spec['name']);self.events.append(('wait',name));self.waits.append(name)
        if 'overlap' in name:raise AssertionError('GC overlap manager must never be exact-waited')
        return terminal(spec,name not in self.failed,name not in self.unknown)
    def current_task(self):return 'r2_X',dict(status='failed_actual_training',task='r2_X')
    def bundle(self,purpose,task,terminal):self.partial.append((purpose,task,terminal));return dict(status='completed_received_all_file_SHA_verified',partial=True,task=task)
    def close(self):self.closed=True


def expect_error(fn,label):
    try:fn()
    except (ValueError,AssertionError,RuntimeError,KeyError):return label
    raise AssertionError(label+' unexpectedly accepted')


def main():
    started=time.monotonic();checks=[]
    a.validate_config(config());checks.append('registered_source_hardware_config_CPU')
    cfg=config();reg={'remote_workspace':a.WORKSPACE,'remote_host':'a100-train'};words=['/usr/bin/python3','-u',str(HERE/'completion.py'),'uniform-evaluation','--gpu','GPU-3090']
    a.validate_evaluator_command(words,cfg,reg);checks.append('bound_completion_default_host_workspace_flags_accepted_CPU')
    a.validate_evaluator_command(words+['--workspace',a.WORKSPACE,'--host','a100-train'],cfg,reg);checks.append('matching_explicit_host_workspace_flags_accepted_CPU')
    checks.append(expect_error(lambda:a.validate_evaluator_command(words+['--workspace','/wrong/workspace'],cfg,reg),'wrong_explicit_workspace_rejected'))
    checks.append(expect_error(lambda:a.validate_evaluator_command(words+['--host','other-host'],cfg,reg),'wrong_explicit_host_rejected'))
    actual_reg_path=OUT/'completion/registration.json'
    if actual_reg_path.exists():
        actual=read(actual_reg_path);path=Path('/proc')/str(actual['pid'])/'cmdline'
        if path.exists():
            actual_words=[x.decode() for x in path.read_bytes().split(b'\0') if x];cfg['hardware']['evaluation']['GPU']=actual['GPU'];cfg['units']['evaluation']['source']=actual['source'];a.validate_evaluator_command(actual_words,cfg,actual);checks.append('actual_live_root_evaluator_default_CLI_CPU_accepted')
    spec=config()['units']['teacher'];bad=terminal(spec);bad['systemd']['LoadState']='not-found'
    checks.append(expect_error(lambda:a.terminal_ok(bad,spec),'GC_default_Result_success_never_exit_proof'))
    bad=terminal(spec);bad['invocation_id']='f'*32
    checks.append(expect_error(lambda:a.terminal_ok(bad,spec),'wrong_invocation_rejected'))
    checks.append(expect_error(lambda:a.terminal_ok(terminal(spec,proved=False),spec),'unknown_exit_rejected'))
    a.terminal_ok(terminal(spec,success=False),spec,False);checks.append('failed_teacher_exact_exit_can_release_resource')
    gpu=config()['hardware']['r1'];busy=gpu['GPU']+', 123, python, 900\n'
    checks.append(expect_error(lambda:a.parse_idle_gpu(gpu,[gpu['GPU']+', '+gpu['model']+', 0, 10000, 0\n',busy]),'foreign_GPU_compute_rejected_no_signals'))
    a.parse_idle_gpu(gpu,[gpu['GPU']+', '+gpu['model']+', 100, 10000, 0\n',gpu['GPU']+', 123, /opt/todesk/ToDesk, 100\n']);checks.append('documented_display_process_only_allowed')
    overlap=a.validate_overlap();assert overlap['manager_exit_proved'] is False;checks.append('actual_B0_closed3000_child_return_SHA_manager_unknown_CPU')
    base=OUT/'advance_checks';base.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='CPU_',dir=base) as directory:
        base=Path(directory)
        def controller(name,cfg=None):
            cfg=copy.deepcopy(cfg or config());events=[];local=Backend(cfg,'1',events);remote=Backend(cfg,'2',events)
            engine=a.Advance(cfg,{'path':'CPU-fixture','sha256':'f'*64},local,remote,False,lambda:dict(status='CPU_finalizer_fixture',no_decision_argument_passed=True),base/name)
            return engine,local,remote,events
        engine,local,remote,events=controller('success');value=engine.run()
        assert value['status']=='ready_for_root_visual_decision' and value['decision_created'] is False and value['new_full_GPU_task_dispatched'] is False
        assert len(local.dispatches)==len(remote.dispatches)==1 and not any('overlap' in w for w in local.waits)
        assert all('--defer-evaluation' in c[-1] and '--operator-resource-resolved' in c[-1] for c in local.dispatches+remote.dispatches)
        assert all('--property=Type=exec' in c and '--property=RemainAfterExit=yes' in c for c in local.dispatches+remote.dispatches)
        assert '/home/ubuntu/3DGS/4dsr/activate_a100.sh' in remote.dispatches[0][-1] and a.WORKSPACE in remote.dispatches[0][-1]
        checks.append('two_fixed_parallel_owned_workers_no_extra_GPU_task_and_no_invented_decision')
        before=(len(local.dispatches),len(remote.dispatches));engine.dispatch('1');engine.dispatch('2');assert before==(len(local.dispatches),len(remote.dispatches));checks.append('restart_attaches_owned_same_invocation_no_double_dispatch')
        local.table[a.UNITS['1']]['InvocationID']='c'*32
        checks.append(expect_error(lambda:engine.dispatch('1'),'restarted_owned_unit_different_invocation_rejected'))
        local.table.clear();checks.append(expect_error(lambda:engine.dispatch('1'),'missing_previously_owned_unit_never_redispatched'))
        engine,local,remote,events=controller('alien');local.table[a.UNITS['1']]=dict(LoadState='loaded',Description='alien')
        checks.append(expect_error(lambda:engine.dispatch('1'),'existing_foreign_unit_not_attached_or_interrupted'));assert not local.dispatches
        engine,local,remote,events=controller('busy');local.busy=True
        result=engine.run();assert result['status']=='failed_paired_dispatch_owned_started_workers_observed_no_success_report' and not local.dispatches and len(remote.dispatches)==1 and a.UNITS['2'] in remote.waits;checks.append('GPU_busy_prelaunch_preserves_foreign_other_independent_owned_worker_observed')
        engine,local,remote,events=controller('remote_dispatch_failure');remote.busy=True
        result=engine.run();assert result['status']=='failed_paired_dispatch_owned_started_workers_observed_no_success_report' and len(local.dispatches)==1 and not remote.dispatches and a.UNITS['1'] in local.waits
        checks.append('second_dispatch_failure_retains_first_worker_exact_exit_observer')
        engine,local,remote,events=controller('prepare_failure');local.failed.add('fixture-prepare_return.service')
        checks.append(expect_error(lambda:engine.run(),'prepare_nonzero_blocks_both_workers'));assert not local.dispatches and not remote.dispatches
        engine,local,remote,events=controller('teacher_failed');local.failed.add('fixture-teacher.service');assert engine.run()['status']=='ready_for_root_visual_decision'
        assert (engine.base/'teacher_failure_independent_core_allowed.json').exists();checks.append('teacher_failed_exact_release_does_not_block_independent_core')
        engine,local,remote,events=controller('teacher_unknown');local.unknown.add('fixture-teacher.service')
        checks.append(expect_error(lambda:engine.run(),'teacher_unknown_exit_blocks_GPU_dispatch'));assert not local.dispatches
        cfg=config();cfg['prerequisite_units']=[dict(name='fixture-prefix.service',invocation_id='d'*32,main_pid=104,source=entry(HERE/'completion.py'))]
        engine,local,remote,events=controller('prefix',cfg);local.failed.add('fixture-prefix.service');assert engine.run()['status']=='ready_for_root_visual_decision'
        assert events.index(('wait','fixture-prefix.service'))<events.index(('dispatch','1'));checks.append('optional_resource_owned_prefix_waits_exit_before_training')
        engine,local,remote,events=controller('remote_failure');remote.failed.add(a.UNITS['2']);result=engine.run()
        assert result['status']=='failed_core_worker_or_observer_no_success_report' and remote.partial[0][:2]==('task','r2_X');checks.append('remote_failure_returns_actual_current_task_partial_and_no_success')
        engine,local,remote,events=controller('eval_failure');local.failed.add('fixture-evaluation.service');assert engine.run()['status']=='failed_uniform_evaluation_no_success_report';checks.append('evaluation_nonzero_never_marks_success')
        engine,local,remote,events=controller('incomplete_callbacks');write(engine.out/'completion/uniform_evaluation_complete.json',dict(status='completed_all_12_uniform_evaluation_callbacks',tasks={},GPU='GPU-3090'))
        checks.append(expect_error(engine.finalize,'fewer_than12_callback_receipts_blocks_CPU_summary'))
        engine,local,remote,events=controller('final_callbacks')
        source=engine.config['units']['evaluation']['source'];artifact=engine.out/'CPU_artifact.json';write(artifact,{'fixture':1});reference=entry(artifact)
        refs={}
        for task in a.TASKS:
            results={key:reference for key in ('training','checkpoint','evaluation','diagnostics')};taskpath=engine.out/'tasks'/(task+'.json');write(taskpath,dict(status='completed_training_evaluation_fixed_diagnostics',**results))
            path=engine.out/'completion/evaluation_receipts'/(task+'.json');write(path,dict(status='completed_uniform_evaluation_event_callback',task=task,GPU='GPU-3090',evaluator_exit_code=0,source=source,results=results,task_receipt=entry(taskpath)));refs[task]=entry(path)
        write(engine.out/'completion/uniform_evaluation_complete.json',dict(status='completed_all_12_uniform_evaluation_callbacks',tasks=refs,GPU='GPU-3090',source=source))
        calls=[]
        def finalize_subprocess(command,**kwargs):
            assert kwargs['env']['CUDA_VISIBLE_DEVICES']=='' and '--decision' not in command;calls.append(command)
            write(engine.out/'suite_integrity.json',dict(status='completed_core',updates=72000,completed=list(a.TASKS)))
            write(engine.out/'summary/complete.json',dict(status='completed_core_quality_evidence',actual_identity_checked_endpoints=12))
            return SimpleNamespace(returncode=0)
        with patch.object(a.subprocess,'run',finalize_subprocess):
            final=engine.finalize();assert final['no_decision_argument_passed'] is True and len(calls)==2
        checks.append('all12_SHA_bound_callbacks_then_CPU_verify_and_summary_no_decision_no_CUDA')
        write(artifact,{'fixture':2});checks.append(expect_error(engine.finalize,'changed_evaluation_artifact_SHA_blocks_finalization'))
        engine,local,remote,events=controller('corrupt_transfer');write(engine.out/'completion/prepare_return_complete.json',dict(status='completed_preparation_return_Exit0_and_all_SHA_verified',terminal=terminal(spec),transfer={'status':'bad'}))
        checks.append(expect_error(lambda:a.validate_preparation_assets(engine.out),'bad_transfer_status_even_when_workspace_field_absent_rejected'))
        engine,local,remote,events=controller('inotify_failure');messages=a.queue.Queue();directory=engine.out/'completion/evaluation_receipts';listener=a.completion.DirectoryEvents(directory)
        watcher=threading.Thread(target=engine.observe_evaluation_failures,args=(directory,'receipts',messages,listener),daemon=True);watcher.start()
        write(directory/'r1_X.json',dict(status='failed_uniform_evaluation_event_callback',task='r1_X',evaluator_exit_code=2))
        kind,_,failure=messages.get(timeout=5);watcher.join(timeout=5)
        assert kind=='evaluation_failure' and failure['evaluation_service_exit_proved'] is False and not watcher.is_alive();checks.append('actual_CPU_inotify_failure_event_no_service_exit_fabricated')
        engine,local,remote,events=controller('recovered_failure');directory=engine.out/'completion';write(directory/'event_failure_123.json',dict(status='failed_queue_event_preserved',error='fixture'))
        messages=a.queue.Queue();engine.observe_evaluation_failures(directory,'queue',messages);assert messages.get_nowait()[0]=='evaluation_failure';checks.append('startup_failure_scan_after_subscribe_no_failure_missed')
        engine,local,remote,events=controller('failed_evaluation_keeps_service_alive')
        original=engine.observe_evaluation
        def event_instead_of_unit_exit(messages):messages.put(('evaluation_failure','',dict(status='failed_uniform_evaluation_event_saved_service_exit_pending',evaluation_service_exit_proved=False)))
        engine.observe_evaluation=event_instead_of_unit_exit;assert engine.run()['status']=='failed_uniform_evaluation_no_success_report';checks.append('evaluation_failed_event_ends_controller_even_without_service_terminal')
        engine,local,remote,events=controller('changed_source');cfg=config();cfg['sources'][0]['sha256']='f'*64
        checks.append(expect_error(lambda:a.validate_config(cfg),'changed_registered_callback_source_SHA_rejected'))
    result=dict(status='passed_CPU_contracts_no_SSH_service_or_GPU_execution',checks=checks,check_count=len(checks),source=entry(HERE/'advance_after_preparation.py'),checks_source=entry(Path(__file__)),seconds=time.monotonic()-started,formal_updates=0,new_model_forwards=0,SSH_calls=0,service_dispatch_calls=0,manager_GC_never_assumed_success=True)
    write(OUT/'advance_checks/complete.json',result);print(json.dumps(result,ensure_ascii=False,indent=2));return result

if __name__=='__main__':main()
