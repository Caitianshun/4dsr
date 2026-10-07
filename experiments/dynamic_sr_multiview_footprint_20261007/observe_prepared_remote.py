"""Root-started CPU observer of the already dispatched exact r2 worker.

No worker dispatch, GPU query, training, service mutation or polling is present.
The frozen observer helper waits on pidfd/D-Bus and retains actual exit evidence.
"""
from __future__ import annotations
import argparse,fcntl,hashlib,json,re,shlex,time,traceback
from pathlib import Path
import advance_after_preparation as advance
from fp_common import ROOT,HERE,OUT,entry,bound,local,read,write,sha

ADVANCE_SHA='30c748217f8dfbf47b0fbd3bf3e1f498fdb5c7b4e928993c7144ea92e995795d'
COMPLETION_SHA='e7b55bf9b504c67e4df7480be078e07b761dcca2f9b7d94fd1da2572d1fdcfeb'
EXIT=OUT/'completion/prepared_remote_worker_exit.json'


def validate_inputs(registration,dispatch):
    if sha(HERE/'advance_after_preparation.py')!=ADVANCE_SHA or sha(HERE/'completion.py')!=COMPLETION_SHA:raise ValueError('Frozen event observer dependency changed')
    registration=local(registration);dispatch=local(dispatch)
    cfg=advance.validate_config(read(registration));reg=entry(registration);early=read(dispatch);worker=early['worker']
    if early.get('status')!='registered_owned_remote_suffix_before_local_immutable_return' or early.get('local_preparation_gate_published') is not False or early.get('formal_updates')!=0:raise ValueError('Actual early dispatch identity required')
    if worker.get('status')!='registered_or_attached_persistent_paired_worker' or worker['registration']!=reg or worker['source']!=entry(HERE/'advance_after_preparation.py'):raise ValueError('Worker root registration/source differs')
    if worker['name']!=advance.UNITS['2'] or not re.fullmatch('[0-9a-f]{32}',worker['invocation_id']) or type(worker['main_pid']) is not int or worker['main_pid']<=0:raise ValueError('Actual fixed r2 unit/invocation/PID required')
    identity=worker['identity'];owner=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    expected=dict(repeat='2',unit=advance.UNITS['2'],GPU=cfg['hardware']['r2'],workspace=cfg['workspace'],python=cfg['remote_train_python'])
    if any(identity.get(k)!=v for k,v in expected.items()) or worker['owner']!=owner:raise ValueError('Owned remote worker hardware/workspace/owner differs')
    if identity['protocol']!=entry(OUT/'protocol.json'):raise ValueError('Closed worker protocol identity differs')
    systemd=worker['systemd']
    if systemd.get('InvocationID')!=worker['invocation_id'] or int(systemd.get('ExecMainPID','0'))!=worker['main_pid'] or systemd.get('Type')!='exec' or systemd.get('RemainAfterExit')!='yes':raise ValueError('Dispatched manager identity/retention differs')
    if systemd.get('Description')!='4dsr-footprint paired owner='+owner+' repeat=2' or 'FOURDSR_ADVANCE_OWNER='+owner not in shlex.split(systemd.get('Environment','')):raise ValueError('Exact remote worker ownership markers required')
    bound(early['source']);snapshot=read(bound(early['remote_preparation']))
    if early['source']['path']!=str((HERE/'dispatch_prepared_remote.py').relative_to(ROOT)) or snapshot.get('status')!='closed_remote_preparation_metadata_SHA_verified' or snapshot['workspace']!=cfg['workspace'] or snapshot['preparation']['sources']!=identity['core_sources'] or snapshot['preparation']['protocol']!=identity['protocol']:raise ValueError('Closed remote preparation/launcher binding differs')
    spec={key:worker[key] for key in ('name','invocation_id','main_pid')}
    provenance=dict(registration=reg,dispatch=entry(dispatch),worker=worker,remote_preparation=early['remote_preparation'],sources={p.name:entry(p) for p in (Path(__file__),HERE/'advance_after_preparation.py',HERE/'completion.py',HERE/'fp_common.py')},launcher_source=early['source'],hardware=cfg['hardware']['r2'],workspace=cfg['workspace'],GPU_calls=0,Adam_calls=0,formal_updates_claimed=0)
    return cfg,spec,provenance


def observe(cfg,spec,provenance,transport_factory=advance.A100Transport,output=EXIT,retry_backoff=advance.completion.backoff):
    output=Path(output)
    if output.exists():raise FileExistsError('Independent observer result preserved; no overwrite or redispatch')
    started=time.monotonic();transport=None;terminal=None;terminal_failed=False;network_events=[];result=dict(provenance=provenance,spec=spec)
    try:
        transport=transport_factory(cfg['remote_host'],cfg['workspace'],OUT,cfg['remote_cpu_python'])
        attempt=0
        while True:
            try:terminal=transport.wait_unit(spec);break
            except ConnectionError as error:
                event=dict(status='network_failure_saved_exact_invocation_retry',unit=spec['name'],invocation_id=spec['invocation_id'],main_pid=spec['main_pid'],attempt=attempt,error=repr(error),unix=time.time(),worker_exit_observed=False,provenance=provenance)
                network_events.append(event);write(output.with_name(output.stem+'.network.json'),dict(**event,events=network_events))
                retry_backoff(attempt);attempt+=1
        advance.terminal_ok(terminal,spec,False)
        terminal_failed=not terminal.get('successful')
        result.update(status='prepared_remote_worker_Exit0' if terminal.get('successful') else 'failed_prepared_remote_worker_saved',terminal=terminal)
        if terminal.get('successful'):advance.terminal_ok(terminal,spec)
    except BaseException as error:
        result.update(status='failed_prepared_remote_worker_observer_saved',terminal=terminal,error=repr(error),traceback=traceback.format_exc())
    finally:
        if transport is not None:
            if terminal_failed:
                try:
                    task,status=transport.current_task();result['remote_worker_status']=status;result['current_task']=task
                    if task:result['partial_return']=transport.bundle('task',task,terminal=terminal)
                except BaseException as error:result['partial_return_error']=dict(error=repr(error),traceback=traceback.format_exc())
            try:transport.close()
            except BaseException as error:result.update(status='failed_prepared_remote_worker_observer_cleanup_saved',cleanup_error=repr(error))
        result.update(wall_seconds=time.monotonic()-started,finished_unix=time.time(),exact_normal_Exit0=result.get('status')=='prepared_remote_worker_Exit0',network_interruptions=len(network_events),network_receipt=entry(output.with_name(output.stem+'.network.json')) if network_events else None,GPU_calls=0,Adam_calls=0,observer_dispatched_worker=False,model_polling_used=False)
        write(output,result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--registration',type=Path,required=True);p.add_argument('--dispatch',type=Path,required=True);a=p.parse_args()
    lockpath=OUT/'locks/prepared_remote_worker_observer.lock';lockpath.parent.mkdir(parents=True,exist_ok=True)
    with lockpath.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            if EXIT.exists():raise FileExistsError('Independent observer exit receipt preserved before registration; no overwrite')
            cfg,spec,provenance=validate_inputs(a.registration,a.dispatch)
            write(OUT/'completion/prepared_remote_worker_observer_registration.json',dict(status='registered_CPU_exact_exit_observer_for_existing_worker',provenance=provenance,spec=spec,registered_unix=time.time(),GPU_calls=0))
            result=observe(cfg,spec,provenance)
            if not result['exact_normal_Exit0']:raise RuntimeError('Remote worker or observer failed; independent receipt and partial-return costs saved')
        except BaseException as error:
            target=OUT/'completion/prepared_remote_observer_errors'/f'{time.time_ns()}.json'
            write(target,dict(status='failed_prepared_remote_observer_saved_no_success_report',registration=entry(local(a.registration)) if local(a.registration).is_file() else None,dispatch=entry(local(a.dispatch)) if local(a.dispatch).is_file() else None,source=entry(Path(__file__)),error=repr(error),traceback=traceback.format_exc(),GPU_calls=0,Adam_calls=0,observer_dispatched_worker=False));raise

if __name__=='__main__':main()
