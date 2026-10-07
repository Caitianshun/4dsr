"""Decouple endpoint evaluation from fixed-budget training without altering it.

Use system Python3 for pidfd support. Training CLI is compatible with worker.py;
--evaluate-endpoint is the local durable evaluator service entry point. Existing
services are attached, never restarted or duplicated after SSH interruption.
--cpu-self-test exercises lifecycle and lock handling without any GPU command.
"""
from __future__ import annotations
import argparse
import ctypes
import fcntl
import os
from pathlib import Path
import select
import shlex
import signal
import subprocess
import sys
import tempfile
import time
import traceback
from cg_common import ROOT, HERE, OUT, read, write, bound, sha, entry

PYTHON='/home/cai_tianshun/Project/4dgs/.venv/bin/python'
SYSTEM_PYTHON='/usr/bin/python3'
REMOTE='/home/ubuntu/3DGS/4dsr'
SPLITS=('test','dev','train_fixed')
UNIT_PROPERTIES=('LoadState','ActiveState','SubState','MainPID','Result','ExecMainStatus')


def wait_pid(pid):
    if not pid:return
    try:fd=os.pidfd_open(int(pid))
    except ProcessLookupError:return
    try:select.select([fd],[],[])
    finally:os.close(fd)


def call(command, log, env=None):
    """Wait/reap the complete private child process group before returning."""
    Path(log).parent.mkdir(parents=True,exist_ok=True)
    if ctypes.CDLL(None,use_errno=True).prctl(36,1,0,0,0)!=0:
        raise OSError(ctypes.get_errno(),'Cannot become child subreaper')
    with Path(log).open('a') as f:
        print(shlex.join(command),file=f,flush=True)
        proc=subprocess.Popen(command,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=proc.wait()
        finally:
            # Any descendants left after their leader exits are owned by this
            # subreaper. They must not keep CUDA alive after the lock is released.
            try:os.killpg(proc.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            proc.wait()
            while True:
                try:os.waitpid(-proc.pid,0)
                except ChildProcessError:break
        if code:raise subprocess.CalledProcessError(code,command)


def free_gpu(gpu):
    uuid=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader'],text=True)
    busy=[r for r in processes.splitlines() if uuid in r and '/opt/todesk/' not in r]
    if busy:raise RuntimeError(('GPU occupied outside shared lock',gpu,busy))
    return dict(uuid=uuid,applications=processes)


def ssh(host, command):
    return subprocess.check_output(['ssh','-o','ConnectTimeout=20','-o','ServerAliveInterval=30',host,command],text=True)


def unit_state(unit, host='local'):
    command=['systemctl','--user','show',unit,'--no-pager',*[f'--property={p}' for p in UNIT_PROPERTIES]]
    if host!='local':command=['ssh','-o','ConnectTimeout=20','-o','ServerAliveInterval=30',host,shlex.join(command)]
    p=subprocess.run(command,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    if p.returncode and 'LoadState=not-found' not in p.stdout:
        raise RuntimeError(p.stderr or p.stdout)
    value=p.stdout
    return dict(line.split('=',1) for line in value.splitlines() if '=' in line)


def ensure_unit(unit, launch, host='local', state_fn=None):
    """A stable unit identity makes ambiguous dispatch outcomes recoverable."""
    state_fn=state_fn or (lambda:unit_state(unit,host))
    state=state_fn()
    if state.get('LoadState')=='not-found':
        launch()
        state=state_fn()
    if state.get('LoadState')!='loaded':raise RuntimeError(('Missing durable unit',unit,state))
    if state.get('ActiveState')=='failed':raise RuntimeError(('Existing unit failed; preserved without redispatch',unit,state))
    return state


def wait_unit(unit, host='local'):
    if host=='local':
        state=unit_state(unit);wait_pid(int(state.get('MainPID','0')))
        state=unit_state(unit)
    else:
        # This waiter can disconnect without affecting the independently owned
        # training unit. A new controller attaches to the same unit identity.
        code="""import os,select,subprocess,sys,json
unit=sys.argv[1]
fields=('LoadState','ActiveState','SubState','MainPID','Result','ExecMainStatus')
def state():
 p=subprocess.run(['systemctl','--user','show',unit,'--no-pager',*[f'--property={k}' for k in fields]],text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
 if p.returncode:raise RuntimeError(p.stderr or p.stdout)
 return dict(s.split('=',1) for s in p.stdout.splitlines() if '=' in s)
s=state();pid=int(s.get('MainPID','0'))
if pid:
 try:fd=os.pidfd_open(pid)
 except ProcessLookupError:pass
 else:
  try:select.select([fd],[],[])
  finally:os.close(fd)
print(json.dumps(state()))
"""
        import json
        state=json.loads(ssh(host,shlex.join(['python3','-c',code,unit])).strip())
    if state.get('Result')!='success' or int(state.get('ExecMainStatus','-1'))!=0 or int(state.get('MainPID','0')):
        raise RuntimeError(('Unit did not finish successfully',host,unit,state))
    return state


def verify_training(task, method=None, repeat=None):
    directory=OUT/'runs'/task;receipt=read(directory/'complete.json')
    if receipt.get('status')!='completed_training' or receipt.get('updates')!=6000:
        raise ValueError(('Incomplete fixed-budget endpoint',task))
    if method is not None and receipt['method']!=method:raise ValueError('Method identity mismatch')
    if repeat is not None and str(receipt['repeat'])!=str(repeat):raise ValueError('Repeat identity mismatch')
    cp=bound(receipt['checkpoint'])
    if cp.resolve()!=(directory/'checkpoint_12000.pt').resolve():raise ValueError('Endpoint path mismatch')
    return cp


def primary_complete(task, cp, require=False):
    directory=OUT/'evaluation'/task;manifest_sha=read(OUT/'protocol.json')['manifest']['sha256'];complete=directory/'complete.json'
    if complete.exists():
        receipt=read(complete)
        if bound(receipt['checkpoint']).resolve()!=cp.resolve():raise ValueError('Primary checkpoint changed')
        for split in SPLITS:bound(receipt['splits'][split])
    elif require:raise ValueError('Missing primary completion receipt '+task)
    for split in SPLITS:
        path=directory/split/'metrics.json'
        if not path.exists():
            if require:raise ValueError('Missing split '+task+'/'+split)
            continue
        metrics=read(path)
        if metrics.get('checkpoint_sha256')!=sha(cp) or metrics.get('manifest_sha256')!=manifest_sha or metrics.get('split')!=split:
            raise ValueError('Existing primary metrics identity mismatch '+str(path))
        if metrics.get('adapter_sha256')!=sha(ROOT/'experiments/dynamic_sr_controlled_headroom_20260926/evaluate.py'):
            raise ValueError('Primary evaluator source identity changed')
        if metrics.get('parameter_updates')!=0:raise ValueError('Evaluator updated parameters')
    return complete.exists()


def extra_complete(task, cp, require=False):
    directory=OUT/'evaluation'/task/'extra';path=directory/'complete.json'
    if not path.exists():
        if require:raise ValueError('Missing extra completion receipt '+task)
        return False
    receipt=read(path);protocol=read(OUT/'protocol.json')
    identity=dict(checkpoint_sha256=sha(cp),manifest_sha256=protocol['manifest']['sha256'],source_sha256=sha(HERE/'evaluate_extra.py'),label=task,protocol_sha256=sha(OUT/'protocol.json'))
    if receipt['identity']!=identity or receipt.get('observations')!=196 or receipt.get('parameter_updates')!=0:
        raise ValueError('Extra evaluation identity/integrity changed '+task)
    for item in receipt['results'].values():bound(item)
    index=read(directory/'float_index.json')
    if len(index['entries'])!=196 or index['identity']!=identity:raise ValueError('Float observation coverage changed')
    expected={(c,f) for c in ('cam00','cam01') for f in range(0,120,2)}
    expected|={(f'cam{c:02d}',f) for c in range(2,21) for f in (0,40,80,118)}
    if {(r['camera'],int(r['frame'])) for r in index['entries']}!=expected:raise ValueError('Float camera/frame coverage changed')
    for item in index['entries']:
        path=Path(item['path']);path=path if path.is_absolute() else directory/path
        if sha(path)!=item['sha256']:raise ValueError('Float cache hash mismatch '+str(path))
    return True


def summarize():
    with (OUT/'summary_gpu_free.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        call([SYSTEM_PYTHON,str(HERE/'summarize.py')],OUT/'async_summary.log')


def evaluate_endpoint(task, dependency=0):
    cp=verify_training(task);wait_pid(dependency);directory=OUT/'evaluation'/task
    directory.mkdir(parents=True,exist_ok=True)
    state=directory/'async_state.json'
    with (OUT/'evaluation_gpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        occupancy=free_gpu('1');primary_complete(task,cp)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
        write(state,dict(status='evaluating_primary',checkpoint=entry(cp),started_unix=time.time(),source_sha256=sha(__file__)))
        for split in SPLITS:
            if (directory/split/'metrics.json').exists():continue
            command=[PYTHON,str(ROOT/'experiments/dynamic_sr_controlled_headroom_20260926/evaluate.py'),
                '--manifest',str(ROOT/'data/dynamic_sr/n3dv_prepared/cook_spinach/manifest.json'),'--checkpoint',str(cp),
                '--out',str(directory/split),'--split',split,'--method',task,
                '--teacher-index',str(ROOT/'output/dynamic_sr_detail_supervision_20260924/teacher_inventory_v1/cook_spinach/teacher_index.json'),
                '--roi-protocol',str(ROOT/'output/dynamic_sr_soft_motion_20260924/roi_registry/cook_spinach/roi_protocol.json'),
                '--cache-dir',str(ROOT/'data/dynamic_sr/n3dv_prepared/cook_spinach/evaluation_cache'),'--lpips-device','cuda','--no-video']
            call(command,directory/f'{split}.log',env)
        primary_complete(task,cp)
        write(directory/'complete.json',dict(status='completed_evaluation',checkpoint=entry(cp),gpu=occupancy,
            splits={s:entry(directory/s/'metrics.json') for s in SPLITS}))
        # The U6000 baseline was already scheduled. Complete it once here if
        # needed, so module probes cannot silently retain a pending parent.
        parent=bound(read(OUT/'protocol.json')['parent'])
        for label,checkpoint in [('U6000',parent),(task,cp)]:
            if extra_complete(label,checkpoint):continue
            write(state,dict(status='evaluating_extra',endpoint=label,checkpoint=entry(checkpoint)))
            call([PYTHON,'-u',str(HERE/'evaluate_extra.py'),'--checkpoint',str(checkpoint),'--label',label,
                  '--out',str(OUT/'evaluation'/label/'extra')],OUT/f'{label}_extra.log',env)
            extra_complete(label,checkpoint,True)
        free_gpu('1')
    primary_complete(task,cp,True);extra_complete(task,cp,True);summarize()
    write(directory/'async_complete.json',dict(status='completed_primary_and_extra',checkpoint=entry(cp),
        primary=entry(directory/'complete.json'),extra=entry(directory/'extra/complete.json'),source_sha256=sha(__file__)))
    write(state,dict(status='completed_primary_and_extra',checkpoint=entry(cp)))


def dispatch_evaluation(task, cp, dependency=0):
    directory=OUT/'evaluation'/task;complete=directory/'async_complete.json'
    if complete.exists():
        if bound(read(complete)['checkpoint']).resolve()!=cp.resolve():raise ValueError('Async checkpoint changed')
        primary_complete(task,cp,True);extra_complete(task,cp,True)
        return None
    if primary_complete(task,cp) and extra_complete(task,cp):
        write(complete,dict(status='completed_primary_and_extra',checkpoint=entry(cp),
            primary=entry(directory/'complete.json'),extra=entry(directory/'extra/complete.json'),
            source_sha256=sha(__file__),reused_completed_artifacts=True))
        return None
    unit='4dsr-cg-eval-'+task.lower().replace('_','-')+'-'+sha(cp)[:8]+'-20261006'
    command=['systemd-run','--user','--remain-after-exit','--unit='+unit,'--property=Type=exec',
             '--property=WorkingDirectory='+str(ROOT),SYSTEM_PYTHON,str(HERE/'worker_async.py'),
             '--evaluate-endpoint',task,'--evaluation-dependency-pid',str(dependency)]
    with (OUT/f'async_dispatch_{task}.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        state=ensure_unit(unit,lambda:call(command,OUT/f'{task}_async_dispatch.log'))
        write(directory/'async_dispatch.json',dict(status='durable_unit_attached',unit=unit,checkpoint=entry(cp),
            unit_state=state,command=command,source_sha256=sha(__file__)))
    return unit


def retrieve_remote(host, task, log):
    """Inventory and retrieve the full run, outer training log and unit journal."""
    remote_out=REMOTE+'/output/'+HERE.name;unit='4dsr-cg-'+task.lower().replace('_','-')+'-20261006'
    code="""import pathlib,hashlib,json,subprocess,sys
out=pathlib.Path(sys.argv[1]);task=sys.argv[2];unit=sys.argv[3];run=out/'runs'/task
run.mkdir(parents=True,exist_ok=True)
with (run/'retrieval_journal.log').open('w') as f:subprocess.run(['journalctl','--user','-u',unit,'--no-pager'],stdout=f,stderr=subprocess.STDOUT,check=True)
paths=[p for p in run.rglob('*') if p.is_file() and p.name!='retrieval_manifest.json']
if (out/(task+'.log')).exists():paths.append(out/(task+'.log'))
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
receipt={'task':task,'unit':unit,'files':[{'path':str(p.relative_to(out)),'sha256':digest(p),'bytes':p.stat().st_size} for p in sorted(paths)]}
path=run/'retrieval_manifest.json';path.write_text(json.dumps(receipt,indent=2)+'\\n')
print(digest(path))
"""
    inventory_sha=ssh(host,shlex.join(['python3','-c',code,remote_out,task,unit])).strip()
    directory=OUT/'runs'/task;directory.mkdir(parents=True,exist_ok=True)
    call(['rsync','-az',host+':'+remote_out+'/runs/'+task+'/',str(directory)+'/'],log)
    call(['rsync','-az',host+':'+remote_out+'/'+task+'.log',str(OUT)+'/'+task+'.log'],log)
    manifest=directory/'retrieval_manifest.json'
    if sha(manifest)!=inventory_sha:raise ValueError('Remote retrieval manifest identity changed')
    receipt=read(manifest)
    for item in receipt['files']:
        path=(OUT/item['path']).resolve()
        if not path.is_relative_to(OUT.resolve()) or sha(path)!=item['sha256']:raise ValueError('Retrieved file mismatch '+str(path))
    write(OUT/f'{task}_retrieval_complete.json',dict(status='all_remote_files_verified',host=host,unit=unit,
        manifest=entry(manifest),file_count=len(receipt['files']),training_complete=(directory/'complete.json').exists()))


def train_endpoint(args, method):
    task=f'r{args.repeat}_{method}';directory=OUT/'runs'/task
    if (directory/'complete.json').exists():return verify_training(task,method,args.repeat)
    unit='4dsr-cg-'+task.lower().replace('_','-')+'-20261006'
    dispatch=OUT/f'dispatch_{task}.json';state=unit_state(unit,args.host)
    if state.get('LoadState')!='not-found' and not dispatch.exists():
        raise ValueError('Existing unit has no local task ownership receipt '+unit)
    if dispatch.exists():
        previous=read(dispatch)
        if previous['host']!=args.host or str(previous['gpu'])!=str(args.gpu):raise ValueError('Existing task ownership mismatch')
    def launch():
        if args.host=='local':
            occupancy=free_gpu(args.gpu)
            command=['systemd-run','--user','--remain-after-exit','--unit='+unit,'--property=Type=exec',
                '--property=WorkingDirectory='+str(ROOT),'--setenv=CUDA_VISIBLE_DEVICES='+args.gpu,
                '--setenv=OMP_NUM_THREADS=4','--setenv=OPENBLAS_NUM_THREADS=4',PYTHON,'-u',str(HERE/'train.py'),
                '--method',method,'--repeat',args.repeat,'--out',str(directory)]
            write(dispatch,dict(host=args.host,gpu=args.gpu,unit=unit,prelaunch=occupancy,command=command))
            call(command,OUT/f'{task}_dispatch.log')
        else:
            code="""import subprocess,sys,json
gpu=sys.argv[1]
uuid=subprocess.check_output(['nvidia-smi','-i',gpu,'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader'],text=True)
busy=[r for r in apps.splitlines() if uuid in r and '/opt/todesk/' not in r]
if busy:raise RuntimeError(('Remote GPU occupied',gpu,busy))
print(json.dumps({'uuid':uuid,'applications':apps}))
"""
            import json
            occupancy=json.loads(ssh(args.host,shlex.join(['python3','-c',code,args.gpu])))
            remote_out=REMOTE+'/output/'+HERE.name
            shell='source '+shlex.quote(REMOTE+'/activate_a100.sh')+' && export CUDA_VISIBLE_DEVICES='+shlex.quote(args.gpu)+' OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 && exec python -u '+shlex.quote(REMOTE+'/experiments/'+HERE.name+'/train.py')+' --method '+shlex.quote(method)+' --repeat '+shlex.quote(args.repeat)+' --out '+shlex.quote(remote_out+'/runs/'+task)+' > '+shlex.quote(remote_out+'/'+task+'.log')+' 2>&1'
            command=shlex.join(['systemd-run','--user','--remain-after-exit','--unit='+unit,'--property=Type=exec',
                '--property=WorkingDirectory='+REMOTE,'--setenv=CUDA_VISIBLE_DEVICES='+args.gpu,
                '--setenv=OMP_NUM_THREADS=4','--setenv=OPENBLAS_NUM_THREADS=4','bash','-lc',shell])
            write(dispatch,dict(host=args.host,gpu=args.gpu,unit=unit,prelaunch=occupancy,command=command))
            ssh(args.host,command)
    try:
        ensure_unit(unit,launch,args.host)
        wait_unit(unit,args.host)
    except BaseException:
        if args.host!='local':
            try:retrieve_remote(args.host,task,OUT/f'{task}_retrieval.log')
            except BaseException:write(OUT/f'{task}_retrieval_failure.json',dict(traceback=traceback.format_exc(),unit_preserved=unit))
        raise
    if args.host!='local':retrieve_remote(args.host,task,OUT/f'{task}_retrieval.log')
    else:
        call(['journalctl','--user','-u',unit,'--no-pager'],OUT/f'{task}.log')
    return verify_training(task,method,args.repeat)


def main(args):
    label=f'{args.host}_r{args.repeat}';status=OUT/f'worker_{label}.json'
    with (OUT/f'worker_async_{label}.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        wait_pid(args.training_dependency_pid);units=[]
        for method in args.methods:
            task=f'r{args.repeat}_{method}'
            write(status,dict(status='training',task=task,host=args.host,gpu=args.gpu,source_sha256=sha(__file__)))
            cp=train_endpoint(args,method)
            unit=dispatch_evaluation(task,cp,args.evaluation_dependency_pid)
            if unit:units.append((task,unit))
            write(status,dict(status='training_complete_evaluation_queued',task=task,unit=unit,training_gpu_continues=True))
        write(status,dict(status='training_batch_complete_waiting_evaluation_events',units=units))
        for task,unit in units:
            wait_unit(unit)
            call(['journalctl','--user','-u',unit,'--no-pager'],OUT/'evaluation'/task/'async_unit.log')
        for method in args.methods:
            task=f'r{args.repeat}_{method}';cp=verify_training(task,method,args.repeat)
            primary_complete(task,cp,True);extra_complete(task,cp,True)
            receipt=read(OUT/'evaluation'/task/'async_complete.json');bound(receipt['primary']);bound(receipt['extra'])
        summarize()
        write(status,dict(status='batch_completed',methods=args.methods,repeat=args.repeat,host=args.host,
            evaluation_units=units,source_sha256=sha(__file__),scheduler_only=True))


def cpu_self_test():
    """Synthetic process/lock/service checks; cannot execute GPU/train/SSH CLI."""
    from types import SimpleNamespace
    receipt=dict(source_sha256=sha(__file__),gpu_calls=0,training_calls=0)
    holder={'LoadState':'not-found'};launches=[]
    def launch():launches.append(1);holder.update(LoadState='loaded',ActiveState='active',MainPID='7')
    for _ in range(2):ensure_unit('synthetic',launch,state_fn=lambda:dict(holder))
    assert len(launches)==1;receipt['reattach_without_redispatch']=True
    holder['ActiveState']='failed'
    try:ensure_unit('synthetic',launch,state_fn=lambda:dict(holder))
    except RuntimeError:pass
    else:raise AssertionError('Failed unit redispatched')
    assert len(launches)==1;receipt['failed_unit_preserved']=True
    with tempfile.TemporaryDirectory(prefix='cg_async_cpu_') as td:
        td=Path(td);process=subprocess.Popen([SYSTEM_PYTHON,'-c','import time;time.sleep(.08)'])
        started=time.monotonic();wait_pid(process.pid);assert process.poll() is not None
        receipt['pidfd_exit_wait']=time.monotonic()-started
        target=td/'late'
        descendant='import time,pathlib;time.sleep(.3);pathlib.Path('+repr(str(target))+').write_text("leaked")'
        code='import subprocess,sys;subprocess.Popen([sys.executable,"-c",'+repr(descendant)+'])'
        call([SYSTEM_PYTHON,'-c',code],td/'process.log')
        # A pidfd waiter has no polling. An adopted leftover process group was
        # killed and reaped before call returned, so it cannot write late.
        assert not target.exists();receipt['descendants_reaped_before_unlock']=True
        lockpath=td/'lock';with_parent=lockpath.open('a');fcntl.flock(with_parent,fcntl.LOCK_EX)
        child=subprocess.Popen([SYSTEM_PYTHON,'-c','import fcntl,sys,pathlib;f=open(sys.argv[1],"a");fcntl.flock(f,fcntl.LOCK_EX);pathlib.Path(sys.argv[2]).write_text("acquired")',str(lockpath),str(td/'acquired')])
        assert not (td/'acquired').exists();fcntl.flock(with_parent,fcntl.LOCK_UN);with_parent.close();wait_pid(child.pid);child.wait()
        assert (td/'acquired').read_text()=='acquired';receipt['shared_lock_serialization']=True
    # Run the real coordinator and real artifact validation against tiny CPU
    # fixtures. Only the actual training/service backends are replaced.
    names=('OUT','train_endpoint','dispatch_evaluation','wait_unit','summarize','verify_training','primary_complete','extra_complete')
    saved={name:globals()[name] for name in names};events=[]
    with tempfile.TemporaryDirectory(prefix='async_cpu_',dir=OUT) as td:
        td=Path(td)
        try:
            globals()['OUT']=td
            (td/'protocol.json').write_bytes((saved['OUT']/'protocol.json').read_bytes())
            cp=td/'checkpoint.pt';cp.write_bytes(b'CPU-only-checkpoint-fixture')
            label='r1_R';directory=td/'evaluation'/label/'extra';directory.mkdir(parents=True)
            protocol=read(td/'protocol.json')
            identity=dict(checkpoint_sha256=sha(cp),manifest_sha256=protocol['manifest']['sha256'],source_sha256=sha(HERE/'evaluate_extra.py'),label=label,protocol_sha256=sha(td/'protocol.json'))
            keys=sorted({(c,f) for c in ('cam00','cam01') for f in range(0,120,2)}|{(f'cam{c:02d}',f) for c in range(2,21) for f in (0,40,80,118)})
            entries=[]
            for camera,frame in keys:
                path=directory/f'{camera}_{frame:04d}.npz';path.write_bytes(b'CPU-float-fixture')
                entries.append(dict(camera=camera,frame=frame,path=path.name,sha256=sha(path)))
            write(directory/'float_index.json',dict(identity=identity,entries=entries))
            write(directory/'complete.json',dict(identity=identity,observations=196,parameter_updates=0,
                results={'float_index.json':entry(directory/'float_index.json')}))
            assert saved['extra_complete'](label,cp,True)
            (directory/entries[0]['path']).write_bytes(b'corrupt')
            try:saved['extra_complete'](label,cp,True)
            except ValueError:pass
            else:raise AssertionError('Corrupt float cache reused')
            receipt['extra_relative_paths_exact196_and_hash_rejection']=True
            def trained(args,method):events.append(('trained',method));return cp
            def dispatched(task,checkpoint,dependency):
                method=task.split('_',1)[1];events.append(('queued',method));out=td/'evaluation'/task;out.mkdir(parents=True,exist_ok=True)
                write(out/'p.json',{});write(out/'e.json',{})
                write(out/'async_complete.json',dict(primary=entry(out/'p.json'),extra=entry(out/'e.json')))
                return method
            globals().update(train_endpoint=trained,dispatch_evaluation=dispatched,
                wait_unit=lambda unit:events.append(('evaluation_exit',unit)),summarize=lambda:events.append(('summary','complete')),
                verify_training=lambda *a,**kw:cp,primary_complete=lambda *a,**kw:True,extra_complete=lambda *a,**kw:True)
            main(SimpleNamespace(host='local',repeat='1',gpu='CPU-fixture',methods=['R','G','RG'],training_dependency_pid=0,evaluation_dependency_pid=0))
            assert events.index(('trained','RG'))<events.index(('evaluation_exit','R'))
            assert read(td/'worker_local_r1.json')['status']=='batch_completed'
            receipt['real_coordinator_training_precedes_evaluation_wait']=events
        finally:globals().update(saved)
    receipt['status']='passed_cpu_mechanism_checks'
    write(OUT/'worker_async_cpu_checks.json',receipt);print(receipt)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--host',choices=['local','a100-train']);p.add_argument('--gpu')
    p.add_argument('--repeat',choices=['1','2']);p.add_argument('--methods',nargs='+',choices=['C1','Jperm','B2perm','R','G','RG'])
    p.add_argument('--training-dependency-pid',type=int,default=0);p.add_argument('--evaluation-dependency-pid',type=int,default=0)
    p.add_argument('--evaluate-endpoint');p.add_argument('--cpu-self-test',action='store_true');args=p.parse_args()
    try:
        if args.cpu_self_test:cpu_self_test()
        elif args.evaluate_endpoint:evaluate_endpoint(args.evaluate_endpoint,args.evaluation_dependency_pid)
        else:
            if not all((args.host,args.gpu,args.repeat,args.methods)):p.error('Training mode requires host, gpu, repeat and methods')
            main(args)
    except BaseException:
        target=OUT/'evaluation'/args.evaluate_endpoint/'async_state.json' if args.evaluate_endpoint else OUT/f'worker_{args.host}_r{args.repeat}.json'
        write(target,dict(status='failed',traceback=traceback.format_exc(),source_sha256=sha(__file__),durable_units_preserved=True));raise
