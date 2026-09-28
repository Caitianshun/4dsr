"""Persistent four-task controller; process exits trigger collection and uniform evaluation."""
import subprocess
import shlex
import threading
import concurrent.futures
import traceback
from dv_common import *
from readiness import validate
from task_state import next_action, fixed_tasks
from summarize import summarize

REMOTE='/home/ubuntu/3DGS/4dsr'

def main():
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--run-root',type=Path,default=OUT);a=ap.parse_args()
    p=require_run_root(a.run_root);validate(p);tasks=fixed_tasks(p)
    controller=OUT/'controller';controller.mkdir(exist_ok=True)
    lock=(OUT/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    started=time.time();status={};commands=read(controller/'commands.json') if (controller/'commands.json').exists() else []
    command_lock=threading.Lock()
    def state(**kw):
        status.update(kw);status.update(updated_unix=time.time(),wall_seconds=time.time()-started);write(controller/'status.json',status)
    def run(cmd,label,env=None,check=True):
        with command_lock:
            n=len(commands)+1;logpath=controller/f'{n:04d}_{label}.log'
            e=dict(label=label,argv=cmd,log=str(logpath.relative_to(ROOT)),started_unix=time.time());commands.append(e);write(controller/'commands.json',commands)
        with logpath.open('w') as f:rc=subprocess.call([str(x) for x in cmd],stdout=f,stderr=subprocess.STDOUT,env=env)
        with command_lock:e.update(returncode=rc,seconds=time.time()-e['started_unix']);write(controller/'commands.json',commands)
        if check and rc:raise RuntimeError(f'{label} exited {rc}: {logpath}')
        return rc,logpath
    def remote(op,task,extra=None):
        argv=['python3',f'experiments/{RUN_ID}/remote_task.py',op,'--run-root',f'output/{RUN_ID}','--task',task]+(extra or [])
        shell='cd '+shlex.quote(REMOTE)+' && '+shlex.join(argv)
        rc,log=run(['ssh','a100-train',shell],task+'_'+op,check=False)
        if rc:raise RuntimeError(f'Remote {op} failed; inspect {log}; no blind retraining')
        return json.loads(log.read_text().splitlines()[-1])
    def collect(task):
        dest=OUT/'runs'/task;dest.mkdir(parents=True,exist_ok=True)
        run(['rsync','-az',f'a100-train:{REMOTE}/output/{RUN_ID}/runs/{task}/',str(dest)+'/'],task+'_return')
        run(['rsync','-az',f'a100-train:{REMOTE}/output/{RUN_ID}/budget.json',str(OUT/'budget.json')],task+'_budget')
    def wait(unit,task):
        shell='cd '+shlex.quote(REMOTE)+' && '+shlex.join(['python3',f'experiments/{RUN_ID}/service_wait.py','--unit',unit])
        rc,log=run(['ssh','a100-train',shell],task+'_wait_exit',check=False)
        if rc:
            # Transport failures are distinct from remote training failures.
            snapshot=remote('inspect',task)
            if snapshot['active']:
                rc,log=run(['ssh','a100-train',shell],task+'_reconnect_exit',check=False)
                if rc:raise RuntimeError(f'Unresolved transport failure; existing service retained: {log}')
    methods=read(OUT/'checkpoint_index.json')['checkpoints'] if (OUT/'checkpoint_index.json').exists() else {}
    evalenv={**os.environ,'FOURDSR_RUN_ROOT':str(OUT),'CUDA_VISIBLE_DEVICES':p['evaluation']['physical_gpu'],'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4'}
    def evaluate(label,checkpoint):
        # Existing complete is validated inside the evaluator before any render.
        if not (OUT/'evaluation'/label/'complete.json').exists():
            occ=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name','--format=csv'],text=True)
            assert p['evaluation']['physical_gpu'] not in occ,'Evaluation GPU occupied; preserve endpoint for later evaluation'
        run([sys.executable,'-u',str(HERE/'evaluate_endpoint.py'),'--run-root',str(OUT),'--checkpoint',str(checkpoint),'--label',label],label+'_eval',evalenv)
        complete=read(OUT/'evaluation'/label/'complete.json');assert complete['checkpoint_sha256']==sha(checkpoint)
        return complete
    pool=concurrent.futures.ThreadPoolExecutor(max_workers=1);pending=[]
    try:
        state(status='running',phase='research_readiness',numerical_replay='warning')
        # All baseline metric implementations are the already evaluated versions.
        baseline=read(bound(p['baseline_reuse']['receipt']))
        assert baseline['checkpoint_sha256']==p['parent']['sha256'] and baseline['physical_gpu']==p['evaluation']['physical_gpu']
        for rel,h in baseline['source_sha256'].items():
            if rel.endswith('dynamic_sr_dynamic_validation_20260927/evaluate_endpoint.py'):
                original=bound(p['baseline_reuse']['original_wrapper']);assert sha(original)==h
                current=(ROOT/rel).read_text().replace("        if not ff:\n            import imageio_ffmpeg\n            ff=imageio_ffmpeg.get_ffmpeg_exe()\n",'')
                assert current==original.read_text(),'Baseline primary wrapper changed'
            else:assert sha(ROOT/rel)==h
        bound(p['baseline_reuse']['endpoint'])
        for ref in p['historical_J1'].values():
            receipt=read(bound(ref['receipt']));bound(ref['endpoint']);bound(ref['checkpoint'])
            assert receipt['physical_gpu']==p['evaluation']['physical_gpu']
            for rel,h in receipt['source_sha256'].items():assert sha(ROOT/rel)==h
        for task in tasks:
            label=task['task_id'];state(phase=label+'_training',current_task=label)
            while True:
                snapshot=remote('inspect',label)
                action=next_action(True,bool(snapshot['active']),bool(snapshot['endpoint']),bool(snapshot['resume']))
                if action=='wait_existing':
                    assert len(snapshot['active'])==1;wait(snapshot['active'][0]['unit'],label);continue
                if action=='evaluate_endpoint':break
                failures=[x for x in snapshot['attempts'] if x['failed']]
                if failures:
                    raise RuntimeError(f'{label} has a real engineering failure: {failures[-1]["failed"]}; repair before resuming')
                if len(snapshot['attempts'])>=3:raise RuntimeError(f'{label}: infrastructure retries require inspection')
                assert time.time()-p['registered_unix']<=p['budget']['execution_wall_seconds_max'],'Two-day execution budget exhausted'
                remaining=12000-(9000 if action=='resume_9000' else 6000)
                used=(snapshot['budget'] or {}).get('actual_updates',0)
                # Reserve remaining effective work, not just the currently launching segment.
                committed=(snapshot['budget'] or {}).get('committed_steps',{})
                outstanding=sum(12000-committed.get(t['task_id'],6000) for t in tasks)
                assert used+outstanding<=p['budget']['actual_formal_max'],'Recovery allowance exhausted'
                extra=['--attempt',str(len(snapshot['attempts'])+1)]
                if action=='resume_9000':extra+=['--resume',snapshot['resume']['path']]
                try:dispatch=remote('launch',label,extra)
                except RuntimeError:
                    snapshot=remote('inspect',label)
                    if not snapshot['active'] and not snapshot['endpoint']:raise
                    continue
                wait(dispatch['unit'],label)
            collect(label);checkpoint=local(snapshot['endpoint']['path']);assert sha(checkpoint)==snapshot['endpoint']['sha256']
            methods[label]=dict(checkpoint=str(checkpoint),sha256=sha(checkpoint),repeat=task['repeat'],arm=task['arm'],step=12000,
                status='returned_evaluation_pending',training_gpu=p['training']['physical_gpu'],evaluation_gpu=p['evaluation']['physical_gpu'],host='a100-train',
                parent=p['parent'],source_identity=entry(OUT/'research_readiness.json'),attempts=snapshot['attempts'])
            write(OUT/'checkpoint_index.json',dict(parent=p['parent'],pair_schedules=p['pair_schedules'],checkpoints=methods,protocol_sha256=sha(OUT/'protocol.json')))
            pending.append((label,pool.submit(evaluate,label,checkpoint)))
            # No first-set quality branch: every registered task is dispatched.
        state(phase='uniform_evaluation')
        for label,future in pending:
            methods[label]['evaluation']=future.result();methods[label]['status']='returned_evaluated'
            write(OUT/'checkpoint_index.json',dict(parent=p['parent'],pair_schedules=p['pair_schedules'],checkpoints=methods,protocol_sha256=sha(OUT/'protocol.json')))
        quality=summarize();assert quality['execution']['completed_formal_endpoints']==len(tasks)
        budget=read(OUT/'budget.json');assert budget['effective_updates']==p['budget']['effective_formal_max']
        costs=dict(**budget,retry_updates=budget['actual_updates']-p['budget']['effective_formal_max'],effective_training_rgb_forwards=p['budget']['effective_RGB'],effective_adam_calls=p['budget']['effective_adam_calls'],
            new_evaluation_rgb_forwards=196*len(tasks),baseline_reused_rgb_forwards=588,controller_wall_seconds=time.time()-started,
            training_gpu=p['training']['physical_gpu'],evaluation_gpu=p['evaluation']['physical_gpu'],
            attempts={label:[dict(path=x['path'],complete=x['complete'],failed=x['failed']) for x in m['attempts']] for label,m in methods.items()})
        write(OUT/'cost.json',costs)
        assert sources()==read(OUT/'research_readiness.json')['sources']
        for e in p['inherited_evidence'].values():bound(e)
        state(status='completed',phase='awaiting_visual_document_review',completed_endpoints=len(tasks),effective_updates=budget['effective_updates'])
    except BaseException:
        state(status='failed',error=traceback.format_exc());raise
    finally:pool.shutdown(wait=True)

if __name__=='__main__':main()
