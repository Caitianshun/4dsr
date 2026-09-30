"""One controller, two A100 branches, process-exit collection then local evaluation.

No minute polling. A remote systemd service survives SSH loss. Linux pidfd exit
events attach to each service, with an hourly exception watchdog. Both returned
checkpoints are verified before serial evaluation in the historical local env.
"""
import argparse
import concurrent.futures
import subprocess
import shlex
import threading
import traceback
import select
from dv_common import *
from readiness import validate
from service_wait import open_pidfd

REMOTE='/home/ubuntu/3DGS/4dsr'
HOST='a100-train'
EVALPY='/home/cai_tianshun/Project/4dgs/.venv/bin/python'

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['diagnose','train'],required=True)
    ap.add_argument('--protocol',type=Path,default=OUT/'protocol.json');a=ap.parse_args()
    if a.stage=='diagnose':
        # Diagnosis is deliberately idempotent; completed immutable evidence is reused.
        required=['audit_manifest.json','observables/complete.json','structure/complete.json',
            'structure/postprocess.json','image_priors/summary.json','spectrum/summary.json']
        missing=[n for n in required if not (OUT/n).is_file()]
        if missing:raise RuntimeError('Required diagnosis assets not completed: '+str(missing))
        write(OUT/'diagnosis_complete.json',dict(status='completed',assets={n:entry(OUT/n) for n in required},
            parameter_updates=0,complete_unix=time.time()))
        return
    assert a.protocol.resolve()==(OUT/'protocol.json').resolve()
    p=require_run_root(OUT);validate(p)
    controller=OUT/'controller';controller.mkdir(exist_ok=True)
    lock=(controller/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    mutex=threading.Lock();eval_lock=threading.Lock();commands=[];states={};start=time.time()
    def state(task,status,**kw):
        with mutex:
            states[task]=dict(status=status,updated_unix=time.time(),**kw)
            write(controller/'status.json',dict(status='running',tasks=states,started_unix=start,updated_unix=time.time()))
    def run(cmd,label,env=None):
        with mutex:
            path=controller/f'{len(commands)+1:03d}_{label}.log'
            r=dict(argv=cmd,label=label,log=str(path.relative_to(ROOT)),started_unix=time.time());commands.append(r)
            write(controller/'commands.json',commands)
        with path.open('w') as f:result=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,env=env)
        with mutex:r.update(returncode=result.returncode,seconds=time.time()-r['started_unix']);write(controller/'commands.json',commands)
        if result.returncode:raise RuntimeError(f'{label} exited {result.returncode}; {path}')
        return path
    def task(t):
        label=t['task_id'];gpu=t['physical_gpu'];base=OUT/'runs'/label;unit='4dsr-prior-diagnosis-'+label.lower().replace('_','-')
        shell='cd '+shlex.quote(REMOTE)+' && '+shlex.join(['python3',f'experiments/{RUN_ID}/dispatch.py','--task',label])
        state(label,'dispatching',physical_gpu=gpu)
        dispatch=run(['ssh','-o','BatchMode=yes',HOST,shell],label+'_dispatch')
        info=json.loads(dispatch.read_text().splitlines()[-1]);unit=info['unit']
        state(label,'training',unit=unit,physical_gpu=gpu)
        waiter='cd '+shlex.quote(REMOTE)+' && '+shlex.join(['python3',f'experiments/{RUN_ID}/service_wait.py','--unit',unit])
        try:run(['ssh','-o','BatchMode=yes',HOST,waiter],label+'_exit_event')
        except RuntimeError:
            # Reattach to the same service; never silently restart a training branch.
            run(['ssh','-o','BatchMode=yes',HOST,waiter],label+'_reattach_exit_event')
        state(label,'returning',unit=unit,physical_gpu=gpu)
        base.mkdir(parents=True,exist_ok=True)
        run(['rsync','-az',f'{HOST}:{REMOTE}/output/{RUN_ID}/runs/{label}/',str(base)+'/'],label+'_return')
        train=base/'attempt_01/train'
        if (train/'failed.json').exists():raise RuntimeError(f'{label} training failed; local error receipt returned')
        complete=read(train/'complete.json');assert complete['status']=='completed' and complete['actual_updates']==6000
        for step in [9000,12000]:
            ck=train/f'checkpoint_{step}.pt';meta=read(ck.with_suffix('.json'));assert sha(ck)==meta['sha256']
        checkpoint=train/'checkpoint_12000.pt'
        state(label,'evaluating',checkpoint=str(checkpoint.relative_to(ROOT)))
        with eval_lock:
            while True:
                occupancy=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader'],text=True)
                pids=[int(line.split(',')[1]) for line in occupancy.splitlines() if p['evaluation']['physical_gpu'] in line]
                if not pids:break
                state(label,'waiting_for_existing_evaluation_gpu_processes',pids=pids)
                fds=[]
                for pid in pids:
                    try:fds.append(open_pidfd(pid))
                    except ProcessLookupError:pass
                try:
                    while fds:
                        done,_,_=select.select(fds,[],[],3600)
                        if not done:state(label,'hourly_exception_watchdog',waiting_pids=pids)
                        for fd in done:os.close(fd);fds.remove(fd)
                finally:
                    for fd in fds:os.close(fd)
            env={**os.environ,'CUDA_VISIBLE_DEVICES':p['evaluation']['physical_gpu'],'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4'}
            run([EVALPY,'-u',str(HERE/'evaluate_endpoint.py'),'--checkpoint',str(checkpoint),'--label',label],label+'_evaluation',env)
        receipt=read(OUT/'evaluation'/label/'complete.json');assert receipt['checkpoint_sha256']==sha(checkpoint)
        state(label,'completed',checkpoint=entry(checkpoint),evaluation=entry(OUT/'evaluation'/label/'complete.json'))
        return label
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            pending=[pool.submit(task,t) for t in p['task_plan']]
            for f in concurrent.futures.as_completed(pending):f.result()
        run(['rsync','-az',f'{HOST}:{REMOTE}/output/{RUN_ID}/budget.json',str(OUT/'budget.json')],'return_budget')
        checkpoints={t['task_id']:dict(**states[t['task_id']]['checkpoint'],task_id=t['task_id'],status='completed',training_gpu=t['physical_gpu']) for t in p['task_plan']}
        write(OUT/'checkpoint_index.json',dict(checkpoints=checkpoints,protocol=entry(OUT/'protocol.json')))
        write(controller/'status.json',dict(status='completed_training_and_evaluation',tasks=states,seconds=time.time()-start))
        run([EVALPY,str(HERE/'summarize.py')],'summarize')
    except BaseException:
        write(controller/'failed.json',dict(status='failed',traceback=traceback.format_exc(),states=states))
        raise

if __name__=='__main__':main()
