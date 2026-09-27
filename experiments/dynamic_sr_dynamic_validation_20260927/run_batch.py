"""Persistent controller: process-exit -> collect/hash -> evaluation -> fixed decision."""
import subprocess
import argparse
import shlex
import threading
import concurrent.futures
import traceback
import time
import fcntl
from dv_common import *
from decide import decide_set,finalize
REMOTE='/home/ubuntu/3DGS/4dsr'

def main():
    p=read(OUT/'protocol.json');controller=OUT/'controller';controller.mkdir(exist_ok=False);lock=(OUT/'controller.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    started=time.time();status={};commands=[]
    def state(**kw):status.update(kw);status.update(updated_unix=time.time(),wall_seconds=time.time()-started);write(controller/'status.json',status)
    def run(cmd,label,env=None,check=True):
        entry=dict(label=label,argv=cmd,started_unix=time.time());commands.append(entry);write(controller/'commands.json',commands)
        with (controller/(label+'.log')).open('w') as log:rc=subprocess.call(cmd,stdout=log,stderr=subprocess.STDOUT,env=env)
        entry.update(returncode=rc,seconds=time.time()-entry['started_unix']);write(controller/'commands.json',commands)
        if check and rc:raise RuntimeError(f'{label} exited {rc}')
        return rc
    def collect(label):
        run(['rsync','-az','--exclude=engineering/**/checkpoint_*.pt','--exclude=engineering_followup/**/checkpoint_*.pt',f'a100-train:{REMOTE}/output/{HERE.name}/',str(OUT)+'/'],label)
    gpu=p['training']['physical_gpu'];evalenv={**os.environ,'CUDA_VISIBLE_DEVICES':p['evaluation']['physical_gpu'],'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4'}
    methods={};queue=[];pool=concurrent.futures.ThreadPoolExecutor(max_workers=1)
    def evaluate(label,checkpoint):
        run([sys.executable,'-u',str(HERE/'evaluate_endpoint.py'),'--checkpoint',str(checkpoint),'--label',label],label+'_eval',evalenv)
        assert read(OUT/'evaluation'/label/'complete.json')['checkpoint_sha256']==sha(checkpoint)
        return read(OUT/'evaluation'/label/'endpoint.json')
    def remote_train(arm,r):
        label=f'r{r}_{arm}';state(status='running',phase=label+'_training')
        trainrel=f'output/{HERE.name}/{label}/train'
        # The service survives SSH loss. A transport failure is recorded, never retrained blindly.
        shell=f'cd {REMOTE} && source activate_a100.sh && export CUDA_VISIBLE_DEVICES={gpu} && exec python -u experiments/{HERE.name}/train.py --method {arm} --repeat {r} --out {trainrel} > output/{HERE.name}/{label}.log 2>&1'
        cmd=['ssh','a100-train','systemd-run','--user',f'--unit=4dsr-dynval-{label.lower().replace("_","-")}-20260927','--wait','--collect','--working-directory='+REMOTE,'/bin/bash','-c',shlex.quote(shell)]
        rc=run(cmd,label+'_remote',check=False);collect(label+'_return')
        train=OUT/label/'train';complete=read(train/'complete.json') if (train/'complete.json').exists() else None
        if rc or not complete or complete['status']!='completed':raise RuntimeError(f'{label} remote exit {rc}; see returned artifacts')
        checkpoint=train/'checkpoint_12000.pt'
        for step in [9000,12000]:
            f=train/f'checkpoint_{step}.pt';assert sha(f)==read(f.with_suffix('.json'))['sha256']
        methods[label]=dict(checkpoint=str(checkpoint),sha256=sha(checkpoint),gpu=gpu,host='a100-train',status='returned_evaluation_pending',train=str(train),repeat=r,arm=arm)
        write(OUT/'checkpoint_index.json',dict(checkpoints=methods))
        queue.append((label,pool.submit(evaluate,label,checkpoint)))
    try:
        state(status='running',phase='engineering_receipt')
        collect('initial_engineering_return')
        eng=read(OUT/'engineering/complete.json');assert eng['status']=='passed' and eng['protocol_sha256']==sha(OUT/'protocol.json')
        assert read(OUT/'decision_tests.json')['status']=='passed'
        # First/second occupancy samples and deployment hashes are stored by preparation.
        run(['ssh','a100-train','nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv'],'prelaunch_occupancy')
        assert gpu not in (controller/'prelaunch_occupancy.log').read_text(),'Training GPU has another process'
        baseline=read(OUT/'evaluation/U6000/endpoint.json');assert read(OUT/'evaluation/U6000/complete.json')['checkpoint_sha256']==p['parent']['sha256']
        decisions=[]
        for r in ['1','2']:
            if r=='2' and not decisions[0]['repeat_allowed']:break
            set_values={'U6000':baseline};queue=[]
            for arm in p['training']['order'][r]:remote_train(arm,r)
            state(phase=f'r{r}_evaluation')
            for label,future in queue:
                arm=methods[label]['arm'];set_values[arm]=future.result();methods[label]['status']='returned_evaluated'
            result=decide_set(set_values,p['decision']);result.update(protocol_sha256=sha(OUT/'protocol.json'),repeat=r,endpoints=set_values,endpoint_files={a:dict(path=str(OUT/'evaluation'/f'r{r}_{a}'/'endpoint.json'),sha256=sha(OUT/'evaluation'/f'r{r}_{a}'/'endpoint.json')) for a in p['training']['order'][r]})
            path=OUT/f'decision_set{r}.json';assert not path.exists();write(path,result);decisions.append(result)
            write(OUT/f'decision_set{r}_frozen.json',dict(path=str(path),sha256=sha(path),eligible_candidates=result['eligible_candidates'],modes_vs_J=result['modes_vs_J'],modes_SA=result['modes_SA'],frozen_before_repeat=r=='1'))
            write(OUT/'checkpoint_index.json',dict(checkpoints=methods))
        final=finalize(decisions[0],decisions[1] if len(decisions)>1 else None,p['decision']);final.update(protocol_sha256=sha(OUT/'protocol.json'),sets=[dict(path=f'decision_set{i+1}.json',sha256=sha(OUT/f'decision_set{i+1}.json')) for i in range(len(decisions))]);write(OUT/'decision.json',final)
        costs=read(OUT/'budget.json');costs.update(segments={label:read(Path(x['train'])/'complete.json') for label,x in methods.items()},legacy_readonly=read(OUT/'legacy_summary.json')['rgb_forwards'],evaluation_rgb_forwards=196*(len(methods)+1),controller_wall_seconds=time.time()-started,training_hardware=gpu,evaluation_hardware=p['evaluation']['physical_gpu'],historical_updates=1214)
        write(OUT/'cost.json',costs)
        from evaluate_endpoint import csvwrite
        import csv
        for name in ['metrics_per_frame.csv','metrics_per_camera.csv']:
            rows=[]
            for label in ['U6000',*methods]:rows.extend(csv.DictReader((OUT/'evaluation'/label/name).open()))
            csvwrite(OUT/name,rows)
        assert sha(local(p['old_decision']['path']))==p['old_decision']['sha256']
        state(status='completed',phase='awaiting_visual_document_review',decision=final,old_decision_unchanged=True)
    except BaseException:
        state(status='failed',error=traceback.format_exc());raise
    finally:pool.shutdown(wait=True)
if __name__=='__main__':main()
