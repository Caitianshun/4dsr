"""Recover completed training through remaining evaluation only, with no updates."""
import subprocess
import traceback
from dv_common import *


def main():
    p=require_run_root(OUT);start=time.time();folder=OUT/'completion_recovery';folder.mkdir(exist_ok=True)
    lock=(folder/'owner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    records=[]
    def run(argv,name,env=None):
        log=folder/(name+'.log');r=dict(argv=[str(x) for x in argv],started_unix=time.time(),log=str(log.relative_to(ROOT)))
        records.append(r);write(folder/'commands.json',records)
        with log.open('w') as f:result=subprocess.run(r['argv'],stdout=f,stderr=subprocess.STDOUT,env=env)
        r.update(returncode=result.returncode,seconds=time.time()-r['started_unix']);write(folder/'commands.json',records)
        assert result.returncode==0,(name,log)
    checkpoints={};inventory=[]
    for task in p['task_plan']:
        label=task['task_id'];base=OUT/'runs'/label/'attempt_01/train'
        completed=read(base/'complete.json');assert completed['status']=='completed' and completed['actual_updates']==6000
        for step in [9000,12000]:
            path=base/f'checkpoint_{step}.pt';meta=read(path.with_suffix('.json'));assert sha(path)==meta['sha256']
            row=dict(**entry(path),task_id=label,step=step,status='completed',training_gpu=task['physical_gpu']);inventory.append(row)
            if step==12000:checkpoints[label]=row
    write(OUT/'checkpoint_index.json',dict(checkpoints=checkpoints,protocol=entry(OUT/'protocol.json')))
    write(OUT/'checkpoint_inventory.json',dict(checkpoints=inventory,protocol=entry(OUT/'protocol.json')))
    run(['rsync','-az',f"a100-train:{p['training']['remote_root']}/output/{RUN_ID}/budget.json",OUT/'budget.json'],'return_budget')
    budget=read(OUT/'budget.json');assert budget['actual_updates']==budget['effective_updates']==12000
    evaluation_gpu=p['evaluation']['physical_gpu']
    occupancy=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader'],text=True)
    assert not any(evaluation_gpu in line for line in occupancy.splitlines()),occupancy
    env={**os.environ,'CUDA_VISIBLE_DEVICES':evaluation_gpu,'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4'}
    for label in ['r2_T','r1_T']:
        run([sys.executable,'-u',HERE/'evaluate_endpoint.py','--checkpoint',local(checkpoints[label]['path']),'--label',label],label+'_evaluation',env)
    run([sys.executable,HERE/'summarize.py'],'summarize',env)
    run([sys.executable,HERE/'evaluate_fixed_regions.py'],'fixed_regions',env)
    run([sys.executable,HERE/'summarize_detail.py'],'detail_summary',env)
    run([sys.executable,HERE/'final_figures.py','--run-root',OUT],'final_figures',env)
    write(folder/'complete.json',dict(status='completed',seconds=time.time()-start,training_updates=0,
        reused_training_updates=12000,commands=entry(folder/'commands.json'),checkpoint_inventory=entry(OUT/'checkpoint_inventory.json')))
    write(OUT/'controller/status.json',dict(status='completed_training_and_evaluation',
        recovered_by=entry(folder/'complete.json'),previous_termination='incomplete evaluation; later local reboot observed; exact cause not recoverable'))


if __name__=='__main__':
    try:main()
    except BaseException:
        write(OUT/'completion_recovery/failed.json',dict(status='failed',traceback=traceback.format_exc()))
        raise
