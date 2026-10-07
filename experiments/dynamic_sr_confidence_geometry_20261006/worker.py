"""Persistent worker: process return triggers retrieval, verification and evaluation."""
import argparse,subprocess,shlex,time,fcntl,select,traceback
from cg_common import *
PYTHON='/home/cai_tianshun/Project/4dgs/.venv/bin/python'
def wait_pid(pid):
    if not pid:return
    try:fd=os.pidfd_open(pid)
    except ProcessLookupError:return
    select.select([fd],[],[]);os.close(fd)
def call(command,log,env=None):
    with Path(log).open('a') as f:
        print(shlex.join(command),file=f,flush=True)
        subprocess.run(command,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
def free_gpu(gpu):
    uuid=subprocess.check_output(['nvidia-smi','-i',gpu,'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader'],text=True)
    busy=[row for row in apps.splitlines() if uuid in row and '/opt/todesk/' not in row]
    assert not busy,('GPU has another compute process',busy)
    return dict(uuid=uuid,applications=apps)
def evaluate(cp,label,dependency):
    wait_pid(dependency)
    with (OUT/'evaluation_gpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);occupancy=free_gpu('1')
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
        directory=OUT/'evaluation'/label;directory.mkdir(parents=True,exist_ok=True)
        for split in ['test','dev','train_fixed']:
            if (directory/split/'metrics.json').exists():continue
            command=[PYTHON,str(ROOT/'experiments/dynamic_sr_controlled_headroom_20260926/evaluate.py'),
              '--manifest',str(ROOT/'data/dynamic_sr/n3dv_prepared/cook_spinach/manifest.json'),
              '--checkpoint',str(cp),'--out',str(directory/split),'--split',split,'--method',label,
              '--teacher-index',str(ROOT/'output/dynamic_sr_detail_supervision_20260924/teacher_inventory_v1/cook_spinach/teacher_index.json'),
              '--roi-protocol',str(ROOT/'output/dynamic_sr_soft_motion_20260924/roi_registry/cook_spinach/roi_protocol.json'),
              '--cache-dir',str(ROOT/'data/dynamic_sr/n3dv_prepared/cook_spinach/evaluation_cache'),'--lpips-device','cuda','--no-video']
            call(command,directory/f'{split}.log',env)
        write(directory/'complete.json',dict(status='completed_evaluation',checkpoint=entry(cp),gpu=occupancy,
          splits={s:entry(directory/s/'metrics.json') for s in ['test','dev','train_fixed']}))
def main(a):
    label=f'{a.host}_r{a.repeat}';status=OUT/f'worker_{label}.json'
    wait_pid(a.training_dependency_pid)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=a.gpu,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
    for method in a.methods:
        task=f'r{a.repeat}_{method}';directory=OUT/'runs'/task;cp=directory/'checkpoint_12000.pt'
        if (directory/'complete.json').exists():
            receipt=read(directory/'complete.json');bound(receipt['checkpoint'])
        else:
            write(status,dict(status='training',method=method,repeat=a.repeat,host=a.host,gpu=a.gpu,task=task,started_unix=time.time()))
            if a.host=='local':
                occupancy=free_gpu(a.gpu);write(OUT/f'dispatch_{task}.json',dict(host='local',gpu=a.gpu,prelaunch=occupancy))
                call([PYTHON,'-u',str(HERE/'train.py'),'--method',method,'--repeat',a.repeat,'--out',str(directory)],OUT/f'{task}.log',env)
            else:
                remote='/home/ubuntu/3DGS/4dsr';rr=remote+'/output/'+HERE.name;unit='4dsr-cg-'+task.lower().replace('_','-')+'-20261006'
                check=f'nvidia-smi -i {shlex.quote(a.gpu)} --query-gpu=memory.used,utilization.gpu --format=csv,noheader; nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name --format=csv,noheader'
                pre=subprocess.check_output(['ssh','-o','ConnectTimeout=20',a.host,check],text=True)
                gpu_uuid=read(OUT/'remote_verification.json').get('reserved_uuid','GPU-b79cd3fe-81f0-f449-30be-432e2857e517')
                assert gpu_uuid not in pre,('Remote GPU occupied',pre)
                shell=f'cd {remote} && source activate_a100.sh && export CUDA_VISIBLE_DEVICES={shlex.quote(a.gpu)} OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 && exec python -u experiments/{HERE.name}/train.py --method {shlex.quote(method)} --repeat {a.repeat} --out {rr}/runs/{task} > {rr}/{task}.log 2>&1'
                command=shlex.join(['systemd-run','--user','--wait','--unit='+unit,'--property=WorkingDirectory='+remote,'bash','-lc',shell])
                write(OUT/f'dispatch_{task}.json',dict(host=a.host,gpu=a.gpu,unit=unit,prelaunch=pre,command=command))
                call(['ssh','-o','ConnectTimeout=20','-o','ServerAliveInterval=30',a.host,command],OUT/f'{task}_remote.log')
                call(['rsync','-az',a.host+':'+rr+'/runs/'+task+'/',str(directory)+'/'],OUT/f'{task}_retrieval.log')
                receipt=read(directory/'complete.json');bound(receipt['checkpoint'])
        assert read(directory/'complete.json')['updates']==6000
        write(status,dict(status='training_complete_evaluating',task=task,host=a.host,gpu=a.gpu))
        evaluate(cp,task,a.evaluation_dependency_pid)
    write(status,dict(status='batch_completed',methods=a.methods,repeat=a.repeat,host=a.host))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--host',choices=['local','a100-train'],required=True);p.add_argument('--gpu',required=True);p.add_argument('--repeat',required=True);p.add_argument('--methods',nargs='+',required=True);p.add_argument('--training-dependency-pid',type=int,default=0);p.add_argument('--evaluation-dependency-pid',type=int,default=0);a=p.parse_args()
    try:main(a)
    except BaseException:
        write(OUT/f'worker_{a.host}_r{a.repeat}.json',dict(status='failed',traceback=traceback.format_exc()));raise
