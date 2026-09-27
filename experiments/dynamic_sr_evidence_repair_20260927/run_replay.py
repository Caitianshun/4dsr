"""Bounded same-GPU ordinary/restart controls; every exit chains evaluation."""
import argparse
import fcntl
import subprocess
import time
import traceback
from context import *

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['tails','long'],required=True);a=parser.parse_args()
    p=paths();protocol=read(OUT/'protocol.json');out=OUT/('replay_'+a.phase);out.mkdir(exist_ok=False)
    lock=(OUT/'pro6000.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    gpu=protocol['physical_gpu'];env={**os.environ,'CUDA_VISIBLE_DEVICES':gpu,'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4'}
    commands=[];methods={};started=time.time()
    def state(**kw):write_json(out/'status.json',dict(pid=os.getpid(),updated=time.time(),seconds=time.time()-started,**kw))
    def run(cmd,label):
        state(status='running',phase=label);row=dict(label=label,argv=[str(x) for x in cmd],started=time.time());commands.append(row);write_json(out/'commands.json',commands)
        with (out/(label+'.log')).open('w') as f:result=subprocess.run(row['argv'],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
        row.update(returncode=result.returncode,seconds=time.time()-row['started']);write_json(out/'commands.json',commands)
        if result.returncode:raise RuntimeError(label+' failed')
    def evaluate(ck,dest,label):
        es={}
        for split in ['test','dev','train_fixed']:
            target=dest/('eval_'+split);es[split]=str(target)
            run([sys.executable,'-u',ROOT/'experiments/dynamic_sr_view_recovery_20260926/evaluate.py','--manifest',p['manifest'],'--checkpoint',ck,'--out',target,'--split',split,'--roi-protocol',p['roi'],'--teacher-index',p['teacher'],'--method',label,'--no-video'],label+'_'+split)
        target=dest/'eval_train76'
        run([sys.executable,'-u',ROOT/'experiments/dynamic_sr_view_recovery_20260926/train76.py','--manifest',p['manifest'],'--checkpoint',ck,'--out',target],label+'_train76')
        return dict(checkpoint=str(ck),checkpoint_sha256=sha256(ck),renderer_id='legacy_direct_v1',evaluations=es,train76=str(target))
    def job(label,parent_id,ck,stop,save_steps):
        d=out/label;d.mkdir();train=d/'train'
        run([sys.executable,'-u',HERE/'train_replay.py','--manifest',p['manifest'],'--resume',ck,'--schedule',p['schedule'],'--protocol',OUT/'protocol.json','--lr-curve',p['lr_curve'],'--target-index',p['teacher'],'--method','C_joint','--run-id',label,'--parent-run-id',parent_id,'--stop',stop,'--save-steps',*save_steps,'--out',train],label+'_train')
        assert read(train/'complete.json')['status']=='completed'
        for step in save_steps:
            checkpoint=train/f'checkpoint_{step}.pt';ed=d/f'endpoint_{step}';ed.mkdir();name=label+'_'+str(step)
            methods[name]=dict(run_id=label,parent_run_id=parent_id,segment=read(train/'complete.json'),train_dir=str(train),**evaluate(checkpoint,ed,name));write_json(out/'methods.json',dict(methods=methods))
        return train
    try:
        apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv'],text=True)
        for line in apps.splitlines()[1:]:
            if line.startswith(gpu):assert '/opt/todesk/' in line,line
        write_json(out/'resources_start.json',dict(gpu=gpu,apps=apps))
        if a.phase=='tails':
            # Read-only repeated evaluation is distinct from stochastic optimization.
            for i in [1,2]:
                d=out/f'readonly{i}';d.mkdir();methods[f'readonly{i}']=dict(run_id=f'readonly{i}',parent_run_id='U6000',**evaluate(p['start'],d,f'readonly{i}'));write_json(out/'methods.json',dict(methods=methods))
            fork=Path(read(PRIOR/'methods.json')['methods']['shared_17550']['checkpoint'])
            for i in [1,2]:job(f'Tail{i}','historical_shared17550',fork,18000,protocol['replay']['save_steps'])
        else:
            live=job('LiveA','U6000',p['start'],18000,[12000,17550,18000])
            job('LiveB','U6000',p['start'],18000,[12000,17550,18000])
            job('Restart12','LiveA_12000',live/'checkpoint_12000.pt',18000,[17550,18000])
            job('Restart17550','LiveA_17550',live/'checkpoint_17550.pt',18000,[18000])
        state(status='completed',phase='all_exits_evaluated')
    except BaseException:state(status='failed',error=traceback.format_exc());raise

if __name__=='__main__':main()
