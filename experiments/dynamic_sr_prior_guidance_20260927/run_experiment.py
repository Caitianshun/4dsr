"""Persistent shared-prefix experiment with exit-triggered evaluation and gates."""
import os
import subprocess
import sys
import time
import traceback
import fcntl
from shared import *
sys.path.insert(0,str(Path(__file__).resolve().parent))
from summarize import values,early_gate,decide,delta


def main():
    p=paths();protocol=read(OUT/'protocol.json');out=OUT/'controller_v1';out.mkdir(exist_ok=False)
    lock=(OUT/'gpu.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    started=time.time();commands=[];status={};methods={};gpu=protocol['physical_gpu']
    env={**os.environ,'CUDA_VISIBLE_DEVICES':gpu,'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4'}
    def state(**kw):
        status.update(kw);status.update(pid=os.getpid(),updated_unix=time.time(),wall_s=time.time()-started)
        write_json(out/'status.json',status)
    def run(cmd,label):
        entry=dict(label=label,argv=[str(v) for v in cmd],started_unix=time.time());commands.append(entry)
        write_json(out/'commands.json',commands);state(status='running',phase=label)
        with (out/(label+'.log')).open('w') as log:
            proc=subprocess.Popen([str(v) for v in cmd],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            state(child_pid=proc.pid);rc=proc.wait()
        entry.update(returncode=rc,seconds=time.time()-entry['started_unix']);write_json(out/'commands.json',commands)
        if rc:raise RuntimeError(f'{label} exited {rc}')
    def job(arm,stop,resume,label):
        directory=OUT/label;directory.mkdir(exist_ok=False);train=directory/'train'
        run([sys.executable,'-u',Path(__file__).with_name('train.py'),'--manifest',p['manifest'],'--resume',resume,'--schedule',p['schedule'],'--protocol',OUT/'protocol.json','--lr-curve',p['lr_curve'],'--target-index',p['teacher'],'--method',arm,'--stop',stop,'--out',train],label+'_train')
        assert read(train/'complete.json')['status']=='completed'
        ck=train/f'checkpoint_{stop}.pt';evaluations={}
        for split in ['test','dev','train_fixed']:
            dest=directory/f'eval_{split}';evaluations[split]=str(dest)
            run([sys.executable,'-u',ROOT/'experiments/dynamic_sr_view_recovery_20260926/evaluate.py','--manifest',p['manifest'],'--checkpoint',ck,'--out',dest,'--split',split,'--roi-protocol',p['roi'],'--teacher-index',p['teacher'],'--method',label,'--no-video'],label+'_'+split)
            assert read(dest/'complete.json')['status']=='completed_evaluation'
        run([sys.executable,'-u',ROOT/'experiments/dynamic_sr_view_recovery_20260926/train76.py','--manifest',p['manifest'],'--checkpoint',ck,'--out',directory/'eval_train76'],label+'_train76')
        methods[label]=dict(checkpoint=str(ck),sha256=sha256(ck),arm=arm,updates=stop,train_dir=str(train),evaluations=evaluations,train76=str(directory/'eval_train76'))
        write_json(OUT/'methods.json',dict(methods=methods))
        write_json(OUT/'checkpoint_index.json',dict(checkpoints=[dict(method=k,path=v['checkpoint'],sha256=v['sha256'],status='completed_evaluated',gpu=gpu) for k,v in methods.items()]))
        return ck
    def val(label):return {c:values(methods[label]['evaluations'][s]) for c,s in [('cam00','test'),('cam01','dev')]}
    try:
        assert read(OUT/'engineering_v1/complete.json')['status'].startswith('passed')
        apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv'],text=True)
        write_json(out/'resource_start.json',dict(gpu=gpu,compute_apps=apps))
        for line in apps.splitlines()[1:]:
            if line.startswith(gpu):assert '/opt/todesk/' in line,line
        prefix12=job('C_joint',12000,protocol['start'],'shared_12000')
        fork=job('C_joint',17550,prefix12,'shared_17550')
        for arm in ['C_joint','P_late','A_late']:job(arm,18000,fork,arm+'_18000')
        early12=job('P_early',12000,protocol['start'],'P_early_12000')
        safety=early_gate(val('P_early_12000'),val('shared_12000'));write_json(OUT/'early_decision.json',safety)
        if safety['continue_to_18000']:job('P_early',18000,early12,'P_early_18000')
        u6={c:values(p['methods']['U6000']['evaluations'][s]) for c,s in [('cam00','test'),('cam01','dev')]}
        control=val('C_joint_18000');results={}
        for arm in ['P_late','A_late','P_early']:
            label=arm+'_18000'
            if label in methods:results[arm]=decide(val(label),control,u6)
            else:results[arm]=dict(status='stopped_at_12000_by_safety',endpoint18000=None)
        write_json(OUT/'decision.json',dict(status='completed',methods=results,early_safety=safety,control=control,u6000=u6,
            endpoints={k:val(k) for k in methods},tails={arm:{c:delta(val(arm+'_18000')[c],val('shared_17550')[c]) for c in control} for arm in ['C_joint','P_late','A_late']},
            main_view='cam00',stress_view='cam01',both_views_development=True,no_depth_training=True))
        costs={}
        for label,v in methods.items():
            c=read(Path(v['train_dir'])/'complete.json');costs[label]={k:c[k] for k in ['actual_updates','total_updates','elapsed_s','train_s','peak_gb']};costs[label]['render_calls']=2*c['actual_updates']
        write_json(OUT/'cost.json',dict(segments=costs,new_updates=sum(c['actual_updates'] for c in costs.values()),training_seconds=sum(c['train_s'] for c in costs.values()),shared_prefix_counted_once=True,gpu=gpu,wall_s=time.time()-started))
        state(status='completed',child_pid=None,phase='awaiting_visual_and_document_review')
    except BaseException:
        state(status='failed',error=traceback.format_exc());raise


if __name__=='__main__':main()
