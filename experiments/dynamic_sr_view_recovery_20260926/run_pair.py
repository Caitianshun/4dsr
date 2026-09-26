"""Persistent exit-chained paired training, evaluation, fixed gate, and archives."""
import argparse
import fcntl
import os
import subprocess
import sys
import time
import traceback
from shared import *
import importlib.util

_spec=importlib.util.spec_from_file_location('recovery_gate',Path(__file__).with_name('summarize.py'))
gate=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(gate)

def main():
    p=paths();protocol=read(OUT/'protocol.json');out=OUT/'controller_v1';out.mkdir(exist_ok=False)
    started=time.time();commands=[];methods={};status={};gpu=protocol['physical_gpu']
    lock=(OUT/'paired_gpu.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    env={**os.environ,'CUDA_VISIBLE_DEVICES':gpu,'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4'}
    def state(**kw):
        status.update(kw);status.update(pid=os.getpid(),updated_unix=time.time(),wall_s=time.time()-started)
        write_json(out/'status.json',status)
    def run(cmd,label):
        entry=dict(label=label,argv=[str(v) for v in cmd],started_unix=time.time())
        commands.append(entry);write_json(out/'commands.json',commands);state(status='running',phase=label)
        with (out/(label+'.log')).open('w') as log:
            proc=subprocess.Popen([str(v) for v in cmd],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            state(child_pid=proc.pid);rc=proc.wait()
        entry.update(returncode=rc,seconds=time.time()-entry['started_unix']);write_json(out/'commands.json',commands)
        if rc:raise RuntimeError(f'{label} exited {rc}')
    try:
        assert read(OUT/'p0_decision.json')['allow_p1']
        assert read(OUT/'engineering_v1/complete.json')['status'].startswith('passed')
        occupancy=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv'],text=True)
        write_json(out/'resource_start.json',dict(gpu=gpu,compute_apps=occupancy,environment=env.get('FOURDSR_UPSTREAM')))
        for line in occupancy.splitlines()[1:]:
            if line.startswith(gpu):assert '/opt/todesk/' in line,line
        baseline={}
        for camera,split in [('cam00','test'),('cam01','dev')]:baseline[camera]=gate.values(p['methods']['U6000']['evaluations'][split])
        for step in [18000,40000]:
            for arm in ['C_joint','F_app']:
                job=OUT/f'{arm}_{step}';train=job/'train';job.mkdir(exist_ok=False)
                resume=Path(protocol['start']) if step==18000 else OUT/f'{arm}_18000/train/checkpoint_18000.pt'
                cmd=[sys.executable,'-u',Path(__file__).with_name('train.py'),'--manifest',p['manifest'],'--resume',resume,
                    '--schedule',p['schedule'],'--protocol',OUT/'protocol.json','--lr-curve',p['lr_curve'],'--target-index',p['teacher'],
                    '--method',arm,'--stop',str(step),'--out',train]
                run(cmd,f'{arm}_{step}_train')
                assert read(train/'complete.json')['status']=='completed'
                checkpoint=train/f'checkpoint_{step}.pt';evaluations={}
                # This call is reached on successful training process exit, with no polling.
                for split in ['dev','test','train_fixed']:
                    evaluation=job/f'eval_{split}';evaluations[split]=str(evaluation)
                    cmd=[sys.executable,'-u',Path(__file__).with_name('evaluate.py'),'--manifest',p['manifest'],'--checkpoint',checkpoint,
                        '--out',evaluation,'--split',split,'--roi-protocol',p['roi'],'--teacher-index',p['teacher'],'--method',f'{arm}{step}','--no-video']
                    run(cmd,f'{arm}_{step}_{split}')
                    assert read(evaluation/'complete.json')['status']=='completed_evaluation'
                run([sys.executable,'-u',Path(__file__).with_name('train76.py'),'--manifest',p['manifest'],'--checkpoint',checkpoint,'--out',job/'eval_train76'],f'{arm}_{step}_train76')
                methods[f'{arm}{step}']=dict(checkpoint=str(checkpoint),sha256=sha256(checkpoint),train_dir=str(train),evaluations=evaluations,train76=str(job/'eval_train76'),arm=arm,updates=step)
                write_json(OUT/'methods.json',dict(methods=methods))
                write_json(OUT/'checkpoint_index.json',dict(checkpoints=[dict(method=k,path=v['checkpoint'],sha256=v['sha256'],status='completed_evaluated',gpu=gpu) for k,v in methods.items()]))
            def vals(arm):return {camera:gate.values(methods[f'{arm}{step}']['evaluations'][split]) for camera,split in [('cam00','test'),('cam01','dev')]}
            decision=gate.decide(protocol,vals('C_joint'),vals('F_app'),baseline,step)
            decision.update(protocol_sha256=sha256(OUT/'protocol.json'),paired_training_gpu=gpu,evaluation_gpu=gpu,completed_unix=time.time())
            write_json(OUT/f'decision_{step}.json',decision);write_json(OUT/'scientific_decision.json',decision)
            if not decision['extend']:break
        run([sys.executable,ROOT/'experiments/dynamic_sr_multi4d_20260924/storage_cost.py','--methods',OUT/'methods.json','--out',OUT/'inference_archives'], 'inference_archives')
        costs={}
        for label,method in methods.items():
            complete=read(Path(method['train_dir'])/'complete.json');config=read(Path(method['train_dir'])/'config.json')
            costs[label]=dict(**{k:complete[k] for k in ['actual_updates','total_updates','elapsed_s','train_s','peak_gb']},render_calls=2*complete['actual_updates'],
                actual_trainable=config['policy']['actual_trainable'],stored_parameters=config['policy']['stored_parameters'],gpu=config['gpu'])
        write_json(OUT/'cost.json',dict(segments=costs,archives=read(OUT/'inference_archives/complete.json'),
            shared_p0_seconds=read(OUT/'p0/drift/summary.json')['seconds'],wall_s=time.time()-started,
            caveat='Incremental experiment cost. Original LR parent/selection/teacher preparation reused, not zero from-scratch cost. Frozen parameters remain in inference archive.'))
        state(status='completed',scientific_status=decision['status'],child_pid=None,phase='awaiting_human_readable_and_visual_review')
    except BaseException:
        state(status='failed',error=traceback.format_exc())
        write_json(OUT/'scientific_decision.json',dict(status='failed',controller_status=str(out/'status.json'),error=traceback.format_exc()))
        raise

if __name__=='__main__':main()
