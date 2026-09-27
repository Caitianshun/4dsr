"""Persistent bounded batch with evaluation attached to actual process exits."""
import argparse
import fcntl
import subprocess
import traceback
from probe_common import *

def main():
    pa=argparse.ArgumentParser();pa.add_argument('--protocol',type=Path,required=True);pa.add_argument('--resume',action='store_true');a=pa.parse_args();p=load_protocol(a.protocol);out=Path(p['_root']);dest=out/'pipeline';dest.mkdir(exist_ok=a.resume);lock=(out/'gpu.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);commands=[];start=time.time();env={**os.environ,'CUDA_VISIBLE_DEVICES':p['physical_gpu'],'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4'}
    def status(**kw):write(dest/'status.json',dict(pid=os.getpid(),seconds=time.time()-start,updated=time.time(),**kw))
    def run(script,args,label):
        if '--out' in args:
            target=Path(args[args.index('--out')+1])
            if a.resume and (target/'complete.json').exists():return
            if a.resume and script=='probe_updates.py' and target.exists():args=[*args,'--resume']
        status(status='running',phase=label);row=dict(label=label,argv=[sys.executable,'-u',str(HERE/script),'--protocol',str(a.protocol.resolve()),*[str(x) for x in args]],started=time.time());commands.append(row);write(dest/'commands.json',commands)
        with (dest/(label+'.log')).open('w') as f:r=subprocess.run(row['argv'],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
        row.update(returncode=r.returncode,seconds=time.time()-row['started']);write(dest/'commands.json',commands)
        if r.returncode:raise RuntimeError(label+' failed; preserve artifacts before repairing')
    def evaluate(ck,target,arm,rid,step,privileged=False):
        args=['--out',target,'--checkpoint',ck,'--arm',arm,'--repeat-id',rid,'--step',step]
        if privileged:args+=['--privileged']
        run('evaluate_probe.py',args,f'r{rid}_{arm}_{step}_'+('HR' if privileged else 'train'))
    try:
        apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv'],text=True)
        for line in apps.splitlines()[1:]:
            if line.startswith(p['physical_gpu']):assert '/opt/todesk/' in line,line
        write(dest/'resource_start.json',dict(apps=apps,gpu=p['physical_gpu'],started=time.time()))
        assert read(out/'engineering/complete.json')['status']=='passed'
        run('probe_updates.py',['--out',out/'updates_initial','--state-id','U6000'],'updates_initial')
        run('footprint_bins.py',['--out',out/'footprints'],'footprint_bins')
        initial=Path(read(out/'initial_state.json')['path']);evaluate(initial,out/'initial_train_eval','U6000',0,0)
        rounds=[]
        for rid in [1,2]:
            order=p['order'] if rid==1 else p['order'][::-1]
            for arm in order:
                target=out/f'repeat{rid}'/arm
                run('train_probe.py',['--out',target/'train','--arm',arm,'--repeat-id',rid],f'r{rid}_{arm}_train')
                for step in [300,600]:evaluate(target/'train'/f'checkpoint_{step}.pt',target/f'eval_train_{step}',arm,rid,step)
            run('probe_updates.py',['--out',out/f'updates_AB_repeat{rid}','--state-id',f'AB600_repeat{rid}','--checkpoint',out/f'repeat{rid}/AB/train/checkpoint_600.pt'],f'updates_AB_repeat{rid}')
            run('decide.py',['--repeat-id',rid,'--out',out/f'round{rid}_decision.json'],f'decide_r{rid}');decision=read(out/f'round{rid}_decision.json');rounds.append(decision)
            if not decision['passed']:break
        write(out/'decision.json',dict(status='frozen',frozen_unix=time.time(),rounds=rounds,round_count=len(rounds),repeat_executed=len(rounds)==2,reproducible_conditional_signal=len(rounds)==2 and all(r['passed'] for r in rounds),decision_information='training LR,teacher and conditional SH diagnostic only',training_updates=len(rounds)*2400,unrun_reason=rounds[-1]['unrun_reason'] if not rounds[-1]['passed'] else 'Bounded maximum complete; no further search'))
        # New HR/dev image loading is possible only after this frozen receipt.
        evaluate(initial,out/'privileged/U6000','U6000',0,0,True)
        for rid in range(1,len(rounds)+1):
            for arm in p['order']:evaluate(out/f'repeat{rid}/{arm}/train/checkpoint_600.pt',out/f'privileged/repeat{rid}/{arm}',arm,rid,600,True)
        status(status='completed',phase='all_training_and_evaluation_complete')
    except BaseException:status(status='failed',error=traceback.format_exc());raise
if __name__=='__main__':main()
