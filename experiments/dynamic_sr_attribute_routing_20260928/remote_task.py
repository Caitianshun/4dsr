"""Inspect immutable attempts and launch only an unoccupied registered physical GPU."""
import argparse
import subprocess
import shlex
from dv_common import *
from readiness import validate
from service_wait import info

def inspect(task_id):
    p=require_run_root(OUT);task=next(t for t in p['task_plan'] if t['task_id']==task_id)
    base=OUT/'runs'/task_id;attempts=[]
    for f in sorted(base.glob('attempt_*/dispatch.json')):
        dispatch=read(f);attempt=f.parent;payload=attempt/'train'
        state=info(dispatch['unit']);points=[]
        for step in [9000,12000]:
            cp=payload/f'checkpoint_{step}.pt';meta=cp.with_suffix('.json')
            if cp.exists() and meta.exists():
                d=read(meta);m=d['metadata']
                assert sha(cp)==d['sha256']
                assert m['task_id']==task_id and m['method']==task['arm'] and m['repeat']==task['repeat']
                assert m['protocol_sha256']==sha(OUT/'protocol.json') and m['schedule_sha256']==p['schedules'][task['repeat']]['sha256']
                config=read(payload/'config.json')
                assert config['sources']==read(OUT/'research_readiness.json')['sources']
                points.append(dict(step=step,path=str(cp.relative_to(ROOT)),sha256=d['sha256'],metadata=m))
        # A written checkpoint is not an accepted endpoint if finite/source checks failed later.
        complete=read(payload/'complete.json') if (payload/'complete.json').exists() else None
        failed=read(payload/'failed.json') if (payload/'failed.json').exists() else None
        attempts.append(dict(path=str(attempt.relative_to(ROOT)),unit=dispatch['unit'],state=state,
            active=state.get('ActiveState') in ['active','activating'],checkpoints=points,complete=complete,failed=failed))
    valid12000=[x for a in attempts if a['complete'] and a['complete']['status']=='completed' and not a['failed'] for x in a['checkpoints'] if x['step']==12000]
    valid9000=[x for a in attempts for x in a['checkpoints'] if x['step']==9000]
    return dict(task=task,attempts=attempts,active=[a for a in attempts if a['active']],
        endpoint=valid12000[-1] if valid12000 else None,
        resume=valid9000[-1] if valid9000 else None,
        budget=read(OUT/'budget.json') if (OUT/'budget.json').exists() else None)

def launch(task_id,attempt_number,resume=None):
    p=require_run_root(OUT);validate(p);snapshot=inspect(task_id)
    assert not snapshot['active'] and snapshot['endpoint'] is None
    assert len(snapshot['attempts'])==attempt_number-1
    task=snapshot['task'];gpu=p['training']['physical_gpu']
    occ=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory','--format=csv'],text=True)
    assert gpu not in occ,'Registered training GPU has another compute process'
    if resume:
        assert snapshot['resume'] and resume==snapshot['resume']['path']
    attempt=OUT/'runs'/task_id/f'attempt_{attempt_number:02d}'
    # Parent dispatch state and training payload are separate so train's exclusive creation stays useful.
    assert not attempt.exists();attempt.mkdir(parents=True)
    unit=f'4dsr-{RUN_ID.replace("_","-")}-{task_id.lower().replace("_","-")}-a{attempt_number:02d}.service'
    payload=attempt/'train'
    argv=['python','-u',str(HERE/'train.py'),'--run-root',str(OUT),'--method',task['arm'],'--repeat',task['repeat'],'--out',str(payload)]
    if resume:argv+=['--resume',str(ROOT/resume)]
    shell='cd '+shlex.quote(str(ROOT))+' && source activate_a100.sh && export CUDA_VISIBLE_DEVICES='+shlex.quote(gpu)+' FOURDSR_RUN_ROOT='+shlex.quote(str(OUT))+' && exec '+shlex.join(argv)+' > '+shlex.quote(str(attempt/'train.log'))+' 2>&1'
    write(attempt/'dispatch.json',dict(unit=unit,task_id=task_id,attempt=attempt_number,resume=resume,protocol_sha256=sha(OUT/'protocol.json'),physical_gpu=gpu,prelaunch_occupancy=occ,argv=argv,started_unix=time.time()))
    subprocess.run(['systemd-run','--user','--unit='+unit,'--property=WorkingDirectory='+str(ROOT),'/bin/bash','-c',shell],check=True)
    return dict(unit=unit,attempt=str(attempt.relative_to(ROOT)))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('operation',choices=['inspect','launch']);ap.add_argument('--task',required=True);ap.add_argument('--attempt',type=int);ap.add_argument('--resume');ap.add_argument('--run-root',type=Path,default=OUT);a=ap.parse_args();require_run_root(a.run_root)
    # Protocol/data and source hashes are checked on every actual launch, not a display-only snapshot.
    result=inspect(a.task) if a.operation=='inspect' else launch(a.task,a.attempt,a.resume)
    print(json.dumps(result),flush=True)
