"""Serial, process-exit-driven comparison on one remote GPU.

The caller starts this under a persistent service. Each training process return
immediately starts its evaluator. No completion polling is used here.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[2]
SCRIPTS=Path(__file__).resolve().parent


def now():
    return datetime.now(timezone.utc).isoformat()


def write(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n');t.replace(p)


def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--scene',required=True)
    ap.add_argument('--gpu',required=True)
    ap.add_argument('--sr4d-root',type=Path,required=True)
    ap.add_argument('--sr4d-python',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--smoke',action='store_true')
    a=ap.parse_args()
    manifest=ROOT/'data/sr4d_author_20260922'/a.scene/'manifest.json'
    m=json.loads(manifest.read_text())
    assert m['comparison_protocol']=='sr4d_author_lr_v1'
    out=a.out.resolve()
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    logs=out/'logs';logs.mkdir()
    wu=ROOT/'.venv/bin/python'
    # Preserve the venv interpreter symlink: resolving it selects system Python
    # and loses the isolated packages/rasterizer.
    sr=a.sr4d_python.expanduser().absolute()
    env=os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=a.gpu,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',
               FOURDSR_UPSTREAM=str(ROOT/'vendor/4dgs'),TORCH_HOME=str(ROOT/'.cache/torch'),
               MPLBACKEND='Agg',PYTHONFAULTHANDLER='1')
    cameras='cam02,cam06,cam12,cam18' if a.scene=='cook_spinach' else 'cam02,cam04,cam08,cam12'
    assert set(cameras.split(',')) <= set(m['splits']['train'])
    coarse,fine=(3200,200) if a.smoke else (20000,40000)
    control_steps=100 if a.smoke else 51800
    state=dict(status='running',started_at=now(),pid=os.getpid(),scene=a.scene,gpu=a.gpu,
               smoke=a.smoke,manifest_sha256=sha(manifest),steps=[],
               completion='Child process return immediately triggers evaluation; no polling.',
               planned_sr4d_loops=coarse+fine,planned_wu_loops=(20+20+20+control_steps if a.smoke else 1000+6000+1200+control_steps))
    snapshots=out/'source_snapshot';snapshots.mkdir()
    import shutil
    state['sources']={}
    for p in SCRIPTS.glob('*.py'):
        shutil.copy2(p,snapshots/p.name);state['sources'][p.name]=sha(p)
    write(out/'status.json',state)

    def run(name,cmd):
        record=dict(name=name,command=list(map(str,cmd)),status='running',started_at=now(),log=str(logs/f'{name}.log'))
        state['steps'].append(record);state['current']=name;write(out/'status.json',state)
        print(json.dumps(dict(event='start',name=name,time=now())),flush=True)
        started=time.monotonic()
        with open(record['log'],'x') as log:
            child=subprocess.Popen(record['command'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            record['pid']=child.pid;write(out/'status.json',state)
            rc=child.wait()
        record.update(returncode=rc,seconds=time.monotonic()-started,finished_at=now(),status='complete' if rc==0 else 'failed')
        write(out/'status.json',state)
        if rc:raise RuntimeError(f'{name} exited {rc}; see {record["log"]}')
        print(json.dumps(dict(event='complete',name=name,time=now())),flush=True)

    def evaluate(name,method,path,stage='fine',iteration=None):
        py=sr if method=='sr4d' else wu
        cmd=[py,SCRIPTS/'evaluate_comparison.py','--method',method,'--manifest',manifest,
             '--out',out/(name+'_evaluation'),'--prior-cameras',cameras]
        if method=='sr4d':
            cmd += ['--upstream',a.sr4d_root,'--run',path,'--stage',stage,'--iteration',str(iteration)]
        else:
            cmd += ['--upstream',ROOT/'vendor/4dgs','--checkpoint',path]
        if a.smoke:cmd+=['--limit-test','3','--skip-train']
        run(name+'_eval',cmd)

    try:
        srout=out/'sr4d'
        cmd=[sr,SCRIPTS/'train_sr4d.py','--upstream',a.sr4d_root,'--manifest',manifest,
             '--output',srout,'--coarse-steps',str(coarse),'--fine-steps',str(fine)]
        if a.smoke:cmd+=['--smoke']
        run('sr4d_train',cmd)
        evaluate('sr4d_final','sr4d',srout,iteration=fine)
        if not a.smoke:
            evaluate('sr4d_coarse','sr4d',srout,stage='coarse',iteration=coarse)
            for it in (6000,18000):evaluate(f'sr4d_fine_{it}','sr4d',srout,iteration=it)
        prior=[wu,SCRIPTS/'generate_author_prior.py','--manifest',manifest,
               '--network',ROOT/'vendor/swinir/network_swinir.py',
               '--checkpoint',ROOT/'weights/001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth',
               '--cameras',cameras]
        # A teacher cache is deterministic and can be reused between smoke/main.
        run('prior_generation',prior)
        parent=out/'wu_parent_native'
        run('wu_native',[wu,SCRIPTS/'wu_entry.py','--entry','warmup','--task','warmup',
             '--manifest',manifest,'--out',parent,'--coarse-steps',str(20 if a.smoke else 1000),
             '--fine-steps',str(20 if a.smoke else 6000),'--observation','native_lr'])
        integrated=out/'wu_parent_integrated'
        run('wu_integrated',[wu,SCRIPTS/'wu_entry.py','--entry','warmup','--task','branch',
             '--manifest',manifest,'--out',integrated,'--checkpoint',parent/'checkpoint_final.pt',
             '--mode','lr_integrated','--steps',str(20 if a.smoke else 1200),'--prior-cameras',cameras])
        evaluate('wu_parent','wu',integrated/'checkpoint_final.pt')
        for name,teacher,weight in [('wu_sr_w01','sr','.1'),('wu_lr_only','none','0')]:
            path=out/name
            milestones=str(control_steps) if a.smoke else '6000,18000,31800,51800'
            run(name,[wu,SCRIPTS/'wu_entry.py','--entry','controlled','--manifest',manifest,
                      '--checkpoint',integrated/'checkpoint_final.pt','--out',path,
                      '--teacher',teacher,'--weight',weight,'--steps',str(control_steps),
                      '--milestones',milestones,'--prior-cameras',cameras])
            evaluate(name,'wu',path/'checkpoint_final.pt')
            if not a.smoke:
                for it in (6000,18000):evaluate(f'{name}_{it}','wu',path/f'checkpoint_{it}.pt')
        state.update(status='complete',finished_at=now(),current=None)
        write(out/'complete.json',dict(status='complete',scene=a.scene,finished_at=now(),
                                      steps=len(state['steps']),smoke=a.smoke))
    except BaseException:
        state.update(status='failed',finished_at=now(),traceback=traceback.format_exc())
        raise
    finally:
        write(out/'status.json',state)
        # A local subscriber can block on this FIFO for immediate rsync after
        # the batch exits. Nonblocking open avoids delaying training teardown
        # if no listener is attached. A ready JSON event is always retained.
        write(out/'completion_event.json',dict(status=state['status'],finished_at=now(),out=str(out)))
        fifo=out.parent/(out.name+'.completion.fifo')
        if fifo.exists():
            try:
                fd=os.open(fifo,os.O_WRONLY|os.O_NONBLOCK)
                os.write(fd,(json.dumps({'status':state['status'],'out':str(out)})+'\n').encode());os.close(fd)
            except OSError:pass


if __name__=='__main__':
    main()
