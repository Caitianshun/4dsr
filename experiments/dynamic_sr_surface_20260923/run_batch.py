"""Three fixed branches on two identical A100s; evaluate on child exit."""
import argparse, csv, json, os, subprocess, sys, time, traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    t=path.with_suffix('.tmp');t.write_text(json.dumps(value,indent=2));t.replace(path)


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--checkpoint',required=True)
    p.add_argument('--out',required=True);p.add_argument('--gpu0',required=True);p.add_argument('--gpu1',required=True)
    args=p.parse_args();root=Path(__file__).resolve().parents[2];out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    assert args.gpu0!=args.gpu1
    scripts=Path(__file__).parent;start=time.monotonic()
    state=dict(status='running',started_utc=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),
        queues={args.gpu0:['joint','local_residual'],args.gpu1:['global_residual']},
        fixed_endpoint=6000,steps=6000,evals=['dev1200','dev6000','test6000'])
    write(out/'status.json',state)
    def child(label,cmd,gpu):
        snap=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv,noheader'],text=True)
        for row in csv.reader(snap.splitlines()):
            if row[0].strip()==gpu and row[2].strip() not in ['/usr/libexec/gnome-remote-desktop-daemon','/opt/todesk/bin/ToDesk_Session']:
                raise RuntimeError(f'GPU occupied {row}')
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=gpu,CUDA_DEVICE_ORDER='PCI_BUS_ID',OMP_NUM_THREADS='4',MPLBACKEND='Agg')
        receipt=dict(status='running',label=label,command=cmd,gpu=gpu,gpu_before=snap,started_utc=datetime.now(timezone.utc).isoformat())
        tick=time.monotonic();path=out/'events'/f'{label}.json'
        with (out/f'{label}.log').open('w') as log:
            proc=subprocess.Popen(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
            receipt['pid']=proc.pid;write(path,receipt);rc=proc.wait()
        receipt.update(status='completed' if rc==0 else 'failed',returncode=rc,elapsed_s=time.monotonic()-tick)
        write(path,receipt)
        if rc:raise RuntimeError(f'{label}: {rc}')
    def queue(gpu,branches):
        for branch in branches:
            folder=out/branch
            child(f'{branch}_train',[sys.executable,str(scripts/'train.py'),'--manifest',args.manifest,'--checkpoint',args.checkpoint,
                '--out',str(folder),'--branch',branch,'--steps','6000','--milestones','1200,6000',
                '--seed','20260923','--prior-cameras','cam00,cam06,cam12,cam18'],gpu)
            for split,step in [('dev',1200),('dev',6000),('test',6000)]:
                child(f'{branch}_eval_{split}_{step}',[sys.executable,str(scripts/'evaluate.py'),'--manifest',args.manifest,
                    '--checkpoint',str(folder/f'checkpoint_{step}.pt'),'--out',str(folder/f'eval_{split}_{step}'),'--split',split],gpu)
    errors=[]
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures={pool.submit(queue,gpu,branches):gpu for gpu,branches in state['queues'].items()}
        for future in as_completed(futures):
            try:future.result()
            except BaseException:
                failure=dict(gpu=futures[future],traceback=traceback.format_exc());errors.append(failure)
                write(out/f'failed_{len(errors)}.json',failure)
    if errors:
        state.update(status='failed',errors=errors);write(out/'status.json',state);raise RuntimeError(errors)
    draws=[]
    for branch in ['joint','global_residual','local_residual']:
        folder=out/branch;c=json.loads((folder/'complete.json').read_text());assert c['parameter_updates']==6000
        draws.append(c['draw_sha256'])
        for split,step in [('dev',1200),('dev',6000),('test',6000)]:
            e=json.loads((folder/f'eval_{split}_{step}/complete.json').read_text());assert e['status']=='completed_evaluation'
    assert len(set(draws))==1
    state.update(status='completed_and_evaluated',elapsed_s=time.monotonic()-start,finished_utc=datetime.now(timezone.utc).isoformat(),draw_sha256=draws[0])
    write(out/'status.json',state);write(out/'complete.json',state)


if __name__=='__main__':main()
