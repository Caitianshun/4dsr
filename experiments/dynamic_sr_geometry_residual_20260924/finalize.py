"""Process-exit continuation: verify return, unified eval, summary, fixed views.

No timer/model polling. Run under systemd, pass the exact parent service MainPID.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from summarize import read,write,sha
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
OUT=ROOT/'output/dynamic_sr_geometry_residual_20260924'
DETAIL=ROOT/'experiments/dynamic_sr_detail_supervision_20260924'


def main():
    p=argparse.ArgumentParser();p.add_argument('--scene',choices=['cook','discussion'],required=True);p.add_argument('--wait-pid',type=int,required=True)
    a=p.parse_args();folder=OUT/f'{a.scene}_finalization_v1';folder.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();state=dict(scene=a.scene,pid=os.getpid(),status='waiting_for_process_exit',wait_pid=a.wait_pid)
    def update(status,**kw):
        state.update(status=status,elapsed_seconds=time.monotonic()-started,**kw);write(folder/'status.json',state)
    def call(label,cmd):
        event=dict(status='running',command=cmd,started_unix=time.time());write(folder/f'{label}_event.json',event)
        tick=time.monotonic()
        with (folder/f'{label}.log').open('x') as f:rc=subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT).returncode
        event.update(status='completed' if rc==0 else 'failed',returncode=rc,seconds=time.monotonic()-tick);write(folder/f'{label}_event.json',event)
        if rc:raise RuntimeError(f'{label} failed: see {folder}/{label}.log')
    try:
        update('waiting_for_process_exit')
        waiter='import os,select\ntry:\n fd=os.pidfd_open('+str(a.wait_pid)+');select.select([fd],[],[]);os.close(fd)\nexcept ProcessLookupError:pass\n'
        subprocess.run(['/usr/bin/python3','-c',waiter],check=True)
        batch=OUT/('cook_batch_v1' if a.scene=='cook' else 'discussion_batch_cts_v1')
        assert read(batch/'complete.json')['status']=='completed_sequential_batch'
        if a.scene=='discussion':
            assert read(batch/'.transport/complete.json')['status']=='completed_remote_batch_and_verified_return'
        common=read(OUT/f'{a.scene}_local_common.json');new={}
        for method in ['G','L']:
            job=batch/method
            if a.scene=='discussion':
                update('unified_evaluation_'+method)
                job=OUT/f'discussion_{method}_unified_v1'
                opts=dict(common,method=method,out=str(job),reuse_train=str(batch/method/'train'))
                cmd=[sys.executable,str(HERE/'run_job.py')]
                for key,value in opts.items():
                    if isinstance(value,bool):
                        if value:cmd.append('--'+key.replace('_','-'))
                    else:cmd+=['--'+key.replace('_','-'),str(value)]
                call('unified_'+method,cmd)
            assert read(job/'complete.json')['status']=='completed_and_evaluated'
            item=read(job/'methods-entry.json')['methods'][method];item['group']='main';new[method]=item
        old=read(ROOT/f'output/dynamic_sr_detail_supervision_20260924/{a.scene}_methods.json')
        methods={name:copy.deepcopy(old['methods'][name]) for name in ['U','W','B4']}
        methods.update(new)
        mapping=OUT/f'{a.scene}_methods.json';write(mapping,dict(scene=old['scene'],methods=methods,
            protocol='U/G/L same initialization, targets, complete sampling and per-scene hardware; W strong supervision, B4 historical coverage'))
        update('summary_and_fixed_views')
        call('summary',[sys.executable,str(HERE/'summarize.py'),'--manifest',common['manifest'],'--methods',str(mapping),'--out',str(OUT/f'{a.scene}_summary_v1')])
        call('views',[sys.executable,str(DETAIL/'export_views.py'),'--manifest',common['manifest'],'--methods',str(mapping),'--roi-protocol',common['roi_protocol'],'--out',str(OUT/f'{a.scene}_views_v1')])
        call('inference_storage',[sys.executable,str(HERE/'package_inference.py'),'--methods',str(mapping),'--out',str(OUT/f'{a.scene}_inference_storage_v1')])
        index=[]
        for method in ['G','L']:
            for step in [1200,6000]:
                checkpoint=batch/method/'train'/f'checkpoint_{step}.pt'
                index.append(dict(scene=old['scene'],method=method,step=step,path=str(checkpoint),sha256=sha(checkpoint),bytes=checkpoint.stat().st_size,status='completed_and_evaluated',training_host='local' if a.scene=='cook' else 'cts'))
        write(folder/'checkpoint_index.json',dict(checkpoints=index))
        update('completed_unified_summary_views_storage',methods_file=str(mapping));write(folder/'complete.json',state)
    except BaseException:
        update('failed',traceback=traceback.format_exc());write(folder/'failed.json',state);raise

if __name__=='__main__':main()
