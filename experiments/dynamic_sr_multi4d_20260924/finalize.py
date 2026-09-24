"""Exit-event join, uniform GPU evaluation, fixed tables and visual artifacts."""
import argparse
import importlib.util
import os
from pathlib import Path
import select
import subprocess
import traceback
from summarize import read,write

ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('leases',ROOT/'experiments/dynamic_sr_soft_motion_20260924/run_job.py')
leases=importlib.util.module_from_spec(spec);spec.loader.exec_module(leases)


def main():
    p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True);a=p.parse_args();s=read(a.spec)
    out=Path(s['out']);out.mkdir(parents=True,exist_ok=False);status=dict(status='waiting_exit_events')
    def state(label):status.update(status=label,utc=leases.stamp());write(out/'status.json',status)
    try:
        state('waiting_exit_events');fds=[]
        for job in s['dependencies']:
            try:fds.append(os.pidfd_open(job['pid']))
            except ProcessLookupError:pass
        while fds:
            ready,_,_=select.select(fds,[],[])
            for fd in ready:os.close(fd);fds.remove(fd)
        for job in s['dependencies']:assert read(job['receipt'])['status']==job['status'],job
        state('uniform_evaluation')
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=s['gpu'],OMP_NUM_THREADS='4',FOURDSR_UPSTREAM=s['wu_upstream'])
        def run(label,cmd,gpu=False):
            if gpu:leases.require_available(leases.gpu_snapshot(s['gpu']))
            state(label)
            with (out/(label+'.log')).open('w') as log:subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        methods={};history=read(ROOT/'output/dynamic_sr_geometry_residual_20260924/cook_methods.json')['methods']
        for old,new in [('U','U6000'),('W','W6000'),('B4','B4'),('G','G')]:
            h=history[old];methods[new]=dict(checkpoint=h['checkpoint'],train_dir=h['train_dir'],updates=6000,
                evaluations={k:h['evaluations'][k+'_6000'] for k in ['train_fixed','dev','test']})
        with leases.lock_file(f'/tmp/4dsr_soft_gpu_{s["gpu"]}.lock'):
            for method in ['U','W']:
                folder=Path(s[method+'40'])
                for step in [18000,40000]:
                    name=method+str(step);checkpoint=folder/f'train/checkpoint_{step}.pt';evaluations={}
                    for split in ['train_fixed','dev','test']:
                        label=f'{name}_{split}';dest=out/label
                        run(label,[s['wu_python'],str(ROOT/'experiments/dynamic_sr_detail_supervision_20260924/evaluate.py'),
                            '--manifest',s['manifest'],'--checkpoint',str(checkpoint),'--out',str(dest),'--split',split,'--method',name,
                            '--original-prior-cameras','cam02,cam06,cam12,cam18','--roi-protocol',s['roi']],gpu=True)
                        evaluations[split]=str(dest)
                    methods[name]=dict(checkpoint=str(checkpoint),train_dir=str(folder/'train'),updates=step,evaluations=evaluations)
        pipeline=Path(s['multi_pipeline'])
        for name in ['M0','M1']:
            methods[name]=dict(checkpoint=str(pipeline/name/'fine_20000.pt'),train_dir=str(pipeline/name),updates=20000,
                evaluations={split:str(pipeline/f'{name}_eval_{split}') for split in ['train_fixed','dev','test']})
        index=out/'methods.json';write(index,dict(methods=methods))
        run('tables',[s['wu_python'],str(HERE/'summarize.py'),'--methods',str(index),'--out',str(out/'tables')])
        run('storage',[s['wu_python'],str(HERE/'storage_cost.py'),'--methods',str(index),'--out',str(out/'storage')])
        run('views',[s['wu_python'],str(HERE/'export_views.py'),'--methods',str(index),'--manifest',s['manifest'],
            '--roi-protocol',s['roi'],'--teacher-index',s['teacher'],'--out',str(out/'views')])
        state('artifacts_ready_for_scientific_review');write(out/'complete.json',dict(status=status['status'],methods=str(index),
            note='No automatic quality claim or discussion dispatch. Review joint metrics, fixed native ROIs/videos and cost before conditional independent-scene validation.'))
    except BaseException:
        status.update(status='failed',traceback=traceback.format_exc());write(out/'failed.json',status);raise
    finally:write(out/'status.json',status)


if __name__=='__main__':main()
