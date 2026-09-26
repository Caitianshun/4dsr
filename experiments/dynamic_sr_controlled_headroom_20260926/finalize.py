"""Join process-exit events, complete U train16 fields, then fixed artifacts."""
import argparse
import os
from pathlib import Path
import select
import subprocess
import traceback
from run_segment import ROOT,leases


def wait_pid(pid,needle):
    try:fd=os.pidfd_open(pid)
    except ProcessLookupError:return
    try:
        try:command=Path(f'/proc/{pid}/cmdline').read_bytes()
        except FileNotFoundError:command=b''
        assert not command or needle.encode() in command,'PID identity differs'
        select.select([fd],[],[])
    finally:os.close(fd)


def main():
    p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True);a=p.parse_args();s=leases.read(a.spec)
    base=a.spec.resolve().parent;controller=ROOT/s['out'];out=base/'final_v1';out.mkdir(parents=True,exist_ok=False)
    status=dict(status='waiting_prefix_exit',pid=os.getpid(),started_utc=leases.stamp());leases.write(out/'status.json',status)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=s['gpu'],OMP_NUM_THREADS='4',FOURDSR_UPSTREAM=s['wu_upstream']);here=Path(__file__).parent
    def run(label,cmd,gpu=False):
        if gpu:leases.require_available(leases.gpu_snapshot(s['gpu']))
        status.update(status=label);leases.write(out/'status.json',status)
        with (out/(label+'.log')).open('w') as log:subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    try:
        wait_pid(leases.read(base/'prefix_service.json')['pid'],'run_prefixes.py')
        assert leases.read(controller/'complete.json')['status']=='prefixes_evaluated_tails_dispatched'
        status['status']='waiting_tail_return_and_uniform_evaluation_events';leases.write(out/'status.json',status)
        for method in ['Z','O']:
            job=leases.read(controller/(method+'_remote.json'));wait_pid(job['return_pid'],'return_tail.py')
            assert leases.read(controller/(method+'_return_status.json'))['status']=='returned_and_uniformly_evaluated'
        methods={};history=leases.read(ROOT/'output/dynamic_sr_multi4d_20260924/final_v2/methods.json')['methods'];spec=s['jobs']['Z']
        with leases.lock_file(f'/tmp/4dsr_soft_gpu_{s["gpu"]}.lock'):
            for step in [6000,18000,40000]:
                name='U'+str(step);h=history[name];dest=out/(name+'_train_fixed')
                run(name+'_train_fixed',[s['wu_python'],str(here/'evaluate.py'),'--manifest',str(ROOT/spec['manifest']),
                    '--checkpoint',h['checkpoint'],'--teacher-index',str(ROOT/spec['teacher_index']),'--roi-protocol',str(ROOT/spec['roi']),
                    '--out',str(dest),'--split','train_fixed','--method',name],gpu=True)
                methods[name]={**h,'evaluations':{**h['evaluations'],'train_fixed':str(dest)},'method':'U','privileged_train_hr':False}
        for method in ['Z','O']:
            for step in [6000,18000,40000]:
                name=method+('_HR_PRIVILEGED' if method=='O' else '')+str(step)
                folder=base/(method+('_prefix_v1' if step==6000 else '_tail_v1'))
                evaluations={split:str(folder/f'eval_{split}_{step}' if step==6000 else base/'uniform_v1'/f'{method}{step}_{split}') for split in ['train_fixed','dev','test']}
                methods[name]=dict(method=method,privileged_train_hr=method=='O',updates=step,checkpoint=str(folder/f'train/checkpoint_{step}.pt'),train_dir=str(folder/'train'),evaluations=evaluations)
        index=out/'methods.json';leases.write(index,dict(methods=methods))
        main_index=out/'primary_methods.json';leases.write(main_index,dict(methods={n:v for n,v in methods.items() if v['updates']==40000}))
        run('target_reference',[s['wu_python'],str(here/'reference_diagnostics.py'),'--manifest',str(ROOT/spec['manifest']),
            '--teacher-index',str(ROOT/spec['teacher_index']),'--out',str(out/'target_reference.json')])
        run('storage',[s['wu_python'],str(ROOT/'experiments/dynamic_sr_multi4d_20260924/storage_cost.py'),'--methods',str(main_index),'--out',str(out/'storage')])
        run('tables',[s['wu_python'],str(here/'summarize.py'),'--methods',str(index),'--root',str(base),'--out',str(out/'tables')])
        run('views',[s['wu_python'],str(here/'export_views.py'),'--methods',str(main_index),'--manifest',str(ROOT/spec['manifest']),
            '--roi-protocol',str(ROOT/spec['roi']),'--teacher-index',str(ROOT/spec['teacher_index']),'--out',str(out/'views')])
        status.update(status='artifacts_ready_for_scientific_and_visual_review',finished_utc=leases.stamp(),methods=str(index))
        leases.write(out/'complete.json',status)
    except BaseException:
        status.update(status='failed',traceback=traceback.format_exc());leases.write(out/'failed.json',status);raise
    finally:leases.write(out/'status.json',status)


if __name__=='__main__':main()
