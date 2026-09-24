"""Durable shared coarse -> preflight -> M0/M1 -> immediate fixed evaluation."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('gpu_leases',ROOT/'experiments/dynamic_sr_soft_motion_20260924/run_job.py')
leases=importlib.util.module_from_spec(spec);spec.loader.exec_module(leases)


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--gpu',required=True)
    p.add_argument('--upstream',type=Path,required=True);p.add_argument('--coarse-only',action='store_true');p.add_argument('--reuse-coarse',type=Path)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    protocol=ROOT/'output/dynamic_sr_multi4d_20260924'
    common=['--upstream',str(a.upstream),'--adapter',str(protocol/'adapter_v1'),'--manifest',str(ROOT/'data/dynamic_sr/n3dv_prepared/cook_spinach/manifest.json'),
        '--init',str(protocol/'cook_init_v1/initialization/points_lr.npz'),'--schedule',str(protocol/'cook_multi_schedule_v1.json')]
    teacher=ROOT/'output/dynamic_sr_detail_supervision_20260924/teacher_inventory_v1/cook_spinach/teacher_index.json'
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=a.gpu,CUDA_HOME='/usr/local/cuda-12.8',TORCH_CUDA_ARCH_LIST='8.6;12.0',MAX_JOBS='6',
        TORCH_EXTENSIONS_DIR=str(a.upstream/'.extensions'),MULTI4D_UPSTREAM=str(a.upstream),OMP_NUM_THREADS='4')
    env['PATH']='/usr/local/cuda-12.8/bin:'+env['PATH']
    status=dict(status='starting',pid=os.getpid(),gpu=a.gpu,started_utc=leases.stamp())
    def run(label,cmd):
        snap=leases.gpu_snapshot(a.gpu);leases.require_available(snap)
        status.update(status=label);leases.write(a.out/'status.json',status)
        leases.write(a.out/(label+'_launch.json'),dict(command=cmd,snapshot=snap))
        with (a.out/(label+'.log')).open('w') as log:subprocess.run(cmd,env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    try:
        with leases.lock_file(f'/tmp/4dsr_soft_gpu_{a.gpu}.lock'):
            coarse=a.reuse_coarse or a.out/'coarse'
            assert not (coarse.parent/'invalidated.json').exists(), 'Refusing invalidated coarse checkpoint'
            if not a.reuse_coarse:run('coarse',[sys.executable,str(HERE/'multi_train.py'),*common,'--out',str(coarse),'--method','coarse'])
            assert leases.read(coarse/'complete.json')['status']=='fine0_prepared'
            fine0=coarse/'fine0.pt'
            if not a.coarse_only:
                check=a.out/'preflight'
                run('preflight',[sys.executable,str(HERE/'multi_preflight.py'),'--checkpoint',str(fine0),'--manifest',common[5],'--teacher-index',str(teacher),'--out',str(check)])
                for method,step in [('M0',0),('M1',0),('M1',2999),('M1',3049),('M1',9999),('M1',15999)]:
                    if step==2999:
                        run('prepare_fixtures',[sys.executable,str(HERE/'prepare_fixtures.py'),'--checkpoint',str(a.out/'check_M1_0/fine_20.pt'),'--out',str(a.out/'fixtures')])
                    endpoint=20 if step==0 else step+2
                    run(f'check_{method}_{step}',[sys.executable,str(HERE/'multi_train.py'),*common,'--out',str(a.out/f'check_{method}_{step}'),
                        '--method',method,'--resume',str(fine0 if step==0 else a.out/'fixtures'/f'fixture_{step}.pt'),'--teacher-index',str(teacher),'--smoke','--stop',str(endpoint)])
                leases.write(a.out/'preflight_passed.json',dict(status='passed',fine0=leases.identity(fine0),engineering_boundary_fixtures=True))
                run('evaluation_compatibility',[sys.executable,str(HERE/'multi_evaluate.py'),'--checkpoint',str(a.out/'check_M0_0/fine_20.pt'),
                    '--manifest',common[5],'--teacher-index',str(teacher),'--out',str(a.out/'evaluation_compatibility'),'--split','train_fixed','--method','M0',
                    '--roi-protocol',str(ROOT/'output/dynamic_sr_soft_motion_20260924/roi_registry/cook_spinach/roi_protocol.json')])
                for method in ['M0','M1']:
                    out=a.out/method
                    run(method,[sys.executable,str(HERE/'multi_train.py'),*common,'--out',str(out),'--method',method,'--resume',str(fine0),'--teacher-index',str(teacher)])
                    assert leases.read(out/'complete.json')['status']=='completed'
                    for split in ['train_fixed','dev','test']:
                        run(f'{method}_eval_{split}',[sys.executable,str(HERE/'multi_evaluate.py'),'--checkpoint',str(out/'fine_20000.pt'),
                            '--manifest',common[5],'--teacher-index',str(teacher),'--out',str(a.out/f'{method}_eval_{split}'),'--split',split,'--method',method,
                            '--roi-protocol',str(ROOT/'output/dynamic_sr_soft_motion_20260924/roi_registry/cook_spinach/roi_protocol.json')])
        status.update(status='fine0_prepared' if a.coarse_only else 'completed_and_evaluated',finished_utc=leases.stamp())
        leases.write(a.out/'complete.json',status)
    except BaseException:
        status.update(status='failed',traceback=traceback.format_exc());leases.write(a.out/'failed.json',status);raise
    finally:leases.write(a.out/'status.json',status)


if __name__=='__main__':main()
