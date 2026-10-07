"""Finite LR-only interventions and frozen cause diagnostics; GPU lock shared."""
import argparse,fcntl,subprocess,select,traceback
from cg_common import *
PYTHON='/home/cai_tianshun/Project/4dgs/.venv/bin/python'
def wait_pid(pid):
    try:fd=os.pidfd_open(pid)
    except ProcessLookupError:return
    select.select([fd],[],[]);os.close(fd)
def call(args,log,env):
    with Path(log).open('a') as f:subprocess.run([PYTHON,'-u',*args],stdout=f,stderr=subprocess.STDOUT,env=env,check=True)
def main(a):
    if a.dependency:wait_pid(a.dependency)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
    with (OUT/'evaluation_gpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        from worker import free_gpu
        occupancy=free_gpu('1');p=read(OUT/'protocol.json');parent=bound(p['parent']);manifest=bound(p['manifest'])
        render=OUT/'cause_render_v1'
        write(OUT/'cause_worker_state.json',dict(status='rendering_causes',gpu=occupancy))
        if not (render/'render_diagnostics.json').exists():
            call([str(HERE/'cause_diagnostics.py'),'--mode','render','--manifest',str(manifest),'--resume',str(parent),'--out',str(render)],OUT/'cause_render.log',env)
        # LR-only causes do not depend on R/G calibration. One shared U6000
        # evaluation supplies step0 for all three identical initial states.
        zero=OUT/'cause_probes/U6000_0'
        if not (zero/'probe_evaluation.json').exists():
            call([str(HERE/'cause_diagnostics.py'),'--mode','probe','--manifest',str(manifest),'--resume',str(parent),
              '--label','U6000','--probe-step','0','--surface-support',str(render/'surface_support.npz'),'--out',str(zero)],OUT/'U6000_cause_evaluation.log',env)
        probes=[zero/'probe_evaluation.json']
        for method in ['P_joint','P_xyz','P_SH']:
            directory=OUT/'runs'/method
            write(OUT/'cause_worker_state.json',dict(status='cause_probe_training',method=method,physical_gpu='1'))
            if not (directory/'complete.json').exists():
                call([str(HERE/'probe_train.py'),'--method',method,'--repeat','1','--out',str(directory),'--stop','500'],OUT/f'{method}.log',env)
            assert read(directory/'complete.json')['updates']==500
            for step in [100,500]:
                cp=directory/f'checkpoint_{6000+step}.pt'
                dest=OUT/'cause_probes'/f'{method}_{step}'
                if not (dest/'probe_evaluation.json').exists():
                    call([str(HERE/'cause_diagnostics.py'),'--mode','probe','--manifest',str(manifest),'--resume',str(cp),
                      '--label',method,'--probe-step',str(step),'--surface-support',str(render/'surface_support.npz'),'--out',str(dest)],OUT/f'{method}_evaluation.log',env)
                probes.append(dest/'probe_evaluation.json')
        c1=OUT/'runs/r1_C1/checkpoint_6500.pt'
        if c1.exists():
            dest=OUT/'cause_probes/C1_500'
            if not (dest/'probe_evaluation.json').exists():
                call([str(HERE/'cause_diagnostics.py'),'--mode','probe','--manifest',str(manifest),'--resume',str(c1),'--label','C1','--probe-step','500',
                  '--surface-support',str(render/'surface_support.npz'),'--out',str(dest)],OUT/'C1_cause_evaluation.log',env)
            probes.append(dest/'probe_evaluation.json')
        call([str(HERE/'cause_matrix.py'),'--cached',str(OUT/'cause_cached_v1/cached_diagnostics.json'),'--render',str(render/'render_diagnostics.json'),
          '--probes',*[str(x) for x in probes],'--out',str(OUT/'cause_matrix')],OUT/'cause_matrix.log',env)
        write(OUT/'cause_worker_state.json',dict(status='completed',updates=1500,probe_training_gpu='RTX3090',
          C1_training_gpu='PRO6000',C1_comparison_hardware_caveat=True,probe_reports=[entry(x) for x in probes]))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--dependency',type=int,default=0);a=ap.parse_args()
    try:main(a)
    except BaseException:write(OUT/'cause_worker_state.json',dict(status='failed',traceback=traceback.format_exc()));raise
