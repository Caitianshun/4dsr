"""Exit-triggered core consolidation; research/visual review remains explicit."""
import argparse, subprocess, traceback, time
from cg_common import *
from worker_async import wait_pid

def run(script, *args):
    # System Python owns pidfd orchestration; numerical CPU utilities need the
    # verified research environment. Disable CUDA for those read-only utilities.
    numerical = script in ['execution_index.py','figures.py']
    python = '/home/cai_tianshun/Project/4dgs/.venv/bin/python' if numerical else '/usr/bin/python3'
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4') if numerical else None
    with (OUT/'core_finalization.log').open('a') as log:
        subprocess.run([python,str(HERE/script),*map(str,args)],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)

def main(a):
    wait_pid(a.dispatcher_pid)
    dispatch=read(OUT/'module_dispatch_state.json')
    assert dispatch['status']=='dispatched_dependency_safe_async'
    for key in ['base_local_pid','base_remote_pid','local_worker_pid','remote_worker_pid']:
        wait_pid(dispatch[key])
    write(OUT/'core_finalization_state.json',dict(status='consolidating_complete_core',started_unix=time.time()))
    # Parent/module extras are safely reused; recover any base extra that became
    # ready after the earlier finite base batch selected its endpoint list.
    run('extra_worker.py','--include-modules')
    run('summarize.py')
    run('final_integrity.py')
    run('execution_index.py','--out',OUT/'accounting_final_core','--verify-checkpoints')
    run('figures.py')
    write(OUT/'core_finalization_state.json',dict(status='core_complete_pending_research_and_visual_review',
      final_integrity=entry(OUT/'final_integrity.json'),quality_summary=entry(OUT/'quality_summary.json'),
      accounting=entry(OUT/'accounting_final_core/execution_index.json'),figures=entry(OUT/'figures/index.json'),
      conditional_phase13_automatically_started=False,chat_notification_claimed=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dispatcher-pid',type=int,required=True);a=p.parse_args()
    try:main(a)
    except BaseException:
        write(OUT/'core_finalization_state.json',dict(status='failed',traceback=traceback.format_exc()));raise
