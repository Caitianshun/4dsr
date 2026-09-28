"""Persistent remote preparation, failure receipt written immediately on exit."""
import argparse
import subprocess
import traceback
from dv_common import *

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['repeat7','video7'],required=True);a=ap.parse_args()
    dest=OUT/'preparation';dest.mkdir(parents=True,exist_ok=True);tick=time.time()
    dep=ROOT/'deployment/temporal_prior_20260928'
    cmd=[str(dep/'venv/bin/python'),'-u',str(HERE/'prepare_prior.py'),'--mode',a.mode,'--repo',str(dep/'BasicVSR_PlusPlus'),
         '--weights',str(dep/'basicvsr_plusplus_reds4.pth'),'--out',str(OUT/'priors'/a.mode)]
    state=dict(status='running',mode=a.mode,pid=os.getpid(),started_unix=tick,argv=cmd,physical_gpu=os.environ['CUDA_VISIBLE_DEVICES'])
    write(dest/(a.mode+'_status.json'),state)
    with (dest/(a.mode+'.log')).open('w') as f:rc=subprocess.call(cmd,stdout=f,stderr=subprocess.STDOUT)
    state.update(status='completed' if rc==0 else 'failed',returncode=rc,seconds=time.time()-tick)
    if rc==0:
        idx=OUT/'priors'/a.mode/'prior_index.json';assert read(idx)['status']=='completed';state['index']=entry(idx)
    write(dest/(a.mode+'_status.json'),state)
    raise SystemExit(rc)

if __name__=='__main__':
    try:main()
    except Exception:
        write(OUT/'preparation'/f'worker_failure_{os.getpid()}.json',dict(status='failed',traceback=traceback.format_exc()));raise
