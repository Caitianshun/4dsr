"""Remote pidfd exit -> local copy/hash -> PRO6000 reevaluation; no polling."""
import argparse
import os
from pathlib import Path
import shlex
import subprocess
import traceback
from run_segment import ROOT,leases


def main():
    p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True);p.add_argument('--method',choices=['Z','O'],required=True);p.add_argument('--pid',type=int,required=True);a=p.parse_args()
    plan=leases.read(a.spec);s=plan['jobs'][a.method];root=ROOT/plan['out'];remote=plan['remote_root']
    tail=leases.read(ROOT/s['remote_spec']);out=ROOT/tail['out'];status=dict(status='waiting_remote_pidfd',method=a.method,remote_pid=a.pid,started_utc=leases.stamp())
    receipt=root/(a.method+'_return_status.json');leases.write(receipt,status)
    try:
        code=f'''import os,select
try:
 fd=os.pidfd_open({a.pid})
except ProcessLookupError:
 fd=None
if fd is not None:
 try:
  cmd=open('/proc/{a.pid}/cmdline','rb').read()
 except FileNotFoundError:
  cmd=b''
 assert not cmd or b'run_segment.py' in cmd, 'PID identity differs'
 select.select([fd],[],[])
 os.close(fd)
'''
        host=s.get('tail_host','cts')
        command=['ssh','cts','python3 -c '+shlex.quote(code)] if host=='cts' else ['/usr/bin/python3','-c',code]
        waited=subprocess.run(command,capture_output=True,text=True)
        status.update(wait_returncode=waited.returncode,wait_stderr=waited.stderr,status='returning_remote_artifacts');leases.write(receipt,status)
        if host=='cts':
            out.mkdir(parents=True,exist_ok=False)
            subprocess.run(['rsync','-aL',f'cts:{remote}/{tail["out"]}/',str(out)+'/'],check=True)
        assert waited.returncode==0,waited.stderr
        assert leases.read(out/'complete.json')['status']=='completed_and_evaluated'
        index=leases.read(out/'checkpoint_index.json');local={}
        for step,row in index.items():
            path=out/f'train/checkpoint_{step}.pt';identity=leases.identity(path);assert identity['sha256']==row['sha256'];local[step]=identity
        leases.write(out/'returned_checkpoint_index.json',local)
        status.update(status='returned_verified_waiting_uniform_gpu',checkpoint_index=local);leases.write(receipt,status)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=plan['gpu'],OMP_NUM_THREADS='4',FOURDSR_UPSTREAM=plan['wu_upstream'])
        with leases.lock_file(f'/tmp/4dsr_soft_gpu_{plan["gpu"]}.lock'):
            for step in [18000,40000]:
                for split in ['train_fixed','dev','test']:
                    label=f'{a.method}{step}_{split}';dest=root.parent/'uniform_v1'/label
                    leases.require_available(leases.gpu_snapshot(plan['gpu']))
                    status.update(status='uniform_evaluation',stage=label);leases.write(receipt,status)
                    cmd=[plan['wu_python'],str(Path(__file__).with_name('evaluate.py')),'--manifest',str(ROOT/s['manifest']),
                        '--checkpoint',str(out/f'train/checkpoint_{step}.pt'),'--teacher-index',str(ROOT/s['teacher_index']),
                        '--roi-protocol',str(ROOT/s['roi']),'--out',str(dest),'--split',split,'--method',a.method+('_HR_PRIVILEGED' if a.method=='O' else '')]
                    with (root/(label+'.log')).open('w') as log:subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
                    assert leases.read(dest/'complete.json')['status']=='completed_evaluation'
        status.update(status='returned_and_uniformly_evaluated',finished_utc=leases.stamp())
    except BaseException:
        status.update(status='failed',traceback=traceback.format_exc());raise
    finally:leases.write(receipt,status)


if __name__=='__main__':main()
