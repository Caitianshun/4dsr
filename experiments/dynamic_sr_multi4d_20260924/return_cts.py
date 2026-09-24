"""Wait on the remote service PID exit event and return verified W40 artifacts."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import traceback


def main():
    p=argparse.ArgumentParser();p.add_argument('--pid',required=True,type=int);a=p.parse_args()
    root=Path(__file__).resolve().parents[2];out=root/'output/dynamic_sr_multi4d_20260924'
    remote='/home/cts/Project/4DSR/runs/multi4d_20260924'
    # pidfd is an exit event; no status polling or minute-level model checks.
    code=f'import os,select; fd=os.pidfd_open({a.pid}); select.select([fd],[],[]); os.close(fd)'
    status={'status':'waiting_for_remote_exit','pid':a.pid,'host':'cts'}
    try:
        result=subprocess.run(['ssh','-o','ServerAliveInterval=30','-o','ServerAliveCountMax=3','cts','python3 -c '+__import__('shlex').quote(code)],text=True,capture_output=True)
        # Already-exited PID is valid only if the remote terminal receipt exists.
        status['wait_returncode']=result.returncode;status['wait_stderr']=result.stderr
        subprocess.run(['ssh','cts',f'test -f {remote}/output/dynamic_sr_multi4d_20260924/W40_v1/complete.json -o -f {remote}/output/dynamic_sr_multi4d_20260924/W40_v1/failed.json'],check=True)
        subprocess.run(['rsync','-aL',f'cts:{remote}/output/dynamic_sr_multi4d_20260924/W40_v1/',str(out/'W40_v1')+'/'],check=True)
        local=out/'W40_v1'
        idx=json.loads((local/'checkpoint_index.json').read_text()) if (local/'checkpoint_index.json').exists() else {}
        for step,row in idx.items():
            path=local/f'train/checkpoint_{step}.pt'
            with path.open('rb') as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
            assert actual==row['sha256'];row['remote_path']=row['path'];row['path']=str(path)
        (out/'returned_checkpoint_index.json').write_text(json.dumps(idx,indent=2)+'\n')
        status['status']='returned_completed' if (local/'complete.json').exists() else 'returned_failed'
    except BaseException:
        status.update(status='return_failed',traceback=traceback.format_exc());raise
    finally:(out/'cts_return_status.json').write_text(json.dumps(status,indent=2)+'\n')


if __name__=='__main__':main()
