"""Run a concrete continuation on a Linux process exit event, never by polling."""
import argparse
import json
import os
from pathlib import Path
import select
import subprocess
import traceback


def main():
    p=argparse.ArgumentParser();p.add_argument('--spec',required=True,type=Path);a=p.parse_args()
    s=json.loads(a.spec.read_text());state={'status':'waiting','pid':s['pid']}
    try:
        try:fd=os.pidfd_open(s['pid'])
        except ProcessLookupError:fd=None
        if fd is not None:select.select([fd],[],[]);os.close(fd)
        receipt=json.loads(Path(s['receipt']).read_text());assert receipt['status']==s['required_status'],receipt
        state['status']='running_continuation'
        Path(s['status_path']).write_text(json.dumps(state)+'\n')
        subprocess.run(s['command'],check=True,cwd=s['cwd'])
        state['status']='completed'
    except BaseException:state.update(status='failed',traceback=traceback.format_exc());raise
    finally:Path(s['status_path']).write_text(json.dumps(state,indent=2)+'\n')


if __name__=='__main__':main()
