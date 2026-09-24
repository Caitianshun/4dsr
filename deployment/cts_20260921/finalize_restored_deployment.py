"""Complete verified archive recovery, incremental sync, and CTS acceptance."""
import json
import os
from pathlib import Path
import selectors
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
D=ROOT/'deployment/cts_20260921'
REMOTE='/home/cts/Project/4DSR'
state=dict(status='running',stage='wait_environment',started_unix=time.time(),pid=os.getpid())

def write():
    (D/'verification/finalize_status.json').write_text(json.dumps(state,indent=2)+'\n')

def run(args,**kwargs):
    print('RUN',args,flush=True)
    return subprocess.run(args,check=True,**kwargs)

try:
    write()
    try:
        fd=os.pidfd_open(int(sys.argv[1]))
    except ProcessLookupError:
        fd=None
    if fd is not None:
        with selectors.DefaultSelector() as selector:
            selector.register(fd,selectors.EVENT_READ);selector.select()
        os.close(fd)
    assert json.loads((D/'verification/environment_reuse.json').read_text())['status']=='complete'
    state['stage']='wait_data_restore';write()
    waiter='import os,selectors; fd=os.pidfd_open('+str(int(sys.argv[2]))+'); s=selectors.DefaultSelector(); s.register(fd,selectors.EVENT_READ); s.select(); os.close(fd)'
    import shlex
    wait=run(['ssh','cts', 'python3 -c '+shlex.quote(waiter)+' || test -f '+REMOTE+'/deployment/cts_20260921/verification/restore_images.json'])
    run(['rsync','-a',f'cts:{REMOTE}/deployment/cts_20260921/verification/restore_images.json',str(D/'verification/restore_images.json')])
    assert json.loads((D/'verification/restore_images.json').read_text())['status']=='complete'
    state['stage']='final_incremental_sync';write()
    run([sys.executable,str(D/'sync_to_cts.py'),'--apply'])
    state['stage']='remote_acceptance';write()
    run(['ssh','cts',f'mkdir -p {REMOTE}/deployment/cts_20260921/logs'])
    run(['ssh','cts','systemd-run --user --wait --unit=cts-sr-acceptance-20260921 '
         '--property=WorkingDirectory='+REMOTE+' '
         '--property=StandardOutput=append:'+REMOTE+'/deployment/cts_20260921/logs/acceptance.log '
         '--property=StandardError=append:'+REMOTE+'/deployment/cts_20260921/logs/acceptance.log '
         '/bin/bash '+REMOTE+'/deployment/cts_20260921/run_acceptance.sh'])
    state['stage']='collect';write()
    run(['rsync','-a',f'cts:{REMOTE}/deployment/cts_20260921/verification/',str(D/'verification/remote')+'/'])
    run(['rsync','-a',f'cts:{REMOTE}/deployment/cts_20260921/logs/',str(D/'logs/remote')+'/'])
    acceptance=json.loads((D/'verification/remote/acceptance_exit.json').read_text())
    assert acceptance['returncode']==0 and acceptance['stage']=='complete',acceptance
    state.update(status='complete',stage=None)
except BaseException as exc:
    state.update(status='failed',error=repr(exc))
    raise
finally:
    state['finished_unix']=time.time();write()
