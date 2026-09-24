"""Wait on exact transfer PIDs, then launch persistent CTS acceptance and collect.

Invocation: python finalize_after_transfers.py ENV_PID DATA_PID DATA_STATUS_JSON
This is a one-off deployment coordinator, not an experiment scheduler.
"""
import json
import os
from pathlib import Path
import selectors
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / 'deployment/cts_20260921'
REMOTE = '/home/cts/Project/4DSR'
state = dict(status='running', stage='wait_transfers', started_unix=time.time(), pid=os.getpid())

def write():
    (D/'verification/finalize_status.json').write_text(json.dumps(state,indent=2)+'\n')

def run(command):
    print('RUN', command, flush=True)
    subprocess.run(command, check=True)

try:
    write()
    for pid in map(int,sys.argv[1:3]):
        try:
            fd=os.pidfd_open(pid)
        except ProcessLookupError:
            continue
        with selectors.DefaultSelector() as selector:
            selector.register(fd,selectors.EVENT_READ)
            selector.select()
        os.close(fd)
    env=json.loads((D/'verification/environment_reuse.json').read_text())
    data=json.loads(Path(sys.argv[3]).read_text())
    assert env['status']=='complete', env
    assert data['status']=='complete', data
    state.update(stage='remote_acceptance');write()
    # These coordinator files were created after the initial data inventory.
    run(['rsync','-a',str(D/'run_acceptance.sh'),f'cts:{REMOTE}/deployment/cts_20260921/'])
    run(['ssh','cts',f'mkdir -p {REMOTE}/deployment/cts_20260921/logs'])
    run(['ssh','cts','systemd-run --user --wait --unit=cts-sr-acceptance-v2-20260921 '
         '--property=WorkingDirectory='+REMOTE+' '
         '--property=StandardOutput=append:'+REMOTE+'/deployment/cts_20260921/logs/acceptance.log '
         '--property=StandardError=append:'+REMOTE+'/deployment/cts_20260921/logs/acceptance.log '
         '/bin/bash '+REMOTE+'/deployment/cts_20260921/run_acceptance.sh'])
    state.update(stage='collect');write()
    run(['rsync','-a',f'cts:{REMOTE}/deployment/cts_20260921/verification/',str(D/'verification/remote')+'/'])
    run(['rsync','-a',f'cts:{REMOTE}/deployment/cts_20260921/logs/',str(D/'logs/remote')+'/'])
    acceptance=json.loads((D/'verification/remote/acceptance_exit.json').read_text())
    assert acceptance['returncode']==0 and acceptance['stage']=='complete', acceptance
    state.update(status='complete',stage=None)
except BaseException as exc:
    state.update(status='failed',error=repr(exc))
    raise
finally:
    state['finished_unix']=time.time();write()
