"""Reconnect to systemd using a Linux process-exit event, not completion polling."""
import argparse
import json
import os
import select
import subprocess

def info(unit):
    result = subprocess.run(['systemctl','--user','show',unit,'--property=MainPID,ActiveState,SubState,Result,ExecMainStatus'],capture_output=True,text=True,check=True)
    return dict(x.split('=',1) for x in result.stdout.splitlines() if '=' in x)

def wait(unit):
    state = info(unit)
    if state.get('ActiveState') not in ['active','activating']: return state
    pid = int(state.get('MainPID','0'))
    if pid <= 0: raise RuntimeError(('Active service has no stable MainPID; inspect before dispatch',state))
    try: fd = os.pidfd_open(pid)
    except ProcessLookupError: return info(unit)
    try:
        while not select.select([fd],[],[],3600)[0]:
            print(json.dumps(dict(event='hourly_exception_watchdog',unit=unit,state=info(unit))),flush=True)
    finally: os.close(fd)
    return info(unit)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--unit',required=True);a=ap.parse_args()
    print(json.dumps(wait(a.unit)),flush=True)
