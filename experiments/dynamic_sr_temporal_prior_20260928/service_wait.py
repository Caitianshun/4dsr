"""Reconnect to systemd using a Linux process-exit event, not completion polling."""
import argparse
import json
import os
import select
import subprocess
from pathlib import Path

def process_matches(pid, start_ticks):
    """Distinguish an existing batch owner from a later process reusing its PID."""
    try:
        fields = (Path('/proc') / str(pid) / 'stat').read_text().rsplit(')', 1)[1].split()
    except (FileNotFoundError, ProcessLookupError):
        return False
    return int(fields[19]) == start_ticks

def open_pidfd(pid, force_libc=False):
    """Open the same Linux exit-event descriptor in Python builds lacking the API."""
    if hasattr(os, 'pidfd_open') and not force_libc:
        return os.pidfd_open(pid)
    import ctypes
    libc = ctypes.CDLL(None, use_errno=True)
    fn = libc.pidfd_open
    fn.argtypes = [ctypes.c_int, ctypes.c_uint]
    fn.restype = ctypes.c_int
    fd = fn(pid, 0)
    if fd < 0:
        errno = ctypes.get_errno()
        raise OSError(errno, os.strerror(errno), pid)
    return fd

def info(unit):
    result = subprocess.run(['systemctl','--user','show',unit,'--property=MainPID,ActiveState,SubState,Result,ExecMainStatus'],capture_output=True,text=True,check=True)
    return dict(x.split('=',1) for x in result.stdout.splitlines() if '=' in x)

def wait(unit):
    state = info(unit)
    if state.get('ActiveState') not in ['active','activating']: return state
    pid = int(state.get('MainPID','0'))
    if pid <= 0: raise RuntimeError(('Active service has no stable MainPID; inspect before dispatch',state))
    try: fd = open_pidfd(pid)
    except ProcessLookupError: return info(unit)
    try:
        while not select.select([fd],[],[],3600)[0]:
            print(json.dumps(dict(event='hourly_exception_watchdog',unit=unit,state=info(unit))),flush=True)
    finally: os.close(fd)
    return info(unit)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--unit',required=True);a=ap.parse_args()
    print(json.dumps(wait(a.unit)),flush=True)
