"""Submit exactly once, wait on a remote exit event, then return artifacts.

No model polling, GPU eviction, lock removal or automatic training replay.
"""
import argparse
import ctypes
from datetime import datetime, timezone
import errno
import hashlib
import json
import os
from pathlib import Path
import select
import shlex
import stat
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
REMOTES = {'cts':'/home/cts/Project/4DSR/tasks/static_sr_wu_20261022'}
SHARED = '/home/cts/Project/4DSR'


def write(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def checked_return_path(path):
    member = Path(path)
    if member.is_absolute() or '..' in member.parts or not member.parts or str(member) == '.':
        raise ValueError('Unsafe return path')
    if member.parts[0] != 'output':
        raise ValueError('Only task output artifacts may be returned')
    return member


def checked_return_destination(path, kind):
    """Do not let an existing local link or wrong artifact type redirect rsync."""
    member = checked_return_path(path)
    if kind not in ('file', 'directory'):
        raise ValueError('Unexpected artifact type')
    root = ROOT.absolute()
    target = root / member

    def inspect():
        current = root
        for part in (None, *member.parts):
            if part is not None:
                current = current / part
            try:
                mode = current.lstat().st_mode
            except FileNotFoundError:
                continue
            if stat.S_ISLNK(mode):
                raise ValueError('Symlink local return path: ' + str(current))
            expected_directory = current != target or kind == 'directory'
            if expected_directory and not stat.S_ISDIR(mode):
                raise ValueError('Local return path must be a directory: ' + str(current))
            if not expected_directory and not stat.S_ISREG(mode):
                raise ValueError('Local return path must be a regular file: ' + str(current))

    inspect()
    (target if kind == 'directory' else target.parent).mkdir(parents=True, exist_ok=True)
    inspect()
    return target


def collect_outputs(host, remote, paths, terminal, directory):
    """Return files and directories; preserve failure evidence if later outputs are absent."""
    members = [str(checked_return_path(path)) for path in paths]
    code = '''import json,sys
from pathlib import Path
request=json.load(sys.stdin);root=Path(request['root']);result={}
for rel in request['paths']:
 p=root/rel
 for parent in [p,*p.parents]:
  if parent==root:break
  if parent.is_symlink():raise ValueError('Symlink return path: '+rel)
 result[rel]='directory' if p.is_dir() else 'file' if p.is_file() else 'missing'
print(json.dumps(result))
'''
    query = subprocess.run(['ssh','-o','BatchMode=yes',host,shlex.join(['/usr/bin/python3','-c',code])],
        input=json.dumps(dict(root=remote,paths=members)),text=True,capture_output=True,check=True,timeout=120)
    kinds = json.loads(query.stdout)
    returned, missing, errors = [], [], []
    for path in members:
        if kinds[path] == 'missing':
            missing.append(path)
            continue
        try:
            target = checked_return_destination(path, kinds[path])
            if kinds[path] == 'directory':
                source, destination = host+':'+remote+'/'+path+'/', str(target)+'/'
            else:
                source, destination = host+':'+remote+'/'+path, str(target)
            subprocess.run(['rsync','-a','--protect-args','--partial',source,destination],check=True,timeout=7200)
            returned.append(dict(path=path,kind=kinds[path]))
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            # A failed large transfer must not hide a later status/error record.
            errors.append(dict(path=path,kind=kinds[path],error_type=type(error).__name__,error=str(error)))
    succeeded = terminal.get('outcome') in ('success','completed')
    status = ('partial_failure_evidence' if missing or errors else
              'collected' if succeeded else 'failure_evidence_collected')
    write(directory/'collected.json',dict(status=status,
        remote_terminal=terminal,returned_paths=returned,missing_paths=missing,
        transfer_errors=errors,local_SHA_verification=False,utc=datetime.now(timezone.utc).isoformat()))
    if missing or errors or not succeeded:
        raise RuntimeError('Remote job failed or declared outputs are missing; available evidence returned, no replay')
    return returned


def wait_transport(job_id, timeout=300, base=Path('/var/lib/claude-training')):
    """Wait for broker dispatch/rejection events before watching the remote host."""
    candidates = [base/'jobs'/job_id/'run/status.json',
                  base/'jobs'/job_id/'submission.json',
                  base/'rejections'/(job_id+'.json')]
    libc = ctypes.CDLL('libc.so.6', use_errno=True)
    descriptor = libc.inotify_init1(os.O_CLOEXEC)
    if descriptor < 0:
        raise OSError(ctypes.get_errno())
    watched = set()
    deadline = time.monotonic() + timeout
    try:
        while True:
            # Check terminal files before installing watches: broker directory
            # ACLs may still be changing after a successful dispatch.
            for path in candidates:
                try:
                    value = json.loads(path.read_text())
                except (FileNotFoundError, PermissionError, json.JSONDecodeError):
                    continue
                if value.get('state') in ('rejected', 'ended', 'failed', 'outcome_unknown'):
                    return value
            for path in candidates:
                parent = path.parent
                while parent not in watched:
                    # Watch the nearest currently readable ancestor. IN_ATTRIB
                    # wakes this loop when the broker finishes setting ACLs.
                    wd = libc.inotify_add_watch(descriptor, os.fsencode(parent), 0x100 | 0x80 | 0x8 | 0x4)
                    if wd >= 0:
                        watched.add(parent)
                        break
                    error = ctypes.get_errno()
                    if error not in (errno.EACCES, errno.ENOENT):
                        raise OSError(error, os.strerror(error), str(parent))
                    if parent == parent.parent:
                        raise PermissionError('No readable broker event ancestor')
                    parent = parent.parent
            # Close the check/watch race if the status appeared while watches
            # were being installed, without polling the training process.
            for path in candidates:
                try:
                    value = json.loads(path.read_text())
                except (FileNotFoundError, PermissionError, json.JSONDecodeError):
                    continue
                if value.get('state') in ('rejected', 'ended', 'failed', 'outcome_unknown'):
                    return value
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError('Broker dispatch event missing; inspect the existing job, do not replay')
            if select.select([descriptor], [], [], remaining)[0]:
                os.read(descriptor, 65536)
    finally:
        os.close(descriptor)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', choices=REMOTES, required=True)
    parser.add_argument('--gpu-index', type=int, required=True)
    parser.add_argument('--job', required=True)
    parser.add_argument('--memory-mib', type=int, required=True)
    parser.add_argument('--timeout-seconds', type=int, default=172800)
    parser.add_argument('--submit-only', action='store_true')
    parser.add_argument('--reattach', action='store_true', help='Only wait/collect the existing matching job; never submit')
    args = parser.parse_args()
    source = Path(args.job).resolve()
    job = json.loads(source.read_text())
    remote = REMOTES[args.host]
    request = dict(version=1, job_id=job['job_id'], project_id='4dsr', role='remote', operation='run',
                   remote_host=args.host, remote_cwd=remote,
                   argv=[SHARED+'/.venv/bin/python','-u',remote+'/deployment/static_sr_cts_20261022/run_job.py','--job',remote+'/'+str(source.relative_to(ROOT))],
                   cuda_visible_devices=str(args.gpu_index), gpu_memory_mib=args.memory_mib,
                   timeout_seconds=args.timeout_seconds)
    directory = HERE/'dispatch'/job['job_id']
    request_path = directory/'broker_request.json'
    if args.reattach:
        intent = json.loads((directory/'intent.json').read_text())
        if intent['job_sha256'] != hashlib.sha256(source.read_bytes()).hexdigest() or intent['host'] != args.host:
            raise ValueError('Reattach requires the exact existing job identity and host')
        if json.loads(request_path.read_text()) != request:
            raise ValueError('Reattach arguments differ from the submitted broker request')
        print(json.dumps(dict(job_id=job['job_id'], action='reattach_only_no_submission')), flush=True)
    else:
        directory.mkdir(parents=True, exist_ok=False)
        write(request_path, request)
        write(directory/'intent.json', dict(utc=datetime.now(timezone.utc).isoformat(), job=str(source.relative_to(ROOT)),
              job_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), host=args.host, no_automatic_replay=True))
        result = subprocess.run(['/usr/local/bin/claude-training','submit',str(request_path)], capture_output=True, text=True, timeout=120)
        (directory/'submit.log').write_text(result.stdout+result.stderr)
        if result.returncode:
            raise RuntimeError('Broker submission rejected; inspect submit.log, do not replay automatically')
        print(result.stdout, flush=True)
    transport = wait_transport(job['job_id'])
    write(directory/'transport_terminal.json', transport)
    if transport.get('outcome') not in ('remote_submitted', 'existing_remote_job'):
        raise RuntimeError('Broker dispatch did not confirm remote submission; inspect transport_terminal.json, do not replay')
    if args.submit_only:
        return
    watch = '''import ctypes,json,os,select,time
from pathlib import Path
p=Path(PATH)
limit=time.monotonic()+TIMEOUT
libc=ctypes.CDLL('libc.so.6',use_errno=True)
fd=libc.inotify_init1(os.O_CLOEXEC)
if fd<0:raise OSError(ctypes.get_errno())
watched=None
try:
 while True:
  parent=p.parent
  while not parent.exists():parent=parent.parent
  if watched!=parent:
   if watched is not None:libc.inotify_rm_watch(fd,wd)
   wd=libc.inotify_add_watch(fd,os.fsencode(parent),0x100|0x80|0x8)
   if wd<0:raise OSError(ctypes.get_errno())
   watched=parent
  if p.exists():
   value=json.loads(p.read_text())
   if value.get('state') in ['ended','failed','completed']:
    print(json.dumps(value),flush=True);break
  remaining=limit-time.monotonic()
  if remaining<=0:raise TimeoutError('Remote terminal event wait expired')
  if select.select([fd],[],[],remaining)[0]:os.read(fd,65536)
finally:os.close(fd)
'''.replace('PATH', repr(SHARED+'/.claude-training/jobs/'+job['job_id']+'/status.json')).replace('TIMEOUT',str(args.timeout_seconds+1800))
    # The initial local transport result is separate from the eventual remote
    # terminal state. The event watcher can be reattached after a network loss.
    ssh=['ssh','-o','BatchMode=yes','-o','ServerAliveInterval=30','-o','ServerAliveCountMax=6',args.host]
    terminal = subprocess.run(ssh+['/usr/bin/python3 -'], input=watch, text=True, capture_output=True, timeout=args.timeout_seconds+1900)
    (directory/'remote_terminal.log').write_text(terminal.stdout+terminal.stderr)
    if terminal.returncode:
        raise RuntimeError('Remote event watch failed; collect existing job instead of resubmitting')
    value=json.loads(terminal.stdout.strip().splitlines()[-1])
    write(directory/'remote_terminal.json',value)
    broker_destination=directory/'remote_supervisor'
    broker_destination.mkdir(exist_ok=True)
    subprocess.run(['rsync','-a',args.host+':'+SHARED+'/.claude-training/jobs/'+job['job_id']+'/',
                    str(broker_destination)+'/'],check=True,timeout=120)
    if value.get('outcome') not in ('success', 'completed') and value.get('task_identity') is None:
        write(directory/'startup_failure.json',dict(remote_terminal=value,artifacts_created=False,
                                                   automatic_replay=False))
        raise RuntimeError('GPU job did not start; supervisor evidence collected, no automatic replay')
    collect_outputs(args.host,remote,job.get('return_paths',[job['out']]),value,directory)
    print(json.dumps(dict(job_id=job['job_id'], status=value.get('state'), outcome=value.get('outcome'), artifacts_returned=True)),flush=True)


if __name__ == '__main__':
    main()
