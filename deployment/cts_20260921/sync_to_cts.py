#!/usr/bin/env python3
"""Incrementally copy the generic dynamic-SR allowlist to cts, never delete.

The default is a remote read-only rsync dry run. --apply performs the transfer,
keeps overwritten destination files in a per-run backup, and verifies SHA256
for every selected file remotely. No environment installation or GPU work runs.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import fnmatch
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import selectors
import shlex
import signal
import subprocess
import tempfile
import time
import uuid


ROOT = Path(__file__).resolve().parents[2]
DEPLOY = ROOT / 'deployment/cts_20260921'
UPSTREAM = Path('/home/cai_tianshun/Project/4dgs')
SWINIR = Path('/home/cai_tianshun/Project/mml/scripts/network_swinir.py')
SWINIR_WEIGHT = Path('/home/cai_tianshun/Project/mml/outputs/week3_swinir/ckpt/001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth')
REMOTE = '/home/cts/Project/4DSR'
# User policy (2026-09-21): 5090 participates in neither computation nor sync.
# The existing cts alias used that host as ProxyJump. Fail closed on direct
# connectivity instead of silently using the inherited jump/ProxyCommand.
SSH_TRANSPORT = ['ssh', '-o', 'ProxyJump=none', '-o', 'ProxyCommand=none',
                 '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20']
SSH = SSH_TRANSPORT + ['cts']
LOCK_RELATIVE = 'deployment/cts_20260921/project_access.lock'
PENDING_RELATIVE = 'deployment/cts_20260921/sync_pending.json'
EXCLUDE_PARTS = {'.git', '.venv', '__pycache__', 'build', 'dist', '.cache',
                 '.pytest_cache', '.mypy_cache', '.rsync-partial', 'hodome'}
EXCLUDE_FILES = ('*.pyc', '*.pyo', '*.so', '*.so.*', '*.o', '*.a', '*.egg-info',
                 '*.tmp', '*.part', 'hodome_loader.py')
PINNED = {
    'vendor/swinir/network_swinir.py': '1d650d2c1c4519d95db771863691c6b239a0822222f287f6a8342a9e0eccbc6e',
    'weights/001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth': '4e78e33f22c1aa8a773db0cf4a7381bae97c2362c717f155439ebc690cbd9215',
}


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            value.update(block)
    return value.hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def json_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def excluded(relative):
    return (any(p in EXCLUDE_PARTS or p.endswith('.egg-info') for p in relative.parts)
            or any(fnmatch.fnmatch(relative.name, pattern) for pattern in EXCLUDE_FILES))


def inventory():
    """Return explicit mappings, without copying environments or historical outputs."""
    entries = []

    def add(source, destination):
        source = Path(source)
        if source.is_symlink() or not source.is_file():
            raise ValueError(f'Source must be a regular, non-symlink file: {source}')
        relative = PurePosixPath(destination)
        if relative.is_absolute() or '..' in relative.parts or '\n' in str(relative):
            raise ValueError(f'Unsafe destination: {destination}')
        entries.append({'source': str(source.absolute()), 'path': str(relative)})

    def tree(source, destination):
        source = Path(source)
        if not source.is_dir() or source.is_symlink():
            raise ValueError(f'Missing or symlink source directory: {source}')
        for base, dirs, files in os.walk(source, followlinks=False):
            base = Path(base)
            dirs[:] = sorted(d for d in dirs if not excluded((base / d).relative_to(source)))
            for d in dirs:
                if (base / d).is_symlink():
                    raise ValueError(f'Unexpected source symlink: {base / d}')
            for name in sorted(files):
                path = base / name
                relative = path.relative_to(source)
                if not excluded(relative):
                    add(path, str(PurePosixPath(destination) / relative.as_posix()))

    tree(ROOT / 'data/dynamic_sr', 'data/dynamic_sr')
    for directory in sorted((ROOT / 'experiments').iterdir()):
        if re.fullmatch(r'dynamic_sr_[0-9]{8}', directory.name):
            tree(directory, directory.relative_to(ROOT).as_posix())
    for relative in ['AGENTS.md', 'docs/cts_deployment_2026-09-21.md', 'docs/remote_resources_2026-09-21.md']:
        if (ROOT / relative).exists():
            add(ROOT / relative, relative)
    # Deployment logs, acceptance outputs, and prior sync records stay host-local.
    for path in sorted(DEPLOY.iterdir()):
        if path.is_file() and path.suffix in {'.py', '.sh', '.md', '.txt', '.toml'}:
            add(path, path.relative_to(ROOT).as_posix())
    # Explicit CPython optional-module overlay; never a separate project env.
    additions = DEPLOY / 'runtime_additions'
    for path in sorted(additions.rglob('*')):
        if path.is_file() and '__pycache__' not in path.parts:
            add(path, path.relative_to(ROOT).as_posix())
    for name in ['run_control.py', 'requirements.txt', 'bootstrap_inventory.json',
                 'weights_inventory.json', 'local_parent_reference.json']:
        relative = f'deployment/a100_20260920/{name}'
        add(ROOT / relative, relative)
    for name in ['arguments', 'gaussian_renderer', 'scene', 'utils', 'submodules', 'lpipsPyTorch']:
        tree(UPSTREAM / name, f'vendor/4dgs/{name}')
    for name in ['train.py', 'render.py', 'metrics.py', 'LICENSE.md', 'README.md', 'requirements.txt']:
        add(UPSTREAM / name, f'vendor/4dgs/{name}')
    add(SWINIR, 'vendor/swinir/network_swinir.py')
    add(SWINIR_WEIGHT, 'weights/' + SWINIR_WEIGHT.name)
    for item in json.loads((ROOT / 'deployment/a100_20260920/bootstrap_inventory.json').read_text()):
        add(item['source'], item['path'])
        PINNED[item['path']] = item['sha256']
    paths = [entry['path'] for entry in entries]
    if len(paths) != len(set(paths)):
        raise ValueError('Duplicate destination in allowlist')
    return sorted(entries, key=lambda entry: entry['path'])


def hash_inventory(entries):
    expected = {}
    frozen = ROOT / 'deployment/a100_20260920/data_required.sha256'
    for line in frozen.read_text().splitlines():
        digest, name = line.split('  ', 1)
        expected[name] = digest
    expected.update(PINNED)
    for entry in entries:
        source = Path(entry['source'])
        before = source.stat()
        entry['sha256'] = sha(source)
        after = source.stat()
        signature = lambda st: (st.st_size, st.st_mtime_ns, st.st_ctime_ns)
        if signature(before) != signature(after):
            raise RuntimeError(f'Source changed during hashing; retry when stable: {source}')
        entry['bytes'] = after.st_size
        entry['mtime_ns'] = after.st_mtime_ns
        entry['ctime_ns'] = after.st_ctime_ns
        if entry['path'] in expected and entry['sha256'] != expected[entry['path']]:
            raise ValueError(f'Frozen source hash mismatch: {entry["path"]}')
    missing = set(expected) - {entry['path'] for entry in entries}
    if missing:
        raise ValueError(f'Frozen source files missing from allowlist: {sorted(missing)[:10]}')


REMOTE_COMMON = r'''
import fcntl,hashlib,json,os,stat,sys,time
from pathlib import Path,PurePosixPath

def safe_path(root,relative,kind=None):
 relative=PurePosixPath(relative)
 if relative.is_absolute() or '..' in relative.parts:raise RuntimeError('Unsafe relative path: '+str(relative))
 p=root/relative
 if not p.resolve().is_relative_to(root.resolve()):raise RuntimeError('Destination escapes project root: '+str(p))
 for part in [p,*p.parents]:
  if part.is_symlink():raise RuntimeError('Refusing destination symlink: '+str(part))
  if part==root:break
  if part!=p and part.exists() and not part.is_dir():raise RuntimeError('Destination parent is not a directory: '+str(part))
 if p.exists() and kind=='file' and not p.is_file():raise RuntimeError('Destination is not a regular file: '+str(p))
 if p.exists() and kind=='directory' and not p.is_dir():raise RuntimeError('Destination is not a directory: '+str(p))
 return p

def check_tree(root,path):
 if not path.exists():return
 for base,dirs,files in os.walk(path,followlinks=False):
  for name in dirs+files:safe_path(root,(Path(base)/name).relative_to(root).as_posix())

def legacy_users(root):
 busy=[];prefix=str(root);venv=prefix+'/.venv'
 for proc in Path('/proc').iterdir():
  if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
  try:cmd=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
  except OSError:cmd=''
  try:exe=os.readlink(proc/'exe')
  except OSError:exe=''
  try:cwd=os.readlink(proc/'cwd')
  except OSError:cwd=''
  # Only the actual rsync executable is exempt, never a shell merely mentioning it.
  if Path(exe).name=='rsync':continue
  reasons=[]
  if cwd==prefix or cwd.startswith(prefix+'/'):reasons.append('cwd inside project')
  if venv in cmd or exe==venv or exe.startswith(venv+'/'):reasons.append('uses project environment')
  if reasons:busy.append({'pid':int(proc.name),'executable':exe,'cwd':cwd,'reasons':reasons})
 return busy

def validate_root(root):
 if not root.is_dir() or root.is_symlink():raise RuntimeError('Remote project root must already exist and not be a symlink')
'''

REMOTE_LOCK = REMOTE_COMMON + r'''
value=json.loads(sys.stdin.readline());root=Path(value['root']);validate_root(root)
lock=safe_path(root,value['lock_relative'],'file')
fd=os.open(lock,os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)
try:
 if not stat.S_ISREG(os.fstat(fd).st_mode):raise RuntimeError('Lock is not a regular file')
 try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:raise RuntimeError('Project is locked by training/evaluation or another sync; no files copied')
 busy=legacy_users(root)
 if busy:raise RuntimeError('Unwrapped project processes are active; refusing sync: '+json.dumps(busy))
 print(json.dumps({'status':'locked','pid':os.getpid(),'path':str(lock),'run_id':value['run_id']}),flush=True)
 # The caller holds this stdin pipe open through rsync and all verification.
 # EOF, SSH disconnect, exceptions, or process exit release the OS flock.
 sys.stdin.read()
finally:
 os.close(fd)
'''

REMOTE_CHECK = REMOTE_COMMON + r'''
value=json.load(sys.stdin);root=Path(value['root']);action=value['action'];validate_root(root)
pending=safe_path(root,'deployment/cts_20260921/sync_pending.json','file')
parents={root}
for row in value['entries']:
 p=safe_path(root,row['path'],'file')
 for part in p.parents:
  if part==root:break
  parents.add(part)
# Check both write destinations even during dry-run without creating them.
for key in ['record','backup']:
 if value.get(key):
  target=safe_path(root,value[key],'directory');check_tree(root,target)
for parent in parents:
 partial=safe_path(root,(parent/'.rsync-partial').relative_to(root).as_posix(),'directory')
 check_tree(root,partial)
if value.get('check_legacy'):
 busy=legacy_users(root)
 # The lock helper has no project cwd or project .venv in its command; it is
 # also explicitly excluded by PID to preserve that invariant after changes.
 busy=[row for row in busy if row['pid']!=value.get('lock_pid')]
 if busy:raise RuntimeError('Unwrapped project processes appeared; refusing sync: '+json.dumps(busy))
if action=='preflight':
 print(json.dumps({'status':'passed','checked_paths':len(value['entries'])}));sys.exit(0)
def manifest_hash(manifest):
 return hashlib.sha256(json.dumps(manifest,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
if action=='begin':
 if not value.get('lock_pid'):raise RuntimeError('Pending marker requires the exclusive sync lock')
 marker={'schema':'4dsr.cts.sync-pending.v1','run_id':value['manifest']['run_id'],
         'manifest_sha256':manifest_hash(value['manifest']),'started_unix':time.time(),
         'status':'pending_full_sha256_verification'}
 if pending.exists():
  previous=pending.read_bytes();marker['replaces_pending_sha256']=hashlib.sha256(previous).hexdigest()
  try:marker['previous_run_id']=json.loads(previous).get('run_id')
  except (ValueError,AttributeError):marker['previous_run_id']=None
 temporary=safe_path(root,'deployment/cts_20260921/.sync_pending.'+marker['run_id']+'.tmp','file')
 fd=os.open(temporary,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
 with os.fdopen(fd,'w') as f:
  json.dump(marker,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(temporary,pending)
 directory_fd=os.open(pending.parent,os.O_RDONLY|os.O_DIRECTORY)
 try:os.fsync(directory_fd)
 finally:os.close(directory_fd)
 print(json.dumps(marker));sys.exit(0)
if action!='verify':raise RuntimeError('Unsupported remote action: '+action)
failures=[];checked=0;total=0
for row in value['entries']:
 p=root/row['path'];actual=None
 if p.is_file():
  h=hashlib.sha256()
  with p.open('rb') as f:
   for b in iter(lambda:f.read(8<<20),b''):h.update(b)
  actual=h.hexdigest()
 if actual!=row['sha256']:failures.append({'path':row['path'],'expected':row['sha256'],'actual':actual})
 else:checked+=1;total+=p.stat().st_size
result={'status':'passed' if not failures else 'failed','checked_files':checked,'checked_bytes':total,'failures':failures,'finished_unix':time.time(),'scope':'File transfer integrity only; environment and GPU acceptance are separate.'}
record=safe_path(root,value['record'],'directory');record.mkdir(parents=True,exist_ok=False)
(record/'manifest.json').write_text(json.dumps(value['manifest'],indent=2)+'\n')
(record/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
if not failures:
 marker=json.loads(pending.read_text())
 if marker.get('run_id')!=value['manifest']['run_id'] or marker.get('manifest_sha256')!=manifest_hash(value['manifest']):
  raise RuntimeError('Pending marker identity changed; refusing to clear it')
 pending.unlink()
 directory_fd=os.open(pending.parent,os.O_RDONLY|os.O_DIRECTORY)
 try:os.fsync(directory_fd)
 finally:os.close(directory_fd)
 result['pending_marker_cleared']=True
print(json.dumps(result));sys.exit(0 if not failures else 2)
'''


def stop_owned(child):
    if child.poll() is not None:
        return
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        child.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL)
        child.wait()


def run_owned(command, *, input=None, stdout=None, stderr=None, text=False):
    """Never release the project lock while this tool's transfer child survives."""
    child = subprocess.Popen(command, stdin=subprocess.PIPE if input is not None else subprocess.DEVNULL,
                             stdout=stdout, stderr=stderr, text=text, start_new_session=True)
    try:
        out, err = child.communicate(input)
        return subprocess.CompletedProcess(command, child.returncode, out, err)
    except BaseException:
        stop_owned(child)
        raise


class RemoteWriteLock:
    def __init__(self, run_id):
        self.run_id = run_id
        self.child = None
        self.receipt = None

    def acquire(self):
        self.child = subprocess.Popen(SSH + ['python3 -u -c ' + shlex.quote(REMOTE_LOCK)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1, start_new_session=True)
        try:
            self.child.stdin.write(json.dumps(dict(root=REMOTE, lock_relative=LOCK_RELATIVE, run_id=self.run_id)) + '\n')
            self.child.stdin.flush()
            with selectors.DefaultSelector() as selector:
                selector.register(self.child.stdout, selectors.EVENT_READ)
                if not selector.select(timeout=45):
                    raise RuntimeError('Timed out acquiring remote project lock')
            line = self.child.stdout.readline()
            if not line:
                raise RuntimeError('Remote project lock refused: ' + self.child.stderr.read())
            self.receipt = json.loads(line)
            if self.receipt.get('status') != 'locked':
                raise RuntimeError(f'Invalid remote lock response: {self.receipt}')
            self.ensure_alive()
            return self.receipt
        except BaseException:
            self.close()
            raise

    def ensure_alive(self):
        if self.child is None or self.child.poll() is not None:
            raise RuntimeError('Remote project lock process exited; refusing further sync work')

    def close(self):
        if self.child is None:
            return
        if self.child.stdin:
            try:
                self.child.stdin.close()
            except BrokenPipeError:
                pass
        try:
            self.child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            stop_owned(self.child)
        for stream in [self.child.stdout, self.child.stderr]:
            if stream:
                stream.close()
        self.child = None


def remote_check(action, entries, record=None, manifest=None, *, backup=None, lock=None):
    if action in {'begin', 'verify'} and lock is None:
        raise RuntimeError('Remote mutation requires an exclusive sync lock')
    if lock:
        lock.ensure_alive()
    payload = dict(action=action, root=REMOTE, entries=entries, record=record, manifest=manifest,
                   backup=backup, check_legacy=bool(lock) and action in {'preflight', 'begin'},
                   lock_pid=lock.receipt['pid'] if lock else None)
    result = run_owned(SSH + ['python3 -c ' + shlex.quote(REMOTE_CHECK)],
                       input=json.dumps(payload), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode:
        raise RuntimeError(f'Remote {action} failed ({result.returncode}): {result.stdout}\n{result.stderr}')
    return json.loads(result.stdout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--apply', action='store_true', help='Copy then verify every file remotely')
    mode.add_argument('--dry-run', action='store_true', help='Default: read-only remote transfer preview')
    parser.add_argument('--inventory-only', action='store_true', help='Local hashing/allowlist check; no SSH')
    args = parser.parse_args()
    if args.inventory_only and args.apply:
        parser.error('--inventory-only cannot be combined with --apply')
    begin = time.monotonic()
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:8]
    local_record = DEPLOY / 'sync_records' / run_id
    local_record.mkdir(parents=True, exist_ok=False)
    state = dict(status='starting', started_at=now(), run_id=run_id, host='cts',
                 remote_root=REMOTE, apply=args.apply, inventory_only=args.inventory_only)
    state_file = local_record / 'status.json'
    json_write(state_file, state)
    write_lock = None
    previous_sigterm = signal.getsignal(signal.SIGTERM)

    def interrupted(signum, _frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, interrupted)
    try:
        if args.apply:
            write_lock = RemoteWriteLock(run_id)
            state['write_lock'] = write_lock.acquire()
            json_write(state_file, state)
        entries = inventory()
        print(f'Hashing {len(entries)} allowlisted files...', flush=True)
        hash_inventory(entries)
        manifest = dict(schema='4dsr.cts.sync.v1', created_at=now(), run_id=run_id,
                        source_root=str(ROOT), remote_root=REMOTE, files=entries,
                        rule='Generic dynamic-SR only; no HOI data/weights/environment; no delete; backup overwritten files.')
        json_write(local_record / 'manifest.json', manifest)
        (local_record / 'files.sha256').write_text(''.join(f'{r["sha256"]}  {r["path"]}\n' for r in entries))
        state.update(files=len(entries), bytes=sum(r['bytes'] for r in entries), status='inventoried')
        json_write(state_file, state)
        print(json.dumps({k: state[k] for k in ['files', 'bytes', 'apply']}) + '\nRecord: ' + str(local_record), flush=True)
        if args.inventory_only:
            state.update(status='inventory_passed', elapsed_seconds=time.monotonic()-begin)
            json_write(state_file, state)
            return
        record_relative = f'deployment/cts_20260921/sync_records/{run_id}'
        backup_relative_root = f'deployment/cts_20260921/sync_backups/{run_id}'
        state['preflight'] = remote_check('preflight', entries, record_relative,
                                        backup=backup_relative_root, lock=write_lock)
        if write_lock:
            state['pending_marker'] = remote_check('begin', entries, record_relative, manifest,
                                                   backup=backup_relative_root, lock=write_lock)
            json_write(state_file, state)
        backup = f'{REMOTE}/{backup_relative_root}'
        # Each source base is paired with its destination base. files-from
        # transmits only allowlisted regular files, including newly added data.
        groups = defaultdict(list)
        for entry in entries:
            source = Path(entry['source'])
            destination = PurePosixPath(entry['path'])
            if source.is_relative_to(ROOT) and source.relative_to(ROOT).as_posix() == entry['path']:
                base, remote_base, name = ROOT, PurePosixPath('.'), source.relative_to(ROOT).as_posix()
            elif source.is_relative_to(UPSTREAM):
                base, remote_base, name = UPSTREAM, PurePosixPath('vendor/4dgs'), source.relative_to(UPSTREAM).as_posix()
            else:
                base, remote_base, name = source.parent, destination.parent, source.name
                # Renamed bootstrap files need a separate exact-file transfer.
                if name != destination.name:
                    groups[(str(source), str(destination), True)].append(name)
                    continue
            groups[(str(base), str(remote_base), False)].append(name)
        commands = []
        with tempfile.TemporaryDirectory(prefix='4dsr_cts_sync_') as temp:
            for index, ((base, remote_base, exact), names) in enumerate(groups.items()):
                backup_relative = str(PurePosixPath(remote_base).parent) if exact else remote_base
                # rsync 3.2.7 fails creating a new backup directory ending in
                # '/.' (code 11). Normalize the project-root group as well.
                group_backup = str(PurePosixPath(backup) / backup_relative)
                command = ['rsync', '-a', '--checksum', '--protect-args', '--itemize-changes', '--stats',
                           '--backup', '--backup-dir=' + group_backup, '--partial-dir=.rsync-partial',
                           '-e', shlex.join(SSH_TRANSPORT)]
                if not args.apply:
                    command.append('--dry-run')
                if exact:
                    # rsync --mkpath is available on the target and creates only
                    # parent directories; the dry run never modifies them.
                    command += ['--mkpath', base, f'cts:{REMOTE}/{remote_base}']
                else:
                    paths_file = Path(temp) / f'files_{index}.txt'
                    paths_file.write_bytes(b''.join(os.fsencode(name) + b'\0' for name in names))
                    command += ['--mkpath', '--from0', '--files-from=' + str(paths_file),
                                base.rstrip('/') + '/', f'cts:{REMOTE}/{remote_base}/']
                commands.append(command)
                print(f'{"APPLY" if args.apply else "DRY-RUN"} group {index+1}/{len(groups)}: {base} -> {remote_base}', flush=True)
                if write_lock:
                    # Legacy jobs do not participate in flock. Recheck them and
                    # all destination paths immediately before each transfer.
                    remote_check('preflight', entries, record_relative,
                                 backup=backup_relative_root, lock=write_lock)
                with (local_record / f'rsync_{index:02d}.log').open('w') as log:
                    result = run_owned(command, stdout=log, stderr=subprocess.STDOUT)
                if write_lock:
                    write_lock.ensure_alive()
                if result.returncode:
                    raise RuntimeError(f'rsync exited {result.returncode}; inspect {local_record / f"rsync_{index:02d}.log"}')
        json_write(local_record / 'commands.json', commands)
        if args.apply:
            state['verification'] = remote_check('verify', entries,
                record_relative, manifest, backup=backup_relative_root, lock=write_lock)
            state['backup_directory'] = backup
            state['status'] = 'complete'
        else:
            state['status'] = 'dry_run_complete'
        state.update(finished_at=now(), elapsed_seconds=time.monotonic()-begin)
        json_write(state_file, state)
        print(json.dumps(state, indent=2), flush=True)
    except BaseException as error:
        state.update(status='failed', error=repr(error), finished_at=now(), elapsed_seconds=time.monotonic()-begin)
        json_write(state_file, state)
        raise
    finally:
        if write_lock:
            write_lock.close()
        signal.signal(signal.SIGTERM, previous_sigterm)


if __name__ == '__main__':
    main()
