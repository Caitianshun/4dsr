#!/usr/bin/env python3
"""Wait on a remote completion event, then return immutable SR4D evidence.

CPU/network only. Uses Linux inotify plus an optional FIFO, never training polls.
Example:
  python collect_on_exit.py --host a100-train \
    --remote-root /home/ubuntu/3DGS/4dsr --run cook_main_v1 \
    --local-root /home/cai_tianshun/Project/4dsr
Run in a persistent LOCAL service. A broken SSH session is recorded as failure;
the hourly fallback may invoke the same command again. No --delete is used.
"""
import argparse
from datetime import datetime, timezone
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import traceback
import uuid


REMOTE_WAIT = r'''
import ctypes,fcntl,json,os,pathlib,select,stat,sys
root=pathlib.Path(sys.argv[1]).resolve(); name=sys.argv[2]
if not root.is_dir():raise FileNotFoundError('Remote project root does not exist: '+str(root))
parent=root/'output/sr4d_20260922'; out=parent/name
parent.mkdir(parents=True,exist_ok=True)
event=out/'completion_event.json'; fifo=parent/(name+'.completion.fifo')
lock=open(parent/(name+'.collection.lock'),'a')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
libc=ctypes.CDLL(None,use_errno=True)
fd=libc.inotify_init1(os.O_NONBLOCK|os.O_CLOEXEC)
if fd<0:raise OSError(ctypes.get_errno(),'inotify_init1')
watched=set()
def watch(p):
 if p in watched:return
 wd=libc.inotify_add_watch(fd,os.fsencode(p),0x100|0x80|0x8|0x400|0x800)
 if wd<0:raise OSError(ctypes.get_errno(),'inotify_add_watch '+str(p))
 watched.add(p)
watch(parent)
try:os.mkfifo(fifo,0o600)
except FileExistsError:pass
if fifo.is_symlink() or not stat.S_ISFIFO(fifo.stat().st_mode):raise RuntimeError('Unsafe pre-existing FIFO path')
pipe=os.open(fifo,os.O_RDWR|os.O_NONBLOCK|os.O_CLOEXEC)
def finished():
 if out.is_dir():watch(out)
 if not event.is_file():return None
 try:d=json.loads(event.read_text())
 except json.JSONDecodeError:return None
 if d.get('status') not in ('complete','failed'):raise RuntimeError('Nonterminal completion event')
 if pathlib.Path(d['out']).resolve()!=out:raise RuntimeError('Completion out path mismatch')
 return d
# Watch registration precedes the first event read. This handles completion
# before subscription, between registration/read, and after blocking begins.
print(json.dumps({'collector_wait':'ready','event':str(event)}),flush=True)
while True:
 d=finished()
 if d is not None:
  print(json.dumps({'completion_event':d}),flush=True);break
 readable,_,_=select.select([fd,pipe],[],[])
 for f in readable:
  try:os.read(f,65536)
  except BlockingIOError:pass
'''


REMOTE_INVENTORY = r'''
import hashlib,json,os,pathlib,sys
root=pathlib.Path(sys.argv[1]).resolve();name=sys.argv[2]
out=root/'output/sr4d_20260922'/name
event=json.loads((out/'completion_event.json').read_text())
assert event['status'] in ('complete','failed')
state=json.loads((out/'status.json').read_text());scene=state['scene']
assert scene and '/' not in scene and scene not in ('.','..')
manifest_value=state.get('manifest_path')
if manifest_value:
 manifest=pathlib.Path(manifest_value)
 if not manifest.is_absolute():manifest=root/manifest
 manifest=pathlib.Path(os.path.abspath(manifest))
 manifest.relative_to(root)
 data=manifest.parent
 data_policy='explicit_manifest_references_only'
else:
 # Historical author-LR smoke batches did not record manifest_path. Their
 # explicit fallback remains reproducible; current runs set manifest_path and
 # never scan or transfer the unrelated author-LR/SwinIR tree.
 data=root/'data/sr4d_author_20260922'/scene
 manifest=data/'manifest.json'
 data_policy='legacy_author_lr_smoke_tree'
files={};omitted=[];references=[]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()
def add(p,role,expected=None):
 # Preserve the requested logical name (notably checkpoint_final.pt). The
 # target must be a regular file inside this same resolved project root.
 logical=pathlib.Path(os.path.abspath(p));rel=str(logical.relative_to(root))
 target=logical.resolve(strict=True)
 try:target_rel=str(target.relative_to(root))
 except ValueError:raise RuntimeError('External symlink target refused: '+str(logical))
 if not target.is_file():raise RuntimeError('Only regular-file artifacts are supported: '+str(logical))
 h=sha(target)
 if expected and h!=expected:raise RuntimeError('Manifest/source identity mismatch: '+str(p))
 files[rel]={'bytes':target.stat().st_size,'sha256':h,'role':role}
 if target!=logical:files[rel]['dereferenced_target']=target_rel
 return rel
for p in sorted(out.rglob('*')):
 if p.is_symlink() and not p.is_file():raise RuntimeError('Broken or directory symlink refused: '+str(p))
 if not p.is_file():continue
 if '__pycache__' in p.parts:continue
 if p.suffix=='.pt' and p.name.startswith('checkpoint_') and p.name not in ('checkpoint_final.pt','checkpoint_6000.pt','checkpoint_18000.pt'):
  omitted.append(str(p.relative_to(root)));continue
 add(p,'run_output')
if data.exists() and not manifest_value:
 for p in sorted(data.rglob('*')):
  if p.is_symlink() and not p.is_file():raise RuntimeError('Broken or directory symlink refused: '+str(p))
  if p.is_file() and '__pycache__' not in p.parts:add(p,'author_lr_or_prior')
if manifest_value and not manifest.is_file():raise FileNotFoundError(manifest)
if manifest.exists():
 add(manifest,'training_manifest',state.get('manifest_sha256'))
 m=json.loads(manifest.read_text())
 for row in m['observations']:
  for kind in ('hr','lr','lr_float'):
   if kind+'_path' not in row:continue
   p=data/row[kind+'_path']
   references.append(add(p,'existing_'+kind+'_reference',row.get(kind+'_sha256')))
 for key in ('npz_path','ply_path','report_path'):
  if key in m['initialization']:
   expected=m['initialization'].get(key.replace('_path','_sha256'))
   references.append(add(data/m['initialization'][key],'existing_initialization_reference',expected))
 if 'comparison_source_manifest' in m:
  references.append(add(data/m['comparison_source_manifest'],'source_manifest_reference',m.get('comparison_source_manifest_sha256')))
print(json.dumps({'schema':'sr4d_remote_collection_v1','root':str(root),'run':name,'scene':scene,
 'training_status':event['status'],'event':event,'files':files,'omitted_nonkey_checkpoints':omitted,
 'manifest_path':str(manifest),'data_policy':data_policy,'referenced_existing_files':sorted(set(references))}))
'''


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(4 << 20), b''):
            h.update(b)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    os.replace(temp, path)


def remote_command(host, source, root, run):
    command = ' '.join(shlex.quote(x) for x in ['python3', '-u', '-c', source, root, run])
    return ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
            '-o', 'ServerAliveInterval=60', '-o', 'ServerAliveCountMax=3', host, command]


def safe_target(root, rel):
    path = Path(rel)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError('Unsafe relative path: ' + rel)
    target = root / path
    target.resolve().relative_to(root)
    if target.is_symlink():
        raise ValueError('Refusing symlink destination: ' + rel)
    return target


def publish_new(source, target, expected_hash):
    """Atomically add absent files; never replace a pre-existing artifact."""
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name('.' + target.name + '.incoming-' + uuid.uuid4().hex)
    try:
        try:
            os.link(source, temp)
        except OSError as exc:
            if exc.errno != errno.EXDEV:
                raise
            shutil.copy2(source, temp)
        try:
            os.link(temp, target)
        except FileExistsError:
            if not target.is_file() or sha(target) != expected_hash:
                raise RuntimeError('Concurrent destination collision: ' + str(target))
    finally:
        temp.unlink(missing_ok=True)


def self_test():
    """Exercise actual inotify/FIFO before/after-completion races using CPU only."""
    results = []
    for completed_early, use_fifo, terminal_status in [(True, False, 'complete'), (False, False, 'complete'), (False, True, 'complete'), (False, True, 'failed')]:
        with tempfile.TemporaryDirectory(prefix='sr4d_collect_test_') as d:
            root = Path(d)
            run = 'test_run'
            out = root / 'output/sr4d_20260922' / run
            event = {'status': terminal_status, 'out': str(out), 'finished_at': now()}
            if completed_early:
                out.mkdir(parents=True)
                atomic_json(out / 'completion_event.json', event)
            p = subprocess.Popen([sys.executable, '-u', '-c', REMOTE_WAIT, str(root), run],
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            try:
                assert json.loads(p.stdout.readline())['collector_wait'] == 'ready'
                if not completed_early:
                    out.mkdir(parents=True)
                    atomic_json(out / 'completion_event.json', event)
                    if use_fifo:
                        try:
                            fd = os.open(out.parent / (run + '.completion.fifo'), os.O_WRONLY | os.O_NONBLOCK)
                            os.write(fd, (json.dumps(event) + '\n').encode())
                            os.close(fd)
                        except OSError as exc:
                            # Completion JSON can already have woken and ended the reader.
                            assert exc.errno == errno.ENXIO
                # readline() can already buffer the following completion line.
                # communicate() then reads the raw fd and can miss that buffered
                # data; wait/read retains it. Output is tiny, so no pipe fill.
                p.wait(timeout=10)
                stdout, stderr = p.stdout.read(), p.stderr.read()
                assert p.returncode == 0, stderr
                assert json.loads(stdout.strip())['completion_event'] == event
                results.append({'completed_before_subscribe': completed_early, 'fifo': use_fifo, 'terminal_status': terminal_status, 'passed': True})
            finally:
                if p.poll() is None:
                    p.kill(); p.wait()
    with tempfile.TemporaryDirectory(prefix='sr4d_publish_test_') as d:
        root = Path(d); source = root / 'source'; target = root / 'published'
        source.write_bytes(b'original'); h = sha(source)
        publish_new(source, target, h); publish_new(source, target, h)
        different = root / 'different'; different.write_bytes(b'different')
        try:
            publish_new(different, target, sha(different))
        except RuntimeError:
            pass
        else:
            raise AssertionError('Conflicting publication was accepted')
        assert target.read_bytes() == b'original'
    print(json.dumps({'status': 'passed', 'event_cases': results, 'immutable_publication': 'passed'}, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--host', choices=['a100-train', 'cts'])
    ap.add_argument('--remote-root')
    ap.add_argument('--run')
    ap.add_argument('--local-root', type=Path)
    ap.add_argument('--self-test', action='store_true')
    a = ap.parse_args()
    if a.self_test:
        self_test(); return 0
    if not all([a.host, a.remote_root, a.run, a.local_root]):
        ap.error('host, remote-root, run, and local-root are required')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', a.run) or a.run in ('.', '..'):
        ap.error('run must be a safe directory basename')
    if not Path(a.remote_root).is_absolute():
        ap.error('remote-root must be absolute')
    local = a.local_root.expanduser().resolve()
    key = a.host + '__' + a.run
    attempt = now().replace(':', '').replace('+', '_') + '__' + uuid.uuid4().hex[:8]
    receipts = local / 'output/sr4d_20260922/.collection_receipts' / key
    work = receipts / attempt
    work.mkdir(parents=True, exist_ok=False)
    lock = open(receipts / 'collector.lock', 'a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = {'status': 'waiting', 'host': a.host, 'remote_root': a.remote_root, 'run': a.run,
             'local_root': str(local), 'started_at': now(), 'attempt': attempt,
             'mechanism': 'Blocking Linux inotify/FIFO plus completion JSON; no polling'}
    status_file = work / 'status.json'
    atomic_json(status_file, state)
    try:
        # Save expanded route; cts is permitted to use its configured 5090 jump host.
        route = subprocess.check_output(['ssh', '-G', a.host], text=True, stderr=subprocess.DEVNULL)
        (work / 'ssh_route.txt').write_text('\n'.join(line for line in route.splitlines()
            if line.split(' ', 1)[0] in {'hostname', 'user', 'port', 'proxyjump', 'proxycommand'}) + '\n')
        with (work / 'event_wait.stderr.log').open('w') as err:
            waited = subprocess.run(remote_command(a.host, REMOTE_WAIT, a.remote_root, a.run),
                                    stdout=subprocess.PIPE, stderr=err, text=True)
        (work / 'event_wait.stdout.log').write_text(waited.stdout)
        if waited.returncode:
            raise RuntimeError(f'Event SSH exited {waited.returncode}; completion not confirmed')
        events = [json.loads(x) for x in waited.stdout.splitlines() if x.strip()]
        event = events[-1]['completion_event']
        state.update(status='inventory', training_status=event['status'], remote_event=event)
        atomic_json(status_file, state)
        with (work / 'inventory.stderr.log').open('w') as err:
            raw = subprocess.check_output(remote_command(a.host, REMOTE_INVENTORY, a.remote_root, a.run), stderr=err, text=True)
        inventory = json.loads(raw)
        if inventory['event'] != event:
            raise RuntimeError('Completion event changed after waiting')
        atomic_json(work / 'remote_inventory.json', inventory)
        needed, reused = [], []
        for rel, info in inventory['files'].items():
            target = safe_target(local, rel)
            if target.exists():
                if not target.is_file() or sha(target) != info['sha256']:
                    raise RuntimeError('Refusing to overwrite differing local artifact: ' + str(target))
                reused.append(rel)
            else:
                needed.append(rel)
        payload = work / 'payload'; payload.mkdir()
        filelist = work / 'transfer_files.nul'
        filelist.write_bytes(b''.join(x.encode() + b'\0' for x in sorted(needed)))
        state.update(status='transferring', needed_files=len(needed), reused_files=len(reused),
                     needed_bytes=sum(inventory['files'][x]['bytes'] for x in needed))
        atomic_json(status_file, state)
        if needed:
            with (work / 'rsync.log').open('w') as log:
                # The inventory has rejected every target outside remote-root.
                # -L materializes those audited file links under their logical
                # names; only the explicit file list is transferred.
                cmd = ['rsync', '-a', '--copy-links', '--relative', '--from0', '--files-from=' + str(filelist),
                       '--protect-args', '--partial-dir=.rsync-partial',
                       '-e', 'ssh -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=60 -o ServerAliveCountMax=3',
                       a.host + ':' + a.remote_root.rstrip('/') + '/', str(payload) + '/']
                p = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
            if p.returncode:
                raise RuntimeError(f'rsync exited {p.returncode}; payload retained, not published')
        for rel in needed:
            source = safe_target(payload, rel)
            info = inventory['files'][rel]
            if not source.is_file() or source.stat().st_size != info['bytes'] or sha(source) != info['sha256']:
                raise RuntimeError('Transferred payload hash mismatch: ' + rel)
        # Check every incoming file before publishing any. Existing targets are
        # never replaced, even when another collector races with this attempt.
        for rel in needed:
            publish_new(payload / rel, safe_target(local, rel), inventory['files'][rel]['sha256'])
        for rel, info in inventory['files'].items():
            if sha(safe_target(local, rel)) != info['sha256']:
                raise RuntimeError('Final local hash mismatch: ' + rel)
        receipt = {'status': 'verified', 'training_status': inventory['training_status'],
                   'host': a.host, 'run': a.run, 'scene': inventory['scene'], 'finished_at': now(),
                   'new_files': len(needed), 'reused_files': len(reused),
                   'total_verified_files': len(inventory['files']),
                   'inventory_sha256': sha(work / 'remote_inventory.json'),
                   'source': a.remote_root, 'destination': str(local),
                   'manifest_references_verified': len(inventory['referenced_existing_files']),
                   'manifest_path': inventory['manifest_path'], 'data_policy': inventory['data_policy'],
                   'materialized_internal_file_links': {rel: info['dereferenced_target']
                       for rel, info in inventory['files'].items() if 'dereferenced_target' in info},
                   'omitted_nonkey_checkpoints': inventory['omitted_nonkey_checkpoints']}
        atomic_json(work / 'receipt.json', receipt)
        state.update(status='complete' if inventory['training_status'] == 'complete' else 'collected_failed_run',
                     finished_at=now(), receipt=str(work / 'receipt.json'))
        atomic_json(status_file, state)
        # Files have been atomically linked/copied to their final paths.
        shutil.rmtree(payload)
        print(json.dumps(state, ensure_ascii=False, indent=2), flush=True)
        return 0 if inventory['training_status'] == 'complete' else 2
    except BaseException:
        state.update(status='failed', finished_at=now(), error=traceback.format_exc(),
                     note='No successful sync is claimed. Preserve this attempt; hourly fallback may retry.')
        atomic_json(status_file, state)
        print(json.dumps(state, ensure_ascii=False, indent=2), flush=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
