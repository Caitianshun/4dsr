"""Additive SHA256-verified deployment into the task's isolated CTS snapshot.

Run by the directly authorized project controller. Does not change the broker,
its policy, GPU locks, existing files, source manifests or other tasks.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
REMOTE = '/home/cts/Project/4DSR/tasks/static_sr_wu_20261022'
SHARED = '/home/cts/Project/4DSR'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


REMOTE_CODE = r'''
import fcntl,hashlib,json,os,shutil,sys,time
from pathlib import Path
request=json.load(sys.stdin);root=Path(request['root']);shared=Path(request['shared'])
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def safe(base,rel):
 p=Path(rel)
 if p.is_absolute() or '..' in p.parts:raise ValueError('Unsafe member')
 q=base/p
 for a in [q,*q.parents]:
  if a==base:break
  if a.is_symlink():raise ValueError('Symlink member: '+str(a))
 return q
root.mkdir(parents=True,exist_ok=True)
with (root/'.deployment.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 missing=[];reused=[];shared_matches=[];conflicts=[];missing_dirs=[];created_dirs=[]
 for entry in request.get('empty_dirs',[]):
  p=safe(root,entry['path'])
  if p.exists():
   if not p.is_dir() or any(p.iterdir()):conflicts.append(entry['path'])
  else:missing_dirs.append(entry['path'])
 for e in request['files']:
  p=safe(root,e['path'])
  if p.exists():
   if p.is_file() and p.stat().st_size==e['bytes'] and sha(p)==e['sha256']:reused.append(e['path'])
   else:conflicts.append(e['path'])
  else:
   s=safe(shared,e['path'])
   immutable=e['path'].startswith(('data/','weight/'))
   if immutable and s.is_file() and s.stat().st_size==e['bytes'] and sha(s)==e['sha256']:shared_matches.append(e['path'])
   else:missing.append(e['path'])
 if conflicts:raise ValueError('Existing task snapshot conflicts; preserved: '+repr(conflicts[:20]))
 needed=sum(e['bytes'] for e in request['files'] if e['path'] in set(missing))
 free=shutil.disk_usage(root).free
 if free<needed+request['reserve_bytes']:raise ValueError('Disk reserve insufficient')
 if request['phase']=='prepare':
  for rel in missing_dirs:
   p=safe(root,rel);p.mkdir(parents=True,exist_ok=False);created_dirs.append(rel)
  for rel in shared_matches:
   p=safe(root,rel);p.parent.mkdir(parents=True,exist_ok=True);os.link(safe(shared,rel),p)
 elif request['phase']=='verify':
  if missing or shared_matches or missing_dirs:raise ValueError('Incomplete deployment')
 else:raise ValueError('Unknown phase')
 print(json.dumps(dict(status='prepared' if request['phase']=='prepare' else 'verified',missing=missing,
   reused=reused,hardlinked_from_verified_shared=shared_matches,bytes_to_transfer=needed,free_bytes=free,
   reserve_bytes=request['reserve_bytes'],files=len(request['files']),empty_gitlinks=request.get('empty_dirs',[]),
   created_empty_dirs=created_dirs,timestamp=time.time())))
'''


def remote(request):
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20', 'cts',
                             shlex.join(['/usr/bin/python3', '-c', REMOTE_CODE])],
                            input=json.dumps(request), text=True, capture_output=True, timeout=1800)
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    return json.loads(result.stdout)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inventory', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--reserve-gib', type=int, default=60)
    a = p.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=False)
    source = Path(a.inventory)
    inv = json.loads(source.read_text())
    if inv.get('symlinks'):
        raise ValueError('Symlink inventories require separate reviewed materialization; sync_assets.py supports regular files and empty Git-link directories only')
    entries = inv['files']
    if isinstance(entries, dict):
        entries = [dict(v, path=k) for k,v in entries.items()]
    normalized = []
    seen = set()
    for e in entries:
        rel = e['path']; local = ROOT / rel
        if Path(rel).is_absolute() or '..' in Path(rel).parts or rel in seen:
            raise ValueError('Invalid/duplicate inventory path')
        seen.add(rel)
        if not local.is_file() or local.is_symlink():
            raise ValueError('Missing/nonregular source: ' + rel)
        size = e.get('bytes', e.get('size_bytes'))
        if size is None: size = local.stat().st_size
        if local.stat().st_size != size or sha(local) != e['sha256']:
            raise ValueError('Local frozen identity changed: ' + rel)
        normalized.append(dict(path=rel, bytes=size, sha256=e['sha256']))
    empty_dirs = []
    for entry in inv.get('empty_dirs', []):
        rel=entry['path']; local=ROOT/rel
        if (Path(rel).is_absolute() or '..' in Path(rel).parts or rel in seen
                or not re.fullmatch(r'[0-9a-f]{40}|[0-9a-f]{64}',entry.get('gitlink_commit',''))):
            raise ValueError('Invalid/duplicate empty Git-link path or commit')
        if local.is_symlink() or not local.is_dir() or any(local.iterdir()):
            raise ValueError('Source Git-link directory is no longer empty: '+rel)
        if any(Path(f['path']).is_relative_to(rel) for f in normalized):
            raise ValueError('Empty Git-link also contains file entries: '+rel)
        seen.add(rel);empty_dirs.append(dict(path=rel,gitlink_commit=entry['gitlink_commit']))
    req = dict(root=REMOTE, shared=SHARED, files=normalized, empty_dirs=empty_dirs,
               reserve_bytes=a.reserve_gib << 30, phase='prepare')
    write(out/'intent.json', dict(inventory=str(source), inventory_sha256=sha(source), root=REMOTE,
                                 files=len(normalized),empty_gitlinks=empty_dirs,created_unix=time.time(), GPU_calls=0))
    prep = remote(req);write(out/'preflight.json', prep)
    print(json.dumps({k:v for k,v in prep.items() if not isinstance(v,list)}), flush=True)
    filelist = out/'missing.txt';filelist.write_text(''.join(rel+'\n' for rel in prep['missing']))
    if prep['missing']:
        # Never replace a file that appeared after preflight; final SHA rejects it.
        argv=['rsync','-rltH','--ignore-existing','--partial-dir=.rsync-partial',
              '--files-from='+str(filelist.resolve()),'-e','ssh -o BatchMode=yes -o ConnectTimeout=20',
              str(ROOT)+'/', 'cts:'+REMOTE+'/']
        with (out/'rsync.log').open('x') as log:
            result=subprocess.run(argv,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError('Transfer incomplete; preserve receipt and inspect rsync.log')
    req['phase']='verify'
    done=remote(req);write(out/'verified.json',dict(done,inventory_sha256=sha(source)))
    print(json.dumps({k:v for k,v in done.items() if not isinstance(v,list)}),flush=True)


if __name__=='__main__':
    main()
