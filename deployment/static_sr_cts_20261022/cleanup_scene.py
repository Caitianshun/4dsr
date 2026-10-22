"""Closed-scope CPU cleanup after verified local return of all four results.

On the local controller, `verify-return` hashes the returned compact files and
successful job records. Transfer that NEW receipt to the execution snapshot.
On CTS, `cleanup` is a dry run by default. It freezes every candidate's device,
inode, size, nanosecond timestamps and SHA256. A subsequent `cleanup --apply
--freeze ...` revalidates all gates, takes the existing project/GPU locks, and
moves exact candidates to a private same-filesystem quarantine before deleting.
No network, CUDA, job submission, raw-data or generic-weight deletion exists.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import ctypes
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import socket
import stat
import sys
import uuid

ROOT=Path(__file__).resolve().parents[2]
SCENES={'Discussion07','Cook07','Cut07','Coffee07','Flame07','VRHeadset07'}
METHODS=('splatsure','iesrgs','sequence_matters','threedsr')
RUN_PREFIX=Path('output/static_sr_wu_20261022/cts_scene_runs')
NOFOLLOW=os.O_NOFOLLOW
DIR_FLAGS=os.O_RDONLY|os.O_DIRECTORY|NOFOLLOW


def now():return datetime.now(timezone.utc).isoformat()


def machine_id():
    return hashlib.sha256(Path('/etc/machine-id').read_bytes().strip()).hexdigest()


def digest(data):return hashlib.sha256(data).hexdigest()


def open_no_links(path,flags,mode=0o600):
    """Pin each parent with openat(O_DIRECTORY|O_NOFOLLOW), not a check/use pair."""
    path=Path(path).absolute()
    if '..' in path.parts:raise ValueError('Path traversal')
    parent=os.open(path.anchor,DIR_FLAGS)
    try:
        for component in path.parts[1:-1]:
            child=os.open(component,DIR_FLAGS,dir_fd=parent);os.close(parent);parent=child
        return os.open(path.name,flags|NOFOLLOW,mode,dir_fd=parent)
    finally:os.close(parent)


def sha(path):
    fd=open_no_links(path,os.O_RDONLY)
    try:return hash_fd(fd)
    finally:os.close(fd)


def hash_fd(fd):
    h=hashlib.sha256();os.lseek(fd,0,os.SEEK_SET)
    while True:
        chunk=os.read(fd,8<<20)
        if not chunk:break
        h.update(chunk)
    return h.hexdigest()


def relpath(value):
    raw=str(value);path=PurePosixPath(raw)
    if path.is_absolute() or '..' in path.parts or not path.parts or raw!=str(path):
        raise ValueError('Require a canonical relative path without traversal: '+raw)
    return Path(raw)


def no_links(path,base):
    """Check every existing component, including base; never resolve symlinks."""
    path=Path(path).absolute();base=Path(base).absolute()
    if '..' in Path(path).parts or not path.is_relative_to(base):raise ValueError('Path escapes closed scope: '+str(path))
    if base.is_symlink() or base.resolve()!=base:raise ValueError('Symlink in scope root')
    current=base
    for part in path.relative_to(base).parts:
        current=current/part
        try:mode=current.lstat().st_mode
        except FileNotFoundError:continue
        if stat.S_ISLNK(mode):raise ValueError('Symlink path refused: '+str(current))
    return path


def scoped(value):
    p=Path(value)
    if p.is_absolute():
        if '..' in p.parts:raise ValueError('Path traversal')
    else:p=ROOT/relpath(p)
    return no_links(p,ROOT)


def load(path):
    with Path(path).open() as stream:return json.load(stream)


def identity(path,base= None):
    path=Path(path);base=ROOT if base is None else Path(base)
    return dict(path=str(path.relative_to(base)) if path.is_relative_to(base) else str(path),bytes=path.stat().st_size,sha256=sha(path))


def write_new(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd=open_no_links(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL)
    with os.fdopen(fd,'w') as stream:
        stream.write(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())


def snapshot_scope(plan,root):
    root=Path(root).absolute()
    if root.name!='static_sr_wu_20261022' or root.parent.name!='tasks' or root.parent.parent.name!='4DSR':
        raise ValueError('Cleanup is restricted to the task snapshot under 4DSR/tasks/static_sr_wu_20261022')
    if plan.get('root')!=str(root) or plan.get('schema')!='static_sr_scene_execution_plan_v1':
        raise ValueError('Plan snapshot root/schema differs')
    if (plan.get('status')!='executable_after_resource_admission' or plan.get('scene') not in SCENES
            or not re.fullmatch(r'[A-Za-z0-9_-]+',str(plan.get('run_id','')))):
        raise ValueError('Not a completed formal scene plan')
    run=RUN_PREFIX/plan['scene']/plan['run_id']
    if relpath(plan['run_root'])!=run:raise ValueError('Run root differs from exact scene/run scope')
    cameras=plan.get('train_cameras',[])
    if not cameras or len(cameras)!=len(set(cameras)) or any(not re.fullmatch(r'cam\d{2}',c) or c=='cam00' for c in cameras):
        raise ValueError('Illegal training-camera list')
    if set(plan['final_outputs'])!=set(METHODS) or set(plan['configs'])!=set(METHODS):
        raise ValueError('All four methods are required')
    for method in METHODS:
        expected=run/method/('coupled' if method=='threedsr' else 'sr')
        if relpath(plan['final_outputs'][method])!=expected:raise ValueError('Foreign final-output directory')
    return root/run


def legal_candidates(plan):
    """Independent closed policy; never trust arbitrary plan cleanup paths."""
    run=relpath(plan['run_root']);cams=plan['train_cameras'];allowed=set()
    for method in METHODS:
        p=run/method
        if method=='splatsure':allowed.update([p/'teacher/generated',p/'contribution/maps',p/'lr/checkpoint.pt'])
        elif method=='iesrgs':
            allowed.add(p/'internal_train/checkpoint.pt')
            allowed.update(p/name/c for name in ('external','internal_cache') for c in cams)
        elif method=='sequence_matters':allowed.update(p/'teacher'/c for c in cams)
        if method!='threedsr':allowed.update([p/'sr/checkpoint.pt',p/'sr/evaluation/renders'])
        else:
            allowed.update([p/'lr/checkpoint.pt',p/'coupled/latents',p/'coupled/evaluation/renders'])
            for i in range(4):
                allowed.update(p/f'coupled/teacher_{i}'/c for c in cams)
                allowed.add(p/f'coupled/reconstruction_{i}/checkpoint.pt')
                if i<3:allowed.add(p/f'coupled/reconstruction_{i}/feedback')
    supplied=[relpath(x) for x in plan['cleanup']['allowlist']]
    if len(supplied)!=len(set(supplied)) or set(supplied)-allowed:
        raise ValueError('Cleanup contains duplicate, input, workdir, shared-weight or foreign-task candidates')
    if set(supplied)!=allowed:
        raise ValueError('Cleanup must cover the complete closed large-output allowlist; no silent partial cleanup')
    return sorted(allowed,key=str)


def validate_job(plan,job,status,complete,job_sha,plan_sha):
    if (not re.fullmatch(r'[A-Za-z0-9_-]+',str(job.get('job_id','')))
            or job.get('scene')!=plan['scene'] or job.get('out')!=plan['run_root']+'/job_logs'
            or job.get('gpu_uuid')!=plan['gpu_uuid']):raise ValueError('Foreign job identity')
    if not any(x.get('sha256')==plan_sha for x in job.get('required_files',[])):
        raise ValueError('Executed job does not bind this plan')
    if status!=complete or complete.get('status')!='completed':raise ValueError('Job is running, failed, or incomplete')
    if (complete.get('schema')!='static_sr_cts_execution_v1' or complete.get('job_sha256')!=job_sha
            or complete.get('job_id')!=job.get('job_id') or complete.get('scene')!=plan['scene']):
        raise ValueError('Job terminal identity differs')
    stages=complete.get('stages',[])
    if [s.get('name') for s in stages]!=[c['name'] for c in job['commands']] or any(s.get('returncode')!=0 for s in stages):
        raise ValueError('Not all registered job stages completed successfully')


def validate_broker(broker,job):
    if (broker.get('job_id')!=job['job_id'] or broker.get('state') not in ('ended','completed')
            or broker.get('outcome') not in ('success','completed')):
        raise ValueError('Resource broker has not confirmed successful terminal state')


def task_still_alive(broker):
    task=broker.get('task_identity')
    if not isinstance(task,dict) or not all(k in task for k in ('pid','startticks','boot_id')):
        raise ValueError('Resource broker task process identity missing')
    if str(task['boot_id'])!=Path('/proc/sys/kernel/random/boot_id').read_text().strip():return False
    try:
        text=(Path('/proc')/str(int(task['pid']))/'stat').read_text()
    except FileNotFoundError:return False
    fields=text[text.rfind(')')+2:].split()
    return int(fields[19])==int(task['startticks']) and fields[0]!='Z'


def validate_metrics(plan,read_original):
    result={}
    for method in METHODS:
        directory=plan['final_outputs'][method]
        completion=read_original(directory+'/complete.json');metrics=read_original(directory+'/evaluation/metrics.json')
        if (completion.get('status')!='completed_with_evaluation' or completion.get('wiring_fixture') is not False
                or completion.get('config')!=plan['configs'][method]):raise ValueError('Incomplete/foreign method: '+method)
        config=read_original(plan['configs'][method]['path'])
        if config.get('method')!=method or config.get('scene')!=plan['scene_name']:raise ValueError('Foreign method configuration')
        if method=='threedsr':
            rounds=completion.get('rounds',[])
            if [r.get('round') for r in rounds]!=list(range(4)):raise ValueError('3DSR four-round completion missing')
            checkpoint=rounds[-1]['checkpoint']
        else:
            if completion.get('steps')!=30000:raise ValueError('Abbreviated training result')
            checkpoint=completion.get('checkpoint')
        if (not checkpoint or metrics.get('checkpoint')!=checkpoint
                or metrics.get('schema')!='static_sr_wu_uniform_png_metrics_v1'
                or metrics.get('scene')!=plan['scene_name'] or metrics.get('camera')!='cam00'
                or sorted(r['frame'] for r in metrics.get('frames',[]))!=list(range(300))):
            raise ValueError('Incomplete/foreign evaluation: '+method)
        for key in ('PSNR','SSIM','LPIPS'):
            values=[float(r[key]) for r in metrics['frames']]
            mean=float(metrics['metrics'][key])
            if not all(math.isfinite(v) for v in [mean,*values]) or not math.isclose(sum(values)/300,mean,rel_tol=1e-10,abs_tol=1e-10):
                raise ValueError('Nonfinite/inconsistent metrics: '+method)
        result[method]=metrics['metrics']
    return result


def verify_compact(compact,expected_sha):
    compact=Path(compact).absolute();no_links(compact,compact)
    index_path=no_links(compact/'index.json',compact)
    if sha(index_path)!=expected_sha:raise ValueError('Compact index differs from remote expected SHA256')
    index=load(index_path)
    if index.get('schema')!='static_sr_scene_compact_results_v1' or index.get('status')!='remote_collected_not_return_verified':
        raise ValueError('Not a complete compact result package')
    originals={};files=[]
    for item in index['files']:
        local=no_links(compact/relpath(item['path']),compact)
        original=str(relpath(item['original_path']))
        if original in originals:raise ValueError('Duplicate original compact record')
        if local.stat().st_size!=item['bytes'] or sha(local)!=item['sha256']:raise ValueError('Returned bytes mismatch: '+str(local))
        originals[original]=local;files.append(dict(path=item['path'],bytes=item['bytes'],sha256=item['sha256']))
    if len(files)!=len({x['path'] for x in files}) or sum(x['bytes'] for x in files)!=index['bytes']:
        raise ValueError('Compact byte inventory differs')
    actual={str(f.relative_to(compact)) for f in compact.rglob('*') if f.is_file()}
    if actual!={x['path'] for x in files}|{'index.json'}:raise ValueError('Foreign/unlisted files in compact return')
    for f in compact.rglob('*'):
        if f.is_symlink():raise ValueError('Symlink in compact return')
    return index,originals,files


def verify_return(args):
    compact=Path(args.compact).absolute()
    index,originals,files=verify_compact(compact,args.expected_index_sha256)
    plan_path=no_links(Path(args.plan).absolute(),compact)
    if plan_path not in originals.values():raise ValueError('Plan was not part of the verified compact transfer')
    plan=load(plan_path);remote_root=Path(plan['root']);snapshot_scope(plan,remote_root)
    if compact.is_relative_to(remote_root):raise ValueError('Verify return on the local controller, outside the execution snapshot')
    if index['scene']!=plan['scene'] or index['run_id']!=plan['run_id']:raise ValueError('Returned scene/run differs')
    values=validate_metrics(plan,lambda name:load(originals[str(relpath(name))]))
    for item in plan['configs'].values():
        if sha(originals[str(relpath(item['path']))])!=item['sha256']:raise ValueError('Returned configuration identity differs')
    job_paths=[Path(getattr(args,n)).absolute() for n in ('job','job_status','job_complete')]
    for path in job_paths:
        no_links(path,path.parent)
    job,status,complete=map(load,job_paths)
    validate_job(plan,job,status,complete,sha(job_paths[0]),sha(plan_path))
    broker_path=Path(args.broker_status).absolute();no_links(broker_path,broker_path.parent)
    validate_broker(load(broker_path),job)
    receipt=dict(schema='static_sr_scene_local_return_verified_v1',status='verified_local_SHA256',
        scene=plan['scene'],run_id=plan['run_id'],remote_snapshot_root=str(remote_root),
        compact_index=dict(path=plan['run_root']+'/compact/index.json',sha256=args.expected_index_sha256,bytes=(compact/'index.json').stat().st_size),
        plan_sha256=sha(plan_path),job_sha256=sha(job_paths[0]),job_complete_sha256=sha(job_paths[2]),
        broker_status_sha256=sha(broker_path),local_broker_status=str(broker_path),
        local_compact_path=str(compact),local_job_paths=list(map(str,job_paths)),files=files,
        bytes=sum(f['bytes'] for f in files),metrics=values,verified_utc=now(),
        verifier_host=socket.gethostname(),verifier_machine_id_sha256=machine_id(),
        verifier_source_sha256=sha(__file__),GPU_calls=0,deleted_files=0)
    write_new(args.out,receipt);return receipt


def stat_record(info):
    return dict(device=info.st_dev,inode=info.st_ino,size=info.st_size,mode=info.st_mode,
                uid=info.st_uid,gid=info.st_gid,nlink=info.st_nlink,
                mtime_ns=info.st_mtime_ns,ctime_ns=info.st_ctime_ns)


def snapshot_fd(fd,kind,prefix='.',allow_npz=False):
    before=os.fstat(fd);record=dict(path=prefix,kind=kind,**stat_record(before));result=[record]
    if kind=='file':
        if before.st_nlink!=1 or before.st_uid!=os.geteuid():raise ValueError('Shared/hardlinked or foreign-owned candidate refused')
        record['sha256']=hash_fd(fd)
    else:
        for name in sorted(os.listdir(fd)):
            info=os.stat(name,dir_fd=fd,follow_symlinks=False)
            rel=name if prefix=='.' else prefix+'/'+name
            if stat.S_ISLNK(info.st_mode):raise ValueError('Symlink in candidate: '+rel)
            if stat.S_ISDIR(info.st_mode):mode=DIR_FLAGS;child_kind='directory'
            elif stat.S_ISREG(info.st_mode):
                suffixes=('.pt','.png','.npy','.npz') if allow_npz else ('.pt','.png','.npy')
                if Path(name).suffix not in suffixes:raise ValueError('Unlisted/non-large artifact in deletion tree: '+rel)
                mode=os.O_RDONLY|NOFOLLOW;child_kind='file'
            else:raise ValueError('Special filesystem object refused')
            child=os.open(name,mode,dir_fd=fd)
            try:result.extend(snapshot_fd(child,child_kind,rel,allow_npz))
            finally:os.close(child)
    if stat_record(os.fstat(fd))!=stat_record(before):raise ValueError('Candidate changed while hashing')
    return result


def snapshot(path,allow_npz=False):
    path=scoped(path);fd=open_no_links(path,os.O_RDONLY|os.O_NONBLOCK)
    try:
        mode=os.fstat(fd).st_mode
        if stat.S_ISDIR(mode):kind='directory'
        elif stat.S_ISREG(mode):kind='file'
        else:raise ValueError('Only regular files and directories may be candidates')
        return snapshot_fd(fd,kind,allow_npz=allow_npz)
    finally:os.close(fd)


def find_writers(paths):
    """Supplement lock admission with same-user open writable descriptor checks."""
    roots=[Path(p) for p in paths];found=[]
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit() or int(entry.name)==os.getpid():continue
        try:
            if entry.stat().st_uid!=os.geteuid():continue
            descriptors=list((entry/'fd').iterdir())
        except FileNotFoundError:continue
        except PermissionError:raise RuntimeError('Cannot inspect a same-user process for active writers: '+entry.name)
        for descriptor in descriptors:
            try:
                target=Path(os.readlink(descriptor).removesuffix(' (deleted)'))
                if not target.is_absolute() or not any(target==p or target.is_relative_to(p) for p in roots):continue
                flags=next(line.split()[1] for line in (entry/'fdinfo'/descriptor.name).read_text().splitlines() if line.startswith('flags:'))
                if int(flags,8)&os.O_ACCMODE != os.O_RDONLY:found.append(dict(pid=int(entry.name),path=str(target),fd=descriptor.name))
            except (FileNotFoundError,ProcessLookupError):continue
    return found


@contextmanager
def admission_locks(run):
    shared=ROOT.parent.parent/'deployment/cts_20260921'
    fds=[]
    try:
        for path in (shared/'project_access.lock',shared/'gpu0.lock'):
            no_links(path,ROOT.parent.parent)
            fd=open_no_links(path,os.O_RDWR);fds.append(fd)
            fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        fd=open_no_links(run/'cleanup.lock',os.O_RDWR|os.O_CREAT);fds.append(fd)
        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (shared/'sync_pending.json').exists():raise RuntimeError('Shared project has an unverified active synchronization')
        yield
    finally:
        for fd in reversed(fds):os.close(fd)


def gate_bundle(plan_path,return_path):
    plan_path=scoped(plan_path);return_path=scoped(return_path);plan=load(plan_path)
    run=snapshot_scope(plan,ROOT);no_links(run,ROOT);candidates=legal_candidates(plan)
    returned=load(return_path)
    if (returned.get('schema')!='static_sr_scene_local_return_verified_v1' or returned.get('status')!='verified_local_SHA256'
            or returned.get('scene')!=plan['scene'] or returned.get('run_id')!=plan['run_id']
            or returned.get('remote_snapshot_root')!=str(ROOT) or returned.get('plan_sha256')!=sha(plan_path)):
        raise ValueError('Missing/foreign local SHA-verified return receipt')
    if returned.get('verifier_machine_id_sha256')==machine_id() or returned.get('verifier_host')==socket.gethostname():
        raise ValueError('Return verification was run on the execution host, not the local controller')
    if (not re.fullmatch(r'[a-f0-9]{64}',str(returned.get('verifier_machine_id_sha256','')))
            or not returned.get('verifier_host') or returned.get('verifier_source_sha256')!=sha(__file__)):
        raise ValueError('Local verifier machine/source identity is missing or changed')
    index_path=run/'compact/index.json'
    if returned['compact_index']['path']!=str(index_path.relative_to(ROOT)):
        raise ValueError('Returned compact index path differs')
    if returned['compact_index'].get('bytes')!=index_path.stat().st_size:raise ValueError('Returned compact index byte size differs')
    index,originals,files=verify_compact(run/'compact',returned['compact_index']['sha256'])
    if returned['files']!=files or returned['bytes']!=index['bytes']:raise ValueError('Return manifest differs from compact bytes')
    values=validate_metrics(plan,lambda name:load(scoped(name)))
    if values!=returned['metrics']:raise ValueError('Remote final metrics differ from local returned metrics')
    critical=[plan_path,return_path,index_path]
    for method in METHODS:
        directory=scoped(plan['final_outputs'][method]);critical += [directory/'complete.json',directory/'evaluation/metrics.json']
        config=scoped(plan['configs'][method]['path'])
        if sha(config)!=plan['configs'][method]['sha256']:raise ValueError('Configuration bytes changed')
        critical.append(config)
        for path in (directory/'complete.json',directory/'evaluation/metrics.json'):
            if sha(path)!=sha(originals[str(path.relative_to(ROOT))]):raise ValueError('Final result changed after compact collection')
    jobs=[run/'job_logs'/name for name in ('job.json','status.json','complete.json')]
    for path in jobs:no_links(path,ROOT)
    validate_job(plan,*map(load,jobs),sha(jobs[0]),sha(plan_path))
    if returned['job_sha256']!=sha(jobs[0]) or returned['job_complete_sha256']!=sha(jobs[2]):raise ValueError('Returned job completion identity differs')
    critical+=jobs
    job=load(jobs[0]);broker_path=ROOT.parent.parent/'.claude-training/jobs'/job['job_id']/'status.json'
    no_links(broker_path,ROOT.parent.parent);broker=load(broker_path);validate_broker(broker,job)
    if returned.get('broker_status_sha256')!=sha(broker_path):raise ValueError('Broker terminal bytes differ from local returned status')
    if task_still_alive(broker):raise RuntimeError('Broker task process is still running')
    critical.append(broker_path)
    paths=[scoped(p) for p in candidates]
    if any(not path.exists() for path in paths):raise ValueError('Missing deletion candidate; no partial/replayed cleanup')
    writers=find_writers(paths)
    if writers:raise RuntimeError('Active writer refused: '+json.dumps(writers))
    return plan,run,paths,[identity(path) for path in critical]


def freeze_candidates(paths):
    result=[]
    for path in paths:
        # The opt-in lossless IE codec changes only these two producer layouts.
        # No extension expansion applies to Splat/PSRT/3DSR or arbitrary trees.
        allow_npz=(path.parts[-3]=='iesrgs' and path.parts[-2] in ('external','internal_cache')
                   and re.fullmatch(r'cam\d{2}',path.parts[-1]) is not None)
        result.append(dict(path=str(path.relative_to(ROOT)),allow_npz=allow_npz,entries=snapshot(path,allow_npz)))
    return result


def comparable_after_rename(entries):
    # Renaming a directory/file can change ctime; pre-rename comparisons include it.
    return [{k:v for k,v in e.items() if k!='ctime_ns'} for e in entries]


def rename_noreplace(source,destination,source_fd,destination_fd):
    libc=ctypes.CDLL(None,use_errno=True)
    if not hasattr(libc,'renameat2'):raise RuntimeError('Linux renameat2(RENAME_NOREPLACE) required')
    call=libc.renameat2;call.argtypes=[ctypes.c_int,ctypes.c_char_p,ctypes.c_int,ctypes.c_char_p,ctypes.c_uint];call.restype=ctypes.c_int
    if call(source_fd,os.fsencode(source),destination_fd,os.fsencode(destination),1):
        error=ctypes.get_errno();raise OSError(error,os.strerror(error))


def delete_verified_tree(path,expected,journal,allow_npz=False):
    """Only unlink frozen regular files via pinned parent dirfds; no rmtree."""
    if comparable_after_rename(snapshot(path,allow_npz))!=comparable_after_rename(expected):raise ValueError('Quarantine identity changed')
    for entry in sorted(expected,key=lambda e:(len(Path(e['path']).parts),e['path']),reverse=True):
        current=path if entry['path']=='.' else path/entry['path']
        parent=scoped(current.parent);fd=open_no_links(parent,DIR_FLAGS)
        try:
            info=os.stat(current.name,dir_fd=fd,follow_symlinks=False)
            if (info.st_dev,info.st_ino)!=(entry['device'],entry['inode']):raise ValueError('Object replaced before unlink')
            if entry['kind']=='directory':os.rmdir(current.name,dir_fd=fd)
            else:
                child=os.open(current.name,os.O_RDONLY|NOFOLLOW,dir_fd=fd)
                try:
                    state=stat_record(os.fstat(child))
                    if any(state[k]!=entry[k] for k in ('device','inode','size','mode','uid','gid','nlink','mtime_ns')) or hash_fd(child)!=entry['sha256']:
                        raise ValueError('File changed immediately before unlink')
                    if stat_record(os.fstat(child))!=state:raise ValueError('File changed during final SHA verification')
                    os.unlink(current.name,dir_fd=fd)
                    journal.write(json.dumps(dict(action='unlinked',original_identity=entry,quarantine_path=str(current.relative_to(ROOT))))+'\n')
                    journal.flush()
                finally:os.close(child)
        finally:os.close(fd)


def cleanup(args):
    if args.apply and not args.freeze:raise ValueError('--apply requires the earlier exact --freeze receipt')
    plan_path=scoped(args.plan);plan=load(plan_path);run=snapshot_scope(plan,ROOT)
    out=scoped(args.out)
    if not out.is_relative_to(run/'cleanup_records') or out.suffix!='.json':raise ValueError('Cleanup receipts must stay under this run/cleanup_records/*.json')
    if out.exists():raise FileExistsError(out)
    no_links(run,ROOT)
    with admission_locks(run):
        plan,run,paths,critical=gate_bundle(plan_path,args.return_receipt)
        frozen=freeze_candidates(paths)
        if not args.apply:
            result=dict(schema='static_sr_scene_cleanup_freeze_v1',status='dry_run_frozen_no_deletion',
                scene=plan['scene'],run_id=plan['run_id'],root=str(ROOT),plan=identity(plan_path),
                return_receipt=identity(scoped(args.return_receipt)),critical_files=critical,candidates=frozen,
                logical_bytes=sum(e['size'] for c in frozen for e in c['entries'] if e['kind']=='file'),
                file_count=sum(e['kind']=='file' for c in frozen for e in c['entries']),
                created_utc=now(),cleanup_source_sha256=sha(__file__),deleted_files=0)
            write_new(out,result);return result
        if not args.freeze:raise ValueError('--apply requires the earlier exact --freeze receipt')
        frozen_path=scoped(args.freeze)
        if not frozen_path.is_relative_to(run/'cleanup_records'):raise ValueError('Foreign freeze receipt')
        prior=load(frozen_path)
        if (prior.get('schema')!='static_sr_scene_cleanup_freeze_v1' or prior.get('status')!='dry_run_frozen_no_deletion'
                or prior.get('root')!=str(ROOT) or prior.get('plan')!=identity(plan_path)
                or prior.get('return_receipt')!=identity(scoped(args.return_receipt))
                or prior.get('cleanup_source_sha256')!=sha(__file__) or prior.get('critical_files')!=critical
                or prior.get('candidates')!=frozen):raise ValueError('Frozen identity changed; no deletion performed')
        # All identities pass before the first mutation. Every candidate is then
        # moved atomically and rehashed before any deletion starts.
        quarantine=run/'cleanup_records'/('quarantine_'+uuid.uuid4().hex)
        quarantine.mkdir(mode=0o700,parents=True,exist_ok=False)
        qfd=open_no_links(quarantine,DIR_FLAGS);moved=[]
        result=dict(schema='static_sr_scene_cleanup_execution_v1',status='staging',scene=plan['scene'],run_id=plan['run_id'],
            freeze=identity(frozen_path),return_receipt=identity(scoped(args.return_receipt)),started_utc=now(),
            quarantine=str(quarantine.relative_to(ROOT)),staged=moved,deleted_candidates=[],GPU_calls=0,SSH_calls=0)
        journal_path=out.with_suffix('.events.jsonl')
        try:
            for number,candidate in enumerate(frozen):
                source=scoped(candidate['path']);destination=quarantine/f'{number:05d}'
                fd=open_no_links(source.parent,DIR_FLAGS)
                try:
                    if snapshot(source,candidate['allow_npz'])!=candidate['entries']:raise ValueError('Candidate changed before atomic staging')
                    rename_noreplace(source.name,destination.name,fd,qfd)
                finally:os.close(fd)
                moved.append(dict(original=candidate['path'],quarantine=str(destination.relative_to(ROOT))))
                if comparable_after_rename(snapshot(destination,candidate['allow_npz']))!=comparable_after_rename(candidate['entries']):
                    raise ValueError('Post-rename identity differs; staged objects preserved')
            writers=find_writers([quarantine])
            if writers:raise RuntimeError('Active quarantine writer refused: '+json.dumps(writers))
            with journal_path.open('x') as journal:
                for candidate,mapping in zip(frozen,moved):
                    delete_verified_tree(scoped(mapping['quarantine']),candidate['entries'],journal,candidate['allow_npz'])
                    result['deleted_candidates'].append(candidate['path'])
                journal.flush();os.fsync(journal.fileno())
            os.rmdir(quarantine)
            result.update(status='applied_verified',finished_utc=now(),deleted_files=prior['file_count'],
                          logical_bytes_unlinked=prior['logical_bytes'],physical_bytes_freed_not_measured=True,journal=identity(journal_path))
            write_new(out,result);return result
        except BaseException as error:
            result.update(status='failed_preserve_quarantine',error=type(error).__name__+': '+str(error),finished_utc=now(),
                          recovery='Do not retry automatically. Quarantine contains any staged but not deleted objects; use staged mapping and journal.')
            if journal_path.is_file():result['journal']=identity(journal_path)
            write_new(out,result);raise
        finally:os.close(qfd)


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='action',required=True)
    p=sub.add_parser('verify-return')
    for name in ('compact','expected-index-sha256','plan','job','job-status','job-complete','broker-status','out'):p.add_argument('--'+name,required=True)
    p=sub.add_parser('cleanup');p.add_argument('--plan',required=True);p.add_argument('--return-receipt',required=True)
    p.add_argument('--out',required=True);p.add_argument('--apply',action='store_true');p.add_argument('--freeze')
    args=parser.parse_args();result=verify_return(args) if args.action=='verify-return' else cleanup(args)
    print(json.dumps({k:result[k] for k in ('status','scene','run_id')}))


if __name__=='__main__':main()
