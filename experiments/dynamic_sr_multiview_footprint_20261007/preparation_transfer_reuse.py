"""CPU/local-only verified immutable transfer reuse; completion owns all gates.

Build independent read-only byte copies from one retained incoming stage, then
hard-link matching blobs into new preparation stages. Inotify handles directory
creation and atomic manifest publication; no SSH, GPU or model polling exists.
"""
from __future__ import annotations
import argparse,ctypes,fcntl,hashlib,json,os,re,select,stat,struct,time
from pathlib import Path,PurePosixPath
from fp_common import ROOT,HERE,OUT,local,read,write,entry,sha
import completion

SUFFIXES={'.npz','.pt','.py'}
COMPLETION_SHA='e7b55bf9b504c67e4df7480be078e07b761dcca2f9b7d94fd1da2572d1fdcfeb'
IN_CREATE=0x100;IN_MOVED_TO=0x80;IN_CLOSE_WRITE=8;IN_ISDIR=0x40000000
IN_FATAL=0x4000|0x400|0x800|0x8000


def relative(value):
    if not isinstance(value,str):raise ValueError('Manifest path must be text')
    p=PurePosixPath(value)
    if not isinstance(value,str) or not value or p.is_absolute() or any(x in ('','..','.') for x in value.split('/')) or '\x00' in value:raise ValueError('Unsafe manifest relative path')
    return Path(*p.parts)


def open_directory(root,parts=(),create=False):
    fd=os.open(root,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:
        for part in parts:
            if create:
                try:os.mkdir(part,dir_fd=fd)
                except FileExistsError:pass
            nxt=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);os.close(fd);fd=nxt
        return fd
    except BaseException:os.close(fd);raise


def file_handle(root,path):
    d=open_directory(root,path.parent.parts)
    try:fd=os.open(path.name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=d)
    finally:os.close(d)
    handle=os.fdopen(fd,'rb');s=os.fstat(fd)
    if not stat.S_ISREG(s.st_mode):handle.close();raise ValueError('Regular file required')
    return handle,s


def signature(s):return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)


def frozen_manifest(stage,workspace,protocol):
    with file_handle(stage,Path('manifest.json'))[0] as f:
        before=os.fstat(f.fileno());raw=f.read();after=os.fstat(f.fileno())
    if signature(before)!=signature(after):raise ValueError('Manifest changed while being read')
    value=json.loads(raw)
    if value.get('kind')!='manifest' or value.get('status')!='frozen_remote_file_SHA_inventory' or value.get('workspace')!=workspace or value.get('purpose')!='prepare' or value.get('task')!='' or value.get('formal_updates')!=0 or value.get('GPU_calls')!=0:raise ValueError('Exact frozen preparation manifest/workspace required')
    exports=Path(workspace)/completion.REL_OUT/'completion_remote/exports';transfer=Path(value['transfer_root'])
    if transfer.parent!=exports or not re.fullmatch(r'preparation_[0-9]+',transfer.name):raise ValueError('Unowned remote snapshot identity')
    paths={}
    for item in value['entries']:
        p=relative(item['path'])
        if str(p) in paths or type(item['bytes']) is not int or item['bytes']<0 or not re.fullmatch('[0-9a-f]{64}',item['sha256']):raise ValueError('Duplicate/invalid manifest entry')
        if item['role'] not in ('artifact','source'):raise ValueError('Unexpected transfer role')
        paths[str(p)]=item
    expected=paths.get(protocol['path'])
    if expected is None or expected['sha256']!=protocol['sha256'] or expected['bytes']!=protocol['bytes'] or expected['role']!='artifact':raise ValueError('Registered preparation protocol identity differs')
    identities={e['path']:e['sha256'] for e in value['source_identities']}
    if any(item['role']=='source' and identities.get(name)!=item['sha256'] for name,item in paths.items()):raise ValueError('Frozen source identity differs from source entry')
    return value,paths,dict(path=str(Path(stage)/'manifest.json'),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw)),raw


def candidates(path):
    yield Path('files')/path
    for special in ('.~tmp~','.completion-rsync-partial'):
        yield Path('files')/path.parent/special/path.name
        yield Path('files')/special/path
        yield Path(special)/path


class Reuse:
    def __init__(self,cache,workspace,protocol,registration):
        if sha(HERE/'completion.py')!=COMPLETION_SHA:raise ValueError('Frozen completion dependency changed')
        self.dependencies={name:entry(HERE/name) for name in ('completion.py','fp_common.py')}
        self.cache=Path(cache);self.workspace=workspace;self.protocol=protocol;self.registration=registration
        self.cache.mkdir(parents=True,exist_ok=True)
        if self.cache.is_symlink():raise ValueError('Reuse cache symlink refused')
        self.blobs=self.cache/'blobs';self.blobs.mkdir(exist_ok=True)
        if self.blobs.is_symlink():raise ValueError('Reuse blob directory symlink refused')
        self.index={};self.processed=set();self.cost=dict(bytes_read_for_cache=0,bytes_copied_to_independent_cache=0,bytes_read_for_seed_SHA=0,seeded_hardlink_bytes=0,seeded_files=0,cache_build_wall_seconds=0.,seed_wall_seconds=0.)
        old=self.cache/'index.json'
        if old.exists():
            value=read(old)
            if value.get('status')!='verified_independent_immutable_reuse_cache_not_completion' or value.get('workspace')!=workspace or value.get('protocol')!=protocol or value.get('registration')!=registration:raise ValueError('Existing reuse cache identity differs; preserve it')
            for row in value['entries']:
                p=relative(row['path'])
                if p.suffix not in SUFFIXES or type(row['bytes']) is not int or not re.fullmatch('[0-9a-f]{64}',row['sha256']) or row['blob']!=row['sha256']+'_'+str(row['bytes'])+'.blob' or row['path'] in self.index:raise ValueError('Invalid cached immutable identity')
                self.index[row['path']]=row
    def receipt(self,label,value):
        path=self.cache/'receipts'/f'{label}_{time.time_ns()}.json'
        write(path,dict(**value,source=entry(Path(__file__)),dependencies=self.dependencies,registration=self.registration,workspace=self.workspace,protocol=self.protocol,GPU_calls=0,SSH_calls=0,preparation_gate_published=False,completion_SHA_acceptance_still_required=True));return path
    def build(self,stage):
        started=time.monotonic();manifest,paths,identity,raw=frozen_manifest(stage,self.workspace,self.protocol);rows=[]
        (self.cache/'manifests').mkdir(exist_ok=True);(self.cache/'manifests'/f"{identity['sha256']}.json").write_bytes(raw)
        for name,item in paths.items():
            p=relative(name)
            if p.suffix not in SUFFIXES:continue
            blob=self.blobs/(item['sha256']+'_'+str(item['bytes'])+'.blob');verified=False;errors=[]
            old=self.index.get(name)
            if old and (old['sha256'],old['bytes'])==(item['sha256'],item['bytes']):
                rows.append(dict(path=name,status='existing_independent_cache_deferred_to_full_seed_SHA_verification'));continue
            for candidate in candidates(p):
                temp=None
                try:
                    handle,start=file_handle(stage,candidate)
                    with handle:
                        if start.st_size!=item['bytes']:continue
                        temp=self.blobs/(blob.name+'.'+str(time.time_ns())+'.pending');digest=hashlib.sha256();copied=0
                        with temp.open('xb') as dest:
                            for chunk in iter(lambda:handle.read(1024*1024),b''):
                                digest.update(chunk);dest.write(chunk);copied+=len(chunk)
                            dest.flush();os.fsync(dest.fileno())
                        self.cost['bytes_read_for_cache']+=copied;self.cost['bytes_copied_to_independent_cache']+=copied
                        os.chmod(temp,0o444)
                        if signature(start)!=signature(os.fstat(handle.fileno())) or copied!=item['bytes'] or digest.hexdigest()!=item['sha256']:raise ValueError('Candidate is partial/mutated/wrong SHA')
                    os.chmod(temp,0o444)
                    try:os.link(temp,blob,follow_symlinks=False)
                    except FileExistsError:
                        with file_handle(self.blobs,Path(blob.name))[0] as old:
                            olddigest=hashlib.sha256();oldcount=0
                            for chunk in iter(lambda:old.read(1024*1024),b''):olddigest.update(chunk);oldcount+=len(chunk)
                            self.cost['bytes_read_for_cache']+=oldcount
                            if oldcount!=item['bytes'] or olddigest.hexdigest()!=item['sha256']:raise ValueError('Existing immutable cache blob differs')
                    # Only our private independent temporary copy is removed.
                    temp.unlink();temp=None;verified=True
                    row=dict(path=name,sha256=item['sha256'],bytes=item['bytes'],blob=blob.name,source_stage=str(stage),source_candidate=str(candidate),independent_copy=True,initial_cache_mode='0444')
                    self.index[name]=row;rows.append(row);break
                except (FileNotFoundError,NotADirectoryError):pass
                except (OSError,ValueError) as error:errors.append(dict(candidate=str(candidate),error=repr(error),retained_pending_copy=str(temp) if temp else None))
            if not verified:rows.append(dict(path=name,status='not_cached_no_verified_complete_candidate',errors=errors))
        self.cost['cache_build_wall_seconds']+=time.monotonic()-started
        write(self.cache/'index.json',dict(status='verified_independent_immutable_reuse_cache_not_completion',workspace=self.workspace,protocol=self.protocol,manifest=identity,entries=list(self.index.values()),cost=self.cost,source=entry(Path(__file__)),dependencies=self.dependencies,registration=self.registration,preparation_gate_published=False))
        return self.receipt('cache_build',dict(status='cache_build_verified_candidates_only_not_preparation_complete',manifest=identity,rows=rows,cost=self.cost))
    def seed(self,stage):
        started=time.monotonic();manifest,paths,identity,raw=frozen_manifest(stage,self.workspace,self.protocol)
        key=(str(stage),identity['sha256'])
        if key in self.processed:return None
        rows=[]
        for name,item in paths.items():
            cached=self.index.get(name)
            if cached is None or relative(name).suffix not in SUFFIXES or (cached['sha256'],cached['bytes'])!=(item['sha256'],item['bytes']):continue
            p=relative(name);fd=None
            try:
                fd=open_directory(stage,('files',)+p.parent.parts,True)
                try:os.stat(p.name,dir_fd=fd,follow_symlinks=False)
                except FileNotFoundError:pass
                else:rows.append(dict(path=name,status='skipped_existing_destination_never_overwritten'));continue
                with file_handle(self.blobs,Path(cached['blob']))[0] as source:
                    before=os.fstat(source.fileno());digest=hashlib.sha256();count=0
                    for chunk in iter(lambda:source.read(1024*1024),b''):digest.update(chunk);count+=len(chunk)
                    self.cost['bytes_read_for_seed_SHA']+=count
                    if signature(before)!=signature(os.fstat(source.fileno())) or count!=item['bytes'] or digest.hexdigest()!=item['sha256']:raise ValueError('Reuse cache SHA/size changed')
                    os.link(self.blobs/cached['blob'],p.name,dst_dir_fd=fd,follow_symlinks=False)
                    installed=os.stat(p.name,dir_fd=fd,follow_symlinks=False)
                    if not stat.S_ISREG(installed.st_mode) or (installed.st_dev,installed.st_ino)!=(before.st_dev,before.st_ino):raise ValueError('Seeded inode differs from the verified cache')
                self.cost['seeded_files']+=1;self.cost['seeded_hardlink_bytes']+=item['bytes'];rows.append(dict(path=name,status='seeded_verified_immutable_hardlink',bytes=item['bytes'],sha256=item['sha256']))
            except FileExistsError:rows.append(dict(path=name,status='skipped_destination_created_during_seed_never_overwritten'))
            except (OSError,ValueError) as error:rows.append(dict(path=name,status='not_seeded_error_saved',error=repr(error)))
            finally:
                if fd is not None:os.close(fd)
        self.processed.add(key);self.cost['seed_wall_seconds']+=time.monotonic()-started
        return self.receipt('seed',dict(status='immutable_reuse_seed_attempt_not_preparation_complete',manifest=identity,stage=str(stage),rows=rows,cost=self.cost,potential_retransmit_bytes_avoided_if_rsync_uses_seeded_destinations=sum(r.get('bytes',0) for r in rows if r['status']=='seeded_verified_immutable_hardlink'),actual_network_bytes_avoided_not_measured=True))


class StageEvents:
    def __init__(self,incoming):
        self.incoming=Path(incoming);self.lib=ctypes.CDLL(None,use_errno=True);self.lib.inotify_init1.argtypes=[ctypes.c_int];self.lib.inotify_add_watch.argtypes=[ctypes.c_int,ctypes.c_char_p,ctypes.c_uint32]
        self.fd=self.lib.inotify_init1(os.O_CLOEXEC)
        if self.fd<0:raise OSError(ctypes.get_errno(),'inotify_init1')
        self.paths={};self.stages=set();self.add(self.incoming,IN_CREATE|IN_MOVED_TO);self.poll=select.poll();self.poll.register(self.fd,select.POLLIN)
    def add(self,path,mask):
        if Path(path).is_symlink():raise ValueError('Incoming stage symlink refused')
        wd=self.lib.inotify_add_watch(self.fd,os.fsencode(path),mask|0x400|0x800)
        if wd<0:raise OSError(ctypes.get_errno(),'inotify_add_watch')
        self.paths[wd]=Path(path)
    def stage(self,path):
        path=Path(path)
        if path in self.stages:return False
        if path.parent!=self.incoming or not re.fullmatch(r'preparation_[0-9]+',path.name):return False
        self.add(path,IN_CLOSE_WRITE|IN_MOVED_TO);self.stages.add(path);return True
    def next(self):
        self.poll.poll();raw=os.read(self.fd,1024*1024);i=0;events=[]
        while i<len(raw):
            wd,mask,cookie,length=struct.unpack_from('iIII',raw,i);i+=16;name=os.fsdecode(raw[i:i+length].split(b'\0',1)[0]);i+=length
            if mask&IN_FATAL:raise RuntimeError('Incoming inotify watch overflow/lost; recover explicitly')
            parent=self.paths[wd]
            if parent==self.incoming and mask&IN_ISDIR and mask&(IN_CREATE|IN_MOVED_TO):events.append(('stage',parent/name))
            elif parent!=self.incoming and name=='manifest.json' and mask&(IN_CLOSE_WRITE|IN_MOVED_TO):events.append(('manifest',parent))
        return events
    def close(self):
        if self.fd is not None:os.close(self.fd);self.fd=None


def serve(reuse,incoming,source_stage):
    watcher=StageEvents(incoming)
    def attempt(stage):
        if Path(stage)==Path(source_stage):return None
        try:return reuse.seed(stage)
        except (FileNotFoundError,ValueError,OSError,json.JSONDecodeError) as error:return reuse.receipt('refused_stage',dict(status='stage_not_seeded_manifest_or_path_error',stage=str(stage),error=repr(error)))
    try:
        # Subscribe first. Initial scan and each newly registered watch get one
        # recovery read; thereafter only close/move/create events trigger work.
        for stage in sorted(Path(incoming).iterdir()):
            if stage.is_dir() and watcher.stage(stage):pass
        reuse.build(source_stage)
        current=[stage for stage in watcher.stages if stage!=Path(source_stage)]
        if current:
            stage=max(current,key=lambda p:int(p.name.split('_',1)[1]))
            if (stage/'manifest.json').exists():attempt(stage)
        while True:
            for kind,stage in watcher.next():
                if kind=='stage':
                    if watcher.stage(stage) and (stage/'manifest.json').exists():attempt(stage)
                else:attempt(stage)
    finally:watcher.close()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=('serve','seed'),default='serve');p.add_argument('--registration',type=Path,required=True);p.add_argument('--source-stage',type=Path,required=True);p.add_argument('--cache',type=Path,default=OUT/'completion/preparation_reuse_cache');p.add_argument('--stage',type=Path);a=p.parse_args()
    cfg=read(local(a.registration))
    if cfg['workspace']!=completion.DEFAULT_WORKSPACE or cfg['remote_host']!='a100-train':raise ValueError('Registered preparation workspace/host required')
    protocol=entry(OUT/'protocol.json');protocol['bytes']=(OUT/'protocol.json').stat().st_size
    incoming=OUT/'completion/incoming';source=local(a.source_stage);cache=local(a.cache)
    if source.parent!=incoming or not re.fullmatch(r'preparation_[0-9]+',source.name) or cache.is_symlink() or not cache.resolve().is_relative_to(OUT/'completion'):raise ValueError('Only owned incoming source and independent local completion cache allowed')
    cache.mkdir(parents=True,exist_ok=True)
    with (cache/'reuse.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);reuse=Reuse(cache,cfg['workspace'],protocol,entry(local(a.registration)))
        if a.mode=='serve':return serve(reuse,incoming,source)
        if not a.stage:p.error('seed requires explicit --stage')
        stage=local(a.stage)
        if stage==source or stage.parent!=incoming or not re.fullmatch(r'preparation_[0-9]+',stage.name):raise ValueError('Seed destination is not a distinct owned incoming stage')
        reuse.build(source);reuse.seed(stage)

if __name__=='__main__':main()
