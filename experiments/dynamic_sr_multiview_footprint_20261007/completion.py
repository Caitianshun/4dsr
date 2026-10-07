"""CPU/network completion callbacks driven by pidfd/inotify, never model polling.

This program registers no services and chooses no resources. The root operator
starts its persistent service with an already resolved evaluation GPU UUID.
"""
from __future__ import annotations
import argparse
import ctypes
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path
import queue
import select
import shlex
import shutil
import socket
import struct
import subprocess
import sys
import threading
import time
import traceback
from fp_common import ROOT, HERE, OUT, read, write, sha, entry, bound, local

TASKS = tuple(f'r{suffix}_{method}' for suffix in ('1', '2')
              for method in ('B0', 'Bsync', 'M', 'X', 'MX', 'E'))
REL_OUT = str(OUT.relative_to(ROOT))
DEFAULT_WORKSPACE = '/home/ubuntu/3DGS/4dsr/execution_workspaces/dynamic_sr_multiview_footprint_20261007/4654f1b7eeb6f6ba'
DEFAULT_UNIT = '4dsr-footprint-native-preparation-a100-gpu1-20261007'
TRAINPY = '/home/cai_tianshun/Project/4dgs/.venv/bin/python'
IN_CLOSE_WRITE, IN_MOVED_TO = 0x8, 0x80
IN_Q_OVERFLOW, IN_DELETE_SELF, IN_MOVE_SELF, IN_IGNORED = 0x4000, 0x400, 0x800, 0x8000


def valid_task(task):
    if task not in TASKS: raise ValueError(f'Unregistered task: {task!r}')
    return task


def safe_relative(value):
    path = Path(value)
    if path.is_absolute() or '..' in path.parts or not path.parts:
        raise ValueError(f'Unsafe relative artifact path: {value!r}')
    return path


def pidfd_wait(pid):
    descriptor = os.pidfd_open(int(pid))
    try:
        watcher = select.poll(); watcher.register(descriptor, select.POLLIN)
        events = watcher.poll()  # Infinite kernel wait; no process/model polling.
        if not events or not events[0][1] & select.POLLIN:
            raise RuntimeError('No persistent pidfd POLLIN exit proof')
        return dict(pid=int(pid), exit_proved=True, authority='persistent_pidfd_POLLIN')
    finally: os.close(descriptor)


class DirectoryEvents:
    """One blocking Linux inotify watch plus optional persistent FIFO."""
    def __init__(self, directory, fifo=None):
        self.directory = Path(directory); self.directory.mkdir(parents=True, exist_ok=True)
        library = ctypes.CDLL(None, use_errno=True)
        library.inotify_init1.argtypes = [ctypes.c_int]; library.inotify_init1.restype = ctypes.c_int
        library.inotify_add_watch.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_uint32]
        library.inotify_add_watch.restype = ctypes.c_int
        self.fd = library.inotify_init1(os.O_CLOEXEC)
        if self.fd < 0: raise OSError(ctypes.get_errno(), 'inotify_init1')
        self.wd = library.inotify_add_watch(self.fd, os.fsencode(self.directory),
            IN_CLOSE_WRITE | IN_MOVED_TO | IN_DELETE_SELF | IN_MOVE_SELF)
        if self.wd < 0:
            error = ctypes.get_errno(); os.close(self.fd); raise OSError(error, 'inotify_add_watch')
        self.fifo, self.fifo_fd, self.fifo_buffer = Path(fifo) if fifo else None, None, b''
        if self.fifo:
            self.fifo.parent.mkdir(parents=True, exist_ok=True)
            if not self.fifo.exists(): os.mkfifo(self.fifo, 0o600)
            if not __import__('stat').S_ISFIFO(self.fifo.stat().st_mode):
                self.close(); raise ValueError('Completion FIFO path is not a FIFO')
            # Holding both ends prevents EOF/readability spin when no writer exists.
            self.fifo_fd = os.open(self.fifo, os.O_RDWR | os.O_NONBLOCK | os.O_CLOEXEC)
        self.poll = select.poll(); self.poll.register(self.fd, select.POLLIN)
        if self.fifo_fd is not None: self.poll.register(self.fifo_fd, select.POLLIN)

    def next(self):
        messages = []
        for descriptor, flags in self.poll.poll():
            if flags & (select.POLLERR | select.POLLHUP | select.POLLNVAL):
                raise RuntimeError('Completion event descriptor lost')
            if descriptor == self.fd:
                raw = os.read(self.fd, 1024*1024); position = 0
                while position < len(raw):
                    wd, mask, cookie, length = struct.unpack_from('iIII', raw, position)
                    position += 16; name = os.fsdecode(raw[position:position+length].split(b'\0', 1)[0]); position += length
                    if mask & (IN_Q_OVERFLOW | IN_DELETE_SELF | IN_MOVE_SELF | IN_IGNORED):
                        raise RuntimeError('Completion inotify overflow/watch invalidated; restart with recovery scan')
                    if mask & (IN_CLOSE_WRITE | IN_MOVED_TO) and name.endswith('.json'):
                        messages.append(dict(name=name, mechanism='inotify', mask=mask))
            elif descriptor == self.fifo_fd:
                self.fifo_buffer += os.read(self.fifo_fd, 65536)
                if len(self.fifo_buffer) > 1024*1024: raise ValueError('Oversized FIFO message')
                while b'\n' in self.fifo_buffer:
                    line, self.fifo_buffer = self.fifo_buffer.split(b'\n', 1)
                    if line:
                        value = json.loads(line); valid_task(value['task'])
                        messages.append(dict(name=value['task']+'.json', mechanism='FIFO'))
        return messages

    def close(self):
        if getattr(self, 'fifo_fd', None) is not None: os.close(self.fifo_fd); self.fifo_fd = None
        if getattr(self, 'fd', None) is not None: os.close(self.fd); self.fd = None


# The same CPU-only script is sent over SSH; it imports no workspace/model code.
REMOTE_HELPER = r'''
import ctypes,hashlib,json,os,pathlib,select,shutil,signal,struct,subprocess,sys,time
mode,workspace,relative_out=sys.argv[1:4]
root=pathlib.Path(workspace);out=root/relative_out
def emit(value):
 try:print(json.dumps(value,ensure_ascii=False),flush=True)
 except BrokenPipeError:
  sys.stdout=open(os.devnull,'w')
def read(path):return json.loads(path.read_text())
def write(path,value):
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.'+str(os.getpid())+'.tmp')
 tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)
def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(1048576),b''):h.update(block)
 return h.hexdigest()
def safe(value):
 p=pathlib.Path(value)
 if p.is_absolute() or '..' in p.parts:raise ValueError('Unsafe manifest path')
 return p
def wait_pid(pid):
 fd=os.pidfd_open(int(pid));p=select.poll();p.register(fd,select.POLLIN)
 try:
  flags=p.poll()
  if not flags or not flags[0][1]&select.POLLIN:raise RuntimeError('pidfd exit not proved')
 finally:os.close(fd)
def starttime(pid):
 text=pathlib.Path('/proc/'+str(pid)+'/stat').read_text();return text[text.rfind(')')+2:].split()[19]
if mode=='events':
 directory=out/'evaluation_queue';directory.mkdir(parents=True,exist_ok=True)
 libc=ctypes.CDLL(None,use_errno=True);libc.inotify_init1.restype=ctypes.c_int
 libc.inotify_add_watch.argtypes=[ctypes.c_int,ctypes.c_char_p,ctypes.c_uint32]
 fd=libc.inotify_init1(os.O_CLOEXEC)
 if fd<0:raise OSError(ctypes.get_errno(),'inotify_init1')
 wd=libc.inotify_add_watch(fd,os.fsencode(directory),0x8|0x80|0x400|0x800)
 if wd<0:raise OSError(ctypes.get_errno(),'inotify_add_watch')
 # Subscribe first, then one recovery scan per connection. Reconnect recovery
 # is necessary to recover durable messages missed during a disconnected stream.
 for path in sorted(directory.glob('*.json')):emit(dict(kind='queue',name=path.name,mechanism='startup_recovery'))
 emit(dict(kind='ready',mechanism='blocking_inotify'))
 while True:
  raw=os.read(fd,1048576);position=0
  while position<len(raw):
   wd,mask,cookie,length=struct.unpack_from('iIII',raw,position);position+=16
   name=os.fsdecode(raw[position:position+length].split(b'\0',1)[0]);position+=length
   if mask&(0x4000|0x400|0x800|0x8000):raise RuntimeError('Remote inotify stream invalidated')
   if mask&(0x8|0x80) and name.endswith('.json'):emit(dict(kind='queue',name=name,mechanism='inotify',mask=mask))
elif mode=='queue-request':
 task=sys.argv[4]
 if task not in {f'r{s}_{m}' for s in ('1','2') for m in ('B0','Bsync','M','X','MX','E')}:raise ValueError('Unregistered endpoint')
 emit(dict(kind='request',request=read(out/'evaluation_queue'/(task+'.json'))))
elif mode=='manifest':
 purpose,task=sys.argv[4:6];paths=set();source_hashes={}
 if purpose=='prepare':
  prefixes=['support','preparation','operator_checks','diagnostics/U6000','schedules','protocol.json','preparation_controller.log','workers/prepare_r1.json']
  prefixes += [p.name for p in out.glob('calibration*')]
  prefixes += [str(p.relative_to(out)) for p in (out/'workers').glob('prepare_*failed*.json')]
  cp=out/'preparation/complete.json'
  if cp.exists():source_hashes=read(cp).get('sources',{})
  elif (out/'support/parent_moments_hr/config.json').exists():
   source_hashes=read(out/'support/parent_moments_hr/config.json').get('sources',{})
 elif purpose=='task':
  allowed={f'r{s}_{m}' for s in ('1','2') for m in ('B0','Bsync','M','X','MX','E')}
  if task not in allowed:raise ValueError('Unregistered endpoint')
  prefixes=['runs/'+task,'logs/'+task+'.log','evaluation_queue/'+task+'.json','first100_timing.json','workers/suffix_'+task[1]+'_platform.json']
  prefixes += [str(p.relative_to(out)) for p in (out/'workers').glob('worker_r'+task[1]+'*.json')]
  config=out/'runs'/task/'config.json'
  if config.exists():source_hashes=read(config).get('source_identity',{})
 else:raise ValueError('Unknown bundle purpose')
 for prefix in prefixes:
  path=out/safe(prefix)
  if path.is_symlink():raise ValueError('Symlink artifacts not accepted')
  if path.is_file():paths.add(path)
  elif path.is_dir():
   for child in path.rglob('*'):
    if child.is_symlink():raise ValueError('Symlink artifacts not accepted')
    if child.is_file() and not child.name.endswith('.tmp'):paths.add(child)
 sources=[]
 for relative,expected in source_hashes.items():
  if relative.startswith('upstream/'):
   sources.append(dict(path=relative,sha256=expected,role='upstream_identity'))
   continue
  path=root/safe(relative)
  if not path.is_file() or digest(path)!=expected:raise ValueError('Bound remote source changed: '+relative)
  paths.add(path);sources.append(dict(path=relative,sha256=expected,role='source'))
 # Freeze atomic JSON/status files before transfer. Hardlink only immutable
 # closed checkpoint/NPZ assets; mutable logs and receipts are copied. A suffix
 # worker may start its next arm while this completed endpoint is returned.
 transfer=out/'completion_remote/exports'/((task or 'preparation')+'_'+str(time.time_ns()))
 transfer.mkdir(parents=True,exist_ok=False);entries=[]
 for path in sorted(paths):
  relative=path.relative_to(root);target=transfer/relative;target.parent.mkdir(parents=True,exist_ok=True)
  if path.suffix in ('.pt','.npz'):os.link(path,target)
  else:shutil.copy2(path,target)
  entries.append(dict(path=str(relative),sha256=digest(target),bytes=target.stat().st_size,
   role='artifact' if path.is_relative_to(out) else 'source'))
 emit(dict(kind='manifest',status='frozen_remote_file_SHA_inventory',workspace=str(root),purpose=purpose,task=task,
  transfer_root=str(transfer),entries=entries,source_identities=sources,formal_updates=0,GPU_calls=0))
elif mode=='wait-unit':
 unit,expected,user_scope=sys.argv[4:7];unit=unit if unit.endswith('.service') else unit+'.service'
 statepath=out/'completion_remote'/(unit+'.json')
 if statepath.exists():
  state=read(statepath)
  if expected and state.get('invocation_id')!=expected:raise ValueError('Previously bound unit InvocationID differs')
  if state.get('kind')=='terminal':emit(state);sys.exit(0)
  alive=False
  try:alive=starttime(state['watcher_pid'])==state['watcher_starttime']
  except (FileNotFoundError,ProcessLookupError):pass
  if alive:
   emit(dict(kind='bound',unit=unit,invocation_id=state['invocation_id'],reconnected_existing_watcher=True))
   try:wait_pid(state['watcher_pid'])
   except ProcessLookupError:pass
   terminal=read(statepath)
   if terminal.get('kind')!='terminal':raise RuntimeError('Existing watcher exited without durable terminal receipt')
   emit(terminal);sys.exit(0)
  raise RuntimeError('Interrupted watcher has no durable exact-exit receipt; manual incident resolution required')
 # Hold a D-Bus RefUnit on a live connection so a transient successful unit
 # remains inspectable after pidfd readiness. Subscribe to its PropertiesChanged
 # before the wait, and block on D-Bus when manager teardown is still pending.
 lib=ctypes.CDLL('libsystemd.so.0');bus=ctypes.c_void_p()
 opener=lib.sd_bus_open_user if user_scope=='user' else lib.sd_bus_open_system
 opener.argtypes=[ctypes.POINTER(ctypes.c_void_p)];opener.restype=ctypes.c_int
 if opener(ctypes.byref(bus))<0:raise RuntimeError('Cannot open systemd D-Bus')
 lib.sd_bus_call_method.restype=ctypes.c_int
 lib.sd_bus_message_read.restype=ctypes.c_int
 lib.sd_bus_unref.argtypes=[ctypes.c_void_p];lib.sd_bus_unref.restype=ctypes.c_void_p
 lib.sd_bus_message_unref.argtypes=[ctypes.c_void_p];lib.sd_bus_message_unref.restype=ctypes.c_void_p
 lib.sd_bus_process.argtypes=[ctypes.c_void_p,ctypes.c_void_p];lib.sd_bus_process.restype=ctypes.c_int
 lib.sd_bus_wait.argtypes=[ctypes.c_void_p,ctypes.c_uint64];lib.sd_bus_wait.restype=ctypes.c_int
 destination=b'org.freedesktop.systemd1';manager=b'/org/freedesktop/systemd1';interface=b'org.freedesktop.systemd1.Manager'
 def call(member):
  reply=ctypes.c_void_p()
  result=lib.sd_bus_call_method(bus,destination,manager,interface,member,None,ctypes.byref(reply),b's',unit.encode())
  if result<0:raise RuntimeError('systemd '+member.decode()+' failed: '+str(result))
  return reply
 reply=call(b'RefUnit');lib.sd_bus_message_unref(reply)
 reply=call(b'GetUnit');objectpath=ctypes.c_char_p()
 if lib.sd_bus_message_read(reply,b'o',ctypes.byref(objectpath))<0:raise RuntimeError('Cannot read systemd unit object')
 objectpath_value=objectpath.value;lib.sd_bus_message_unref(reply)
 changes=[0]
 CALLBACK=ctypes.CFUNCTYPE(ctypes.c_int,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p)
 @CALLBACK
 def changed(message,userdata,error):changes[0]+=1;return 1
 slot=ctypes.c_void_p()
 lib.sd_bus_match_signal.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_void_p),ctypes.c_char_p,ctypes.c_char_p,ctypes.c_char_p,ctypes.c_char_p,CALLBACK,ctypes.c_void_p]
 lib.sd_bus_match_signal.restype=ctypes.c_int
 if lib.sd_bus_match_signal(bus,ctypes.byref(slot),destination,objectpath_value,b'org.freedesktop.DBus.Properties',b'PropertiesChanged',changed,None)<0:
  raise RuntimeError('Cannot subscribe systemd unit lifecycle events')
 def inspect():
  cmd=['systemctl']+(['--user'] if user_scope=='user' else [])+['show',unit,'--no-pager']
  result=subprocess.run(cmd,capture_output=True,text=True,check=True)
  return dict(line.split('=',1) for line in result.stdout.splitlines() if '=' in line)
 initial=inspect();invocation=initial.get('InvocationID')
 if not invocation or expected and invocation!=expected:raise ValueError('Unit InvocationID missing/different')
 pid=int(initial.get('MainPID','0'))
 try:os.setsid()
 except PermissionError:pass
 signal.signal(signal.SIGHUP,signal.SIG_IGN)
 state=dict(kind='bound',unit=unit,invocation_id=invocation,main_pid=pid,watcher_pid=os.getpid(),watcher_starttime=starttime(os.getpid()),bound_at_unix=time.time(),initial=initial)
 write(statepath,state);emit(state)
 pidfd_proved=False
 if pid:
  try:wait_pid(pid);pidfd_proved=True
  except ProcessLookupError:pass
 current=inspect()
 def finished(properties):
  return properties.get('ActiveState') in ('inactive','failed') or (
   properties.get('ActiveState')=='active' and properties.get('SubState')=='exited' and
   properties.get('MainPID')=='0' and int(properties.get('ExecMainExitTimestampMonotonic','0'))>0)
 while not finished(current):
  changes[0]=0
  while lib.sd_bus_process(bus,None)>0:pass
  if changes[0]:current=inspect();continue
  while not changes[0]:
   if lib.sd_bus_wait(bus,2**64-1)<0:raise RuntimeError('systemd event wait failed')
   while lib.sd_bus_process(bus,None)>0:pass
  current=inspect() # Only on actual unit lifecycle signal, never timed polling.
 # Some systemd versions clear InvocationID when entering inactive. Retained
 # ExecMainPID and the exact monotonic start timestamp still bind that final
 # status to this invocation; a new invocation/PID or start is rejected.
 current_invocation=current.get('InvocationID')
 same_exec=(current.get('ExecMainPID')==initial.get('ExecMainPID') and
  int(initial.get('ExecMainStartTimestampMonotonic','0'))>0 and
  current.get('ExecMainStartTimestampMonotonic')==initial.get('ExecMainStartTimestampMonotonic'))
 if current_invocation not in (invocation,'') or current_invocation=='' and not same_exec:
  raise ValueError('Unit invocation changed while waiting')
 exit_code=int(current.get('ExecMainStatus','-1'));exit_kind=int(current.get('ExecMainCode','0'))
 exit_stamp=int(current.get('ExecMainExitTimestampMonotonic','0'))
 proved=exit_kind!=0 and exit_stamp>0
 successful=proved and exit_kind==1 and exit_code==0 and current.get('Result')=='success' and current.get('MainPID')=='0'
 terminal=dict(kind='terminal',status='systemd_exit0_proved' if successful else 'systemd_unsuccessful_or_unknown_exit',unit=unit,
  invocation_id=invocation,initial_main_pid=pid,pidfd_exit_proof=pidfd_proved,exit_proved=proved,exit_kind=exit_kind,exit_code=exit_code,
  terminal_InvocationID=current_invocation,retained_ExecMainPID_and_start_match=same_exec,
  successful=successful,authority='persistent_pidfd_and_RefUnit_retained_exact_systemd_ExecMainStatus' if pidfd_proved else 'retained_exact_systemd_terminal_metadata_recovery',systemd=current,
  bound_at_unix=state['bound_at_unix'],finished_unix=time.time(),formal_updates=0,GPU_calls=0)
 write(statepath,terminal);emit(terminal);lib.sd_bus_unref(bus)
else:raise ValueError('Unknown helper mode')
'''


class Transport:
    def __init__(self, host, workspace, out=OUT, remote_python='/usr/bin/python3', remote_activate=None):
        self.host, self.workspace, self.out = host, str(workspace), Path(out)
        self.remote_python, self.remote_activate = remote_python, remote_activate
        self.base = self.out/'completion'; self.base.mkdir(parents=True, exist_ok=True)
        self.children, self.children_lock = set(), threading.Lock()
        self.ssh = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20',
                    '-o', 'ServerAliveInterval=30', '-o', 'ServerAliveCountMax=3', host]

    def command(self, mode, *arguments):
        words = [self.remote_python, '-u', '-c', REMOTE_HELPER, mode, self.workspace, REL_OUT, *map(str, arguments)]
        command = shlex.join(words)
        if self.remote_activate:
            command = 'source '+shlex.quote(self.remote_activate)+' && exec '+command
            command = shlex.join(['bash', '-lc', command])
        return self.ssh+[command]

    def stream(self, mode, *arguments):
        log = self.base/(mode+'_ssh.stderr.log')
        with log.open('a') as error:
            child = subprocess.Popen(self.command(mode, *arguments), stdout=subprocess.PIPE, stderr=error, text=True)
            with self.children_lock: self.children.add(child)
            try:
                for line in child.stdout:
                    if line.strip(): yield json.loads(line)
                status = child.wait()
                if status == 255: raise ConnectionError(f'SSH helper {mode} transport exited {status}; {log}')
                if status: raise RuntimeError(f'Remote helper {mode} failed with {status}; {log}')
            finally:
                if child.poll() is None:
                    # Only this private SSH child; remote jobs/services are never signaled.
                    child.terminate(); child.wait()
                with self.children_lock: self.children.discard(child)

    def close(self):
        """Reap only SSH clients created by this transport, never remote units."""
        with self.children_lock: children = list(self.children)
        for child in children:
            if child.poll() is None:
                child.terminate()
                try: child.wait(timeout=10)
                except subprocess.TimeoutExpired: child.kill(); child.wait()

    def bundle(self, purpose, task='', terminal=None):
        started = time.monotonic()
        manifest_messages = list(self.stream('manifest', purpose, task))
        if len(manifest_messages) != 1 or manifest_messages[0].get('kind') != 'manifest':
            raise ValueError('No single remote artifact manifest')
        manifest = manifest_messages[0]
        if manifest['workspace'] != self.workspace: raise ValueError('Remote workspace changed')
        transfer_root = manifest['transfer_root']
        owned_exports = Path(self.workspace)/REL_OUT/'completion_remote/exports'
        if not Path(transfer_root).is_relative_to(owned_exports) or Path(transfer_root).parent != owned_exports:
            raise ValueError('Transfer snapshot outside the owned remote export directory')
        label = 'preparation' if purpose == 'prepare' else valid_task(task)
        receipt_id = label+'_'+str(time.time_ns()); stage = self.base/'incoming'/receipt_id
        stage.mkdir(parents=True, exist_ok=False)
        write(stage/'manifest.json', manifest)
        paths = [safe_relative(item['path']) for item in manifest['entries']]
        if len(set(paths)) != len(paths): raise ValueError('Duplicate manifest path')
        filelist = stage/'files.from0'; filelist.write_bytes(b''.join(os.fsencode(path)+b'\0' for path in paths))
        command = ['rsync', '-a', '--checksum', '--protect-args', '--from0', '--files-from='+str(filelist),
                   '--relative', '--delay-updates', '--partial-dir=.completion-rsync-partial',
                   '-e', 'ssh -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=30 -o ServerAliveCountMax=3',
                   self.host+':'+transfer_root+'/', str(stage/'files')+'/']
        with (stage/'rsync.log').open('w') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode: raise ConnectionError(f'Artifact rsync exited {result.returncode}; staging retained at {stage}')
        for item in manifest['entries']:
            path = stage/'files'/safe_relative(item['path'])
            if path.is_symlink() or not path.is_file() or path.stat().st_size != item['bytes'] or sha(path) != item['sha256']:
                raise ValueError(f'Received artifact SHA/size mismatch: {item["path"]}')
        # Refuse every conflicting immutable destination before publishing any
        # completion marker or replacing mutable logs/registration records.
        for item in manifest['entries']:
            if item['role'] != 'artifact': continue
            relative = safe_relative(item['path'])
            try: tail = relative.relative_to(REL_OUT)
            except ValueError: raise ValueError('Artifact outside registered output root')
            target = self.out/tail
            if target.is_symlink(): raise ValueError(f'Local artifact destination is a symlink: {target}')
            immutable = target.suffix in ('.npz', '.pt', '.py') or str(tail) == 'protocol.json' or tail.parts[0] == 'schedules'
            if immutable and target.exists() and sha(target) != item['sha256']:
                raise ValueError(f'Existing immutable local artifact differs; preserve incident: {target}')
        snapshot = self.base/'source_snapshots'/receipt_id
        verified_sources = []
        for identity in manifest['source_identities']:
            relative = safe_relative(identity['path'])
            if str(relative).startswith('upstream/'):
                source = Path(os.environ.get('FOURDSR_UPSTREAM', '/home/cai_tianshun/Project/4dgs'))/Path(*relative.parts[1:])
                if not source.is_file() or sha(source) != identity['sha256']:
                    raise ValueError(f'Local bound upstream bytes differ: {relative}')
                target = snapshot/relative; target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(source, target)
                verified_sources.append(dict(identity, byte_source='identical_SHA_local_upstream_copy'))
            else:
                source = ROOT/relative
                if not source.is_file() or sha(source) != identity['sha256']:
                    raise ValueError(f'Local execution source differs from trained source: {relative}')
                received = stage/'files'/relative
                if sha(received) != identity['sha256']: raise ValueError(f'Source snapshot mismatch: {relative}')
                target = snapshot/relative; target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(received, target)
                verified_sources.append(dict(identity, byte_source='remote_source_snapshot'))
        published = []
        # Publish completion markers only after all underlying artifacts.
        records = sorted(manifest['entries'], key=lambda item: (Path(item['path']).name == 'complete.json', item['path']))
        for item in records:
            if item['role'] != 'artifact': continue
            relative = safe_relative(item['path'])
            try: tail = relative.relative_to(REL_OUT)
            except ValueError: raise ValueError('Artifact outside registered output root')
            source, target = stage/'files'/relative, self.out/tail
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and sha(target) != item['sha256']:
                immutable = target.suffix in ('.npz', '.pt', '.py') or str(tail) == 'protocol.json' or tail.parts[0] == 'schedules'
                if immutable: raise ValueError(f'Existing immutable local artifact differs; preserve incident: {target}')
                archived = self.base/'replaced_mutable_archive'/receipt_id/tail
                archived.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(target, archived)
            if not target.exists() or sha(target) != item['sha256']:
                # Staging and final output share a filesystem. Promote verified
                # bytes atomically without retaining a second 30+ GB cache copy.
                try: source.replace(target)
                except OSError as error:
                    if error.errno != errno.EXDEV: raise
                    temporary = target.with_name(target.name+'.completion.tmp')
                    shutil.copy2(source, temporary); temporary.replace(target); source.unlink()
            elif target.suffix in ('.npz', '.pt'):
                source.unlink() # This private verified staging duplicate only.
            if sha(target) != item['sha256']: raise ValueError(f'Published SHA mismatch: {target}')
            published.append(dict(path=str(target), sha256=item['sha256'], bytes=item['bytes']))
        receipt = dict(status='completed_received_all_file_SHA_verified', purpose=purpose, task=task,
            host=self.host, remote_workspace=self.workspace, manifest=entry(stage/'manifest.json'),
            verified_files=published, source_snapshots=verified_sources, source_snapshot_root=str(snapshot),
            immutable_local_sources_not_overwritten=True, terminal=terminal, formal_updates=0, GPU_calls=0,
            wall_seconds=time.monotonic()-started, scope='CPU SHA verification and network/disk transfer; no new model forwards',
            successful_large_artifacts_promoted_without_duplicate_staging_storage=True)
        write(self.base/'transfers'/(receipt_id+'.json'), receipt)
        return receipt

    def request(self, task):
        events = list(self.stream('queue-request', valid_task(task)))
        if len(events) != 1 or events[0].get('kind') != 'request': raise ValueError('Missing remote queue request')
        return events[0]['request']


def backoff(attempt):
    # Network outages are retried by a CPU service, not model/GPU checks.
    time.sleep(min(600, 30*2**min(attempt, 5)))


def validate_preparation(terminal, out=OUT):
    if not terminal.get('successful') or not terminal.get('exit_proved') or terminal.get('exit_code') != 0 or terminal.get('exit_kind') != 1:
        raise ValueError('Preparation service lacks exact normal Exit0 proof')
    path = Path(out)/'preparation/complete.json'; value = read(path)
    if value.get('status') != 'completed_native_support_single_calibration' or value.get('formal_updates') != 0:
        raise ValueError('Preparation artifact status is incomplete/failed')
    for key in ('protocol', 'fixture', 'parent', 'support', 'calibration'): bound(value[key])
    if read(bound(value['calibration'])).get('status') != 'passed': raise ValueError('Calibration did not pass')
    return entry(path)


def prepare_return(args, transport=None):
    transport = transport or Transport(args.host, args.workspace, args.out, args.remote_python, args.remote_activate)
    binding = hashlib.sha256((args.host+'\0'+str(args.workspace)+'\0'+args.unit).encode()).hexdigest()[:16]
    statepath = Path(args.out)/'completion/bindings'/('prepare_'+binding+'.json'); attempt = 0; started = time.monotonic()
    expected = read(statepath).get('invocation_id', '') if statepath.exists() else ''
    while True:
        try:
            terminal = None
            for event in transport.stream('wait-unit', args.unit, expected, 'system' if args.system_scope else 'user'):
                if event.get('kind') == 'bound':
                    invocation = event['invocation_id']
                    if expected and invocation != expected: raise ValueError('Remote unit InvocationID changed')
                    expected = invocation
                    write(statepath, dict(status='bound_waiting_remote_pidfd_exit', unit=args.unit, invocation_id=expected,
                        host=args.host, remote_workspace=args.workspace, event=event, formal_updates=0, GPU_calls=0))
                elif event.get('kind') == 'terminal': terminal = event
            if not terminal: raise ValueError('SSH helper returned without a durable terminal event')
            if expected and terminal['invocation_id'] != expected: raise ValueError('Terminal event belongs to another unit invocation')
            transfer = transport.bundle('prepare', terminal=terminal) # Return failure logs/assets too.
            receipt = dict(status='failed_preparation_returned_artifacts', terminal=terminal, transfer=transfer,
                source=entry(Path(__file__)), formal_updates=0, GPU_calls=0, wall_seconds=time.monotonic()-started,
                observer_wall_includes_blocked_training_wait=True, observer_wall_is_not_GPU_compute_cost=True)
            try:
                receipt['preparation_complete'] = validate_preparation(terminal, args.out)
                receipt['status'] = 'completed_preparation_return_Exit0_and_all_SHA_verified'
            except Exception as error: receipt['error'] = repr(error)
            write(Path(args.out)/'completion/prepare_return_complete.json', receipt)
            return receipt['status'].startswith('completed_')
        except ConnectionError as error:
            attempt += 1
            write(Path(args.out)/'completion/prepare_return_connection.json', dict(status='network_disconnected_backoff_pending',
                attempts=attempt, error=repr(error), invocation_id=expected, formal_updates=0, GPU_calls=0))
            backoff(attempt-1)


class EvaluationConsumer:
    """Sequential durable evaluator; only registered queue events invoke it."""
    def __init__(self, args, transport, evaluate=None):
        self.args, self.transport, self.out = args, transport, Path(args.out)
        self.evaluate = evaluate or self._evaluate
        self.receipts = self.out/'completion/evaluation_receipts'; self.receipts.mkdir(parents=True, exist_ok=True)
        self.seen_failures = set()

    def _evaluate(self, task):
        command = [sys.executable, '-u', str(HERE/'run_suite.py'), '--phase', 'evaluate', '--task', task,
            '--gpu', self.args.gpu, '--python', self.args.python, '--operator-resource-resolved']
        logpath = self.out/'completion'/('evaluate_'+task+'.log')
        with logpath.open('a') as log: result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        return result.returncode, command, str(logpath)

    def consume(self, event):
        task = valid_task(Path(event['name']).stem); started = time.monotonic()
        receiptpath = self.receipts/(task+'.json')
        if event['origin'] == 'remote':
            # A duplicate remote event does not copy a multi-GB checkpoint again.
            # Its actual immutable request is checked before the idempotent return.
            if receiptpath.exists() and read(receiptpath).get('status') == 'completed_uniform_evaluation_event_callback':
                old, remote = read(receiptpath), self.transport.request(task)
                if remote.get('task') != task or remote.get('checkpoint') != old['checkpoint']:
                    raise ValueError('Remote duplicate changed its endpoint checkpoint')
                for identity in old['results'].values(): bound(identity)
                return old
            self.transport.bundle('task', task)
        requestpath = self.out/'evaluation_queue'/(task+'.json'); request = read(requestpath)
        if request.get('task') != task or request.get('status') != 'pending_uniform_evaluation':
            raise ValueError('Wrong or incomplete queue request')
        checkpoint = bound(request['checkpoint']); checkpoint_sha = sha(checkpoint)
        if checkpoint != self.out/'runs'/task/'checkpoint_12000.pt': raise ValueError('Queue checkpoint outside its registered endpoint')
        if receiptpath.exists():
            old = read(receiptpath)
            if old['checkpoint_sha256'] != checkpoint_sha: raise ValueError('Previously evaluated checkpoint changed')
            if old['status'] == 'completed_uniform_evaluation_event_callback':
                for identity in old['results'].values(): bound(identity)
                return old
        failed_key = (task, checkpoint_sha)
        if failed_key in self.seen_failures: return dict(status='failed_task_duplicate_event_ignored', task=task)
        # Queue content is immutable; local/remote duplicates cannot make GPU
        # work parallel. The outer consumer service owns its serial flock.
        code, command, log = self.evaluate(task)
        result = dict(status='failed_uniform_evaluation_event_callback', task=task, checkpoint=request['checkpoint'],
            checkpoint_sha256=checkpoint_sha, trigger=event, command=command, log=log, evaluator_exit_code=code,
            formal_updates=0, Adam_calls=0, elapsed_seconds=time.monotonic()-started,
            GPU=self.args.gpu, source=entry(Path(__file__)))
        if code == 0:
            try:
                task_receipt = self.out/'tasks'/(task+'.json'); value = read(task_receipt)
                if value.get('status') != 'completed_training_evaluation_fixed_diagnostics' or value['checkpoint'] != request['checkpoint']:
                    raise ValueError('Evaluator Exit0 without matching training/evaluation/diagnostic receipt')
                for key in ('training', 'checkpoint', 'evaluation', 'diagnostics'): bound(value[key])
                result['status'] = 'completed_uniform_evaluation_event_callback'
                result['results'] = {key: value[key] for key in ('training', 'checkpoint', 'evaluation', 'diagnostics')}
                result['task_receipt'] = entry(task_receipt)
            except Exception as error:
                result['error'] = repr(error); self.seen_failures.add(failed_key)
        else:
            self.seen_failures.add(failed_key)
        write(receiptpath, result)
        return result


def evaluation_service(args, transport=None):
    if not args.operator_resource_resolved or not args.gpu or not args.gpu.startswith('GPU-'):
        raise ValueError('Root-resolved physical evaluation GPU UUID is required')
    if Path(args.out).resolve() != OUT.resolve():
        raise ValueError('Registered run_suite evaluation must use the fixed experiment OUT')
    out = Path(args.out); (out/'locks').mkdir(parents=True, exist_ok=True)
    # The run_suite child holds the controller GPU lock itself. This distinct
    # consumer lock serializes multiple callbacks without deadlocking that child.
    with (out/'locks'/('uniform_completion_'+args.gpu+'.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        transport = transport or Transport(args.host, args.workspace, out, args.remote_python, args.remote_activate)
        messages = queue.Queue(); stopping = threading.Event()
        listener = DirectoryEvents(out/'evaluation_queue', out/'completion/evaluation_events.fifo')
        preparation_events = DirectoryEvents(out/'preparation')
        consumer = EvaluationConsumer(args, transport)
        def local_events():
            try:
                for path in sorted((out/'evaluation_queue').glob('*.json')):
                    messages.put(dict(origin='local', name=path.name, mechanism='startup_recovery'))
                while not stopping.is_set():
                    for event in listener.next(): messages.put(dict(origin='local', **event))
            except BaseException:
                if not stopping.is_set(): messages.put(dict(origin='local_stream_error', traceback=traceback.format_exc()))
        def remote_events():
            attempt = 0
            while not stopping.is_set():
                try:
                    for event in transport.stream('events'):
                        if event.get('kind') == 'queue': messages.put(dict(origin='remote', **event))
                        elif event.get('kind') == 'ready': attempt = 0
                    raise ConnectionError('Remote event stream ended')
                except ConnectionError as error:
                    if stopping.is_set(): return
                    write(out/'completion/remote_events_connection.json', dict(status='network_disconnected_backoff_pending',
                        attempts=attempt+1, error=repr(error), formal_updates=0, GPU_calls=0))
                    backoff(attempt); attempt += 1
                except BaseException:
                    if not stopping.is_set(): messages.put(dict(origin='remote_stream_error', traceback=traceback.format_exc()))
                    return
        threading.Thread(target=local_events, daemon=True).start()
        threading.Thread(target=remote_events, daemon=True).start()
        registration = dict(status='registered_durable_uniform_evaluation_callback',
            pid=os.getpid(), source=entry(Path(__file__)), source_sha256=sha(Path(__file__)), tasks=list(TASKS), GPU=args.gpu, host=socket.gethostname(),
            remote_host=args.host, remote_workspace=args.workspace, model_polling=False,
            startup_recovery_scan_only=True, blocking_stage='waiting_preparation_complete_inotify',
            local_queue=str(out/'evaluation_queue'), FIFO=str(out/'completion/evaluation_events.fifo'),
            registered_unix=time.time(), formal_updates=0)
        stat = Path('/proc/self/stat').read_text()
        registration['pid_starttime'] = stat[stat.rfind(')')+2:].split()[19]
        write(out/'completion/registration.json', registration)
        wait_for_preparation(out, preparation_events)
        preparation_events.close()
        registration['blocking_stage'] = 'waiting_queue_events'; write(out/'completion/registration.json', registration)
        while True:
            event = messages.get() # Infinite blocking event queue, no sleep/model polling.
            if event['origin'] in ('local_stream_error', 'remote_stream_error'): raise RuntimeError(event['traceback'])
            try:
                registration['blocking_stage'] = 'receiving_or_evaluating_'+Path(event['name']).stem
                write(out/'completion/registration.json', registration)
                result = consumer.consume(event)
                write(out/'completion/evaluation_last_event.json', dict(status='event_consumed', event=event, result=result['status']))
            except ConnectionError as error:
                # Retry this failed transfer through the same serialized queue.
                write(out/'completion/evaluation_transfer_connection.json', dict(status='network_disconnected_backoff_pending',
                    event=event, error=repr(error), formal_updates=0, GPU_calls=0))
                backoff(0); messages.put(event)
            except Exception as error:
                write(out/'completion'/('event_failure_'+str(time.time_ns())+'.json'), dict(status='failed_queue_event_preserved',
                    event=event, error=repr(error), traceback=traceback.format_exc(), formal_updates=0))
            registration['blocking_stage'] = 'waiting_queue_events'; write(out/'completion/registration.json', registration)
            complete = [task for task in TASKS if (consumer.receipts/(task+'.json')).exists()
                        and read(consumer.receipts/(task+'.json')).get('status') == 'completed_uniform_evaluation_event_callback']
            if len(complete) == 12:
                write(out/'completion/uniform_evaluation_complete.json', dict(status='completed_all_12_uniform_evaluation_callbacks',
                    tasks={task: entry(consumer.receipts/(task+'.json')) for task in complete},
                    GPU=args.gpu, formal_updates=0, source=entry(Path(__file__))))
                registration['status'] = 'completed_durable_uniform_evaluation_callback'
                registration['blocking_stage'] = 'completed_all_12'
                write(out/'completion/registration.json', registration)
                stopping.set(); listener.close(); transport.close(); return


def wait_for_preparation(out, listener):
    """Subscribe before the startup check; then wait only for complete rename."""
    path = Path(out)/'preparation/complete.json'
    def completed():
        if not path.exists(): return False
        value = read(path)
        if value.get('status') != 'completed_native_support_single_calibration' or value.get('formal_updates') != 0:
            raise ValueError('Existing preparation marker is not completed registered preparation')
        for key in ('protocol', 'fixture', 'parent', 'support', 'calibration'): bound(value[key])
        return True
    if completed(): return
    while True:
        for event in listener.next():
            if event['name'] == 'complete.json' and completed(): return


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('mode', choices=('prepare-return', 'uniform-evaluation'))
    result.add_argument('--host', default='a100-train'); result.add_argument('--workspace', default=DEFAULT_WORKSPACE)
    result.add_argument('--unit', default=DEFAULT_UNIT); result.add_argument('--system-scope', action='store_true')
    result.add_argument('--out', type=Path, default=OUT); result.add_argument('--remote-python', default='/usr/bin/python3')
    result.add_argument('--remote-activate'); result.add_argument('--gpu'); result.add_argument('--python', default=TRAINPY)
    result.add_argument('--operator-resource-resolved', action='store_true')
    return result


def main():
    args = parser().parse_args()
    try:
        if args.mode == 'prepare-return': return 0 if prepare_return(args) else 1
        evaluation_service(args); return 0
    except BaseException as error:
        write(Path(args.out)/'completion'/('callback_failure_'+str(time.time_ns())+'.json'),
            dict(status='failed_callback_preserved_not_completed', mode=args.mode, error=repr(error),
                traceback=traceback.format_exc(), source=entry(Path(__file__)), formal_updates=0))
        raise


if __name__ == '__main__': sys.exit(main())
