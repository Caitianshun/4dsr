"""Root-registered event continuation; imports CPU code only.

Writing this module never registers a service. Its main entry is a root-started
CPU service, and only the two fixed paired workers are permitted dispatches.
"""
from __future__ import annotations
import argparse, csv, fcntl, hashlib, json, os, queue, re, shlex, socket, subprocess, sys, threading, time, traceback
from pathlib import Path
import completion
from fp_common import ROOT, HERE, OUT, read, write, entry, sha, bound, local

UNITS={'1':'4dsr-footprint-r1-paired-pro6000-20261007.service','2':'4dsr-footprint-r2-paired-a100-gpu1-20261007.service'}
WORKSPACE=completion.DEFAULT_WORKSPACE
TASKS=completion.TASKS


def unit_name(value):
    if not re.fullmatch(r'[A-Za-z0-9_.@:-]+',value):raise ValueError('Unsafe unit name')
    return value if value.endswith('.service') else value+'.service'


def process_starttime(pid):
    value=Path(f'/proc/{int(pid)}/stat').read_text();return value[value.rfind(')')+2:].split()[19]


def terminal_ok(value,spec,success=True):
    if value.get('kind')!='terminal' or unit_name(value['unit'])!=unit_name(spec['name']) or value.get('invocation_id')!=spec['invocation_id']:
        raise ValueError('Wrong exact unit/invocation terminal')
    systemd=value.get('systemd',{})
    if not value.get('exit_proved') or int(value.get('exit_kind',0))==0 or systemd.get('LoadState')=='not-found':raise ValueError('Unknown/GC exit is not proof GPU process exited')
    if systemd.get('MainPID')!='0' or int(systemd.get('ExecMainExitTimestampMonotonic','0'))<=0:raise ValueError('Live process/missing true exit timestamp')
    if int(systemd.get('ExecMainCode','0'))!=int(value['exit_kind']) or int(systemd.get('ExecMainStatus','-1'))!=int(value['exit_code']):raise ValueError('Terminal and retained systemd exit disagree')
    if systemd.get('InvocationID') not in (spec['invocation_id'],'') or systemd.get('InvocationID')=='' and not value.get('retained_ExecMainPID_and_start_match'):raise ValueError('Terminal retained invocation does not match')
    if spec.get('main_pid') and value.get('systemd',{}).get('ExecMainPID'):
        if int(value['systemd']['ExecMainPID'])!=int(spec['main_pid']):raise ValueError('Retained execution PID changed')
    if success and not (value.get('successful') and value.get('exit_kind')==1 and value.get('exit_code')==0):raise ValueError('Normal exact Exit0 required')
    return value


def validate_config(config):
    if config.get('schema')!=1 or config['workspace']!=WORKSPACE or config['remote_host']!='a100-train':raise ValueError('Unregistered paired workspace/host')
    for label,model in [('r1','PRO'),('r2','A100'),('evaluation','3090')]:
        value=config['hardware'][label]
        if not value['GPU'].startswith('GPU-') or model not in value['model']:raise ValueError('Fixed GPU model/UUID missing')
    if config['hardware']['r1']['GPU']==config['hardware']['evaluation']['GPU']:raise ValueError('Training/evaluation GPU overlap')
    for label in ('prepare_return','overlap','teacher','evaluation'):
        spec=config['units'][label];unit_name(spec['name'])
        if not re.fullmatch('[0-9a-f]{32}',spec['invocation_id']):raise ValueError('True systemd InvocationID required')
        bound(spec['source'])
    for spec in config.get('prerequisite_units',[]):
        unit_name(spec['name']);bound(spec['source'])
        if not re.fullmatch('[0-9a-f]{32}',spec['invocation_id']):raise ValueError('Prerequisite lacks invocation')
    identities={e['path']:e for e in config['sources']}
    for name in ('advance_after_preparation.py','completion.py','run_suite.py','summarize.py'):
        if str((HERE/name).relative_to(ROOT)) not in identities:raise ValueError('Required callback/core source not registered')
    for identity in identities.values():bound(identity)
    if config['hardware']['r1']['host']!=socket.gethostname() or config['hardware']['evaluation']['host']!=socket.gethostname():raise ValueError('Local hardware host changed')
    return config


def validate_preparation_assets(out=OUT):
    receipt=read(out/'completion/prepare_return_complete.json')
    if receipt.get('status')!='completed_preparation_return_Exit0_and_all_SHA_verified':raise ValueError('Preparation return did not succeed')
    remote=receipt['terminal']
    if not remote.get('successful') or not remote.get('exit_proved') or remote['exit_kind']!=1 or remote['exit_code']!=0:raise ValueError('Remote preparation normal exit not proved')
    transfer=receipt['transfer']
    if transfer.get('status')!='completed_received_all_file_SHA_verified':raise ValueError('Transfer status mismatch')
    if transfer.get('remote_workspace',WORKSPACE)!=WORKSPACE:raise ValueError('Transfer workspace mismatch')
    manifest=read(bound(transfer['manifest']))
    if manifest['workspace']!=WORKSPACE or manifest['purpose']!='prepare':raise ValueError('Returned different preparation bundle')
    expected={}
    for item in manifest['entries']:
        if item['role']!='artifact':continue
        relative=completion.safe_relative(item['path'])
        try:tail=relative.relative_to(completion.REL_OUT)
        except ValueError:raise ValueError('Preparation artifact outside registered output root')
        expected[str(out/tail)]=(item['bytes'],item['sha256'])
    actual={value['path']:(value['bytes'],value['sha256']) for value in transfer['verified_files']}
    if not expected or expected!=actual or len(actual)!=len(transfer['verified_files']):raise ValueError('Not every declared preparation artifact was transferred exactly once')
    for value in transfer['verified_files']:
        path=Path(value['path'])
        if path.is_symlink() or not path.resolve().is_relative_to(out.resolve()) or path.stat().st_size!=value['bytes'] or sha(path)!=value['sha256']:raise ValueError('Transferred preparation artifact changed')
    prep_path=bound(receipt['preparation_complete'])
    if prep_path!=out/'preparation/complete.json':raise ValueError('Preparation marker path changed')
    prep=read(prep_path)
    if prep.get('status')!='completed_native_support_single_calibration' or prep.get('formal_updates')!=0:raise ValueError('Preparation marker incomplete')
    for key in ('protocol','fixture','parent','support','calibration'):bound(prep[key])
    if read(bound(prep['calibration'])).get('status')!='passed':raise ValueError('Calibration failed')
    for relative,expected in prep['sources'].items():
        path=Path(os.environ.get('FOURDSR_UPSTREAM','/home/cai_tianshun/Project/4dgs'))/relative[len('upstream/'):] if relative.startswith('upstream/') else ROOT/relative
        if sha(path)!=expected:raise ValueError('Bound preparation/core source changed')
    return dict(return_receipt=entry(out/'completion/prepare_return_complete.json'),preparation=entry(out/'preparation/complete.json'),verified_asset_count=len(transfer['verified_files']))


def validate_overlap(out=OUT,spec=None):
    # The old transient overlap manager was GC'd. Only the successful child
    # return/closed checkpoints are proved; LoadState=not-found proves no exit.
    path=out/'completion/B0_curve_exit_acceptance.json';receipt=read(path)
    if spec and spec.get('exit_receipt') and bound(spec['exit_receipt'])!=path:raise ValueError('Different overlap acceptance receipt')
    if receipt.get('status')!='accepted_registered_checkpoint_with_successful_training_child_return_manager_exit_unknown':raise ValueError('Overlap manager Exit0 is not proved')
    if receipt.get('successful_training_child_exit_code')!=0 or receipt.get('formal_suffix_updates')!=3000 or receipt.get('parent_manager_exit_code')!='unknown_GC_not_found_is_not_Exit0_proof':raise ValueError('Child closure/unknown manager exit must be explicit')
    if bound(receipt['source'])!=HERE/'overlap_baseline.py' or spec and receipt['source']!=spec['source']:raise ValueError('Overlap source binding changed')
    status_path=bound(receipt['post_return_status'])
    if status_path!=out/'workers/B0_preparation_overlap.json':raise ValueError('Different overlap post-return receipt')
    status=read(status_path)
    if status.get('status')!='completed_registered_B0_curve_checkpoint' or status['formal_suffix_updates']!=3000 or status['added_updates']!=2900:raise ValueError('B0 overlap did not finish registered3000')
    segment=read(bound(status['segment']));checkpoint=bound(status['checkpoint'])
    initial_path=out/'runs/r1_B0/attempt_complete_0000_0100.json';initial=read(initial_path);bound(initial['checkpoint'])
    for value,start,stop,updates in ((initial,0,100,100),(segment,100,3000,2900)):
        if value['status']!='completed_formal_segment' or value['method']!='B0' or value['repeat']!='1' or value['start']!=start or value['suffix_endpoint']!=stop or value['updates']!=updates:raise ValueError('Closed B0 segment counters wrong')
        if value['training_rgb_forwards']!=3*updates or value['adam_calls']!=2*updates or value['moment_forwards']!=0 or not value['topology_unchanged']:raise ValueError('Closed B0 segment work identity wrong')
        if value['source_identity']!=initial['source_identity'] or value['physical_gpu']!=initial['physical_gpu']:raise ValueError('B0 hardware/source changed across segments')
    for relative,expected in initial['source_identity'].items():
        source=Path(os.environ.get('FOURDSR_UPSTREAM','/home/cai_tianshun/Project/4dgs'))/relative[len('upstream/'):] if relative.startswith('upstream/') else ROOT/relative
        if sha(source)!=expected:raise ValueError('B0 training source changed')
    if receipt['segment']!=status['segment'] or checkpoint!=out/'runs/r1_B0/checkpoint_9000.pt' or segment['checkpoint']!=status['checkpoint']:raise ValueError('B0 checkpoint identity wrong')
    side=read(checkpoint.with_suffix('.json'))
    if side['metadata']['suffix_step']!=3000:raise ValueError('B0 retained checkpoint cursor wrong')
    return dict(status='accepted_successful_training_child_return_closed3000_manager_exit_unknown',exit_receipt=entry(path),post_return_status=receipt['post_return_status'],initial_segment=entry(initial_path),segment=status['segment'],checkpoint=status['checkpoint'],successful_training_child_exit_code=0,manager_exit_proved=False,parent_manager_exit_code='unknown',GPU_process_release_requires_prelaunch_metadata=True)


def validate_evaluator_command(words,config,registration,out=OUT):
    source=bound(config['units']['evaluation']['source'])
    positions=[i for i,word in enumerate(words) if word==str(source)]
    if len(positions)!=1:raise ValueError('Uniform evaluator exact executable source missing')
    # The bound frozen parser is authoritative for omitted default flags.
    # Explicit values are parsed and must match that same registered workspace.
    try:args=completion.parser().parse_args(words[positions[0]+1:])
    except SystemExit as error:raise ValueError('Uniform evaluator CLI could not be parsed') from error
    if args.mode!='uniform-evaluation' or args.gpu!=config['hardware']['evaluation']['GPU']:raise ValueError('Uniform evaluator mode/GPU differs')
    if args.workspace!=WORKSPACE or args.workspace!=registration['remote_workspace'] or args.host!=config['remote_host'] or args.host!=registration['remote_host']:raise ValueError('Uniform evaluator explicit/default host/workspace differs')
    if args.out.resolve()!=Path(out).resolve():raise ValueError('Uniform evaluator output directory differs')
    return dict(workspace=args.workspace,host=args.host,GPU=args.gpu,default_flags_from_bound_source=True)


def validate_evaluator(config,properties,out=OUT):
    registration=read(out/'completion/registration.json');spec=config['units']['evaluation'];hardware=config['hardware']['evaluation']
    if registration['status'] not in ('registered_durable_uniform_evaluation_callback','completed_durable_uniform_evaluation_callback'):raise ValueError('Uniform callback unregistered')
    if registration['source']!=spec['source'] or registration['source_sha256']!=spec['source']['sha256']:raise ValueError('Uniform evaluator source changed')
    bound(registration['source'])
    if registration['GPU']!=hardware['GPU'] or registration['host']!=hardware['host'] or registration['remote_host']!='a100-train' or registration['remote_workspace']!=WORKSPACE:raise ValueError('Uniform hardware/workspace registration changed')
    if len(registration['tasks'])!=12 or set(registration['tasks'])!=set(TASKS):raise ValueError('Uniform evaluator task set wrong')
    if properties.get('InvocationID') not in (spec['invocation_id'],''):raise ValueError('Uniform evaluator invocation changed')
    if properties.get('MainPID','0')!='0':
        pid=registration['pid']
        if int(properties['MainPID'])!=pid or spec.get('main_pid') and spec['main_pid']!=pid:raise ValueError('Uniform evaluator real PID mismatch')
        if process_starttime(pid)!=str(registration['pid_starttime']):raise ValueError('Uniform evaluator PID reused')
        command=Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0');words=[x.decode() for x in command if x]
        validate_evaluator_command(words,config,registration,out)
    else:
        if registration['status']!='completed_durable_uniform_evaluation_callback' or not (out/'completion/uniform_evaluation_complete.json').exists():raise ValueError('Uniform evaluator not truly alive or completed')
    return entry(out/'completion/registration.json')


GPU_QUERIES=(['nvidia-smi','--query-gpu=uuid,name,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],
             ['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv,noheader,nounits'])


def parse_idle_gpu(gpu,outputs):
    rows=[[x.strip() for x in r] for r in csv.reader(outputs[0].splitlines(),skipinitialspace=True)];matches=[r for r in rows if r and r[0]==gpu['GPU']]
    if len(matches)!=1 or matches[0][1]!=gpu['model']:raise ValueError('Physical GPU identity changed before dispatch')
    apps=[[x.strip() for x in r] for r in csv.reader(outputs[1].splitlines(),skipinitialspace=True) if r and r[0].strip()==gpu['GPU']]
    foreign=[r for r in apps if len(r)<3 or '/opt/todesk/' not in r[2]]
    if foreign:raise ValueError('Prelaunch GPU has foreign compute; preserve other processes: '+repr(foreign))
    return dict(status='prelaunch_registered_GPU_no_foreign_compute',GPU=gpu,hardware_row=matches[0],compute_rows=apps,checked_unix=time.time(),metadata_only=True,new_model_forwards=0,foreign_processes_signaled=False)


def props_command(name):return ['systemctl','--user','show',unit_name(name),'--no-pager']


class LocalBackend:
    def __init__(self,config):self.config=config;self.children=[]
    def properties(self,name):
        result=subprocess.run(props_command(name),capture_output=True,text=True)
        values=dict(line.split('=',1) for line in result.stdout.splitlines() if '=' in line)
        if result.returncode and values.get('LoadState')!='not-found':raise RuntimeError(result.stderr)
        return values
    def idle_gpu(self,gpu):
        outputs=[]
        for command in GPU_QUERIES:
            result=subprocess.run(command,capture_output=True,text=True)
            if result.returncode:raise RuntimeError('Prelaunch GPU metadata read failed: '+result.stderr)
            outputs.append(result.stdout)
        return parse_idle_gpu(gpu,outputs)
    def wait_unit(self,spec):
        if spec.get('terminal') is not None:
            receipt=read(bound(spec['exit_receipt']))
            if receipt.get('terminal')!=spec['terminal']:raise ValueError('Embedded exact terminal differs from bound receipt')
            return terminal_ok(spec['terminal'],spec,False)
        command=[self.config['local_cpu_python'],'-u','-c',completion.REMOTE_HELPER,'wait-unit',str(ROOT),completion.REL_OUT,unit_name(spec['name']),spec['invocation_id'],'user']
        child=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True);self.children.append(child);terminal=None
        for line in child.stdout:
            event=json.loads(line)
            if event.get('kind')=='terminal':terminal=event
        error=child.stderr.read();code=child.wait()
        if code or terminal is None:raise RuntimeError('Local exact observer failed: '+error)
        return terminal_ok(terminal,spec,False)
    def dispatch(self,command):
        result=subprocess.run(command,capture_output=True,text=True)
        if result.returncode:raise RuntimeError('Worker dispatch failed: '+result.stderr)
        return dict(stdout=result.stdout,stderr=result.stderr,returncode=result.returncode)


class A100Transport(completion.Transport):
    def properties(self,name):
        result=subprocess.run(self.ssh+[shlex.join(props_command(name))],capture_output=True,text=True)
        if result.returncode==255:raise ConnectionError(result.stderr)
        values=dict(line.split('=',1) for line in result.stdout.splitlines() if '=' in line)
        if result.returncode and values.get('LoadState')!='not-found':raise RuntimeError(result.stderr)
        return values
    def idle_gpu(self,gpu):
        outputs=[]
        for command in GPU_QUERIES:
            result=subprocess.run(self.ssh+[shlex.join(command)],capture_output=True,text=True)
            if result.returncode==255:raise ConnectionError(result.stderr)
            if result.returncode:raise RuntimeError('Remote prelaunch GPU metadata read failed: '+result.stderr)
            outputs.append(result.stdout)
        return parse_idle_gpu(gpu,outputs)
    def wait_unit(self,spec):
        if spec.get('terminal') is not None:raise ValueError('Remote worker observer requires actual helper event')

        terminal=None
        for event in self.stream('wait-unit',unit_name(spec['name']),spec['invocation_id'],'user'):
            if event.get('kind')=='terminal':terminal=event
        if terminal is None:raise ValueError('Remote observer returned without exact terminal')
        return terminal_ok(terminal,spec,False)
    def dispatch(self,command):
        result=subprocess.run(self.ssh+[shlex.join(command)],capture_output=True,text=True)
        if result.returncode==255:raise ConnectionError(result.stderr)
        if result.returncode:raise RuntimeError('Remote dispatch failed: '+result.stderr)
        return dict(stdout=result.stdout,stderr=result.stderr,returncode=result.returncode)
    def current_task(self):
        script='import json,pathlib,sys; p=pathlib.Path(sys.argv[1])/sys.argv[2]; print(p.read_text() if p.exists() else "{}")'
        command=self.ssh+[shlex.join([self.remote_python,'-c',script,self.workspace,completion.REL_OUT+'/workers/worker_r2.json'])]
        result=subprocess.run(command,capture_output=True,text=True)
        if result.returncode==255:raise ConnectionError(result.stderr)
        if result.returncode:raise RuntimeError(result.stderr)
        value=json.loads(result.stdout);task=value.get('task')
        if task is not None and (task not in TASKS or not task.startswith('r2_')):raise ValueError('Unexpected remote failed task')
        return task,value


def worker_command(config,repeat,owner):
    remote=repeat=='2';workspace=config['workspace'] if remote else str(ROOT);cpu=config['remote_cpu_python'] if remote else config['local_cpu_python'];train=config['remote_train_python'] if remote else config['local_train_python'];gpu=config['hardware']['r'+repeat]['GPU']
    log=Path(workspace)/completion.REL_OUT/'logs'/('paired_worker_r'+repeat+'_controller.log')
    command=[cpu,'-u',str(Path(workspace)/str((HERE/'run_suite.py').relative_to(ROOT))),'--phase','worker','--repeat',repeat,'--gpu',gpu,'--defer-evaluation','--operator-resource-resolved','--python',train]
    shell='cd '+shlex.quote(workspace)+' && exec '+shlex.join(command)+' > '+shlex.quote(str(log))+' 2>&1'
    if remote:shell='source '+shlex.quote(config['remote_activate'])+' && '+shell
    return ['systemd-run','--user','--unit='+UNITS[repeat],'--property=Type=exec','--property=RemainAfterExit=yes','--property=Nice=5',
      '--property=WorkingDirectory='+workspace,'--property=Description=4dsr-footprint paired owner='+owner+' repeat='+repeat,
      '--setenv=FOURDSR_ADVANCE_OWNER='+owner,'/bin/bash','-lc',shell]


class Advance:
    def __init__(self,config,registration,local_backend=None,remote=None,checks=True,finalizer=None,out=OUT):
        self.config=config;self.registration=registration;self.out=Path(out);self.base=self.out/'advance';self.base.mkdir(parents=True,exist_ok=True)
        self.local=local_backend or LocalBackend(config);self.remote=remote or A100Transport(config['remote_host'],config['workspace'],self.out,config['remote_cpu_python']);self.checks=checks;self.finalizer=finalizer or self.finalize;self.failure_stop=threading.Event();self.failure_listeners=[];self.failure_listener_lock=threading.Lock()
    def save(self,name,value):write(self.base/name,value);return value
    def wait(self,backend,spec,success=True):
        attempt=0
        while True:
            try:
                result=backend.wait_unit(spec);self.save(unit_name(spec['name'])+'.terminal.json',result);return terminal_ok(result,spec,success)
            except ConnectionError as error:
                self.save(unit_name(spec['name'])+'.network.json',dict(status='network_failure_saved_exact_invocation_retry',invocation_id=spec['invocation_id'],attempt=attempt,error=repr(error)));completion.backoff(attempt);attempt+=1
    def prerequisites(self):
        config=self.config;units=config['units'];proofs={}
        proofs['prepare_return']=self.wait(self.local,units['prepare_return'])
        if self.checks:
            proofs['preparation_assets']=validate_preparation_assets(self.out)
            proofs['overlap_assets']=validate_overlap(self.out,units['overlap'])
        else:proofs['overlap_assets']={'status':'CPU_fixture_successful_child_manager_unknown','manager_exit_proved':False}
        proofs['teacher']=self.wait(self.local,units['teacher'],False)
        if not proofs['teacher']['successful']:self.save('teacher_failure_independent_core_allowed.json',dict(status='teacher_failed_exact_GPU_exit_proved_core_independent',terminal=proofs['teacher']))
        for spec in config.get('prerequisite_units',[]):proofs[spec['name']]=self.wait(self.local,spec,spec.get('required_success',False))
        if self.checks:proofs['evaluation_registration']=validate_evaluator(config,self.local.properties(units['evaluation']['name']),self.out)
        return self.save('prerequisites_complete.json',dict(status='passed_preparation_exact_exit_B0_child_closure_resource_release_and_durable_evaluator',proofs=proofs,registration=self.registration,formal_updates=0))
    def dispatch(self,repeat):
        backend=self.local if repeat=='1' else self.remote;config=self.config;gpu=config['hardware']['r'+repeat]
        identity=dict(repeat=repeat,unit=UNITS[repeat],GPU=gpu,workspace=str(ROOT) if repeat=='1' else WORKSPACE,python=config['local_train_python'] if repeat=='1' else config['remote_train_python'],protocol=entry(self.out/'protocol.json') if self.checks else {'path':'fake_protocol','sha256':'f'*64},core_sources=read(self.out/'preparation/complete.json')['sources'] if self.checks else {})
        owner=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest();command=worker_command(config,repeat,owner);path=self.base/('dispatch_r'+repeat+'.json');intent=self.base/('dispatch_intent_r'+repeat+'.json')
        old=read(path) if path.exists() else None;properties=backend.properties(UNITS[repeat]);exists=properties.get('LoadState') not in ('not-found',None)
        if old:
            if old['identity']!=identity or old['owner']!=owner or not exists:raise ValueError('Owned worker disappeared/identity changed; no redispatch')
        if exists:
            if properties.get('Description')!='4dsr-footprint paired owner='+owner+' repeat='+repeat or 'FOURDSR_ADVANCE_OWNER='+owner not in shlex.split(properties.get('Environment','')):raise ValueError('Existing unit is not owned by this exact worker registration')
            if old and properties.get('InvocationID')!=old['invocation_id']:raise ValueError('Owned worker was restarted; refuse new invocation')
            dispatch=dict(status='attached_existing_owned_unit',command=command)
        else:
            if intent.exists():raise ValueError('Previous dispatch intent lacks owned unit; explicit incident resolution required')
            prelaunch=backend.idle_gpu(gpu);self.save('prelaunch_r'+repeat+'.json',prelaunch)
            self.save(intent.name,dict(status='intent_before_exact_single_dispatch',identity=identity,owner=owner,command=command,registration=self.registration))
            dispatch=backend.dispatch(command);properties=backend.properties(UNITS[repeat])
        invocation=properties.get('InvocationID');pid=int(properties.get('MainPID','0') or 0) or int(properties.get('ExecMainPID','0') or 0)
        if not invocation or not pid:raise ValueError('Dispatched worker has no real invocation/execution PID')
        if properties.get('Description')!='4dsr-footprint paired owner='+owner+' repeat='+repeat or 'FOURDSR_ADVANCE_OWNER='+owner not in shlex.split(properties.get('Environment','')):raise ValueError('Worker owner markers missing')
        if properties.get('Type')!='exec' or properties.get('RemainAfterExit')!='yes':raise ValueError('Persistent worker execution/retention properties changed')
        result=dict(status='registered_or_attached_persistent_paired_worker',name=UNITS[repeat],invocation_id=invocation,main_pid=pid,owner=owner,identity=identity,dispatch=dispatch,systemd=properties,registration=self.registration,source=entry(Path(__file__)),formal_updates=0)
        self.save(path.name,result);return result
    def observe_worker(self,repeat,spec,events):
        backend=self.local if repeat=='1' else self.remote
        try:
            terminal=self.wait(backend,spec,False);result=dict(repeat=repeat,status='worker_Exit0' if terminal['successful'] else 'failed_worker_saved',terminal=terminal)
            if repeat=='2' and not terminal['successful']:
                task,status=self.remote.current_task();result['remote_worker_status']=status
                if task:result['partial_return']=self.remote.bundle('task',task,terminal=terminal)
            self.save('worker_r'+repeat+'_exit.json',result);events.put(('worker',repeat,result))
        except BaseException as error:
            result=dict(repeat=repeat,status='failed_observer_or_partial_return_saved',error=repr(error),traceback=traceback.format_exc());self.save('worker_r'+repeat+'_exit.json',result);events.put(('worker',repeat,result))
    def observe_evaluation(self,events):
        try:terminal=self.wait(self.local,self.config['units']['evaluation'],False);value=dict(status='evaluation_Exit0' if terminal['successful'] else 'failed_evaluation_service_saved',terminal=terminal)
        except BaseException as error:value=dict(status='failed_evaluation_observer_saved',error=repr(error),traceback=traceback.format_exc())
        self.save('evaluation_unit_exit.json',value);events.put(('evaluation','',value))
    def close_failure_listener(self,listener):
        with self.failure_listener_lock:listener.close()
    def observe_evaluation_failures(self,directory,label,events,listener=None):
        # Subscribe before the initial recovery scan; no timed model polling.
        listener=listener or completion.DirectoryEvents(directory);self.failure_listeners.append(listener)
        def inspect(path):
            if not path.exists() or not path.name.endswith('.json'):return None
            if label=='queue' and not path.name.startswith('event_failure_'):return None
            value=read(path)
            if not str(value.get('status','')).startswith('failed_'):return None
            return dict(status='failed_uniform_evaluation_event_saved_service_exit_pending',failure_receipt=entry(path),failure=value,evaluation_service_exit_proved=False,other_owned_training_preserved=True)
        try:
            for path in sorted(Path(directory).glob('*.json')):
                value=inspect(path)
                if value:self.save('evaluation_failure_event.json',value);events.put(('evaluation_failure','',value));return
            while not self.failure_stop.is_set():
                for event in listener.next():
                    value=inspect(Path(directory)/event['name'])
                    if value:self.save('evaluation_failure_event.json',value);events.put(('evaluation_failure','',value));return
        except BaseException as error:
            if not self.failure_stop.is_set():
                value=dict(status='failed_evaluation_failure_observer_saved',error=repr(error),traceback=traceback.format_exc());self.save('evaluation_failure_observer_error.json',value);events.put(('evaluation_failure','',value))
        finally:self.close_failure_listener(listener)

    def finalize(self):
        complete=read(self.out/'completion/uniform_evaluation_complete.json')
        if complete.get('status')!='completed_all_12_uniform_evaluation_callbacks' or set(complete['tasks'])!=set(TASKS) or complete['GPU']!=self.config['hardware']['evaluation']['GPU']:raise ValueError('All12 uniform callbacks not completed')
        if complete.get('source')!=self.config['units']['evaluation']['source']:raise ValueError('Uniform completion source differs from registered evaluator')
        bound(complete['source'])
        for task,reference in complete['tasks'].items():
            if bound(reference)!=self.out/'completion/evaluation_receipts'/(task+'.json'):raise ValueError('Uniform callback path/task changed')
            value=read(bound(reference))
            if value.get('task')!=task or value.get('evaluator_exit_code')!=0 or value.get('source')!=complete['source']:raise ValueError('Uniform callback task/Exit0/source mismatch')
            if value.get('status')!='completed_uniform_evaluation_event_callback' or value['GPU']!=self.config['hardware']['evaluation']['GPU']:raise ValueError('Uniform endpoint callback failed')
            if set(value['results'])!=set(('training','checkpoint','evaluation','diagnostics')):raise ValueError('Uniform callback results incomplete')
            for item in value['results'].values():bound(item)
            task_receipt=read(bound(value['task_receipt']))
            if task_receipt.get('status')!='completed_training_evaluation_fixed_diagnostics' or any(task_receipt[key]!=value['results'][key] for key in value['results']):raise ValueError('Uniform callback and task artifact receipt disagree')
        commands=[[self.config['local_cpu_python'],str(HERE/'run_suite.py'),'--phase','verify'],[self.config['local_cpu_python'],str(HERE/'summarize.py')]]
        for number,command in enumerate(commands):
            with (self.base/('CPU_finalization_'+str(number)+'.log')).open('a') as log:process=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
            if process.returncode:raise ValueError('CPU verification/summary subprocess failed')
        integrity=read(self.out/'suite_integrity.json');summary=read(self.out/'summary/complete.json')
        if integrity.get('status')!='completed_core' or integrity['updates']!=72000 or len(integrity['completed'])!=12 or set(integrity['completed'])!=set(TASKS):raise ValueError('Formal core integrity incomplete')
        if summary.get('status')!='completed_core_quality_evidence' or summary['actual_identity_checked_endpoints']!=12:raise ValueError('Quality evidence summary incomplete')
        return dict(suite_integrity=entry(self.out/'suite_integrity.json'),summary=entry(self.out/'summary/complete.json'),no_decision_argument_passed=True)
    def run(self):
        try:return self._run()
        finally:
            self.failure_stop.set()
            for listener in self.failure_listeners:self.close_failure_listener(listener)
    def _run(self):
        self.save('state.json',dict(status='waiting_exact_preparation_and_resource_prerequisites',registration=self.registration,source=entry(Path(__file__)),formal_updates=0))
        self.prerequisites();events=queue.Queue();workers={};dispatch_failures={}
        if self.checks:
            for directory,label in ((self.out/'completion/evaluation_receipts','receipts'),(self.out/'completion','queue')):
                listener=completion.DirectoryEvents(directory)
                threading.Thread(target=self.observe_evaluation_failures,args=(directory,label,events,listener),daemon=True).start()
        threading.Thread(target=self.observe_evaluation,args=(events,),daemon=True).start()
        for repeat in ('1','2'):
            try:
                workers[repeat]=self.dispatch(repeat)
                # Bind each successfully dispatched worker immediately, even if
                # the independent second host later fails dispatch.
                threading.Thread(target=self.observe_worker,args=(repeat,workers[repeat],events),daemon=True).start()
            except BaseException as error:
                value=dict(status='failed_worker_dispatch_saved_other_owned_worker_observer_preserved',repeat=repeat,error=repr(error),traceback=traceback.format_exc());dispatch_failures[repeat]=value
                self.save('dispatch_failure_r'+repeat+'.json',value);self.save('state.json',dict(status='failure_saved_other_owned_workers_preserved',event=value,registration=self.registration))
        results={};evaluation=None;evaluation_failure=None
        while len(results)<len(workers):
            kind,repeat,value=events.get()
            if kind=='worker':results[repeat]=value
            elif kind=='evaluation_failure':evaluation_failure=value
            else:evaluation=value
            if value['status'].startswith('failed'):self.save('state.json',dict(status='failure_saved_other_owned_workers_preserved',event=value,registration=self.registration,formal_updates=0))
        if dispatch_failures:return self.save('state.json',dict(status='failed_paired_dispatch_owned_started_workers_observed_no_success_report',dispatch_failures=dispatch_failures,workers=results,evaluation=evaluation,evaluation_failure=evaluation_failure,registration=self.registration))
        if any(v['status']!='worker_Exit0' for v in results.values()):return self.save('state.json',dict(status='failed_core_worker_or_observer_no_success_report',workers=results,evaluation=evaluation,evaluation_failure=evaluation_failure,registration=self.registration))
        while evaluation is None and evaluation_failure is None:
            kind,repeat,value=events.get()
            if kind=='evaluation':evaluation=value
            elif kind=='evaluation_failure':evaluation_failure=value
        if evaluation_failure is not None or evaluation['status']!='evaluation_Exit0':return self.save('state.json',dict(status='failed_uniform_evaluation_no_success_report',workers=results,evaluation=evaluation,evaluation_failure=evaluation_failure,registration=self.registration))
        final=self.finalizer();return self.save('state.json',dict(status='ready_for_root_visual_decision',workers=results,evaluation=evaluation,final=final,registration=self.registration,source=entry(Path(__file__)),decision_created=False,new_full_GPU_task_dispatched=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--registration',type=Path,required=True);args=parser.parse_args();config=validate_config(read(args.registration));identity=entry(args.registration);(OUT/'locks').mkdir(parents=True,exist_ok=True)
    with (OUT/'locks/advance_after_preparation.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);controller=Advance(config,identity)
        try:result=controller.run()
        except BaseException as error:
            controller.save('state.json',dict(status='failed_saved_no_success_report',error=repr(error),traceback=traceback.format_exc(),registration=identity,source=entry(Path(__file__))));raise
        finally:controller.remote.close()
    return 0 if result['status']=='ready_for_root_visual_decision' else 1

if __name__=='__main__':sys.exit(main())
