"""Phase-scoped immutable generic-SR deployment to a separate remote workspace.

The default inventory command uses no network and initializes no GPU. Sync is
explicit, requires a frozen source hash record, creates missing files only and
verifies every selected SHA256 remotely. Existing environments and existing
project/output files are never overwritten. GPU scheduling is the controller's
separate responsibility; CPU environment verification does not reserve a GPU.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import time

from fp_common import ROOT,HERE,OUT,read,write,sha,entry,local

HOSTS={
    'a100-train':dict(project='/home/ubuntu/3DGS/4dsr',activate='activate_a100.sh'),
    'cts':dict(project='/home/cts/Project/4DSR',activate='activate_cts.sh'),
}
PHASES=('code','prepare','calibrate','train','evaluate','full')
FORBIDDEN_COMPONENTS={'.git','.venv','__pycache__','third_party','HOI','hoi','HODome','hodome'}
EXPECTED_TORCH='2.7.1+cu128'
EXPECTED_CUDA='12.8'


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()


def source_paths():
    """Read the maintained publication whitelist without executing its publisher."""
    publisher=ROOT/'scripts/publish_generic_sr.py'
    tree=ast.parse(publisher.read_text())
    declaration=next(node for node in tree.body if isinstance(node,ast.Assign) and
        any(isinstance(target,ast.Name) and target.id=='EXPERIMENTS' for target in node.targets))
    folders=ast.literal_eval(declaration.value)
    if HERE.name not in folders:raise ValueError('New experiment is absent from the generic-SR publication whitelist')
    paths=[]
    for folder in folders:
        directory=ROOT/'experiments'/folder
        if not directory.is_dir():raise FileNotFoundError(directory)
        paths.extend(path for path in sorted(directory.glob('*.py')) if path.is_file() and not path.is_symlink())
    return paths+[publisher]


def relative_file(path):
    path=local(path)
    if not path.is_absolute():path=ROOT/path
    relative=path.relative_to(ROOT)
    if any(part in FORBIDDEN_COMPONENTS for part in relative.parts) or '\n' in str(relative) or '\r' in str(relative):
        raise ValueError(f'Forbidden deployment path: {relative}')
    if path.is_symlink():raise ValueError(f'Symlinks are not deployment payload files: {path}')
    return path,relative.as_posix()


def build_inventory(phase='prepare',host='a100-train',checkpoints=(),include_reference_floats=False):
    """All identities derive from existing registrations; missing is explicit."""
    started=time.monotonic();items={};missing=[];mismatched=[]
    protocol_path=OUT/'protocol.json'
    if not protocol_path.exists():
        return dict(status='missing_required_inputs',phase=phase,host=host,files=[],
            missing=[dict(path=str(protocol_path.relative_to(ROOT)),reason='registered protocol missing')],
            network_used=False,GPU_initialized=False,seconds=time.monotonic()-started)
    protocol=read(protocol_path)
    def add(value,role,required=True):
        identity=value if isinstance(value,dict) else dict(path=str(value))
        path,relative=relative_file(identity['path'])
        if not path.is_file():
            if required:missing.append(dict(path=relative,role=role,reason='required file absent'))
            return
        observed=sha(path)
        if identity.get('sha256') and observed!=identity['sha256']:
            mismatched.append(dict(path=relative,role=role,expected=identity['sha256'],observed=observed));return
        if relative in items:
            if items[relative]['sha256']!=observed:raise ValueError('File changed during inventory')
            if role not in items[relative]['roles']:items[relative]['roles'].append(role)
        else:items[relative]=dict(path=relative,sha256=observed,bytes=path.stat().st_size,roles=[role])
    code=[]
    for path in source_paths():
        add(path,'generic_source');code.append(str(path.relative_to(ROOT)))
    add(protocol_path,'registered_protocol')
    for key in ('source_attachment','previous_protocol','previous_final_delivery','previous_execution_log'):
        if key in protocol:add(protocol[key],'provenance_record')
    schedule_index=OUT/'schedules/index.json';add(schedule_index,'schedule_inventory')
    if schedule_index.is_file():
        index=read(schedule_index)
        if index.get('status')!='completed':missing.append(dict(path=str(schedule_index.relative_to(ROOT)),reason='schedules not completed'))
        for identity in index.get('schedules',[]):add(identity,'registered_schedule')
    else:
        for value in protocol.get('schedules',{}).values():add(value,'registered_schedule')
    # Text operator records establish prerequisites; toy CPU fixture weights and
    # old incident assets are never needed to run a remote training branch.
    for path in sorted((OUT/'operator_checks').rglob('*')):
        if path.is_file() and path.suffix in ('.json','.jsonl','.py'):
            add(path,'new_operator_prerequisite_record')
    if phase!='code':
        for key in ('parent','manifest','teacher','old_schedule','lr_curve','roi'):
            add(protocol[key],'shared_legal_input_identity')
        manifest_path=local(protocol['manifest']['path'])
        if manifest_path.is_file():
            manifest=read(manifest_path)
            legal={(f'cam{camera:02d}',frame) for camera in range(2,21) for frame in range(0,120,2)}
            observed={(o['camera_id'],int(o['frame_index'])) for o in manifest['observations'] if o['split']=='train'}
            if observed!=legal:raise ValueError('Deployment manifest is not the registered 1140 legal training observations')
        if phase in ('prepare','calibrate','train','full'):
            for identity in protocol['training_files']:
                if identity.get('role') not in ('LR','SR','teacher'):raise ValueError('Unexpected privileged role in training whitelist')
                add(identity,'legal_train_'+identity['role'])
    if phase in ('calibrate','train','full'):
        frozen=OUT/'support/frozen/index.json';add(frozen,'completed_frozen_support_index')
        if frozen.is_file():
            index=read(frozen)
            if index.get('status')!='completed_frozen_X_support':
                missing.append(dict(path=str(frozen.relative_to(ROOT)),reason='frozen support not complete'))
            else:
                for name in ('tau_registration',):
                    if name in index:add(index[name],'frozen_dispersion_registration')
                for value in index.get('identity',{}).get('schedules',[]):add(value,'cache_schedule_identity')
                parent_index=index.get('identity',{}).get('parent_index')
                if parent_index:add(parent_index,'HR_parent_cache_provenance_index')
                for row in index.get('observations',[])+index.get('entries',[]):
                    add(dict(path=str(frozen.parent/row['path']),sha256=row['sha256']),'frozen_X_support')
                add(frozen.parent/'config.json','frozen_X_support_registration')
    if phase in ('train','full'):
        calibration=OUT/'calibration.json';add(calibration,'completed_frozen_calibration')
        if calibration.is_file():
            result=read(calibration)
            if result.get('status')!='passed':missing.append(dict(path=str(calibration.relative_to(ROOT)),reason='calibration not passed'))
            if 'gdiag' in result:add(result['gdiag'],'calibration_gradient_record')
            for value in result.get('source_files',{}).values():add(value,'calibration_source')
        for suffix in ('_registration.json','_last_attempt.json'):
            add(OUT/f'calibration{suffix}','calibration_provenance',required=False)
    if phase in ('evaluate','full'):
        for identity in protocol['evaluation_files']:add(identity,'evaluation_only_'+identity.get('role','image'))
        # The imported evaluator computes train76 teacher fit, not all 1140.
        for identity in protocol['training_files']:
            if identity.get('role') in ('SR','teacher') and int(identity.get('frame',-1)) in (0,40,80,118):
                add(identity,'evaluation_train76_teacher')
        add(ROOT/'output/dynamic_sr_prior_diagnosis_20260929/spectrum/roi_protocol.json','fixed_evaluation_ROI')
        parent=ROOT/'output/dynamic_sr_confidence_geometry_20261006/evaluation/U6000/extra/complete.json'
        add(parent,'historical_U6000_reference_receipt')
        if parent.is_file():
            reference=read(parent)
            for value in reference.get('results',{}).values():add(value,'historical_U6000_reference_statistics')
            if include_reference_floats:
                floats=read(parent.parent/'float_index.json')
                for value in floats['entries']:
                    add(dict(path=str(parent.parent/value['path']),sha256=value['sha256']),'historical_U6000_reference_float')
        for path in checkpoints:
            candidate,relative=relative_file(path)
            if not relative.startswith(str(OUT.relative_to(ROOT))+'/runs/') or candidate.suffix!='.pt':
                raise ValueError('Only this experiment endpoint checkpoints may be dispatched for evaluation')
            add(candidate,'current_endpoint_checkpoint')
            add(candidate.with_suffix('.json'),'current_endpoint_checkpoint_receipt')
    files=[items[name] for name in sorted(items)]
    code_identity={name:items[name]['sha256'] for name in sorted(code) if name in items}
    code_sha=canonical_sha(code_identity)
    target=HOSTS[host]
    workspace=f"{target['project']}/execution_workspaces/{HERE.name}/{code_sha[:16]}"
    upstream=Path(os.environ.get('FOURDSR_UPSTREAM','/home/cai_tianshun/Project/4dgs'))
    upstream_identity={}
    for relative in ('scene/gaussian_model.py','scene/deformation.py','scene/hexplane.py','gaussian_renderer/__init__.py'):
        source=upstream/relative
        if source.is_file():upstream_identity[relative]=sha(source)
        else:missing.append(dict(path=str(source),role='existing_upstream_source',reason='local upstream source missing'))
    return dict(status='ready_input_inventory' if not missing and not mismatched else 'missing_or_mismatched_required_inputs',
        phase=phase,host=host,origin_root=str(ROOT),remote_workspace=workspace,existing_project=target['project'],
        existing_activate_script=f"{target['project']}/{target['activate']}",
        files=files,missing=missing,mismatched=mismatched,code_sha256=code_sha,source_sha256=code_identity,
        upstream_source_sha256=upstream_identity,protocol=entry(protocol_path),
        bytes=sum(item['bytes'] for item in files),file_count=len(files),
        scope='generic allowlisted source; registered legal inputs; only phase-required new assets; evaluation HR/LR explicitly separate',
        existing_files_never_overwritten=True,delete_used=False,environment_copied=False,HOI_assets_copied=False,
        source_record_paths_not_rewritten=True,network_used=False,GPU_initialized=False,
        endpoint_checkpoints_requested=len(checkpoints),include_historical_reference_floats=include_reference_floats,
        seconds=time.monotonic()-started)


def require_frozen_sources(inventory,path):
    if path is None:raise ValueError('Explicit --sync requires --frozen-source-manifest from the root source freeze')
    record=read(path)
    values=record.get('sources',record.get('source_identity',record.get('sha256')))
    if values is None:
        values={item['path']:item['sha256'] for item in record.get('source_files',[])}
    for relative,expected in inventory['source_sha256'].items():
        actual=values.get(relative)
        if isinstance(actual,dict):actual=actual.get('sha256')
        if actual!=expected:raise ValueError(f'Frozen source identity missing/different: {relative}')
    frozen_path=Path(path).resolve()
    try:return entry(frozen_path)
    except ValueError:return dict(path=str(frozen_path),sha256=sha(frozen_path),location='local external frozen publication record; not synced')


def verify_inventory(inventory_path,report_path=None):
    """Run inside the existing activated environment, without CUDA init."""
    started=time.monotonic();inventory=read(inventory_path);failures=[]
    if inventory['status']!='ready_input_inventory':raise ValueError('Missing inventory cannot be declared ready remotely')
    if ROOT.resolve()!=Path(inventory['remote_workspace']).resolve():
        raise ValueError('Verification must execute inside the registered isolated target workspace')
    for item in inventory['files']:
        path=ROOT/item['path']
        if not path.is_file():failures.append(dict(path=item['path'],reason='missing'))
        elif sha(path)!=item['sha256']:failures.append(dict(path=item['path'],reason='SHA256 mismatch'))
    import torch
    if str(torch.__version__)!=EXPECTED_TORCH or str(torch.version.cuda)!=EXPECTED_CUDA:
        failures.append(dict(reason='torch/CUDA runtime mismatch',torch=str(torch.__version__),cuda=str(torch.version.cuda)))
    upstream=Path(os.environ.get('FOURDSR_UPSTREAM',''))
    for relative,expected in inventory['upstream_source_sha256'].items():
        path=upstream/relative
        if not path.is_file() or sha(path)!=expected:failures.append(dict(path=str(path),reason='upstream source missing/different'))
    extension=None
    try:
        import diff_gaussian_rasterization._C as native
        extension=dict(path=str(native.__file__),sha256=sha(native.__file__))
    except Exception as error:failures.append(dict(reason='existing rasterizer extension import failed',error=repr(error)))
    for item in inventory['files']:
        if 'generic_source' in item['roles'] and item['path'].endswith('.py'):
            try:compile((ROOT/item['path']).read_text(),item['path'],'exec')
            except Exception as error:failures.append(dict(path=item['path'],reason='source compilation failed',error=repr(error)))
    initialized=bool(torch.cuda.is_initialized())
    if initialized:failures.append(dict(reason='CPU deployment verification unexpectedly initialized CUDA'))
    result=dict(status='passed_phase_files_and_CPU_runtime_verification' if not failures else 'failed_phase_verification',
        inventory=entry(Path(inventory_path).resolve()),files_checked=len(inventory['files']),failures=failures,
        phase=inventory['phase'],hostname=socket.gethostname(),workspace=str(ROOT),
        torch=str(torch.__version__),cuda=str(torch.version.cuda),python=sys_version(),
        rasterizer_extension=extension,CUDA_initialized=initialized,GPU_forwards=0,
        native_GPU_fixture_passed=False,parameter_updates=0,seconds=time.monotonic()-started,
        claim='Selected file identities and existing CPU imports only; GPU occupancy and native execution require separate controller checks.')
    target=Path(report_path or OUT/'deployment/verifications'/f"{inventory['phase']}_{time.time_ns()}.json")
    write(target,result)
    if failures:raise RuntimeError(json.dumps(result,ensure_ascii=False))
    return target,result


def sys_version():
    import sys
    return sys.version


REMOTE_PREFLIGHT='''import hashlib,json,os,pathlib,sys
p=json.load(sys.stdin);root=pathlib.Path(p['remote_workspace']);project=pathlib.Path(p['existing_project'])
try:parts=root.relative_to(project).parts
except ValueError:print(json.dumps({'status':'refused_workspace_outside_existing_project'}));sys.exit(2)
cursor=project
for part in parts:
 cursor=cursor/part
 if cursor.is_symlink():print(json.dumps({'status':'refused_workspace_symlink','path':str(cursor)}));sys.exit(2)
root.mkdir(parents=True,exist_ok=True)
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
conflicts=[];same=0;missing=0
for e in p['files']:
 q=root/e['path']
 try:q.resolve().relative_to(root.resolve());inside=True
 except ValueError:inside=False
 if not inside or q.is_symlink():conflicts.append({'path':e['path'],'reason':'symlink/path escape'});continue
 if q.exists():
  if not q.is_file() or sha(q)!=e['sha256']:conflicts.append({'path':e['path'],'reason':'existing destination differs; overwrite forbidden'})
  else:same+=1
 else:missing+=1
if conflicts:print(json.dumps({'status':'refused_existing_file_conflict','conflicts':conflicts}));sys.exit(2)
print(json.dumps({'status':'passed_no_overwrite_preflight','identical_existing':same,'missing_new_files':missing}))
'''


def sync_inventory(inventory,inventory_path,frozen_manifest,report_directory):
    if inventory['status']!='ready_input_inventory':raise ValueError('Missing/mismatched local assets block synchronization')
    frozen=require_frozen_sources(inventory,frozen_manifest)
    host=inventory['host'];workspace=inventory['remote_workspace'];report_directory.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();commands=[]
    ssh=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=20',host]
    preflight=ssh+['python3 -c '+shlex.quote(REMOTE_PREFLIGHT)]
    commands.append(preflight)
    result=subprocess.run(preflight,input=json.dumps(inventory),text=True,capture_output=True,check=False)
    (report_directory/'preflight.stdout').write_text(result.stdout);(report_directory/'preflight.stderr').write_text(result.stderr)
    if result.returncode:raise RuntimeError('Remote no-overwrite preflight failed; inspect recorded output')
    # The inventory itself is uniquely named and is an additional payload file.
    filelist=report_directory/'files.txt'
    names=[item['path'] for item in inventory['files']]+[str(inventory_path.relative_to(ROOT))]
    filelist.write_text('\n'.join(sorted(set(names)))+'\n')
    command=['rsync','-az','--checksum','--ignore-existing','--delay-updates',
             '--files-from='+str(filelist),'--copy-dest='+inventory['existing_project'],
             str(ROOT)+'/',host+':'+workspace+'/']
    commands.append(command)
    with (report_directory/'rsync.log').open('w') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
    relative_inventory=str(inventory_path.relative_to(ROOT))
    relative_report=str((OUT/'deployment/verification'/f"{host}_{inventory['phase']}_{time.time_ns()}.json").relative_to(ROOT))
    remote_command='source '+shlex.quote(inventory['existing_activate_script'])+' && cd '+shlex.quote(workspace)+\
        ' && python '+shlex.quote('experiments/'+HERE.name+'/deploy.py')+' --verify --inventory '+\
        shlex.quote(relative_inventory)+' --verification-out '+shlex.quote(relative_report)
    command=ssh+['bash -lc '+shlex.quote(remote_command)];commands.append(command)
    with (report_directory/'verification.log').open('w') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
    download=['rsync','-az',host+':'+workspace+'/'+relative_report,str(report_directory/'remote_verification.json')]
    commands.append(download);subprocess.run(download,check=True)
    verified=read(report_directory/'remote_verification.json')
    if verified['status']!='passed_phase_files_and_CPU_runtime_verification':raise ValueError('Remote verification did not pass')
    if verified['inventory']['sha256']!=sha(inventory_path):raise ValueError('Remote inventory differs from the exact local payload')
    # Source/data must not change while the immutable copy was prepared.
    for item in inventory['files']:
        if sha(ROOT/item['path'])!=item['sha256']:raise ValueError('Local input changed during deployment: '+item['path'])
    receipt=dict(status='completed_immutable_incremental_phase_deployment',host=host,phase=inventory['phase'],
        remote_workspace=workspace,inventory=entry(inventory_path),frozen_source_manifest=frozen,
        remote_verification=entry(report_directory/'remote_verification.json'),commands=commands,
        seconds=time.monotonic()-started,files=inventory['file_count'],bytes=inventory['bytes'],
        existing_files_overwritten=0,delete_used=False,HOI_assets_copied=False,
        GPU_reserved=False,GPU_forwards=0,training_updates=0)
    write(report_directory/'receipt.json',receipt)
    return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=PHASES,default='prepare');parser.add_argument('--host',choices=tuple(HOSTS),default='a100-train')
    mode=parser.add_mutually_exclusive_group();mode.add_argument('--inventory-only',action='store_true');mode.add_argument('--sync',action='store_true');mode.add_argument('--verify',action='store_true')
    parser.add_argument('--inventory',type=Path);parser.add_argument('--verification-out',type=Path)
    parser.add_argument('--frozen-source-manifest',type=Path);parser.add_argument('--checkpoint',type=Path,action='append',default=[])
    parser.add_argument('--include-reference-floats',action='store_true')
    a=parser.parse_args()
    if a.verify:
        if a.inventory is None:parser.error('--verify requires --inventory')
        path,result=verify_inventory(a.inventory,a.verification_out)
        print(json.dumps(dict(status=result['status'],report=str(path),files=result['files_checked'])),flush=True);return
    inventory=build_inventory(a.phase,a.host,a.checkpoint,a.include_reference_floats)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    path=a.inventory or OUT/'deployment/inventories'/f'{a.host}_{a.phase}_{stamp}_{time.time_ns()}.json'
    if path.exists():raise FileExistsError('Preserve the prior inventory; select a new --inventory path')
    write(path,inventory)
    if a.sync:
        directory=OUT/'deployment/sync_records'/f'{a.host}_{a.phase}_{stamp}_{time.time_ns()}'
        try:receipt=sync_inventory(inventory,path,a.frozen_source_manifest,directory)
        except BaseException as error:
            write(directory/'failure.json',dict(status='failed_phase_deployment',error=repr(error),inventory=entry(path),
                existing_files_overwritten=0,parameter_updates=0,GPU_reserved=False));raise
        print(json.dumps(receipt,ensure_ascii=False),flush=True)
    else:
        print(json.dumps(dict(status=inventory['status'],phase=a.phase,files=inventory.get('file_count',0),bytes=inventory.get('bytes',0),
            missing=len(inventory.get('missing',[])),mismatched=len(inventory.get('mismatched',[])),
            inventory=str(path),network_used=False,GPU_initialized=False,seconds=inventory['seconds']),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
