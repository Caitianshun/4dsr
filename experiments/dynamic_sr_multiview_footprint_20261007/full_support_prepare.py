"""Selected full-native-parent raw HR moments and frozen X support.

The frozen support_cache.prepare function body is reused unchanged. Only its
population, grids, paths and six-edge schedule interface are adapted in a
private module instance. This entry never reads image pixels, teachers, HR or
held-out data and never updates the native parent. Lambda/kappa calibration is
owned by full_refine --mode calibrate, consuming this exact canonical index.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import fcntl
import json
import math
import os
from pathlib import Path
import socket
import sys
import time
import traceback
from types import SimpleNamespace

from fp_common import ROOT,HERE,OUT,local,read,write,sha,entry,bound,module
import full_prefix as prefix
import full_refine as refine
from config import ARMS,METHODS

SCHEMA='selected_full_native_X_support_prepare_v1'
CALIBRATION_FRAMES=(0,100,200,299)
OUTPUT_ROOT=OUT/'full_support'
PIXEL_SUFFIXES={'.png','.jpg','.jpeg','.npz','.npy','.exr','.tif','.tiff'}


def source_files(upstream):
    paths=(Path(__file__),HERE/'fp_common.py',HERE/'full_prefix.py',HERE/'full_refine.py',
        HERE/'support_cache.py',HERE/'footprint.py',HERE/'config.py',prefix.LEGACY,
        ROOT/'experiments/dynamic_sr_confidence_geometry_20261006/depth_prior.py')
    return dict(project={str(p.relative_to(ROOT)):sha(p) for p in paths},
                upstream={p:sha(Path(upstream)/p) for p in prefix.RUNTIME_FILES})


def legal_records(manifest):
    cameras=list(manifest['splits']['train'])
    frames=list(manifest['frame_indices'])
    if 'cam00' in cameras or 'cam01' not in cameras or frames!=list(range(300)):
        raise ValueError('Only official full-time training keys including cam01 are legal')
    rows=sorted((r for r in manifest['observations'] if r['split']=='train'),
                key=lambda r:(int(r['frame_index']),r['camera_id']))
    expected={(c,f) for c in cameras for f in frames}
    if len(rows)!=len(expected) or {(r['camera_id'],int(r['frame_index'])) for r in rows}!=expected:
        raise ValueError('Missing, duplicate or foreign full training observation')
    return rows


def all_edges(tables):
    edges=set()
    for table in tables:edges.update(refine.required_edges(table,ARMS['X']))
    return sorted(edges,key=lambda k:(k[2],k[0],k[1]))


def build_plan(a):
    """CPU identity registration with explicit missing; no CUDA or GPU query."""
    m,data,_=prefix.validate_manifest(local(a.manifest),a.seed)
    rows=legal_records(m)
    if m['resolutions']['hr']!=[1344,1008] or m['resolutions']['lr']!=[336,252]:
        raise ValueError('Separate audited protocol required for another grid')
    upstream=Path(a.upstream or os.environ.get('FOURDSR_UPSTREAM','/home/cai_tianshun/Project/4dgs'))
    sources=source_files(upstream);missing=[];deps={};parent=None;parent_plan=None
    def available(name,path):
        if path is None or not local(path).is_file():
            missing.append(dict(dependency=name,path=str(path) if path else None));return None
        deps[name]=entry(local(path));return read(bound(deps[name]))
    selection=available('selection',a.selection)
    if selection is not None:refine.validate_selection(selection,a.method)
    need_X=ARMS[a.method]['X']
    if not need_X:
        return dict(schema=SCHEMA,status='pending_selection_no_X_cache' if missing else 'selected_method_requires_no_X_cache',
            method=a.method,seed=a.seed,scene=m['scene'],requires_X=False,missing=missing,dependencies=deps,
            manifest=data['manifest'],source_files=sources,planned_moment_forwards=0,planned_edges=0,
            Adam_calls=0,parameter_updates=0,HR_reference_images_read=False,heldout_pixels_read=False)
    if not a.parent or not a.parent_complete or not local(a.parent).is_file() or not local(a.parent_complete).is_file():
        missing.append(dict(dependency='complete_independent_native_LR_parent'))
    else:
        parent,parent_plan=refine.full_parent(a.parent,a.parent_complete,data['manifest'],a.seed)
    table=available('schedule',a.schedule)
    base=dict(manifest=data['manifest'],parent=parent['checkpoint'] if parent else None)
    context=dict(base,schedule=deps.get('schedule'))
    edges=[]
    if table is not None:
        if parent is None:missing.append(dict(dependency='schedule_native_parent_binding'))
        else:
            refine.validate_schedule(table,m,base,a.seed)
            edges=all_edges([table])
            if {(s,f) for s,t,f in edges}!={(r['camera_id'],int(r['frame_index'])) for r in rows}:
                raise ValueError('Selected schedule does not cover every legal full source observation')
    return dict(schema=SCHEMA,status='pending_selected_full_native_support_dependencies' if missing else 'registered_ready_selected_full_native_support',
        scene=m['scene'],seed=a.seed,method=a.method,requires_X=True,missing=missing,manifest=data['manifest'],
        parent=parent,dependencies=deps,full_native_context=context,source_files=sources,
        configuration=parent_plan['configuration'] if parent_plan else None,author_commit=prefix.PIN,
        representation=refine.REPRESENTATION,hr_size=[1008,1344],lr_size=[252,336],
        full_legal_keys=[[r['camera_id'],int(r['frame_index'])] for r in rows],
        calibration_keys=[[c,f] for c in m['splits']['train'] for f in CALIBRATION_FRAMES],
        planned_moment_forwards=len(rows),planned_edges=len(edges),planned_RGB_forwards=0,
        Adam_calls=0,parameter_updates=0,teacher_reads=0,HR_reference_images_read=False,heldout_pixels_read=False,
        implementation='private population/path adapter of unchanged frozen support_cache.prepare body',
        calibration_contract=dict(entry='full_refine.py --mode calibrate --method X or MX',
            balanced_keys='all actual training cameras x 0/100/200/299',tau='.75 quantile of positive finite supported HR c_z; floor1e-6',
            lambda_X='.1 median mean-three-LR/X summed-native-index xyz RMS; no coefficient grid',
            kappa='median anchor RGB L1/MSE RMS',Adam_calls=0,parameter_updates=0,
            calibration_separate=True,support_index='frozen/index.json',parent_buffers_must_not_be_reset=True))


def verify_plan(plan,upstream):
    if source_files(upstream)!=plan['source_files']:raise ValueError('Frozen full support source tree changed')
    bound(plan['manifest'])
    for ref in plan['dependencies'].values():bound(ref)
    if plan.get('parent'):
        for key in ('checkpoint','complete','sidecar','plan'):bound(plan['parent'][key])


def freeze_sources(plan,out,upstream):
    refs={}
    for group,items in plan['source_files'].items():
        for rel,identity in items.items():
            origin=ROOT/rel if group=='project' else Path(upstream)/rel
            if sha(origin)!=identity:raise ValueError('Source changed before freezing: '+rel)
            target=out/'source_snapshot'/group/rel;target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists():
                if sha(target)!=identity:raise ValueError('Existing immutable source snapshot differs')
            else:target.write_bytes(origin.read_bytes());target.chmod(0o444)
            refs[rel if group=='project' else 'upstream/'+rel]=entry(target)
    return refs


def preserve_incomplete(root):
    """Keep interrupted orphan assets before the original core may regenerate."""
    root=Path(root);orphans=[]
    for pattern in ('*.npz','observations/*.npz','edges/*.npz'):
        for path in root.glob(pattern):
            if not path.with_suffix('.json').exists():orphans.append(path)
    for pattern in ('*.json','observations/*.json','edges/*.json'):
        for path in root.glob(pattern):
            if path.name in ('config.json','index.json','tau_registration.json'):continue
            if not path.with_suffix('.npz').exists():orphans.append(path)
    # An archived interrupted write remains immutable on later resumes.
    orphans+=[p for p in root.rglob('*.tmp') if p.relative_to(root).parts[0]!='incidents']
    if not orphans:return None
    dst=root/'incidents'/f'incomplete_{time.time_ns()}';dst.mkdir(parents=True)
    entries=[]
    for path in sorted(set(orphans)):
        ref=dict(relative=str(path.relative_to(root)),sha256=sha(path),bytes=path.stat().st_size)
        target=dst/ref['relative'];target.parent.mkdir(parents=True,exist_ok=True);path.replace(target);entries.append(ref)
    write(dst/'receipt.json',dict(status='incomplete_assets_preserved_before_regeneration',entries=entries,
        unreturned_operation_cost='unknown',parameter_updates=0,Adam_calls=0))
    return entry(dst/'receipt.json')


@contextmanager
def pixel_boundary(out):
    """Model caches may be read/written in this revision; no image is opened."""
    active=[True];opened={};root=Path(out).resolve()
    def audit(event,args):
        if not active[0] or event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        path=Path(os.fsdecode(args[0])).resolve()
        if path.suffix.lower() not in PIXEL_SUFFIXES:return
        if path.suffix.lower() not in ('.npz','.npy') or not path.is_relative_to(root):
            raise AssertionError('Pixel/cache read outside selected full support revision: '+str(path))
        name=str(path);opened[name]=opened.get(name,0)+1
    sys.addaudithook(audit)
    try:yield opened
    finally:active[0]=False


def core_adapter(manifest,protocol_dir):
    """No function-body patch: bind only populations, grids, roots and edges."""
    core=module('native_full_support_core_'+str(time.time_ns()),HERE/'support_cache.py')
    core.CAMERAS=tuple(manifest['splits']['train']);core.FRAMES=tuple(manifest['frame_indices'])
    core.CALIBRATION_FRAMES=CALIBRATION_FRAMES
    core.HR_SIZE=tuple(reversed(manifest['resolutions']['hr']));core.LR_SIZE=tuple(reversed(manifest['resolutions']['lr']))
    core.OUT=Path(protocol_dir)
    core.legal_records=lambda value:[r for r in value['observations'] if r['split']=='train']
    core.required_edges=all_edges
    return core


def native_camera(manifest,row,uid):
    """Calibrated LR-sized zero payload gives the same HR projection without pixels."""
    import numpy as np
    import torch
    width,height=manifest['resolutions']['lr'];cal=manifest['cameras'][row['camera_id']]
    item=dict(image=torch.zeros(3,height,width),time=float(row['time']),camera_id=row['camera_id'],
        frame_index=int(row['frame_index']),width=width,height=height,K=np.asarray(cal['K_lr'],np.float64),
        w2c=np.asarray(cal['w2c'],np.float64))
    return refine.hr_camera(item,uid,manifest)


def export_parent(plan,out,core,upstream,journal):
    """One actual HR raw-moment render per legal key; no LR interpolation/RGB."""
    import numpy as np
    import torch
    manifest=read(bound(plan['manifest']));rows=legal_records(manifest)
    identity=dict(full_native_context=plan['full_native_context'],plan_sha256=prefix.digest(plan),
        parent=plan['parent']['checkpoint'],manifest=plan['manifest'],hr_size=plan['hr_size'],
        quantity='native raw unclamped HR [A,M1,M2]; no LR-depth interpolation',sources=plan['source_files'],
        training_images_opened=0,HR_reference_images_read=False,heldout_pixels_read=False)
    directory=out/'parent_moments_hr';directory.mkdir(parents=True,exist_ok=True)
    config=directory/'config.json'
    if config.exists() and read(config)!=identity:raise ValueError('Full raw moment identity changed; preserve revision')
    if not config.exists():write(config,identity)
    index=directory/'index.json'
    if index.exists():
        existing=read(index)
        if existing['status']!='completed_HR_parent_moments' or existing['identity']!=identity:
            raise ValueError('Existing full raw index differs')
        expected={(r['camera_id'],int(r['frame_index'])) for r in rows}
        if len(existing['entries'])!=len(expected) or {(r['camera'],int(r['frame'])) for r in existing['entries']}!=expected:
            raise ValueError('Existing raw index is incomplete')
        return index
    preserve_incomplete(directory)
    closed={};missing=[]
    for row in rows:
        key=(row['camera_id'],int(row['frame_index']));path=directory/f'{key[0]}_{key[1]:04d}.npz';receipt=path.with_suffix('.json')
        if path.exists() and receipt.exists():
            value=read(receipt)
            if value.get('config_sha256')!=sha(config) or value.get('camera')!=key[0] or value.get('frame')!=key[1] or sha(path)!=value['sha256']:
                raise ValueError('Closed full raw parent asset changed')
            closed[key]=value
        else:missing.append(row)
    g=None;before=None;render=None;state=None;frame=None
    torch.set_num_threads(journal['cpu_threads']);torch.cuda.reset_peak_memory_stats();started=time.monotonic();rendered=0
    def persist():write(Path(journal['path']),journal)
    try:
        if missing:
            sys.path.insert(0,str(upstream));sys.path.insert(1,str(prefix.LEGACY.parent))
            from scene.gaussian_model import GaussianModel
            h=SimpleNamespace(**plan['configuration']['ModelHiddenParams']);opt=SimpleNamespace(**plan['configuration']['OptimizationParams'])
            g=GaussianModel(plan['configuration']['ModelParams']['sh_degree'],h);g._deformation=g._deformation.cuda()
            parent_plan=read(bound(plan['parent']['plan']))
            payload=refine.checkpoint_payload(bound(plan['parent']['checkpoint']),torch,parent_plan,'full_author_LR_prefix_checkpoint_v1','cuda')
            refine.restore_native_model(g,payload,opt,torch,np,parent_plan,'full_author_LR_prefix_checkpoint_v1');del payload
            def immutable():return dict(model=refine.tree_digest(g.capture()),deformation_accum=refine.tree_digest(g._deformation_accum),
                gradients=refine.tree_digest({k:p.grad for k,p in prefix.parameter_map(g).items()}),RNG=refine.tree_digest(prefix.rng_state(torch,np)),
                versions={k:p._version for k,p in prefix.parameter_map(g).items()})
            before=immutable()
            render=module('full_support_native_depth_'+str(time.time_ns()),ROOT/'experiments/dynamic_sr_confidence_geometry_20261006/depth_prior.py').render_moments
            with torch.no_grad():
                for n,row in enumerate(missing):
                    key=(row['camera_id'],int(row['frame_index']));tick=time.monotonic()
                    journal['active_operation']=dict(kind='native_HR_raw_moment',key=list(key));journal['attempted_moments']+=1;persist()
                    camera=native_camera(manifest,row,n)
                    if frame!=key[1]:
                        frame=key[1];state=refine.native_effective_state(g,camera.time,torch);cov=refine.state_covariance(state,torch)
                    raw=render(camera,state['xyz'],cov,state['opacity'])['hr_moments']
                    if tuple(raw.shape)!=(3,*plan['hr_size']):raise ValueError('Native full frame did not produce true HR raw moments')
                    path=directory/f'{key[0]}_{key[1]:04d}.npz'
                    core.atomic_npz(path,hr_moments=raw.detach().cpu().numpy().astype(np.float32));torch.cuda.synchronize()
                    result=dict(camera=key[0],frame=key[1],path=path.name,sha256=sha(path),config_sha256=sha(config),
                        time=float(row['time']),hr_size=plan['hr_size'],moment_forwards=1,rgb_forwards=0,parameter_updates=0,
                        seconds=time.monotonic()-tick,nonfinite_moment_elements=int((~torch.isfinite(raw)).sum()))
                    write(path.with_suffix('.json'),result);closed[key]=result;rendered+=1
                    journal['completed_moments']+=1;journal['active_operation']=None;persist();del camera,raw
                    if (n+1)%60==0:print(json.dumps(dict(stage='full_native_raw_HR',new_moments=rendered,reused=len(rows)-len(missing),planned=len(rows))),flush=True)
            if immutable()!=before:raise ValueError('Zero-update export changed native parent/Adam/buffers/gradients/RNG')
        verify_plan(plan,upstream)
        entries=[closed[(r['camera_id'],int(r['frame_index']))] for r in rows]
        value=dict(status='completed_HR_parent_moments',scope='full_0_299_train_only',full_native_context=plan['full_native_context'],
            identity=identity,entries=entries,parent_sha256=plan['parent']['checkpoint']['sha256'],manifest_sha256=plan['manifest']['sha256'],
            total_moment_forwards=len(entries),new_moment_forwards=rendered,reused_entries=len(rows)-len(missing),rgb_forwards=0,
            parameter_updates=0,Adam_calls=0,parent_state_unchanged=True,seconds=time.monotonic()-started,
            per_entry_seconds_sum=sum(r['seconds'] for r in entries),peak_gb=torch.cuda.max_memory_allocated()/1e9,
            gpu=torch.cuda.get_device_name(),physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),torch=str(torch.__version__))
        write(index,value);return index
    finally:del g,render,state


def prepare_support(plan,out,core,parent_index,producer_sources,device):
    """Invoke the actual frozen production formula and add full context only."""
    verify_parent=read(parent_index)
    if verify_parent.get('full_native_context')!=plan['full_native_context'] or verify_parent.get('scope')!='full_0_299_train_only':
        raise ValueError('Short U6000 or another full seed parent cache refused')
    expected={tuple(k) for k in plan['full_legal_keys']}
    if len(verify_parent['entries'])!=len(expected) or {(r['camera'],int(r['frame'])) for r in verify_parent['entries']}!=expected:
        raise ValueError('Incomplete full HR raw parent cache')
    directory=out/'frozen';preserve_incomplete(directory)
    prior={}
    for path in sorted((directory/'edges').glob('*.json')):
        if path.with_suffix('.npz').exists():
            row=read(path);prior[row['path']]=dict(path=row['path'],declared_NPZ_sha256=row['sha256'],
                closed_receipt=entry(path),seconds=row['seconds'])
    # The immutable formula's compute_device means this dispatch. Retained
    # historical edges must not become CUDA work merely by changing that field.
    started=time.monotonic();dispatch=out/'support_dispatches'/f'{time.time_ns()}_{os.getpid()}.json'
    execution=dict(status='running_frozen_full_support_dispatch',compute_device=str(device),
        prior_closed_edges=list(prior.values()),prior_closed_edge_seconds_sum=sum(r['seconds'] for r in prior.values()),
        prior_NPZ_sha_semantics='Receipt declarations; immutable core verifies actual bytes before reuse',
        source=entry(Path(__file__)),parent_index=entry(parent_index),parameter_updates=0,Adam_calls=0)
    write(dispatch,execution)
    try:index=core.prepare(parent_index=parent_index,schedules=[bound(plan['dependencies']['schedule'])],out=directory,device=device)
    except BaseException as exc:
        execution.update(status='failed_frozen_full_support_dispatch_preserved',seconds=time.monotonic()-started,
            error=repr(exc),unreturned_operation_cost='unknown')
        write(dispatch,execution);raise
    value=read(index)
    reused=[r for r in value['entries'] if r['path'] in prior and r['sha256']==prior[r['path']]['declared_NPZ_sha256']]
    reused_paths={r['path'] for r in reused}
    new=[r for r in value['entries'] if r['path'] not in reused_paths]
    if len(new)!=value['new_edges']:raise ValueError('Actual support generation/reuse count differs from immutable core')
    execution.update(status='completed_frozen_full_support_dispatch',seconds=time.monotonic()-started,
        reused_closed_edges=len(reused),new_edges=len(new),new_closed_edge_seconds_sum=sum(r['seconds'] for r in new),
        generated_edges=[dict(path=r['path'],sha256=r['sha256'],seconds=r['seconds']) for r in new],
        reused_closed_edges_not_charged_to_current_device=True,
        note='Current dispatch wall time includes SHA reads and cache loading; prior closed costs are separate. Prior devices follow retained dispatch receipts; undisclosed historical devices remain unknown.')
    write(dispatch,execution)
    value.update(scope='full_0_299_train_only',full_native_context=plan['full_native_context'],selection=plan['dependencies']['selection'],
        producer_registration=entry(out/'registration.json'),calibration_scope='all actual train cameras x 0/100/200/299',
        legacy_train76_field_label='retained core field name; keys contain the full balanced training calibration set',
        execution_receipt=entry(dispatch),execution_history=[entry(p) for p in sorted((out/'support_dispatches').glob('*.json'))])
    value['identity']['producer_sources']=producer_sources
    value['identity']['fulltrain_calibration_keys']=plan['calibration_keys']
    value['identity']['population_adapter_source_sha256']=sha(__file__)
    refine.validate_support(value,plan['full_native_context'],read(bound(plan['dependencies']['schedule'])),read(bound(plan['manifest'])))
    write(index,value);return index


def registered(plan,out):
    path=out/'registration.json'
    if path.exists() and read(path)!=plan:raise ValueError('Registration is immutable; use a new revision after dependency/source changes')
    if not path.exists():write(path,plan)
    return path


def prepare(a,plan):
    out=local(a.out).resolve();upstream=Path(a.upstream or os.environ.get('FOURDSR_UPSTREAM','/home/cai_tianshun/Project/4dgs'))
    if plan['status']=='selected_method_requires_no_X_cache':return dict(status='skipped_selected_non_X_method_no_cache',method=a.method,Adam_calls=0,GPU_calls=0)
    if plan['status']!='registered_ready_selected_full_native_support':raise ValueError('Pending dependencies refuse all generation before CUDA: '+str(plan['missing']))
    if not out.is_relative_to(OUTPUT_ROOT.resolve()):raise ValueError('Production support output must be an independent output/full_support revision')
    gpu_needed=a.phase!='support' or a.support_device!='cpu'
    if gpu_needed and (not a.operator_resource_resolved or not a.gpu_uuid or os.environ.get('CUDA_VISIBLE_DEVICES')!=a.gpu_uuid):
        raise ValueError('Root must resolve resource ownership and bind allocated physical UUID before CUDA')
    verify_plan(plan,upstream);registered(plan,out)
    journal=dict(status='running_selected_full_support',path=str(out/'attempts'/f'{time.time_ns()}_{os.getpid()}.json'),
        host=socket.gethostname(),pid=os.getpid(),gpu_uuid=a.gpu_uuid if gpu_needed else None,phase=a.phase,
        cpu_threads=a.cpu_threads,attempted_moments=0,completed_moments=0,active_operation=None,
        formal_updates=0,Adam_calls=0,started_unix=time.time(),source_guard=plan['source_files'])
    started=time.monotonic();write(Path(journal['path']),journal)
    try:
        observations=read(bound(plan['manifest']));core=core_adapter(observations,out)
        producers=freeze_sources(plan,out,upstream)
        write(out/'protocol.json',dict(parent=plan['parent']['checkpoint'],manifest=plan['manifest']))
        with pixel_boundary(out) as opened:
            parent_index=out/'parent_moments_hr/index.json'
            if a.phase in ('all','export'):parent_index=export_parent(plan,out,core,upstream,journal)
            support_index=None
            if a.phase in ('all','support'):
                journal['active_operation']=dict(kind='frozen_full_support');write(Path(journal['path']),journal)
                support_index=prepare_support(plan,out,core,parent_index,producers,a.support_device)
                journal['active_operation']=None
            verify_plan(plan,upstream)
        journal.update(status='completed_selected_full_support_phase',seconds=time.monotonic()-started,
            parent_index=entry(parent_index),support_index=entry(support_index) if support_index else None,
            actual_model_cache_opens=opened,HR_reference_images_read=False,heldout_pixels_read=False,
            closed_edge_seconds_sum=sum(r['seconds'] for r in read(support_index)['entries']) if support_index else None,
            note='Per-edge closed receipt costs include reused historical work; this invocation wallclock is separate. Interrupted unreturned operations unknown.')
        write(Path(journal['path']),journal);write(out/('complete.json' if a.phase in ('all','support') else 'export_complete.json'),journal)
        return journal
    except BaseException as exc:
        journal.update(status='failed_selected_full_support_preserved',seconds=time.monotonic()-started,error=repr(exc),
            traceback=traceback.format_exc(),unreturned_operation_cost='unknown' if journal['active_operation'] else None)
        write(Path(journal['path']),journal);raise


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('plan','register','prepare','contract'),default='plan')
    p.add_argument('--manifest',type=Path);p.add_argument('--seed',type=int);p.add_argument('--method',choices=METHODS)
    p.add_argument('--parent',type=Path);p.add_argument('--parent-complete',type=Path);p.add_argument('--schedule',type=Path)
    p.add_argument('--selection',type=Path);p.add_argument('--out',type=Path);p.add_argument('--upstream',type=Path)
    p.add_argument('--phase',choices=('all','export','support'),default='all')
    p.add_argument('--support-device',choices=('cpu','cuda:0'),default='cuda:0');p.add_argument('--cpu-threads',type=int,default=4)
    p.add_argument('--gpu-uuid');p.add_argument('--lock',type=Path);p.add_argument('--operator-resource-resolved',action='store_true')
    p.add_argument('--idle-check-seconds',type=float,default=5.)
    return p


def main():
    a=parser().parse_args()
    if a.mode=='contract':print(json.dumps(cpu_contract(),ensure_ascii=False));return
    if any(getattr(a,name) is None for name in ('manifest','seed','method','out')):raise ValueError('manifest, seed, method and out required')
    if a.cpu_threads<1 or not 1<=a.idle_check_seconds<=60:raise ValueError('Positive threads and separated resource checks required')
    plan=build_plan(a);out=local(a.out)
    if a.mode=='plan':print(json.dumps(plan,ensure_ascii=False));return
    if a.mode=='register':print(json.dumps(dict(status=plan['status'],registration=entry(registered(plan,out)),missing=plan['missing']),ensure_ascii=False));return
    if plan['status']!='registered_ready_selected_full_native_support':print(json.dumps(prepare(a,plan),ensure_ascii=False));return
    gpu_needed=a.phase!='support' or a.support_device!='cpu'
    if gpu_needed and (not a.gpu_uuid or not a.lock or not a.operator_resource_resolved):raise ValueError('Root UUID, explicit shared lock and resource resolution required')
    lock=local(a.lock) if a.lock else out/'CPU_prepare.lock';lock.parent.mkdir(parents=True,exist_ok=True)
    with lock.open('a+') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if gpu_needed:
            first=prefix.check_gpu(a.gpu_uuid);time.sleep(a.idle_check_seconds);second=prefix.check_gpu(a.gpu_uuid)
            resource=dict(first=first,second=second,separated_seconds=a.idle_check_seconds,
                gpu_uuid=a.gpu_uuid,lock=str(lock),host=socket.gethostname(),pid=os.getpid())
            launch=out/'resource_launches'/f'{time.time_ns()}_{os.getpid()}.json';write(launch,resource)
            write(out/'resource_launch.json',dict(**resource,immutable_receipt=entry(launch)))
        print(json.dumps(prepare(a,plan),ensure_ascii=False))


# The compact executable CPU contract is added below; it must never initialize CUDA.
def cpu_contract():
    """Small grids exercise the real unchanged core, reader and full gates."""
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError("CPU contract requires CUDA_VISIBLE_DEVICES='' before importing torch")
    import copy
    import io
    from contextlib import redirect_stdout
    from unittest.mock import patch
    import numpy as np
    import torch
    if torch.cuda.is_initialized():raise RuntimeError('CPU contract cannot run after CUDA initialization')
    torch.set_num_threads(1)
    out=OUT/'operator_checks/full_support'/f'{time.time_ns()}';out.mkdir(parents=True)
    (out/'full_support_prepare.py').write_bytes(Path(__file__).read_bytes())
    started=time.monotonic();checks=[];record=dict(status='running_CPU_contract',source=entry(Path(__file__)),GPU_calls=0,Adam_calls=0)
    def check(name,call):
        tick=time.monotonic();value=call();checks.append(dict(name=name,seconds=time.monotonic()-tick,result=value))
    def refuses(call,kind=ValueError):
        try:call()
        except kind as exc:return type(exc).__name__
        raise AssertionError('Expected refusal did not occur')
    try:
        # Complete 0..299 schedule population; no pixel fixture is decoded.
        cams=['cam01','cam02','cam03'];keys=[[c,f] for c in cams for f in range(300)]
        manifest=dict(splits=dict(train=cams),frame_indices=list(range(300)),observations=[dict(camera_id=c,frame_index=f,split='train') for c,f in keys])
        context=dict(manifest=dict(path='fixture_manifest',sha256='a'*64),parent=dict(path='fixture_native_parent',sha256='b'*64))
        rows=[]
        for n in range(6000):
            f=(n//3)%300;c=n%3;rows.append([((c+j)%3)*300+f for j in range(3)])
        calibration=[[i*300+f,((i+1)%3)*300+f,((i+2)%3)*300+f] for i in range(3) for f in CALIBRATION_FRAMES]
        table=dict(schema=refine.SCHEDULE_SCHEMA,identity=context,seed=7,updates=6000,record_keys=keys,rows=rows,random_rows=copy.deepcopy(rows),calibration_triplets=calibration,audit=dict(passed=True))
        def scheduling():
            refine.validate_schedule(table,manifest,context,7)
            edges=all_edges([table]);assert len(edges)==1800
            assert {(s,f) for s,t,f in edges}=={tuple(k) for k in keys}
            bad=copy.deepcopy(table);bad['rows'][0][1]=bad['rows'][0][0]
            refuses(lambda:refine.validate_schedule(bad,manifest,context,7))
            leaked=copy.deepcopy(manifest);leaked['splits']['train'][0]='cam00'
            refuses(lambda:legal_records(leaked))
            return dict(all_900_legal_keys_covered=True,six_directed_edges_per_triplet=True,duplicate_camera_and_cam00_refused=True)
        check('full_schedule_and_information_boundary',scheduling)
        upstream=Path(os.environ.get('FOURDSR_UPSTREAM','/home/cai_tianshun/Project/4dgs'))
        def production_gates():
            args=SimpleNamespace(out=OUTPUT_ROOT/'CPU_contract_never_generated',upstream=upstream,method='X',phase='all',support_device='cuda:0',operator_resource_resolved=False,gpu_uuid=None)
            refuses(lambda:prepare(args,dict(status='pending_selected_full_native_support_dependencies',missing=['selection'])))
            refuses(lambda:prepare(args,dict(status='registered_ready_selected_full_native_support')))
            args.method='M'
            skipped=prepare(args,dict(status='selected_method_requires_no_X_cache'))
            assert skipped['GPU_calls']==0 and not args.out.exists()
            no_evidence=dict(status='registered_full_development_selection',selected_candidates=['X'],baseline_method='B0',reason='CPU fixture',short_window_evidence=[],uses_confirmation_for_selection=False)
            refuses(lambda:refine.validate_selection(no_evidence,'X'))
            return dict(pending_selection_refused_before_CUDA=True,unallocated_GPU_refused=True,non_X_no_cache=True,empty_selection_evidence_refused=True)
        check('selection_and_resource_gates',production_gates)
        # Use four calibration times on 12x16 native raw grids. This is an
        # explicitly synthetic CPU cache, never a full production registration.
        small=out/'synthetic_cache';small.mkdir();rawdir=small/'parent_moments_hr';rawdir.mkdir()
        k=np.array([[8.,0.,7.5],[0.,8.,5.5],[0.,0.,1.]]);w=np.eye(4)
        mini=dict(splits=dict(train=cams),frame_indices=list(CALIBRATION_FRAMES),resolutions=dict(hr=[16,12],lr=[4,3]),
            cameras={c:dict(K_hr=k.tolist(),w2c=w.tolist(),c2w=w.tolist()) for c in cams},
            observations=[dict(camera_id=c,frame_index=f,split='train') for c in cams for f in CALIBRATION_FRAMES])
        mini_manifest=small/'manifest.json';write(mini_manifest,mini)
        mini_keys=[[r['camera_id'],r['frame_index']] for r in mini['observations']]
        mini_rows=[[i*4+j,((i+1)%3)*4+j,((i+2)%3)*4+j] for i in range(3) for j in range(4)]
        mini_table=small/'schedule.json';write(mini_table,dict(record_keys=mini_keys,rows=mini_rows,calibration_triplets=mini_rows))
        evidence=small/'selection.json';write(evidence,dict(status='CPU_fixture_only_not_a_method_selection'))
        fullcontext=dict(manifest=entry(mini_manifest),parent=dict(path='CPU_fixture_native',sha256='c'*64),schedule=entry(mini_table))
        write(small/'protocol.json',dict(parent=fullcontext['parent'],manifest=fullcontext['manifest']))
        core=core_adapter(mini,small);entries=[]
        for camera,frame in mini_keys:
            z=float(cams.index(camera)+1);raw=np.empty((3,12,16),np.float32);raw[0]=1.;raw[1]=z;raw[2]=np.nan
            if camera=='cam03':raw[1,5,7]=np.inf
            path=rawdir/f'{camera}_{frame:04d}.npz';core.atomic_npz(path,hr_moments=raw)
            entries.append(dict(camera=camera,frame=frame,path=path.name,sha256=sha(path),seconds=0.))
        parent_index=rawdir/'index.json'
        write(parent_index,dict(status='completed_HR_parent_moments',scope='full_0_299_train_only',full_native_context=fullcontext,
            entries=entries,parent_sha256=fullcontext['parent']['sha256'],manifest_sha256=fullcontext['manifest']['sha256']))
        plan=dict(full_native_context=fullcontext,full_legal_keys=mini_keys,calibration_keys=mini_keys,
            dependencies=dict(schedule=entry(mini_table),selection=entry(evidence)),manifest=entry(mini_manifest),source_files=source_files(upstream))
        write(small/'registration.json',plan);producers=freeze_sources(plan,small,upstream)
        def raw_and_frozen_support():
            with redirect_stdout(io.StringIO()),pixel_boundary(small):index=prepare_support(plan,small,core,parent_index,producers,'cpu')
            value=read(index);assert len(value['entries'])==24 and len(value['observations'])==12
            assert value['tau_z']==1e-6 and read(bound(value['tau_registration']))['fallback_no_positive_finite']
            assert all(row['parent_occlusion_fraction']==0. for row in value['entries'])
            cache=core.SupportCache(index)
            result=cache.edge(('cam01',0),('cam02',0));assert bool(torch.isfinite(result['weight_lr']).all())
            assert bool((result['weight_lr']>=.25).all()) and bool((result['weight_lr']<=1.).all())
            assert torch.allclose(result['weight_lr'][1:-1,1:-1],torch.ones_like(result['weight_lr'][1:-1,1:-1]))
            observation=cache.observation('cam03',0);assert not bool(observation['valid_hr'][5,7])
            edge=cache.edge(('cam01',0),('cam03',0));assert not bool(edge['mask_hr'][5,7])
            import footprint
            assert not bool(footprint.degradation_support(edge['mask_hr'],(3,4)).all())
            known=core.normalize_hr_moments(torch.tensor([[[1.,1.]],[[1.,3.]],[[1.,9.]]]))
            pooled=known['raw_moments'].mean(dim=2,keepdim=True)
            physical=core.normalize_hr_moments(pooled)
            assert abs(float(physical['z'])-2.)<1e-5 and abs(float(physical['variance'])-1.)<1e-5
            return dict(real_frozen_core_and_generic_reader=True,unknown1_and_tau_floor=True,unknown_not_hard_occlusion=True,
                invalid_HR_mask_and_full_D0_support=True,raw_nonnegative_pool_precedes_normalization=True)
        check('unchanged_formula_and_raw_HR_contract',raw_and_frozen_support)
        def reuse_and_failures():
            before={r['path']:r['sha256'] for r in read(small/'frozen/index.json')['entries']}
            with redirect_stdout(io.StringIO()):idx=prepare_support(plan,small,core,parent_index,producers,'cpu')
            after=read(idx);assert after['new_edges']==0 and before=={r['path']:r['sha256'] for r in after['entries']}
            dispatch=read(bound(after['execution_receipt']))
            assert dispatch['new_edges']==0 and dispatch['reused_closed_edges']==24
            assert dispatch['new_closed_edge_seconds_sum']==0 and dispatch['reused_closed_edges_not_charged_to_current_device']
            foreign=copy.deepcopy(read(parent_index));foreign['full_native_context']['parent']['sha256']='f'*64
            bad=rawdir/'foreign.json';write(bad,foreign)
            refuses(lambda:prepare_support(plan,small,core,bad,producers,'cpu'))
            asset=small/'frozen'/after['entries'][0]['path'];original=asset.read_bytes();asset.write_bytes(original+b'changed')
            with redirect_stdout(io.StringIO()):refuses(lambda:prepare_support(plan,small,core,parent_index,producers,'cpu'))
            assert asset.read_bytes()==original+b'changed';asset.write_bytes(original)
            orphan=small/'orphan';orphan.mkdir();p=orphan/'cam01_0000.npz';p.write_bytes(b'partial')
            receipt=preserve_incomplete(orphan);assert receipt and not p.exists()
            assert list((orphan/'incidents').rglob('cam01_0000.npz'))[0].read_bytes()==b'partial'
            interrupted=orphan/'cam02_0000.npz.tmp';interrupted.write_bytes(b'partial temporary write')
            tmp_receipt=preserve_incomplete(orphan);assert tmp_receipt
            immutable_archive=bound(tmp_receipt);archive_sha=sha(immutable_archive)
            assert preserve_incomplete(orphan) is None and sha(immutable_archive)==archive_sha
            try:
                with pixel_boundary(small):open(out/'forbidden_HR.png','wb')
            except AssertionError:pass
            else:raise AssertionError('Image boundary did not reject forbidden HR')
            return dict(closed_NPZ_SHA_reuse=True,foreign_parent_refused=True,corruption_refused_without_overwrite=True,
                interrupted_orphan_preserved=True,historical_incomplete_archive_immutable_on_resume=True,
                reused_edges_have_separate_historical_cost=True,HR_image_open_refused=True)
        check('reuse_parent_identity_integrity_and_read_boundary',reuse_and_failures)
        assert not torch.cuda.is_initialized()
        record.update(status='passed_CPU_contract_requires_separate_native_GPU_acceptance',checks=checks,CUDA_initialized=False,
            scope='Synthetic small-grid formula/interface tests only; no production cache, method choice or native rendering acceptance')
    except BaseException as exc:
        record.update(status='failed_CPU_contract_preserved',checks=checks,error=repr(exc),traceback=traceback.format_exc());raise
    finally:
        record['seconds']=time.monotonic()-started;write(out/'receipt.json',record)
    return dict(**record,receipt=entry(out/'receipt.json'))


if __name__=='__main__':main()
