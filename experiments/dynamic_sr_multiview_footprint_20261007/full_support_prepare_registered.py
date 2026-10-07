"""Registered full-native-parent raw HR moments and frozen X support.

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
import full_prefix_registered as prefix
import full_refine_registered as refine
import full_native_protocol as domain
from config import ARMS,METHODS

SCHEMA='selected_registered_full_native_X_support_prepare_v1'
CALIBRATION_FRAMES=(0,100,200,299)
OUTPUT_ROOT=OUT/'full_support_registered'
PROPOSAL=OUT/'full_native_protocol_proposals/1791391845797437620/proposal.json'
PROPOSAL_SHA='77d9191c25c4a9ec315fa1fdc2787013ccc829e185841b2b4c3f3870628d09b9'
FROZEN_CORE_SHA='2df909b241456e18889fd669d6b3e7cebc5dcdfc010e09988678399ac78b122f'
PIXEL_SUFFIXES={'.png','.jpg','.jpeg','.npz','.npy','.exr','.tif','.tiff'}


def source_files(upstream):
    if sha(HERE/'support_cache.py')!=FROZEN_CORE_SHA:raise ValueError('Frozen support formula source changed')
    if sha(HERE/'full_support_prepare.py')!='fc5431cd35304db6723f64ef855413796eb468fdaf13fb0d53732bf1ee26f8d1':
        raise ValueError('Original accepted full support producer changed')
    paths=(Path(__file__),HERE/'fp_common.py',HERE/'full_prefix_registered.py',HERE/'full_refine_registered.py',
        HERE/'full_native_protocol.py',HERE/'full_support_prepare.py',HERE/'full_prefix.py',HERE/'full_refine.py',
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


def support_population(manifest):
    """The registered native grid and every actual train camera, never train76."""
    descriptor=domain.validate_grid(manifest,manifest['scene'])
    cameras=manifest['splits']['train']
    if len(cameras)!=len(set(cameras)) or set(cameras)!=set(manifest['cameras'])-{'cam00'}:
        raise ValueError('Every actual supplied non-test camera is required once')
    if manifest['splits']['test']!=['cam00'] or manifest['splits']['dev']:
        raise ValueError('Official complete training split with no development holdout required')
    rows=legal_records(manifest)
    calibration=[[camera,frame] for camera in manifest['splits']['train'] for frame in CALIBRATION_FRAMES]
    return dict(hr_size=list(reversed(descriptor['native_HR'])),lr_size=list(reversed(descriptor['native_LR'])),
        full_legal_keys=[[r['camera_id'],int(r['frame_index'])] for r in rows],calibration_keys=calibration,
        training_camera_count=len(manifest['splits']['train']),calibration_anchor_count=len(calibration),
        planned_moment_forwards=len(rows),planned_calibration_RGB_forwards=3*len(calibration),
        planned_calibration_moment_forwards=3*len(calibration))


def domain_registration(manifest,data,readiness):
    if sha(PROPOSAL)!=PROPOSAL_SHA:raise ValueError('Frozen full-domain proposal changed')
    proposal=read(PROPOSAL)
    if proposal.get('schema')!='private_full_native_domain_protocol_proposal_v1':raise ValueError('Foreign domain proposal')
    rows=[row for row in proposal['scenes'] if row['scene']==manifest['scene']]
    if len(rows)!=1:raise ValueError('Scene absent/duplicated in frozen domain proposal')
    configuration,_=domain.configuration(manifest['scene'],readiness)
    if configuration!=rows[0]['author_configuration']:raise ValueError('Configuration differs from frozen scene-domain proposal')
    return dict(domain_protocol=data['domain_protocol'],frozen_domain_proposal=entry(PROPOSAL),
        required_parent_state=domain.expected_parent_state(configuration))


def build_plan(a):
    """CPU identity registration; X dependencies are required only when selected."""
    if a.seed not in domain.SEEDS or a.method not in METHODS:raise ValueError('Unregistered full seed/method')
    readiness=local(getattr(a,'readiness',domain.READINESS))
    m,data,_=domain.validate_manifest(local(a.manifest),a.seed,readiness)
    rows=legal_records(m);population=support_population(m)
    registration=domain_registration(m,data,readiness)
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
            planned_calibration_RGB_forwards=0,planned_calibration_moment_forwards=0,
            Adam_calls=0,parameter_updates=0,HR_reference_images_read=False,heldout_pixels_read=False,**registration)
    if not a.parent or not a.parent_complete or not local(a.parent).is_file() or not local(a.parent_complete).is_file():
        missing.append(dict(dependency='complete_independent_native_LR_parent'))
    else:
        parent,parent_plan=refine.full_parent(a.parent,a.parent_complete,data['manifest'],a.seed)
        domain.validate_parent_state(parent['state'],parent_plan['configuration'])
        if parent['state']!=registration['required_parent_state']:raise ValueError('Parent batch/RGB count differs from registered domain')
        expected,_=domain.configuration(m['scene'],readiness)
        if parent_plan['configuration']!=expected:raise ValueError('Parent configuration differs from registered domain')
    table=available('schedule',a.schedule)
    base=dict(manifest=data['manifest'],parent=parent['checkpoint'] if parent else None)
    context=dict(base,schedule=deps.get('schedule'));edges=[]
    if table is not None:
        if parent is None:missing.append(dict(dependency='schedule_native_parent_binding'))
        else:
            refine.validate_schedule(table,m,base,a.seed);edges=all_edges([table])
            if {(s,f) for s,t,f in edges}!={(r['camera_id'],int(r['frame_index'])) for r in rows}:
                raise ValueError('Selected schedule does not cover every legal full source observation')
    return dict(schema=SCHEMA,status='pending_selected_full_native_support_dependencies' if missing else 'registered_ready_selected_full_native_support',
        scene=m['scene'],seed=a.seed,method=a.method,requires_X=True,missing=missing,manifest=data['manifest'],
        parent=parent,dependencies=deps,full_native_context=context,source_files=sources,
        configuration=parent_plan['configuration'] if parent_plan else None,author_commit=prefix.PIN,
        representation=refine.REPRESENTATION,**population,**registration,
        planned_edges=len(edges),planned_RGB_forwards=0,Adam_calls=0,parameter_updates=0,teacher_reads=0,
        HR_reference_images_read=False,heldout_pixels_read=False,
        implementation='registered population/path adapter of unchanged frozen support_cache.prepare body',
        calibration_contract=dict(entry='full_refine_registered.py --mode calibrate --method X or MX',
            balanced_keys='every actual training camera x 0/100/200/299; Cook80/Cut76/Meet48/Coffee68',
            tau='.75 quantile of positive finite supported HR c_z; floor1e-6',
            lambda_X='.1 median mean-three-LR/X summed-native-index xyz RMS; no coefficient grid',
            kappa='median anchor RGB L1/MSE RMS',Adam_calls=0,parameter_updates=0,calibration_separate=True,
            support_index='frozen/index.json',parent_buffers_must_not_be_reset=True,
            old_short_train76_or_1140_moments_reusable=False))


def verify_plan(plan,upstream):
    if plan.get('schema')!=SCHEMA:raise ValueError('Short/unregistered support plan refused')
    if source_files(upstream)!=plan['source_files']:raise ValueError('Frozen full support source tree changed')
    manifest=read(bound(plan['manifest']));population=support_population(manifest)
    if plan.get('scene')!=manifest['scene'] or plan.get('seed') not in domain.SEEDS:raise ValueError('Registered full scene/seed differs')
    if plan.get('method') not in METHODS or plan.get('requires_X')!=ARMS[plan['method']]['X']:
        raise ValueError('Selected method and X dependency contract differ')
    if plan.get('status')=='registered_ready_selected_full_native_support' and (not plan.get('requires_X') or not plan.get('parent') or not {'selection','schedule'}<=set(plan['dependencies'])):
        raise ValueError('Ready support requires the selected complete native parent and schedule')
    if plan.get('frozen_domain_proposal')!=entry(PROPOSAL) or sha(PROPOSAL)!=PROPOSAL_SHA:raise ValueError('Domain proposal identity changed')
    descriptor=domain.descriptor(manifest['scene']);d=plan['domain_protocol']
    if d.get('scene')!=manifest['scene'] or d.get('schema')!=descriptor['schema'] or d.get('seed')!=plan['seed']:
        raise ValueError('Registered full domain/seed identity differs')
    if any(d.get(key)!=value for key,value in descriptor.items()):raise ValueError('Registered domain interface changed')
    for key in ('protocol_source','readiness','legacy_accepted_prefix_source'):bound(d[key])
    configuration,sources=domain.configuration(manifest['scene'],bound(d['readiness']))
    if d['configuration_sha256']!=domain.digest(configuration):raise ValueError('Registered domain configuration digest differs')
    if d['configuration_sources']!={key:entry(sources[key]) for key in descriptor['configuration_paths']}:
        raise ValueError('Registered pinned configuration sources differ')
    if plan['required_parent_state']!=domain.expected_parent_state(configuration):raise ValueError('Registered full budget differs')
    for ref in plan['dependencies'].values():bound(ref)
    if plan.get('requires_X'):
        if not ARMS[plan['method']]['X']:raise ValueError('Non-X selection cannot allocate moment support')
        for key,value in population.items():
            if plan.get(key)!=value:raise ValueError('Registered full support population/grid differs: '+key)
        if plan.get('parent'):
            for key in ('checkpoint','complete','sidecar','plan'):bound(plan['parent'][key])
            domain.validate_parent_state(plan['parent']['state'],plan['configuration'])
            if plan['parent']['state']!=plan['required_parent_state']:raise ValueError('Parent registered domain budget differs')

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
    if not out.is_relative_to(OUTPUT_ROOT.resolve()):raise ValueError('Production support output must be an independent output/full_support_registered revision')
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
    p.add_argument('--readiness',type=Path,default=domain.READINESS)
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


# CPU contracts below parse metadata/source only; native GPU acceptance is separate.
def cpu_contract():
    """Metadata, dimensions and source contracts; no tensor/model imports."""
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError("CPU contracts require CUDA_VISIBLE_DEVICES=''")
    import ast
    import copy
    from unittest.mock import patch
    if 'torch' in sys.modules or any(k=='scene' or k.startswith('scene.') for k in sys.modules):
        raise RuntimeError('Metadata contracts must not import tensors or models')
    out=OUT/'operator_checks/full_support_registered'/str(time.time_ns());out.mkdir(parents=True)
    (out/'full_support_prepare_registered.py').write_bytes(Path(__file__).read_bytes())
    started=time.monotonic();cpu_started=time.process_time();checks=[]
    record=dict(status='running_CPU_metadata_contract',source=entry(Path(__file__)),GPU_calls=0,Adam_calls=0,
        tensor_or_model_imports=False,actual_moment_or_support_generation=False,production_selection_performed=False)
    def check(name,call):
        tick=time.monotonic();result=call();checks.append(dict(name=name,seconds=time.monotonic()-tick,result=result))
    def refuses(call):
        try:call()
        except (ValueError,AssertionError) as exc:return type(exc).__name__
        raise AssertionError('Expected refusal')
    manifests={}
    try:
        def frozen_computation():
            old=HERE/'full_support_prepare.py'
            assert sha(old)=='fc5431cd35304db6723f64ef855413796eb468fdaf13fb0d53732bf1ee26f8d1'
            definitions=lambda p:{n.name:n for n in ast.parse(p.read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
            previous=definitions(old);current=definitions(Path(__file__))
            names=('legal_records','all_edges','freeze_sources','preserve_incomplete','pixel_boundary','core_adapter',
                'native_camera','export_parent','prepare_support','registered')
            for name in names:assert ast.dump(previous[name],include_attributes=False)==ast.dump(current[name],include_attributes=False),name
            frozen=OUT/'operator_checks/full_support/1791383904176820611/synthetic_cache/source_snapshot/project/experiments/dynamic_sr_multiview_footprint_20261007/support_cache.py'
            assert sha(HERE/'support_cache.py')==sha(frozen)==FROZEN_CORE_SHA
            return dict(exact_unchanged_function_AST=list(names),no_formula_or_native_export_body_changed=True,
                original_frozen_formula_snapshot=entry(frozen),current_formula_source=entry(HERE/'support_cache.py'))
        check('unchanged_native_raw_HR_and_frozen_support_computation_AST',frozen_computation)
        for scene,anchors,count in (('cook_spinach',80,6000),('cut_roasted_beef',76,5700),
                ('meetroom_discussion',48,3600),('meetroom_vrheadset',48,3600),('coffee_martini',68,5100)):
            manifest=ROOT/'data/dynamic_sr/full_time_20261007/prepared'/scene/'manifest_train_ready_seed20261007.json'
            value=read(manifest);manifests[scene]=(manifest,value)
            def population_case(value=value,anchors=anchors,count=count,manifest=manifest):
                result=support_population(value)
                assert result['calibration_anchor_count']==anchors and result['planned_moment_forwards']==count
                assert result['planned_calibration_RGB_forwards']==result['planned_calibration_moment_forwards']==anchors*3
                assert len(result['full_legal_keys'])==count and len({tuple(k) for k in result['calibration_keys']})==anchors
                assert {'cam01'}<={k[0] for k in result['calibration_keys']} and 'cam00' not in {k[0] for k in result['full_legal_keys']}
                return dict(manifest=entry(manifest),LR_size_hw=result['lr_size'],HR_size_hw=result['hr_size'],
                    train_cameras=result['training_camera_count'],anchors=anchors,moments=count,calibration_RGB=anchors*3)
            check('actual_full_manifest_grid_and_calibration_population_'+scene,population_case)
            def budget(scene=scene):
                cfg,_=domain.configuration(scene);state=domain.expected_parent_state(cfg)
                assert state['accepted_RGB']==(68000 if scene in domain.MEETROOM+('coffee_martini',) else 34000)
                assert state['accepted_backward']==17000 and state['accepted_Adam']==16999
                bad=copy.deepcopy(state);bad['accepted_RGB']=34000 if state['accepted_RGB']==68000 else 68000
                refuses(lambda:domain.validate_parent_state(bad,cfg))
                return dict(expected=state,wrong_batch_RGB_count_refused=True)
            check('registered_author_batch_prefix_budget_'+scene,budget)
        meet_path,meet=manifests['meetroom_discussion'];cook_path,cook=manifests['cook_spinach']
        for name,mutate in (
            ('foreign_Cook_grid_for_MeetRoom',lambda m:m['resolutions'].update(lr=[336,252],hr=[1344,1008])),
            ('wrong_pixel_center',lambda m:m['cameras']['cam01']['K_lr'][0].__setitem__(2,m['cameras']['cam01']['K_lr'][0][2]+.5)),
            ('duplicate_train_camera',lambda m:m['splits']['train'].append('cam01')),
            ('test_camera_as_train',lambda m:m['splits']['train'].append('cam00')),
            ('short60_frames',lambda m:m.__setitem__('frame_indices',list(range(60)))),
            ('missing_full_observation',lambda m:m['observations'].pop()),
            ('cam01_dev_holdout',lambda m:(m['splits']['train'].remove('cam01'),m['splits'].__setitem__('dev',['cam01'])))):
            bad=copy.deepcopy(meet);mutate(bad);check('reject_'+name,lambda bad=bad:refuses(lambda:support_population(bad)))
        readiness=domain.READINESS;configuration,sources=domain.configuration(meet['scene'])
        descriptor=domain.descriptor(meet['scene'])
        identity=dict(**descriptor,seed=20261007,configuration_sources={key:entry(sources[key]) for key in descriptor['configuration_paths']},
            configuration_sha256=domain.digest(configuration),readiness=entry(readiness),protocol_source=entry(HERE/'full_native_protocol.py'),
            legacy_accepted_prefix_source=entry(HERE/'full_prefix.py'))
        data=dict(manifest=entry(meet_path),domain_protocol=identity)
        args=SimpleNamespace(manifest=meet_path,seed=20261007,readiness=readiness,upstream=None,method='X',selection=None,
            parent=None,parent_complete=None,schedule=None,out=out/'no_production_dispatch',phase='all',support_device='cuda:0',
            operator_resource_resolved=False,gpu_uuid=None)
        with patch.object(domain,'validate_manifest',return_value=(meet,data,[])):
            X=build_plan(args)
            def pending_X():
                assert X['status']=='pending_selected_full_native_support_dependencies' and X['requires_X']
                assert X['required_parent_state']['accepted_RGB']==68000
                assert X['calibration_anchor_count']==48 and X['hr_size']==[720,1280] and X['lr_size']==[180,320]
                assert {m['dependency'] for m in X['missing']}=={'selection','schedule','complete_independent_native_LR_parent'}
                refuses(lambda:prepare(args,X));assert not args.out.exists()
                return dict(pending_before_CUDA=True,new_native_parent_required=True,no_short_moment_or_train76_fallback=True)
            check('X_requires_selected_parent_schedule_before_generation',pending_X)
            args.method='M';M=build_plan(args)
            def no_X():
                assert M['planned_moment_forwards']==M['planned_edges']==M['planned_calibration_moment_forwards']==0
                assert {m['dependency'] for m in M['missing']}=={'selection'}
                return dict(non_X_no_parent_or_schedule_dependency=True,no_moment_cache_or_X_calibration_planned=True)
            check('unselected_X_no_extra_geometry_dependency',no_X)
            fixture=out/'selection_fixture.json';write(fixture,dict(status='CPU_contract_fixture_not_a_development_selection'))
            args.selection=fixture
            with patch.object(refine,'validate_selection',side_effect=lambda value,method:None):selected_M=build_plan(args)
            result=prepare(args,selected_M);assert result['GPU_calls']==0 and result['Adam_calls']==0 and not args.out.exists()
            check('selected_non_X_skips_all_generation',lambda:dict(**result,synthetic_selection_validation_stub=True))
            check('real_selection_validator_refuses_CPU_fixture',lambda:refuses(lambda:refine.validate_selection(read(fixture),'M')))
            args.seed=7;check('unregistered_seed_refused',lambda:refuses(lambda:build_plan(args)))
        check('verify_pending_registered_metadata_identity',lambda:(verify_plan(X,Path('/home/cai_tianshun/Project/4dgs')),True)[1])
        bad=copy.deepcopy(X);bad['domain_protocol']['seed']=20261008
        check('verify_foreign_seed_rejected',lambda:refuses(lambda:verify_plan(bad,Path('/home/cai_tianshun/Project/4dgs'))))
        bad=copy.deepcopy(X);bad['source_files']['project'][str(Path(__file__).relative_to(ROOT))]='0'*64
        check('verify_source_drift_rejected',lambda:refuses(lambda:verify_plan(bad,Path('/home/cai_tianshun/Project/4dgs'))))
        bad=copy.deepcopy(X);bad['schema']='selected_full_native_X_support_prepare_v1'
        check('old_unregistered_support_schema_refused',lambda:refuses(lambda:verify_plan(bad,Path('/home/cai_tianshun/Project/4dgs'))))
        bad=copy.deepcopy(X);bad['calibration_keys']=bad['calibration_keys'][:-1]
        check('unbalanced_calibration_population_refused',lambda:refuses(lambda:verify_plan(bad,Path('/home/cai_tianshun/Project/4dgs'))))
        bad=copy.deepcopy(X);bad['status']='registered_ready_selected_full_native_support'
        check('missing_dependencies_cannot_be_marked_ready',lambda:refuses(lambda:verify_plan(bad,Path('/home/cai_tianshun/Project/4dgs'))))
        bad=copy.deepcopy(X);bad['requires_X']=False
        check('method_X_dependency_mismatch_refused',lambda:refuses(lambda:verify_plan(bad,Path('/home/cai_tianshun/Project/4dgs'))))
        parent=out/'old_short_parent_index.json';write(parent,dict(scope='short60',full_native_context={},entries=[]))
        check('old_short_parent_moments_refused_before_core',lambda:refuses(lambda:prepare_support(X,out,None,parent,{},'cpu')))
        assert 'torch' not in sys.modules and not any(k=='scene' or k.startswith('scene.') for k in sys.modules)
        record.update(status='passed_CPU_metadata_contract_native_GPU_acceptance_pending',checks=checks,
            scope='Actual full manifest metadata plus controlled dependency fixtures; no LR/HR pixel SHA sweep or tensor/model/native render performed',
            calibration_scope_correction='Cook full20x4=80; Cut19x4=76; Meet12x4=48; Coffee17x4=68. Historical short train76/calibration assets unchanged.',
            native_parent_models_or_independent_training_completed=False,domain_proposal=entry(PROPOSAL),unchanged_old_support_source=entry(HERE/'full_support_prepare.py'))
    except BaseException as exc:
        record.update(status='failed_CPU_metadata_contract_preserved',checks=checks,error=repr(exc),traceback=traceback.format_exc());raise
    finally:
        record.update(wall_seconds=time.monotonic()-started,process_CPU_seconds=time.process_time()-cpu_started)
        write(out/'receipt.json',record)
    return dict(**record,receipt=entry(out/'receipt.json'))


if __name__=='__main__':main()
