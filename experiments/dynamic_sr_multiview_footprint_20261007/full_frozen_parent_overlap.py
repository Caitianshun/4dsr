"""Fixed coarse native-parent visibility estimates for full-time assignment.

This producer renders no RGB and decodes no image. It is an explicitly
registered parent-model approximation, not ground-truth geometry or X support.
Metadata planning is stdlib-only; CUDA preparation requires explicit owned execution.
This portable source has a distinct SHA and creates new plans; it does not rewrite
or reuse plans registered by the private frozen producer. FOURDSR_ORIGIN_ROOT
optionally maps original absolute input references without rewriting source records.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import socket
import sys
import time
import traceback
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = ROOT/'output/dynamic_sr_multiview_footprint_20261007'
OUTPUT_ROOT = OUT/'native_full_parent_coarse_overlap_20261007'
ORIGIN = Path(os.environ.get('FOURDSR_ORIGIN_ROOT', str(ROOT))).resolve()
PLAN_SCHEMA = 'registered_frozen_native_parent_coarse_overlap_plan_v2'
OBSERVER_RNG_POLICY = dict(name='initialize_CUDA_before_observer_snapshot_then_exact_selfcheck_v2',
    cuda_initialization_is_actual=True,fail_fast_before_native_model_restore=True,
    native_parent_RNG_checked_before_observer_restore=True,serialized_observer_snapshot=True)
INDEX_SCHEMA = 'frozen_native_parent_coarse_overlap_v1'
POLICY = dict(name='native_LR_quarter_coarse_visibility_v1',coarse_LR_divisor=4,
    epsilon=1e-6,alpha_support_min=1e-3,behind_relative=0.02,behind_sigma=3.0,
    source_support='all nonzero bilinear taps supported; zero taps impose no constraint',
    normalization='raw native A/M1/M2; Z=M1/(A+epsilon), variance=max(M2/(A+epsilon)-Z^2,0)',
    unknown_variance='no explicit hard occlusion; count separately',
    directed_ratio='visible supported target coarse pixels / all supported target coarse pixels',
    pair_ratio='minimum of the two directed ratios; null if either denominator is zero',
    interpretation='frozen parent estimate, not ground truth; no HR-error or quality selection')
PINS = {
    'full_refine_registered.py':'9662c17f32212d6fdbd8c0f1a0cf405274b5239eb1a1a3b48acd6d60a11b6d09',
    'full_prefix_registered.py':'92a19ed94c2bf7dab881341375c4bdc486e6f4032536e977592c7911c3ce6605',
    'full_native_protocol.py':'cb4773700bbf535ebb0d9cc94c51f06ca457e04bf305d65ef9afe911746d6713',
    'footprint.py':'3c2645449d065e032fef38cefd1e269ec971e86735a86ef28aefc1677819c789'}
DEPTH = ROOT/'experiments/dynamic_sr_confidence_geometry_20261006/depth_prior.py'
DEPTH_SHA = 'a9979ec78e40fee06178c1a1b25c8d60a8e9078263f2ef321bee43f0ad2735aa'
PIXELS = {'.png','.jpg','.jpeg','.exr','.tif','.tiff','.npy','.npz'}

def require(condition,message):
    if not condition: raise ValueError(message)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):h.update(block)
    return h.hexdigest()
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n');temporary.replace(path)
def local(path):
    path=Path(path)
    if path.is_absolute():
        try:path=ROOT/path.relative_to(ORIGIN)
        except ValueError:pass
    else:path=ROOT/path
    require(path.resolve().is_relative_to(ROOT.resolve()),'Input/output path outside execution workspace')
    return path
def entry(path):
    path=local(path);return dict(path=str(path.relative_to(ROOT)),sha256=sha(path))
def bound(ref):
    require(set(ref)=={'path','sha256'},'Malformed immutable entry')
    path=local(ref['path']);require(sha(path)==ref['sha256'],'Bound bytes changed: '+str(path));return path
def science():
    for name,value in PINS.items():require(sha(HERE/name)==value,'Frozen science source differs: '+name)
    require(sha(DEPTH)==DEPTH_SHA,'Frozen moment renderer differs')
    if str(HERE) not in sys.path:sys.path.insert(0,str(HERE))
    import full_refine_registered as refine
    import full_prefix_registered as prefix
    import full_native_protocol as domain
    return refine,prefix,domain

def coarse_size(lr_size):
    require(len(lr_size)==2 and list(lr_size) in ([336,252],[320,180]),'Unregistered native grid')
    require(all(int(x)==x and x%4==0 for x in lr_size),'Coarse grid must be exact LR/4')
    return [int(x)//4 for x in lr_size]
def scaled_K(matrix,old_size,new_size):
    require(len(matrix)==3 and all(len(row)==3 for row in matrix),'Invalid K')
    require(all(math.isfinite(float(v)) for row in matrix for v in row),'Nonfinite K')
    result=[[float(v) for v in row] for row in matrix]
    sx,sy=new_size[0]/old_size[0],new_size[1]/old_size[1]
    for j in range(3):result[0][j]*=sx;result[1][j]*=sy
    result[0][2]=(float(matrix[0][2])+.5)*sx-.5
    result[1][2]=(float(matrix[1][2])+.5)*sy-.5
    return result
def legal_rows(manifest):
    cameras=list(manifest['splits']['train'])
    require(manifest['splits']['test']==['cam00'] and not manifest['splits']['dev'] and 'cam00' not in cameras and 'cam01' in cameras,'Official train-only population required')
    require(len(cameras)==len(set(cameras)) and set(cameras)==set(manifest['cameras'])-{'cam00'},'Missing/duplicate train camera')
    require(manifest['frame_indices']==list(range(300)),'Full 0..299 required')
    rows=[r for r in manifest['observations'] if r['split']=='train']
    keys=[(r['camera_id'],int(r['frame_index'])) for r in rows]
    require(len(keys)==len(set(keys)) and set(keys)=={(c,f) for c in cameras for f in range(300)},'Incomplete/foreign legal keys')
    require(all(math.isfinite(float(r['time'])) and abs(float(r['time'])-int(r['frame_index'])/300)<1e-7 for r in rows),'Registered frame/300 time differs')
    return sorted(rows,key=lambda r:(int(r['frame_index']),r['camera_id']))
def validate_identity(plan):
    require(plan.get('schema')==PLAN_SCHEMA and plan.get('status')=='registered_CPU_ready_frozen_parent_coarse_overlap','Pending/foreign plan')
    require(plan.get('policy')==POLICY,'Unregistered overlap policy')
    require(plan.get('observer_RNG_policy')==OBSERVER_RNG_POLICY,'Observer RNG lifecycle policy differs')
    require(plan['scene'] in ('cook_spinach','cut_roasted_beef','meetroom_discussion','meetroom_vrheadset') and plan['seed'] in (20261007,20261008),'Only eight authorized development parents')
    require(plan['context']==dict(manifest=plan['data']['manifest'],parent=plan['parent']['checkpoint']),'Parent/manifest context differs')
    require(plan['coarse_size']==coarse_size(plan['data']['resolution_LR']),'Coarse grid differs')
    cameras=plan['data']['train_cameras'];n=len(cameras)
    require(plan['planned']==dict(native_moment_forwards=n*300,shared_deformation_states=300,directed_pairs=n*(n-1)*300,
        unordered_pairs=n*(n-1)*150,RGB_forwards=0,autograd_calls=0,Adam_calls=0,parameter_updates=0,
        raw_float32_uncompressed_bytes=n*300*3*math.prod(plan['coarse_size'])*4),'Planned cost/population differs')
    require(plan['author_commit']=='843d5ac636c37e4b611242287754f3d4ed150144','Author pin differs')
    require(plan['boundary']==dict(training_pixel_decodes=0,HR_pixels_read=False,heldout_pixels_read=False,
        teacher_reads=0,quality_based_pair_selection=False,coarse_output_is_X_support=False),'Input/evidence boundary differs')
    return True
def source_files(upstream):
    refine,prefix,_=science();sources=refine.source_files(upstream)
    sources['project'][str(Path(__file__).relative_to(ROOT))]=sha(__file__)
    return sources
def build_plan(a):
    refine,prefix,domain=science()
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU planning must explicitly hide CUDA')
    upstream=Path(a.upstream).resolve();require(os.environ.get('FOURDSR_UPSTREAM')==str(upstream),'Explicit upstream/environment mismatch')
    m,data,inputs=domain.validate_manifest(local(a.manifest),a.seed,local(a.readiness))
    require(m['role']=='already_used_development' and m['scene'] in ('cook_spinach','cut_roasted_beef','meetroom_discussion','meetroom_vrheadset'),'Confirmation needs a later final single-main authorization')
    rows=legal_rows(m)
    parent,parent_plan=refine.full_parent(local(a.parent),local(a.parent_complete),data['manifest'],a.seed)
    require(parent_plan['scene']==m['scene'],'Parent scene differs')
    require(parent_plan['runtime_source_sha256']==source_files(upstream)['upstream'],'Author runtime differs from native parent')
    acceptance=entry(local(a.parent_acceptance));accepted_index=read(local(a.parent_index))
    matches=[r for r in accepted_index['entries'] if r['scene']==m['scene'] and int(r['seed'])==a.seed]
    require(len(matches)==1 and matches[0]['status']=='completed_independent_full_author_LR_prefix','Exact accepted parent entry required')
    require(all(matches[0][k]==v for k,v in [('checkpoint',parent['checkpoint']),('complete',parent['complete']),('exit_acceptance',acceptance)]),'Parent accepted-entry identity differs')
    coarse=coarse_size(m['resolutions']['lr']);n=len(m['splits']['train'])
    plan=dict(schema=PLAN_SCHEMA,status='registered_CPU_ready_frozen_parent_coarse_overlap',scene=m['scene'],seed=a.seed,
        context=dict(manifest=data['manifest'],parent=parent['checkpoint']),data=data,parent=parent,
        parent_acceptance=acceptance,accepted_parent_entry=matches[0],policy=POLICY,observer_RNG_policy=OBSERVER_RNG_POLICY,coarse_size=coarse,
        camera_K_coarse={c:scaled_K(m['cameras'][c]['K_lr'],m['resolutions']['lr'],coarse) for c in m['splits']['train']},
        source_files=source_files(upstream),author_commit=prefix.PIN,readiness=entry(local(a.readiness)),
        legal_LR_SHA_inputs=inputs,legal_keys=[[r['camera_id'],int(r['frame_index'])] for r in rows],
        planned=dict(native_moment_forwards=n*300,shared_deformation_states=300,directed_pairs=n*(n-1)*300,
            unordered_pairs=n*(n-1)*150,RGB_forwards=0,autograd_calls=0,Adam_calls=0,parameter_updates=0,
            raw_float32_uncompressed_bytes=n*300*3*math.prod(coarse)*4),
        boundary=dict(training_pixel_decodes=0,HR_pixels_read=False,heldout_pixels_read=False,teacher_reads=0,
            quality_based_pair_selection=False,coarse_output_is_X_support=False),
        estimated_storage='Raw float32 size only; compressed NPZ/JSON size and time are measured by actual prepare',
        metadata_runtime=dict(python=sys.version.split()[0],numpy_distribution=__import__('importlib.metadata',fromlist=['version']).version('numpy')),
        selection_dependency=False,method='Frozen native LR parent approximation shared by candidate schedules; no new training',
        original_parent_plan=parent['plan'])
    require(plan['metadata_runtime']['numpy_distribution']=='1.26.4','Use native TRAINPY NumPy1.26.4 for exact metadata registration')
    validate_identity(plan);return plan
def verify_plan(plan,upstream):
    validate_identity(plan)
    require(source_files(upstream)==plan['source_files'],'Source tree changed')
    for key in ('checkpoint','complete','plan','sidecar'):bound(plan['parent'][key])
    bound(plan['parent_acceptance']);bound(plan['readiness'])
    for ref in plan['legal_LR_SHA_inputs']:require(sha(local(ref['path']))==ref['sha256'],'Legal LR bytes changed')
    m=read(bound(plan['context']['manifest']));rows=legal_rows(m)
    require([[r['camera_id'],int(r['frame_index'])] for r in rows]==plan['legal_keys'],'Legal population changed')
    require({c:scaled_K(m['cameras'][c]['K_lr'],m['resolutions']['lr'],plan['coarse_size']) for c in m['splits']['train']}==plan['camera_K_coarse'],'Camera metadata changed')
    p=read(bound(plan['parent']['plan']))
    require(p['seed']==plan['seed'] and p['data']['manifest']==plan['context']['manifest'] and p['runtime_source_sha256']==plan['source_files']['upstream'],'Native parent provenance changed')
    parent,_=science()[0].full_parent(bound(plan['parent']['checkpoint']),bound(plan['parent']['complete']),plan['context']['manifest'],plan['seed'])
    require(parent==plan['parent'],'Complete native parent state/audit/provenance differs')
    require(plan['metadata_runtime']==dict(python=sys.version.split()[0],numpy_distribution=__import__('importlib.metadata',fromlist=['version']).version('numpy')),'Planning and native preparation CPU metadata runtime differs')
    return m

def normalize_raw(raw):
    """Finite-before-division native moment normalization; variance unknown is separate."""
    import torch
    require(raw.ndim==3 and raw.shape[0]==3 and raw.is_floating_point(),'Expected raw [A,M1,M2]')
    alpha,m1,m2=raw.detach().unbind(0)
    valid=torch.isfinite(alpha)&torch.isfinite(m1)&(alpha>POLICY['alpha_support_min'])
    denominator=torch.where(valid,alpha,torch.ones_like(alpha))+POLICY['epsilon']
    z=torch.where(valid,m1,torch.zeros_like(m1))/denominator
    support=valid&torch.isfinite(z)&(z>0)
    z=torch.where(support,z,torch.zeros_like(z))
    second=torch.where(support&torch.isfinite(m2),m2,torch.zeros_like(m2))/denominator
    variance=second-z.square()
    known=support&torch.isfinite(m2)&torch.isfinite(variance)
    variance=torch.where(known,variance.clamp_min(0),torch.zeros_like(variance))
    return dict(Z=z,variance=variance,support=support,variance_known=known,
        nonfinite_raw_elements=int((~torch.isfinite(raw)).sum()))
def sample_bilinear(packet,xy):
    """Every positive-weight tap must be in bounds and supported."""
    import torch
    require(xy.ndim==3 and xy.shape[-1]==2,'Expected projected HW2 pixels')
    finite=torch.isfinite(xy).all(-1);xy=torch.where(finite[...,None],xy,torch.zeros_like(xy))
    height,width=packet['Z'].shape
    # Far out-of-frame coordinates are masked before conversion to int64.
    representable=finite&(xy[...,0]>=-1)&(xy[...,0]<=width)&(xy[...,1]>=-1)&(xy[...,1]<=height)
    xy=torch.where(representable[...,None],xy,torch.zeros_like(xy));base=xy.floor()
    frac=xy-base;ix,iy=base[...,0].long(),base[...,1].long();fx,fy=frac.unbind(-1)
    values=torch.zeros_like(xy[...,0]);variance=torch.zeros_like(values)
    valid=representable.clone();known=representable.clone()
    for dx,dy,weight in ((0,0,(1-fx)*(1-fy)),(1,0,fx*(1-fy)),(0,1,(1-fx)*fy),(1,1,fx*fy)):
        x,y=ix+dx,iy+dy;active=weight>0
        inside=(x>=0)&(x<width)&(y>=0)&(y<height);xx=x.clamp(0,width-1);yy=y.clamp(0,height-1)
        valid&=~active|(inside&packet['support'][yy,xx]);known&=~active|(inside&packet['variance_known'][yy,xx])
        values+=weight*packet['Z'][yy,xx];variance+=weight*packet['variance'][yy,xx]
    valid&=torch.isfinite(values)&torch.isfinite(variance)
    return dict(Z=values,variance=variance,support=valid,variance_known=known&valid)
def directed_overlap(source,target,camera_source,camera_target):
    """Direction is target supported-pixel reprojection into source, not RGB warp."""
    import torch
    from footprint import project
    p=project(camera_source,camera_target,target['Z']);s=sample_bilinear(source,p['xy'])
    target_support=target['support'];geometry=target_support&p['valid']&s['support']
    known=geometry&s['variance_known']
    margin=torch.maximum(POLICY['behind_relative']*s['Z'],POLICY['behind_sigma']*s['variance'].sqrt())
    occluded=known&(p['source_z']>s['Z']+margin)
    visible=geometry&~occluded;den=int(target_support.sum());num=int(visible.sum())
    return dict(ratio=num/den if den else None,visible_target_pixels=num,supported_target_pixels=den,
        projection_invalid_pixels=int((target_support&~p['valid']).sum()),
        source_footprint_unsupported_pixels=int((target_support&p['valid']&~s['support']).sum()),
        explicit_occlusion_pixels=int(occluded.sum()),unknown_source_variance_pixels=int((geometry&~known).sum()))
def pair_overlap(left,right,camera_left,camera_right):
    # a_to_b describes supported a pixels visible in b.
    ab=directed_overlap(right,left,camera_right,camera_left)
    ba=directed_overlap(left,right,camera_left,camera_right)
    ratios=[ab['ratio'],ba['ratio']]
    return dict(a_to_b=ab,b_to_a=ba,ratio=min(ratios) if all(x is not None for x in ratios) else None)
def pose_pair(cal_a,cal_b):
    ca,cb=cal_a['c2w'],cal_b['c2w'];a=[float(ca[i][3]) for i in range(3)];b=[float(cb[i][3]) for i in range(3)]
    axis_a=[float(ca[i][2]) for i in range(3)];axis_b=[float(cb[i][2]) for i in range(3)]
    norm=math.sqrt(sum(x*x for x in axis_a)*sum(x*x for x in axis_b))
    require(norm>0 and math.isfinite(norm),'Degenerate optical axis')
    cosine=max(-1.,min(1.,sum(x*y for x,y in zip(axis_a,axis_b))/norm))
    return dict(camera_center_distance=math.sqrt(sum((x-y)**2 for x,y in zip(a,b))),optical_axis_angle_radians=math.acos(cosine))

def input_inventory(plan,upstream):
    """Explicit deploy roles; never recursively follow manifest HR/test references."""
    m=read(bound(plan['context']['manifest']));init=read(bound(plan['data']['initialization_receipt']))
    refs=[('manifest',plan['data']['manifest']),('original_manifest',plan['data']['original_manifest']),
        ('initialization_receipt',plan['data']['initialization_receipt']),('initial_points_metadata_SHA_only',plan['data']['initial_points']),
        ('readiness',plan['readiness']),('parent_acceptance',plan['parent_acceptance'])]
    refs += [('native_parent_'+key,plan['parent'][key]) for key in ('checkpoint','complete','plan','sidecar')]
    refs += [('initialization_artifact_SHA_only',ref) for ref in init['artifacts']]
    refs += [('source',ref) for ref in init['identity'].values() if isinstance(ref,dict) and set(ref)=={'path','sha256'}]
    refs += [('source',dict(path=path,sha256=value)) for path,value in plan['source_files']['project'].items()]
    refs += [('author_configuration',ref) for ref in read(bound(plan['parent']['plan']))['author_sources'].values()]
    # Frozen legacy full_parent consults this canonical accepted-entry table.
    refs += [('accepted_native_parent_metadata_index',entry(OUT/'full_LR_prefixes/checkpoint_index.json'))]
    refs += [('legal_train_LR_SHA_only',dict(path=i['path'],sha256=i['sha256'])) for i in plan['legal_LR_SHA_inputs']]
    rows={}
    for role,ref in refs:
        p=bound(ref);key=ref['path']
        require(p.suffix.lower() not in {'.png','.jpg','.jpeg'} or role=='legal_train_LR_SHA_only','Image outside legal LR inventory')
        if key not in rows:rows[key]=dict(**ref,bytes=p.stat().st_size,roles=[])
        require(rows[key]['sha256']==ref['sha256'],'Conflicting source entry');rows[key]['roles'].append(role)
    return dict(schema='frozen_parent_coarse_overlap_input_inventory_v1',context=plan['context'],
        entries=sorted(rows.values(),key=lambda x:x['path']),external_author=dict(root=str(Path(upstream).resolve()),
            entries=[dict(relative=name,sha256=value) for name,value in plan['source_files']['upstream'].items()]),
        runtime_pixel_decodes=0,HR_test_teacher_assets_included=False,
        note='Legal LR and initial arrays are bytes-hashed by the frozen CPU metadata validator, not decoded/rendered by this producer.')

def ownership(authorization,plan_ref,uuid,lock):
    require(authorization.get('schema')=='root_authorized_frozen_native_parent_coarse_overlap_v1','Root operational authorization required')
    require(authorization['plan']==plan_ref and authorization['producer_source']==entry(__file__) and authorization['gpu_uuid']==uuid,'Authorized plan/source/UUID differs')
    require(authorization['runtime_root']==str(ROOT) and authorization['physical_gpu_lock']==str(lock),'Authorized root/physical lock differs')
    unit=authorization['unit'];require(unit.endswith('.service') and '/' not in unit,'Owned service unit required')
    cgroup=Path('/proc/self/cgroup').read_text();require(cgroup.count('\n')==1 and cgroup.startswith('0::/') and cgroup.rstrip('\n').endswith('/'+unit),'Process is not in the exact owned unit')
    argv=[os.fsdecode(v) for v in Path('/proc/self/cmdline').read_bytes().rstrip(b'\0').split(b'\0')]
    require(argv==authorization['manager_command'],'Owned executable/argv differs')
    require(len(argv)>=3 and argv[1]=='-u' and Path(argv[0]).is_absolute(),'Exact unbuffered native interpreter command required')
    manager_source=authorization.get('manager_source',authorization['producer_source'])
    require(Path(argv[2]).resolve()==bound(manager_source).resolve(),'Owned manager source differs')
    require(Path('/proc/self/exe').resolve()==Path(argv[0]).resolve(),'Owned interpreter executable differs')
    require(os.environ.get('CUDA_VISIBLE_DEVICES')==uuid and os.environ.get('FOURDSR_ROOT')==str(ROOT),'Actual root/CUDA environment differs')
    return dict(pid=os.getpid(),cmdline=argv,cgroup=cgroup,unit=unit,authorization=authorization,manager_source=manager_source,
        resource_adapter=authorization.get('resource_adapter'),check_gpu_policy='frozen strict all-foreign-compute rejection unless separately root-bound exact resource adapter')
@contextmanager
def physical_lock(path):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a+') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:yield
        finally:fcntl.flock(stream,fcntl.LOCK_UN)
def resource_api_source(prefix,authorization):
    original=Path(prefix.__file__).resolve();actual=Path(prefix.check_gpu.__code__.co_filename).resolve()
    if actual==original:
        require(authorization.get('resource_adapter') is None,'Declared resource adapter is not installed')
        return dict(resource_API='frozen_prefix_strict_all_foreign_compute',source=entry(original))
    adapter=authorization.get('resource_adapter')
    require(adapter is not None and bound(adapter).resolve()==actual,'In-memory resource API substitution lacks exact root-bound source')
    return dict(resource_API='explicit_root_resource_adapter',source=adapter,
        interpretation='Operational GPU-occupancy API adapted; native model/moment/projection source bytes remain frozen')
def pixel_boundary(directory):
    directory=directory.resolve();opened=dict(own_arrays=0,image_pixels=0)
    def guard(event,args):
        if event!='open' or not args or not isinstance(args[0],(str,bytes,os.PathLike)):return
        p=Path(os.fsdecode(args[0]));suffix=p.suffix.lower()
        if suffix not in PIXELS:return
        p=p.resolve()
        require(suffix=='.npz' and p.is_relative_to(directory) and p.parent==directory/'raw_moments','Image/array read outside owned coarse moments')
        opened['own_arrays']+=1
    return guard,opened
def coarse_camera(manifest,row,uid,size):
    import numpy as np
    import torch
    from n3dv_data import observation_to_4dgs_camera
    cal=manifest['cameras'][row['camera_id']]
    item=dict(image=torch.zeros(3,size[1],size[0]),time=float(row['time']),camera_id=row['camera_id'],
        frame_index=int(row['frame_index']),width=size[0],height=size[1],
        K=np.asarray(scaled_K(cal['K_lr'],manifest['resolutions']['lr'],size),np.float64),w2c=np.asarray(cal['w2c'],np.float64))
    return observation_to_4dgs_camera(item,uid)
def immutable(g,prefix,refine,torch,np):
    params=prefix.parameter_map(g)
    return dict(native_capture=refine.tree_digest(g.capture()),Adam=refine.tree_digest(g.optimizer.state_dict()),
        gradients=refine.tree_digest({k:p.grad for k,p in params.items()}),
        buffers=refine.tree_digest({k:getattr(g,k) for k in ('_deformation_accum','_deformation_table','max_radii2D','xyz_gradient_accum','denom')}),
        RNG=refine.tree_digest(prefix.rng_state(torch,np)),versions={k:p._version for k,p in params.items()})
def atomic_npz(path,raw,np):
    temporary=path.with_name(path.stem+f'.{os.getpid()}.tmp.npz')
    np.savez_compressed(temporary,raw_moments=raw);temporary.replace(path)
def rng_description(value,refine):
    return dict(tree_sha256=refine.tree_digest(value),
        component_sha256={key:refine.tree_digest(value[key]) for key in ('python','numpy','torch','cuda')},
        CUDA_state_count=len(value['cuda']))
def observer_rng_start(torch,np,prefix,refine,out):
    """Actual CUDA init, then prove exact observer restoration before any model/MOM."""
    initialized_before=torch.cuda.is_initialized();tick=time.monotonic()
    torch.cuda.init()
    require(torch.cuda.is_initialized(),'CUDA initialization did not complete')
    observer=prefix.rng_state(torch,np);before=rng_description(observer,refine)
    require(before['CUDA_state_count']==1,'Explicit single UUID must expose one CUDA RNG state')
    path=out/'observer_RNG_initial.pt';temporary=path.with_suffix('.tmp.pt')
    torch.save(observer,temporary);temporary.replace(path)
    prefix.restore_rng(observer,torch,np);after=rng_description(prefix.rng_state(torch,np),refine)
    value=dict(status='passed_exact_observer_RNG_startup_selfcheck' if after==before else 'failed_observer_RNG_startup_selfcheck',
        policy=OBSERVER_RNG_POLICY,initialized_before=initialized_before,initialized_after=torch.cuda.is_initialized(),
        snapshot=entry(path),before=before,after=after,exact_equal=after==before,
        CUDA_init_calls=1,native_model_restore_started=False,native_moment_forwards=0,Adam_calls=0,
        seconds=time.monotonic()-tick)
    proof=out/'observer_RNG_startup_selfcheck.json';write(proof,value)
    require(after==before,'Observer RNG startup selfcheck differs before native model restore')
    return observer,entry(proof)
def observer_rng_restore(observer,torch,np,prefix,refine,out,scope):
    before=rng_description(observer,refine);prefix.restore_rng(observer,torch,np)
    after=rng_description(prefix.rng_state(torch,np),refine)
    value=dict(status='passed_exact_observer_RNG_restoration' if after==before else 'failed_observer_RNG_restoration',
        scope=scope,policy=OBSERVER_RNG_POLICY,before=before,after=after,exact_equal=after==before)
    path=out/('observer_RNG_'+scope+'.json');write(path,value)
    require(after==before,'Observer RNG restoration differs: '+scope)
    return entry(path)

def validate_result(plan,complete_path):
    """CPU all-SHA/result acceptance; caller separately proves actual unit Exit0."""
    validate_identity(plan);done=read(local(complete_path));index=read(bound(done['index']))
    require(done['status']=='completed_frozen_native_parent_coarse_overlap' and index['schema']==INDEX_SCHEMA and index['status']==done['status'],'Coarse preparation not completed')
    require(done['context']==index['context']==plan['context'] and index['scene']==plan['scene'] and index['seed']==plan['seed'],'Completed parent/manifest/seed differs')
    require(read(bound(index['registration']))==plan and index['policy']==POLICY and index['policy_sha256']==digest(POLICY),'Completed plan/policy differs')
    require(index['source_files']==plan['source_files'] and index['source']==done['source']==entry(__file__),'Completed source identity differs')
    require(index['coarse_size']==plan['coarse_size'] and index['total_native_moment_forwards']==plan['planned']['native_moment_forwards'],'Completed grid/count differs')
    require(index['prior_reused_moments']==0 and index['RGB_forwards']==index['Adam_calls']==index['formal_updates']==index['autograd_calls']==0,'Unexpected model/optimizer operations')
    native_proof=read(bound(index['invariants']['native_invariance']))
    require(native_proof==dict(before=index['invariants']['before'],after=index['invariants']['after'],exact_equal=True,scope='native_parent_before_observer_restore'),'Native invariant closure differs')
    observer_checks=index['invariants']['observer_RNG_checks']
    startup=read(bound(observer_checks['startup']));restoration=read(bound(observer_checks['restoration']))
    bound(startup['snapshot']);require(startup['status']=='passed_exact_observer_RNG_startup_selfcheck' and startup['policy']==OBSERVER_RNG_POLICY and startup['exact_equal'] is True and startup['before']==startup['after'] and startup['native_moment_forwards']==0 and startup['native_model_restore_started'] is False and startup['before']['CUDA_state_count']==1,'Observer startup proof incomplete')
    require(restoration['status']=='passed_exact_observer_RNG_restoration' and restoration['policy']==OBSERVER_RNG_POLICY and restoration['scope']=='after_native_parent_check' and restoration['exact_equal'] is True and restoration['before']==restoration['after']==startup['before'],'Observer completion closure differs')
    require(index['invariants']['before']==index['invariants']['after'] and index['invariants']['parent_state_unchanged'] is True and index['invariants']['observer_RNG_restored'] is True,'Native state/RNG not invariant')
    require(all(index['invariants']['restore'].get(k) is True for k in ('one_Adam_exact','parameters_exact','topology_buffers_exact','saved_gradients_exact','global_RNG_exact','no_children')) and index['invariants']['restore'].get('optimizer_conversion') is False,'Native14 restore proof incomplete')
    require(index['training_pixel_decodes']==index['teacher_reads']==0 and index['HR_pixels_read'] is False and index['heldout_pixels_read'] is False and index['numerical_ground_truth_claimed'] is False,'Coarse input/evidence boundary differs')
    journal=read(bound(done['attempt']));require(journal==done['actual_cost'],'Attempt cost differs')
    require(journal['status']=='completed_frozen_parent_coarse_overlap' and journal['plan_sha256']==digest(plan) and journal['active_operation'] is None,'Attempt not closed')
    require(journal['attempted_moments']==journal['completed_moments']==plan['planned']['native_moment_forwards'] and journal['attempted_shared_states']==journal['completed_shared_states']==300 and journal['completed_directed_pairs']==plan['planned']['directed_pairs'],'Actual dispatch/completed cost differs')
    require(journal['seconds']>0 and journal['moment_seconds']>0 and journal['pair_seconds']>0,'Missing actual cost')
    raw=index['raw_moments'];expected={tuple(k) for k in plan['legal_keys']}
    require(len(raw)==len(expected) and {(r['camera'],r['frame']) for r in raw}==expected,'Incomplete/duplicate native moments')
    for row in raw:
        path=bound(row['raw']);require(read(path.with_suffix('.json'))==row,'Raw closed receipt differs')
        require(row['shape']==[3,plan['coarse_size'][1],plan['coarse_size'][0]] and row['dtype']=='float32' and row['native_moment_forwards']==1 and row['seconds']>0,'Raw shape/cost differs')
    require(len(index['frames'])==300 and len({ref['path'] for ref in index['frames']})==300,'Incomplete frame ratio inventory')
    names=sorted(plan['data']['train_cameras']);pairs={(a,b) for i,a in enumerate(names) for b in names[i+1:]};seen=set()
    for ref in index['frames']:
        frame=read(bound(ref));f=frame['frame'];require(f in range(300) and f not in seen,'Duplicate/foreign ratio frame');seen.add(f)
        require(frame['context']==plan['context'] and frame['policy_sha256']==digest(POLICY) and frame['time']==f/300,'Ratio context/time/policy differs')
        require(len(frame['pairs'])==len(pairs) and {(r['a'],r['b']) for r in frame['pairs']}==pairs,'Incomplete ratio pairs')
        for row in frame['pairs']:
            scores=[]
            for direction in ('a_to_b','b_to_a'):
                value=row[direction];num,den=value['visible_target_pixels'],value['supported_target_pixels']
                require(isinstance(num,int) and isinstance(den,int) and 0<=num<=den,'Invalid ratio counts')
                require(value['ratio']==(num/den if den else None),'Directed ratio differs from counts');scores.append(value['ratio'])
            require(row['ratio']==(min(scores) if all(x is not None for x in scores) else None),'Conservative pair ratio differs')
    return dict(status='CPU_all_SHA_coarse_parent_overlap_accepted_pending_exact_unit_Exit0',complete=entry(local(complete_path)),
        index=done['index'],parent_state_unchanged=True,unit_exit_observed_by_this_function=False,actual_cost=journal,
        model_imported=False,GPU_queries=0,actual_overlap_recomputed=False)

def prepare(a,plan):
    refine,prefix,_=science();upstream=Path(a.upstream).resolve();out=local(a.out).resolve()
    require(a.operator_resource_resolved and a.gpu_uuid and os.environ.get('FOURDSR_UPSTREAM')==str(upstream),'Explicit root resource/environment resolution required')
    require(out.is_relative_to(OUTPUT_ROOT.resolve()) and out!=OUTPUT_ROOT.resolve(),'Independent producer output revision required')
    require(not out.exists(),'Fresh-only producer preserves existing outputs; recovery requires separate explicit authorization')
    registered=entry(local(a.registered_plan));require(read(bound(registered))==plan,'Exact registered plan required')
    require(not bound(registered).resolve().is_relative_to(out),'Plan must be separate from runtime output')
    manifest=verify_plan(plan,upstream)
    lock=local(a.lock).resolve();require(lock.name in (a.gpu_uuid+'.coarse-overlap.lock',a.gpu_uuid+'.native-fixture.lock'),'Physical UUID lock filename differs')
    authorization_ref=entry(local(a.authorization));authorization=read(bound(authorization_ref))
    owned=ownership(authorization,registered,a.gpu_uuid,lock)
    resource=resource_api_source(prefix,authorization)
    with physical_lock(lock):
        out.mkdir(parents=True);write(out/'registration.json',plan);write(out/'ownership.json',owned);write(out/'resource_API.json',resource)
        journal=dict(status='running_frozen_parent_coarse_overlap',plan_sha256=digest(plan),host=socket.gethostname(),pid=os.getpid(),
            attempted_moments=0,completed_moments=0,attempted_shared_states=0,completed_shared_states=0,completed_directed_pairs=0,
            moment_seconds=0.,pair_seconds=0.,active_operation=None,Adam_calls=0,RGB_forwards=0,autograd_calls=0,
            parameter_updates=0,started_unix=time.time(),source=entry(__file__))
        started=time.monotonic();attempt=out/'attempts'/f'{time.time_ns()}_{os.getpid()}.json';write(attempt,journal)
        torch=None;np=None;observer=None;g=None;before=None;model_restore_started=False
        try:
            first=dict(unix=time.time(),inventory=prefix.check_gpu(a.gpu_uuid));write(out/'resource_first.json',first)
            time.sleep(30)
            second=dict(unix=time.time(),inventory=prefix.check_gpu(a.gpu_uuid));write(out/'resource_second.json',second)
            require(second['unix']-first['unix']>=30,'Resource double read interval differs')
            prelaunch=dict(unix=time.time(),inventory=prefix.check_gpu(a.gpu_uuid));write(out/'resource_prelaunch.json',prelaunch)
            import numpy as np
            import torch
            torch.set_num_threads(a.cpu_threads)
            journal['active_operation']=dict(kind='CUDA_initialization_and_observer_RNG_selfcheck',native_moment_forwards=0);write(attempt,journal)
            observer,observer_startup=observer_rng_start(torch,np,prefix,refine,out)
            journal['active_operation']=None;write(attempt,journal)
            sys.path.insert(0,str(upstream));sys.path.insert(1,str(prefix.LEGACY.parent))
            from scene.gaussian_model import GaussianModel
            journal['active_operation']=dict(kind='native_model_restore');write(attempt,journal)
            parent_plan=read(bound(plan['parent']['plan']));opt=SimpleNamespace(**parent_plan['configuration']['OptimizationParams'])
            h=SimpleNamespace(**parent_plan['configuration']['ModelHiddenParams'])
            model_restore_started=True
            g=GaussianModel(parent_plan['configuration']['ModelParams']['sh_degree'],h);g._deformation=g._deformation.cuda()
            payload=refine.checkpoint_payload(bound(plan['parent']['checkpoint']),torch,parent_plan,'full_author_LR_prefix_checkpoint_v1','cuda')
            restore=refine.restore_native_model(g,payload,opt,torch,np,parent_plan,'full_author_LR_prefix_checkpoint_v1');del payload
            before=immutable(g,prefix,refine,torch,np);write(out/'native_restore.json',dict(restore=restore,before=before,zero_updates=True))
            # Metadata SHA verification precedes this permanent process-local hook.
            # Its owned-array set stays a set; postchecks never reopen image paths.
            guard,reads=pixel_boundary(out);sys.addaudithook(guard)
            spec=importlib.util.spec_from_file_location('coarse_native_moments',DEPTH)
            depth=importlib.util.module_from_spec(spec);spec.loader.exec_module(depth)
            from footprint import project  # frozen projection API, loaded once
            torch.cuda.reset_peak_memory_stats();directory=out/'raw_moments';directory.mkdir()
            rows=legal_rows(manifest);by_frame={f:[r for r in rows if int(r['frame_index'])==f] for f in range(300)}
            raw_entries=[];frames=[];mom_shape=(3,plan['coarse_size'][1],plan['coarse_size'][0])
            with torch.no_grad():
                for frame,frame_rows in by_frame.items():
                    journal['active_operation']=dict(kind='shared_native_deformation',frame=frame);journal['attempted_shared_states']+=1;write(attempt,journal)
                    state=refine.native_effective_state(g,float(frame_rows[0]['time']),torch);cov=refine.state_covariance(state,torch)
                    journal['completed_shared_states']+=1;packets={};cameras={}
                    for uid,row in enumerate(frame_rows):
                        camera=row['camera_id'];tick=time.monotonic()
                        journal['active_operation']=dict(kind='coarse_camera_creation',camera=camera,frame=frame);write(attempt,journal)
                        c=coarse_camera(manifest,row,uid,plan['coarse_size'])
                        journal['attempted_moments']+=1;journal['active_operation']=dict(kind='native_coarse_moment',camera=camera,frame=frame);write(attempt,journal)
                        raw=depth.render_moments(c,state['xyz'],cov,state['opacity'])['hr_moments']
                        require(tuple(raw.shape)==mom_shape,'Native coarse full-frame size differs')
                        packets[camera]=normalize_raw(raw);cal=manifest['cameras'][camera]
                        cameras[camera]=dict(K=plan['camera_K_coarse'][camera],w2c=cal['w2c'],c2w=cal['c2w'])
                        path=directory/f'{camera}_{frame:04d}.npz';atomic_npz(path,raw.detach().cpu().numpy().astype(np.float32),np)
                        torch.cuda.synchronize();seconds=time.monotonic()-tick
                        value=dict(camera=camera,frame=frame,raw=entry(path),seconds=seconds,native_moment_forwards=1,
                            shape=list(mom_shape),dtype='float32',nonfinite_elements=packets[camera]['nonfinite_raw_elements'],
                            supported_pixels=int(packets[camera]['support'].sum()),unknown_variance_pixels=int((packets[camera]['support']&~packets[camera]['variance_known']).sum()))
                        write(path.with_suffix('.json'),value);raw_entries.append(value)
                        journal['completed_moments']+=1;journal['moment_seconds']+=seconds;journal['active_operation']=None;write(attempt,journal)
                        del c,raw
                    tick=time.monotonic();journal['active_operation']=dict(kind='coarse_pair_geometry',frame=frame);write(attempt,journal)
                    pairs=[];names=sorted(packets)
                    for i,left in enumerate(names):
                        for right in names[i+1:]:
                            pairs.append(dict(a=left,b=right,**pair_overlap(packets[left],packets[right],cameras[left],cameras[right]),
                                **pose_pair(manifest['cameras'][left],manifest['cameras'][right])))
                    torch.cuda.synchronize();seconds=time.monotonic()-tick
                    path=out/'frames'/f'{frame:04d}.json';write(path,dict(frame=frame,time=float(frame_rows[0]['time']),pairs=pairs,
                        scope='train-only frozen-parent coarse approximation',context=plan['context'],policy_sha256=digest(POLICY),seconds=seconds))
                    frames.append(entry(path));journal['completed_directed_pairs']+=2*len(pairs);journal['pair_seconds']+=seconds
                    journal['active_operation']=None;write(attempt,journal);del packets,cameras,state,cov
            after=immutable(g,prefix,refine,torch,np)
            write(out/'native_invariance.json',dict(before=before,after=after,exact_equal=after==before,scope='native_parent_before_observer_restore'))
            native_invariance=entry(out/'native_invariance.json')
            require(after==before,'Native parameters/Adam/buffers/gradients/RNG/versions changed')
            # No legal-LR paths are reopened after the audit hook. Source/parent
            # files are still rehashed; image identity was proven before dispatch.
            require(source_files(upstream)==plan['source_files'],'Producer/science source changed during generation')
            for key in ('checkpoint','complete','plan','sidecar'):bound(plan['parent'][key])
            bound(plan['context']['manifest']);bound(plan['parent_acceptance']);bound(plan['readiness']);bound(registered)
            bound(authorization_ref);bound(owned['manager_source']);require(resource_api_source(prefix,authorization)==resource,'Resource API source changed')
            for key in ('native_moment_forwards','directed_pairs','shared_deformation_states'):
                observed={'native_moment_forwards':'completed_moments','directed_pairs':'completed_directed_pairs','shared_deformation_states':'completed_shared_states'}[key]
                require(journal[observed]==plan['planned'][key],'Actual complete budget differs: '+key)
            # The native-parent RNG equality has already passed above. Restore
            # the observer separately before publishing a completed receipt.
            observer_restoration=observer_rng_restore(observer,torch,np,prefix,refine,out,'after_native_parent_check')
            observer=None
            journal.update(status='completed_frozen_parent_coarse_overlap',seconds=time.monotonic()-started,active_operation=None)
            write(attempt,journal)
            value=dict(schema=INDEX_SCHEMA,status='completed_frozen_native_parent_coarse_overlap',context=plan['context'],
                scene=plan['scene'],seed=plan['seed'],policy=POLICY,policy_sha256=digest(POLICY),coarse_size=plan['coarse_size'],
                registration=entry(out/'registration.json'),source=entry(__file__),source_files=plan['source_files'],
                frames=frames,raw_moments=raw_entries,actual=dict(journal,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    output_bytes_before_index_complete=sum(p.stat().st_size for p in out.rglob('*') if p.is_file()),
                    raw_cache_bytes=sum(bound(r['raw']).stat().st_size for r in raw_entries),frame_ratio_JSON_bytes=sum(bound(r).stat().st_size for r in frames),torch=str(torch.__version__),
                    device_name=torch.cuda.get_device_name(),gpu_uuid=a.gpu_uuid,cpu_threads=a.cpu_threads),
                invariants=dict(before=before,after=after,parent_state_unchanged=True,restore=restore,observer_RNG_restored=True,
                    observer_RNG_checks=dict(startup=observer_startup,restoration=observer_restoration),native_invariance=native_invariance),
                input_reads=reads,HR_pixels_read=False,heldout_pixels_read=False,training_pixel_decodes=0,teacher_reads=0,
                total_native_moment_forwards=len(raw_entries),prior_reused_moments=0,RGB_forwards=0,Adam_calls=0,
                formal_updates=0,autograd_calls=0,numerical_ground_truth_claimed=False,
                image_identity_guard='all legal LR SHA before CUDA; images cannot be opened during render/pair/postcheck',
                approximate_overlap_not_X_support=True)
            write(out/'index.json',value)
            write(out/'complete.json',dict(status=value['status'],index=entry(out/'index.json'),attempt=entry(attempt),
                context=plan['context'],source=entry(__file__),planned=plan['planned'],actual_cost=journal,
                parent_state_unchanged=True,observer_RNG_restored_after_parent_check=True))
            return out/'complete.json'
        except BaseException as exc:
            journal.update(status='failed_frozen_parent_coarse_overlap_preserved',seconds=time.monotonic()-started,error=repr(exc),
                unresolved_active_operation_cost='unknown' if journal['active_operation'] else None)
            write(attempt,journal);write(out/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),attempt=entry(attempt),
                model_dispatched=model_restore_started,CUDA_initialized=torch.cuda.is_initialized() if torch is not None else False,
                native_moment_forwards_completed=journal['completed_moments'],formal_updates=0,source=entry(__file__)))
            raise
        finally:
            if observer is not None:
                # Preserve the primary preparation failure if cleanup also fails.
                try:observer_rng_restore(observer,torch,np,prefix,refine,out,'failure_cleanup')
                except BaseException as cleanup:
                    write(out/'observer_RNG_failure_cleanup_error.json',dict(error=repr(cleanup),traceback=traceback.format_exc(),primary_failure_preserved=True))
            del g

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=('plan','prepare'),required=True)
    p.add_argument('--manifest');p.add_argument('--seed',type=int);p.add_argument('--parent');p.add_argument('--parent-complete')
    p.add_argument('--parent-acceptance');p.add_argument('--parent-index',default=str(OUT/'full_LR_prefixes/checkpoint_index.json'))
    p.add_argument('--readiness',default=str(OUT/'full_data_readiness/full_prefix_readiness.json'))
    p.add_argument('--upstream',required=True);p.add_argument('--registered-plan',required=True);p.add_argument('--out')
    p.add_argument('--inventory');p.add_argument('--gpu-uuid');p.add_argument('--lock');p.add_argument('--authorization')
    p.add_argument('--operator-resource-resolved',action='store_true');p.add_argument('--cpu-threads',type=int,default=2)
    a=p.parse_args()
    if a.mode=='plan':
        for k in ('manifest','seed','parent','parent_complete','parent_acceptance'):require(getattr(a,k) is not None,'Required metadata input: '+k)
        plan=build_plan(a);path=local(a.registered_plan)
        if path.exists():require(read(path)==plan,'Immutable registered plan differs')
        else:write(path,plan)
        if a.inventory:write(local(a.inventory),input_inventory(plan,a.upstream))
        print(json.dumps(dict(status=plan['status'],plan=entry(path),planned=plan['planned'],CUDA_queries=0,model_imported=False)))
    else:
        for k in ('out','gpu_uuid','lock','authorization'):require(getattr(a,k) is not None,'Required owned execution input: '+k)
        require(a.cpu_threads>0,'Positive CPU threads required')
        path=prepare(a,read(bound(entry(local(a.registered_plan)))));print(json.dumps(dict(complete=entry(path))))
if __name__=='__main__':main()
