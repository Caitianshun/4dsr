"""CPU registries for unchanged training consumers; generic data-grid adaptation.

N3DV native manifests dispatch to the original register_native functions, retaining
their byte-exact existing registrations. Fresh COLMAP conditions use default.py.
"""
from __future__ import annotations
import argparse, ast, collections, itertools, math, random
from pathlib import Path
from common import *
NATIVE=ROOT/'experiments/dynamic_sr_native_20261010/register_native.py'
AUTHOR_COMMIT='843d5ac636c37e4b611242287754f3d4ed150144'
RUNTIME=ROOT/'third_party/4dgs_843d5ac_runtime'
DEFINITION=dict(schedule='rows',lr_weights=[1.,0.,0.],sr_weights=[0.,.05,.05],X=False,E=False)
def config_default():
    native=module('highres_native_register_metadata',NATIVE)
    paths={'arguments/__init__.py':RUNTIME/'arguments/__init__.py','arguments/dynerf/default.py':RUNTIME/'arguments/dynerf/default.py'}
    result={}
    for cls in ast.parse(paths['arguments/__init__.py'].read_text()).body:
        if not isinstance(cls,ast.ClassDef) or cls.name not in ('ModelParams','ModelHiddenParams','OptimizationParams','PipelineParams'):continue
        init=next(f for f in cls.body if isinstance(f,ast.FunctionDef) and f.name=='__init__')
        result[cls.name]={n.targets[0].attr:native.pure(n.value) for n in init.body if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Attribute) and isinstance(n.targets[0].value,ast.Name) and n.targets[0].value.id=='self'}
    for n in ast.parse(paths['arguments/dynerf/default.py'].read_text()).body:
        if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id in result:result[n.targets[0].id].update(native.pure(n.value))
    if (result['OptimizationParams']['coarse_iterations'],result['OptimizationParams']['iterations'],result['OptimizationParams']['batch_size'])!=(3000,14000,4):raise ValueError('Author default changed')
    return result,paths
def inventory(path,seed):
    import numpy as np
    path=local(path);m=read(path)
    if not path.is_relative_to(DATA/'prepared') or m['scene'] not in CONDITIONS or seed not in SEEDS:raise ValueError('Unsupported highres namespace/seed')
    W,H=CONDITIONS[m['scene']]
    if m['resolutions']!=dict(hr=[W,H],lr=[W//4,H//4]) or not m.get('full_time_decode_verified') or m['frame_indices']!=list(range(300)):raise ValueError('Unverified grid/time')
    train=m['splits']['train']
    if train[0]!='cam01' or m['splits']['test']!=['cam00'] or m['splits']['dev'] or set(train)!=set(m['cameras'])-{'cam00'}:raise ValueError('Split/diagnostic boundary differs')
    if m['initialization']['seed']!=seed or not m['initialization']['train_only_lr'] or m['initialization']['official_COLMAP_initialization']:raise ValueError('LR-only independent initialization required')
    original=local(m['original_full_manifest']['path'])
    if sha(original)!=m['original_full_manifest']['sha256']:raise ValueError('Producer changed')
    complete=read(path.parent/'complete.json');z=local(complete['z1']['path'])
    if sha(z)!=complete['z1']['sha256'] or not read(z)['passed']:raise ValueError('Z1 not passed')
    rows=[r for r in m['observations'] if r['split']=='train'];keys=[(r['camera_id'],r['frame_index']) for r in rows]
    if len(keys)!=len(set(keys)) or set(keys)!=set(itertools.product(train,range(300))):raise ValueError('Training LR inventory incomplete')
    files=[]
    for r in rows:
        p=path.parent/r['lr_path']
        if sha(p)!=r['lr_sha256'] or abs(r['time']-r['frame_index']/300)>1e-7:raise ValueError('Training LR identity/time differs')
        files.append(dict(camera=r['camera_id'],frame=r['frame_index'],**entry(p)))
    points=path.parent/m['initialization']['npz_path']
    if sha(points)!=m['initialization']['npz_sha256']:raise ValueError('Initialization changed')
    centers=np.array([np.array(m['cameras'][c]['c2w'])[:3,3] for c in train]);center=centers.mean(0);extent=float(np.linalg.norm(centers-center,axis=1).max()*1.1)
    native=module('highres_native_registry_helpers',NATIVE)
    data=dict(manifest=entry(path),original_manifest=entry(original),initialization_receipt=entry(path.parent/f'initialization_seed_{seed}/complete.json'),initial_points=entry(points),train_cameras=train,training_observations=len(rows),training_LR_sources_sha256=native.digest(files),train_camera_center=center.tolist(),cameras_extent=extent,extent_cam01_included=True,heldout_camera='cam00',frames=[0,299],time='frame_index/300',resolution_LR=[W//4,H//4])
    return path,m,data,files
def sources(paths):
    project=[Path(__file__),HERE/'entry.py',ROOT/'experiments/dynamic_sr_cull_retrain_20261009/cullretrain.py',ROOT/'experiments/dynamic_sr_modules_20261010/modsr.py',ROOT/'experiments/dynamic_sr_modules_20261010/refs.py']
    runtime={str(p.relative_to(RUNTIME)):sha(p) for d in ('arguments','scene','utils') for p in sorted((RUNTIME/d).rglob('*.py'))}
    return dict(author_sources={k:entry(p) for k,p in paths.items()},runtime_source_sha256=runtime,project_sources={str(p.relative_to(ROOT)):sha(p) for p in project})
def lr(path,seed):
    path,m,data,files=inventory(path,seed);config,paths=config_default();s=sources(paths)
    native=module('highres_planned_counts',NATIVE)
    value=dict(schema='full_author_LR_prefix_v1',status='registered_highres_independent_LR_prefix_pending_GPU',seed=seed,scene=m['scene'],data=data,author_commit=AUTHOR_COMMIT,configuration=config,**s,observation='native_lr',white_background=config['ModelParams']['_white_background'],fresh_seed_model=True,representation='native_author_GaussianModel_no_children',planned=native.planned_counts(config),teacher_used=False,HR_used=False,heldout_pixels_used=False,CUDA_initialized=False,GPU_forwards=0,Adam_calls=0,author_adaptations=['Published COLMAP calibration; train-LR-only known-camera initialization; complete background.','Default dynerf configuration unchanged (batch4); supplied grids and time window are explicit data adaptations.'])
    return write(REGISTRY/'full_LR_prefixes'/m['scene']/f'seed_{seed}'/'config.json',value)
def schedule(m,manifest,seed):
    native=module('highres_schedule_helpers',NATIVE)
    cameras=m['splits']['train']
    if len(cameras)<=20:return native.schedule(m,manifest,seed)
    keys=[(r['camera_id'],int(r['frame_index'])) for r in m['observations'] if r['split']=='train'];lookup={k:i for i,k in enumerate(keys)};rng=random.Random(seed);camorder=cameras[:];rng.shuffle(camorder);rows=[];calibration=[]
    # 32*300 observations cannot fit 6000 one-anchor rows. Preserve6000budget and
    # equal corresponding-column exposure; rotate20camera subsets per realframe.
    for frame in range(300):
        subset=[camorder[(frame*20+j)%len(camorder)] for j in range(20)];rng.shuffle(subset)
        shift1=rng.randrange(1,20);shift2=rng.choice([x for x in range(1,20) if x!=shift1])
        block=[[lookup[(subset[j],frame)],lookup[(subset[(j+shift1)%20],frame)],lookup[(subset[(j+shift2)%20],frame)]] for j in range(20)];rows.extend(block)
        if frame in (0,100,200,299):calibration.extend(block)
    rng.shuffle(rows);columns=[[r[i] for r in rows] for i in range(3)]
    for c in columns:rng.shuffle(c)
    random_rows=[list(x) for x in zip(*columns)];exposures=[collections.Counter(r[i] for r in rows) for i in range(3)]
    if len(rows)!=6000 or not exposures[0]==exposures[1]==exposures[2]:raise ValueError('Schedule exposure differs')
    counts=collections.Counter(keys[r[0]][0] for r in rows)
    if max(counts.values())-min(counts.values())>1:raise ValueError('Unbalanced camera exposure')
    return dict(schema='full_native_SR_schedule_v1',identity=dict(manifest=manifest),seed=seed,updates=6000,record_keys=[list(k) for k in keys],rows=rows,random_rows=random_rows,calibration_triplets=calibration,audit=dict(passed=True,full_legal_observations=len(keys),exposed_legal_observations=len(exposures[0]),every_legal_key_covered=False,three_column_exact_exposure=True,full6000_block_only=True,camera_exposure=dict(counts)),assignment_protocol=dict(source=entry(__file__),cost='seeded balanced rotating20of32camera subset per frame; identical schedule for all arms',HR_or_test_or_quality_cost=False,coverage_limitation='6000/9600 camera-time keys per column; all32cameras included but not everycamera-time; preserves6000update budget.'))
def sr(path,seed,teacher_index):
    path,m,data,lr_files=inventory(path,seed);config,paths=config_default();index_path=local(teacher_index);index=read(index_path)
    if index['status']!='completed_teacher_inventory' or index['manifest_sha256']!=sha(path) or index['seed_consumer']!=seed:raise ValueError('Teacher index differs')
    expected={(r['camera_id'],r['frame_index']):r for r in m['observations'] if r['split']=='train'};files=[]
    if {(e['camera'],e['frame']) for e in index['entries']}!=set(expected):raise ValueError('Teacher population differs')
    for e in index['entries']:
        p=path.parent/e['relative_path']
        if sha(p)!=e['sha256'] or e['lr_sha256']!=expected[e['camera'],e['frame']]['lr_sha256']:raise ValueError('Teacher identity differs')
        files.append(dict(camera=e['camera'],frame=e['frame'],**entry(p)))
    s=sources(paths);table=schedule(m,data['manifest'],seed);sp=write(REGISTRY/'schedules'/m['scene']/f'seed_{seed}'/'schedule.json',table)
    value=dict(schema='registered_full_native_SR_refinement_v1',status='registered_highres_SR_consumer',missing=[],scene=m['scene'],seed=seed,method='Bsync',definition=DEFINITION,data=dict(data,resolution_HR=list(CONDITIONS[m['scene']])),dependencies=dict(schedule=entry(sp),teacher=entry(index_path)),full_native_context=dict(manifest=data['manifest'],schedule=entry(sp)),configuration=config,source_files=dict(project=s['project_sources'],upstream=s['runtime_source_sha256']),representation='native_author_GaussianModel_no_children',author_commit=AUTHOR_COMMIT,training_files=dict(LR=lr_files,teacher=files),white_background=config['ModelParams']['_white_background'],HR_training_images_read=False,heldout_pixels_read=False,CPU_only_registration=True,GPU_forwards=0,Adam_calls=0,adaptations=['New COLMAP dataset/grids; default dynerf config batch4 unchanged.','Shared6000balanced same-time three-view schedule; Long360 coverage limitation recorded in schedule.'])
    return write(REGISTRY/'full_SR_refinement'/m['scene']/f'seed_{seed}'/'Bsync'/'config.json',value)
def main():
    p=argparse.ArgumentParser();p.add_argument('kind',choices=['lr','sr']);p.add_argument('--manifest',required=True,type=Path);p.add_argument('--seed',required=True,type=int);p.add_argument('--teacher-index',type=Path);a=p.parse_args();path=local(a.manifest)
    if path.is_relative_to(ROOT/'data/dynamic_sr/native_20261010/prepared'):
        native=module('highres_native_passthrough',NATIVE);result=native.lr(path,a.seed) if a.kind=='lr' else native.sr(path,a.seed,a.teacher_index);print(result);return
    print(lr(path,a.seed) if a.kind=='lr' else sr(path,a.seed,a.teacher_index))
if __name__=='__main__':main()
