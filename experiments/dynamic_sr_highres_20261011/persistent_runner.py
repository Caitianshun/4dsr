"""Persistent per-condition DAG runner: wait on process exit events, no polling.

Root allocates GPU UUIDs and invokes this process through a persistent service.
One subprocess per selected GPU; independent branches can use both A100 cards.
Existing partial outputs are preserved. Artifacts are checked before successors.
"""
from __future__ import annotations
import argparse,concurrent.futures as cf,contextlib,csv,json,math,os,re,socket,subprocess,sys,time,traceback
from pathlib import Path
from common import *
METRIC_SOURCE_PATHS=[ROOT/rel for rel in [
    'experiments/dynamic_sr_detail_supervision_20260924/evaluate.py',
    'experiments/dynamic_sr_same_observation_20260930/frequency.py',
    'experiments/dynamic_sr_prior_diagnosis_20260929/evaluation_adapter.py',
    'experiments/dynamic_sr_detail_supervision_20260924/detail_loss.py']]
def validate(tasks):
    names={t['name'] for t in tasks}
    if len(names)!=len(tasks):raise ValueError('Duplicate task names')
    outputs=[t['done'] for t in tasks]
    if len(set(outputs))!=len(outputs):raise ValueError('Duplicate completion paths')
    for t in tasks:
        if not re.fullmatch(r'[A-Za-z0-9_]+',t['name']):raise ValueError('Unsafe task name')
        if t['script'] not in ['entry.py','register_highres.py','teacher_screened.py','prior_fit.py']:raise ValueError('Unsupported fresh screening task script')
        if not set(t['deps'])<=names or t['device'] not in ['cpu','gpu']:raise ValueError('Unresolved dependency/device')
        script=(HERE/t['script']).resolve()
        if script.parent!=HERE or not script.is_file():raise ValueError('Script outside fresh code directory')
        done=local(t['done'])
        if not done.is_relative_to(OUT):raise ValueError('Completion outside fresh namespace')
        if t['script']=='entry.py':
            argv=t['argv']
            if argv[0] not in ['lr','sr','eval','extras'] or t['device']!='gpu':raise ValueError('Endpoint command/device differs')
            if any(flag in argv for flag in ['--smoke-coarse-steps','--stop','--frames','--region-stride','--region-limit']):raise ValueError('Screening graph forbids shortened diagnostic budgets')
            if '--out' not in argv or local(argv[argv.index('--out')+1])!=done.parent:raise ValueError('Output/completion mismatch')
        elif t['script']=='register_highres.py':
            if t['argv'][0]!='sr' or t['device']!='cpu':raise ValueError('Only CPU SR registration belongs to this graph')
        elif t['script']=='prior_fit.py':
            argv=t['argv']
            if t['device']!='gpu' or '--out' not in argv or local(argv[argv.index('--out')+1])!=done.parent or '--check-only' in argv:raise ValueError('Prior-fit output/device differs or check-only diagnostic requested')
        elif t['argv'][0] not in ['cost','run'] or t['device']!='gpu':raise ValueError('Teacher command/device differs')
    pending={t['name']:set(t['deps']) for t in tasks}
    while pending:
        ready={n for n,d in pending.items() if not d}
        if not ready:raise ValueError('Cyclic graph')
        pending={n:d-ready for n,d in pending.items() if n not in ready}
def validate_table(table):
    if table['schema']!='highres_screening_event_DAG_v1' or table['condition'] not in CONDITIONS or table['seed']!=20261007:raise ValueError('Unapproved screening condition/seed/schema')
    if table['budgets']!=dict(LRparent17k=17000,lam0_base_oracle=23000,SRrecon_HRref=17000):raise ValueError('Approved budgets changed')
    tasks=table['tasks'];validate(tasks)
    teacher=next(t for t in tasks if t['script']=='teacher_screened.py')
    arg=lambda flag:teacher['argv'][teacher['argv'].index(flag)+1]
    import make_screening_tasks
    if tasks!=make_screening_tasks.build(table['condition'],table['seed'],arg('--z5-receipt'),arg('--z5-samples')):raise ValueError('Task graph differs from the approved six-endpoint builder')
def check_completion(task):
    done=local(task['done']);value=read(done)
    if task['script']=='entry.py':
        dispatch=read(done.parent/'dispatch.json')
        if dispatch['argv']!=task['argv'] or dispatch['entry']['sha256']!=sha(HERE/'entry.py'):raise ValueError('Existing endpoint belongs to another command/source')
        kind=task['argv'][0]
        if kind in ['lr','sr']:
            if value.get('status')!='completed' or not math.isfinite(value.get('seconds',math.nan)):raise ValueError('Incomplete training receipt')
            ref=value['checkpoint'];p=local(ref['path'])
            if sha(p)!=ref['sha256']:raise ValueError('Checkpoint hash differs')
            if (done.parent/'smoke_override.json').exists():raise ValueError('Smoke checkpoint cannot satisfy formal task')
            if kind=='sr' and value.get('cursor')!=6000:raise ValueError('SR budget incomplete')
            if kind=='lr':
                coarse='--coarse-only' in task['argv'];s=value['state']
                if s['coarse_iteration']!=3000 or (not coarse and s['fine_iteration']!=14000):raise ValueError('LR budget incomplete')
        elif kind=='eval':
            if value.get('status')!='completed' or value.get('frames')!=300:raise ValueError('Main evaluation is not full300')
            if not all(math.isfinite(value['test'][k]) for k in ['psnr','ssim','lpips']):raise ValueError('Nonfinite main metrics')
            with (done.parent/'metrics_per_observation.csv').open() as f:rows=list(csv.DictReader(f))
            test=[r for r in rows if r['scope']=='test']
            if sorted((r['camera'],int(r['frame'])) for r in test)!=[('cam00',i) for i in range(300)]:raise ValueError('Evaluation CSV must contain cam00 uniquely at all300frames')
        elif kind=='extras':
            if 'cam00_region_band' not in value or not value.get('train_LR_closure'):raise ValueError('ROI/closure diagnostics missing')
            for region,cell in value['cam00_region_band'].items():
                if not all(math.isfinite(cell[k]) for k in ['area','low','mid','high']):raise ValueError('Nonfinite ROI bands')
    elif task['script']=='teacher_screened.py':
        if task['argv'][0]=='cost' and value.get('status')!='completed_three_image_untiled_teacher_cost':raise ValueError('Teacher cost incomplete')
        if task['argv'][0]=='run' and value.get('status')!='completed_full_teacher_CPU_acceptance':raise ValueError('Teacher acceptance incomplete')
        planpath=local(task['argv'][task['argv'].index('--plan')+1])
        if task['argv'][0]=='cost':
            import teacher_highres
            if value['plan']!=entry(planpath) or value['images']!=3 or value['tile']!=0 or value['frozen_recipe']!=teacher_highres.FROZEN:raise ValueError('Teacher cost belongs to another plan/recipe')
        else:
            if value['producer_manifest']!=read(planpath)['producer_manifest']:raise ValueError('Teacher acceptance belongs to another producer')
            for ref in value['indexes']:
                if entry(local(ref['path']))!=ref:raise ValueError('Teacher index changed after acceptance')
    elif task['script']=='register_highres.py':
        if value.get('status')!='registered_highres_SR_consumer':raise ValueError('SR registration incomplete')
    elif task['script']=='prior_fit.py':
        if value.get('status')!='completed_train_view_prior_fit' or value.get('training') is not False or value.get('optimizer_calls')!=0 or value.get('parameter_updates')!=0 or value.get('heldout_pixels_used') is not False:raise ValueError('Prior fit did not remain a completed train-only evaluation')
        cameras=value['train_cameras'];keys=[(c,f) for c in cameras for f in [15,150,285]]
        if 'cam00' in cameras or len(set(cameras))!=len(cameras) or value['sampled_frames']!=[15,150,285] or value['observations']!=len(keys) or sorted(map(tuple,value['population_keys']))!=sorted(keys):raise ValueError('Prior-fit population missing, duplicated or leaked heldout views')
        for label in ['source','checkpoint','manifest','baseline_sr','teacher_index','teacher_acceptance','hr_storage','z5_samples','metrics','csv','observation_index']:
            ref=value[label]
            if entry(local(ref['path']))!=ref:raise ValueError(f'Prior-fit {label} identity changed')
        if value['source']!=entry(HERE/'prior_fit.py'):raise ValueError('Prior fit source differs')
        if set(cameras)!=set(read(local(value['manifest']['path']))['splits']['train']):raise ValueError('Prior fit omitted some training cameras')
        arg=lambda flag:local(task['argv'][task['argv'].index(flag)+1])
        expected=dict(checkpoint=arg('--checkpoint'),baseline_sr=arg('--baseline-sr')/'config.json',z5_samples=arg('--z5-samples'),hr_storage=arg('--hr-root')/'hr_full_storage.json')
        for label,path in expected.items():
            if local(value[label]['path'])!=path:raise ValueError(f'Prior fit {label} belongs to another command')
        with local(value['csv']['path']).open() as f:rows=list(csv.DictReader(f))
        if sorted((r['camera'],int(r['frame'])) for r in rows)!=sorted(keys):raise ValueError('Prior-fit CSV population differs')
    return entry(done)
def resolve_gpu(identifier):
    p=subprocess.run(['nvidia-smi','--id',identifier,'--query-gpu=index,uuid,name','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    rows=list(csv.reader(p.stdout.strip().splitlines()))
    if len(rows)!=1:raise ValueError('GPU selector must resolve exactly one card')
    index,uuid,name=[x.strip() for x in rows[0]];return dict(index=index,uuid=uuid,name=name)
def check_gpu_idle(gpu,allowed_names):
    resource=subprocess.run(['nvidia-smi','--id',gpu['uuid'],'--query-gpu=uuid,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    rows=list(csv.reader(resource.stdout.strip().splitlines()))
    if len(rows)!=1:raise ValueError('Allocated GPU resource query ambiguous')
    uuid,used,total,util=[x.strip() for x in rows[0]]
    state=dict(uuid=uuid,memory_used_MiB=float(used),memory_total_MiB=float(total),utilization_percent=float(util))
    if uuid!=gpu['uuid'] or state['memory_used_MiB']>1024 or state['utilization_percent']>10:raise RuntimeError('Allocated GPU is not idle: memory>1024MiB or utilization>10%; current tasks preserved')
    p=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    occupants=[]
    for row in csv.reader(p.stdout.strip().splitlines()):
        if not row or row[0].strip()!=gpu['uuid']:continue
        uuid,pid,name,memory=[x.strip() for x in row];occupants.append(dict(uuid=uuid,pid=pid,name=name,memory_MiB=memory))
        if Path(name).name not in allowed_names or float(memory)>600:raise RuntimeError(f'Allocated GPU has unknown/oversized compute process: {name} PID{pid}; allowed graphics limit600MiB')
    return dict(gpu=state,processes=occupants,limits=dict(memory_used_MiB=1024,utilization_percent=10,allowed_graphics_process_MiB=600))
@contextlib.contextmanager
def gpu_claims(gpus,lockdir,allowed_names):
    owned=[]
    try:
        for g in gpus:
            lock=mkdir(lockdir)/f'{g["uuid"]}.lock';fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o664)
            stat=os.fstat(fd);owned.append((lock,stat.st_ino,stat.st_dev))
            with os.fdopen(fd,'w') as f:json.dump(dict(kind='highres_screening_runner',PID=os.getpid(),gpu=g,host=socket.gethostname(),started_unix=time.time()),f)
            check_gpu_idle(g,allowed_names)
        yield
    finally:
        for lock,inode,dev in owned:
            try:current=lock.stat()
            except FileNotFoundError:continue
            if (current.st_ino,current.st_dev)==(inode,dev):lock.unlink()
def atomic_status(path,value):
    p=Path(path);tmp=p.with_name(p.name+f'.{os.getpid()}.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');os.replace(tmp,p)
def prepared_inputs(condition,seed,manifest):
    m=read(manifest);points=manifest.parent/m['initialization']['npz_path'];original=local(m['original_full_manifest']['path'])
    if m['scene']!=condition or m['initialization']['seed']!=seed or not m['initialization']['train_only_lr'] or sha(points)!=m['initialization']['npz_sha256']:raise ValueError('Initialized LR manifest/point identity differs')
    if sha(original)!=m['original_full_manifest']['sha256']:raise ValueError('Prepared producer differs from initialized manifest')
    preparation=manifest.parent/'complete.json';prepared=read(preparation)
    if prepared['manifest']!=entry(original):raise ValueError('Prepared completion belongs to another producer')
    z1=local(prepared['z1']['path']);window=local(prepared['window']['path'])
    if entry(z1)!=prepared['z1'] or entry(window)!=prepared['window'] or not read(z1)['passed']:raise ValueError('Z1/window provenance differs or gate failed')
    if read(z1)['rung']!=('full' if condition=='dgF_cube' else 'half') or read(window)['selected_start']!=m['original_start']:raise ValueError('Z1/window condition mismatch')
    initialization_receipt=manifest.parent/f'initialization_seed_{seed}/complete.json'
    return m,points,original,[preparation,z1,window,initialization_receipt]
def check_frozen_identity(source_identity,python_inventory=None):
    if python_inventory is not None and sorted(p.name for p in HERE.glob('*.py'))!=python_inventory:raise ValueError('Frozen namespace Python inventory changed')
    for ref in source_identity:
        if sha(local(ref['path']))!=ref['sha256']:raise ValueError('Frozen code/input changed before task')
def disk_estimate(task):
    if task['script']=='teacher_screened.py':return 1.0 if task['argv'][0]=='cost' else 35.0
    if task['script']=='register_highres.py' or task['script']=='prior_fit.py':return .25
    return 15.0 if task['argv'][0] in ['lr','sr'] else 3.0
def execute(task,gpu,out,source_identity,allowed_names,python_inventory=None):
    check_frozen_identity(source_identity,python_inventory)
    storage=disk_guard(ROOT,disk_estimate(task))
    occupants=check_gpu_idle(gpu,allowed_names) if gpu else []
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES=gpu['uuid'] if gpu else '',OMP_NUM_THREADS='4')
    argv=[sys.executable,'-u','-B',str(HERE/task['script']),*task['argv']];begin=time.time();logpath=out/'logs'/f'{task["name"]}.log'
    with logpath.open('xb') as log:
        proc=subprocess.Popen(argv,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        write(out/'events'/f'{task["name"]}_start.json',dict(event='started',pid=proc.pid,argv=argv,gpu=gpu,occupants=occupants,disk_guard=storage,started_unix=begin,host=socket.gethostname()))
        code=proc.wait() # This blocking wait is a process-exit event, not model polling.
    result=dict(exit_code=code,seconds=time.time()-begin,gpu=gpu,log=entry(logpath),completed_unix=time.time())
    if code!=0:raise RuntimeError(f'Task exited {code}; see {logpath}')
    result['artifact']=check_completion(task);result['disk_after']=disk_guard(ROOT,0);write(out/'events'/f'{task["name"]}_complete.json',dict(status='awaiting_review',**result));return result
def main():
    p=argparse.ArgumentParser();p.add_argument('--tasks',required=True,type=Path);p.add_argument('--out',required=True,type=Path);p.add_argument('--gpus',required=True,nargs='+');p.add_argument('--gpu-lock-dir',type=Path,default=OUT/'claims');p.add_argument('--allowed-graphics-name',nargs='*',default=[]);p.add_argument('--validate-only',action='store_true');a=p.parse_args();tablepath=local(a.tasks);table=read(tablepath);tasks=table['tasks'];validate_table(table)
    if a.validate_only:print(json.dumps(dict(status='DAG_validated_not_launched',tasks=len(tasks),budgets=table['budgets'])));return
    out=local(a.out)
    if not out.is_relative_to(OUT/'pipelines') or out.exists():raise FileExistsError('Pipeline requires absent fresh output; review interrupted runs before restart')
    if table['condition'] not in CONDITIONS or table['seed']!=20261007:raise ValueError('Initial screening is restricted to the approved condition and one seed')
    gpus=[resolve_gpu(x) for x in a.gpus]
    if len({g['uuid'] for g in gpus})!=len(gpus) or len({g['name'] for g in gpus})!=1:raise ValueError('Distinct GPUs of same model required for paired condition')
    if local(a.gpu_lock_dir)!=OUT/'claims':raise ValueError('Use the shared screening/runner GPU UUID claims directory')
    mkdir(out);mkdir(out/'logs');mkdir(out/'events')
    write(out/'allocation_request.json',dict(tasks=entry(tablepath),gpus=gpus,PID=os.getpid(),host=socket.gethostname(),requested_unix=time.time()))
    try:
        with gpu_claims(gpus,a.gpu_lock_dir,a.allowed_graphics_name):
            run_pipeline(a,tablepath,table,tasks,gpus,out)
    except BaseException:
        write(out/f'preflight_or_pipeline_failure_{time.time_ns()}.json',dict(status='failed_preserved',error=traceback.format_exc(),automatic_restart=False,existing_foreign_claims_preserved=True));raise
def run_pipeline(a,tablepath,table,tasks,gpus,out):
    condition=table['condition'];seed=table['seed'];manifest=DATA/'prepared'/condition/f'manifest_train_ready_seed{seed}.json';hrindex=DATA/'hr_full'/condition/'hr_full_storage.json';parentconfig=REGISTRY/'full_LR_prefixes'/condition/f'seed_{seed}'/'config.json';teacherplan=REGISTRY/'teachers'/condition/'plan.json'
    for q in [manifest,hrindex,parentconfig,teacherplan]:
        if not q.is_file():raise FileNotFoundError(f'External prerequisite missing: {q}')
    import teacher_screened
    screened=next(t for t in tasks if t['script']=='teacher_screened.py')
    arg=lambda flag:screened['argv'][screened['argv'].index(flag)+1]
    screening=teacher_screened.gate(teacherplan,arg('--z5-receipt'),arg('--z5-samples'))
    m,points,original,preparation_inputs=prepared_inputs(condition,seed,manifest)
    dataset_config=local(read(local(arg('--z5-samples')))['dataset_config']['path'])
    python_inventory=sorted(q.name for q in HERE.glob('*.py'))
    generator_ref=read(teacherplan)['generator'];generator=local(generator_ref['path'])
    if entry(generator)!=generator_ref:raise ValueError('Frozen teacher generator differs from plan')
    source_identity=[entry(q) for q in [tablepath,manifest,original,points,hrindex,parentconfig,teacherplan,generator,*METRIC_SOURCE_PATHS,*preparation_inputs,dataset_config,local(arg('--z5-receipt')),local(arg('--z5-samples')),*sorted(HERE.glob('*.py'))]]
    parent=read(parentconfig)
    if parent['data']['manifest']!=entry(manifest) or parent['data']['initial_points']!=entry(points):raise ValueError('LR registration input/point identity differs')
    hr=read(hrindex)
    if hr['manifest']!=entry(original) or hr.get('LR_roundtrip_all_frames_SHA_equal') is not True or set(hr['cameras'])!=set(m['splits']['train']):raise ValueError('Privileged HR storage belongs to another producer/population')
    for camera in hr['cameras'].values():
        if camera['frames']!=300 or sorted(map(int,camera['hr_sha256']))!=list(range(300)):raise ValueError('Privileged HR storage incomplete')
    for rel,expected in parent['runtime_source_sha256'].items():
        q=ROOT/'third_party/4dgs_843d5ac_runtime'/rel
        if sha(q)!=expected:raise ValueError('Frozen upstream runtime differs')
        source_identity.append(entry(q))
    for rel,expected in parent['project_sources'].items():
        q=ROOT/rel
        if sha(q)!=expected:raise ValueError('Frozen project source differs')
        source_identity.append(entry(q))
    mkdir(out);mkdir(out/'logs');mkdir(out/'events');write(out/'identity.json',dict(tasks=entry(tablepath),sources=source_identity,python_inventory=python_inventory,Z5_binding=screening,gpus=gpus,host=socket.gethostname(),pid=os.getpid(),start_unix=time.time(),scope='six endpoints; no new module branches',extras_scope=table['extras_scope'],prior_fit_scope=table['prior_fit_scope']))
    state={t['name']:dict(status='pending',deps=t['deps']) for t in tasks};active={};available=gpus[:];cpu_busy=False;fatal=None
    def persist():atomic_status(out/'status.json',dict(tasks=state,updated_unix=time.time(),active=list(active.values()),available_gpus=available))
    try:
        with cf.ThreadPoolExecutor(max_workers=len(gpus)+1) as pool:
            while True:
                for t in tasks:
                    if fatal is not None:break
                    name=t['name']
                    if state[name]['status']!='pending' or not all(state[d]['status'] in ['completed','already_complete'] for d in t['deps']):continue
                    if local(t['done']).exists():state[name].update(status='already_complete',artifact=check_completion(t));continue
                    if t['script']=='entry.py' and local(t['done']).parent.exists():raise FileExistsError(f'Partial task output preserved: {local(t["done"]).parent}')
                    if t['device']=='gpu':
                        if not available:continue
                        gpu=available.pop(0)
                    else:
                        if cpu_busy:continue
                        gpu=None;cpu_busy=True
                    state[name].update(status='running',gpu=gpu);f=pool.submit(execute,t,gpu,out,source_identity,a.allowed_graphics_name,python_inventory);active[f]=(name,gpu);persist()
                if not active:
                    if fatal is not None:raise RuntimeError(fatal)
                    if all(s['status'] in ['completed','already_complete'] for s in state.values()):break
                    raise RuntimeError('No runnable task; unresolved dependency or failure')
                completed,_=cf.wait(active,return_when=cf.FIRST_COMPLETED)
                for f in completed:
                    name,gpu=active.pop(f)
                    if gpu:available.append(gpu)
                    else:cpu_busy=False
                    try:state[name].update(status='completed',result=f.result())
                    except BaseException:
                        fatal=traceback.format_exc();state[name].update(status='failed',error=fatal)
                        write(out/'events'/f'{name}_failed.json',dict(status='failed',error=fatal,completed_unix=time.time(),healthy_running_branches_preserved=True));persist()
                persist()
        write(out/'complete.json',dict(status='awaiting_review',tasks=state,identity=entry(out/'identity.json'),completion='all tasks exited; checkpoint hashes/full300metrics and sampledROI diagnostics accepted automatically'))
    except BaseException:
        write(out/f'failure_{time.time_ns()}.json',dict(status='failed_preserved',tasks=state,error=traceback.format_exc(),active_tasks=list(active.values()),automatic_restart=False));raise
if __name__=='__main__':main()
