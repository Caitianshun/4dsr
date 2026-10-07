"""Durable preparation/paired workers; child exit directly triggers verification.

Run with system Python; the CUDA runtime is a separately named subprocess.
GPU ownership is never inferred from a low utilization sample. No worker waits
for a foreign training PID or captures an inter-job gap. The root operator must
dispatch only after resolving known queued jobs or a reserved resource window.
"""
from __future__ import annotations
import argparse,fcntl,os,platform,shlex,socket,subprocess,time,traceback,sys
from fp_common import ROOT,HERE,OUT,read,write,sha,entry,bound,module,source_identity
from config import METHODS,ARMS

PYTHON='/home/cai_tianshun/Project/4dgs/.venv/bin/python'
sys.path.insert(1,str(ROOT/'experiments/dynamic_sr_confidence_geometry_20261006'))
legacy=module('fp_proven_process_lifecycle',ROOT/'experiments/dynamic_sr_confidence_geometry_20261006/worker_async.py')

class ResourceBusy(RuntimeError): pass

def sample(gpu):
    raw=subprocess.check_output(['nvidia-smi','-i',gpu,
        '--query-gpu=uuid,name,memory.total,memory.used,utilization.gpu,driver_version',
        '--format=csv,noheader,nounits'],text=True).strip()
    uuid,name,total,used,util,driver=[s.strip() for s in raw.split(',')]
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory',
                                 '--format=csv,noheader,nounits'],text=True)
    rows=[r for r in apps.splitlines() if uuid in r]
    # Display-only ToDesk is already documented; every other compute process
    # remains protected, even a newly launched zero-utilization process.
    foreign=[r for r in rows if '/opt/todesk/' not in r]
    return dict(at_unix=time.time(),uuid=uuid,name=name,total_MiB=float(total),used_MiB=float(used),
                utilization=float(util),driver=driver,compute_rows=rows,foreign=foreign,
                load=list(os.getloadavg()),host=socket.gethostname(),kernel=platform.release())

def free_sample(gpu):
    result=sample(gpu)
    if result['foreign']:raise ResourceBusy(str(result))
    return result

def check_resource(gpu,status,interval=30):
    first=free_sample(gpu)
    write(status,dict(status='resource_second_sample_pending',first=first,interval_seconds=interval))
    time.sleep(interval)
    second=free_sample(gpu)
    assert first['uuid']==second['uuid'] and first['name']==second['name']
    write(status,dict(status='resource_double_sample_passed',first=first,second=second))
    return second

def run(command,log,env,resource_gpu=None):
    if resource_gpu:free_sample(resource_gpu) # immediately before every GPU child
    legacy.call(command,log,env) # reaps private descendants before releasing lock

def environment(gpu):
    return dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')

def stage_files():
    # Bind actual GPU-preparation/training dependencies. Data reception and
    # result writers are independent and may finish while caches are prepared.
    result=source_identity()
    for name in ('calibrate.py','diagnostics.py','footprint_cuda_checks.py','run_suite.py'):
        path=HERE/name;result[str(path.relative_to(ROOT))]=sha(path)
    return result

def plan():
    protocol=read(OUT/'protocol.json');bound(protocol['parent']);bound(protocol['manifest'])
    table=read(OUT/'schedules/index.json');assert table['status']=='completed'
    for identity in table['schedules']:bound(identity)
    result=dict(status='registered_waiting_for_native_preparation_and_resources',
        protocol=entry(OUT/'protocol.json'),schedules=entry(OUT/'schedules/index.json'),
        preparation=['native CUDA footprint fixture','1140 native HR parent moments',
                     '8570 frozen directed pairs, CPU','single balanced train76 calibration'],
        six_arm_order=['B0','X','Bsync','M','MX','E'],repeats=['1','2'],
        first100=dict(repeat='1',methods=['B0','X'],formal_budget_included=True),
        formal_updates=72000,RGB_forwards=216000,moment_forwards=72000,Adam_calls=144000,
        endpoint_observations=196,endpoint_fixed_diagnostic_RGB=228,endpoint_fixed_diagnostic_moments=228,
        protected_foreign_jobs=True,automatic_gap_capture=False,
        hardware='each suffix all six arms same host/platform/GPU model/software; evaluation one fixed platform',
        continuation='atomic complete model/two Adams/RNG; archive failed work; never cold restart silently',
        full_scene_followthrough=protocol['full_scene_followthrough'])
    path=OUT/'suite_plan.json'
    if path.exists():assert read(path)==result,'Suite registration changed'
    else:write(path,result)
    return result

def prepare(a,status,env):
    # File identities are held fixed across preparation. Any amended source
    # requires an explicit archived incident, never silent cache reuse.
    frozen=stage_files()
    fixture=OUT/'operator_checks/footprint_cuda/latest.json'
    if not fixture.exists() or read(fixture).get('status')!='passed_native_CUDA_X_gradient_fixture':
        write(status,dict(status='running_native_footprint_fixture',gpu=a.gpu,sources=frozen))
        run([a.python,'-u',str(HERE/'footprint_cuda_checks.py'),'--gpu-uuid',a.gpu,
             '--lock',str(OUT/'locks'/f'{a.gpu}.native_fixture.lock')],OUT/'preparation/native_fixture.log',env,a.gpu)
    # Verify the fixture's own bound source/identity fields before parent export.
    verify_fixture()
    parent=OUT/'support/parent_moments_hr/index.json'
    if not parent.exists():
        write(status,dict(status='exporting_1140_native_HR_parent_moments',gpu=a.gpu,sources=frozen))
        run([a.python,'-u',str(HERE/'support_cache.py'),'--mode','export'],OUT/'preparation/parent_export.log',env,a.gpu)
    value=read(parent);assert value['status']=='completed_HR_parent_moments' and len(value['entries'])==1140
    assert value['parent_sha256']==read(OUT/'protocol.json')['parent']['sha256']
    support=OUT/'support/frozen/index.json'
    if not support.exists():
        write(status,dict(status='preparing_frozen_directed_support_CPU',parent=entry(parent),gpu_not_used=True))
        run([a.python,'-u',str(HERE/'support_cache.py'),'--mode','prepare','--device','cpu'],
            OUT/'preparation/support_prepare.log',dict(env,CUDA_VISIBLE_DEVICES=''))
    value=read(support);assert value['status']=='completed_frozen_X_support'
    calibration=OUT/'calibration.json'
    if not calibration.exists():
        write(status,dict(status='calibrating_train76_once',gpu=a.gpu,support=entry(support)))
        run([a.python,'-u',str(HERE/'calibrate.py')],OUT/'preparation/calibration.log',env,a.gpu)
    value=read(calibration);assert value['status']=='passed'
    for identity in value['source_files'].values():bound(identity)
    assert frozen==stage_files(),'Preparation source changed; preserve outputs before amendment'
    run([a.python,str(HERE/'diagnostics.py'),'--reuse-parent','--label','U6000'],
        OUT/'preparation/reuse_parent_diagnostics.log',dict(env,CUDA_VISIBLE_DEVICES=''))
    write(OUT/'preparation/complete.json',dict(status='completed_native_support_single_calibration',
        protocol=entry(OUT/'protocol.json'),fixture=entry(verify_fixture()),parent=entry(parent),support=entry(support),
        calibration=entry(calibration),sources=frozen,formal_updates=0))

def verify_fixture():
    latest=read(OUT/'operator_checks/footprint_cuda/latest.json')
    assert latest['status']=='passed_native_CUDA_X_gradient_fixture','CPU tests are insufficient'
    path=bound(latest['receipt']);value=read(path)
    assert value['status']==latest['status'] and value['counters']['parameter_updates']==0
    for identity in value['source'].values():bound(identity)
    protocol=read(OUT/'protocol.json')
    assert value['parent']==protocol['parent'] and value['manifest']==protocol['manifest']
    return path

def endpoint(task):
    directory=OUT/'runs'/task;receipt=read(directory/'complete.json')
    method=task.split('_',1)[1];assert receipt['status']=='completed_training' and receipt['method']==method
    assert receipt['updates']==6000 and receipt['training_rgb_forwards']==18000
    assert receipt['moment_forwards']==(18000 if ARMS[method]['X'] else 0) and receipt['adam_calls']==12000
    cp=bound(receipt['checkpoint']);assert cp==directory/'checkpoint_12000.pt'
    position=0
    for identity in receipt['segments']:
        segment=read(bound(identity));assert segment['start']==position
        bound(segment['checkpoint']);position=segment['suffix_endpoint']
    assert position==6000
    return cp

def evaluate(a,task,env):
    cp=endpoint(task)
    evaluation=OUT/'evaluation'/task/'extra/adapter_complete.json'
    if not evaluation.exists():
        run([a.python,'-u',str(HERE/'evaluate.py'),'--checkpoint',str(cp),'--label',task],
            OUT/'evaluation'/task/'evaluation.log',env,a.gpu)
    value=read(evaluation);assert value['observations']==196 and value['checkpoint']==entry(cp)
    for key in ('adapter','original_evaluator','extra_complete'):bound(value[key])
    extra=read(bound(value['extra_complete']))
    platform_path=OUT/'evaluation_platform.json'
    identity=dict(host=socket.gethostname(),GPU_model=extra['gpu'],torch=str(extra['torch']),
        cuda=extra['cuda'],extension_sha256=extra['extension_sha256'],evaluator=extra['identity']['source_sha256'])
    with (OUT/'locks/evaluation_platform.lock').open('a') as platform_lock:
        fcntl.flock(platform_lock,fcntl.LOCK_EX)
        if platform_path.exists():assert read(platform_path)==identity,'Uniform evaluation platform changed'
        else:write(platform_path,identity)
    diag=OUT/'diagnostics'/task/'diagnostics.json'
    if not diag.exists():
        run([a.python,'-u',str(HERE/'diagnostics.py'),'--checkpoint',str(cp),'--label',task],
            OUT/'diagnostics'/task/'diagnostics.log',env,a.gpu)
    value=read(diag);assert value['cost']['formal_updates']==0 and value['cost']['Adam_calls']==0
    assert value['checkpoint']==entry(cp)
    bound(value['immutability_audit']);bound(value['gdiag'])
    write(OUT/'tasks'/f'{task}.json',dict(status='completed_training_evaluation_fixed_diagnostics',
        task=task,training=entry(OUT/'runs'/task/'complete.json'),checkpoint=entry(cp),
        evaluation=entry(evaluation),diagnostics=entry(diag),GPU=a.gpu,host=socket.gethostname()))

def continuation(directory):
    if not directory.exists():return None
    if not (directory/'config.json').exists():raise ValueError('Existing unregistered run directory preserved')
    checkpoints=[]
    for sidecar in directory.glob('checkpoint_*.json'):
        value=read(sidecar);cp=bound(value);assert cp.suffix=='.pt'
        checkpoints.append((value['metadata']['suffix_step'],cp))
    if not checkpoints:raise ValueError('Interrupted before first durable checkpoint; preserve cost and explicitly resolve before retry')
    return max(checkpoints)[1]

def train(a,method,stop,env,status):
    task=f'r{a.repeat}_{method}';directory=OUT/'runs'/task
    if (directory/'complete.json').exists():endpoint(task);return
    resume=continuation(directory)
    if resume:
        cursor=read(resume.with_suffix('.json'))['metadata']['suffix_step']
        if cursor==6000:
            run([a.python,str(HERE/'finalize_endpoint.py'),'--task',task],OUT/'logs'/f'{task}_finalization.log',
                dict(env,CUDA_VISIBLE_DEVICES=''))
            endpoint(task);return
        if cursor>=stop:return
    write(status,dict(status='training',task=task,stop=stop,gpu=a.gpu,
                      resume=str(resume) if resume else None,started_unix=time.time()))
    command=[a.python,'-u',str(HERE/'train.py'),'--method',method,'--repeat',a.repeat,
             '--out',str(directory),'--stop',str(stop)]
    if resume:command+=['--resume',str(resume)]
    run(command,OUT/'logs'/f'{task}.log',env,a.gpu)
    if stop==6000:endpoint(task)

def worker(a,status,env,resource):
    prep=read(OUT/'preparation/complete.json');assert prep['status']=='completed_native_support_single_calibration'
    for key in ('protocol','fixture','parent','support','calibration'):bound(prep[key])
    platform_path=OUT/'workers'/f'suffix_{a.repeat}_platform.json'
    identity=dict(host=socket.gethostname(),GPU_model=resource['name'],physical_GPU=resource['uuid'],
                  driver=resource['driver'],python=a.python,protocol=entry(OUT/'protocol.json'))
    if platform_path.exists():assert read(platform_path)==identity,'A suffix training platform changed'
    else:write(platform_path,identity)
    if a.repeat=='1':
        for method in ('B0','X'):train(a,method,100,env,status)
        for method in ('B0','X'):
            directory=OUT/'runs'/f'r1_{method}'
            assert directory.exists()
        write(OUT/'first100_timing.json',dict(status='completed_formal_100_update_segments',
            B0=entry(OUT/'runs/r1_B0/attempt_complete_0000_0100.json'),
            X=entry(OUT/'runs/r1_X/attempt_complete_0000_0100.json'),formal_updates_included=True,
            quality_not_used_for_early_elimination=True))
    for method in ('B0','X','Bsync','M','MX','E'):
        train(a,method,6000,env,status)
        task=f'r{a.repeat}_{method}'
        write(status,dict(status='endpoint_completed_immediate_evaluation',task=task,gpu=a.gpu))
        if a.evaluate_here:evaluate(a,task,env)
        else:
            write(OUT/'evaluation_queue'/f'{task}.json',dict(status='pending_uniform_evaluation',
                checkpoint=entry(endpoint(task)),task=task,trigger='successful training child process return'))
    write(status,dict(status='suffix_completed',repeat=a.repeat,gpu=a.gpu,methods=list(METHODS),
                      evaluation_here=a.evaluate_here))

def main(a):
    plan()
    if a.phase=='plan':return
    if a.phase=='verify':
        tasks=[f'r{r}_{m}' for r in ('1','2') for m in METHODS]
        completed=[]
        for task in tasks:
            if (OUT/'tasks'/f'{task}.json').exists():endpoint(task);completed.append(task)
        write(OUT/'suite_integrity.json',dict(status='completed_core' if len(completed)==12 else 'partial_core',
            completed=completed,updates=6000*len(completed),planned_updates=72000))
        return
    assert a.gpu and a.gpu.startswith('GPU-'),'Bind the verified physical GPU UUID to avoid aliasing resource locks'
    assert a.operator_resource_resolved,'Known queued jobs require root scheduling resolution before dispatch'
    (OUT/'locks').mkdir(parents=True,exist_ok=True)
    status=OUT/'workers'/f'{a.phase}_r{a.repeat}.json'
    with (OUT/'locks'/f'{a.gpu}.controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        resource=check_resource(a.gpu,status);env=environment(a.gpu)
        if a.phase=='prepare':prepare(a,status,env)
        elif a.phase=='worker':worker(a,status,env,resource)
        elif a.phase=='evaluate':
            assert a.task;evaluate(a,a.task,env)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=['plan','prepare','worker','evaluate','verify'],required=True)
    p.add_argument('--gpu');p.add_argument('--python',default=PYTHON);p.add_argument('--repeat',choices=['1','2'],default='1')
    p.add_argument('--task');p.add_argument('--evaluate-here',action='store_true',default=True)
    p.add_argument('--defer-evaluation',dest='evaluate_here',action='store_false',
        help='Only for an already registered durable unified evaluator callback; not the normal dispatch')
    p.add_argument('--operator-resource-resolved',action='store_true',help='Root resolved known competing queue/reservation; this does not bypass occupancy checks')
    a=p.parse_args()
    try:main(a)
    except ResourceBusy:
        write(OUT/'workers'/f'{a.phase}_r{a.repeat}.json',dict(status='waiting_for_free_GPU_no_foreign_process_interrupted',
              traceback=traceback.format_exc(),formal_work_not_launched=True))
        raise SystemExit(75)
    except BaseException:
        write(OUT/'workers'/f'{a.phase}_r{a.repeat}_failed_{time.time_ns()}.json',dict(status='failed_preserved',traceback=traceback.format_exc()))
        raise
