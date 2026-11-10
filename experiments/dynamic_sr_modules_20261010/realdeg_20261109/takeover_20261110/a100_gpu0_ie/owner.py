#!/usr/bin/env python3
"""One hash-bound IE-Cook migration; CPU parent, isolated GPU children.

Default validates/displays the plan without importing Torch. Execution requires
the real fixed-service context and the root-activated local ownership handoff.
No original local-job completion is emitted. Existing output is never adopted.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import socket
import stat
import subprocess
import sys
import time
import traceback

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
BASE_REL = 'experiments/dynamic_sr_modules_20261010/realdeg_20261109/baselines'
BASE = ROOT / BASE_REL
sys.path.insert(0, str(BASE))  # CPU-only helpers are also used by the parent.
REMOTE_ROOT = '/home/ubuntu/3DGS/4dsr'
PYTHON = REMOTE_ROOT + '/.venv/bin/python'
GPU = 'GPU-5c08f287-3ffd-edf9-91ed-cd6db2690f3b'
JOB = 'rd4-g0-ie-cook-01'
ORIGINAL_JOB = 'rd-ie-cook'
SCENE = 'cook_spinach'
METHOD = 'iesrgs'
MIN_FREE = 20 * 1024 ** 3


def require(value, message):
    if not value:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def safe_path(root, relative):
    p = PurePosixPath(relative)
    require(isinstance(relative, str) and str(p) == relative and not p.is_absolute()
            and '..' not in p.parts and '\\' not in relative, 'Unsafe relative path')
    full = Path(root) / relative
    require(not any(p.is_symlink() for p in (full, *full.parents)), 'Symlink path refused: ' + str(full))
    return full


def ref(root, path):
    path = Path(path)
    return dict(path=path.relative_to(root).as_posix(), bytes=path.stat().st_size, sha256=sha(path))


def check_ref(root, value):
    relative = PurePosixPath(value['path'])
    require(not relative.is_absolute() and '..' not in relative.parts and str(relative) == value['path'], 'Unsafe input path')
    path = Path(root) / value['path']
    require(path.resolve().is_relative_to(Path(root).resolve()), 'Read-only alias escapes project root')
    require(path.is_file() and stat.S_ISREG(path.stat().st_mode), 'Missing regular input: ' + str(path))
    require(sha(path) == value['sha256'], 'Bound input changed: ' + str(path))
    return path


def write_new(path, value):
    path = Path(path)
    require(not any(p.is_symlink() for p in (path, *path.parents)), 'Symlink output refused')
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644)
    with os.fdopen(fd, 'w') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def atomic(path, value):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp-' + str(os.getpid()))
    write_new(temp, value)
    os.replace(temp, path)


def adapted_task(original, plan):
    task = copy.deepcopy(original)
    task['python'] = plan['python']
    if '--task-root' in task['argv']:
        index = task['argv'].index('--task-root') + 1
        require(task['argv'][index] == plan['original_task_root'], 'Unexpected original task root')
        task['argv'][index] = plan['task_root']
    return task


def validate_plan(plan, root=ROOT):
    require(plan['schema'] == 'rd4_ie_cook_gpu0_owner_plan_v1'
            and plan['job_id'] == JOB and plan['original_job_id'] == ORIGINAL_JOB
            and plan['method'] == METHOD and plan['scene'] == SCENE, 'Foreign owner cell')
    require(plan['project_root'] == REMOTE_ROOT and plan['python'] == PYTHON
            and plan['gpu_uuid'] == GPU and plan['cuda_visible_devices'] == '0'
            and plan['min_free_bytes'] == MIN_FREE, 'Changed host/GPU/interpreter/guard')
    require(plan['task_root'] == REMOTE_ROOT + '/tasks/static_sr_shared_wu_20261024_a100', 'Foreign frozen task root')
    require(plan['coarse_iterations'] == 3000 and plan['fine_iterations'] == 14000, 'Changed scientific schedule')
    require(plan['output_root'] == 'output/dynamic_sr_modules_20261010/realdeg_20261109/baselines/iesrgs/cook_spinach'
            and plan['ownership_file'] == plan['output_root'] + '/owner_a100_gpu0.json', 'Foreign output/ownership scope')
    check_ref(root, plan['config'])
    check_ref(root, plan['original_request'])
    check_ref(root, plan['original_source_lock'])
    for item in plan['source_bindings']:
        check_ref(root, item)
    lane = read(check_ref(root, plan['original_lane']))
    require(lane['job_id'] == ORIGINAL_JOB and lane['method'] == METHOD and lane['scene'] == SCENE
            and lane['host'] == 'local' and lane['no_parallel_training'] is True, 'Unexpected original lane')
    require([t['id'] for t in lane['tasks']] == [ORIGINAL_JOB + x for x in ('-teacher', '-s0', '-s1', '-s2', '-s3')],
            'Changed original task order')
    tasks = [adapted_task(t, plan) for t in lane['tasks']]
    require(tasks == plan['adapted_tasks'], 'Host adaptation differs from exact allowed transformation')
    for task in tasks:
        if task['source']:
            check_ref(root, task['source'])
        require(task['method'] == METHOD and task['scene'] == SCENE and task['host'] == 'local', 'Foreign task')
    require(tasks[1]['argv'][tasks[1]['argv'].index('--stage') + 1] == 'internal'
            and tasks[2]['argv'][1] == 'iedepth'
            and tasks[3]['argv'][tasks[3]['argv'].index('--stage') + 1] == 'sr'
            and tasks[4]['argv'][tasks[4]['argv'].index('--scope') + 1] == 'test', 'Changed stage semantics')
    return lane


def runtime():
    sys.path.insert(0, str(BASE))
    import runtime_rd
    return runtime_rd


def execution_context(plan):
    require(str(ROOT) == REMOTE_ROOT and str(Path(sys.executable).absolute()) == PYTHON, 'Wrong execution root/interpreter')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '0', 'Exactly physical A100 GPU0 required')
    jobdir = os.environ.get('CLAUDE_REMOTE_JOB_DIR')
    require(jobdir, 'Fixed remote service context required')
    jobdir = Path(jobdir)
    require(jobdir.name == JOB and jobdir.is_relative_to(ROOT / '.claude-training'), 'Foreign remote service job')
    context = read(jobdir / 'execution_context.json')
    request = read(jobdir / 'request.json')
    require(context.get('job_id') == JOB and context.get('project_id') == '4dsr'
            and context.get('project_root') == REMOTE_ROOT and context.get('selected_gpu_uuids') == [GPU],
            'Wrong fixed executor admission context')
    require(request.get('job_id') == JOB and request.get('operation') == 'run'
            and request.get('remote_host') == 'a100-train' and request.get('project_id') == '4dsr'
            and request.get('cuda_visible_devices') == '0', 'Foreign fixed executor request')
    require(context['boot_id'] == Path('/proc/sys/kernel/random/boot_id').read_text().strip(), 'Stale service context')
    pid = os.getpid()
    ancestors = set()
    while pid > 1:
        require(pid not in ancestors, 'Invalid process ancestry')
        ancestors.add(pid)
        fields = Path('/proc', str(pid), 'stat').read_text().rsplit(')', 1)[1].split()
        if pid == context['supervisor_pid']:
            require(int(fields[19]) == context['startticks'] and fields[0] not in ('Z', 'X'), 'Stale supervisor identity')
            return dict(context=context, context_file=str(jobdir / 'execution_context.json'),
                        request_sha256=sha(jobdir / 'request.json'), ancestors=sorted(ancestors))
        pid = int(fields[1])
    raise RuntimeError('Runner is not a descendant of the admitted fixed-service supervisor')


def handoff(plan):
    delegation_path = safe_path(ROOT, plan['handoff']['delegation_path'])
    delegation = read(delegation_path)
    require(delegation.get('schema') == 'rd4_local_baseline_owner_delegation_v1'
            and delegation.get('original_job_id') == ORIGINAL_JOB
            and delegation.get('method') == METHOD and delegation.get('scene') == SCENE
            and delegation.get('remote_owner_job_id') == JOB
            and delegation.get('target_host') == 'a100-train' and delegation.get('target_gpu_uuid') == GPU
            and delegation.get('state') == 'ownership_assigned_not_completion'
            and delegation.get('remote_completion_claimed') is False
            and delegation.get('auto_local_fallback') is False, 'Wrong ownership delegation')
    for field, path in [('remote_runner', HERE / 'owner.py'), ('remote_plan', HERE / 'plan.json')]:
        declared = delegation[field]
        require(declared['sha256'] == sha(path)
                and declared['path'] == path.relative_to(ROOT).as_posix(), 'Delegation does not bind final runner/plan')
    for field, bound in [('lane', 'original_lane'), ('config', 'config'),
                         ('request', 'original_request'), ('source_lock', 'original_source_lock')]:
        require(delegation['original'][field]['sha256'] == plan[bound]['sha256'], 'Delegated original ' + field + ' differs')
    activation_path = safe_path(ROOT, plan['handoff']['activation_path'])
    activation = read(activation_path)
    require(activation.get('status') == 'active' and activation.get('original_job_id') == ORIGINAL_JOB
            and activation.get('remote_owner_job_id') == JOB
            and activation.get('old_waiting_controller_stopped') is True
            and activation.get('replacement_excludes_original_job') is True
            and activation.get('delegation_sha256') == sha(delegation_path)
            and activation.get('owner_plan_sha256') == sha(HERE / 'plan.json')
            and activation.get('owner_runner_sha256') == sha(HERE / 'owner.py'),
            'Local controller has not activated the duplicate-preventing ownership handoff')
    return dict(delegation=ref(ROOT, delegation_path), activation=ref(ROOT, activation_path))


def asset_checks(plan, *, depth=False):
    require(plan.get('asset_status') == 'accepted', 'Asset plan is not yet accepted by the owner')
    bindings = [*plan['frozen_sources'], *plan['gpu_runtime_bindings']]
    if depth:
        bindings += [*plan['depth_source_bindings'], plan['depth_weights']]
    verified = []
    for item in bindings:
        path = check_ref(ROOT, item)
        verified.append(dict(alias=item['path'], resolved=str(path.resolve()), sha256=item['sha256']))
    return verified


def fresh_outputs(plan):
    base = safe_path(ROOT, plan['output_root'])
    require(not base.exists(), 'Existing IE-Cook output needs explicit owner review; never adopt/rerun')
    return base


def gpu_preflight(plan, *, depth=False):
    """Fresh child only. No optimizer steps and no real image pixels are read."""
    verified_assets = asset_checks(plan, depth=depth)
    rt = runtime()
    cfg = rt.check_active_config(METHOD, SCENE)
    rt.metadata(SCENE, verify_init=True)
    driver = rt.load('_ie_gpu0_preflight_driver', plan['adapted_tasks'][1]['source']['path'])
    rt.setup_static_adapter(driver)
    mods = driver.setup(plan['task_root'])
    torch = mods['torch']
    if depth:
        package = (ROOT / plan['depth_package']).resolve()
        require(package.is_relative_to(ROOT), 'Depth package alias escapes project')
        sys.path.insert(0, str(package))
        from depth_anything_v2.dpt import DepthAnythingV2
        model = DepthAnythingV2(encoder='vitl', features=256, out_channels=[256, 512, 1024, 1024])
        state = torch.load(ROOT / plan['depth_weights']['path'], map_location='cpu', weights_only=True, mmap=True)
        expected = model.state_dict()
        require(set(state) == set(expected) and all(state[k].shape == expected[k].shape
                and state[k].dtype == expected[k].dtype for k in expected), 'DA-V2-L class/key/shape/dtype mismatch')
        return dict(status='passed', kind='DA_weights_class_shape_CPU_acceptance', GPU_executed=False,
                    weights=plan['depth_weights'], architecture=dict(encoder='vitl', features=256,
                    out_channels=[256, 512, 1024, 1024]), tensors=len(state), no_inference=True,
                    verified_assets=verified_assets)
    sys.path.insert(0, str(ROOT / 'experiments/dynamic_sr_cull_retrain_20261009'))
    import rt as wu_runtime
    wu = wu_runtime.check_runtime()
    require(torch.cuda.device_count() == 1, 'Exactly one visible CUDA device required')
    actual = subprocess.check_output(['/usr/bin/nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'], text=True)
    masks = {row.split(',')[0].strip(): row.split(',')[1].strip() for row in actual.strip().splitlines()}
    require(masks.get('0') == GPU, 'Physical GPU0 UUID changed')
    require('A100' in torch.cuda.get_device_name(0), 'A100 hardware required')
    from simple_knn._C import distCUDA2
    points = torch.tensor([[0., 0., 2.], [.1, 0., 2.], [0., .1, 2.], [0., 0., 2.1]], device='cuda')
    distances = distCUDA2(points)
    require(bool(torch.isfinite(distances).all()) and bool((distances > 0).all()), 'Wu simple_knn CUDA failure')
    from types import SimpleNamespace
    from utils.graphics_utils import getProjectionMatrix
    from methods.iesrgs_renderer import raster_api, render_state
    api = raster_api()
    projection = getProjectionMatrix(znear=.01, zfar=10., fovX=math.pi/3, fovY=math.pi/3).transpose(0, 1).cuda()
    camera = SimpleNamespace(image_width=64, image_height=64, FoVx=math.pi/3, FoVy=math.pi/3,
        world_view_transform=torch.eye(4, device='cuda'), full_proj_transform=projection,
        camera_center=torch.zeros(3, device='cuda'), time=0.)
    xyz = torch.tensor([[0., 0., 2.]], device='cuda', requires_grad=True)
    state = dict(xyz=xyz, scales=torch.full((1, 3), .25, device='cuda', requires_grad=True),
        opacity=torch.full((1, 1), .95, device='cuda', requires_grad=True),
        rotation=torch.tensor([[1., 0., 0., 0.]], device='cuda', requires_grad=True),
        sh=torch.full((1, 1, 3), .1, device='cuda', requires_grad=True))
    packet = render_state(camera, state, torch.zeros(3, device='cuda'), 0, [camera], require_depth=True, api=api)
    for field in ('render', 'median_depth', 'expected_depth', 'mask'):
        require(bool(torch.isfinite(packet[field]).all()), 'Nonfinite RADE ' + field)
    gradient, = torch.autograd.grad(packet['median_depth'].sum(), xyz, retain_graph=True)
    require(bool(torch.isfinite(gradient).all()) and float(gradient.abs().sum()) > 0,
            'RADE median-depth backward is detached, zero, or nonfinite')
    rgb_gradient, = torch.autograd.grad(packet['render'].sum(), state['sh'])
    require(bool(torch.isfinite(rgb_gradient).all()) and float(rgb_gradient.abs().sum()) > 0, 'RADE RGB backward failed')
    torch.cuda.synchronize()
    return dict(status='passed', kind='isolated_GPU_dependencies', GPU_executed=True,
        scientific_result=False, optimizer_steps=0, full_scene_model_created=False,
        hardware=dict(host=socket.gethostname(), uuid=GPU, visible='0', model=torch.cuda.get_device_name(0),
                      torch=torch.__version__, cuda=torch.version.cuda),
        wu_runtime=wu, RADE_eight_tensor_contract=True, verified_assets=verified_assets,
        median_depth_xyz_gradient_absolute_sum=float(gradient.abs().sum()),
        RGB_SH_gradient_absolute_sum=float(rgb_gradient.abs().sum()), frozen_sources=driver.frozen_identity())


def execute_task(plan, task_id):
    task = next((t for t in plan['adapted_tasks'] if t['id'] == task_id), None)
    require(task is not None, 'Unknown original task')
    asset_checks(plan, depth=task_id in (ORIGINAL_JOB + '-s1', ORIGINAL_JOB + '-s2'))
    rt = runtime()
    rt.metadata(SCENE, verify_init=True)
    rt.check_active_config(METHOD, SCENE)
    if task['mode'] == 'teacher_index':
        from teacher_rd import verify_consumer
        return verify_consumer(SCENE)
    rt.dispatch_original(task)
    return dict(status='original_task_returned_exit0', task=task_id)


def run_child(plan, flag, value, log_root, env, *, invoke=subprocess.run):
    name = value if flag == '--task' else flag.removeprefix('--')
    argv = [plan['python'], '-u', str(HERE / 'owner.py'), '--plan', str(HERE / 'plan.json'), flag]
    if value:
        argv.append(value)
    argv.append('--execute')
    logfile = log_root / (name + '.log')
    started = time.monotonic()
    with logfile.open('x') as stream:
        result = invoke(argv, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
    record = dict(task=name, argv=argv, returncode=result.returncode,
                  seconds=round(time.monotonic() - started, 3), log=ref(ROOT, logfile))
    write_new(log_root / (name + '.exit.json'), record)
    require(result.returncode == 0, 'Child failed; no subsequent task launched: ' + name)
    return record


def final_returns(plan, log_root):
    from returns_rd import render_files, selected_record
    files = render_files(METHOD, SCENE)
    base = ROOT / plan['output_root']
    required = ['internal/checkpoint.pt', 'internal/run.json', 'internal/complete.json',
                'sr/checkpoint.pt', 'sr/run.json', 'sr/complete.json', 'depth/index.json',
                'renders_test/render_index.json', plan['ownership_file'].split(plan['output_root'] + '/')[1]]
    for relative in required:
        require((base / relative).is_file(), 'Missing return evidence/checkpoint: ' + relative)
    for stage, expected in [('internal', 'iesrgs_internal'), ('sr', 'sr')]:
        receipt = read(base / stage / 'complete.json')
        require(receipt['status'] == 'completed' and receipt['stage'] == stage
                and receipt['coarse'] == 3000 and receipt['fine'] == 14000
                and receipt['metadata']['stage'] == expected
                and receipt['metadata']['optimizer_updates'] == 16999
                and receipt['config']['sha256'] == plan['config']['sha256'], 'Changed/incomplete scientific stage')
        checkpoint = receipt['checkpoint']
        require(sha(base / stage / 'checkpoint.pt') == checkpoint['sha256'], 'Checkpoint receipt SHA mismatch')
    selected = {str(ROOT / f['path']) for f in files}
    for p in sorted(base.rglob('*')):
        if p.is_file() and not p.is_symlink():
            relative = p.relative_to(base).as_posix()
            if relative in required or selected_record(METHOD, relative) or (p.is_relative_to(log_root) and p.name != 'owner.lock'):
                selected.add(str(p))
    for binding in [*plan['source_bindings'], plan['original_lane'], plan['config'], plan['original_request']]:
        selected.add(str(check_ref(ROOT, binding)))
    selected.update(str(ROOT / plan['handoff'][k]) for k in ('delegation_path', 'activation_path'))
    selected.update(str(p) for p in (HERE / 'owner.py', HERE / 'plan.json'))
    return [ref(ROOT, Path(p)) for p in sorted(selected)]


def run(plan):
    context = execution_context(plan)
    binding = handoff(plan)
    asset_checks(plan)
    base = fresh_outputs(plan)
    from disk_guard import check as disk_guard
    disk_guard(ROOT, 20)
    base.mkdir(parents=True, exist_ok=False)
    owner = dict(schema='rd4_ie_migrated_owner_v1', state='claimed_not_completed', original_job_id=ORIGINAL_JOB,
        remote_owner_job_id=JOB, GPU_UUID=GPU, original_lane=plan['original_lane'], config=plan['config'],
        owner_source=ref(ROOT, HERE / 'owner.py'), plan=ref(ROOT, HERE / 'plan.json'),
        handoff=binding, execution=context, claimed_unix=time.time(), original_local_completion_emitted=False)
    write_new(ROOT / plan['ownership_file'], owner)
    log_root = base / 'jobs' / JOB
    log_root.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
        IESRGS_RASTERIZER_ROOT=str(ROOT / plan['rade_runtime']))
    status = dict(schema='rd4_ie_gpu0_execution_v1', status='running', job_id=JOB, original_job_id=ORIGINAL_JOB,
        began_unix=time.time(), owner=ref(ROOT, ROOT / plan['ownership_file']), steps=[], current=None)
    def persist():
        atomic(log_root / 'status.json', status)
    def step(flag, value=''):
        disk_guard(ROOT, 20)
        status['current'] = value or flag
        persist()
        item = run_child(plan, flag, value, log_root, env)
        status['steps'].append(item)
        persist()
    with (log_root / 'owner.lock').open('x') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            step('--preflight')
            require(read(log_root / 'gpu_preflight.json').get('status') == 'passed', 'GPU dependency receipt absent')
            for task in plan['adapted_tasks']:
                if task['id'] == ORIGINAL_JOB + '-s1':
                    step('--preflight-depth')
                    require(read(log_root / 'depth_preflight.json').get('status') == 'passed', 'Depth dependency receipt absent')
                step('--task', task['id'])
            from returns_rd import verify_render
            verified = verify_render(METHOD, SCENE)
            # Check both scientific stages and hash all required return assets
            # before creating the completion receipt. Do not claim success on
            # render count alone.
            files = final_returns(plan, log_root)
            status.update(status='completed', current=None, ended_unix=time.time(), render_verification=verified)
            persist()
            write_new(log_root / 'complete.json', status)
            fresh = {r['path']: r for r in files}
            for path in (log_root / 'status.json', log_root / 'complete.json'):
                item = ref(ROOT, path)
                fresh[item['path']] = item
            files = [fresh[k] for k in sorted(fresh)]
            write_new(log_root / 'return-full.json', dict(version=1, files=files))
            write_new(log_root / 'download.json', dict(version=1, files=[dict(path=f['path']) for f in files]))
            print(json.dumps(dict(status='completed_return_ready_not_transferred', job_id=JOB,
                                 cam00_PNGs=verified['cam00_PNGs'], files=len(files))), flush=True)
        except BaseException:
            status.update(status='failed', ended_unix=time.time(), error=traceback.format_exc())
            persist()
            write_new(log_root / 'failure.json', status)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', default=str(HERE / 'plan.json'))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--task')
    mode.add_argument('--preflight', action='store_true')
    mode.add_argument('--preflight-depth', action='store_true')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    require(Path(args.plan).resolve() == HERE / 'plan.json', 'Only the colocated owner plan accepted')
    plan = read(args.plan)
    validate_plan(plan)
    if not args.execute:
        print(json.dumps(dict(status='CPU_plan_valid_no_execution', plan=ref(ROOT, Path(args.plan)),
                             asset_status=plan['asset_status'], GPU_calls=0, service_calls=0), indent=2))
        return
    execution_context(plan)
    handoff(plan)
    if args.task or args.preflight or args.preflight_depth:
        owner = read(ROOT / plan['ownership_file'])
        require(owner['remote_owner_job_id'] == JOB and owner['plan'] == ref(ROOT, Path(args.plan)),
                'Missing/different exclusive output owner')
        state = read(ROOT / plan['output_root'] / 'jobs' / JOB / 'status.json')
        expected = args.task or ('--preflight-depth' if args.preflight_depth else '--preflight')
        require(state.get('status') == 'running' and state.get('current') == expected,
                'Child was not admitted by the serial CPU owner')
        completed = {s['task'] for s in state['steps'] if s['returncode'] == 0}
        if args.task:
            require('preflight' in completed, 'Initial GPU preflight has not exited successfully')
            ids = [t['id'] for t in plan['adapted_tasks']]
            require(set(ids[:ids.index(args.task)]).issubset(completed), 'Prior original task has not exited successfully')
            if args.task in (ORIGINAL_JOB + '-s1', ORIGINAL_JOB + '-s2'):
                require('preflight-depth' in completed, 'DA class/weights preflight has not exited successfully')
        if args.task:
            print(json.dumps(execute_task(plan, args.task)), flush=True)
        else:
            receipt = gpu_preflight(plan, depth=args.preflight_depth)
            name = 'depth_preflight.json' if args.preflight_depth else 'gpu_preflight.json'
            write_new(ROOT / plan['output_root'] / 'jobs' / JOB / name, receipt)
            print(json.dumps(receipt), flush=True)
        return
    run(plan)


if __name__ == '__main__':
    main()
