#!/usr/bin/env python3
"""Fixed-service PRO6000 host adapter for the eight unchanged original IE lanes.

Default validation is CPU-only. Preflight creates no scientific cell. Formal
execution imports the original lane runner and changes only its LOCAL_GPU value.
The original runner retains source/dependency/disk/serial/render verification.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import stat
import sys
import time

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[5]
BASE = ROOT / 'experiments/dynamic_sr_modules_20261010/realdeg_20261109/baselines'
PLAN = HERE / 'prepared_plan.json'
GPU = 'GPU-ddc2c5d8-3507-6293-837c-b1ea6b5e1cc5'
PYTHON = '/home/cai_tianshun/miniconda3/envs/mml_bw/bin/python'
IDS = ('rd-ie-cook','rd-ie-cut','rd-ie-flame','rd-ie-coffee','rd-ie-sear','rd-ie-salmon','rd-ie-disc','rd-ie-vr')
SERVICE_BASE = Path('/var/lib/claude-training/jobs')


def require(value, message):
    if not value:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def check_ref(reference, root=ROOT):
    p = Path(reference['path'])
    require(not p.is_absolute() and '..' not in p.parts and p.as_posix() == reference['path'], 'Unsafe source path')
    p = root / p
    require(p.resolve().is_relative_to(root.resolve()) and p.is_file(), 'Missing or foreign source: ' + str(p))
    require(sha(p) == reference['sha256'], 'Bound SHA changed: ' + str(p))
    return p


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_plan(plan, root=ROOT):
    require(plan.get('schema') == 'rd4_pro6000_original_IE_plan_v2'
            and plan.get('project_id') == '4dsr' and plan.get('gpu_uuid') == GPU
            and plan.get('python') == PYTHON and plan.get('env_id') == 'mml_bw'
            and tuple(j['original_job_id'] for j in plan['jobs']) == IDS,
            'Only the frozen eight IE cells and authorized environment/GPU are allowed')
    for reference in plan['source_bindings']:
        check_ref(reference, root)
    for reference in plan['recovery'].values():
        check_ref(reference, root)
    source_lock = read(check_ref(plan['source_lock'], root))
    artifacts = {r['path']:r['sha256'] for r in read(check_ref(plan['artifact_lock'], root))['files']}
    for reference in source_lock['files']:
        check_ref(reference, root)
    require(len({j['service_job_id'] for j in plan['jobs']}) == 8, 'Duplicate service job identity')
    for row in plan['jobs']:
        for key in ('original_request','lane','config'):
            reference = row[key]
            check_ref(reference, root)
            require(artifacts.get(reference['path']) == reference['sha256'], 'Original artifact lock differs')
        original = read(root / row['original_request']['path'])
        lane = read(root / row['lane']['path'])
        require(original == row['original_request_value'] and lane['job_id'] == row['original_job_id']
                and lane['method'] == 'iesrgs' and lane['scene'] == row['scene']
                and lane['host'] == 'local' and lane['no_parallel_training'] is True,
                'Original IE lane/request identity changed')
        expected_id = row['original_job_id'].replace('rd-ie-', 'rd4-p6-ie-') + ('-02' if row['original_job_id'] == 'rd-ie-cook' else '-01')
        require(row['service_job_id'] == expected_id,
                'Unregistered PRO service identity')
        expected = dict(original, job_id=row['service_job_id'], gpu_uuids=[GPU],
                        argv=[PYTHON,'-u',str(root / plan['runner_path']),'--lane',row['lane']['path'],'--execute'])
        require(row['service_manifest'] == expected, 'Only job_id, GPU and host-adapter argv may differ')
        require(original['project_id'] == '4dsr' and original['env_id'] == 'mml_bw'
                and original['argv'][0] == PYTHON and original['cwd'] == str(root), 'Changed original environment/project')
        for task in lane['tasks']:
            require(task.get('python') == PYTHON, 'Foreign child interpreter')
            if task.get('source'):
                check_ref(task['source'], root)
        require(row['output_root'] == 'output/dynamic_sr_modules_20261010/realdeg_20261109/baselines/iesrgs/' + row['scene'],
                'Changed scientific output scope')
    check_ref(plan['rade_receipt'], root)
    check_ref(plan['rade_binary'], root)
    check_ref(plan['resource_gate_reassignment'], root)


def proc_identity(pid, proc=Path('/proc')):
    p = proc / str(pid)
    fields = (p / 'stat').read_text().rsplit(')',1)[1].split()
    return dict(pid=pid, ppid=int(fields[1]), state=fields[0], start_ticks=int(fields[19]),
                cgroup=(p / 'cgroup').read_text().strip().split('::',1)[1],
                netns=os.readlink(p / 'ns/net'))


def execution_context(plan, expected, *, service_base=SERVICE_BASE, proc=Path('/proc')):
    require(str(Path(sys.executable).absolute()) == PYTHON, 'Use the registered absolute method interpreter')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == GPU, 'Exactly the authorized PRO GPU UUID must be visible')
    directory = service_base / expected['job_id']
    frozen = directory / 'manifest.json'
    info = frozen.lstat()
    require(stat.S_ISREG(info.st_mode) and info.st_uid == 0 and info.st_nlink == 1 and not info.st_mode & 0o022,
            'The fixed-service manifest must be a safe root-frozen regular file')
    manifest = read(frozen)
    require(all(manifest.get(k) == v for k,v in expected.items()), 'Fixed manifest differs from prepared scientific/host adaptation fields')
    submission = read(directory / 'submission.json')
    require(submission.get('manifest_sha256') == sha(frozen), 'Submission must bind normalized root manifest bytes')
    status = read(directory / 'run/status.json')
    require(status.get('job_id') == expected['job_id'] and status.get('project_id') == '4dsr'
            and status.get('gpu_uuids') == [GPU] and status.get('state') == 'running'
            and status.get('command_started') is True, 'No matching running fixed-service admission')
    runner = status['runner']
    actual = proc_identity(runner['pid'], proc)
    cg = '/claudevpn.slice/claude-local-' + expected['job_id'] + '.service'
    require(actual['state'] not in ('Z','X') and runner['start_ticks'] == actual['start_ticks']
            and runner['cgroup'].removeprefix('0::') == actual['cgroup'] == cg
            and runner['netns'] == actual['netns'], 'Stale or foreign fixed runner identity')
    current = proc_identity(os.getpid(), proc)
    require(current['cgroup'] == actual['cgroup'] and current['netns'] == actual['netns'], 'Foreign execution cgroup/netns')
    child = status.get('child', {})
    require(child.get('pid') == os.getpid() and child.get('start_ticks') == current['start_ticks']
            and child.get('cgroup','').removeprefix('0::') == cg and child.get('netns') == current['netns'],
            'This process is not the admitted scientific command child')
    seen = set()
    pid = os.getpid()
    while pid > 1 and pid not in seen:
        seen.add(pid)
        if pid == runner['pid']:
            return dict(job_id=expected['job_id'], root_manifest_sha256=sha(frozen), runner=runner, child=child)
        pid = proc_identity(pid, proc)['ppid']
    raise RuntimeError('Fixed-service runner is not an ancestor')


def authorize(plan, row, root=ROOT):
    delegation_path = root / plan['authorization']['delegation_path']
    activation_path = root / plan['authorization']['activation_path']
    delegation, activation = read(delegation_path), read(activation_path)
    require(activation.get('schema') == 'rd4_pro6000_split_activation_v2'
            and activation.get('status') == 'active'
            and activation.get('old_v6_waiting_controller_stopped') is True
            and activation.get('original_jobs_empty') is True and activation.get('local_mode_excludes_ie') is True
            and activation.get('pro_mode_owns_original_ids') == list(IDS)
            and activation.get('delegation_sha256') == sha(delegation_path), 'No active duplicate-preventing CPU ownership handoff')
    # The final root partition binds new manifests and this plan/runner without a hash cycle.
    partition = root / plan['authorization']['partition_path']
    require(activation.get('partition_sha256') == sha(partition), 'Changed root ownership partition')
    resource = plan['resource_gate_reassignment']
    resource_path = check_ref(resource,root)
    resource_ref = dict(path=str(resource_path),sha256=resource['sha256'])
    require(all(value.get('resource_gate_reassignment') == resource_ref
                for value in (delegation,activation,read(partition))),
            'Resource-only predecessor reassignment must be bound by all ownership records')
    require(delegation.get('schema') == 'rd4_pro6000_split_owner_delegation_v1'
            and delegation.get('runner',{}).get('sha256') == sha(HERE / 'runner.py')
            and delegation.get('runner',{}).get('path') == plan['runner_path']
            and delegation.get('prepared_plan',{}).get('sha256') == sha(PLAN)
            and delegation.get('prepared_plan',{}).get('path') == PLAN.relative_to(ROOT).as_posix()
            and activation.get('original_plan_sha256') == plan['original_owner_plan']['sha256']
            and activation.get('v6_controller_sha256') == plan['predecessor_controller']['sha256']
            and [j['original_job_id'] for j in delegation['jobs']] == list(IDS),
            'Delegation does not bind final runner/plan or original CPU ownership')
    matches = [j for j in delegation['jobs'] if j['original_job_id'] == row['original_job_id']]
    require(len(matches) == 1, 'Missing or duplicate original IE delegation')
    selected = matches[0]
    require(selected['service_job_id'] == row['service_job_id']
            and selected['original_request']['sha256'] == row['original_request']['sha256']
            and selected['original_lane']['sha256'] == row['lane']['sha256']
            and selected['config']['sha256'] == row['config']['sha256'], 'Delegated source or service identity differs')
    manifest_ref = selected['service_manifest']
    manifest_path = Path(manifest_ref['path'])
    require(manifest_path.is_absolute() and manifest_path.resolve().is_relative_to(root.resolve())
            and sha(manifest_path) == manifest_ref['sha256']
            and read(manifest_path) == row['service_manifest'], 'Delegated service manifest differs')
    # The GPU unit can see its own /var/lib job only. The CPU owner separately
    # verifies the actual root files and records byte-identical project mirrors.
    actual_proof = activation['preflight']
    proof = activation['gpu_visible_preflight']
    verified = {}
    for key in ('manifest','actual_status','root_submission','root_manifest','native_report'):
        reference = proof[key]
        p = Path(reference['path'])
        require(reference['sha256'] == actual_proof[key]['sha256']
                and p.is_absolute() and p.resolve().is_relative_to(root.resolve())
                and p == root / plan['authorization']['preflight_mirror_directory'] / (key + '.json')
                and stat.S_ISREG(p.lstat().st_mode)
                and sha(p) == reference['sha256'], 'Changed GPU-visible preflight mirror or root binding: ' + key)
        verified[key] = read(p)
    expected = plan['preflight_manifest']
    require(verified['manifest'] == expected
            and all(verified['root_manifest'].get(k) == v for k,v in expected.items())
            and verified['root_submission'].get('manifest_sha256') == proof['root_manifest']['sha256']
            and verified['actual_status'].get('job_id') == expected['job_id']
            and verified['actual_status'].get('state') == 'ended'
            and verified['actual_status'].get('reason') == 'completed'
            and type(verified['actual_status'].get('exit_code')) is int
            and verified['actual_status']['exit_code'] == 0,
            'Real native preflight successful fixed-service Exit0 required')
    report = verified['native_report']
    require(report.get('status') == 'passed' and report.get('optimizer_steps') == 0
            and report.get('runner_sha256') == sha(HERE / 'runner.py')
            and report.get('prepared_plan_sha256') == sha(PLAN)
            and report.get('fixed_context',{}).get('root_manifest_sha256') == proof['root_manifest']['sha256'],
            'Native preflight did not execute the final runner/plan/root manifest bytes')
    return dict(delegation_sha256=sha(delegation_path), activation_sha256=sha(activation_path))


def gpu_preflight(plan):
    """Native API acceptance; no optimizer, real image or scientific output."""
    sys.path.insert(0,str(BASE))
    import runtime_rd
    driver = runtime_rd.load('_pro_IE_capability_driver', plan['preflight_driver']['path'])
    runtime_rd.setup_static_adapter(driver)
    mods = driver.setup(str(ROOT))
    torch = mods['torch']
    require(torch.cuda.device_count() == 1, 'Exactly one CUDA device required')
    props = torch.cuda.get_device_properties(0)
    actual_uuid = str(props.uuid)
    if not actual_uuid.startswith('GPU-'):
        actual_uuid = 'GPU-' + actual_uuid
    require(actual_uuid == GPU and torch.cuda.get_device_capability(0) == (12,0), 'Expected PRO6000 sm120 hardware')
    sys.path.insert(0,str(ROOT / 'experiments/dynamic_sr_cull_retrain_20261009'))
    import rt
    wu = rt.check_runtime()
    from simple_knn._C import distCUDA2
    points = torch.tensor([[0.,0.,2.],[.1,0.,2.],[0.,.1,2.],[0.,0.,2.1]],device='cuda')
    distance = distCUDA2(points)
    require(bool(torch.isfinite(distance).all()) and bool((distance > 0).all()), 'Native Wu KNN failed')
    from types import SimpleNamespace
    from utils.graphics_utils import getProjectionMatrix
    from methods.iesrgs_renderer import raster_api,render_state
    api = raster_api()
    projection = getProjectionMatrix(znear=.01,zfar=10.,fovX=math.pi/3,fovY=math.pi/3).transpose(0,1).cuda()
    camera = SimpleNamespace(image_width=64,image_height=64,FoVx=math.pi/3,FoVy=math.pi/3,
        world_view_transform=torch.eye(4,device='cuda'),full_proj_transform=projection,
        camera_center=torch.zeros(3,device='cuda'),time=0.)
    xyz = torch.tensor([[0.,0.,2.]],device='cuda',requires_grad=True)
    state = dict(xyz=xyz,scales=torch.full((1,3),.25,device='cuda',requires_grad=True),
        opacity=torch.full((1,1),.95,device='cuda',requires_grad=True),
        rotation=torch.tensor([[1.,0.,0.,0.]],device='cuda',requires_grad=True),
        sh=torch.full((1,1,3),.1,device='cuda',requires_grad=True))
    packet = render_state(camera,state,torch.zeros(3,device='cuda'),0,[camera],require_depth=True,api=api)
    require(all(bool(torch.isfinite(packet[k]).all()) for k in ('render','median_depth','expected_depth','mask')), 'Nonfinite RADE output')
    depth_gradient, = torch.autograd.grad(packet['median_depth'].sum(),xyz,retain_graph=True)
    rgb_gradient, = torch.autograd.grad(packet['render'].sum(),state['sh'])
    require(all(bool(torch.isfinite(g).all()) and float(g.abs().sum()) > 0 for g in (depth_gradient,rgb_gradient)), 'Detached/zero/nonfinite RADE backward')
    weights = check_ref(plan['depth_weights'])
    sys.path.insert(0,str(ROOT / plan['depth_package']))
    from depth_anything_v2.dpt import DepthAnythingV2
    model = DepthAnythingV2(encoder='vitl',features=256,out_channels=[256,512,1024,1024])
    state_dict = torch.load(weights,map_location='cpu',weights_only=True,mmap=True)
    model.load_state_dict(state_dict,strict=True)
    model.eval().cuda()
    with torch.inference_mode():
        depth = model(torch.zeros(1,3,28,28,device='cuda'))
    require(tuple(depth.shape) == (1,28,28) and bool(torch.isfinite(depth).all()), 'DA-V2-L native small-shape inference failed')
    torch.cuda.synchronize()
    return dict(status='passed',gpu_uuid=actual_uuid,model=props.name,capability=list(torch.cuda.get_device_capability(0)),
        torch=torch.__version__,cuda=torch.version.cuda,wu_runtime=wu,native_KNN=True,
        RADE_median_depth_backward=float(depth_gradient.abs().sum()),RADE_RGB_backward=float(rgb_gradient.abs().sum()),
        DA_weights=plan['depth_weights'],DA_shape=list(depth.shape),scientific_cell_created=False,optimizer_steps=0)


def execute_lane(plan,row):
    require(not (ROOT / row['output_root']).exists(), 'Scientific cell exists; never adopt or overwrite partial/completed output')
    sys.path.insert(0,str(BASE))
    original = load('_original_IE_lane_PRO_host_adapter',BASE / 'run_lane.py')
    require(original.LOCAL_GPU == plan['original_gpu_uuid'], 'Unexpected original host binding')
    original.LOCAL_GPU = GPU
    sys.argv = [str(BASE / 'run_lane.py'),'--lane',row['lane']['path'],'--execute']
    original.main()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight',action='store_true')
    parser.add_argument('--lane')
    parser.add_argument('--execute',action='store_true')
    args = parser.parse_args()
    require(not (args.preflight and args.lane), 'Choose preflight or one original lane')
    plan = read(PLAN)
    validate_plan(plan)
    if args.preflight:
        context = execution_context(plan,plan['preflight_manifest'])
        report = gpu_preflight(plan)
        report.update(runner_sha256=sha(HERE / 'runner.py'),prepared_plan_sha256=sha(PLAN),fixed_context=context)
        print(json.dumps(report,ensure_ascii=False),flush=True)
    elif args.execute:
        rows = [r for r in plan['jobs'] if r['lane']['path'] == args.lane]
        require(len(rows) == 1,'Only an exact frozen original IE lane may execute')
        execution_context(plan,rows[0]['service_manifest'])
        authorize(plan,rows[0])
        execute_lane(plan,rows[0])
    else:
        print(json.dumps(dict(status='validated_CPU_only',local_jobs=8,GPU_calls=0,
                             preflight_manifest=plan['preflight_manifest']),ensure_ascii=False))


if __name__ == '__main__':
    main()
