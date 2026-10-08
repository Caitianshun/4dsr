"""Source-aware native full-SR manager and exit-triggered local evaluator.

This additive operator leaves the hash-bound scientific and historical runtime
files untouched. Root supplies one fresh Discussion07 B0 or Bsync spec, binds
the actual two service identities, and publishes an immutable release file.
CPU validation never queries GPUs or imports tensors. There are no service
creation, global-index updates, scientific retries, or held-scene operations.
"""
from __future__ import annotations
import argparse, copy, fcntl, hashlib, importlib.metadata, importlib.util, json, os, re, shutil, socket
import subprocess, sys, time, traceback
from pathlib import Path

SCHEMA = 'registered_native_full_source_aware_operator_v2'
OUT = 'output/dynamic_sr_multiview_footprint_20261007'
NATIVE_SHA = 'c278928f56b0515adedf15eef9a30850e8bd2a4c41e6f6e0f828068ca774b75d'
EVALUATOR_SHA = 'ffdfd66ea15e1bac4c5f5e285568d087baac912a06e6d4d7f15845cd37e30199'
CORE_SHA = '590f38a55da141de8a9617c66484d7cad1dbc36ba5ae5411bb11030972f73d2d'
TRANSPORT_SHA = '7704aa2142456d890eaf32104f0b62f1647b606f75c28c76b03a0bdc2305b580'
PREFIX_SHA = '92a19ed94c2bf7dab881341375c4bdc486e6f4032536e977592c7911c3ce6605'
TEMPORAL_SHA = 'fccf82a5a8d04694f6327bfb791117ded0f345c10c66c623ceaf4503994322c8'
DISPLAY_SHA = 'e256267d9d07397325bc757b20fecc4f5b35d772b78b79bf7d8e1b5599bf0925'
HELPER_SHA = '7f2a46ecf46f32c62b8fa76c32ea4ad68644594bfc07cb21bbab61ccf47bd44b'
COMPLETION_SHA = 'e7b55bf9b504c67e4df7480be078e07b761dcca2f9b7d94fd1da2572d1fdcfeb'
GPU0 = 'GPU-5c08f287-3ffd-edf9-91ed-cd6db2690f3b'


def require(value, message):
    if not value: raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''): h.update(block)
    return h.hexdigest()


def bare(reference):
    return {k: reference[k] for k in ('path', 'sha256')}


def load(name, path):
    sp = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(sp)
    sys.modules[name] = module; sp.loader.exec_module(module)
    return module


def services(s, role):
    return s['manager_unit'] if role == 'manager' else s['callback_unit']


def task_key(s):
    return s['scene'] + '/' + str(s['seed']) + '/' + s['method']


def validate_no_repeat(s, index):
    rows = [r for r in index['entries'] if r['task_key'] == task_key(s)]
    require(len(rows) == 1 and rows[0]['status'] == 'planned_only'
            and rows[0]['actual_units_started'] is False,
            'Completed, dispatched, missing, or duplicate endpoint refused')
    require(s['no_automatic_scientific_retry'] is True, 'No implicit recovery')


def validate_acceptance(s, plan, value, checks):
    require(value['status'] == 'passed_native_full_refinement_CUDA_acceptance'
            and value['source_sha256'] == NATIVE_SHA
            and value['author_commit'] == plan['author_commit']
            and value['Adam_calls'] == value['formal_updates'] == 0
            and value['observer_RNG_restored'] is True,
            'Exact v2 zero-update native CUDA acceptance required')
    grid = plan['data']['resolution_LR']
    require(grid in value['accepted_grids'], 'This actual domain grid unaccepted')
    cases = [v for v in value['results'] if v['LR'] == grid
             and v['scene'] == v['parent_scene'] == s['scene']
             and v['seed'] == v['parent_seed'] == s['seed']]
    require(len(cases) == 1 and cases[0]['execution'] == 'actual_native_CUDA'
            and cases[0]['synthetic'] is False
            and cases[0]['parent'] == plan['parent']
            and set(cases[0]['checks']) == set(checks)
            and all(cases[0]['checks'].values()),
            'Every real same-parent native CUDA check must pass')


def references(value):
    if isinstance(value, dict):
        if 'path' in value and 'sha256' in value: yield bare(value)
        for v in value.values(): yield from references(v)
    elif isinstance(value, list):
        for v in value: yield from references(v)


def metric_external(platform):
    lp=platform['metric_runtime']['LPIPS']
    return [bare(lp[k]) for k in ('learned_weights','AlexNet_pretrained_weights')] + [
        dict(path=p,sha256=h) for p,h in lp['source_files'].items()]


def validate_platform(s,e):
    platform=e.read(s['assets']['uniform_evaluation_platform']);ev=s['evaluation']
    require(platform['status']=='registered_same_platform_native_full_uniform_evaluation'
            and (platform['host'],platform['GPU'],platform['GPU_model'])==(ev['host'],ev['GPU'],ev['GPU_model'])
            and (platform['runtime_python'],platform['upstream'])==(ev['runtime_python'],ev['upstream'])
            and platform['torch']=='2.7.1+cu128' and platform['cuda']=='12.8'
            and ev['transport']=='local' and ev['workspace']=='/home/cai_tianshun/Project/4dsr'
            and ev['wait_for_shared_physical_lock'] is True,
            'Exact previously registered uniform local evaluation platform required')
    require(type(ev['cpu_threads']) is int and ev['cpu_threads']>0
            and all(type(ev[k]) is int and 0<=ev[k]<=limit for k,limit in
                    (('idle_max_used_MiB',1024),('idle_max_utilization_percent',10))),
            'Root must declare bounded actual evaluation resource thresholds')
    for ref in platform['runtime_asset_refs']:e.bound(ref)
    return platform


def required_role_files(s, e, plan, role):
    """Enumerate the whole legal role inventory from registered inputs."""
    refs=list(s['sources'].values())+[s['production_core_source'],s['CPU_temporal']['source']]
    refs += [v for v in s['assets'].values() if isinstance(v,dict) and 'path' in v]
    refs += [dict(path=p,sha256=h) for p,h in plan['source_files']['project'].items()]
    # Both native training and evaluation reconstruct the complete registered
    # data/initializer identity and require every legal teacher file to exist.
    # The callback decodes only its registered diagnostic subset.
    refs += [v for rows in plan['training_files'].values() for v in rows]
    refs += list(references(plan['data']))
    init=e.read(plan['data']['initialization_receipt']);refs += list(references(init))
    teacher=e.read(s['assets']['teacher'])
    refs += [teacher[k] for k in ('producer_config','producer_summary','plan')]
    parent_plan=e.read(s['assets']['parent_plan'])
    refs += list(references(parent_plan['author_sources']))
    readiness=e.read(plan['data']['domain_protocol']['readiness'])
    refs += list(references(readiness['original_Wu']))
    refs += e.read(s['assets']['uniform_evaluation_platform'])['runtime_asset_refs']
    if role == 'callback':
        protocol=e.read(s['assets']['evaluation_protocol']);manifest=e.read(s['assets']['manifest'])
        refs += [dict(path=p,sha256=h) for p,h in protocol['sources'].items()]
        refs += list(references(protocol))
        keys={tuple(k) for k in protocol['test_keys']+protocol['train_teacher_diagnostics']['keys']}
        base=e.path(s['assets']['manifest']['path']).parent
        for obs in manifest['observations']:
            if (obs['camera_id'],int(obs['frame_index'])) in keys:
                for grid in ('lr','hr'):
                    refs.append(dict(path=str((base/obs[grid+'_path']).relative_to(e.root)),sha256=obs[grid+'_sha256']))
    by={}
    for ref in refs:
        if ref['path'] in by: require(bare(by[ref['path']])==bare(ref),'One role path has conflicting SHA')
        by[ref['path']]=bare(ref)
    return list(by.values())


def validate_role_deployment(s,e,plan,role):
    dep=s['role_deployments'][role]
    host=s['host'] if role=='manager' else s['evaluation']['host']
    workspace=s['workspace'] if role=='manager' else s['evaluation']['workspace']
    require(dep['schema']=='actual_native_full_v2_role_CPU_all_SHA' and dep['role']==role
            and dep['host']==host and dep['workspace']==workspace and dep['full_inventory_verified'] is True
            and dep['GPU_calls']==dep['model_imports']==dep['image_decodes']==dep['formal_updates']==0,
            'Actual whole-role CPU-only deployment receipt required')
    expected={v['path']:v for v in required_role_files(s,e,plan,role)}
    actual={v['path']:v for v in dep['verified_files']}
    require(len(actual)==len(dep['verified_files']) and set(actual)==set(expected),
            'Partial or duplicated role inventory cannot authorize a worker')
    for p,v in actual.items():
        require(bare(v)==bare(expected[p]) and type(v['bytes']) is int and v['bytes']>=0,'Role byte/source identity differs')
        require(e.bound(v).stat().st_size==v['bytes'],'Actual closed role file bytes differ')
    require(dep['files']==len(actual) and dep['bytes']==sum(v['bytes'] for v in actual.values()),'Complete role counts differ')
    runtime=s['training'] if role=='manager' else s['evaluation']
    require(dep['python']==runtime['runtime_python'] and Path(dep['python']).resolve().is_file()
            and Path(sys.executable).resolve()==Path(dep['python']).resolve()
            and sha(Path(dep['python']).resolve())==dep['python_binary_sha256'],'Role actual interpreter changed')
    require(dep['versions']=={'torch':'2.7.1+cu128','numpy':'1.26.4'}
            and all(importlib.metadata.version(k)==v for k,v in dep['versions'].items()),'Actual native distribution recipe changed')
    require(type(dep['min_free_bytes']) is int and dep['min_free_bytes']>0
            and dep['observed_free_bytes']>=dep['min_free_bytes']
            and shutil.disk_usage(e.root).free>=dep['min_free_bytes'],'Declared actual role storage reserve unavailable')
    for row in dep['external_runtime_files']:
        p=Path(row['path']);require(p.is_absolute() and p.is_file() and sha(p)==row['sha256'],'Actual external native/metric runtime changed')
    required_external={str(Path(runtime['upstream'])/p):h for p,h in plan['source_files']['upstream'].items()}
    extensions=runtime['native_extensions']
    require(len(extensions)==2 and all(Path(v['path']).is_absolute() and Path(v['path']).name.startswith('_C')
            and Path(v['path']).suffix=='.so' for v in extensions)
            and {next((part for part in Path(v['path']).parts if part in ('diff_gaussian_rasterization','simple_knn')),None)
                 for v in extensions}=={'diff_gaussian_rasterization','simple_knn'},
            'Both actual native CUDA extension binary identities required')
    required_external.update({v['path']:v['sha256'] for v in extensions})
    if role=='callback':required_external.update({v['path']:v['sha256'] for v in metric_external(e.read(s['assets']['uniform_evaluation_platform']))})
    external={v['path']:v for v in dep['external_runtime_files']}
    require(set(required_external)<=set(external) and all(external[p]['sha256']==h for p,h in required_external.items()),
            'Complete pinned author runtime is required')


def close_role(s,e,role,output,min_free_bytes,external_files):
    """Create an actual CPU all-SHA receipt; this does not enable dispatch."""
    began=time.monotonic()
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','Role closure must be CPU-only')
    runtime=s['training'] if role=='manager' else s['evaluation']
    host=s['host'] if role=='manager' else s['evaluation']['host']
    workspace=s['workspace'] if role=='manager' else s['evaluation']['workspace']
    require(socket.gethostname()==host and str(e.root)==workspace,'Actual closure actor differs')
    require(type(min_free_bytes) is int and min_free_bytes>0,'Root must declare a storage reserve')
    free=shutil.disk_usage(e.root).free;require(free>=min_free_bytes,'Storage reserve unavailable')
    plan=e.read(s['assets']['plan']);rows=[]
    for ref in required_role_files(s,e,plan,role):
        p=e.bound(ref);rows.append(dict(bare(ref),bytes=p.stat().st_size))
    external=[]
    for ref in external_files:
        p=Path(ref['path']);require(p.is_absolute() and p.is_file() and sha(p)==ref['sha256'],'External runtime SHA differs')
        external.append(dict(path=str(p),sha256=ref['sha256'],bytes=p.stat().st_size))
    python=Path(runtime['runtime_python']);require(python.is_absolute() and python.resolve().is_file(),'Actual runtime Python required')
    value=dict(schema='actual_native_full_v2_role_CPU_all_SHA',role=role,host=host,workspace=workspace,
               full_inventory_verified=True,verified_files=rows,files=len(rows),bytes=sum(v['bytes'] for v in rows),
               python=str(python),python_binary_sha256=sha(python.resolve()),
               versions={k:importlib.metadata.version(k) for k in ('torch','numpy')},
               min_free_bytes=min_free_bytes,observed_free_bytes=free,external_runtime_files=external,
               GPU_calls=0,model_imports=0,image_decodes=0,formal_updates=0,wall_seconds=time.monotonic()-began,
               operator_source=e.entry(str(Path(__file__).resolve().relative_to(e.root))),dispatch_authorized=False)
    temporary=copy.deepcopy(s);temporary['role_deployments']={role:value}
    validate_role_deployment(temporary,e,plan,role)
    return e.new(output,value)


def validate(s, e, role, operator_source):
    require(s['schema'] == SCHEMA and s['status'] == 'root_registered_ready_operator_v2'
            and s['dispatch_enabled'] is True and s['synthetic_contract'] is False,
            'Root must explicitly enable the completed real registration')
    require((s['scene'], s['seed']) == ('meetroom_discussion', 20261007)
            and s['method'] in ('B0', 'Bsync') and s['role'] == 'already_used_development',
            'Only registered Discussion07 development B0/Bsync pair')
    require(s['GPU'] == GPU0 and s['GPU_model'] == 'NVIDIA A100-SXM4-40GB'
            and s['transport'] == 'a100-train' and s['transport_python'] == '/usr/bin/python3',
            'Exact registered A100 GPU0 training transport required')
    expected_root = s['workspace'] if role == 'manager' else s['evaluation']['workspace']
    expected_host = s['host'] if role == 'manager' else s['evaluation']['host']
    require(str(e.root) == expected_root and socket.gethostname() == expected_host,
            'Actual role workspace/host differs')
    require(bare(s['assets']['runtime_adapter_source']) == bare(operator_source),
            'Actual v2 operator source differs')
    refs = [(s['production_core_source'], CORE_SHA),
            (s['assets']['transport_runtime_source'], TRANSPORT_SHA),
            (s['sources']['full_refine_registered.py'], NATIVE_SHA),
            (s['sources']['full_evaluate_registered.py'], EVALUATOR_SHA),
            (s['sources']['full_prefix_registered.py'], PREFIX_SHA),
            (s['sources']['completion.py'], COMPLETION_SHA),
            (s['CPU_temporal']['source'], TEMPORAL_SHA),
            (s['assets']['display_guard_source'], DISPLAY_SHA),
            (s['assets']['remote_helper_source'], HELPER_SHA)]
    for reference, pin in refs:
        require(reference['sha256'] == pin, 'Source revision differs'); e.bound(reference)
    for name in ('plan', 'manifest', 'parent_complete', 'parent_plan', 'parent_sidecar',
                 'selection', 'schedule', 'teacher', 'native_acceptance',
                 'evaluation_protocol', 'uniform_evaluation_platform',
                 'root_endpoint_index', 'remote_helper_source', 'display_guard_source'):
        e.bound(s['assets'][name])
    for reference in s['sources'].values(): e.bound(reference)
    plan = e.read(s['assets']['plan'])
    require(plan['status'] == 'registered_ready_full_native_SR_refinement'
            and plan['missing'] == [] and plan['representation'] == 'native_author_GaussianModel_no_children'
            and (plan['scene'], plan['seed'], plan['method']) == (s['scene'], s['seed'], s['method'])
            and plan['source_files']['project'][s['sources']['full_refine_registered.py']['path']] == NATIVE_SHA,
            'Exact complete v2 scientific plan required')
    require(plan['full_native_context'] == dict(manifest=s['assets']['manifest'],
            parent=s['assets']['parent_checkpoint'], schedule=s['assets']['schedule']),
            'Original immutable parent/manifest/schedule context differs')
    require(plan['schedule']['steps'] == plan['schedule']['parameter_updates'] == 6000
            and plan['schedule']['Adam_calls_per_update'] == 1
            and plan['schedule']['optimizer_resets'] == 0,
            'Formal6000/oneAdam/no-reset budget differs')
    for name in ('manifest',):
        require(bare(plan['data'][name]) == bare(s['assets'][name]), 'Plan input differs')
    for name in ('checkpoint', 'complete', 'plan', 'sidecar'):
        require(bare(plan['parent'][name]) == bare(s['assets']['parent_' + name]), 'Parent identity differs')
    for name in ('selection', 'schedule', 'teacher'):
        require(bare(plan['dependencies'][name]) == bare(s['assets'][name]), 'Dependency differs')
    selection = e.read(s['assets']['selection'])
    require(selection['baseline_method'] == 'B0' and selection['selected_candidates'] == ['Bsync']
            and selection['uses_confirmation_for_selection'] is False, 'Frozen selected pair differs')
    validate_no_repeat(s, e.read(s['assets']['root_endpoint_index']))
    core = load('_full_operator_v2_integrity_core', e.bound(s['production_core_source']))
    validate_acceptance(s, plan, e.read(s['assets']['native_acceptance']), core.CUDA_CHECKS)
    protocol = e.read(s['assets']['evaluation_protocol'])
    require(protocol['schema'] == core.EVAL and protocol['status'] == 'registered_full_native_evaluation_before_prediction_reads'
            and protocol['scene'] == s['scene'] and protocol['seed'] == s['seed']
            and protocol['test_keys'] == [['cam00', f] for f in range(300)]
            and bare(protocol['manifest']) == bare(s['assets']['manifest'])
            and protocol['sources'][s['sources']['full_evaluate_registered.py']['path']] == EVALUATOR_SHA
            and protocol['sources'][s['sources']['full_refine_registered.py']['path']] == NATIVE_SHA,
            'Explicit v2 full300 evaluation protocol required')
    validate_platform(s,e)
    guard = load('_full_operator_v2_exact_display_guard', e.bound(s['assets']['display_guard_source']))
    require(s['display_exemption'] == guard.GNOME and guard.GPU == s['GPU'],
            'Original audited exact-display resource guard required')
    require(s['physical_lock'] == OUT + '/locks/' + s['GPU'] + '.controller.lock'
            and s['evaluation']['physical_lock'] == OUT + '/locks/' + s['evaluation']['GPU'] + '.controller.lock',
            'Shared physical controller locks differ')
    require(s['training']['physical_controller_lock']=='/home/ubuntu/3DGS/4dsr/'+s['physical_lock'],
            'Isolated training workspace must use the base-project shared physical controller lock')
    require(all(re.fullmatch(r'4dsr-footprint-[A-Za-z0-9-]+\.service', services(s, r)) for r in ('manager', 'callback')),
            'Root must declare both unique services')
    require(s['paths']['train_out'].startswith(OUT + '/full_SR_refinement/'), 'Native output namespace differs')
    paths = [e.path(s['paths'][k]) for k in ('train_out', 'training_manager_out', 'evaluation_out', 'callback_out', 'returned_out')]
    t=s['CPU_temporal']
    require(t['required'] is True and t['runtime_python']==s['evaluation']['runtime_python']
            and t['out'].startswith(OUT+'/full_temporal_diagnostics/'),
            'Same CPU runtime and separate native full temporal output required')
    paths.append(e.path(t['out']))
    require(not any(a == b or a.is_relative_to(b) or b.is_relative_to(a)
                    for i, a in enumerate(paths) for b in paths[i + 1:]), 'Separate fresh task output paths required')
    validate_role_deployment(s,e,plan,role)
    return plan, core, guard


def transport(s, e):
    return load('_full_operator_v2_transport', e.bound(s['assets']['transport_runtime_source']))


def role_command(s,e,role,spec_ref):
    workspace=s['workspace'] if role=='manager' else s['evaluation']['workspace']
    runtime=s['training' if role=='manager' else 'evaluation']['runtime_python']
    return [runtime,'-u',str(Path(workspace)/s['assets']['runtime_adapter_source']['path']),
            '--mode',role,'--root',workspace,'--spec',str(Path(workspace)/spec_ref['path']),
            '--spec-sha',spec_ref['sha256'],'--release',s['release_path'][role]]


def register_owner(s, e, role, spec_ref, argv, legacy):
    unit = services(s, role); inv = os.environ.get('INVOCATION_ID', '')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and re.fullmatch('[a-f0-9]{32}', inv),
            'CPU-only owned persistent service invocation required')
    cg = Path('/proc/self/cgroup').read_text().strip()
    require(cg.startswith('0::/user.slice/') and cg.endswith('/' + unit), 'Actual service cgroup differs')
    props = subprocess.run(['systemctl', '--user', 'show', unit, '-p', 'ExecMainStartTimestampMonotonic',
                            '-p', 'ExecMainPID', '-p', 'InvocationID'], capture_output=True, text=True, check=True)
    values = dict(line.split('=', 1) for line in props.stdout.splitlines() if '=' in line)
    require(values['ExecMainPID'] == str(os.getpid()) and values['InvocationID'] == inv,
            'Actual systemd owner differs')
    require(argv == role_command(s,e,role,spec_ref),
            'Actual service command must match the registered root command')
    r = dict(unit=unit, invocation_id=inv, pid=os.getpid(), start_ticks=legacy.start_ticks(os.getpid()),
             start_monotonic=values['ExecMainStartTimestampMonotonic'], host=socket.gethostname(),
             workspace=str(e.root), boot_id=legacy.boot(), owned_cgroup=cg[3:],
             command=argv, source=s['assets']['runtime_adapter_source'], spec=spec_ref,
             empty_cgroup_helper_source=s['assets']['remote_helper_source'])
    directory=s['paths']['training_manager_out'] if role == 'manager' else s['paths']['callback_out']
    return r, e.new(directory + '/actual_registration.json', r)


def resolve_release(s, e, spec_ref, owner, role, path, legacy):
    bound_path = legacy.wait_immutable_binding(path)
    require(bound_path.stat().st_mode & 0o222 == 0, 'Root must freeze release before atomic publication')
    release = json.loads(bound_path.read_text())
    require(release['schema'] == 'root_released_source_aware_native_full_v2'
            and bare(release['spec']) == bare(spec_ref) and release['task_key'] == task_key(s)
            and release['root_only_registration'] is True and release['no_scientific_asset_changes'] is True,
            'Explicit root release of this exact immutable spec required')
    references=release['registrations']; actual=e.read(references[role])
    require(actual == owner, 'Root release binds a different actual manager')
    for r in ('manager','callback'):
        registered=e.read(references[r]);workspace=s['workspace'] if r=='manager' else s['evaluation']['workspace']
        host=s['host'] if r=='manager' else s['evaluation']['host']
        require(registered['unit']==services(s,r) and bare(registered['source'])==bare(s['assets']['runtime_adapter_source'])
                and bare(registered['spec'])==bare(spec_ref) and registered['command']==role_command(s,e,r,spec_ref)
                and registered['host']==host and registered['workspace']==workspace
                and re.fullmatch('[a-f0-9]{32}',registered['invocation_id'])
                and int(registered['pid'])>0 and int(registered['start_ticks'])>0
                and int(registered['start_monotonic'])>0 and registered['owned_cgroup'].endswith('/'+services(s,r)),
                'Both actual service source/spec/argv/host/process identities required')
    result=copy.deepcopy(s)
    result['assets']['training_registration']=references['manager']
    result['assets']['callback_registration']=references['callback']
    result['assets']['operator_spec']=spec_ref
    result['assets']['root_release']=e.entry(str(bound_path.resolve().relative_to(e.root)))
    return result


def science_argv(s, e, stop, resume=None):
    a=s['assets']; absolute=lambda r: str(e.bound(a[r]))
    argv=[s['training']['runtime_python'], '-u', str(e.bound(s['sources']['full_refine_registered.py'])),
          '--mode', 'train', '--method', s['method'], '--manifest', absolute('manifest'),
          '--seed', str(s['seed']), '--parent', absolute('parent_checkpoint'),
          '--parent-complete', absolute('parent_complete'), '--teacher', absolute('teacher'),
          '--schedule', absolute('schedule'), '--selection', absolute('selection'),
          '--out', str(e.path(s['paths']['train_out'])), '--upstream', s['training']['upstream'],
          '--gpu-uuid', s['GPU'], '--native-acceptance', absolute('native_acceptance'),
          '--steps', '6000', '--topology-clock', 'refinement', '--stop', str(stop),
          '--checkpoint-interval', '100', '--cpu-threads', str(s['training']['cpu_threads'])]
    if resume: argv += ['--resume', str(e.path(resume))]
    return argv


def train_manager(s, e, plan, guard, legacy, spec_ref, release_path):
    out=s['paths']['training_manager_out']
    require(not e.path(out + '/intent.json').exists() and not e.path(s['paths']['train_out']).exists(),
            'Existing scientific intent/output protected; no automatic retry')
    e.new(out + '/intent.json', dict(task_key=task_key(s), spec=spec_ref, plan=s['assets']['plan'],
                                   no_automatic_scientific_retry=True))
    physical=Path(s['training']['physical_controller_lock']); physical.parent.mkdir(parents=True, exist_ok=True)
    require(not any(p.is_symlink() for p in [physical,*physical.parents]),'Shared physical lock cannot follow a symlink')
    backend=guard.Backend(); own=backend.process(os.getpid()); phases=[]; began=time.monotonic()
    with physical.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for start, stop in ((0, 100), (100, 6000)):
            first=guard.resource(backend.sample(), backend.process, own, idle=True)
            e.new(out + '/resource_first_' + str(stop) + '.json', first); backend.pause(30)
            second=guard.resource(backend.sample(), backend.process, own, idle=True)
            pre=guard.resource(backend.sample(), backend.process, own, idle=True)
            require(second['at_monotonic'] - first['at_monotonic'] >= 30, 'Separated resource reads required')
            resource=e.new(out + '/resource_' + str(stop) + '.json', dict(first=first, second=second, prelaunch=pre))
            command=[s['training']['runtime_python'], '-u', str(e.bound(s['assets']['runtime_adapter_source'])),
                     '--mode', 'science', '--root', str(e.root), '--spec', str(e.bound(spec_ref)),
                     '--spec-sha', spec_ref['sha256'], '--release', str(release_path), '--stop', str(stop)]
            if start: command += ['--resume', s['paths']['train_out'] + '/checkpoint_00100.pt']
            env=dict(os.environ, CUDA_VISIBLE_DEVICES=s['GPU'], FOURDSR_ROOT=str(e.root),
                     FOURDSR_UPSTREAM=s['training']['upstream'], OMP_NUM_THREADS=str(s['training']['cpu_threads']),
                     OPENBLAS_NUM_THREADS=str(s['training']['cpu_threads']))
            child=legacy.scientific_child(command, env, e.path(out + '/science_' + str(stop) + '.log'), e, s['assets']['runtime_adapter_source'])
            exit_ref=e.new(out + '/child_exit_' + str(stop) + '.json', child)
            require(child['exit_code'] == 0 and child['private_process_group_reaped'] is True,
                    'Scientific child failed/leaked; preserve and stop')
            cp=e.entry(s['paths']['train_out'] + '/checkpoint_' + str(stop).zfill(5) + '.pt')
            side=e.entry(cp['path'][:-3] + '.json'); metadata=e.read(side)['metadata']
            require(metadata['cursor'] == stop and metadata['plan_sha256'] == legacy.c.digest(plan)
                    and metadata['terminal_zero_grad'] is True and metadata['density_statistical_boundary_applied'] is True,
                    'Committed full native state invalid')
            counts=metadata['completed_operations']; span=stop-start
            require(all(counts[k] == v for k,v in dict(iterations=span, RGB=3*span, moments=0, Adam=span, backward=span).items()),
                    'Formal segment update/RGB/oneAdam budget differs')
            phases.append(e.new(out + '/phase_' + str(stop) + '.json', dict(start=start, stop=stop, delta_updates=span,
                          child=child, child_exit=exit_ref, checkpoint=cp, sidecar=side, resource=resource)))
            if start == 0: backend.pause(15)
    result=dict(status='completed_source_aware_native_full_v2_manager', task_key=task_key(s), spec=spec_ref,
                plan=s['assets']['plan'], phase_receipts=phases, accepted_delta_updates=6000,
                no_automatic_scientific_retry=True, wall_seconds=time.monotonic()-began)
    return e.new(out + '/complete.json', result)


def science(s, e, guard, stop, resume, release_path, spec_ref):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == s['GPU'], 'Only science child receives the UUID')
    require(Path(release_path).stat().st_mode & 0o222 == 0,'Science release must remain immutable')
    release=json.loads(Path(release_path).read_text()); r=e.read(release['registrations']['manager'])
    require(release['schema']=='root_released_source_aware_native_full_v2'
            and bare(release['spec'])==bare(spec_ref) and release['task_key']==task_key(s)
            and r['unit']==services(s,'manager') and bare(r['source'])==bare(s['assets']['runtime_adapter_source']),
            'Science release source/spec/service differs')
    require(os.getppid() == r['pid'] and Path('/proc/self/cgroup').read_text().strip() == '0::' + r['owned_cgroup'],
            'Science wrapper is not a child of the exact released manager')
    require(stop in (100,6000) and ((stop == 100 and resume is None) or
            (stop == 6000 and resume == s['paths']['train_out'] + '/checkpoint_00100.pt')), 'Only formal100→5900 phases')
    path=e.bound(s['sources']['full_refine_registered.py']); sys.path.insert(0, str(path.parent))
    native=load('_full_operator_v2_science', path); prefix=native.prefix
    require(sha(prefix.__file__) == PREFIX_SHA, 'Original shared prefix resource API differs')
    original=prefix.check_gpu; backend=guard.Backend(); own=backend.process(os.getpid()); count=0
    def check(uuid):
        nonlocal count
        require(uuid == s['GPU'], 'Wrong science UUID')
        value=guard.resource(backend.sample(), backend.process, own)
        count += 1
        e.new(s['paths']['training_manager_out'] + '/science_resource_API_' + str(stop) + '_' + str(count) + '.json', value)
        return ', '.join(str(value[k]) for k in ('uuid','name','memory_total','memory_used','driver'))
    prefix.check_gpu=check; old=sys.argv; sys.argv=science_argv(s,e,stop,resume)[2:]
    try: return native.main()
    finally: prefix.check_gpu=original; sys.argv=old


def completion_flow(s, e, backend, core):
    """Ordered exit, all-SHA return, integrity, uniform evaluation, final return."""
    r=e.read(s['assets']['training_registration'])
    terminal=core.exact_success(backend.wait_training(r),r)
    terminal_ref=e.new(s['paths']['callback_out'] + '/training_exact_exit.json', terminal)
    complete,segments,inventory=backend.inspect_training(s,terminal)
    training=core.validate_training(s,e,e.read(s['assets']['plan']),complete,segments,inventory)
    training_ref=e.new(s['paths']['callback_out'] + '/training_integrity.json',training)
    command=core.evaluator_argv(s,training['checkpoint']); resource=backend.prepare_evaluation(command,s)
    evaluated=backend.evaluate_once(command,s)
    require(evaluated['exit_code'] == 0 and evaluated['private_process_group_reaped'] is True,
            'Evaluation failed/leaked; preserve actual outputs')
    child_ref=e.new(s['paths']['callback_out'] + '/evaluation_child_exit.json',evaluated)
    evaluation=core.validate_evaluation(s,e,training['checkpoint'],evaluated['complete'])
    refs=training['required_refs']+evaluation['required_refs']+[terminal_ref,training_ref,resource,child_ref]
    refs += list(s['sources'].values())+[s['production_core_source']]
    refs += [v for v in s['assets'].values() if v is not None and isinstance(v,dict) and 'path' in v]
    returned=backend.return_closed(s,refs)
    require(returned['status'] == 'completed_native_full_training_and_evaluation_all_file_SHA_return',
            'Final entire returned inventory must pass')
    by={v['path']:v for v in returned['entries']}
    require(len(by)==len(returned['entries']), 'Duplicate return identities')
    for ref in refs:
        require(ref['path'] in by and bare(by[ref['path']])==bare(ref), 'Missing final returned reference')
        e.bound(by[ref['path']])
    value=dict(status='completed_registered_native_full_SR_exit_integrity_evaluation_and_SHA_return',
               task_key=task_key(s), spec_digest=core.digest(s), training_exit=terminal_ref,training=training,evaluation=evaluation,
               transfer=returned,returned_refs=list(by.values()),selection_performed=False,
               global_state_modified=False,new_Adam_calls=0)
    e.new(s['paths']['callback_out'] + '/complete.json',value)
    return value


def callback_backend(s,e,legacy,core):
    class Backend(legacy.Runtime):
        def inspect_training(self,s,terminal):
            r=e.read(s['assets']['training_registration'])
            inventory=self.helper_call('export',dict(workspace=s['workspace'],train_out=s['paths']['train_out'],
                manager_out=s['paths']['training_manager_out'],registration=r,terminal=terminal,purpose='native_full_SR_final6000'))
            self.export_inventory=inventory; self.transfer_export(inventory)
            manager=e.read(e.entry(s['paths']['training_manager_out']+'/complete.json'))
            require(manager['status']=='completed_source_aware_native_full_v2_manager'
                    and manager['task_key']==task_key(s) and manager['accepted_delta_updates']==6000,
                    'Real source-aware manager completion required')
            segments=[dict(path=v['path'],sha256=v['sha256']) for v in inventory['entries']
                      if re.fullmatch(re.escape(s['paths']['train_out'])+r'/segment_[0-9]+_[0-9]+\.json',v['path'])]
            return e.entry(s['paths']['train_out']+'/training_complete.json'),segments,inventory
        def evaluate_once(self,command,s):
            ev=s['evaluation']; env=dict(os.environ,CUDA_VISIBLE_DEVICES=ev['GPU'],FOURDSR_ROOT=ev['workspace'],
                FOURDSR_UPSTREAM=ev['upstream'],OMP_NUM_THREADS=str(ev['cpu_threads']),OPENBLAS_NUM_THREADS=str(ev['cpu_threads']))
            result=legacy.scientific_child(command,env,self.out/'evaluation_child.log',e,s['sources']['full_evaluate_registered.py'])
            if result['exit_code']==0:
                result['complete']=e.entry(s['paths']['evaluation_out']+'/complete.json')
                self.lock.close();self.lock=None
                t=s['CPU_temporal']; env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
                cpu=legacy.scientific_child([t['runtime_python'],str(e.bound(t['source'])),'--evaluation-complete',
                     str(e.bound(result['complete'])),'--manifest',str(e.bound(s['assets']['manifest'])),'--out',
                     str(e.path(t['out']))],env,self.out/'temporal_CPU_child.log',e,t['source'])
                e.new(s['paths']['callback_out']+'/temporal_CPU_child_exit.json',cpu)
                require(cpu['exit_code']==0 and cpu['private_process_group_reaped'] is True,'CPU temporal failed')
                done=e.read(e.entry(t['out']+'/complete.json'))
                summary=e.read(done['summary'])
                require(done['status']=='completed_registered_full_native_temporal_residual_diagnostic'
                        and done['observations']==300 and done['adjacent_pairs']==299
                        and done['identity']==summary['identity']
                        and bare(done['identity']['source'])==bare(t['source'])
                        and bare(done['identity']['evaluation_complete'])==bare(result['complete'])
                        and bare(done['identity']['manifest'])==bare(s['assets']['manifest'])
                        and bare(done['identity']['protocol'])==bare(s['assets']['evaluation_protocol'])
                        and done['formal_updates']==done['Adam_calls']==done['model_forwards']==done['GPU_calls']==0,
                        'Full CPU temporal identity/counts differ')
                self.temporal_refs=[e.entry(t['out']+'/complete.json'),done['summary'],*done['results'].values()]
            return result
    return Backend(s,e)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('close-role','validate','manager','callback','science'),required=True)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--spec',type=Path,required=True)
    p.add_argument('--spec-sha',required=True);p.add_argument('--release',type=Path)
    p.add_argument('--role',choices=('manager','callback'),default='manager')
    p.add_argument('--out');p.add_argument('--min-free-bytes',type=int);p.add_argument('--external-files',type=Path)
    p.add_argument('--stop',type=int);p.add_argument('--resume');a=p.parse_args()
    require(sha(a.spec)==a.spec_sha,'Spec SHA differs');s=json.loads(a.spec.read_text())
    core_path=a.root/s['production_core_source']['path']
    require(s['production_core_source']['sha256']==CORE_SHA and sha(core_path)==CORE_SHA,'Untouched integrity core required')
    core=load('_full_operator_v2_core_bootstrap',core_path);e=core.Evidence(a.root)
    spec_ref=e.entry(str(a.spec.resolve().relative_to(e.root)));source=e.entry(str(Path(__file__).resolve().relative_to(e.root)))
    if a.mode=='close-role':
        require(a.out and a.min_free_bytes and a.external_files,'Actual closure output/reserve/external inventory required')
        print(json.dumps(close_role(s,e,a.role,a.out,a.min_free_bytes,json.loads(a.external_files.read_text()))));return
    role=a.role if a.mode=='validate' else ('callback' if a.mode=='callback' else 'manager')
    plan,core,guard=validate(s,e,role,source);legacy=transport(s,e)
    if a.mode=='validate':
        require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU validate visibility must be empty')
        print(json.dumps(dict(status='passed_CPU_source_aware_operator_v2_registration',task_key=task_key(s),
                             model_imports=0,GPU_calls=0,formal_updates=0)));return
    require(a.release is not None,'Root release event path required')
    if a.mode=='science':return science(s,e,guard,a.stop,a.resume,a.release,spec_ref)
    argv=legacy.argv_of(os.getpid());owner,_=register_owner(s,e,role,spec_ref,argv,legacy)
    bound=resolve_release(s,e,spec_ref,owner,role,a.release,legacy)
    backend=None
    try:
        if a.mode=='manager':return train_manager(bound,e,plan,guard,legacy,spec_ref,a.release)
        backend=callback_backend(bound,e,legacy,core);return completion_flow(bound,e,backend,core)
    except BaseException as exc:
        directory=s['paths']['training_manager_out'] if role=='manager' else s['paths']['callback_out']
        partial=None
        if backend is not None:
            try:partial=backend.return_partial(bound,repr(exc))
            except BaseException as problem:partial=dict(status='partial_return_failed_preserved',error=repr(problem))
        e.new(directory+'/failure_'+str(time.time_ns())+'.json',dict(status='failed_source_aware_operator_preserved',
              error=repr(exc),traceback=traceback.format_exc(),automatic_scientific_retry=False,partial=partial))
        raise
    finally:
        if backend is not None:backend.close()


if __name__=='__main__':main()
