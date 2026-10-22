"""CPU contracts for a sealed, resumable four-stage IE preparation prefix.

SHA256 binds evidence; it is not a cryptographic signature or a claim that a
budget guarantees convergence. No submission, GPU API, SSH, or deletion here.
"""
from __future__ import annotations

import resource
import stat
import subprocess

STAGES = ('ie_external', 'ie_internal_train', 'ie_internal_cache', 'ie_merge')
SCHEMA = 'static_sr_ie_preparation_completion_v1'
SELF = 'deployment/static_sr_cts_20261022/scene_prefix.py'
GIB = 1 << 30


def prefix_budget(api, nodes, count, hr_wh):
    """Incompressible raw bytes plus ample DEFLATE/ZIP/JSON/filesystem overhead.

    32 GiB is an enforced maximum individual file for checkpoint generation,
    not a point-count cap. Exceeding it fails the stage without changing any
    training iteration. The extra GiB covers small training records/logs.
    """
    raw = count * hr_wh[0] * hr_wh[1] * 16
    arrays = (raw * 101 + 99) // 100 + count * 2 * 8192 + (64 << 20)
    evidence = [api.identity(api.ROOT / SELF), api.identity(api.ROOT / api.SELF)]
    sizes = dict(ie_external=arrays, ie_internal_train=33*GIB,
                 ie_internal_cache=arrays, ie_merge=64 << 20)
    return {'schema': 'static_sr_scene_capacity_budget_v1', 'stages': {
        n['id']: {'additional_bytes': sizes[n['id']], 'basis': 'conservative_bound',
                  'maximum_file_bytes': 32*GIB if n['id']=='ie_internal_train' else sizes[n['id']],
                  'evidence': evidence} for n in nodes}}


def expected_commands(api, plan, plan_path):
    commands = []
    for stage in STAGES:
        for name, action in ((stage+'_disk_guard', 'guard'), (stage, 'stage-run')):
            commands.append(dict(name=name, argv=[plan['base_python'], '{ROOT}/'+api.SELF,
                                                  action, '--plan', api.token(plan_path), '--stage', stage]))
    commands.append(dict(name='seal_ie_prefix', argv=[plan['base_python'], '{ROOT}/'+api.SELF,
                                                     'seal-prefix', '--plan', api.token(plan_path)]))
    return commands


def safe_output(api, path, run):
    value = api.resolve(path).absolute()
    scope = api.resolve(run).absolute()
    if '..' in value.parts or not value.is_relative_to(scope):
        raise ValueError('Prefix output outside the exact scene/run')
    cursor = value
    while cursor != api.ROOT:
        if cursor.is_symlink():
            raise ValueError('Prefix output symlink is forbidden')
        if cursor == cursor.parent:
            raise ValueError('Prefix output outside snapshot')
        cursor = cursor.parent
    return value


def artifact_inventory(api, plan):
    result = []
    for node in plan['prefix_dag']:
        top = safe_output(api, node['output'], plan['run_root'])
        if not top.exists():
            raise ValueError('Missing prefix output: ' + str(top))
        paths = [top] if top.is_file() else sorted(top.rglob('*'))
        for path in paths:
            safe_output(api, path, plan['run_root'])
            st = path.lstat()
            if stat.S_ISDIR(st.st_mode):
                continue
            if not stat.S_ISREG(st.st_mode) or st.st_nlink != 1:
                raise ValueError('Prefix outputs must be private regular files')
            result.append(dict(api.identity(path), bytes=st.st_size))
    if len({i['path'] for i in result}) != len(result):
        raise ValueError('Duplicate prefix artifact')
    return sorted(result, key=lambda i: i['path'])


def verify_bundle(api, plan, files):
    """Check exact train-key coverage and every indexed identity against inventory."""
    config = api.read(api.checked(plan['configs']['iesrgs']))
    manifest = api.read(api.checked(config['manifest']))
    expected = {(r['camera_id'], r['frame_index']): r for r in manifest['observations'] if r['split'] == 'train'}
    root = plan['run_root'] + '/iesrgs'
    by = {r['path']: r for r in files}

    def bound(item):
        relative = api.relative(item['path'])
        if relative not in by or by[relative]['sha256'] != item['sha256']:
            raise ValueError('Prefix index references an unsealed artifact')

    for suffix, kind, fields in (
        ('external/index.json', 'iesrgs_external', ('external_rgb', 'external_inverse_depth')),
        ('internal_cache/index.json', 'iesrgs_internal', ('internal_rgb', 'internal_depth')),
        ('bundle.json', 'iesrgs_complete', ('external_rgb', 'external_inverse_depth', 'internal_rgb', 'internal_depth'))):
        index = api.read(root + '/' + suffix)
        entries = {(r['camera'], r['frame']): r for r in index['entries']}
        if (index.get('schema') != 'static_sr_wu_prior_bundle_v1' or index.get('kind') != kind
                or index.get('manifest') != config['manifest'] or index.get('scene') != config['scene']
                or index.get('training_only') is not True or len(entries) != len(index['entries'])
                or set(entries) != set(expected) or not index.get('producer') or not index.get('model_identities')):
            raise ValueError('Incomplete or foreign IE prefix bundle')
        for key, row in entries.items():
            if row['lr_sha256'] != expected[key]['lr_sha256'] or row['time'] != expected[key]['time']:
                raise ValueError('Prefix LR/time identity changed')
            for field in fields:
                bound(row[field])
        for item in index['producer'] + index['model_identities']:
            api.checked(item)
        if kind != 'iesrgs_external':
            bound(index['internal_teacher']['checkpoint'])
    complete = api.read(root + '/internal_train/complete.json')
    if (complete.get('status') != 'stage_completed' or complete.get('wiring_fixture') is not False
            or complete.get('stage') != 'internal' or complete.get('steps') != 30000
            or complete.get('config') != plan['configs']['iesrgs']):
        raise ValueError('IE internal parent is not the complete formal 30000-step run')
    bound(complete['checkpoint'])


def seal(api, plan_path):
    plan = api.read(plan_path)
    if plan.get('phase') != 'ie_prepare' or plan['status'] != 'executable_after_resource_admission':
        raise ValueError('Only an executable IE preparation plan can seal')
    for item in plan['required_files']:
        api.checked(item)
    log = plan['run_root'] + '/job_logs_ie_prepare'
    job = api.read(log + '/job.json')
    status = api.read(log + '/status.json')
    if (status['job_sha256'] != api.sha(api.resolve(log + '/job.json'))
            or job['commands'] != expected_commands(api, plan, plan_path)):
        raise ValueError('Prefix job identity changed')
    expected = [name for stage in STAGES for name in (stage+'_disk_guard', stage)]
    completed = status['stages'][:-1]
    if ([s['name'] for s in completed] != expected or any(s.get('returncode') != 0 for s in completed)
            or status['stages'][-1]['name'] != 'seal_ie_prefix'
            or job['commands'][-1]['name'] != 'seal_ie_prefix'):
        raise ValueError('Four prefix stages did not exit successfully')
    command_receipts = []
    for node in plan['prefix_dag']:
        path = plan['run_root'] + '/prefix_stage_receipts/' + node['id'] + '.json'
        receipt = api.read(path)
        if (receipt.get('returncode') != 0 or receipt.get('argv') != node['argv']
                or receipt.get('maximum_file_bytes') != plan['storage'][node['id']]['maximum_file_bytes']):
            raise ValueError('Scientific prefix command identity differs')
        command_receipts.append(api.identity(path))
    files = artifact_inventory(api, plan)
    verify_bundle(api, plan, files)
    value = dict(schema=SCHEMA, status='sealed_awaiting_job_terminal', scene=plan['scene'], run_id=plan['run_id'],
                 run_root=plan['run_root'], plan=api.identity(plan_path), job=api.identity(log+'/job.json'),
                 configs=plan['configs'], LR_identity=plan['LR_identity'], prefix_dag=plan['prefix_dag'],
                 required_files=plan['required_files'], command_receipts=command_receipts,
                 files=files, bytes=sum(i['bytes'] for i in files),
                 cleanup_allowed=False, full_scene_completed=False)
    api.write_new(plan['run_root'] + '/ie_prefix_completion.json', value)


def validate(api, manifest_path, current, require_fresh=True):
    receipt = api.read(manifest_path)
    if (receipt.get('schema') != SCHEMA or receipt.get('status') != 'sealed_awaiting_job_terminal'
            or any(receipt.get(k) != current[k] for k in ('scene', 'run_id', 'run_root', 'configs', 'LR_identity', 'prefix_dag'))):
        raise ValueError('Completed prefix belongs to a different plan/scene/run/command')
    prior = api.read(api.checked(receipt['plan']))
    if prior.get('phase') != 'ie_prepare' or prior.get('prefix_dag') != receipt['prefix_dag']:
        raise ValueError('Prefix preparation plan differs')
    if receipt.get('required_files') != prior.get('required_files'):
        raise ValueError('Prefix source/input freeze differs')
    for item in receipt['required_files']:
        api.checked(item)
    job = api.read(api.checked(receipt['job']))
    if job['commands'] != expected_commands(api, prior, receipt['plan']['path']):
        raise ValueError('Prefix frozen job commands differ from the original scientific plan')
    expected_receipts = [current['run_root']+'/prefix_stage_receipts/'+stage+'.json' for stage in STAGES]
    if [r['path'] for r in receipt.get('command_receipts', [])] != expected_receipts:
        raise ValueError('Prefix stage receipt coverage differs')
    for node, item in zip(current['prefix_dag'], receipt['command_receipts']):
        stage_receipt = api.read(api.checked(item))
        if (stage_receipt.get('returncode') != 0 or stage_receipt.get('argv') != node['argv']
                or stage_receipt.get('maximum_file_bytes') != prior['storage'][node['id']]['maximum_file_bytes']):
            raise ValueError('Scientific prefix execution receipt changed')
    log = current['run_root'] + '/job_logs_ie_prepare'
    terminal = api.read(log+'/complete.json')
    status = api.read(log+'/status.json')
    if (terminal != status or terminal.get('status') != 'completed' or terminal.get('scene') != current['scene']
            or terminal.get('job_id') != job['job_id'] or terminal.get('job_sha256') != receipt['job']['sha256']
            or terminal.get('gpu_uuid') != current['gpu_uuid'] or job['gpu_uuid'] != current['gpu_uuid']):
        raise ValueError('Prefix job terminal evidence is missing, failed, or foreign')
    expected = [name for stage in STAGES for name in (stage+'_disk_guard', stage)] + ['seal_ie_prefix']
    if ([s['name'] for s in terminal['stages']] != expected
            or [s['name'] for s in job['commands']] != expected
            or any(s.get('returncode') != 0 for s in terminal['stages'])):
        raise ValueError('Prefix terminal stages are incomplete')
    for command, stage in zip(job['commands'], terminal['stages']):
        expanded = [str(x).replace('{ROOT}', str(api.ROOT)) for x in command['argv']]
        if stage.get('argv') != expanded:
            raise ValueError('Prefix executed command differs from frozen job')
    actual = artifact_inventory(api, current)
    if actual != receipt['files'] or sum(i['bytes'] for i in actual) != receipt['bytes']:
        raise ValueError('Completed prefix artifacts changed')
    verify_bundle(api, current, actual)
    if require_fresh:
        allowed = {'iesrgs', 'job_logs_ie_prepare', 'prefix_stage_receipts', 'ie_prefix_completion.json'}
        if {p.name for p in api.resolve(current['run_root']).iterdir()} != allowed:
            raise ValueError('Resume run contains output beyond the sealed prefix')
        if {p.name for p in api.resolve(current['run_root']+'/iesrgs').iterdir()} != {'external','internal_train','internal_cache','bundle.json'}:
            raise ValueError('IE prefix contains unregistered remaining-stage output')
    return dict(completion=api.identity(manifest_path), terminal=api.identity(log+'/complete.json'),
                status=api.identity(log+'/status.json'), reused_stages=list(STAGES), retained_bytes=receipt['bytes'])


def run_stage(api, plan_path, stage):
    """A file-size resource limit fails oversized output; it never changes training."""
    plan = api.read(plan_path)
    if plan.get('status') != 'executable_after_resource_admission':
        raise ValueError('Blocked plan cannot run')
    api.guard(plan_path, stage)
    node = next(n for n in plan['dag'] if n['id'] == stage)
    limit = plan['storage'][stage].get('maximum_file_bytes')
    if type(limit) is not int or limit <= 0:
        raise ValueError('Resource wrapper requires a positive file limit from the frozen plan')
    def limits():
        _, current_hard = resource.getrlimit(resource.RLIMIT_FSIZE)
        hard = limit if current_hard == resource.RLIM_INFINITY else min(limit, current_hard)
        resource.setrlimit(resource.RLIMIT_FSIZE, (hard, hard))
    argv = [str(x).replace('{ROOT}', str(api.ROOT)) for x in node['argv']]
    result = subprocess.run(argv, cwd=api.ROOT, preexec_fn=limits, check=False)
    receipt_dir = 'prefix_stage_receipts' if plan.get('phase') == 'ie_prepare' else 'stage_resource_receipts'
    api.write_new(plan['run_root']+'/'+receipt_dir+'/'+stage+'.json',
                  dict(argv=node['argv'], returncode=result.returncode, maximum_file_bytes=limit,
                       cap_semantics='Failure boundary only; no iterations or Gaussian points are removed'))
    if result.returncode:
        raise RuntimeError('Prefix scientific command failed: '+stage+' rc='+str(result.returncode))
