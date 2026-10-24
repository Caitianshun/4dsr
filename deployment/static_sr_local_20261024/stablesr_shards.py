"""Explicit, broker-admitted StableSR shards; no submission, transfer or deletion.

The pinned teacher bridge and its author command are unchanged. Each shard is a
new author process with seed 42. The author additionally resets that seed for
each outer tile. Neither policy asserts cross-hardware bitwise equivalence.
"""
from __future__ import annotations
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import resource
import shutil
import socket
import subprocess
import sys
import time
import traceback

EXP = Path('experiments/static_sr_wu_20261022')
BRIDGE = EXP/'methods/splatsure_stablesr.py'
OUTPUT = Path('output/static_sr_wu_20261024')
GiB = 1 << 30
FILE_CAP = 128 << 20
RESERVE = 20 * GiB
RNG = 'seed42_reset_per_shard; author_outer_tile_seed42_reset_preserved; no_serial_or_cross_hardware_bitwise_equivalence_claim'


def read(path): return json.loads(Path(path).read_text())
def now(): return datetime.now(timezone.utc).isoformat()
def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''): h.update(block)
    return h.hexdigest()
def identity(path): return dict(path=str(Path(path).absolute()), sha256=sha(path))
def canonical(value): return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
def write_new(path, value):
    with Path(path).open('x') as f: json.dump(value, f, indent=2, allow_nan=False); f.write('\n')
def normalize_uuid(value): return str(value).lower().removeprefix('gpu-')
def checked(item):
    p = Path(item['path'])
    if sha(p) != item['sha256']: raise ValueError('Changed frozen file: ' + str(p))
    return p
def load_bridge(root):
    spec = importlib.util.spec_from_file_location('frozen_splatsure_shard_bridge', Path(root)/BRIDGE)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module
def no_symlink(path):
    p = Path(path).absolute()
    if '..' in p.parts or any(q.is_symlink() for q in [p, *p.parents]): raise ValueError('Symlink or traversal path: ' + str(p))
    return p
def key(row): return f"{row['camera']}/{row['frame']:04d}"
def portable(row): return {k: v for k, v in row.items() if k not in ('lr_path', 'sr')}


def full_targets(bridge, manifest):
    m, rows = bridge.training_targets(manifest)
    if (m['scene'] != 'cook_spinach' or len(rows) != 6000 or m['frame_indices'] != list(range(300))
            or m['splits']['test'] != ['cam00'] or m['resolutions'] != {'lr': [336, 252], 'hr': [1344, 1008]}):
        raise ValueError('Current shard protocol requires complete Cook07 area4 x4, 6000 training keys')
    if len({key(r) for r in rows}) != len(rows) or any(r['camera'] == 'cam00' or r['split'] != 'train' for r in rows):
        raise ValueError('Duplicate or held-out target')
    return m, rows


def partition(root, manifest, output, size=100):
    if type(size) is not int or not 1 <= size <= 1000: raise ValueError('Invalid shard size')
    bridge = load_bridge(root); m, targets = full_targets(bridge, manifest)
    portable_targets = [portable(r) for r in targets]
    shards = [dict(id=f'shard_{i//size:03d}', start=i, stop=min(i+size, len(targets)),
                   keys=[key(r) for r in targets[i:i+size]]) for i in range(0, len(targets), size)]
    roster = dict(schema='stablesr_fixed_shard_roster_v1', scene=m['scene'], manifest=identity(manifest),
                  count=len(targets), shard_size=size, targets=portable_targets, shards=shards,
                  target_digest=canonical(portable_targets), rng_policy=RNG)
    write_new(output, roster); return roster


def verify_roster(roster, bridge, manifest):
    m, rows = full_targets(bridge, manifest)
    targets = [portable(r) for r in rows]
    if (roster.get('schema') != 'stablesr_fixed_shard_roster_v1' or roster['manifest']['sha256'] != sha(manifest)
            or roster['targets'] != targets or roster['target_digest'] != canonical(targets)
            or roster['scene'] != m['scene'] or roster['count'] != len(rows) or roster.get('rng_policy') != RNG):
        raise ValueError('Roster differs from the complete training manifest')
    size = roster['shard_size']
    expected = [dict(id=f'shard_{i//size:03d}', start=i, stop=min(i+size, len(rows)),
                     keys=[key(r) for r in rows[i:i+size]]) for i in range(0, len(rows), size)]
    if not 1 <= size <= 1000 or roster['shards'] != expected: raise ValueError('Shard partition changed')
    return rows


def png_bound(w, h):
    # RGB8 scanlines; generous zlib/chunk allowance. No producer PNG metadata.
    return ((math.ceil(((3*w+1)*h)*1.15)+65536+4095)//4096)*4096


def budget(rows):
    return sum(png_bound(*r['expected_sr_shape']) + Path(r['lr_path']).stat().st_size for r in rows) + 2*FILE_CAP + (8 << 20)


def storage_guard(directory, remaining):
    if os.statvfs(directory).f_frsize > 4096: raise ValueError('PNG budget requires allocation blocks <=4096 B')
    free = shutil.disk_usage(directory).free
    needed = RESERVE + sum(n['additional_bytes'] for n in remaining)
    if free < needed: raise ValueError(f'Insufficient space: free={free}, reserve_and_remaining={needed}')
    return dict(free_bytes=free, required_bytes=needed, reserve_bytes=RESERVE)


def gate(receipt_path, plan, bridge):
    r = read(receipt_path)
    if (r.get('schema') != 'static_sr_wu_teacher_acceptance_fixture_v1' or r.get('status') != 'fixture_passed'
            or r.get('method') != 'stablesr' or not r.get('gpu_executed', r.get('GPU_executed', False)) or not r.get('teacher_generated')
            or normalize_uuid(r.get('gpu_uuid', r.get('GPU_uuid', r.get('hardware', {}).get('gpu_uuid')))) != normalize_uuid(plan['gpu_uuid'])
            or r.get('host', r.get('hostname')) != plan['host'] or r.get('upstream_commit') != bridge.STABLESR_COMMIT):
        raise ValueError('A successful current-host/current-GPU StableSR fixture is required')
    sources = [r['wrapper'], r['registered_records'], *r['adapter_sources'], *r['model_identities'], r['runtime_assets_registration']]
    sources += r['outputs']
    for item in sources: checked(item)
    if not any(v['sha256'] == sha(Path(plan['root'])/BRIDGE) for v in r['adapter_sources']):
        raise ValueError('Fixture did not use this bridge')
    if {v['sha256'] for v in r['model_identities']} != {v[1] for v in bridge.WEIGHTS.values()}:
        raise ValueError('Fixture model mismatch')
    if any(r['runtime_assets_registration'].get(k) != plan['runtime_assets_identity'].get(k) for k in ('path','sha256')):
        raise ValueError('Fixture runtime cache registration differs')
    records = read(r['registered_records']['path'])
    if (records.get('schema') != 'static_sr_wu_train_records_v1' or records.get('heldout_cameras') != ['cam00']
            or records.get('manifest', {}).get('sha256') != plan['manifest']['sha256']):
        raise ValueError('Fixture registered training manifest differs')
    if not r['outputs'] or any(v['size_wh'] != [1344, 1008] for v in r['outputs']): raise ValueError('Fixture output grid differs')
    return sources


def build_plan(a):
    root = no_symlink(a.root).resolve(); out = no_symlink(a.out)
    if not out.is_relative_to(root/OUTPUT): raise ValueError('Use a fresh output/static_sr_wu_20261024 directory under task root')
    if not re.fullmatch(r'[a-z][a-z0-9-]{0,63}', a.job_id): raise ValueError('Invalid broker job ID')
    if not re.fullmatch(r'GPU-[0-9a-fA-F-]{36}', a.gpu_uuid): raise ValueError('Explicit full GPU UUID required')
    bridge = load_bridge(root); roster_path = Path(a.roster).absolute(); roster = read(roster_path)
    targets = verify_roster(roster, bridge, a.manifest)
    selected = a.shards.split(',')
    by_id = {s['id']: s for s in roster['shards']}
    if not selected or len(set(selected)) != len(selected) or set(selected)-set(by_id): raise ValueError('Unknown or repeated shard')
    # Explicit order is recorded. There is no dynamic queue or reassignment.
    out.mkdir(parents=True, exist_ok=False)
    template = bridge.plan(a.manifest, out/'template', a.source, a.weight_root, a.teacher_python, a.runtime_assets)
    if template['missing_dependencies']: raise ValueError('Missing teacher dependencies: ' + str(template['missing_dependencies']))
    if template['targets'] != targets: raise ValueError('Bridge changed full target order')
    required = [Path(__file__).absolute(), root/BRIDGE, roster_path, Path(a.manifest).absolute(),
                Path(a.teacher_python).absolute(), Path(a.runtime_assets).absolute(), Path(a.teacher_receipt).absolute()]
    required += list(map(Path, a.freeze_file or []))
    required.append(Path(template['runtime_assets']['cache_environment']['HUGGINGFACE_HUB_CACHE'])/bridge.OPENCLIP_CACHE_REPO/'refs/main')
    source = Path(template['source_root'])
    tracked = subprocess.check_output(['git', '-C', str(source), 'ls-files', '-z']).decode().split('\0')
    required += [source/p for p in tracked if p and (source/p).is_file()]
    value = dict(schema='stablesr_shard_job_v1', status='CPU_plan_ready_requires_broker', root=str(root), output_root=str(out),
        host=socket.gethostname(), broker=a.broker, broker_project_root=str(Path(a.broker_project_root or root).resolve()),
        job_id=a.job_id, gpu_uuid=a.gpu_uuid, visible_device=a.visible_device or a.gpu_uuid,
        roster=identity(roster_path), manifest=template['manifest'], recipe=template['recipe'],
        runtime_assets_identity=template['runtime_assets_identity'], teacher_receipt=identity(a.teacher_receipt),
        template=identity(out/'template/plan.json'), rng_policy=RNG, maximum_file_bytes=FILE_CAP,
        wrapper=identity(__file__), nodes=[])
    for item in gate(a.teacher_receipt, value, bridge): required.append(Path(item['path']))
    (out/'shards').mkdir()
    for sid in selected:
        s = by_id[sid]; path = out/'shards'/sid; path.mkdir()
        p = copy.deepcopy(template)
        p.update(output_root=str(path), execution_workdir=str(path/'execution_workdir'), targets=targets[s['start']:s['stop']],
                 shard=dict(roster_sha256=value['roster']['sha256'], id=sid, start=s['start'], stop=s['stop'], keys=s['keys'],
                            full_training_count=len(targets), full_manifest_sha256=value['manifest']['sha256'], rng_policy=RNG))
        p['command'] = bridge.command(p)
        write_new(path/'plan.json', p)
        value['nodes'].append(dict(id=sid, plan=identity(path/'plan.json'), keys=s['keys'], additional_bytes=budget(p['targets'])))
    value['required_files'] = [identity(p) for p in dict.fromkeys(required)]
    value['storage'] = storage_guard(out, value['nodes'])
    write_new(out/'job_plan.json', value)
    return value


def process_info(pid):
    fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
    return int(fields[1]), int(fields[19])


def admission(plan):
    if socket.gethostname() != plan['host'] or os.environ.get('CUDA_VISIBLE_DEVICES') != plan['visible_device']:
        raise ValueError('Execution host or visible GPU differs')
    group = Path('/proc/self/cgroup').read_text().strip()
    result = dict(host=socket.gethostname(), cgroup=group, CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES'], gpu_uuid=plan['gpu_uuid'])
    if plan['broker'] == 'local':
        if (group != '0::/claudevpn.slice/claude-local-'+plan['job_id']+'.service'
                or plan['visible_device'] != plan['gpu_uuid']): raise ValueError('Local broker job identity differs')
    elif plan['broker'] == 'remote':
        job_dir = no_symlink(os.environ.get('CLAUDE_REMOTE_JOB_DIR', '/missing'))
        base = Path(plan['broker_project_root'])/'.claude-training/jobs'
        if job_dir != base/plan['job_id']: raise ValueError('Remote broker job directory differs')
        context_path = job_dir/'execution_context.json'; context = read(context_path)
        if (context.get('version') != 1 or context.get('job_id') != plan['job_id']
                or Path(context['project_root']).resolve() != Path(plan['broker_project_root'])
                or not Path(plan['root']).is_relative_to(Path(context['project_root']))
                or context['boot_id'] != Path('/proc/sys/kernel/random/boot_id').read_text().strip()
                or list(map(normalize_uuid, context['selected_gpu_uuids'])) != [normalize_uuid(plan['gpu_uuid'])]):
            raise ValueError('Remote supervisor context differs')
        pid = int(context['supervisor_pid']); parent, ticks = process_info(pid)
        if ticks != context['startticks'] or Path(f'/proc/{pid}/cgroup').read_text().strip() != group:
            raise ValueError('Remote supervisor process identity differs')
        ancestor = os.getpid(); visited = set()
        while ancestor != pid and ancestor > 1 and ancestor not in visited:
            visited.add(ancestor); ancestor, _ = process_info(ancestor)
        if ancestor != pid: raise ValueError('Current process is not the admitted supervisor descendant')
        result['execution_context'] = identity(context_path)
        result['supervisor_context'] = context
    else: raise ValueError('Unknown broker')
    return result


def verify_job(plan_path):
    p = read(plan_path)
    if p.get('schema') != 'stablesr_shard_job_v1' or p.get('status') != 'CPU_plan_ready_requires_broker' or p.get('rng_policy') != RNG:
        raise ValueError('Invalid shard job plan')
    if p['maximum_file_bytes'] != FILE_CAP: raise ValueError('Storage failure cap changed')
    if Path(plan_path).absolute() != Path(p['output_root'])/'job_plan.json': raise ValueError('Plan moved')
    if p['wrapper'] != identity(__file__): raise ValueError('Shard wrapper changed')
    for item in p['required_files']: checked(item)
    bridge = load_bridge(p['root']); template = read(checked(p['template'])); bridge.revalidate(template)
    roster = read(checked(p['roster'])); targets = verify_roster(roster, bridge, checked(p['manifest']))
    by_id = {s['id']: s for s in roster['shards']}
    if len({n['id'] for n in p['nodes']}) != len(p['nodes']) or not p['nodes']: raise ValueError('Repeated or empty assignment')
    for n in p['nodes']:
        s = by_id[n['id']]; shard = read(checked(n['plan']))
        expected = copy.deepcopy(template); out = Path(p['output_root'])/'shards'/n['id']
        expected.update(output_root=str(out), execution_workdir=str(out/'execution_workdir'), targets=targets[s['start']:s['stop']],
            shard=dict(roster_sha256=p['roster']['sha256'], id=s['id'], start=s['start'], stop=s['stop'], keys=s['keys'],
                       full_training_count=len(targets), full_manifest_sha256=p['manifest']['sha256'], rng_policy=RNG))
        expected['command'] = bridge.command(expected)
        if (shard != expected or n['keys'] != s['keys'] or n['additional_bytes'] != budget(shard['targets'])
                or Path(n['plan']['path']) != out/'plan.json'): raise ValueError('Shard selection/command/budget changed')
    gate(checked(p['teacher_receipt']), p, bridge)
    return p, bridge


def run_one(job_path, sid):
    p, bridge = verify_job(job_path); proof = admission(p)
    claim = read(Path(p['output_root'])/'claim.json')
    if claim.get('plan') != identity(job_path) or claim.get('broker') != proof:
        raise ValueError('Shard child requires the matching admitted parent claim')
    node = next(n for n in p['nodes'] if n['id'] == sid)
    index = p['nodes'].index(node)
    storage_guard(p['output_root'], p['nodes'][index:])
    out = Path(node['plan']['path']).parent
    write_new(out/'claim.json', dict(job_plan=identity(job_path), broker=proof, started_at=now()))
    # A serialization failure boundary, not a scientific truncation. This also
    # bounds the author's verbose log. Reads of multi-GB model files are exempt.
    enforce_file_cap(p['maximum_file_bytes'])
    result = bridge.run(node['plan']['path'])
    verify_job(job_path); admission(p)
    seal = dict(schema='stablesr_shard_completion_v1', status='completed', shard_id=sid, keys=node['keys'],
        job_plan=identity(job_path), plan=node['plan'], receipt=identity(out/'receipt.json'), execution=identity(out/'execution.json'),
        roster=p['roster'], manifest=p['manifest'], broker=proof, completed_at=now(), rng_policy=RNG,
        files=[r['sr'] for r in result['targets']], inference_log=identity(out/'inference.log'))
    if p['broker'] == 'remote': seal['broker_context'] = identity(Path(p['output_root'])/'broker_execution_context.json')
    write_new(out/'completion.json', seal)


def enforce_file_cap(cap):
    if cap != FILE_CAP: raise ValueError('Unregistered file failure cap')
    resource.setrlimit(resource.RLIMIT_FSIZE, (cap, cap))


def run(job_path):
    p, _ = verify_job(job_path); proof = admission(p); out = Path(p['output_root'])
    write_new(out/'claim.json', dict(plan=identity(job_path), broker=proof, started_at=now()))
    started = time.monotonic(); completed = []; current = None
    try:
        if p['broker'] == 'remote':
            source = checked(proof['execution_context'])
            with (out/'broker_execution_context.json').open('xb') as target: target.write(source.read_bytes())
        for i, node in enumerate(p['nodes']):
            current = node['id']; verify_job(job_path); admission(p)
            storage_guard(out, p['nodes'][i:])
            target = Path(node['plan']['path']).parent
            python = read(node['plan']['path'])['python']
            with (target/'runner.log').open('x') as log:
                code = subprocess.run([python, str(Path(__file__).absolute()), '_one', '--plan', str(Path(job_path).absolute()),
                                       '--shard', current], stdout=log, stderr=subprocess.STDOUT, check=False).returncode
            if code: raise RuntimeError(f'Shard {current} failed (exit {code}); no later shard started')
            completion = identity(target/'completion.json'); record = read(checked(completion))
            if record['status'] != 'completed' or record['job_plan'] != identity(job_path): raise ValueError('Invalid child completion')
            completed.append(completion)
        verify_job(job_path); admission(p)
        result = dict(status='completed', plan=identity(job_path), completed=completed, seconds=time.monotonic()-started, broker=proof)
        write_new(out/'completion.json', result)
    except BaseException as e:
        write_new(out/'failure.json', dict(status='failed', plan=identity(job_path), failed_shard=current, completed=completed,
                   seconds=time.monotonic()-started, error=str(e), traceback=traceback.format_exc(), broker=proof))
        raise
    return result


class Mapper:
    """Explicit longest-prefix relocation, without changing original receipts."""
    def __init__(self, mappings):
        self.pairs = []
        for value in mappings:
            old, new = value.split('=', 1); old = Path(old); new = no_symlink(new)
            if not old.is_absolute() or '..' in old.parts or any(old == x[0] for x in self.pairs): raise ValueError('Invalid/ambiguous mapping')
            self.pairs.append((old, new))
        self.pairs.sort(key=lambda p: len(p[0].parts), reverse=True)
    def path(self, original):
        p = Path(original)
        if not p.is_absolute() or '..' in p.parts: raise ValueError('Invalid original path')
        for old, new in self.pairs:
            if p.is_relative_to(old): return no_symlink(new/p.relative_to(old))
        raise ValueError('No explicit path mapping: ' + str(p))
    def checked(self, item):
        p = self.path(item['path'])
        if sha(p) != item['sha256']: raise ValueError('Returned/reused file SHA differs: ' + str(p))
        return p


def verify_broker_provenance(seal, job, mapper):
    proof = seal['broker']
    if (proof.get('gpu_uuid') != job['gpu_uuid'] or proof.get('host') != job['host']
            or proof.get('CUDA_VISIBLE_DEVICES') != job['visible_device']):
        raise ValueError('Recorded broker GPU/host differs')
    if job['broker'] == 'local':
        if (proof.get('cgroup') != '0::/claudevpn.slice/claude-local-'+job['job_id']+'.service'
                or job['visible_device'] != job['gpu_uuid']): raise ValueError('Recorded local broker differs')
    elif job['broker'] == 'remote':
        original = proof['execution_context']; copied = seal['broker_context']
        if (original['sha256'] != copied['sha256'] or Path(copied['path']) != Path(job['output_root'])/'broker_execution_context.json'
                or Path(original['path']) != Path(job['broker_project_root'])/'.claude-training/jobs'/job['job_id']/'execution_context.json'):
            raise ValueError('Original broker context identity differs')
        context = read(mapper.checked(copied))
        if (context != proof.get('supervisor_context') or context.get('version') != 1 or context.get('job_id') != job['job_id']
                or context.get('project_root') != job['broker_project_root']
                or not Path(job['root']).is_relative_to(Path(context['project_root']))
                or list(map(normalize_uuid, context['selected_gpu_uuids'])) != [normalize_uuid(job['gpu_uuid'])]
                or type(context.get('supervisor_pid')) is not int or context['supervisor_pid'] <= 1
                or type(context.get('startticks')) is not int or context['startticks'] <= 0
                or not context.get('boot_id') or not proof.get('cgroup')):
            raise ValueError('Recorded remote supervisor binding differs')
    else: raise ValueError('Unknown recorded broker')


def verify_aggregate(root, manifest, roster_path, seals, mappings):
    """Require every roster key exactly once, even across hardware and jobs."""
    bridge = load_bridge(root); roster = read(roster_path); targets = verify_roster(roster, bridge, manifest)
    mapper = Mapper(mappings); seen = set(); all_targets = {}; originals = []; first = None; model_set = None
    by_id = {s['id']: s for s in roster['shards']}; job_cache = {}
    for seal_path in seals:
        seal_path = no_symlink(seal_path); seal = read(seal_path); sid = seal['shard_id']
        if (seal.get('schema') != 'stablesr_shard_completion_v1' or seal.get('status') != 'completed' or sid not in by_id
                or sid in seen or seal.get('rng_policy') != RNG or seal['keys'] != by_id[sid]['keys']
                or seal['roster']['sha256'] != sha(roster_path) or seal['manifest']['sha256'] != sha(manifest)):
            raise ValueError('Failed, duplicate, foreign or incomplete shard seal')
        seen.add(sid)
        job_path = mapper.checked(seal['job_plan']); job_key = str(job_path)
        if job_key not in job_cache:
            job = read(job_path)
            if (job.get('schema') != 'stablesr_shard_job_v1' or job['status'] != 'CPU_plan_ready_requires_broker'
                    or job['rng_policy'] != RNG or job['roster']['sha256'] != sha(roster_path)
                    or job['manifest']['sha256'] != sha(manifest) or job['wrapper']['sha256'] != sha(__file__)):
                raise ValueError('Foreign shard job plan')
            for item in job['required_files']: mapper.checked(item)
            template = read(mapper.checked(job['template']))
            for item in template['producer'] + template['model_identities']: mapper.checked(item)
            assets = template['runtime_assets']
            if ({x['role']: x['sha256'] for x in assets['files']} != bridge.RUNTIME_SHA256
                    or len(assets['files']) != len(bridge.RUNTIME_SHA256)):
                raise ValueError('Runtime model identities differ')
            for item in assets['files'] + assets['cache_mappings']: mapper.checked(item)
            if read(mapper.checked(template['runtime_assets_identity'])) != assets:
                raise ValueError('Runtime registration differs from plan')
            ref = Path(assets['cache_environment']['HUGGINGFACE_HUB_CACHE'])/bridge.OPENCLIP_CACHE_REPO/'refs/main'
            if mapper.path(str(ref)).read_text() != bridge.OPENCLIP_REVISION:
                raise ValueError('Returned offline revision ref differs, including whitespace')
            job_cache[job_key] = (job, template)
        job, template = job_cache[job_key]
        verify_broker_provenance(seal, job, mapper)
        nodes = [n for n in job['nodes'] if n['id'] == sid]
        if len(nodes) != 1 or nodes[0]['plan'] != seal['plan'] or nodes[0]['keys'] != seal['keys']: raise ValueError('Shard not in assigned job')
        p = read(mapper.checked(seal['plan'])); r = read(mapper.checked(seal['receipt'])); e = read(mapper.checked(seal['execution']))
        s = by_id[sid]; expected_rows = targets[s['start']:s['stop']]
        if (p['recipe'] != bridge.RECIPE or p['upstream_commit'] != bridge.STABLESR_COMMIT or p['seed'] != 42
                or p['command'] != bridge.command(p) or p['manifest']['sha256'] != sha(manifest)
                or [portable(x) for x in p['targets']] != [portable(x) for x in expected_rows]
                or p['shard']['id'] != sid or p['shard']['keys'] != s['keys']
                or p['shard']['roster_sha256'] != sha(roster_path)):
            raise ValueError('Altered recipe, command, inputs or shard identity')
        # All non-shard bridge parameters must equal the full author plan.
        fixed = copy.deepcopy(p)
        for field in ('output_root', 'execution_workdir', 'targets', 'command', 'shard'): fixed.pop(field, None)
        base = copy.deepcopy(template)
        for field in ('output_root', 'execution_workdir', 'targets', 'command'): base.pop(field, None)
        if fixed != base: raise ValueError('Shard changed a non-shard teacher parameter')
        expected_receipt = copy.deepcopy(p)
        expected_receipt.update(status='completed', execution=seal['execution'], plan=seal['plan'])
        expected_receipt['targets'] = r['targets']
        if r != expected_receipt or r['targets'] != [dict(x, sr=y.get('sr')) for x,y in zip(p['targets'], r['targets'])]:
            raise ValueError('Receipt no longer matches its original plan')
        if (e.get('status') != 'completed' or e.get('returncode') != 0 or e['plan'] != seal['plan'] or e['command'] != p['command']
                or e['execution_workdir'] != p['execution_workdir'] or e['source_root'] != p['source_root']
                or e['runtime_assets_identity'] != p['runtime_assets_identity'] or e['cuda_visible_devices'] != job['visible_device']
                or seal['broker'].get('gpu_uuid') != job['gpu_uuid'] or seal['broker'].get('host') != job['host']):
            raise ValueError('Missing successful plan-bound author execution')
        if seal['files'] != [x['sr'] for x in r['targets']]: raise ValueError('Sealed output set differs')
        mapper.checked(seal['inference_log'])
        current_models = {(x['role'], x['sha256']) for x in p['model_identities']}
        if len(p['model_identities']) != len(bridge.WEIGHTS) or current_models != {(role, spec[1]) for role, spec in bridge.WEIGHTS.items()}:
            raise ValueError('Wrong official models')
        if model_set is not None and current_models != model_set: raise ValueError('Different shard models')
        model_set = current_models
        for row in r['targets']:
            k = key(row)
            if k in all_targets: raise ValueError('Repeated training key')
            path = mapper.checked(row['sr'])
            if Path(row['sr']['path']) != Path(p['output_root'])/'generated'/row['staged_name']:
                raise ValueError('SR PNG did not come from this shard generated directory')
            if bridge.png_shape(path) != row['expected_sr_shape']: raise ValueError('Returned PNG grid changed')
            from PIL import Image
            with Image.open(path) as img: img.verify()
            local = copy.deepcopy(next(x for x in expected_rows if key(x) == k)); local['sr'] = identity(path)
            all_targets[k] = local
        originals.append(dict(seal=identity(seal_path), original_job_plan=seal['job_plan'],
            returned_job_plan=identity(job_path), original_receipt=seal['receipt'], returned_receipt=identity(mapper.checked(seal['receipt'])),
            original_execution=seal['execution'], returned_execution=identity(mapper.checked(seal['execution'])), hardware=seal['broker']))
        if first is None: first = p
    if seen != set(by_id) or set(all_targets) != {key(r) for r in targets}: raise ValueError('Full roster coverage missing')
    result = dict(schema='wu_splatsure_stablesr_teacher_v1', status='completed', method='splatsure', recipe=bridge.RECIPE,
        scene=roster['scene'], manifest=identity(manifest), training_only=True, parameter_updates=0, HR_images_read=0, heldout_images_read=0,
        model_identities=[dict(x, path=str(mapper.checked(x))) for x in first['model_identities']],
        targets=[all_targets[key(r)] for r in targets], producer=[identity(__file__), identity(Path(root)/BRIDGE)],
        sharding=dict(schema='stablesr_aggregated_shards_v1', roster=identity(roster_path), count=len(seen),
                      target_count=len(targets), rng_policy=RNG, mappings=list(mappings), original_receipts_preserved=True, shards=originals))
    return result


def aggregate(root, manifest, roster_path, seals, mappings, output):
    result = verify_aggregate(root, manifest, roster_path, seals, mappings)
    write_new(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest='action', required=True)
    p = sub.add_parser('partition'); p.add_argument('--root', required=True); p.add_argument('--manifest', required=True)
    p.add_argument('--out', required=True); p.add_argument('--size', type=int, default=100)
    p = sub.add_parser('plan')
    for name in ('root', 'manifest', 'roster', 'shards', 'out', 'job-id', 'gpu-uuid', 'teacher-python', 'teacher-receipt', 'runtime-assets', 'source', 'weight-root'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--broker', choices=('local', 'remote'), required=True); p.add_argument('--broker-project-root')
    p.add_argument('--visible-device'); p.add_argument('--freeze-file', action='append', default=[])
    for action in ('run', '_one'):
        p = sub.add_parser(action); p.add_argument('--plan', required=True)
        if action == '_one': p.add_argument('--shard', required=True)
    p = sub.add_parser('aggregate')
    for name in ('root', 'manifest', 'roster', 'out'): p.add_argument('--'+name, required=True)
    p.add_argument('--seal', action='append', required=True); p.add_argument('--map', action='append', required=True)
    a = parser.parse_args()
    if a.action == 'partition': result = partition(a.root, a.manifest, a.out, a.size)
    elif a.action == 'plan': result = build_plan(a)
    elif a.action == 'run': result = run(a.plan)
    elif a.action == '_one': run_one(a.plan, a.shard); result = {'status': 'shard_completed'}
    else: result = aggregate(a.root, a.manifest, a.roster, a.seal, a.map, a.out)
    print(json.dumps({k: result[k] for k in ('status', 'schema', 'count', 'storage') if k in result}))


if __name__ == '__main__': main()
