"""Read-only target-GPU admission, without conversation or remote-host checks.

A verifiable host view, real target inventory, UUID-bound root reservations,
memory budget are required. Atomic admission and cgroup reservations belong to
the existing root broker. This module never takes a lease, clears a reservation,
starts CUDA, submits a job or sends a message.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess

import idle_gate as common

UUID = re.compile(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}\Z")
EXPECTED_GPU_NAMES = {
    'GPU-ddc2c5d8-3507-6293-837c-b1ea6b5e1cc5':
        'NVIDIA RTX PRO 6000 Blackwell Workstation Edition',
}


def inventory():
    command = ['nvidia-smi', '--query-gpu=uuid,name,memory.total,memory.free',
               '--format=csv,noheader,nounits']
    process = subprocess.run(command, text=True, capture_output=True, timeout=20)
    if process.returncode:
        raise RuntimeError('GPU inventory failed: ' + process.stderr[-500:])
    devices = {}
    for row in csv.reader(process.stdout.splitlines()):
        if len(row) != 4:
            raise ValueError('Malformed GPU inventory record')
        gpu, name, total, free = [value.strip() for value in row]
        if not UUID.fullmatch(gpu) or gpu in devices or not name:
            raise ValueError('Unrecognized or duplicate host GPU identity')
        devices[gpu] = dict(uuid=gpu, name=name, total_mib=total, free_mib=free)
    return devices


def gpu_uuids(value, known):
    if (not isinstance(value, list) or not value or len(value) != len(set(value))
            or any(not isinstance(gpu, str) or not UUID.fullmatch(gpu) for gpu in value)):
        raise ValueError('Reservation GPU UUIDs must be a nonempty unique list of full UUIDs')
    if not set(value) <= set(known):
        raise ValueError('Reservation refers to a GPU outside the verified host inventory')
    return value


def reservations(known):
    """Validate membership even for records the common helper proves finished."""
    errors = []
    try:
        for path in common.RESERVATIONS.iterdir():
            if path.suffix != '.json':
                continue
            info = path.lstat()
            if not stat.S_ISREG(info.st_mode) or info.st_uid != 0 or info.st_mode & 0o022:
                raise ValueError('Untrusted root reservation record: ' + str(path))
            value = json.loads(path.read_text())
            gpu_uuids(value['gpu_uuids'], known)
    except (OSError, ValueError, KeyError, TypeError) as error:
        errors.append(dict(reason='reservation_gpu_membership_unverifiable', error=str(error)))
    active, common_errors = common.local_reservations()
    errors.extend(common_errors)
    for record in active:
        try:
            gpu_uuids(record['gpu_uuids'], known)
        except (ValueError, KeyError, TypeError) as error:
            errors.append(dict(reason='active_reservation_gpu_membership_unverifiable', error=str(error)))
    return active, errors


def process_details(pid):
    state = common.proc_state(pid)
    if state is None:
        return None
    raw = (Path('/proc') / str(pid) / 'stat').read_text()
    fields = raw[raw.rfind(')') + 2:].split()
    if len(fields) < 20 or fields[19] != state['start_ticks'] or fields[0] in ('Z', 'X'):
        raise ValueError('Process identity changed during ancestry read')
    return dict(pid=int(pid), ppid=int(fields[1]), start_ticks=fields[19])


def capture_owned_identity(pid):
    domain, errors = common.domain_context()
    if errors or not domain.get('host_view_verified'):
        raise ValueError('Cannot bind the owned child to a verified host namespace')
    process = process_details(pid)
    if process is None:
        raise ValueError('Owned child exited before its process identity was captured')
    return dict(pid=int(pid), start_ticks=process['start_ticks'], boot_id=domain['boot_id'],
                pid_domain=domain['pid_domain'])


def owned_process(pid, identity, domain):
    """Ignore only a stable exact PID/start or a double-checked descendant chain."""
    if not identity:
        return False, None
    if identity.get('boot_id') != domain['boot_id'] or identity.get('pid_domain') != domain['pid_domain']:
        raise ValueError('Owned process identity belongs to another host/boot/PID namespace')
    root = process_details(identity['pid'])
    if root is None or root['start_ticks'] != str(identity['start_ticks']):
        raise ValueError('Owned root PID exited or was reused')
    chain, seen = [], set()
    current = int(pid)
    for _ in range(128):
        if current in seen or current <= 0:
            return False, None
        seen.add(current)
        record = process_details(current)
        if record is None:
            return False, None
        chain.append(record)
        if current == root['pid']:
            if record['start_ticks'] != root['start_ticks']:
                raise ValueError('Owned root PID changed while proving ancestry')
            for child, parent in zip(chain, chain[1:]):
                if int(child['start_ticks']) < int(parent['start_ticks']):
                    raise ValueError('Impossible parent/child process chronology')
            for record in chain:
                if process_details(record['pid']) != record:
                    raise ValueError('Process ancestry changed during ownership proof')
            return True, chain
        current = record['ppid']
    return False, None


def _root_json(path):
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_uid != 0 or info.st_mode & 0o022:
        raise ValueError('Untrusted root broker identity file: ' + str(path))
    return json.loads(path.read_text())


def _cgroup(pid):
    value = (Path('/proc') / str(pid) / 'cgroup').read_text().strip()
    if not value.startswith('0::/') or '\n' in value:
        raise ValueError('Unverifiable process cgroup')
    return value


def frozen_manifest_proof(job, gpu_uuid):
    """Verify both requested fields and the expected root-derived closure."""
    jid = job['job_id']
    if not isinstance(jid, str) or not re.fullmatch(r'[a-z][a-z0-9-]{0,63}', jid):
        raise ValueError('Invalid owned broker job ID')
    directory = common.ROOT_JOBS / jid
    info = directory.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != 0 or info.st_mode & 0o022:
        raise ValueError('Broker job directory is not root-controlled')
    manifest_path = directory / 'manifest.json'
    manifest = _root_json(manifest_path)
    submission = _root_json(directory / 'submission.json')
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if submission.get('manifest_sha256') != manifest_sha:
        raise ValueError('Frozen root manifest differs from its root submission identity')
    expected = job['request_manifest']
    for key in ('version', 'job_id', 'project_id', 'role', 'operation', 'argv', 'cwd', 'env_id',
                'gpu_uuids', 'gpu_memory_mib', 'timeout_seconds'):
        if manifest.get(key) != expected.get(key):
            raise ValueError('Owned root manifest does not match this submitted request: ' + key)
    if (manifest.get('job_id') != jid or manifest.get('role') != 'local'
            or manifest.get('operation') != 'run' or manifest.get('project_id') != '4dsr'
            or manifest.get('gpu_uuids') != [gpu_uuid]):
        raise ValueError('Owned broker job is not this target-GPU 4dsr local run')
    derived = job['expected_broker_identity']
    for key in ('project_root', 'runner_uid', 'env_root'):
        if manifest.get(key) != derived[key]:
            raise ValueError('Root-derived broker runtime identity changed: ' + key)
    for key, value in derived['gpu_admission'].items():
        if manifest.get('gpu_admission', {}).get(key) != value:
            raise ValueError('Root-derived GPU admission identity changed: ' + key)
    return dict(directory=str(directory), manifest=manifest, manifest_sha256=manifest_sha,
                manifest_path=str(manifest_path), submission=submission)


def broker_job_proof(job, gpu_uuid, domain):
    """Bind an exemption to this controller's frozen root job and live cgroup."""
    if job['host_boot_id'] != domain['boot_id'] or job['host_pid_domain'] != domain['pid_domain']:
        raise ValueError('Owned broker job was submitted in a different host/boot/PID domain')
    frozen = frozen_manifest_proof(job, gpu_uuid)
    jid = job['job_id']
    directory, manifest = Path(frozen['directory']), frozen['manifest']
    reservation = _root_json(common.RESERVATIONS / (jid + '.json'))
    guard = _root_json(directory / 'verified-guard.json')
    path = directory / 'run/status.json'
    if path.is_symlink() or not path.resolve().is_relative_to(directory.resolve()):
        raise ValueError('Broker status path escapes its root-bound job directory')
    status = json.loads(path.read_text())
    if status.get('job_id') != jid or status.get('gpu_uuids') != [gpu_uuid] or status.get('state') != 'running':
        raise ValueError('Owned broker status is not a live matching target-GPU run')
    runner = status['runner']
    pid = int(runner['pid'])
    live = common.proc_state(pid)
    expected_cgroup = '/claudevpn.slice/claude-local-' + jid + '.service'
    if (reservation.get('job_id') != jid or reservation.get('gpu_uuids') != [gpu_uuid]
            or reservation.get('state') != 'active' or reservation.get('main_pid') != pid
            or reservation.get('expected_cgroup') != expected_cgroup
            or guard.get('main_pid') != pid or guard.get('cgroup') != expected_cgroup
            or guard.get('role') != 'local' or not guard.get('private_network')
            or live is None or live['start_ticks'] != str(runner['start_ticks'])
            or runner.get('cgroup') != '0::' + expected_cgroup or _cgroup(pid) != runner['cgroup']):
        raise ValueError('Root reservation/guard/status/live runner identity mismatch')
    uid_line = next(line for line in (Path('/proc') / str(pid) / 'status').read_text().splitlines()
                    if line.startswith('Uid:'))
    if {int(value) for value in uid_line.split()[1:]} != {int(manifest['runner_uid'])}:
        raise ValueError('Live root worker UID differs from frozen broker identity')
    unit = 'claude-local-' + jid + '.service'
    systemd = subprocess.run(['systemctl', 'show', unit, '--no-pager',
                             '--property=MainPID,ControlGroup,ActiveState,PrivateNetwork'],
                            capture_output=True, text=True, timeout=15)
    values = dict(line.split('=', 1) for line in systemd.stdout.splitlines() if '=' in line)
    if (systemd.returncode or values.get('MainPID') != str(pid)
            or values.get('ControlGroup') != expected_cgroup or values.get('ActiveState') != 'active'
            or values.get('PrivateNetwork') != 'yes'):
        raise ValueError('Systemd does not confirm this root service runner/cgroup/isolation')
    network_visibility = 'readable'
    try:
        network = os.readlink('/proc/' + str(pid) + '/ns/net')
    except PermissionError:
        network = None  # Equivalent root systemd + guard proof remains required.
        network_visibility = 'unavailable_permission_denied'
    if network == '':
        # This host also returns zero bytes for a cross-UID namespace link.
        # Keep the same mandatory root/systemd proof as for PermissionError.
        network = None
        network_visibility = 'unavailable_empty_readlink'
    if network is not None and not re.fullmatch(r'net:\[[0-9]+\]', network):
        raise ValueError('Malformed current service network namespace identity')
    if (guard.get('netns') != runner.get('netns')
            or guard.get('netns') == guard.get('host_netns')
            or (network is not None and network != guard.get('netns'))):
        raise ValueError('Root guard and current service network identity differ')
    return dict(job_id=jid, gpu_uuids=[gpu_uuid], cgroup=expected_cgroup,
                manifest_sha256=frozen['manifest_sha256'], manifest=frozen['manifest_path'], status=status,
                systemd=values, runner=runner,
                netns_verification=dict(visibility=network_visibility, live_netns=network,
                                        guard_netns=guard.get('netns'), runner_netns=runner.get('netns'),
                                        host_netns=guard.get('host_netns'),
                                        root_systemd_fallback=network is None),
                owned_identity=dict(pid=pid, start_ticks=live['start_ticks'],
                                    boot_id=domain['boot_id'], pid_domain=domain['pid_domain']))


def check(*, gpu_uuid, minimum_free_vram_gib, broker_job=None, expected_gpu_name=None):
    result = dict(schema='target_gpu_idle_admission_v1', checked_utc=common.stamp(), safe=False,
                  condition='target GPU only: verified host, no external compute or effective target reservation, sufficient VRAM; atomic admission delegated to existing root broker',
                  gpu_uuid=gpu_uuid, blockers=[], policy='target_gpu_broker',
                  conversations_checked=False, remote_checked=False)
    domain, errors = common.domain_context()
    result['domain'] = domain
    result['blockers'].extend(errors)
    if not domain.get('host_view_verified'):
        return result
    try:
        if not isinstance(gpu_uuid, str) or not UUID.fullmatch(gpu_uuid):
            raise ValueError('A full target GPU UUID is required')
        minimum = float(minimum_free_vram_gib) * 1024
        if isinstance(minimum_free_vram_gib, bool) or not math.isfinite(minimum) or minimum <= 0:
            raise ValueError('An explicit positive minimum_free_vram_gib is required')
        devices = inventory()
        if gpu_uuid not in devices:
            raise ValueError('Target GPU does not exist in this host inventory')
        target = dict(devices[gpu_uuid])
        if '5090' in target['name']:
            raise ValueError('RTX 5090 is not authorized for this training queue')
        expected_name = expected_gpu_name or EXPECTED_GPU_NAMES.get(gpu_uuid)
        if (not isinstance(expected_name, str) or not expected_name.strip()
                or expected_name.strip().lower() in ('n/a', 'unknown')
                or ' '.join(target['name'].split()) != ' '.join(expected_name.split())):
            raise ValueError('Target GPU model differs from its frozen expected identity')
        target['expected_name'] = expected_name
        total, free = float(target['total_mib']), float(target['free_mib'])
        if not (math.isfinite(total) and math.isfinite(free) and 0 <= free <= total and total > 0):
            raise ValueError('Target GPU memory state is unverifiable')
        target.update(total_mib=total, free_mib=free, minimum_free_mib=minimum)
        result['target'] = target
        active, errors = reservations(devices)
        result['blockers'].extend(errors)
        proof = broker_job_proof(broker_job, gpu_uuid, domain) if broker_job else None
        result['owned_broker_job'] = proof
        owned_identity = proof['owned_identity'] if proof else None
        own_reservations = [value for value in active if proof and value.get('job_id') == proof['job_id']
                            and value.get('gpu_uuids') == [gpu_uuid] and value.get('cgroup') == proof['cgroup']]
        result['owned_reservations'] = own_reservations
        result['reservations'] = [value for value in active if value not in own_reservations
                                  and isinstance(value.get('gpu_uuids'), list) and gpu_uuid in value['gpu_uuids']]
        result['other_gpu_reservations'] = [value for value in active if value not in result['reservations']]
        result['blockers'].extend(dict(reason='effective_target_gpu_reservation', **value) for value in result['reservations'])
        sample = common.gpu_sample()
        own, outside, display = [], [], []
        for row in sample['occupants'] + sample.get('own', []):
            if row['gpu_uuid'] != gpu_uuid:
                continue
            proven, chain = owned_process(row['pid'], owned_identity, domain)
            if proven and _cgroup(row['pid']) != '0::' + proof['cgroup']:
                raise ValueError('An owned GPU descendant escaped its reserved service cgroup')
            if proven:
                own.append(dict(row, ownership_chain=chain))
            else:
                outside.append(row)
        display = [row for row in sample['display'] if row['gpu_uuid'] == gpu_uuid]
        result['compute'] = dict(own=own, external_or_unknown=outside, display=display)
        result['blockers'].extend(dict(reason='external_or_unknown_target_compute', **row) for row in outside)
        if len({row['pid'] for row in own}) > 1:
            result['blockers'].append(dict(reason='multiple_owned_cuda_processes_in_single_stage',
                                          pids=sorted({row['pid'] for row in own})))
        # While monitoring, put back only memory attributed to proven owned
        # compute processes. Our own allocation must not invalidate admission.
        own_mib = sum(float(row['memory_mib']) for row in own)
        if not math.isfinite(own_mib) or own_mib < 0:
            raise ValueError('Owned GPU memory accounting is unverifiable')
        effective_free = min(total, free + own_mib)
        target.update(owned_mib=own_mib, free_excluding_owned_mib=effective_free)
        if effective_free < minimum:
            result['blockers'].append(dict(reason='target_gpu_insufficient_free_memory',
                                          available_mib=effective_free, required_mib=minimum))
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, StopIteration, subprocess.SubprocessError) as error:
        result['blockers'].append(dict(reason='target_gpu_state_unverifiable', error=str(error)))
    result['safe'] = not result['blockers']
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--minimum-free-vram-gib', type=float, required=True)
    parser.add_argument('--expected-gpu-name')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = check(gpu_uuid=args.gpu_uuid, minimum_free_vram_gib=args.minimum_free_vram_gib,
                   expected_gpu_name=args.expected_gpu_name)
    common.write(args.out, result)
    print(json.dumps(dict(safe=result['safe'], blockers=len(result['blockers']), policy=result['policy'])))


if __name__ == '__main__':
    main()
