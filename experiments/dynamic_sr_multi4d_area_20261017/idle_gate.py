"""Read-only admission gate: all live Claude conversations idle, all GPUs free.

No Claude messages, model requests, process signals, lock removal, or policy edits.
An unreadable/unknown session, process owner, reservation, or remote probe blocks
admission. Display programs are excluded from model-compute occupancy.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys

HOME_DIR = Path('/home/cai_tianshun')
PROFILES = (HOME_DIR / '.claude-runtimeforge', HOME_DIR / '.claude', HOME_DIR / '.claude-new-subscription')
DISPLAY = {'/opt/todesk/bin/ToDesk_Session', '/usr/libexec/gnome-remote-desktop-daemon',
           '/usr/lib/xorg/Xorg', '/usr/bin/gnome-shell'}
REMOTE_ROOTS = {'a100-train': '/home/ubuntu/3DGS/4dsr', 'cts': '/home/cts/Project/4DSR'}
RESERVATIONS = Path('/var/lib/claude-training/reservations')
ROOT_JOBS = Path('/var/lib/claude-training/jobs')
TERMINAL = {'ended', 'completed', 'failed', 'rejected'}
# Linux reserves these inode numbers for the initial namespaces. A private
# PID/cgroup view cannot establish absence of host work and must not admit it.
INITIAL_PID_NAMESPACE = 'pid:[4026531836]'
INITIAL_CGROUP_NAMESPACE = 'cgroup:[4026531835]'


def stamp():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def proc_state(pid):
    try:
        raw = (Path('/proc') / str(pid) / 'stat').read_text()
    except FileNotFoundError:
        return None
    if not raw or raw.rfind(')') < 0:
        raise ValueError('Empty or malformed process identity')
    suffix = raw[raw.rfind(')') + 2:].split()
    if len(suffix) < 20:
        raise ValueError('Incomplete process identity')
    if suffix[0] in ('Z', 'X'):
        return None
    return {'pid': int(pid), 'state': suffix[0], 'start_ticks': suffix[19]}


def domain_context():
    """Prove which host PID/cgroup view the following absence checks use.

    Claude's pidDomain uses /etc/machine-id, not boot_id. Both are recorded;
    each registry must match machine-id + current PID namespace, and PID
    start ticks are checked only within this proven current host view.
    """
    context, blockers = {}, []
    try:
        context.update(machine_id=Path('/etc/machine-id').read_text().strip(),
                       boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                       pid_namespace=os.readlink('/proc/self/ns/pid'),
                       cgroup_namespace=os.readlink('/proc/self/ns/cgroup'))
        if not re.fullmatch(r'[0-9a-f]{32}', context['machine_id']):
            raise ValueError('Unrecognized machine-id')
        if not re.fullmatch(r'[0-9a-f-]{36}', context['boot_id']):
            raise ValueError('Unrecognized boot_id')
        if context['pid_namespace'] != INITIAL_PID_NAMESPACE or context['cgroup_namespace'] != INITIAL_CGROUP_NAMESPACE:
            raise ValueError('PID/cgroup namespace is not the proven initial host view')
        nspid = next(line.split(':', 1)[1].split() for line in Path('/proc/self/status').read_text().splitlines()
                     if line.startswith('NSpid:'))
        if nspid != [str(os.getpid())]:
            raise ValueError('Nested or translated PID view')
        own = Path('/proc/self/cgroup').read_text().strip().splitlines()
        if len(own) != 1 or not own[0].startswith('0::/'):
            raise ValueError('Unrecognized cgroup v2 view')
        own_path = own[0].split(':', 2)[2]
        visible = Path('/sys/fs/cgroup') / own_path.lstrip('/') / 'cgroup.procs'
        if str(os.getpid()) not in visible.read_text().split():
            raise ValueError('Own host cgroup membership is not visible')
        context['pid_domain'] = 'linux:' + context['machine_id'] + ':' + context['pid_namespace']
        context['host_view_verified'] = True
    except (OSError, ValueError, StopIteration) as error:
        context['host_view_verified'] = False
        blockers.append(dict(reason='host_namespace_unverifiable', error=str(error)))
    return context, blockers


def conversation_state():
    live, registered = [], set()
    domain, blockers = domain_context()
    if not domain['host_view_verified']:
        return dict(live_sessions=live, blockers=blockers, idle=False, domain=domain)
    for profile in PROFILES:
        directory = profile / 'sessions'
        try:
            if not directory.exists():
                continue
            paths = sorted(path for path in directory.iterdir() if path.suffix == '.json')
        except OSError as error:
            blockers.append(dict(reason='claude_registry_directory_unverifiable', path=str(directory), error=str(error)))
            continue
        for path in paths:
            try:
                record = json.loads(path.read_text())
                if not isinstance(record, dict):
                    raise ValueError('Registry record is not an object')
                if record.get('pidDomain') != domain['pid_domain']:
                    blockers.append(dict(reason='claude_registry_other_or_unknown_pid_domain', path=str(path),
                                         registry_domain=record.get('pidDomain'), current_domain=domain['pid_domain']))
                    continue
                if not str(record.get('procStart', '')).isdigit():
                    raise ValueError('Registry has no verifiable process start ticks')
                recorded_boot = record.get('boot_id', record.get('bootId'))
                if recorded_boot is not None and recorded_boot != domain['boot_id']:
                    blockers.append(dict(reason='claude_registry_other_boot_unverifiable', path=str(path),
                                         registry_boot_id=recorded_boot, current_boot_id=domain['boot_id']))
                    continue
                pid = int(record['pid'])
                process = proc_state(pid)
                if process is None:
                    continue
                if str(record.get('procStart')) != process['start_ticks']:
                    continue  # stale record for a recycled PID
                registered.add(pid)
                row = {key: record.get(key) for key in ('pid', 'sessionId', 'name', 'cwd', 'status', 'updatedAt')}
                row['registry'] = str(path)
                live.append(row)
                if record.get('status') != 'idle':
                    blockers.append(dict(reason='claude_conversation_not_explicitly_idle', **row))
            except (OSError, ValueError, KeyError, TypeError) as error:
                blockers.append(dict(reason='claude_registry_unverifiable', path=str(path), error=str(error)))
    # A Claude process without a current matching registry is not assumed idle.
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        try:
            executable = str((path / 'exe').readlink())
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            executable = ''
        try:
            comm = (path / 'comm').read_text().strip()
        except FileNotFoundError:
            continue
        except OSError as error:
            blockers.append(dict(reason='live_process_identity_unverifiable', pid=int(path.name), error=str(error)))
            continue
        if '/claude/versions/' not in executable and Path(executable).name != 'claude' and comm != 'claude':
            continue
        pid = int(path.name)
        if pid not in registered and proc_state(pid) is not None:
            blockers.append(dict(reason='unregistered_live_claude_process', pid=pid))
    return dict(live_sessions=live, blockers=blockers, idle=not blockers, domain=domain)


def reservation_finished(value):
    """Release stale root records only with terminal status AND original runner exit."""
    job_id = value['job_id']
    if not isinstance(job_id, str) or not re.fullmatch(r'[a-zA-Z0-9_-]+', job_id):
        raise ValueError('Invalid reservation job ID')
    path = ROOT_JOBS / job_id / 'run/status.json'
    # The root-owned job directory binds this status to the root reservation.
    job_stat = (ROOT_JOBS / job_id).lstat()
    if not stat.S_ISDIR(job_stat.st_mode) or job_stat.st_uid != 0 or job_stat.st_mode & 0o022:
        raise ValueError('Root job directory identity is untrusted')
    if path.is_symlink() or not path.resolve().is_relative_to((ROOT_JOBS / job_id).resolve()):
        raise ValueError('Job status escapes its root-controlled job directory')
    status = json.loads(path.read_text())
    if not isinstance(status, dict):
        raise ValueError('Root job status is not an object')
    if status.get('job_id') != job_id or status.get('gpu_uuids') != value['gpu_uuids']:
        raise ValueError('Root status/reservation identity mismatch')
    if status.get('state') not in TERMINAL or not isinstance(status.get('exit_code'), int) or 'end_time_unix' not in status:
        return False, dict(status_path=str(path), state=status.get('state'), reason='root_status_not_terminal')
    runner = status.get('runner')
    if not isinstance(runner, dict) or not isinstance(runner.get('pid'), int) or not runner.get('start_ticks'):
        raise ValueError('Terminal root status lacks original runner PID/start ticks')
    if value.get('main_pid') != runner['pid']:
        raise ValueError('Reservation main PID differs from the status runner')
    process = proc_state(runner['pid'])
    exited = process is None or process['start_ticks'] != str(runner['start_ticks'])
    return exited, dict(status_path=str(path), state=status['state'], runner=runner,
                        original_runner_exited=exited, current_process=process)


def local_reservations():
    active, errors = [], []
    directory = RESERVATIONS
    domain, domain_errors = domain_context()
    if domain_errors:
        return [], domain_errors
    try:
        info = directory.lstat()
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != 0 or info.st_gid != 0 or stat.S_IMODE(info.st_mode) != 0o755:
            raise ValueError('Root reservation directory must be root:root mode 0755, without symlink')
        paths = sorted(path for path in directory.iterdir() if path.suffix == '.json')
    except (OSError, ValueError):
        return [], [dict(reason='reservation_inventory_missing')]
    for path in paths:
        try:
            file_info = path.lstat()
            if not stat.S_ISREG(file_info.st_mode) or file_info.st_uid != 0 or file_info.st_mode & 0o022:
                raise ValueError('Reservation record must be an immutable root-owned regular file')
            value = json.loads(path.read_text())
            if not isinstance(value, dict):
                raise ValueError('Reservation record is not an object')
            if value.get('state') not in {'reserved', 'active', 'ended'}:
                raise ValueError('Unrecognized root reservation state')
            cgroup = value['expected_cgroup']
            if not isinstance(cgroup, str) or not cgroup.startswith('/') or '..' in Path(cgroup).parts:
                raise ValueError('Invalid reservation cgroup')
            events = Path('/sys/fs/cgroup') / cgroup.lstrip('/') / 'cgroup.events'
            try:
                populated = dict(line.split() for line in events.read_text().splitlines()).get('populated')
            except FileNotFoundError:
                populated = None  # absence is not evidence of release
            if populated == '1':
                active.append(dict(job_id=value['job_id'], gpu_uuids=value['gpu_uuids'], cgroup=cgroup))
            elif populated not in ('0', None):
                raise ValueError('Unknown cgroup populated state')
            else:
                finished, proof = reservation_finished(value)
                if not finished:
                    active.append(dict(job_id=value['job_id'], gpu_uuids=value['gpu_uuids'], cgroup=cgroup,
                                       cgroup_populated=populated, release_proof=proof))
        except (OSError, ValueError, KeyError, TypeError) as error:
            errors.append(dict(reason='reservation_unverifiable', path=str(path), error=str(error)))
    return active, errors


def gpu_sample(ignore_pids=()):
    command = ['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory',
               '--format=csv,noheader,nounits']
    process = subprocess.run(command, text=True, capture_output=True, timeout=20)
    if process.returncode:
        raise RuntimeError('Local GPU query failed: ' + process.stderr[-500:])
    occupants, display, own = [], [], []
    for values in csv.reader(process.stdout.splitlines()):
        if not values:
            continue
        if len(values) != 4:
            raise ValueError('Malformed compute-process record')
        gpu, pid, name, memory = [value.strip() for value in values]
        row = dict(gpu_uuid=gpu, pid=int(pid), process_name=name, memory_mib=memory)
        if name in DISPLAY:
            display.append(row)
        elif int(pid) in ignore_pids:
            own.append(row)
        else:
            occupants.append(row)
    return dict(occupants=occupants, display=display, own=own)


def project_jobs(roots):
    jobs, errors = [], []
    for root in roots:
        directory = Path(root) / '.claude-training/jobs'
        try:
            if not directory.exists():
                continue
            paths = [path / 'status.json' for path in directory.iterdir() if path.is_dir()]
        except OSError as error:
            errors.append(dict(path=str(directory), error=str(error)))
            continue
        for path in paths:
            try:
                value = json.loads(path.read_text())
                if not isinstance(value, dict):
                    raise ValueError('Project status is not an object')
                if value.get('state') not in TERMINAL:
                    jobs.append(dict(job_id=path.parent.name, state=value.get('state'), root=str(root)))
            except (OSError, ValueError) as error:
                errors.append(dict(path=str(path), error=str(error)))
    return jobs, errors


def remote_probe_script():
    # Execute this exact read-only gate source in memory on the remote host.
    # It is neither installed nor written remotely and cannot recursively SSH.
    source = Path(__file__).read_text()
    return ("__name__ = '_remote_idle_gate'\n__file__ = '<remote-idle-gate>'\n"
            + 'exec(compile(' + repr(source) + ", '<remote-idle-gate>', 'exec'), globals())\n") + '''
homes = {Path.home(), Path('/home/ubuntu'), Path('/home/cts'), Path('/home/cai_tianshun')}
PROFILES = tuple(home / profile for home in homes for profile in ('.claude-runtimeforge','.claude','.claude-new-subscription'))
value = check(remote=False)
jobs, errors = project_jobs([Path('/home/ubuntu/3DGS/4dsr'), Path('/home/ubuntu/3DGS/HOI'), Path('/home/cts/Project/4DSR'), Path('/home/cts/Project/HOI')])
value['project_jobs'] = jobs
value['blockers'].extend(dict(reason='remote_project_job_not_terminal', **job) for job in jobs)
value['blockers'].extend(dict(reason='remote_project_job_unverifiable', **error) for error in errors)
value['safe'] = not value['blockers']
print(json.dumps(value))
'''


def check(*, remote=False, ignore_pids=()):
    result = dict(schema='claude_idle_admission_v1', checked_utc=stamp(), safe=False,
                  condition='all live Claude sessions explicitly idle and no managed/unknown GPU work',
                  conversations=conversation_state(), remote_checked=False, blockers=[])
    result['blockers'].extend(result['conversations']['blockers'])
    reservations, errors = local_reservations()
    result['reservations'] = reservations
    result['blockers'].extend(errors)
    result['blockers'].extend(dict(reason='live_managed_gpu_reservation', **r) for r in reservations)
    try:
        result['local_gpu'] = gpu_sample(ignore_pids)
        result['blockers'].extend(dict(reason='other_or_unknown_local_compute', **r)
                                  for r in result['local_gpu']['occupants'])
    except (OSError, ValueError, subprocess.SubprocessError, RuntimeError) as error:
        result['blockers'].append(dict(reason='local_gpu_probe_unverifiable', error=str(error)))
    if remote and not result['blockers']:
        result['remote_checked'] = True
        result['remote'] = {}
        for host in REMOTE_ROOTS:
            try:
                probe = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=12',
                                       host, 'python3', '-'], input=remote_probe_script(),
                                       text=True, capture_output=True, timeout=35)
                if probe.returncode:
                    raise RuntimeError(f'SSH exit {probe.returncode}: {probe.stderr[-300:]}')
                value = json.loads(probe.stdout)
                result['remote'][host] = value
                if not value['safe'] or not value['conversations']['domain']['host_view_verified']:
                    result['blockers'].append(dict(reason='remote_gpu_or_managed_work_not_free', host=host, details=value))
            except (OSError, ValueError, KeyError, subprocess.SubprocessError, RuntimeError) as error:
                result['blockers'].append(dict(reason='remote_state_unverifiable', host=host, error=str(error)))
    result['safe'] = not result['blockers'] and (not remote or result['remote_checked'])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--remote', action='store_true')
    args = parser.parse_args()
    result = check(remote=args.remote)
    write(args.out, result)
    print(json.dumps(dict(safe=result['safe'], live_claude_sessions=len(result['conversations']['live_sessions']),
                          blockers=len(result['blockers']), remote_checked=result['remote_checked'])))


if __name__ == '__main__':
    main()
