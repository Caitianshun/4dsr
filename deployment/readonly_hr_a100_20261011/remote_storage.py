"""Owned Long360 HR-only additive storage helper. No CUDA, sudo or system install."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path('/home/ubuntu/3DGS/4dsr/workspaces/highres_20261011')
DRIVE = Path('/media/ubuntu/Game Drive')
EXTERNAL = DRIVE / '4dsr-highres-20261011/hr_readonly_inputs'
PREFIX = 'data/dynamic_sr/highres_20261011/hr_full/l3H_long360/'
PROBE = ROOT / '.tools/bindfs_probe_20261011T055600Z'
BINARY = PROBE / 'extracted/usr/bin/bindfs'
PACKAGE = PROBE / 'bindfs_1.14.7-1_amd64.deb'
BINARY_SHA = '9f0da916b05f19558f4e272d1d75b2107df1e2eaadaa072cd5860576e299fc3a'
PACKAGE_SHA = '22a1e0f43a6ae348f779c0a18090bda96b0285dd13c5c092d363b42bed57ba80'
RESERVE = 60 * 10**9


def digest(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def no_symlinks(p):
    p = Path(p)
    for q in [p, *p.parents]:
        if q.is_symlink():
            raise ValueError('Symlink/ancestor rejected: ' + str(q))
    return p


def owned(p):
    no_symlinks(p)
    if Path(p).stat().st_uid != os.getuid():
        raise ValueError('Path must belong to current UID: ' + str(p))


def validate_inventory(files):
    expected = {'hr_full_storage.json'} | {f'hr/cam{c:02d}/{f:04d}.png' for c in range(1, 33) for f in range(300)}
    normalized = {}
    for rel, value in files.items():
        path = PurePosixPath(rel)
        if path.is_absolute() or '..' in path.parts or not rel.startswith(PREFIX):
            raise ValueError('Only canonical Long360 training-HR prefix is accepted')
        short = rel[len(PREFIX):]
        if short not in expected or rel != PREFIX + str(PurePosixPath(short)):
            raise ValueError('HR member outside exact 32-camera/300-frame allowlist')
        if type(value.get('bytes')) is not int or value['bytes'] <= 0 or not re.fullmatch('[0-9a-f]{64}', value.get('sha256', '')):
            raise ValueError('Invalid byte count or SHA256')
        normalized[short] = dict(bytes=value['bytes'], sha256=value['sha256'])
    if set(normalized) != expected:
        raise ValueError('Incomplete full training-HR population')
    return normalized


def paths(run_id):
    if not re.fullmatch('[a-z0-9_-]{1,64}', run_id):
        raise ValueError('Unsafe run identity')
    return EXTERNAL / run_id / 'hr_tree', ROOT / PREFIX.rstrip('/'), ROOT / 'output/dynamic_sr_highres_20261011/storage' / run_id


def guard(path, needed):
    free = shutil.disk_usage(path).free
    if free - needed < RESERVE:
        raise RuntimeError('60GB storage reserve would be violated')
    return dict(path=str(path), free_bytes=free, needed_bytes=needed, reserve_bytes=RESERVE)


def capture(argv):
    p = subprocess.run(argv, text=True, capture_output=True)
    return dict(argv=argv, returncode=p.returncode, stdout=p.stdout, stderr=p.stderr)


def mountinfo(target):
    def decode(s):
        return s.replace('\\040', ' ').replace('\\011', '\t').replace('\\012', '\n').replace('\\134', '\\')
    rows = []
    for line in Path('/proc/self/mountinfo').read_text().splitlines():
        fields = line.split()
        if decode(fields[4]) == str(target):
            split = fields.index('-')
            rows.append(dict(raw=line, mountpoint=decode(fields[4]), options=fields[5].split(','),
                             fstype=fields[split + 1], source=decode(fields[split + 2]), super_options=fields[split + 3].split(',')))
    return rows


def binary_identity():
    for path, expected in ((BINARY, BINARY_SHA), (PACKAGE, PACKAGE_SHA)):
        no_symlinks(path)
        if not path.is_file() or digest(path) != expected:
            raise ValueError('Pinned package/binary identity differs')
    if PACKAGE.stat().st_size != 30540:
        raise ValueError('Pinned package size differs')
    fields = capture(['dpkg-deb', '-f', str(PACKAGE), 'Package', 'Version', 'Architecture'])
    if fields['returncode'] or fields['stdout'].splitlines() != ['Package: bindfs', 'Version: 1.14.7-1', 'Architecture: amd64']:
        raise ValueError('Pinned package metadata differs')
    version, libraries = capture([str(BINARY), '--version']), capture(['ldd', str(BINARY)])
    if version['returncode'] or libraries['returncode'] or 'not found' in libraries['stdout']:
        raise ValueError('Existing runtime libraries unavailable')
    return dict(binary=dict(path=str(BINARY), sha256=BINARY_SHA), package=dict(path=str(PACKAGE), sha256=PACKAGE_SHA), version=version, libraries=libraries, package_fields=fields)


def context(run_id):
    source, target, out = paths(run_id)
    for p in (ROOT, DRIVE, EXTERNAL, source, target, out):
        no_symlinks(p)
    owned(ROOT)
    owned(DRIVE)
    drive = capture(['findmnt', '-J', '-T', str(DRIVE)])
    if drive['returncode']:
        raise ValueError('External filesystem is absent')
    row = json.loads(drive['stdout'])['filesystems'][0]
    if row['target'] != str(DRIVE) or row['fstype'] != 'exfat' or 'rw' not in row['options'].split(','):
        raise ValueError('Expected writable existing Game Drive exFAT mount required')
    return source, target, out, drive, binary_identity()


def verify_tree(directory, files):
    started = time.monotonic()
    no_symlinks(directory)
    owned(directory)
    found = set()
    for p in directory.rglob('*'):
        no_symlinks(p)
        if p.is_file():
            found.add(str(p.relative_to(directory)))
        elif not p.is_dir():
            raise ValueError('Nonregular source member rejected')
    if found != set(files):
        raise ValueError('Unexpected or missing external source members')
    for rel, value in files.items():
        p = directory / rel
        if p.stat().st_size != value['bytes'] or digest(p) != value['sha256']:
            raise ValueError('Preserve differing external HR member: ' + rel)
    storage = json.loads((directory / 'hr_full_storage.json').read_text())
    if storage.get('schema') != 'highres_hr_full_storage_v1' or storage.get('LR_roundtrip_all_frames_SHA_equal') is not True or storage.get('GPU_calls') != 0:
        raise ValueError('Original complete CPU HR expansion required')
    cams = storage.get('cameras', {})
    if set(cams) != {f'cam{c:02d}' for c in range(1, 33)}:
        raise ValueError('HR storage camera population differs')
    for camera, row in cams.items():
        if row['frames'] != 300 or set(row['hr_sha256']) != {str(f) for f in range(300)}:
            raise ValueError('HR storage frame population differs')
        if any(row['hr_sha256'][str(f)] != files[f'hr/{camera}/{f:04d}.png']['sha256'] for f in range(300)):
            raise ValueError('Original HR storage hashes differ from transfer inventory')
    return dict(files=len(files), bytes=sum(v['bytes'] for v in files.values()), sha256=hashlib.sha256(json.dumps(files, sort_keys=True, separators=(',', ':')).encode()).hexdigest(), hash_verify_seconds=time.monotonic()-started)


def service_identity(state, source, target, proc_root=Path('/proc')):
    fields = {}
    for line in state['stdout'].splitlines():
        key, sep, value = line.partition('=')
        if not sep or key in fields:
            raise ValueError('Malformed or duplicated user service state')
        fields[key] = value
    if state['returncode'] or fields.get('ActiveState') != 'active' or fields.get('SubState') != 'running' or not re.fullmatch('[1-9][0-9]*', fields.get('MainPID', '')):
        raise ValueError('Persistent user mount service is not running')
    pid = int(fields['MainPID'])
    proc = Path(proc_root) / str(pid)
    if proc.stat().st_uid != os.getuid():
        raise ValueError('Mount service PID belongs to another UID')
    stat_before = (proc / 'stat').read_text().rsplit(')', 1)[1].split()[19]
    argv = [os.fsdecode(value) for value in (proc / 'cmdline').read_bytes().rstrip(b'\0').split(b'\0')]
    expected = [str(BINARY), '--no-allow-other', '-r', '-f', '-o', 'fsname=' + str(source) + ',subtype=bindfs', str(source), str(target)]
    if argv != expected:
        raise ValueError('Unknown or changed mount service process arguments')
    exe = proc / 'exe'
    if exe.resolve() != BINARY.resolve() or digest(exe) != BINARY_SHA or digest(BINARY) != BINARY_SHA:
        raise ValueError('Running mount executable identity differs')
    if (proc / 'stat').read_text().rsplit(')', 1)[1].split()[19] != stat_before:
        raise ValueError('Mount service PID was reused during identity check')
    return dict(pid=pid, uid=os.getuid(), start_ticks=int(stat_before), argv=argv, executable=str(exe.resolve()), executable_sha256=BINARY_SHA, systemd=state)


def readonly_mount(target, source):
    rows = mountinfo(target)
    if len(rows) != 1 or rows[0]['fstype'] != 'fuse.bindfs' or rows[0]['source'] != str(source):
        raise ValueError('Exact owned bindfs mapping required')
    if 'ro' not in rows[0]['options'] or 'rw' in rows[0]['options'] or any(o in ('allow_other', 'allow_root') for o in rows[0]['options'] + rows[0]['super_options']):
        raise ValueError('Kernel read-only, current-user-only mount required')
    if 'user_id=' + str(os.getuid()) not in rows[0]['super_options']:
        raise ValueError('FUSE mount owner UID differs')
    return rows[0]


def apply(mode, run_id, full_files):
    files = validate_inventory(full_files)
    source, target, out, drive, identity = context(run_id)
    unit = 'highres-hr-' + run_id + '.service'
    if mode == 'prepare':
        if source.parent.exists() or out.exists():
            raise FileExistsError('Preserve existing storage run; no automatic retry')
        if mountinfo(target) or (target.exists() and (not target.is_dir() or any(target.iterdir()))):
            raise FileExistsError('Canonical HR target is occupied; never overwrite it')
        if target.exists():
            owned(target)
        disk = guard(DRIVE, sum(v['bytes'] for v in files.values()))
        guard(ROOT, 10**6)
        EXTERNAL.mkdir(parents=True, exist_ok=True)
        owned(EXTERNAL)
        source.mkdir(parents=True, exist_ok=False)
        out.mkdir(parents=True, exist_ok=False)
        target.mkdir(parents=True, exist_ok=True)
        owned(target)
        return dict(status='new_owned_HR_stage_accepted', source=str(source), target=str(target), unit=unit, files=list(files), disk=disk, parent_drive=drive, package_binary=identity)
    if mode not in ('verify', 'mount', 'mount-check'):
        raise ValueError('Unknown operation')
    for p in (source, out, target):
        owned(p)
    verified = verify_tree(source, files)
    disk = guard(DRIVE, 0)
    if mode == 'verify':
        return dict(status='external_HR_all_SHA_verified', source=str(source), target=str(target), verification=verified, disk=disk, package_binary=identity)
    if mode == 'mount':
        if mountinfo(target) or any(target.iterdir()):
            raise FileExistsError('Mount target already in use; never remount/restart')
        inspect = capture(['systemctl', '--user', 'show', unit, '--property=LoadState', '--value'])
        if inspect['stdout'].strip() not in ('not-found', ''):
            raise FileExistsError('Existing service name protected')
        receipt = out / 'mount_started.json'
        if receipt.exists():
            raise FileExistsError('Preserve existing mount attempt')
        log = out / 'bindfs.service.log'
        argv = ['systemd-run', '--user', '--unit=' + unit, '--property=Type=exec', '--property=Restart=no',
                '--property=StandardOutput=append:' + str(log), '--property=StandardError=append:' + str(log),
                '--', str(BINARY), '--no-allow-other', '-r', '-f', '-o', 'fsname=' + str(source) + ',subtype=bindfs', str(source), str(target)]
        receipt.write_text(json.dumps(dict(status='mount_dispatch_attempted_no_retry', argv=argv, package_binary=identity), indent=2) + '\n')
        started = capture(argv)
        if started['returncode']:
            raise RuntimeError('Persistent user bindfs service failed; inspect preserved logs: ' + started['stderr'])
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline and not mountinfo(target):
            time.sleep(.1)
        mapping = readonly_mount(target, source)
    else:
        mapping = readonly_mount(target, source)
    canonical = verify_tree(target, files)
    state = capture(['systemctl', '--user', 'show', unit, '--property=ActiveState', '--property=SubState', '--property=MainPID', '--property=ExecStart', '--property=Result'])
    identity_pid = service_identity(state, source, target)
    result = dict(status='complete_readonly_HR_canonical_mapping', source=str(source), target=str(target), unit=unit,
                  verification=verified, canonical_verification=canonical, mountinfo=mapping, service=identity_pid,
                  disk=disk, package_binary=identity, parent_drive=drive, training=False, optimizer_calls=0,
                  input_record_rewrites=0, automatic_retry=False)
    if mode == 'mount':
        with (out / 'complete.json').open('x') as f:
            json.dump(result, f, indent=2); f.write('\n')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['prepare', 'verify', 'mount', 'mount-check'], required=True)
    p.add_argument('--run-id', required=True)
    args = p.parse_args()
    payload = json.load(sys.stdin)
    try:
        print(json.dumps(apply(args.mode, args.run_id, payload['files'])))
    except BaseException:
        try:
            _, _, out = paths(args.run_id)
            if out.exists():
                no_symlinks(out); owned(out)
                with (out / ('failure_' + str(time.time_ns()) + '.json')).open('x') as f:
                    json.dump(dict(status='failed_preserved_no_retry', mode=args.mode, error=traceback.format_exc(), training=False), f, indent=2)
        finally:
            raise


if __name__ == '__main__':
    main()
