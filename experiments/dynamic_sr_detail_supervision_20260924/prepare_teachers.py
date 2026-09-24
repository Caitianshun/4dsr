#!/usr/bin/env python3
"""Extend legal training LR SwinIR caches with the exact frozen old generator.

Does not open HR, change old cache metadata, or schedule any training. The only
permitted accelerator is the specifically assigned local RTX 3090 UUID.
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import select
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
GPU_UUID = 'GPU-c40035c3-0f06-e88b-73f5-fa40d62ec4ec'
LEGACY_SHA = 'd7be23d15473082c57e1e4091bfea9eaafad0de97ef7379db39308ae14c6cbcc'
LEGACY = ROOT / 'output/dynamic_sr_20260918/cook_spinach_pilot_v1_lr_integrated/source_snapshot/06_generate_prior.py'
SCENES = [
    ROOT / 'data/dynamic_sr/n3dv_prepared/cook_spinach/manifest.json',
    ROOT / 'data/dynamic_sr/meetroom_prepared/discussion/manifest.json',
]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)


def guard(out, label):
    gpu = subprocess.check_output([
        'nvidia-smi', f'--id={GPU_UUID}',
        '--query-gpu=uuid,name,memory.used,memory.free,utilization.gpu',
        '--format=csv,noheader,nounits'], text=True).strip()
    processes = subprocess.check_output([
        'nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name,used_memory',
        '--format=csv,noheader,nounits'], text=True).strip()
    fields = [x.strip() for x in gpu.split(',')]
    ours = [line for line in processes.splitlines() if line.startswith(GPU_UUID)]
    record = {'utc': now(), 'label': label, 'gpu': gpu, 'compute_processes': ours}
    with (out / 'gpu_checks.jsonl').open('a', buffering=1) as f:
        f.write(json.dumps(record) + '\n')
    if fields[0] != GPU_UUID or '3090' not in fields[1]:
        raise RuntimeError(f'Assigned GPU identity mismatch: {gpu}')
    if ours or float(fields[2]) > 256 or float(fields[4]) > 10:
        raise RuntimeError(f'Assigned GPU busy; no fallback GPU permitted: {record}')
    return record


def inventory(manifest):
    m = json.loads(manifest.read_text())
    train = m['splits']['train']
    forbidden = {'cam00', 'cam01'} | set(m['splits'].get('dev', [])) | set(m['splits'].get('test', []))
    assert len(train) == len(set(train)) and not set(train) & forbidden
    rows = sorted([x for x in m['observations'] if x['split'] == 'train'],
                  key=lambda x: (x['camera_id'], int(x['frame_index'])))
    assert {x['camera_id'] for x in rows} == set(train)
    frame_sets = {c: sorted(int(x['frame_index']) for x in rows if x['camera_id'] == c) for c in train}
    assert all(len(v) == 60 and len(set(v)) == 60 for v in frame_sets.values())
    assert all(v == next(iter(frame_sets.values())) for v in frame_sets.values())
    old_config_path = manifest.parent / 'sr_swinir_x4/prior_config.json'
    cfg = json.loads(old_config_path.read_text())
    assert cfg['manifest_sha256'] == sha(manifest)
    assert cfg['generator_sha256'] == LEGACY_SHA
    assert cfg['scale'] == 4 and cfg['window'] == 8 and cfg['tile'] == 0 and cfg['overlap'] == 32
    assert cfg['precision'] == 'float32' and cfg['padding'] == 'official_mirror_concat_next_window'
    assert sha(cfg['checkpoint']) == cfg['checkpoint_sha256']
    network = Path('/home/cai_tianshun/Project/mml/scripts/network_swinir.py')
    assert sha(network) == cfg['network_sha256']
    entries = []
    preserved = {str(old_config_path): sha(old_config_path)}
    for row in rows:
        camera, frame = row['camera_id'], int(row['frame_index'])
        source = (manifest.parent / row['lr_path']).resolve()
        assert source.is_relative_to(manifest.parent.resolve())
        assert Path(row['lr_path']).parts[0] in ('lr', 'lr_x4')
        assert Path(row['lr_path']).parts[1] == camera
        input_sha = sha(source)
        assert input_sha == row['lr_sha256'], f'Changed manifest LR input: {source}'
        target = manifest.parent / 'sr_swinir_x4' / camera / source.name
        receipt = target.with_suffix('.json')
        assert target.exists() == receipt.exists(), f'Incomplete output/receipt pair: {target}'
        hit = target.exists()
        output_sha = receipt_sha = None
        if hit:
            r = json.loads(receipt.read_text())
            output_sha, receipt_sha = sha(target), sha(receipt)
            assert r['input_sha256'] == input_sha and r['output_sha256'] == output_sha
            preserved[str(target)] = output_sha
            preserved[str(receipt)] = receipt_sha
        entries.append({'camera': camera, 'frame': frame, 'path': str(target),
                        'relative_path': str(target.relative_to(manifest.parent)),
                        'sha256': output_sha, 'lr_sha256': input_sha,
                        'lr_path': str(source), 'lr_relative_path': row['lr_path'],
                        'receipt_path': str(receipt), 'receipt_sha256': receipt_sha,
                        'cache_hit': hit})
    assert len({(e['camera'], e['frame']) for e in entries}) == len(entries)
    result = {'schema': 1, 'status': 'preparing', 'scene': m['scene'],
              'manifest': str(manifest), 'manifest_sha256': sha(manifest),
              'prior_subdir': 'sr_swinir_x4', 'train_cameras': train,
              'frame_indices': next(iter(frame_sets.values())),
              'input_policy': 'Manifest training LR only; cam00/cam01 excluded; no HR image reads',
              'teacher_config': cfg, 'teacher_config_path': str(old_config_path),
              'teacher_config_sha256': sha(old_config_path),
              'cache_hits_initial': sum(e['cache_hit'] for e in entries),
              'missing_initial': sum(not e['cache_hit'] for e in entries),
              'entries': entries, 'created_utc': now()}
    return result, preserved


def run_generator(args, out, frozen, index, label, extra):
    # nvidia-smi utilization integrates a preceding sampling interval. Let our
    # just-exited generator's interval expire, then perform the same busy check.
    time.sleep(2)
    guard(out, f"{index['scene']}_{label}_immediate_prelaunch")
    cmd = [sys.executable, str(frozen), '--manifest', index['manifest'],
           '--checkpoint', index['teacher_config']['checkpoint'],
           '--network', '/home/cai_tianshun/Project/mml/scripts/network_swinir.py',
           '--dependency-path', str(ROOT / 'experiments/dynamic_sr_20260918/vendor'),
           '--device', 'cuda:0', '--tile', '0', '--overlap', '32', '--log-every', '60', *extra]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=GPU_UUID, PYTHONUNBUFFERED='1')
    scene_out = out / index['scene']
    stage = {'command': cmd, 'cuda_visible_devices': GPU_UUID, 'started_utc': now()}
    save(scene_out / f'{label}_status.json', stage)
    t0 = time.monotonic()
    with (scene_out / f'{label}.log').open('x', buffering=1) as log:
        process = subprocess.Popen(cmd, env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        stage['pid'] = process.pid
        save(scene_out / f'{label}_status.json', stage)
        rc = process.wait()
    stage.update(returncode=rc, seconds=time.monotonic() - t0, completed_utc=now())
    save(scene_out / f'{label}_status.json', stage)
    if rc:
        raise RuntimeError(f'Original generator failed: {index["scene"]} {label} rc={rc}; see {scene_out / (label + ".log")}')
    return stage


def adopt_cook(args):
    """CPU event handoff after ownership of discussion moved to another host.

    The stopped original dispatcher cannot launch discussion. Its existing cook
    child is allowed to finish uninterrupted; pidfd observes the actual exit.
    """
    out = args.out.resolve()
    scene_out = out / 'cook_spinach'
    pid = args.adopt_running_cook
    parent_pid = args.stopped_dispatcher_pid
    assert hasattr(os, 'pidfd_open'), 'Use /usr/bin/python3 for this CPU-only handoff'
    cmdline = Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0', b' ').decode()
    assert 'generate_prior_original.py' in cmdline and 'cook_spinach/manifest.json' in cmdline
    index = json.loads((scene_out / 'teacher_index.json').read_text())
    preserved = json.loads((scene_out / 'preserved_files_before.json').read_text())
    status = json.loads((out / 'status.json').read_text())
    status['handoff'] = {'utc': now(), 'cook_generator_pid': pid,
                         'stopped_dispatcher_pid': parent_pid,
                         'reason': 'Discussion teachers exclusively assigned to A100 GPU0; cook child not interrupted',
                         'finalizer_pid': os.getpid(), 'finalizer_source_sha256': sha(__file__),
                         'wait': 'Linux pidfd exit event using system Python'}
    status['scenes']['meetroom_discussion']['status'] = 'delegated_to_a100'
    save(out / 'status.json', status)
    exited = False
    try:
        fd = os.pidfd_open(pid)
        try:
            select.select([fd], [], [])
        finally:
            os.close(fd)
        exited = True
        # Our stopped dispatcher retains the zombie, so Linux exposes the real
        # wait status in /proc/PID/stat field 52 until we clean up the dispatcher.
        procstat = Path(f'/proc/{pid}/stat').read_text()
        fields = procstat[procstat.rfind(')') + 2:].split()
        assert fields[0] == 'Z'
        returncode = os.waitstatus_to_exitcode(int(fields[49]))
        stage_path = scene_out / 'all_train_status.json'
        stage = json.loads(stage_path.read_text())
        stage.update(returncode=returncode, completed_utc=now(),
                     seconds=(dt.datetime.now(dt.timezone.utc) - dt.datetime.fromisoformat(stage['started_utc'])).total_seconds(),
                     completion_detection='pidfd exit; /proc zombie wait status; dispatcher handoff')
        save(stage_path, stage)
        assert returncode == 0, f'Original cook generator failed with rc={returncode}'
        summary = json.loads((scene_out / 'all_train.log').read_text().strip().splitlines()[-1])
        assert summary['scene'] == 'cook_spinach' and summary['selection_complete']
        assert summary['selected'] == len(index['entries'])
        generated_seconds = 0.
        generated_bytes = existing_bytes = 0
        for e in index['entries']:
            target, receipt, source = Path(e['path']), Path(e['receipt_path']), Path(e['lr_path'])
            r = json.loads(receipt.read_text())
            assert sha(source) == e['lr_sha256'] == r['input_sha256']
            e['sha256'], e['receipt_sha256'] = sha(target), sha(receipt)
            assert e['sha256'] == r['output_sha256']
            assert r['output_hw'] == [v * 4 for v in r['input_hw']]
            e['bytes'] = target.stat().st_size
            if e['cache_hit']:
                existing_bytes += e['bytes']
            else:
                generated_bytes += e['bytes']
                generated_seconds += r['seconds']
        for name, expected in preserved.items():
            assert sha(name) == expected, f'Previously existing file changed: {name}'
        assert sha(index['manifest']) == index['manifest_sha256']
        index.update(status='completed_teacher_inventory', completed_utc=now(),
                     generated_count=index['missing_initial'], old_file_identity_preserved=True,
                     preserved_file_count=len(preserved),
                     costs={'stages': {'benchmark': json.loads((scene_out / 'benchmark_cost.json').read_text()),
                                       'full_generation': stage},
                            'new_image_generation_seconds': generated_seconds,
                            'new_png_bytes': generated_bytes, 'existing_png_bytes': existing_bytes},
                     gpu_uuid=GPU_UUID, gpu_name='NVIDIA GeForce RTX 3090',
                     completion_handoff=status['handoff'])
        save(scene_out / 'teacher_index.json', index)
        status['status'] = 'cook_completed_discussion_delegated'
        status['scenes']['cook_spinach'] = {'status': 'completed_teacher_inventory',
            'teacher_index': str(scene_out / 'teacher_index.json'),
            'teacher_index_sha256': sha(scene_out / 'teacher_index.json'),
            'cache_hits': index['cache_hits_initial'], 'new': index['missing_initial']}
        save(out / 'status.json', status)
        print(json.dumps({'event': 'scene_complete', **status['scenes']['cook_spinach']}), flush=True)
    except BaseException as exc:
        status.update(status='cook_handoff_failed', error=repr(exc), traceback=traceback.format_exc())
        save(out / 'status.json', status)
        raise
    finally:
        if exited:
            # The child has already exited. This only removes the intentionally
            # stopped obsolete dispatcher; it cannot terminate GPU generation.
            os.kill(parent_pid, signal.SIGTERM)
            os.kill(parent_pid, signal.SIGCONT)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--benchmark-frames', type=int, default=4)
    p.add_argument('--idle-interval-seconds', type=float, default=15)
    p.add_argument('--inventory-only', action='store_true')
    p.add_argument('--resume', action='store_true', help='Resume failed wrapper; preserve its failure record and initial cache identity')
    p.add_argument('--adopt-running-cook', type=int, help='CPU-only pidfd completion for existing cook generator PID')
    p.add_argument('--stopped-dispatcher-pid', type=int)
    args = p.parse_args()
    if args.adopt_running_cook:
        assert args.stopped_dispatcher_pid
        adopt_cook(args)
        return
    assert 1 <= args.benchmark_frames <= 8
    assert args.idle_interval_seconds >= 10
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=args.resume)
    if args.resume:
        previous = json.loads((out / 'status.json').read_text())
        assert previous['status'] == 'failed'
        save(out / f'failed_attempt_{time.time_ns()}.json', previous)
    status = {'schema': 1, 'status': 'preparing', 'started_utc': now(), 'pid': os.getpid(),
              'host': os.uname().nodename, 'gpu_uuid': GPU_UUID, 'scenes': {}}
    save(out / 'status.json', status)
    started = time.monotonic()
    lock = None
    try:
        assert sha(LEGACY) == LEGACY_SHA
        source_out = out / 'source_snapshot'
        source_out.mkdir(exist_ok=args.resume)
        frozen = source_out / 'generate_prior_original.py'
        if frozen.exists():
            assert sha(frozen) == LEGACY_SHA
        else:
            shutil.copyfile(LEGACY, frozen)
        driver_snapshot = source_out / ('prepare_teachers.py' if not args.resume else f'prepare_teachers_resume_{time.time_ns()}.py')
        shutil.copyfile(__file__, driver_snapshot)
        status['sources'] = {str(frozen): sha(frozen), str(driver_snapshot): sha(__file__)}
        planned = []
        for manifest in SCENES:
            index, preserved = inventory(manifest)
            if args.resume:
                scene_out = out / index['scene']
                fresh = index
                index = json.loads((scene_out / 'teacher_index.json').read_text())
                preserved = json.loads((scene_out / 'preserved_files_before.json').read_text())
                assert index['manifest_sha256'] == fresh['manifest_sha256']
                assert len(index['entries']) == len(fresh['entries'])
                for name, expected in preserved.items():
                    assert sha(name) == expected
            index['preparation_script_sha256'] = sha(__file__)
            index['original_generator_path'] = str(LEGACY)
            index['executed_generator_path'] = str(frozen)
            index['original_generator_sha256'] = LEGACY_SHA
            scene_out = out / index['scene']
            save(scene_out / 'teacher_index.json', index)
            if not args.resume:
                save(scene_out / 'preserved_files_before.json', preserved)
            planned.append((index, preserved))
            status['scenes'][index['scene']] = {'status': 'inventoried', 'cache_hits': index['cache_hits_initial'], 'missing': index['missing_initial']}
        save(out / 'status.json', status)
        if args.inventory_only:
            status.update(status='inventory_only_complete', seconds=time.monotonic() - started)
            save(out / 'status.json', status)
            print(json.dumps(status), flush=True)
            return
        lock = open(f'/tmp/4dsr_soft_gpu_{GPU_UUID}.lock', 'a')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        guard(out, 'idle_check_1')
        time.sleep(args.idle_interval_seconds)
        guard(out, 'idle_check_2')
        for index, preserved in planned:
            scene = index['scene']
            scene_out = out / scene
            missing = [e for e in index['entries'] if not e['cache_hit']]
            costs = {}
            if missing:
                first_camera = missing[0]['camera']
                assert all(not e['cache_hit'] for e in index['entries'] if e['camera'] == first_camera)
                if args.resume and (scene_out / 'benchmark_cost.json').exists():
                    costs['benchmark'] = json.loads((scene_out / 'benchmark_cost.json').read_text())
                    assert costs['benchmark']['returncode'] == 0
                else:
                    costs['benchmark'] = run_generator(args, out, frozen, index, 'benchmark',
                                                        ['--cameras', first_camera, '--limit', str(args.benchmark_frames)])
                    bench_rows = [e for e in missing if e['camera'] == first_camera][:args.benchmark_frames]
                    costs['benchmark']['image_generation_seconds'] = [json.loads(Path(e['receipt_path']).read_text())['seconds'] for e in bench_rows]
                    mean = sum(costs['benchmark']['image_generation_seconds']) / len(bench_rows)
                    costs['benchmark']['estimated_remaining_image_seconds'] = mean * (len(missing) - len(bench_rows))
                    save(scene_out / 'benchmark_cost.json', costs['benchmark'])
                    print(json.dumps({'event': 'benchmark_complete', 'scene': scene, 'new_frames': len(bench_rows), 'mean_image_seconds': mean,
                                      'remaining_estimated_seconds': costs['benchmark']['estimated_remaining_image_seconds']}), flush=True)
                costs['full_generation'] = run_generator(args, out, frozen, index, 'all_train', [])
            generated_seconds = 0.
            generated_bytes = existing_bytes = 0
            for e in index['entries']:
                target, receipt, source = Path(e['path']), Path(e['receipt_path']), Path(e['lr_path'])
                r = json.loads(receipt.read_text())
                assert sha(source) == e['lr_sha256'] == r['input_sha256']
                e['sha256'], e['receipt_sha256'] = sha(target), sha(receipt)
                assert e['sha256'] == r['output_sha256']
                assert r['output_hw'] == [v * 4 for v in r['input_hw']]
                e['bytes'] = target.stat().st_size
                if e['cache_hit']:
                    existing_bytes += e['bytes']
                else:
                    generated_bytes += e['bytes']
                    generated_seconds += r['seconds']
            for name, expected in preserved.items():
                assert sha(name) == expected, f'Previously existing file changed: {name}'
            assert sha(index['manifest']) == index['manifest_sha256']
            index.update(status='completed_teacher_inventory', completed_utc=now(), generated_count=len(missing),
                         old_file_identity_preserved=True, preserved_file_count=len(preserved),
                         costs={'stages': costs, 'new_image_generation_seconds': generated_seconds,
                                'new_png_bytes': generated_bytes, 'existing_png_bytes': existing_bytes},
                         gpu_uuid=GPU_UUID, gpu_name='NVIDIA GeForce RTX 3090')
            save(scene_out / 'teacher_index.json', index)
            status['scenes'][scene] = {'status': 'complete', 'teacher_index': str(scene_out / 'teacher_index.json'),
                                       'teacher_index_sha256': sha(scene_out / 'teacher_index.json'),
                                       'cache_hits': index['cache_hits_initial'], 'new': len(missing)}
            save(out / 'status.json', status)
            print(json.dumps({'event': 'scene_complete', 'scene': scene, **status['scenes'][scene]}), flush=True)
        status.update(status='complete', completed_utc=now(), seconds=time.monotonic() - started)
        save(out / 'status.json', status)
        print(json.dumps(status), flush=True)
    except BaseException as exc:
        status.update(status='failed', failed_utc=now(), seconds=time.monotonic() - started,
                      error=repr(exc), traceback=traceback.format_exc())
        save(out / 'status.json', status)
        raise
    finally:
        if lock:
            lock.close()


if __name__ == '__main__':
    main()
