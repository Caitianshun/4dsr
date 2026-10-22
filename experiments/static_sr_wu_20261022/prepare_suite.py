"""Register the fixed six-scene area4 suite and prepare CPU-only ALS plans.

Registration hashes every training LR image. Existing directories must match
exactly and are never overwritten. Plans contain camera orderings only; no SR
teacher is run. Run `register`, then `plan --workers 4` in a CPU OpenCV environment.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from common import ROOT, METHODS, checked, identity, read, records, resolve, sha, training_manifest, validate_records_bundle, write_new
from prepare import METHOD_SPEC, build

SCENES = (
    ('Cook07', 'cook_spinach'), ('Cut07', 'cut_roasted_beef'),
    ('Coffee07', 'coffee_martini'), ('Flame07', 'flame_steak'),
    ('Discussion07', 'meetroom_discussion'), ('VRHeadset07', 'meetroom_vrheadset'),
)
DEFAULT_PREPARED = ROOT / 'output/static_sr_wu_20261022/prepared_final'
DEFAULT_PLANS = ROOT / 'output/static_sr_wu_20261022/cpu_prior_plans'
DEFAULT_UPSTREAM = ROOT / 'tmp/static_sr_wu_20261022/upstream/sequence_matters'
CPU_ENV = {'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
           'MKL_NUM_THREADS': '1', 'NUMEXPR_NUM_THREADS': '1', 'PYTHONDONTWRITEBYTECODE': '1'}


def scene_paths(short):
    scene = dict(SCENES)[short]
    return (ROOT / f'data/dynamic_sr/full_time_20261007/prepared/{scene}_area4/manifest_train_ready_seed20261007.json',
            ROOT / f'output/dynamic_sr_physlr_20261016/registry/{scene}_area4/seed_20261007/swinir_teacher/Bsync/config.json')


def validate_scope(short, manifest):
    if short not in dict(SCENES) or manifest['scene'] != dict(SCENES)[short]:
        raise ValueError('Scene is outside the fixed six-scene suite or its identity differs')
    if manifest['frame_indices'] != list(range(300)):
        raise ValueError('The suite requires all original frames 0 through 299')


def registration_payloads(short, manifest, source):
    m, rows = training_manifest(manifest, verify_pixels=True)
    validate_scope(short, m)
    payloads = {}
    for method in METHODS:
        cfg = build(manifest, source, method, verify_pixels=False)
        if method == 'splatsure':
            cfg['label'] += ' / stablesr'
        payloads[method + '.json'] = cfg
    payloads['training_records.json'] = dict(schema='static_sr_wu_train_records_v1', manifest=identity(manifest),
        records=records(m, rows), heldout_cameras=m['splits']['test'])
    payloads['registration.json'] = dict(status='registered_CPU_only', manifest=identity(manifest),
        training_observations=len(rows), LR_bytes_verified=True, HR_images_read=0, heldout_images_read=0,
        GPU_calls=0, parameter_updates=0,
        requirements_before_training={key: spec['dependencies'] for key, spec in METHOD_SPEC.items()})
    return m, rows, payloads


def materialize_registration(out, payloads):
    """Reject partial/stale directories before writing anything; preserve reused bytes."""
    out = Path(out)
    if out.exists():
        if not out.is_dir():
            raise FileExistsError(f'Registration destination is not a directory: {out}')
        mismatches = [name for name, expected in payloads.items()
                      if not (out / name).is_file() or read(out / name) != expected]
        if mismatches:
            raise ValueError(f'Existing registration differs; preserved without changes: {out}: {mismatches}')
        return 'reused_verified'
    out.mkdir(parents=True, exist_ok=False)
    for name, value in payloads.items():
        write_new(out / name, value)
    return 'created'


def dependency_paths(short, prepared, plans):
    result = {}
    for method in METHODS:
        upstream = ROOT / 'tmp/static_sr_wu_20261022/upstream' / method
        producer = ROOT / 'experiments/static_sr_wu_20261022/methods' / (method + '.py')
        result[method] = dict(configuration=str(prepared / short / (method + '.json')),
            training_records=str(prepared / short / 'training_records.json'),
            producer=identity(producer), upstream=str(upstream), upstream_present=upstream.is_dir(),
            requirements_before_training=METHOD_SPEC[method]['dependencies'])
    result['sequence_matters']['sequence_plan_receipt'] = str(plans / (short + '_full300') / 'receipt.json')
    return result


def register_suite(prepared=DEFAULT_PREPARED, plans=DEFAULT_PLANS):
    prepared, plans = resolve(prepared), resolve(plans)
    entries, actions = [], {}
    # Validate every input and every existing registration before adding any scene.
    staged = []
    for short, scene in SCENES:
        manifest, source = scene_paths(short)
        m, rows, payloads = registration_payloads(short, manifest, source)
        out = prepared / short
        if out.exists():
            materialize_registration(out, payloads)
        staged.append((short, scene, manifest, source, m, rows, payloads))
    for short, scene, manifest, source, m, rows, payloads in staged:
        out = prepared / short
        actions[short] = materialize_registration(out, payloads)
        entries.append(dict(short=short, scene=scene, frames=300, frame_indices=list(range(300)),
            train_camera_count=len(m['splits']['train']), train_cameras=m['splits']['train'],
            heldout_cameras=m['splits']['test'], lr_count=len(rows),
            lr_bytes=sum((Path(m['_root']) / row['lr_path']).stat().st_size for row in rows),
            manifest=identity(manifest), source_configuration=identity(source),
            initialization=identity(Path(m['_root']) / m['initialization']['npz_path']),
            initialization_points=m['initialization']['point_count'],
            configurations={method: identity(out / (method + '.json')) for method in METHODS},
            training_records=identity(out / 'training_records.json'), registration=identity(out / 'registration.json'),
            dependency_paths=dependency_paths(short, prepared, plans)))
    index = dict(schema='static_sr_wu_six_scene_suite_v1', status='registered_CPU_only_GPU_acceptance_pending',
        scope='fixed_six_current_scenes_full300_area4', scenes=entries, methods=list(METHODS),
        totals=dict(scenes=6, method_configurations=24, scene_frames=1800,
                    train_camera_count=sum(e['train_camera_count'] for e in entries),
                    lr_observations=sum(e['lr_count'] for e in entries), lr_bytes=sum(e['lr_bytes'] for e in entries)),
        LR_bytes_verified=True, HR_images_read=0, heldout_images_read=0, GPU_calls=0, parameter_updates=0,
        teacher_targets_generated=False)
    path = prepared / 'suite_index.json'
    if path.exists():
        if read(path) != index:
            raise ValueError(f'Suite index differs; existing index preserved: {path}')
    else:
        write_new(path, index)
    return dict(suite_index=identity(path), actions=actions, totals=index['totals'])


def plan_worker(short, prepared=DEFAULT_PREPARED, plans=DEFAULT_PLANS, upstream=DEFAULT_UPSTREAM):
    if 'torch' in sys.modules or any(os.environ.get(k) != v for k, v in CPU_ENV.items()):
        raise RuntimeError('Plan worker requires explicit CPU-only single-thread environment and no torch import')
    import cv2
    from methods.sequence_matters import prepare_sequences, validate_sequence_plan
    cv2.setNumThreads(1)
    prepared, plans, upstream = resolve(prepared), resolve(plans), resolve(upstream)
    manifest_path, _ = scene_paths(short)
    m, rows = training_manifest(manifest_path, verify_pixels=False)
    validate_scope(short, m)
    record_file = prepared / short / 'training_records.json'
    bundle = read(record_file)
    validate_records_bundle(bundle, identity(manifest_path), m, rows)
    output = plans / (short + '_full300')
    summary_path, receipt_path = output / 'cpu_preparation_summary.json', output / 'receipt.json'
    source = upstream / 'vsr/utils_vsr.py'
    producer_path = ROOT / 'experiments/static_sr_wu_20261022/methods/sequence_matters.py'
    producer_before = sha(producer_path)
    begin, started = time.perf_counter(), datetime.now(timezone.utc).isoformat()
    if output.exists():
        if not summary_path.is_file() or not receipt_path.is_file():
            raise ValueError(f'Incomplete existing plan preserved: {output}')
        receipt, summary = read(receipt_path), read(summary_path)
        validate_sequence_plan(receipt, bundle, source)
        if (summary.get('scene') != short or summary.get('observations') != len(rows)
                or summary.get('synchronous_groups') != 300 or summary.get('teacher_generated') is not False
                or summary.get('torch_imported') is not False or summary.get('targets') != 0
                or summary.get('receipt', {}).get('sha256') != sha(receipt_path)):
            raise ValueError(f'Existing plan summary differs: {summary_path}')
        action = 'reused_verified'
    else:
        receipt = prepare_sequences(bundle, upstream, output, plan_only=True, device='cpu')
        valid_start = time.perf_counter()
        validate_sequence_plan(receipt, bundle, source)
        validation_elapsed = time.perf_counter() - valid_start
        if 'torch' in sys.modules or sha(producer_path) != producer_before:
            raise RuntimeError('Torch imported or producer changed while the CPU plan was running')
        summary = dict(status='cpu_sequence_plan_complete_teacher_not_generated', scene=short,
            scene_identity=dict(SCENES)[short], started_utc=started, finished_utc=datetime.now(timezone.utc).isoformat(),
            elapsed_seconds=time.perf_counter()-begin, observations=len(rows), synchronous_groups=len(receipt['groups']),
            als_sequences=sum(len(g['sequences']) for g in receipt['groups']), teacher_generated=False, targets=0,
            torch_imported=False, cuda_visible_devices='', cv2_threads=cv2.getNumThreads(),
            worker_environment={k:os.environ[k] for k in CPU_ENV}, heldout_cameras=bundle['heldout_cameras'],
            heldout_or_hr_inputs=0, manifest=identity(record_file), receipt=identity(receipt_path),
            producer_sha256=producer_before, producer_source_sha256_observed_at_end=sha(producer_path),
            all_training_keys_covered_exactly_once=True,
            cache_consumer_validation=dict(status='passed', elapsed_seconds=validation_elapsed,
                consumer_sha256=producer_before, torch_imported=False))
        write_new(summary_path, summary)
        action = 'created'
    if 'torch' in sys.modules or cv2.getNumThreads() != 1:
        raise RuntimeError('CPU-only worker invariant changed')
    return dict(scene=short, action=action, observations=len(rows), groups=300,
        als_sequences=summary['als_sequences'], preparation_elapsed_seconds=summary['elapsed_seconds'],
        worker_elapsed_seconds=time.perf_counter()-begin, receipt=identity(receipt_path), summary=identity(summary_path),
        validated_against=identity(record_file), cache_consumer_validation='passed', torch_imported=False)


def plan_suite(prepared=DEFAULT_PREPARED, plans=DEFAULT_PLANS, upstream=DEFAULT_UPSTREAM, workers=4):
    if workers not in (2, 3, 4):
        raise ValueError('Use 2, 3, or 4 independent single-thread CPU workers')
    prepared, plans, upstream = resolve(prepared), resolve(plans), resolve(upstream)
    suite = read(prepared / 'suite_index.json')
    if [(e['short'], e['scene']) for e in suite['scenes']] != list(SCENES):
        raise ValueError('Registered suite scene list differs')
    for entry in suite['scenes']:
        for key in ('manifest', 'source_configuration', 'initialization', 'training_records', 'registration'):
            checked(entry[key])
        for config in entry['configurations'].values():
            checked(config)
    begin, started = time.perf_counter(), datetime.now(timezone.utc).isoformat()
    def launch(short):
        command = [sys.executable, str(Path(__file__).resolve()), 'plan-worker', '--scene', short,
                   '--prepared', str(prepared), '--plans', str(plans), '--upstream', str(upstream)]
        process = subprocess.run(command, env={**os.environ, **CPU_ENV}, capture_output=True, text=True)
        if process.returncode:
            raise RuntimeError(f'{short} CPU plan failed: {process.stdout}\n{process.stderr}')
        return json.loads(process.stdout)
    results = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(launch, short):short for short, _ in SCENES}
        for future in as_completed(futures):
            result = future.result()
            results[result['scene']] = result
            print(json.dumps(dict(event='scene_plan_complete', **result)), flush=True)
    summary = dict(schema='static_sr_wu_six_scene_cpu_plans_v1', status='CPU_plan_complete_teacher_not_generated',
        started_utc=started, finished_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.perf_counter()-begin,
        workers=workers, threads_per_worker=1, environment=CPU_ENV,
        suite_index=identity(prepared / 'suite_index.json'), scenes=[results[short] for short, _ in SCENES],
        totals=dict(scenes=6, synchronous_groups=sum(r['groups'] for r in results.values()),
            lr_observations=sum(r['observations'] for r in results.values()),
            als_sequences=sum(r['als_sequences'] for r in results.values())),
        teacher_targets_generated=False, torch_imported=False, GPU_calls=0, parameter_updates=0,
        heldout_or_hr_inputs=0)
    destination = plans / 'six_scene_suite_summary.json'
    # Keep each completed run immutable; reruns validate the existing receipt.
    if destination.exists():
        previous = read(destination)
        if (previous['suite_index'] != summary['suite_index'] or previous['totals'] != summary['totals']
                or [(r['scene'],r['receipt']) for r in previous['scenes']]
                != [(r['scene'],r['receipt']) for r in summary['scenes']]):
            raise ValueError('Existing plan suite summary differs; preserved without changes')
    else:
        write_new(destination, summary)
    return dict(summary=identity(destination), totals=summary['totals'], elapsed_seconds=summary['elapsed_seconds'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['register','plan','plan-worker'])
    parser.add_argument('--prepared', type=Path, default=DEFAULT_PREPARED)
    parser.add_argument('--plans', type=Path, default=DEFAULT_PLANS)
    parser.add_argument('--upstream', type=Path, default=DEFAULT_UPSTREAM)
    parser.add_argument('--workers', type=int, choices=[2,3,4], default=4)
    parser.add_argument('--scene', choices=[short for short,_ in SCENES])
    args = parser.parse_args()
    begin = time.perf_counter()
    if args.command == 'register':
        result = register_suite(args.prepared, args.plans)
    elif args.command == 'plan':
        result = plan_suite(args.prepared, args.plans, args.upstream, args.workers)
    else:
        if args.scene is None:
            parser.error('plan-worker requires --scene')
        result = plan_worker(args.scene, args.prepared, args.plans, args.upstream)
    print(json.dumps(dict(**result, command_elapsed_seconds=time.perf_counter()-begin)), flush=True)


if __name__ == '__main__':
    main()
