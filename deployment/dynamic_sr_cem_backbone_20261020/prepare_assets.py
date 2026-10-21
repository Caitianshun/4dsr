"""Prepare closed CEM area4 input inventories and safely materialize missing files.

Inventory generation uses immutable SHA declarations in the existing area4
manifest and frozen teacher indexes. Materialization never overwrites a target:
only a verified byte-identical existing original-scene HR file may be hardlinked
(or copied across filesystems) into a missing area4 path. Other missing files are
reported for an explicit allowlisted incremental transfer. No GPU calls occur.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import errno
import json
import os
from pathlib import Path
import shutil
import socket
import time

from verify_assets import atomic_json, contained, sha

SCENES = ('cook_spinach', 'cut_roasted_beef')
PREPARED = Path('data/dynamic_sr/full_time_20261007/prepared')


def build(root, scene, target):
    root = root.resolve()
    folder = PREPARED / (scene + '_area4')
    manifest_path = folder / 'manifest_train_ready_seed20261007.json'
    manifest = json.loads((root / manifest_path).read_text())
    manifest_sha = sha(root / manifest_path)
    observations = manifest['observations']
    training = [row for row in observations if row['split'] == 'train']
    evaluation = [row for row in observations if row['camera_id'] == 'cam00']
    assert manifest['frame_indices'] == list(range(300))
    assert training and not any(row['camera_id'] == 'cam00' for row in training)
    assert len(evaluation) == 300 and all(row['split'] == 'test' for row in evaluation)
    assert len(training) in (6000, 5700)
    files = {}

    def add(path, expected, role, camera=None, frame=None, alternatives=()):
        path = Path(path)
        source = contained(root, str(path))
        row = dict(sha256=expected, bytes=source.stat().st_size, role=role, scene=scene)
        if camera is not None:
            row.update(camera=camera, frame=frame)
        if alternatives:
            row['existing_identical_source_candidates'] = list(map(str, alternatives))
        if str(path) in files:
            assert files[str(path)] == row
        files[str(path)] = row

    add(manifest_path, manifest_sha, 'area4_manifest')
    for row in training + evaluation:
        path = folder / row['hr_path']
        alternative = PREPARED / scene / row['hr_path']
        add(path, row['hr_sha256'], 'training_HR_oracle_or_HRref' if row['split'] == 'train' else 'heldout_HR_evaluation_only', row['camera_id'], row['frame_index'], [alternative])
    for row in training:
        if target == 'a100' or row['frame_index'] == 0:
            add(folder / row['lr_path'], row['lr_sha256'], 'training_area4_LR' if target == 'a100' else 'training_area4_LR_frame0_initialization', row['camera_id'], row['frame_index'])
    teacher_provenance = {}
    if target == 'a100':
        lr_by_key = {(row['camera_id'], row['frame_index']): row for row in training}
        for tag, role in [('sr_swinir_x4', 'original_SwinIR'), ('sr_swinir_x4_areaft', 'matched_SwinIR')]:
            index_path = folder / ('teacher_index_' + tag + '_seed20261007.json')
            index = json.loads((root / index_path).read_text())
            assert index['manifest_sha256'] == manifest_sha and index['oracle'] is False
            entries = index['entries']
            assert len(entries) == len(training)
            assert {(entry['camera'], entry['frame']) for entry in entries} == set(lr_by_key)
            add(index_path, sha(root / index_path), role + '_index')
            teacher_provenance[role] = dict(index_path=str(index_path), index_sha256=sha(root / index_path), teacher_config=index['teacher_config'], frozen_cache_reuse=True, weights_required_for_training=False)
            for entry in entries:
                original = lr_by_key[(entry['camera'], entry['frame'])]
                assert entry['lr_sha256'] == original['lr_sha256']
                add(folder / entry['relative_path'], entry['sha256'], role + '_training_cache', entry['camera'], entry['frame'])
    counts = Counter(row['role'] for row in files.values())
    sizes = Counter()
    for row in files.values():
        sizes[row['role']] += row['bytes']
    return dict(schema='CEM_area4_closed_input_inventory_v1', scene=scene, target=target,
                created_utc=datetime.now(timezone.utc).isoformat(),
                scope='A100 area4 native LR parent + original/matched teacher SR arms + HR oracle' if target == 'a100' else 'CTS native HRref training/evaluation HR plus area4 frame0 training LR for independent native geometry',
                source_manifest=dict(path=str(manifest_path), sha256=manifest_sha),
                train_test_boundary=dict(heldout_camera='cam00', training_HR='only explicit areaOracle/native_HRref arms', heldout_HR='evaluation only; excluded from all geometry/teacher/training', initializer='independent native geometry from training camera area4 frame0 LR; old original-LR baseline geometry is not an area4 initializer'),
                frozen_teacher_provenance=teacher_provenance,
                files=dict(sorted(files.items())), file_count=len(files),
                total_bytes=sum(row['bytes'] for row in files.values()),
                role_counts=dict(counts), role_bytes=dict(sizes),
                expected_SHA_source='existing immutable area4 manifest and frozen teacher indexes; source bytes are rehashed by verify_assets.py before use',
                all_local_image_bytes_rehashed=False)


def materialize(root, inventory_path, workers):
    root = root.resolve()
    inventory = json.loads(inventory_path.read_text())
    assert inventory['schema'] == 'CEM_area4_closed_input_inventory_v1'
    began = time.monotonic()

    def one(item):
        relative, expected = item
        destination = contained(root, relative)
        result = dict(path=relative)
        if destination.exists():
            return dict(result, action='preserved_existing_target')
        for relative_source in expected.get('existing_identical_source_candidates', []):
            source = contained(root, relative_source)
            if not source.is_file():
                continue
            size = source.stat().st_size
            digest = sha(source)
            if size != expected['bytes'] or digest != expected['sha256']:
                return dict(result, action='candidate_mismatch_not_used', source=relative_source, bytes=size, sha256=digest)
            destination.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(source, destination)
                action = 'hardlinked_verified_identical_source'
            except FileExistsError:
                return dict(result, action='preserved_racing_existing_target')
            except OSError as error:
                if error.errno != errno.EXDEV:
                    raise
                # Exclusive creation also protects targets created by another process.
                with source.open('rb') as reader, destination.open('xb') as writer:
                    shutil.copyfileobj(reader, writer, 8 << 20)
                action = 'copied_verified_identical_source_cross_filesystem'
            assert destination.stat().st_size == expected['bytes']
            return dict(result, action=action, source=relative_source, sha256=digest, bytes=size,
                        source_inode=source.stat().st_ino, target_inode=destination.stat().st_ino)
        return dict(result, action='missing_requires_allowlisted_transfer')

    with ThreadPoolExecutor(max_workers=workers) as pool:
        records = list(pool.map(one, inventory['files'].items()))
    return dict(schema='CEM_area4_nonoverwriting_materialization_v1', status='completed_materialization_only_not_verification',
                inventory_sha256=sha(inventory_path), root=str(root), host=socket.gethostname(),
                finished_utc=datetime.now(timezone.utc).isoformat(), seconds=time.monotonic()-began,
                actions=dict(Counter(row['action'] for row in records)), records=records,
                GPU_calls=0, overwritten_existing_files=0,
                warning='A complete verify_assets.py byte-SHA receipt is required after materialization/transfer.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    sub = parser.add_subparsers(dest='command', required=True)
    inv = sub.add_parser('inventory')
    inv.add_argument('--scene', choices=SCENES, required=True)
    inv.add_argument('--target', choices=['a100', 'cts'], required=True)
    inv.add_argument('--output', type=Path, required=True)
    mat = sub.add_parser('materialize')
    mat.add_argument('--inventory', type=Path, required=True)
    mat.add_argument('--receipt', type=Path, required=True)
    mat.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    if args.command == 'inventory':
        assert not args.output.exists(), 'Frozen inventories are not overwritten; use a new identity.'
        value = build(args.root, args.scene, args.target)
        atomic_json(args.output, value)
        print(json.dumps(dict(path=str(args.output), sha256=sha(args.output), file_count=value['file_count'], total_bytes=value['total_bytes'], role_counts=value['role_counts'])), flush=True)
    else:
        assert 1 <= args.workers <= 8
        value = materialize(args.root, args.inventory, args.workers)
        atomic_json(args.receipt, value)
        print(json.dumps({key: value[key] for key in ['status', 'inventory_sha256', 'actions', 'seconds']}), flush=True)


if __name__ == '__main__':
    main()
