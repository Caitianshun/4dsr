"""CPU-only SplatSuRe config/bundle adapter for verified existing SwinIR PNGs.

The frozen train_stage.py accepts the resulting bundle without monkeypatching.
The LR checkpoint supplies contribution maps only; final SR stays fresh.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

try:
    from . import shared_cache as shared
except ImportError:
    import shared_cache as shared

ROOT = shared.ROOT
EXP = ROOT / 'experiments/static_sr_wu_20261022'
UPSTREAM = 'df40abb8d92a02656268cb0a0b97acc2495b9307'
RASTERIZER = '83814a5d942991a6a201dfd84f81606bc0382579'
MAP_CONFIG = dict(ratio_threshold=1.1, transition_width=0.05, minimum_views=3,
                  gamma=0.4, lambda_dssim=0.2, lr_operator='bicubic')
LR_IMPORT_SCHEMA = 'shared_splatsure_portable_LR_import_v1'


def _core(config):
    value = copy.deepcopy(config)
    for key in ('teacher_variant', 'label'):
        value.pop(key, None)
    value.get('information', {}).pop('shared_teacher', None)
    return value


def _base(path, project_root):
    config = shared.read(path, project_root)
    if (config.get('schema') != 'static_sr_wu_adaptation_v1'
            or config.get('method') != 'splatsure'
            or config.get('initialization') != 'fresh_scene_local_train_LR_points'
            or config.get('degradation') != 'area4'
            or config.get('training', {}).get('steps') != 30000):
        raise ValueError('Require the formal fresh-SR area4 SplatSuRe registration')
    source = shared.read(shared.checked(config['source_configuration'], project_root), project_root)
    if source['configuration'] != config['wu_configuration']:
        raise ValueError('Wu configuration differs from its registered source')
    return config


def _config_document(base, base_identity, cache_identity, cache):
    result = copy.deepcopy(base)
    result['teacher_variant'] = shared.VARIANT
    result['label'] = 'SplatSuRe on Wu-4DGS / shared area4-finetuned SwinIR RGB8 PNG'
    result['information']['shared_teacher'] = dict(
        cache=cache_identity, source_config=base_identity, tag=cache['tag'],
        decode=cache['decode'], model_identities=cache['model_identities'],
        protocol='Shared external SR images; geometry selection and original losses retained',
        original_author_teacher_reproduction=False)
    if _core(result) != _core(base):
        raise AssertionError('Teacher adaptation changed Wu/science configuration')
    return result


def build_config(base_config, shared_cache, *, project_root=ROOT):
    base = _base(base_config, project_root)
    manifest_path = shared.checked(base['manifest'], project_root)
    cache = shared.verify_cache(shared_cache, manifest_path, project_root=project_root)
    if base['scene'] != cache['scene']:
        raise ValueError('Shared cache and SplatSuRe config refer to different scenes')
    return _config_document(base, shared.identity(base_config, project_root),
                            shared.identity(shared_cache, project_root), cache)


def _lr_stage(record):
    if (record.get('status') != 'stage_completed' or record.get('stage') != 'lr'
            or record.get('steps') != 30000 or record.get('wiring_fixture') is not False
            or record.get('prior_bundle') is not None
            or record.get('training_HR_read') is not False
            or record.get('heldout_pixels_read') is not False):
        raise ValueError('An actual completed 30k LR stage is required; fixtures cannot supply maps')


def _same_bytes(origin, imported, description):
    """Origin paths are provenance; only the imported destination is opened."""
    if (origin.get('sha256') != imported['sha256']
            or ('bytes' in origin and origin['bytes'] != imported['bytes'])):
        raise ValueError(description + ' is not the exact original file content')


def _lr_payload(parent, original, config, record, project_root):
    _lr_stage(record)
    if _core(original) != _core(config):
        raise ValueError('LR parent and final SR training configurations differ')
    _, rows = shared.legal_manifest(shared.checked(config['manifest'], project_root), project_root)
    if record.get('training_keys') != len(rows):
        raise ValueError('LR completion does not cover this exact scene training set')
    # Loading on CPU is solely for provenance. No model is constructed/restored,
    # no CUDA availability probe is performed, and no optimizer step is run.
    import torch
    payload = torch.load(shared.checked(parent, project_root), map_location='cpu', weights_only=False)
    shared._common().validate_lr_parent_payload(payload, 'splatsure')
    if (payload.get('schema') != 'static_sr_wu_checkpoint_v1'
            or _core(payload.get('adaptation_config', {})) != _core(original)
            or payload.get('affine') is not None):
        raise ValueError('Parent payload is not this uncorrected scene-local SplatSuRe LR model')
    del payload


def _lr_evidence(lr_completion, project_root):
    """Resolve either a local completion or an explicit portable import.

    A portable record is never substituted for a stage_completed receipt. Its
    copied original JSON is checked byte-for-byte, and its original paths are
    retained as provenance without attempting to open a foreign filesystem.
    """
    document = shared.read(lr_completion, project_root)
    if document.get('schema') != LR_IMPORT_SCHEMA:
        original_path = shared.checked(document['config'], project_root)
        parent = shared.identity(shared.checked(document['checkpoint'], project_root), project_root)
        return document, original_path, parent, dict(
            mode='same_project_original_completion',
            original_completion=shared.identity(lr_completion, project_root),
            origin_checkpoint=document['checkpoint'], imported_checkpoint=parent,
            checkpoint_source_size_recorded='bytes' in document['checkpoint'])
    if (document.get('status') != 'verified_imported_LR_parent'
            or document.get('original_bytes_preserved') is not True
            or document.get('checkpoint_retagged') is not False):
        raise ValueError('Invalid portable LR import status or preservation policy')
    original_complete = shared.checked(document['original_completion'], project_root)
    record = shared.read(original_complete, project_root)
    _lr_stage(record)
    # Origin completion path names the source host. It is never resolved here.
    origin_complete = document['origin_completion']
    if not Path(origin_complete['path']).is_absolute() or '..' in Path(origin_complete['path']).parts:
        raise ValueError('Portable origin completion must record an explicit absolute source path')
    _same_bytes(origin_complete, shared.identity(original_complete, project_root), 'Original completion copy')
    if document['origin_checkpoint'] != record['checkpoint'] or document['origin_config'] != record['config']:
        raise ValueError('Portable provenance differs from the preserved original completion')
    original_path = shared.checked(document['imported_config'], project_root)
    _same_bytes(record['config'], shared.identity(original_path, project_root), 'Original config copy')
    parent = shared.identity(shared.checked(document['imported_checkpoint'], project_root), project_root)
    _same_bytes(record['checkpoint'], parent, 'LR checkpoint copy')
    original = _base(original_path, project_root)
    if document['scene'] != original['scene'] or document['manifest'] != original['manifest']:
        raise ValueError('Portable import scene/manifest differs from its original configuration')
    producer = shared.checked(document['producer'], project_root)
    if (producer != Path(project_root).absolute() / Path(__file__).resolve().relative_to(ROOT)
            or document['producer']['sha256'] != shared.sha(__file__)):
        raise ValueError('Portable LR import does not identify this verified import implementation')
    return record, original_path, parent, dict(
        mode='explicit_portable_import', import_record=shared.identity(lr_completion, project_root),
        original_completion=document['original_completion'], origin_completion=origin_complete,
        origin_checkpoint=document['origin_checkpoint'], imported_checkpoint=parent,
        origin_config=document['origin_config'], imported_config=document['imported_config'],
        checkpoint_source_size_recorded='bytes' in document['origin_checkpoint'],
        identity_rule='SHA256 equality; imported byte size always recorded; origin size compared when originally recorded')


def import_lr_parent(original_completion, imported_checkpoint, imported_config, config_path,
                     origin_completion_path, *, project_root=ROOT):
    """Register copied LR bytes explicitly; never edit or relabel the originals."""
    original_completion = shared.path(original_completion, project_root)
    record = shared.read(original_completion, project_root)
    _lr_stage(record)
    origin = Path(origin_completion_path)
    if not origin.is_absolute() or '..' in origin.parts:
        raise ValueError('Explicit absolute origin-completion path required')
    config_copy = shared.identity(imported_config, project_root)
    _same_bytes(record['config'], config_copy, 'Original config copy')
    parent = shared.identity(imported_checkpoint, project_root)
    _same_bytes(record['checkpoint'], parent, 'LR checkpoint copy')
    original = _base(imported_config, project_root)
    target = _base(config_path, project_root)
    _lr_payload(parent, original, target, record, project_root)
    original_id = shared.identity(original_completion, project_root)
    return dict(schema=LR_IMPORT_SCHEMA, status='verified_imported_LR_parent',
        scene=original['scene'], manifest=original['manifest'],
        original_completion=original_id,
        origin_completion=dict(original_id, path=str(origin)),
        origin_checkpoint=record['checkpoint'], imported_checkpoint=parent,
        origin_config=record['config'], imported_config=config_copy,
        original_bytes_preserved=True, checkpoint_retagged=False,
        producer=shared.identity(__file__),
        identity_rule='SHA256 equality; destination size recorded; origin size compared when present',
        intended_use='Frozen contribution geometry only; independently initialized final SR')


def lr_parent_provenance(lr_completion, *, project_root=ROOT):
    return _lr_evidence(lr_completion, project_root)[3]


def lr_dependency_paths(lr_completion, *, project_root=ROOT):
    _, config_path, parent, provenance = _lr_evidence(lr_completion, project_root)
    paths = [shared.path(lr_completion, project_root), config_path, shared.checked(parent, project_root)]
    paths.append(shared.checked(provenance['original_completion'], project_root))
    return list(dict.fromkeys(paths))


def verify_lr_parent(lr_completion, config, *, project_root=ROOT):
    """Validate original stage and actual CPU payload, with explicit relocation."""
    record, config_path, parent, _ = _lr_evidence(lr_completion, project_root)
    original = _base(config_path, project_root)
    _lr_payload(parent, original, config, record, project_root)
    return parent, record


def _maps(receipt_path, parent, config, manifest, rows, project_root):
    receipt_path = shared.path(receipt_path, project_root)
    receipt = shared.read(receipt_path, project_root)
    if (receipt.get('schema') != 'wu_splatsure_weights_v1' or receipt.get('status') != 'prepared'
            or receipt.get('scene') != manifest['scene']
            or receipt.get('manifest_sha256') != config['manifest']['sha256']
            or receipt.get('training_only') is not True
            or receipt.get('time_policy') != 'same_time_train_cameras_only'
            or receipt.get('requested_frame_indices') is not None
            or receipt.get('parameter_updates') != 0 or receipt.get('HR_image_reads') != 0
            or receipt.get('upstream_commit') != UPSTREAM
            or receipt.get('rasterizer_commit') != RASTERIZER
            or receipt.get('rasterizer_settings') != {'antialiasing': False, 'num_contributions': 10}
            or receipt.get('config') != MAP_CONFIG):
        raise ValueError('Require full source contribution maps for this exact scene/protocol')
    manifest_path = shared.checked(config['manifest'], project_root)
    if shared.path(receipt['manifest_source'], project_root) != manifest_path:
        raise ValueError('Map receipt names another manifest')
    provenance = receipt['provenance']
    actual_parent = shared.identity(shared.checked(provenance['parent_checkpoint'], project_root), project_root)
    if actual_parent != parent or provenance.get('parent_checkpoint_sha256') != parent['sha256']:
        raise ValueError('Contribution maps do not bind the audited LR parent')
    map_config = _base(shared.checked(provenance['source_config'], project_root), project_root)
    if _core(map_config) != _core(config):
        raise ValueError('Contribution producer used a different training configuration')
    producer_paths = {shared.checked(i, project_root) for i in receipt['producer_files']}
    required = {EXP / 'methods/splatsure_prepare.py', EXP / 'methods/splatsure.py'}
    if producer_paths != required:
        raise ValueError('Maps were not made by the actual frozen contribution producer')
    if shared.checked(provenance['backend'], project_root) != EXP / 'prepare_priors.py':
        raise ValueError('Map CLI does not identify the formal parent-validation entry')
    binaries = receipt.get('rasterizer_binary_identities', [])
    if not binaries:
        raise ValueError('Map receipt lacks the actual rasterizer binary identity')
    for item in binaries:
        shared.checked(item, project_root)
    expected = {(r['camera_id'], r['frame_index']): r for r in rows}
    records = receipt['records']
    actual = {(r['camera_id'], r['frame_index']): r for r in records}
    if len(actual) != len(records) or set(actual) != set(expected) or receipt.get('render_calls') != 3 * len(rows):
        raise ValueError('Incomplete, duplicate or foreign contribution maps')
    import torch
    maps = {}
    for key, row in actual.items():
        if row['time'] != expected[key]['time'] or row['path'] != f'maps/{key[0]}_{key[1]:06d}.pt':
            raise ValueError('Map time/path differs from its exact training observation')
        item = dict(path=str(shared.path(row['path'], project_root, receipt_path.parent)), sha256=row['sha256'])
        p = shared.checked(item, project_root)
        value = torch.load(p, map_location='cpu', weights_only=True)
        shape = list(reversed(manifest['resolutions']['lr']))
        if (not torch.is_tensor(value) or value.dtype != torch.float32 or list(value.shape) != shape
                or row['shape'] != shape or not torch.isfinite(value).all()
                or bool((value < 0).any()) or bool((value > 1).any())):
            raise ValueError('Invalid scalar LR contribution map tensor')
        maps[key] = shared.identity(p, project_root)
    return maps, receipt


def build_bundle(config_path, shared_cache, weights_receipt, lr_completion, *, project_root=ROOT):
    config = _base(config_path, project_root)
    manifest_path = shared.checked(config['manifest'], project_root)
    cache = shared.verify_cache(shared_cache, manifest_path, project_root=project_root)
    marker = config.get('information', {}).get('shared_teacher', {})
    base_path = shared.checked(marker['source_config'], project_root)
    base = _base(base_path, project_root)
    expected_config = _config_document(base, shared.identity(base_path, project_root),
                                      shared.identity(shared_cache, project_root), cache)
    if config != expected_config:
        raise ValueError('Shared config changed fields beyond the teacher label/information')
    manifest, rows = shared.legal_manifest(manifest_path, project_root)
    parent, completed = verify_lr_parent(lr_completion, config, project_root=project_root)
    if completed.get('training_keys') != len(rows):
        raise ValueError('LR completion does not cover this manifest training set')
    maps, weight_document = _maps(weights_receipt, parent, config, manifest, rows, project_root)
    entries = [dict(camera=e['camera'], frame=e['frame'], time=e['time'], lr_sha256=e['lr_sha256'],
                    sr=e['sr'], weight=maps[(e['camera'], e['frame'])]) for e in cache['entries']]
    return dict(schema='static_sr_wu_prior_bundle_v1', method='splatsure',
        manifest=config['manifest'], scene=manifest['scene'], training_only=True,
        teacher_variant=shared.VARIANT, teacher_recipe='shared_existing_areaft_SwinIR_RGB8_PNG',
        entries=entries, model_identities=cache['model_identities'] + [parent],
        producer=[shared.identity(__file__), shared.identity(Path(shared.__file__)),
                  shared.identity(shared_cache, project_root), shared.identity(weights_receipt, project_root),
                  shared.identity(lr_completion, project_root)] + cache['producer'] + weight_document['producer_files'],
        shared_cache=shared.identity(shared_cache, project_root),
        teacher_producer_receipt=shared.identity(shared_cache, project_root),
        weight_producer_receipt=shared.identity(weights_receipt, project_root),
        LR_parent_reuse=dict(checkpoint=parent, completion=shared.identity(lr_completion, project_root),
                             provenance=lr_parent_provenance(lr_completion, project_root=project_root),
                             use='Frozen contribution maps only; final SR independently initializes',
                             original_checkpoint_or_metadata_rewritten=False),
        input_decode=cache['decode'], teacher_provenance=cache['model_provenance'],
        author_inference_receipt=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    c = sub.add_parser('config')
    c.add_argument('--base-config', required=True)
    b = sub.add_parser('bundle')
    b.add_argument('--config', required=True)
    b.add_argument('--weights-receipt', required=True)
    b.add_argument('--lr-completion', required=True)
    lr = sub.add_parser('import-lr')
    lr.add_argument('--original-completion', required=True, help='Copied original stage JSON, unchanged bytes')
    lr.add_argument('--origin-completion-path', required=True, help='Absolute source-host path, provenance only')
    lr.add_argument('--imported-checkpoint', required=True)
    lr.add_argument('--imported-config', required=True, help='Copied original registered config, unchanged bytes')
    lr.add_argument('--config', required=True, help='New shared-teacher target config')
    lr.add_argument('--out', required=True)
    for p in (c, b):
        p.add_argument('--shared-cache', required=True)
        p.add_argument('--out', required=True)
    a = parser.parse_args()
    if a.command == 'config':
        result = build_config(a.base_config, a.shared_cache)
    elif a.command == 'import-lr':
        result = import_lr_parent(a.original_completion, a.imported_checkpoint, a.imported_config,
                                  a.config, a.origin_completion_path)
    else:
        result = build_bundle(a.config, a.shared_cache, a.weights_receipt, a.lr_completion)
    shared.write_new(a.out, result)
    print(json.dumps(dict(output=shared.identity(a.out), GPU_calls=0, status='CPU_contract_prepared')))


if __name__ == '__main__':
    main()
