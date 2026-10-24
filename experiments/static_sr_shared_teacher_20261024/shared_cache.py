"""Import existing SwinIR PNGs; no inference, image rewriting, or scheduling.

The source teacher inventory remains the evidence for production. This module
verifies its bytes and registers a shared, quantized input. It never creates an
author StableSR/DF2K inference receipt for the area-finetuned project teacher.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
COMMON_PATH = ROOT / 'experiments/static_sr_wu_20261022/common.py'
SCHEMA = 'shared_swinir_cache_v1'
VARIANT = 'shared_swinir_areaft_png'
DEFAULT_TAG = 'sr_swinir_x4_areaft'
# Read and audited producer: frozen SwinIR, mirror padding, rounding to RGB8;
# it contains no w4, colour correction, target-view pixels, or learned fusion.
GENERATOR_SHA256 = 'dde031195a6db7f4b64eaa5c921788a9f8f181e7d047363a37052243836acde6'
DECODE = dict(container='PNG', mode='RGB', bit_depth=8,
              tensor='float32_CHW_div255', resize=False,
              w4=False, colour_correction=False,
              source_quantization='round(clamp(SwinIR_output,0,1)*255)')


def path(value, project_root=ROOT, base=None):
    """Project-confined, lexical paths; hard links are allowed, symlinks are not."""
    root = Path(project_root).absolute()
    p = Path(value)
    if '..' in p.parts:
        raise ValueError('Path traversal is not a cache identity')
    p = p if p.is_absolute() else (Path(base) if base is not None else root) / p
    p = p.absolute()
    if not p.is_relative_to(root) or any(q.is_symlink() for q in [p, *p.parents]):
        raise ValueError('Cache assets must be regular paths inside this project')
    return p


def sha(p):
    digest = hashlib.sha256()
    with Path(p).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def identity(value, project_root=ROOT):
    p = path(value, project_root)
    if not p.is_file():
        raise ValueError(f'Missing regular file: {p}')
    return dict(path=str(p.relative_to(Path(project_root).absolute())),
                sha256=sha(p), bytes=p.stat().st_size)


def checked(item, project_root=ROOT):
    if not isinstance(item, dict) or not re.fullmatch('[a-f0-9]{64}', str(item.get('sha256', ''))):
        raise ValueError('An exact SHA256 file identity is required')
    p = path(item['path'], project_root)
    actual = identity(p, project_root)
    if actual['sha256'] != item['sha256'] or actual['bytes'] != item.get('bytes', actual['bytes']):
        raise ValueError(f'Identity changed: {p}')
    return p


def _unique(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f'Duplicate JSON field: {key}')
        value[key] = item
    return value


def read(value, project_root=ROOT):
    return json.loads(path(value, project_root).read_text(), object_pairs_hook=_unique)


def write_new(value, document, project_root=ROOT):
    p = path(value, project_root)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x') as stream:
        stream.write(json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    return p


def _common():
    spec = importlib.util.spec_from_file_location('_shared_teacher_frozen_common', COMMON_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def legal_manifest(manifest_path, project_root=ROOT):
    p = path(manifest_path, project_root)
    raw = read(p, project_root)
    frames = raw['frame_indices']
    if (len(frames) != len(set(frames)) or
            any(type(f) is not int or not 0 <= f < 300 for f in frames)):
        raise ValueError('Invalid original frame indices')
    # Check paths before the frozen validator opens the initializer. Neither
    # validator opens held-out images or training HR images.
    path(raw['initialization']['npz_path'], project_root, p.parent)
    for row in raw['observations']:
        if row['split'] == 'train':
            if type(row['frame_index']) is not int or not math.isfinite(row['time']):
                raise ValueError('Invalid training frame/time')
            expected = f"lr/{row['camera_id']}/{row['frame_index']:04d}.png"
            if row['lr_path'] != expected:
                raise ValueError('LR path must identify this exact camera/frame')
            path(row['lr_path'], project_root, p.parent)
    return _common().training_manifest(p, verify_pixels=False)


def png_identity(value, expected_sha, expected_size, project_root=ROOT):
    """Hash and fully decode the same bytes, rejecting palette/alpha/16-bit PNG."""
    from PIL import Image
    p = path(value, project_root)
    raw = p.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha:
        raise ValueError(f'PNG bytes differ: {p}')
    if raw[:8] != b'\x89PNG\r\n\x1a\n' or raw[12:16] != b'IHDR' or raw[24:26] != bytes([8, 2]):
        raise ValueError('Shared inputs must be RGB 8-bit PNGs')
    with Image.open(io.BytesIO(raw)) as image:
        if image.format != 'PNG' or image.mode != 'RGB' or list(image.size) != list(expected_size):
            raise ValueError(f'PNG format/size differs: {p}')
        if getattr(image, 'n_frames', 1) != 1:
            raise ValueError('Animated PNG is not one observation')
        image.load()
    return dict(path=str(p.relative_to(Path(project_root).absolute())),
                sha256=expected_sha, bytes=len(raw))


def _model_provenance(config, summary_path, manifest, project_root):
    if config.get('teacher') != 'swinir_area_finetuned_x4' or not summary_path:
        raise ValueError('This adapter requires an explicit area-finetuned SwinIR summary')
    model = identity(checked(config['weights'], project_root), project_root)
    summary_path = path(summary_path, project_root)
    summary = read(summary_path, project_root)
    if summary.get('status') != 'completed' or identity(checked(summary['weights'], project_root), project_root) != model:
        raise ValueError('Fine-tuning summary does not identify the actual teacher')
    if summary.get('degradation') != 'round(4x4 box mean), 8-bit sRGB':
        raise ValueError('Teacher fine-tuning degradation differs from area4')
    base = identity(checked(summary['base_weights'], project_root), project_root)
    data_path = checked(summary['data_index'], project_root)
    data = read(data_path, project_root)
    if data.get('purpose') != 'area-matched SwinIR fine-tuning (non-evaluation sources)' or not data.get('frames'):
        raise ValueError('Missing declared non-evaluation fine-tuning provenance')
    sources = sorted({row['source'] for row in data['frames'].values()})
    if manifest['scene'] in sources:
        raise ValueError('This target scene occurs in teacher fine-tuning sources')
    raw_directory = manifest.get('raw_directory')
    if raw_directory and any(Path(row['video']).is_relative_to(Path(raw_directory)) for row in data['frames'].values()):
        raise ValueError('Teacher fine-tuning includes this target scene raw data')
    # Historical training code was subsequently extended. Preserve the recorded
    # identity; do not pretend a current file with a different SHA is that code.
    declared = summary['code']
    current_path = path(declared['path'], project_root)
    current = identity(current_path, project_root) if current_path.is_file() else None
    return model, dict(
        kind='SwinIR-M_x4_external_area4_finetuned', original_DF2K_checkpoint=False,
        finetune_summary=identity(summary_path, project_root), base_weights=base,
        data_index=identity(data_path, project_root), source_names=sources,
        data_purpose=data['purpose'], train_frames=summary['train_frames'], val_frames=summary['val_frames'],
        degradation=summary['degradation'], historical_training_code=declared,
        historical_training_code_current_identity=current,
        historical_training_code_currently_matches=bool(current and current['sha256'] == declared['sha256']),
        provenance_scope='Recorded generic external fine-tuning inventory; no target-scene HR is read by this import')


def import_cache(manifest, teacher_index, tag=DEFAULT_TAG, *, finetune_summary=None, project_root=ROOT):
    """Verify every legal LR/SR pair and return a new deterministic cache record."""
    if not re.fullmatch('[A-Za-z0-9_-]+', tag):
        raise ValueError('Tag must be a single directory component')
    manifest_path = path(manifest, project_root)
    m, rows = legal_manifest(manifest_path, project_root)
    index_path = path(teacher_index, project_root)
    source = read(index_path, project_root)
    if (source.get('schema') != 1 or source.get('status') != 'completed_teacher_inventory'
            or source.get('scene') != m['scene'] or source.get('oracle') is not False
            or source.get('seed_independent_frozen_prior') is not True):
        raise ValueError('Not a complete, non-oracle inventory for this scene')
    if (path(source['manifest'], project_root) != manifest_path
            or source['manifest_sha256'] != sha(manifest_path)
            or source.get('train_cameras') != sorted(m['splits']['train'])
            or source.get('frame_indices') != m['frame_indices']):
        raise ValueError('Teacher manifest or full training split differs')
    config = source['teacher_config']
    if config.get('kind') != 'physlr_teacher' or config.get('tag') != tag or config.get('scale') != 4:
        raise ValueError('Teacher type/tag/scale differs')
    generator = identity(checked(config['code'], project_root), project_root)
    if generator['sha256'] != GENERATOR_SHA256:
        raise ValueError('Unreviewed producer; no-w4/no-colour provenance cannot be asserted')
    model, provenance = _model_provenance(config, finetune_summary, m, project_root)
    expected = {(o['camera_id'], o['frame_index']): o for o in rows}
    supplied = {}
    for row in source['entries']:
        if type(row['frame']) is not int:
            raise ValueError('Teacher frames must be integer original frame indices')
        key = (row['camera'], row['frame'])
        if key in supplied or key not in expected:
            raise ValueError('Duplicate, foreign or held-out teacher observation')
        supplied[key] = row
    if set(supplied) != set(expected):
        raise ValueError('Teacher inventory does not cover all legal training observations')
    entries = []
    for key, observation in expected.items():
        row = supplied[key]
        sr_relative = f'{tag}/{key[0]}/{key[1]:04d}.png'
        if (row['relative_path'] != sr_relative or row['lr_relative_path'] != observation['lr_path']
                or row['lr_sha256'] != observation['lr_sha256']
                or ('time' in row and row['time'] != observation['time'])):
            raise ValueError('Teacher key/path/LR identity/time differs')
        lr = png_identity(path(row['lr_relative_path'], project_root, manifest_path.parent),
                          observation['lr_sha256'], m['resolutions']['lr'], project_root)
        sr = png_identity(path(sr_relative, project_root, manifest_path.parent),
                          row['sha256'], m['resolutions']['hr'], project_root)
        entries.append(dict(key=f'{key[0]}/{key[1]:04d}', camera=key[0], frame=key[1],
                            time=observation['time'], lr_sha256=observation['lr_sha256'],
                            lr=lr, sr=sr, shape=list(m['resolutions']['hr'])))
    return dict(schema=SCHEMA, status='verified_existing_cache', scene=m['scene'],
                manifest=identity(manifest_path, project_root), source_index=identity(index_path, project_root),
                tag=tag, teacher_variant=VARIANT, training_only=True, entries=entries,
                model_identities=[model], model_provenance=provenance,
                source_teacher_config=config, producer=[generator],
                importer=dict(path=str(Path(__file__).resolve()), sha256=sha(__file__)),
                validator=dict(path=str(COMMON_PATH), sha256=sha(COMMON_PATH)),
                decode=DECODE, time_source='Exact camera/frame join to registered manifest; source inventory has no time field',
                HR_images_read=0, heldout_images_read=0, inference_executed=False,
                cached_pngs_rewritten=False,
                original_method_receipt=False)


def verify_cache(cache_path, manifest_path=None, *, project_root=ROOT):
    """Revalidate source identities and all image bytes; never trusts status alone."""
    document = read(cache_path, project_root)
    if document.get('schema') != SCHEMA or document.get('status') != 'verified_existing_cache':
        raise ValueError('Expected an imported shared SwinIR cache')
    mp = checked(document['manifest'], project_root)
    if manifest_path is not None and path(manifest_path, project_root) != mp:
        raise ValueError('Shared cache belongs to a different manifest')
    index = checked(document['source_index'], project_root)
    summary = checked(document['model_provenance']['finetune_summary'], project_root)
    regenerated = import_cache(mp, index, document['tag'], finetune_summary=summary, project_root=project_root)
    if document != regenerated:
        raise ValueError('Shared record differs from its exact source inventory and verified images')
    return document


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--teacher-index', required=True)
    parser.add_argument('--tag', default=DEFAULT_TAG)
    parser.add_argument('--finetune-summary', required=True)
    parser.add_argument('--output', required=True)
    a = parser.parse_args()
    result = import_cache(a.manifest, a.teacher_index, a.tag, finetune_summary=a.finetune_summary)
    output = write_new(a.output, result)
    print(json.dumps(dict(status=result['status'], observations=len(result['entries']), output=identity(output))))


if __name__ == '__main__':
    main()
