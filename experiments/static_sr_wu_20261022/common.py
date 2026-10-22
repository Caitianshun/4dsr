"""CPU-only identities and train-only input contracts for explicitly adapted baselines."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = 'static_sr_wu_adaptation_v1'
METHODS = ('splatsure', 'iesrgs', 'sequence_matters', 'threedsr')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def resolve(path):
    p = Path(path)
    return (ROOT / p).resolve() if not p.is_absolute() else p.resolve()


def identity(path):
    p = resolve(path)
    return {'path': str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p), 'sha256': sha(p)}


def checked(item):
    p = resolve(item['path'])
    if sha(p) != item['sha256']:
        raise ValueError(f'Identity changed: {p}')
    return p


def load_prior_array(item, require_float32=False):
    """Read one hashed NPY or single-member, lossless NPZ; never enable pickle.

    Legacy NPY identities contain only path/sha256 and remain supported. New
    storage metadata is optional, but every supplied value is verified. NPZ
    is reserved for the float32 IE cache contract; no dtype conversion occurs.
    """
    import numpy as np
    import zipfile
    p = checked(item)
    if p.suffix == '.npy':
        storage_format = 'npy'
        array = np.load(p, allow_pickle=False)
    elif p.suffix == '.npz':
        storage_format = 'npz_deflate'
        with zipfile.ZipFile(p) as archive:
            members = archive.infolist()
            if (len(members) != 1 or members[0].filename != 'array.npy'
                    or members[0].compress_type != zipfile.ZIP_DEFLATED):
                raise ValueError('Prior NPZ must contain exactly one deflated array.npy member')
        with np.load(p, allow_pickle=False) as archive:
            array = archive['array']
        require_float32 = True
    else:
        raise ValueError('Expected .npy or .npz prior array')
    if not isinstance(array, np.ndarray) or array.dtype.hasobject:
        raise ValueError('Prior cache must contain a numeric array')
    if require_float32 and array.dtype != np.dtype('float32'):
        raise ValueError('IE prior cache must preserve float32 dtype')
    if not np.isfinite(array).all():
        raise ValueError('Nonfinite prior array')
    storage = item.get('storage', {})
    expected = {'format': storage_format, 'dtype': str(array.dtype),
                'shape': list(array.shape), 'array_order': 'C'}
    if 'array_sha256' in storage:
        expected['array_sha256'] = hashlib.sha256(array.tobytes(order='C')).hexdigest()
    for key, value in storage.items():
        if key not in expected or value != expected[key]:
            raise ValueError(f'Prior storage metadata differs: {key}')
    return array


def read(path):
    return json.loads(resolve(path).read_text())


def write_new(path, value):
    p = resolve(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x') as f:
        f.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def training_manifest(path, verify_pixels=False):
    """Never opens an HR image or a held-out image; hash metadata is not supervision."""
    path = resolve(path)
    m = read(path)
    if m.get('schema') != 'n3dv_dynamic_sr_pilot_v1':
        raise ValueError('Unsupported dataset schema')
    train = m['splits']['train']
    held = set(m['splits'].get('test', [])) | set(m['splits'].get('dev', []))
    if len(train) != len(set(train)) or not train or set(train) & held or 'cam00' in train:
        raise ValueError('Illegal camera split')
    if m.get('scale') != 4:
        raise ValueError('This registration is x4 only')
    if m.get('physical_lr_derivation', {}).get('formation') != 'area4':
        raise ValueError('This comparison requires the registered area4 input')
    rows = [o for o in m['observations'] if o['split'] == 'train']
    keys = {(o['camera_id'], o['frame_index']) for o in rows}
    expected = {(c, f) for c in train for f in m['frame_indices']}
    if len(rows) != len(keys) or keys != expected:
        raise ValueError('Duplicate, missing, or foreign train observations')
    for o in rows:
        if abs(o['time'] - o['frame_index'] / 300.) > 1e-8:
            raise ValueError('Expected original frame_index / 300, not per-window renormalization')
        p = (path.parent / o['lr_path']).resolve()
        # Existing area4 assets can be hard-linked; reject accidental HR pathname use.
        if 'hr' in Path(o['lr_path']).parts or not p.is_file():
            raise ValueError(f'Invalid LR path: {p}')
        if verify_pixels and sha(p) != o['lr_sha256']:
            raise ValueError(f'LR bytes changed: {p}')
    if not m.get('initialization', {}).get('train_only_lr'):
        raise ValueError('Initializer is not declared training-LR-only')
    ip = path.parent / m['initialization']['npz_path']
    if sha(ip) != m['initialization']['npz_sha256']:
        raise ValueError('Initializer identity changed')
    m['_root'] = str(path.parent)
    m['_manifest_path'] = str(path)
    return m, sorted(rows, key=lambda x: (x['frame_index'], x['camera_id']))


def records(m, rows):
    return [dict(key=f"{o['camera_id']}/{o['frame_index']:04d}", scene=m['scene'],
                 camera=o['camera_id'], time=o['time'], frame=o['frame_index'], split='train',
                 lr_path=str(Path(m['_root']) / o['lr_path']), lr_sha256=o['lr_sha256'],
                 camera_center=[r[3] for r in m['cameras'][o['camera_id']]['c2w'][:3]]) for o in rows]


def validate_records_bundle(bundle, manifest_id, m, rows):
    if bundle.get('manifest') != manifest_id or bundle.get('heldout_cameras') != m['splits']['test']:
        raise ValueError('Teacher records manifest or held-out split differs')
    expected = {r['key']: r for r in records(m, rows)}
    supplied = bundle.get('records', [])
    actual = {r['key']: r for r in supplied}
    if len(actual) != len(supplied) or actual != expected:
        raise ValueError('Teacher record content differs from exact scene/camera/time/LR input')


def validate_lr_parent_payload(payload, method):
    meta = payload.get('metadata', {})
    if (meta.get('stage') != 'lr' or meta.get('iterations') != 30000 or
            meta.get('optimizer_updates') != 29999 or
            payload.get('adaptation_config', {}).get('method') != method):
        raise ValueError('Formal parent must complete the source LR 30000 iterations / 29999 updates')


def verify_prior_index(path, manifest_id, m, rows, required_fields):
    """A producer must bind every target to this exact scene, LR bytes and source model."""
    idx = read(path)
    if idx.get('schema') != 'static_sr_wu_prior_bundle_v1' or idx.get('manifest') != manifest_id:
        raise ValueError('Prior index schema/manifest mismatch')
    if idx.get('scene') != m['scene'] or idx.get('training_only') is not True:
        raise ValueError('Priors must come only from this scene training inputs')
    if not idx.get('producer') or not idx.get('model_identities'):
        raise ValueError('Prior producer and model identities required')
    expected = {(o['camera_id'], o['frame_index']): o for o in rows}
    entries = {(r['camera'], r['frame']): r for r in idx['entries']}
    if len(entries) != len(idx['entries']) or set(entries) != set(expected):
        raise ValueError('Prior keys are missing, duplicated or held out')
    for k, e in entries.items():
        if e.get('lr_sha256') != expected[k]['lr_sha256']:
            raise ValueError('Prior derived from a different LR input')
        for field in required_fields:
            checked(e[field])
    for item in idx['model_identities']:
        checked(item)
    for item in idx['producer']:
        checked(item)
    return entries
