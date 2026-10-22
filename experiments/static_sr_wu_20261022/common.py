"""CPU-only identities and train-only input contracts for explicitly adapted baselines."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = 'static_sr_wu_adaptation_v1'
METHODS = ('splatsure', 'iesrgs', 'sequence_matters', 'threedsr')
SHUFFLE_CODEC = 'float32_byte_shuffle'
SHUFFLE_VERSION = 1
MAX_SHUFFLED_PRIOR_BYTES = 512 * (1 << 20)


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


def _shuffle_metadata(value):
    """Closed version-1 schema; reject sizes before allocating the decoded array."""
    import math
    import re
    keys = {'format', 'codec', 'codec_version', 'dtype', 'dtype_str', 'shape', 'array_order', 'array_sha256'}
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError('Byte-shuffle metadata fields differ')
    if (value['format'] != 'npz_byte_shuffle' or value['codec'] != SHUFFLE_CODEC
            or type(value['codec_version']) is not int or value['codec_version'] != SHUFFLE_VERSION
            or value['dtype'] != 'float32' or value['dtype_str'] != '<f4' or value['array_order'] != 'C'):
        raise ValueError('Unknown byte-shuffle codec/version/dtype/order')
    shape = value['shape']
    if (not isinstance(shape, list) or not 1 <= len(shape) <= 4
            or any(type(d) is not int or not 0 < d <= 65536 for d in shape)
            or math.prod(shape) * 4 > MAX_SHUFFLED_PRIOR_BYTES):
        raise ValueError('Invalid or oversized byte-shuffle shape')
    if not isinstance(value['array_sha256'], str) or not re.fullmatch('[a-f0-9]{64}', value['array_sha256']):
        raise ValueError('Invalid byte-shuffle raw SHA256')
    return math.prod(shape)


def _read_shuffle_member(archive, name, expected_shape=None, max_bytes=4096):
    """Inspect the NPY header and exact byte count before allocation; no pickle."""
    import math
    import numpy as np
    info = archive.getinfo(name)
    if not 0 < info.file_size <= max_bytes + 256:
        raise ValueError('Oversized byte-shuffle archive member')
    with archive.open(info) as stream:
        if np.lib.format.read_magic(stream) != (1, 0):
            raise ValueError('Unsupported byte-shuffle NPY header version')
        shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream, max_header_size=128)
        if dtype != np.dtype('uint8') or fortran:
            raise ValueError('Byte-shuffle members must be C-order uint8 arrays')
        if (expected_shape is not None and shape != expected_shape) or (
                expected_shape is None and (len(shape) != 1 or not 0 < shape[0] <= max_bytes)):
            raise ValueError('Byte-shuffle member shape differs')
        size = math.prod(shape)
        if size > max_bytes or info.file_size != stream.tell() + size:
            raise ValueError('Byte-shuffle member byte count differs')
        raw = stream.read(size + 1)
        if len(raw) != size:
            raise ValueError('Byte-shuffle member payload length differs')
    return np.frombuffer(raw, dtype=np.uint8).reshape(shape)


def _load_shuffled_array(archive):
    import numpy as np
    def unique_fields(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('Duplicate byte-shuffle metadata field')
            result[key] = value
        return result
    raw_metadata = _read_shuffle_member(archive, 'metadata.npy').tobytes()
    metadata = json.loads(raw_metadata.decode('ascii'), object_pairs_hook=unique_fields)
    count = _shuffle_metadata(metadata)
    shuffled = _read_shuffle_member(archive, 'array.npy', (4, count), count * 4)
    array = np.array(shuffled.T, copy=True, order='C').reshape(-1).view('<f4').reshape(metadata['shape'])
    if hashlib.sha256(array.tobytes(order='C')).hexdigest() != metadata['array_sha256']:
        raise ValueError('Byte-shuffle restored raw SHA256 differs')
    return array, metadata


def load_prior_array(item, require_float32=False):
    """Read hashed NPY, plain NPZ, or versioned byte-shuffle NPZ; no pickle.

    Legacy NPY identities contain only path/sha256 and remain supported. New
    storage metadata is optional, but every supplied value is verified. NPZ
    is reserved for the float32 IE cache contract; no dtype conversion occurs.
    """
    import numpy as np
    import zipfile
    p = checked(item)
    encoded_metadata = {}
    if p.suffix == '.npy':
        storage_format = 'npy'
        array = np.load(p, allow_pickle=False)
    elif p.suffix == '.npz':
        storage_format = 'npz_deflate'
        with zipfile.ZipFile(p) as archive:
            members = archive.infolist()
            names = [m.filename for m in members]
            if (len(set(names)) != len(names) or any(m.compress_type != zipfile.ZIP_DEFLATED or m.flag_bits & 1 for m in members)):
                raise ValueError('Prior NPZ requires unique, unencrypted deflated members')
            if names == ['array.npy']:
                with np.load(p, allow_pickle=False) as plain:
                    array = plain['array']
            elif set(names) == {'array.npy', 'metadata.npy'}:
                storage_format = 'npz_byte_shuffle'
                array, encoded_metadata = _load_shuffled_array(archive)
            else:
                raise ValueError('Prior NPZ must contain exactly one deflated array.npy member or the versioned byte-shuffle pair')
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
    if not isinstance(storage, dict):
        raise ValueError('Prior storage metadata must be an object')
    if encoded_metadata and storage:
        _shuffle_metadata(storage)
    expected = {'format': storage_format, 'dtype': str(array.dtype),
                'shape': list(array.shape), 'array_order': 'C', **encoded_metadata}
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
