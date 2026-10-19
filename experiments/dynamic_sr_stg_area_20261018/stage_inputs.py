"""Stage the registered STG inputs with an explicit file list and SHA256 receipt.

No source manifest is rewritten. Existing remote files are never overwritten;
the verifier rejects any byte mismatch. This command does not start CUDA.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
DESTINATIONS = {'a100-train': '/home/ubuntu/3DGS/4dsr', 'cts': '/home/cts/Project/4DSR'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def prepare(scene, out):
    folder = ROOT / 'data/dynamic_sr/full_time_20261007/prepared' / (scene + '_area4')
    manifest = folder / 'manifest_train_ready_seed20261007.json'
    teacher = folder / 'teacher_index_sr_swinir_x4_areaft_seed20261007.json'
    m, t = json.loads(manifest.read_text()), json.loads(teacher.read_text())
    if t['manifest_sha256'] != sha(manifest) or t.get('oracle'):
        raise ValueError('Teacher is not bound to the legal manifest')
    paths = {manifest: sha(manifest), teacher: sha(teacher)}
    for r in m['observations']:
        for kind in ('hr', 'lr'):
            paths[folder / r[kind + '_path']] = r[kind + '_sha256']
    for r in t['entries']:
        paths[folder / r['relative_path']] = r['sha256']
    records = []
    for p, expected in sorted(paths.items()):
        if not p.resolve().is_relative_to(ROOT) or sha(p) != expected:
            raise ValueError('Input identity mismatch: ' + str(p))
        records.append(dict(path=str(p.relative_to(ROOT)), bytes=p.stat().st_size, sha256=expected))
    out.mkdir(parents=True, exist_ok=False)
    receipt = out / 'transfer.json'
    receipt.write_text(json.dumps(dict(schema='stg_inputs_transfer_v1', scene=scene,
        source_root=str(ROOT), files=records, manifest_sha256=sha(manifest),
        bytes=sum(r['bytes'] for r in records), files_verified=True,
        HR_role='evaluation and train-camera areaOracle only; excluded from legal training'), indent=2) + '\n')
    (out / 'files.txt').write_text('\n'.join(r['path'] for r in records) + '\n')
    print(json.dumps(dict(receipt=str(receipt), files=len(records), bytes=sum(r['bytes'] for r in records))))
    return receipt


def verify(receipt):
    data = json.loads(receipt.read_text())
    errors = []
    started = time.monotonic()
    for r in data['files']:
        p = ROOT / r['path']
        if not p.resolve().is_relative_to(ROOT) or not p.is_file():
            errors.append(dict(path=r['path'], reason='missing_or_escape'))
        elif p.stat().st_size != r['bytes'] or sha(p) != r['sha256']:
            errors.append(dict(path=r['path'], reason='identity_mismatch'))
    result = dict(status='failed' if errors else 'verified', errors=errors,
        files=len(data['files']), bytes=data['bytes'], seconds=time.monotonic()-started,
        receipt_sha256=sha(receipt), destination_root=str(ROOT))
    receipt.with_name('verified.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'errors'}))
    if errors:
        raise RuntimeError(str(errors[:5]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene', choices=('cut_roasted_beef', 'cook_spinach'))
    p.add_argument('--out', type=Path)
    p.add_argument('--verify', type=Path)
    a = p.parse_args()
    if a.verify:
        verify(a.verify)
    elif a.scene and a.out:
        prepare(a.scene, a.out)
    else:
        p.error('Provide --verify or --scene and --out')


if __name__ == '__main__':
    main()
