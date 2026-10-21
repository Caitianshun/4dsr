"""CPU-only complete byte-SHA verification of a frozen, closed input inventory.

No torch/CUDA import, file repair, GPU query, synchronization or reservation occurs.
The receipt is accepted only when every listed file has its expected size and SHA.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import socket
import time


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def contained(root, relative):
    path = Path(relative)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError('Inventory paths must be project relative: ' + relative)
    target = root / path
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError('Input resolves outside project: ' + relative)
    return target


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def verify(root, inventory_path, workers):
    root = Path(root).resolve()
    inventory_path = Path(inventory_path).resolve()
    identity = sha(inventory_path)
    inventory = json.loads(inventory_path.read_text())
    assert inventory['schema'] == 'CEM_area4_closed_input_inventory_v1'
    started = time.monotonic()

    def check(item):
        relative, expected = item
        path = contained(root, relative)
        value = dict(role=expected['role'], scene=expected['scene'])
        try:
            before = path.stat()
            value['bytes'] = before.st_size
            value['sha256'] = sha(path)
            after = path.stat()
            value['stable_during_read'] = (before.st_size, before.st_mtime_ns, before.st_ino) == (after.st_size, after.st_mtime_ns, after.st_ino)
            value['status'] = 'verified' if (value['bytes'] == expected['bytes'] and value['sha256'] == expected['sha256'] and value['stable_during_read']) else 'mismatch'
        except FileNotFoundError:
            value['status'] = 'missing'
        except Exception as error:
            value.update(status='error', error=repr(error))
        return relative, value

    with ThreadPoolExecutor(max_workers=workers) as executor:
        files = dict(executor.map(check, inventory['files'].items()))
    failures = {name: value for name, value in files.items() if value['status'] != 'verified'}
    return dict(schema='CEM_area4_complete_input_verification_v1',
                status='verified' if not failures else 'failed',
                inventory_path=str(inventory_path.relative_to(root)) if inventory_path.is_relative_to(root) else str(inventory_path),
                inventory_sha256=identity, host=socket.gethostname(), root=str(root),
                finished_utc=datetime.now(timezone.utc).isoformat(), seconds=time.monotonic()-started,
                hash_algorithm='SHA256', all_listed_file_bytes_rehashed=True, CPU_only=True,
                file_count=len(files), verified_count=len(files)-len(failures),
                total_expected_bytes=inventory['total_bytes'], failure_count=len(failures),
                failures=failures, files=files,
                scope=inventory['scope'], train_test_boundary=inventory['train_test_boundary'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    assert 1 <= args.workers <= 8
    result = verify(args.root, args.inventory, args.workers)
    atomic_json(args.receipt, result)
    print(json.dumps({key: value for key, value in result.items() if key not in {'files', 'failures'}}, ensure_ascii=False), flush=True)
    raise SystemExit(0 if result['status'] == 'verified' else 2)


if __name__ == '__main__':
    main()
