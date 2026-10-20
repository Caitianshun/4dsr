"""Check a frozen inventory without loading any image into a model."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--inventory', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    inventory = json.loads(Path(args.inventory).read_text())
    def check(item):
        name, value = item
        path = root / name
        if root not in path.resolve().parents:
            return name, 'escaped_root'
        if not path.is_file():
            return name, 'missing'
        return name, 'verified' if path.stat().st_size == value['bytes'] and sha(path) == value['sha256'] else 'mismatch'
    with ThreadPoolExecutor(max_workers=4) as pool:
        result = list(pool.map(check, inventory['files'].items()))
    value = dict(utc=datetime.now(timezone.utc).isoformat(), root=str(root),
                 status='verified' if all(status == 'verified' for _,status in result) else 'incomplete',
                 inventory_sha256=sha(Path(args.inventory)), verified=sum(status=='verified' for _,status in result),
                 missing=[name for name,status in result if status=='missing'],
                 mismatch=[name for name,status in result if status not in ('missing','verified')])
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')
    print(json.dumps({k:(len(v) if isinstance(v,list) else v) for k,v in value.items()}))


if __name__ == '__main__':
    main()
