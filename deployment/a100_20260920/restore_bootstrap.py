"""Unpack the three unchanged common parents and verify every original byte."""
import gzip
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]
rows = json.loads((ROOT / 'deployment/a100_20260920/bootstrap_inventory.json').read_text())
for row in rows:
    dest = ROOT / row['path']
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        part = dest.with_suffix('.unpack')
        with gzip.open(ROOT / 'checkpoints/bootstrap' / row['payload'], 'rb') as src, part.open('wb') as target:
            shutil.copyfileobj(src, target)
        assert hashlib.sha256(part.read_bytes()).hexdigest() == row['sha256'], dest
        part.replace(dest)
    assert hashlib.sha256(dest.read_bytes()).hexdigest() == row['sha256'], dest
    print('VERIFIED', row['path'])
(ROOT / 'deployment/a100_20260920/bootstrap_verified.json').write_text(json.dumps(dict(status='passed', files=rows), indent=2))
