#!/usr/bin/env python3
"""Seal deployment receipts after the service exits, including its final stdout."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time

p = argparse.ArgumentParser()
p.add_argument('--root', required=True)
p.add_argument('--cuobjdump', required=True)
a = p.parse_args()
root = Path(a.root)
out = root / 'remote_deployment'
status = json.loads((out / 'status.json').read_text())
assert status['status'] == 'complete', status
assert status['original_environment_unchanged']
modules = json.loads((out / 'sr4d_extension_identity.log').read_text())['modules']
arch = []
for name, info in modules.items():
    if not info['file'].endswith('.so'):
        continue
    p = subprocess.run([a.cuobjdump, '--list-elf', info['file']], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    found = sorted(set(re.findall(r'sm_[0-9]+', p.stdout)))
    expected = 'sm_' + status['arch'].replace('.', '')
    assert p.returncode == 0 and found == [expected], (name, p.returncode, p.stdout)
    arch.append({'module': name, 'tool': a.cuobjdump, 'architectures': found, 'output': p.stdout})
(out / 'binary_architecture.json').write_text(json.dumps(arch, indent=2) + '\n')
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
files = {str(p.relative_to(root)): {'bytes': p.stat().st_size, 'sha256': sha(p)}
         for p in out.iterdir() if p.is_file() and p.name != 'final_receipt.json'}
for name in ['activate_sr4d.sh', 'deploy_remote.py', 'source_manifest.json', 'local_source.patch', 'upstream_commit.txt', 'source_snapshot.tar.gz']:
    p = root / name
    files[name] = {'bytes': p.stat().st_size, 'sha256': sha(p)}
receipt = {'finished_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'status': 'complete',
           'files': files, 'note': 'This final receipt is generated after the deployment service exits. The in-service status receipt_hashes/service.log was captured before its final status print, so use this sealed service.log hash.'}
(out / 'final_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps({'status': 'complete', 'files': len(files), 'architectures': arch}, indent=2))
