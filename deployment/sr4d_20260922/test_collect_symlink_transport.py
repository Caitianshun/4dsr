#!/usr/bin/env python3
"""Real CTS CPU/network regression: internal final links and frozen manifests."""
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import uuid


HERE = Path(__file__).resolve().parent
attempt = uuid.uuid4().hex[:8]
base = HERE / 'verification' / ('collector_symlink_fixture_' + attempt)
base.mkdir(parents=True)
remote = '/tmp/sr4d_collector_symlink_fixture_20260922_' + attempt
setup = r'''
import pathlib,hashlib,json,sys
r=pathlib.Path(sys.argv[1]);data=r/'data/dynamic_sr/n3dv_prepared/fixture';data.mkdir(parents=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for name,content in [('hr/cam00/0000.png',b'fixed HR transport fixture'),('lr/cam00/0000.png',b'fixed LR transport fixture'),('initialization/points_lr.npz',b'fixed initialization transport fixture')]:
 p=data/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(content)
m={'observations':[{'hr_path':'hr/cam00/0000.png','hr_sha256':sha(data/'hr/cam00/0000.png'),'lr_path':'lr/cam00/0000.png','lr_sha256':sha(data/'lr/cam00/0000.png')}],'initialization':{'npz_path':'initialization/points_lr.npz','npz_sha256':sha(data/'initialization/points_lr.npz')}}
manifest=data/'manifest.json';manifest.write_text(json.dumps(m))
poison=r/'data/sr4d_author_20260922/fixture/sr_swinir_x4/unrelated_teacher.png';poison.parent.mkdir(parents=True);poison.write_bytes(b'Must not be scanned or transferred')
outside=r.parent/(r.name+'_outside.pt');outside.write_bytes(b'External fixture must never transfer')
for name,external in [('internal_run',False),('external_run',True)]:
 o=r/'output/sr4d_20260922'/name;p=o/'wu_sr_w01';p.mkdir(parents=True)
 for step in ('6000','18000','51800'):(p/('checkpoint_'+step+'.pt')).write_bytes(('transport checkpoint '+step).encode())
 (p/'checkpoint_final.pt').symlink_to(outside if external else 'checkpoint_51800.pt')
 (o/'status.json').write_text(json.dumps({'scene':'fixture','manifest_path':str(manifest),'manifest_sha256':sha(manifest)}))
 (o/'completion_event.json').write_text(json.dumps({'status':'complete','finished_at':'CPU/network synthetic fixture','out':str(o)}))
print(r)
'''
command = ' '.join(shlex.quote(x) for x in ['python3', '-c', setup, remote])
subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', 'cts', command], check=True)
local = base / 'local_project'
hr = local / 'data/dynamic_sr/n3dv_prepared/fixture/hr/cam00/0000.png'
hr.parent.mkdir(parents=True)
hr.write_bytes(b'fixed HR transport fixture')
results = {}
for name in ['internal_run', 'internal_run', 'external_run']:
    label = name if name not in results else 'internal_retry'
    p = subprocess.run([sys.executable, str(HERE / 'collect_on_exit.py'), '--host', 'cts',
                        '--remote-root', remote, '--run', name, '--local-root', str(local)],
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    (base / (label + '.log')).write_text(p.stdout)
    state = json.loads(p.stdout)
    if name == 'external_run':
        assert p.returncode == 1 and state['status'] == 'failed', state
        assert not (local / 'output/sr4d_20260922/external_run').exists()
        log = next((local / 'output/sr4d_20260922/.collection_receipts/cts__external_run').glob('*/inventory.stderr.log'))
        assert 'External symlink target refused' in log.read_text()
        results[label] = {'status': 'correctly_rejected', 'returncode': p.returncode}
        continue
    assert p.returncode == 0, p.stdout
    receipt = json.loads(Path(state['receipt']).read_text())
    final = local / 'output/sr4d_20260922/internal_run/wu_sr_w01/checkpoint_final.pt'
    assert final.is_file() and not final.is_symlink()
    assert final.read_bytes() == b'transport checkpoint 51800'
    assert not final.with_name('checkpoint_51800.pt').exists()
    assert final.with_name('checkpoint_6000.pt').is_file()
    assert final.with_name('checkpoint_18000.pt').is_file()
    assert not (local / 'data/sr4d_author_20260922').exists()
    assert receipt['data_policy'] == 'explicit_manifest_references_only'
    assert receipt['manifest_references_verified'] == 3
    assert receipt['materialized_internal_file_links'] == {
        'output/sr4d_20260922/internal_run/wu_sr_w01/checkpoint_final.pt':
        'output/sr4d_20260922/internal_run/wu_sr_w01/checkpoint_51800.pt'}
    if label == 'internal_retry':
        assert receipt['new_files'] == 0
    results[label] = receipt
report = {'status': 'passed', 'kind': 'Synthetic CPU/network transport bytes, not training or real weights',
          'remote_root': remote, 'collector_sha256': hashlib.sha256((HERE / 'collect_on_exit.py').read_bytes()).hexdigest(),
          'results': results}
(base / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({'status': 'passed', 'evidence': str(base / 'verification.json'), 'results': results}, indent=2))
