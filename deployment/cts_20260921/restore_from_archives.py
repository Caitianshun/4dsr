"""Restore only missing raw members, then verify exact original HR/LR hashes."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
VERIFY = ROOT/'deployment/cts_20260921/verification'
spec = json.loads((VERIFY/'archive_hashes.json').read_text())
restored = []
for scene, expected in spec.items():
    archive=ROOT/f'data/dynamic_sr/n3dv_archives/{scene}.zip'
    h=hashlib.sha256()
    with archive.open('rb') as handle:
        for b in iter(lambda:handle.read(8<<20),b''):h.update(b)
    assert h.hexdigest()==expected, archive
    raw=ROOT/'data/dynamic_sr/n3dv_raw'
    with zipfile.ZipFile(archive) as z:
        for member in z.infolist():
            p=PurePosixPath(member.filename)
            assert not p.is_absolute() and '..' not in p.parts and p.parts[0]==scene,p
            target=raw/p
            if member.is_dir():target.mkdir(parents=True,exist_ok=True);continue
            target.parent.mkdir(parents=True,exist_ok=True)
            assert not target.is_symlink()
            if target.exists():continue
            with z.open(member) as source, target.open('xb') as destination:
                shutil.copyfileobj(source,destination,8<<20)
            restored.append(str(p))
    print('ARCHIVE_OK',scene,flush=True)
(VERIFY/'raw_archive_restore.json').write_text(json.dumps(dict(status='complete',archives=spec,restored=restored),indent=2)+'\n')
subprocess.run([sys.executable,str(ROOT/'deployment/a100_20260920/restore_images.py'),
    '--project-root',str(ROOT),'--scene','cook_spinach','--scene','cut_roasted_beef',
    '--allow-regenerate','--report',str(VERIFY/'restore_images.json')],check=True,cwd=ROOT)
