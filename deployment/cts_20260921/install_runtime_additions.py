"""Install the recorded CPython 3.10 Tk binding inside the CTS project only."""
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[2]
D=ROOT/'deployment/cts_20260921'
assert Path(sys.prefix).resolve()==(ROOT/'.venv').resolve()
source=D/'runtime_additions'
mapping={source/'tkinter': ROOT/'.venv/lib/python3.10/site-packages/tkinter',
         source/'lib-dynload': ROOT/'.venv/lib/python3.10/site-packages',
         source/'runtime-libs': ROOT/'tools/runtime-libs'}
records=[]
for base,target in mapping.items():
    for path in sorted(base.rglob('*')):
        if not path.is_file() or '__pycache__' in path.parts:continue
        assert not path.is_symlink()
        dest=target/path.relative_to(base)
        dest.parent.mkdir(parents=True,exist_ok=True)
        assert not dest.is_symlink()
        shutil.copy2(path,dest)
        sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
        assert sha(path)==sha(dest)
        records.append(dict(source=str(path.relative_to(ROOT)),path=str(dest.relative_to(ROOT)),sha256=sha(dest)))
report=dict(status='passed',source_host='a100-train',source_python='Ubuntu CPython 3.10.12',
            source_debian_package='python3-tk 3.10.8-1~22.04; /lib/libBLT.2.5.so.8.6',
            reason='Upstream scene/deformation.py imports tkinter; no training code change.',files=records)
(D/'verification/runtime_additions.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(status='passed',files=len(records))))
