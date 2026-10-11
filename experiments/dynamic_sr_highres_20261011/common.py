"""Fresh high-resolution validation namespace; no mutation of historical assets."""
from __future__ import annotations
import hashlib, importlib.util, json, os, shutil, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DATA = ROOT / 'data/dynamic_sr/highres_20261011'
OUT = ROOT / 'output/dynamic_sr_highres_20261011'
REGISTRY = OUT / 'registry'
CONDITIONS = {'dgF_cube': (2520, 1416), 'dgH_cube': (1260, 708), 'l3H_long360': (1868, 1048)}
SEEDS = (20261007, 20261008)
sys.dont_write_bytecode = True
def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()
def read(path): return json.loads(Path(path).read_text())
def local(path):
    p = Path(path)
    return p.resolve() if p.is_absolute() else (ROOT / p).resolve()
def entry(path):
    p = local(path)
    return dict(path=str(p.relative_to(ROOT)), sha256=sha(p))
def mkdir(path):
    p=Path(path); p.mkdir(parents=True, exist_ok=True); p.chmod(0o2775); return p
def write(path, value):
    """Create immutable artifact, accepting identical replay only."""
    p=Path(path); mkdir(p.parent)
    data=(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode()
    if p.exists():
        if p.read_bytes()!=data: raise FileExistsError(f'Preserve differing artifact: {p}')
        return p
    temp=p.with_name(p.name+f'.{os.getpid()}.partial')
    with temp.open('xb') as f: f.write(data); f.flush(); os.fsync(f.fileno())
    os.link(temp,p); temp.unlink(); p.chmod(0o664); return p
def module(name, path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec); sys.modules[name]=m; spec.loader.exec_module(m); return m
def disk_guard(path, estimate_gb):
    p=Path(path)
    while not p.exists(): p=p.parent
    free=shutil.disk_usage(p).free
    if free-estimate_gb*10**9 < 60*10**9:
        raise RuntimeError(f'Disk guard: free={free/10**9:.2f} GB, estimated={estimate_gb:.2f} GB, reserve=60 GB')
    return dict(free_bytes=free, estimated_bytes=int(estimate_gb*10**9), reserve_bytes=60*10**9)
