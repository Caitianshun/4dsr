"""Isolated execution identities and portable paths; no historical files edited."""
from pathlib import Path
import os, sys, json, hashlib, importlib.util
ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = ROOT / 'output' / HERE.name
ORIGIN = Path('/home/cai_tianshun/Project/4dsr')
def local(p):
    p=Path(p)
    if not p.is_absolute(): return ROOT/p
    try: return ROOT/p.relative_to(ORIGIN)
    except ValueError: return p
def read(p): return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()
def write(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    q=p.with_name(p.name+f'.{os.getpid()}.tmp')
    q.write_text(json.dumps(d,indent=2,ensure_ascii=False,allow_nan=False)+'\n');q.replace(p)
def entry(p):
    p=Path(p);return {'path':str(p.relative_to(ROOT)), 'sha256':sha(p)}
def bound(e):
    p=local(e['path']);assert sha(p)==e['sha256'],p;return p
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def setup():
    sys.path.insert(0,str(ROOT/'experiments/dynamic_sr_prior_guidance_20260927'))
    import shared
    import gradient_policy
    sys.path.insert(0,str(HERE))
    return shared,gradient_policy
def source_identity():
    folders={'dynamic_sr_confidence_geometry_20261006':['cg_common.py','schedules.py','train.py'],
             'dynamic_sr_prior_guidance_20260927':['shared.py','gradient_policy.py'],
             'dynamic_sr_detail_supervision_20260924':['training_support.py','detail_loss.py'],
             'dynamic_sr_motion_bound_20260923':['motion_model.py'],
             'dynamic_sr_20260918':['common.py','n3dv_data.py','run_experiment.py'],
             'dynamic_sr_20260920':['resume_control.py']}
    paths=[ROOT/'experiments'/folder/n for folder,names in folders.items() for n in names]
    d={str(p.relative_to(ROOT)):sha(p) for p in paths}
    upstream=Path(os.environ.get('FOURDSR_UPSTREAM','/home/cai_tianshun/Project/4dgs'))
    for n in ['scene/gaussian_model.py','scene/deformation.py','scene/hexplane.py','gaussian_renderer/__init__.py']:
        d['upstream/'+n]=sha(upstream/n)
    return d
