"""Relocatable execution paths; original provenance paths are never rewritten."""
from pathlib import Path
import hashlib
import json
import sys
import time
import os
import fcntl
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
OUT=ROOT/'output'/HERE.name
ORIGIN=Path('/home/cai_tianshun/Project/4dsr')
def local(path):
    p=Path(path)
    if not p.is_absolute():return ROOT/p
    try:return ROOT/p.relative_to(ORIGIN)
    except ValueError:return p

def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def write(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp')
    t.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n');t.replace(p)
def bound(entry):
    p=local(entry['path']);assert sha(p)==entry['sha256'],p;return p

def sources():
    names=['train.py','routing.py','schedule.py','dv_common.py','engineering.py']
    paths=[HERE/x for x in names]
    for folder,files in {
        'dynamic_sr_prior_guidance_20260927':['gradient_policy.py','shared.py'],
        'dynamic_sr_detail_supervision_20260924':['training_support.py','detail_loss.py'],
        'dynamic_sr_motion_bound_20260923':['motion_model.py'],
        'dynamic_sr_20260918':['common.py','n3dv_data.py','run_experiment.py'],
        'dynamic_sr_20260920':['resume_control.py']}.items():paths.extend(ROOT/'experiments'/folder/x for x in files)
    upstream=Path(os.environ.get('FOURDSR_UPSTREAM','/home/cai_tianshun/Project/4dgs'))
    result={str(p.relative_to(ROOT)):sha(p) for p in paths}
    result.update({'upstream/'+x:sha(upstream/x) for x in ['scene/gaussian_model.py','scene/deformation.py','scene/hexplane.py','gaussian_renderer/__init__.py']})
    return result

def setup():
    sys.path.insert(0,str(ROOT/'experiments/dynamic_sr_prior_guidance_20260927'))
    import shared
    sys.path.insert(0,str(HERE))
    return shared

def debit(kind,label,updates=0,adam=0,rgb=0,aux=0):
    path=OUT/'budget.json';lock=(OUT/'budget.lock').open('a')
    with lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        d=read(path) if path.exists() else dict(engineering_updates=0,formal_updates=0,adam_calls=0,training_rgb_forwards=0,auxiliary_forwards=0,by_run={})
        key=kind+'_updates';assert kind in ['engineering','formal'];limit=32 if kind=='engineering' else 36000
        assert d[key]+updates<=limit,(kind,d[key],updates,limit)
        d[key]+=updates;d['adam_calls']+=adam;d['training_rgb_forwards']+=rgb;d['auxiliary_forwards']+=aux
        row=d['by_run'].setdefault(label,dict(kind=kind,started_updates=0,adam_calls=0,rgb_forwards=0,auxiliary_forwards=0))
        row['started_updates']+=updates;row['adam_calls']+=adam;row['rgb_forwards']+=rgb;row['auxiliary_forwards']+=aux
        d['updated_unix']=time.time();write(path,d)
