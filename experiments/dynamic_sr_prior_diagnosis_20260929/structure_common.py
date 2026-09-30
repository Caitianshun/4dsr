"""Frozen-model diagnostic I/O. No optimizer updates or image-derived training masks."""
import hashlib, importlib.util, json, os, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'output/dynamic_sr_prior_diagnosis_20260929'
OLD=ROOT/'output/dynamic_sr_temporal_prior_20260928'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def write(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+f'.{os.getpid()}.tmp');t.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n');t.replace(p)
def local(p):
 p=Path(p)
 return p if p.is_absolute() else ROOT/p
def module(name,relative):
 p=ROOT/relative
 if name in sys.modules:return sys.modules[name]
 spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def setup():
 m=module('structure_motion','experiments/dynamic_sr_motion_bound_20260923/motion_model.py')
 ev=module('structure_eval','experiments/dynamic_sr_detail_supervision_20260924/evaluate.py')
 return m,ev

def registry():
 p=read(OLD/'protocol.json');b=read(OLD/'baseline_registry.json');idx=read(ROOT/'output/dynamic_sr_sync_multiview_20260928/checkpoint_index.json')
 entries={}
 for name,label in [('LR6k','LR-direct-HRrender'),('HR6k','HR-direct-6k')]:
  r=b['references'][label];entries[name]=dict(checkpoint=r['checkpoint'],supervision=r['supervision'],total_updates=r['total_updates'],expected_points=r['points'])
 entries['U6000']=dict(checkpoint=p['parent'],supervision='train LR and frozen SwinIR',total_updates=14200,expected_points=132972)
 entries['r1_J1']=dict(checkpoint=p['historical_J1']['1']['checkpoint'],supervision='train LR and frozen SwinIR',total_updates=20200,expected_points=132972)
 r=idx['checkpoints']['r1_Async2'];entries['r1_Async2']=dict(checkpoint=dict(path=r['checkpoint'],sha256=r['sha256']),supervision='train LR and frozen SwinIR',total_updates=20200,expected_points=132972)
 return entries

def effective_state(model,t,motion):
 import torch
 g=model.g
 def deform(x,s,q,o,h):return g._deformation(x,s,q,o,h,torch.full((len(x),1),float(t),device=x.device,dtype=x.dtype))
 states=[deform(g._xyz,g._scaling,g._rotation,g._opacity,g.get_features)]
 if model.children is not None:
  assert model.branch=='ordinary_split';c=model.children;states.append(deform(c.xyz(),c.logscale,c.quaternion,c.opacity,c.features()))
 return dict(xyz=torch.cat([s[0] for s in states]),cov=torch.cat([motion.covariance(s[1],s[2]) for s in states]),opacity=torch.cat([s[3] for s in states]).sigmoid(),sh=torch.cat([s[4] for s in states]))

def save_npz(p,**arrays):
 import numpy as np
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);temp=p.with_suffix('.npz.tmp')
 with temp.open('wb') as f:np.savez(f,**arrays)
 temp.replace(p)
