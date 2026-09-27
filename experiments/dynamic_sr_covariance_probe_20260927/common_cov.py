"""Explicit AB600 parent and unchanged fixed-time renderer/data adapters."""
import random
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PREVIOUS=ROOT/'experiments/dynamic_sr_conflict_probe_20260927'
sys.path.insert(0,str(PREVIOUS))
from probe_common import *
import probe_common as prior
assert Path(prior.__file__).resolve()==PREVIOUS/'probe_common.py'
HERE=Path(__file__).resolve().parent
OUT=ROOT/'output/dynamic_sr_covariance_probe_20260927'

def sources():
    return {**prior.sources(),**{str(f):sha(f) for f in HERE.glob('*.py')}}

def initial(p):
    entry=p['parent'];assert sha(entry['path'])==entry['sha256']
    ck=torch.load(entry['path'],map_location='cpu',weights_only=False)
    assert ck['metadata']['arm']=='AB' and ck['metadata']['step']==600 and ck['metadata']['frame']==40
    m=fixed.load_baked(entry['path'])
    assert (len(m.xyz),m.base_count,m.degree)==(132972,88648,3)
    assert model_hash(m)==entry['model_hash']
    return m

def configure(model,p,active_names):
    active=list(active_names);names=dict(model.named_parameters())
    assert len(set(active))==len(active) and set(active)<=set(names)
    for n,v in names.items():v.grad=None;v.requires_grad_(n in active)
    groups=[]
    for n in active:
        cfg=p['rates'][n];assert set(cfg)=={'lr','betas','eps'}
        groups.append(dict(name=n,params=[names[n]],**cfg))
    opt=torch.optim.Adam(groups,lr=0.)
    assert not opt.state
    assert {id(v) for g in opt.param_groups for v in g['params']}=={id(names[n]) for n in active}
    return opt

def covariance(m):
    n=m.base_count
    return torch.cat((motion.covariance(m.logscale[:n],m.quaternion[:n]),motion.covariance(m.logscale[n:],m.quaternion[n:])))

def contract_state(m,arm):
    names=['xyz','opacity']+(['logscale','quaternion'] if arm=='C' else [])
    state={n:cpu(getattr(m,n)) for n in names}
    state.update(base_count=m.base_count,points=len(m.xyz),degree=m.degree,point_order=cpu(torch.arange(len(m.xyz))))
    state['activated_opacity']=cpu(m.opacity.sigmoid())
    if arm=='C':state['covariance']=cpu(covariance(m))
    return state

def check_contract(m,arm,frozen_hash):
    assert digest(contract_state(m,arm))==frozen_hash
    for n,v in m.named_parameters():
        assert bool(torch.isfinite(v).all()),n
        if not v.requires_grad:assert v.grad is None,n

def rng_state():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())

def set_rng(s):
    random.setstate(s['python']);np.random.set_state(s['numpy']);torch.set_rng_state(s['torch']);torch.cuda.set_rng_state_all(s['cuda'])

def snapshot(model,opt,p,arm,repeat,step):
    return dict(baked=fixed.state(model),optimizer=cpu(opt.state_dict()),rng=rng_state(),sampler=dict(position=step,sequence=p['sequences'][str(repeat)],sequence_sha=digest(p['sequences'][str(repeat)])),metadata=dict(stage='A',arm=arm,repeat=repeat,step=step,frame=40,parent_sha256=p['parent']['sha256'],protocol_sha256=sha(p['_protocol_path'])))

def restore(path,p,arm,repeat):
    ck=torch.load(path,map_location='cpu',weights_only=False);m=fixed.load_baked(path);opt=configure(m,p,p['active'][arm]);opt.load_state_dict(ck['optimizer']);set_rng(ck['rng'])
    assert ck['metadata']['parent_sha256']==p['parent']['sha256'] and ck['metadata']['arm']==arm and ck['metadata']['repeat']==repeat
    assert ck['sampler']['sequence']==p['sequences'][str(repeat)] and ck['sampler']['sequence_sha']==digest(p['sequences'][str(repeat)])
    return m,opt,ck['sampler']['position']

def update(model,opt,data,c,p,ledger):
    opt.zero_grad(set_to_none=True)
    r=render(model,data[c]['camera']);ra=render(model,data['cam02']['camera']);rb=render(model,data['cam03']['camera'])
    lr=(downsample(r,(252,336))-data[c]['lr']).abs().mean();a=(ra-data['cam02']['teacher']).abs().mean();b=(rb-data['cam03']['teacher']).abs().mean();loss=lr+.05*a+.05*b
    assert bool(torch.isfinite(loss));loss.backward()
    for n,v in model.named_parameters():
        if v.requires_grad:assert v.grad is not None and bool(torch.isfinite(v.grad).all()),n
        else:assert v.grad is None,n
    ledger['started_updates']+=1;write(ledger['path'],ledger)
    opt.step();ledger['adam_calls']+=1;write(ledger['path'],ledger)
    torch.cuda.synchronize()
    return dict(LR=float(lr),teacher_a=float(a),teacher_b=float(b),loss=float(loss),camera=c,forwards=3)

def regions(p,c,pred,masks):
    out={'full':None}
    if c in masks:out['M']=masks[c]
    for name,xy in p['rois'].get(c,{}).items():
        x0,y0,x1,y1=[4*v for v in xy];mask=torch.zeros_like(pred[0],dtype=torch.bool);mask[y0:y1,x0:x1]=True;out[name]=mask
    return out
