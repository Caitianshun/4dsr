"""Complete branch/optimizer/controller checkpoint, including non-module tensors."""
import hashlib
import os
import random
import numpy as np
import torch


def cpu(x):
    if isinstance(x,torch.Tensor):return x.detach().cpu().clone()
    if isinstance(x,dict):return {k:cpu(v) for k,v in x.items()}
    if isinstance(x,list):return [cpu(v) for v in x]
    if isinstance(x,tuple):return tuple(cpu(v) for v in x)
    return x


def digest(x):
    h=hashlib.sha256()
    def visit(v):
        h.update(type(v).__name__.encode()+b':')
        if isinstance(v,torch.Tensor):
            a=v.detach().cpu().contiguous();h.update(str((a.dtype,tuple(a.shape))).encode());h.update(a.numpy().tobytes())
        elif isinstance(v,np.ndarray):h.update(str((v.dtype,v.shape)).encode());h.update(v.tobytes())
        elif isinstance(v,dict):
            for k in sorted(v,key=lambda k:(type(k).__name__,str(k))):visit(k);visit(v[k])
        elif isinstance(v,(tuple,list)):
            for a in v:visit(a)
        else:h.update(repr(v).encode())
    visit(x);return h.hexdigest()


def rng():return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())
def restore_rng(s):
    random.setstate(s['python']);np.random.set_state(s['numpy']);torch.set_rng_state(s['torch']);torch.cuda.set_rng_state_all(s['cuda'])


def branch_state(g):
    result=dict(parameters={},tensors={},modules={},values={},callables=[])
    for k,v in vars(g).items():
        if k=='optimizer':continue
        if isinstance(v,torch.nn.Parameter):result['parameters'][k]=dict(value=cpu(v),requires_grad=v.requires_grad)
        elif isinstance(v,torch.Tensor):result['tensors'][k]=cpu(v)
        elif isinstance(v,torch.nn.Module):result['modules'][k]=dict(state=cpu(v.state_dict()),training=v.training)
        elif callable(v):result['callables'].append(k)
        elif isinstance(v,(int,float,str,bool,list,tuple,dict,type(None),np.ndarray,np.number)):result['values'][k]=cpu(v)
        else:raise TypeError(f'Uncaptured branch attribute {k}: {type(v)}')
    result['callables'].sort();result['optimizer']=cpu(g.optimizer.state_dict())
    return result


def restore_branch(g,s,opt):
    for k,v in s['values'].items():setattr(g,k,v)
    for k,v in s['parameters'].items():setattr(g,k,torch.nn.Parameter(v['value'].cuda(),requires_grad=v['requires_grad']))
    for k,v in s['modules'].items():
        module=getattr(g,k).cuda();module.load_state_dict(v['state']);module.train(v['training'])
    # Rebuild scheduler closures and parameter references using frozen native args.
    g.training_setup(opt)
    # setup intentionally resets buffers; restore all captured control statistics after it.
    for k,v in s['tensors'].items():setattr(g,k,v.cuda())
    for k,v in s['values'].items():setattr(g,k,v)
    g.optimizer.load_state_dict(s['optimizer'])
    assert digest(branch_state(g))==digest(s),'Full branch identity failed after reload'


def snapshot(branches,opt,hyper,stage,step,metadata,sampler):
    return dict(schema='multi4d_full_state_v1',branches=[branch_state(g) for g in branches],
        optimizer_config=vars(opt),hidden_config=vars(hyper),stage=stage,completed_updates=step,
        phase=3 if stage=='fine' and step>=10000 else 2,rng=cpu(rng()),metadata=metadata,sampler=sampler)


def save(path,state):
    tmp=path.with_suffix('.tmp')
    with tmp.open('wb') as f:torch.save(state,f);f.flush();os.fsync(f.fileno())
    tmp.replace(path)


def restore(branches,state,opt):
    assert state['schema']=='multi4d_full_state_v1'
    assert vars(opt)==state['optimizer_config']
    for g,s in zip(branches,state['branches']):restore_branch(g,s,opt)
    restore_rng(state['rng'])
    assert digest(rng())==digest(state['rng'])
