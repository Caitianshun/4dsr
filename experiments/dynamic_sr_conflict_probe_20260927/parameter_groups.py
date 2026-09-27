"""Named parameter units, SH-only optimization, and structural invariants."""
from probe_common import *
UNITS={'xyz':'world coordinate','logscale':'log world scale','quaternion':'unnormalized quaternion','opacity':'logit','sh_dc':'SH DC coefficient','sh_rest':'directional SH coefficients'}

def configure(model,protocol,all_attributes=False,sh_state=None):
    names=dict(model.named_parameters());active=list(protocol['rates']) if all_attributes else protocol['trainable']
    assert set(names)==set(protocol['rates'])
    for n,v in names.items():v.grad=None;v.requires_grad_(n in active)
    opt=torch.optim.Adam([dict(name=n,params=[names[n]],**protocol['rates'][n]) for n in active],lr=0.,eps=1e-15)
    if sh_state is not None:
        for g in sh_state['param_groups']:
            name=g['name'];assert name in ['sh_dc','sh_rest'];old=g['params'][0]
            for k in ['lr','eps']:assert g[k]==protocol['rates'][name][k]
            assert tuple(g['betas'])==tuple(protocol['rates'][name]['betas'])
            if old in sh_state['state']:
                opt.state[names[name]]={k:v.clone().to(names[name].device) if torch.is_tensor(v) and k!='step' else v.clone() if torch.is_tensor(v) else copy.deepcopy(v) for k,v in sh_state['state'][old].items()}
    assert {id(v) for g in opt.param_groups for v in g['params']}=={id(names[n]) for n in active}
    return opt

def geometry_state(model):
    with torch.no_grad():
        return dict(xyz=cpu(model.xyz),logscale=cpu(model.logscale),quaternion=cpu(model.quaternion),opacity=cpu(model.opacity),covariance=cpu(motion.covariance(model.logscale,model.quaternion)),activated_opacity=cpu(model.opacity.sigmoid()),point_order=cpu(torch.arange(len(model.xyz))),base_count=model.base_count,degree=model.degree)

def geometry_hash(model):return digest(geometry_state(model))

def assert_frozen(model,before):
    assert geometry_hash(model)==before,'Frozen effective structure changed'
    for n in ['xyz','logscale','quaternion','opacity']:
        v=getattr(model,n);assert not v.requires_grad and v.grad is None,n

def named_statistics(values,base_count):
    rows=[]
    for n,v in values.items():
        if v is None:
            rows.append(dict(parameter=n,part='all',none=True));continue
        for part,x in [('all',v),('base',v[:base_count]),('child',v[base_count:])]:
            if x is None:rows.append(dict(parameter=n,part=part,none=True));continue
            x=x.detach().double().flatten();finite=torch.isfinite(x);a=x[finite].abs();q=torch.quantile(a,torch.tensor([.5,.95],dtype=torch.float64,device=a.device)) if a.numel() else [0.,0.]
            rows.append(dict(parameter=n,part=part,unit=UNITS[n],none=False,count=x.numel(),nonfinite=int((~finite).sum()),l2=float(torch.linalg.vector_norm(x)),rms=float(x.square().mean().sqrt()),p50=float(q[0]),p95=float(q[1]),max=float(a.max()) if a.numel() else 0.))
    return rows
