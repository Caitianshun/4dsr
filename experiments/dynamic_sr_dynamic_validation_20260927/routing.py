"""Route only the added SR derivative, preserving LR/reg gradients and both Adams."""
from dv_common import setup
shared=setup()
import torch
from gradient_policy import effective_state, render_model
all_named=shared.all_named
APP=['_features_dc','_features_rest','children.sh_dc','children.sh_rest']
SHAPE=['_scaling','_rotation','children.logscale','children.quaternion']

def selected(model,arm):
    assert arm in ['J_joint','A_sh','S_cov'];assert model.branch=='ordinary_split'
    named=all_named(model)
    assert len({id(p) for p in named.values()})==len(named),'Aliased parameters in effective namespace'
    names=set(APP)
    if not model.h.no_dshs:names.update(n for n in named if n.startswith('deformation.deformation_net.shs_deform.'))
    if arm=='S_cov':
        names.update(SHAPE)
        for flag,head in [('no_ds','scales_deform'),('no_dr','rotations_deform')]:
            if not getattr(model.h,flag):names.update(n for n in named if n.startswith('deformation.deformation_net.'+head+'.'))
    if arm=='J_joint':names={n for n,p in named.items() if p.requires_grad}
    assert names<=set(named)
    chosen={n:p for n,p in named.items() if n in names and p.requires_grad}
    if arm!='J_joint':
        # Inspect module aliases without PyTorch's default deduplication.
        allowed_ids={id(p) for p in chosen.values()}
        prefixes=('shs_deform.',)+(('scales_deform.','rotations_deform.') if arm=='S_cov' else ())
        for n,p in model.g._deformation.named_parameters(remove_duplicate=False):
            head=n.removeprefix('deformation_net.')
            if not head.startswith(prefixes):assert id(p) not in allowed_ids,('shared output head',n)
    return chosen

def policy_info(model,arm):
    named=all_named(model);chosen=selected(model,arm)
    ids={id(p):n for n,p in named.items()}
    opts=[]
    for name,opt in [('base',model.g.optimizer),('children',model.child_optimizer)]:
        opts.append(dict(name=name,state_sha256=shared.digest_state(opt.state_dict()),groups=[dict(name=g['name'],lr=g['lr'],parameters=[ids[id(p)] for p in g['params']],numel=sum(p.numel() for p in g['params'])) for g in opt.param_groups]))
    return dict(arm=arm,sr_names=sorted(chosen),sr_reachable_parameters=sum(p.numel() for p in chosen.values()),stored_parameters=sum(p.numel() for p in named.values()),all_trainable_parameters=sum(p.numel() for p in named.values() if p.requires_grad),parameter_shapes={n:list(p.shape) for n,p in named.items()},configuration={k:getattr(model.h,k,None) for k in ['no_ds','no_dr','no_dshs','no_dx','no_do','apply_rotation','static_mlp','empty_voxel']},lr_full_path=True,position_detach=False,global_freeze=False,shared_head_alias_check=True,optimizers=opts,interpretation='Only incremental SR gradients are restricted; LR and inherited Adam moments may change centers.')

def backward_prior(loss,model,arm,audit=False):
    if arm=='J_joint':loss.backward();return None
    chosen=selected(model,arm);params=list(chosen.values())
    grads=torch.autograd.grad(loss,params,allow_unused=True)
    actual={n:None if g is None else g.detach().clone() for n,g in zip(chosen,grads)} if audit else None
    for p,g in zip(params,grads):
        if g is None:continue
        if p.grad is None:p.grad=g
        else:p.grad.add_(g)

    return actual
