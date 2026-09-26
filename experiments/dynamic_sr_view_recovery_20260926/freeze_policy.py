"""Freeze by Parameter identity, preserving Adam state and dynamic forward flags."""
from shared import all_named, digest_state, gamma_digest

def apply_policy(model, arm):
    assert arm in ['C_joint','F_app']
    named=all_named(model)
    allowed={n for n in named if n in ['_features_dc','_features_rest','children.sh_dc','children.sh_rest']
             or (not model.h.no_dshs and n.startswith('deformation.deformation_net.shs_deform.'))}
    for n,p in named.items():
        # Preserve intrinsically nontrainable buffers exposed as Parameters (AABB).
        if arm=='F_app':
            p.requires_grad_(n in allowed)
        p.grad=None
    actual={n for n,p in named.items() if p.requires_grad}
    if arm=='F_app': assert actual==allowed
    return dict(schema=1,arm=arm,allowed_names=sorted(actual),
        inactive_color_head=bool(model.h.no_dshs),
        frozen_names=sorted(set(named)-actual),
        actual_trainable=sum(p.numel() for p in named.values() if p.requires_grad),
        stored_parameters=sum(p.numel() for p in named.values()),
        rule='Apply after EVERY load; frozen grad is None; keep inherited per-parameter Adam state')

def frozen_state(model):
    named=all_named(model)
    params={n:p for n,p in named.items() if not p.requires_grad}
    states={}
    for opt in [model.g.optimizer,model.child_optimizer]:
        for n,p in params.items():
            if p in opt.state: states[n]=opt.state[p]
    buffers={'deformation.'+n:b for n,b in model.g._deformation.named_buffers()}
    buffers.update({'children.'+n:b for n,b in model.children.named_buffers()})
    for n in ['_deformation_table','_deformation_accum','max_radii2D','xyz_gradient_accum','denom']:
        if hasattr(model.g,n): buffers['base.'+n]=getattr(model.g,n)
    return dict(parameters_sha256=digest_state(params),buffers_sha256=digest_state(buffers),
        adam_per_parameter_sha256=digest_state(states))

def assert_no_frozen_grad(model):
    bad=[n for n,p in all_named(model).items() if not p.requires_grad and p.grad is not None]
    assert not bad, bad

def color_digest(model):
    return digest_state({n:p for n,p in all_named(model).items() if p.requires_grad})
