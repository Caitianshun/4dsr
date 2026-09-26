"""Source-specific gradient routing; all original LR parameters stay trainable.

The optional switch cuts the *concatenated posed center* input, including native
SH direction derivatives. Shared-field gradients through other attributes remain.
"""
import torch
from shared import all_named
from motion_model import rasterize, covariance


def effective_state(model, time):
    assert model.branch == 'ordinary_split' and model.children is not None
    g, c = model.g, model.children
    def deform(xyz, scale, quat, opacity, sh):
        times = torch.full((len(xyz), 1), float(time), device=xyz.device, dtype=xyz.dtype)
        return g._deformation(xyz, scale, quat, opacity, sh, times)
    a = deform(g._xyz, g._scaling, g._rotation, g._opacity, g.get_features)
    b = deform(c.xyz(), c.logscale, c.quaternion, c.opacity, c.features())
    return dict(xyz=torch.cat((a[0], b[0])),
                cov=torch.cat((covariance(a[1], a[2]), covariance(b[1], b[2]))),
                opacity=torch.sigmoid(torch.cat((a[3], b[3]))),
                sh=torch.cat((a[4], b[4])))


def render_model(model, camera, detach_prior_position=False, trace=False):
    state = effective_state(model, camera.time)
    if trace:
        for value in state.values():
            if value.requires_grad:
                value.retain_grad()
    xyz = state['xyz'].detach() if detach_prior_position else state['xyz']
    result = rasterize(model.g, camera, xyz, state['cov'], state['opacity'], state['sh'])
    if trace:
        result['effective_state'] = state
    return result


def appearance(model):
    return {n: p for n, p in all_named(model).items()
            if n in ['_features_dc', '_features_rest', 'children.sh_dc', 'children.sh_rest']
            or (not model.h.no_dshs and n.startswith('deformation.deformation_net.shs_deform.'))}


def policy_info(model, arm):
    named = all_named(model)
    return dict(arm=arm, sr_appearance_names=sorted(appearance(model)),
                actual_trainable=sum(p.numel() for p in named.values() if p.requires_grad),
                stored_parameters=sum(p.numel() for p in named.values()),
                lr_full_path=True, global_freeze=False, switch_step=17550,
                indirect_center_change_allowed=True)


def position_locked(arm, step, smoke=False):
    return arm == 'P_early' or (arm == 'P_late' and (step > 17550 or smoke))


def backward_prior(loss, model, arm, step, smoke=False):
    if arm == 'A_late' and (step > 17550 or smoke):
        params = list(appearance(model).values())
        grads = torch.autograd.grad(loss, params, allow_unused=True)
        for p, grad in zip(params, grads):
            if grad is not None:
                if p.grad is None:
                    p.grad = grad
                else:
                    p.grad.add_(grad)
    else:
        loss.backward()


def group_name(name):
    if name in ['_features_dc', '_features_rest', 'children.sh_dc', 'children.sh_rest'] or '.shs_deform.' in name:
        return 'appearance'
    if name in ['_xyz', 'children.offset_raw'] or '.pos_deform.' in name:
        return 'center'
    if name in ['_scaling', '_rotation', 'children.logscale', 'children.quaternion'] or '.scales_deform.' in name or '.rotations_deform.' in name:
        return 'shape'
    if name in ['_opacity', 'children.opacity'] or '.opacity_deform.' in name:
        return 'opacity'
    return 'shared_field'
