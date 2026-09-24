"""Add world-space child-center residual AFTER shared deformation, before rasterization.

The old ordinary-split path is untouched. Rasterizer SH directions, visibility,
projection, depth sorting and covariance projection receive corrected means3D.
"""
from pathlib import Path
import sys
import torch
from torch import nn
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'experiments/dynamic_sr_motion_bound_20260923'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import motion_model as old
from time_basis import TimeBasis


class CenterResidual(nn.Module):
    def __init__(self, model, kind, manifest):
        super().__init__()
        c = model.children
        self.basis = TimeBasis(kind).to(model.g._xyz.device)
        self.coeff = nn.Parameter(torch.zeros(len(c.parent_ids), 8, 3, device=model.g._xyz.device))
        self.register_buffer('child_indices', torch.arange(len(model.g._xyz), len(model.g._xyz)+len(c.parent_ids), device=self.coeff.device))
        self.register_buffer('initial_parent_max_scale', torch.linalg.vector_norm(c.chart, dim=1).max(-1).values.detach().clone())
        mapping = {}
        for record in manifest['observations']:
            f, t = int(record['frame_index']), float(record['time'])
            if f in mapping and abs(mapping[f]-t)>1e-8: raise ValueError('Camera-dependent time map')
            mapping[f] = t
        if sorted(mapping) != list(range(0,120,2)) or any(abs(t-f/300)>1e-7 for f,t in mapping.items()):
            raise ValueError('Unexpected source time map')
        self.frame_times = mapping

    def camera_frame(self, camera):
        # Real frame identifier is present in the existing camera adapter.
        frame = float(camera.image_name.rsplit('/', 1)[-1])
        if abs(float(camera.time)-frame/300)>1e-7:
            raise ValueError('Frame identifier disagrees with original manifest time')
        return frame

    def displacement(self, frame):
        return torch.einsum('k,nkc->nc', self.basis(frame).to(self.coeff.dtype), self.coeff)


def xyz_group(model):
    found = [p for p in model.g.optimizer.param_groups if p['name']=='xyz']
    if len(found)!=1 or found[0]['params'][0] is not model.g._xyz:
        raise ValueError('Cannot resolve original world-coordinate _xyz optimizer group')
    return found[0]


def attach(model, kind, manifest):
    if model.branch != 'ordinary_split': raise ValueError('Geometry residual requires ordinary U split')
    model.geometry = CenterResidual(model, kind, manifest)
    base = xyz_group(model)
    settings = {k: base[k] for k in ['betas','eps','weight_decay','amsgrad','maximize','foreach','capturable','differentiable','fused'] if k in base}
    model.geometry_optimizer = torch.optim.Adam([dict(params=[model.geometry.coeff],name='world_center_coeff',lr=base['lr'])], **settings)
    return model


def synchronize_lr(model):
    value = xyz_group(model)['lr']
    model.geometry_optimizer.param_groups[0]['lr'] = value
    return value


def geometry_state(model):
    return dict(schema=1, kind=model.geometry.basis.kind, tensors=model.geometry.state_dict(),
                frame_times=model.geometry.frame_times, basis=model.geometry.basis.protocol(),
                optimizer=model.geometry_optimizer.state_dict(),
                units='world XYZ; no scale, rotation, clamp, projection or extra penalty',
                sh_directions='Original rasterizer evaluates SH using final corrected world means3D')


def load_model(checkpoint_path, manifest=None, restore_rng=False):
    model = old.load_model(checkpoint_path, manifest, restore_rng)
    saved = model.checkpoint.get('geometry_residual')
    if saved is None: return model
    if manifest is None: raise ValueError('Manifest required for residual frame map validation')
    attach(model, saved['kind'], manifest)
    if saved['frame_times'] != model.geometry.frame_times: raise ValueError('Changed frame map')
    expected = model.geometry.state_dict()
    for key in expected:
        if key != 'coeff' and not torch.equal(expected[key], saved['tensors'][key]):
            raise ValueError(f'Geometry buffer identity mismatch: {key}')
    model.geometry.load_state_dict(saved['tensors'], strict=True)
    model.geometry_optimizer.load_state_dict(saved['optimizer'])
    return model


def capacity_summary(model):
    result = old.capacity_summary(model)
    if not hasattr(model,'geometry'): return result
    p = model.geometry.coeff
    result.update(geometry_parameters=p.numel(), geometry_coefficient_bytes=p.numel()*p.element_size(),
                  geometry_buffer_bytes=sum(v.numel()*v.element_size() for v in model.geometry.buffers()))
    for key in ['trainable_parameters','effective_trainable_parameters']:
        result[key] += p.numel()
    # Inference tensor storage only: no Adam, gradients, metadata or inactive parent checkpoint copy.
    tensors = list(old.named_parameters(model.g).values()) + list(model.children.parameters()) + list(model.children.buffers()) + list(model.geometry.parameters()) + list(model.geometry.buffers())
    tensors += list(model.g._deformation.buffers())
    result['inference_tensor_bytes'] = sum(p.numel()*p.element_size() for p in tensors)
    return result


def render_model(model, camera, enabled=True):
    if not enabled or not hasattr(model,'geometry'): return old.render_model(model,camera)
    g,c = model.g,model.children
    def deform(xyz,scale,quat,opacity,sh):
        times=torch.full((len(xyz),1),float(camera.time),device=xyz.device,dtype=xyz.dtype)
        return g._deformation(xyz,scale,quat,opacity,sh,times)
    base=deform(g._xyz,g._scaling,g._rotation,g._opacity,g.get_features)
    child=deform(c.xyz(),c.logscale,c.quaternion,c.opacity,c.features())
    means = torch.cat((base[0], child[0]),0)
    delta = model.geometry.displacement(model.geometry.camera_frame(camera))
    means = means.index_add(0, model.geometry.child_indices, delta)
    cov = torch.cat((old.covariance(base[1],base[2]),old.covariance(child[1],child[2])),0)
    opacity = torch.sigmoid(torch.cat((base[3],child[3]),0))
    sh = torch.cat((base[4],child[4]),0)
    return old.rasterize(g,camera,means,cov,opacity,sh)


@torch.no_grad()
def residual_statistics(model):
    c = model.geometry.coeff.double()
    phi = model.geometry.basis.normalized60.double()
    mu = phi.mean(0)
    covariance = phi.T@phi/len(phi)-mu[:,None]*mu[None,:]
    mean = torch.einsum('k,nkc->nc',mu,c)
    static = mean.norm(dim=-1)
    temporal = torch.einsum('nkc,kl,nlc->n',c,covariance,c).clamp_min(0).sqrt()
    scale = model.geometry.initial_parent_max_scale.double()
    def quantiles(x):
        return dict(zip(['median','p90','p95','p99','max'],[float(torch.quantile(x,q)) for q in [.5,.9,.95,.99,1.]]))
    return dict(static_world=quantiles(static),temporal_world=quantiles(temporal),
                static_parent_scale=quantiles(static/scale),temporal_parent_scale=quantiles(temporal/scale),
                children=len(c),coefficient_l2=float(c.norm()),
                interpretation='All child coefficients; static mean and temporal RMS, not visible-point image benefit or true motion')
