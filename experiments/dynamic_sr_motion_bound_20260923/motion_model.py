"""Bounded-budget spatial refinement; one rasterization for base and children.

Children share a fixed canonical chart in both arms. Ordinary children query
the existing deformation at their own canonical xyz. Bound children use the
live parent geometry and its complete affine covariance transport. Both arms
query child opacity/SH at the child's canonical xyz with the same network.
"""
from dataclasses import dataclass
import math
from pathlib import Path
import sys
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / 'experiments/dynamic_sr_20260918'
sys.path.insert(0, str(OLD))
from common import load_checkpoint, named_parameters, render_image

BRANCHES = ('ordinary_split', 'bound_split')


def rotation(q):
    q = torch.nn.functional.normalize(q, dim=-1)
    w, x, y, z = q.unbind(-1)
    return torch.stack((1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y),
                        2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x),
                        2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)), -1).reshape(-1, 3, 3)


def frame(logscale, quaternion):
    return rotation(quaternion) * logscale.exp()[:, None, :]


def covariance(logscale, quaternion):
    a = frame(logscale, quaternion)
    return a @ a.transpose(-1, -2)


def packed_covariance(cov):
    return torch.stack((cov[:,0,0],cov[:,0,1],cov[:,0,2],cov[:,1,1],cov[:,1,2],cov[:,2,2]), -1).contiguous()


class Children(nn.Module):
    def __init__(self, values):
        super().__init__()
        for name in ['offset_raw', 'logscale', 'quaternion', 'opacity', 'sh_dc', 'sh_rest']:
            self.register_parameter(name, nn.Parameter(values[name].detach().clone()))
        for name in ['origin', 'chart', 'chart_inverse', 'parent_ids', 'keep_ids']:
            self.register_buffer(name, values[name].detach().clone())

    def offset(self):
        return 1.5 * self.offset_raw.tanh()

    def xyz(self):
        return self.origin + (self.chart @ self.offset().unsqueeze(-1)).squeeze(-1)

    def features(self):
        return torch.cat((self.sh_dc, self.sh_rest), 1)


@dataclass
class MotionModel:
    g: object
    h: object
    o: object
    checkpoint: dict
    branch: str
    children: Children | None
    child_optimizer: object
    original_count: int
    initialization: dict


def child_optimizer(child, o):
    groups = [('offset_raw', .0025), ('logscale', o.scaling_lr), ('quaternion', o.rotation_lr),
              ('opacity', o.opacity_lr), ('sh_dc', o.feature_lr), ('sh_rest', o.feature_lr / 20)]
    return torch.optim.Adam([{'params':[getattr(child,n)],'lr':lr,'name':n} for n,lr in groups], lr=0., eps=1e-15)


def make_model(g, h, o, checkpoint, manifest, branch, selection):
    ids = selection['selected_ids'].to(g._xyz.device).long()
    n = len(g._xyz)
    assert ids.ndim == 1 and len(ids.unique()) == len(ids) and len(ids) > 0
    assert int(ids.min()) >= 0 and int(ids.max()) < n
    keep = torch.ones(n, dtype=torch.bool, device=g._xyz.device)
    keep[ids] = False
    child_ids = ids.repeat_interleave(2)
    with torch.no_grad():
        chart = frame(g._scaling[child_ids], g._rotation[child_ids])
        inv = torch.diag_embed(g._scaling[child_ids].neg().exp()) @ rotation(g._rotation[child_ids]).transpose(-1,-2)
        offsets = torch.zeros((len(child_ids), 3), device=chart.device)
        axis = g._scaling[child_ids].argmax(-1)
        offsets[torch.arange(len(child_ids), device=chart.device), axis] = torch.tensor([-.5,.5],device=chart.device).repeat(len(ids))
        values = dict(offset_raw=torch.atanh(offsets/1.5), logscale=g._scaling[child_ids]-math.log(1.6),
                      quaternion=g._rotation[child_ids], opacity=g._opacity[child_ids],
                      sh_dc=g._features_dc[child_ids], sh_rest=g._features_rest[child_ids],
                      origin=g._xyz[child_ids], chart=chart, chart_inverse=inv,
                      parent_ids=child_ids, keep_ids=torch.where(keep)[0])
    children = Children(values)
    mapping = {'xyz':'_xyz','f_dc':'_features_dc','f_rest':'_features_rest',
               'scaling':'_scaling','rotation':'_rotation','opacity':'_opacity'}
    # Verify the preserved optimizer rows by value, not merely file identity.
    expected = {}
    for group in g.optimizer.param_groups:
        if group['name'] in mapping:
            state = g.optimizer.state.get(group['params'][0], {})
            expected[group['name']] = {k:(v[keep].clone() if torch.is_tensor(v) and v.ndim else v.clone() if torch.is_tensor(v) else v)
                                       for k,v in state.items()}
    if branch == 'ordinary_split':
        g.prune_points(~keep)
    elif branch != 'bound_split':
        raise ValueError(branch)
    for group in g.optimizer.param_groups:
        name = group['name']
        if name not in expected:
            continue
        for key, value in g.optimizer.state.get(group['params'][0], {}).items():
            actual = value[keep] if branch == 'bound_split' and torch.is_tensor(value) and value.ndim else value
            reference = expected[name][key]
            if torch.is_tensor(actual):
                assert torch.equal(actual, reference), (name, key)
            else:
                assert actual == reference
    if branch == 'bound_split':
        # These selected rows are stored only for compatibility with g.capture;
        # their appearance never renders. Prevent stale Adam momentum drift.
        for group in g.optimizer.param_groups:
            if group['name'] in ['f_dc', 'f_rest', 'opacity']:
                state = g.optimizer.state.get(group['params'][0], {})
                for key in ['exp_avg', 'exp_avg_sq']:
                    if key in state:
                        state[key][ids] = 0
    init = dict(optimizer_unselected_rows_exactly_preserved=True,
                child_optimizer='new separate Adam, empty state, step0; no parent moments',
                selected_parent_controller_state='all original rows retained; geometry moments inherited' if branch=='bound_split' else 'selected parent rows pruned',
                unused_selected_parent_appearance='SH/opacity moments zeroed at split, zero gradient, no weight decay; stored for checkpoint compatibility',
                chart='fixed original canonical parent R diag(s); not deformed time zero',
                position='same bounded canonical coordinate in B/C, componentwise 1.5*tanh; not unconstrained author split',
                split='at additional step0, antipodal +/-0.5 along largest canonical axis, scale/1.6, parent opacity copied; not exactly image preserving',
                covariance='C exact At A0^-1 Sigma_child A0^-T At^T; B covariance of own deformation output',
                selected_count=len(ids), original_count=n)
    return MotionModel(g,h,o,checkpoint,branch,children,child_optimizer(children,o),n,init)


def capacity_summary(model):
    child = model.children
    if child is None:
        return dict(active_gaussians=len(model.g._xyz), stored_gaussians=len(model.g._xyz), parent_control_count=0,
                    trainable_parameters=sum(p.numel() for p in named_parameters(model.g).values()), child_parameters=0)
    m = len(child.parent_ids)//2
    return dict(active_gaussians=model.original_count+m,
                stored_gaussians=len(model.g._xyz)+len(child.parent_ids),
                parent_control_count=m if model.branch=='bound_split' else 0,
                trainable_parameters=sum(p.numel() for p in named_parameters(model.g).values())+sum(p.numel() for p in child.parameters()),
                child_parameters=sum(p.numel() for p in child.parameters()),
                bound_unused_parent_appearance_scalars=m*49 if model.branch=='bound_split' else 0,
                effective_trainable_parameters=sum(p.numel() for p in named_parameters(model.g).values())+sum(p.numel() for p in child.parameters())-(m*49 if model.branch=='bound_split' else 0),
                fixed_chart_buffer_bytes=sum(b.numel()*b.element_size() for b in child.buffers()))


def refinement_state(model):
    return dict(schema=1,branch=model.branch,original_count=model.original_count,initialization=model.initialization,
                children=model.children.state_dict(),child_optimizer=model.child_optimizer.state_dict(),
                capacity=capacity_summary(model))


def load_model(checkpoint_path, manifest=None, restore_rng=False):
    g,h,o,ck = load_checkpoint(checkpoint_path)
    if 'motion_refinement' not in ck:
        if ck.get('surface_residual',{}).get('branch','joint') != 'joint':
            raise ValueError('Color residual checkpoint is not an A/B/C motion-refinement model')
        return MotionModel(g,h,o,ck,'joint',None,None,len(g._xyz),{})
    state = ck['motion_refinement']
    children = Children(state['children'])
    opt = child_optimizer(children,o)
    opt.load_state_dict(state['child_optimizer'])
    result = MotionModel(g,h,o,ck,state['branch'],children,opt,state['original_count'],state['initialization'])
    if restore_rng:
        sys.path.insert(0,str(ROOT/'experiments/dynamic_sr_20260920'))
        from resume_control import restore_global_rng
        restore_global_rng(ck['rng'])
    return result


def render_model(model, camera):
    g,c = model.g,model.children
    if c is None:
        return render_image(g,camera)
    def deform(xyz, scale, quat, opacity, sh):
        times = torch.full((len(xyz),1),float(camera.time),device=xyz.device,dtype=xyz.dtype)
        return g._deformation(xyz,scale,quat,opacity,sh,times)
    base = deform(g._xyz,g._scaling,g._rotation,g._opacity,g.get_features)
    child = deform(c.xyz(),c.logscale,c.quaternion,c.opacity,c.features())
    if model.branch == 'bound_split':
        parent_ids = c.parent_ids
        a = frame(base[1][parent_ids],base[2][parent_ids])
        xyz = base[0][parent_ids] + (a @ c.offset().unsqueeze(-1)).squeeze(-1)
        transported = a @ c.chart_inverse @ frame(c.logscale,c.quaternion)
        cov = transported @ transported.transpose(-1,-2)
        base = tuple(value[c.keep_ids] for value in base)
    else:
        xyz = child[0]
        cov = covariance(child[1],child[2])
    xyz = torch.cat((base[0],xyz),0)
    cov = torch.cat((covariance(base[1],base[2]),cov),0)
    opacity = torch.sigmoid(torch.cat((base[3],child[3]),0))
    sh = torch.cat((base[4],child[4]),0)
    return rasterize(g,camera,xyz,cov,opacity,sh)


def rasterize(g,camera,xyz,cov,opacity,sh):
    """Shared full-covariance rasterizer, including the native-path parity check."""
    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
    settings = GaussianRasterizationSettings(image_height=int(camera.image_height),image_width=int(camera.image_width),
        tanfovx=math.tan(camera.FoVx*.5),tanfovy=math.tan(camera.FoVy*.5),
        bg=torch.zeros(3,device=xyz.device),scale_modifier=1.,viewmatrix=camera.world_view_transform.to(xyz.device),
        projmatrix=camera.full_proj_transform.to(xyz.device),sh_degree=g.active_sh_degree,
        campos=camera.camera_center.to(xyz.device),prefiltered=False,debug=False)
    screenspace = torch.zeros_like(xyz,requires_grad=torch.is_grad_enabled())
    if screenspace.requires_grad:
        screenspace.retain_grad()
    image,radii,depth = GaussianRasterizer(raster_settings=settings)(means3D=xyz,means2D=screenspace,
        shs=sh.contiguous(),colors_precomp=None,opacities=opacity.contiguous(),scales=None,rotations=None,
        cov3D_precomp=packed_covariance(cov))
    return dict(render=image,viewspace_points=screenspace,visibility_filter=radii>0,radii=radii,depth=depth,
                active_gaussians=len(xyz))
