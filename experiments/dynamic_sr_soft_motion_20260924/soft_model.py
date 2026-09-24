"""Training-only frozen motion increments; inference is the unchanged B model."""
from pathlib import Path
import sys
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
MOTION = ROOT / 'experiments/dynamic_sr_motion_bound_20260923'
sys.path.insert(0, str(MOTION))
from motion_model import (capacity_summary, covariance, frame, load_model, make_model,
                          named_parameters, rasterize, refinement_state, render_model)


def deform_children(model, time):
    c = model.children
    xyz = c.xyz()
    times = torch.full((len(xyz), 1), float(time), device=xyz.device, dtype=xyz.dtype)
    return model.g._deformation(xyz, c.logscale, c.quaternion, c.opacity, c.features(), times)


def child_centers(model, time):
    return deform_children(model, time)[0]


def render_with_centers(model, camera):
    """Same operations/order as old ordinary_split, exposing an image ancestor."""
    assert model.branch == 'ordinary_split'
    g = model.g
    times = torch.full((len(g._xyz), 1), float(camera.time), device=g._xyz.device, dtype=g._xyz.dtype)
    base = g._deformation(g._xyz, g._scaling, g._rotation, g._opacity, g.get_features, times)
    child = deform_children(model, camera.time)
    xyz = torch.cat((base[0], child[0]), 0)
    cov = torch.cat((covariance(base[1], base[2]), covariance(child[1], child[2])), 0)
    opacity = torch.sigmoid(torch.cat((base[3], child[3]), 0))
    sh = torch.cat((base[4], child[4]), 0)
    package = rasterize(g, camera, xyz, cov, opacity, sh)
    package['child_centers'] = child[0]
    return package


class FrozenReference:
    def __init__(self, path, device='cuda'):
        state = torch.load(path, map_location='cpu', weights_only=False)
        self.metadata = state['metadata']
        self.frames = [int(x) for x in state['frames']]
        self.index = {f:i for i,f in enumerate(self.frames)}
        self.reference_frame = int(self.metadata['reference_frame'])
        self.reference_time = float(state['times'][self.index[self.reference_frame]])
        self.centers = state['reference_child_centers'].to(device).detach()
        self.length = state['child_length'].to(device).detach()
        self.valid = state['valid_children'].to(device).bool()
        self.initial_offset_raw = state['initial_offset_raw']
        self.parent_ids = state['child_parent_ids']
        assert not self.centers.requires_grad and not self.length.requires_grad
        assert bool(self.valid.any()) and bool(torch.isfinite(self.centers[:,self.valid]).all())
        assert bool((self.length[self.valid] > 0).all())

    def validate(self, model, parent_sha, selection_sha, manifest_sha):
        for key,value in [('parent_sha256',parent_sha),('selection_sha256',selection_sha),('manifest_sha256',manifest_sha)]:
            assert self.metadata[key] == value, key
        assert torch.equal(model.children.parent_ids.cpu(), self.parent_ids)
        assert torch.equal(model.children.offset_raw.detach().cpu(), self.initial_offset_raw)

    def residual(self, current, anchor, frame_index):
        i, r = self.index[int(frame_index)], self.index[self.reference_frame]
        v=self.valid
        return ((current[v]-anchor[v]) - (self.centers[i,v]-self.centers[r,v])) / self.length[v,None]

    def loss(self, current, anchor, frame_index):
        residual = self.residual(current, anchor, frame_index)
        return F.huber_loss(residual, torch.zeros_like(residual), reduction='mean', delta=1.0)


def child_geometry_statistics(model):
    with torch.no_grad():
        c=model.children
        xyz=c.xyz().reshape(-1,2,3)
        distance=(xyz[:,0]-xyz[:,1]).norm(dim=-1)
        scale=c.logscale.exp()
        offset=c.offset().abs()
        def stats(x):
            return {'min':float(x.min()),'median':float(x.median()),'max':float(x.max()),
                    'finite':bool(torch.isfinite(x).all())}
        return {'canonical_sibling_distance':stats(distance),'canonical_axis_scale':stats(scale),
                'absolute_local_offset':stats(offset),
                'offset_components_above_1_49_fraction':float((offset>1.49).float().mean())}
