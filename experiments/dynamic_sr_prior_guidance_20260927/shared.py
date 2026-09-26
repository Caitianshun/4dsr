"""Shared identities and unchanged Wu rendering state for the recovery control."""
import json
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'output/dynamic_sr_prior_guidance_20260927'
sys.path.insert(0, str(ROOT / 'experiments/dynamic_sr_detail_supervision_20260924'))
from training_support import digest_state, initial_identity, load_teacher_index, TeacherCache
from motion_model import load_model, render_model, rasterize, covariance, capacity_summary, refinement_state
from common import named_parameters, appearance_parameters, sha256, write_json, UPSTREAM, downsample
from n3dv_data import load_manifest

def evaluator():
    import importlib.util
    spec=importlib.util.spec_from_file_location('recovery_original_evaluator',ROOT/'experiments/dynamic_sr_detail_supervision_20260924/evaluate.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

def read(path):
    return json.loads(Path(path).read_text())

def paths():
    old = read(ROOT / 'output/dynamic_sr_controlled_headroom_20260926/final_v1/methods.json')['methods']
    return dict(manifest=ROOT/'data/dynamic_sr/n3dv_prepared/cook_spinach/manifest.json',
        teacher=ROOT/'output/dynamic_sr_detail_supervision_20260924/teacher_inventory_v1/cook_spinach/teacher_index.json',
        schedule=ROOT/'output/dynamic_sr_multi4d_20260924/cook_wu_schedule_v1.json',
        lr_curve=ROOT/'output/dynamic_sr_multi4d_20260924/U40_v1/train/learning_rates.json',
        roi=ROOT/'output/dynamic_sr_soft_motion_20260924/roi_registry/cook_spinach/roi_protocol.json',
        methods=old)

def all_named(model):
    return {**named_parameters(model.g), **{'children.'+n:p for n,p in model.children.named_parameters()}}

@torch.no_grad()
def effective_state(model, time):
    """Same ordinary_split forward, before the existing shared rasterizer."""
    assert model.branch == 'ordinary_split'
    g, c = model.g, model.children
    def deform(xyz, scale, quat, opacity, sh):
        times=torch.full((len(xyz),1),float(time),device=xyz.device,dtype=xyz.dtype)
        return g._deformation(xyz,scale,quat,opacity,sh,times)
    a=deform(g._xyz,g._scaling,g._rotation,g._opacity,g.get_features)
    b=deform(c.xyz(),c.logscale,c.quaternion,c.opacity,c.features())
    return dict(xyz=torch.cat((a[0],b[0])),
        cov=torch.cat((covariance(a[1],a[2]),covariance(b[1],b[2]))),
        opacity=torch.sigmoid(torch.cat((a[3],b[3]))),sh=torch.cat((a[4],b[4])))

def topology(model):
    return dict(base_count=len(model.g._xyz),child_count=len(model.children.parent_ids),
        point_order='base preserved keep_ids order, then original child order; parent_ids are NOT current base row IDs',
        fixed_buffers_sha256=digest_state(dict(model.children.named_buffers())),
        degree=model.g.active_sh_degree)

def gamma_digest(model):
    return {str(f):digest_state({k:v for k,v in effective_state(model,f/300).items() if k!='sh'}) for f in [0,40,80,118]}
