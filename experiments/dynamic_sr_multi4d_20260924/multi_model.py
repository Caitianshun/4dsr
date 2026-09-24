"""Complete native three-branch inference from a full Multi4D checkpoint."""
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import torch
import full_state as fs

UPSTREAM=Path(os.environ.get('MULTI4D_UPSTREAM','/home/cai_tianshun/Project/Multi4D-c483e82'))
sys.path.insert(0,str(UPSTREAM))


def load_model(checkpoint_path,manifest=None,restore_rng=False):
    from scene.gaussian_model import GaussianModel_dynamic,GaussianModel,GaussianModelTransient
    from arguments import PipelineParams
    from argparse import ArgumentParser
    state=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
    hyper=SimpleNamespace(**state['hidden_config']);opt=SimpleNamespace(**state['optimizer_config'])
    fg=GaussianModel_dynamic(3,hyper);bg=GaussianModel(3);tr=GaussianModelTransient(3,gaussian_dim=4,time_duration=[0.,10.],rot_4d=True,sh_degree_t=2)
    for g,s in zip([fg,bg,tr],state['branches']):fs.restore_branch(g,s,opt)
    if restore_rng:fs.restore_rng(state['rng'])
    p=ArgumentParser();pp=PipelineParams(p);pipe=pp.extract(p.parse_args([]))
    return SimpleNamespace(g=fg,bg=bg,tr=tr,branches=[fg,bg,tr],pipe=pipe,branch='multi4d',
        checkpoint={'metadata':dict(state['metadata'],stage=state['stage'],completed_updates=state['completed_updates'])})


def capacity_summary(model):
    counts=[len(g._xyz) for g in model.branches]
    params,buffers=0,0
    # Full saved inference parameters + network buffers. Controller/Adam tensors excluded.
    for g in model.branches:
        for v in vars(g).values():
            if isinstance(v,torch.nn.Parameter):params+=v.numel()*v.element_size()
            elif isinstance(v,torch.nn.Module):
                params+=sum(p.numel()*p.element_size() for p in v.parameters())
                buffers+=sum(p.numel()*p.element_size() for p in v.buffers())
    return dict(stored_gaussians=sum(counts),per_branch_stored=dict(zip(['persistent_dynamic','static','transient'],counts)),
        inference_parameter_bytes=params,inference_network_buffer_bytes=buffers,inference_tensor_bytes=params+buffers,
        effective_per_frame_counts='recorded by render visibility diagnostics; stored points include inactive transient times')


def render_model(model,cam):
    from gaussian_renderer import render_rawall3pc
    result=render_rawall3pc(cam,model.g,model.bg,model.tr,model.pipe,
        torch.tensor([1e-7,1e-7,1e-7],device='cuda'),stage='fine',dropout_use=False)
    model.last_visibility={name:result[key] for name,key in [('persistent_dynamic','visibility_filter_fg'),('static','visibility_filter_bg'),('transient','visibility_filter')]}
    return result
