"""Pinned legacy imports for evidence repair; no ambiguous script-name imports."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = ROOT / 'output/dynamic_sr_evidence_repair_20260927'
PRIOR = ROOT / 'output/dynamic_sr_prior_guidance_20260927'

def module(name, relative):
    path = (ROOT / relative).resolve()
    if name in sys.modules:
        assert Path(sys.modules[name].__file__).resolve() == path, (name, path, sys.modules[name].__file__)
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result

motion = module('motion_model', 'experiments/dynamic_sr_motion_bound_20260923/motion_model.py')
from common import sha256, write_json, downsample, image_tensor, named_parameters, resized_camera, UPSTREAM
from n3dv_data import load_manifest
detail = module('detail_loss', 'experiments/dynamic_sr_detail_supervision_20260924/detail_loss.py')
support = module('training_support', 'experiments/dynamic_sr_detail_supervision_20260924/training_support.py')
# gradient_policy's explicit legacy dependency is pinned and verified here.
shared = module('shared', 'experiments/dynamic_sr_prior_guidance_20260927/shared.py')
policy = module('evidence_original_policy', 'experiments/dynamic_sr_prior_guidance_20260927/gradient_policy.py')
legacy_train = module('evidence_train18', 'experiments/dynamic_sr_20260918/run_experiment.py')
resume = module('evidence_resume20', 'experiments/dynamic_sr_20260920/resume_control.py')
evalmod = module('evidence_evaluate24', 'experiments/dynamic_sr_detail_supervision_20260924/evaluate.py')

read = lambda p: json.loads(Path(p).read_text())
all_named = shared.all_named
digest = support.digest_state

def paths():
    p = shared.paths()
    p['start'] = Path(read(PRIOR/'protocol.json')['start'])
    return p

def imports():
    return {n: str(Path(m.__file__).resolve()) for n,m in sys.modules.items()
            if getattr(m,'__file__',None) and '/experiments/' in str(m.__file__)}

def rng():
    import numpy as np, random
    return dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all(),numpy=np.random.get_state(),python=random.getstate())

def render(model, camera, renderer_id):
    if renderer_id == 'legacy_direct_v1': return motion.render_model(model,camera)
    if renderer_id == 'prior_refactor_v1': return policy.render_model(model,camera)
    raise ValueError(renderer_id)

def guard_training_images():
    def guard(event,args):
        if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
            s=os.fsdecode(args[0])
            if s.endswith('.png'):
                assert '/hr/' not in s and '/cam00/' not in s and '/cam01/' not in s, s
    sys.addaudithook(guard)

def cpu(x):
    if torch.is_tensor(x): return x.detach().cpu().clone()
    if isinstance(x,dict): return {k:cpu(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return type(x)(cpu(v) for v in x)
    return x

def optimizer_named(model):
    names={id(p):n for n,p in all_named(model).items()};rows={}
    for optlabel,opt in [('base',model.g.optimizer),('children',model.child_optimizer)]:
        for group in opt.param_groups:
            for p in group['params']:
                n=names[id(p)]
                rows[n]=dict(optimizer=optlabel,group=group['name'],shape=list(p.shape),lr=group['lr'],
                    betas=list(group['betas']),eps=group['eps'],state=cpu(opt.state.get(p,{})))
    assert set(rows)==set(names.values())
    return rows
