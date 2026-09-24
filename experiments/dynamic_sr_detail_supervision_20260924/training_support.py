"""Identity and train-only diagnostics shared by preparation and training."""
from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import random
import sys
import time
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
MOTION = ROOT / 'experiments/dynamic_sr_motion_bound_20260923'
sys.path.insert(0, str(MOTION))
from motion_model import make_model, render_model, capacity_summary
from common import sha256, image_tensor, named_parameters, downsample
from detail_loss import teacher_loss


def read(path):
    return json.loads(Path(path).read_text())


def load_teacher_index(path, manifest, records):
    value = read(path)
    if value['manifest_sha256'] != sha256(manifest['_manifest_path']):
        raise ValueError('Teacher manifest mismatch')
    if not value.get('status', '').startswith('completed'):
        raise ValueError('Teacher inventory must be completed')
    entries = {(r['camera'], int(r['frame'])): r for r in value['entries']}
    expected = {(r['camera_id'], int(r['frame_index'])) for r in records}
    if set(entries) != expected or len(entries) != len(value['entries']):
        raise ValueError('Teacher inventory must contain exactly all train observations, no held-out camera')
    paths, identity = {}, []
    for i, r in enumerate(records):
        entry = entries[r['camera_id'], int(r['frame_index'])]
        path = Path(manifest['_root']) / entry['relative_path']
        if sha256(path) != entry['sha256'] or sha256(r['image_path']) != entry['lr_sha256']:
            raise ValueError(f'Teacher/source LR changed: {path}')
        paths[i] = path
        identity.append(dict(camera=r['camera_id'], frame=int(r['frame_index']), path=str(path),
                             relative_path=entry['relative_path'], sha256=entry['sha256'], lr_sha256=entry['lr_sha256']))
    return paths, identity


class TeacherCache:
    """Bounded float32 CPU LRU: identical original image_tensor conversion.

    Full-camera pools must not silently allocate tens of GB of RGB/H tensors.
    H(T) is computed online with no_grad, so there is no separate H cache.
    """
    def __init__(self, paths, shape, maximum=64):
        self.paths, self.shape, self.maximum = paths, tuple(shape), maximum
        self.cache, self.hits, self.misses = OrderedDict(), 0, 0

    def get(self, index):
        if index in self.cache:
            self.hits += 1
            value = self.cache.pop(index)
        else:
            self.misses += 1
            value = image_tensor(self.paths[index], device='cpu')
            if tuple(value.shape) != self.shape:
                raise ValueError(f'Teacher shape mismatch: {self.paths[index]}')
        self.cache[index] = value
        if len(self.cache) > self.maximum:
            self.cache.popitem(last=False)
        return value.cuda()


def digest_state(value):
    digest = hashlib.sha256()
    def visit(x):
        digest.update(type(x).__name__.encode() + b':')
        if isinstance(x, torch.Tensor):
            a = x.detach().cpu().contiguous()
            digest.update(str((a.dtype, tuple(a.shape))).encode())
            digest.update(a.numpy().tobytes())
        elif isinstance(x, np.ndarray):
            digest.update(str((x.dtype, x.shape)).encode()); digest.update(x.tobytes())
        elif isinstance(x, dict):
            for k in sorted(x, key=lambda z: (type(z).__name__, str(z))):
                visit(k); visit(x[k])
        elif isinstance(x, (tuple, list)):
            for a in x: visit(a)
        else:
            digest.update(repr(x).encode())
    visit(value)
    return digest.hexdigest()


def initial_identity(model):
    return dict(model_and_optimizer_sha256=digest_state((model.g.capture(), model.children.state_dict(),
                                                         model.child_optimizer.state_dict())),
                capacity=capacity_summary(model))


def parameter_groups(model):
    groups = {k: [] for k in ['base_geometry', 'base_appearance', 'shared_deformation',
                              'child_geometry', 'child_appearance']}
    for name, p in named_parameters(model.g).items():
        if not p.requires_grad:
            continue
        key = 'shared_deformation' if name.startswith('deformation.') else (
            'base_geometry' if name in ['_xyz','_scaling','_rotation'] else 'base_appearance')
        groups[key].append(p)
    for name, p in model.children.named_parameters():
        if not p.requires_grad:
            continue
        key = 'child_geometry' if name in ['offset_raw','logscale','quaternion'] else 'child_appearance'
        groups[key].append(p)
    return groups


def fixed_gradient_audit(model, records, cameras, cache, method, alpha, original_cameras):
    """Two fixed train observations, after the endpoint update; no .grad writes."""
    before = dict(torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all(),
                  numpy=np.random.get_state(), python=random.getstate())
    groups = parameter_groups(model)
    params = [p for group in groups.values() for p in group]
    result, started = [], time.monotonic()
    def norms(grads):
        rows, start = {}, 0
        for key, group in groups.items():
            part = grads[start:start+len(group)]; start += len(group)
            rows[key] = dict(l2=sum(float(g.detach().double().square().sum()) for g in part if g is not None)**.5,
                             parameters_with_gradient=sum(g is not None for g in part), parameters=len(part))
        return rows
    try:
        for frame in [0,118]:
            camera = sorted(original_cameras)[0]
            i = next(i for i,r in enumerate(records) if (r['camera_id'],int(r['frame_index'])) == (camera,frame))
            prediction = render_model(model,cameras[i])['render']
            target = cache.get(i); lr = records[i]['image'].cuda()
            teacher, rgb, detail = teacher_loss(prediction,target,method,alpha,lr.shape[-2:])
            lr_loss = (downsample(prediction,lr.shape[-2:])-lr).abs().mean()
            teacher_output = torch.autograd.grad(teacher,prediction,retain_graph=True)[0]
            lr_output = torch.autograd.grad(lr_loss,prediction,retain_graph=True)[0]
            tg = torch.autograd.grad(teacher,params,allow_unused=True,retain_graph=True)
            lg = torch.autograd.grad(lr_loss,params,allow_unused=True)
            result.append(dict(camera=camera,frame=frame,teacher_loss=float(teacher),lr_loss=float(lr_loss),
                               teacher_output_grad_norm=float(teacher_output.double().square().sum().sqrt()),
                               lr_output_grad_norm=float(lr_output.double().square().sum().sqrt()),
                               teacher_parameter_groups=norms(tg),lr_parameter_groups=norms(lg)))
        torch.cuda.synchronize()
    finally:
        torch.set_rng_state(before['torch']); torch.cuda.set_rng_state_all(before['cuda'])
        np.random.set_state(before['numpy']); random.setstate(before['python'])
    return dict(observations=result,seconds=time.monotonic()-started,parameter_updates=0,
                protocol='Post-update fixed first-original-teacher camera frames0/118; separate teacher-only and LR-only gradients; no original regularizer; .grad unmodified')
