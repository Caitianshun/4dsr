"""Uniform 8-bit PNG rendering/scoring; held-out HR is read only in this evaluator."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from common import ROOT, checked, identity, read, resolve, write_new


def render_observation(backend, obs, allow_heldout=False):
    cfg = backend.config
    state = backend.effective_state(obs['time'])
    camera = backend.camera(obs, 'hr', allow_heldout=allow_heldout)
    if cfg['method'] in ('iesrgs','threedsr'):
        if cfg['method'] == 'iesrgs':
            from methods.iesrgs_renderer import render_state
        else:
            from methods.threedsr_renderer import render_state
        # Filter reference cameras are TRAIN cameras only even for held-out rendering.
        references = [backend.camera(o, 'hr') for o in backend.observations if o['frame_index'] == obs['frame_index']]
        return render_state(camera,state,backend.background,backend.g.active_sh_degree,references)['render']
    return backend.native.native_rgb(camera,backend.g,state,backend.pipe,backend.background)['render']


def write_rgb(path, tensor):
    import numpy as np
    from PIL import Image
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        raise FileExistsError(p)
    pixels = tensor.detach().cpu().numpy()
    if pixels.ndim != 3 or pixels.shape[0] != 3 or not np.isfinite(pixels).all():
        raise ValueError('Invalid rendered RGB')
    Image.fromarray(np.rint(np.clip(pixels.transpose(1,2,0),0,1)*255).astype(np.uint8)).save(p)
    return str(p)


def render_training(backend, output):
    result = {}
    with backend.torch.no_grad():
        for obs in backend.observations:
            key = f"{obs['camera_id']}/{obs['frame_index']:04d}"
            result[key] = write_rgb(Path(output) / (key+'.png'), render_observation(backend,obs))
    return result


def evaluate(backend, checkpoint, out, camera='cam00'):
    import numpy as np
    import torch
    from PIL import Image
    from utils.loss_utils import ssim
    import sys
    evaluator_root = ROOT / 'third_party/static_sr_eval_20261022'
    if not (evaluator_root / 'lpipsPyTorch/__init__.py').is_file():
        raise FileNotFoundError('Prepare the pinned LPIPS evaluator dependency: ' + str(evaluator_root))
    sys.path.insert(0, str(evaluator_root))
    from lpipsPyTorch import LPIPS
    cache = ROOT / 'weight/static_sr_wu_20261022/evaluation/torch/hub'
    evaluator_weights = [
        dict(path=str(cache / 'checkpoints/alexnet-owt-7be5be79.pth'),
             sha256='7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02'),
        dict(path=str(cache / 'checkpoints/alex.pth'),
             sha256='df73285e35b22355a2df87cdb6b70b343713b667eddbda73e1977e0c860835c0'),
    ]
    for asset in evaluator_weights:
        checked(asset)
    torch.hub.set_dir(str(cache))
    # Same network and [0,1] input convention as the unified Wu scorer.
    # Instantiate once; the former convenience function reloaded it per frame.
    criterion = LPIPS(net_type='alex', version='0.1').to('cuda').eval()
    m = backend.manifest
    if camera not in m['splits']['test']:
        raise ValueError('This evaluator is only the fixed held-out split; train diagnostics must be labeled separately')
    rows=[]
    with torch.no_grad():
        for obs in m['observations']:
            if obs['camera_id'] != camera or obs['split'] != 'test':
                continue
            p = Path(out)/'renders'/camera/(f"{obs['frame_index']:04d}.png")
            write_rgb(p, render_observation(backend, obs, allow_heldout=True))
            pred = torch.from_numpy(np.asarray(Image.open(p).convert('RGB')).copy()).permute(2,0,1).float().cuda()/255
            gt_path=Path(m['_root'])/obs['hr_path']
            from common import sha
            if sha(gt_path) != obs['hr_sha256']:
                raise ValueError('Evaluation HR identity changed')
            gt=torch.from_numpy(np.asarray(Image.open(gt_path).convert('RGB')).copy()).permute(2,0,1).float().cuda()/255
            if pred.shape!=gt.shape:
                raise ValueError('Render/evaluation resolution differs')
            rows.append(dict(frame=obs['frame_index'],PSNR=float(-10*torch.log10((pred-gt).square().mean().clamp_min(1e-12))),
                SSIM=float(ssim(pred,gt)),LPIPS=float(criterion(pred[None],gt[None]).mean()),
                render=identity(p)))
    if len(rows)!=len(m['frame_indices']):
        raise ValueError('Held-out evaluation incomplete')
    result=dict(schema='static_sr_wu_uniform_png_metrics_v1',checkpoint=checkpoint,scene=m['scene'],camera=camera,
        quantization='np.rint(clip(rgb,0,1)*255), all methods same 8-bit PNG scorer',
        lpips_implementation=identity(evaluator_root / 'lpipsPyTorch/modules/lpips.py'),
        lpips_model_identities=evaluator_weights,
        lpips_input_range='[0,1], unified Wu scorer convention',
        metrics={k:float(np.mean([r[k] for r in rows])) for k in ('PSNR','SSIM','LPIPS')},frames=rows,
        visual_temporal_acceptance=False, training_view_metrics=False)
    write_new(Path(out)/'metrics.json',result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--checkpoint',required=True)
    p.add_argument('--out',required=True);a=p.parse_args()
    from backend import WuBackend
    out=resolve(a.out);out.mkdir(parents=True,exist_ok=False)
    backend=WuBackend(read(a.config),checkpoint=identity(a.checkpoint))
    evaluate(backend,identity(a.checkpoint),out)

if __name__=='__main__':main()
