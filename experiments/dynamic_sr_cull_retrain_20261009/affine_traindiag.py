"""Training-camera diagnostic with the learned per-camera colour affine applied (cullaff/gateaff runs).

The standard eval renders training cameras without their affine, which is not what training optimised.
This re-renders the same cam01 frames (15..285 step 30) and reports metrics both without and with the affine;
the 'without' numbers must reproduce the standard eval and serve as a consistency check.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import rt
import cullretrain as cr
import full_refine_registered_v2 as native
import full_evaluate_registered_v2 as ev


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    rt.check_runtime()  # pins torch.hub to the project-local AlexNet copy (the service has no network)
    import lpips
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    started = time.time()
    payload = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    if payload.get('affine') is None:
        raise ValueError('checkpoint has no colour affine')
    plan = payload['plan']
    g, _ = cr.load_for_render(a.checkpoint, plan)
    h, opt, pipe, sh = cr.configuration(plan)
    manifest_path, manifest = cr.plan_scene(plan)
    names, delta = payload['affine']['names'], payload['affine']['delta'].float().cuda()
    eff = delta - delta.mean(0)
    legacy, closure, _, _ = ev.metric_helpers()
    metric = lpips.LPIPS(net='alex', version='0.1', spatial=False, pnet_rand=False, pretrained=True, verbose=False).cuda().eval().requires_grad_(False)
    background = torch.ones(3, device='cuda') if cr.white_background(plan) else torch.zeros(3, device='cuda')
    cam = manifest['splits']['train'][0]
    d = eff[names.index(cam)]
    root = Path(manifest['_root'])
    rows = []
    obs_list = sorted([o for o in manifest['observations'] if o['split'] == 'train' and o['camera_id'] == cam
                       and int(o['frame_index']) in range(15, 300, 30)], key=lambda o: int(o['frame_index']))
    for i, obs in enumerate(obs_list):
        with torch.no_grad():
            camera = native.hr_camera(ev.camera_item(manifest, obs, torch), i, manifest)
            state = native.native_effective_state(g, float(obs['time']), torch)
            raw = native.native_rgb(camera, g, state, pipe, background)['render'].detach()
            corrected = torch.einsum('ij,jhw->ihw', torch.eye(3, device='cuda') + d[:, :3], raw) + d[:, 3, None, None]
        hr, lr = legacy.read_rgb(root / obs['hr_path']), legacy.read_rgb(root / obs['lr_path'])
        row = dict(camera=cam, frame=int(obs['frame_index']))
        for tag, img in (('raw', raw), ('affine', corrected)):
            pred = legacy.image_array(img)
            empty = np.zeros(pred.shape[:2], bool)
            q = legacy.spatial_metrics(pred, hr, empty, metric, 'cuda')['full']
            c = closure(img, lr, empty)['full']
            row.update({f'{tag}_psnr': q['psnr'], f'{tag}_ssim': q['ssim'], f'{tag}_lpips': q['lpips_alex'], f'{tag}_LR_psnr': c['psnr']})
        rows.append(row)
    mean = lambda k: float(np.mean([r[k] for r in rows]))
    summary = dict(checkpoint=cr.entry(a.checkpoint), camera=cam, frames=len(rows), affine_matrix=(torch.eye(3) + d[:, :3].cpu()).tolist(),
                   affine_offset=d[:, 3].cpu().tolist(), rows=rows, seconds=time.time() - started,
                   raw={k: mean(f'raw_{k}') for k in ('psnr', 'ssim', 'lpips', 'LR_psnr')},
                   affine={k: mean(f'affine_{k}') for k in ('psnr', 'ssim', 'lpips', 'LR_psnr')},
                   gpu=torch.cuda.get_device_name(), sources={cr.rel(Path(__file__)): rt.sha(__file__)})
    rt.write_json(out / 'summary.json', summary)
    print(json.dumps({k: summary[k] for k in ('camera', 'raw', 'affine', 'seconds')}))


if __name__ == '__main__':
    main()
