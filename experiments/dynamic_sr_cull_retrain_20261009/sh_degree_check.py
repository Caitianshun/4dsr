"""Zero-training check: render with the active SH degree truncated to 0..3 (no parameter updates).

If held-out-view colour artifacts come from high-order SH extrapolation, truncating the degree at render time
removes them on cam00 while costing some view-dependent fidelity elsewhere. cam00 every 10th frame (30) and the
first training camera (10 frames) are scored per degree with the official legacy metrics, the FFT low band and the
artifact area (blurred RGB error above a threshold, as in artifact_check.py).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import gaussian_filter

import rt
import cullretrain as cr
import full_refine_registered_v2 as native
import full_evaluate_registered_v2 as ev


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--plan', help='config.json for registered checkpoints')
    p.add_argument('--cull', action='store_true', help='post-hoc culling before rendering')
    p.add_argument('--out', required=True)
    p.add_argument('--degrees', type=int, nargs='+', default=[0, 1, 2, 3])
    a = p.parse_args()
    rt.check_runtime()  # pins torch.hub to the project-local AlexNet copy (the service has no network)
    import lpips
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    started = time.time()
    plan = json.loads(Path(a.plan).read_text()) if a.plan else torch.load(a.checkpoint, map_location='cpu', weights_only=False)['plan']
    g, _ = cr.load_for_render(a.checkpoint, plan)
    h, opt, pipe, sh = cr.configuration(plan)
    manifest_path, manifest = cr.plan_scene(plan)
    culled = cr.cull(g, cr.Culler(manifest), None, 'posthoc', optimizer_backed=False) if a.cull else None
    trained_degree = int(g.active_sh_degree)
    legacy, closure, _, fft = ev.metric_helpers()
    metric = lpips.LPIPS(net='alex', version='0.1', spatial=False, pnet_rand=False, pretrained=True, verbose=False).cuda().eval().requires_grad_(False)
    background = torch.ones(3, device='cuda') if cr.white_background(plan) else torch.zeros(3, device='cuda')
    root = Path(manifest['_root'])
    diag_cam = manifest['splits']['train'][0]
    obs = [o for o in manifest['observations'] if o['camera_id'] == 'cam00' and int(o['frame_index']) % 10 == 0]
    obs += [o for o in manifest['observations'] if o['split'] == 'train' and o['camera_id'] == diag_cam and int(o['frame_index']) in range(15, 300, 30)]
    obs.sort(key=lambda o: (o['camera_id'] != 'cam00', int(o['frame_index'])))
    pixels = {i: (legacy.read_rgb(root / o['hr_path']), legacy.read_rgb(root / o['lr_path'])) for i, o in enumerate(obs)}
    result = {}
    for degree in a.degrees:
        if degree > trained_degree:
            continue
        g.active_sh_degree = degree
        rows = []
        for i, o in enumerate(obs):
            with torch.no_grad():
                cam = native.hr_camera(ev.camera_item(manifest, o, torch), i, manifest)
                state = native.native_effective_state(g, float(o['time']), torch)
                raw = native.native_rgb(cam, g, state, pipe, background)['render'].detach()
            hr, lr = pixels[i]
            pred = legacy.image_array(raw)
            empty = np.zeros(pred.shape[:2], bool)
            q = legacy.spatial_metrics(pred, hr, empty, metric, 'cuda')['full']
            row = dict(camera=o['camera_id'], frame=int(o['frame_index']), psnr=q['psnr'], ssim=q['ssim'], lpips=q['lpips_alex'],
                       LR_psnr=closure(raw, lr, empty)['full']['psnr'])
            if o['camera_id'] == 'cam00':
                band = fft.decompose(pred, hr)
                row['low_psnr'] = band['low_psnr']
                err = gaussian_filter(np.sqrt(((np.clip(pred, 0, 1) - hr) ** 2).sum(-1)), 12.0)
                row['artifact_area'] = float((err > 0.12).mean())
            rows.append(row)
        mean = lambda cam, k: float(np.mean([r[k] for r in rows if (r['camera'] == 'cam00') == (cam == 'cam00')]))
        result[str(degree)] = dict(cam00={k: mean('cam00', k) for k in ('psnr', 'ssim', 'lpips', 'LR_psnr', 'low_psnr', 'artifact_area')},
                                   train={k: mean('train', k) for k in ('psnr', 'ssim', 'lpips', 'LR_psnr')}, rows=rows)
    g.active_sh_degree = trained_degree
    summary = dict(checkpoint=cr.entry(a.checkpoint), posthoc_cull=culled, trained_sh_degree=trained_degree, train_camera=diag_cam,
                   cam00_frames='every 10th (30)', degrees=result, seconds=time.time() - started, gpu=torch.cuda.get_device_name(),
                   sources={cr.rel(Path(__file__)): rt.sha(__file__)}, parameter_updates=0)
    rt.write_json(out / 'summary.json', summary)
    print(json.dumps({d: v['cam00'] for d, v in result.items()}))


if __name__ == '__main__':
    main()
