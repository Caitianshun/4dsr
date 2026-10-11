"""Service smoke test: pinned runtime import, CUDA render vs saved official float, LPIPS offline load, one backward."""
import argparse, json, socket, sys, time
from pathlib import Path
import rt

p = argparse.ArgumentParser()
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
t0 = time.time()
result = dict(host=socket.gethostname(), started_unix=t0, runtime=rt.check_runtime())
import numpy as np
import torch
from types import SimpleNamespace
import full_refine_registered_v2 as native
import full_evaluate_registered_v2 as ev
from scene.gaussian_model import GaussianModel

result['device'] = torch.cuda.get_device_name()
result['visible'] = torch.cuda.device_count()
run = rt.BASE / 'full_SR_refinement/Cook_both_seeds_actual_revisionB_1791421128150050152/cook_spinach/seed_20261007/B0'
evald = rt.BASE / 'full_materialized/Cook_both_seeds_actual_revisionB_1791421128150050152/cook_spinach/seed_20261007/B0/evaluation'
plan = json.load(open(run / 'config.json'))
m = json.load(open(rt.DATA / 'cook_spinach/manifest_train_ready_seed20261007.json'))
m['_root'] = str(rt.DATA / 'cook_spinach')
payload = torch.load(run / 'checkpoint_06000.pt', map_location='cuda', weights_only=False)
h = SimpleNamespace(**plan['configuration']['ModelHiddenParams'])
g = GaussianModel(plan['configuration']['ModelParams']['sh_degree'], h)
(g.active_sh_degree, xyz, deform_state, g._deformation_table, f_dc, f_rest, scaling, rotation, opacity,
 _r, _a, _d, _o, g.spatial_lr_scale) = payload['model']
g._deformation.load_state_dict(deform_state); g._deformation = g._deformation.cuda().eval()
for name, t in (('_xyz', xyz), ('_features_dc', f_dc), ('_features_rest', f_rest), ('_scaling', scaling), ('_rotation', rotation), ('_opacity', opacity)):
    setattr(g, name, torch.nn.Parameter(t.detach().clone().float().cuda(), requires_grad=True))
pipe = SimpleNamespace(**plan['configuration']['PipelineParams'])
bg = torch.ones(3, device='cuda')
rows = []
for f in (0, 150):
    obs = next(o for o in m['observations'] if o['camera_id'] == 'cam00' and int(o['frame_index']) == f)
    cam = native.hr_camera(ev.camera_item(m, obs, torch), f, m)
    with torch.no_grad():
        st = native.native_effective_state(g, float(obs['time']), torch)
        img = native.native_rgb(cam, g, st, pipe, bg)['render']
    ref = torch.from_numpy(np.load(evald / f'floats/cam00_{f:04d}.npz')['rgb_raw']).cuda()
    ref = ref.permute(2, 0, 1) if ref.shape[-1] == 3 else ref
    d = (img - ref).abs()
    rows.append(dict(frame=f, mean_abs=float(d.mean()), max_abs=float(d.max()), shape=list(img.shape), img=img, obs=obs))

st = native.native_effective_state(g, 0.5, torch)
img = native.native_rgb(cam, g, st, pipe, bg)['render']
img.mean().backward()
result['backward_finite'] = all(bool(torch.isfinite(t.grad).all()) for t in (g._xyz, g._opacity, g._features_dc) if t.grad is not None)
import csv, lpips
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False; torch.backends.cudnn.benchmark = False
metric = lpips.LPIPS(net='alex', version='0.1', spatial=False, pnet_rand=False, pretrained=True).cuda().eval().requires_grad_(False)
legacy, closure, _, _ = ev.metric_helpers()
official = {int(r['frame']): r for r in csv.DictReader(open(evald / 'metrics_per_observation.csv'))}
for r in rows:
    raw, obs = r.pop('img'), r.pop('obs')
    hr = legacy.read_rgb(rt.DATA / 'cook_spinach' / obs['hr_path']); lr = legacy.read_rgb(rt.DATA / 'cook_spinach' / obs['lr_path'])
    pred = legacy.image_array(raw); empty = np.zeros(pred.shape[:2], bool)
    q = legacy.spatial_metrics(pred, hr, empty, metric, 'cuda')['full']; c = closure(raw, lr, empty)['full']
    o = official[r['frame']]
    r.update(psnr=q['psnr'], ssim=q['ssim'], lpips=q['lpips_alex'], LR_psnr=c['psnr'],
             d_psnr=q['psnr'] - float(o['psnr']), d_ssim=q['ssim'] - float(o['ssim']), d_lpips=q['lpips_alex'] - float(o['lpips']),
             d_LR_psnr=c['psnr'] - float(o['LR_diagnostic_psnr']))
result['render_vs_official_float'] = rows
result['alexnet_cache'] = str(Path(torch.hub.get_dir()))
result['peak_GB'] = torch.cuda.max_memory_allocated() / 1e9
result['seconds'] = time.time() - t0
rt.write_json(a.output / 'smoke.json', result)
print(json.dumps(result))
