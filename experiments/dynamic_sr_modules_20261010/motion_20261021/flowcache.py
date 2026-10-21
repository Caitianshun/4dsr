"""Optical-flow cache for the motion module of the SR stage (Claude, session 4dsr-25, 2026-10-21; takes over 4dsr-70's design).

Premise (tmp/claude_perscene_design_20261021/out/motion_premise_*.json, CPU, training cameras, D = 2 frames): in moving regions the
w4 parent's own Gaussian-centre motion is off by 1.98 / 0.92 HR px (Cook07 / Discussion07, EPE against RAFT on HR), RAFT on the LR
video by 2.93 / 1.68, RAFT on the matched SR teacher video by 0.75 / 0.39. This script precomputes those flows for every training
camera and frame pair (f -> f + D) so the SR stage can supervise the projected Gaussian velocity (modkit13.py).

Sources (all from training cameras only; cam00 is never read):
  teacher  RAFT on the matched SR teacher frames (sr_swinir_x4_areaft, the registered teacher index) at HR      -- the method
  lrup     RAFT on the bicubic x4 upsampled LR frames at HR (same resolution as the teacher)                    -- SR-specificity control
  lr       RAFT on the native LR frames, flow x 4                                                                  -- premise control
  hr       RAFT on the HR frames                                       -- DIAGNOSTIC ORACLE ONLY (training-view HR; never a method input)
Storage on the LR grid (HR px units): flow = 4x4 mean of the HR-grid flow (lr: LR flow x 4); fb = 4x4 max of the forward-backward
inconsistency |F(p) + B(p + F(p))| in HR px (lr: LR fb x 4), quantised to 0.1 px (uint8, 25.5 = saturated). Per camera five
.npy files (memory-mappable): <cam>.flow.npy (N, 2, h, w) float16 (x, y) = F (f -> f + D, on frame f's grid), <cam>.fb.npy (N, h, w)
uint8, <cam>.bflow.npy / <cam>.bfb.npy = the backward flow B (f + D -> f, on frame f + D's grid) and its own forward-backward error,
<cam>.frames.npy (N,) int16 = f of the pair f -> f + D; the .frames.npy file is written last and marks the camera complete.
RAFT: torchvision raft_large, weights C_T_SKHT_V2 (generic public weights; local copy, sha256 recorded), 12 updates, inputs in [-1, 1],
replicate padding to a multiple of 8, TF32 off.

  flowcache.py --prep data/dynamic_sr/full_time_20261007/prepared/cook_spinach_area4 --sources teacher lrup lr hr --D 2
               --out output/dynamic_sr_modules_20261010/motion_20261021/flows/cook_spinach_area4 [--cameras cam01 ...]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
WEIGHTS = ROOT / 'output/dynamic_sr_modules_20261010/motion_20261021/third_party/raft_large_C_T_SKHT_V2-ff5fadd5.pth'
MANIFEST = 'manifest_train_ready_seed20261007.json'
TEACHER_INDEX = 'teacher_index_sr_swinir_x4_areaft_seed20261007.json'


def sha(path, n=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(n), b''):
            h.update(block)
    return h.hexdigest()


def read_rgb(path):
    with Image.open(path) as im:
        return torch.from_numpy(np.asarray(im.convert('RGB'), dtype=np.float32) / 255.0).permute(2, 0, 1)


def raft_model(device):
    from torchvision.models.optical_flow import raft_large
    m = raft_large(weights=None)
    m.load_state_dict(torch.load(WEIGHTS, map_location='cpu'))
    return m.to(device).eval()


def pad8(x):
    h, w = x.shape[-2:]
    return F.pad(x, (0, (-w) % 8, 0, (-h) % 8), mode='replicate'), h, w


def warp(img, fl):
    """sample img (N,C,H,W) at p + fl(p) (fl in px, (N,2,H,W)); bilinear, border."""
    n, _, h, w = fl.shape
    gy, gx = torch.meshgrid(torch.arange(h, device=fl.device, dtype=fl.dtype), torch.arange(w, device=fl.device, dtype=fl.dtype),
                            indexing='ij')
    x = (gx + fl[:, 0]) * (2 / (w - 1)) - 1
    y = (gy + fl[:, 1]) * (2 / (h - 1)) - 1
    return F.grid_sample(img, torch.stack([x, y], -1), mode='bilinear', padding_mode='border', align_corners=True)


@torch.no_grad()
def pair_flows(model, a, b, iters):
    """a, b: (3,H,W) in [0,1] -> forward flow a->b and fb inconsistency at a's grid, both (.., H, W) in input px."""
    x1, h, w = pad8(torch.stack([a, b]) * 2 - 1)
    x2, _, _ = pad8(torch.stack([b, a]) * 2 - 1)
    fl = model(x1, x2, num_flow_updates=iters)[-1][:, :, :h, :w]
    fw, bw = fl[:1], fl[1:]
    fb = (fw + warp(bw, fw)).norm(dim=1)[0]
    bfb = (bw + warp(fw, bw)).norm(dim=1)[0]
    return fw[0], fb, bw[0], bfb


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--prep', required=True)
    p.add_argument('--sources', nargs='+', default=['teacher', 'lrup', 'lr'])
    p.add_argument('--D', type=int, default=2)
    p.add_argument('--out', required=True)
    p.add_argument('--cameras', nargs='*')
    p.add_argument('--iters', type=int, default=12)
    p.add_argument('--device', default='cuda')
    p.add_argument('--limit', type=int, default=0, help='smoke test: first N pairs per camera only')
    p.add_argument('--frames', type=int, nargs='*', help='only these f (pairs f -> f + D); default all')
    a = p.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = True
    prep = ROOT / a.prep
    man = json.loads((prep / MANIFEST).read_text())
    tix = json.loads((prep / TEACHER_INDEX).read_text())
    tpath = {(e['camera'], int(e['frame'])): prep / e['relative_path'] for e in tix['entries']}
    obs = {(o['camera_id'], int(o['frame_index'])): o for o in man['observations']}
    train = list(man['splits']['train'])
    cams = a.cameras or train
    if set(cams) - set(train):
        raise SystemExit('only training cameras are allowed')
    frames = sorted({int(o['frame_index']) for o in man['observations'] if o['camera_id'] == train[0]})
    hr_w, hr_h = man['resolutions']['hr']
    lr_w, lr_h = man['resolutions']['lr']
    k = hr_w // lr_w
    model = raft_model(a.device)
    out = ROOT / a.out
    started = time.time()
    record = dict(kind='flowcache', prep=a.prep, manifest_sha256=sha(prep / MANIFEST), teacher_index_sha256=sha(prep / TEACHER_INDEX),
                  weights=str(WEIGHTS.relative_to(ROOT)), weights_sha256=sha(WEIGHTS), D=a.D, iters=a.iters, cameras=cams,
                  source_code_sha256=sha(Path(__file__)), gpu=torch.cuda.get_device_name() if a.device == 'cuda' else 'cpu',
                  storage='LR grid; flow = 4x4 mean (HR px); fb = 4x4 max (HR px, uint8 x10)', sources={})

    def load(src, c, f):
        if src == 'teacher':
            return read_rgb(tpath[(c, f)])
        if src == 'hr':
            return read_rgb(prep / obs[(c, f)]['hr_path'])
        y = read_rgb(prep / obs[(c, f)]['lr_path'])
        if src == 'lr':
            return y
        return F.interpolate(y[None], size=(hr_h, hr_w), mode='bicubic', align_corners=False)[0].clamp(0, 1)  # lrup

    for src in a.sources:
        if src not in ('teacher', 'lrup', 'lr', 'hr'):
            raise SystemExit(f'unknown source {src}')
        sdir = out / f'{src}_D{a.D}'
        sdir.mkdir(parents=True, exist_ok=True)
        stats = {}
        t_src = time.time()
        for c in cams:
            path = sdir / f'{c}.frames.npy'
            if path.exists():
                continue
            pairs = [f for f in frames if f + a.D in frames and (not a.frames or f in a.frames)]
            if a.limit:
                pairs = pairs[:a.limit]
            flow = np.zeros((len(pairs), 2, lr_h, lr_w), np.float16)
            fbq = np.zeros((len(pairs), lr_h, lr_w), np.uint8)
            bflow = np.zeros((len(pairs), 2, lr_h, lr_w), np.float16)
            bfbq = np.zeros((len(pairs), lr_h, lr_w), np.uint8)
            cache = {}
            for i, f in enumerate(pairs):
                for g in (f, f + a.D):
                    if g not in cache:
                        cache[g] = load(src, c, g).to(a.device)
                fw, fb, bw, bfb = pair_flows(model, cache[f], cache[f + a.D], a.iters)
                for old in [g for g in cache if g <= f]:
                    cache.pop(old)
                for src_f, src_e, dst_f, dst_e in ((fw, fb, flow, fbq), (bw, bfb, bflow, bfbq)):
                    if src == 'lr':
                        f_lr, e_lr = src_f * k, src_e * k
                    else:
                        f_lr = F.avg_pool2d(src_f[None], k)[0]
                        e_lr = F.max_pool2d(src_e[None, None], k)[0, 0]
                    dst_f[i] = f_lr.cpu().numpy().astype(np.float16)
                    dst_e[i] = (e_lr * 10).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            for name, arr in (('flow', flow), ('fb', fbq), ('bflow', bflow), ('bfb', bfbq), ('frames', np.asarray(pairs, np.int16))):
                tmp = sdir / f'{c}.{name}.tmp.npy'
                np.save(tmp, arr)
                tmp.replace(sdir / f'{c}.{name}.npy')
            mag = np.linalg.norm(flow.astype(np.float32), axis=1)
            stats[c] = dict(pairs=len(pairs), mean_mag=float(mag.mean()), p99_mag=float(np.percentile(mag, 99)),
                            fb_lt1=float((fbq < 10).mean()), moving_fb_lt1=float((fbq[mag > 1] < 10).mean()) if (mag > 1).any() else None)
            print(json.dumps(dict(source=src, camera=c, **stats[c], elapsed=round(time.time() - started))), flush=True)
        record['sources'][src] = dict(dir=str(sdir.relative_to(ROOT)), seconds=time.time() - t_src, cameras=stats)
        (out / f'flowcache_{src}_D{a.D}.json').write_text(json.dumps(record['sources'][src], indent=1))
    record['seconds'] = time.time() - started
    (out / f'flowcache_D{a.D}_{"_".join(a.sources)}.json').write_text(json.dumps(record, indent=1))
    print('done', round(record['seconds']), flush=True)


if __name__ == '__main__':
    main()
