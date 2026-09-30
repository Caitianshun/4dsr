"""Check whether a resize residual is in the actual linear LR operator's nullspace."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from frequency import ROOT, OUT, read, write, sha


def resize(x, size):
    return F.interpolate(x[None], size=size, mode='bicubic', align_corners=False, antialias=True)[0]


def main():
    torch.set_num_threads(2)
    p = read(ROOT / 'output/dynamic_sr_same_observation_20260930/protocol.json')
    manifest = read(ROOT / p['manifest_path']); data = ROOT / p['manifest_path']; data = data.parent
    observations = {(o['camera_id'], o['frame_index']): o for o in manifest['observations']}
    rows = []
    for item in p['observations']:
        if item['camera'] not in ['cam02', 'cam06', 'cam12', 'cam18'] or item['frame'] not in [0, 40, 80, 118]: continue
        path = data / item['sr_path']; assert sha(path) == item['sr_sha256']
        sr = torch.from_numpy(np.asarray(Image.open(path).convert('RGB'), dtype=np.float64).transpose(2, 0, 1).copy()) / 255
        lrsize = tuple(reversed(manifest['resolutions']['lr'])); d = resize(sr, lrsize)
        h = sr - resize(d, sr.shape[-2:]); leak = resize(h, lrsize)
        rows.append(dict(camera=item['camera'], frame=item['frame'], sr_sha256=item['sr_sha256'],
                         d0_resize_residual_l1=float(leak.abs().mean()), d0_resize_residual_mse=float((leak ** 2).mean())))
    assert len(rows) == 16
    yy, xx = np.indices((64, 64)); wave = np.cos(2 * np.pi * (7 * xx + 7 * yy) / 64)
    x = torch.from_numpy(np.repeat(wave[None], 3, 0)); dx = resize(x, (16, 16))
    write(OUT / 'operator_check.json', dict(status='completed', script_sha256=sha(__file__),
          operator='Linear torch bicubic align_corners=False antialias=True, float64 CPU; no clamp/round',
          rows=rows, mean_d0_resize_residual_l1=float(np.mean([r['d0_resize_residual_l1'] for r in rows])),
          mean_d0_resize_residual_mse=float(np.mean([r['d0_resize_residual_mse'] for r in rows])),
          diagonal_wave=dict(fx=7/64, fy=7/64, radius=float(np.sqrt(2)*7/64), fft_band='mid',
                             lr_grid_rms=float(torch.mean(dx ** 2).sqrt()), hr_grid_rms=float(torch.mean(x ** 2).sqrt())),
          interpretation='H(T)=T-Up(D0(T)) is not an exact LR-nullspace projection; use the actual D0 and its adjoint for such a claim.',
          parameter_updates=0))
    print('Completed 16 actual-operator leakage checks')


if __name__ == '__main__': main()
