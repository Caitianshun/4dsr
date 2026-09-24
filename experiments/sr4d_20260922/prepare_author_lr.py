"""Create separate, author-code bilinear float LR inputs; never overwrite data."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def write(p, d):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix('.tmp')
    tmp.write_text(json.dumps(d, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source-manifest', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    torch.set_num_threads(4)
    src = a.source_manifest.resolve()
    out = a.out.resolve()
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    old = json.loads(src.read_text())
    assert old['schema'] == 'n3dv_dynamic_sr_pilot_v1'
    m = copy.deepcopy(old)
    start = time.monotonic()
    errors = []
    for obs in m['observations']:
        hp = src.parent / obs['hr_path']
        assert sha(hp) == obs['hr_sha256'], hp
        hr = np.array(Image.open(hp).convert('RGB'), copy=True)
        x = torch.from_numpy(hr).permute(2, 0, 1).float().div(255)
        h, w = x.shape[-2:]
        assert (w, h) == tuple(m['resolutions']['hr']) and h % 4 == w % 4 == 0
        lr = F.interpolate(x[None], scale_factor=.25, mode='bilinear', align_corners=False)[0]
        # At integer x4 reduction, this kernel samples the central 2x2 of
        # each 4x4 footprint. This independent expression checks phase/kernel.
        ref = (x[:, 1::4, 1::4] + x[:, 1::4, 2::4] +
               x[:, 2::4, 1::4] + x[:, 2::4, 2::4]) * .25
        error = float((lr - ref).abs().max())
        assert error < 2e-7, error
        errors.append(error)
        rel = Path('lr_float') / obs['camera_id'] / (Path(obs['lr_path']).stem + '.npy')
        npy = out / rel
        npy.parent.mkdir(parents=True, exist_ok=True)
        np.save(npy, lr.numpy(), allow_pickle=False)
        preview = Path('lr') / obs['camera_id'] / (npy.stem + '.png')
        pp = out / preview
        pp.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(lr.permute(1, 2, 0).mul(255).round().byte().numpy()).save(pp)
        obs['hr_path'] = os.path.relpath(hp, out)
        obs['lr_path'] = str(preview)
        obs['lr_sha256'] = sha(pp)
        obs['lr_float_path'] = str(rel)
        obs['lr_float_sha256'] = sha(npy)
    for key in ('npz_path', 'ply_path', 'report_path'):
        if key in m['initialization']:
            m['initialization'][key] = os.path.relpath(src.parent / old['initialization'][key], out)
    m['initialization']['reuse_note'] = (
        'Fixed shared auxiliary point cloud previously constructed from training-only '
        'bicubic-AA LR; reused identically by both systems, not re-estimated from author LR.')
    m['degradation'] = dict(mode='bilinear', scale_factor=.25, align_corners=False,
                            antialias=False, dtype='float32', input='RGB PNG / 255',
                            training_target='lr_float_path, NOT quantized lr_path preview',
                            provenance='SR4D author Camera code; paper does not specify input kernel')
    m['comparison_source_manifest'] = os.path.relpath(src, out)
    m['comparison_source_manifest_sha256'] = sha(src)
    m['comparison_protocol'] = 'sr4d_author_lr_v1'
    m['author_lr_generator_sha256'] = sha(Path(__file__))
    write(out / 'manifest.json', m)
    write(out / 'preparation_complete.json', dict(
        status='complete', manifest_sha256=sha(out / 'manifest.json'),
        source_manifest_sha256=sha(src), observations=len(m['observations']),
        independent_kernel_max_error=max(errors), seconds=time.monotonic()-start,
        device='CPU', preserved_camera_time_split=True))
    print(json.dumps(dict(status='complete', observations=len(m['observations']), out=str(out))))


if __name__ == '__main__':
    main()
