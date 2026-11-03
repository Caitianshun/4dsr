"""Render speed and cam00 PNG renders of the SR4D x4 endpoints in the local sr4d environment (Claude 4dsr-ac, 2026-11-02).

The main-table SR4D endpoints (author code + the audited 09-22 fixes, trained on the A100: fine 40000 + coarse 20000) are
rendered here with the sr4d environment's own SR4D rasterizer build (sm_86 + sm_120; its Python wrapper has the same sha256
as the A100 ext_a100 build used for the main table, checked below) instead of ext_a100 (sm_80 only).
  fps   SR4D's evaluation call (sr4d_common.render_endpoint -> author gaussian_renderer.render) rasterizes the LR and the HR image
        every frame; the HR image is the output. Two timings: 'sr4d' = HR-only (the author HR branch, same inputs, checked
        bit-identical to the author render_hr on frames 0/150/299) and 'sr4d_author_call' = the full author call.
  render all 300 cam00 frames at HR exactly as sr4d_area4.py render (author call, black background, clamp, rint to uint8 PNG),
        with render_index.json in the scorer format, plus the count of PNGs identical to the main-table render_index sha256.
  fps_sr4d.py fps    --spec SPEC.json --scene S --out FPS_DIR
  fps_sr4d.py render --spec SPEC.json --scene S --out RENDER_DIR
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
S4 = ROOT / 'experiments/dynamic_sr_modules_20261010/sr4d_area4_20261028'
WRAPPER_SHA = 'ebb6635a8d7b6e000d42eb793c712f7364c1bc18e2e9d6516868007896da1f32'  # ext_a100/diff_gaussian_rasterization/__init__.py
sys.path.insert(0, str(HERE))
import bench_core  # noqa: E402


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def setup():
    import importlib
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    sys.path[:0] = [str(S4 / 'vendor'), str(S4)]
    import sr4d_common as SC
    SC.configure_upstream(str(S4 / 'upstream'))
    mods = {}
    for name in ('diff_gaussian_rasterization', 'diff_gaussian_rasterization._C', 'simple_knn._C', 'kornia'):
        f = Path(importlib.import_module(name).__file__).resolve()
        mods[name] = dict(file=str(f), sha256=sha(f))
    if mods['diff_gaussian_rasterization']['sha256'] != WRAPPER_SHA:
        raise SystemExit('SR4D rasterizer wrapper differs from the main-table build')
    return torch, SC, mods


class Adapter:
    def __init__(self, torch, SC, spec):
        import score_views_area4 as SA
        self.torch, self.SC = torch, SC
        mp = ROOT / spec['manifest']
        manifest = json.loads(mp.read_text())
        rows = SA.select_observations_area4(manifest, 'test')
        data = SC.ManifestData(mp)
        run = ROOT / spec['sr4d']['run']
        self.g, self.c, self.f = SC.load_endpoint(str(S4 / 'upstream'), run, 'fine', 40000)
        by_key = {(o['camera_id'], o['frame_index']): o for o in manifest['observations']}
        self.obs = [by_key[(r['camera_id'], r['frame_index'])] for r in rows]
        self.frames = [int(o['frame_index']) for o in self.obs]
        if self.frames != list(range(len(self.frames))):
            raise SystemExit('cam00 frames 0..299 in order required')
        self.cams = [SC.Camera(data, o) for o in self.obs]
        for cam in self.cams:  # what render_endpoint does on every call; done once here
            for name in ('fid', 'world_view_transform', 'full_proj_transform', 'full_proj_transform_lr', 'camera_center'):
                setattr(cam, name, getattr(cam, name).cuda())
        self.bg = torch.zeros(3, device='cuda')
        from argparse import Namespace
        self.pipe = Namespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
        from gaussian_renderer import render
        from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
        self.author_render, self.Settings, self.Rasterizer = render, GaussianRasterizationSettings, GaussianRasterizer
        self.hr_only = True
        self.info = dict(points=int(self.g.get_xyz.shape[0]), run=spec['sr4d']['run'],
                         endpoint={p: sha(run / p) for p in ('point_cloud_hr/iteration_40000/point_cloud.ply', 'deform_hr/iteration_40000/deform_hr.pth')})

    def deform(self, i):
        g, cam = self.g, self.cams[i]
        time_ = cam.fid[None].expand(len(g.get_xyz), -1)
        values = self.c.step(g.get_xyz.detach(), time_)
        return tuple(a + b for a, b in zip(values, self.f.step(g.get_xyz.detach(), time_)))

    def extra(self, i, st):
        return st

    def raster(self, i, st):
        if not self.hr_only:
            return self.author_render(self.cams[i], self.g, self.pipe, self.bg, *st, False)['render_hr']
        torch, g, cam = self.torch, self.g, self.cams[i]
        d_xyz, d_rotation, d_scaling = st
        sp = torch.zeros_like(g.get_xyz, dtype=g.get_xyz.dtype, requires_grad=True, device='cuda') + 0
        spd = torch.zeros_like(g.get_xyz, dtype=g.get_xyz.dtype, requires_grad=True, device='cuda') + 0
        settings = self.Settings(image_height=int(cam.image_height), image_width=int(cam.image_width),
                                 tanfovx=math.tan(cam.FoVx * 0.5), tanfovy=math.tan(cam.FoVy * 0.5), bg=self.bg, scale_modifier=1.0,
                                 viewmatrix=cam.world_view_transform, projmatrix=cam.full_proj_transform, sh_degree=g.active_sh_degree,
                                 campos=cam.camera_center, prefiltered=False, debug=False)
        return self.Rasterizer(raster_settings=settings)(means3D=g.get_xyz + d_xyz, means2D=sp, means2D_densify=spd, shs=g.get_features,
                                                         colors_precomp=None, opacities=g.get_opacity, scales=g.get_scaling + d_scaling,
                                                         rotations=g.get_rotation + d_rotation, cov3D_precomp=None)[0]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('mode', choices=('fps', 'render'))
    p.add_argument('--spec', required=True)
    p.add_argument('--scene', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    sys.path.insert(0, str(ROOT / 'experiments/dynamic_sr_modules_20261010/area4_baselines_20261024'))
    spec = json.loads((ROOT / a.spec).read_text())[a.scene]
    torch, SC, mods = setup()
    import numpy as np
    began = time.time()
    ad = Adapter(torch, SC, spec)
    load_seconds = time.time() - began
    if a.mode == 'fps':
        env = dict(bench_core.environment(torch, sys), modules=mods)
        ident = {}
        with torch.no_grad():
            for i in (0, 150, 299):
                st = ad.deform(i)
                ad.hr_only = True
                x = ad.raster(i, st)
                ad.hr_only = False
                y = ad.raster(i, st)
                ident[str(i)] = float((x - y).abs().max().item())
        for name, hr_only in (('sr4d', True), ('sr4d_author_call', False)):
            out = ROOT / a.out / f'{a.scene}_area4' / f'{name}.json'
            if out.exists():
                continue
            ad.hr_only = hr_only
            rec = dict(schema='qualfig_fps_v1', scene=a.scene, method=name, environment=env, load_seconds=load_seconds, info=ad.info,
                       hr_only_vs_author_render_hr_max_abs=ident, resolution_hr=list(SC.ManifestData(ROOT / spec['manifest']).hr_wh),
                       script=dict(path=str(Path(__file__).relative_to(ROOT)), sha256=sha(__file__)),
                       note='author call also rasterizes the LR image' if not hr_only else 'HR branch of the author render only')
            rec.update(bench_core.run(ad, torch))
            rec['status'] = 'ok'
            bench_core.write(out, rec)
            print(json.dumps(dict(scene=a.scene, method=name, fps=round(rec['fps_from_mean_latency'], 1), fps_tp=round(rec['fps_throughput'], 1))), flush=True)
        return
    from PIL import Image
    out = ROOT / a.out
    if (out / 'render_index.json').exists():
        raise FileExistsError(out)
    (out / 'cam00').mkdir(parents=True, exist_ok=True)
    ad.hr_only = False  # exactly the main-table render call
    ref = {}
    ri = ROOT / spec['sr4d'].get('main_table_render_index', '')
    if ri.is_file():
        ref = {int(r['frame']): r['render_sha256'] for r in json.loads(ri.read_text())['records']}
    records, same = [], 0
    with torch.no_grad():
        for i, o in enumerate(ad.obs):
            img = ad.raster(i, ad.deform(i)).detach().cpu().numpy()
            if not np.isfinite(img).all():
                raise ValueError(f'invalid render {i}')
            png = np.rint(np.clip(img.transpose(1, 2, 0), 0, 1) * 255).astype(np.uint8)
            rel = f'cam00/{i:04d}.png'
            Image.fromarray(png).save(out / rel)
            h = sha(out / rel)
            same += int(ref.get(i) == h)
            records.append(dict(camera_id='cam00', frame=i, path=rel, render_sha256=h))
    mp = ROOT / spec['manifest']
    index = dict(schema='dynamic_lr_multiview_render_v1', manifest_sha256=sha(mp), scope='test', resolution=list(SC.ManifestData(mp).hr_wh),
                 input_representation='uint8_RGB_PNG', render_camera_transform='identity', records=records, status='completed',
                 method='sr4d_author_area4', scene=json.loads(mp.read_text())['scene'], endpoint=ad.info,
                 renderer='author gaussian_renderer.render, render_hr (coarse + fine deformation), black background, rint uint8',
                 environment=dict(bench_core.environment(torch, sys), modules=mods), main_table_render_index=str(ri.relative_to(ROOT)) if ref else None,
                 pngs_identical_to_main_table=same, evaluator_source=dict(path=str(Path(__file__).relative_to(ROOT)), sha256=sha(__file__)),
                 no_HR_or_LR_pixels_read=True, seconds=time.time() - began)
    (out / 'render_index.json').write_text(json.dumps(index, indent=1) + '\n')
    print(json.dumps(dict(scene=a.scene, rendered=len(records), identical_to_main_table=same)), flush=True)


if __name__ == '__main__':
    main()
