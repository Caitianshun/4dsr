"""Render speed at the cam00 test view for the methods that run in the mml_bw environment (Claude 4dsr-ac, 2026-11-02).

Methods (one JSON per scene x method under --out):
  ours         main-table model: PVCD student checkpoint, post-hoc training-camera cull (ens_eval.load_model), Wu deformation
               query, P+C residuals of the frame (pre-staged on the GPU; same arithmetic as eval_resid.apply), Wu rasterizer at HR.
               cam00 is not a training camera, so the per-camera module is not applied (as in the main table).
  ours_noresid the same model without the residuals (cost of the per-frame residual step).
  wu           main-table Wu 4DGS endpoint (no cull), Wu rasterizer.
  splatsure / iesrgs / sequence_matters / threedsr
               static-SR baselines on the shared Wu backbone, through sv2d.render (the same renderer as their main-table renders:
               Wu rasterizer for SplatSuRe and Sequence Matters, the RADE/Mip renderer with the same-time training-camera 3D filter
               for IE-SRGS, the 3DSR Mip renderer for 3DSR). The main-table rows are the uncut models; no cull here.
The timing loop is bench_core.run (see its docstring). Before timing, ours checks on frames 0/150/299 that the pre-staged
residual step gives bit-identical images to eval_resid.apply, and every method compares frames 0/150/299 with its main-table
8-bit PNG when that PNG still exists (count of differing 8-bit values; renders were made on other GPUs for most methods).
  fps_bench.py --spec SPEC.json --scene Cook07 --methods ours,ours_noresid,wu,splatsure,iesrgs,sequence_matters,threedsr --out DIR
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
MOD = ROOT / 'experiments/dynamic_sr_modules_20261010'
for p in (HERE, MOD / 'geoprobe_20261017', MOD / 'perframe_20261021', MOD / 'area4_baselines_20261024', MOD / 'static_sr_v2_20261026'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import bench_core  # noqa: E402


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def git_safe_directory():
    """The 3DSR renderer checks its upstream clone with `git rev-parse HEAD` / `git diff --quiet HEAD`; under the training service
    the clone belongs to another user, so git refuses it (dubious ownership). Mark exactly that clone as safe for this process."""
    import os
    os.environ['GIT_CONFIG_COUNT'] = '1'
    os.environ['GIT_CONFIG_KEY_0'] = 'safe.directory'
    os.environ['GIT_CONFIG_VALUE_0'] = str(ROOT / 'tmp/static_sr_wu_20261022/upstream/threedsr')


def u8(img, np):
    a = img.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy().astype(np.float64)
    return np.floor(a * 255.0 + 0.5).clip(0, 255).astype(np.uint8)


def png_compare(adapter, png_dir, np, rounding):
    """8-bit values that differ from the stored main-table PNG on frames 0/150/299 (None if the PNGs are gone)."""
    from PIL import Image
    out = {}
    for i in (0, 150, 299):
        p = ROOT / png_dir / f'cam00/{i:04d}.png'
        if not p.exists():
            return None
        with adapter.torch.no_grad():
            img = adapter.raster(i, adapter.extra(i, adapter.deform(i))).clamp(0, 1)
        a = img.permute(1, 2, 0).cpu().numpy().astype(np.float64)
        mine = (np.floor(a * 255.0 + 0.5) if rounding == 'floor' else np.rint(a * 255.0)).clip(0, 255).astype(np.uint8)
        ref = np.asarray(Image.open(p).convert('RGB'))
        d = np.abs(mine.astype(int) - ref.astype(int))
        out[str(i)] = dict(values_differing=int((d > 0).sum()), max_abs=int(d.max()))
    return out


class OursAdapter:
    def __init__(self, spec, residuals):
        import eval_full
        import eval_resid
        import numpy as np
        import torch
        self.torch, self.np = torch, np
        self.E, self.R = eval_full, eval_resid
        m = eval_full.ens_eval.load_model('ours', spec['ours']['checkpoint'], True)
        self.m, self.g = m, m['g']
        self.obs = m['test']
        self.frames = [int(o['frame_index']) for o in self.obs]
        self.cams = [eval_full.native.hr_camera(eval_full.ev.camera_item(m['manifest'], o, torch), int(o['frame_index']), m['manifest'])
                     for o in self.obs]
        self.times = [float(o['time']) for o in self.obs]
        self.resid = None
        self.info = dict(points=int(self.g.get_xyz.shape[0]), cull=m['info'].get('cull'), checkpoint=m['info']['checkpoint'])
        if self.frames != list(range(len(self.frames))):
            raise SystemExit('cam00 frames 0..299 in order required')
        if residuals:
            d = ROOT / spec['ours']['residuals']
            meta = json.loads(((d / 'run.json') if (d / 'run.json').exists() else (d / 'meta.json')).read_text())
            rsha = eval_resid.meta_value(meta, ('checkpoint_sha256', 'parent_sha256', 'parent.sha256', 'checkpoint.sha256'))
            if rsha != m['info']['checkpoint']['sha256']:
                raise SystemExit('residuals belong to another checkpoint')
            self.npz, self.resid = {}, {}
            sizes = []
            for f in self.frames:
                with np.load(d / f'f{f:04d}.npz') as z:
                    raw = {k: eval_resid.first(z, eval_resid.ALIASES[k]) for k in eval_resid.ALIASES}
                    raw = {k: (None if v is None else np.asarray(v)) for k, v in raw.items()}
                self.npz[f] = {k: v for k, v in raw.items() if v is not None}
                t = {'idx': torch.from_numpy(np.asarray(raw['idx'], dtype=np.int64)).cuda()}
                for k in ('dxyz', 'dsh0', 'dlogscale', 'dlogit'):
                    if raw[k] is not None:
                        t[k] = torch.from_numpy(np.asarray(raw[k], dtype=np.float32)).cuda()
                self.resid[f] = t
                sizes.append(int(raw['idx'].shape[0]))
            self.info['residual_gaussians_per_frame'] = dict(mean=float(np.mean(sizes)), min=int(min(sizes)), max=int(max(sizes)))
            self.info['residual_fields'] = sorted(k for k in self.resid[self.frames[0]] if k != 'idx')

    def deform(self, i):
        return self.E.native.native_effective_state(self.g, self.times[i], self.torch)

    def extra(self, i, st):
        if self.resid is None:
            return st
        z = self.resid[self.frames[i]]
        st = dict(st)
        idx = z['idx']
        torch = self.torch
        if 'dxyz' in z:
            st['xyz'] = st['xyz'].index_add(0, idx, z['dxyz'].to(st['xyz'].dtype))
        if 'dsh0' in z:
            sh = st['sh'].clone()
            sh[idx, 0, :] = sh[idx, 0, :] + z['dsh0'].to(sh.dtype)
            st['sh'] = sh
        if 'dlogscale' in z:
            st['scales'] = st['scales'].index_put((idx,), st['scales'][idx] * torch.exp(z['dlogscale']))
        if 'dlogit' in z:
            o = st['opacity'][idx].clamp(1e-6, 1 - 1e-6)
            st['opacity'] = st['opacity'].index_put((idx,), torch.sigmoid(torch.log(o / (1 - o)) + z['dlogit']))
        return st

    def raster(self, i, st):
        return self.E.native.native_rgb(self.cams[i], self.g, st, self.m['pipe'], self.m['background'])['render']

    def residual_identity(self):
        """Pre-staged residual step vs eval_resid.apply on frames 0/150/299: max |difference| of the float images."""
        if self.resid is None:
            return None
        torch = self.torch
        out = {}
        with torch.no_grad():
            for i in (0, 150, 299):
                st = self.deform(i)
                a = self.raster(i, self.extra(i, st))
                b = self.raster(i, self.R.apply(st, self.npz[self.frames[i]]))
                out[str(i)] = float((a - b).abs().max().item())
        return out


class WuAdapter(OursAdapter):
    def __init__(self, spec):
        import eval_full
        import numpy as np
        import torch
        self.torch, self.np = torch, np
        self.E = eval_full
        m = eval_full.ens_eval.load_model('wu', spec['wu']['checkpoint'], False)
        self.m, self.g = m, m['g']
        self.obs = m['test']
        self.frames = [int(o['frame_index']) for o in self.obs]
        self.cams = [eval_full.native.hr_camera(eval_full.ev.camera_item(m['manifest'], o, torch), int(o['frame_index']), m['manifest'])
                     for o in self.obs]
        self.times = [float(o['time']) for o in self.obs]
        self.resid = None
        self.info = dict(points=int(self.g.get_xyz.shape[0]), checkpoint=m['info']['checkpoint'])


class Sv2Adapter:
    def __init__(self, spec, method):
        import sv2d as sv2
        import numpy as np
        if not sv2.MODS:
            sv2.setup(ROOT)
        torch, common = sv2.MODS['torch'], sv2.MODS['common']
        self.torch, self.np, self.sv2, self.method = torch, np, sv2, method
        e = spec[method]
        config = sv2.read_config(ROOT / e['config'])
        if config['method'] != method:
            raise SystemExit(f'config method {config["method"]} != {method}')
        import score_views_area4 as SA
        manifest = json.loads((ROOT / config['manifest']['path']).read_text())
        rows = SA.select_observations_area4(manifest, 'test')
        from backend import WuBackend
        self.backend = WuBackend(config, checkpoint=common.identity(ROOT / e['checkpoint']))
        self.backend.g._deformation.eval()
        self.cams = sv2.Cameras(self.backend)
        by_key = {(o['camera_id'], o['frame_index']): o for o in manifest['observations']}
        self.obs = [by_key[(r['camera_id'], r['frame_index'])] for r in rows]
        self.frames = [int(o['frame_index']) for o in self.obs]
        if self.frames != list(range(len(self.frames))):
            raise SystemExit('cam00 frames 0..299 in order required')
        for o in self.obs:  # camera objects and same-time reference cameras are built once, outside the timing
            self.cams.get(o, 'hr', True)
            if method in ('iesrgs', 'threedsr'):
                self.cams.references(o['frame_index'], 'hr')
        self.info = dict(points=int(self.backend.g.get_xyz.shape[0]), checkpoint=dict(path=e['checkpoint'], sha256=sha(ROOT / e['checkpoint'])),
                         renderer={'splatsure': 'Wu native rasterizer', 'sequence_matters': 'Wu native rasterizer',
                                   'iesrgs': 'IE-SRGS RADE rasterizer + Mip 3D filter from same-time training cameras',
                                   'threedsr': '3DSR Mip rasterizer + 3D filter from same-time training cameras'}[method])

    def deform(self, i):
        return self.backend.effective_state(self.obs[i]['time'])

    def extra(self, i, st):
        return st

    def raster(self, i, st):
        return self.sv2.render(self.method, 'hr', self.backend, self.cams, self.obs[i], st, heldout=True)['render']


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--spec', required=True)
    p.add_argument('--scene', required=True)
    p.add_argument('--methods', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--frames', type=int, default=300)
    a = p.parse_args()
    import numpy as np
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    spec = json.loads((ROOT / a.spec).read_text())[a.scene]
    git_safe_directory()
    env = bench_core.environment(torch, sys)
    for method in a.methods.split(','):
        out = ROOT / a.out / f'{a.scene}_area4' / f'{method}.json'
        if out.exists() and json.loads(out.read_text()).get('status') == 'ok':
            print(json.dumps(dict(skip=str(out))), flush=True)
            continue
        began = time.time()
        rec = dict(schema='qualfig_fps_v1', scene=a.scene, method=method, environment=env, script=dict(path=str(Path(__file__).relative_to(ROOT)), sha256=sha(__file__)))
        try:
            torch.cuda.empty_cache()
            t0 = time.time()
            if method in ('ours', 'ours_noresid'):
                ad = OursAdapter(spec, residuals=(method == 'ours'))
                png, rounding = spec['ours']['renders'], 'floor'
            elif method == 'wu':
                ad = WuAdapter(spec)
                png, rounding = spec['wu']['renders'], 'rint'  # render_wu_multi.py: np.rint
            else:
                ad = Sv2Adapter(spec, method)
                png, rounding = None, 'rint'
            torch.cuda.synchronize()
            rec['load_seconds'] = time.time() - t0
            ad.frames = ad.frames[:a.frames]
            rec['info'] = ad.info
            rec['resolution_hr'] = list(ad.raster(0, ad.extra(0, ad.deform(0))).shape[1:][::-1])
            if method == 'ours':
                rec['residual_step_identity_max_abs'] = ad.residual_identity()
            if png and method != 'ours_noresid':
                rec['png_compare_main_table'] = png_compare(ad, png, np, rounding)
            rec.update(bench_core.run(ad, torch))
            rec['status'] = 'ok'
            print(json.dumps(dict(scene=a.scene, method=method, points=ad.info['points'], fps=round(rec['fps_from_mean_latency'], 1),
                                  fps_tp=round(rec['fps_throughput'], 1), ms=round(rec['latency_ms']['mean'], 2))), flush=True)
            del ad
        except Exception:
            rec['status'] = 'failed'
            rec['error'] = traceback.format_exc()[-4000:]
            print(rec['error'], flush=True)
        rec['seconds'] = time.time() - began
        bench_core.write(out, rec)
        import gc
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
