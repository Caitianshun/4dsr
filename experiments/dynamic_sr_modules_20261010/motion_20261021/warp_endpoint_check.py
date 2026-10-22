"""Zero-training check of the warp-field approximation in modkit13's flow term (Claude 4dsr-46, 2026-10-22; requested in the
gpt-6-astra review ft-stack-20261022 round 1 after a Codex remark relayed by the user; minimal form and closure rule agreed with
4dsr-e3, pre-registered in tmp/claude_perscene_design_20261021/prereg_motion.md Addendum H before any result).

modkit13 maps the model flow image to the observation frame with the camera's learned warp field by resampling only (to_obs: the
flow is read at T(x) = x + w(x), bilinear). The exact observed motion of the point seen at x is u = T^-1(T(x) + d0) - x with
d0 = the normalised resampled flow, i.e. the fixed point u = d0 - [w(x + u) - w(x)] (iterated from u = d0). On a fixed checkpoint
(no parameter update) and for every training camera at --frames, the LR-pooled model flow is computed the old way and with the
endpoint fixed point (normalise by alpha on HR, solve, multiply alpha back, pool exactly as modkit13) and compared on the cached
teacher target with modkit13's definitions (valid = fb < 1 HR px and alpha_lr > 0.95, active = valid and (|target| > 0.5 or
|flow| > 0.5), term = sum_valid relu(e - 0.25) / max(#active, 64)). Reported: |new - old| on active LR pixels; hinge flips across
the 0.25 dead zone (per active pixel); active-set change; relative change of the term; the cosine between the gradients of the
old and new term w.r.t. the deformed centres (both frames); and the bilinear-vs-bicubic resampling difference (the image
observation model samples bicubic). The flow math (project, cov2d, sqrt2, inv2, splat, model_flow) is modkit13's, copied unchanged.
Usage: warp_endpoint_check.py --checkpoint CK --flow-dir DIR --out JSON [--frames 0,50,100,150,200,250] [--D 2] [--iters 6]"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'absorb_20261021'))
import absorb_bound2 as ab  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

cr, native, ev, rt = ab.cr, ab.native, ab.ev, ab.rt
ROOT = ab.ROOT
DEAD, KEEP = 0.25, 100_000


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--flow-dir', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--frames', default='0,50,100,150,200,250')
    p.add_argument('--D', type=int, default=2)
    p.add_argument('--iters', type=int, default=6)
    a = p.parse_args()
    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(0)
    dev = 'cuda'
    started = time.time()
    ck = ROOT / a.checkpoint
    out = ROOT / a.out
    if out.exists():
        raise FileExistsError(f'completed run preserved: {out}')
    payload = torch.load(ck, map_location='cpu', weights_only=False)
    plan, raw_aff = payload['plan'], payload['affine']
    del payload
    manifest_path, manifest = cr.plan_scene(plan)
    train = list(manifest['splits']['train'])
    obs = {(o['camera_id'], int(o['frame_index'])): o for o in manifest['observations']}
    index = {c: i for i, c in enumerate(train)}
    g = ab.load_model(ck, plan, dev)
    names = list(raw_aff['names'])
    wfield = raw_aff['wfield'].float().to(dev)
    wmean = wfield.mean(0)
    fdir = Path(a.flow_dir) if Path(a.flow_dir).is_absolute() else ROOT / a.flow_dir

    def target(c, f):
        if not (fdir / f'{c}.frames.npy').exists():
            return None
        rows_ = {int(x): i for i, x in enumerate(np.load(fdir / f'{c}.frames.npy'))}
        if int(f) not in rows_:
            return None
        i = rows_[int(f)]
        fl = np.asarray(np.load(fdir / f'{c}.flow.npy', mmap_mode='r')[i], dtype=np.float32)
        fb = np.asarray(np.load(fdir / f'{c}.fb.npy', mmap_mode='r')[i])
        return torch.from_numpy(fl).to(dev), torch.from_numpy(fb).to(dev)

    def project(xyz, cam, w, h):
        P = cam.full_proj_transform.to(xyz.device)
        q = torch.cat([xyz, torch.ones_like(xyz[:, :1])], 1) @ P
        ndc = q[:, :2] / (q[:, 3:4] + 1e-7)
        return torch.stack([((ndc[:, 0] + 1) * w - 1) * 0.5, ((ndc[:, 1] + 1) * h - 1) * 0.5], 1)

    def cov2d(st, cam, w, h):
        V = cam.world_view_transform.to(st['xyz'].device)
        pc = torch.cat([st['xyz'], torch.ones_like(st['xyz'][:, :1])], 1) @ V
        tz = pc[:, 2].clamp_min(1e-4)
        tanx, tany = math.tan(cam.FoVx * .5), math.tan(cam.FoVy * .5)
        fx, fy = w / (2 * tanx), h / (2 * tany)
        tx = (pc[:, 0] / tz).clamp(-1.3 * tanx, 1.3 * tanx) * tz
        ty = (pc[:, 1] / tz).clamp(-1.3 * tany, 1.3 * tany) * tz
        J = torch.zeros(tz.shape[0], 2, 3, device=tz.device)
        J[:, 0, 0], J[:, 0, 2] = fx / tz, -fx * tx / tz ** 2
        J[:, 1, 1], J[:, 1, 2] = fy / tz, -fy * ty / tz ** 2
        T = J @ V[:3, :3].T
        S = T @ native.state_covariance(st, torch) @ T.transpose(1, 2)
        return S + 0.3 * torch.eye(2, device=S.device)

    def sqrt2(M):
        s_ = (M[:, 0, 0] * M[:, 1, 1] - M[:, 0, 1] * M[:, 1, 0]).clamp_min(1e-12).sqrt()
        t_ = (M[:, 0, 0] + M[:, 1, 1] + 2 * s_).clamp_min(1e-12).sqrt()
        return (M + s_[:, None, None] * torch.eye(2, device=M.device)) / t_[:, None, None]

    def inv2(M):
        det = (M[:, 0, 0] * M[:, 1, 1] - M[:, 0, 1] * M[:, 1, 0]).clamp_min(1e-12)
        return torch.stack([torch.stack([M[:, 1, 1], -M[:, 0, 1]], -1), torch.stack([-M[:, 1, 0], M[:, 0, 0]], -1)], 1) / det[:, None, None]

    def splat(cam, st, colors):
        w, h = int(cam.image_width), int(cam.image_height)
        settings = GaussianRasterizationSettings(image_height=h, image_width=w, tanfovx=math.tan(cam.FoVx * .5),
                                                 tanfovy=math.tan(cam.FoVy * .5), bg=torch.zeros(3, device=colors.device),
                                                 scale_modifier=1., viewmatrix=cam.world_view_transform.to(colors.device),
                                                 projmatrix=cam.full_proj_transform.to(colors.device), sh_degree=0,
                                                 campos=cam.camera_center.to(colors.device), prefiltered=False, debug=False)
        points = torch.zeros_like(st['xyz'], requires_grad=False)
        out_ = GaussianRasterizer(raster_settings=settings)(means3D=st['xyz'].detach(), means2D=points, shs=None,
                                                            colors_precomp=colors, opacities=st['opacity'].detach(),
                                                            scales=st['scales'].detach(), rotations=st['rotation'].detach(),
                                                            cov3D_precomp=None)
        return out_[0]

    def model_flow(cam, st, st2):
        w, h = int(cam.image_width), int(cam.image_height)
        mu1, mu2 = project(st['xyz'], cam, w, h), project(st2['xyz'], cam, w, h)
        d = (mu2 - mu1).clamp(-64, 64)
        one = torch.ones_like(d[:, :1])
        p1 = splat(cam, st, torch.cat([d, one], 1))
        B = sqrt2(cov2d(st2, cam, w, h)) @ inv2(sqrt2(cov2d(st, cam, w, h))) - torch.eye(2, device=d.device)
        B = B.clamp(-4, 4)
        v = (B @ mu1[:, :, None])[:, :, 0]
        p2 = splat(cam, st, torch.stack([B[:, 0, 0], B[:, 0, 1], B[:, 1, 0]], 1))
        p3 = splat(cam, st, torch.cat([B[:, 1, 1:2], v], 1))
        py, px = torch.meshgrid(torch.arange(h, device=d.device, dtype=torch.float32),
                                torch.arange(w, device=d.device, dtype=torch.float32), indexing='ij')
        fx_ = p1[0] + p2[0] * px + p2[1] * py - p3[1]
        fy_ = p1[1] + p2[2] * px + p3[0] * py - p3[2]
        return torch.stack([fx_, fy_, p1[2]])

    def field(cid, h, w):
        d = wfield[names.index(cid)] - wmean
        return F.interpolate(d[None], size=(h, w), mode='bilinear', align_corners=True)[0]

    def sample(img, disp, mode):
        """img read at x + disp(x) (pixel centres, as modkit13 to_obs)"""
        h, w = img.shape[-2:]
        py, px = torch.meshgrid(torch.arange(h, device=img.device, dtype=torch.float32) + 0.5,
                                torch.arange(w, device=img.device, dtype=torch.float32) + 0.5, indexing='ij')
        grid = torch.stack([(px + disp[0]) * (2 / w) - 1, (py + disp[1]) * (2 / h) - 1], -1)
        return F.grid_sample(img[None], grid[None], mode=mode, padding_mode='border', align_corners=False)[0]

    def pooled(weighted, alpha, k):
        """alpha-weighted HR flow sums -> LR flow, exactly as modkit13 (pool the sums, divide by the pooled alpha)"""
        a_lr = F.avg_pool2d(alpha[None, None], k)[0, 0]
        return F.avg_pool2d(weighted[None], k)[0] / a_lr.clamp_min(1e-4), a_lr

    def term(fl, tgt, valid):
        e = (fl - tgt).norm(dim=0)
        act = valid & ((tgt.norm(dim=0) > 0.5) | (fl.detach().norm(dim=0) > 0.5))
        return torch.relu(e - DEAD)[valid].sum() / max(int(act.sum()), 64), e.detach(), act

    def keep(t):
        return t[torch.randperm(t.numel(), device=t.device)[:KEEP]].cpu() if t.numel() > KEEP else t.cpu()

    frames = [int(x) for x in a.frames.split(',')]
    rows, pool = [], dict(d_act=[], d_valid=[], bicubic_act=[], hr_corr=[])
    tot = dict(valid=0, act_old=0, act_new=0, act_sym=0, hinge_flip_valid=0, hinge_flip_act=0)
    for f in frames:
        o2 = obs.get((train[0], f + a.D))
        if o2 is None:
            continue
        with torch.no_grad():
            st = native.native_effective_state(g, float(obs[(train[0], f)]['time']), torch)
            st2 = native.native_effective_state(g, float(o2['time']), torch)
        for c in train:
            if c not in names:
                continue
            tg = target(c, f)
            if tg is None:
                continue
            tgt, fbq = tg
            cam = native.hr_camera(ev.camera_item(manifest, obs[(c, f)], torch), index[c], manifest)
            x1 = st['xyz'].detach().clone().requires_grad_(True)
            x2 = st2['xyz'].detach().clone().requires_grad_(True)
            img = model_flow(cam, dict(st, xyz=x1), dict(st2, xyz=x2))
            h, w = img.shape[-2:]
            up = field(c, h, w)
            old = sample(img, up, 'bilinear')
            alpha = old[2]
            d0 = old[:2] / alpha.clamp_min(1e-4)
            u, last = d0, None
            for _ in range(a.iters):
                u_next = d0 - (sample(up, u, 'bilinear') - up)
                last = float((u_next - u).detach().abs().max())
                u = u_next
            k = w // tgt.shape[-1]
            fl_old, a_lr = pooled(old[:2], alpha, k)
            fl_new, _ = pooled(u * alpha, alpha, k)
            valid = (fbq < 10) & (a_lr.detach() > 0.95)
            t_old, e_old, act_old = term(fl_old, tgt, valid)
            t_new, e_new, act_new = term(fl_new, tgt, valid)
            g_old = torch.autograd.grad(t_old, [x1, x2], retain_graph=True)
            g_new = torch.autograd.grad(t_new, [x1, x2])
            go, gn = torch.cat([q.flatten() for q in g_old]), torch.cat([q.flatten() for q in g_new])
            cos = float((go @ gn) / (go.norm() * gn.norm()).clamp_min(1e-30))
            rel_grad = float((gn - go).norm() / go.norm().clamp_min(1e-30))
            with torch.no_grad():
                cub = sample(img.detach(), up, 'bicubic')
                fl_cub, _ = pooled(cub[:2], cub[2], k)
                diff = (fl_new - fl_old).norm(dim=0)
                flip = (e_old > DEAD) != (e_new > DEAD)
                nv, na_o, na_n = int(valid.sum()), int(act_old.sum()), int(act_new.sum())
                n_sym = int((act_old ^ act_new).sum())
                fv, fa = int((flip & valid).sum()), int((flip & act_old).sum())
                pool['d_act'].append(keep(diff[act_old]))
                pool['d_valid'].append(keep(diff[valid]))
                pool['bicubic_act'].append(keep((fl_cub - fl_old).norm(dim=0)[act_old]))
                pool['hr_corr'].append(keep(((u - d0).norm(dim=0))[alpha > 0.95]))
                for key, val in (('valid', nv), ('act_old', na_o), ('act_new', na_n), ('act_sym', n_sym), ('hinge_flip_valid', fv),
                                 ('hinge_flip_act', fa)):
                    tot[key] += val
                rows.append(dict(frame=f, camera=c, valid=nv, active_old=na_o, active_new=na_n, active_sym=n_sym,
                                 hinge_flips_valid=fv, hinge_flips_active=fa, term_old=float(t_old), term_new=float(t_new),
                                 grad_cos=cos, grad_rel_change=rel_grad, fixed_point_last_step=last,
                                 diff_active_mean=float(diff[act_old].mean()) if na_o else None,
                                 diff_active_max=float(diff[act_old].max()) if na_o else None,
                                 epe_active_old=float(e_old[act_old].mean()) if na_o else None))
            del img, old, u, d0, g_old, g_new
        print(json.dumps(dict(frame=f, rows=len(rows), elapsed=round(time.time() - started, 1))), flush=True)
    if not rows:
        raise SystemExit('no (camera, frame) pair had a cached target')

    def q(t):
        t = t.float()
        return dict(mean=float(t.mean()), p50=float(torch.quantile(t, .5)), p90=float(torch.quantile(t, .9)),
                    p99=float(torch.quantile(t, .99)), max=float(t.max()), n=int(t.numel()))

    to, tn = np.array([r['term_old'] for r in rows]), np.array([r['term_new'] for r in rows])
    cosv = np.array([r['grad_cos'] for r in rows])
    summary = dict(pairs=len(rows), valid_lr_pixels=tot['valid'], active_lr_pixels_old=tot['act_old'],
                   flow_diff_active_lr=q(torch.cat(pool['d_act'])), flow_diff_valid_lr=q(torch.cat(pool['d_valid'])),
                   endpoint_correction_hr_px=q(torch.cat(pool['hr_corr'])),
                   bicubic_vs_bilinear_active_lr=q(torch.cat(pool['bicubic_act'])),
                   hinge_flips_per_active=tot['hinge_flip_act'] / max(tot['act_old'], 1),
                   hinge_flips_per_valid=tot['hinge_flip_valid'] / max(tot['valid'], 1),
                   active_set_change=(tot['act_new'] - tot['act_old']) / max(tot['act_old'], 1),
                   active_set_symdiff_per_active=tot['act_sym'] / max(tot['act_old'], 1),
                   term_mean_old=float(to.mean()), term_mean_new=float(tn.mean()),
                   term_rel_change_pooled=float((tn.mean() - to.mean()) / max(to.mean(), 1e-12)),
                   term_rel_change_max_pair=float(np.max(np.abs(tn - to) / np.maximum(to, 1e-12))),
                   grad_cos_mean=float(cosv.mean()), grad_cos_min=float(cosv.min()),
                   grad_rel_change_mean=float(np.mean([r['grad_rel_change'] for r in rows])),
                   fixed_point_last_step_max=float(max(r['fixed_point_last_step'] for r in rows)))
    summary['closure_rule_met'] = bool(summary['hinge_flips_per_active'] < 0.01 and abs(summary['term_rel_change_pooled']) < 0.01)
    rec = dict(kind='warp_endpoint_check', doc=__doc__, checkpoint=cr.entry(ck), manifest=cr.entry(manifest_path),
               flow_dir=str(fdir.relative_to(ROOT)) if fdir.is_relative_to(ROOT) else str(fdir), D=a.D, frames=frames,
               iters=a.iters, summary=summary, rows=rows, host=__import__('socket').gethostname(),
               gpu=torch.cuda.get_device_name(), seconds=time.time() - started,
               sources_code={cr.rel(x): rt.sha(x) for x in (Path(__file__), HERE.parent / 'absorb_20261021/absorb_bound2.py')})
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=1))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
