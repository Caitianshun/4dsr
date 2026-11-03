"""Zero-training split of the P+C v2 residuals by candidate free-set gates, at a LOCO-excluded camera (Claude 4dsr-4e, 2026-11-01).
Follow-up of pc_heldout_diag.py, whose result rejected the occlusion hypothesis: the cam10 loss of Cook07 x4 / Coffee07 x2 comes from
free Gaussians that are visible in >= 2 fitted cameras, in the rings around the person and in view-dependent background (window
glass, outdoor trees). Render only: no training, no HR image, no cam00.
Per frame f, for the free set F of the LOCO fit (residual mu = dxyz / px_world in px units, c = dsh0), features from the parent state
S(f) and the fitted cameras only (never C):
  agr    multi-view gradient agreement at the parent state (residual 0): for every fitted camera j the gradient g_ij of the fitter's
         camera loss L_j = |area(camera model_j(render)) - y_j|_1 / (3 |mask_j|) with respect to Gaussian i's residual;
         agr_P = |sum_j g_ij^mu| / sum_j |g_ij^mu|, agr_C likewise for c, agr = (agr_P + agr_C) / 2 (1 = all cameras agree).
         Gaussians without gradient in any fitted camera form the class 'nograd'; gates never drop them.
  rho    median over fitted cameras of the camera-space depth / the fitter's reference depth (px_world * fx)
  defo   parent deformation |x_i(f) - x_i(canonical)| / px_world (HR px at the reference depth)
  rmask  in-mask fraction of the fitted cameras' alpha weight, sum_j W_ij / sum_j A_ij (W, A as in pc_heldout_diag.py)
The fitter's free set and reference depth are recomputed from the fitted cameras and checked against the fit (idx, px_world;
abort below Jaccard 0.99 or 1 % depth difference, else recorded).
Arms = the fit's residuals (dxyz and dsh0) applied to a subset of F:
  agr_drop25 / agr_drop50 / agr_drop75   drop the 25 / 50 / 75 % lowest-agr Gaussians (per-frame quantiles among those with gradient)
  agr_bot50only                          only the 50 % lowest-agr Gaussians
  agrP_drop50 / agrC_drop50              drop the 50 % lowest by agr_P / agr_C
  rand_drop50                            drop a random 50 % of the Gaussians with gradient (seed 0; equal-size control)
  rmask_drop50 / rmask_bot50only         by rmask (same quantile rule)
  rho_near (1/1.5 <= rho < 1.5) / rho_dropfar (rho < 1.5) / rho_faronly (rho >= 1.5)
  defo_moving (defo >= 1) / defo_still (defo < 1)
Read-outs per arm: at C as loco_eval (LR, observation = area(camera model of C applied to the render), clamped): pooled PSNR change
against the parent per region (M, R2, R6, R16, far as in pc_heldout_diag.py, and all) and each region's share of the change; at the
fitted cameras every --closure-every frames: pooled LR PSNR change against the parent, whole image and inside their masks (gain of
the fitting objective kept by the subset). Feature quantiles, per-frame Spearman correlations and per-arm kept fractions (count,
residual energy) are recorded. Writes OUT/gate.json.
  pc_gate_diag.py --checkpoint CK --fit LOCO_FIT_DIR --camera cam10 --out OUT [--frames 0-299] [--closure-every 5]"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import socket
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ABS = HERE.parent / 'absorb_20261021'
if str(ABS) not in sys.path:
    sys.path.insert(0, str(ABS))

import absorb_bound2 as ab  # noqa: E402
import perframe_fit as pf  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

cr, native, ev = ab.cr, ab.native, ab.ev
ROOT = ab.ROOT
SH_C0 = 0.28209479177387814
REGIONS = ('M', 'R2', 'R6', 'R16', 'far')
ARMS = ('parent', 'full', 'agr_drop25', 'agr_drop50', 'agr_drop75', 'agr_bot50only', 'agrP_drop50', 'agrC_drop50', 'rand_drop50',
        'rmask_drop50', 'rmask_bot50only', 'rho_near', 'rho_dropfar', 'rho_faronly', 'defo_moving', 'defo_still')
FEATURES = ('agr', 'agr_P', 'agr_C', 'rho', 'defo', 'rmask', 'mu_px', 'dsh0')


def frames_arg(s):
    out = []
    for part in s.split(','):
        a, _, b = part.partition('-')
        out += list(range(int(a), int(b) + 1)) if b else [int(a)]
    return out


def psnr(se, n):
    return -10 * math.log10(max(se / max(n, 1), 1e-12))


def subset(res, sel):
    return dict(idx=res['idx'][sel], dxyz=None if res.get('dxyz') is None else res['dxyz'][sel],
                dsh0=None if res.get('dsh0') is None else res['dsh0'][sel])


def spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    if rx.size < 3 or rx.std() == 0 or ry.std() == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--fit', required=True)
    p.add_argument('--camera', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--frames', default='0-299')
    p.add_argument('--closure-every', type=int, default=5)
    a = p.parse_args()
    out = ROOT / a.out
    if (out / 'gate.json').exists():
        raise FileExistsError(f'completed diagnostic preserved: {out}')
    out.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dev = 'cuda'
    started = time.time()
    ck = ROOT / a.checkpoint
    payload = torch.load(ck, map_location='cpu', weights_only=False)
    plan, raw_aff = payload['plan'], payload.get('affine')
    del payload
    manifest_path, manifest = cr.plan_scene(plan)
    root = Path(manifest['_root'])
    train = list(manifest['splits']['train'])
    c = a.camera
    if c not in train or c == 'cam00':
        raise SystemExit('--camera must be a training camera (never cam00)')
    model = ab.camera_model(raw_aff, dev)
    g = ab.load_model(ck, plan, dev)
    h, opt, pipe, sh = cr.configuration(plan)
    cull_record = cr.cull(g, cr.Culler(manifest, device=dev), None, 'posthoc', optimizer_backed=False)
    xyz_sha = hashlib.sha256(g._xyz.detach().float().cpu().numpy().tobytes()).hexdigest()
    ck_sha = cr.entry(ck)['sha256']
    fit = ROOT / a.fit
    run = json.loads((fit / 'run.json').read_text())
    if run.get('checkpoint_sha256') != ck_sha or int(run.get('points_after_cull', -1)) != int(g._xyz.shape[0]) \
            or run.get('post_cull_xyz_sha256') != xyz_sha:
        raise SystemExit('the fit does not belong to this checkpoint / cull')
    excluded = list(run.get('excluded_cameras') or [])
    if c not in excluded:
        raise SystemExit(f'the fit did not exclude {c} (excluded={excluded})')
    fitted = [cc for cc in train if cc not in excluded]
    lr_w, lr_h = manifest['resolutions']['lr']
    hr_w, hr_h = manifest['resolutions']['hr']
    s = hr_w // lr_w
    if hr_w != s * lr_w or hr_h != s * lr_h:
        raise SystemExit('HR is not an integer multiple of LR')
    obs = {(o['camera_id'], int(o['frame_index'])): o for o in manifest['observations']}
    index = {cc: i for i, cc in enumerate(train)}
    background = torch.ones(3, device=dev) if cr.white_background(plan) else torch.zeros(3, device=dev)
    names = list(raw_aff['names'])
    delta = raw_aff['delta'].double() - raw_aff['delta'].double().mean(0)

    def canon_fn(cc):
        Ainv = torch.linalg.inv(torch.eye(3, dtype=torch.float64) + delta[names.index(cc), :, :3]).float().to(dev)
        b = delta[names.index(cc), :, 3].float().to(dev)
        return lambda y: torch.einsum('ij,jhw->ihw', Ainv, y - b[:, None, None])

    cams_all = [c] + fitted
    canon = {cc: canon_fn(cc) for cc in cams_all}
    k3 = torch.ones(3, 3, device=dev)
    medbg = {cc: torch.stack([canon[cc](ab.read_rgb(root / obs[(cc, f)]['lr_path'], dev)) for f in range(0, 300, 5)]).median(0).values
             for cc in cams_all}
    rings = {r: ab.ellipse(r, dev) for r in (2, 6, 16)}
    fx = float(np.asarray(manifest['cameras'][fitted[0]]['K_hr'])[0][0])
    gen = torch.Generator(device=dev).manual_seed(0)
    acc = {arm: {r: [0.0, 0] for r in REGIONS + ('all',)} for arm in ARMS}
    clo = {arm: dict(se=0.0, n=0, se_d=0.0, n_d=0) for arm in ARMS}
    kept = {arm: dict(count=[], mu2=[], c2=[]) for arm in ARMS}
    fq = {k: [] for k in FEATURES}
    corr_rows, checks = [], []
    frames = frames_arg(a.frames)

    def render(cam_, st):
        return native.native_rgb(cam_, g, st, pipe, background)['render']

    def observe(cc, cam_, st):
        return ab.area(model.apply(render(cam_, st), cc), (lr_h, lr_w))

    for f in frames:
        o = obs[(c, f)]
        with torch.no_grad():
            S = {k: v.detach() for k, v in native.native_effective_state(g, float(o['time']), torch).items()}
        res = pf.load_residual(fit / f'f{f:04d}.npz', dev)
        idx = res['idx']
        n = int(idx.numel())
        with np.load(fit / f'f{f:04d}.npz') as z:
            px_world = float(z['px_world'])
        ys = {cc: ab.read_rgb(root / obs[(cc, f)]['lr_path'], dev) for cc in fitted}
        masks = {cc: ab.dilate((canon[cc](ys[cc]) - medbg[cc]).abs().amax(0) > 0.06, k3) for cc in fitted}
        cams = {cc: native.hr_camera(ev.camera_item(manifest, obs[(cc, f)], torch), index[cc], manifest) for cc in fitted}
        # the fitter's free-set rule and reference depth, recomputed (check), plus each Gaussian's depth per fitted camera
        count = torch.zeros(S['xyz'].shape[0], dtype=torch.int32, device=dev)
        hit_depth, zs = [], []
        hom = torch.cat([S['xyz'], torch.ones_like(S['xyz'][:, :1])], 1)
        for cc in fitted:
            ph = hom @ cams[cc].full_proj_transform.to(dev)
            wv = hom @ cams[cc].world_view_transform.to(dev)
            ndc = ph[:, :2] / ph[:, 3:4].clamp_min(1e-6)
            px_ = ((ndc[:, 0] + 1) * hr_w - 1) * 0.5
            py_ = ((ndc[:, 1] + 1) * hr_h - 1) * 0.5
            lx = torch.floor((px_ + 0.5) / (hr_w // lr_w)).long()
            ly = torch.floor((py_ + 0.5) / (hr_h // lr_h)).long()
            ok = (wv[:, 2] > 0.2) & (lx >= 0) & (lx < lr_w) & (ly >= 0) & (ly < lr_h)
            hit = torch.zeros_like(ok)
            hit[ok] = masks[cc][ly[ok], lx[ok]]
            count += hit.int()
            hit_depth.append(wv[hit, 2])
            zf = wv[idx, 2].clone()
            zf[zf <= 0.2] = float('nan')
            zs.append(zf)
        free = torch.nonzero(count >= 2).squeeze(1)
        same = free.numel() == n and bool(torch.equal(free, idx.to(free.dtype)))
        inter = int(torch.isin(free, idx).sum()) if free.numel() and n else 0
        jac = inter / max(free.numel() + n - inter, 1)
        px_re = float(torch.cat(hit_depth).median()) / fx if free.numel() else 0.0
        checks.append(dict(frame=f, free_equal=same, jaccard=jac, px_world=px_world, px_world_recomputed=px_re))
        if jac < 0.99 or abs(px_re - px_world) > 1e-2 * max(px_world, 1e-12):
            raise SystemExit(f'frame {f}: recomputed free set / reference depth differ from the fit: {checks[-1]}')
        y = ab.read_rgb(root / o['lr_path'], dev)
        M = ab.dilate((canon[c](y) - medbg[c]).abs().amax(0) > 0.06, k3)
        D2, D6, D16 = (ab.dilate(M, rings[r]) for r in (2, 6, 16))
        reg = dict(M=M, R2=D2 & ~M, R6=D6 & ~D2, R16=D16 & ~D6, far=~D16)
        cam = native.hr_camera(ev.camera_item(manifest, o, torch), index[c], manifest)
        sel = {'parent': None}
        if n:
            # features
            mu0 = torch.zeros(n, 3, device=dev, requires_grad=True)
            c0 = torch.zeros(n, 3, device=dev, requires_grad=True)
            Gm = torch.zeros(n, 3, device=dev)
            Gc = torch.zeros(n, 3, device=dev)
            Nm = torch.zeros(n, device=dev)
            Nc = torch.zeros(n, device=dev)
            Wsum = torch.zeros(n, device=dev)
            Asum = torch.zeros(n, device=dev)
            for cc in fitted:
                st = dict(S)
                st['xyz'] = S['xyz'].index_add(0, idx, mu0 * px_world)
                sh_ = S['sh'].clone()
                sh_[idx, 0, :] = sh_[idx, 0, :] + c0
                st['sh'] = sh_
                L = (observe(cc, cams[cc], st) - ys[cc]).abs().sum() / (3 * max(int(masks[cc].sum()), 1))
                gm, gc = torch.autograd.grad(L, [mu0, c0])
                Gm += gm
                Gc += gc
                Nm += gm.norm(dim=1)
                Nc += gc.norm(dim=1)
                shz = torch.zeros_like(S['sh']).requires_grad_(True)
                img = render(cams[cc], dict(S, sh=shz))
                up = masks[cc].float().repeat_interleave(s, 0).repeat_interleave(s, 1)
                Lw = (img[0] * up).sum() / s ** 2 + img[1].sum() / s ** 2
                gsh, = torch.autograd.grad(Lw, shz)
                Wsum += gsh[idx, 0, 0] / SH_C0
                Asum += gsh[idx, 0, 1] / SH_C0
            has = (Nm + Nc) > 0
            agr_P = torch.where(Nm > 0, Gm.norm(dim=1) / Nm.clamp_min(1e-30), torch.zeros_like(Nm))
            agr_C = torch.where(Nc > 0, Gc.norm(dim=1) / Nc.clamp_min(1e-30), torch.zeros_like(Nc))
            agr = (agr_P + agr_C) / 2
            rho = torch.nanmedian(torch.stack(zs), dim=0).values / (px_world * fx)
            rho = torch.nan_to_num(rho, nan=float('inf'))
            defo = (S['xyz'][idx] - g.get_xyz[idx].detach()).norm(dim=1) / px_world
            rmask = torch.where(Asum > 0, Wsum / Asum.clamp_min(1e-30), torch.zeros_like(Asum))
            mu_px = res['dxyz'].norm(dim=1) / px_world if res.get('dxyz') is not None else torch.zeros(n, device=dev)
            dc = res['dsh0'].norm(dim=1) if res.get('dsh0') is not None else torch.zeros(n, device=dev)
            feats = dict(agr=agr, agr_P=agr_P, agr_C=agr_C, rho=rho, defo=defo, rmask=rmask, mu_px=mu_px, dsh0=dc)
            for k, v in feats.items():
                vv = v[has] if k.startswith('agr') else v
                vv = vv[torch.isfinite(vv)]
                if vv.numel():
                    fq[k].append(torch.quantile(vv.float(), torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9], device=dev)).tolist())
            if a.closure_every > 0 and f % a.closure_every == 0:
                hv = has.cpu().numpy()
                fn = {k: v.detach().cpu().numpy() for k, v in feats.items()}
                corr_rows.append({f'{x}~{y_}': spearman(fn[x][hv], fn[y_][hv]) for i, x in enumerate(FEATURES)
                                  for y_ in FEATURES[i + 1:]})

            def drop_low(v, q):
                if not has.any():
                    return torch.ones(n, dtype=torch.bool, device=dev)
                thr = torch.quantile(v[has].float(), q)
                return ~(has & (v < thr))

            def low_only(v, q):
                if not has.any():
                    return torch.zeros(n, dtype=torch.bool, device=dev)
                thr = torch.quantile(v[has].float(), q)
                return has & (v < thr)

            rnd = torch.rand(n, generator=gen, device=dev)
            masks_sel = dict(
                full=torch.ones(n, dtype=torch.bool, device=dev),
                agr_drop25=drop_low(agr, 0.25), agr_drop50=drop_low(agr, 0.5), agr_drop75=drop_low(agr, 0.75),
                agr_bot50only=low_only(agr, 0.5), agrP_drop50=drop_low(agr_P, 0.5), agrC_drop50=drop_low(agr_C, 0.5),
                rand_drop50=~(has & (rnd < 0.5)),
                rmask_drop50=drop_low(rmask, 0.5), rmask_bot50only=low_only(rmask, 0.5),
                rho_near=(rho >= 1 / 1.5) & (rho < 1.5), rho_dropfar=rho < 1.5, rho_faronly=rho >= 1.5,
                defo_moving=defo >= 1.0, defo_still=defo < 1.0)
            mu2t = max(float((mu_px ** 2).sum()), 1e-30)
            c2t = max(float((dc ** 2).sum()), 1e-30)
            for arm, m in masks_sel.items():
                sel[arm] = subset(res, m)
                kept[arm]['count'].append(float(m.float().mean()))
                kept[arm]['mu2'].append(float((mu_px[m] ** 2).sum()) / mu2t)
                kept[arm]['c2'].append(float((dc[m] ** 2).sum()) / c2t)
        else:
            for arm in ARMS[1:]:
                sel[arm] = None
        with torch.no_grad():
            states = {arm: (S if sel[arm] is None else pf.apply_residual(S, sel[arm])) for arm in ARMS}
            for arm in ARMS:
                e2 = ((observe(c, cam, states[arm]) - y) ** 2).sum(0)
                for r, m in reg.items():
                    acc[arm][r][0] += float(e2[m].sum())
                    acc[arm][r][1] += int(m.sum()) * 3
                acc[arm]['all'][0] += float(e2.sum())
                acc[arm]['all'][1] += e2.numel() * 3
            if a.closure_every > 0 and f % a.closure_every == 0:
                for arm in ARMS:
                    for cc in fitted:
                        e2 = ((observe(cc, cams[cc], states[arm]) - ys[cc]) ** 2).sum(0)
                        clo[arm]['se'] += float(e2.sum())
                        clo[arm]['n'] += e2.numel() * 3
                        clo[arm]['se_d'] += float(e2[masks[cc]].sum())
                        clo[arm]['n_d'] += int(masks[cc].sum()) * 3
            del states
        if f % 25 == 0:
            print(json.dumps(dict(frame=f, free=n, check=checks[-1]['jaccard'],
                                  d_all={arm: round(psnr(acc[arm]['all'][0], acc[arm]['all'][1])
                                                    - psnr(acc['parent']['all'][0], acc['parent']['all'][1]), 3)
                                         for arm in ('full', 'agr_drop50', 'rand_drop50', 'rho_dropfar', 'defo_moving')},
                                  seconds=round(time.time() - started, 1))), flush=True)
    summ = {}
    p0 = {r: psnr(*acc['parent'][r]) for r in REGIONS + ('all',)}
    c0_ = dict(all=psnr(clo['parent']['se'], clo['parent']['n']), dyn=psnr(clo['parent']['se_d'], clo['parent']['n_d']))
    full_all = psnr(*acc['full']['all']) - p0['all']
    full_clo = psnr(clo['full']['se_d'], clo['full']['n_d']) - c0_['dyn']
    for arm in ARMS:
        d = {r: psnr(*acc[arm][r]) - p0[r] for r in REGIONS + ('all',)}
        tot = acc[arm]['all'][0] - acc['parent']['all'][0]
        share = {r: ((acc[arm][r][0] - acc['parent'][r][0]) / tot if abs(tot) > 0 else None) for r in REGIONS}
        ca = psnr(clo[arm]['se'], clo[arm]['n']) - c0_['all']
        cd = psnr(clo[arm]['se_d'], clo[arm]['n_d']) - c0_['dyn']
        summ[arm] = dict(cam_d_psnr=d, cam_share_of_change=share, closure_d_psnr=dict(all=ca, dyn=cd),
                         cam_all_fraction_of_full=(d['all'] / full_all if abs(full_all) > 1e-9 else None),
                         closure_dyn_fraction_of_full=(cd / full_clo if abs(full_clo) > 1e-9 else None),
                         kept_count=float(np.mean(kept[arm]['count'])) if kept[arm]['count'] else None,
                         kept_mu2=float(np.mean(kept[arm]['mu2'])) if kept[arm]['mu2'] else None,
                         kept_c2=float(np.mean(kept[arm]['c2'])) if kept[arm]['c2'] else None)
    corr = {}
    for k in (corr_rows[0].keys() if corr_rows else []):
        v = [r[k] for r in corr_rows if r[k] is not None]
        corr[k] = float(np.median(v)) if v else None
    feat_q = {k: (np.median(np.asarray(v), axis=0).tolist() if v else None) for k, v in fq.items()}
    rec = dict(doc=__doc__, kind='pc_gate_diag', checkpoint=cr.entry(ck), fit=a.fit, camera=c, fitted=fitted,
               frames=[frames[0], frames[-1], len(frames)], closure_every=a.closure_every, cull_record=cull_record,
               post_cull_xyz_sha256=xyz_sha, points=int(g._xyz.shape[0]), parent=dict(cam=p0, closure=c0_), summary=summ,
               feature_quantiles=dict(q=[0.1, 0.25, 0.5, 0.75, 0.9], median_over_frames=feat_q), spearman_median=corr,
               free_set_check=dict(all_equal=all(x['free_equal'] for x in checks), min_jaccard=min(x['jaccard'] for x in checks),
                                   max_px_world_rel_diff=max(abs(x['px_world_recomputed'] - x['px_world']) / max(x['px_world'], 1e-12)
                                                             for x in checks)),
               host=socket.gethostname(), gpu=torch.cuda.get_device_name(), seconds=time.time() - started)
    (out / 'gate.json').write_text(json.dumps(rec, indent=1))
    print(json.dumps(dict(cam_all={arm: round(v['cam_d_psnr']['all'], 3) for arm, v in summ.items()},
                          closure_dyn={arm: round(v['closure_d_psnr']['dyn'], 3) for arm, v in summ.items()},
                          kept={arm: (round(v['kept_count'], 3) if v['kept_count'] is not None else None) for arm, v in summ.items()},
                          check=rec['free_set_check'], seconds=round(rec['seconds'], 1))), flush=True)


if __name__ == '__main__':
    main()
