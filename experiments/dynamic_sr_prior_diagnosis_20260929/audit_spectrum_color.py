#!/usr/bin/env python3
"""Section 6 and 10.2: additive RGB audit, five counterfactuals, legal IBP.

Usage --stage anchor-prepare (CPU), --stage anchor-metrics --device cuda,
--stage render --device cuda [--model NAME]. GPU allocation is external.
"""
from __future__ import annotations
import argparse
import csv
import importlib.util
import os
import sys
import time
from pathlib import Path
from spectral_common import *

OLD = ROOT / 'output/dynamic_sr_temporal_prior_20260928'
OBS = ROOT / 'output/dynamic_sr_prior_diagnosis_20260929/observables'

def dump_csv(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if not rows: return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, 'w') as f:
        w = csv.DictWriter(f, keys); w.writeheader(); w.writerows(rows)

def dataset():
    p = read(OLD / 'protocol.json'); mp = ROOT / p['manifest']['path']; m = read(mp)
    return p, m, mp.parent

def train_observations():
    p, m, dr = dataset()
    oo = sorted([o for o in m['observations'] if o['split'] == 'train' and o['frame_index'] in p['evaluation']['train_frames']], key=lambda o: (o['camera_id'], o['frame_index']))
    assert len(oo) == 76
    return p, oo, dr

def anchor_prepare():
    legal = OUT / 'legal_anchored_targets'; legal.mkdir(parents=True, exist_ok=True)
    pp = OUT / 'backprojection_protocol.json'
    if pp.exists(): prot = read(pp)
    else: prot = norm_protocol(); write(pp, prot)
    p, oo, dr = train_observations(); teachers = read(ROOT / p['teacher']['path'])['entries']
    ts = {(t['camera'], t['frame']): t for t in teachers}
    rows = []; begin = time.monotonic()
    for o in oo:
        c, f = o['camera_id'], o['frame_index']; t = ts[c, f]
        lp, sp = dr / o['lr_path'], ROOT / t['path']
        assert sha(lp) == o['lr_sha256'] and sha(sp) == t['sha256']
        dst = legal / c / f'{f:04d}.npy'; receipt = dst.with_suffix('.json')
        if receipt.exists(): rows.append(read(receipt)); continue
        l, s = rgb(lp), rgb(sp); a, trace = anchored_target(s, l, prot)
        dst.parent.mkdir(parents=True, exist_ok=True)
        # This subtree is legal: no HR path, values, labels, or weights.
        np.save(dst, np.clip(a, 0, 1).astype(np.float32))
        row = dict(camera=c, frame=f, path=str(dst.relative_to(ROOT)), sha256=sha(dst),
                   lr_path=str(lp.relative_to(ROOT)), lr_sha256=sha(lp), teacher_path=str(sp.relative_to(ROOT)), teacher_sha256=sha(sp),
                   protocol_sha256=sha(pp), original_closure=closure(s, l), raw_closure=closure(a, l),
                   target_closure=closure(np.clip(a, 0, 1), l), iteration_d0_mse=trace,
                   raw_out_of_range_fraction=float(((a < 0) | (a > 1)).mean()),
                   asset_class='legal_training_derived', shape=list(s.shape), format='float32 HWC NPY; endpoint clipped; unquantized')
        write(receipt, row); rows.append(row)
        print(f'anchor {len(rows)}/76 {c}/{f}', flush=True)
    write(legal / 'index.json', dict(status='completed_fixed_diagnostic_train76', entries=rows,
          inputs='registered train LR and frozen SwinIR only; no HR accessed by preparation',
          observations=76, seconds=time.monotonic() - begin, full_training_cache=False))

def evaluator(device):
    sys.path.insert(0, str(ROOT / 'experiments/dynamic_sr_20260918'))
    spec = importlib.util.spec_from_file_location('spectral_original_evaluate', ROOT / 'experiments/dynamic_sr_20260918/evaluate.py')
    legacy = importlib.util.module_from_spec(spec); spec.loader.exec_module(legacy)
    import lpips
    metric = lpips.LPIPS(net='alex', spatial=False).to(device).eval().requires_grad_(False)
    def quality(pred, gt):
        p = np.ascontiguousarray(np.clip(pred, 0, 1).astype(np.float32))
        g = np.ascontiguousarray(gt.astype(np.float32))
        # Identical primary arithmetic, avoiding unneeded second spatial LPIPS.
        mse = float(np.square(p - g).mean(axis=2).mean())
        sm = legacy.ssim_map_rgb(p, g)
        a = torch.from_numpy(p).permute(2, 0, 1)[None].to(device) * 2 - 1
        b = torch.from_numpy(g).permute(2, 0, 1)[None].to(device) * 2 - 1
        with torch.inference_mode(): lp = float(metric(a, b).item())
        return dict(psnr=legacy.metric_psnr(mse), ssim=float(sm[5:-5, 5:-5].mean()), lpips=lp,
                    clamped_metric_mse=mse)
    return quality

def means(rows, fields):
    return {k: float(np.mean([r[k] for r in rows])) for k in fields}

def anchor_metrics(device):
    q = evaluator(device); idx = read(OUT / 'legal_anchored_targets/index.json')
    p, oo, dr = train_observations(); observations = {(o['camera_id'], o['frame_index']): o for o in oo}
    rows = []; budgets = []; regions = []; rois = read(OUT / 'roi_protocol.json')['regions_by_camera_xyxy_exclusive']
    begin = time.monotonic()
    for e in idx['entries']:
        c, f = e['camera'], e['frame']; o = observations[c, f]; hp = dr / o['hr_path']
        assert sha(hp) == o['hr_sha256']; g = rgb(hp)
        for method, a in [('SwinIR', rgb(ROOT / e['teacher_path'])), ('Anchored8', np.load(ROOT / e['path']))]:
            identity = dict(model=method, camera=c, frame=f, split='train76_hr_diagnostic_only')
            row, _ = spectral_budget(a, g)
            budgets.append(dict(**identity, **row)); quality = q(a, g)
            closure_key = 'original_closure' if method == 'SwinIR' else 'target_closure'
            rows.append(dict(**identity, **quality, **e[closure_key]))
            regions.extend(dict(**identity, **rr) for rr in region_budget(a, g, rois.get(c, {})))
        print(f'anchor metric {c}/{f}', flush=True)
    dump_csv(OUT / 'anchor_quality.csv', rows); dump_csv(OUT / 'anchor_error_budget.csv', budgets); dump_csv(OUT / 'anchor_regions.csv', regions)
    fields = ['psnr', 'ssim', 'lpips', 'd0_mse', 'd0_l1', 'actual_D_mse', 'actual_D_l1']
    agg = {m: means([r for r in rows if r['model'] == m], fields) for m in ['SwinIR', 'Anchored8']}
    delta = {k: agg['Anchored8'][k] - agg['SwinIR'][k] for k in fields}
    bm = {m: means([r for r in budgets if r['model'] == m], ['dc_mse', 'low_mse', 'mid_mse', 'high_mse']) for m in agg}
    write(OUT / 'anchor_decision.json', dict(status='completed', recommend_target_anchoring=None,
          rule='Controller considers low-frequency bias and detail evidence with all three metrics; no unanimous-metric gate, 2% veto, or parameter scan',
          decision_status='quality_tradeoff_for_controller',
          aggregate=agg, delta_anchored_minus_swinir=delta, bands=bm, observations=76,
          source_split='76 fixed training views with genuine HR used only for diagnostic selection',
          formal_cache='76 is diagnostic; full legal training cache needed only if anchoring selected',
          gpu=os.environ.get('CUDA_VISIBLE_DEVICES'), device=device, seconds=time.monotonic() - begin,
          protocol_sha256=sha(OUT / 'backprojection_protocol.json'), script_sha256=sha(__file__),
          source_arithmetic_sha256=sha(ROOT / 'experiments/dynamic_sr_20260918/evaluate.py')))

def metadata(row, model):
    return dict(model=model, camera=row.get('camera', row.get('camera_id')), frame=row.get('frame', row.get('frame_index')),
                split=row.get('split'))

def render_audit(device, model_names):
    q = evaluator(device); rois = read(OUT / 'roi_protocol.json')['regions_by_camera_xyxy_exclusive']
    bp = read(OUT / 'backprojection_protocol.json'); p, mm, dr = dataset()
    observations = {(o['camera_id'], o['frame_index']): o for o in mm['observations']}
    for model in model_names:
        modelroot = OBS / model; dest = OUT / model; dest.mkdir(parents=True, exist_ok=True)
        if not (modelroot / 'complete.json').exists(): raise ValueError(f'incomplete observables: {modelroot}')
        budget = []; regions = []; counter = []; begin = time.monotonic()
        for npz in sorted(modelroot.glob('*.npz')):
            info = read(npz.with_suffix('.json')); c, fstr = npz.stem.rsplit('_', 1); f = int(fstr)
            o = observations[c, f]; key = dict(model=model, camera=c, frame=f, split='train' if o['split']=='train' else 'development', original_manifest_split=o['split'])
            rr = dest / f'{c}_{f:04d}.json'
            if rr.exists():
                cached = read(rr); budget.append(cached['budget']); regions.extend(cached['regions']); counter.extend(cached['counterfactuals']); continue
            raw = np.load(npz)['rgb_raw'].transpose(1, 2, 0); pred = np.clip(raw, 0, 1)
            hp = dr / o['hr_path']; assert sha(hp) == o['hr_sha256']; g = rgb(hp)
            row, trans = spectral_budget(pred, g, raw); br = dict(**key, **row, float_asset_sha256=sha(npz), hr_sha256=sha(hp))
            rs = [dict(**key, **r) for r in region_budget(pred, g, rois.get(c, {}))]
            cs = []; base = q(pred, g)
            cs.append(dict(**key, counterfactual='unchanged', **base, unclamped_mse=row['mse'], expected_unclamped_mse=row['mse'], exact_mse_residual=0.))
            expected = {'replace_dc': row['mse'] - row['dc_mse'], 'replace_low_non_dc': row['mse'] - row['low_mse'],
                        'replace_mid_high': row['mse'] - row['mid_mse'] - row['high_mse'], 'replace_chroma_q1q2': row['q0_mse']}
            for label, a in oracle_images(pred, trans):
                mse = float(np.square(a - g).mean()); residual = abs(mse - expected[label]); assert residual < 1e-11
                qs = q(a, g)
                cs.append(dict(**key, counterfactual=label, **qs, unclamped_mse=mse, expected_unclamped_mse=expected[label],
                               exact_mse_residual=residual, outside_01_fraction=float(((a < 0) | (a > 1)).mean()),
                               diagnostic_only=True, legal_training_LR_used=False))
            if o['split'] == 'train':
                lp = dr / o['lr_path']; assert sha(lp) == o['lr_sha256']; lr = rgb(lp)
                a, trace = anchored_target(pred, lr, bp); qs = q(a, g)
                cs.append(dict(**key, counterfactual='LR_backprojection8', **qs, unclamped_mse=float(np.square(a - g).mean()),
                               expected_unclamped_mse=None, exact_mse_residual=None, diagnostic_only=True, legal_training_LR_used=True,
                               original_D0_mse=closure(pred, lr)['d0_mse'], corrected_D0_mse=closure(np.clip(a, 0, 1), lr)['d0_mse']))
            else:
                cs.append(dict(**key, counterfactual='LR_backprojection8', status='NA', reason='No legal observed LR at held-out novel view; no cam00/01 LR read'))
            write(rr, dict(budget=br, regions=rs, counterfactuals=cs, observation_metadata_sha256=sha(npz.with_suffix('.json'))))
            budget.append(br); regions.extend(rs); counter.extend(cs)
            if len(budget) % 10 == 0: print(f'{model}: {len(budget)}/196', flush=True)
        assert len(budget) == 196
        dump_csv(dest / 'error_budget.csv', budget); dump_csv(dest / 'region_budget.csv', regions); dump_csv(dest / 'counterfactuals.csv', counter)
        write(dest / 'complete.json', dict(status='completed', observations=len(budget), seconds=time.monotonic() - begin,
              counterfactual_rows=len(counter), valid_LR_backprojection_rows=76, gpu=os.environ.get('CUDA_VISIBLE_DEVICES'), device=device,
              source_sha256={Path(x).name:sha(x) for x in [__file__, ROOT / 'experiments/dynamic_sr_prior_diagnosis_20260929/spectral_common.py']},
              metric_arithmetic_sha256=sha(ROOT / 'experiments/dynamic_sr_20260918/evaluate.py')))

def summarize():
    model_names = ['LR6k', 'HR6k', 'U6000', 'r1_J1', 'r1_Async2']; rows = []; regs = []; cf = []
    for model in model_names:
        for path in sorted((OUT / model).glob('cam*.json')):
            d = read(path); rows.append(d['budget']); regs.extend(d['regions']); cf.extend(d['counterfactuals'])
    dump_csv(OUT / 'error_budget.csv', rows); dump_csv(OUT / 'region_budget.csv', regs); dump_csv(OUT / 'counterfactuals.csv', cf)
    agg = []; ca = []; fields = ['mse', 'dc_mse', 'low_mse', 'mid_mse', 'high_mse', 'q0_mse', 'q1_mse', 'q2_mse', 'q_chroma_mse', 'psnr_from_frame_mse']
    for model in model_names:
        for camera in ['cam00', 'cam01', 'train76']:
            select = lambda r: r['model'] == model and (r['split'] == 'train' if camera == 'train76' else r['camera'] == camera)
            rr = [r for r in rows if select(r)]
            if not rr: continue
            a = dict(model=model, camera=camera, observations=len(rr), **means(rr, fields))
            for band in ['dc', 'low', 'mid', 'high']: a[band + '_error_share'] = a[band + '_mse'] / a['mse']
            agg.append(a)
            for label in sorted({r['counterfactual'] for r in cf}):
                cr = [r for r in cf if select(r) and r['counterfactual'] == label and 'psnr' in r]
                if cr: ca.append(dict(model=model, camera=camera, counterfactual=label, observations=len(cr), **means(cr, ['psnr','ssim','lpips'])))
    dump_csv(OUT / 'error_budget_aggregate.csv', agg); dump_csv(OUT / 'counterfactuals_aggregate.csv', ca)
    # Preserve signed comparison to HR-direct; regions remain spatial-only.
    hrr = {(r['camera'],r['frame'],r['region']): r for r in regs if r['model'] == 'HR6k'}
    diffs = []
    for r in regs:
        if r['model'] not in ['U6000','r1_J1','r1_Async2']: continue
        h = hrr.get((r['camera'],r['frame'],r['region']))
        if h: diffs.append(dict(model=r['model'], camera=r['camera'], frame=r['frame'], region=r['region'],
                                current_minus_HRmodel_mse_contribution=r['mse_contribution']-h['mse_contribution']))
    dump_csv(OUT / 'regional_current_minus_hrmodel.csv', diffs)
    write(OUT / 'arithmetic_audit.json', dict(status='completed' if len(rows)==980 else 'partial', rows=len(rows),
          max_parseval_abs_residual=max([r['parseval_abs_residual'] for r in rows],default=None),
          max_color_additivity_abs_residual=max([r['color_additivity_abs_residual'] for r in rows],default=None),
          max_oracle_mse_abs_residual=max([r['exact_mse_residual'] for r in cf if r.get('exact_mse_residual') is not None],default=None),
          bands='DC; max(kx/W,ky/H)<=1/4 excluding DC; 1/4<max<=1/2; max>1/2; exact integer membership',
          dct='full image 2D per RGB channel, norm=ortho, float64; contributions divide 3HW',
          color='orthogonal q0=(R+G+B)/sqrt3 q1=(R-G)/sqrt2 q2=(R+G-2B)/sqrt6; q0 not standard luminance',
          spatial='fixed disjoint priority ROIs plus rest; no spatial-frequency matrix or additive patch-frequency claim',
          primary_metrics='clipped float32 sRGB; mean per-frame PSNR, original 11x11 sigma1.5 SSIM interior, AlexNet v0.1 scalar LPIPS',
          counterfactual='oracle replacements diagnostic only; exact MSE before clipping, three metrics after clip; IBP only legal training views',
          roi_sha256=sha(OUT/'roi_protocol.json')))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--stage',choices=['anchor-prepare','anchor-metrics','render','summarize'],required=True)
    ap.add_argument('--device',default='cpu',choices=['cpu','cuda']); ap.add_argument('--model',action='append'); a=ap.parse_args()
    torch.set_num_threads(4)
    if a.stage=='anchor-prepare': anchor_prepare()
    elif a.stage=='anchor-metrics': anchor_metrics(a.device)
    elif a.stage=='render': render_audit(a.device,a.model or ['LR6k','HR6k','U6000','r1_J1','r1_Async2'])
    else: summarize()

if __name__=='__main__': main()
