"""Frozen LR-only temporal target construction, followed by separate HR evaluation.

This is a teacher diagnostic, not a reconstruction result. No model is updated.
Run build first; evaluate verifies its source/protocol/input and output hashes.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import shutil
import socket
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
SCENES = {
    'cook_spinach': ('n3dv_prepared/cook_spinach', ['cam02', 'cam06']),
    'meetroom_discussion': ('meetroom_prepared/discussion', ['cam02', 'cam04']),
}
ANCHORS = [16, 32, 48, 64, 80, 104]
OFFSETS = [-4, -2, 2, 4]
FLOW_ARGS = (.5, 4, 21, 5, 7, 1.5, 0)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    tmp.replace(path)


def rgb(path):
    return np.asarray(Image.open(path).convert('RGB'), dtype=np.float32) / 255


def tensor(x):
    return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))).unsqueeze(0)


def resize(x, size, aa=False):
    return F.interpolate(tensor(x), size=size, mode='bicubic', align_corners=False,
                         antialias=aa)[0].permute(1, 2, 0).numpy()


def down(x, size):
    return np.clip(resize(x, size, True), 0, 1)


def high(x, lrsize):
    # Linear filtering for residual extraction; clipping belongs only to the
    # observed LR operator / image metrics, not to the detail decomposition.
    return x - resize(resize(x, lrsize, True), x.shape[:2])


def warp(x, flow):
    h, w = flow.shape[:2]
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    return cv2.remap(x, xx + flow[..., 0], yy + flow[..., 1], cv2.INTER_CUBIC,
                     borderMode=cv2.BORDER_REFLECT_101)


def lift(x, shape, nearest=False):
    return cv2.resize(x, (shape[1], shape[0]), interpolation=(cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR))


def combine(base, neighbor_high, weights, eta):
    den = weights.sum(0)
    avg = (neighbor_high * weights[..., None]).sum(0) / np.maximum(den[..., None], 1e-12)
    bh = high(base, (base.shape[0] // 4, base.shape[1] // 4))
    actual = eta * (den > 1e-12)
    return base + actual[..., None] * (avg - bh)


def checks():
    rng = np.random.default_rng(20260923)
    base = rng.uniform(.1, .9, (32, 40, 3)).astype(np.float32)
    hs = rng.normal(0, .1, (4, 32, 40, 3)).astype(np.float32)
    zero = np.zeros((32, 40), np.float32)
    a = combine(base, hs, np.ones((4, 32, 40), np.float32), zero)
    b = combine(base, hs, np.zeros((4, 32, 40), np.float32), zero + .5)
    assert np.array_equal(a, base) and np.array_equal(b, base)
    pred = tensor(base * .8).requires_grad_()
    losses = [F.l1_loss(pred, tensor(t)) for t in [base, a, b]]
    grads = [torch.autograd.grad(loss, pred, retain_graph=True)[0] for loss in losses]
    assert torch.equal(grads[0], grads[1]) and torch.equal(grads[0], grads[2])
    # A known horizontal translation tests pull-flow direction and x4 scaling.
    x = rng.random((24, 32, 3), dtype=np.float32)
    y = np.roll(x, 1, axis=1)
    flow = np.zeros((24, 32, 2), np.float32); flow[..., 0] = 1
    assert np.max(np.abs(warp(y, flow)[:, 2:-2] - x[:, 2:-2])) < 1e-6
    return dict(eta_zero_exact=True, empty_support_exact=True, target_loss_gradient_exact=True,
                pull_flow_direction=True, scope='CPU target/loss derivative; model runtime audited separately')


def flow_states(lrs, anchor):
    ref = lrs[anchor]; h, w = ref.shape[:2]
    gray = {f: cv2.cvtColor(np.round(x * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY) for f, x in lrs.items()}
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    interior = (xx >= 10) & (xx < w-10) & (yy >= 10) & (yy < h-10)
    states = []
    for off in OFFSETS:
        frame = anchor + off
        f = cv2.calcOpticalFlowFarneback(gray[anchor], gray[frame], None, *FLOW_ARGS)
        b = cv2.calcOpticalFlowFarneback(gray[frame], gray[anchor], None, *FLOW_ARGS)
        d = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM).calc(gray[anchor], gray[frame], None)
        fb = np.linalg.norm(f + warp(b, f), axis=2)
        disagreement = np.linalg.norm(f-d, axis=2)
        inside = interior & (xx+f[..., 0] >= 10) & (xx+f[..., 0] < w-10) & (yy+f[..., 1] >= 10) & (yy+f[..., 1] < h-10)
        valid0 = inside & (fb < .5) & (disagreement < .5)
        # Exclude neighboring warp discontinuities from the D/U filter support.
        valid = cv2.erode(valid0.astype(np.uint8), np.ones((11, 11), np.uint8)).astype(bool)
        r = valid * np.exp(-.5 * (fb/.25)**2 - .5 * (disagreement/.35)**2)
        states.append(dict(frame=frame, flow=f, fb=fb, disagreement=disagreement,
                           valid=valid, r=r.astype(np.float32), warped_lr=warp(lrs[frame], f)))
    eigen = cv2.cornerMinEigenVal(gray[anchor].astype(np.float32)/255, 5, ksize=3)
    return states, eigen


def patches(states, eigen):
    h, w = eigen.shape
    motion = np.median(np.stack([np.linalg.norm(s['flow'], axis=2) for s in states]), axis=0)
    support = np.mean(np.stack([s['valid'] for s in states]), axis=0)
    candidates = [(x, y) for y in range(18, h-18, 16) for x in range(18, w-18, 16)]
    selected = []
    def add(x, y, kind):
        if all((x-v['x'])**2+(y-v['y'])**2 >= 18**2 for v in selected):
            selected.append(dict(x=int(x), y=int(y), selection=kind, texture=float(eigen[y, x]),
                                 motion=float(motion[y, x]), valid_fraction=float(support[y, x])))
            return True
        return False
    for yy in [h//3, 2*h//3]:
        for xx in [w//3, 2*w//3]:
            add(xx, yy, 'grid')
    for kind, score, n in [('texture', eigen, 3), ('motion', motion, 3), ('uncertain', 1-support, 2)]:
        count = 0
        for x, y in sorted(candidates, key=lambda xy: (-float(score[xy[1], xy[0]]), xy[1], xy[0])):
            if add(x, y, kind):
                count += 1
                if count == n:
                    break
    return selected


def exposure_fit(ref, other, valid, patch_list):
    # All evaluated patches + margin are excluded from exposure estimation.
    fit = valid.copy()
    for p in patch_list:
        x, y = p['x'], p['y']; fit[max(y-12, 0):y+12, max(x-12, 0):x+12] = False
    fit &= np.all((other > .02) & (other < .98) & (ref > .02) & (ref < .98), axis=2)
    if fit.sum() < 512:
        return np.ones(3, np.float32), np.zeros(3, np.float32), dict(known=False, pixels=int(fit.sum()))
    gains, biases = [], []
    for ch in range(3):
        x, y = other[..., ch][fit].astype(np.float64), ref[..., ch][fit].astype(np.float64)
        if np.std(x) < .02:
            return np.ones(3, np.float32), np.zeros(3, np.float32), dict(known=False, pixels=int(fit.sum()), reason='low_variance')
        design = np.stack([x, np.ones_like(x)], axis=1)
        weight = np.ones_like(x)
        for _ in range(3):
            a, b = np.linalg.lstsq(design * np.sqrt(weight[:, None]), y * np.sqrt(weight), rcond=None)[0]
            a, b = np.clip(a, .8, 1.25), np.clip(b, -.1, .1)
            residual = np.abs(a*x+b-y)
            weight = np.minimum(1, .02 / np.maximum(residual, 1e-6))
        gains.append(a); biases.append(b)
    return np.array(gains, np.float32), np.array(biases, np.float32), dict(known=True, pixels=int(fit.sum()), gain=gains, bias=biases)


def build(out):
    started = time.monotonic(); out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(__file__, out/'source.py')
    protocol = dict(version=1, scenes=SCENES, anchors=ANCHORS, offsets_original_frames=OFFSETS,
        calibration_anchors=[16, 48, 80], check_anchors=[32, 64, 104], patch_radius_lr=8,
        r='inbounds margin10; FB<0.5 and Farneback-DIS<0.5; erode11x11 for D/U support; exp(-0.5*(FB/.25)^2-0.5*(DIS/.35)^2); no photometric rejection',
        p='exp(-(7x7 LR mean absolute exposure-corrected RGB error/.03)^2); unknown exposure => p=0',
        exposure='3-iteration Huber affine RGB on reliable LR support, all scored patches+margin excluded; >=512 pixels; gain[.8,1.25],bias[-.1,.1]',
        targets='A=single; C=A+.5*(r-weighted Hneighbor-H(A)); D uses r*p and eta=.5*mean_p; >=2 valid neighbors; D mean_p>=.25; G=C global-matched borrowing amplitude using only LR eta maps',
        sampling='4 grid,3 texture,3 motion,2 correspondence-uncertain; LR-only; radius8 =>64 HR patch',
        high_operator='H=x-U(D_raw(x)), linear bicubic-AA down and bicubic up, neither clipped; observed LR comparison clips D_raw; H is not an exact nullspace projection',
        hr_use='separate evaluate command only after ALL targets locked',
        claims='image correspondence proxies, not geometric/visibility ground truth; same-camera temporal teacher diagnostic only',
        gate='both scenes check anchors: D improves A and C/G in at least 2 of mean PSNR,SSIM,LPIPS (all reported); >=4/6 observations improve PSNR vs C; effective D coverage >=5%; no detail-energy collapse; directional screen, not significance or independent generalization; metric tradeoffs require review',
        cpu_threads=4, source_sha256=sha(__file__), started_utc=datetime.now(timezone.utc).isoformat())
    write(out/'protocol.json', protocol)
    write(out/'compatibility_checks.json', checks())
    inputs, observations = {}, []
    for scene, (sub, cams) in SCENES.items():
        data = ROOT/'data/dynamic_sr'/sub
        mp = data/'manifest.json'; m = json.loads(mp.read_text()); inputs[str(mp)] = sha(mp)
        obs = {(o['camera_id'], o['frame_index']): o for o in m['observations']}
        for cam in cams:
            assert cam in m['splits']['train']
            for anchor in ANCHORS:
                tick = time.monotonic(); lrs, teachers = {}, {}
                for frame in [anchor] + [anchor+v for v in OFFSETS]:
                    o = obs[(cam, frame)]
                    lp = data/o['lr_path']; sp = data/'sr_swinir_x4'/cam/Path(o['lr_path']).name
                    assert sha(lp) == o['lr_sha256']
                    for path in [lp, sp]: inputs[str(path)] = sha(path)
                    lrs[frame], teachers[frame] = rgb(lp), rgb(sp)
                states, eigen = flow_states(lrs, anchor)
                selected = patches(states, eigen)
                base = teachers[anchor]; shape = base.shape[:2]; lrsize = lrs[anchor].shape[:2]
                assert shape == (lrsize[0]*4, lrsize[1]*4)
                bh = high(base, lrsize); highs, raw_highs, w_r, w_p, pair_meta = [], [], [], [], []
                for s in states:
                    gain, bias, exp_info = exposure_fit(lrs[anchor], s['warped_lr'], s['valid'], selected)
                    error = np.mean(np.abs(s['warped_lr']*gain+bias-lrs[anchor]), axis=2)
                    p = np.exp(-(cv2.boxFilter(error, -1, (7, 7), normalize=True)/.03)**2).astype(np.float32)
                    if not exp_info['known']:
                        p.fill(0)
                        # Both C and D exclude unknown exposure, avoiding a
                        # hidden difference in their correspondence sets.
                        s['r'].fill(0)
                    s['exposure_known'] = exp_info['known']
                    s['p'] = p
                    flow_hr = lift(s['flow'], shape)*4
                    aligned = warp(teachers[s['frame']], flow_hr)
                    raw_highs.append(high(aligned, lrsize))
                    highs.append(high(aligned*gain+bias, lrsize))
                    w_r.append(lift(s['r'], shape, True)); w_p.append(lift(p, shape, True))
                    pair_meta.append(dict(frame=s['frame'], exposure=exp_info,
                                          normalized_dt=obs[(cam, s['frame'])]['time']-obs[(cam, anchor)]['time']))
                hs, wr, ps = np.stack(highs), np.stack(w_r), np.stack(w_p)
                support = np.stack([lift(s['valid'].astype(np.float32), shape, True)*s['exposure_known'] for s in states]).sum(0)
                mpmap = (wr*ps).sum(0)/np.maximum(wr.sum(0), 1e-12)
                eta_c = .5*(support >= 2).astype(np.float32)
                eta_d = eta_c*mpmap*(mpmap >= .25)
                C = combine(base, hs, wr, eta_c)
                D = combine(base, hs, wr*ps, eta_d)
                ratio = float(eta_d.mean()/max(eta_c.mean(), 1e-12))
                G = base + ratio*(C-base)
                pair_crops, raw_crops, cells = [], [], []
                for idx, s in enumerate(states):
                    validhr = lift(s['valid'].astype(np.float32), shape, True)
                    candidate = base + .5*validhr[..., None]*(hs[idx]-bh)
                    raw_candidate = base + .5*validhr[..., None]*(raw_highs[idx]-bh)
                    for pi, p in enumerate(selected):
                        x, y = p['x'], p['y']; sl = np.s_[(y-8)*4:(y+8)*4, (x-8)*4:(x+8)*4]
                        slr = np.s_[y-8:y+8, x-8:x+8]
                        den = float(s['r'][slr].sum())
                        cells.append(dict(patch=pi, neighbor=idx, offset=s['frame']-anchor, exposure_known=s['exposure_known'], p=float((s['r'][slr]*s['p'][slr]).sum()/max(den, 1e-12)),
                                          reliable_fraction=float(s['valid'][slr].mean()), mean_r=float(s['r'][slr].mean())))
                        pair_crops.append(candidate[sl]); raw_crops.append(raw_candidate[sl])
                stem = f'{scene}_{cam}_{anchor:04d}'
                file = out/f'{stem}.npz'
                np.savez_compressed(file, C=C, D=D, G=G, pair_crops=np.stack(pair_crops), raw_pair_crops=np.stack(raw_crops),
                    flow=np.stack([s['flow'] for s in states]), valid=np.stack([s['valid'] for s in states]),
                    r=np.stack([s['r'] for s in states]), p=np.stack([s['p'] for s in states]),
                    eta_c_lr=eta_c[::4, ::4], eta_d_lr=eta_d[::4, ::4])
                row = dict(scene=scene, sub=sub, camera=cam, anchor=anchor, split='calibration' if anchor in [16,48,80] else 'check',
                    data_file=file.name, sha256=sha(file), patches=selected, pairs=cells, neighbors=pair_meta,
                    C_coverage=float((eta_c>0).mean()), D_coverage=float((eta_d>0).mean()),
                    mean_eta_C=float(eta_c.mean()), mean_eta_D=float(eta_d.mean()), G_ratio=ratio,
                    elapsed_seconds=time.monotonic()-tick)
                observations.append(row)
                print(stem, 'locked', round(row['elapsed_seconds'], 2), flush=True)
    write(out/'inputs.json', inputs)
    write(out/'observations.json', observations)
    write(out/'targets_locked.json', dict(source_sha256=sha(__file__), protocol_sha256=sha(out/'protocol.json'),
        inputs_sha256=sha(out/'inputs.json'), observations_sha256=sha(out/'observations.json'),
        observations=len(observations), patches=sum(len(o['patches']) for o in observations),
        pairs=sum(len(o['pairs']) for o in observations), elapsed_seconds=time.monotonic()-started,
        host=socket.gethostname(), python=platform.python_version(), torch=torch.__version__, cv2=cv2.__version__,
        finished_utc=datetime.now(timezone.utc).isoformat(), no_hr_images_read=True, no_gpu=True))


def measure(pred, gt, lr, lp):
    pc = np.clip(pred, 0, 1); mse = float(np.mean((pc-gt)**2))
    blur = lambda a: cv2.GaussianBlur(a.astype(np.float64), (11, 11), 1.5)
    mx, my = blur(pc), blur(gt)
    vx, vy, cov = blur(pc**2)-mx**2, blur(gt**2)-my**2, blur(pc*gt)-mx*my
    smap = ((2*mx*my+.01**2)*(2*cov+.03**2))/((mx**2+my**2+.01**2)*(vx+vy+.03**2))
    with torch.inference_mode(): perceptual = float(lp(tensor(pc)*2-1, tensor(gt)*2-1))
    ph, gh = high(pc, lr.shape[:2]), high(gt, lr.shape[:2])
    edge = lambda a: np.stack([cv2.Sobel(a, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(a, cv2.CV_32F, 0, 1, ksize=3)])
    return dict(mse=mse, raw_mse=float(np.mean((pred-gt)**2)), psnr=float(-10*np.log10(max(mse, 1e-20))),
        ssim=float(smap[5:-5, 5:-5].mean()), lpips=perceptual,
        high_mse=float(np.mean((ph-gh)**2)), high_energy_ratio=float(np.mean(ph**2)/max(np.mean(gh**2), 1e-15)),
        edge_mse=float(np.mean((edge(pc)-edge(gt))**2)), lr_mse=float(np.mean((down(pc, lr.shape[:2])-lr)**2)),
        clipped_fraction=float(np.mean((pred<0)|(pred>1))))


def evaluate(out):
    import lpips
    from scipy.stats import spearmanr
    start = time.monotonic(); lock = json.loads((out/'targets_locked.json').read_text())
    assert lock['source_sha256'] == sha(__file__)
    for name in ['protocol', 'inputs', 'observations']:
        assert sha(out/f'{name}.json') == lock[f'{name}_sha256']
    assert not (out/'evaluation_complete.json').exists()
    for path, digest in json.loads((out/'inputs.json').read_text()).items(): assert sha(path) == digest
    rows = json.loads((out/'observations.json').read_text())
    # Verify every locked target BEFORE evaluation, no adaptive build between rows.
    for row in rows: assert sha(out/row['data_file']) == row['sha256']
    lp = lpips.LPIPS(net='alex').eval().cpu()
    results, pairs, hr_inputs, plots = [], [], {}, []
    for row in rows:
        data = ROOT/'data/dynamic_sr'/row['sub']; m = json.loads((data/'manifest.json').read_text())
        ob = next(o for o in m['observations'] if o['camera_id']==row['camera'] and o['frame_index']==row['anchor'])
        hp = data/ob['hr_path']; assert sha(hp) == ob['hr_sha256']; hr_inputs[str(hp)] = sha(hp)
        gt, lr = rgb(hp), rgb(data/ob['lr_path'])
        base = rgb(data/'sr_swinir_x4'/row['camera']/Path(ob['lr_path']).name)
        with np.load(out/row['data_file']) as arrays:
            predictions = {'A': base, **{k: arrays[k] for k in ['C','D','G']}}
            measures = {k: measure(x, gt, lr, lp) for k, x in predictions.items()}
            for k, x in predictions.items():
                measures[k]['lr_change_mse'] = float(np.mean((down(x, lr.shape[:2])-down(base, lr.shape[:2]))**2))
                measures[k]['lr_change_mse_clipped'] = float(np.mean((down(np.clip(x,0,1), lr.shape[:2])-down(base, lr.shape[:2]))**2))
            result = {k:row[k] for k in ['scene','camera','anchor','split','C_coverage','D_coverage','mean_eta_C','mean_eta_D','G_ratio']}
            result['metrics'] = measures
            result['patches'] = []
            for pi, p in enumerate(row['patches']):
                x, y = p['x'], p['y']; sl=np.s_[(y-8)*4:(y+8)*4, (x-8)*4:(x+8)*4]
                pm = {k:float(np.mean((np.clip(im[sl],0,1)-gt[sl])**2)) for k,im in predictions.items()}
                result['patches'].append({**p, 'mse':pm})
            for idx, pair in enumerate(row['pairs']):
                p = row['patches'][pair['patch']]; x,y=p['x'],p['y']; sl=np.s_[(y-8)*4:(y+8)*4,(x-8)*4:(x+8)*4]
                candidate = arrays['pair_crops'][idx]; rawcandidate = arrays['raw_pair_crops'][idx]
                err0 = float(np.mean((base[sl]-gt[sl])**2))
                err1 = float(np.mean((np.clip(candidate,0,1)-gt[sl])**2))
                replacement = 2*candidate-base[sl]
                pairs.append(dict(**{k:row[k] for k in ['scene','camera','anchor','split']}, **pair,
                    selection=p['selection'], motion=p['motion'], texture=p['texture'], baseline_mse=err0,
                    candidate_mse=err1, gain=err0-err1,
                    replacement_gain=err0-float(np.mean((np.clip(replacement,0,1)-gt[sl])**2)),
                    uncorrected_gain=err0-float(np.mean((np.clip(rawcandidate,0,1)-gt[sl])**2))))
            # The example is chosen by LR motion only, before any HR metric.
            pi = max(range(len(row['patches'])), key=lambda i:row['patches'][i]['motion'])
            p=row['patches'][pi]; x,y=p['x'],p['y']; sl=np.s_[(y-8)*4:(y+8)*4,(x-8)*4:(x+8)*4]
            panel=Image.new('RGB',(5*192,218),'white'); draw=ImageDraw.Draw(panel)
            for i,(k,im) in enumerate([('HR',gt),*predictions.items()]):
                tile=Image.fromarray(np.round(np.clip(im[sl],0,1)*255).astype(np.uint8)).resize((192,192),Image.Resampling.NEAREST)
                panel.paste(tile,(192*i,26));draw.text((192*i+5,6),k,fill='black')
            panel.save(out/f"{row['scene']}_{row['camera']}_{row['anchor']:04d}_panel.png")
        results.append(result)
        print(row['scene'],row['camera'],row['anchor'],'evaluated',flush=True)
    summary={}
    for scene in SCENES:
        summary[scene]={}
        for split in ['calibration','check']:
            rr=[r for r in results if r['scene']==scene and r['split']==split]
            pp=[p for p in pairs if p['scene']==scene and p['split']==split and p['reliable_fraction']>=.75 and p['exposure_known']]
            means={k:{metric:float(np.mean([r['metrics'][k][metric] for r in rr])) for metric in rr[0]['metrics'][k]} for k in ['A','C','D','G']}
            corr=spearmanr([p['p'] for p in pp],[p['gain'] for p in pp]).statistic if len(pp)>2 else float('nan')
            bins=[]
            for lo,hi in [(0,.25),(.25,.5),(.5,.75),(.75,1.00001)]:
                bb=[p for p in pp if lo<=p['p']<hi]
                bins.append(dict(lo=lo,hi=min(hi,1),n=len(bb),mean_gain=float(np.mean([p['gain'] for p in bb])) if bb else None,
                                 beneficial_fraction=float(np.mean([p['gain']>0 for p in bb])) if bb else None))
            summary[scene][split]=dict(n=len(rr),means=means,
                D_better_C_psnr=sum(r['metrics']['D']['psnr']>r['metrics']['C']['psnr'] for r in rr),
                D_better_G_psnr=sum(r['metrics']['D']['psnr']>r['metrics']['G']['psnr'] for r in rr),
                C_coverage=float(np.mean([r['C_coverage'] for r in rr])),D_coverage=float(np.mean([r['D_coverage'] for r in rr])),
                reliable_pairs=len(pp),p_gain_spearman=float(corr) if np.isfinite(corr) else None,p_bins=bins,
                beneficial_pair_fraction=float(np.mean([p['gain']>0 for p in pp])) if pp else None)
    gates={}
    for scene in SCENES:
        s=summary[scene]['check']; mm=s['means']
        metric_signs={f'D_beats_{k}_{metric}':mm['D'][metric]>mm[k][metric] if metric!='lpips' else mm['D'][metric]<mm[k][metric]
                    for k in ['A','C','G'] for metric in ['psnr','ssim','lpips']}
        conditions={f'D_beats_{k}_at_least_two_metrics':sum(metric_signs[f'D_beats_{k}_{metric}'] for metric in ['psnr','ssim','lpips'])>=2 for k in ['A','C','G']}
        conditions.update(coverage=s['D_coverage']>=.05, most_observations=s['D_better_C_psnr']>=4,
                          no_energy_collapse=mm['D']['high_energy_ratio']>=.9*mm['A']['high_energy_ratio'])
        gates[scene]=dict(conditions=conditions,metric_signs=metric_signs,passed=all(conditions.values()))
    write(out/'evaluation_rows.json',results);write(out/'pair_evaluation.json',pairs);write(out/'hr_evaluation_inputs.json',hr_inputs)
    write(out/'summary.json',summary)
    write(out/'decision.json',dict(gates=gates,proceed_p2=all(g['passed'] for g in gates.values()),
        meaning='pre-registered directional development screen; failure stops this prototype, not all temporal sharing or semantic methods'))
    write(out/'evaluation_complete.json',dict(elapsed_seconds=time.monotonic()-start,source_sha256=sha(__file__),
        targets_lock_sha256=sha(out/'targets_locked.json'),summary_sha256=sha(out/'summary.json'),
        finished_utc=datetime.now(timezone.utc).isoformat(),gpu_used=False,parameter_updates=0))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['build','evaluate','check'])
    parser.add_argument('--out',type=Path,default=ROOT/'output/dynamic_sr_20260923/temporal_sharing_v1')
    args=parser.parse_args();torch.set_num_threads(4);cv2.setNumThreads(2)
    if args.mode=='build':build(args.out)
    elif args.mode=='evaluate':evaluate(args.out)
    else:print(json.dumps(checks(),indent=2))
