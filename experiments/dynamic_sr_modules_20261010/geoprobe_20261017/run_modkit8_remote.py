"""Remote driver v8 (Claude, session 4dsr-e9, 2026-10-18 09:30): run_modkit7_remote.py (byte-frozen) with modkit8.py as the training
entry and the key hold (--holdout-camera CAM: CAM takes no part in the SR stage). For an arm with hold, after the usual cam00
evaluation vsel_eval.py scores CAM against its teacher / LR (/ HR when present, diagnostic) into vsel/<label>_<CAM>/summary.json and
the row gets the v_* fields (selection signal: v_teacher_apsnr) and hold_reads / hold_replaced from modkit8.json; an arm whose
schedule was never modified is marked as an error. Arms without hold train exactly as under v7.
v7 docstring: run_modkit6_remote.py (byte-frozen) plus the key op (LR operator of the
SR stage, bicaa or area; default area as before), so that the naive row of an ablation chain (bicaa operator + off-the-shelf
teacher) runs through the same entry as the other rows. v6 docstring: run_modkit5_remote.py with modkit6.py as the training
entry (key dstop = --densify-stop K added; modkit6 passes everything else to modkit5; the row gets dstop_skipped from modkit6.json).
v5 docstring: run_modkit4_remote.py with modkit5.py as the training
entry (keys dprior = --depth-prior W, dpatch = --depth-patch P added; modkit5 passes everything else to modkit4; the row gets
dprior_loss_first / dprior_loss_last / dprior_xyz_grad / dprior_error from modkit5.json). v4 docstring: (keys psf, psflr added; modkit4 passes everything else to modkit2 / modkit unchanged). v3 docstring: eval_full.py as the scorer, so
every arm also gets the colour-gauge-aligned PSNR (apsnr, psnr_odd) and dynamic / static region PSNR (dyn_psnr, static_psnr,
adyn_psnr, astatic_psnr) next to the unchanged standard metrics. Original docstring: each arm is trained, scored at once with
group_eval_bands.py (cam00, all 300 frames, post-hoc cull, TF32 off, FFT bands) and written to results/<label>.json, so
results stream in arm by arm. Arms run one after another (one training process at a time); a trained arm is not retrained,
a scored arm is not rescored, a failing arm does not stop the others.

  run_modkit_remote.py --parent CK --baseline-sr DIR [--lambda-sr 0.2] [--teacher-zeros H W] --out-root DIR --tag T
                       [--view cam00] --arm NAME=[KEY:VALUE[,KEY:VALUE...]] ...
modkit keys : calib, warp (warp lr), densify, xyz, deform, lowpass, drop, field (grid), fieldlr, probe (every K steps)
modkit2 keys: toff (--time-offset lr), wfield (--warp-field grid), wflr (--warp-field-lr); modkit2.py passes the modkit keys on,
              so an arm without modkit2 keys trains exactly as under run_modkit_remote.py
SR keys     : lambda (--lambda-sr), geo (--geo-weight), geosrc (--geo-source), geohp (--geo-highpass-sigma),
              geotol (--geo-depth-tol); any geo* key selects the csc entry (CSC-G), otherwise cullretrain.
Every arm: area operator, srfix --mode fix, --prune-gate none, --affine, D-SSIM 0.2, 6000 steps (the 4dsr-7c area4 recipe;
areaTm when the registry is the area-matched teacher). An empty spec (NAME=) is the control.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
KIT = HERE / 'modkit8.py'
VSEL = HERE / 'vsel_eval.py'
EVAL = HERE / 'eval_full.py'
OWN = dict(calib='--calib', warp='--warp-lr', densify='--densify-scale', xyz='--xyz-lr-scale', deform='--deform-lr-scale',
           lowpass='--teacher-lowpass', drop='--drop-rate', field='--cam-field', fieldlr='--cam-field-lr', probe='--grad-probe',
           toff='--time-offset', wfield='--warp-field', wflr='--warp-field-lr', psf='--psf', psflr='--psf-lr',
           dprior='--depth-prior', dpatch='--depth-patch', dstop='--densify-stop', hold='--holdout-camera')
SR = dict(lambda_='--lambda-sr', geo='--geo-weight', geosrc='--geo-source', geohp='--geo-highpass-sigma', geotol='--geo-depth-tol')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--parent', required=True)
    p.add_argument('--baseline-sr', required=True)
    p.add_argument('--lambda-sr', default='0.2')
    p.add_argument('--teacher-zeros', nargs=2)
    p.add_argument('--out-root', required=True)
    p.add_argument('--tag', required=True)
    p.add_argument('--view', default='cam00')
    p.add_argument('--arm', action='append', required=True)
    a = p.parse_args()
    root = ROOT / a.out_root
    (root / 'results').mkdir(parents=True, exist_ok=True)
    report = []
    for spec in a.arm:
        name, rest = spec.split('=', 1)
        sw = dict(kv.split(':', 1) for kv in rest.split(',') if kv)
        sw = {('lambda_' if k == 'lambda' else k): v for k, v in sw.items()}
        if set(sw) - set(OWN) - set(SR) - {'op'}:
            raise SystemExit(f'unknown switch in {spec}')
        if sw.get('op', 'area') not in ('area', 'bicaa'):
            raise SystemExit(f'op must be area or bicaa in {spec}')
        lam = sw.get('lambda_', a.lambda_sr)
        entry = 'csc' if any(k.startswith('geo') for k in sw) else 'cullretrain'
        label = f'{a.tag}_{name}'
        out = root / 'sr' / label
        ev = root / 'eval' / f'{label}_{a.view}'
        ck = out / 'checkpoint_06000.pt'
        res_file = root / 'results' / f'{label}.json'
        if not (out / 'complete.json').exists():
            cmd = [sys.executable, '-u', str(KIT), '--lr-operator', sw.get('op', 'area')]
            if a.teacher_zeros and float(lam) == 0:
                cmd += ['--teacher-zeros', *a.teacher_zeros]
            for k, v in sw.items():
                if k in OWN:
                    cmd += [OWN[k], v]
            cmd += ['--', '--mode', 'fix', entry, 'sr', '--parent-checkpoint', a.parent, '--baseline-sr', a.baseline_sr,
                    '--arm', label, '--prune-gate', 'none', '--affine', '--lambda-sr', lam, '--dssim', '0.2',
                    '--out', str(out.relative_to(ROOT))]
            for k, v in sw.items():
                if k in SR and k != 'lambda_':
                    cmd += [SR[k], v]
            t = time.time()
            r = subprocess.run(cmd, cwd=ROOT)
            print(json.dumps(dict(arm=name, rc=r.returncode, seconds=round(time.time() - t, 1))), flush=True)
        row = dict(arm=name, label=label, tag=a.tag, switches={('lambda' if k == 'lambda_' else k): v for k, v in sw.items()},
                   entry=entry, lambda_sr=lam, parent=a.parent, baseline_sr=a.baseline_sr)
        if not ck.exists():
            row['error'] = 'no checkpoint'
        else:
            if not (ev / 'summary.json').exists():
                subprocess.run([sys.executable, '-u', str(EVAL), '--checkpoint', f'{name}={ck.relative_to(ROOT)}', '--view', a.view,
                                '--out', str(ev.relative_to(ROOT))], cwd=ROOT)
            if (ev / 'summary.json').exists():
                s = json.loads((ev / 'summary.json').read_text())
                m = s['summary'][name]
                c = json.loads((out / 'complete.json').read_text())
                row.update(points=c.get('points'), train_seconds=round(c.get('seconds', 0)), host=s.get('host'), gpu=s.get('gpu'),
                           **{k: m[k] for k in ('psnr', 'ssim', 'lpips', 'low_psnr', 'mid_psnr', 'high_psnr', 'psnr_odd', 'apsnr',
                                                'dyn_psnr', 'static_psnr', 'adyn_psnr', 'astatic_psnr', 'dyn_share') if k in m})
                kit = out / 'modkit.json'
                if kit.exists():
                    k = json.loads(kit.read_text())
                    row.update(drop=k.get('drop'), probe=k.get('probe'))
                kit4 = out / 'modkit4.json'
                if kit4.exists():
                    k = json.loads(kit4.read_text())
                    sg = sorted(v for v in (k.get('psf_sigma2_px2') or {}).values() if v is not None)
                    row.update(psf_sigma2_min=sg[0] if sg else None, psf_sigma2_max=sg[-1] if sg else None)
                kit8 = out / 'modkit8.json'
                if sw.get('hold'):
                    k = json.loads(kit8.read_text()) if kit8.exists() else {}
                    st = k.get('stats') or {}
                    row.update(hold=sw['hold'], hold_reads=st.get('reads'), hold_replaced=st.get('replaced_per_position'))
                    vdir = root / 'vsel' / f"{label}_{sw['hold']}"
                    if not (vdir / 'summary.json').exists():
                        subprocess.run([sys.executable, '-u', str(VSEL), '--checkpoint', f'{name}={ck.relative_to(ROOT)}', '--camera',
                                        sw['hold'], '--baseline-sr', a.baseline_sr, '--out', str(vdir.relative_to(ROOT))], cwd=ROOT)
                    if (vdir / 'summary.json').exists():
                        v = json.loads((vdir / 'summary.json').read_text())['models'][name]
                        row.update({f'v_{q}': x for q, x in v.items() if isinstance(x, (int, float))})
                    if not st.get('reads') or 'v_teacher_apsnr' not in row:
                        row['error'] = 'holdout not applied' if not st.get('reads') else 'vsel failed'
                kit6 = out / 'modkit6.json'
                if kit6.exists():
                    row.update(dstop_skipped=json.loads(kit6.read_text()).get('topology_skipped'))
                kit5 = out / 'modkit5.json'
                if kit5.exists():
                    k = json.loads(kit5.read_text())
                    st = k.get('stats') or {}
                    row.update(dprior_loss_first=(k.get('first_1000_calls') or {}).get('loss'),
                               dprior_loss_last=(k.get('last_1000_calls') or {}).get('loss'),
                               dprior_xyz_grad=st.get('first_call_xyz_grad_norm'), dprior_error=st.get('error'))
                kit2 = out / 'modkit2.json'
                if kit2.exists():
                    k = json.loads(kit2.read_text())
                    to = [v for v in (k.get('time_offsets_frames') or {}).values() if v is not None]
                    wf = [v for v in (k.get('warp_field_rms_px') or {}).values() if v is not None]
                    row.update(time_offset_abs_max=max(map(abs, to)) if to else None, time_offset_rms=(sum(v * v for v in to) / len(to)) ** 0.5 if to else None,
                               warp_field_rms_px_median=sorted(wf)[len(wf) // 2] if wf else None)
                probe = out / 'grad_probe.jsonl'
                if probe.exists():
                    rows = [json.loads(x) for x in probe.read_text().splitlines() if x.strip()]
                    row['grad_probe'] = {key: {q: sum(r[key][q] for r in rows if key in r) / max(1, sum(key in r for r in rows))
                                               for q in ('cos', 'neg_frac', 'neg_sr_norm_share')}
                                         for key in ('xyz', 'f_dc', 'opacity', 'scaling')}
                    row['grad_probe_rows'] = len(rows)
            else:
                row['error'] = 'eval failed'
        res_file.write_text(json.dumps(row, indent=1))
        report.append(row)
        print(json.dumps({k: v for k, v in row.items() if k not in ('parent', 'baseline_sr')}), flush=True)
    (root / f'{a.tag}_report.json').write_text(json.dumps(report, indent=1))
    failed = [r['arm'] for r in report if 'error' in r]
    print(json.dumps(dict(done=[r['arm'] for r in report if 'error' not in r], failed=failed)), flush=True)
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
