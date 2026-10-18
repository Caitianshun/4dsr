"""Remote driver for modkit.py arms (Claude, session 4dsr-e9, 2026-10-17): each arm is trained, scored at once with
group_eval_bands.py (cam00, all 300 frames, post-hoc cull, TF32 off, FFT bands) and written to results/<label>.json, so
results stream in arm by arm. Arms run one after another (one training process at a time); a trained arm is not retrained,
a scored arm is not rescored, a failing arm does not stop the others.

  run_modkit_remote.py --parent CK --baseline-sr DIR [--lambda-sr 0.2] [--teacher-zeros H W] --out-root DIR --tag T
                       [--view cam00] --arm NAME=[KEY:VALUE[,KEY:VALUE...]] ...
modkit keys : calib, warp (warp lr), densify, xyz, deform, lowpass, drop, field (grid), fieldlr, probe (every K steps)
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
KIT = HERE / 'modkit.py'
EVAL = HERE / 'group_eval_bands.py'
OWN = dict(calib='--calib', warp='--warp-lr', densify='--densify-scale', xyz='--xyz-lr-scale', deform='--deform-lr-scale',
           lowpass='--teacher-lowpass', drop='--drop-rate', field='--cam-field', fieldlr='--cam-field-lr', probe='--grad-probe')
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
        if set(sw) - set(OWN) - set(SR):
            raise SystemExit(f'unknown switch in {spec}')
        lam = sw.get('lambda_', a.lambda_sr)
        entry = 'csc' if any(k.startswith('geo') for k in sw) else 'cullretrain'
        label = f'{a.tag}_{name}'
        out = root / 'sr' / label
        ev = root / 'eval' / f'{label}_{a.view}'
        ck = out / 'checkpoint_06000.pt'
        res_file = root / 'results' / f'{label}.json'
        if not (out / 'complete.json').exists():
            cmd = [sys.executable, '-u', str(KIT), '--lr-operator', 'area']
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
                           **{k: m[k] for k in ('psnr', 'ssim', 'lpips', 'low_psnr', 'mid_psnr', 'high_psnr') if k in m})
                kit = out / 'modkit.json'
                if kit.exists():
                    k = json.loads(kit.read_text())
                    row.update(drop=k.get('drop'), probe=k.get('probe'))
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
