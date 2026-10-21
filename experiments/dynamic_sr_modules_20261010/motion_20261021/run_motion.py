"""Arm runner for the motion module (Claude, session 4dsr-25, 2026-10-21): the w4 main recipe's SR stage (area operator, matched
teacher, per-camera 4x4 warp field, colour affine, srfix --mode fix, prune gate none, D-SSIM 0.2, 6000 steps) with modkit13.py as
the training entry, then eval_full.py on the test view (all 300 frames, post-hoc cull, TF32 off). Arms run one after another
(one training process at a time); a trained arm is not retrained, a scored arm is not rescored, a failing arm does not stop the
others. An ABORT file in --out-root stops the runner before the next arm. Results go to <out-root>/results/<tag>_<arm>.json.

  run_motion.py --parent CK --baseline-sr DIR --out-root DIR --tag T [--view cam00] [--lambda-sr 0.2]
                --arm NAME=[KEY:VALUE[,KEY:VALUE...]] ...
keys: fw (--flow-weight), fdir (--flow-dir), fD (--flow-D), fviews (--flow-views, '+'-separated), ffb (--flow-fb),
      fdead (--flow-dead), fstart (--flow-start), fmodel (--flow-model), aw (--app-weight), ash (--app-sh-weight),
      tf (--teacher-tf), lambda (--lambda-sr)
An empty spec (NAME=) is the registered w4 recipe (control).
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
KIT = HERE / 'modkit13.py'
EVAL = ROOT / 'experiments/dynamic_sr_modules_20261010/geoprobe_20261017/eval_full.py'
OWN = dict(fw='--flow-weight', fdir='--flow-dir', fD='--flow-D', fviews='--flow-views', ffb='--flow-fb', fdead='--flow-dead',
           fstart='--flow-start', fmodel='--flow-model', aw='--app-weight', ash='--app-sh-weight', tf='--teacher-tf')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--parent', required=True)
    p.add_argument('--baseline-sr', required=True)
    p.add_argument('--lambda-sr', default='0.2')
    p.add_argument('--out-root', required=True)
    p.add_argument('--tag', required=True)
    p.add_argument('--view', default='cam00')
    p.add_argument('--arm', action='append', required=True)
    a = p.parse_args()
    root = ROOT / a.out_root
    (root / 'results').mkdir(parents=True, exist_ok=True)
    report = []
    for spec in a.arm:
        if (root / 'ABORT').exists():
            print(json.dumps(dict(aborted_before=spec)), flush=True)
            break
        name, rest = spec.split('=', 1)
        sw = dict(kv.split(':', 1) for kv in rest.split(',') if kv)
        if set(sw) - set(OWN) - {'lambda'}:
            raise SystemExit(f'unknown switch in {spec}')
        lam = sw.get('lambda', a.lambda_sr)
        label = f'{a.tag}_{name}'
        out = root / 'sr' / label
        ev = root / 'eval' / f'{label}_{a.view}'
        ck = out / 'checkpoint_06000.pt'
        res_file = root / 'results' / f'{label}.json'
        if not (out / 'complete.json').exists():
            cmd = [sys.executable, '-u', str(KIT)]
            for k, v in sw.items():
                if k in OWN:
                    cmd += [OWN[k], v.replace('+', ',') if k == 'fviews' else v]
            cmd += ['--lr-operator', 'area', '--warp-field', '4', '--', '--mode', 'fix', 'cullretrain', 'sr',
                    '--parent-checkpoint', a.parent, '--baseline-sr', a.baseline_sr, '--arm', label, '--prune-gate', 'none',
                    '--affine', '--lambda-sr', lam, '--dssim', '0.2', '--out', str(out.relative_to(ROOT))]
            t = time.time()
            r = subprocess.run(cmd, cwd=ROOT)
            print(json.dumps(dict(arm=name, rc=r.returncode, seconds=round(time.time() - t, 1))), flush=True)
        row = dict(arm=name, label=label, tag=a.tag, switches=sw, lambda_sr=lam, parent=a.parent, baseline_sr=a.baseline_sr)
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
                kit = out / 'modkit13.json'
                if kit.exists():
                    k = json.loads(kit.read_text())
                    row.update(motion_stats=k.get('stats'), motion_terms=k.get('terms'))
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
