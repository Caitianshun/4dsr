"""Remote driver for geosr.py arms (Claude, session 4dsr-e9, 2026-10-17): each arm is trained, then scored at once with
group_eval_bands.py (cam00, all 300 frames, post-hoc cull, TF32 off, FFT band PSNR), so results stream in arm by arm.
Arms run one after another (one training process at a time); an arm whose complete.json exists is not retrained, a scored
arm is not rescored, a failing arm does not stop the others.

  run_geo_remote.py --parent CK --baseline-sr DIR [--lambda-sr 0.2] [--teacher-zeros H W] [--wait MARKER] [--wait-timeout S]
                    --out-root DIR --tag T [--view cam00] --arm NAME=[KEY:VALUE[,KEY:VALUE...]] ...
KEY: calib (none|shift|full), warp (warp lr), densify, xyz, deform (scales), lowpass (teacher low-pass sigma, HR px).
An empty spec (NAME=) is the control. Every arm: area operator, srfix --mode fix, cullretrain sr, --prune-gate none,
--affine, D-SSIM 0.2, 6000 steps (the 4dsr-7c area4 gate recipe; areaTm when the registry is the area-matched teacher).
--wait MARKER: before the first arm, wait (60 s sleeps) until MARKER exists (uploaded after the teacher images).
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
GEOSR = HERE / 'geosr.py'
EVAL = HERE / 'group_eval_bands.py'
FLAG = dict(calib='--calib', warp='--warp-lr', densify='--densify-scale', xyz='--xyz-lr-scale', deform='--deform-lr-scale',
            lowpass='--teacher-lowpass')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--parent', required=True)
    p.add_argument('--baseline-sr', required=True)
    p.add_argument('--lambda-sr', default='0.2')
    p.add_argument('--teacher-zeros', nargs=2)
    p.add_argument('--wait')
    p.add_argument('--wait-timeout', type=float, default=10800)
    p.add_argument('--out-root', required=True)
    p.add_argument('--tag', required=True)
    p.add_argument('--view', default='cam00')
    p.add_argument('--arm', action='append', required=True)
    a = p.parse_args()
    root = ROOT / a.out_root
    root.mkdir(parents=True, exist_ok=True)
    if a.teacher_zeros and float(a.lambda_sr) != 0:
        raise SystemExit('--teacher-zeros requires --lambda-sr 0')
    if a.wait:
        marker, t0 = ROOT / a.wait, time.time()
        while not marker.exists() and time.time() - t0 < a.wait_timeout:
            time.sleep(60)
        print(json.dumps(dict(wait=a.wait, present=marker.exists(), waited=round(time.time() - t0))), flush=True)
        if not marker.exists():
            raise SystemExit('teacher inputs not ready')
    report = []
    for spec in a.arm:
        name, rest = spec.split('=', 1)
        switches = dict(kv.split(':', 1) for kv in rest.split(',') if kv)
        if set(switches) - set(FLAG):
            raise SystemExit(f'unknown switch in {spec}')
        label = f'{a.tag}_{name}'
        out = root / 'sr' / label
        ev = root / 'eval' / f'{label}_{a.view}'
        ck = out / 'checkpoint_06000.pt'
        if not (out / 'complete.json').exists():
            cmd = [sys.executable, '-u', str(GEOSR), '--lr-operator', 'area']
            if a.teacher_zeros:
                cmd += ['--teacher-zeros', *a.teacher_zeros]
            for k, v in switches.items():
                cmd += [FLAG[k], v]
            cmd += ['--', '--mode', 'fix', 'cullretrain', 'sr', '--parent-checkpoint', a.parent, '--baseline-sr', a.baseline_sr,
                    '--arm', label, '--prune-gate', 'none', '--affine', '--lambda-sr', a.lambda_sr, '--dssim', '0.2',
                    '--out', str(out.relative_to(ROOT))]
            t = time.time()
            r = subprocess.run(cmd, cwd=ROOT)
            print(json.dumps(dict(arm=name, rc=r.returncode, seconds=round(time.time() - t, 1))), flush=True)
        if not ck.exists():
            report.append(dict(arm=name, switches=switches, error='no checkpoint'))
            print(json.dumps(report[-1]), flush=True)
            continue
        if not (ev / 'summary.json').exists():
            subprocess.run([sys.executable, '-u', str(EVAL), '--checkpoint', f'{name}={ck.relative_to(ROOT)}', '--view', a.view,
                            '--out', str(ev.relative_to(ROOT))], cwd=ROOT)
        if (ev / 'summary.json').exists():
            s = json.loads((ev / 'summary.json').read_text())['summary'][name]
            c = json.loads((out / 'complete.json').read_text())
            report.append(dict(arm=name, switches=switches, points=c.get('points'), seconds=round(c.get('seconds', 0)),
                               **{k: round(s[k], 4) for k in ('psnr', 'ssim', 'lpips', 'low_psnr', 'mid_psnr', 'high_psnr')}))
        else:
            report.append(dict(arm=name, switches=switches, error='eval failed'))
        print(json.dumps(report[-1]), flush=True)
    (root / f'{a.tag}_report.json').write_text(json.dumps(report, indent=1))
    failed = [r['arm'] for r in report if 'error' in r]
    print(json.dumps(dict(done=[r['arm'] for r in report if 'error' not in r], failed=failed)), flush=True)
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
