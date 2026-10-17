"""Remote driver for the injected-misregistration dose test (Claude, session 4dsr-e9, 2026-10-17): perturbsr.py LR fine stages
(parents) and SR stages (arms), each finished SR arm scored at once with group_eval_bands.py (cam00, all 300 frames, post-hoc
cull, TF32 off, plus FFT band PSNR). Steps run one after another in command-line order (one training process at a time).
A step whose complete.json exists is skipped; a failing step does not stop the others (arms of a failed parent fail).

  run_perturb_remote.py --tag T --out-root DIR --lr-registry DIR --coarse CK --reg-zero DIR --reg-matched DIR
                        --hr-width 1344 --teacher-zeros 1008 1344 [--wait SUP=MARKER ...] [--wait-timeout 14400]
                        --parent NAME=SIGMA[:CALIB[:WARP_LR]] ... --arm NAME=PARENT:SUP[:CALIB[:WARP_LR]] ...

Parents: perturbsr ... -- lr --parent LR_REGISTRY --fine0 COARSE --arm <T>_<NAME> --prune-gate none --affine --out ...
         (the 4dsr-7c area4 gate recipe of the LR fine stage, i.e. Cook07_area4_gateaff, plus the perturbation).
Arms (all: area operator, srfix --mode fix, --prune-gate none, --affine, D-SSIM 0.2, 6000 steps, sigma of the parent):
  L0  lambda_SR 0, zero teachers (REG_ZERO)           -- the areaL0 recipe
  Tm  lambda_SR 0.2, area-matched SwinIR (REG_MATCHED) -- the areaTm recipe
  Or  lambda_SR 0.2, HR teacher via refs --teacher-source hr (REG_ZERO) -- the oracle recipe
--wait SUP=MARKER: before the first arm of that supervision, wait (60 s sleeps, up to --wait-timeout) until MARKER exists --
the marker is uploaded after that supervision's inputs (teacher images / training HR) have finished uploading.
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
PERTURB = HERE / 'perturbsr.py'
EVAL = HERE / 'group_eval_bands.py'


def run(cmd):
    t = time.time()
    r = subprocess.run([sys.executable, '-u'] + [str(c) for c in cmd], cwd=ROOT)
    return r.returncode, round(time.time() - t, 1)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--tag', required=True)
    p.add_argument('--out-root', required=True)
    p.add_argument('--lr-registry', required=True)
    p.add_argument('--coarse', required=True)
    p.add_argument('--reg-zero', required=True)
    p.add_argument('--reg-matched', required=True)
    p.add_argument('--hr-width', required=True)
    p.add_argument('--teacher-zeros', nargs=2, required=True)
    p.add_argument('--perturb-seed', default='20261017')
    p.add_argument('--view', default='cam00')
    p.add_argument('--wait', action='append', default=[])
    p.add_argument('--wait-timeout', type=float, default=14400)
    p.add_argument('--parent', dest='steps', action='append', type=lambda s: ('parent', s))
    p.add_argument('--arm', dest='steps', action='append', type=lambda s: ('arm', s))
    a = p.parse_args()
    root = ROOT / a.out_root
    root.mkdir(parents=True, exist_ok=True)
    waits = dict(w.split('=', 1) for w in a.wait)
    parents, report = {}, []
    for kind, spec in a.steps:
        name, rest = spec.split('=', 1)
        label = f'{a.tag}_{name}'
        if kind == 'parent':
            f = rest.split(':')
            sigma, calib, wlr = f[0], (f[1] if len(f) > 1 else 'none'), (f[2] if len(f) > 2 else '2e-3')
            out = root / 'lr' / label
            ck = out / 'checkpoint_fine_14000.pt'
            parents[name] = dict(sigma=sigma, ck=ck)
            if (out / 'complete.json').exists():
                report.append(dict(step=name, skipped=True))
                continue
            rc, sec = run([PERTURB, '--perturb-sigma', sigma, '--perturb-seed', a.perturb_seed, '--hr-width', a.hr_width,
                           '--calib', calib, '--warp-lr', wlr, '--', 'lr', '--parent', a.lr_registry, '--fine0', a.coarse,
                           '--arm', label, '--prune-gate', 'none', '--affine', '--out', out.relative_to(ROOT)])
            ok = rc == 0 and ck.exists()
            report.append(dict(step=name, kind='parent', sigma=sigma, calib=calib, rc=rc, seconds=sec, ok=ok))
            print(json.dumps(report[-1]), flush=True)
            continue
        f = rest.split(':')
        parent, sup = f[0], f[1]
        calib, wlr = (f[2] if len(f) > 2 else 'none'), (f[3] if len(f) > 3 else '2e-3')
        out = root / 'sr' / label
        ev = root / 'eval' / f'{label}_{a.view}'
        ck = out / 'checkpoint_06000.pt'
        if parent not in parents or not parents[parent]['ck'].exists():
            report.append(dict(step=name, kind='arm', error=f'parent {parent} missing'))
            print(json.dumps(report[-1]), flush=True)
            continue
        if sup in waits:
            marker, t0 = ROOT / waits.pop(sup), time.time()
            while not marker.exists() and time.time() - t0 < a.wait_timeout:
                time.sleep(60)
            print(json.dumps(dict(wait=sup, marker=str(marker.relative_to(ROOT)), present=marker.exists(),
                                  waited=round(time.time() - t0))), flush=True)
            if not marker.exists():
                waits[sup] = str(marker.relative_to(ROOT))
                report.append(dict(step=name, kind='arm', error=f'{sup} inputs not ready'))
                continue
        if not (out / 'complete.json').exists():
            own = [PERTURB, '--perturb-sigma', parents[parent]['sigma'], '--perturb-seed', a.perturb_seed, '--hr-width', a.hr_width,
                   '--lr-operator', 'area', '--calib', calib, '--warp-lr', wlr]
            core = ['sr', '--parent-checkpoint', parents[parent]['ck'].relative_to(ROOT), '--arm', label, '--prune-gate', 'none',
                    '--affine', '--dssim', '0.2', '--out', out.relative_to(ROOT)]
            if sup == 'L0':
                cmd = own + ['--teacher-zeros', *a.teacher_zeros, '--', '--mode', 'fix', 'cullretrain', *core,
                             '--baseline-sr', a.reg_zero, '--lambda-sr', '0']
            elif sup == 'Tm':
                cmd = own + ['--', '--mode', 'fix', 'cullretrain', *core, '--baseline-sr', a.reg_matched, '--lambda-sr', '0.2']
            elif sup == 'Or':
                cmd = own + ['--', '--mode', 'fix', 'refs', '--teacher-source', 'hr', *core, '--baseline-sr', a.reg_zero,
                             '--lambda-sr', '0.2']
            else:
                raise SystemExit(f'unknown supervision {sup}')
            rc, sec = run(cmd)
            print(json.dumps(dict(step=name, kind='arm', rc=rc, seconds=sec)), flush=True)
        if not ck.exists():
            report.append(dict(step=name, kind='arm', error='no checkpoint'))
            continue
        if not (ev / 'summary.json').exists():
            rc, _ = run([EVAL, '--checkpoint', f'{name}={ck.relative_to(ROOT)}', '--view', a.view, '--out', ev.relative_to(ROOT)])
        if (ev / 'summary.json').exists():
            s = json.loads((ev / 'summary.json').read_text())['summary'][name]
            report.append(dict(step=name, kind='arm', parent=parent, sup=sup, calib=calib,
                               **{k: round(s[k], 4) for k in ('psnr', 'ssim', 'lpips', 'low_psnr', 'mid_psnr', 'high_psnr')}))
        else:
            report.append(dict(step=name, kind='arm', error='eval failed'))
        print(json.dumps(report[-1]), flush=True)
    (root / f'{a.tag}_report.json').write_text(json.dumps(report, indent=1))
    print(json.dumps(dict(done=[r['step'] for r in report if 'error' not in r and r.get('ok', True)],
                          failed=[r['step'] for r in report if 'error' in r or r.get('ok') is False])), flush=True)
    sys.exit(1 if any('error' in r or r.get('ok') is False for r in report) else 0)


if __name__ == '__main__':
    main()
