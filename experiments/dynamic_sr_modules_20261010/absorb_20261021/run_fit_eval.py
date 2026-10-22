"""run_fit_eval.py: perframe_fit.py, then (only if it exits 0) 4dsr-25's eval_resid.py, in one remote job (Claude 4dsr-f3, 2026-10-21).
usage: run_fit_eval.py <perframe_fit.py args> --eval <eval_resid.py args>
Both children use this interpreter and inherit the environment (remote_run_fp32.py sets NVIDIA_TF32_OVERRIDE=0); exit code = the
first non-zero child exit code."""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
argv = sys.argv[1:]
if '--eval' not in argv:
    raise SystemExit('usage: run_fit_eval.py <fit args> --eval <eval args>')
cut = argv.index('--eval')
r = subprocess.run([sys.executable, '-u', str(HERE / 'perframe_fit.py')] + argv[:cut], cwd=ROOT)
print(f'[run_fit_eval] fit exit {r.returncode}', flush=True)
if r.returncode:
    sys.exit(r.returncode)
r = subprocess.run([sys.executable, '-u', str(ROOT / 'experiments/dynamic_sr_modules_20261010/perframe_20261021/eval_resid.py')]
                   + argv[cut + 1:], cwd=ROOT)
print(f'[run_fit_eval] eval exit {r.returncode}', flush=True)
sys.exit(r.returncode)
