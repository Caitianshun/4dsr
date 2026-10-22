"""run_scenes2.py: run absorb_bound2.py on the given scenes (default Cook07 then Discussion07) with the same interpreter (Claude 4dsr-f3, 2026-10-21).
usage: run_scenes2.py [SCENE ...] [-- extra absorb_bound2.py args]
Each scene writes its own output directory; a failure on one scene does not stop the next; exit code = number of failures."""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SR = 'output/dynamic_sr_modules_20261010/geoprobe_20261017/mk_local/sr'
OUT = 'output/dynamic_sr_modules_20261010/absorb_20261021/v2'
argv = sys.argv[1:]
cut = argv.index('--') if '--' in argv else len(argv)
scenes, extra = (argv[:cut] or ['Cook07', 'Discussion07']), argv[cut + 1:]
fails = 0
for scene in scenes:
    cmd = [sys.executable, '-u', str(HERE / 'absorb_bound2.py'), '--checkpoint', f'{SR}/{scene}_area4_ch3w_w4/checkpoint_06000.pt',
           '--out', f'{OUT}/{scene}'] + extra
    print('[run_both]', ' '.join(cmd[2:]), flush=True)
    r = subprocess.run(cmd, cwd=ROOT)
    print(f'[run_both] {scene} exit {r.returncode}', flush=True)
    fails += r.returncode != 0
sys.exit(fails)
