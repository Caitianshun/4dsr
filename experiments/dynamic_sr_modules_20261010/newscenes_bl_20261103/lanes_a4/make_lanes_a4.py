"""Derived lane files for the new-scene baselines (Claude 4dsr-a4, 2026-11-03). System python3 -I, stdlib only.

Source: ../lanes_exec/<lane>.sh of Codex task dsr-newscene-baselines-sol-max-20261103 (one command per line, each with its
exact interpreter). Changes, nothing else:
  * A100 (every line runs in the remote project .venv): the Codex sv2_lanes.py lane ../lanes/<lane>.txt is kept (including
    its CRITICAL marks); every line is checked against the .sh command (interpreter = remote .venv). --torch-home is set to
    output/dynamic_sr_modules_20261010/bl4_20261024/torch_home, the LPIPS cache that scored the main-table A100 baselines
    (the new model_cache directory of the .sh files does not exist on the A100).
  * CTS: every line keeps its interpreter as an '@<interpreter>' prefix for lanes_interp.py (CEM steps in the CEM leaf env,
    scoring/cleanup/collect in the .venv, as the historical CTS CEM chain). The score lines drop --torch-home: the
    historical CTS CEM scoring used the default TORCH_HOME of the remote .venv.
  * The .sh environment exports are not applied (no historical baseline lane set them; TORCH_HOME is handled above).
Each written line is checked against its .sh line: same interpreter, script and arguments apart from the listed edits.
  make_lanes_a4.py            (writes lanes_a4/<lane>.txt and lanes_a4/derivation.json)"""
import hashlib, json, shlex
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / 'lanes_exec'
ROOT = HERE.parents[2]
NEW_TH = 'output/dynamic_sr_modules_20261010/newscenes_bl_20261103/model_cache/torch_home'
A100_TH = 'output/dynamic_sr_modules_20261010/bl4_20261024/torch_home'
VENV = {'a100': '/home/ubuntu/3DGS/4dsr/.venv/bin/python', 'cts': '/home/cts/Project/4DSR/.venv/bin/python'}
LANES = [f'a100_{s}_{m}' for s in ('sear_steak', 'flame_salmon') for m in ('sr4d', 'splatsure', 'iesrgs')] + \
        [f'cts_{s}_cem' for s in ('sear_steak', 'flame_salmon')]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def commands(sh):
    out = []
    for line in sh.read_text().splitlines():
        if not line.strip() or line.startswith(('#', 'set ', 'cd ', 'export ')):
            continue
        parts = shlex.split(line)
        assert parts[1] == '-u', line
        out.append((parts[0], parts[2:]))
    return out


def main():
    record = []
    for lane in LANES:
        host = lane.split('_')[0]
        sh = SRC / f'{lane}.sh'
        lines, edits = [], []
        for interp, args in commands(sh):
            args = list(args)
            if '--torch-home' in args:
                i = args.index('--torch-home')
                assert args[i + 1] == NEW_TH, (lane, args)
                if host == 'a100':
                    args[i + 1] = A100_TH
                    edits.append(f'{args[0]}: --torch-home {A100_TH}')
                else:
                    del args[i:i + 2]
                    edits.append(f'{args[0]}: --torch-home removed')
            if host == 'a100':
                assert interp == VENV['a100'], (lane, interp)
                lines.append(args)
            else:
                assert interp in (VENV['cts'], '/home/cts/Project/4DSR/envs/dynamic_lr_baselines_20261020/cem/bin/python'), interp
                lines.append('@' + interp + ' ' + shlex.join(args))
        if host == 'a100':
            txt = HERE.parent / 'lanes' / f'{lane}.txt'
            src = [l for l in txt.read_text().splitlines() if l.strip() and not l.startswith('#')]
            assert len(src) == len(lines), lane
            out_lines = []
            for l, args in zip(src, lines):
                crit = l.startswith('CRITICAL ')
                orig = shlex.split(l[9:] if crit else l)
                assert [a if a != NEW_TH else A100_TH for a in orig] == args, (lane, l)
                out_lines.append(('CRITICAL ' if crit else '') + shlex.join(args))
            lines = out_lines
        head = f'# Claude 4dsr-a4, 2026-11-03: derived from ../lanes_exec/{lane}.sh (sha256 {sha(sh)[:16]}) by make_lanes_a4.py'
        out = HERE / f'{lane}.txt'
        out.write_text('\n'.join([head, *lines]) + '\n')
        record.append(dict(lane=lane, host=host, source=str(sh.relative_to(ROOT)), source_sha256=sha(sh),
                           output=str(out.relative_to(ROOT)), output_sha256=sha(out), commands=len(lines), edits=edits))
    (HERE / 'derivation.json').write_text(json.dumps(dict(script=str(Path(__file__).relative_to(ROOT)),
                                                          script_sha256=sha(__file__), lanes=record), indent=1) + '\n')
    for r in record:
        print(r['lane'], r['commands'], len(r['edits']))


if __name__ == '__main__':
    main()
