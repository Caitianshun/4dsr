"""SR-stage module kit, part 6: stop densification before the end of the SR stage (Claude, session 4dsr-e9, 2026-10-17). New file;
modkit5.py (hash pinned below) and everything it imports are imported, never modified.

  modkit6.py [--densify-stop K] <modkit5 / modkit4 / modkit2 / modkit options> -- --mode fix {cullretrain|csc} sr <SR args>

Motivation: in the registered SR stage (6000 steps) densification, pruning and growth run every 100 steps up to the end
(opt.densify_until_iter exceeds the SR step count; srfix --mode fix only skips the final step). Gaussians added at steps 5800 / 5900
get at most 200 optimisation steps before the checkpoint is saved. One Cut07 run (toffLR parent + calib, 10-17) diverged right after
the densifications at steps 5800 and 5900 (anchor LR L1 x2-4, cam00 PSNR 25.6), and 10-16 found that most of the 2-model ensemble
gain is SR-stage chaos. Letting the last part of the stage refine a fixed set of Gaussians may lower both.

--densify-stop K : fine-stage topology (densification statistics, densify, prune, grow, opacity reset) is skipped for SR steps
                   >= K, exactly as the author code does for iterations >= densify_until_iter; earlier steps are unchanged (the
                   annealed thresholds still use densify_until_iter). Implemented by wrapping srfix._original_topology, the function
                   srfix's own final-step guard calls, so both guards compose.
modkit6.json in --out records the switch, the number of skipped topology calls, the first skipped step and source hashes.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MOD = HERE.parent
ROOT = MOD.parents[1]
for _p in (MOD, HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import modkit5  # noqa: E402

PINNED = {'modkit5.py': '6d7108e73adf085a'}


def make_stop(original, k, stats):
    def topology(g, opt, stage, iteration, *args, **kwargs):
        if stage == 'fine' and iteration >= k:
            stats['skipped'] += 1
            if stats['first'] is None:
                stats['first'] = int(iteration)
            return None
        return original(g, opt, stage, iteration, *args, **kwargs)
    return topology


def main():
    argv = sys.argv[1:]
    if '--' not in argv:
        raise SystemExit(__doc__)
    sha = modkit5.modkit4.modkit2.modkit.calibsr_v2.sha
    for name, digest in PINNED.items():
        if not sha(HERE / name).startswith(digest):
            raise SystemExit(f'{name} changed')
    k = argv.index('--')
    own, rest = argv[:k], argv[k + 1:]
    s = dict(densify_stop=0)
    passthrough = []
    i = 0
    while i < len(own):
        if own[i] == '--densify-stop':
            s['densify_stop'] = int(own[i + 1]); i += 2
        else:
            passthrough.append(own[i]); i += 1
    if s['densify_stop'] < 0:
        raise SystemExit('switch out of range')
    if len(rest) < 4 or rest[0] != '--mode' or rest[3] != 'sr':
        raise SystemExit('expected -- --mode fix {cullretrain|csc} sr ...')
    sr_args = rest[3:]
    out = Path(sr_args[sr_args.index('--out') + 1])
    out = out if out.is_absolute() else ROOT / out
    import srfix  # noqa: E402
    stats = dict(skipped=0, first=None)
    real_main = srfix.main

    def stop_main():
        """innermost wrapper: srfix's final-step guard calls srfix._original_topology at call time"""
        if s['densify_stop']:
            srfix._original_topology = make_stop(srfix._original_topology, s['densify_stop'], stats)
        return real_main()

    srfix.main = stop_main
    sys.argv = [str(HERE / 'modkit5.py')] + passthrough + ['--'] + rest
    modkit5.main()
    record = dict(switches=s, passthrough=passthrough, argv=rest, topology_skipped=stats['skipped'], first_skipped_step=stats['first'],
                  sources={str(Path(f).resolve().relative_to(ROOT)): sha(f) for f in (__file__, modkit5.__file__, srfix.__file__)})
    (out / 'modkit6.json').write_text(json.dumps(record, indent=1))
    print(json.dumps(dict(modkit6=s, skipped=stats['skipped'], first=stats['first']), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
