"""SR-stage module kit, part 8: hold one training camera out of the SR stage (Claude, session 4dsr-e9, 2026-10-18). New file;
modkit6.py (hash pinned below) and everything it imports are imported, never modified.

  modkit8.py [--holdout-camera CAM] <modkit6 / modkit5 / modkit4 / modkit2 / modkit options> -- --mode fix cullretrain sr <SR args>

User 10-18: the goal is per-scene optimisation; different parameters per scene are allowed. A per-scene choice made on the test
view cam00 is a test-set selection. This switch gives a selection signal that needs no HR data: camera CAM takes no part in the
SR stage, so its teacher image (a function of its own LR only) and its LR can score an SR-stage configuration on a view the SR
stage has not seen (vsel_eval.py). CAM is fixed by camera poses only: the training camera nearest the test view (cam06 for the
N3DV scenes, cam03 for MeetRoom). The LR parent is unchanged (it was trained on the LR of every training camera, CAM included).

--holdout-camera CAM : in the registered schedule (rows of three same-time training views: a = LR closure, b / c = teacher
                       views) every occurrence of CAM is replaced by another training camera at the same frame, drawn uniformly
                       from the training cameras not in the row (random.Random(20261018 + row index)); the number of steps, the
                       frames and the other views of each row are unchanged. CAM's per-camera parameters (colour affine, calib,
                       fields) never receive a gradient and keep their initial values. cullretrain entry only.
modkit8.json in --out records the switch, the replaced slots per view position, the number of schedule reads and source hashes.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MOD = HERE.parent
ROOT = MOD.parents[1]
for _p in (MOD, HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import modkit6  # noqa: E402

PINNED = {'modkit6.py': '6eab5d7124643c5b'}
SEED = 20261018


def hold_rows(table, cam, seed=SEED):
    """rows of the registered schedule with every slot of `cam` replaced (see the module docstring); returns rows, per-position counts"""
    keys = [(c, int(f)) for c, f in table['record_keys']]
    index = {k: i for i, k in enumerate(keys)}
    cams = sorted({c for c, _ in keys})
    if cam not in cams:
        raise SystemExit(f'{cam} is not a training camera of the schedule')
    rows, counts = [], [0, 0, 0]
    for r, row in enumerate(table['rows']):
        row_cams = [keys[i][0] for i in row]
        if cam not in row_cams:
            rows.append(list(row))
            continue
        frame = keys[row[0]][1]
        rng = random.Random(seed + r)
        pool = [c for c in cams if c != cam and c not in row_cams]
        new = list(row)
        for j, c in enumerate(row_cams):
            if c == cam:
                sub = rng.choice(pool)
                pool.remove(sub)
                new[j] = index[(sub, frame)]
                counts[j] += 1
        rows.append(new)
    return rows, counts


def install(cr, cam, stats):
    """wrap cullretrain.read: the registered schedule (the JSON with rows and record_keys) comes back with `cam` held out"""
    original = cr.read

    def read(path):
        value = original(path)
        if isinstance(value, dict) and 'rows' in value and 'record_keys' in value:
            rows, counts = hold_rows(value, cam)
            value = dict(value, rows=rows)
            stats['reads'] += 1
            stats['replaced_per_position'] = counts
            stats['schedule'] = str(path)
        return value
    cr.read = read


def main():
    argv = sys.argv[1:]
    if '--' not in argv:
        raise SystemExit(__doc__)
    sha = modkit6.modkit5.modkit4.modkit2.modkit.calibsr_v2.sha
    for name, digest in PINNED.items():
        if not sha(HERE / name).startswith(digest):
            raise SystemExit(f'{name} changed')
    k = argv.index('--')
    own, rest = argv[:k], argv[k + 1:]
    s = dict(holdout_camera=None)
    passthrough = []
    i = 0
    while i < len(own):
        if own[i] == '--holdout-camera':
            s['holdout_camera'] = own[i + 1]; i += 2
        else:
            passthrough.append(own[i]); i += 1
    if len(rest) < 4 or rest[0] != '--mode' or rest[3] != 'sr':
        raise SystemExit('expected -- --mode fix {cullretrain|csc} sr ...')
    if s['holdout_camera'] and rest[2] != 'cullretrain':
        raise SystemExit('--holdout-camera needs the cullretrain entry')
    sr_args = rest[3:]
    out = Path(sr_args[sr_args.index('--out') + 1])
    out = out if out.is_absolute() else ROOT / out
    import srfix  # noqa: E402
    stats = dict(reads=0, replaced_per_position=None, schedule=None)
    real_main = srfix.main

    def hold_main():
        if s['holdout_camera']:
            install(srfix.cr, s['holdout_camera'], stats)
        return real_main()

    srfix.main = hold_main
    sys.argv = [str(HERE / 'modkit6.py')] + passthrough + ['--'] + rest
    modkit6.main()
    error = 'the schedule was never read' if s['holdout_camera'] and not stats['reads'] else None
    record = dict(switches=s, passthrough=passthrough, argv=rest, stats=stats, error=error,
                  sources={str(Path(f).resolve().relative_to(ROOT)): sha(f) for f in (__file__, modkit6.__file__, srfix.__file__)})
    (out / 'modkit8.json').write_text(json.dumps(record, indent=1))
    print(json.dumps(dict(modkit8=s, stats=stats, error=error), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
