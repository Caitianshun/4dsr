"""HR-teacher upper bound of the final recipe (Claude, session 4dsr-a4, 2026-11-02).

User 11-02: the main table may carry an HR upper bound. The same-backbone HR-input row is beaten by Ours in several cells, so the
upper bound is the full final recipe (three w4 SR sources -> PVCD student -> P+C on LR -> post-hoc cull) with every SwinIR-areaft
teacher image replaced by the HR image of the same training observation. Nothing else changes: same LR parent, Bsync
registration, schedule, seeds, modkit4 / pvcd_modkit / srfix / cullretrain code and arguments. The swap is
refs.patch_teacher_source, the mechanism of the 10-16 oracle arm (each HR file SHA256-checked against the manifest). Reads HR
training pixels: a diagnostic upper bound, never a method row.
  hrteacher.py modkit4 <modkit4.py arguments, unchanged>
  hrteacher.py pvcd    <pvcd_modkit.py arguments, unchanged>
hrteacher.json in the SR --out records the entry, argv, the number of redirected teachers, the number of teacher reads served
with HR during the run and the source hashes.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MOD = HERE.parent
ROOT = MOD.parents[1]
GEO = MOD / 'geoprobe_20261017'
THIRD = MOD / 'third_20261022'
for _p in (MOD, GEO, THIRD):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import refs  # noqa: E402  (imports modsr -> cullretrain; no side effects besides the runtime paths)


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def main():
    if len(sys.argv) < 3 or sys.argv[1] not in ('modkit4', 'pvcd'):
        raise SystemExit(__doc__)
    entry, rest = sys.argv[1], sys.argv[2:]
    if '--mode' not in rest or 'sr' not in rest:
        raise SystemExit('expected the srfix part: ... -- --mode fix cullretrain sr ...')
    sr = rest[rest.index('--mode'):]
    baseline = sr[sr.index('--baseline-sr') + 1]
    out = Path(sr[sr.index('--out') + 1])
    out = out if out.is_absolute() else ROOT / out
    if (out / 'complete.json').exists():
        raise FileExistsError('Completed run preserved: ' + str(out))
    redirected = refs.patch_teacher_source(baseline)
    served = refs.cr.read_teacher
    reads = [0]

    def read_teacher(path):
        reads[0] += 1
        return served(path)

    refs.cr.read_teacher = read_teacher
    if entry == 'modkit4':
        import modkit4 as module
        sys.argv = [str(GEO / 'modkit4.py')] + rest
    else:
        import pvcd_modkit as module
        sys.argv = [str(THIRD / 'pvcd_modkit.py')] + rest
    module.main()
    if refs.cr.read_teacher is not read_teacher:
        raise SystemExit('the teacher reader was replaced during the run')
    if reads[0] == 0:
        raise SystemExit('no teacher image was read')
    files = [Path(__file__), Path(refs.__file__), Path(refs.cr.__file__), Path(module.__file__)]
    record = dict(kind='hr_teacher_upper_bound', entry=entry, argv=rest, baseline_sr=baseline, redirected_teachers=redirected,
                  teacher_reads_served_with_HR=reads[0], HR_training_pixels_read=True,
                  sources={str(f.resolve().relative_to(ROOT)): sha(f) for f in files})
    (out / 'hrteacher.json').write_text(json.dumps(record, indent=1) + '\n')
    print(json.dumps(dict(hrteacher=entry, redirected=redirected, reads=reads[0])), flush=True)


if __name__ == '__main__':
    main()
