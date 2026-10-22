"""Diagnostic test-view phase check (Claude 4dsr-46, 2026-10-22; suggested by 4dsr-e3; pre-registered in
tmp/claude_perscene_design_20261021/prereg_motion.md Addendum H before any result).

eval_full.py (geoprobe_20261017, imported, never modified) with the evaluated view's render time shifted by --shift-frames x dt,
dt = the view's median frame spacing in manifest time; everything else (post-hoc cull, TF32 off, metrics, regions) is eval_full's.
On 10-17 the camres_probe2 time scan put Cook07's cam00 at -0.35 frame from the training-camera consensus
(docs/sr_modules_area4_2026-10-17.md): a model whose motion is sharper at the consensus time is penalised at cam00 by that phase.
This uses the test view's own phase, so it is an analysis of the evaluation, never a method or a selection rule.
Writes <out>/phase_shift.json next to eval_full's outputs.
Usage: phase_eval.py --shift-frames K <eval_full.py arguments>"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
GEO = HERE.parent / 'geoprobe_20261017'
sys.path.insert(0, str(GEO))


def main():
    i = sys.argv.index('--shift-frames')
    k = float(sys.argv[i + 1])
    del sys.argv[i:i + 2]
    import numpy as np
    import eval_full
    orig = eval_full.ens_eval.view_observations
    used = {}

    def shifted(manifest, view, label):
        obs = orig(manifest, view, label)
        dt = float(np.median(np.diff(np.array([float(o['time']) for o in obs]))))
        used[label] = dict(view=view, dt=dt, shift_time=k * dt)
        return [dict(o, time=float(o['time']) + k * dt) for o in obs]

    eval_full.ens_eval.view_observations = shifted
    eval_full.main()
    out = eval_full.ROOT / sys.argv[sys.argv.index('--out') + 1]
    (out / 'phase_shift.json').write_text(json.dumps(dict(
        shift_frames=k, per_model=used, doc=__doc__,
        sources_code={eval_full.cr.rel(q): eval_full.rt.sha(q) for q in (Path(__file__), GEO / 'eval_full.py')}), indent=1))


if __name__ == '__main__':
    main()
