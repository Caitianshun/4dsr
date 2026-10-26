"""PVCD training-view ablation (Claude 4dsr-33, 2026-10-26; pre-registration tmp/claude_pvcd_k45_20261026/prereg_k45.md).

Runs pvcd_modkit.py (CTS / local) or its A100 copy pvcd_modkit_a.py unchanged, except for the pose of the consistency term:
the camera is the training camera i itself instead of the pseudo-view between i and its neighbour j.
The sampler draws i, j, alpha and t from the same numpy stream and in the same order as consistency_distill.PseudoViews
(alpha is drawn and discarded), so with the same --seed the student sees exactly the same (i, t) sequence as the matched
pseudo-view run; only the pose differs (alpha = 0: rotation and centre of camera i, intrinsics of i as before).
Loss, weight, D-SSIM share, scale, sources, stop-gradient, native rendering and the densification exclusion are untouched.
The kit's pvcd.json keeps its constant alpha field [0.25, 0.75]; the pose actually used is recorded in pvcd.jsonl ('last')
and in pvcd_tv.json written here.

usage: pvcd_tv.py {pvcd_modkit|pvcd_modkit_a} <pvcd_modkit.py arguments, unchanged>
"""
from __future__ import annotations

import hashlib
import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def main():
    if len(sys.argv) < 3 or sys.argv[1] not in ('pvcd_modkit', 'pvcd_modkit_a'):
        raise SystemExit(__doc__)
    name, rest = sys.argv[1], sys.argv[2:]
    if '--dry-run' in rest:
        raise SystemExit('no dry run here; use tmp/claude_pvcd_k45_20261026/test_tv_sampler.py')
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    kit = importlib.import_module(name)
    cd = kit.cd
    base = cd.PseudoViews
    stats = dict(samples=0, cameras={})

    class TrainingViews(base):
        def sample(self, ev, native, torch):
            i = self.cams[int(self.rng.integers(len(self.cams)))]
            j = self.neighbours[i][int(self.rng.integers(2))]
            self.rng.uniform(0.25, 0.75)  # drawn and discarded: same random stream as PseudoViews
            t = self.times[int(self.rng.integers(len(self.times)))]
            R = np.array(self.w2c[i][:3, :3], dtype=np.float64)
            c = np.array(self.center[i], dtype=np.float64)
            w2c = np.eye(4)
            w2c[:3, :3], w2c[:3, 3] = R, -R @ c
            obs = ev.camera_item(self.manifest, self.first[i], torch)  # camera i's intrinsics and LR size (payload only)
            obs.update(w2c=w2c, time=t)
            camera = native.hr_camera(obs, 0, self.manifest)
            stats['samples'] += 1
            stats['cameras'][i] = stats['cameras'].get(i, 0) + 1
            return camera, t, dict(i=i, j=j, alpha=0.0, time=t, center=c.tolist(), mode='training_view')

    cd.PseudoViews = TrainingViews
    kit_argv = rest
    cut = kit_argv.index('--')
    sr_args = kit_argv[cut + 1:]
    sr_args = sr_args[sr_args.index('--') + 1:]
    out = Path(sr_args[sr_args.index('--out') + 1])
    out = out if out.is_absolute() else ROOT / out
    started = time.time()
    sys.argv = [str(HERE / (name + '.py'))] + kit_argv
    try:
        kit.main()
    finally:
        cd.PseudoViews = base
        rec = dict(kind='pvcd_training_view_ablation', kit=name, kit_sha256=hashlib.sha256((HERE / (name + '.py')).read_bytes()).hexdigest(),
                   wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), pose='training camera i (alpha = 0)',
                   samples=stats['samples'], samples_per_camera=stats['cameras'], seconds=time.time() - started,
                   argv=kit_argv, complete=(out / 'complete.json').exists())
        if out.exists():
            (out / 'pvcd_tv.json').write_text(json.dumps(rec, indent=1))


if __name__ == '__main__':
    main()
