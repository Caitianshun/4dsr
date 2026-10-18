"""SR-stage module kit, part 10: gauge-corrected held-out camera (Claude, session 4dsr-e9, 2026-10-18). New file; modkit9.py (hash
pinned below) and everything it imports are imported, never modified.

  modkit10.py [--holdout-gauge] <modkit9 / modkit8 (--holdout-camera CAM) / ... options> -- --mode fix cullretrain sr <SR args>

prereg_vsel.txt amendment 1 (10-18 09:46): with modkit8 --holdout-camera CAM, CAM stays in the camera list, so its rows of every
per-camera parameter (colour affine 'delta', calib 'warp', warp field 'wfield', time offset 'toff') receive gradients through the
re-centring mean and drift freely; the trained cameras then share a common correction -d_CAM / (N - 1). The Cook07 held-out runs of
10-18 drifted far (CAM radial term 5.4 / 3.4 px vs 0.14 / -0.04 in the all-camera runs).
--holdout-gauge : after construction, after load_state_dict and after every optimiser step of the camera model, CAM's rows of those
                  tensors are set to the mean of the other cameras' rows, so the applied corrections of the trained cameras are
                  mean-zero among themselves and CAM's own applied correction is the identity (as for the test view). Per-camera
                  PSF ('psf_d') is not supported (the switch then stops). Needs --holdout-camera.
modkit10.json in --out records the switch, the camera, the number of resets and CAM's largest re-centred row value at the end.
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

PINNED = {'modkit9.py': '96b0d847c2f5f68b'}
NAMES = ('delta', 'warp', 'wfield', 'toff')


def make_gauge(base, cam, stats):
    import torch

    class GaugeAffine(base):
        def _reset_cam(self):
            if getattr(self, 'psf_d', None) is not None:
                raise SystemExit('--holdout-gauge does not support the per-camera PSF')
            v = self.index[cam]
            with torch.no_grad():
                for name in NAMES:
                    t = getattr(self, name, None)
                    if not isinstance(t, torch.Tensor) or t.shape[0] != len(self.names):
                        continue
                    keep = torch.ones(t.shape[0], dtype=torch.bool, device=t.device)
                    keep[v] = False
                    t[v] = t[keep].mean(0)
            stats['resets'] += 1

        def __init__(self, cameras, *args, **kwargs):
            super().__init__(cameras, *args, **kwargs)
            if cam not in self.index:
                raise SystemExit(f'{cam} is not in the camera list')
            self._reset_cam()
            stats['instances'] += 1
            stats['last'] = self

        def load_state_dict(self, value):
            super().load_state_dict(value)
            self._reset_cam()

        def step(self, clock):
            super().step(clock)
            self._reset_cam()

    return GaugeAffine


def max_recentred(aff, cam):
    """largest |row(CAM) - mean over cameras| over the tensors (0 when the gauge is held exactly)"""
    import torch
    v = aff.index[cam]
    out = {}
    for name in NAMES:
        t = getattr(aff, name, None)
        if isinstance(t, torch.Tensor) and t.shape[0] == len(aff.names):
            out[name] = float((t[v] - t.mean(0)).abs().max())
    return out


def main():
    argv = sys.argv[1:]
    if '--' not in argv:
        raise SystemExit(__doc__)
    import modkit9  # noqa: E402
    import modkit8  # noqa: E402
    sha = modkit8.modkit6.modkit5.modkit4.modkit2.modkit.calibsr_v2.sha
    for name, digest in PINNED.items():
        if not sha(HERE / name).startswith(digest):
            raise SystemExit(f'{name} changed')
    k = argv.index('--')
    own, rest = argv[:k], argv[k + 1:]
    s = dict(holdout_gauge=False, camera=None)
    passthrough = []
    for x in own:
        if x == '--holdout-gauge':
            s['holdout_gauge'] = True
        else:
            passthrough.append(x)
    if '--holdout-camera' in passthrough:
        s['camera'] = passthrough[passthrough.index('--holdout-camera') + 1]
    if s['holdout_gauge'] and not s['camera']:
        raise SystemExit('--holdout-gauge needs --holdout-camera')
    sr_args = rest[3:]
    out = Path(sr_args[sr_args.index('--out') + 1])
    out = out if out.is_absolute() else ROOT / out
    import srfix  # noqa: E402
    stats = dict(resets=0, instances=0, last=None)
    real_main = srfix.main

    def gauge_main():
        """innermost wrapper (installed before modkit9 / modkit8 / ... install theirs): cr.CameraAffine is final here"""
        if s['holdout_gauge']:
            srfix.cr.CameraAffine = make_gauge(srfix.cr.CameraAffine, s['camera'], stats)
        return real_main()

    srfix.main = gauge_main
    sys.argv = [str(HERE / 'modkit9.py')] + passthrough + ['--'] + rest
    modkit9.main()
    last = stats.pop('last')
    error = None
    if s['holdout_gauge'] and (not stats['instances'] or not stats['resets']):
        error = 'gauge requested but never applied'
    record = dict(switches=s, passthrough=passthrough, argv=rest, stats=stats, error=error,
                  cam_recentred_max=max_recentred(last, s['camera']) if (last is not None and s['camera']) else None,
                  sources={str(Path(f).resolve().relative_to(ROOT)): sha(f) for f in (__file__, modkit9.__file__)})
    (out / 'modkit10.json').write_text(json.dumps(record, indent=1))
    print(json.dumps(dict(modkit10=s, stats=stats, error=error, cam_recentred_max=record['cam_recentred_max']), ensure_ascii=False), flush=True)
    if error:
        raise SystemExit(error)


if __name__ == '__main__':
    main()
