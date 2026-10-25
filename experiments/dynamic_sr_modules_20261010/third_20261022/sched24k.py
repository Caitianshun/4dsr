"""Equal-update single-run control for PVCD (Claude 4dsr-71, 2026-10-24; pre-registration
tmp/claude_pvcd_reopt_20261024/prereg_sched24k.md). The schedule was extracted by Codex P1 dsr-pvcd-sched24k-p1-sol-20261024 and
checked against the code and the Bsync configuration by Claude.

One continuous SR stage of A x 6000 = 24000 steps on the plain w4 recipe (no PVCD), every step-indexed schedule stretched by A = 4:
- Gaussian learning rates: cullretrain calls g.update_learning_rate(LR_OFFSET + s); the hook evaluates the original curves at
  LR_OFFSET + s / A (xyz, deformation and grid share position_lr_max_steps = 20000, so new step 4k gets the old step-k rate);
  the constant groups (features, opacity, scaling, rotation) are untouched.
- SH: the loop's oneupSHdegree calls are forwarded only where (LR_OFFSET + s / A) % 1000 == 0 (s = 4000, ..., 24000).
- Topology: densify_from / until / interval, pruning_from / interval and opacity_reset_interval of the opt namespace returned by
  cr.configuration are multiplied by A (the plan and the baseline configuration-equality check are untouched). The threshold ramps
  divide by densify_until_iter and stretch with it; end points, the 360k admission cap and the size rule are unchanged.
- cr.SR_STEPS = A x 6000: default stop, colour-affine / warp-field max_steps (warp lr 0.002 * 0.1 ** (s / 24000)) and srfix's
  final-step fix (topology skipped once, at 24000).
- Sampling: the registered 6000-row table is cycled A times (row (s - 1) mod 6000); nothing is reset between cycles.
Parent, teacher, warp field, colour affine, lambda_SR, D-SSIM, prune gate and RNG inheritance are the ctlp6 recipe.
The hooks are installed before modkit4.main(), so modkit's wrappers (installed inside main) see the real cursor s.
Self-checks: at s = 1 (scaled opt, affine max_steps, cycled table), every step (contiguous s), at s = 4000 / 24000 (learning rates);
after training: counts, topology cursors, SH calls; any failure stops the job before evaluation. Audit: OUT/sched24k_audit.jsonl,
record: OUT/sched24k.json.

usage: sched24k.py [--factor 4] -- <modkit4.py arguments of the plain w4 run, including its own --; no --stop>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
MOD = ROOT / 'experiments/dynamic_sr_modules_20261010'
GEO = MOD / 'geoprobe_20261017'
for _p in (MOD, GEO):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

TOPOLOGY = ('densify_from_iter', 'densify_until_iter', 'densification_interval', 'pruning_from_iter', 'pruning_interval',
            'opacity_reset_interval')
SCHEDULED = {'xyz': 'xyz_scheduler_args', 'deformation': 'deformation_scheduler_args', 'grid': 'grid_scheduler_args'}
AUDIT = (1, 400, 2400, 4000, 8000, 12000, 16000, 20000, 23600, 24000)


def install(cr, GaussianModel, A, state, log=None):
    """Installs the stretched schedule on the imported cullretrain module and GaussianModel class (no file is modified)."""
    N0 = cr.SR_STEPS
    if (N0, cr.LR_OFFSET) != (6000, 14000):
        raise SystemExit(f'unexpected cullretrain constants SR_STEPS={N0} LR_OFFSET={cr.LR_OFFSET}')
    cr.SR_STEPS = A * N0
    C0, P0, I0 = cr.configuration, cr.Prefetch, cr.CameraAffine.__init__
    U0, H0 = GaussianModel.update_learning_rate, GaussianModel.oneupSHdegree
    state.update(A=A, N0=N0, s=0, last=0, sh_calls=[], affine_max_steps=[], opt=None, rows_checked=False, lr_checks=[])

    def configuration(plan):
        h, opt, pipe, sh = C0(plan)
        state['opt'] = {k: getattr(opt, k) for k in TOPOLOGY}
        for k in TOPOLOGY:
            setattr(opt, k, A * getattr(opt, k))
        return h, opt, pipe, sh

    class RepeatPrefetch(P0):
        def __init__(self, fn, keys, workers=3):
            if not state['rows_checked']:
                fn(N0 - 1)
                try:
                    fn(N0)
                except IndexError:
                    pass
                else:
                    raise SystemExit(f'the registered schedule has more than {N0} rows; cycling it would skip rows')
                state['rows_checked'] = True
            super().__init__(lambda i: fn(i % N0), keys, workers)

    def affine_init(self, *args, **kwargs):
        I0(self, *args, **kwargs)
        state['affine_max_steps'].append(self.max_steps)

    def update_learning_rate(self, iteration):
        s = iteration - cr.LR_OFFSET
        result = U0(self, cr.LR_OFFSET + s / A)
        if s >= 1:
            if s != state['last'] + 1:
                raise SystemExit(f'SR steps not contiguous: {state["last"]} -> {s}')
            state['last'] = s
        state['s'] = s
        if s == 1:
            scaled = {k: A * v for k, v in (state['opt'] or {}).items()}
            if not state['opt'] or not state['rows_checked'] or state['affine_max_steps'] != [A * N0]:
                raise SystemExit(f'schedule not installed: opt={state["opt"]} rows={state["rows_checked"]} '
                                 f'affine={state["affine_max_steps"]}')
            state['opt_scaled'] = scaled
        if s in (1, A * 1000, A * N0):
            clock = cr.LR_OFFSET + s / A
            for group in self.optimizer.param_groups:
                name = 'grid' if 'grid' in group['name'] else group['name']
                if name in SCHEDULED:
                    want = getattr(self, SCHEDULED[name])(clock)
                    plain = getattr(self, SCHEDULED[name])(iteration)
                    if abs(group['lr'] - want) > 1e-12 * max(1.0, abs(want)):
                        raise SystemExit(f'lr mapping failed at s={s}: {group["name"]} {group["lr"]} != {want}')
                    state['lr_checks'].append(dict(s=s, group=group['name'], lr=group['lr'], unmapped=plain))
        if log is not None and s in AUDIT:
            log(dict(s=s, clock=cr.LR_OFFSET + s / A, points=int(self.get_xyz.shape[0]),
                     active_sh_degree=int(self.active_sh_degree), spatial_lr_scale=float(self.spatial_lr_scale),
                     lr={g['name']: g['lr'] for g in self.optimizer.param_groups}))
        return result

    def oneupSHdegree(self):
        s = state['s']
        if s >= 1:
            if (cr.LR_OFFSET + s / A) % 1000 != 0:
                return None
            state['sh_calls'].append(s)
        return H0(self)

    cr.configuration, cr.Prefetch = configuration, RepeatPrefetch
    cr.CameraAffine.__init__ = affine_init
    GaussianModel.update_learning_rate, GaussianModel.oneupSHdegree = update_learning_rate, oneupSHdegree


def expected_topology(opt, A, stop):
    """Cursors at which the original topology() runs densify / prune, for the A-scaled opt (srfix skips stop)."""
    cursors = []
    for it in range(1, stop):
        if it >= A * opt['densify_until_iter']:
            continue
        if it > A * opt['densify_from_iter'] and it % (A * opt['densification_interval']) == 0:
            cursors.append(it)
    prune = [it for it in range(1, stop) if it < A * opt['densify_until_iter'] and it > A * opt['pruning_from_iter']
             and it % (A * opt['pruning_interval']) == 0]
    resets = [it for it in range(1, stop) if it < A * opt['densify_until_iter'] and it % (A * opt['opacity_reset_interval']) == 0]
    return cursors, prune, resets


def verify(out, state):
    A, N0 = state['A'], state['N0']
    stop = A * N0
    done = json.loads((out / 'complete.json').read_text())
    counts = done['counts']
    dens, prune, resets = expected_topology(state['opt'], A, stop)
    changed = []
    rows = 0
    with (out / 'trajectory.jsonl').open() as f:
        for k, line in enumerate(f, 1):
            r = json.loads(line)
            if r['cursor'] != k:
                raise SystemExit(f'trajectory cursor {r["cursor"]} at row {k}')
            if r['points_after'] != r['points_before']:
                changed.append(k)
            rows = k
    checks = dict(
        cursor=done['cursor'] == stop,
        trajectory_rows=rows == stop,
        steps_contiguous=state['last'] == stop,
        prune=counts['prune'] == len(prune),
        densify_at_most=counts['densify'] <= len(dens),
        opacity_reset=counts['opacity_reset'] == len(resets),
        grow=counts['grow'] == 0,
        final_step_skipped=counts.get('final_step_topology_skipped') == 1,
        topology_cursors=set(changed) <= set(dens) | set(prune),
        sh_calls=state['sh_calls'] == [A * k for k in range(1000, N0 + 1, 1000)],
        affine_max_steps=state['affine_max_steps'] == [stop],
        lr_checks=len(state['lr_checks']) == 9)
    return checks, dict(expected_densify=len(dens), expected_prune=len(prune), expected_resets=len(resets),
                        first_event=min(dens) if dens else None, last_event=max(dens) if dens else None,
                        point_change_cursors=len(changed), counts=counts, seconds=done['seconds'], peak_GB=done['peak_GB'],
                        points=done['points'])


def main():
    argv = sys.argv[1:]
    if '--' not in argv:
        raise SystemExit(__doc__)
    cut = argv.index('--')
    own, kit = argv[:cut], argv[cut + 1:]
    p = argparse.ArgumentParser(prog='sched24k.py')
    p.add_argument('--factor', type=int, default=4)
    a = p.parse_args(own)
    if '--' not in kit:
        raise SystemExit('the modkit4 arguments must contain their own -- before the srfix arguments')
    sr_args = kit[kit.index('--') + 1:]
    if sr_args[:4] != ['--mode', 'fix', 'cullretrain', 'sr']:
        raise SystemExit('expected the w4 chain: ... -- --mode fix cullretrain sr <SR args>')
    if '--stop' in sr_args:
        raise SystemExit('do not pass --stop: the stop is factor x SR_STEPS')
    out = Path(sr_args[sr_args.index('--out') + 1])
    out = out if out.is_absolute() else ROOT / out
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('a fresh output directory is required: ' + str(out))
    out.mkdir(parents=True, exist_ok=True)

    import srfix  # puts cullretrain on sys.path and imports it (unmodified) and torch
    import rt  # noqa: F401  (pinned runtime and registered experiment code on sys.path; already imported by cullretrain)
    from scene.gaussian_model import GaussianModel
    cr = srfix.cr
    audit = (out / 'sched24k_audit.jsonl').open('a', buffering=1)
    state = {}
    install(cr, GaussianModel, a.factor, state, log=lambda row: audit.write(json.dumps(row) + '\n'))
    import modkit4
    record = dict(kind='sched24k', factor=a.factor, sr_steps=cr.SR_STEPS, modkit4_argv=kit,
                  prereg='tmp/claude_pvcd_reopt_20261024/prereg_sched24k.md',
                  source={str(Path(f).resolve().relative_to(ROOT)): rt.sha(Path(f))
                          for f in (__file__, srfix.__file__, cr.__file__, modkit4.__file__, sys.modules['modkit2'].__file__,
                                    sys.modules['modkit'].__file__)})
    rt.write_json(out / 'sched24k.json', record)
    sys.argv = [str(GEO / 'modkit4.py')] + kit
    try:
        modkit4.main()
    finally:
        audit.close()
    checks, summary = verify(out, state)
    record.update(opt=state['opt'], opt_scaled=state.get('opt_scaled'), sh_calls=state['sh_calls'],
                  affine_max_steps=state['affine_max_steps'], lr_checks=state['lr_checks'], checks=checks, **summary)
    rt.write_json(out / 'sched24k.json', record)
    failed = [k for k, v in checks.items() if not v]
    if failed:
        raise SystemExit('sched24k checks failed: ' + ', '.join(failed))


if __name__ == '__main__':
    main()
