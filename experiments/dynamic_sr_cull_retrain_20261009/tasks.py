"""Build task graphs for the 2026-10-09 culling retrain (validation replay and wave 1)."""
import argparse
import json
from pathlib import Path

import rt

BASE = 'output/dynamic_sr_multiview_footprint_20261007'
OUT = 'output/dynamic_sr_cull_retrain_20261009'
COOK = 'Cook_both_seeds_actual_revisionB_1791421128150050152'
CUT7 = 'Cut_seed20261007_actual_revisionB_1791421833156833604'
CUT8 = 'Cut_seed20261008_actual_revisionB_1791424843421006692'


def endpoint(scene, seed, sr, ev):
    return dict(scene=scene, seed=seed, parent=f'{BASE}/full_LR_prefixes/{scene}/seed_{seed}', sr=sr, eval=ev)


def registry():
    r = {}
    for seed, tag in ((20261007, '07'), (20261008, '08')):
        for arm in ('B0', 'Bsync'):
            r[f'Cook{tag}_{arm}'] = endpoint('cook_spinach', seed, f'{BASE}/full_SR_refinement/{COOK}/cook_spinach/seed_{seed}/{arm}',
                                             f'{BASE}/full_materialized/{COOK}/cook_spinach/seed_{seed}/{arm}/evaluation')
            cut = CUT7 if tag == '07' else CUT8
            r[f'Cut{tag}_{arm}'] = endpoint('cut_roasted_beef', seed, f'{BASE}/full_SR_refinement/{cut}/cut_roasted_beef/seed_{seed}/{arm}',
                                            f'{BASE}/full_materialized/{cut}/cut_roasted_beef/seed_{seed}/{arm}/evaluation')
            suffix = '_zero_science_operational_recovery1' if tag == '08' else ''
            r[f'Discussion{tag}_{arm}'] = endpoint('meetroom_discussion', seed, f'{BASE}/full_SR_refinement/meetroom_discussion/seed_{seed}/{arm}_sourceaware_v2_20261008',
                                                   f'{BASE}/full_evaluation/meetroom_discussion/seed{seed}/{arm}_sourceaware_v2_20261008{suffix}')
            vr = f'{arm}_sourceaware_GPU1_v1_20261008' if tag == '07' else f'{arm}_local_sourceaware_v2'
            r[f'VRHeadset{tag}_{arm}'] = endpoint('meetroom_vrheadset', seed, f'{BASE}/full_SR_refinement/meetroom_vrheadset/seed_{seed}/{vr}',
                                                  f'{BASE}/full_evaluation/meetroom_vrheadset/seed{seed}/{vr}')
    for key, e in r.items():
        for field in ('parent', 'sr', 'eval'):
            if not (rt.ROOT / e[field]).is_dir():
                raise FileNotFoundError(f'{key} {field}: {e[field]}')
        if not (rt.ROOT / e['sr'] / 'checkpoint_06000.pt').is_file():
            raise FileNotFoundError(f'{key} checkpoint_06000.pt')
    return r


ARMS = {
    'cull': ['--cull', '--prune-gate', 'none'],
    'cullaff': ['--cull', '--prune-gate', 'none', '--affine'],
}


def wave1(r, out):
    tasks, posthoc = [], []
    seeds = sorted({k.rsplit('_', 1)[0] for k in r})
    for key in sorted(r):
        e = r[key]
        posthoc.append(dict(name=f'posthoc_{key}', deps=[], done=f'{out}/posthoc/{key}_cull/summary.json',
                            argv=['eval', '--checkpoint', f"{e['sr']}/checkpoint_06000.pt", '--plan', f"{e['sr']}/config.json",
                                  '--cull', '--out', f'{out}/posthoc/{key}_cull', '--label', f'{key}_posthoc_cull']))
    lr, sr, ev = [], [], []
    for seed_key in seeds:
        e = r[f'{seed_key}_Bsync']
        parent = rt.ROOT / e['parent']
        fine0, deps = [], []
        if not (parent / 'checkpoint_fine_00000.pt').is_file():
            coarse = f'{out}/lr/{seed_key}_coarse_replay'
            lr.append(dict(name=f'lr_{seed_key}_coarse', deps=[], done=f'{coarse}/complete.json',
                           argv=['lr', '--parent', e['parent'], '--coarse-only', '--arm', 'coarse_replay', '--out', coarse]))
            fine0, deps = ['--fine0', f'{coarse}/checkpoint_fine_00000.pt'], [f'lr_{seed_key}_coarse']
        for arm, flags in ARMS.items():
            run = f'{seed_key}_{arm}'
            lr.append(dict(name=f'lr_{run}', deps=deps, done=f'{out}/lr/{run}/complete.json',
                           argv=['lr', '--parent', e['parent'], *fine0, '--arm', arm, *flags, '--out', f'{out}/lr/{run}']))
            sr.append(dict(name=f'sr_{run}', deps=[f'lr_{run}'], done=f'{out}/sr/{run}/complete.json',
                           argv=['sr', '--parent-checkpoint', f'{out}/lr/{run}/checkpoint_fine_14000.pt', '--baseline-sr', e['sr'],
                                 '--arm', arm, *flags, '--out', f'{out}/sr/{run}']))
            ev.append(dict(name=f'eval_{run}', deps=[f'sr_{run}'], done=f'{out}/eval/{run}/summary.json',
                           argv=['eval', '--checkpoint', f'{out}/sr/{run}/checkpoint_06000.pt', '--out', f'{out}/eval/{run}', '--label', run]))
    # Coarse replays first, LR runs interleaved with the independent post-hoc evaluations, then SR, then evaluation.
    head = [t for t in lr if t['name'].endswith('_coarse')]
    body = [t for t in lr if not t['name'].endswith('_coarse')]
    mixed = []
    for i, t in enumerate(body):
        mixed.append(t)
        if i < len(posthoc):
            mixed.append(posthoc[i])
    mixed += posthoc[len(body):]
    # Wave 2: SR-weight / D-SSIM re-test on the cull-only parents (same SR culling), lambda_SR=0.05 is wave 1's cull arm.
    sr2, ev2 = [], []
    for seed_key in seeds:
        e = r[f'{seed_key}_Bsync']
        for tag, extra in (('lam0p1', ['--lambda-sr', '0.1']), ('lam0p2', ['--lambda-sr', '0.2']), ('lam0p5', ['--lambda-sr', '0.5']),
                           ('dssim0p2', ['--lambda-sr', '0.05', '--dssim', '0.2'])):
            run = f'{seed_key}_cull_{tag}'
            sr2.append(dict(name=f'sr_{run}', deps=[f'lr_{seed_key}_cull'], done=f'{out}/sr/{run}/complete.json',
                            argv=['sr', '--parent-checkpoint', f'{out}/lr/{seed_key}_cull/checkpoint_fine_14000.pt', '--baseline-sr', e['sr'],
                                  '--arm', f'cull_{tag}', *ARMS['cull'], *extra, '--out', f'{out}/sr/{run}']))
            ev2.append(dict(name=f'eval_{run}', deps=[f'sr_{run}'], done=f'{out}/eval/{run}/summary.json',
                            argv=['eval', '--checkpoint', f'{out}/sr/{run}/checkpoint_06000.pt', '--out', f'{out}/eval/{run}', '--label', run]))
    return head + mixed + ev + sr + ev2 + sr2


def control(r, out):
    """Same-hardware controls: 'nocull' (author topology), 'gate' (200k prune gate removed, no culling) and 'gateaff'
    (gate removed + colour affine, no culling), LR+SR; each evaluated as-is and post-hoc culled. With 'cull'/'cullaff'
    this gives culling x affine at the same gate setting, plus the gate change itself."""
    tasks = []
    for seed_key in sorted({k.rsplit('_', 1)[0] for k in r}):
        e = r[f'{seed_key}_Bsync']
        parent = rt.ROOT / e['parent']
        fine0 = [] if (parent / 'checkpoint_fine_00000.pt').is_file() else ['--fine0', f'{out}/lr/{seed_key}_coarse_replay/checkpoint_fine_00000.pt']
        for arm, flags in (('nocull', []), ('gate', ['--prune-gate', 'none']), ('gateaff', ['--prune-gate', 'none', '--affine'])):
            run = f'{seed_key}_{arm}'
            tasks.append(dict(name=f'lr_{run}', deps=[], done=f'{out}/lr/{run}/complete.json',
                              argv=['lr', '--parent', e['parent'], *fine0, '--arm', arm, *flags, '--out', f'{out}/lr/{run}']))
            tasks.append(dict(name=f'sr_{run}', deps=[f'lr_{run}'], done=f'{out}/sr/{run}/complete.json',
                              argv=['sr', '--parent-checkpoint', f'{out}/lr/{run}/checkpoint_fine_14000.pt', '--baseline-sr', e['sr'],
                                    '--arm', arm, *flags, '--out', f'{out}/sr/{run}']))
            tasks.append(dict(name=f'eval_{run}', deps=[f'sr_{run}'], done=f'{out}/eval/{run}/summary.json',
                              argv=['eval', '--checkpoint', f'{out}/sr/{run}/checkpoint_06000.pt', '--out', f'{out}/eval/{run}', '--label', run]))
            tasks.append(dict(name=f'eval_{run}_posthoc', deps=[f'sr_{run}'], done=f'{out}/eval/{run}_posthoc_cull/summary.json',
                              argv=['eval', '--checkpoint', f'{out}/sr/{run}/checkpoint_06000.pt', '--cull', '--out', f'{out}/eval/{run}_posthoc_cull',
                                    '--label', f'{run}_posthoc_cull']))
    order = [t for t in tasks if t['name'].startswith('eval')] + [t for t in tasks if t['name'].startswith('sr')] + [t for t in tasks if t['name'].startswith('lr')]
    return order


def control_lr(r):
    """3090 job after the wave: the 24 control LR runs, then affine-corrected cam01 diagnostics for the affine arms.
    SR dependencies run elsewhere (claimed on the PRO6000) and are only awaited through their done files."""
    out = f'{OUT}/wave1'
    lr = sorted([t for t in control(r, out) if t['name'].startswith('lr_')],
                key=lambda t: (not t['name'].endswith('_nocull'), t['name'].split('_')[1][:-2] not in ('Cook', 'Cut')))
    wave = {t['name']: t for t in wave1(r, out)}
    ctrl = {t['name']: t for t in control(r, out)}
    extra, diag = [], []
    for seed_key in sorted({k.rsplit('_', 1)[0] for k in r}):
        for arm, source in (('cullaff', wave), ('gateaff', ctrl)):
            run = f'{seed_key}_{arm}'
            if arm == 'cullaff':
                extra += [source[f'lr_{run}'], source[f'sr_{run}']]
            else:
                extra.append(source[f'sr_{run}'])
            diag.append(dict(name=f'diag_{run}', deps=[f'sr_{run}'], script='affine_traindiag.py', done=f'{out}/traindiag_affine/{run}/summary.json',
                             argv=['--checkpoint', f'{out}/sr/{run}/checkpoint_06000.pt', '--out', f'{out}/traindiag_affine/{run}']))
    pool = {**wave, **ctrl}
    have = {t['name'] for t in lr + diag + extra}
    queue = [d for t in lr + diag + extra for d in t['deps']]
    while queue:
        d = queue.pop()
        if d not in have:
            have.add(d)
            extra.append(pool[d])
            queue += pool[d]['deps']
    return lr + diag + extra


def wave3(r, out):
    """SR-loss confirmation on the cull+affine parents: D-SSIM 0.2 at lambda_SR 0.05 and 0.2."""
    tasks = []
    for seed_key in sorted({k.rsplit('_', 1)[0] for k in r}):
        e = r[f'{seed_key}_Bsync']
        for tag, extra in (('dssim0p2', ['--lambda-sr', '0.05', '--dssim', '0.2']), ('lam0p2_dssim0p2', ['--lambda-sr', '0.2', '--dssim', '0.2'])):
            run = f'{seed_key}_cullaff_{tag}'
            tasks.append(dict(name=f'sr_{run}', deps=[f'lr_{seed_key}_cullaff'], done=f'{out}/sr/{run}/complete.json',
                              argv=['sr', '--parent-checkpoint', f'{out}/lr/{seed_key}_cullaff/checkpoint_fine_14000.pt', '--baseline-sr', e['sr'],
                                    '--arm', f'cullaff_{tag}', *ARMS['cullaff'], *extra, '--out', f'{out}/sr/{run}']))
            tasks.append(dict(name=f'eval_{run}', deps=[f'sr_{run}'], done=f'{out}/eval/{run}/summary.json',
                              argv=['eval', '--checkpoint', f'{out}/sr/{run}/checkpoint_06000.pt', '--out', f'{out}/eval/{run}', '--label', run]))
    pool = {t['name']: t for t in wave1(r, out)}
    have = {t['name'] for t in tasks}
    queue = [d for t in tasks for d in t['deps']]
    while queue:
        d = queue.pop()
        if d not in have:
            have.add(d); tasks.append(pool[d]); queue += pool[d]['deps']
    return [t for t in tasks if t['name'].startswith('eval')] + [t for t in tasks if t['name'].startswith('sr')] + [t for t in tasks if t['name'].startswith('lr')]


def wave4(r, out):
    """Fair baseline for the SR loss: no-cull parents (author topology) + D-SSIM 0.2 at lambda_SR 0.05, no culling in SR;
    evaluated as-is and post-hoc culled (the SRGS-v3-style recipe on the same backbone)."""
    tasks = []
    for seed_key in sorted({k.rsplit('_', 1)[0] for k in r}):
        e = r[f'{seed_key}_Bsync']
        run = f'{seed_key}_nocull_dssim0p2'
        tasks.append(dict(name=f'sr_{run}', deps=[f'lr_{seed_key}_nocull'], done=f'{out}/sr/{run}/complete.json',
                          argv=['sr', '--parent-checkpoint', f'{out}/lr/{seed_key}_nocull/checkpoint_fine_14000.pt', '--baseline-sr', e['sr'],
                                '--arm', 'nocull_dssim0p2', '--lambda-sr', '0.05', '--dssim', '0.2', '--out', f'{out}/sr/{run}']))
        tasks.append(dict(name=f'eval_{run}', deps=[f'sr_{run}'], done=f'{out}/eval/{run}/summary.json',
                          argv=['eval', '--checkpoint', f'{out}/sr/{run}/checkpoint_06000.pt', '--out', f'{out}/eval/{run}', '--label', run]))
        tasks.append(dict(name=f'eval_{run}_posthoc', deps=[f'sr_{run}'], done=f'{out}/eval/{run}_posthoc_cull/summary.json',
                          argv=['eval', '--checkpoint', f'{out}/sr/{run}/checkpoint_06000.pt', '--cull', '--out', f'{out}/eval/{run}_posthoc_cull',
                                '--label', f'{run}_posthoc_cull']))
    pool = {t['name']: t for t in control(r, out)}
    pool.update({t['name']: t for t in wave1(r, out)})
    have = {t['name'] for t in tasks}
    queue = [d for t in tasks for d in t['deps']]
    while queue:
        d = queue.pop()
        if d not in have:
            have.add(d); tasks.append(pool[d]); queue += pool[d]['deps']
    return [t for t in tasks if t['name'].startswith('eval')] + [t for t in tasks if t['name'].startswith('sr')] + [t for t in tasks if t['name'].startswith('lr')]


def shcheck(r, out):
    """Zero-training SH-degree truncation check on finished/pending SR checkpoints (sh_degree_check.py)."""
    pool = {t['name']: t for t in wave1(r, out)}
    pool.update({t['name']: t for t in control(r, out)})
    pool.update({t['name']: t for t in wave3(r, out)})
    tasks = []
    for seed_key in sorted({k.rsplit('_', 1)[0] for k in r}):
        e = r[f'{seed_key}_Bsync']
        specs = [('official_posthoc', f"{e['sr']}/checkpoint_06000.pt", ['--plan', f"{e['sr']}/config.json", '--cull'], None),
                 ('cull', f'{out}/sr/{seed_key}_cull/checkpoint_06000.pt', [], f'sr_{seed_key}_cull'),
                 ('cull_r2', f'{out}/sr_repeat/{seed_key}_cull_r2/checkpoint_06000.pt', [], None),
                 ('cullaff', f'{out}/sr/{seed_key}_cullaff/checkpoint_06000.pt', [], f'sr_{seed_key}_cullaff'),
                 ('cullaff_dssim0p2', f'{out}/sr/{seed_key}_cullaff_dssim0p2/checkpoint_06000.pt', [], f'sr_{seed_key}_cullaff_dssim0p2'),
                 ('nocull_posthoc', f'{out}/sr/{seed_key}_nocull/checkpoint_06000.pt', ['--cull'], f'sr_{seed_key}_nocull')]
        for tag, ckpt, extra, dep in specs:
            run = f'{seed_key}_{tag}'
            tasks.append(dict(name=f'shcheck_{run}', deps=[dep] if dep else [], script='sh_degree_check.py', done=f'{out}/shcheck/{run}/summary.json',
                              argv=['--checkpoint', ckpt, *extra, '--out', f'{out}/shcheck/{run}']))
    have = {t['name'] for t in tasks}
    queue = [d for t in tasks for d in t['deps']]
    extra_tasks = []
    while queue:
        d = queue.pop()
        if d not in have:
            have.add(d); extra_tasks.append(pool[d]); queue += pool[d]['deps']
    return tasks + extra_tasks


def wave5(r, out):
    """Candidate simple recipe: affine-trained parents without in-training culling (gateaff) + D-SSIM 0.2 at lambda_SR 0.05,
    no culling in SR; evaluated as-is and post-hoc culled."""
    tasks = []
    for seed_key in sorted({k.rsplit('_', 1)[0] for k in r}):
        e = r[f'{seed_key}_Bsync']
        run = f'{seed_key}_gateaff_dssim0p2'
        tasks.append(dict(name=f'sr_{run}', deps=[f'lr_{seed_key}_gateaff'], done=f'{out}/sr/{run}/complete.json',
                          argv=['sr', '--parent-checkpoint', f'{out}/lr/{seed_key}_gateaff/checkpoint_fine_14000.pt', '--baseline-sr', e['sr'],
                                '--arm', 'gateaff_dssim0p2', '--prune-gate', 'none', '--affine', '--lambda-sr', '0.05', '--dssim', '0.2',
                                '--out', f'{out}/sr/{run}']))
        tasks.append(dict(name=f'eval_{run}', deps=[f'sr_{run}'], done=f'{out}/eval/{run}/summary.json',
                          argv=['eval', '--checkpoint', f'{out}/sr/{run}/checkpoint_06000.pt', '--out', f'{out}/eval/{run}', '--label', run]))
        tasks.append(dict(name=f'eval_{run}_posthoc', deps=[f'sr_{run}'], done=f'{out}/eval/{run}_posthoc_cull/summary.json',
                          argv=['eval', '--checkpoint', f'{out}/sr/{run}/checkpoint_06000.pt', '--cull', '--out', f'{out}/eval/{run}_posthoc_cull',
                                '--label', f'{run}_posthoc_cull']))
    pool = {t['name']: t for t in control(r, out)}
    pool.update({t['name']: t for t in wave1(r, out)})
    have = {t['name'] for t in tasks}
    queue = [d for t in tasks for d in t['deps']]
    while queue:
        d = queue.pop()
        if d not in have:
            have.add(d); tasks.append(pool[d]); queue += pool[d]['deps']
    return [t for t in tasks if t['name'].startswith('eval')] + [t for t in tasks if t['name'].startswith('sr')] + [t for t in tasks if t['name'].startswith('lr')]


def validation(r, out):
    c7, c8 = r['Cook07_Bsync'], r['Cook08_Bsync']
    return [
        dict(name='v_lr_replay_cook07', deps=[], argv=['lr', '--parent', c7['parent'], '--arm', 'nocull', '--stop', '300', '--out', f'{out}/lr_replay_cook07_nocull_300']),
        dict(name='v_sr_replay_cook07', deps=[], argv=['sr', '--parent-checkpoint', f"{c7['parent']}/checkpoint_fine_14000.pt", '--baseline-sr', c7['sr'],
                                                      '--arm', 'nocull', '--stop', '200', '--out', f'{out}/sr_replay_cook07_Bsync_nocull_200']),
        dict(name='v_eval_cook08B0_cull', deps=[], argv=['eval', '--checkpoint', f"{r['Cook08_B0']['sr']}/checkpoint_06000.pt", '--plan', f"{r['Cook08_B0']['sr']}/config.json",
                                                        '--cull', '--frames', '0', '150', '--out', f'{out}/eval_cook08B0_cull_2frames', '--label', 'v_cook08B0_cull']),
        dict(name='v_lr_cull_cook08', deps=[], argv=['lr', '--parent', c8['parent'], '--arm', 'cullaff', *ARMS['cullaff'], '--stop', '1000', '--out', f'{out}/lr_cook08_cullaff_1000']),
        dict(name='v_sr_cull_cook08', deps=[], argv=['sr', '--parent-checkpoint', f"{c8['parent']}/checkpoint_fine_14000.pt", '--baseline-sr', c8['sr'],
                                                    '--arm', 'cullaff', *ARMS['cullaff'], '--stop', '100', '--out', f'{out}/sr_cook08_cullaff_100']),
    ]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('which', choices=('validation', 'wave1', 'control', 'control_lr', 'control_lr_only', 'wave3', 'wave4', 'wave5', 'shcheck'))
    p.add_argument('--out', required=True)
    a = p.parse_args()
    r = registry()
    tasks = dict(validation=lambda: validation(r, f'{OUT}/validation'), wave1=lambda: wave1(r, f'{OUT}/wave1'),
                 control=lambda: control(r, f'{OUT}/wave1'),
                 control_lr=lambda: control_lr(r),
                 wave3=lambda: wave3(r, f'{OUT}/wave1'),
                 wave4=lambda: wave4(r, f'{OUT}/wave1'),
                 shcheck=lambda: shcheck(r, f'{OUT}/wave1'),
                 wave5=lambda: wave5(r, f'{OUT}/wave1'),
                 control_lr_only=lambda: sorted([t for t in control(r, f'{OUT}/wave1') if t['name'].startswith('lr_')],
                                           key=lambda t: (not t['name'].endswith('_nocull'), t['name'].split('_')[1][:-2] not in ('Cook', 'Cut'))))[a.which]()
    Path(a.out).write_text(json.dumps(tasks, indent=1))
    print(len(tasks), 'tasks ->', a.out)


if __name__ == '__main__':
    main()
