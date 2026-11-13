"""Job of the SplatSuRe-on-our-base-configuration control (Claude, 2026-11-12). PRO6000, one training process at a time.
Lane T: selection maps from our LR parent, SplatSuRe SR stage (sps_layer.py), Flame Steak first, then Cook Spinach; then a
same-card base model per scene (the areaTm command of run_modkit7_remote.py, no SplatSuRe layer). Lane E renders and scores
each finished run with the main-table evaluator (cam00, 300 frames). Parents and teacher registries are those of the
existing base models (Flame07: holdout_w15 areaTm; Cook07: Tab. 3 base, geo_a100 TmB_ctlB).
  /usr/bin/python3 -I make_sps_jobs.py"""
import json
import os
from pathlib import Path

R = Path('/home/cai_tianshun/Project/4dsr')
E = 'experiments/dynamic_sr_modules_20261010'
O = 'output/dynamic_sr_modules_20261010/sps_base_20261112'
HERE = R / E / 'sps_base_20261112'
PY = '/home/cai_tianshun/miniconda3/envs/mml_bw/bin/python'
PRO6000 = 'GPU-ddc2c5d8-3507-6293-837c-b1ea6b5e1cc5'
P = 'data/dynamic_sr/full_time_20261007/prepared'
LRP = 'output/dynamic_sr_cull_retrain_20261009/wave1/lr/physlr_20261016/runs'
TORCH_HOME = 'output/dynamic_sr_modules_20261010/bl4_20261024/torch_home'
GUARD = f'CRITICAL {E}/third_20261022/disk_guard.py --min-gib 15'
mk = f'{O}/markers'
SC = {'Flame07': dict(m='flame_steak_area4', reg='output/dynamic_sr_physlr_20261016/registry/flame_steak_area4/seed_20261007/areaft_teacher/Bsync'),
      'Cook07': dict(m='cook_spinach_area4', reg=f'{E.replace("experiments", "output")}/geoprobe_20261017/registry/cook_spinach_area4/seed_20261007/areaft_teacher/Bsync')}
assert not (R / O).exists(), O
for s, c in SC.items():
    assert (R / LRP / f'{s}_area4_gateaff/checkpoint_fine_14000.pt').is_file() and (R / c['reg'] / 'config.json').is_file(), s


def sr_args(s, arm):
    return (f"-- --lr-operator area -- --mode fix cullretrain sr --parent-checkpoint {LRP}/{s}_area4_gateaff/checkpoint_fine_14000.pt "
            f"--baseline-sr {SC[s]['reg']} --arm {s}_area4_{arm} --prune-gate none --affine --lambda-sr 0.2 --dssim 0.2 "
            f"--out {O}/sr/{s}_area4_{arm}")


def row_lines(s, arm):
    m = f"{P}/{SC[s]['m']}/manifest_train_ready_seed20261007.json"
    o = f'{O}/rows/{s}/{arm}'
    return [f'{E}/area4_baselines_20261024/ours/render_ours_png.py --checkpoint {O}/sr/{s}_area4_{arm}/checkpoint_06000.pt '
            f'--manifest {m} --scope test --out {o}/render_test',
            f'{E}/area4_baselines_20261024/score_views_area4.py --manifest {m} --scope test --render-dir {o}/render_test --out {o}/score_test '
            f'--method sps_{s}_{arm} --device cuda --execute --torch-home {TORCH_HOME}']


T = ['# lane T: training on the PRO6000, one training process at a time']
Ev = ['# lane E: renders and scores (evaluation only)']
for s in SC:
    T += [GUARD, f'CRITICAL {E}/sps_base_20261112/sps_weights.py --parent {LRP}/{s}_area4_gateaff/checkpoint_fine_14000.pt --out {O}/weights/{s}',
          f"CRITICAL {E}/sps_base_20261112/sps_layer.py --weights {O}/weights/{s}/receipt.json {sr_args(s, 'sps')}",
          f'TOUCH {mk}/{s}_sps.attempted']
    Ev += [f'WAIT {mk}/{s}_sps.attempted'] + row_lines(s, 'sps')
for s in SC:
    T += [GUARD, f"{E}/geoprobe_20261017/modkit6.py {sr_args(s, 'basep6')[3:]}", f'TOUCH {mk}/{s}_base.attempted']
    Ev += [f'WAIT {mk}/{s}_base.attempted'] + row_lines(s, 'basep6')
T += [f'TOUCH {mk}/laneT.done']
Ev += [f'TOUCH {mk}/laneE.done']
(HERE / 'req').mkdir(exist_ok=True)
(HERE / 'lane_T.txt').write_text('\n'.join(T) + '\n')
(HERE / 'lane_E.txt').write_text('\n'.join(Ev) + '\n')
job = dict(version=1, job_id='sps-p6-02', project_id='4dsr', role='local', operation='run', env_id='mml_bw', cwd=str(R),
           argv=[PY, '-u', str(R / E / 'static_sr_v2_20261026/sv2_lanes.py'), '--abort', f'{O}/ABORT',
                 str(HERE / 'lane_T.txt'), str(HERE / 'lane_E.txt')],
           gpu_uuids=[PRO6000], gpu_memory_mib=40000, timeout_seconds=5 * 3600)
(HERE / 'req/sps-p6-02.json').write_text(json.dumps(job, indent=1))
for p in ('lane_T.txt', 'lane_E.txt', 'req/sps-p6-02.json'):
    os.chmod(HERE / p, 0o640)
print('T', len(T), 'E', len(Ev))
