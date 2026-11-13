"""Second PRO6000 job of the attribution runs (Claude, 2026-11-12). Job 1 (attr-p6-01) could not start the new-scene SR stages:
teacher_paths_new.py only writes under output/.../newscenes_reg_20261103 ("Write outside authorized namespace"), so the Sear07
base and the Salmon07 same-card pair (w4 + base) now go to newscenes_reg_20261103/attrib_20261112. Same launch, recipe and arms
as in make_attrib_jobs.py; rows of this job are written to attrib_20261112/rows2/.
  /usr/bin/python3 -I make_attrib_jobs2.py"""
import json
import os
from pathlib import Path

R = Path('/home/cai_tianshun/Project/4dsr')
E = 'experiments/dynamic_sr_modules_20261010'
A = 'output/dynamic_sr_modules_20261010/attrib_20261112'
NS = 'output/dynamic_sr_modules_20261010/newscenes_reg_20261103'
NA = f'{NS}/attrib_20261112'
HERE = R / E / 'attrib_20261112'
PY = '/home/cai_tianshun/miniconda3/envs/mml_bw/bin/python'
PRO6000 = 'GPU-ddc2c5d8-3507-6293-837c-b1ea6b5e1cc5'
P = 'data/dynamic_sr/full_time_20261007/prepared'
SCHED = 'output/dynamic_sr_modules_20261010/newscenes_sched_20261103/sr/registry'
TORCH_HOME = 'output/dynamic_sr_modules_20261010/bl4_20261024/torch_home'
FITARGS = '--arm PC --target lr --steps 100 --steps-first 300 --c-reg'
GUARD = f'CRITICAL {E}/third_20261022/disk_guard.py --min-gib 15'
mk = f'{A}/markers'

MAN = dict(Coffee07='coffee_martini_area4', Cut07='cut_roasted_beef_area4', Flame07='flame_steak_area4', Discussion07='meetroom_discussion_area4',
           VRHeadset07='meetroom_vrheadset_area4')
SC = {'Sear07': dict(m='sear_steak_area4', base=f'{NA}/sear/sr/Sear07_area4_at_areaTm/checkpoint_06000.pt'),
      'Salmon07': dict(m='flame_salmon_area4', base=f'{NA}/pair/sr/Salmon07_area4_at_areaTm/checkpoint_06000.pt',
                       field=f'{NA}/pair/sr/Salmon07_area4_at_w4/checkpoint_06000.pt', fit=f'{A}/fit/Salmon07_atw4_PCv2_lr',
                       resid=f'{A}/tv2/Salmon07_pc/SMX1')}
for p in (NA, f'{A}/fit/Salmon07_atw4_PCv2_lr', f'{A}/tv2/Salmon07_pc', f'{A}/rows2'):
    assert not (R / p).exists(), p


def sr_line(s, mname, sub, arms):
    return (f'{E}/newscenes_reg_20261103/teacher_paths_new.py --index {NS}/teacher_cache/{mname}_area4/teacher_index_sr_swinir_x4_areaft_seed20261007.json '
            f'--script {E}/geoprobe_20261017/run_modkit7_remote.py --receipt {NA}/receipts/{s}_{sub}_teacher_paths.json -- '
            f'--parent {NS}/lr/runs/{s}_area4_gateaff/checkpoint_fine_14000.pt '
            f'--baseline-sr {SCHED}/{mname}_area4/seed_20261007/areaft_teacher/Bsync --lambda-sr 0.2 --out-root {NA}/{sub} '
            f'--tag {s}_area4_at --view cam00 ' + ' '.join(f'--arm {x}' for x in arms))


def row_lines(s, row, ck, resid=None):
    m = f"{P}/{SC[s]['m']}/manifest_train_ready_seed20261007.json"
    o = f'{A}/rows2/{s}/{row}'
    r = f' --resid {resid}' if resid else ''
    return [f'{E}/area4_baselines_20261024/ours/render_ours_png.py --checkpoint {ck}{r} --manifest {m} --scope test --out {o}/render_test',
            f'{E}/area4_baselines_20261024/score_views_area4.py --manifest {m} --scope test --render-dir {o}/render_test --out {o}/score_test '
            f'--method attr_{s}_{row} --device cuda --execute --torch-home {TORCH_HOME}']


S = SC['Salmon07']
T = ['# lane T2: training on the PRO6000, one training process at a time', GUARD,
     sr_line('Sear07', 'sear_steak', 'sear', ['areaTm=']), f'TOUCH {mk}/U1_sear_base.attempted', GUARD,
     sr_line('Salmon07', 'flame_salmon', 'pair', ['w4=wfield:4', 'areaTm=']), f'TOUCH {mk}/U2_salmon_pair.attempted', GUARD,
     f"{E}/absorb_20261021/perframe_fit2.py --checkpoint {S['field']} --out {S['fit']} {FITARGS}",
     f"{E}/third_20261022/resid_temporal_variants2.py --fit {S['fit']} --out-root {A}/tv2/Salmon07_pc --variants SMX1",
     f'TOUCH {mk}/U3_salmon_fit.attempted', f'TOUCH {mk}/laneT2.done']
# 'nores' = the final PinSR student rendered without frame residuals (Tab. 3 row "PinSR w/o frame residuals"), 7 scenes; Cook07 is in Tab. 3
STUDENT = {}
for p in sorted((R / 'output/dynamic_lr_area4_20261024/ours').glob('*_area4_ours/renders_test/render_index.json')):
    s = p.parts[-3].split('_')[0]
    if s != 'Cook07':
        STUDENT[s] = (json.loads(p.read_text())['checkpoint']['path'], MAN[s])
for s, m in (('Sear07', 'sear_steak_area4'), ('Salmon07', 'flame_salmon_area4')):
    STUDENT[s] = (json.loads((R / f'{NS}/final/{s}_area4/renders_test/render_index.json').read_text())['checkpoint']['path'], m)
assert len(STUDENT) == 7
Ev = ['# lane E2: renders and scores (evaluation only)']
for s, (ck, m) in STUDENT.items():
    SC.setdefault(s, dict(m=m))
    Ev += row_lines(s, 'nores', ck)
Ev += [f'WAIT {mk}/U1_sear_base.attempted'] + row_lines('Sear07', 'base', SC['Sear07']['base'])
Ev += [f'WAIT {mk}/U2_salmon_pair.attempted'] + row_lines('Salmon07', 'field', S['field']) + row_lines('Salmon07', 'base', S['base'])
Ev += [f'WAIT {mk}/U3_salmon_fit.attempted'] + row_lines('Salmon07', 'pc', S['field'], S['resid']) + [f'TOUCH {mk}/laneE2.done']
(HERE / 'lane_T2.txt').write_text('\n'.join(T) + '\n')
(HERE / 'lane_E2.txt').write_text('\n'.join(Ev) + '\n')
job = dict(version=1, job_id='attr-p6-02', project_id='4dsr', role='local', operation='run', env_id='mml_bw', cwd=str(R),
           argv=[PY, '-u', str(R / E / 'static_sr_v2_20261026/sv2_lanes.py'), '--abort', f'{A}/ABORT2',
                 str(HERE / 'lane_T2.txt'), str(HERE / 'lane_E2.txt')],
           gpu_uuids=[PRO6000], gpu_memory_mib=40000, timeout_seconds=6 * 3600)
(HERE / 'req/attr-p6-02.json').write_text(json.dumps(job, indent=1))
for p in ('lane_T2.txt', 'lane_E2.txt', 'req/attr-p6-02.json'):
    os.chmod(HERE / p, 0o640)
print('T2', len(T), 'E2', len(Ev))
