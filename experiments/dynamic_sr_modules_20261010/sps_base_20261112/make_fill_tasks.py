"""Task files for the 11-12 fill-in runs (Claude, 2026-11-12; user: fill the empty 8-scene cells of Tab. 3 and extend the SplatSuRe
control to every scene, as fast as possible).
Tab. 3 rows, recipes as on Cook Spinach (ablation_rescore.json), each followed by the PinSR residual fit (perframe_fit2, PC v2) and,
on the SMX1 scenes of PinSR (Coffee, Discussion, Salmon), the SMX1 temporal variant; render + score with the main-table evaluator:
  k1   'Consensus from one source': pvcd_modkit.py, K = 1 (source A of the PinSR student), weight 0.5, D-SSIM 0.2, scale 4, seed 20261014
  drop 'DropGaussian for consensus': modkit6 w4 recipe with --drop-rate 0.2 (drop seed 20261016)
Discussion has both runs already (mk_pvcdk1s4p6, mk_drop02p6, with fits and SMX1 variants): render + score only.
Queues: queue_local (PRO6000 / 3090 workers): SplatSuRe scenes, Discussion scoring, Coffee + Flame rows (their sources and
registries are outside the remote sync prefixes); queue_cts: Cut and VR Headset rows; queue_cts2: Sear and Salmon rows
(uploaded and started after their teacher caches reach the CTS).
  /usr/bin/python3 -I make_fill_tasks.py"""
import json
from pathlib import Path

R = Path('/home/cai_tianshun/Project/4dsr')
HERE = Path(__file__).resolve().parent
E = 'experiments/dynamic_sr_modules_20261010'
OS = 'output/dynamic_sr_modules_20261010/sps_base_20261112'
OF = 'output/dynamic_sr_modules_20261010/t3fill_20261112'
NS = 'output/dynamic_sr_modules_20261010/newscenes_reg_20261103'
ONS = f'{NS}/t3fill_20261112'
P = 'data/dynamic_sr/full_time_20261007/prepared'
TORCH_HOME = 'output/dynamic_sr_modules_20261010/bl4_20261024/torch_home'
GUARD = f'CRITICAL {E}/third_20261022/disk_guard.py --min-gib 15'
FIT = '--arm PC --target lr --steps 100 --steps-first 300 --c-reg'
SMX1 = ('Coffee07', 'Discussion07', 'Salmon07')
ST = json.loads((R / 'tmp/claude_sps_remote_20261112/pinsr_students.json').read_text())
GR = 'output/dynamic_sr_modules_20261010/geoprobe_20261017/registry'
NR = 'output/dynamic_sr_modules_20261010/newscenes_sched_20261103/sr/registry'
M = dict(Coffee07='coffee_martini', Flame07='flame_steak', Cut07='cut_roasted_beef', VRHeadset07='meetroom_vrheadset',
         Sear07='sear_steak', Salmon07='flame_salmon', Discussion07='meetroom_discussion')
REG = dict(Coffee07=ST['Coffee07']['reg'], Flame07=ST['Flame07']['reg'],
           Cut07=f'{GR}/cut_roasted_beef_area4/seed_20261007/areaft_teacher/Bsync',
           VRHeadset07=f'{GR}/meetroom_vrheadset_area4/seed_20261007/areaft_teacher/Bsync',
           Sear07=f'{NR}/sear_steak_area4/seed_20261007/areaft_teacher/Bsync', Salmon07=f'{NR}/flame_salmon_area4/seed_20261007/areaft_teacher/Bsync')
NEW = ('Sear07', 'Salmon07')


def manifest(s):
    return f'{P}/{M[s]}_area4/manifest_train_ready_seed20261007.json'


def evals(s, row, ck, resid):
    o = f'{OF}/rows/{s}/{row}'
    return [f'{E}/area4_baselines_20261024/ours/render_ours_png.py --checkpoint {ck} --resid {resid} --manifest {manifest(s)} --scope test '
            f'--out {o}/render_test',
            f'{E}/area4_baselines_20261024/score_views_area4.py --manifest {manifest(s)} --scope test --render-dir {o}/render_test '
            f'--out {o}/score_test --method t3fill_{s}_{row} --device cuda --execute --torch-home {TORCH_HOME}']


def row_task(s, row):
    base = ONS if s in NEW else OF
    arm = f'{s}_area4_t3{row}_w4'
    out = f'{base}/{row}/sr/{arm}'
    sr = (f"--parent-checkpoint {ST[s]['parent']} --baseline-sr {REG[s]} --arm {arm} --prune-gate none --affine --lambda-sr 0.2 "
          f'--dssim 0.2 --out {out}')
    if row == 'k1':
        script, args = f'{E}/third_20261022/pvcd_modkit.py', (f"--teacher A={ST[s]['sources'][0]} --weight 0.5 --dssim 0.2 --scale 4 "
                                                               f'--seed 20261014 -- --lr-operator area --warp-field 4 -- --mode fix cullretrain sr {sr}')
    else:
        script, args = f'{E}/geoprobe_20261017/modkit6.py', f'--lr-operator area --drop-rate 0.2 --warp-field 4 -- --mode fix cullretrain sr {sr}'
    if s in NEW:
        idx = f"{NS}/teacher_cache/{M[s]}_area4/teacher_index_sr_swinir_x4_areaft_seed20261007.json"
        train = (f'{E}/newscenes_reg_20261103/teacher_paths_new.py --index {idx} --script {script} '
                 f'--receipt {ONS}/receipts/{s}_{row}_teacher_paths.json -- {args}')
    else:
        train = f'{script} {args}'
    fit = f'{OF}/fit/{s}_{row}_PCv2_lr'
    ck = f'{out}/checkpoint_06000.pt'
    lines = [f'# Tab. 3 fill: {s} {row}', GUARD, f'CRITICAL {train}', f'CRITICAL {E}/absorb_20261021/perframe_fit2.py --checkpoint {ck} --out {fit} {FIT}']
    resid = fit
    if s in SMX1:
        lines.append(f'CRITICAL {E}/third_20261022/resid_temporal_variants2.py --fit {fit} --out-root {OF}/tv2/{s}_{row} --variants SMX1')
        resid = f'{OF}/tv2/{s}_{row}/SMX1'
    return lines + evals(s, row, ck, resid)


def sps_task(s):
    lanes = HERE / 'lanes'
    lines = [x for x in (lanes / f'{s}_T.txt').read_text().splitlines() + (lanes / f'{s}_E.txt').read_text().splitlines()
             if x.strip() and not x.startswith(('#', 'WAIT ', 'TOUCH '))]
    return [f'# SplatSuRe on our base configuration: {s}'] + lines


T3 = 'output/dynamic_sr_modules_20261010/third_20261022'
disc = (['# Tab. 3 fill: Discussion07 k1 and drop (existing runs, scoring only)']
        + evals('Discussion07', 'k1', f'{T3}/mk_pvcdk1s4p6/sr/Discussion07_area4_pvcdk1s4p6_w4/checkpoint_06000.pt', f'{T3}/tv2/Discussion07_pvcdk1s4p6/SMX1')
        + evals('Discussion07', 'drop', f'{T3}/mk_drop02p6/sr/Discussion07_area4_drop02p6_w4/checkpoint_06000.pt', f'{T3}/tv2/Discussion07_drop02p6/SMX1'))
local = {'10_sps_Discussion07': sps_task('Discussion07'), '15_t3_Discussion07_eval': disc,
         '20_t3_Coffee07_k1': row_task('Coffee07', 'k1'), '21_t3_Flame07_k1': row_task('Flame07', 'k1'),
         '22_t3_Coffee07_drop': row_task('Coffee07', 'drop'), '23_t3_Flame07_drop': row_task('Flame07', 'drop')}
for i, s in enumerate(('Cut07', 'Coffee07', 'VRHeadset07', 'Sear07', 'Salmon07')):
    local[f'3{i}_sps_{s}'] = sps_task(s)
cts = {f'1{i}_t3_{s}_{row}': row_task(s, row)
       for i, (s, row) in enumerate([('VRHeadset07', 'k1'), ('Cut07', 'k1'), ('VRHeadset07', 'drop'), ('Cut07', 'drop')])}
cts2 = {f'1{i}_t3_{s}_{row}': row_task(s, row)
        for i, (s, row) in enumerate([('Sear07', 'k1'), ('Salmon07', 'k1'), ('Sear07', 'drop'), ('Salmon07', 'drop')])}
for qname, tasks in (('queue_local', local), ('queue_cts', cts), ('queue_cts2', cts2)):
    pend = R / OS / qname / 'pending'
    pend.mkdir(parents=True, exist_ok=True)
    for name, lines in tasks.items():
        (pend / f'{name}.txt').write_text('\n'.join(lines) + '\n')
    print(qname, sorted(tasks))
(HERE / 'fill_tasks.json').write_text(json.dumps(dict(local=local, cts=cts, cts2=cts2), indent=1))
