"""Writes the lane file and job manifest of the mock-review controls on the RTX PRO 6000 (Claude 4dsr-a3, 2026-11-12).
Recipes are copied from tmp/claude_review_polish_20261112/recipes/RECIPES.md (ctlp6 / c3p6 / U4 / e24p6); only arm and output names
are new. Lane T trains (one process at a time); lane E only renders.  /usr/bin/python3 -I make_rc1.py"""
import json
from pathlib import Path

ROOT = Path('/home/cai_tianshun/Project/4dsr')
HERE = ROOT / 'experiments/dynamic_sr_modules_20261010/review_ctl_20261112'
R = 'output/dynamic_sr_modules_20261010/review_ctl_20261112'
E = 'experiments/dynamic_sr_modules_20261010'
PARENT = dict(Cook07='output/dynamic_sr_cull_retrain_20261009/wave1/lr/physlr_20261016/runs/Cook07_area4_gateaff/checkpoint_fine_14000.pt',
              Discussion07='output/dynamic_sr_cull_retrain_20261009/wave1/lr/physlr_20261016/runs/Discussion07_area4_gateaff/checkpoint_fine_14000.pt')
REG = dict(Cook07='output/dynamic_sr_modules_20261010/geoprobe_20261017/registry/cook_spinach_area4/seed_20261007/areaft_teacher/Bsync',
           Discussion07='output/dynamic_sr_modules_20261010/geoprobe_20261017/registry/meetroom_discussion_area4/seed_20261007/areaft_teacher/Bsync')
SR = '--prune-gate none --affine --lambda-sr 0.2 --dssim 0.2'
TEACH = ('--teacher A=output/dynamic_sr_modules_20261010/geoprobe_20261017/mk_local/sr/Cook07_area4_ch3w_w4/checkpoint_06000.pt '
         '--teacher B=output/dynamic_sr_modules_20261010/geoprobe_20261017/mk_a100/sr/Cook07_area4_mkw_wf4/checkpoint_06000.pt '
         '--teacher C=output/dynamic_sr_modules_20261010/geoprobe_20261017/mk_a100/sr/Cook07_area4_pairg1_w4/checkpoint_06000.pt')
CTLP6_W4 = 'output/dynamic_sr_modules_20261010/third_20261022/warm_20261024/same/Cook07_area4_ctlp6_w4/checkpoint_06000.pt'
GUARD = f'CRITICAL {E}/third_20261022/disk_guard.py --min-gib 15'


def sr_tail(scene, arm, out, extra=''):
    return f'-- --mode fix cullretrain sr --parent-checkpoint {PARENT[scene]} --baseline-sr {REG[scene]} --arm {arm} {SR}{extra} --out {out}'


def field(scene, g, tag):
    arm = f'{scene}_area4_rcp6_{tag}'
    wf = f' --warp-field {g}' if g else ''
    return arm, f'{E}/geoprobe_20261017/modkit4.py --lr-operator area{wf} ' + sr_tail(scene, arm, f'{R}/sr/{arm}')


def fit_eval(ck, name):
    fit, ev = f'{R}/fit/Cook07_{name}_PCv2_lr', f'{R}/eval/Cook07_{name}_PCv2_lr'
    return (f'{E}/absorb_20261021/run_fit_eval2.py --checkpoint {ck} --out {fit} --arm PC --target lr --steps 100 --steps-first 300 --c-reg '
            f'--eval --resid PC={fit} --require-all --checkpoint parent={ck} --checkpoint PC={ck} --view cam00 --out {ev}')


lines = ['# lane T: mock-review controls on the RTX PRO 6000, one process at a time (Claude 4dsr-a3, 2026-11-12; make_rc1.py)', GUARD]
smoke = 'Cook07_area4_rcsmoke_w1'
lines.append(f'{E}/geoprobe_20261017/modkit4.py --lr-operator area --warp-field 1 '
             + sr_tail('Cook07', smoke, f'{R}/smoke/{smoke}', extra=' --stop 30'))
lines.append(f'{E}/review_ctl_20261112/check_smoke_w1.py {R}/smoke/{smoke}')
for scene, grids in (('Cook07', ((0, 'nofield'), (1, 'w1'), (2, 'w2'))), ('Discussion07', ((0, 'nofield'), (1, 'w1'), (2, 'w2'), (4, 'w4')))):
    cks = []
    for g, tag in grids:
        arm, cmd = field(scene, g, tag)
        lines.append(cmd)
        cks.append(f'--checkpoint {tag}={R}/sr/{arm}/checkpoint_06000.pt')
    if scene == 'Cook07':
        cks.append(f'--checkpoint w4={CTLP6_W4}')
    lines.append(f'{E}/perframe_20261021/eval_resid.py ' + ' '.join(cks) + f' --view cam00 --out {R}/eval/{scene}_grid_cam00')
lines.append(f'TOUCH {R}/markers/RC1_GRID_DONE')

for i in (1, 2):
    lines.append(GUARD)
    pv = f'Cook07_area4_rcpv{i}_w4'
    lines.append(f'{E}/third_20261022/pvcd_modkit.py {TEACH} --scale 4 --seed 20261014 -- --lr-operator area --warp-field 4 '
                 + sr_tail('Cook07', pv, f'{R}/pvcd/{pv}'))
    lines.append(fit_eval(f'{R}/pvcd/{pv}/checkpoint_06000.pt', f'rcpv{i}'))
    cor = f'{R}/cor/Cook07_rccor{i}'
    lines.append(f'{E}/moduleaudit_20261101/u4/cotrain_modkit.py --weight 0.5 --dssim 0.2 --scale 4 --seed 20261014 --log-every 50 '
                 f'--out-b {cor}/b -- --lr-operator area --warp-field 4 ' + sr_tail('Cook07', f'Cook07_area4_rccor{i}_w4', f'{cor}/a'))
    lines.append(fit_eval(f'{cor}/a/checkpoint_06000.pt', f'rccor{i}A'))
    e24 = f'Cook07_area4_rce24n{i}_w4'
    lines.append(f'{E}/third_20261022/sched24k.py --factor 4 -- --lr-operator area --warp-field 4 ' + sr_tail('Cook07', e24, f'{R}/e24/{e24}'))
    lines.append(fit_eval(f'{R}/e24/{e24}/checkpoint_24000.pt', f'rce24n{i}'))
    lines.append(f'TOUCH {R}/markers/RC1_PVCD{i}_DONE')
lines.append(f'TOUCH {R}/markers/RC1_T_DONE')
(HERE / 'rc1_T.txt').write_text('\n'.join(lines) + '\n')

MAN = 'data/dynamic_sr/full_time_20261007/prepared/cook_spinach_area4/manifest_train_ready_seed20261007.json'
RENDER = f'{E}/area4_baselines_20261024/ours/render_ours_png.py'
SRC_A = 'output/dynamic_sr_modules_20261010/geoprobe_20261017/mk_local/sr/Cook07_area4_ch3w_w4/checkpoint_06000.pt'
lane_e = ['# lane E: Cook07 test renders for the frame-residual temporal check (render only; no training, no scoring)',
          f'{RENDER} --checkpoint output/dynamic_sr_modules_20261010/third_20261022/mk_pvcds4k3a/sr/Cook07_area4_pvcds4k3a_w4/checkpoint_06000.pt '
          f'--manifest {MAN} --scope test --out {R}/rows/Cook07/nores/render_test '
          '--check-float output/dynamic_sr_modules_20261010/third_20261022/eval/Cook07_pvcds4k3a_PCv2_lr --check-name parent',
          f'{RENDER} --checkpoint {SRC_A} --manifest {MAN} --scope test --out {R}/rows/Cook07/field/render_test',
          f'{RENDER} --checkpoint {SRC_A} --resid output/dynamic_sr_modules_20261010/third_20261022/fit/Cook07_ch3w_PCv2_lr_3090 '
          f'--manifest {MAN} --scope test --out {R}/rows/Cook07/pc/render_test',
          f'TOUCH {R}/markers/RC1_E_DONE']
(HERE / 'rc1_E.txt').write_text('\n'.join(lane_e) + '\n')

job = dict(version=1, job_id='rc1-p6-a', project_id='4dsr', role='local', operation='run', env_id='mml_bw', cwd=str(ROOT),
           argv=['/home/cai_tianshun/miniconda3/envs/mml_bw/bin/python', '-u', str(ROOT / f'{E}/motion_20261021/lanes.py'),
                 '--abort', f'{R}/ABORT_rc1-p6-a', str(HERE / 'rc1_T.txt'), str(HERE / 'rc1_E.txt')],
           gpu_uuids=['GPU-ddc2c5d8-3507-6293-837c-b1ea6b5e1cc5'], gpu_memory_mib=16384, timeout_seconds=43200)
jd = ROOT / 'tmp/claude_review_polish_20261112/jobs'
jd.mkdir(parents=True, exist_ok=True)
(jd / 'rc1-p6-a.json').write_text(json.dumps(job, indent=1) + '\n')
print(len(lines), 'lines;', jd / 'rc1-p6-a.json')
