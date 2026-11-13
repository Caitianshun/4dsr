"""Base-vs-components attribution on all 8 scenes (Claude, 2026-11-12; mock-review W1). Writes the two PRO6000 lanes and the job request.

Rows per scene (test view cam00, 300 frames, main-table renderer + evaluator):
  base   SR stage of Eq. (2) without displacement field (same LR model, teacher, schedule and code as the field run)
  field  the same SR stage with the 4x4 field (Sear07: source A; Salmon07: a new same-card pair with the new base)
  pc     field model + frame residuals; smoothing variant = the scene's held-out-camera (LOCO) decision, made on that field model
  full   PinSR as in Tab. 1 (not re-run).  Cook07 rows come from Tab. 3 (ablation_rescore.json).
Descriptive attribution only; no gate is applied to these numbers.
  /usr/bin/python3 -I make_attrib_jobs.py"""
import hashlib
import json
import os
from pathlib import Path

R = Path('/home/cai_tianshun/Project/4dsr')
E = 'experiments/dynamic_sr_modules_20261010'
A = 'output/dynamic_sr_modules_20261010/attrib_20261112'
HERE = R / E / 'attrib_20261112'
PY = '/home/cai_tianshun/miniconda3/envs/mml_bw/bin/python'
PRO6000 = 'GPU-ddc2c5d8-3507-6293-837c-b1ea6b5e1cc5'
P = 'data/dynamic_sr/full_time_20261007/prepared'
NS = 'output/dynamic_sr_modules_20261010/newscenes_reg_20261103'
SCHED = 'output/dynamic_sr_modules_20261010/newscenes_sched_20261103/sr/registry'
TORCH_HOME = 'output/dynamic_sr_modules_20261010/bl4_20261024/torch_home'
FITARGS = '--arm PC --target lr --steps 100 --steps-first 300 --c-reg'

SC = {
    'Coffee07': dict(m='coffee_martini_area4', smx=True,
                     base='output/dynamic_sr_physlr_20261016/holdout_w15/sr/Coffee07_area4_ho_areaTm/checkpoint_06000.pt',
                     field='output/dynamic_sr_physlr_20261016/holdout_w15/sr/Coffee07_area4_ho_w4/checkpoint_06000.pt',
                     fit='output/dynamic_sr_modules_20261010/perframe_20261021/fit/Coffee07_PCv2_lr'),
    'Cut07': dict(m='cut_roasted_beef_area4', smx=False,
                  base='output/dynamic_sr_physlr_20261016/sr/Cut07_area4_areaTm/checkpoint_06000.pt',
                  field='output/dynamic_sr_modules_20261010/geoprobe_20261017/mk_local/sr/Cut07_area4_mkfo_w4/checkpoint_06000.pt',
                  fit=f'{A}/fit/Cut07_mkfo_PCv2_lr', new_fit=True),
    'Flame07': dict(m='flame_steak_area4', smx=False,
                    base='output/dynamic_sr_physlr_20261016/holdout_w15/sr/Flame07_area4_ho_areaTm/checkpoint_06000.pt',
                    field='output/dynamic_sr_physlr_20261016/holdout_w15/sr/Flame07_area4_ho_w4/checkpoint_06000.pt',
                    fit='output/dynamic_sr_modules_20261010/perframe_20261021/fit/Flame07_PCv2_lr'),
    'Discussion07': dict(m='meetroom_discussion_area4', smx=True,
                         base='output/dynamic_sr_physlr_20261016/sr/Discussion07_area4_areaTm/checkpoint_06000.pt',
                         field='output/dynamic_sr_modules_20261010/geoprobe_20261017/mk_local/sr/Discussion07_area4_ch3w_w4/checkpoint_06000.pt',
                         fit='output/dynamic_sr_modules_20261010/perframe_20261021/fit/Discussion07_PCv2_lr'),
    'VRHeadset07': dict(m='meetroom_vrheadset_area4', smx=False,
                        base='output/dynamic_sr_modules_20261010/geoprobe_20261017/mk_local/sr/VRHeadset07_area4_f2mr12_tm/checkpoint_06000.pt',
                        field='output/dynamic_sr_modules_20261010/geoprobe_20261017/mk_local/sr/VRHeadset07_area4_f2mr12_w4/checkpoint_06000.pt',
                        fit='output/dynamic_sr_modules_20261010/perframe_20261021/fit/VRHeadset07_PCv2_lr'),
    'Sear07': dict(m='sear_steak_area4', smx=False, new_base=True, new_fit=True,
                   base=f'{A}/sear/sr/Sear07_area4_at_areaTm/checkpoint_06000.pt',
                   field=f'{NS}/sources/A/sr/Sear07_area4_ho_w4/checkpoint_06000.pt',
                   fit=f'{A}/fit/Sear07_hoA_PCv2_lr'),
    'Salmon07': dict(m='flame_salmon_area4', smx=True, new_base=True, new_field=True, new_fit=True,
                     base=f'{A}/pair/sr/Salmon07_area4_at_areaTm/checkpoint_06000.pt',
                     field=f'{A}/pair/sr/Salmon07_area4_at_w4/checkpoint_06000.pt',
                     fit=f'{A}/fit/Salmon07_atw4_PCv2_lr'),
}
LOCO = {'Coffee07': f'{E[:0]}output/dynamic_sr_modules_20261010/third_20261022/eval/Coffee07_loco10/loco.json',
        'Cut07': 'output/dynamic_sr_modules_20261010/third_20261022/eval/Cut07_loco10/loco.json',
        'Flame07': 'output/dynamic_sr_modules_20261010/third_20261022/eval/Flame07_loco10b/loco.json',
        'Discussion07': 'output/dynamic_sr_modules_20261010/third_20261022/eval/Disc07_loco10/loco.json',
        'VRHeadset07': 'output/dynamic_sr_modules_20261010/third_20261022/eval/VRH07_loco10b/loco.json',
        'Sear07': f'{NS}/loco/eval/Sear07/loco.json', 'Salmon07': f'{NS}/loco/eval/Salmon07/loco.json'}


def sha(p):
    h = hashlib.sha256()
    with open(R / p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def run_meta(ck):
    d = (R / ck).parent
    r = json.loads((d / 'run.json').read_text())
    a = r['args']
    fix = json.loads((d / 'srfix.json').read_text())['mode']
    return dict(parent=r['parent_checkpoint']['sha256'], teacher=r['teacher_index']['sha256'], schedule=r['schedule']['sha256'],
                code=r['sources'], lam=a['lambda_sr'], dssim=a['dssim'], affine=a['affine'], gate=a['prune_gate'], srfix=fix,
                gpu=r.get('gpu'), host=r.get('host'))


inputs = {}
for s, c in SC.items():
    dec = json.loads((R / LOCO[s]).read_text())['decision']['choose']
    assert (dec == 'SMX1') == c['smx'], (s, dec)
    rec = dict(manifest=f"{P}/{c['m']}/manifest_train_ready_seed20261007.json", loco=LOCO[s], loco_choice=dec)
    rec['manifest_sha256'] = sha(rec['manifest'])
    if not c.get('new_field'):
        rec['field'] = dict(path=c['field'], sha256=sha(c['field']), **run_meta(c['field']))
    if not c.get('new_base'):
        rec['base'] = dict(path=c['base'], sha256=sha(c['base']), **run_meta(c['base']))
        b, f = rec['base'], rec['field']
        for k in ('parent', 'teacher', 'schedule', 'code', 'lam', 'dssim', 'affine', 'gate', 'srfix'):
            assert b[k] == f[k], (s, k, b[k], f[k])
    if not c.get('new_fit'):
        fr = json.loads((R / c['fit'] / 'run.json').read_text())
        assert fr['checkpoint']['path'] == c['field'] and fr['checkpoint']['sha256'] == rec['field']['sha256'], s
        code = {Path(k).name: v for k, v in fr['sources_code'].items()}
        assert any(v.startswith('778995a3') for v in code.values()), (s, code)
        assert len(list((R / c['fit']).glob('f*.npz'))) == 300 and (R / c['fit'] / 'complete.json').exists(), s
        rec['fit'] = dict(path=c['fit'], code=code, args=fr['args'])
    inputs[s] = rec

# the new runs reuse the exact launch of source A (teacher-path adapter + run_modkit7_remote.py, area op, srfix, affine, gate none)
for s, mname in (('Sear07', 'sear_steak'), ('Salmon07', 'flame_salmon')):
    for p in (f'{NS}/lr/runs/{s}_area4_gateaff/checkpoint_fine_14000.pt',
              f'{SCHED}/{mname}_area4/seed_20261007/areaft_teacher/Bsync',
              f'{NS}/teacher_cache/{mname}_area4/teacher_index_sr_swinir_x4_areaft_seed20261007.json'):
        assert (R / p).exists(), p
assert run_meta(SC['Sear07']['field'])['gpu'].startswith('NVIDIA RTX PRO 6000')


def sr_line(s, mname, out_root, arms):
    return (f'{E}/newscenes_reg_20261103/teacher_paths_new.py --index {NS}/teacher_cache/{mname}_area4/teacher_index_sr_swinir_x4_areaft_seed20261007.json '
            f'--script {E}/geoprobe_20261017/run_modkit7_remote.py --receipt {A}/receipts/{s}_{Path(out_root).name}_teacher_paths.json -- '
            f'--parent {NS}/lr/runs/{s}_area4_gateaff/checkpoint_fine_14000.pt '
            f'--baseline-sr {SCHED}/{mname}_area4/seed_20261007/areaft_teacher/Bsync --lambda-sr 0.2 --out-root {out_root} '
            f'--tag {s}_area4_at --view cam00 ' + ' '.join(f'--arm {x}' for x in arms))


def fit_line(ck, out):
    return f'{E}/absorb_20261021/perframe_fit2.py --checkpoint {ck} --out {out} {FITARGS}'


def smx_line(fit, out_root):
    return f'{E}/third_20261022/resid_temporal_variants2.py --fit {fit} --out-root {out_root} --variants SMX1'


def resid_of(s):
    c = SC[s]
    return f"{A}/tv2/{s}_pc/SMX1" if c['smx'] else c['fit']


def row_lines(s, row):
    c = SC[s]
    m = f"{P}/{c['m']}/manifest_train_ready_seed20261007.json"
    ck = c['base'] if row == 'base' else c['field']
    resid = f' --resid {resid_of(s)}' if row == 'pc' else ''
    o = f'{A}/rows/{s}/{row}'
    return [f'{E}/area4_baselines_20261024/ours/render_ours_png.py --checkpoint {ck}{resid} --manifest {m} --scope test --out {o}/render_test',
            f'{E}/area4_baselines_20261024/score_views_area4.py --manifest {m} --scope test --render-dir {o}/render_test --out {o}/score_test '
            f'--method attr_{s}_{row} --device cuda --execute --torch-home {TORCH_HOME}']


GUARD = f'CRITICAL {E}/third_20261022/disk_guard.py --min-gib 15'
mk = f'{A}/markers'
T = ['# lane T: training on the PRO6000, one training process at a time', GUARD,
     sr_line('Sear07', 'sear_steak', f'{A}/sear', ['areaTm=']), f'TOUCH {mk}/T1_sear_base.attempted', GUARD,
     sr_line('Salmon07', 'flame_salmon', f'{A}/pair', ['w4=wfield:4', 'areaTm=']), f'TOUCH {mk}/T2_salmon_pair.attempted', GUARD,
     fit_line(SC['Salmon07']['field'], SC['Salmon07']['fit']), smx_line(SC['Salmon07']['fit'], f'{A}/tv2/Salmon07_pc'),
     f'TOUCH {mk}/T3_salmon_fit.attempted', GUARD,
     fit_line(SC['Cut07']['field'], SC['Cut07']['fit']), f'TOUCH {mk}/T4_cut_fit.attempted', GUARD,
     fit_line(SC['Sear07']['field'], SC['Sear07']['fit']), f'TOUCH {mk}/T5_sear_fit.attempted', f'TOUCH {mk}/laneT.done']
Ev = ['# lane E: renders and scores (evaluation only)', GUARD,
      smx_line(SC['Coffee07']['fit'], f'{A}/tv2/Coffee07_pc'), smx_line(SC['Discussion07']['fit'], f'{A}/tv2/Discussion07_pc')]
for s in ('Coffee07', 'Discussion07', 'Flame07', 'VRHeadset07'):
    for row in ('base', 'field', 'pc'):
        Ev += row_lines(s, row)
Ev += row_lines('Cut07', 'base') + row_lines('Cut07', 'field') + row_lines('Sear07', 'field')
Ev += [f'WAIT {mk}/T1_sear_base.attempted'] + row_lines('Sear07', 'base')
Ev += [f'WAIT {mk}/T2_salmon_pair.attempted'] + row_lines('Salmon07', 'field') + row_lines('Salmon07', 'base')
Ev += [f'WAIT {mk}/T3_salmon_fit.attempted'] + row_lines('Salmon07', 'pc')
Ev += [f'WAIT {mk}/T4_cut_fit.attempted'] + row_lines('Cut07', 'pc')
Ev += [f'WAIT {mk}/T5_sear_fit.attempted'] + row_lines('Sear07', 'pc') + [f'TOUCH {mk}/laneE.done']

(HERE / 'lane_T.txt').write_text('\n'.join(T) + '\n')
(HERE / 'lane_E.txt').write_text('\n'.join(Ev) + '\n')
(HERE / 'inputs.json').write_text(json.dumps(dict(scenes=SC, verified=inputs), indent=1))
job = dict(version=1, job_id='attr-p6-01', project_id='4dsr', role='local', operation='run', env_id='mml_bw', cwd=str(R),
           argv=[PY, '-u', str(R / E / 'static_sr_v2_20261026/sv2_lanes.py'), '--abort', f'{A}/ABORT',
                 str(HERE / 'lane_T.txt'), str(HERE / 'lane_E.txt')],
           gpu_uuids=[PRO6000], gpu_memory_mib=40000, timeout_seconds=8 * 3600)
(HERE / 'req').mkdir(exist_ok=True)
(HERE / 'req/attr-p6-01.json').write_text(json.dumps(job, indent=1))
for p in ('lane_T.txt', 'lane_E.txt', 'inputs.json', 'req/attr-p6-01.json'):
    os.chmod(HERE / p, 0o640)
print('lines T', len(T), 'E', len(Ev))
print(json.dumps({s: {k: (v.get('gpu') if isinstance(v, dict) else v) for k, v in r.items() if k in ('base', 'field', 'loco_choice')}
                  for s, r in inputs.items()}, indent=0))
