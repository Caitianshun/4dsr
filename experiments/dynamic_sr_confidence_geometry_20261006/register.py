"""Freeze scope and provenance before any formal optimizer updates."""
import copy,shutil
from cg_common import *
from schedules import generate
def main():
    OUT.mkdir(parents=True,exist_ok=True);assert not (OUT/'protocol.json').exists()
    old=read(ROOT/'output/dynamic_sr_sync_multiview_20260928/protocol.json')
    keys=read(bound(old['old_schedule']))['record_keys']
    ss={}
    for repeat,seed in [('1',2026100601),('2',2026100602)]:
        path=OUT/f'schedule_{repeat}.json';write(path,generate(keys,seed));ss[repeat]=entry(path)
    arms={'C1':'same rendered HR image: LR L1 + .1 complete SR L1 + inherited reg',
      'Jperm':'LR image + one permuted SR image .1, accumulate before both existing Adams',
      'B2perm':'LR image + two permuted SR images .05+.05, accumulate before both Adams',
      'R':'C1; .1*k_R*mean(c_view*c_LR*abs(Q(render-teacher))) replaces complete SR',
      'G':'C1 + fixed lambda_G (500-step ramp) relative-depth rank prior',
      'RG':'R + G; identical frozen coefficients/caches',
      'P_joint':'500 LR-only updates all original parameters',
      'P_xyz':'500 LR-only updates canonical xyz and children offset only; frozen Adams do not advance',
      'P_SH':'500 LR-only updates direct base/children SH only; frozen Adams do not advance',
      'RG_time':'RG with two temporal evidence edges', 'RG_ST':'RG with one view and one time evidence edge',
      'RG_mass':'RG with per-observation mean ST confidence broadcast',
      'G_FullSR_ST':'G and complete SR residual weighted by ST confidence',
      'G_Q_uniform':'G and Q residual with uniform confidence'}
    p={k:copy.deepcopy(old[k]) for k in ['parent','manifest','teacher','old_schedule','lr_curve','roi','training_files','evaluation_files','evaluation']}
    p.update(run_id=HERE.name,run_root=str(OUT.relative_to(ROOT)),status='registered',arms=arms,schedules=ss,
      parent_prefix='Validate native 6000 sampler draws and two Adam/RNG values; new independently registered suffix',
      training=dict(parent_step=6000,suffix_updates=6000,lr_offset=7200,points=132972,sr_weight=.1,
        save_suffix=[500,3000,6000],density='fixed',repeats=2,whole_scene=True,original_HR_training=False),
      budget=dict(core_updates=72000,cause_probe_updates=1500,conditional_mechanism_updates=60000,
        optional_SV_updates=24000,total_max=157500,core_RGB=108000,core_moments=24000,
        no_weight_grid=True,failures_and_replays_counted=True),
      core_task_plan=[dict(method=arm,repeat=r,updates=6000) for r in ['1','2'] for arm in ['C1','Jperm','B2perm','R','G','RG']],
      source_attachment=dict(path=str(OUT/'attachment_source.codex.md'),sha256=sha(OUT/'attachment_source.codex.md')),
      previous_execution_logs=[entry(ROOT/'docs/dynamic_sr_frequency_execution_log_2026-09-30.md'),entry(ROOT/'docs/dynamic_sr_same_observation_execution_log_2026-09-30.md')],
      cache_rules=dict(frozen=True,observations=1140,HR_derived_training=False,dev_camera_training=False,
        unknown_view=.5,evidence_max=2,calibration_train76=[0,40,80,118]),
      modules=dict(R_actual_linear_bicubic_nullspace=True,G_nonnegative_area_moments=True,
        k_R_fixed_train32=True,lambda_G_LR_xyz_RMS_fraction=.1,G_ramp_steps=500),
      scope='Core six arms and finite causes first; conditional mechanisms only after complete quality review. Full benchmark requires separate frozen protocol.')
    for e in p['training_files']+p['evaluation_files']:bound(e)
    for k in ['parent','manifest','teacher','old_schedule','lr_curve','roi']:bound(p[k])
    write(OUT/'protocol.json',p)
    write(OUT/'state.json',dict(status='registered_implementing_and_verifying',protocol=entry(OUT/'protocol.json')))
    print(json.dumps(dict(protocol=entry(OUT/'protocol.json'),schedules=ss)))
if __name__=='__main__':main()
