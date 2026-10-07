"""Register the fixed six-arm experiment without changing historical assets.

The calibration is a separately immutable prerequisite. Its coefficients are
bound by each consuming run; they are never chosen from development scores.
"""
import argparse
from fp_common import ROOT,HERE,OUT,read,write,entry,bound

METHODS=('B0','Bsync','M','X','MX','E')
ARMS={
 'B0':dict(schedule='random_rows',lr_weights=[1.,0.,0.],sr_weights=[0.,.05,.05],X=False,E=False),
 'Bsync':dict(schedule='rows',lr_weights=[1.,0.,0.],sr_weights=[0.,.05,.05],X=False,E=False),
 'M':dict(schedule='rows',lr_weights=[1/3,1/3,1/3],sr_weights=[0.,.05,.05],X=False,E=False),
 'X':dict(schedule='rows',lr_weights=[1.,0.,0.],sr_weights=[0.,.05,.05],X=True,E=False),
 'MX':dict(schedule='rows',lr_weights=[1/3,1/3,1/3],sr_weights=[0.,.05,.05],X=True,E=False),
 'E':dict(schedule='random_rows',lr_weights=[1.,0.,0.],sr_weights=[0.,.05,.05],X=False,E=True),
}

def register():
 old=ROOT/'output/dynamic_sr_confidence_geometry_20261006/protocol.json'
 p=read(old)
 out={k:p[k] for k in ('parent','manifest','teacher','old_schedule','lr_curve','roi','training_files','evaluation_files','evaluation')}
 for k in ('parent','manifest','teacher','old_schedule','lr_curve'):
  bound(out[k])
 assert len(out['training_files'])==2280 and len(out['evaluation_files'])==392
 out.update(run_id=HERE.name,run_root=str(OUT.relative_to(ROOT)),status='registered_fixed_core_requires_new_operator_support_calibration_checks',
  source_attachment=entry(OUT/'attachment_source.codex.md'),previous_protocol=entry(old),
  previous_final_delivery=entry(ROOT/'output/dynamic_sr_confidence_geometry_20261006/final_delivery_manifest.json'),
  previous_execution_log=entry(ROOT/'docs/dynamic_sr_confidence_geometry_execution_log_2026-10-06.md'),
  base_publication_commit='1bc6ac2d82dec320e023ffd4221406742fbf5268',
  arms=ARMS,
  training=dict(parent_step=6000,suffix_updates=6000,lr_offset=7200,points=132972,repeats=2,
    save_suffix=[100,3000,6000],sr_weight=.1,density='fixed',whole_scene=True,original_HR_training=False,
    optimizer_calls_per_update=2,optimizer_parameters_disjoint=True,regularizer_evaluations_per_update=1,
    same_state_rgb_per_update=3,auxiliary_moments_per_X_update=3),
  budget=dict(core_updates=72000,core_RGB=216000,core_moments=72000,adam_calls=144000,
    first100_are_formal_updates=True,preparation_calibration_evaluation_separate=True,no_weight_grid=True),
  schedules={str(i):dict(path=str((OUT/'schedules'/f'schedule_{i}.json').relative_to(ROOT)),
    frozen_hash_record='schedules/index.json; consumed schedule hash saved in each run') for i in (1,2)},
  calibration=dict(path=str((OUT/'calibration.json').relative_to(ROOT)),training_cameras=[f'cam{i:02d}' for i in range(2,21)],
    frames=[0,40,80,118],observations=76,lambda_X_xyz_RMS_fraction=.1,E_L1_weight=.8,E_MSE_weight=.2,
    tau_z_quantile=.75,tau_z_floor=1e-6,robust_delta=.01,ramp_steps=500,one_frozen_coefficient_for_X_and_MX=True,
    no_HR_or_development_input=True),
  support_cache=dict(parent='same complete U6000',moment_grid='native HR; no LR depth upsampling',
    soft_weight='single frozen .25+.75/(1+c_z/tau_z); unknown variance neutral 1',
    directed_edges_per_triplet=6,empty_edge_loss=0,outer_normalizer_always=6,
    no_photo_Census_FB_threshold_mask=True,freeze_parent_weights_and_visibility=True),
  hardware_policy='All six arms within each suffix use the same GPU model/platform/software; paired suffix reporting; protect other jobs.',
  selection=dict(allow_metric_tradeoffs=True,no_fixed_PSNR_threshold=True,no_all_three_or_two_of_three_gate=True,
    Pareto_and_paired_changes=True,maximum_development_candidates=2),
  full_scene_followthrough=dict(prepare_in_parallel=True,development_scenes_previously_used=True,
    confirmation_scene_minimum=2,configuration_frozen_before_confirmation=True,
    independent_seed_prefixes=True,official_protocol_requires_verification=True),
  scope='New M×X modular six-arm experiment plus E; no restart of old RG conditional queue. Short window identifies modules; full-time development and unused full-scene confirmation follow recommendation.')
 target=OUT/'protocol.json'
 if target.exists():
  assert read(target)==out,'Registered protocol changed; preserve prior registration and explain amendment before use'
 else:write(target,out)
 return out

if __name__=='__main__':
 argparse.ArgumentParser(description=__doc__).parse_args()
 p=register();print(dict(status=p['status'],arms=list(p['arms']),formal_updates=p['budget']['core_updates']))
