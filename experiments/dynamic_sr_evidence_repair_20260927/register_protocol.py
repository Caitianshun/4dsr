"""Register the bounded evidence-repair protocol before any new measurement.

Requires the prepared generic scene and immutable historical prior-guidance inputs.
Refuses to overwrite an existing registration. GPU availability is rechecked by
execution controllers, not inferred from this registration.
"""
import argparse
import shutil
import time
from context import *

DEFAULTS = {'gates': {'early_psnr_drop': 0.2,
           'early_dynamic_relative': 0.05,
           'early_temporal_relative': 0.05,
           'cam00_psnr_drop': 0.2,
           'cam00_dynamic_relative': 0.05,
           'cam00_temporal_relative': 0.05,
           'cam01_psnr_drop': 0.2,
           'cam01_lpips_relative': 0.05,
           'signal_psnr_gain': 0.2,
           'signal_psnr_tolerance': 0.1,
           'signal_dynamic_gain': 0.05,
           'signal_temporal_tolerance': 0.02,
           'signal_dynamic_tolerance': 0.02,
           'u6_psnr_tolerance': 0.1,
           'u6_dynamic_gain': 0.03,
           'u6_temporal_tolerance': 0.02},
 'schema': 1,
 'status': 'registered_before_new_measurements',
 'frames': [0, 40, 80, 118],
 'train_cameras': ['cam02',
                   'cam03',
                   'cam04',
                   'cam05',
                   'cam06',
                   'cam07',
                   'cam08',
                   'cam09',
                   'cam10',
                   'cam11',
                   'cam12',
                   'cam13',
                   'cam14',
                   'cam15',
                   'cam16',
                   'cam17',
                   'cam18',
                   'cam19',
                   'cam20'],
 'probe_cameras': ['cam02', 'cam06', 'cam12', 'cam18'],
 'parity_frames': [40, 80],
 'lr_offset': 7200,
 'sr_weight': 0.1,
 'renderer_choices': ['legacy_direct_v1', 'prior_refactor_v1'],
 'parity': {'repeat_multiplier': 3.0,
            'ulp_multiplier': 8.0,
            'absolute_floor': 1e-10,
            'none_masks_exact': True,
            'tolerance_registration': 'For each named tensor '
                                      'max(3*max(old_repeat,new_repeat),8*float32_eps*reference_maxabs,1e-10); '
                                      'register repeat envelope before computing cross-implementation '
                                      'difference; never widen after failure'},
 'replay': {'tails': 2,
            'tail_start': 17550,
            'tail_stop': 18000,
            'save_steps': [17600, 17700, 17850, 18000],
            'long_max_updates': 30450,
            'sensitivity_psnr_db': 0.05,
            'sensitivity_relative': 0.01},
 'geometry': {'sift_nfeatures': 2500,
              'sift_contrast': 0.012,
              'sift_edge': 12,
              'ratio': 0.8,
              'ransac_px': 1.5,
              'ransac_confidence': 0.999,
              'known_sampson_px': 1.5,
              'source_reprojection_px': 0.5,
              'angle_deg': 2.0,
              'third_reprojection_px': 1.0,
              'perturb_px': 0.5,
              'track_selection': 'Sorted pair identities; earliest valid source edge with a mutually matched '
                                 'triangle; one source pair per conflict-free connected track; held-out '
                                 'third view never fitted',
              'track_conflict': 'Reject connected components containing multiple feature IDs in one camera',
              'point_pair_distance': [4.0, 40.0],
              'relative_depth_difference': 0.03,
              'min_tracks': 10,
              'min_pairs': 30,
              'wrong_shift': 48},
 'moments': {'alpha_epsilon': 1e-06,
             'alpha_valid': 1e-06,
             'variance_relative_tolerance': 2e-06,
             'variance_absolute_tolerance': 1e-08,
             'area_factor': 4,
             'reference_alpha': 0.9,
             'reference_var_ratio': 0.02},
 'depth': {'patch_size': 16,
           'patch_stride': 16,
           'border': 16,
           'patches_per_observation': 247,
           'stability': 0.9,
           'LR_gradient_threshold': 0.08,
           'LR_interior_fraction': 0.5,
           'min_support_points': 2,
           'model_valid_fraction': 0.5,
           'coverage': 0.2,
           'min_cameras': 12,
           'min_geometry_cameras': 3,
           'geometry_spaced_cameras': ['cam02', 'cam06', 'cam12', 'cam18'],
           'order_accuracy': 0.8,
           'wrong_margin': 0.1,
           'min_temporal_clips': 3,
           'W_radius': 4,
           'W_patch_fraction': 0.5,
           'min_nonzero_training_steps': 500},
 'fixed_time': {'frame': 40,
                'steps': 1200,
                'save_steps': [600, 1200],
                'seed': 20260927,
                'zero_adam': True,
                'disable_regularization': True,
                'loss': 'LR L1 + 0.1 teacher RGB L1',
                'teacher_highpass_gain': 0.1,
                'lr_harm_relative': 0.02},
 'max_major_updates': 63750,
 'information_boundary': 'Training uses cam02..20 LR and frozen SwinIR only; HR and cam00/01 '
                         'evaluation-only. No camera roles changed; no LPL search.',
 'phase_status': {'P0': 'pending', 'P1': 'pending', 'P2': 'pending', 'P3': 'pending'}}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--gpu',required=True);a=parser.parse_args()
    assert not (OUT/'protocol.json').exists(),'Never overwrite a preregistration'
    p=paths();config=dict(DEFAULTS);config.update(start=str(p['start']),start_sha256=sha256(p['start']),physical_gpu=a.gpu,registered_unix=time.time(),adopted_plan_sha256=sha256(a.plan))
    for k in ['manifest','teacher','schedule','lr_curve']:config[k]=dict(path=str(p[k]),sha256=sha256(p[k]))
    m=load_manifest(p['manifest']);entries=read(p['teacher'])['entries'];assert len(entries)==1140
    for e in entries:assert sha256(Path(m['_root'])/e['relative_path'])==e['sha256']
    config['teacher'].update(count=len(entries),all_image_hashes_checked=True);config['schedule']['steps']=read(p['schedule'])['steps'];assert config['schedule']['steps']==40000
    OUT.mkdir(parents=True,exist_ok=True);shutil.copyfile(a.plan,OUT/'adopted_plan.codex.md');write_json(OUT/'protocol.json',config)

if __name__=='__main__':main()
