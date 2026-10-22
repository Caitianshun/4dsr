"""Register train-only inputs and explicit Wu adaptation recipes without using CUDA."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from common import ROOT, SCHEMA, METHODS, identity, read, records, training_manifest, write_new

METHOD_SPEC = {
    'splatsure': dict(stages=['lr', 'splatsure_cache', 'sr'], lr_steps=30000, sr_steps=30000,
        init_sr='fresh', dependencies=['StableSR main teacher recipe/weights and training-view SR cache (SwinIR only as labeled variant)',
        'official SplatSuRe rasterizer compiled separately', 'frozen LR Wu projected contribution cache']),
    'iesrgs': dict(stages=['internal', 'iesrgs_cache', 'sr'], lr_steps=30000, sr_steps=30000,
        init_sr='fresh', dependencies=['SwinIR and Depth Anything V2 checkpoints and caches',
        'official IE-SRGS RADE rasterizer compiled separately', 'internal Wu Mip RGB and median-depth cache']),
    'sequence_matters': dict(stages=['sequence_cache', 'sr'], lr_steps=0, sr_steps=30000,
        init_sr='fresh', dependencies=['official PSRT checkpoint', 'same-time ALS cross-view sequence SR cache']),
    'threedsr': dict(stages=['lr', 'diffusion_rounds'], lr_steps=30000, sr_steps=20000,
        init_sr='lr_parent', dependencies=['StableSR-Turbo and VQGAN-CFW checkpoints',
        'per-view per-frame latent-state diffusion worker', 'four shared-Wu optimize/render feedback rounds']),
}


def build(manifest, source_plan, method, verify_pixels=False):
    m, rows = training_manifest(manifest, verify_pixels)
    source = read(source_plan)
    if source['scene'] != m['scene']:
        raise ValueError('Wu configuration is from a different scene')
    if source['data']['manifest'] != identity(manifest):
        raise ValueError('Wu configuration plan is bound to different inputs')
    return dict(schema=SCHEMA, method=method, label='Wu + ' + method + ' (dynamic adaptation)',
        manifest=identity(manifest), scene=m['scene'], seed=20261007,
        source_configuration=identity(source_plan), wu_configuration=source['configuration'],
        cameras_extent=source['data']['cameras_extent'], specification=METHOD_SPEC[method],
        initialization='fresh_scene_local_train_LR_points',
        training=dict(steps=METHOD_SPEC[method]['sr_steps'], batch_views=1, internal_batch_views=3 if method == 'iesrgs' else None,
            stage='fine', lr_clock='stage_local_30k', topology='source_3DGS_schedule_on_Wu_clone_split',
            final_step_topology=False, optimizer_updates='source_final_iteration_skips_step', coarse_warmup_steps=0),
        degradation='area4', teacher_variant='stablesr' if method == 'splatsure' else method,
        source_native_degradation='bicubic for SplatSuRe and SequenceMatters; see source audit',
        dynamic_adaptations=[
            'Static 3DGS replaced by one Wu 843d5ac model shared over all original_frame_index/300 times',
            'Wu native parameter groups and time-plane regularization retained; clone/split uses source 500/100/15000 schedule',
            'Source method loss and view grouping retained; full same-time camera groups only',
            'area4 is an explicitly matched observation-operator adaptation; native bicubic is separate',
            'Fresh Wu fine optimization starts directly, without silently importing a trained project parent',
            'Source final iteration skips optimizer.step; 30000 iterations mean 29999 parameter updates'],
        excluded_project_modules=['colour_affine', 'w4', 'culling', 'MVFB', 'cross_scene_learned_prior'],
        information=dict(train_cameras=m['splits']['train'], heldout_cameras=m['splits']['test'],
            train_observations=len(rows), frames=m['frame_indices'], training_HR_read=False,
            heldout_pixels_read=False, source_model_scene_transfer=False),
        prior_bundle=None, checkpoint=None, readiness='registered_CPU_only_GPU_acceptance_pending')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True)
    p.add_argument('--source-plan', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--verify-lr-bytes', action='store_true')
    p.add_argument('--splatsure-teacher', choices=['stablesr','swinir'], default='stablesr')
    a = p.parse_args()
    m, rows = training_manifest(a.manifest, a.verify_lr_bytes)
    out = Path(a.out)
    for method in METHODS:
        config = build(a.manifest, a.source_plan, method, False)
        if method == 'splatsure':
            config['teacher_variant'] = a.splatsure_teacher
            config['label'] += ' / ' + a.splatsure_teacher
        write_new(out / (method + '.json'), config)
    write_new(out / 'training_records.json', dict(schema='static_sr_wu_train_records_v1',
        manifest=identity(a.manifest), records=records(m, rows), heldout_cameras=m['splits']['test']))
    write_new(out / 'registration.json', dict(status='registered_CPU_only', manifest=identity(a.manifest),
        training_observations=len(rows), LR_bytes_verified=a.verify_lr_bytes,
        HR_images_read=0, heldout_images_read=0, GPU_calls=0, parameter_updates=0,
        requirements_before_training={k:v['dependencies'] for k,v in METHOD_SPEC.items()}))
    print(json.dumps(dict(status='registered_CPU_only', out=str(out), records=len(rows))))

if __name__ == '__main__':
    main()
