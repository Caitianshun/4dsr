"""Lock actual private assets, historical controls, and the immutable P1 protocol."""
import os
import platform
import subprocess
import time
import torch
from shared import *
from freeze_policy import apply_policy, frozen_state
from run_experiment import load_training

def main():
    torch.set_num_threads(4)
    p=paths();out=OUT/'p0';out.mkdir(parents=True,exist_ok=True)
    m=load_manifest(p['manifest']);schedule=read(p['schedule'])
    records,cameras=load_training(m)
    assert len(records)==1140 and {r['camera_id'] for r in records}=={f'cam{x:02d}' for x in range(2,21)}
    assert all(r['time']==r['frame_index']/300 for r in records)
    assert schedule['record_keys']==[[r['camera_id'],r['frame_index']] for r in records]
    teacher_paths,teacher_inputs=load_teacher_index(p['teacher'],m,records)
    endpoints={}
    for label in ['U6000','U18000','U40000']:
        entry=p['methods'][label];cp=Path(entry['checkpoint']);model=load_model(cp,m)
        ck=model.checkpoint;assert ck['metadata']['intervention_step']==int(label[1:])
        assert ck['metadata']['manifest_sha']==sha256(p['manifest'])
        original=read(cp.parent/'config.json');sources=[]
        for source in original['sources']:
            path=Path(source['path'])
            actual=sha256(path);assert actual==source['sha256'],str(path)
            sources.append(dict(path=str(path),sha256=actual))
        endpoints[label]=dict(path=str(cp),sha256=sha256(cp),identity=initial_identity(model),
            hidden=ck['hidden'],optim=ck['optim'],metadata=ck['metadata'],topology=topology(model),
            flags={n:getattr(model.h,n,None) for n in ['no_dx','no_dr','no_ds','no_do','no_dshs']},
            aabb=[v.detach().cpu().tolist() for v in model.g._deformation.deformation_net.get_aabb],
            source_identity=sources,training_gpu=original['gpu'])
        if label=='U6000':
            policy=apply_policy(model,'F_app')
            endpoints[label]['freeze_policy']=policy
            endpoints[label]['frozen_state']=frozen_state(model)
        del model
    assert len({json.dumps(v['topology'],sort_keys=True) for v in endpoints.values()})==1
    previous=read(ROOT/'output/dynamic_sr_controlled_headroom_20260926/scientific_decision.json')
    assert previous['status']=='completed_no_expansion'
    assets=dict(status='passed',manifest=dict(path=str(p['manifest']),sha256=sha256(p['manifest'])),
        teacher=dict(path=str(p['teacher']),sha256=sha256(p['teacher']),count=len(teacher_inputs),all_image_hashes_checked=True),
        schedule=dict(path=str(p['schedule']),sha256=sha256(p['schedule']),steps=schedule['steps']),
        lr_curve=dict(path=str(p['lr_curve']),sha256=sha256(p['lr_curve'])),endpoints=endpoints,
        resolutions=m['resolutions'],time='frame/300',lr_parent_fine_steps=7200,
        input_range='uint8 PNG /255; HR center crop [y3:1011,x4:1348] of 1352x1014 resize; fixed calibrated pixel centers',
        rendering_range='RGB losses use raw float; D bicubic antialias then clamp; full RGB eval clamp before metrics',
        environment=dict(torch=str(torch.__version__),cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),
            visible_cuda=os.environ.get('CUDA_VISIBLE_DEVICES'),python=platform.python_version()),
        upstream_commit=subprocess.check_output(['git','-C',str(UPSTREAM),'rev-parse','HEAD'],text=True).strip(),
        upstream_patch_sha256=__import__('hashlib').sha256(subprocess.check_output(['git','-C',str(UPSTREAM),'diff','HEAD'])).hexdigest(),
        history=dict(previous_decision=previous['status'],equivalent_control_found=False,
            explanation='9/20 appearance_only used unsplit base and four-camera teacher, not this all-camera U6000/child topology/suffix; negative detail/temporal evidence retained',
            report=str(ROOT/'docs/dynamic_sr_continuation_findings_2026-09-20.md')),
        user_document_sha256=sha256(OUT/'user_protocol.md'))
    write_json(OUT/'assets.json',assets)
    protocol=dict(schema=1,status='registered_before_any_update',created_unix=time.time(),
        arms=['C_joint','F_app'],start=endpoints['U6000']['path'],start_sha256=endpoints['U6000']['sha256'],
        manifest=assets['manifest'],teacher=assets['teacher'],schedule=assets['schedule'],lr_curve=assets['lr_curve'],
        endpoints=[18000,40000],initial_sr_updates=6000,stage1_new_updates_each=12000,conditional_new_updates_each=22000,
        physical_gpu='GPU-ddc2c5d8-3507-6293-837c-b1ea6b5e1cc5',paired_same_physical_gpu=True,
        sr_weight=.1,lr_schedule='7200 + cumulative_sr_step; exact original values including fixed child rates',
        freeze_policy=endpoints['U6000']['freeze_policy'],
        information_boundary='Only cam02..20 LR and existing frozen SwinIR enter training; cam00/01 and HR masks/flow evaluation-only development data',
        decision_order=[
            dict(rule=1,action='stop_harm',psnr_loss_gt_db=.20,dynamic_lpips_or_temporal_relative_worse_gt=.05),
            dict(rule=2,action='extend_both_recovery',cam00_psnr_gain_ge=.10,cam01_psnr_loss_le=.05,full_lpips_absolute_worse_le=.003,dynamic_lpips_temporal_relative_worse_le=.03),
            dict(rule=3,action='extend_both_late_decline',control_loss_from_U6000_each_lt=.10,frozen_psnr_loss_each_le=.05,full_lpips_absolute_worse_le=.003,dynamic_lpips_temporal_relative_worse_le=.03),
            dict(rule=4,action='stop_other')],
        candidate40000=dict(psnr_loss_from_U6000_each_le=.10,dynamic_lpips_improvement_from_U6000_each_ge=.03,temporal_worse_from_U6000_each_le=.03,
            cam00_psnr_gain_vs_control_ge=.10,cam01_psnr_loss_vs_control_le=.05,full_lpips_absolute_worse_vs_control_le=.003,dynamic_lpips_temporal_worse_vs_control_le=.03),
        near_threshold='Within 0.001 dB PSNR, 0.00005 absolute LPIPS, or 0.001 relative change: mark pending and no automatic extension; one targeted uncertainty check only if needed',
        numerical_test_tolerance=dict(short_resume_parameter_maxabs=1e-4,short_resume_render_maxabs=1e-3,frozen_exact=True),
        costs='Measure new updates, two renders per update, audit renders separately, wall time, peak allocated VRAM and full inference bytes',
        no_extra_branches=True,history_negative_report=assets['history']['report'])
    target=OUT/'protocol.json'
    if target.exists(): raise FileExistsError(target)
    write_json(target,protocol)
    print(json.dumps(dict(status='passed',flags=endpoints['U6000']['flags'],policy=endpoints['U6000']['freeze_policy'],points=endpoints['U6000']['topology'])))

if __name__=='__main__':main()
