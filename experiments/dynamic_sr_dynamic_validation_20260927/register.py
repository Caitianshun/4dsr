"""Freeze the user-authorized dynamic experiment after the legacy read-only audit."""
from dv_common import *
from schedule import generate

def entry(path):return dict(path=str(path.relative_to(ROOT)),sha256=sha(path))
def rule(path,change,operator,threshold,reference):return dict(path=path,change=change,operator=operator,threshold=threshold,reference=reference)
def main():
    assert not (OUT/'protocol.json').exists()
    old=read(ROOT/'output/dynamic_sr_prior_guidance_20260927/protocol.json')
    legacy=read(OUT/'legacy_summary.json');assert legacy['status']=='completed' and legacy['rgb_forwards']==63
    old_schedule=bound(old['schedule']);schedules={}
    for i,mode in [('1','original_continuation'),('2','independent_suffix')]:
        path=OUT/f'schedule_{i}.json';write(path,generate(read(old_schedule),mode));schedules[i]=entry(path)
    m=read(bound(old['manifest']));tr=[o for o in m['observations'] if o['split']=='train'];assert len(tr)==1140
    root=bound(old['manifest']).parent;teacher=read(bound(old['teacher']));by={(r['camera'],r['frame']):r for r in teacher['entries']}
    assert set(by)=={(o['camera_id'],o['frame_index']) for o in tr}
    files=[];evals=[]
    for o in m['observations']:
        if o['split']=='train':
            t=by[o['camera_id'],o['frame_index']]
            for role,rel,h in [('LR',o['lr_path'],o['lr_sha256']),('teacher',t['relative_path'],t['sha256'])]:
                path=root/rel;assert sha(path)==h
                files.append(dict(role=role,camera=o['camera_id'],frame=o['frame_index'],path=str(path.relative_to(ROOT)),sha256=h))
        if o['split']!='train' or o['frame_index'] in [0,40,80,118]:
            for role in ['hr','lr']:
                path=root/o[role+'_path'];assert sha(path)==o[role+'_sha256']
                evals.append(dict(role=role,camera=o['camera_id'],frame=o['frame_index'],path=str(path.relative_to(ROOT)),sha256=o[role+'_sha256']))
    old_methods=read(ROOT/'output/dynamic_sr_controlled_headroom_20260926/final_v1/methods.json')['methods'];caches={}
    for c,split in [('cam00','test'),('cam01','dev')]:
        metrics=read(Path(old_methods['U6000']['evaluations'][split])/'metrics.json');item=metrics['evaluation_caches'][c];cache=local(item['path'])
        assets=[cache/'dynamic_mask.png',cache/'complete.json'] if (cache/'complete.json').exists() else [cache/'dynamic_mask.png']
        assets+=list(cache.rglob('*.npz'))+list(cache.glob('*.json'))
        caches[c]=dict(previous_metrics=entry(Path(old_methods['U6000']['evaluations'][split])/'metrics.json'),identity=item['identity'],path=str(cache.relative_to(ROOT)),assets=[entry(p) for p in sorted(set(assets))])
        assert sha(cache/'dynamic_mask.png')==item['dynamic_mask_sha256'];assert len(list(cache.rglob('*.npz')))>=59
    r=lambda path,t,ref='same_set_reference':rule(path,'relative','le',t,ref)
    d=lambda path,t,ref='same_set_reference':rule(path,'absolute','ge',t,ref)
    gates=[r('train76_lr',.05),r('cameras.cam00.temporal',.05),r('cameras.cam01.temporal',.05),d('cameras.cam01.psnr',-.2),r('cameras.cam01.lpips',.05)]
    for c in m['splits']['train']:gates.extend([d('train.'+c+'.psnr',-.2),r('train.'+c+'.lpips',.05)])
    ugates=[r('train76_lr',.05,'U6000')]
    for c in ['cam00','cam01']:ugates.extend([d('cameras.'+c+'.psnr',-.2,'U6000'),r('cameras.'+c+'.lpips',.05,'U6000'),r('cameras.'+c+'.temporal',.05,'U6000')])
    p=dict(schema=1,experiment=HERE.name,registered_unix=time.time(),adopted_plan=entry(OUT/'adopted_plan.codex.md'),
        manifest=entry(bound(old['manifest'])),teacher=entry(bound(old['teacher'])),lr_curve=entry(bound(old['lr_curve'])),old_schedule=entry(old_schedule),schedules=schedules,
        parent=dict(path=str(local(old['start']).relative_to(ROOT)),sha256=old['start_sha256']),
        legacy_summary=entry(OUT/'legacy_summary.json'),old_decision=dict(path='output/dynamic_sr_covariance_probe_20260927/decision.json',sha256=legacy['old_decision_sha256'],preserved_status='fail'),
        roi=entry(ROOT/'output/dynamic_sr_soft_motion_20260924/roi_registry/cook_spinach/roi_protocol.json'),
        training_files=files,evaluation_files=evals,temporal_caches=caches,
        training=dict(host='a100-train',physical_gpu='GPU-b79cd3fe-81f0-f449-30be-432e2857e517',parent_step=6000,stop=12000,save=[9000,12000],sr_weight=.1,lr_offset=7200,points=132972,order={'1':['J_joint','A_sh','S_cov'],'2':['S_cov','A_sh','J_joint']},objective='LR L1 + unchanged regularization + 0.1 teacher RGB L1',route_start=6001,position_lock=False),
        evaluation=dict(host='local',physical_gpu='GPU-c40035c3-0f06-e88b-73f5-fa40d62ec4ec',train_frames=[0,40,80,118],train_cameras=m['splits']['train'],dev_frames=list(range(0,120,2)),train16_cameras=['cam02','cam06','cam12','cam18'],aggregate='per-frame metrics, per-camera means, camera equal weighting',temporal_metric='temporal_aggregate.dynamic.gt_relative_warp_l1_mean'),
        engineering=old['engineering'],budget=dict(engineering_max=32,formal_max=36000,total_max=36032,formal_RGB_max=72000,formal_adam_max=72000),
        decision=dict(minimum_denominator=1e-6,comparison_roundoff_tolerance=1e-12,comparison=dict(reference='J_joint',early_reference='U6000',candidates=['A_sh','S_cov'],extra_path=['S_cov','A_sh'],fallback='J/U6000'),
            benefit_modes={'fidelity':[d('cameras.cam00.psnr',.2),r('cameras.cam00.dynamic_lpips',.02)],'perceptual':[r('cameras.cam00.dynamic_lpips',-.05),d('cameras.cam00.psnr',-.1)]},guardrails_vs_reference=gates,guardrails_vs_U6000=ugates,observation_lines=[r('train76_lr',.02),r('cameras.cam00.temporal',.02),r('cameras.cam01.temporal',.02)],diagnostics_only=['teacher_H','H_HR','SSIM']),
        information_boundary='Only registered training LR and frozen teacher files may enter training. HR/flow/masks are separate evaluation-process inputs. cam00/01 are development views.',previous_observed_evidence='Static S ROI PSNR +1.104 dB and LPIPS -5.42%; teacher H remaining17 +3.48%. New tolerances authorized by attached plan; no static requalification.',deadline='Two working days execution budget; no parameter search, no new scene in this run.')
    assert sha(local(p['parent']['path']))==p['parent']['sha256'];write(OUT/'protocol.json',p)
    write(OUT/'schedule_manifest.json',dict(schedules=schedules,old_source=entry(old_schedule),protocol_sha256=sha(OUT/'protocol.json')))
    print(json.dumps(dict(status='registered',protocol_sha256=sha(OUT/'protocol.json'),training_files=len(files),evaluation_files=len(evals))))
if __name__=='__main__':main()
