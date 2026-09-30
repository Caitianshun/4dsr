"""Compare the single independent from-start run with immutable historical results."""
import csv
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'output/dynamic_sr_same_observation_20260930'
OLD = ROOT / 'output/dynamic_sr_prior_diagnosis_20260929'
ARMS = ['LR-direct-HRrender','LR-direct-Bicubic','HR-direct-6k','U6000','J1','Async2','Sync2','T']


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(path, rows):
    with Path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    complete = read(OUT/'train/complete.json')
    assert complete['updates']==20200 and complete['combined_supervision_from_first_step']
    for stage in ['coarse','fine']:
        audit = read(OUT/'train'/f'first_update_{stage}.json')
        assert audit['same_render_tensor'] and audit['same_LR_and_SR_observation']
        assert audit['render_calls']==audit['parameter_backward_calls']==audit['optimizer_calls']==1
        assert audit['combined_output_gradient_error']<1e-7 and audit['SR_active_from_first_update']
    checkpoint = OUT/'train'/complete['final_checkpoint']
    assert sha(checkpoint)==complete['final_sha256']
    old = read(OLD/'final_summary.json')
    rows = [{k:r[k] for k in ['arm','scope','repeat_count','psnr','ssim','lpips']}
            for r in old['averages'] if r['arm'] in ARMS]
    per_frame, auxiliary, new = [], [], []
    for split,camera,count in [('test','cam00',60),('dev','cam01',60),('train_fixed',None,16)]:
        path=OUT/'eval'/split/'metrics.json';e=read(path)
        assert len(e['rows'])==count and e['checkpoint_sha256']==complete['final_sha256']
        assert e['lpips_status']=='alex_v0.1_standard_full_and_fixed_mask_spatial_map'
        assert e['parameter_updates']==0
        if camera:
            q=e['aggregate']['full']
            row=dict(arm='P1-from-start',scope=camera,repeat_count=1,
                     psnr=q['psnr_mean'],ssim=q['ssim_mean'],lpips=q['lpips_alex_mean'])
            new.append(row)
            auxiliary.append(dict(camera=camera,
                dynamic_lpips=e['aggregate']['dynamic']['lpips_alex_spatial_mask_mean'],
                dynamic_temporal=e['temporal_aggregate']['dynamic']['gt_relative_warp_l1_mean'],
                full_temporal=e['temporal_aggregate']['full']['gt_relative_warp_l1_mean'],
                lr_l1=e['lr_reprojection_aggregate']['full']['l1_mean']))
        for r in e['rows']:
            q=r['spatial']['full']
            per_frame.append(dict(arm='P1-from-start',split=split,camera=r['camera_id'],frame=r['frame_index'],
                                  psnr=q['psnr'],ssim=q['ssim'],lpips=q['lpips_alex'],
                                  lr_l1=r['lr_reprojection']['full']['l1'],
                                  teacher_rgb_l1=r.get('rgb_teacher_l1'),
                                  h_hr_l1=r['h_hr']['full']['l1']))
    mean=dict(arm='P1-from-start',scope='equal_camera_mean',repeat_count=1,
              **{k:statistics.mean(r[k] for r in new) for k in ['psnr','ssim','lpips']})
    new.append(mean);rows+=new
    gaps=[]
    for scope in ['cam00','cam01','equal_camera_mean']:
        n=next(r for r in new if r['scope']==scope)
        for arm in ARMS:
            r=next(r for r in rows if r['scope']==scope and r['arm']==arm)
            gaps.append(dict(method='P1-from-start',reference=arm,scope=scope,
                             **{k:n[k]-r[k] for k in ['psnr','ssim','lpips']}))
    budgets=[]
    for arm in ARMS+['P1-from-start']:
        updates=7000 if arm.startswith(('LR-','HR-')) else (14200 if arm=='U6000' else 20200)
        points=(110810 if arm.startswith('LR-') else 121884 if arm.startswith('HR-') else
                complete['points'] if arm=='P1-from-start' else 132972)
        forwards=(7000 if arm.startswith(('LR-','HR-')) else 20200 if arm in ['U6000','P1-from-start'] else
                  38200 if arm in ['Async2','Sync2'] else 32200)
        budgets.append(dict(arm=arm,total_updates=updates,training_rgb_forwards=forwards,points=points,
                            repetitions=2 if arm in ['J1','Async2','Sync2','T'] else 1,
                            SR_observations=20200 if arm=='P1-from-start' else 6000 if arm=='U6000' else
                                            18000 if arm in ['Async2','Sync2'] else 12000 if arm in ['J1','T'] else 0,
                            SR_updates=20200 if arm=='P1-from-start' else 6000 if arm=='U6000' else
                                       12000 if arm in ['J1','Async2','Sync2','T'] else 0))
    write_csv(OUT/'comparison_quality.csv',rows)
    write_csv(OUT/'quality_gaps.csv',gaps)
    write_csv(OUT/'metrics_per_frame.csv',per_frame)
    write_csv(OUT/'auxiliary_quality.csv',auxiliary)
    write_csv(OUT/'training_budgets.csv',budgets)
    inventory=[]
    for count in [7000,14200,20200]:
        path=OUT/'train'/f'checkpoint_{count}.pt';receipt=read(path.with_suffix('.json'))
        assert sha(path)==receipt['sha256']
        inventory.append(dict(method='P1-from-start',total_updates=count,path=str(path.relative_to(ROOT)),
                              sha256=receipt['sha256'],status='completed',evaluated=count==20200,
                              remote_path='/home/ubuntu/3DGS/4dsr/'+str(path.relative_to(ROOT))))
    (OUT/'checkpoint_inventory.json').write_text(json.dumps(inventory,ensure_ascii=False,indent=2))
    train_rows=[r for r in per_frame if r['split']=='train_fixed']
    train_summary={k:statistics.mean(r[k] for r in train_rows) for k in ['psnr','ssim','lpips','lr_l1','teacher_rgb_l1','h_hr_l1']}
    result=dict(status='completed_comparison',new_run=new,comparisons=gaps,training=complete,
                auxiliary=auxiliary,train16=train_summary,training_budgets=budgets,
                evaluated_observations=len(per_frame),checkpoint_inventory=inventory,
                historical_source=dict(path=str(OLD/'final_summary.json'),sha256=sha(OLD/'final_summary.json')),
                evidence='One independent from-start run compared with historical single models and two shared-U6000 suffix means; not a paired single-factor ablation',
                controlled='Same scene, camera split, LR degradation, frozen SwinIR, SR coefficient0.1 and float evaluation metric arithmetic',
                differing='Supervision from step1, same-observation coupling, one render/backward per update, native topology/densification, optimizer-stage history and SR exposure count')
    (OUT/'comparison_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps(dict(status=result['status'],metrics=mean,training=complete),ensure_ascii=False))


if __name__=='__main__':
    main()
