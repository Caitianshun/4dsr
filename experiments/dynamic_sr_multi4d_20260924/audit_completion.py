"""CPU-only closeout integrity, sample exposure, costs, and video decode audit.

Reads completed fixed endpoints. Does not select checkpoints or dispatch runs.
"""
import argparse
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
import statistics
import cv2
from summarize import read,write,sha

ROOT=Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--final',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    summary=read(a.final/'tables/summary.json');methods=read(a.final/'methods.json')['methods'];checkpoints=[];evaluations=[]
    for name,item in methods.items():
        checkpoint=Path(item['checkpoint']);h=sha(checkpoint);checkpoints.append(dict(method=name,path=str(checkpoint),sha256=h,updates=item['updates'],bytes=checkpoint.stat().st_size))
        for split,folder in item['evaluations'].items():
            folder=Path(folder);receipt=read(folder/'complete.json');m=read(folder/'metrics.json')
            assert receipt['status']=='completed_evaluation' and receipt['parameter_updates']==0
            assert receipt['checkpoint_sha256']==m['checkpoint_sha256']==h and receipt['metrics_sha256']==sha(folder/'metrics.json')
            assert receipt['observations']==len(m['rows'])==(16 if split=='train_fixed' else 60)
            assert m['gpu']=='NVIDIA RTX PRO 6000 Blackwell Workstation Edition'
            evaluations.append(dict(method=name,split=split,metrics_sha256=receipt['metrics_sha256'],observations=receipt['observations']))
    pipe=a.root/'pipeline_v3'
    initial=[read(pipe/n/'reload_audit.json') for n in ['M0','M1']];assert initial[0]==initial[1]
    ends=[read(pipe/n/'fine_20000.json') for n in ['M0','M1']]
    assert all(e['completed_updates']==20000 and e['render_calls']==99994 for e in ends)
    assert ends[0]['teacher_samples']==0 and ends[1]['teacher_samples']==40000
    assert ends[0]['sampler']['exposure']==ends[1]['sampler']['exposure']
    assert sum(ends[0]['sampler']['exposure'].values())==40000
    schedule=read(a.root/'cook_multi_schedule_v1.json');expected=Counter()
    for batch in schedule['fine']:
        for i in batch:
            c,f=schedule['record_keys'][i];expected[f'fine/{c}/{f:04d}']+=1
    assert dict(expected)==ends[0]['sampler']['exposure']
    wu=[read(a.root/f'{n}40_v1/train/complete.json') for n in ['U','W']]
    for d in wu:assert d['parameter_updates']==34000 and d['total_sr_updates']==40000 and d['source_unchanged'] and not d['smoke']
    for key in ['draw_sha256','lr_draw_sha256','sr_frame_sha256']:assert wu[0][key]==wu[1][key]
    assert read(a.root/'cts_return_status.json')['status']=='returned_completed'
    for step,row in read(a.root/'returned_checkpoint_index.json').items():assert sha(row['path'])==row['sha256']
    storage=read(a.final/'storage/complete.json')
    for row in storage['rows']:assert sha(row['archive'])==row['archive_sha256']
    videos=[]
    for path in sorted((a.final/'views').glob('*/*.mp4')):
        cap=cv2.VideoCapture(str(path));assert cap.isOpened();fps=cap.get(cv2.CAP_PROP_FPS);count=0;shape=None
        while True:
            ok,frame=cap.read()
            if not ok:break
            if shape is None:shape=list(frame.shape)
            assert list(frame.shape)==shape;count+=1
        cap.release();assert count==60 and abs(fps-15)<.01
        videos.append(dict(path=str(path),frames=count,fps=fps,shape=shape,sha256=sha(path)))
    assert len(videos)==6
    manifest_path=ROOT/'data/dynamic_sr/n3dv_prepared/cook_spinach/manifest.json';manifest=read(manifest_path)
    teacher_path=ROOT/'output/dynamic_sr_detail_supervision_20260924/teacher_inventory_v1/cook_spinach/teacher_index.json';teacher=read(teacher_path)
    assert teacher['manifest_sha256']==sha(manifest_path)
    teacher_seconds=[]
    for row in teacher['entries']:
        path=manifest_path.parent/row['relative_path'];assert sha(path)==row['sha256']
        receipt=read(path.with_suffix('.json'));assert receipt['output_sha256']==row['sha256'];teacher_seconds.append(receipt['seconds'])
    assert len(teacher_seconds)==1140
    old=read(ROOT/'output/dynamic_sr_20260918/prior_quality_audit/cook_spinach.json')
    fixed=[v for v in old['rows'] if v['frame_index'] in [0,40,80,118]];assert len(fixed)==16
    for row in fixed:
        for key in ['lr','hr']:
            assert sha(manifest_path.parent/row[key+'_path'])==row[key+'_sha256']
        assert sha(manifest_path.parent/row['prior_path'])==row['prior_sha256']
    prior={name:{metric:statistics.mean(v[name]['full'][metric] for v in fixed) for metric in ['psnr','ssim','lpips_alex']} for name in ['bicubic','swinir']}
    storage_by={r['method']:r for r in storage['rows']};costs=[]
    for c in summary['costs']:
        name=c['method'];metadata=c['endpoint_training_metadata'];elapsed=metadata.get('elapsed_s');prefix=0.
        if name.startswith(('U','W')) and name not in ['U6000','W6000']:
            oldmethod=name[0]+'6000';prefix=read(Path(methods[oldmethod]['train_dir'])/'complete.json')['elapsed_s']
        if name in ['M0','M1']:elapsed=read(pipe/name/'complete.json')['elapsed_s']
        costs.append(dict(method=name,stored_points=c['capacity']['stored_gaussians'],capacity=c['capacity'],
            optimization_segment_s=elapsed,prefix_6000_s=prefix,total_fine_or_sr_s=elapsed+prefix,
            inference_MB=storage_by[name]['inference_archive_bytes']/1e6,training_checkpoint_MB=c['checkpoint_bytes']/1e6,
            render_ms_median=c['render_ms_median'],render_ms_p95=c['render_ms_p95'],peak_allocated_GB=c['training_metadata'].get('peak_gb')))
    write(a.out/'complete.json',dict(status='verified_completed_fixed_round',checked_utc=datetime.now(timezone.utc).isoformat(),
        checkpoints=checkpoints,evaluations=evaluations,videos=videos,costs=costs,
        initial_M0_M1_identity=initial[0],paired_fine_sample_exposure=40000,M0_teacher_samples=0,M1_teacher_samples=40000,
        teacher_inventory_count=1140,teacher_per_image_inference_s_sum=sum(teacher_seconds),teacher_fixed16_reference=prior,
        teacher_reference_note='Reused original pinned audit; same 16 LR/HR/teacher hashes verified. Train-view diagnostic, not novel-view upper bound.',
        multi_shared_coarse_s=read(a.root/'pipeline_v2/coarse/complete.json')['elapsed_s'],
        lr_parent_history_s=sum(read(ROOT/p)['elapsed_s'] for p in ['output/dynamic_sr_20260918/cook_warmup_s20260918/complete.json','output/dynamic_sr_20260918/cook_spinach_pilot_v1_lr_integrated/complete.json']),
        limitations=['Cross-hardware training times do not compare optimization efficiency.','Frame-zero triangulation and environment-build costs were not separately timed.','Video decode integrity is not visual quality review.','Inference archives are uncompressed explicit-schema state, not an upstream loader or codec.']))


if __name__=='__main__':main()
