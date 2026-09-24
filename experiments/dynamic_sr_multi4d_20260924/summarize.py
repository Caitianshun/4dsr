"""Validate and tabulate fixed Multi4D/Wu endpoints; no automatic quality verdict."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False,ensure_ascii=False)+'\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('--methods',required=True,type=Path);p.add_argument('--out',required=True,type=Path)
    p.add_argument('--validate-only',action='store_true',help='Validate real existing endpoints without forming the final comparison groups')
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);methods=read(a.methods)['methods'];identities={};rows=[];costs=[]
    for name,item in methods.items():
        checkpoint_sha=sha(item['checkpoint'])
        for split,folder in item['evaluations'].items():
            folder=Path(folder);v=read(folder/'metrics.json');receipt=read(folder/'complete.json')
            assert receipt['status']=='completed_evaluation' and receipt['parameter_updates']==0
            assert receipt['metrics_sha256']==sha(folder/'metrics.json')
            assert v['checkpoint_sha256']==receipt['checkpoint_sha256']==checkpoint_sha
            assert v['split']==split and len(v['rows'])==(16 if split=='train_fixed' else 60)
            assert v['lpips_status']=='alex_v0.1_standard_full_and_fixed_mask_spatial_map'
            expected_step=v['checkpoint_metadata'].get('completed_updates',v['checkpoint_metadata'].get('intervention_step'))
            assert expected_step==item['updates']
            identity={k:v[k] for k in ['manifest_sha256','metric_helpers_sha256','observation_keys','versions','lr_reprojection_protocol','gpu','lpips_device','dynamic_threshold','flow_scale']}
            identity['mask_hashes']={k:d['dynamic_mask_sha256'] for k,d in v['evaluation_caches'].items()}
            if split in identities:assert identities[split]==identity,(name,split,'metric identity mismatch')
            else:identities[split]=identity
            for region in ['full','dynamic','static']:
                d=v['aggregate'][region];t=v.get('temporal_aggregate',{}).get(region,{});lr=v['lr_reprojection_aggregate'][region]
                rows.append(dict(method=name,split=split,region=region,updates=item['updates'],
                    psnr=d['psnr_mean'],ssim=d['ssim_mean'],lpips=d['lpips_alex_mean' if region=='full' else 'lpips_alex_spatial_mask_mean'],
                    temporal=t.get('gt_relative_warp_l1_mean'),lr_psnr=lr['psnr_mean'],lr_l1=lr['l1_mean'],
                    h_hr=v['h_hr_aggregate'].get(region),h_teacher=v['h_teacher_aggregate'].get(region)))
            if split=='test':
                times=sorted(r['render_seconds'] for r in v['rows'])
                costs.append(dict(method=name,capacity=v['capacity'],checkpoint_bytes=Path(item['checkpoint']).stat().st_size,
                    render_ms_median=1000*statistics.median(times),render_ms_p95=1000*times[round(.95*(len(times)-1))],
                    render_scope=v['render_timing'],training_metadata=read(Path(item['train_dir'])/'complete.json'),
                    endpoint_training_metadata=v['checkpoint_metadata'],
                    raster_visible_per_frame=[r.get('raster_visible_points') for r in v['rows']],checkpoint_sha256=checkpoint_sha))
    if a.validate_only:
        write(a.out/'input_validation.json',dict(status='verified_metric_inputs',methods=list(methods),rows=len(rows),identities=identities));return
    index={(r['method'],r['split'],r['region']):r for r in rows}
    lines=['# 固定端点结果：自动提取，待视觉与机制复核','',
        '完整短窗开发比较；cam00/cam01 均已用于开发，非最终盲测。M0/M1主端点为fine20000，U/W主长端点为总SR40000；不按结果选best。','']
    groups=[('Multi4D 内部教师增量',['M0','M1']),('Wu 延长训练',['U6000','U18000','U40000','W6000','W18000','W40000']),
        ('完整系统与历史参照',['M0','M1','U40000','W40000','B4','U6000','W6000','G'])]
    for label,names in groups:
        lines += ['## '+label,'','| 视角 | 方法 | PSNR | SSIM | 全图 LPIPS | 变化区 LPIPS | 其余区 LPIPS | 变化区时序 L1×1000 | LR PSNR |',
            '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
        for split,cam in [('test','cam00'),('dev','cam01')]:
            for n in names:
                f=index[n,split,'full'];d=index[n,split,'dynamic'];s=index[n,split,'static']
                lines.append(f'| {cam} | {n} | {f["psnr"]:.5f} | {f["ssim"]:.6f} | {f["lpips"]:.6f} | {d["lpips"]:.6f} | {s["lpips"]:.6f} | {1000*d["temporal"]:.5f} | {f["lr_psnr"]:.5f} |')
        lines += ['']
    write(a.out/'summary.json',dict(status='verified_metrics_need_visual_decision',metrics=rows,costs=costs,identities=identities,methods_sha256=sha(a.methods)))
    (a.out/'tables.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
