"""Three fixed comparison tables; preserve per-view differences and privilege."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]


def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser()
    for k in ['methods','root','out']:p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);methods=read(a.methods)['methods'];rows=[];checkpoints={}
    expected={m+('_HR_PRIVILEGED' if m=='O' else '')+str(k) for m in ['Z','U','O'] for k in [6000,18000,40000]}
    assert set(methods)==expected
    subprocess.run([sys.executable,str(ROOT/'experiments/dynamic_sr_multi4d_20260924/summarize.py'),
        '--methods',str(a.methods),'--out',str(a.out/'metric_identity'),'--validate-only'],check=True)
    identity=read(a.root/'inputs_v1/start_identity.json')['identity'];schedule=read(ROOT/read(a.root/'controller_spec.json')['jobs']['Z']['schedule'])
    for name,item in methods.items():
        checkpoints[name]=dict(path=item['checkpoint'],sha256=sha(item['checkpoint']),bytes=Path(item['checkpoint']).stat().st_size)
        for split,folder in item['evaluations'].items():
            v=read(Path(folder)/'metrics.json');meta=v['checkpoint_metadata'];step=item['updates']
            assert v['capacity']==identity['capacity'] and meta['initial_identity']==identity
            assert meta['intervention_step']==step and v['parameter_updates']==0
            if item['method']=='O':assert v['privileged_train_hr'] and 'PRIVILEGED' in v['method']
            if step==40000:
                for key in ['draw_sha256','lr_draw_sha256','sr_frame_sha256']:assert meta[key]==schedule[key]
            f=v['aggregate']['full'];d=v['aggregate']['dynamic'];s=v['aggregate']['static'];lr=v['lr_reprojection_aggregate']['full']
            row=dict(name=name,method=item['method'],privileged_train_hr=item['method']=='O',step=step,split=split,
                psnr=f['psnr_mean'],ssim=f['ssim_mean'],lpips=f['lpips_alex_mean'],dynamic_lpips=d['lpips_alex_spatial_mask_mean'],
                static_lpips=s['lpips_alex_spatial_mask_mean'],dynamic_temporal=v.get('temporal_aggregate',{}).get('dynamic',{}).get('gt_relative_warp_l1_mean'),
                lr_psnr=lr['psnr_mean'],lr_l1=lr['l1_mean'],h_hr_l1=v['h_hr_aggregate']['full']['l1_mean'],
                h_teacher_l1=v['h_teacher_aggregate'].get('full',{}).get('l1_mean'),metrics_path=str(Path(folder)/'metrics.json'))
            if split=='train_fixed':
                assert all(x['rgb_teacher_l1'] is not None for x in v['rows'])
                row['rgb_teacher_l1']=statistics.mean(x['rgb_teacher_l1'] for x in v['rows'])
            rows.append(row)
    storage=read(a.methods.parent/'storage/complete.json');storage_by={x['method']:x for x in storage['rows']};costs=[]
    for method in ['Z','U','O']:
        name=method+('_HR_PRIVILEGED' if method=='O' else '')+'40000';item=methods[name]
        dirs=[Path(methods[method+('_HR_PRIVILEGED' if method=='O' else '')+'6000']['train_dir']),Path(item['train_dir'])]
        completions=[read(p/'complete.json') for p in dirs]
        if method!='U':
            assert [c['actual_updates'] for c in completions]==[6000,34000]
            for p in dirs:
                exposure=read(p/'exposure.json');c=read(p/'complete.json')
                assert exposure['actual_segment_render_calls']==2*c['actual_updates']
                if method=='Z':assert not read(p/'image_reads.json')['actual_high_resolution_opens']
        assert all(c['status']=='completed' and c['source_unchanged'] for c in completions)
        b=storage_by[name];assert b['source_checkpoint_sha256']==checkpoints[name]['sha256'] and sha(b['archive'])==b['archive_sha256']
        costs.append(dict(name=name,method=method,privileged_train_hr=method=='O',target={'Z':'none after shared teacher-influenced selection','U':'frozen SwinIR train RGB','O':'PRIVILEGED legal train HR'}[method],
            lr_samples=40000,second_supervised_samples=0 if method=='Z' else 40000,render_calls=80000,actual_updates=40000,
            training_seconds_by_segment=[c['train_s'] for c in completions],training_seconds_total=sum(c['train_s'] for c in completions),
            segment_wall_seconds=[c.get('wall_s',c['elapsed_s']) for c in completions],peak_gb_by_segment=[c['peak_gb'] for c in completions],
            hardware_by_segment=['RTX PRO6000','RTX3090 '+('cts' if method=='Z' else 'local')],
            inference_archive_MB=b['inference_archive_bytes']/1e6,training_checkpoint_MB=b['training_checkpoint_bytes']/1e6,
            cost_boundary='Reported continuation training only. Shared LR parent, original B/SR selection and frozen prior preparation reused; Z empty second render is experimental control cost. Evaluation/IO excluded from train_s.',
            source_train_dirs=[str(p) for p in dirs]))
    index={(x['method'],x['step'],x['split']):x for x in rows};differences=[]
    for split in ['test','dev']:
        z,u,o=[index[m,40000,split] for m in ['Z','U','O']]
        for ref,candidate,label in [(z,u,'teacher_increment_Z_minus_U'),(u,o,'supervision_replacement_U_minus_O')]:
            differences.append(dict(split=split,label=label,psnr_gain_db=candidate['psnr']-ref['psnr'],ssim_gain=candidate['ssim']-ref['ssim'],
                errors={k:dict(absolute_reduction=ref[k]-candidate[k],relative_reduction_percent=100*(ref[k]-candidate[k])/ref[k]) for k in ['lpips','dynamic_lpips','static_lpips','dynamic_temporal','lr_l1','h_hr_l1']}))
    lines=['# Z/U/O 固定端点表（等待科学与视觉复核）','',
        'O_HR_PRIVILEGED 为真实训练 HR 特权诊断；不列为合法 LR 方法或严格上界。cam00/cam01 均为开发视角。保留共同主端点，不选 best。60帧相关，未作独立种子显著性声明。','',
        '## 1. 40000 主表','']
    header=['| 视角 | 路径 | PSNR | SSIM | 全图 LPIPS | 变化区 LPIPS | 其余区 LPIPS | 变化区时间 L1×1000 | LR PSNR |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    def table_row(r):
        cam={'test':'cam00','dev':'cam01'}[r['split']]
        return f'| {cam} | {r["name"]} | {r["psnr"]:.5f} | {r["ssim"]:.6f} | {r["lpips"]:.6f} | {r["dynamic_lpips"]:.6f} | {r["static_lpips"]:.6f} | {1000*r["dynamic_temporal"]:.5f} | {r["lr_psnr"]:.5f} |'
    lines+=header+[table_row(index[m,40000,sp]) for sp in ['test','dev'] for m in ['Z','U','O']]
    lines+=['','## 2. 三个预定过程端点','']+header
    lines += [table_row(index[m,k,sp]) for sp in ['test','dev'] for m in ['Z','U','O'] for k in [6000,18000,40000]]
    lines+=['','## 3. 训练观察与成本','',
        'train16 是固定训练观察抽样，不等于随机训练批次均值或全部训练分布。RGB 指标沿用原浮点评价范围处理；LR 回投和 H 从原始未量化渲染计算。H 含 clamp，并非严格线性高通/零空间。','',
        '| 路径 | LR L1 | LR PSNR | HR PSNR | HR SSIM | HR LPIPS | 同一教师 RGB L1 | H教师 L1 | H真实HR L1 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for m in ['Z','U','O']:
        for k in [6000,18000,40000]:
            r=index[m,k,'train_fixed'];lines.append(f'| {r["name"]} | {r["lr_l1"]:.7f} | {r["lr_psnr"]:.5f} | {r["psnr"]:.5f} | {r["ssim"]:.6f} | {r["lpips"]:.6f} | {r["rgb_teacher_l1"]:.7f} | {r["h_teacher_l1"]:.7f} | {r["h_hr_l1"]:.7f} |')
    reference=read(a.methods.parent/'target_reference.json')
    for kind in ['teacher','hr']:
        r=reference['aggregate'][kind];lines += ['',f'D({kind}) 对存储 LR：train16 L1={r["l1"]:.7f}，PSNR={r["psnr"]:.5f} dB（逐帧 dB 均值）。']
    lines+=['','| 路径/监督源 | LR/第二路监督次数 | 完整模型 MB | 训练检查点 MB | PRO6000/3090 训练秒 | 两段峰值 GB |',
        '| --- | --- | ---: | ---: | --- | --- |']
    for r in costs:
        times='/'.join(f'{x:.2f}' for x in r['training_seconds_by_segment']);peak='/'.join(f'{x:.3f}' for x in r['peak_gb_by_segment'])
        lines.append(f'| {r["name"]}: {r["target"]} | 40000/{r["second_supervised_samples"]} | {r["inference_archive_MB"]:.3f} | {r["training_checkpoint_MB"]:.3f} | {times} | {peak} |')
    lines+=['','三路均 40000 次更新、80000 次渲染，模型包含基座、全部子点、形变网络及固定图表缓冲。时间仅为追加训练，两段相加；共同 LR7200、原 B 选点、已有 SwinIR 准备是复用成本，未宣称从零总成本。Z 空第二路属于实验控制开销，不能据此评价高效 LR-only。固定推理结构相同，本轮不做额外 FPS 竞赛。','',
        '解释限制：留出 LR 分数下降首先是新视角泛化变化；只有结合 train16 与 Z 才能讨论训练观察符合度及教师作用。O−U 不能区分教师单图误差、跨视角/时间不一致和低频偏差；O≈U 不证明真实上限。']
    write(a.out/'summary.json',dict(status='verified_tables_need_scientific_visual_review',metrics=rows,costs=costs,differences=differences,checkpoints=checkpoints,target_reference=reference))
    write(a.root/'checkpoint_index.json',dict(status='all_fixed_endpoints_local_sha256_verified',checkpoints=checkpoints))
    (a.out/'tables.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
