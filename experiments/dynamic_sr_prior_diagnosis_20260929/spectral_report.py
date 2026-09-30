#!/usr/bin/env python3
"""Compact machine-readable and human-readable audit after all five models."""
from spectral_common import *
from audit_spectrum_color import dump_csv, summarize

def mean(rr,k): return float(np.mean([r[k] for r in rr]))

def main():
    names=['LR6k','HR6k','U6000','r1_J1','r1_Async2']
    for n in names: assert read(OUT/n/'complete.json')['observations']==196
    summarize()
    rows=[];regions=[];counter=[]
    for n in names:
        for f in sorted((OUT/n).glob('cam*.json')):
            d=read(f);rows.append(d['budget']);regions+=d['regions'];counter+=d['counterfactuals']
    ag=[];rg=[];cg=[]
    for model in names:
        for camera in ['cam00','cam01','cam02','dev_equal','train76']:
            def chosen(r):
                return r['model']==model and (r['split']=='train' if camera=='train76' else (r['split']=='development' if camera=='dev_equal' else r['camera']==camera))
            r=[r for r in rows if chosen(r)]; c=[r for r in counter if chosen(r) and 'psnr' in r]
            scalar=['mse','dc_mse','low_mse','mid_mse','high_mse','q0_mse','q1_mse','q2_mse','q_chroma_mse','raw_out_of_range_fraction']
            a=dict(model=model,camera=camera,observations=len(r),**{k:mean(r,k) for k in scalar})
            for b in ['dc','low','mid','high']:a[b+'_error_share']=a[b+'_mse']/a['mse']
            for b in ['low','mid','high']:a[b+'_gt_structure_share']=mean(r,b+'_gt_structure_share')
            a['q0_error_share']=a['q0_mse']/a['mse'];a['chroma_error_share']=a['q_chroma_mse']/a['mse'];ag.append(a)
            base={k:mean([x for x in c if x['counterfactual']=='unchanged'],k) for k in ['psnr','ssim','lpips']}
            for label in sorted(set(x['counterfactual'] for x in c)):
                rr=[x for x in c if x['counterfactual']==label]
                res=dict(model=model,camera=camera,counterfactual=label,observations=len(rr),**{k:mean(rr,k) for k in base})
                res.update({k+'_delta':res[k]-base[k] for k in base});cg.append(res)
            rr=[x for x in regions if chosen(x)]
            for label in sorted(set(x['region'] for x in rr)):
                rrr=[x for x in rr if x['region']==label]
                # For dev_equal different ROI sets cannot average only present
                # observations and call it a full-view contribution. Skip.
                if camera in ['dev_equal','train76']:continue
                v=mean(rrr,'mse_contribution')
                rg.append(dict(model=model,camera=camera,region=label,observations=len(rrr),
                               area_fraction=mean(rrr,'area_fraction'),mse_contribution=v,
                               full_error_share=v/a['mse'],within_region_mse=mean(rrr,'within_region_mse')))
    dump_csv(OUT/'band_color_summary.csv',ag);dump_csv(OUT/'region_summary.csv',rg);dump_csv(OUT/'counterfactual_delta_summary.csv',cg)
    oldp=ROOT/'output/dynamic_sr_sync_multiview_20260928/evaluation/r1_Async2/endpoint.json';old=read(oldp);parity=[]
    for cam in ['cam00','cam01']:
        x=next(r for r in cg if r['model']=='r1_Async2' and r['camera']==cam and r['counterfactual']=='unchanged')
        parity.append(dict(camera=cam,**{k+'_difference':x[k]-old['cameras'][cam][k] for k in ['psnr','ssim','lpips']}))
    assert max(abs(r[k]) for r in parity for k in r if k!='camera')==0
    write(OUT/'summary.json',dict(status='completed',models=names,observations=980,band_color=ag,regions=rg,
         counterfactuals=cg,old_Async2_metric_parity=parity,old_endpoint_sha256=sha(oldp),
         j1_two_device_metric_parity=read(OUT/'j1_two_gpu_parity.json'),
         hardware_ledger_sha256=sha(OUT/'diagnostic_hardware_ledger.json'),
         backprojection=read(OUT/'anchor_decision.json'),arithmetic=read(OUT/'arithmetic_audit.json'),
         interpretation='Full image DCT and orthogonal color are additive. Fixed spatial ROIs plus rest are independently additive. Their projections do not commute; no additive ROI-by-band claim.',
         training_decision='Controller owns freeze. Teacher target choice and map scale are never chosen independently here.'))
    cur=[r for r in ag if r['model']=='r1_Async2' and r['camera'] in ['cam00','cam01']]
    text=['# 频带、颜色和离线反事实诊断',
          '', '五个固定模型各196观察的原始浮点导出均完成诊断；共980观察。cam00与cam01是已用于开发的相机。每个模型的第五种回投影仅使用76个合法训练观察，开发相机对应项标为NA，未读取其真实LR作图像修正。',
          '', '各频带采用全图正交DCT，误差贡献先逐帧计算、再平均。表中比例是平均绝对MSE之比；PSNR另按逐帧指标平均，未由均值PSNR倒算MSE。',
          '', '| Async2首套 | 总MSE | DC占比 | 低频非DC占比 | 中频占比 | 高频占比 | 亮度方向占比 |',
          '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in cur:text.append(f"| {r['camera']} | {r['mse']:.9f} | {r['dc_error_share']:.2%} | {r['low_error_share']:.2%} | {r['mid_error_share']:.2%} | {r['high_error_share']:.2%} | {r['q0_error_share']:.2%} |")
    lamp=next(r for r in rg if r['model']=='r1_Async2' and r['camera']=='cam01' and r['region']=='lamp_wall_corner_reference')
    text += ['',f"cam01预先冻结的灯具墙角ROI覆盖{lamp['area_fraction']:.2%}像素，贡献{lamp['full_error_share']:.2%}总MSE。这个空间结果与低频结果各自可加，但不能把二者直接相乘或当成相同子空间。",'',
        '低频错误明显，并不表示只需低频先验。真实HR替换低频可大幅改善cam01的PSNR；替换中高频对LPIPS仍有更大改善。Oracle只表明潜在收益位置，不证明现有教师、深度、光流或高斯增点能实现它。', '',
        '原始SwinIR教师8步回投影将训练LR闭环MSE从约6.39×10⁻⁷降到9.59×10⁻¹⁰，但真实HR低频MSE仅下降约0.232%。PSNR均值−0.00417 dB、SSIM−0.000227、LPIPS−0.003535；76个观察的LPIPS均改善，SSIM均微降，旧cam02四个ROI的MSE均微升。其合法训练目标和HR诊断表已分流保存；这里不把闭环变好等同于真实细节变好。', '',
        '同一D0核的有限网格范数平方为0.0640420347298，固定步长14.8340071331，8次更新均使用自动微分真实伴随。数据实际还包含clamp与uint8量化，因此该步骤是线性部分上的有限步观测校正，不称为完整实际退化的精确正交投影。', '',
        '本次Async2两开发相机原图PSNR、SSIM、LPIPS与历史同检查点评价逐位一致。可加残差、每帧原值、原始超范围比例、颜色偏差和各反事实clamp前后算术见JSON/CSV，所有数字直接来自浮点资产。', '',
        '主要文件：`error_budget.csv`、`band_color_summary.csv`、`region_summary.csv`、`regional_current_minus_hrmodel.csv`、`counterfactual_delta_summary.csv`、`anchor_decision.json`、`demand_coverage_summary.json`。两张固定frame40频谱图仅作解释，显示范围固定；频率残差平方图不能逐像素相加。']
    (OUT/'findings.md').write_text('\n'.join(text)+'\n')

if __name__=='__main__':main()
