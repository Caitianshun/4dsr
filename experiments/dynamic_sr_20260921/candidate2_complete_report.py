#!/usr/bin/env python3
import hashlib,json,shutil
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
REPORT=ROOT/'docs/dynamic_sr_candidate2_findings_2026-09-21.md'
NAMES={'cook':'菠菜','meeting':'会议室'}
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def fmt(x,n=7):return f'{x:+.{n}f}'
def main():
 allm={};audit={};lines=['','## 扩大 B 覆盖与匹配更新预算后的补充','',
 '主结果不足以区分“候选思路无效”和“两个编号相近的核验相机覆盖不足”。因此做一次有界的探索性补充：依标定相机中心的最大最小距离选四个分散训练相机，各取两个时刻，共八观察；比较固定随机八观察。这里只依据相机标定与 LR 选 B，未由 HR 挑相机；但已看过主结果，故不称独立确认。完全复用旧方向、旧步幅与旧检查点。','',
 '| 场景／8 观察 B | 接受数 | 其中 PSNR 改善／持平／有害 | LR 分数与新视角 PSNR 相关 | 门控 ΔPSNR | ΔSSIM | ΔLPIPS |','| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
 for scene in NAMES:
  path=ROOT/f'output/dynamic_sr_20260921/candidate2_{scene}_coverage_budget_v1';m=json.loads((path/'metrics.json').read_text());lock=json.loads((path/'lr_locked.json').read_text());comp=json.loads((path/'complete.json').read_text())
  assert sha(path/'metrics.json')==comp['metrics_sha256'];assert sha(path/'lr_locked.json')==m['lock_sha256'];assert sha(path/'source.py')==lock['source_sha256'];assert all(x['role']!='evaluation_only_hr' for x in lock['inputs'].values());assert m['rollback_exact'] and m['optimizer_unchanged']
  for f,v in m['inputs'].items():assert sha(f)==v['sha256']
  budget_errors={k:abs(v['gate_total_l2']-v['uniform_total_l2']) for k,v in m['budget'].items()};assert max(budget_errors.values())<1e-12
  audit[scene]=dict(metrics_sha256=sha(path/'metrics.json'),locked_lr_sha256=m['lock_sha256'],files_hashed=len(m['inputs']),equal_budget_errors=budget_errors,cross_score_replay_max_error=m['parent_cross_score_replay_max_abs_error'],elapsed_s=m['elapsed_s'],gpu=m['gpu'],peak_gb=m['peak_gb'])
  log=Path(f'/tmp/candidate2_{scene}_coverage_budget_v1.log')
  if log.exists():shutil.copy2(log,path/'run.log')
  for kind,label in [('spread8','几何分散8'),('random8','随机8')]:
   r=m['summary']['0.005'][kind];v=r['gated_mean_delta'];neutral=r['accepted']-r['harmed_accepted']-r['improved_accepted']
   lines.append(f"| {NAMES[scene]}／{label} | {r['accepted']}/12 | {r['improved_accepted']}／{neutral}／{r['harmed_accepted']} | {r['spearman_psnr']:+.3f} | {fmt(v['psnr'],6)} | {fmt(v['ssim'])} | {fmt(v['lpips'])} |")
  allm[scene]=m
 lines+=['','分散覆盖在会议室有弱的正向线索，但菠菜仍放过四个有害提议；不能认为增加真实 LR 数量便解决新视角可靠性。相机中心分散也不等于同一表面可见或子像素相位互补。','',
 '随后增加等总步长的简单对照。只用 LR 接受名单和冻结方向范数，计算统一较小的步幅，让全部 12 个提议的参数 L2 步长总和与门控完全一致。主步幅、门控及统一缩步全部在 3090 上重评；源自原 PRO 6000 的菠菜 cross LR 分数重放最大绝对差为 0。此处预算是参数坐标下的更新幅度，不是物理位移或图像变化量。','',
 '| 场景／规则 | 统一步幅 | 门控／统一缩步 ΔPSNR | 门控／统一缩步 ΔSSIM | 门控／统一缩步 ΔLPIPS | 门控／统一缩步 Δ高残差 MSE ×10⁶ |','| --- | ---: | ---: | ---: | ---: | ---: |']
 for scene,m in allm.items():
  for k,label in [('cross2','不同相机2'),('spread8','几何分散8')]:
   b=m['budget'][k];g=b['results']['novel']['gate'];u=b['results']['novel']['uniform']
   lines.append(f"| {NAMES[scene]}／{label} | {b['alpha']:.8f} | {fmt(g['psnr'],6)}／{fmt(u['psnr'],6)} | {fmt(g['ssim'])}／{fmt(u['ssim'])} | {fmt(g['lpips'])}／{fmt(u['lpips'])} | {fmt(g['hf_mse']*1e6,4)}／{fmt(u['hf_mse']*1e6,4)} |")
 lines+=['','不同相机2门控在两域都没有胜过等总参数步长的统一缩步 PSNR；几何分散8在会议室优于对应统一缩步，但提升约 0.00072 dB，且菠菜没有同样优势。LPIPS、SSIM和高残差并非全部同向，因此不能只凭 PSNR 认定任何更新完全无价值；现有证据仍不足以把这套接受规则投入长训练。','',
 '## 一条可追踪的反例','',
 '菠菜 cam06/frame56 的主步提议：A 的 SR L1 从 0.009181918 降至 0.009031285；A 的 LR L1 下降 0.000146813。相邻训练相机 cam05/cam07 同帧的 B-LR MSE 下降 1.229866×10⁻⁶，B 同位置 HR 的 PSNR／SSIM／LPIPS 也分别改善 +0.010793 dB／+0.00000307／−0.00002205。可是 cam00 的对应三指标变为 −0.091724 dB／−0.00037986／+0.00021259。它清楚说明“另一组训练观察支持”仍不能自动推出新视角正确。cam06 三个提议时刻都出现同类问题，不是单个随机行；但 cam00 的尺度高残差 MSE 同时略改善，所以也不能把全部退化归为高频细节失真。','',
 '## 研究判断','',
 '1. **本轮支持的是有限能力。** cross 的真实 LR 变化能较好排序其同位置 HR 的 PSNR 变化，两域 Spearman 都为 0.930；这可以成为跨分辨率局部诊断，但尚不能推广为新视角的更新信任度。','2. **当前形式不进入长训练。** 小步、真实教师下降、精确回滚和等更新预算都已控制，依然没有两域一致的新视角收益。直接把全图 B-LR 误差下降用作 SR 接受规则，证据不足。','3. **也不全盘否定更严格的局部核验。** 本轮 B 只按时间、相机编号或相机中心覆盖选择，没有验证与 A 同一表面的可见性，也没有测互补采样相位。若后续候选一能提供经实际 LR 验证的局部对应和可辨识方向，可在这些局部内再次测试候选更新；需要独立新样例，不能继续在这 24 个提议上改规则直到获益。','4. **优先保留简单对照。** 后续任何门控原型都应先超过统一缩步、同批 LR 一阶/有限步检查以及同预算随机观察，并同时报告 PSNR、SSIM、LPIPS和细节误差。LR 零空间中的假细节依旧可能不可检验。','',
 f"补充实际耗时：菠菜 {audit['cook']['elapsed_s']:.2f} 秒、会议室 {audit['meeting']['elapsed_s']:.2f} 秒；两域均在本机 3090 顺序执行，无远端计算。结果和日志已保存在本机。",'',
 '- [补充源码](../experiments/dynamic_sr_20260921/candidate2_coverage_budget.py)','- [菠菜覆盖及预算结果](../output/dynamic_sr_20260921/candidate2_cook_coverage_budget_v1/metrics.json)','- [会议室覆盖及预算结果](../output/dynamic_sr_20260921/candidate2_meeting_coverage_budget_v1/metrics.json)','']
 auditpath=ROOT/'output/dynamic_sr_20260921/candidate2_assessment_v2/supplement_audit.json';auditpath.write_text(json.dumps(audit,indent=2,ensure_ascii=False))
 text=REPORT.read_text();intro='**结论：当前“只看另一组训练 LR 是否改善”的门控，尚不足以保证新视角 SR 更新可靠。** 它对核验位置自己的 HR 有预测力，但没有在两域稳定超过等总步长的统一缩步。已补更广相机覆盖、随机等预算与统一缩步对照；本轮不据此启动长期门控训练，也不把这项有限测试推广为所有跨观察核验都无效。\n\n'
 text=text.replace('日期：2026-09-21。',intro+'日期：2026-09-21。',1);REPORT.write_text(text+'\n'.join(lines))
 # Plot primary LR score vs development HR change, preserving camera clusters.
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 fig,axs=plt.subplots(2,2,figsize=(9,6.5),layout='constrained')
 for i,scene in enumerate(NAMES):
  parent=json.loads((ROOT/f'output/dynamic_sr_20260921/candidate2_{scene}_v2/metrics.json').read_text());p={r['proposal_id']:r for r in parent['rows'] if r['scale']==.005};ext={r['proposal_id']:r for r in allm[scene]['rows'] if r['scale']==.005}
  cams=sorted({k.split('_')[0] for k in p})
  for j,kind in enumerate(['cross2','spread8']):
   ax=axs[i,j]
   for cam in cams:
    ks=[k for k in p if k.startswith(cam+'_')];x=[(p[k]['scores']['cross'] if kind=='cross2' else ext[k]['scores']['spread8'])*1e6 for k in ks];y=[p[k]['eval_delta']['novel']['psnr'] for k in ks]
    ax.scatter(x,y,label=cam,s=35)
   ax.axvline(.01,color='gray',lw=.8);ax.axhline(0,color='gray',lw=.8);ax.set_title(f'{scene} / {kind}');ax.set_xlabel('LR MSE decrease (x 1e6)');ax.set_ylabel('Novel-view PSNR change (dB)');ax.legend(fontsize=7,ncol=2);ax.grid(alpha=.15)
 plot=ROOT/'output/dynamic_sr_20260921/candidate2_assessment_v2/score_vs_novel_psnr.png';fig.savefig(plot,dpi=160);plt.close(fig)
 with REPORT.open('a') as f:f.write('\n![LR 分数与开发新视角 PSNR 变化](../output/dynamic_sr_20260921/candidate2_assessment_v2/score_vs_novel_psnr.png)\n\n图中每个颜色代表提出更新的 A 相机。右侧且下方的点表示 B-LR 改善但新视角 PSNR 变差；三个同色点对应不同 A 时刻，不能视为完全独立样本。\n')
 print(json.dumps(audit,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
