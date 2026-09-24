#!/usr/bin/env python3
"""Independent receipt/score audit and descriptive reporting for candidate 2."""
import hashlib,json,math
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'output/dynamic_sr_20260921/candidate2_assessment_v2'
NAMES={'cook':'菠菜','meeting':'会议室'}
GATES={'cross':'不同相机 B','far':'间隔时间 B','near':'相邻时间 B','random':'随机 B','A_train_L1_actual':'同批 A-LR L1 实测','A_train_L1_derivative':'同批 A-LR L1 导数','A_actual':'同批 A-LR MSE 实测','A_derivative':'同批 A-LR MSE 导数'}
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def fmt(v,n=5):return '—' if v is None else f'{v:+.{n}f}'
def main():
 OUT.mkdir(parents=True,exist_ok=False)
 data={};audit={}
 for scene in NAMES:
  p=ROOT/f'output/dynamic_sr_20260921/candidate2_{scene}_v2';m=json.loads((p/'metrics.json').read_text());locked=json.loads((p/'lr_scores_locked.json').read_text());inputs=json.loads((p/'inputs.json').read_text());complete=json.loads((p/'complete.json').read_text())
  assert sha(p/'metrics.json')==complete['metrics_sha256'];assert sha(p/'source.py')==m['config']['source_sha256'];assert sha(p/'lr_scores_locked.json')==m['locked_lr_sha256']
  assert m['final_parameter_hash']==m['config']['initial_parameter_hash'];assert m['optimizer_unchanged']
  for file,v in inputs.items():assert sha(file)==v['sha256'],file
  assert all(v['role']!='evaluation_only_hr' for v in locked['inputs'].values())
  for prop,hash_ in locked['direction_hashes'].items():assert sha(p/'directions'/f'{prop}.pt')==hash_
  max_error=0
  for r in m['rows']:
   for gate,ks in r['B'].items():
    score=float(np.mean([r['lr_before'][k]['mse']-r['lr_after'][k]['mse'] for k in ks]));max_error=max(max_error,abs(score-r['scores'][gate]));assert abs(score-r['scores'][gate])<1e-12
   for gate,score in r['scores'].items():
    t=r['score_thresholds'][gate];state='improve' if score>t else 'worsen' if score < -t else 'unresolved';assert state==r['gate_state'][gate]
  fs=[r['finite_difference'][0] for r in m['rows'] if r['scale']==.005]
  audit[scene]=dict(completed=True,files_hashed=len(inputs),score_max_abs_error=max_error,rollback_exact=True,optimizer_unchanged=True,hr_absent_from_locked_input_manifest=True,
    metrics_sha256=sha(p/'metrics.json'),sr_fd_median_relative_error=float(np.median([x['sr_relative_error'] for x in fs])),sr_fd_max_relative_error=max(x['sr_relative_error'] for x in fs),
    lr_mse_fd_median_relative_error=float(np.median([x['lr_relative_error'] for x in fs])),lr_l1_fd_median_relative_error=float(np.median([x['lr_l1_relative_error'] for x in fs])),
    primary_teacher_improved=sum(r['A_teacher_before']-r['A_teacher_after']>1e-7 for r in m['rows'] if r['scale']==.005),
    repeat_noise_max=max(r['repeat_render_mse_noise'] for r in m['rows']),elapsed_s=m['elapsed_s'],peak_gb=m['peak_memory_gb'],gpu=m['config']['gpu'])
  data[scene]=m
 (OUT/'audit.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False))
 lines=['# 候选二：真实 LR 能否核验一次 SR 更新——实际结果','',
 '日期：2026-09-21。保持 Wu 4DGaussians，不增密、不改骨干。两场景共 24 个 A 提议，每个提议从相同历史 SR0.1 追加 6k 检查点独立恢复；三个固定小步仅作敏感性分析，不能当 72 个独立样本。', '',
 '## 这项验证在问什么','',
 '厨师袖口的 SR 教师可能推动模型增加一条细纹。我们先只让这个观察 A 提出参数改变，再看另外两张真实 LR（B）是否也被解释得更好。要检验的不是 B 自己变清晰，而是：B 的改善能否在不知道 HR 的情况下，预示这一步对其他观察的 HR 是否有益。若能，它才可能成为训练时允许、缩小或拒绝 SR 更新的依据。', '',
 'A 当前 SR L1 梯度按参数组的当前梯度均方根及原组学习率缩放，不带入 Adam 历史，不更新原优化器。这只是固定方向的一步诊断，不等于直接复现历史训练的一步。B 是历史训练内、未参与本次梯度提议的观察，不是统计独立的最终测试。','',
 '主步幅为 0.005。全部 LR 分数、三类决策和方向文件先锁定并保存 SHA256，再读取正式 HR 标签。B 分数是两图 LR MSE 的下降；大于固定 1e−8 阈值才算支持，阈内是无分辨力。历史训练 LR L1 的同批导数和实际变化作为对照，L1 阈值为 1e−7。','',
 'cam00 已参与先前问题研究，是开发新视角。每场景的 12 个提议只覆盖 cam00 的 24/56/88 三个时刻，不能冒充完整时间窗或独立大样本。cross HR 就是 cross B 的两个相机，反映跨分辨率预测；只有 cam00 结果用于检验向新视角转移。','',
 '## 主步幅：新视角结果','',
 '表内均为更新后减去更新前。PSNR/SSIM 越高越好，LPIPS 与尺度高残差 MSE 越低越好。门控后的均值覆盖全部 12 个提议；被拒绝的提议记为零更新，避免只报接受子集的均值。尺度高残差为图像减去降采样后再放大的图，不是正交频率分解。','',
 '| 场景／规则 | 接受数 | 接受后 PSNR 真改善率 | 全提议 ΔPSNR | ΔSSIM | ΔLPIPS | Δ高残差 MSE ×10⁶ |',
 '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
 for scene,m in data.items():
  s=m['summary']['0.005']['novel'];v=s['unfiltered_mean_delta'];lines.append(f"| {NAMES[scene]}／全部更新 | 12/12 | {s['improved']}/12 | {fmt(v['psnr'])} | {fmt(v['ssim'],7)} | {fmt(v['lpips'],7)} | {fmt(v['hf_mse']*1e6,4)} |")
  for gate in GATES:
   g=s['gates'][gate];v=g['gated_mean_delta_all_proposals'];precision='—' if g['psnr_improvement_precision'] is None else f"{100*g['psnr_improvement_precision']:.1f}%"
   lines.append(f"| {NAMES[scene]}／{GATES[gate]} | {g['accepted']}/12 | {precision} | {fmt(v['psnr'])} | {fmt(v['ssim'],7)} | {fmt(v['lpips'],7)} | {fmt(v['hf_mse']*1e6,4)} |")
 lines+=['','PSNR 真改善的描述阈值为 +0.001 dB；−0.001 到 +0.001 dB 为持平。它不是统计显著性阈值。','',
 '## B 分数的预测关系','',
 '| 场景／B | 新视角 Spearman 相关 | 固定 cross 两视角 HR 的相关 | 支持／持平／退化 |', '| --- | ---: | ---: | ---: |']
 for scene,m in data.items():
  s=m['summary']['0.005']
  for gate in ['cross','far','near','random','A_train_L1_actual','A_train_L1_derivative']:
   g=s['novel']['gates'][gate];r=s['train_cross']['gates'][gate]
   lines.append(f"| {NAMES[scene]}／{GATES[gate]} | {fmt(g['spearman_psnr'],3)} | {fmt(r['spearman_psnr'],3)} | {g['accepted']}／{g['neutral']}／{g['rejected']} |")
 lines+=['','相关性只描述各场景 12 个提议的排序关系；相机和时刻有重复、存在相关性，不据此声明统计显著或泛化成功。','',
 '## 步幅、教师拟合和有限差分','', '| 场景／步幅 | 教师 L1 实际改善数 | 无筛选新视角 ΔPSNR | cross B 接受数 | cross 门控 ΔPSNR |','| --- | ---: | ---: | ---: | ---: |']
 for scene,m in data.items():
  for scale,s in m['summary'].items():
   g=s['novel']['gates']['cross'];lines.append(f"| {NAMES[scene]}／{scale} | {s['teacher_improved']}/12 | {fmt(s['novel']['unfiltered_mean_delta']['psnr'])} | {g['accepted']}/12 | {fmt(g['gated_mean_delta_all_proposals']['psnr'])} |")
 lines+=['','原 smoke 的 0.1/0.4 步幅使 A 教师自身误差上升，已经保留为过冲证据。正式小步调整只依据 A 监督数值；smoke 自动产生过 HR，团队中检查明细时曾看到该标签，因此不宣称 smoke 完全盲评。正式两场景的 LR 决策锁定后才进行 HR 评价，期间没有据 HR 改变协议。','',
 '| 场景 | SR 导数中心差分中位相对误差／最大 | A-LR MSE 导数中位误差 | A-LR L1 导数中位误差 | 用时 | 峰值显存 |', '| --- | ---: | ---: | ---: | ---: | ---: |']
 for scene,a in audit.items():
  lines.append(f"| {NAMES[scene]} | {a['sr_fd_median_relative_error']:.2%}／{a['sr_fd_max_relative_error']:.2%} | {a['lr_mse_fd_median_relative_error']:.2%} | {a['lr_l1_fd_median_relative_error']:.2%} | {a['elapsed_s']:.2f} s | {a['peak_gb']:.2f} GB |")
 lines+=['','中心差分使用 ±0.001；完整 ±0.005 结果保存在每条记录。L1 尖点、可见性和渲染数值使有限步不必严格等于一阶预测；导数误差按实际值报告，不据参数哈希通过就声称梯度全都精确。所有门控仍使用真实重渲染的有限步 LR 误差。','',
 '## 可复核边界和原始产物','',
 '- 两场景正式输入、源码、方向与结果的 SHA256 已独立复查；重算 B 分数和三类决策，核查参数精确回滚及优化器哈希不变。','- 每场景分别在同一硬件比较前后，未把两个 GPU 的细微绝对差异当方法收益。','- 未运行门控长期训练；本轮只检验选择信号，不能据此声称收敛质量提高。','- 未验证互补子像素相位选择；far 只表示时间间隔，cross 只表示不同训练相机。','- 所有 B 都看不见的 LR 零空间伪细节仍可能通过。即使预测有用，也不是真实性证明。','',
 '- [预先固定协议](dynamic_sr_candidate2_protocol_2026-09-21.md)','- [正式执行源码](../experiments/dynamic_sr_20260921/candidate2_update_validation_v2.py)','- [独立完整性与评分审计](../output/dynamic_sr_20260921/candidate2_assessment_v2/audit.json)','- [菠菜逐提议数据](../output/dynamic_sr_20260921/candidate2_cook_v2/metrics.json)','- [会议室逐提议数据](../output/dynamic_sr_20260921/candidate2_meeting_v2/metrics.json)','']
 report=ROOT/'docs/dynamic_sr_candidate2_findings_2026-09-21.md';report.write_text('\n'.join(lines))
 print(json.dumps(audit,indent=2,ensure_ascii=False))
if __name__=='__main__':main()
