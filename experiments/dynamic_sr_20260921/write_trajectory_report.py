#!/usr/bin/env python3
"""Write evidence report without transcribing numbers manually."""
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'output/dynamic_sr_20260921'
read=lambda p:json.loads(p.read_text())
KEYS=['psnr','ssim','lpips_alex']
NAMES={'cook_spinach':'菠菜','cut_roasted_beef':'牛肉','meetroom_discussion':'会议室'}


def triple(m):return f'{m["psnr"]:.4f} / {m["ssim"]:.6f} / {m["lpips_alex"]:.6f}'
def change(a,b):return ' / '.join(f'{b[k]-a[k]:+.6f}' for k in KEYS)
def trainmean(v,mode):
    rr=[r[mode] for r in v['rows'] if r['group']!='novel_cam00']
    return {k:float(np.mean([r[k] for r in rr])) for k in KEYS}


def main():
    matched=read(OUT/'resolution_trajectory/metrics.json')
    native=read(OUT/'native_resolution_trajectory/metrics.json')
    full=read(OUT/'native_resolution_trajectory/cam00_full60/metrics.json')
    closed=read(OUT/'resolution_trajectory/lr_only_cam00_full60/metrics.json')
    total=sum(x['elapsed_seconds'] for x in [matched,native,full,closed])
    lines=['# LR、SR 与 HR 随迭代的训练—新视角轨迹核验','',
        '日期：2026-09-21。全部为现有检查点评价，没有重训、参数更新或更换骨干。', '',
        '## 1. 结论和应当修正的叙述','',
        '**训练观察继续改善而新视角 PSNR/SSIM 下降，并非只在 SR 监督下发生；但“所有分辨率都会发生”同样超出了证据。** 共同父检查点后的 LR-only 分支已出现训练 LR 拟合提高、新视角下降；独立的完整 HR 训练在菠菜后段也出现 PSNR/SSIM 下降。另一方面，原生 LR 训练在厨房两场景的全 60 帧后段均值仍改善，牛肉完整 HR 也改善。', '',
        '因此，当前更准确的问题是：既有骨干与数据条件下存在与观察覆盖、优化阶段、成像路径及目标有关的泛化限制，SR 会改变该限制的具体表现；不能把退化现象本身称为 SR 独有机制，也不能把任一时段下降称为普遍必然。PSNR/SSIM 下降而 LPIPS 改善时，应逐指标陈述，不能概括成“三指标全面恶化”。', '',
        '用户所指“测试视角上涨、新视角下降”在此区分为：**参与拟合的训练相机**与**未参与拟合的 cam00 相机**。cam00 已在此前研究中反复用于诊断，属于开发证据，不能作为独立最终验证集。', '',
        '## 2. 两套实验协议不能混在一起','',
        '| 协议 | 训练数据与成像 | 起点、步数 | 能回答什么 |',
        '|---|---|---|---|',
        '| 同父检查点 LR-only | 全训练相机真实 LR；以 HR 栅格渲染后 bicubic 抗混叠回降得到 LR | 同一采样适配后父状态；新增 1.2k/6k/12k/18k；无增密 | 当前 SR 干预协议中，去掉 SR 后是否仍有泛化下降 |',
        '| 同父检查点 SR0.1 | 上述 LR + 四训练相机的冻结 SwinIR SR，权重 0.1 | 同上 | 与此前现象直接对照 |',
        '| 同父检查点 HR oracle | 上述 LR + 四训练相机真实 HR，权重 1 | 同上 | 用真实教师替换 SR 后的诊断；**不是所有相机 HR 原生训练** |',
        '| 从初始化的 native LR / full HR | 所有训练相机分别使用原生 LR 栅格 LR 监督，或原生 HR 栅格真实 HR 监督 | 1k coarse + fine 2k/4k/6k；配置记录相同 seed 与训练规模 | 两种实际原生训练的阶段性表现；不可与以上新增 18k 曲线拼接 |','',
        '特别注意：旧 `hr_oracle_w10` 与 `sr_w01` 同时改变教师类型与权重（1 对 0.1），不能把两者差异仅归于教师真假。native LR / full HR 虽然现有配置与实现记录的增密阈值和触发上限相同，实际点数不同；旧原生 LR 完整源码快照缺失，也不能把当前代码一致当作历史源码完全一致的证明。这是可运行原生设置的趋势比较，不是仅改变分辨率、其他变量完全锁定的因果试验。详见[分辨率因果审计](/home/cai_tianshun/Project/4dsr/docs/dynamic_sr_resolution_causal_audit_2026-09-21.md)。', '',
        '四帧主诊断固定 0、40、80、118，先于本次指标选取。每场景按旧先验配置取四个训练相机（A 组）；其余训练相机按 ID 取首、较小中位和末尾三个（B 组）；C 组为 cam00。A/B 名称在原生协议中仅表示相机位置分层，两个组都接收各自完整的 LR 或 HR 监督。下文训练均值仅为这七个所选相机，不冒充全部训练相机。', '',
        '- 菠菜、牛肉：A = cam02/06/12/18；B = cam03/11/20。',
        '- 会议室：A = cam02/04/08/12；B = cam03/07/11。',
        '- 三指标顺序统一为 **PSNR / SSIM / LPIPS-Alex**，前两项越高越好，LPIPS 越低越好。逐图平均，不把不同尺寸 LPIPS 的绝对值横向解释成难度大小。', '',
        '## 3. 当前 LR-only 干预：在它实际优化的 LR 成像路径上判断','',
        'LR 图来自 `D(Render_HR)`，D 是训练所用 bicubic、antialias=True 回降与裁剪。不是另换成原生 LR 栅格；后者在逐图 JSON 中独立保存，不混进本节。下面训练层为同四帧，所有三场景、A/B 两组 LR 三指标均改善。', '',
        '| 场景 | 所选训练相机 6k | 所选训练相机 18k | 三指标变化 |',
        '|---|---|---|---|']
    for scene,cases in matched['scenes'].items():
        a,b=[trainmean(cases['lr_long'][s],'lr_integrated') for s in ['6000','18000']]
        lines.append(f'| {NAMES[scene]} | {triple(a)} | {triple(b)} | {change(a,b)} |')
    lines += ['', '为控制四帧抽样风险，对 LR-only 的 cam00 在相同 6k/18k 端点补齐全部 60 个时间点。此处不择帧、不改窗口、不据评价挑检查点。','',
        '| 场景／cam00 全60帧 | 6k | 18k | 三指标变化 |','|---|---|---|---|']
    for scene,steps in closed['scenes'].items():
        a,b=[steps[s]['mean'] for s in ['6000','18000']]
        lines.append(f'| {NAMES[scene]} | {triple(a)} | {triple(b)} | {change(a,b)} |')
    lines += ['', '全 60 帧的 PSNR/SSIM 在三个场景均下降；LPIPS 在两厨房略改善、会议室恶化。这将“是否只有 SR 才引起观察外退化”的问题，放回真实 LR 目标空间检查，而不是只看 LR 模型放大后对 HR 的误差。它仍不能单凭指标指出几何、运动、颜色哪一项是真正原因。', '',
        '## 4. 同一批检查点的 HR 输出：与 SR0.1 和稀疏 HR oracle 比较','',
        '为保持相机与时间样本一致，本表均使用 cam00 固定四帧。1.2k、6k、12k、18k 全曲线均保留；6k→18k 是此前问题所对应窗口，不按每个分支另选最佳起点。','',
        '| 场景 | 分支 | 6k，HR评价 | 18k，HR评价 | 三指标变化 |','|---|---|---|---|---|']
    for scene,cases in matched['scenes'].items():
        for case,steps in cases.items():
            a,b=[steps[s]['groups']['novel_cam00']['hr'] for s in ['6000','18000']]
            lines.append(f'| {NAMES[scene]} | {case} | {triple(a)} | {triple(b)} | {change(a,b)} |')
    lines += ['', 'LR-only 在 HR 图像上的训练指标并非始终提高。例如会议室训练 HR 质量降低，但训练 LR 三指标提高。这说明模型可以越来越符合低分辨率观察，同时放大的细节越来越差；不能把这两种评价混为“训练拟合”。', '',
        '旧的两厨房 cam00 全 60 帧 HR 评价也可核验：LR-only 菠菜 PSNR 30.0936→29.6506，牛肉 31.0034→30.4099，SSIM 与 LPIPS 同时恶化。这些旧评价已通过检查点/manifest 哈希、当前四帧重新渲染指标核对后关联，未重新计算或挑选帧。','',
        '![相同父检查点的 HR 输出轨迹](/home/cai_tianshun/Project/4dsr/output/dynamic_sr_20260921/resolution_trajectory/curves_hr.png)','',
        '## 5. 真正的原生 LR 与完整 HR：固定后段 4k→6k','',
        '另查两个厨房从相同数据初始化的原生训练。表中 native LR 直接在 LR 栅格上渲染、对真实 LR 评价；full HR 直接在 HR 栅格上渲染、对真实 HR 评价。所有四分支均检查同一 fine 4k→6k 窗口；并保留此前 2k 点，避免只展示下降段。两个时刻每个分支点数保持一致，因而该窗口内没有增密事件变化。','',
        '| 场景 | 原生分支／训练七相机四帧 | 4k | 6k | 三指标变化 |','|---|---|---|---|---|']
    for scene,cases in native['scenes'].items():
        for case,steps in cases.items():
            mode='lr_native' if case=='native_lr' else 'hr'
            a,b=[trainmean(steps[s],mode) for s in ['4000','6000']]
            lines.append(f'| {NAMES[scene]} | {case} | {triple(a)} | {triple(b)} | {change(a,b)} |')
    lines += ['', '**cam00 的全 60 帧确认：**','',
        '| 场景 | 原生分支／自身监督分辨率 | 4k | 6k | 三指标变化 |','|---|---|---|---|---|']
    for scene,cases in full['scenes'].items():
        for case,steps in cases.items():
            a,b=[steps[s]['mean'] for s in ['4000','6000']]
            lines.append(f'| {NAMES[scene]} | {case} | {triple(a)} | {triple(b)} | {change(a,b)} |')
    lines += ['', '**必须保留的反例与修正：** 菠菜 native LR 的预固定四帧 PSNR 从 33.4698 降至 33.3066，但全 60 帧从 33.5320 升至 33.6496，SSIM/LPIPS 也改善；PSNR 改善 46/60 帧，SSIM 改善 52/60，LPIPS 改善 57/60。不能以四帧结果声称原生 LR 整个短窗下降。这里保留原四帧表和全 60 帧确认，不删掉相反证据。', '',
        '菠菜 full HR 的全 60 帧 PSNR/SSIM 下降则仍存在：53/60 帧 PSNR 下降，51/60 帧 SSIM 下降；LPIPS 在 53/60 帧改善。因此它支持完整 HR 原生重建也可能出现指标分歧和 PSNR/SSIM 泛化下降，不支持“全部感知质量退化”。牛肉原生 LR 和 full HR 的 60/60 帧三指标都改善。', '',
        '原生训练 2k→6k 的总体趋势仍是两场景、两分辨率均明显进步；后段行为不能外推到整个训练。菠菜 LR 点数 110810、HR 121884；牛肉 LR 110557、HR 121148（4k/6k）。配置 max_points=120000 在旧实现是触发增密前的检查而非硬点数上限，HR 分支越过该值是原实现行为，未通过本次评价改动。','',
        '![从初始化的原生 LR 与完整 HR 轨迹](/home/cai_tianshun/Project/4dsr/output/dynamic_sr_20260921/resolution_trajectory/curves_native_protocol.png)','',
        '## 6. 对后续研究判断的影响','',
        '1. **不再把“训练上涨、新视角下降”本身作为 SR 特有创新动机。** 已有 LR-only 和完整 HR 反例；应研究 SR 在何处改变了错误类型、恢复能力或泛化代价。',
        '2. **保留动态局部多帧验证的价值。** 如果真实互补 LR 能支持新细节，需检验它是否在相同输入/预算下提高未参与提议观察的质量，而不是只增加训练纹理拟合。当前曲线没有证明多帧方法一定有效，也没有否定该方向。',
        '3. **成像路径必须明确。** 当前长分支 LR-only 以 HR 栅格回降训练，早期 native LR 直接以 LR 栅格训练；二者迭代阶段、容量与学习历史不同，不能从结果差异推出哪一种栅格因果更好。若未来专门评价抗混叠，需重新固定共同起点和训练预算。',
        '4. **优先减少归因错误，不立即长训穷举。** 本轮已有足够反证否定 SR 唯一性；无须为证明“所有原生设置都会下降”增加训练。若需“LR 放大某类问题”的论文主张，则另做相同父状态、容量、目标路径与充分预算的分辨率干预，或缩小主张。',
        '5. **不要把短时指标变化解释为真实几何改善/退化。** 这里没有运动或几何真值，也没有独立随机种子重复；相邻帧相关，不能把 60 帧当 60 个独立场景统计。','',
        '## 7. 完整性、成本与复现','',
        f'- 36 个同父检查点、12 个原生检查点共 1536 个固定四帧模型—观察组合；另补原生 8 个端点的 480 个全时域组合及 LR-only 6 个端点的 360 个组合。总计 2376 次模型—观察评价，部分四帧重叠用于校验，不宣称独立样本。',
        f'- 累计评价脚本墙钟时间 {total:.1f} 秒；GPU0 NVIDIA RTX PRO 6000，PyTorch 2.7.1+cu128；最高记录 allocated 显存约 {max(matched["peak_gpu_gb"],native["peak_gpu_gb"]):.3f} GB（不等于整卡占用）。没有新增训练或后台长任务。',
        '- 660 个旧 HR 逐图记录重新渲染核对：PSNR、SSIM、MSE 最大绝对差均为 0，LPIPS 最大差 5.96×10⁻⁷。全 60 帧与原四帧重叠记录全部通过 10⁻⁶ 以内检查。',
        '- 输入图像、manifest、源配置、检查点和执行代码哈希均记录；场景选择及相机位置未按 HR 指标筛选。完整路径和每幅图分数保存在下列 JSON。','',
        '- [同父分支全部指标](/home/cai_tianshun/Project/4dsr/output/dynamic_sr_20260921/resolution_trajectory/metrics.json)',
        '- [同父分支 LR-only 全60帧确认](/home/cai_tianshun/Project/4dsr/output/dynamic_sr_20260921/resolution_trajectory/lr_only_cam00_full60/metrics.json)',
        '- [从初始化原生 LR/HR 全部指标](/home/cai_tianshun/Project/4dsr/output/dynamic_sr_20260921/native_resolution_trajectory/metrics.json)',
        '- [原生 LR/HR 全60帧确认](/home/cai_tianshun/Project/4dsr/output/dynamic_sr_20260921/native_resolution_trajectory/cam00_full60/metrics.json)',
        '- [完整逐步表格](/home/cai_tianshun/Project/4dsr/output/dynamic_sr_20260921/resolution_trajectory/tables.md)',
        '- [同父分支评价脚本](/home/cai_tianshun/Project/4dsr/experiments/dynamic_sr_20260921/trajectory_eval.py)',
        '- [原生分支评价脚本](/home/cai_tianshun/Project/4dsr/experiments/dynamic_sr_20260921/native_trajectory_eval.py)',
        '- [原生全60帧补验脚本](/home/cai_tianshun/Project/4dsr/experiments/dynamic_sr_20260921/native_full60_eval.py)',
        '- [LR-only全60帧补验脚本](/home/cai_tianshun/Project/4dsr/experiments/dynamic_sr_20260921/integrated_full60_eval.py)',
        '- [本报告生成脚本](/home/cai_tianshun/Project/4dsr/experiments/dynamic_sr_20260921/write_trajectory_report.py)','']
    path=ROOT/'docs/dynamic_sr_resolution_trajectory_2026-09-21.md'
    path.write_text('\n'.join(lines))
    print(path)


if __name__=='__main__':main()
