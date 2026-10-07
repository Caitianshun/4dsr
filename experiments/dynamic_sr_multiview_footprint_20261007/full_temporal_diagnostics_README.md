# 完整原生评价的时序残差诊断

当前只有小数组 CPU 合同验收，尚未生成任何生产时序结果。该入口不运行模型、不训练、不调用 GPU，也不创建 ROI、光流或遮挡分层。根代理验收后才能将它接到真实完成事件。

本诊断回答一个有限问题：相邻帧的重建误差是否在变化。设正式评价的截断渲染为 `clamp(raw_render, 0, 1)`，误差为 `e_t = clamped_render_t - HR_GT_t`。主结果计算全部 RGB 像素上的 `mean((e_t - e_(t-1))²)`，生成 299 个相邻帧结果与等权平均。残差以正式浮点图像的 float32 方式相减，再以 float64 计算差平方及均值。单独输出未截断 raw 残差结果和每帧空间误差 MSE，避免把截断收益或持续偏差隐去。

例如，一直存在同样的偏差会得到零时序残差变化，但空间误差仍然非零。误差正负交替则会得到正值。该量比较相同图像位置，未对运动建立对应关系。因此它反映误差闪动及真实变化的重建偏差，不能证明几何正确、感知时序稳定或运动补偿后的一致性。帧和相邻帧对存在相关性，标准差只描述结果分布，不是独立样本置信区间。

入口要求已完成的 `full_evaluate_registered.py` cam00×300 原生评价。先核验 complete、summary、float_index、protocol、native checkpoint sidecar、计划、源码和模型/RNG不变证据；再核验原始 300 个浮点文件的 SHA256。只在这些检查通过后读取 GT，并在读取每帧时核验浮点文件及 GT 的原 SHA。输出完成之前再次核验冻结身份和全部选中浮点文件。不会读取检查点张量、训练像素或 LR 像素。

Cook、Cut、MeetRoom discussion/vrheadset 已用于开发。两确认场景 Coffee/Flame 必须额外提供 `root_frozen_single_main_configuration_before_confirmation_HR`：绑定原 development selection、单一 main method、配置文件、四开发场景×两个种子的完整 native SR6000/test300 证据和统一评价硬件，并声明确认 HR 未用于选择。≤2 候选的开发选择不能替代此冻结。确认诊断只允许最终主方法及其已登记基线。不存在真实冻结与完整结果时严格拒绝。

```bash
CUDA_VISIBLE_DEVICES='' /home/cai_tianshun/Project/4dgs/.venv/bin/python \
  experiments/dynamic_sr_multiview_footprint_20261007/full_temporal_diagnostics.py \
  --evaluation-complete /ABS/FULL_NATIVE_EVALUATION/complete.json \
  --manifest /ABS/FULL_TRAIN_READY_MANIFEST.json \
  --out output/dynamic_sr_multiview_footprint_20261007/full_temporal_diagnostics/NEW_LABEL
```

确认场景另传 `--final-main-freeze /ABS/ROOT_FROZEN_MAIN.json`、`--confirmation-authorization /ABS/PREAUTH.json` 和 `--confirmation-launch-proof /ABS/OWNED_CHILD_EXIT.json`。不能在确认评价之后补一份选择冻结，就宣称此前未使用确认质量。原冻结 evaluator 在空目录内自行创建 `registration.json`，所以采用两阶段运营证据：实际 backend 在调用前先写授权与期望 registration 字典；真实子进程结束后绑定实际 registration，并要求两者逐项相等。授权还绑定最终配置、原 protocol/checkpoint/evaluator 源码、后台源码、经理 spec 和精确 unit/invocation/PID/start/host/boot 身份。后续证明须有同一经理的单调写入→spawn→wait 顺序、原 argv/子进程 PID/start 和真实 child Exit0。经理和评价子进程必须同 host、同 boot，不能比较跨机单调时间。所有证据引用必须通过 SHA。后台必须在原 evaluator 调用前核验授权、来源和哈希。当前只有 CPU 假合同，不能证明任何真实生产前置，也未生成确认结果；真实后台及根验收缺失时不执行此路径。

上述路径是接口示例，不表示已有生产计划或已派发。输出为 `registration.json`、`per_adjacent_frame.csv`、`spatial_frame_mse.csv`、`summary.json` 和 `complete.json`。已有输出目录或部分历史不会覆盖；失败保存独立错误与已完成读取计数，未完成操作成本记为未知。时序后处理时间及读取成本单列，不计入正式训练预算。

合同入口 `full_temporal_diagnostics_checks.py` 只使用小型假数组与临时 JSON 身份文件。覆盖完美重建、固定偏差、交替误差、截断/未截断差异、缺帧/重复/形状/非有限值、错误角色、未冻结确认和哈希/路径拒绝。没有真实 PNG/NPZ/PT、模型或资源访问。
