# 面积采样下的 Multi4D 模型验证

2026-10-18授权更新：用户已允许使用显卡开展训练。当前入口为 `broker_suite.py`，按目标GPU的进程、有效预约、显存和可核验身份准入，不再等待所有Claude对话闲置。不联系Claude，不清锁或修改共享政策；优先保障已有任务和预约，PRO6000全局最多一个实际训练进程，5090 GPU禁训。

新路径通过现有 `claude-training` broker 逐阶段执行。项目内 `runtime_v2` 独立保存固定源码与原生环境，`plan_v2.json` 为新计划位置，`run_v2` 保存本轮阶段输出；详见下文。旧 `run_suite.py`、`idle_gate.py` 的保守默认行为和旧服务记录保留为历史入口。CPU部署、资源准入和真实CUDA验收分别记录；真实CUDA仍待验收，当前没有本轮Multi4D完整GPU评价结论。

## 验证什么

当前面积采样保留了不同视角的混叠差异。同一份 LR 能否转化成新视角细节，还取决于重建模型。这里比较 Multi4D 和现用 Wu 4DGaussians，并把三个问题分开：合法流程的绝对质量；匹配 SR 教师相对纯 LR 的增益；换成训练视角真实 HR 教师后仍能提高多少。

旧 Multi4D 实验来自 2026-09-24 的 bicubic 抗混叠短窗，只有原帧 0、2、…、118，并额外留出 cam01。其 M0/M1 结果不能作为本次 area4、全300帧、cam00 单独留出的结果。旧代码和完整状态恢复机制可复用，旧权重与指标不混入本轮表格。

## 固定输入与三个内控

- 首景为 `cut_roasted_beef_area4`：HR 1344×1008、LR 336×252；300 帧 0–299；时间为 frame/300。cam00 测试，其余实际存在的19台相机训练，缺 cam04，包含 cam01。
- LR 为每个 HR 的4×4块均值，在8位 sRGB上求均值后用 `np.round` 量化；不加额外模糊或噪声。训练时对 HR 渲染做可微的4×4平均，不对预测量化。
- Multi4D 固定上游 `c483e82cd8daa1b4fdc460ed45882fc2fe7a22b0`，保留静态、持续动态和瞬态三分支，以及原生阶段、损失和控制器；适配与必要修复由补丁、源码哈希和输入清单披露，不能称完全未修改的官方复现。
- 计划共用一份 coarse 结束的 `fine0.pt`，三个分支各执行20000次 fine 更新，逐步使用同一登记采样；是否完成以各臂终态回执为准。初始化只读训练相机的第0帧 LR。

| 内控 | 可读训练监督 | 用途 |
| --- | --- | --- |
| `areaL0` | 面积 LR；保留 Multi4D 原生其他损失 | 测该流程只靠 LR 能达到什么水平 |
| `areaTm` | 同一面积 LR＋冻结、按面积退化匹配的 SwinIR 教师 | 合法 SR 流程；默认教师权重0.2、D-SSIM混合系数0.2，以 spec 为准 |
| `areaOracle` | 同一面积 LR＋训练相机真实 HR 教师 | 诊断先验质量余量，禁止作为合法方法排名 |

`areaTm − areaL0` 是教师收益。`areaOracle − areaTm` 是更好教师可能触及的剩余空间。这两项不能互相替代。原生拓扑控制器会响应梯度，三个分支最终点数可能不同，因此这里是完整训练流程对照。

## Wu 参照与比较边界

`assess.py`读取已识别的完整评价文件并记录SHA256，分别比较匹配教师强基线、加入自标定的当前完整方法，以及各模型内部的教师收益和oracle差距。研究数值和汇总仅保存在本地，不随源码发布。缺失结果保持null。

Wu内部收益应使用同硬件配对，避免把跨硬件差异归因于教师。Wu参照包含基于训练相机的posthoc cull；Multi4D保持原生三分支推理。两者排名是完整系统比较，包含训练日程、初始化、容量和后处理差异，不能把差值全部归因于骨干。

## 运行范围与停止条件

新入口为 `broker_suite.py --plan output/dynamic_sr_multi4d_area_20261017/plan_v2.json`，计划须声明 `admission_policy=target_gpu_broker`。它使用项目内 `runtime_v2/Multi4D-c483e82/.venv/bin/python` 和独立上游副本，将每个已准入阶段交给现有root broker。新计划、运行输出和runtime均位于 `output/dynamic_sr_multi4d_area_20261017/` 下，分别使用 `plan_v2.json`、`run_v2/`、`runtime_v2/`；计划存在、服务登记和GPU执行成功须分别以实际记录确认，路径约定不构成启动证明。

此前的 `plan.json` 与服务名 `codex-multi4d-area-20261017` 属于旧等待入口记录。直接使用旧 `run_suite.py` / `idle_gate.py` 时，仍要求所有Claude会话明确闲置并检查本地与远端资源；这个历史默认没有被改写。新broker入口通过 `target_gate.py` 只审查目标GPU，不用其他对话或远端主机的忙闲代替目标卡准入。

1. 目标GPU需通过至少间隔20秒的两次资源核对，确认没有其他计算进程或有效目标卡预约、显存满足预算，且主机视图、GPU UUID与型号可核验；未知状态拒绝准入。最终原子准入和预约归现有root broker管理。先做真实CUDA冒烟与状态恢复检查，任一项失败则保存错误，不继续正式训练。
2. 先完成 Cut 的三臂、评价及完整性核验。训练退出后直接串接评价，不用模型轮询发现正常完成。
3. Cut 三臂完整后，若下列任一**差值绝对值**达门槛，则在 Cook 用相同三臂确认：Multi4D `areaTm` 相对 Wu `areaTm` 的 PSNR差至少0.05 dB；LPIPS差至少0.003；两模型教师收益的PSNR差至少0.05 dB。Multi4D胜出或落后都触发确认。
4. 若三项均低于门槛，记录“该开发协议下差别小”，不扩大。缺失、失败、评价不完整不充零，也不被解释为方法持平。
5. 最多 Cut＋条件触发的Cook；不扩展更多场景，不做参数搜索。若目标GPU出现其他工作、有效预约或不可核验状态，控制器请求自身阶段保存完整状态并退出，随后核验原生子进程退出、broker终态与预约释放；不得抢占或清理别人的工作。只有重新通过目标卡准入才继续。其他Claude对话活跃本身不触发新入口让卡，真实CUDA下的暂停、恢复与释放链仍待验收。

## 文件与验收边界

| 文件 | 作用 |
| --- | --- |
| `prepare_inputs.py`、`data.py` | CPU清单、纯LR初始化、采样与当前相机/时间接口 |
| `build_adapter.py` | 生成可审阅的面积观测补丁和来源记录 |
| `train.py` | 三臂、完整状态恢复、暂停保存；GPU执行入口 |
| `evaluate.py` | 同Wu空间指标、全300帧cam00、训练各相机0/150/299固定诊断、帧差时序、固定ROI与预览 |
| `assess.py` | 原始Wu身份、绝对质量、教师收益及oracle差距；缺失明确记为 incomplete |
| `idle_gate.py`、`run_suite.py` | 历史保守入口：全部Claude会话闲置、本地与远端资源核对；默认保持不变 |
| `target_gate.py` | 新入口的只读目标GPU核对：UUID/型号、计算进程、有效预约、显存及主机视图；未知拒绝准入 |
| `broker_suite.py` | 新CPU控制器：两次目标卡核对，逐阶段提交现有root broker，核验自身作业身份与终态，继承退出触发评价和有界场景确认 |
| `stage_wrapper.py` | broker内校验worker UID、cgroup、冻结源码及原生环境，只监管自身原生子进程、暂停和超时，并保存回执 |
| `make_render_comparison.py` | CPU读取已有summary与 `rgb_raw` NPZ，统一转换同帧整图和旧固定ROI，记录checkpoint/manifest/NPZ哈希、GPU与cull；缺结果记incomplete，缺真实连续帧缓存记pending，盲评accepted保持false |

已完成的CPU检查包括空间指标函数一致性、Cut完整划分、旧固定ROI参考HR与当前像素的SHA一致性，以及缺失结果保持null。独立runtime的源码、环境文件身份和内部符号链接已核验，原生目标Python可导入torch、numpy、mmcv、PIL及 `simple_knn._C`，CUDA没有初始化。真实hybrid扩展加载会触发JIT编译，已单列为待受控smoke验收，不能以namespace导入冒充通过。

`make_render_comparison.py` 默认展示cam00的0/150/299三个离散时间点；只有实际缓存齐全时才组合149/150/151连续帧。旧固定ROI名称只描述原HR第40帧的内容，不保证每帧仍是完整人脸或可辨文字。已有Wu图可作CPU核对材料；Multi4D缺失时不能标比较完成，也不产生替代预测。当前材料带方法标签，人类盲评尚未进行，accepted=false。

上述CPU检查不代表目标卡broker实际准入、CUDA训练、暂停恢复或最终评价已经通过。真实CUDA仍待验收，本轮尚无Multi4D完整PSNR、SSIM、LPIPS或胜负结论。
