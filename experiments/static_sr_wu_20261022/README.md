# 静态超分方法到 Wu 4DGS 的动态迁移

本目录准备四个明确标注为 **dynamic adaptation** 的比较分支：SplatSuRe、IE-SRGS、Sequence Matters、3DSR。不含 SR4D。静态三维高斯替换为 Wu 843d5ac 的规范高斯与时间形变网络；各时间共享同一套参数。不能称为未修改作者实现，也不能把 CPU 检查、CUDA 编译或初始化成功称为实验结果。

## 已实现的连接

| 方法 | 保留的关键机制 | Wu 连接 | 作者日程 |
| --- | --- | --- | --- |
| SplatSuRe | 实际像素贡献标记、未扩张投影半径、跨视角 fidelity、closest-view 修正、加权 SR | 冻结 LR Wu，在每个时间独立计算训练相机权重图；重新初始化 SR Wu | LR 30k + fresh SR 30k |
| IE-SRGS | 内部多视角重建、外部 SwinIR 与 Depth Anything V2、channel mask、median depth | 同时间 3 训练视角的内部 Wu+Mip；训练相机 HR 先验；fresh final Wu+Mip | internal 30k + fresh final 30k |
| Sequence Matters | PSRT、视角夹角排序、ORB 排名、ALS 自适应序列 | 每个真实时间的训练相机组成伪视频；不同时间不混排 | PSRT 准备 + fresh SR 30k |
| 3DSR | StableSR-Turbo 逐轮去噪、共享三维表示反馈、Mip 过滤 | LR Wu+Mip → 四轮目标更新与共享 Wu 重建；每轮回渲全部训练 camera/time | LR 30k + 4×5k |

SplatSuRe 主论文使用 StableSR。`splatsure_teacher.py` 提供论文支持的 **SwinIR 变体**，必须单独命名；不能用它冒充 StableSR 主配方。主配方的外部生成桥接与登记见方法文件和来源审计。本机六场景主教师计划为 `output/static_sr_wu_20261022/splatsure_stablesr_plans_v4/<Scene>/plan.json`；重新规划时显式指定 `--weight-root weight/static_sr_wu_20261022/stablesr --runtime-assets output/static_sr_wu_20261022/splatsure_stablesr_assets/runtime_assets_v2.json`，不要沿用旧的默认占位权重目录。

3DSR 源码在跨图/patch 循环中复用最后一个 `x_T`，且没有持久化下一轮状态。本适配按论文 Algorithm 1 为每个 scene/camera/time/patch 保存独立状态；同时修复反馈图的 padding/crop 次序。必须标为论文机制修复适配版，并保留源码差异。它不能退化为一次性离线图像 SR 缓存。

StableSR 外层分块适配：CTS `diffusion_v3` 在 Cook 完成 200 步后，作者 `scripts/util_image.py:797` 将高度 1024 的矩形 patch 与固定 1280×1280 高斯权重相乘而失败，尚未生成教师图。作者 [StableSR README:179–180](https://github.com/IceClear/StableSR/blob/398ee9383777e255540ea027a704c8ce1f32145b/README.md#L179-L180) 明确允许调整 `--vqgantile_size/stride`，要求 tile≥512、stride<tile，并提示小 tile 可能增加边缘伪影。[SplatSuRe README:36](https://github.com/pranav-asthana/SplatSuRe/blob/df40abb8d92a02656268cb0a0b97acc2495b9307/README.md#L36) 只指定采用 StableSR 生成 SISR 输入，没有公开固定 1280 的调用约束。因此固定采用几何规则：仅当原 1280 阈值触发分块时，`tile=min(1280,padded_height,padded_width)`，`stride=min(1000,tile−32)`；否则保留原不分块路径。该规则不依赖指标，不额外补边到 1280，不修改作者源码。这里的外层 tile 是图像空间编码/解码分块；**扩散 `input_size=512`、latent tile=64、overlap=32、DDPM200、模型/权重均保持原样**。配方标识保留原主族标识，具体接口适配由 plan/receipt 的 `tiling` 字段明确区分；旧计划必须重新生成。

| 场景 | LR 尺寸 W×H | StableSR 实际补边 H×W；tile/stride | 3DSR 实际补边 H×W；tile/stride |
| --- | --- | --- | --- |
| Cook、Cut、Coffee、Flame | 336×252 | 1024×1376；1024/992 | 1024×1344；原1280/1000，矩形边块 |
| Discussion、VRHeadset | 320×180 | 736×1312；736/704 | 736×1280；保留不分块 |

两条路径补边不同：StableSR 原脚本在任一轴未对齐32时将两轴都推进到下一个32；3DSR 已披露适配采用最小 modulo32 补边。3DSR 的作者分块器 `utils/util_image.py:797–803` 已将高斯权重裁到实际矩形 patch，因此没有相同累加错误；保持原1280/1000分块参数和 Turbo 四步/四轮机制；每个观测的 `tiling` 和 patch region 写入 latent 状态，后续轮次必须完全一致。CPU 测试使用原作者分块器验证全覆盖、形状和四轮状态，不代表真实扩散图像质量已通过。最终 Cook 输出仍为1344×1008，MeetRoom 输出仍为1280×720；实际效果及边缘伪影需真实 GPU 教师验收与后续评价。

3DSR 每轮 5k 更新采样少于 Cook 的 6000 个训练观测。因此单轮不保证所有 camera/time 都被采样。这是作者迭代预算用于动态数据后的覆盖率差异，结果解释和后续匹配预算对照必须记录。

## 数据和比较协议

`prepare_suite.py` 默认注册当前全部六场景：Cook07、Cut07、Coffee07、Flame07、Discussion07、VRHeadset07；统一使用 area4 输入、0–299 帧、cam00 留出。`prepare.py` 保留单场景入口。相机内参、外参、时间及逐图 LR SHA256 保留。六场景分别有 6000、5700、5100、6000、3600、3600 个训练观测，共 30000 个；不假设相机数量或图像分辨率相同。HR 路径只供独立评价器读取。训练和缓存准备拒绝 cam00、缺帧、跨场景检查点和不匹配的 LR 来源。

骨干设置来自当前场景 Wu 原始配置。新的参数从该场景 LR 点云初始化，单独训练；不继承项目 gateaff、w4、裁点、MVFB 等模块。原始静态方法的单视角采样改为 camera/time 采样；IE 内部的三个视角固定在同一时间。Wu 的时间平面正则和形变参数继续更新。高斯的 clone/split/prune 日程采用源 3DGS 的 500/100/15000 设置（3DSR SR 每轮截至 2500）；具体张量与拓扑维护由 Wu 完成。

动态适配还有明确边界：Mip 滤波在变形后的同时间状态重算；opacity reset 使用 Wu 规范高斯 opacity，未静默声称与静态 Mip-aware reset 完全等价。背景沿当前 Wu 场景配置。高斯和形变学习率共享明确的 30k 时钟。IE 内部每步损失求和，拓扑统计保留作者“最后一个视角”的行为。源代码最终迭代跳过 optimizer.step，故 30k 迭代为 29999 次更新；3DSR 每轮 5k 为 4999 次更新。

area4 输入对应的 LR 闭环显式采用 area 降采样；这属于共同观测协议适配，作者 bicubic 原设置另外保留在 loss API。不能把 area4 与 bicubic 输入混在同一公平表，也不能默认不同教师、训练步数或耗时相同。原论文配方比较与统一教师/预算消融须分列。

## 入口

六场景 CPU 注册入口为 `python3 experiments/static_sr_wu_20261022/prepare_suite.py register`，序列计划为同脚本 `plan --workers 4`（使用已验证的 CPU OpenCV 环境）。本机登记索引在 `output/static_sr_wu_20261022/prepared_final/`。单场景 CPU 注册（输出必须是新目录）：

```bash
python3 experiments/static_sr_wu_20261022/prepare.py \
  --manifest MANIFEST --source-plan EXISTING_WU_CONFIGURATION_JSON \
  --out NEW_REGISTRATION --verify-lr-bytes
```

真正执行前由现有资源准入服务提供一个 `CUDA_VISIBLE_DEVICES`，不要直接绕过 GPU 锁。当前已编译扩展位于本地 `output/static_sr_wu_20261022/runtime/`，使用 Python 3.10、torch 2.7.1+cu128、sm80/sm86/sm120；其他环境必须重新编译和验收。

```bash
# 先做真实小步前后向连接检查。该产物永远不作为完整比较。
python train_stage.py --config CONFIG --stage lr --out NEW_FIXTURE --wiring-fixture-steps 3
# Splat LR / 3DSR LR，IE 使用 --stage internal
python train_stage.py --config CONFIG --stage lr --out NEW_LR
# 完整真实先验 bundle 才能启动 SR；完成立即调用固定 cam00 PNG 评价器。
python train_stage.py --config CONFIG --stage sr --prior-bundle PRIORS --out NEW_SR
```

以上后三行需在本目录运行，或使用脚本完整路径。`--check-only` 只核验输入，不能验证 CUDA。禁止将 `--wiring-fixture-steps` 产物登记为正式父模型或正式先验。

缓存入口：

- Splat 主配方/变体：`methods/splatsure_stablesr.py` / `methods/splatsure_teacher.py`；权重图用 `prepare_priors.py --kind splatsure_weights --checkpoint LR_CK --rasterizer-source PINNED_SPLAT_RASTERIZER ...`。`bundle_priors.py` 合并已完成教师与权重图并核对身份。
- IE：`prepare_priors.py --kind iesrgs_external --upstream ... --swinir-checkpoint ... --depth-checkpoint ...`，随后 `--kind iesrgs_internal --checkpoint INTERNAL_CK`，再用 `--kind iesrgs_merge --external-index ... --internal-index ...`。
  两类生产入口可显式传 `--cache-format npz_deflate`，无损压缩原 float32 数组。训练读取同时核验文件 SHA、数组形状、精度和原始数组 SHA；默认 `npy` 保留兼容。两种实际尺寸、两条读取路径的损失与 RGB/深度梯度已通过逐位一致检查。压缩率依赖内容，完整场景仍须独立登记容量预算。
  新增 `--cache-format npz_byte_shuffle`：先将每个 float32 的四个字节按字节位置重排，再作相同 DEFLATE 压缩；解码反向重排，不量化、不改变位值。版本 1 固定为 little-endian float32，单数组不超过 512 MiB。每个 NPZ 只含 `array.npy`（uint8 字节平面）和 `metadata.npy`（uint8 编码的严格 JSON）；元数据保存原形状、类型、编码名/版本及原始数组 SHA。读取在分配数组前核对成员、头部和字节数，拒绝未知字段/版本、重复成员、对象数组、非法形状和 SHA 不符。旧 NPY 和普通 NPZ 保持兼容；新格式的 CPU/GPU 验收范围须查独立收据，不能仅凭压缩探针宣布训练验收通过。
- Sequence：`methods/sequence_matters.py --manifest TRAIN_RECORDS --upstream ... --psrt-checkpoint ... --spynet-checkpoint ... --output ...`。`--plan-only` 提前完成 CPU ORB/ALS 分组。正式推理传 `--sequence-plan 已完成的receipt.json` 直接复用，并核对输入/分组/源码；不重新计算 ORB。之后 `bundle_priors.py` 绑定真实 PSRT 输出。
- 3DSR：`run_threedsr.py --config CONFIG --records TRAIN_RECORDS --lr-checkpoint LR_CK --upstream ... --stable-checkpoint ... --vqgan-checkpoint ... --teacher-python TEACHER_PYTHON --runtime-assets RUNTIME_ASSETS_JSON --out NEW_RUN`。扩散进程结束后串接重建，再串接回渲，不使用定时轮询。LR→SR 重置 Adam 一次；四轮之间保留模型和 Adam（检查点往返），学习率时钟按作者每轮重置。视角采样及增密所用的 Python/NumPy/Torch/CUDA 随机数状态跨轮连续，避免每轮重复相同的观测子集。

所有入口 `--help` 给出实际参数。输出目录独立，不修改已有实验。新检查点带配置、Adam、RNG、实际学习率参数与身份；当前不是自动断点恢复调度器，失败会记录 failure.json，不能静默覆盖或重跑。

## 验收与指标

CPU 测试包括作者公式与梯度对照、源 ALS 控制流对照、Mip 滤波、同时间分组、测试相机隔离和实际相机投影。三个作者扩展已分别编译，且在一个 CPU 进程中共存导入；StableSR 与 3DSR 可共用本机独立 `output/static_sr_wu_20261022/runtime/stablesr_env/bin/python`，已完成实际模块导入、离线 OpenCLIP 缓存解析与依赖检查，环境兼容记录位于 `output/static_sr_wu_20261022/runtime/stablesr_env_audit/acceptance.json`。上述本机环境记录只证明 CPU 导入。后续 CTS 验收已在 Cook07 与 Discussion07 两种原始尺寸上通过四方法共 8 组短程前后向、参数更新和检查点恢复，并通过 IE 外部教师真实推理及统一指标测试。PSRT、StableSR、3DSR 的真实教师验收和正式六场景训练状态，以独立 CTS 收据为准；不能从短程连接测试推断方法效果。逐场景执行、容量门和指标回传后清理见 `deployment/static_sr_cts_20261022/README.md`。

Sequence Matters 的独立解释器为 `output/static_sr_wu_20261022/runtime/sequence_env/bin/python`，固定入口为同级 `sequence_env_audit/run_teacher.sh`。BasicSR 1.4.2、timm 0.6.12、真实 PSRT 模块导入与权重 CPU 张量解析已通过；原作者未锁定这两个包的版本。CTS 已在 Cook07 与 Discussion07 的帧 0 分别完成 20 和 12 个同步训练视角的实际 FP32 PSRT 推理，输出原始 HR 尺寸并经回传 SHA 核验；完整 300 帧先验与正式训练仍须单独完成。

本轮已下载或复用所需官方教师权重并通过哈希校验。训练前仍需核对独立教师依赖、生成真实缓存，并完成各方法的真实 GPU 小步检查、median-depth→形变梯度检查、显存/计时及 checkpoint roundtrip。使用明确固定的 SwinIR-M、Depth Anything V2-Large、PSRT/SpyNet、StableSR/StableSR-Turbo/VQGAN；不能用其他模型替代。无需 CUDA 的权重下载、哈希与可行的 CPU 加载已按本地收据准备；下载记录不能当作教师输出。需要记录 source model、权重 SHA 和参数，不能复用其他项目场景专用权重。

`render_eval.py` 固定保存 8-bit PNG，再计算同一实现的 PSNR、SSIM、AlexNet LPIPS；cam00 全 300 帧。训练视角诊断应另列。正式比较还需要预先固定全图/人脸/文字 ROI、视频连续性、盲评、时间、峰值显存、点数、模型大小和教师准备成本。CPU 通过不说明画质；本目录没有新增方法指标。

## 来源与许可

- SplatSuRe: https://github.com/pranav-asthana/SplatSuRe, `df40abb8d92a02656268cb0a0b97acc2495b9307`, CVPR 2026。
- IE-SRGS: https://github.com/ZheXues/IE-SRGS, `74d60d5bb0f49974ac0efa3497dbbd9e246480f4`, AAAI 2026。
- Sequence Matters: https://github.com/DHPark98/SequenceMatters, `06d9e7fa359f65e6fdb6b96e48786d6028de0caf`, AAAI 2025。
- 3DSR: https://github.com/Consistent3DSR/3DSR, `8c70a9f64301b4de255672e4aa7da42acc0001a2`, ICCV 2025。

来源审计与本地证据在 `docs/static_sr_wu_20261022_sources/` 和 `output/static_sr_wu_20261022/`。上游完整仓库、编译产物、权重和数据不随公开代码发布。方法函数属于带来源的适配，不作为本项目原创算法；上游代码使用须遵守其许可，见第三方声明。
