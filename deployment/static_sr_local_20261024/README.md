# 本机 Cook07 SplatSuRe 串行执行入口

此入口只生成 CPU 计划，并在主控已经通过本机 broker 准入后执行。它不提交 GPU 作业、不访问远端、不清理任何资产。当前协议固定为 Cook07 全 300 帧、20 个训练相机共 6000 个观测、area4×4（336×252 → 1344×1008）；cam00 的 300 帧只用于评价。

主配方是 Wu + SplatSuRe 的动态适配：LR 30000 迭代 → StableSR 全部训练视角教师 → 作者贡献栅格器生成同时间权重 → 完整 bundle 验证 → 全新初始化的 SR 30000 迭代 → 统一 PNG PSNR/SSIM/LPIPS。没有步数覆盖或 fixture 参数。根据已冻结作者日程，每 30000 迭代执行 29999 次参数更新。

## 两个独立阶段

`--phase lr` 执行 LR 训练，成功退出后立即执行完整 cam00 评价。它不会自动启动教师。`--phase full` 必须读取同一 run 的成功 LR 收据，并重新校验配置、代码、检查点及所有评价 PNG 的 SHA；随后串行完成剩余步骤。最终 SR 训练入口已经内置即时评价，所以不会重复评价或重复渲染。

CPU 生成 LR 计划的示例：

```bash
python3 deployment/static_sr_local_20261024/run_splatsure.py plan \
  --phase lr --run-id cook07-splatsure-v1 \
  --gpu-uuid GPU_UUID_FROM_RESOURCE_OWNER \
  --wiring-receipt LOCAL_WIRING_RECEIPT \
  --metric-receipt LOCAL_METRIC_RECEIPT \
  --out NEW_PLAN_JSON
```

本机实际 GPU 验收必须与当前主机、GPU、Cook07 输入尺寸、源代码和扩展二进制一致；旧 CTS 证据不能作为本机准入。计划固定 broker 合法且最长 64 字符的 job ID，保留原 run ID 的哈希，避免下划线或大小写归一化碰撞。主控将该 ID 原样用于 `role=local, env_id=sr4d`，单 GPU broker 作业执行以下 argv：

```text
/home/cai_tianshun/miniconda3/envs/sr4d/bin/python -s
ABS_PROJECT/deployment/static_sr_local_20261024/run_splatsure.py run --plan ABS_PLAN_JSON
```

运行器验证真实 broker cgroup 与单个 GPU UUID，不会自行调用 broker。持久性由现有 systemd broker 服务提供。GPU 子进程始终串行。

本地环境由任务单独准备：默认 `output/static_sr_wu_20261024/local_runtime/launch_v2.py` 和 `stable_python`（调用 v2 launcher）。可通过 `--base-launcher`、`--teacher-python` 指定已验收版本；不修改既有解释器或 `.pth`。运行器冻结 base overlay 清单及其实际文件、Wu 共享源代码和扩展、实验源代码；full 阶段另冻结教师 overlay 和缓存注册。每个子进程启动前及最终成功写入前重新校验冻结文件，避免长训练期间的源修改被报告为原版本结果。

full 计划增加 `--teacher-receipt LOCAL_STABLESR_RECEIPT`、`--lr-completion RUN/lr_phase/complete.json`，其余输入与 LR 相同，并使用新 plan 文件。全量 StableSR 的计算预算必须根据本机单图实测另行决定；不因历史时间过长而自动换成 SwinIR、减少 DDPM 步数或缩小场景。

## 空间与失败行为

每阶段启动前重新检查“剩余阶段预算 + 20 GiB 预留”。PNG 预算按固定 RGB8 尺寸、扫描行开销、15% DEFLATE 保守膨胀与每文件 64 KiB 容器余量计算，并要求文件系统分配单元不超过 4096 字节；不依赖样本压缩率。LR 暂存复制退路、每个阶段日志、float32 权重图和评价 PNG 分别计入。检查点采用每文件 2 GiB 的序列化失败边界，超出即失败；它不是有效检查点截断、点数上限或科学日程修改。教师与纯评价的每文件边界为 1 GiB。

计划标为 `blocked_storage` 时不能执行。一个失败或已启动的阶段目录不能重放；修复后必须由主控明确决定新 run。中断与失败保留 `status.json`、各阶段原日志、SHA、耗时及最后 64 KiB 诊断。正常结束保存完成收据、检查点和 300 帧评价身份；所有大资产留在原处。

CPU 测试：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  python3 -B -m unittest discover -s deployment/static_sr_local_20261024 -p 'test_*.py' -v
```

CPU 测试不代表本机真实前后向、正式训练收敛或方法比较完成。

## Sequence Matters 的独立完整链

`run_sequence.py` 使用同一 Cook07 area4×4 协议。它先执行帧 0 的全部 20 个训练相机 PSRT（Pseudo Sequence Recurrent Transformer）GPU 验收，再生成正式 6000 张训练教师图。相机按既有 ALS 序列计划组织：每组固定同一时刻，跨训练相机组成伪视频；不把真实时间视频替代作者的跨视角序列。完整计划包含 300 组、2100 条最多 8 张图的序列。旧 CPU 计划必须通过记录、全部 LR、作者版本及源码 SHA 核验，才可跳过 ORB/ALS 重算。

```text
20 相机同步组 GPU 验收（独立目录，不能作为正式先验）
  → 6000 张正式 PSRT 教师图
  → 正式 producer、逐图 SHA、模型和 LR 绑定检查
  → bundle + 训练入口 check-only
  → 全新 Wu SR 30000 迭代（29999 次更新）
  → 内置 cam00 全 300 帧 PSNR / SSIM / LPIPS
```

PSRT 与 SpyNet 使用已核验的作者权重；保留作者 FP32、色彩处理、序列长度与模型设置。Sequence SR 从新模型开始，不读取 SplatSuRe 的 LR 检查点。单组验收必须证明真实 GPU 执行、全部 ALS 前向顺序、×4 输出尺寸、有限数值和完整 20 张图；旧 CTS 收据与纯 CPU 计划均不能代替此步骤。

CPU 计划命令如下。`--out` 必须在未来运行目录外，且必须是新文件。

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  /home/cai_tianshun/miniconda3/envs/sr4d/bin/python -B \
  deployment/static_sr_local_20261024/run_sequence.py plan \
  --run-id cook07-sequence-v1 \
  --gpu-uuid GPU_UUID_FROM_RESOURCE_OWNER \
  --wiring-receipt LOCAL_WIRING_RECEIPT \
  --metric-receipt LOCAL_METRIC_RECEIPT \
  --out NEW_SEQUENCE_PLAN_JSON
```

执行只由主控经既有本机 broker 准入；须使用计划生成的完整 job ID、相同 PRO6000 UUID、`role=local` 和 `env_id=sr4d`，由 broker 持久服务串行运行：

```text
/home/cai_tianshun/miniconda3/envs/sr4d/bin/python -s
ABS_PROJECT/deployment/static_sr_local_20261024/run_sequence.py run --plan ABS_SEQUENCE_PLAN_JSON
```

默认 base launcher 为任务内已冻结的 `launch_v2.py`；教师使用 `sequence_python`，不改变已有环境。新目录固定在 `output/static_sr_wu_20261024/local_sequence_runs/Cook07/RUN_ID`。每个子进程前重验源码、权重、overlay 文件与剩余磁盘预算。失败保留 `execution/status.json`、`diagnostics.json`、各阶段日志和耗时；成功才写 `execution/complete.json`。没有自动提交、续跑、删除、远端访问或方法替换。

全链初始要求 **61,115,334,656 字节（56.917 GiB）** 可用空间，其中包括 20 GiB 预留。PNG 每张上界为 4,743,168 字节，分别计入 6000 张正式教师、20 张验收图和 300 张评价图。其余空间计入阶段日志与受每文件上界约束的检查点；不使用样本压缩率承诺容量。`blocked_storage` 计划不能执行；资源改变后须由主控生成新计划，不能改旧 JSON 绕过门槛。单组教师成本仅用于估计，不能视为全场景实测成本或训练收益。
