# A100 的 CEM 私有 CPU 构建

此入口只为已授权的两景 CEM 比较准备 sm80 环境，不启动训练或 GPU 验证。GPU 调度仍由根代理统一准入。运行时隐藏全部 CUDA 设备，并把编译并行数固定为 4。

`build_isolated_env.sh`、`prepare_cuda_overlay.py` 与辅助脚本从本轮 `localdygs_cem` 工作流按字节冻结。作者源固定为 CEM `7b5dfe3c05456f95c56b3318e406ed6d0f3c05bd`；PyTorch3D 为官方 v0.7.9 / `33824be3cbc87a7dd1db0f6a9a9de9ac81b2d0ba`；Faiss 为官方 v1.12.0 / `e8234e563f1ecef5f036e83c3cfee366d3f1fbca`。CEM 的 simple-knn 已自带 `<cfloat>`，本入口不施加 LocalDyGS 的头文件补丁。

A100 上的独立环境为 `envs/dynamic_lr_baselines_20261020/cem_a100_sm80`。它通过 `.pth` 只读借用 base PyTorch 2.7.1+cu128，补充依赖与编译结果只写私有环境。CUDA overlay 也只在该 leaf 中建立软链接。独立作者源与校准 v3 runtime 位于 `output/dynamic_lr_baselines_20261020/runtime/CEM-4DGS-{7b5dfe3,calibrated-v3}`。训练或渲染时使用 v3 的数据接口；编译使用带原始 `.git` 的固定作者源。

先同步并校验 `output/dynamic_lr_baselines_20261020/a100_cem_cpu_build/source_inventory.json` 绑定的文件，再在 A100 项目根执行：

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python deployment/dynamic_lr_baselines_20261020/methods/cem_a100_build/run_cpu_build.py
```

状态、日志、版本清单与 CPU import 证据写入 `output/dynamic_lr_baselines_20261020/a100_cem_cpu_build/`。`completed_CPU_build_no_GPU_execution` 仅表示编译及无 GPU 分配的导入检查完成，不能替代 GPU 冒烟、正式训练或评价。Faiss 必须保留真实 GPU 类，不使用 CPU Faiss 替代。

首轮构建在 Faiss 14% 因精简 toolkit 缺少 `cuda_profiler_api.h` 失败，原状态与日志保留。`continue_cpu_build.py` 只恢复这项已定位失败。补充的头文件来自本机官方 CUDA 12.8.1，SHA256 为 `bb8e03c217df1309ccfe3249c4fac1bcd577bdcff5c65944217548195204118e`，只复制到 CEM 私有 overlay，不改作者或 Faiss 源码，也不改共享 CUDA。恢复记录另存 `recovery_01/`，不能覆盖首轮失败证据。

本次 CPU 恢复已完成。`finalize_cpu_build.py` 重新验证了全部 5389 项源码 SHA 和上游 commit，冻结 leaf 的 10 项环境/扩展/头文件身份，写入 `build_identity.json`；其本次 SHA 为 `57878e1847f6d426c12c1f82a9cc527fd2da438d881531de5b7ac4703e756f95`。完整原生 `train.py` 在禁止下载且隐藏 CUDA 的条件下导入通过，未实例化 LPIPS 网络。该验收仍不表示真实 GPU 算子或训练已经完成。

`build_cem_a100_job.py` 只生成一个场景的冻结任务。它要求原生初始化、相机标定审计及 CPU 构建身份均已回传，本身不提交 GPU 作业。场景独立生成，`--gpu-index 0/1` 冻结物理编号及对应 UUID，默认 GPU1。两阶段固定为作者 stage1 的 40000 步，以及从本次 `chkpnt40000.pth` 继续到 stage2 总 50000 步；随后立即渲染 300 张 HR PNG、统一评价和打包来源元数据。

CEM 原始 `train.py` 会将 `--gpu` 写入 `CUDA_VISIBLE_DEVICES`。因此，A100 物理 GPU1 必须传 `--gpu 1`。冻结的 `run_author.py` 在导入作者代码前要求该参数与 broker 的单整数 mask 完全相同；设备内的 CUDA/Faiss 操作仍使用可见逻辑设备 0。禁止把物理 GPU1 改写为 `--gpu 0`。

`frozen_execution_helpers_v2/` 是共享 CTS 工作流的独立字节快照，身份见其 `frozen_helper_identity.json`。它支持独立 helper 目录以及 `--unified-metrics`；较早的未提交快照保留来源。最终 `source_metadata/` 保留 protocol 补丁、上游 commit、配置、执行 helpers、训练时环境文本、构建身份及初始化标定记录，避免只留下远端安装目录。主表使用统一的 8-bit PNG 评价，写入 `output/dynamic_lr_baselines_20261020/shared_A100_metrics/cem/{scene}`，不调用原生 LPIPS 测试报告。静态调用链审计见本地构建输出的 `pretrained_weights_audit.json`；本轮 CEM 训练和 HR 渲染不加载外部预训练网络。

重复 helper 快照不发布到源码仓库。公开的 `freeze_execution_helpers.py` 从 canonical `methods/localdygs_cem/` 读取 5 个自有 helper，逐一核对本轮固定 SHA、只解析语法，再原样复制到新目录并生成身份清单。它不导入作者代码、不运行 GPU，也拒绝覆盖已有目录。新部署可执行：

```bash
python3 deployment/dynamic_lr_baselines_20261020/methods/cem_a100_build/freeze_execution_helpers.py \
  --output deployment/dynamic_lr_baselines_20261020/methods/cem_a100_build/frozen_execution_helpers_v2
```

已有本轮 v2 快照时不要重跑上式；为另一份新快照指定新目录，并把它传给 job 生成器的 `--helper-directory`。作者源、官方 PyTorch3D/Faiss、私有 leaf、固定输入和 SHA source inventory 仍须按上述路径及版本手动准备。source inventory 和输入回执是部署产物，不是随公开仓库提供的数据；本轮没有承诺一键下载、数据授权或完整安装树。
