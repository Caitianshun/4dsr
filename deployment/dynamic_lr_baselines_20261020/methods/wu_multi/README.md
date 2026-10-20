# 原始 LR 的 Wu / Multi4D 比较

本轮只使用 `full_time_20261007/prepared/{cook_spinach,cut_roasted_beef}/` 的最初登记 LR。禁止使用 `_area4`、`phys_*`、`gateaff`、`areaL0` 或 SR 教师替代它。两景均为 0–299 全 300 帧，时间 `frame/300`，LR 336×252，HR 1344×1008。cam00 留出；Cook 的其余 20 相机训练；Cut 缺 cam04，其余 19 相机训练。

## Wu

`wu_*_native_lr.json` 绑定原始 LR manifest、作者配置、已完成原生端点及必需源码的 SHA256。端点位于 `output/dynamic_sr_multiview_footprint_20261007/full_LR_prefixes/{scene}/seed_20261007/checkpoint_fine_14000.pt`。原训练为 coarse3000＋fine14000，共 16999 次 Adam；保留作者末步跳过行为。历史训练使用 PRO6000，应披露训练硬件。

`complete.json` 明确不使用 HR、教师或测试像素，保留作者高斯表示与拓扑。它采用已登记的训练 LR 三角化初始化、已标定相机接口及可恢复的作者等价采样。这些属于公开披露的数据/执行适配。没有颜色仿射、后处理剔除或本项目超分模块。本轮复用端点，重新渲染和统一评价，不重复训练。

## Multi4D

上游固定 `c483e82cd8daa1b4fdc460ed45882fc2fe7a22b0`。独立远端副本为 `output/dynamic_lr_baselines_20261020/runtime/Multi4D-c483e82`。`prepare_multi_native.py` 只生成本轮新入口，不改历史源或上游副本。新入口保留作者三分支、损失、2000＋20000 循环和 batch2。训练直接在 LR 相机网格渲染并比较原始 LR，不进行 HR 渲染后降采样。

数据接口使用训练相机第 0 帧 LR 的 SIFT/已知相机三角化点代替作者提供的 COLMAP 点；动态分支仍按作者随机初始化。该初始化差异必须在论文中披露，不能称完全未修改的原始复现。`prepared_original/` 保存初始化、输入清单与固定原生采样；没有读取 HR、测试图或 SR 图。Cook/Cut 静态初始化分别为 8877/7752 点。

继承 `upstream_fixes_v3.json` 的 6 项已记录源码修复（CUDA 头、混合渲染贡献统计及空 TR 分支处理），并保留既有单点 TR KNN 尺度修复。它们必须与本轮数据适配一起披露。作者最后一步 fine Adam 跳过行为恢复；`fine_20000.pt` 表示完成 20000 个循环，实际 fine Adam 每分支为 19999 次。

`multi4d_native/cpu_contract.json` 核对 19 项作者 loss 赋值表达式逐项相同，区域 mask/L1/depth 赋值也相同。`camera_projection_cpu.json` 覆盖两景所有相机首末帧，使用精确 K_lr/K_hr 和主点：最大投影误差分别小于 0.000024 LR 像素、0.000094 HR 像素。这些是 CPU 检查，不能替代 CUDA 冒烟、完整训练或指标完成。

冻结入口顶部仍保留继承模板的 `area4` 字样，以及 `Context` 中未激活的教师诊断字段。它们不能用来判定本轮训练方式。实际 CLI 只允许 `coarse` 和 `nativeLR`；`nativeLR` 的教师权重为零，不进入教师读取分支。正式完成记录必须满足 `teacher_samples=0`，并与上述 19 项原作者损失表达式合同共同核对。为保持运行源码身份，本轮不清理这些继承名称。

## 统一评价与执行

`evaluate_native_lr.py` 同时支持两种模型。它只在渲染之后读取 cam00 的真实 HR，禁止读取测试 LR、训练 HR 或教师。全 300 帧均通过原有 `write_rgb` 的 `np.rint(clip(rgb,0,1)*255)` 保存至 `evaluation/renders_hr/0000.png` 等文件。浮点指标保存在该评价目录，仅作诊断。主表由共同的 `deployment/dynamic_lr_baselines_20261020/evaluate_images.py` 在同一 A100 base 环境重新评价 8 位 PNG，不能与历史浮点均分混用。

`run_wu_native_lr.py` 只做端点评价。`run_multi_native_lr.py` 依次做 coarse20 冒烟、coarse2000、fine20 冒烟、fine20000，然后立即串接完整 HR 评价。GPU 选择、准入、持久执行与回传由外部 broker 负责。`jobs/a100_gpu1_{wu,multi4d}_original_lr.json` 依次安排 Cook/Cut，并在每景之后执行共同 PNG 指标评价；准备 job 不等于已经提交或完成。

独立 Multi4D 环境 `runtime/multi_env` 只读共享 base Python 包。补充依赖和 sm80 扩展只写独立环境；CPU 编译通过 `build_a100_multi_runtime.py` 隐藏 CUDA 设备执行。训练前仍须检查实际编译状态、公共输入回执、GPU 准入及真实 CUDA 冒烟。

## 公开源码的 CPU 合同再生成

`adapter/adapted_train.py` 是第三方完整训练代码的本地产物，不发布该快照。先按上游固定 commit 准备作者目录，再用 `prepare_multi_native.py` 从已发布的 builder 和模板生成本地 adapter。`check_native_loss_contract.py` 只用 Python AST（抽象语法树）及 `git show` 对照作者源，不导入 PyTorch，也不运行 CUDA。示例输出必须选择尚不存在的文件；本轮已冻结合同不能覆盖。

```bash
python3 deployment/dynamic_lr_baselines_20261020/methods/wu_multi/check_native_loss_contract.py \
  --upstream output/dynamic_sr_multi4d_area_20261017/runtime_v2/Multi4D-c483e82 \
  --out output/rebuilt_multi4d_native_cpu_contract.json
```

检查覆盖 19 项 loss 赋值、区域 mask/L1、深度约束、17 处以目标 RGB 为输入的调用，以及 3 处作者末步 Adam 条件。它只证明这些源表达式一致；输入是否合法、教师采样是否为零，仍由数据接口和真实运行记录单独核验。公开目录不包含数据、初始化点、采样表、权重或执行验收结果。既有 `prepare_inputs.py` 可重建原始 LR 初始化与采样表；依赖和输入需按固定版本手动准备，不能只克隆仓库便启动本轮冻结任务。
