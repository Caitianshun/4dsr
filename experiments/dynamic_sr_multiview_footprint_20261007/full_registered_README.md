# 注册版完整时间窗 native SR 与评价入口

`full_refine_registered.py` 与 `full_evaluate_registered.py` 是原冻结入口的独立适配。原 `full_refine.py`、`full_evaluate.py`、`full_prefix.py`、已验收父模型及其 plan 均保留原字节。

父模型仍是 Wu `843d5ac636c37e4b611242287754f3d4ed150144` 的原生 14 元素 checkpoint，一个含八个参数组的 Adam。Cook/Cut 已验收的四个父模型沿用旧 `full_author_LR_prefix_v1` plan、实际执行来源及原 RNG 兼容修复记录。其他父模型必须使用 `registered_native_full_LR_prefix_v1`、冻结的场景配置和完整 300 帧合法 LR 数据。陌生 schema 不被接受。

| 接口 | LR / HR 尺寸 | 前缀 batch | 完整前缀 RGB 次数 |
| --- | --- | --- | --- |
| Cook / Cut，原作者场景配置 | 336×252 / 1344×1008 | 2 | 34,000 |
| MeetRoom Discussion / VR，明确登记的 Wu generic 域适配 | 320×180 / 1280×720 | 4 | 68,000 |
| Coffee，原作者场景配置 | 336×252 / 1344×1008 | 4 | 68,000 |

完整前缀各有 3,000 coarse + 14,000 fine 循环、17,000 次反向、16,999 次 Adam、两个阶段优化器重置。最后 fine 循环原作者未应用 Adam 的语义保留。这不是未修改的原作者 MeetRoom 实验，也不是完整官方 benchmark 结果。

SR 后缀仍为 6,000 次更新、每次 3 个 HR RGB 渲染、一次累加反向与一个原生 Adam。X/MX 每次额外 3 个原生 HR raw moment 渲染，并用六条有向边的冻结完整足迹支持。学习率/SH 时钟为 `fine14000 + SRcursor`；密度时钟统一从 1 开始，密度统计仍按同一高斯索引先求梯度和、半径最大值及可见性的逻辑或。新 SR 边界仅清零 `max_radii2D`、`xyz_gradient_accum`、`denom` 一次；恢复后不重清零，也不重置 Adam/RNG。

训练必须有已冻结的最多两个候选选择证据、完整教师、同一父模型绑定的 schedule；X/MX 还须完整支持及零更新训练校准，E 须其校准。缺少资产会登记 `pending_full_native_refinement_dependencies`，训练在 CUDA 初始化前拒绝。确认场景不能参与候选选择。教师尺寸按该场景 manifest 校验。投影、足迹、raw moment、权重、损失和校准公式未改变。

CPU 合同命令：

```bash
CUDA_VISIBLE_DEVICES='' /home/cai_tianshun/Project/4dgs/.venv/bin/python \
  experiments/dynamic_sr_multiview_footprint_20261007/full_registered_checks.py
```

该合同核验实际四个旧父、Meet/Coffee 完整合法 LR SHA、实际 batch/grid/source、错误数据/计数/选择的拒绝，并以 AST 等价与可执行 CPU 模型验证公式、一次 SR 更新和密度统计。CPU fixture 不代表 native CUDA 验收、训练完成或质量结果；所有失败与成本保留在独立 `operator_checks/full_registered/` 目录。

注册版训练 CLI 与原入口参数相同，以新文件运行：

```bash
python full_refine_registered.py --mode plan --method M \
  --manifest <seed-specific-train-ready-manifest> --seed 20261007 \
  --parent <native-final-checkpoint> --parent-complete <native-complete.json> \
  --teacher <complete-seed-teacher-index> --schedule <full-schedule.json> \
  --selection <frozen-development-selection.json> --out <new-registration-directory>
```

X/MX 添加 `--support-index` 和 `--calibration`；E 添加 `--calibration`。生产训练或校准还要求该注册版源码对应的 `--native-acceptance`、根代理分配的 `--gpu-uuid` 及资源锁。已有旧 CUDA 收据不能冒充新源码的验收。

评价先执行 `full_evaluate_registered.py --mode register --manifest ... --protocol <new-file>`。确认场景还须 `--selection`。实际 `--mode evaluate` 需要明确 checkpoint、输出、label、GPU UUID 和 `--operator-resource-resolved`。评价保持零 Adam、零反向、全 300 帧指标、原生模型/Adam/拓扑/RNG 不变；测试 LR 仅在渲染后作诊断。ROI 沿冻结来源，未给新场景臆造区域。

`full_native_context` 与旧支持读取器格式兼容，仍为 `{manifest, parent, schedule}` 三个文件身份。对外 API `full_parent(...)`、`restore_native_model(...)`、`native_effective_state(...)`、`state_covariance(...)`、`native_rgb(...)`、`hr_camera(...)` 的参数位置保持；新增 `validate_parent_plan(plan, complete=None)` 返回场景 grid/batch/完整状态契约。原生 checkpoint 本身不做模型转换。
