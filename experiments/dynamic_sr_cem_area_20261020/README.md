# CEM area4 骨干与监督诊断

本目录把当前项目的六种监督设定接到 CEM 的原生表示上，用来区分教师信息、观测修正和骨干拟合能力。它是受控迁移实验，不能称为原论文提出的动态超分方法。

上游为 `cem4dgs_20261020`，commit `7b5dfe3c05456f95c56b3318e406ed6d0f3c05bd`。运行依赖已经验收的 calibrated runtime v3、CEM 私有环境，以及独立准备的 **area4 训练 LR 第 0 帧 COLMAP 稀疏初始化**。不附带第三方完整源码、数据、权重、环境或结果。`prepare_native_runtime.py` 只复制已核对身份的 v3，替换观测接口，并写逐文件身份；不会修改旧运行目录。

## 为什么这样连接

CEM 用可渲染高斯和虚拟运动中心组成父子层级。高斯的位置和旋转继承父节点的运动，运动中心通过时间控制点插值；时间不透明度控制高斯在何时可见。这与 Wu 的共享形变网络不同。迁移只调用 CEM 自己的 `init_model`、`restore`、`render` 和 RAdam 优化器，不替换运动表示或套用 Wu 的增密函数。

先从 area4 LR 独立训练作者完整两阶段：0→40k，再恢复该分支的完整检查点到总 50k。父模型保持原生图像损失、运动正则、采样、渐进时间范围、增密、跨视角纠错（CEC）和更新顺序；只适配精确相机、像素路径及只读测试日志。父模型不增加颜色仿射或 w4。

六个 SR 臂恢复同一场景的 50k 父模型及全部 11 组 RAdam 状态，在绝对迭代 50001→56000 各做 6000 次更新。作者 CEC 已在 50k 前结束；这一续训阶段固定高斯数、虚拟运动中心数、父子索引和时间控制点数量，但位置、球谐颜色、不透明度、尺度、旋转、位移、运动控制点和时间不透明度参数继续训练。保留作者学习率调度器的实际行为，包括其首组更新后的提前返回，不自行修正。

这能诊断既有 CEM 表示如何吸收新监督。它不能单独判断“允许重新增密的 CEM 是否更好”，也不能把与 Wu 或 Multi4D 的差异归因为纯骨干差异。当前 Wu SR 会重新增密，训练路径不同。

## 六臂与额外 HR 参考

每步使用同一时间的三个不同训练相机。a 约束 LR 闭环，b、c 各接受权重 0.2 的教师监督。图像项均为 `0.8 L1 + 0.2 (1−SSIM)`；SSIM 使用 CEM 的实际实现。原生 CEM 正则每步加一次。

| 臂 | a 的 HR→LR 运算 | b、c 的目标 | w4 |
| --- | --- | --- | --- |
| areaL0 | 4×4 平均并裁到 [0,1] | 无，仍执行共同三视角渲染 | 无 |
| naive | bicubic，antialias=True，align_corners=False | 原始 SwinIR | 无 |
| areaT | 4×4 平均并裁到 [0,1] | 原始 SwinIR | 无 |
| areaTm | 同上 | area4 匹配 SwinIR | 无 |
| areaTm_w4 | 同上 | area4 匹配 SwinIR | 有 |
| areaOracle | 同上 | **训练相机 HR**，只替换 areaTm 的教师 | 无 |

所有 SR 臂共享一个从零初始化的逐训练相机 3×4 颜色仿射，去除相机均值使平均变换为恒等，Adam 学习率固定 0.001。原生 CEM 父模型没有可继承的颜色仿射；这一点与当前 Wu 链不同。

w4 是逐训练相机的 2×4×4 平滑二维位移场，单位为 HR 像素。它去除相机均值，先以 bilinear、align_corners=True 放大，再以 bicubic grid_sample、border、align_corners=False 作用于渲染图，之后做颜色仿射。Adam 学习率在 6000 步内从 0.002 衰减到 0.0002。不加入时间偏移、五参数标定或 Wu 的裁点模块。cam00 的颜色和位移始终恒等。

另做独立 `native_HR_reference`：从同一 **LR-only** COLMAP 初始化出发，用训练 HR 完成作者 0→40k→总 50k。它不是 areaOracle；前者检验完整 HR 拟合路径，后者只检验同 LR 父模型上 6000 步续训时教师信息的替换效果。两者均属额外 HR 诊断，不能放进合法 LR-only 方法主表。

## 准备和执行接口

CPU 准备不启动 CUDA：

```bash
python3 experiments/dynamic_sr_cem_area_20261020/prepare_native_runtime.py
python3 experiments/dynamic_sr_cem_area_20261020/check_native_contract.py \
  --runtime output/dynamic_sr_cem_backbone_20261020/runtime/cem-native-area4-v1 \
  --output /new/path/native_CPU_contract.json
```

在已准入的设备上，由调度器提供唯一物理 `CUDA_VISIBLE_DEVICES`。`native_stage.py` 的 `--gpu` 必须等于该物理编号；CEM 算子使用掩码后的逻辑设备 0。两个阶段使用同一个新 model 目录，stage2 只能恢复该目录的 `chkpnt40000.pth`：

```bash
python native_stage.py --runtime RUNTIME --manifest AREA4_MANIFEST \
  --source NATIVE_LR_INIT_CEM_DATA --model NEW_MODEL --stage 1 \
  --level lr --gpu PHYSICAL_GPU --receipt NEW_STAGE1_JSON
python native_stage.py --runtime RUNTIME --manifest AREA4_MANIFEST \
  --source NATIVE_LR_INIT_CEM_DATA --model NEW_MODEL --stage 2 \
  --level lr --gpu PHYSICAL_GPU --receipt NEW_STAGE2_JSON
```

独立 HR 参考使用新的 model/receipt 路径和 `--level hr`。原生 HR 容量接线可用 `native_capacity_fixture.py --iterations 5`，仅把隔离配置的迭代终点改为 5。作者最后一步不执行优化器更新，因此该 fixture 是 5 次迭代、4 次更新；其初始点数也不能代表成熟 50k 模型的显存。

SR 入口：

```bash
python sr_stage.py --runtime RUNTIME --manifest AREA4_MANIFEST \
  --schedule FROZEN_SCENE_SCHEDULE --parent-checkpoint NATIVE_50K_CHECKPOINT \
  --parent-stage2-receipt NATIVE_STAGE2_JSON --arm areaTm_w4 \
  --teacher-index MATCHED_TEACHER_INDEX --gpu PHYSICAL_GPU --output NEW_SR_DIR
```

areaL0 和 areaOracle 不传 teacher-index。短接线必须显式 `--steps 3 --wiring-fixture`，不会标为完整 6000 步结果。正式计算前须实际验收原生 CUDA 栅格器、11 组梯度/状态连接、HR 三视角显存和计时；CPU 合同不能代替这些证据。

`render_hr.py` 对 native 50k / SR 56k 统一保存 HR RGB PNG。`--camera cam00` 输出完整 300 帧；`--camera cam01` 只输出固定 0、150、299 帧供训练视角诊断，SR 同时保留未经观测修正的 `_raw.png`。全图与人脸面板由外部固定 ROI 工具制作，渲染器不挑区域。所有图片在裁剪和量化前检查形状与有限值，然后采用 `np.rint(clip(rgb,0,1)*255)`。统一 evaluator 读取同一批 8-bit PNG；cam01 诊断与 cam00 新视角指标分开。

## 验收边界

CPU 合同检查两景全部帧/相机/教师映射和 exact schedule、主点与时间约定、真实源代码的颜色/w4/降采样值及梯度、CEM SSIM 和正则表达式。它不构造原生 CUDA 模型、不更新参数，也不证明收敛或画质。正式记录还必须保存输入与源码身份、初始化版本、训练硬件、各组 RAdam 更新计数、初末点数、零拓扑事件、时间和显存口径。原生检查点不包含 RNG；六臂在恢复后共同显式使用 SR seed 20261007。
