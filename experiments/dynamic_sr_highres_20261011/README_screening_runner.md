# 一条件、一个种子的持久验证流程

这个流程用于判断当前场景是否存在值得继续优化的空间。它保留六个终点：LRparent17k、SRrecon17k、HRref17k，以及从相同 LRparent 接着训练 6000 步的 lam0、base 和 oracleHR。后三个终点总计 23000 步。17k 与 23k 不能宣称训练预算相同；同迭代数也不能宣称计算量相同。最终比较由 headroom.py 根据完整指标、预算和实际硬件记录完成。

本文件只说明新调度接口。历史 cullretrain.py、modsr.py、refs.py 与冻结教师生成器保持原实现。Long360 的 32 个训练相机采用共享的 6000 行采样表：每个时间从 32 个相机中轮转选择 20 个相机，三列曝光相同，所有相机都出现，但每列只覆盖 62.5% 的合法相机—时间组合。这是披露的数据接口适配。

## CPU 前置步骤

Z1 必须在当前 HR 网格通过。Z5 必须在全部训练相机的帧 15、150、285 上通过，并绑定相同数据集、时间窗、降采样图像、冻结模型和生成源码。三张教师图像的成本记录不能代替 Z5。读取原始视频的完整准备、LR 初始化和 HR 展开均在本地执行。远端仅接收原样文件及原样来源记录。

远端启动前需要以下相对路径：

- `data/dynamic_sr/highres_20261011/prepared/<condition>/manifest_train_ready_seed20261007.json` 与其输入文件、初始化点云和收据。
- `data/dynamic_sr/highres_20261011/hr_full/<condition>/hr_full_storage.json` 与所有 HR 监督图像。
- `output/dynamic_sr_highres_20261011/registry/full_LR_prefixes/<condition>/seed_20261007/config.json`。
- `output/dynamic_sr_highres_20261011/registry/teachers/<condition>/plan.json`。
- Z1、时间窗、Z5 结果和 Z5 样本清单，保持来源路径与内容不变。

先在目标主机生成任务表，再进行只读验证：

```bash
.venv/bin/python -B experiments/dynamic_sr_highres_20261011/make_screening_tasks.py \
  --condition l3H_long360 --seed 20261007 \
  --z5-receipt output/dynamic_sr_highres_20261011/zero_checks/long360/gpu_screen_half/z5_half.json \
  --z5-samples output/dynamic_sr_highres_20261011/zero_checks/long360/samples_half/samples.json \
  --out output/dynamic_sr_highres_20261011/pipelines/tasks_l3H_seed20261007.json

.venv/bin/python -B experiments/dynamic_sr_highres_20261011/persistent_runner.py \
  --tasks output/dynamic_sr_highres_20261011/pipelines/tasks_l3H_seed20261007.json \
  --out output/dynamic_sr_highres_20261011/pipelines/l3H_seed20261007_<fresh_suffix> \
  --gpus 0 1 --allowed-graphics-name gnome-remote-desktop-daemon --validate-only
```

`--validate-only` 不申请 GPU、不调用 nvidia-smi、不初始化 CUDA、不执行训练。正式运行由主控再次确认资源、同步哈希后，通过 systemd 持久服务运行同一命令并移除 `--validate-only`。A100 可指定 `--gpus 0 1`；3090 使用一张卡。每项任务只使用一张卡。GPU 可用索引或 UUID 指定；运行记录保存解析后的 UUID。两张卡必须属于相同型号。

## 完成事件与失败保护

runner 对每个子进程调用等待退出接口。子进程成功退出后，立即核验产物，再启动后继任务。两张卡可以同时生成教师和训练 LRparent。每个训练终点接完整 300 帧评价，随后接区域诊断。调度与完成发现均不依赖模型轮询。

完整评价核验 `metrics_per_observation.csv` 的 300 个唯一 `(cam00, frame)` 键，以及 PSNR、SSIM、LPIPS 是否有限。训练核验步数、检查点哈希，并拒绝 smoke 诊断检查点。区域诊断沿用 modsr 默认采样：cam00 每 10 帧取一帧，共 30 帧；cam01 取 15、45、…、285，共 10 帧；训练 LR 闭环检查取 15、150、285。区域诊断不能描述为 300 帧统计。

LRparent17k、SRrecon、HRref 和 base 还执行 `prior_fit.py`。它在全部训练相机的 15、150、285 帧比较教师图像、渲染图像与真实 HR，并计算渲染与教师的拟合误差。这项诊断用于分清“教师本身损失多少信息”和“三维重建又损失多少信息”。它不更新模型参数。它属于训练视角诊断，须与 cam00 新视角的 300 帧评价分开解释。流程总计 28 项任务，增加的四项只评价，不增加训练分支。

每项任务前检查全部冻结源码与输入身份，并检查该命名空间的 Python 文件清单。新加入、修改或删除源码均使任务拒绝继续。运行期间不要修改该主机的源码副本。GPU 锁与 screen_gpu.py 共用 `output/dynamic_sr_highres_20261011/claims/<GPU-uuid>.lock`，使用排他创建。已有锁阻断调度；退出时只删除本 runner 创建的锁。

启动前要求 GPU 总占用不超过 1024 MiB、利用率不超过 10%。只有明确传入的桌面进程名称可以保留，且单进程不超过 600 MiB。未知计算进程会阻断任务。失败立即写入失败收据并停止新增后继任务，同时保留已经运行的健康分支，等待其正常退出。部分输出和历史收据不覆盖、不自动重启。

CPU 验收命令为 `python3 -B experiments/dynamic_sr_highres_20261011/selfcheck_runner.py`。它使用合成收据和一个已完成 Cook07 评价的真实格式验证拒绝路径，覆盖 Z5 身份、步数、哈希、帧完整性、GPU 占用和锁保护。它不运行教师或训练。结果收据不等于实际训练验收。
