# Yang 4DGS 与 4C4D：原始 LR 动态重建部署

本目录只提供作者实现的数据接口、隔离环境构建和可追踪执行外壳。训练输入固定为 `full_time_20261007/prepared/{cook_spinach,cut_roasted_beef}/manifest_train_ready_seed20261007.json`。这两个文件没有 `_area4` 后缀。每景保留 300 帧、原始 4× LR、完整场景，cam00 只用于训练结束后的新视角渲染。

## 作者代码身份

| 方法 | 官方仓库 | 固定提交 |
| --- | --- | --- |
| Yang 4DGS | https://github.com/fudan-zvg/4d-gaussian-splatting | `3fe460f1f2342526bccf6c01b63da125faf0fa94` |
| 4C4D | https://github.com/yangzf-1023/4C4D | `ed6a3cb69782c4147151b3898944bc38132bae00` |
| 4C4D 初始化依赖 MAtCha / MASt3R | https://github.com/anttwo/MAtCha | `b119fd96e484fc81eb40623c1ea92ad3dbd3c21e` |
| MAtCha 的 Python 相机工具依赖 PyTorch3D v0.7.4 | https://github.com/facebookresearch/pytorch3d | `297020a4b1d7492190cb4a909cafbd2c81a12cb5` |

Yang 使用作者 2024 年 9 月的 4D 旋转及球谐函数反向传播修正版本；未使用仓库 2026 年新增的时间预过滤实现。第三方原始仓库与运行副本分开保存，运行副本的 `adapter.patch` 和 `adapter_identity.json` 记录每处适配及前后哈希。

公开源码只包含本项目接口和小型适配补丁。`patches/UPSTREAM.json` 给出每个补丁对应的官方仓库、提交和 SHA256；这些补丁已对固定原码执行 dry-run 检查。复现还需单独准备上述第三方仓库及子模块、官方预训练权重、COLMAP 和合法数据，不能仅克隆本项目就完成全部实验。

## 输入与相机

`prepare_native.py` 对照 manifest SHA256 检查每张训练 LR，再创建硬链接或副本。Cook 使用 20 台训练相机，共 6000 张；Cut 缺 cam04，使用其余 19 台训练相机，共 5700 张。没有读取 cam00 或 HR 像素。

相机世界坐标使用 Yang 作者的上向量、观察中心和平均相机半径 4 的归一化规则。`stage_receipt.json` 保存 manifest 世界坐标到原生世界坐标的完整相似变换。初始化点云必须处于该原生坐标系，不能直接复用其他方法在原世界坐标中生成的点云。

原始 manifest 的像素中心遵循 OpenCV 整数像素中心约定。作者 CUDA 栅格器采用角点坐标约定，因此相机接口将主点加 0.5，以保持相同射线。没有将主点改成图像中心。`utils/camera_utils.py` 还需要从同一焦距补出有效 FoV，使 CUDA 椭圆投影和完整投影矩阵使用一致内参。

`audit_projection.py` 实际执行作者 `Camera` 和 `loadCam`。它将原世界中多个点映射至原生世界，再通过实际完整投影矩阵回投图像，检查全部训练 LR 相机及 cam00 HR 相机。CPU 验收误差上限为 0.002 像素。它只读相机元数据，不读图像。

时间只转换一次：`native_time = 10 * manifest_time = frame_index / 30`，作者原生时间域保留 `[0,10]`。训练与最终 HR 渲染使用同一映射。不会把时间压到 `[0,1]` 后继续沿用作者未缩放的时间超参数。

## 初始化和额外先验

- Yang：从合法训练相机的第 0 帧原始 LR 进行 CPU SIFT、已知相机位姿三角化，再用作者 COLMAP PatchMatch / stereo fusion 得到稠密点。GPU 稠密阶段必须通过公共 broker。没有随机点、项目旧点云或 HR 初始化后备分支。作者 N3DV 配置的最多 300000 个初始化点抽样行为保持不变。
- 4C4D：使用作者 MAtCha posed MASt3R 流程，输入同样只有训练相机第 0 帧原始 LR。作者 MASt3R 内部将输入调整至模型工作分辨率 512；这是预训练几何先验，需要在论文中单列，不属于使用 SR 图像监督重建。保持作者 coarse/refine 各 1000 步、标定相机初始化、全部合法训练相机和原生置信度阈值。原生选项包含 `--fix_focal --fix_principal_point --fix_rotation --fix_translation`；但源码对平移仍设置 `opt_tran=True`、对尺度设置 `opt_size=True`，结束后将点云和相机位置按尺度、平移对齐到给定相机，并将输出相机恢复为给定位姿。这里保留作者实际实现，不能把该过程描述成优化全程完全固定内外参。训练和 HR 渲染始终使用 manifest 的给定相机。保留作者 CLI 实际默认 `initial_num_pts=-1`，即全部 MASt3R 点；不自行限点以适配显存。

两种方法的输入目录独立，防止 MASt3R 覆写 Yang 的 COLMAP 初始化。原始 LR 通过硬链接共享字节，几何和收据不共享路径。

Yang 的已完成初始化收据记录实际执行文件为 A100 的 `/usr/local/bin/colmap`。当前只读二进制检查为 COLMAP 3.9.1、Commit Unknown、支持 CUDA，SHA256 为 `861adf9a3ea43a3b2469f3c093c010713001c893c06043dfb3b512791210cbc5`。历史初始化收据未单独保存当时的二进制哈希，当前检查不能冒充历史冻结证明。若与其他机器的 COLMAP 版本不同，需随初始化条件一同披露。

## 训练和评价边界

保留作者完整 30000 步、batch size 4、损失、采样、优化器、球谐函数阶数及增密日程。4C4D 的不透明度衰减及相应日程使用作者 CLI 默认行为。只替换数据/输出路径、将已经是 336×252 的 LR 设置为 resolution 1，并关闭训练中的测试图像读取及测试指标检查点选择。

原生无时间点云的时间初始化也保留：Yang 均匀分布范围 `[-1,11]`，4C4D 在作者默认 redundant ratio 0 时为 `[0,10]`。初始时间方差均为 2。`train_exec.py` 在原生初始化返回及第 10/100 步读取点数、时间统计和显存；它不增加随机采样，不更新参数。Yang 原生循环的首个循环索引为 2，外壳保留这一行为，没有为记录第 1 步而改写原循环。随后同一进程继续原作者完整日程。

固定最终 `chkpnt30000.pth` 后，`render_heldout.py --render-only` 才渲染 cam00 的全部 300 张 1344×1008 RGB PNG。该阶段不读 cam00 HR 像素。渲染器在 clamp 和整数化之前逐张断言原始预测的形状为 `(3,1008,1344)`，且所有数值有限，避免 NaN 被转成黑色而掩盖失败。PNG 位于每个运行目录的 `render/renders_hr/{frame:04d}.png`；收据必须记录 300 张全部通过。主表的 PSNR / SSIM / LPIPS 由公共 A100 环境和相同权重统一重算；作者内部指标不作为主表数字。

如果作者完整容量出现 OOM，保存原配置、失败、实际点数和显存，不自动减少帧数、初始点数或模型容量。

## A100 部署位置和执行

远端根目录为 `/home/ubuntu/3DGS/4dsr`。独立环境位于 `envs/dynamic_lr_baselines_20261020/{yang,4c4d,mast3r}`。只从现有 base 环境只读导入 Torch 及其依赖，没有改写 base `.venv` 或安装系统包。

`remote_prepare_scene.sh` 是无 GPU 的持久 CPU 准备入口。输入 verified receipt 和清单 SHA 通过后，依次执行 LR staging、实际相机 QA、CPU COLMAP sparse。`make_job.py` 创建 GPU 作业冻结清单；公共 `submit_remote.py` 负责准入、持久执行、退出事件和回传。`queue_after.py` 只在上一作业成功结束并回传后提交下一作业，使用文件退出事件，无模型轮询和自动重放。

完整 checkpoint 可达数 GB，因此当前有限队列使用 `queue_remote_endpoints.py`：读取真实 broker 成功终态后，复核远端最终索引、完整 300 张 PNG、有限数值渲染收据和 GPU 无训练残留，再让下一作业重新通过 broker 准入。原 submitter 同时继续回传全部已注册文件。远端完成和本地回传完成分别记录，不以构造提前的 `collected.json` 跳过回传。控制器支持在尚未派发的项进入前读取明确的重新分配记录；它不会自动重放失败或身份未知的作业，也不改作者训练文件或配置。

`hourly_observer.py` 是独立的持久 CPU 异常兜底。它每 3600 秒通过 SSH 只读一次队列、各作业阶段、训练资源和日志末尾，并保存 GPU 占用清单；不加载 CUDA、不提交或重试作业、不修改远端文件、不发送聊天通知。队列进入完成或失败终态后，本地文件事件会使观察器自动退出。正常完成仍由既有退出事件立即触发渲染、评价及回传，小时观察器不承担正常完成的发现任务。

正式入场前还须通过完整 CPU 部署合同：`audit_full_loader.py` 以原生 12 个 worker、batch size 4 读完 Cook 的全部 6000 张或 Cut 的全部 5700 张训练 LR；实际调用原生相机和图像加载器。`build_cpu_contract.py` 将 manifest、staging、投影 QA、完整 loader、环境出口及训练外壳身份绑定到收据；`gate_native.py` 在 GPU 作业入口复核。MASt3R 的门禁另要求固定官方权重的实际 CPU 模型、检索器及码本加载通过；只运行 `--help` 不算权重加载验收。

部署兼容性集中在自有子进程处理：将 soft `RLIMIT_NOFILE` 提至 `min(65536, hard)`，保留系统和进程的原 hard limit；将 `TMPDIR` 设置为任务独占的短 `/tmp` 目录，避免多进程套接字路径过长。Yang 的旧 `np.byte` 改为 `np.uint8`，并验证修改前后的原始 RGB 字节及真实 loader 张量逐字节一致。Torch 2.7 的 `weights_only=False` 只用于哈希固定且来源明确的原生本地 checkpoint 和官方 MASt3R 权重。以上均为运行兼容与数据接口适配，未改训练采样、作者容量、损失或日程。

MASt3R 的独立构建环境另固定 `setuptools==69.5.1`，因为 ASMK 使用的 NumPy distutils 仍导入新版本已移除的 `distutils.msvccompiler`。MAtCha 的 COLMAP 读取模块还通过焦距工具模块间接导入 `Transform3d`，因此安装作者环境使用的官方 PyTorch3D v0.7.4（提交 `297020a4b1d7492190cb4a909cafbd2c81a12cb5`）。此 SfM 路径不调用任何 PyTorch3D 原生算子，故使用官方 `PYTORCH3D_NO_EXTENSION=1` 选项安装其原样 Python 变换模块；未另写替代变换或渲染实现。其他需要 PyTorch3D C++/CUDA 算子的路径不属于此 MASt3R 环境的验收范围。

每次完整运行的 `result_index.json` 由 `finalize_run.py` 生成，包含数据、最终 checkpoint、全部 PNG、配置、作者代码及补丁身份、训练秒数、显存和时间坐标。`status.json`、CPU 准备收据和环境日志用于区分准备完成、运行中、失败与正式完成。

尚未派发的作业通过 `add_shared_eval_job.py` 生成新身份，在原生渲染成功后立即调用 A100 base `.venv` 的公共 `evaluate_images.py`，然后生成原生索引，再由 `attach_shared_metrics.py` 仅补充已完成指标的路径、哈希和统一 LPIPS 权重身份。评价与该作业共用已准入的 GPU0，串行执行，不等待 GPU1。共同 evaluator、legacy 指标函数及索引附加工具均纳入冻结 SHA。`shared_A100_metrics/{yang|4c4d}/{scene}` 保存统一结果。运行中的 Cook r5 不修改；它的共同评价安排在下一项 Yang Cut 新作业的最前面，且在已经冻结 Cut 训练配置后执行。新增评价不会把测试图像或分数传给原生训练进程，也不改变最终端点选择。

两次已确认未启动的入口拒绝保留为部署证据：第一次 job ID 含下划线被拒绝；第二次声明 39000 MiB 加 broker 固定 2048 MiB 余量超过实际空闲显存，`task_identity=null`。后续使用 37000 MiB 准入估计，模型配置未变。该字段是准入预算，没有设置 Torch 显存上限。

Cook 的早期兼容失败也保留，不列为正式完成实验：r2 在首次 batch 前遇 Pillow 不接受 int8；r3 在首次 batch 前遇套接字路径过长，核实本作业身份后仅停止自有 supervisor；r4 真实执行至 650 步后遇文件描述符上限。它们分别保留原运行目录、配置和失败成本。完成完整 CPU 合同后的 r5 使用新任务身份和新输出目录，重新执行原作者完整 30000 步，没有续接失败分支或改用低容量配置。实际完成状态以各运行的 `result_index.json` 和 broker 终态为准，本说明不将排队或运行中记为成功。
