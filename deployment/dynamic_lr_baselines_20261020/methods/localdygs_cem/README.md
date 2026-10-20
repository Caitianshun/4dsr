# LocalDyGS / CEM-4DGS 原始 LR 接入（2026-10-20）

本目录仅含数据、环境、评价接口和上游补丁。第三方完整实现放在 `third_party/`，不作为本项目原创源码发布。本轮使用最初 `full_time_20261007/prepared/{cook_spinach,cut_roasted_beef}/manifest_train_ready_seed20261007.json`：每景 300 帧，cam00 留出，336×252 原始 LR 训练，1344×1008 HR 留出评价。禁止替换 `_area4`、SR 图或本项目增益模块。

| 方法 | 官方仓库及固定提交 | 完整原生日程 |
| --- | --- | --- |
| LocalDyGS (ICCV 2025) | https://github.com/WuJH2001/LocalDyGS · `39dacdcd8ef6d2b93824df79041713b4a29fb828` | Cook/Cut 各 fine 30k |
| CEM-4DGS (SIGGRAPH Asia 2025) | https://github.com/tho-kn/CEM-4DGS · `7b5dfe3c05456f95c56b3318e406ed6d0f3c05bd` | stage1 到 40k，恢复该检查点到总 50k，stage2 实际再训 10k |

## 数据和相机接口

正式运行副本为 `third_party/localdygs_calibrated_runtime_v3_20261020` 和 `third_party/cem4dgs_calibrated_runtime_v3_20261020`。由 `prepare_runtime.py` 从未改上游生成，补丁保存在副本和本目录 `*_calibrated_interface_v3.patch`。旧的 width-only、calibrated v1/v2 副本仅留档，不能用于本轮正式训练。

`DYNAMIC_LR_PROTOCOL_MANIFEST` 指向原始 manifest。训练图路径、相机名称、帧号、w2c、K 全部按 manifest 的真实 camera_id 绑定，避免 Cut 缺 cam04 时将 cam05 错配到第 4 行。保留真实主点 LR (167.5625,125.5625)，不采用图片宽高的几何中心；Graphdeco 栅格化的半像素约定为 `ndc2pix(v,S)=((v+1)*S-1)/2`。COLMAP 使用角点原点，其主点等于 manifest 主点加 0.5。

公共输入刻意没有 test LR。Local 原生 Dataset 在 test getitem 返回零张量；CEM 使用 `protocol-zero://cam00/` metadata URI，原生 `im_reader` 的必要接口分支返回零张量。两法原生 `training_report` 只读测试分支在 manifest 模式下跳过；Local 只供独立诊断 `evaluate()` 的模块级 VGG 实例化一并跳过。这些零图不参与训练 loss，不输出成真实指标。核心训练函数、loss、采样和优化日程保持上游不变，`audit_training_integrity.py` 比较其抽象语法树哈希。CEM 上游 stage2 和最终 Scene checkpoint load 已显式 `weights_only=False`；Local 最终模型为纯 state_dict。

CPU 验收文件位于 `output/dynamic_lr_baselines_20261020/localdygs_cem/`：

- `camera_projection_cpu_acceptance.json`：两场景全 41 相机、两分辨率、两方法共 164 组 native full projection，最大误差 0.0001367504 像素，门槛 0.005 像素。
- `{cook,cut}_{localdygs,cem}_loader_cpu.json`：实际 Local Dataset，以及 CEM 原生 CameraInfo 工厂/拆分/图像读取函数。Cook 6000、Cut 5700 训练观察均按真实路径核对；每景 300 LR metadata、300 HR metadata 测试图为零，训练/测试位姿误差为 0。该验收未创建完整 GPU Scene，也不能替代训练和渲染验收。
- `{localdygs,cem}_native_training_integrity.json`：核心训练函数 AST 与上游一致，报告函数只增加测试门控。Local 的 VGG 引用仅出现在未被训练入口调用的 `evaluate()`。
- `*_full_native_worker_cpu.json`：Local 实际 FourDGSdataset 和原16worker DataLoader 完整读过 Cook6000/Cut5700 LR；CEM 原 getTrainCameras/getTCams 方法体、im_reader 和2个Loky worker 完整读过 Cook6000/Cut5700 LR。CPU Camera metadata 代替 CEM GPU 存储，未创建模型或分配CUDA。四份 receipt 的 SHA 另汇总在 `native_worker_acceptance_index.json`。

## 作者初始化

`prepare_native_colmap.py` 只复制训练 LR 和已知标定，不读 test/HR 图、不创建点云。`run_native_colmap.py` 在 broker 已分配的单卡中依次运行作者原生初始化，失败立即保存并停止，不自动重放或换初始化。

- LocalDyGS：10、20、…、290、299 共 30 时刻独立 COLMAP 三角化、PatchMatch、stereo fusion。使用作者 affine/domain-size SIFT、最大 16384 特征；CPU SIFT/matcher 仅为硬件接口选择。合并稠密点云，按作者初始 0.02、每轮加 0.01 的 voxel 下采样，最多 90k 点。作者脚本写 `_10.ply`、读取器读 `_50.ply`，仅映射文件名。按本轮信息边界排除 cam00，须在论文披露。
- CEM：第 0 帧 STG 风格 COLMAP 稀疏点。保持原始 manifest 世界坐标，不混用 Yang 已归一化世界的点；Local 多时刻稠密点不代替 CEM sparse。

Cook 初始化产物位于 CTS `output/dynamic_lr_baselines_20261020/cts_native_localdygs_cem/cook_spinach/native_init/{localdygs_data,cem_data}`。初始 broker ID 为 `codex-dynlr-localdygs-cem-ctsg0-20261020-cook-spinach-init`，使用 GPU0、8GiB 申报预算。首个含下划线 ID 已被入口明确拒绝，未派发，失败记录保留。

Cook 初次运行完成12个稠密时点（frame10至120），每时点138.16–153.85秒，均值144.64秒；frame130 CPU matcher 因 `FeatureMatcherCache::GetCamera` 抛出 `std::out_of_range` 失败。数据库20个相机/20张图无孤儿ID。部署 COLMAP `af2b8659` 的[官方 matcher.cc 第189–200行](https://github.com/colmap/colmap/blob/af2b8659/src/colmap/feature/matcher.cc#L189-L200)先发布 map 指针再逐项填充，多个 verifier 可能读到尚未填完的相机缓存。修复只将 CPU matcher 线程设为1，保留特征和几何选项；初始化失败状态、DB、日志、旧源码及12份点云 SHA 已冻结并回传 `cook_failed_initialization_20261020/`。获准的新 broker ID `codex-dynlr-cook-colmap-continue1-cts0-20261020` 显式续跑，跳过成功阶段，不重建完成点云。`--resume-failed` 只接受该已诊断 matcher SIGABRT，不能用于自动重放任意失败。

Cut 准备转至 A100 GPU1。CPU staging 为31时点×19相机，保留缺cam04/存在cam05的真实名称；689项源/输入 SHA、589相机标定检查通过（最大误差1.44e-15）。冻结 job 是 `jobs/a100_cut_native_init_gpu1.json`，是否派发由主控当时准入决定。该机 COLMAP 为3.9.1（commit Unknown），binary SHA `861adf9a3ea43a3b2469f3c093c010713001c893c06043dfb3b512791210cbc5`；CTS 为3.12.0.dev0 `af2b8659`，SHA `1098f840af01c745aa011839cafdb5f36bda4219cbe82ff8d4d92056bc0bf1b3`。二者初始化软件版本不同，不能声称仅硬件不同；保持同一作者流程并披露此差异。A100 冻结接口在 `output/.../localdygs_cem/a100_cut_init_preparation/`，输出在 `output/.../a100_native_localdygs_cem/cut_roasted_beef/`。

## 隔离环境

CTS 环境路径为 `envs/dynamic_lr_baselines_20261020/{localdygs,cem}`。通过只读 `.pth` 借用 base torch 2.7.1+cu128，其余依赖和 CUDA 扩展安装在独立 leaf；不改系统/base。`build_isolated_env.sh` 只编译，不分配训练 GPU。缺 Debian ensurepip 时使用无 pip venv，再只读借用 base pip。

Local 需要 torch_scatter 2.1.2、Tiny CUDA NN 1.7、mmcv 1.6 和自己的 rasterizer。torch_scatter 使用官方 pt27cu128 wheel，SHA256 `3a20acd4777bc1d996756df5bb0c2ac42440fd434315e37464715567699b4bfb`。Tiny CUDA NN 没有 v1.7 Git tag，固定声明 VERSION 1.7 的官方提交 `3294b150084266f24a04009d47dcb530d0a8fb03`。

CEM 需要 PyTorch3D v0.7.9（提交 `33824be3cbc87a7dd1db0f6a9a9de9ac81b2d0ba`）、真正带 `StandardGpuResources` / `GpuIndexFlatL2` 的 Faiss GPU v1.12.0（提交 `e8234e563f1ecef5f036e83c3cfee366d3f1fbca`），不能用 faiss-cpu 替代。私有 NVCC toolkit 缺 cuSPARSE/cuBLAS 头文件，但 base 的匹配 NVIDIA 包已有；`prepare_cuda_overlay.py` 在 leaf 内建立只读 symlink overlay，不改共享 toolkit。SWIG 安装在 leaf，不 sudo。

远端直连 Git/data.pyg DNS/TLS 失败已留日志；官方源码和 wheel 经本机获取后同步，7007 个依赖源文件 SHA 全验收通过，见 `dependency_source_verified.json`。设置 `FOURDSR_DEPS_PREVERIFIED=1` 仅允许使用这些已验收固定提交，不再网络克隆。构建恢复需显式 `FOURDSR_ENV_RESUME=1`；该开关不允许重放失败训练。

环境构建日志保存在 CTS `output/dynamic_lr_baselines_20261020/environment_builds/`。Local r5 已实证 simple-knn 在 CUDA12.8 下缺少 `FLT_MAX` 定义；`prepare_simple_knn_build.py` 只在 leaf 私有 build-source 添加标准 `<cfloat>` 头文件，保存 `localdygs_simple_knn_cfloat.patch` 和源/补丁 SHA，所有 kernel 函数体不变，未修改上游仓库。Local r6 已成功编译 tcnn、原生 rasterizer、simple-knn，并通过 CPU 导入/二进制存在检查；真实 CUDA 依赖检查仍须 broker 内执行，不能称为训练通过。CEM 上游已带 `<cfloat>`，不需要该补丁；CTS PyTorch3D 已编译，Faiss GPU 仍在编译。另由主控分配的 A100 独立 sm80 CEM 环境正在准备。

## 训练、HR 渲染与结果索引

`build_job.py` 冻结 required_files SHA、输入 receipt、GPU UUID、串行命令和回传路径；不提交 GPU。`--initialization-only` 只准备作者点云，`--skip-initialization` 只训练已有点云，`--inputs-already-prepared` 复用已核验 CPU staging。正式训练仍由主控 broker 做当时资源准入，不手工 nohup CUDA 进程。

每景流程：Local 30k → HR 渲染 → 索引；CEM 40k → 原 checkpoint 恢复到 50k → HR 渲染 → 索引。目前调度为 CTS 优先 Cook Local、随后 Cut Local；A100 GPU1 待现有训练释放后由主控安排 Cut 初始化和 Cook/Cut CEM，避免在两机重复训练。`after_initialization.py` 监听已存在初始化 collector 的 PID 退出，在成功回传及标定通过后冻结单个 Local job 并交 broker；失败只保存状态，不重放，不派发 CEM。CEM prepared LR 配置只将 resolution 2 改为 1，避免已准备 LR 二次缩小；其他核心配置沿用作者。`run_author.py` 保存原始 argv、接口 receipt、耗时及 Torch 分配器 peak memory（不包括外部 Faiss/COLMAP）。不能以 held-out test 指标挑模型，固定作者完整日程最终端点。Local 原入口没有保存完整 optimizer 状态，其正式最终资产是 iteration_30000 的点云、hash/MLP 权重；不得称为可完整恢复 optimizer 的 checkpoint。

正式作业将 TMPDIR/TMP/TEMP 设为短路径 `/tmp`，只提高自有训练进程的 soft RLIMIT_NOFILE 至 min(65536,hard)，保留系统/hard限制和原生worker数，避免已实证的Unix socket长路径、1024文件句柄上限。OMP/MKL/OpenBLAS/NUMEXPR线程池各1，限制CPU资源竞争；不改训练采样、GPU算子、模型或迭代数。GPU依赖检查在broker内执行，使用固定算子测试，确认方法自己的rasterizer/simple-knn等均从对应leaf导入并记录binary SHA；Tiny CUDA NN 顶层import会查询真实GPU，不能在CPU构建服务运行。

CEM 原生 `train.py` 的 main 通过 `--gpu` 重写 `CUDA_VISIBLE_DEVICES`，因此该参数必须是 broker 选择的物理卡编号；A100 GPU1 必须传 `--gpu 1`。`run_author.py` 在导入上游代码前严格验证唯一参数与 broker 单整数 mask 相同，不匹配就退出。全仓检查确认 `args.gpu` 没有其他用途；作者 `safe_state` 和 Faiss 之后使用 mask 内 logical0。九个 CPU guard 合同已通过且没有导入 Torch，记录在 `cem_broker_device_guard_cpu.json`。HR 自有渲染入口不执行上游 `render.py` main 的 mask 赋值，直接继承 broker 可见范围。

`render_hr.py` 直接调用作者最终模型加载器和 rasterizer，设置 `DYNAMIC_LR_RENDER_HR=1` 仅改变 test K/宽高。Local 按帧构造 image/camera，CEM 不获取 test 图像生成器，不一次性堆 300 张 HR GPU 图。每帧先断言原始 prediction 为 `(3,1008,1344)` 且全部 finite，再量化。输出完整 `renders_hr/0000.png` 至 `0299.png`。统一量化是 `np.rint(np.clip(CHW_to_HWC_float32,0,1)*255).astype(np.uint8)` 后 PIL RGB PNG，receipt 记录300帧 finite、rounding 和逐帧 SHA。

`index_result.py` 仅在完整训练和 300 张 HR RGB PNG 均通过后写每法每景 `result_index.json`，包含 manifest、最终模型资产和 hash、源补丁/配置身份、硬件/耗时/内存范围、渲染清单。最终指标由主控在 A100 base 环境对这些 PNG 和真实 HR cam00 统一复算。此目录的 native LR 指标和零占位诊断均不能报为 HR 主表。

Local 正式 CTS 作业在 HR render 成功退出后，立即用已准入GPU0和 CTS base `.venv` 调用公共 `evaluate_images.py`。结果写入 `output/.../provisional_CTS_metrics/localdygs/{scene}`，只作即时质量核查；A100 对同一 PNG 的统一复算才进入主表。job 同时冻结公共 evaluator、旧指标 helper 和官方 AlexNet cache 的 SHA。共用 index 接受 `--provisional-metrics` 或 `--unified-metrics`，明确记录可否进入主表，并保留逐帧报告。

`hourly_observer.py` 是这轮有限队列的 CPU 异常兜底。它监听既有 Local follow-on 服务的 PID 退出；等待期间每小时只读记录两个本地服务、已有 broker/阶段状态和失败阶段末尾错误。它不派发GPU、不重试、不清锁、不终止作业，也不声明聊天通知；follow-on 完成或失败后保存末次快照并自行退出。Cook 实例为 `4dsr-cook-local-hourly-observer-20261020.service`，首次兜底时间19:35:58 UTC，之后间隔至少3600秒；receipt 和 JSONL 快照位于本地 `output/.../localdygs_cem/cook_hourly_observer_*`。该兜底不承担正常完成后的渲染和评价，后者仍由原退出事件链立即执行。

各方法输出内的 `source_metadata/` 回传小型复现材料：上游提交、原始接口 receipt/patch、场景配置、执行 helpers、训练时环境版本/编译器、私有构建身份、必要头文件修复 receipt，以及初始化和标定记录。`metadata_index.json` 同时保留原来源路径、结果内相对路径和 SHA；不回传第三方完整安装树。跨主机接入通过 `audit_transferred_initialization.py` 验证原31时点标定和当前点云 SHA，保留 A100/CTS 原来源路径，不为适配接收机改写科学记录。
