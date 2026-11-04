"""Chinese handoff report and evidence tables, from authorized saved material."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_bl import *


def md_table(headers, rows):
    def cell(v):
        return str(v).replace("|", "\\|").replace("\n", " ")
    return "\n".join(["| " + " | ".join(headers) + " |",
                      "| " + " | ".join("---" for _ in headers) + " |"] +
                     ["| " + " | ".join(cell(c) for c in row) + " |" for row in rows])


def build():
    audit = read(OUT / "comparisons/argv_comparison.json")
    checks = read(OUT / "checks/cpu_v1/checks.json")
    batches = read(OUT / "sync/batches.json")["batches"]
    commands = read(OUT / "commands.json")
    catalog = read(OUT / "sync/catalog.json")
    old = ROOT / "output/dynamic_sr_modules_20261010/area4_baselines_20261024/localdygs_cem/flame_steak"
    historical = []
    for stage in (1, 2):
        path = old / f"cem/training_stage{stage}_receipt.json"
        value = read(path)
        historical.append(dict(method="CEM", stage=f"stage{stage}", scene="Flame07", seconds=value["elapsed_seconds"],
                               hardware=value["gpu_name"], evidence=identity(path)))
    for stage in ("splatsure_lr", "splatsure_sr", "iesrgs_internal", "iesrgs_sr"):
        path = ROOT / f"output/dynamic_sr_modules_20261010/static_sr_v2_20261026/runs/Flame07/{stage}/complete.json"
        value = read(path)
        historical.append(dict(method=stage.split("_")[0], stage=stage, scene="Flame07",
                               seconds=value["seconds"], hardware=value["gpu"], batch=value["batch"], evidence=identity(path)))
    path = ROOT / "output/dynamic_sr_modules_20261010/sr4d_area4_20261028/runs/Flame07/sr4d_b/fine_complete.json"
    historical.append(dict(method="SR4D", stage="fine40000 only", scene="Flame07", seconds=read(path)["seconds"],
                           hardware="A100 (historical config.environment)", evidence=identity(path)))
    queue = ROOT / "tmp/claude_codex_materials_manifest_costtable_20261102/files/tmp/claude_8a_static_sr_20261026"
    # Exact event intervals; these are shared-job/transfer wall times, not
    # fabricated independent per-method GPU measurements.
    intervals = [
        dict(name="Flame07 teacher upload", seconds=2100, kind="transfer_job_wall",
             began="10-26 23:40:40", ended="10-27 00:15:40", evidence=identity(queue / "queue_a1up.log")),
        dict(name="Coffee07 teacher upload", seconds=1041, kind="transfer_job_wall",
             began="10-26 23:23:14", ended="10-26 23:40:35", evidence=identity(queue / "queue_a1up.log")),
        dict(name="Flame07 static mixed job", seconds=16718, kind="multi_lane_job_wall_not_sum_GPU",
             began="10-27 04:54:42", ended="10-27 09:33:20", evidence=identity(queue / "queue_a1g1.log")),
        dict(name="Coffee07 static mixed job", seconds=20340, kind="multi_lane_job_wall_not_sum_GPU",
             began="10-27 09:33:20", ended="10-27 15:12:20", evidence=identity(queue / "queue_a1g1.log")),
        dict(name="Coffee07+Flame07 SR4D shared job", seconds=20944, kind="two_lane_job_wall_not_per_scene_GPU",
             began="10-28 13:41:31", ended="10-28 19:30:35", evidence=identity(queue / "queue_a1g0.log"))]
    estimates = [
        dict(host="cts", method="cem", scene="sear_steak", GPU_hours=[1.6, 2.6],
             rationale="Flame stage1 20000->21000; stage2 10000->9000; different scene thresholds/points uncertain; includes evaluation"),
        dict(host="cts", method="cem", scene="flame_salmon", GPU_hours=[2.8, 5.0],
             rationale="author40000+10000; stage1 twice Flame budget; actual extracted points uncertain; includes evaluation"),
        dict(host="a100", method="sr4d", scene="sear_steak", GPU_hours=[5.0, 7.0],
             rationale="Flame fine alone4.268h; coarse timing unavailable; no fake coarse receipt"),
        dict(host="a100", method="sr4d", scene="flame_salmon", GPU_hours=[4.5, 7.0],
             rationale="same author budget;18 training views; geometry and convergence cost uncertain"),
        dict(host="a100", method="splatsure", scene="sear_steak", GPU_hours=[1.5, 2.0],
             rationale="Flame LR+SR1.497h, same batch2, add weights/evaluation"),
        dict(host="a100", method="splatsure", scene="flame_salmon", GPU_hours=[2.3, 4.0],
             rationale="batch4 author recipe, Coffee mixed-job wall as coarse bound only"),
        dict(host="a100", method="iesrgs", scene="sear_steak", GPU_hours=[4.0, 5.0],
             rationale="Flame internal+SR3.752h, add DA-V2-L depth/evaluation"),
        dict(host="a100", method="iesrgs", scene="flame_salmon", GPU_hours=[5.0, 8.0],
             rationale="internal remains3 same-time; SR batch4; Coffee job5.65h includes other concurrent work")]
    write_new(OUT / "costs.json", dict(
        schema="newscene_cost_estimates_v1", measurements=historical, intervals=intervals, estimates=estimates,
        cautions=["All new-scene values are estimates", "Mixed historical jobs contain concurrency and unrelated commands",
                  "Historical elapsed wall seconds are not measured CUDA active time", "SR4D coarse timing and Coffee per-stage receipts unavailable"]))

    rows = []
    for n, r in enumerate(audit, 1):
        changed = [t for t in r["token_comparison"] if not t["equal"]]
        label = r.get("historical_step", f"line {r.get('historical_line')}")
        rows.append([n, r["scene"], r["method"], r["historical_label"], label,
                     len(r["token_comparison"]), ", ".join(str(t["position"]) for t in changed) or "无",
                     "通过；完整old/new及每个参数见JSON"])
    text_new(OUT / "comparisons/ARGV_COMPARISON.md", "\n\n".join([
        "# 新场景逐命令、逐参数对照",
        "参数位置从0开始。每一个相同参数也保存在 `argv_comparison.json` 的 token_comparison 中；"
        "只接受精确列举的路径、场景名和作者端点映射，不按“含斜杠”泛化放行。"
        "静态两方法和SR4D同时对照Flame07/Coffee07。CEM读取真实Flame07 steps/native_init收据；Coffee07 CEM实测argv不在授权材料中，"
        "因此没有伪造该项回归，作者Coffee配置仅作为配置差异参考。",
        md_table(["#", "新场景", "方法", "历史", "命令", "参数数", "变化位置", "结果"], rows),
        "CEM train/render表比较展开后的历史stage-b/作者/renderer参数；实际lane入口是cem_new.py，"
        "其包装CLI和绝对解释器另见commands.json。CEM --gpu=0只用作历史比较夹具；实际执行继承固定执行器给出的单个数值mask。"
        "stage2的chkpnt21000/chkpnt40000来自本景作者配置。共享评分器名、scope、--device cuda、--execute均不变。",
        "额外参数：CEM评分显式加 --torch-home 到新输出目录（历史用默认目录）；权重必须与主表相同。"
        "A100评分也仅重定向该缓存路径。shell版补绝对解释器、离线变量和set -e；这些改变执行/缓存位置，不改变训练算法。",
        "新增的frame0 prepare/check、教师索引转换、逐图SHA比较和结果回收验收没有伪装为历史训练命令。"
        "CPU负例验证种子标量变化、伪装成路径的种子均被拒绝。"]))

    totals = defaultdict(lambda: [0, 0, 0])
    for b in batches:
        sums = totals[b["host"], b["group"]]
        sums[0] += b["files"]
        sums[1] += b["estimated_bytes"]
        sums[2] += b["planning_budget_bytes"]
    inventory_table = md_table(
        ["主机", "组", "文件数", "预计GiB", "分批预算GiB"],
        [[host, group, v[0], f"{v[1]/2**30:.4f}", f"{v[2]/2**30:.4f}"] for (host, group), v in sorted(totals.items())])
    teacher_rows, schemes = [], []
    for scene in SCENES:
        g = totals["a100", scene + "_teacher_a"]
        estimate, budget = g[1], g[2]
        # Same file-count scaling of the observed Flame transfer; no assumed
        # network connection or current link measurement.
        historical_seconds = 2100 * g[0] / 6000
        teacher_rows.append([scene, g[0], f"{estimate/2**30:.3f}", f"{budget/2**30:.3f}",
                             f"{historical_seconds/60:.1f}分钟（旧Flame文件数外推）"])
        schemes.append(dict(scene=scene, scheme_a_estimated_bytes=estimate,
                            scheme_a_planning_budget_bytes=budget,
                            scheme_a_wall_seconds_from_Flame_transfer=historical_seconds,
                            scheme_a_seconds_at_MiBps={str(rate): estimate/(rate*2**20) for rate in (10, 30, 100)},
                            scheme_b_uploaded_PNG_bytes=0, scheme_b_generated_bytes_estimate=estimate,
                            scheme_b_GPU_hours_estimate=[0.4, 1.8], scheme_b_SHA_verification_minutes_estimate=[1, 5],
                            scheme_b_weight_and_code_bytes="unknown until owner enumerates teacher checkpoint/dependencies",
                            scheme_b_exact_SHA_acceptance=False))
    write_new(OUT / "teacher_schemes.json", dict(
        schema="newscene_teacher_schemes_v1", preferred="a (local cache upload)",
        estimates=schemes, transfer_rate_measured_now=False,
        historical_teacher_mean_bytes=catalog["byte_assumptions"]["teacher_estimated_each"],
        warning="Same weights/script/batch do not guarantee cross-GPU byte identity. Any SHA mismatch stops scheme b; use original local PNGs."))
    text_new(OUT / "sync/SYNC_PLAN.md", "\n\n".join([
        "# 远端同步与回收清单",
        "本文件只交付清单，没有联网/同步。完整逐文件理由、预计字节、规划预算、已知SHA、远端状态与证据均见catalog.json。"
        "每批upload/*.json严格为 {\"version\":1,\"files\":[{\"path\":\"项目相对路径\"}]}，没有目录或glob占位符。",
        inventory_table,
        "已知代码/清单的字节数为实际只读stat；新场景LR/HR/NPZ/教师PNG没有读取权限，大小为显式估计。"
        "LR每张预计110000字节、规划256KiB；评分HR每张预计1.8MB、规划4MiB；教师按历史Flame全部6000张PNG的平均字节估计、规划4MiB。"
        "4MiB是预算余量，不是对任意附带元数据PNG的硬上界；实际同步前必须重新stat/SHA并检查总量。"
        "metadata含area4两清单、冻结原始非area manifest.json及新registry；没有上传测试LR。"
        "HR只上传cam00全300帧和训练相机0,10,…,290。shared_initialization仅A100需要；CTS在远端实际重算CEM帧0。",
        "a100使用teacher_a组；方案b省略全部*_teacher_a组，增加teacher_b_code和真实权重/模型依赖。"
        "两方案均需原本机producer索引、权重身份summary。dependencies未核实项必须先解决；"
        "缺少权重文件名/第三方目录库存时，这些预装依赖不能被称为完整bootstrap上传清单。"
        "两台主机新目录下的LPIPS AlexNet缓存须从各自主表缓存按SHA复制，源缓存的当前库存本任务不可读。",
        md_table(["场景", "教师PNG", "方案a预计GiB", "方案a预算GiB", "旧网络外推耗时"], teacher_rows),
        "方案a：历史Flame缓存上传35.0分钟，Coffee17.35分钟，均为队列运行到完成墙钟，包含文件传输/写盘等开销。"
        "新sear约35分钟、salmon约31.5分钟；建议每景预留20–60分钟。"
        "另给10/30/100MiB/s有效速率的理想字节下界于teacher_schemes.json；这不是当前网络实测。",
        "方案b：用teacher_bl_new.py→原teacher_cache_new.py→固定SHA teacher_physlr3.py，"
        "同一swinir_area_ft_v1、batch6、mask缓存/补边/量化不变，在新baseline namespace生成。"
        "每景预计0.4–1.8 A100 GPU小时，逐图SHA/PNG验收另预留1–5分钟；生成字节与a近似相同，"
        "只上传原索引和必要代码/权重，权重大小及转运时间当前未知。"
        "所有PNG编码SHA逐个对本机索引，任一不一致即失败；不能以像素接近或指标接近放行。",
        "执行前：由Claude从固定同步入口取得真实前缀列表（本任务未提供）；"
        "用/usr/bin/python3 -I .../finalize_sync_new.py --manifest <批JSON> --allowed-prefixes <真实导出JSON> "
        "--catalog .../sync/catalog.json --out <新验收JSON>核对实际文件数、每文件16GiB、每批32GiB及已知SHA。"
        "然后才交固定入口重新验收/同步；本工具不注册前缀、不替代固定入口。"
        "每批最多20000个文件；当前最大预算见batches.json，未把未知大小当作0。",
        "download/*.json给出8个结果批，含16组scope的summary/per_view.csv/area4_selection/render_index及全部480张固定样本。"
        "样本：test每cam00取0/150/299；train每真实训练相机取0/150/290。"
        "先评分再sv2_cleanup，保留render_index及样本；collect_results_new验证计数、manifest/render SHA绑定和样本SHA后产生result_acceptance。"
        "下载后应在本地再次核对该收据中的所有SHA。检查点回收清单另见checkpoint_download.json；"
        "每批的实际大小仍需下载入口验收。"]))

    # Formal checkpoint recovery, separate from lightweight metrics/samples.
    checkpoints = []
    for scene in SCENES:
        for method in METHOD_NAMES:
            base = OUT / method / scene
            if method == "cem":
                end = read(CODE / f"configs/{scene}/cem_recipe.json")["endpoint"]
                point = base / f"model/point_cloud/iteration_{end}"
                files = [base / "model/cfg_args", base / f"model/chkpnt{end}.pth"] + [
                    point / n for n in ("point_cloud.ply", "dynamic_point_cloud.ply", "shared_point_cloud.ply", "duration.ply")]
                files += [base / "native_init/cem_data/colmap_0/sparse/0" / n
                          for n in ("cameras.bin", "images.bin", "points3D.bin", "points3D.ply")]
            elif method == "sr4d":
                files = [base / "model" / n for n in (
                    "point_cloud/iteration_20000/point_cloud.ply", "deform_lr/iteration_20000/deform_lr.pth",
                    "point_cloud_hr/iteration_40000/point_cloud.ply", "deform_hr/iteration_40000/deform_hr.pth",
                    "deform_hr/iteration_40000/stage_metadata.json")]
            else:
                stages = ("lr", "sr") if method == "splatsure" else ("internal", "sr")
                files = [base / s / "checkpoint.pt" for s in stages]
            path = OUT / f"sync/checkpoints/{method}_{scene}.json"
            write_new(path, dict(version=1, files=[dict(path=rel(f)) for f in files]))
            checkpoints.append(dict(host="cts" if method == "cem" else "a100", method=method, scene=scene,
                                    manifest=rel(path), actual_size_and_SHA="not yet produced; fixed download gate required"))
    write_new(OUT / "sync/checkpoint_download.json", dict(version=1, batches=checkpoints, downloaded=False))

    cost_table = md_table(["主机", "方法", "场景", "估计GPU小时"],
                          [[e["host"], e["method"], e["scene"], f"{e['GPU_hours'][0]}–{e['GPU_hours'][1]}"]
                           for e in estimates])
    geometry_table = md_table(
        ["场景", "训练相机", "训练帧数", "center（训练相机）", "extent", "评分test/train"],
        [[s, len(read(CODE / f"configs/{s}/geometry.json")["train_cameras"]),
          len(read(CODE / f"configs/{s}/geometry.json")["train_cameras"]) * 300,
          read(CODE / f"configs/{s}/geometry.json")["train_camera_center"],
          read(CODE / f"configs/{s}/geometry.json")["cameras_extent"],
          "300/" + str(len(read(CODE / f"configs/{s}/geometry.json")["train_cameras"]) * 30)]
         for s in SCENES])
    report = "\n\n".join([
        "# 新确认场景四基线准备报告（P2；2026-11-03命名空间）",
        "**代码、配置、lane与已知文件清单准备完成；不是训练/远端环境验收报告。** "
        "只在本次两个write_paths新建文件，没有联网、同步、GPU调用、训练提交或派生子代理。"
        "真实LR/HR/NPZ、教师缓存和权重不在本次read_paths，未读取或假造这些结果；"
        "固定同步入口当前前缀及部分预装依赖库存缺失，故全依赖bootstrap清单仍有明确缺项。",
        "## 协议、几何和配置来源",
        geometry_table,
        "两景均为336×252面积LR→1344×1008HR，cam00测试300帧；训练视角评分为每个真实训练相机0,10,…,290。"
        "flame_salmon对应官方flame_salmon_1前300帧，保留camNN真实编号，缺cam03/cam17，不补齐或重编号。"
        "几何从本景训练相机inverse(w2c)重算，逐项对registry验证；没复制旧center/extent或日程。"
        "本任务不生成Ours训练日程，四条基线也不依赖该日程。",
        "CEM配置直接读作者N3V_s1/s2本景文件，仅resolution2→1。sear端点21000→30000，"
        "salmon端点40000→50000；stage2从对应chkpnt21000/chkpnt40000开始，mini_batch=1不改。"
        "batch2/4是Wu静态动态适配的作者场景batch：sear=2、salmon=4；IE internal仍是作者3个同时间视图求和，"
        "SR4D仍是作者单视图迭代，不给它们误加Wu batch参数。",
        "SR4D沿用sr4d_b的train_sr4d_area4.py、sr4d_common.py和固定A100扩展，coarse20000+fine40000、"
        "seed0、edgemap0.01、tex_rate0.001、opacity_reg0.1、save20000/40000全部保持，不使用教师。",
        "SplatSuRe/IE-SRGS直接运行原sv2c.py，从本景新registry完整取Wu配置。"
        "保留主表Wu coarse3000+fine14000、position max_steps20000、densify500/100/10000、opacity_reset60000、"
        "fresh SR及最后fine迭代不更新，作者损失/采样/渲染器不变。新配置纠正历史元数据残留的30k/StableSR标签，"
        "明确实际共同SwinIR areaft教师；这些元数据修正没有改变原sv2c执行。",
        "## CEM帧0初始化",
        "cem_frame0_new.py plan只读元数据；prepare只复制训练LR帧0并用原rotation_quaternion和本景K/w2c生成已知位姿。"
        "run保持历史frame0的默认SIFT、PINHOLE、CPU匹配、单匹配线程、已知位姿数据库ID重映射、"
        "point_triangulator及ba_global_function_tolerance=0.000001，导出TXT/bin/PLY。"
        "新增三角化后K/位姿不漂移、非空有限点及逐文件SHA验收。"
        "不运行非0帧、undistort、PatchMatch、fusion、LocalDyGS合并或poses_bounds复制；这些不向CEM sparse导出提供输入。"
        "新schema为cem_frame0_inputs_v1/cem_frame0_native_init_v1，文件frame0_complete.json；"
        "既不伪造旧联合complete.json，也不接受旧完成schema。",
        "后续cem_new.py通过原run_area4.stage_b和run_author运行作者训练。渲染导入原render_local_cem_area4.py，"
        "只把recipe查询映射到本景新配置和flame_salmon_1别名，协议相机、checkpoint及PNG算法不改。"
        "实际CEM sparse尚未计算，实际COLMAP及作者加载仍必须由Claude验收。",
        "## lane与逐参数对照",
        f"8条方法lane、4条教师备选lane；commands.json列明每条命令的cwd、绝对解释器和实际argv。"
        f"对照{len(audit)}组、{sum(len(r['token_comparison']) for r in audit)}个参数位置全部通过，"
        "详情见[ARGV_COMPARISON.md](comparisons/ARGV_COMPARISON.md)及机器JSON。"
        "A100方法逐行同时对照Flame07和Coffee07，包含训练、权重/深度、render、score和cleanup。"
        "CEM对照真实Flame07 stages/render/score/native COLMAP收据；Coffee07 CEM实测收据不可读，"
        "没有声称做过该实测对照。作者本景JSON相对Flame/Coffee的全部科学参数差异另见cem_author_scene_differences.json。",
        "允许差异均为明确路径、场景名、相机及作者场景参数；额外项逐项披露："
        "CEM独立frame0收据/验收包装；CEM评分增加显式--torch-home；所有缓存/输出重定向新目录；"
        "新教师索引格式转换与逐图SHA；新增结果验收。SplatSuRe历史lane最后附带的3DSR LR任务不在本次范围，"
        "旧TOUCH完成标记由新真实result_acceptance替代。没有把任何数值改名作为本景科学配置。",
        "lanes/*.txt保持历史<script> argv格式，可交历史runner（CTS native与评分需不同解释器，优先用shell版）；"
        "lanes_exec/*.sh补齐绝对解释器且set -e串行失败即停。"
        "从对应远端项目根运行shell文件；这些文件不选择GPU、不提交作业，不复制运行中资产。"
        "A100为/home/ubuntu/3DGS/4dsr/.venv/bin/python；CEM leaf为"
        "/home/cts/Project/4DSR/envs/dynamic_lr_baselines_20261020/cem/bin/python；CTS评分为项目.venv。",
        "## CPU检查与边界",
        f"`/usr/bin/python3 -I {rel(CODE / 'prepare_new.py')}`：通过，生成两景配置。",
        f"`/usr/bin/python3 -I {rel(CODE / 'make_lanes_new.py')}`：通过，生成lane/argv审计。",
        f"`/usr/bin/python3 -I {rel(CODE / 'make_sync_new.py')}`：通过，输出{len(batches)}批已知上传文件。",
        f"`/usr/bin/python3 -I {rel(CODE / 'cem_frame0_new.py')} plan --scene <scene>`：两景通过，只生成计划。",
        f"`/usr/bin/python3 -I {rel(CODE / 'check_cpu_new.py')} --out {rel(OUT / 'checks/cpu_v1')}`："
        f"{checks['status']}，{len(checks['checks'])}项；详见checks/cpu_v1/checks.json。",
        "检查包括真实metadata/registry/SHA/几何、共享冻结选择器300/600/540视角、实际训练相机投影和原四元数、"
        "作者配置仅resolution适配、SR4D/Wu配方一致；合成SQLite覆盖任意COLMAP image/camera ID重映射、"
        "已知位姿/有限点验收；合成PNG覆盖教师SHA/路径/重复key/LR身份/batch/跨主机SHA负例；"
        "参数标量篡改、越界同步路径/前缀/文件数负例；12份shell语法、绝对解释器、源文件保全。"
        "合成文件明确存于checks/SYNTHETIC_ONLY_*，不在任何正式初始化/教师目录，不能作为实验产物。",
        "未验证：真实PNG/NPZ字节、实际COLMAP sparse、teacher库存、权重/扩展的远端SHA、GPU显存/梯度、"
        "训练完成、最终质量和指标、同步入口当前前缀与网络速率。没有把CPU成功称为GPU就绪。",
        "## Claude执行顺序与估计成本",
        "1. 用dependency_inventory逐项核实两机现存代码/解释器/扩展/权重、当前SHA及固定同步前缀；"
        "缺项取得真实文件库存后补精确清单。LPIPS骨干与head在主表版本一致、离线可用；"
        "不要用另一个版本替换静态task-root内已钉住的冻结模块。",
        "2. 每个上传批先运行finalize_sync_new.py实际stat/SHA/限额校验，再由固定同步入口验收/增量传输；"
        "本P2没有执行这些动作。运行preflight_commands.json的四条CPU真实LR/评分HR/初始化验收。",
        "3. CTS只执行cem_frame0 prepare→run→check，检查真实稀疏点与收据后，经固定GPU执行器准入运行两阶段→"
        "test/train render→同共享评分器→cleanup→collect。可以在A100运行独立方法时完成CTS的CPU初始化。",
        "4. A100先按二选一方案准备同一教师索引；方案a推荐上传本机缓存。"
        "方案b必须原脚本/权重/batch6重新生成并全量逐图SHA等于本机；失败转回a，不使用近似缓存。"
        "SR4D无需等待教师；SplatSuRe LR/weights、IE internal亦可先做，SR/depth必须等待教师验收。"
        "两张A100可由Claude在各自固定执行器准入后分配独立链；本交付不预留GPU或并发提交。",
        "5. SplatSuRe按lr→weights→sr，IE按internal→iedepth(DA-V2-L)→sr；"
        "每链随后test/train render、共享score、cleanup、collect。失败保存记录，不自动重试。"
        "最后下载sync/download与checkpoints清单，按result_acceptance重核所有样本/CSV/索引SHA，"
        "将各scope PSNR/SSIM/LPIPS汇入主表；跨硬件细小差异交统一评价环境复核。",
        cost_table,
        "以上均为新景估计，A100合计22.3–33.0 GPU小时，CTS合计4.4–7.6 GPU小时；"
        "方案b另加A100约0.8–3.6 GPU小时，两机/两卡并行的墙钟不能直接相加。"
        "历史Flame CEM训练约1.112+0.465小时，render/score合计约0.252小时；"
        "SplatSuRe训练0.536+0.961小时，IE训练2.139+1.613小时，SR4D仅fine实测4.268小时。"
        "Coffee静态混合作业墙钟5.65小时、Flame4.64小时；Coffee+Flame SR4D混合lane5.82小时，"
        "包含并发/其他任务，不能当作独立每方法GPU小时。SR4D coarse及Coffee逐阶段收据不在材料中，估计保留较宽范围。"
        "历史CEM帧0这五个COLMAP命令合计约11.14秒，而旧联合初始化4351秒包含LocalDyGS；"
        "新景帧0CPU暂预留10–120秒，不把4351秒计入CEM GPU预算。",
        "## 同步、教师方案与回收",
        "详见[sync/SYNC_PLAN.md](sync/SYNC_PLAN.md)、sync/catalog.json逐文件表与sync/batches.json批次。"
        "每批≤20000文件、预算≤32GiB、每文件预算≤16GiB；真实大小须后续验收。"
        "所有批都是具体项目相对文件路径，没有把未知目录写成可上传文件。"
        "现有remote_inventory.py仅是脚本，不是某台主机实际库存；历史成功lane可证明过去使用，不能证明当前仍存在。",
        "本机教师PNG/producer索引不在P2读权限，故新景教师SHA不写假值；"
        "索引转换必须在Claude拿到真实库存后运行。方案a预计sear35/salmon31.5分钟（历史文件数外推，建议每景20–60分钟），"
        "方案b每景0.4–1.8 A100小时另加1–5分钟SHA核验，未知权重/代码传输时间另计。"
        "方案a的PNG字节、方案b生成字节及带宽假设均在teacher_schemes.json。"
        "二者都必须保持同一外部冻结areaft权重，不能重新训练教师。",
        "统一评分method为cem_area4、sr4d_author_area4、splatsure_wu_v2、iesrgs_wu_v2。"
        "共享score_views_area4.py及其frozen副本均不修改；两个scope均评，不能用cam00 LR监督训练。"
        "下载清单包含summary、per_view.csv、render_index、固定样本及训练/初始化/资源收据，"
        "formal checkpoints单列，未虚报已生成或已下载。",
        "## 必须披露的适配与风险",
        "- 共同area4输入、标定/划分/评分及训练LR-only初始化为主表统一接口适配；不是作者原始数据降采样条件。"
        "CEM仅resolution=1和已知位姿LR帧0 sparse；SR4D保留已部署的2026-09-22审计修复和固定avg-pool x4。",
        "- SplatSuRe/IE-SRGS为主表相同的Wu动态适配，保留方法损失/先验/渲染，使用共同Wu日程与areaft外部RGB。"
        "SplatSuRe外部RGB为SwinIR替代其原StableSR教师，必须标注；IE增加DA-V2-L深度。"
        "IE internal为3同时间视图求和，sv2c Mip hypot稳定计算补丁同主表，不能称完全未修改作者方法。",
        "- 三种初始化/先验预算要分开报告：CEM真实LR帧0COLMAP；SR4D/静态方法已登记本景LR-only NPZ；"
        "静态两方法另用共同冻结SR，IE再用外部深度。没有导入Ours gateaff父权重或跨场景模型。",
        "- 当前主要阻断是权限边界造成的真实资产/远端依赖/前缀验收缺失，不是已知训练失败。"
        "未知预装目录和权重文件名不能编造为全量文件同步清单；由Claude取得实际库存后补齐。"
        "没有读取/修改这些未授权材料，也没有扩权或绕过拒绝。",
        "- COLMAP点数、作者场景配置的实际显存/耗时、静态优化稳定性尚未知；"
        "实际triangle相机漂移或任何SHA不符应失败保存证据，不换旧点云或放宽门槛。"
        "跨GPU教师即便相同batch/权重也可能PNG SHA不同，方案b没有bitwise保证。",
        "## 新文件与证据索引",
        "代码与配置完整清单见FILE_LIST.md（代码、config、lane逐项），机器身份清单见DELIVERABLES.json；"
        "config_differences.json记录静态元数据/本景Wu配置的所有差异；"
        "cem_author_scene_differences.json记录作者本景相对Flame/Coffee的所有参数差异；"
        "argv_comparison.json保存每个参数，costs.json保存实测来源SHA和估计依据；"
        "checks/cpu_v1记录CPU证据与合成夹具。",
        "本报告使用中文Markdown，按本次明确要求交付；不写未授权docs/、不发布仓库、不执行AGENTS中更宽的主控自主动作。"])
    text_new(OUT / "REPORT.md", report + "\n")
    # Compare CEM author configs to both historical authors; all non-resolution
    # values are the official scene choices, not copied training schedules.
    cem_diffs = []
    for scene, author in SCENES.items():
        for stage in (1, 2):
            source = ROOT / f"third_party/cem4dgs_20261020/configs/N3V_s{stage}/{author}.json"
            for label, historic in (("Flame07", "flame_steak"), ("Coffee07", "coffee_martini")):
                other = ROOT / f"third_party/cem4dgs_20261020/configs/N3V_s{stage}/{historic}.json"
                cem_diffs.append(dict(scene=scene, stage=stage, historical_label=label,
                                      source=identity(source), historical=identity(other),
                                      differences=differences(read(other), read(source)),
                                      reason="official author per-scene configuration; no manual tuning"))
    write_new(OUT / "comparisons/cem_author_scene_differences.json", cem_diffs)
    print(__import__("json").dumps(dict(report=rel(OUT / "REPORT.md"), checks=len(checks["checks"]),
                                      historical_cost_measurements=len(historical), new_GPU_executions=0)))


if __name__ == "__main__":
    build()
