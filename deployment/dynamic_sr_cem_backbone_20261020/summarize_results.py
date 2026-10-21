"""CPU-only, receipt-based final comparison of CEM area4 and fixed Wu references.

Reads accepted indexes, training receipts, RGB PNG identities and common A100
metric CSV/summary files. Never imports a model, opens a checkpoint, parses a
training log, uses CUDA or connects to a host. Default output requires all28
endpoints. --allow-incomplete creates a marked research record, never completion.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import struct
import sys

ROOT = Path(__file__).resolve().parents[2]
BATCH = ROOT / "output/dynamic_sr_cem_backbone_20261020"
SCENES = ("cook_spinach", "cut_roasted_beef")
ARMS = ("areaL0", "naive", "areaT", "areaTm", "areaTm_w4", "areaOracle")
WU_ARMS = dict(zip(ARMS, ("areaL0", "naive", "areaT", "areaTm", "w4", "oracle")))
METRICS = ("psnr", "ssim", "lpips")
GROUPS = ("xyz", "f_dc", "f_rest", "opacity", "opacity_center", "opacity_var",
          "scaling", "rotation", "xyz_disp", "motion_xyz", "motion_rotation")
ENV_KEYS = ("torch", "numpy", "opencv", "gpu", "evaluator_sha256", "helper_source_sha256", "lpips_state_dict_sha256")
REMOTE_ROOTS = ("/home/ubuntu/3DGS/4dsr", "/home/cts/Project/4DSR")
CONTROLS = {
    "areaL0": dict(lr_operator="area", teacher=None, teacher_weight=0.0, w4=False),
    "naive": dict(lr_operator="bicaa", teacher="original", teacher_weight=0.2, w4=False),
    "areaT": dict(lr_operator="area", teacher="original", teacher_weight=0.2, w4=False),
    "areaTm": dict(lr_operator="area", teacher="matched", teacher_weight=0.2, w4=False),
    "areaTm_w4": dict(lr_operator="area", teacher="matched", teacher_weight=0.2, w4=True),
    "areaOracle": dict(lr_operator="area", teacher="train_HR", teacher_weight=0.2, w4=False),
}
NOTES = [
    "输入为4×4面积平均生成的area4 LR（336×252），HR为1344×1008。Cook用cam01–cam20共20台训练相机；Cut有19台训练相机，cam04缺失；精确列表由各manifest给出。cam00留出，主指标覆盖固定全部300帧。旧bicubic LR CEM基线不进入本表。",
    "全部主指标来自同一A100环境、同一评价源码及AlexNet LPIPS权重的8位RGB PNG。PSNR为逐帧PSNR的算术平均；两景均值为两场景均分的等权平均。数值差分统一为目标减来源，PSNR/SSIM正值更好，LPIPS负值更好。",
    "CEM native HRref从同一仅LR初始化点云独立开始，以训练HR直接监督作者40k→总50k日程；它是有额外信息的拟合诊断。areaOracle从已完成LR父模型继续6000步、固定拓扑且无w4，仍用混合目标：第一个采样训练相机的LR项系数为1，另外两台同时间训练相机的HR项各为0.2，并保留原生正则；所有SR臂共有的颜色仿射先作用于渲染。系数不等于实测梯度贡献，不能据此断言LR梯度占主导。二者还改变监督目标、优化历史、拓扑成长机会和预算，不能互相替代，也不是严格质量上界。",
    "areaL0是相同6000步SR预算下的无teacher继续训练对照。原生父模型到areaL0还共同改变HR尺寸渲染后的area观测约束、学得SR颜色仿射、固定三视角采样与继续训练流程，不是仅追加6000步的单因素对照。",
    "CEM六SR臂共享原生LR父端点，继续原生11组RAdam状态和绝对时钟50001–56000，共6000次参数更新；固定拓扑但仍更新原生几何、外观、运动和寿命参数。RAdam各组实际步数按有梯度参与计数保留，不能仅由全局6000步推定各组均更新6000次；实际计数以逐组收据为准。原生motion scheduler未另行修复。",
    "Wu引用固定历史开发检查点，先前选择受开发评价影响；重渲染不消除此选择。Wu父流程为coarse3000+fine14000，继承训练相机颜色affine；SR重新计时1–6000并再次增密/剪枝。CEM父流程为作者40k→总50k、不加颜色affine，六SR臂affine从零开始、固定拓扑。二者不是仅替换骨干的单因素实验。",
    "训练硬件逐项保留：CEM LR父与SR在A100，native HRref在CTS RTX3090；Wu历史分支混用PRO6000/RTX3090。统一A100评价只能减少评价实现与硬件差异，不能消除训练硬件、优化日程、初始容量和历史选择差异；细微差异不能直接归因于骨干。",
    "所有cam00输出为原始渲染，无额外posthoc cull、训练相机颜色或位移校正、测试图像拟合。cam01 raw/校正图、固定ROI与时序诊断由独立diagnostics_images.py输出；本汇总不把全图均分或锐度当作几何、时间一致性或人工验收。",
    "资源表的SR秒数为分支增量，父模型耗时单列并由六臂共享，不重复计入六臂总成本。原生两阶段秒数求和；显存是各阶段峰值的最大值。PyTorch allocator峰值不含外部Faiss等分配；不同硬件耗时不可当作等算力比较。初始化/教师生成/传输/失败尝试成本不包含在训练秒数内。",
    "本工具不读取大checkpoint。模型身份来自已完成原生索引或经过CPU checkpoint核验的SR索引；重新核验索引/收据/评价CSV/全部300PNG字节SHA与PNG头，以及GT的manifest身份。完成标记只证明本结果矩阵及评价身份齐备，不是科学主张或人工视觉验收。",
]
ARM_DESCRIPTIONS = {
    "areaL0": "相同6000步SR预算下的无teacher继续训练；HR渲染经4×4面积平均约束到LR。原生父模型到areaL0还改变了SR颜色仿射和固定三视角采样流程，并非仅增加步数。",
    "naive": "原版SwinIR teacher；LR约束用带抗混叠的bicubic降采样，作为直接接入SR监督的对照。",
    "areaT": "保留原版teacher，仅将LR约束换为与输入一致的area算子；areaT−naive检验前向退化一致性。",
    "areaTm": "保留area约束，将teacher换为针对area退化微调并冻结的版本；areaTm−areaT检验先验匹配。",
    "areaTm_w4": "areaTm加每训练相机4×4平滑二维位移场；w4−areaTm检验观测校正。cam00不使用该校正。",
    "areaOracle": "areaTm的teacher替换为训练相机真实HR，权重0.2且不加w4；Oracle−areaTm检验先验信息不足的影响。仅作额外信息诊断。",
}


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def identity(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size)


def finite(value, label):
    require(type(value) in (int, float) and math.isfinite(value), "Non-finite/non-number " + label)
    return value


class Context:
    def __init__(self, root=ROOT, batch=BATCH):
        self.root, self.batch = Path(root).resolve(), Path(batch).resolve()
        self.files = {}

    def local(self, path):
        p = Path(path)
        if not p.is_absolute():
            p = self.root / p
        elif not p.is_relative_to(self.root):
            matches = [prefix for prefix in REMOTE_ROOTS if p.is_relative_to(prefix)]
            require(len(matches) == 1, "Unrecognized original host path: " + str(p))
            p = self.root / p.relative_to(matches[0])
        p = p.resolve()
        require(p.is_relative_to(self.root), "Path escapes project: " + str(p))
        return p

    def check(self, path, expected=None, png=False):
        path = self.local(path)
        stat = path.stat()
        signature = (stat.st_ino, stat.st_size, stat.st_mtime_ns)
        key = str(path)
        if key not in self.files:
            value = identity(path)
            after = path.stat()
            require(signature == (after.st_ino, after.st_size, after.st_mtime_ns), "File changed while hashing: " + key)
            self.files[key] = (signature, value)
        cached_signature, value = self.files[key]
        require(signature == cached_signature, "File changed after verification: " + key)
        if expected:
            require(value["sha256"] == expected["sha256"], "SHA differs: " + key)
            if "bytes" in expected:
                require(value["bytes"] == expected["bytes"], "Size differs: " + key)
        if png:
            with path.open("rb") as stream:
                header = stream.read(29)
            require(len(header) == 29 and header[:8] == b"\x89PNG\r\n\x1a\n" and header[12:16] == b"IHDR", "Not PNG: " + key)
            require(struct.unpack(">IIBB", header[16:26]) == (1344, 1008, 8, 2), "Not HR8bit RGB PNG: " + key)
        return value

    def metadata(self, item):
        self.check(item["path"], item)
        return read(self.local(item["path"]))


def manifest_for(ctx, scene):
    path = ctx.root / "data/dynamic_sr/full_time_20261007/prepared" / (scene + "_area4") / "manifest_train_ready_seed20261007.json"
    value = read(path)
    require(value["scene"] == scene and value["physical_lr_derivation"]["formation"] == "area4", "Wrong scene/degradation")
    require(value["frame_indices"] == list(range(300)) and value["splits"]["test"] == ["cam00"], "Wrong frames/split")
    require(value["resolutions"] == dict(lr=[336, 252], hr=[1344, 1008]), "Wrong resolutions")
    test = sorted((row for row in value["observations"] if row["split"] == "test"), key=lambda row: row["frame_index"])
    require([(row["camera_id"], row["frame_index"]) for row in test] == [("cam00", i) for i in range(300)], "Wrong heldout records")
    return path, value, test


def verify_metrics(ctx, directory, render_dir, scene, manifest, test, rendered, expected_env, method):
    directory, render_dir = Path(directory), Path(render_dir)
    summary_path, csv_path = directory / "summary.json", directory / "per_frame.csv"
    summary, rows = read(summary_path), list(csv.DictReader(csv_path.open()))
    require(summary["schema"] == "dynamic_lr_shared_HR_image_metrics_v1" and summary["status"] == "completed", "Incomplete/wrong common metric schema")
    require(summary["scene"] == scene and summary["method"] == method and summary["camera"] == "cam00", "Wrong metric method/scene/camera")
    require(summary["frames"] == 300 and summary["resolution"] == [1344, 1008], "Wrong metric size/count")
    require(summary["manifest_sha256"] == ctx.check(manifest)["sha256"], "Metric manifest differs")
    require(summary["input_representation"] == "8-bit rendered RGB PNG for every method", "Raw/PNG metric mixing forbidden")
    require(summary["device"] == "cuda" and "A100" in summary["gpu"], "Common evaluation must use A100")
    require({key: summary[key] for key in ENV_KEYS} == expected_env, "Common evaluator/environment/LPIPS differs")
    require(summary["psnr_aggregation"] == "arithmetic mean of per-frame full-image PSNR", "PSNR aggregation differs")
    require(summary["ssim_definition"] == "RGB 11x11 Gaussian sigma1.5 population covariance; exclude outer5px", "SSIM differs")
    require(summary["lpips_definition"] == "AlexNet v0.1 input[-1,1] full-image scalar", "LPIPS differs")
    require(ctx.local(summary["render_directory"]) == render_dir.resolve(), "Metric render-directory differs")
    require([int(row["frame"]) for row in rows] == list(range(300)), "Missing/extra/reordered CSV frame")
    require([entry["frame"] for entry in rendered] == list(range(300)), "Missing render identity")
    for frame, row in enumerate(rows):
        require(row["render_sha256"] == rendered[frame]["sha256"], "CSV/render SHA differs")
        require(row["reference_sha256"] == test[frame]["hr_sha256"], "CSV/GT SHA differs")
        for key in (*METRICS, "mse"):
            finite(float(row[key]), "CSV " + key)
    for key in METRICS:
        finite(summary["test"][key], "summary " + key)
        require(abs(math.fsum(float(row[key]) for row in rows) / 300 - summary["test"][key]) < 1e-10, "CSV mean differs: " + key)
    return dict(values=dict(summary["test"]), summary=ctx.check(summary_path), per_frame=ctx.check(csv_path),
                original_summary=summary, evaluation_environment={key: summary[key] for key in (*ENV_KEYS, "device", "host")})


def render_contract(ctx, run, manifest_hash, checkpoint_hash, kind, index):
    path = run / "renders_hr/render_receipt.json"
    rendering = read(path)
    require(rendering["status"] == "completed" and rendering["kind"] == kind, "Incomplete/wrong rendering")
    require(rendering["camera"] == "cam00" and rendering["scope"] == "heldout_300", "Wrong rendering scope")
    require(rendering["frames"] == rendering["finite_before_clip_quantization_passed_frames"] == 300, "Invalid render frames")
    require(rendering["frame_indices"] == list(range(300)), "Wrong render indices")
    require(rendering["manifest"]["sha256"] == manifest_hash and rendering["checkpoint"]["sha256"] == checkpoint_hash, "Render source identity differs")
    require(rendering["colour_affine"] == rendering["warp"] == "identity", "Heldout correction forbidden")
    require(not rendering["heldout_pixels_read"] and not rendering["training_pixels_read"], "Renderer fitted/read target pixels")
    require(rendering["iteration"] == (56000 if kind == "sr" else 50000), "Wrong final endpoint")
    receipt = ctx.check(path)
    expected_receipt = index["heldout_rendering"]["receipt"]["sha256"] if kind == "sr" else index["renders_receipt_sha256"]
    require(receipt["sha256"] == expected_receipt, "Indexed render receipt differs")
    require([row["frame"] for row in rendering["records"]] == list(range(300)), "Wrong render records")
    records = []
    for row in rendering["records"]:
        require(row["path"] == f"{row['frame']:04d}.png", "Unexpected render path")
        file_id = ctx.check(run / "renders_hr" / row["path"], row, png=True)
        records.append(dict(frame=row["frame"], **file_id))
    require({p.name for p in (run / "renders_hr").glob("*.png")} == {f"{i:04d}.png" for i in range(300)}, "Extra/missing render PNGs")
    return rendering, receipt, records


def native(ctx, run, scene, key, manifest, test, expected_env, metric_dir):
    run = Path(run).resolve()
    index_path = run / "result_index.json"
    index = read(index_path)
    level = "hr" if key == "HRref" else "lr"
    manifest_hash = ctx.check(manifest)["sha256"]
    require(index["status"] == "native_endpoints_and_300_renders_verified" and index["train_level"] == level, "Wrong/incomplete native index")
    require(index["manifest_sha256"] == manifest_hash and not index["heldout_used_for_training"], "Native manifest/leakage differs")
    stages, resource_stages, identities = [], [], [ctx.check(index_path)]
    for number, endpoint in [(1, 40000), (2, 50000)]:
        path = run / f"stage{number}.json"
        state = read(path)
        entries = [row for row in index["stages"] if Path(row["path"]).name == path.name]
        require(len(entries) == 1, "Native index stage missing/duplicated")
        identities.append(ctx.check(path, entries[0]))
        require(state["status"] == "completed" and state["stage"] == number and state["train_level"] == level and state["endpoint"] == endpoint, "Wrong native stage/endpoint")
        require(state["manifest"]["sha256"] == manifest_hash and not state["heldout_pixels_used"], "Wrong stage manifest/heldout use")
        require(state["native_schedule_preserved"] and not state["parent_added_affine"] and not state["parent_added_warp"], "Native core altered")
        require(state["author_optimizer_restored"] == (number == 2), "Native stage optimizer restoration differs")
        require(state["train_HR_used"] == (level == "hr"), "Native supervision differs")
        require(entries[0]["checkpoint_sha256"] == state["final_checkpoint"]["sha256"], "Native checkpoint identity differs")
        require(entries[0]["elapsed_seconds"] == state["elapsed_seconds"] and entries[0]["peak_reserved"] == state.get("torch_peak_reserved_bytes"), "Native indexed resource differs")
        finite(state["elapsed_seconds"], "native training seconds")
        stages.append(state)
        resource_stages.append(dict(stage=number, endpoint_iteration=endpoint, training_GPU=state["gpu_name"],
            seconds=state["elapsed_seconds"], torch_peak_reserved_bytes=state.get("torch_peak_reserved_bytes"),
            peak_memory_scope=state.get("peak_memory_scope"), receipt=identities[-1]))
    require(stages[1]["start_checkpoint"]["sha256"] == stages[0]["final_checkpoint"]["sha256"], "Native stage transition differs")
    require(len({s["source"]["LR_initialization_pointcloud_sha256"] for s in stages}) == 1, "Native initialization differs across stages")
    rendering, render_id, records = render_contract(ctx, run, manifest_hash, stages[-1]["final_checkpoint"]["sha256"], "native", index)
    require(rendering["training_receipt"]["sha256"] == identities[-1]["sha256"], "Render/native stage2 receipt differs")
    method = "CEM_area4_native_HRref" if level == "hr" else "CEM_area4_native_parent"
    metrics = verify_metrics(ctx, metric_dir, run / "renders_hr", scene, manifest, test, records, expected_env, method)
    if level == "lr":
        require(index["metrics_status"] == "shared_A100_complete" and index["metrics"] == metrics["original_summary"], "Native parent index/common metrics differ")
    else:
        require(index["metrics_status"] in {"shared_A100_evaluation_pending", "shared_A100_complete"}, "Unexpected HRref index status")
    peaks = [s["torch_peak_reserved_bytes"] for s in resource_stages if s["torch_peak_reserved_bytes"] is not None]
    return dict(system="CEM", scene=scene, arm=key, run=str(run), kind="native_HR_reference" if level == "hr" else "native_LR_parent",
        status="verified", metrics=metrics, manifest=ctx.check(manifest), checkpoint=stages[-1]["final_checkpoint"],
        initialization_LR_pointcloud_sha256=stages[-1]["source"]["LR_initialization_pointcloud_sha256"],
        provenance=identities + [render_id], native_original_index_metrics_status=index["metrics_status"],
        render_receipt=render_id, renders=records, rendering_GPU=rendering.get("GPU_name"),
        training_HR=level == "hr", heldout_training=False,
        budget=dict(author_stage_endpoints=[40000, 50000], final_iteration=50000, actual_parameter_updates=None,
            note="作者迭代端点；保存/optimizer.step顺序沿原码，不把编号50000等同精确50000次参数更新。"),
        resources=dict(training_GPU=[s["training_GPU"] for s in resource_stages], training_seconds=math.fsum(s["seconds"] for s in resource_stages),
            torch_peak_reserved_bytes=max(peaks) if peaks else None, stages=resource_stages, shared_parent=False),
        original_training_receipts=stages)


def sr(ctx, run, scene, arm, manifest, test, expected_env):
    run = Path(run).resolve()
    path = run / "result_index.json"
    index = read(path)
    require(index["schema"] == "CEM_area4_SR_verified_result_v1" and index["status"] == "SR_endpoint_and_303_renders_unified_metrics_verified", "Wrong/incomplete SR index")
    require(not index["fixture_only"] and index["CPU_checkpoint_verification"], "Fixture/unverified checkpoint cannot enter main table")
    require(index["scene"] == scene and index["arm"] == arm and index["manifest"]["sha256"] == ctx.check(manifest)["sha256"], "SR method/manifest differs")
    require(index["actual_parameter_updates"] == 6000 and index["absolute_final_iteration"] == 56000 and index["fixed_topology"], "SR budget/topology differs")
    require(not index["heldout_used_for_training"] and index["training_HR"] == (arm == "areaOracle"), "SR supervision differs")
    state = ctx.metadata(index["training_receipt"])
    require(state["status"] == "completed" and state["arm"] == arm, "SR training incomplete")
    require(state["controls"] == CONTROLS[arm] and state["train_HR_used"] == (arm == "areaOracle") and not state["heldout_pixels_used"], "SR controls/information boundary differs")
    require(state["steps_requested"] == state["steps_completed"] == state["parameter_updates"] == 6000, "SR actual updates differ")
    require(state["absolute_start"] == 50001 and state["final_iteration"] == 56000, "SR clock differs")
    require(state["optimizer_state_inherited"] and not state["native_scheduler_repaired"] and not state["parent_affine"], "SR native optimization contract differs")
    require(not state["gradient_error_statistics_updated"] and state["topology_parent_and_level_unchanged"], "SR topology/statistics changed")
    require(state["initial_point_counts"] == state["final_point_counts"], "SR point counts changed")
    require(state["parent_checkpoint"]["sha256"] == index["parent_checkpoint"]["sha256"] and state["parent_stage2_receipt"]["sha256"] == index["parent_stage2_receipt"]["sha256"], "SR parent identities differ")
    require(state["manifest"]["sha256"] == index["manifest"]["sha256"] and state["schedule"]["sha256"] == index["schedule"]["sha256"], "SR input identities differ")
    counters = index["native_RAdam_step_counters"]
    require(set(counters) == set(GROUPS), "Not native11RAdam groups")
    for name in GROUPS:
        counter = counters[name]
        require(all(type(x) is int and x >= 0 for key in ("initial", "final", "updates") for x in counter[key]), "Invalid RAdam step counter")
        require(len(counter["initial"]) == len(counter["final"]) == len(counter["updates"]) == 1, "Unexpected group parameter shape")
        require(counter["updates"] == [counter["final"][0] - counter["initial"][0]] == [counter["steps_with_grad"]], "RAdam update/gradient participation differs")
        require(0 <= counter["steps_with_grad"] <= 6000, "RAdam participation exceeds update count")
        require(state["native_RAdam_step_counters"][name] == {k: counter[k] for k in ("initial", "final", "updates")}, "Receipt/RAdam count differs")
        require(state["native_gradient_participation"][name]["steps_with_grad"] == counter["steps_with_grad"], "Receipt/gradient participation differs")
    require(index["training_GPU"] == state["gpu_name"] and index["training_seconds"] == state["elapsed_seconds"], "SR resource differs")
    finite(index["training_seconds"], "SR training seconds")
    require(index["training_seconds"] >= 0 and type(index["torch_peak_reserved_bytes"]) is int and index["torch_peak_reserved_bytes"] >= 0, "Invalid SR resources")
    rendering, render_id, records = render_contract(ctx, run, ctx.check(manifest)["sha256"], index["checkpoint"]["sha256"], "sr", index)
    require(rendering["training_receipt"]["sha256"] == index["training_receipt"]["sha256"], "Render/SR receipt differs")
    metrics = verify_metrics(ctx, run / "metrics", run / "renders_hr", scene, manifest, test, records, expected_env, "CEM_area4_" + arm)
    require(index["metrics_status"] == "shared_A100_8bit_PNG_complete" and index["metrics"]["values"] == metrics["values"], "SR indexed metric values differ")
    ctx.check(run / "metrics/summary.json", index["metrics"]["summary"])
    ctx.check(run / "metrics/per_frame.csv", index["metrics"]["per_frame"])
    provenance = [ctx.check(path), ctx.check(ctx.local(index["training_receipt"]["path"]), index["training_receipt"]), render_id]
    for name in ("parent_stage2_receipt", "schedule", "teacher_index", "job"):
        if index.get(name):
            provenance.append(ctx.check(index[name]["path"], index[name]))
    return dict(system="CEM", scene=scene, arm=arm, run=str(run), kind="fixed_topology_SR_continuation", status="verified",
        manifest=ctx.check(manifest), metrics=metrics, checkpoint=index["checkpoint"], parent_checkpoint=index["parent_checkpoint"],
        parent_stage2_receipt=index["parent_stage2_receipt"], provenance=provenance, render_receipt=render_id, renders=records,
        frozen_source_identities=index.get("source_identities", []), training_HR=arm == "areaOracle", heldout_training=False,
        controls=state["controls"], rendering_GPU=rendering.get("GPU_name"), native_RAdam_step_counters=counters,
        budget=dict(SR_parameter_updates=6000, absolute_iterations=[50001, 56000], parent_endpoint=50000, fixed_topology=True),
        resources=dict(training_GPU=index["training_GPU"], training_seconds=index["training_seconds"],
            torch_peak_reserved_bytes=index["torch_peak_reserved_bytes"], peak_memory_scope=index["peak_memory_scope"],
            shared_parent=True, cost_scope="incremental SR continuation; LR parent/teacher/initialization excluded"),
        original_training_receipt=state)


def wu(ctx, folder, scene, arm, manifest, test, expected_env, inventory, inventory_id):
    task_rows = [task for task in inventory["tasks"] if task["scene"] == scene and task["arm"] == WU_ARMS[arm]]
    require(len(task_rows) == 1, "Wu inventory task absent/duplicated")
    task = task_rows[0]
    path = folder / "render_receipt.json"
    rendering = read(path)
    require(rendering["schema"] == "Wu_area4_fixed_raw_no_posthoc_render_v1" and rendering["status"] == "completed", "Wrong Wu render receipt")
    require(rendering["task"] == task and rendering["inventory_sha256"] == inventory_id["sha256"], "Wu frozen inventory/task differs")
    require(rendering["frames"] == 300 and rendering["parameter_updates"] == 0, "Incomplete Wu render/extra fitting")
    require(rendering["points_loaded"] == rendering["points_rendered"] == task["expected_points_loaded"], "Wu culling/point identity differs")
    require(task["manifest"]["sha256"] == ctx.check(manifest)["sha256"], "Wu manifest differs")
    policy = rendering["inference_policy"]
    require(policy == inventory["inference_policy"], "Wu inference policy differs")
    require(all(policy[k] is False for k in ("posthoc_cull", "color_affine", "spatial_warp", "test_color_fit")) and policy["parameter_updates"] == 0, "Wu extra correction/update forbidden")
    original, completion = ctx.metadata(task["original_run"]), ctx.metadata(task["completion"])
    require(completion["status"] == "completed" and completion["cursor"] == 6000 and completion["checkpoint"]["sha256"] == task["checkpoint"]["sha256"], "Wu final training identity differs")
    require(completion["args"]["cull"] is False and completion["culls"] == [] and original["affine_inherited"], "Wu historical setup differs")
    require(original["parent_checkpoint"] == completion["parent_checkpoint"], "Wu parent differs")
    require([row["frame"] for row in rendering["PNGs"]] == list(range(300)), "Incomplete Wu300PNGs")
    records = []
    for row in rendering["PNGs"]:
        require(row["path"] == f"renders_hr/{row['frame']:04d}.png" and row["gt_sha256"] == test[row["frame"]]["hr_sha256"], "Wrong Wu PNG/GT identity")
        records.append(dict(frame=row["frame"], **ctx.check(folder / row["path"], row, png=True)))
    require({p.name for p in (folder / "renders_hr").glob("*.png")} == {f"{i:04d}.png" for i in range(300)}, "Extra/missing Wu PNGs")
    metrics = verify_metrics(ctx, folder / "common_A100_metrics", folder / "renders_hr", scene, manifest, test, records, expected_env, "wu_area4_" + WU_ARMS[arm])
    parent_run = ctx.local(original["parent_checkpoint"]["path"]).parent
    parent_path = parent_run / "complete.json"
    parent = read(parent_path)
    require(parent["status"] == "completed" and parent["checkpoint"]["sha256"] == original["parent_checkpoint"]["sha256"], "Wu LR parent incomplete")
    require(parent["state"]["coarse_iteration"] == 3000 and parent["state"]["fine_iteration"] == 14000, "Wu LR parent budget differs")
    coarse_path = ctx.local(parent["fine0"]["path"]).parent / "complete.json"
    coarse = read(coarse_path)
    require(coarse["status"] == "completed", "Wu LR coarse incomplete")
    peak_gb = finite(completion["peak_GB"], "Wu peakGB")
    finite(completion["seconds"], "Wu training seconds")
    require(peak_gb >= 0 and completion["seconds"] >= 0, "Invalid Wu resources")
    return dict(system="Wu", scene=scene, arm=arm, historical_arm=WU_ARMS[arm], run=str(folder), kind="historically_selected_development_reference",
        status="verified", metrics=metrics, manifest=ctx.check(manifest), checkpoint=task["checkpoint"], parent_checkpoint=original["parent_checkpoint"],
        provenance=[inventory_id, ctx.check(path), task["original_run"], task["completion"], ctx.check(parent_path), ctx.check(coarse_path)],
        historical_selection=task["historical_selection"], historical_metrics_excluded=task["historical_metric_protocol"],
        heldout_training=False, training_HR=arm == "areaOracle", rendering_GPU=rendering["gpu"], render_receipt=ctx.check(path), renders=records,
        budget=dict(parent_coarse_iteration=3000, parent_fine_iteration=14000, SR_cursor=6000, SR_schedule_clock=[1, 6000],
            affine_inherited=True, fixed_topology=False, topology_counts=completion["counts"], native_RAdam_step_counters=None),
        resources=dict(training_GPU=original["gpu"], training_host=original["host"], training_seconds=completion["seconds"],
            original_peak_GB=peak_gb, peak_allocated_bytes=round(peak_gb * 1e9), peak_allocated_GiB=peak_gb * 1e9 / (1024 ** 3),
            peak_memory_scope="original peak_GB=CUDA max_memory_allocated()/1e9; allocator only", shared_parent=True,
            cost_scope="historical incremental SR; LR prefix listed separately", parent_prefix_stages=[
                dict(stage="coarse", receipt=ctx.check(coarse_path), GPU=coarse.get("gpu"), seconds=coarse.get("seconds"), peak_GB=coarse.get("peak_GB")),
                dict(stage="fine", receipt=ctx.check(parent_path), GPU=parent.get("gpu"), seconds=parent.get("seconds"), peak_GB=parent.get("peak_GB"))]),
        original_training_completion=completion, original_training_run=original)


def deltas(endpoints):
    lookup = {(x["system"], x["scene"], x["arm"]): x for x in endpoints}
    result = []
    pairs = [("areaT-naive", "areaT", "naive"), ("areaTm-areaT", "areaTm", "areaT"),
             ("w4-areaTm", "areaTm_w4", "areaTm"), ("Oracle-areaTm", "areaOracle", "areaTm")]
    for system in ("CEM", "Wu"):
        for label, target, source in pairs + ([("nativeHR-parent", "HRref", "native_parent")] if system == "CEM" else []):
            values = []
            for scene in SCENES:
                a, b = lookup.get((system, scene, target)), lookup.get((system, scene, source))
                delta = {k: a["metrics"]["values"][k] - b["metrics"]["values"][k] for k in METRICS} if a and b else None
                result.append(dict(system=system, scene=scene, contrast=label, target=target, source=source,
                    status="paired_available" if delta else "missing_pair", delta_target_minus_source=delta))
                if delta:
                    values.append(delta)
            result.append(dict(system=system, scene="two_scene_equal_weight_mean", contrast=label, target=target, source=source,
                status="both_scenes_available" if len(values) == 2 else "missing_scene_no_mean",
                delta_target_minus_source={k: math.fsum(x[k] for x in values) / 2 for k in METRICS} if len(values) == 2 else None))
    return result


def aggregate(endpoints):
    means = []
    for system, arms in [("CEM", ("native_parent", *ARMS, "HRref")), ("Wu", ARMS)]:
        for arm in arms:
            selected = [x for x in endpoints if x["system"] == system and x["arm"] == arm]
            full = {x["scene"] for x in selected} == set(SCENES) and len(selected) == 2
            means.append(dict(system=system, arm=arm, scenes=[x["scene"] for x in selected],
                status="both_scenes_available" if full else "missing_scene_no_mean",
                values={k: math.fsum(x["metrics"]["values"][k] for x in selected) / 2 for k in METRICS} if full else None))
    return means


def mappings(values, label):
    result = {}
    for text in values:
        scene, path = text.split("=", 1)
        require(scene in SCENES and scene not in result, "Unknown/duplicated " + label + " scene")
        result[scene] = Path(path).resolve()
    return result


def collect(args, ctx):
    inventory_path = args.wu_inventory.resolve()
    inventory_id, inventory = ctx.check(inventory_path), read(inventory_path)
    require(inventory["schema"] == "Wu_area4_fixed_no_posthoc_reference_v1", "Wrong Wu inventory")
    expected = inventory["expected_metric_environment"]
    require(set(expected) == set(ENV_KEYS), "Incomplete frozen common evaluation environment")
    require(expected["evaluator_sha256"] == ctx.check(ctx.root / "deployment/dynamic_lr_baselines_20261020/evaluate_images.py")["sha256"], "Evaluator changed from frozen environment")
    require(expected["helper_source_sha256"] == ctx.check(ctx.root / "experiments/dynamic_sr_20260918/evaluate.py")["sha256"], "Legacy metric helper changed")
    hr_runs, hr_metrics = mappings(args.hrref_run, "HRref run"), mappings(args.hrref_metrics, "HRref metric")
    endpoints, missing, manifests = [], [], []
    for scene in args.scene:
        manifest, data, test = manifest_for(ctx, scene)
        manifests.append(dict(**ctx.check(manifest), scene=scene, train_cameras=data["splits"]["train"],
                              training_camera_count=len(data["splits"]["train"])))
        for row in test:
            ctx.check(manifest.parent / row["hr_path"], dict(sha256=row["hr_sha256"]))
        specs = [("CEM", "native_parent", args.runs_root / (scene + "_native_parent_v1"))]
        specs += [("CEM", arm, args.runs_root / (scene + "_" + arm + "_v1")) for arm in ARMS]
        specs += [("CEM", "HRref", hr_runs.get(scene, args.runs_root / (scene + "_native_HRref_v2")))]
        label = "Cook07" if scene == "cook_spinach" else "Cut07"
        specs += [("Wu", arm, args.wu_root / label / WU_ARMS[arm]) for arm in ARMS]
        for system, arm, run in specs:
            try:
                if system == "Wu":
                    value = wu(ctx, run, scene, arm, manifest, test, expected, inventory, inventory_id)
                elif arm in ARMS:
                    value = sr(ctx, run, scene, arm, manifest, test, expected)
                else:
                    metric_dir = run / "metrics" if arm == "native_parent" else hr_metrics.get(scene, ctx.batch / "shared_A100_metrics/native_HRref" / scene)
                    value = native(ctx, run, scene, arm, manifest, test, expected, metric_dir)
                endpoints.append(value)
            except (OSError, ValueError, KeyError, TypeError, AssertionError) as error:
                missing.append(dict(system=system, scene=scene, arm=arm, run=str(run.resolve()), reason=f"{type(error).__name__}: {error}"))
    # Cross-arm parent binding is a separate gate; accepted individual receipts
    # cannot silently be combined with a different scene parent checkpoint.
    rejected = []
    lookup = {(e["system"], e["scene"], e["arm"]): e for e in endpoints}
    for e in endpoints:
        if e["system"] != "CEM":
            continue
        parent = lookup.get(("CEM", e["scene"], "native_parent"))
        reason = None
        if e["arm"] in ARMS and parent:
            if e["parent_checkpoint"]["sha256"] != parent["checkpoint"]["sha256"] or e["parent_stage2_receipt"]["sha256"] != parent["provenance"][2]["sha256"]:
                reason = "SR references a different parent endpoint"
        if e["arm"] == "HRref" and parent:
            if e["initialization_LR_pointcloud_sha256"] != parent["initialization_LR_pointcloud_sha256"]:
                reason = "native HRref and LR parent use different LR initial pointcloud"
        if reason:
            rejected.append(e)
            missing.append(dict(system=e["system"], scene=e["scene"], arm=e["arm"], run=e["run"], reason=reason))
    endpoints = [e for e in endpoints if e not in rejected]
    complete = len(endpoints) == 28 and not missing and set(args.scene) == set(SCENES)
    require(complete or args.allow_incomplete, "Matrix incomplete; no output written. --allow-incomplete is research-only. " + json.dumps(missing, ensure_ascii=False))
    return dict(schema="CEM_Wu_area4_final_receipt_comparison_v1", status="complete" if complete else "partial_research_record",
        created_utc=datetime.now(timezone.utc).isoformat(), expected_endpoints=28, verified_endpoints=len(endpoints), complete=complete,
        selected_scenes=args.scene, missing_or_invalid=missing, source=identity(__file__), wu_inventory=inventory_id,
        manifest_identities=manifests, common_metric_environment=expected, endpoints=endpoints, two_scene_means=aggregate(endpoints),
        paired_deltas=deltas(endpoints), resource_rows=[dict(system=e["system"], scene=e["scene"], arm=e["arm"], budget=e["budget"],
            resources=e["resources"], native_RAdam_step_counters=e.get("native_RAdam_step_counters")) for e in endpoints],
        arm_descriptions=ARM_DESCRIPTIONS, interpretation_and_boundaries=NOTES, acceptance=dict(input_result_matrix_complete=complete, human_visual_acceptance=False,
            temporal_or_geometry_acceptance=False, pure_backbone_causal_attribution=False),
        verification_scope=dict(checkpoint_bytes_opened=False, logs_parsed=False, GPU_used=False, SSH_used=False,
            SHA_verified_files=len(ctx.files), source_of_numeric_results="accepted result indexes and common metric summary/300-rowCSV"))


def markdown(result):
    lines = ["# CEM / Wu area4固定端点比较", "", f"状态：**{result['status']}**；已核验 {result['verified_endpoints']}/28 项。", "",
        "以下JSON原值保留完整浮点精度；表格仅作展示。PSNR↑、SSIM↑、LPIPS↓。", "",
        "| 场景 | 系统 | 端点 | PSNR | SSIM | LPIPS |", "|---|---|---|---:|---:|---:|"]
    for endpoint in result["endpoints"]:
        v = endpoint["metrics"]["values"]
        lines.append(f"| {endpoint['scene']} | {endpoint['system']} | {endpoint['arm']} | {v['psnr']:.4f} | {v['ssim']:.6f} | {v['lpips']:.6f} |")
    lines += ["", "## 两景等权均值", "", "缺一景不计算两景均值。", "", "| 系统 | 端点 | PSNR | SSIM | LPIPS |", "|---|---|---:|---:|---:|"]
    for item in result["two_scene_means"]:
        v = item["values"]
        values = f"{v['psnr']:.4f} | {v['ssim']:.6f} | {v['lpips']:.6f}" if v else "缺项 | 缺项 | 缺项"
        lines.append(f"| {item['system']} | {item['arm']} | {values} |")
    lines += ["", "## 成对增量（目标−来源）", "", "PSNR/SSIM正值表示改善；LPIPS负值表示改善。未完成的配对不计算。", "",
        "| 场景 | 系统 | 对照 | ΔPSNR | ΔSSIM | ΔLPIPS |", "|---|---|---|---:|---:|---:|"]
    for item in result["paired_deltas"]:
        v = item["delta_target_minus_source"]
        values = f"{v['psnr']:+.4f} | {v['ssim']:+.6f} | {v['lpips']:+.6f}" if v else "缺项 | 缺项 | 缺项"
        lines.append(f"| {item['scene']} | {item['system']} | {item['contrast']} | {values} |")
    lines += ["", "## 对照臂与问题", ""]
    lines += [f"- **{arm}**：{description}" for arm, description in result["arm_descriptions"].items()]
    lines += ["", "链条增量是同一系统内的配对观察。若teacher匹配或w4在两景、PSNR/SSIM/LPIPS与固定区域上共同改善，才支持其在当前系统内有效。Oracle改善只能说明当前teacher与真实训练HR仍有差距；HRref表现也受独立完整训练和拓扑成长影响。跨CEM/Wu的绝对分数只比较完整系统，不能单凭它识别骨干因果作用。"]
    lines += ["", "## 训练资源与预算", "", "原生端点编号不等于参数更新次数；SR计数为实际更新。秒数不含教师生成、初始化、传输与失败尝试。", "",
        "| 场景 | 系统/端点 | 训练GPU | 训练秒数 | allocator峰值GiB | 显存口径 | 预算 |", "|---|---|---|---:|---:|---|---|"]
    for e in result["endpoints"]:
        r, b = e["resources"], e["budget"]
        gpu = ", ".join(dict.fromkeys(r["training_GPU"])) if isinstance(r["training_GPU"], list) else r["training_GPU"]
        peak = r.get("torch_peak_reserved_bytes", r.get("peak_allocated_bytes"))
        peak_text = f"{peak / (1024 ** 3):.3f}" if peak is not None else "未记录"
        budget = "coarse3000+fine14000后，SR编号1–6000/再增密" if e["system"] == "Wu" else ("LR父50k后实际6000步/固定拓扑" if e["arm"] in ARMS else "原生40k→总50k端点")
        memory_kind = "allocated（原peak_GB为十进制GB）" if e["system"] == "Wu" else "reserved"
        lines.append(f"| {e['scene']} | {e['system']}/{e['arm']} | {gpu} | {r['training_seconds']:.3f} | {peak_text} | {memory_kind} | {budget} |")
    lines += ["", "## CEM SR原生11组RAdam计数", "", "每格为初始→最终（实际增加）；有梯度参与次数保留在JSON，不强行认定每组6000。", "",
        "| 场景/臂 | 参数组 | 初始 | 最终 | 实际增加 | 有梯度步数 |", "|---|---|---:|---:|---:|---:|"]
    for e in result["endpoints"]:
        for name, c in (e.get("native_RAdam_step_counters") or {}).items():
            lines.append(f"| {e['scene']}/{e['arm']} | {name} | {c['initial'][0]} | {c['final'][0]} | {c['updates'][0]} | {c['steps_with_grad']} |")
    lines += ["", "## 解释与适用边界", ""] + [f"- {note}" for note in result["interpretation_and_boundaries"]]
    if result["missing_or_invalid"]:
        lines += ["", "## 缺项或身份核验失败", ""] + [f"- {item['scene']} / {item['system']} / {item['arm']}: {item['reason']}" for item in result["missing_or_invalid"]]
    lines += ["", "所有原始数值、索引/收据/summary/CSV/PNG的本地绝对路径与SHA、训练资源及原始收据见同目录comparison.json。未执行人工、时序或几何验收。", ""]
    return "\n".join(lines)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene", nargs="+", choices=SCENES, default=list(SCENES))
    p.add_argument("--runs-root", type=Path, default=BATCH / "runs")
    p.add_argument("--wu-root", type=Path, default=BATCH / "wu_reference")
    p.add_argument("--wu-inventory", type=Path, default=BATCH / "wu_reference_inventory.json")
    p.add_argument("--hrref-run", action="append", default=[], metavar="SCENE=RUN")
    p.add_argument("--hrref-metrics", action="append", default=[], metavar="SCENE=DIR")
    p.add_argument("--allow-incomplete", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    return p


def main():
    args = parser().parse_args()
    require(len(args.scene) == len(set(args.scene)), "Duplicate scene")
    require(not args.output.exists(), "Preserve existing output; choose a new directory")
    result = collect(args, Context())
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "comparison.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    (args.output / "results.md").write_text(markdown(result))
    if result["complete"]:
        gate = dict(schema="CEM_Wu_area4_complete_matrix_gate_v1", status="complete", endpoints=28,
                    comparison=identity(args.output / "comparison.json"), report=identity(args.output / "results.md"),
                    scope="receipt/PNG/common-A100-metric completeness only; not human/scientific acceptance")
        (args.output / "complete_matrix_gate.json").write_text(json.dumps(gate, indent=2) + "\n")
    print(json.dumps(dict(status=result["status"], verified_endpoints=result["verified_endpoints"], output=str(args.output.resolve()),
        source_sha256=sha(__file__), torch_imported="torch" in sys.modules)))


if __name__ == "__main__":
    main()
