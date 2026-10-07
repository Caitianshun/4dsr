"""Evidence-limited cause matrix; missing/contrary evidence stays explicit."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from depth_prior import sha256, write_json

STATUSES = ["已确认实现问题", "有干预支持的贡献因素", "相关现象", "证据不足"]


def read(path):
    return json.loads(Path(path).read_text()) if path is not None and Path(path).exists() else None


def mean(values):
    good = [value for value in values if value is not None and np.isfinite(value)]
    return float(np.mean(good)) if good else None


def oracle_differences(cached, variant):
    rows = []
    if cached is None:
        return rows
    for oracle in cached["oracles"]:
        if oracle["model"] != "U6000":
            continue
        evaluations = oracle["evaluations"]
        baseline = {(e["frame"], e["split"]): e for e in evaluations if e["variant"] == "original"}
        selected = [e for e in evaluations if e["variant"] == variant and e["split"] == "validation"]
        for e in selected:
            original = baseline[e["frame"], e["split"]]["fixed_region"]["mse"]
            changed = e["fixed_region"]["mse"]
            rows.append(dict(camera=oracle["camera"], frame=e["frame"], baseline_mse=original,
                changed_mse=changed, delta_mse=changed - original if original is not None and changed is not None else None,
                coefficients=oracle["gain_bias"] if variant == "gain_bias" else oracle["translation"]))
    return rows


def summarize_probes(reports):
    summary = []
    for report in reports:
        rows = report["rows"]
        train = [r for r in rows if r["split"] == "train"]
        cam01 = [r for r in rows if r["camera"] == "cam01"]
        summary.append(dict(probe=report["label"], step=report["probe_step"],
            checkpoint_sha256=report["checkpoint_sha256"],
            train_lr_mse=mean([r["lr_mse"] for r in train]),
            train_lr_l1=mean([r["lr_l1"] for r in train]),
            train_surface_l1=mean([r["surface_lr_l1"] for r in train]),
            cam01_hr_low_mse=mean([r["hr_low_mse"] for r in cam01]),
            cam01_hr_mse=mean([r["hr_total_mse"] for r in cam01]),
            offset_saturation=report["children_offset"]))
    return summary


def build(cached, rendered, probe_reports, files, verified_bug=None):
    probes = summarize_probes(probe_reports)
    table = {(r["probe"], r["step"]): r for r in probes}
    entries = []

    def add(cause, status, observation, counterevidence, unknown, values=None):
        entries.append(dict(cause=cause, status=status, observation=observation,
            counterevidence=counterevidence, remaining_unknown=unknown,
            evidence_files="; ".join(files), values_json=json.dumps(values, ensure_ascii=False) if values is not None else ""))

    calibration = cached.get("calibration") if cached is not None else None
    if verified_bug is not None and verified_bug.get("status") == "confirmed_reproduced_bug":
        add("数据/相机实现错误", STATUSES[0], verified_bug["observation"],
            "协议修复必须同时覆盖所有受影响比较分支。", verified_bug.get("remaining_unknown", ""), verified_bug)
    else:
        calibration_values = dict(
            maximum_intrinsics_half_pixel_error=max(r["intrinsics_half_pixel_maxabs"] for r in calibration["rows"]),
            maximum_pose_inverse_error=max(r["pose_inverse_maxabs"] for r in calibration["rows"]),
            duplicate_keys=calibration["duplicate_camera_frame_keys"], time_map_error=calibration["time_frame_over300_maxabs"]) if calibration else None
        add("数据/相机实现错误", STATUSES[3], "已核对内部相机/时间映射、HR→LR半像素内参缩放与位姿逆矩阵；没有独立复现并修复的bug证据。" if calibration else "数据/相机核验未完成。",
            "内部一致与稀疏重投影较小不证明真实标定完全正确。",
            "无法从当前观测排除真实相机小偏差、低纹理区匹配缺失或投影约定的共有偏差。", calibration_values)

    if rendered is not None:
        values = dict(sources=rendered["sources"], actual_support_histogram=rendered["actual_support_histogram"])
        add("观测覆盖不足", STATUSES[2], "从cam01墙角实际alpha贡献点出发，已统计19个训练视图的真实渲染贡献、角度、HR渲染对应LR足迹及同位置LR拟合。",
            "这些覆盖/遮挡仍由U6000位置与渲染器推断；错误父几何可能制造虚假遮挡。",
            "墙角低纹理区域缺少密集独立LR几何证据，不能据模型贡献数断言真实不可见。", values)
    else:
        add("观测覆盖不足", STATUSES[3], "尚未完成同一实际贡献位置到训练视图的核验。",
            "最近相机和重复二维矩形不构成同一表面支持。", "真实覆盖与模型假设覆盖仍未分离。")

    dc = rendered.get("dc_only", []) if rendered else []
    validation_dc = [r for r in dc if r["camera"] == "cam01" and r["frame"] in [80, 118]]
    training_dc = [r for r in dc if r["split"] == "train"]
    dc_values = dict(cam01_validation_delta_lr_mse=mean([r["delta_lr_mse"] for r in validation_dc]),
                     train_delta_lr_mse=mean([r["delta_lr_mse"] for r in training_dc]))
    supported_dc = dc_values["cam01_validation_delta_lr_mse"] is not None and dc_values["cam01_validation_delta_lr_mse"] < 0
    add("视角相关外观/SH", STATUSES[1] if supported_dc else STATUSES[2] if dc else STATUSES[3],
        "固定几何关闭SH非DC后，独立登记验证帧的cam01 LR误差下降，支持视角外观参与残差。" if supported_dc else "已执行固定几何DC-only诊断；当前结果未显示稳定的cam01验证帧改善。" if dc else "DC-only诊断未完成。",
        "关闭方向颜色会改变整个图像，可能损害训练拟合；不能由此证明几何正确。",
        "仍需结合同一表面训练误差、P_SH干预和实际颜色漂移；单父状态未验证跨场景。", dc_values)

    gain = oracle_differences(cached, "gain_bias")
    add("曝光或宽尺度颜色偏差", STATUSES[2] if gain else STATUSES[3],
        "以开发LR第0/40帧拟合每相机RGB gain+bias，在80/118帧独立计算剩余误差。" if gain else "分帧gain+bias oracle未完成或静态支持不足。",
        "低维颜色校正也可能补偿几何错配，且使用目标开发LR，不属于合法新视角成绩。",
        "不能确认真实曝光错误；校正参数禁止写回训练或主表。", gain)

    shift = oracle_differences(cached, "translation")
    add("配准/相机小偏差", STATUSES[2] if shift else STATUSES[3],
        "与颜色校正分别拟合固定±2 LR像素平移，仅使用第0/40帧；80/118帧验证。" if shift else "小范围平移oracle未完成。",
        "对错误几何平移图像同样可能改善；二维位移并不等于真实刚性相机扰动。",
        "没有独立LR轨迹证实相机误差；当前没有把oracle收益称作标定修复。", shift)

    parent = table.get(("U6000", 0))
    xyz = table.get(("P_xyz", 500))
    geometry_supported = (parent is not None and xyz is not None and
        xyz["train_lr_mse"] < parent["train_lr_mse"] and xyz["cam01_hr_low_mse"] < parent["cam01_hr_low_mse"])
    add("几何/遮挡或大footprint", STATUSES[1] if geometry_supported else STATUSES[2] if rendered else STATUSES[3],
        "位置限定LR-only干预同时减少训练LR和cam01低频误差，支持可调整位置是贡献因素。" if geometry_supported else "已记录归一矩、多层原始方差、真实贡献及投影足迹；尚无充分干预支持将其认定为主因。",
        "厚射线或大高斯本身不是错误；均值深度不等于唯一表面，位置探针未更新所有几何能力。",
        "需要P_xyz/P_SH/P_joint与C1 500步完整对照、children饱和/Adam位移；不能仅靠深度先验自洽证明几何真值。",
        dict(parent=parent, xyz=xyz, target=rendered["target"] if rendered else None))

    frozen = rendered.get("background_frozen", []) if rendered else []
    frozen_validation = [r for r in frozen if r["camera"] == "cam01" and r["frame"] in [80, 118]]
    frozen_delta = mean([r["frozen_frequency"]["low_mse"] - r["baseline_frequency"]["low_mse"] for r in frozen_validation])
    add("动态表示污染静态背景", STATUSES[1] if frozen_delta is not None and frozen_delta < 0 else STATUSES[2] if frozen else STATUSES[3],
        "训练LR跨时稳定支持点固定到参考形变输出后，cam01验证帧低频误差下降。" if frozen_delta is not None and frozen_delta < 0 else "已执行训练LR稳定支持点的形变固定诊断；验证帧未显示同向低频改善。" if frozen else "未取得足够训练LR稳定支持点或未完成固定诊断。",
        "LR颜色稳定是代理支持，恒定纹理的运动面仍可能被误纳入；不能把人体/显隐区域全面冻结。",
        "真实背景是否静态、共享形变对细节和遮挡的因果作用仍需独立证据。",
        dict(cam01_validation_delta_low_mse=frozen_delta, background_motion=rendered.get("background_motion") if rendered else None))

    joint = table.get(("P_joint", 500))
    c1 = table.get(("C1", 500))
    conflict_supported = joint is not None and c1 is not None and joint["train_lr_mse"] < c1["train_lr_mse"]
    add("SR监督妨碍观测优化", STATUSES[1] if conflict_supported else STATUSES[3],
        "同父状态、500更新LR-only joint的真实训练LR误差低于C1，支持辅助SR在该轨迹中妨碍观测拟合。" if conflict_supported else "P_joint与C1第500步同观察对照尚不完整，或未呈现这一模式。",
        "即使训练拟合改善，新视角残差仍可能有独立几何/覆盖瓶颈；不能仅凭负梯度cosine推出有害冲突。",
        "须核对相同前500观察、继承Adam状态、固定冻结状态不推进与活动实际位移记录。", dict(joint=joint, C1=c1))

    add("输出采样问题", STATUSES[2], "历史原生HR与2HR area诊断已记录不同足迹采样的质量变化，当前未新增超采样训练。",
        "既有2HR三指标整体变差；采样变化不能作为默认修复，也未单独解释墙角。",
        "仅属历史相关现象；需要读取原完整三指标和同位置图，不能从本原因矩阵推断改善。")
    return entries, probes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cached", type=Path)
    parser.add_argument("--render", type=Path)
    parser.add_argument("--probes", type=Path, nargs="*", default=[])
    parser.add_argument("--verified-bug", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    files = [str(p) for p in [args.cached, args.render, *args.probes, args.verified_bug] if p is not None and p.exists()]
    reports = [read(path) for path in args.probes if path.exists()]
    rows, probes = build(read(args.cached), read(args.render), reports, files, read(args.verified_bug))
    with (args.out / "cause_matrix.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(args.out / "cause_matrix.json", dict(status="completed_with_explicit_unknowns", rows=rows,
        statuses=STATUSES, probe_summary=probes, sources=[dict(path=p, sha256=sha256(p)) for p in files],
        source_sha256=sha256(__file__), limitation="Finite descriptive/intervention evidence, no single-cause certainty or significance claim; oracles never enter legal main metrics."))
    print(json.dumps(dict(status="completed_with_explicit_unknowns", rows=len(rows),
        counts={status: sum(r["status"] == status for r in rows) for status in STATUSES}), ensure_ascii=False))


if __name__ == "__main__":
    main()
