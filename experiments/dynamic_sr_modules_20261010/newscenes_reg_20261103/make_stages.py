"""Generate pipeline.py task rows and a token-by-token Flame07 comparison.

No executor, GPU, lane, remote job or training is started. script is relative to
pipeline.py's directory (its real interface); argv and done are project-relative.
Owner-supplied schedule/selection/assignment receipts must exist before execution.
"""
from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path
import shlex
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_new import *
import register_new

E = "experiments/dynamic_sr_modules_20261010"
T = E + "/third_20261022"
G = E + "/geoprobe_20261017"
OURS = E + "/area4_baselines_20261024/ours"
THIRD = "output/dynamic_sr_modules_20261010/third_20261022"
OLD_LR = "output/dynamic_sr_cull_retrain_20261009/wave1/lr/physlr_20261016"
OLD_REG = "output/dynamic_sr_physlr_20261016/registry/flame_steak_area4/seed_20261007/areaft_teacher/Bsync"
OLD_A_ROOT = "output/dynamic_sr_physlr_20261016/holdout_w15"
OLD_A = OLD_A_ROOT + "/sr/Flame07_area4_ho_w4/checkpoint_06000.pt"
OLD_B = THIRD + "/mk_ctl6000/sr/Flame07_area4_ctl6000_w4/checkpoint_06000.pt"
OLD_C = THIRD + "/mk_ctl3g9/sr/Flame07_area4_ctl3g9_w4/checkpoint_06000.pt"
OLD_STUDENT = THIRD + "/mk_pvcds4k3g9/sr/Flame07_area4_pvcds4k3g9_w4/checkpoint_06000.pt"
OLD_PARENT = OLD_LR + "/runs/Flame07_area4_gateaff/checkpoint_fine_14000.pt"
OLD_MANIFEST = f"data/dynamic_sr/full_time_20261007/prepared/flame_steak_area4/manifest_train_ready_seed{SEED}.json"
OLD_PC = THIRD + "/fit/Flame07_pvcds4k3g9_PCv2_lr"
OLD_EVAL_PC = THIRD + "/eval/Flame07_pvcds4k3g9_PCv2_lr"
PIPELINE_HERE = ROOT / "experiments/dynamic_sr_cull_retrain_20261009"


def lane_line(filename, script_basename, contains=""):
    path = ROOT / T / "jobs" / filename
    matches = []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        tokens = shlex.split(line)
        if tokens and tokens[0] == "CRITICAL":
            tokens = tokens[1:]
        if tokens and Path(tokens[0]).name == script_basename and contains in line:
            matches.append((tokens, number))
    require(len(matches) == 1, "Historical argv ambiguous: " + filename + "/" + script_basename)
    tokens, number = matches[0]
    return dict(script=tokens[0], argv=tokens[1:],
                evidence=dict(file=entry(path), line=number, type="literal_historical_lane"))


def historical():
    jobs = ROOT / T / "jobs"
    c = lane_line("dsr-tm-3090-k3_laneB.txt", "modkit4.py", "ctl3g9")
    student = lane_line("dsr-tm-3090-k3_laneB.txt", "pvcd_modkit.py")
    pc = lane_line("dsr-tm-3090-k3_laneB.txt", "run_fit_eval2.py",
                   "--out " + OLD_PC + " ")
    loco_fit = lane_line("f3t9_loco_Flame07.txt", "perframe_fit_loco.py")
    loco_smx = lane_line("f3t9_loco_Flame07.txt", "resid_temporal_variants2.py", "--fit " + THIRD + "/fit/")
    loco_eval = lane_line("f3t9_loco_Flame07.txt", "loco_eval.py")
    smx = lane_line("f3t9_loco_Flame07.txt", "resid_temporal_variants2.py",
                    "--fit output/dynamic_sr_modules_20261010/perframe_20261021/")
    source_generator = ROOT / T / "run_alphap_p6.py"
    a_run = ROOT / OLD_A
    b_run = ROOT / OLD_B
    out = dict(
        teacher=dict(script="experiments/dynamic_sr_physlr_20261016/teacher_physlr3.py",
                     argv=["--manifest", OLD_MANIFEST, "--batch", "6", "--weights",
                           "output/dynamic_sr_physlr_20261016/teacher_ft/swinir_area_ft_v1"],
                     evidence=dict(file=entry(PREP / f"flame_steak_area4/teacher_index_sr_swinir_x4_areaft_seed{SEED}.json"),
                                   type="producer_index_plus_authorized_recipe")),
        source_A=dict(script=G + "/run_modkit7_remote.py",
                      argv=["--parent", OLD_PARENT, "--baseline-sr", OLD_REG, "--lambda-sr", "0.2",
                            "--out-root", OLD_A_ROOT, "--tag", "Flame07_area4_ho",
                            "--view", "cam00", "--arm", "w4=wfield:4"],
                      evidence=dict(file=entry(a_run.parent / "run.json"),
                                    type="effective_run_args_plus_authorized_driver_mapping")),
        source_B=dict(script=G + "/run_modkit4_remote.py",
                      argv=["--parent", OLD_PARENT, "--baseline-sr", OLD_REG, "--lambda-sr", "0.2",
                            "--out-root", THIRD + "/mk_ctl6000", "--tag", "Flame07_area4_ctl6000",
                            "--view", "cam00", "--arm", "w4=wfield:4"],
                      evidence=dict(file=entry(source_generator), run=entry(b_run.parent / "run.json"),
                                    type="historical_generator_expanded_argv")),
        source_C=c, student=student, loco_fit=loco_fit, loco_smx=loco_smx,
        loco_eval=loco_eval, pc_fit_eval=pc, smx=smx)
    main_index = read(ROOT / "output/dynamic_lr_area4_20261024/ours/Flame07_area4_ours/renders_test/render_index.json")
    require(main_index["checkpoint"]["path"] == OLD_STUDENT and main_index["residuals"] == OLD_PC,
            "Historical main table does not use the expected actual final model")
    # Explicit constructor expansions for final rendering/scoring; their evidence
    # is distinguished from the literal source/student/LOCO lane text.
    render_root = "output/dynamic_lr_area4_20261024/ours/Flame07_area4_ours"
    out["render_test"] = dict(
        script=OURS + "/render_ours_png.py",
        argv=["--checkpoint", OLD_STUDENT, "--resid", OLD_PC, "--manifest", OLD_MANIFEST,
              "--scope", "test", "--out", render_root + "/renders_test",
              "--check-float", OLD_EVAL_PC, "--check-name", "PC"],
        evidence=dict(file=entry(ROOT / render_root / "renders_test/render_index.json"),
                      type="render_index_effective_arguments"))
    out["render_train"] = dict(
        script=OURS + "/render_ours_png_cam.py",
        argv=["--checkpoint", OLD_STUDENT, "--resid", OLD_PC, "--manifest", OLD_MANIFEST,
              "--out", "output/dynamic_sr_modules_20261010/ourscam_20261029/Flame07_area4/render_train"],
        evidence=dict(file=entry(ROOT / E / "area4_baselines_20261024/ours/lanes_cam_20261029/lane_C.txt"),
                      type="literal_camera_render_lane"))
    for stage, script, scope, rd, od, method in (
        ("score_test", E + "/area4_baselines_20261024/score_views_area4.py", "test",
         render_root + "/renders_test", render_root + "/scores_test", "ours"),
        ("score_train", OURS + "/score_views_cam.py", "train",
         "output/dynamic_sr_modules_20261010/ourscam_20261029/Flame07_area4/render_train/cam",
         "output/dynamic_sr_modules_20261010/ourscam_20261029/Flame07_area4/score_train_cam", "ours_with_camera_model"),
        ("score_train_identity", E + "/area4_baselines_20261024/score_views_area4.py", "train",
         "output/dynamic_sr_modules_20261010/ourscam_20261029/Flame07_area4/render_train/identity",
         "output/dynamic_sr_modules_20261010/ourscam_20261029/Flame07_area4/score_train_identity", "ours_identity_rerun")):
        out[stage] = dict(script=script, argv=[
            "--manifest", OLD_MANIFEST, "--scope", scope, "--render-dir", rd,
            "--out", od, "--method", method, "--device", "cuda", "--execute",
            "--torch-home", "output/dynamic_sr_modules_20261010/bl4_20261024/torch_home"],
            evidence=dict(file=entry(ROOT / "tmp/claude_8a_ourscam_20261029/make_ourscam_lanes.py"),
                          type="historical_scorer_constructor_expansion"))
    return out


def paths(scene):
    short = NEW[scene] + "07"
    base = rel(OUT)
    return dict(
        short=short, parent=f"{base}/lr/runs/{short}_area4_gateaff/checkpoint_fine_14000.pt",
        registry=f"{base}/sr/registry/{scene}_area4/seed_{SEED}/areaft_teacher/Bsync",
        teacher=f"{base}/teacher_cache/{scene}_area4",
        manifest=f"data/dynamic_sr/full_time_20261007/prepared/{scene}_area4/manifest_train_ready_seed{SEED}.json",
        a=f"{base}/sources/A/sr/{short}_area4_ho_w4/checkpoint_06000.pt",
        b=f"{base}/sources/B/sr/{short}_area4_ctl6000_w4/checkpoint_06000.pt",
        c=f"{base}/sources/C/sr/{short}_area4_ctl3g9_w4/checkpoint_06000.pt",
        student=f"{base}/student/sr/{short}_area4_pvcds4k3g9_w4/checkpoint_06000.pt",
        pc=f"{base}/fit/{short}_pvcds4k3g9_PCv2_lr",
        pc_eval=f"{base}/eval/{short}_pvcds4k3g9_PCv2_lr",
        loco_fit=f"{base}/loco/fit/{short}_PCv2_loco10",
        loco_smx=f"{base}/loco/variants/{short}",
        decision=f"{base}/loco/eval/{short}/loco.json",
        tv=f"{base}/tv2/{short}", final=f"{base}/final/{short}_area4",
        selected_eval=f"{base}/eval/{short}_selected_final")


def translate(scene, tokens):
    p = paths(scene)
    base = rel(OUT)
    replacements = {
        OLD_PARENT: p["parent"], OLD_REG: p["registry"], OLD_MANIFEST: p["manifest"],
        OLD_A: p["a"], OLD_B: p["b"], OLD_C: p["c"], OLD_STUDENT: p["student"],
        OLD_PC: p["pc"], OLD_EVAL_PC: p["pc_eval"],
        THIRD + "/fit/Flame07_PCv2_loco10b": p["loco_fit"],
        THIRD + "/tv_loco/Flame07b": p["loco_smx"],
        THIRD + "/eval/Flame07_loco10b": str(Path(p["decision"]).parent),
        "output/dynamic_sr_modules_20261010/perframe_20261021/fit/Flame07_PCv2_lr": p["pc"],
        THIRD + "/tv2/Flame07b": p["tv"],
        "output/dynamic_sr_modules_20261010/mvfb_remote_20261021/parents/Flame07_area4_ho_w4/checkpoint_06000.pt": p["a"],
        THIRD + "/mk_ctl6000": base + "/sources/B",
        THIRD + "/mk_ctl3g9": base + "/sources/C",
        THIRD + "/mk_pvcds4k3g9": base + "/student",
        OLD_A_ROOT: base + "/sources/A",
        "output/dynamic_lr_area4_20261024/ours/Flame07_area4_ours/renders_test": p["final"] + "/renders_test",
        "output/dynamic_lr_area4_20261024/ours/Flame07_area4_ours/scores_test": p["final"] + "/scores_test",
        "output/dynamic_sr_modules_20261010/ourscam_20261029/Flame07_area4/render_train": p["final"] + "/renders_train",
        "output/dynamic_sr_modules_20261010/ourscam_20261029/Flame07_area4/score_train_cam": p["final"] + "/scores_train_cam",
        "output/dynamic_sr_modules_20261010/ourscam_20261029/Flame07_area4/score_train_identity": p["final"] + "/scores_train_identity",
        "output/dynamic_sr_modules_20261010/bl4_20261024/torch_home": base + "/model_cache/torch_home",
    }
    result = []
    for token in tokens:
        for old_value, new_value in sorted(replacements.items(), key=lambda kv: -len(kv[0])):
            if old_value in token:
                token = token.replace(old_value, new_value)
                break
        token = token.replace("Flame07", p["short"])
        result.append(token)
    return result


def task(name, script, argv, deps, done):
    return dict(name=name, script=os.path.relpath(local(script), PIPELINE_HERE), argv=argv,
                deps=list(deps), done=rel(done))


def generate_scene(scene, refs):
    p = paths(scene)
    s = p["short"]
    rows, comparisons = [], []
    own = rel(CODE)
    def add(key, done, deps, *, argv=None, script=None):
        ref = refs[key]
        effective = translate(scene, ref["argv"]) if argv is None else argv
        source = script or ref["script"]
        execution_script, execution_argv = source, effective
        if key in ("source_A", "source_B", "source_C", "student"):
            # External teacher images cannot be consumed through the original
            # dataset-only safe_child. Keep the scientific argv unchanged and
            # bind a narrowly scoped producer/consumer path adapter.
            done = rel(OUT / f"rows/receipts/{s}_{key}_teacher_paths.json")
            execution_script = own + "/teacher_paths_new.py"
            execution_argv = [
                "--index", p["teacher"] + f"/teacher_index_sr_swinir_x4_areaft_seed{SEED}.json",
                "--script", source, "--receipt", done, "--", *effective]
        rows.append(task(f"{s}_{key}", execution_script, execution_argv, deps, done))
        comparisons.append(dict(scene=scene, stage=key, historical=ref, new_script=source, new_argv=effective,
                                execution_script=execution_script, execution_argv=execution_argv))
        return f"{s}_{key}"

    register_name = f"{s}_register_LR"
    rows.append(task(register_name, own + "/register_new.py",
                     ["register", "--scene", scene], [], OUT / f"lr/registry/{scene}_area4/seed_{SEED}/config.json"))
    lr_rows = register_new.tasks([scene])
    lr_rows[0]["deps"] = [register_name]
    rows.extend(lr_rows)
    fine_name = lr_rows[-1]["name"]
    tensor_done = rel(OUT / f"lr/validation/{scene}_parent_tensors.json")
    tensor_name = f"{s}_parent_tensors"
    rows.append(task(tensor_name, own + "/check_parent_torch_new.py",
                     ["--scene", scene, "--parent", p["parent"], "--out", tensor_done],
                     [fine_name], tensor_done))
    teacher_args = translate(scene, refs["teacher"]["argv"]) + ["--out-root", p["teacher"]]
    teacher = add("teacher", p["teacher"] + f"/teacher_index_sr_swinir_x4_areaft_seed{SEED}.json",
                  [register_name], argv=teacher_args, script=own + "/teacher_cache_new.py")
    owner = rel(OUT / f"owner_inputs/{scene}")
    schedule = rel(OUT / f"schedules/{scene}/schedule.json")
    clean = rel(OUT / f"clean_bsync/{scene}/seed{SEED}/Bsync/scientific_plan.json")
    schedule_done = rel(OUT / f"schedules/{scene}/validated.json")
    rows.append(task(f"{s}_schedule", own + "/materialize_bsync_new.py", [
        "require-schedule", "--scene", scene, "--parent", p["parent"], "--schedule", schedule,
        "--assignment-acceptance", owner + "/assignment_acceptance.json",
        "--out", schedule_done], [tensor_name], schedule_done))
    rows.append(task(f"{s}_clean_bsync", own + "/materialize_bsync_new.py", [
        "materialize", "--scene", scene, "--parent", p["parent"],
        "--teacher", p["teacher"] + f"/teacher_index_sr_swinir_x4_areaft_seed{SEED}.json",
        "--schedule", schedule, "--selection", owner + "/selection.json",
        "--assignment-acceptance", owner + "/assignment_acceptance.json", "--out", clean],
        [f"{s}_schedule", teacher], clean))
    rows.append(task(f"{s}_sr_register", own + "/sr_register_new.py", ["--scene", scene],
                     [f"{s}_clean_bsync"], p["registry"] + "/config.json"))
    a = add("source_A", str(Path(p["a"]).parent / "complete.json"), [f"{s}_sr_register"])
    b = add("source_B", str(Path(p["b"]).parent / "complete.json"), [f"{s}_sr_register"])
    c = add("source_C", str(Path(p["c"]).parent / "complete.json"), [f"{s}_sr_register"])
    student = add("student", str(Path(p["student"]).parent / "complete.json"), [a, b, c])
    loco_fit = add("loco_fit", p["loco_fit"] + "/complete.json", [a])
    loco_smx = add("loco_smx", p["loco_smx"] + "/SMX1/complete.json", [loco_fit])
    loco = add("loco_eval", p["decision"], [loco_smx])
    pc = add("pc_fit_eval", p["pc_eval"] + "/summary.json", [student, loco])
    def gate(name, script, args, deps, *, condition=False, inventory=False):
        receipt = p["final"] + f"/receipts/{name}.json"
        own_args = ["--decision", p["decision"], "--v2", p["pc"],
                    "--smx1", p["tv"] + "/SMX1", "--out", receipt]
        if condition:
            own_args += ["--if-smx1"]
        if inventory:
            own_args += ["--checkpoint", p["student"]]
        else:
            own_args += ["--", script, *args]
        rows.append(task(f"{s}_{name}", own + "/stage_gate_new.py", own_args, deps, receipt))
        return f"{s}_{name}"
    smx_args = translate(scene, refs["smx"]["argv"])
    smx = gate("smx", refs["smx"]["script"], smx_args, [pc], condition=True)
    comparisons.append(dict(scene=scene, stage="smx", historical=refs["smx"],
                            new_script=refs["smx"]["script"], new_argv=smx_args,
                            adapter="stage_gate_new.py: execute iff frozen cam10 rule chooses SMX1"))
    final_model = gate("final_model", "", [], [smx], inventory=True)
    selected_args = ["--resid", "PC=@RESID@", "--require-all",
                     "--checkpoint", "parent=" + p["student"], "--checkpoint", "PC=" + p["student"],
                     "--view", "cam00", "--out", p["selected_eval"]]
    selected_eval = gate("selected_float_eval", E + "/perframe_20261021/eval_resid.py",
                         selected_args, [final_model])
    pc_tail = refs["pc_fit_eval"]["argv"]
    comparisons.append(dict(scene=scene, stage="selected_float_eval",
                            historical=dict(script=E + "/perframe_20261021/eval_resid.py",
                                            argv=pc_tail[pc_tail.index("--eval") + 1:],
                                            evidence=refs["pc_fit_eval"]["evidence"]),
                            new_script=E + "/perframe_20261021/eval_resid.py", new_argv=selected_args,
                            adapter="selected real residual, same evaluator arguments"))
    render_test = translate(scene, refs["render_test"]["argv"])
    render_test[render_test.index("--resid") + 1] = "@RESID@"
    render_test[render_test.index("--check-float") + 1] = p["selected_eval"]
    rt = gate("render_test", refs["render_test"]["script"], render_test, [selected_eval])
    comparisons.append(dict(scene=scene, stage="render_test", historical=refs["render_test"],
                            new_script=refs["render_test"]["script"], new_argv=render_test,
                            adapter="stage_gate_new.py substitutes the selected residual path"))
    add("score_test", p["final"] + "/scores_test/area4_selection.json", [rt])
    render_train = translate(scene, refs["render_train"]["argv"])
    render_train[render_train.index("--resid") + 1] = "@RESID@"
    rr = gate("render_train", refs["render_train"]["script"], render_train, [final_model])
    comparisons.append(dict(scene=scene, stage="render_train", historical=refs["render_train"],
                            new_script=refs["render_train"]["script"], new_argv=render_train,
                            adapter="selected residual path; camera-model PNGs required"))
    add("score_train", p["final"] + "/scores_train_cam/camera_model_scoring.json", [rr])
    add("score_train_identity", p["final"] + "/scores_train_identity/area4_selection.json", [rr])
    return rows, comparisons


def comparison_report(comparisons):
    lines = ["# Flame07 与新场景逐参数对照", "",
             "序号从 0 开始；完整 argv、证据类型及原文件 SHA 见 argv_comparison.json。",
             "来源 A/B 部分由历史生成器和 run.json 展开；并非伪称存在完整命令行日志。", "",
             "学生不显式传 --weight/--seed/--dssim，沿用历史默认 0.5/20261014/0.2；"
             "三个来源均继承同一 seed20261007 的 LR 父模型。", "",
             "| 场景 / 阶段 | 位置 | Flame07 | 新场景 | 判定 |", "| --- | ---: | --- | --- | --- |"]
    results = []
    for item in comparisons:
        old_args = item["historical"]["argv"]
        new_args = item["new_argv"]
        expected = translate(item["scene"], old_args)
        p = paths(item["scene"])
        if item["stage"] == "teacher":
            expected += ["--out-root", p["teacher"]]
        if item["stage"] in ("render_test", "render_train"):
            expected[expected.index("--resid") + 1] = "@RESID@"
        if item["stage"] == "render_test":
            expected[expected.index("--check-float") + 1] = p["selected_eval"]
        if item["stage"] == "selected_float_eval":
            expected[expected.index("--resid") + 1] = "PC=@RESID@"
            expected[expected.index("--out") + 1] = p["selected_eval"]
        require(new_args == expected,
                f"Unexpected historical argv change in {item['scene']}/{item['stage']}")
        checks = []
        for i in range(max(len(old_args), len(new_args))):
            a = old_args[i] if i < len(old_args) else ""
            b = new_args[i] if i < len(new_args) else ""
            allowed = b == (expected[i] if i < len(expected) else "")
            require(allowed, f"Non-path historical parameter changed: {item['stage']} token {i}: {a} -> {b}")
            label = "一致" if a == b else ("输出路径适配" if item["stage"] == "teacher" and i >= len(old_args)
                                      else "真实选中残差路径" if "@RESID@" in b else "路径或场景名")
            checks.append(dict(index=i, historical=a, new=b, allowed=allowed, category=label))
            escape = lambda v: str(v).replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {item['scene']} / {item['stage']} | {i} | `{escape(a)}` | `{escape(b)}` | {label} |")
        results.append(dict(**item, tokens=checks, passed=True))
    lines += ["", "教师新增 --out-root 和路径适配入口，以遵守只写新命名空间；其余算法不变。",
              "来源 A/B/C 与学生经 teacher_paths_new.py 调用表中原脚本和完整 argv；"
              "它只放行面积清单对应教师索引中逐文件 SHA 验证过的新缓存路径。A/B 的训练子进程也显式套用此适配。",
              "SMX1 与最终渲染的外层 gate 只读 LR LOCO 决策并选择真实残差目录。",
              "LOCO 保留历史做法：在来源 A 上拟合/判定场景规则，随后应用于学生的 P+C；"
              "没有用学生或 cam00 HR 分数重新挑选规则。", ""]
    return results, "\n".join(lines)


def validate_graph(rows):
    by_name = {r["name"]: r for r in rows}
    require(len(by_name) == len(rows), "Duplicate task name")
    for row in rows:
        require(set(row) == {"name", "script", "argv", "deps", "done"}, "Unexpected task schema")
        require((PIPELINE_HERE / row["script"]).resolve().is_file(), "pipeline.py script resolution fails")
        require(all(d in by_name for d in row["deps"]), "Unknown dependency")
        require(local(row["done"]).resolve().is_relative_to(OUT.resolve()), "Done outside new namespace")
        require(not any("gpu-uuid" in v or "CUDA_VISIBLE_DEVICES" in v for v in row["argv"]),
                "GPU assignment in task")
        for i, token in enumerate(row["argv"][:-1]):
            if token in ("--out", "--out-root", "--torch-home", "--receipt"):
                require(local(row["argv"][i + 1]).resolve().is_relative_to(OUT.resolve()),
                        "Stage output outside the new namespace")
    pending = set(by_name)
    while pending:
        ready = {name for name in pending if not (set(by_name[name]["deps"]) & pending)}
        require(bool(ready), "Dependency cycle")
        pending -= ready
    return dict(tasks=len(rows), unique_names=True, scripts_resolve_as_pipeline_py=True, acyclic=True,
                GPU_assignment=False, training_submitted=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-root", default=rel(OUT / "tasks"))
    a = p.parse_args()
    refs = historical()
    rows, comparisons = [], []
    for scene in NEW:
        scene_rows, scene_comparisons = generate_scene(scene, refs)
        rows += scene_rows
        comparisons += scene_comparisons
        write_new(local(a.out_root) / f"{scene}_stages.json", scene_rows)
    results, table = comparison_report(comparisons)
    graph = validate_graph(rows)
    write_new(local(a.out_root) / "all_stages.json", rows)
    write_new(local(a.out_root) / "argv_comparison.json", results)
    text_new(local(a.out_root) / "ARGV_COMPARISON.md", table)
    write_new(local(a.out_root) / "graph_validation.json", graph)
    write_new(local(a.out_root) / "external_inputs.json", dict(
        status="planning_only_not_ready_for_end_to_end_execution",
        LR_registry="strict register still needs all training LR PNG hashes",
        schedule="new cullretrain_lr_v1 parents need a compatible actual coarse-overlap producer; "
                 "the pinned producer and old confirmation authorization only support two old native parents",
        scenes={scene: dict(
            actual_schedule=rel(OUT / f"schedules/{scene}/schedule.json"),
            assignment_acceptance=rel(OUT / f"owner_inputs/{scene}/assignment_acceptance.json"),
            selection=rel(OUT / f"owner_inputs/{scene}/selection.json")) for scene in NEW},
        no_receipt_created_to_satisfy_missing_dependencies=True,
        fixed_executor_acceptance_required_before_owner_dispatch=True,
        GPU_assignment=False, training_submitted=False))
    print(__import__("json").dumps(graph))


if __name__ == "__main__":
    main()
