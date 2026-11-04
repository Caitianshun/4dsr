"""Meaningful CPU regressions/negative checks. No torch, pixels, GPU or training.

Regression classifications are explicit: constructor/metadata equality is not
pixel-byte acceptance, a loaded checkpoint, or a reproduced actual schedule.
"""
from __future__ import annotations

import argparse
import ast
import copy
import math
from pathlib import Path
import sys
from unittest import mock

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_new import *
import protocol_new as protocol
import register_new
import materialize_bsync_new as bsync
import make_stages
import stage_gate_new as gate
import teacher_cache_new as teacher
import teacher_paths_new as teacher_paths


def annotate(rows):
    for row in rows:
        name = row["field"]
        if "training_LR_bytes_SHA_verified" in name:
            reason = "本任务无 LR PNG 读权限；0 表示未验收，不能冒称历史的6000"
        elif "protocol_source" in name or "project_sources" in name:
            reason = "新协议适配器和公共工具 SHA；旧源码未修改"
        elif "original_plan" in name:
            reason = "回归使用新建模板路径及其真实 SHA，未改历史登记"
        elif "physical_lr/registry_source/sha256" in name:
            reason = "历史登记绑定旧版physlr整文件SHA，当前只读physlr的SHA不同；本次未修改原文件，未取得旧版源码，不能据此声称整文件字节复现"
        elif name == "/status":
            reason = "新模板/草稿仅 CPU 元数据，不训练、不冒称原登记器已通过"
        elif "CPU_seconds" in name:
            reason = "运行耗时不属于固定配方；模板不记录不稳定耗时"
        elif "template_validation" in name or "draft_validation" in name:
            reason = "新增显式验收范围和待验收状态"
        elif "train_camera_center" in name or "cameras_extent" in name:
            reason = "从实际训练相机重新计算，历史值与本次值仅有浮点末位差异；最大容差1e-12，不复制历史几何"
        else:
            reason = "待审阅；不能自动视作身份差异"
        row["reason"] = reason
    return rows


def template_regression(template):
    old_value = read(historical_prefix())
    for field in ("configuration", "author_sources", "runtime_source_sha256", "planned"):
        require(template[field] == old_value[field], "LR regression changed " + field)
    allowed = {
        "/training_LR_bytes_SHA_verified", "/domain_protocol/protocol_source/path",
        "/domain_protocol/protocol_source/sha256", "/train_camera_center", "/cameras_extent"}
    data_diff = differences(old_value["data"], template["data"])
    require({r["field"] for r in data_diff} <= allowed, "Unexpected LR data regression")
    require(geometry_equal(old_value["data"], template["data"]), "Recomputed LR geometry differs")
    require(old_value["data"]["train_camera_center"] == template["data"]["train_camera_center"]
            and old_value["data"]["cameras_extent"] == template["data"]["cameras_extent"],
            "Historical ordered-sum geometry must reproduce Flame07 exactly")
    return dict(
        status="passed_constructor_and_metadata_regression_only",
        reference=entry(historical_prefix()), configuration_equal=True, author_sources_equal=True,
        runtime_sources_equal=True, planned_counts_equal=True,
        actual_camera_center_and_extent_equal_within_1e12=True,
        actual_camera_center_and_extent_byte_equal=True,
        max_center_absolute_difference=max(abs(a-b) for a,b in zip(
            old_value["data"]["train_camera_center"], template["data"]["train_camera_center"])),
        all_fields_differences=annotate(differences(old_value, template)),
        LR_PNG_byte_validation="not_run_read_scope_excludes_pixels",
        trained=False)


def registry_regression(path):
    original = ROOT / ("output/dynamic_sr_cull_retrain_20261009/wave1/lr/physlr_20261016/"
                       f"registry/flame_steak_area4/seed_{SEED}/config.json")
    reference = snapshot_file(original)
    old_value, value = read(reference), read(path)
    for key in ("configuration", "author_sources", "planned", "runtime_source_sha256", "author_adaptations"):
        require(old_value[key] == value[key], "Registry metadata differs: " + key)
    require(geometry_equal(old_value["data"], value["data"]),
            "Registry geometry changed")
    require(old_value["data"]["train_camera_center"] == value["data"]["train_camera_center"]
            and old_value["data"]["cameras_extent"] == value["data"]["cameras_extent"],
            "Area4 registry historical ordered-sum geometry differs")
    permitted = {
        "/data/domain_protocol/protocol_source/path", "/data/domain_protocol/protocol_source/sha256",
        "/data/train_camera_center", "/data/cameras_extent", "/data/training_LR_bytes_SHA_verified",
        "/status", "/CPU_seconds", "/physical_lr/original_plan/path", "/physical_lr/original_plan/sha256",
        "/physical_lr/registry_source/sha256", "/template_validation", "/draft_validation"}
    diff = differences(old_value, value)
    require(all(r["field"] in permitted or r["field"].startswith(
        "/project_sources/experiments/dynamic_sr_modules_20261010/newscenes_reg_20261103/")
                for r in diff), "Unexplained registry regression field")
    return dict(status="passed_metadata_draft_comparison_original_registrar_not_executed",
                reference_snapshot=entry(reference), reference_original_path=rel(original),
                configuration_author_sources_planned_and_adaptations_equal=True,
                actual_geometry_equal_within_1e12=True, all_fields_differences=annotate(diff),
                all_original_physlr_register_checks="not_run_LR_PNG_outside_read_scope")


def bsync_regression():
    history = read(historical_bsync())
    manifest = read(bound(history["data"]["manifest"]))
    result = bsync.assemble(
        scene_manifest=manifest, data=history["data"], configuration=history["configuration"],
        parent=history["parent"], dependencies=history["dependencies"],
        lr_files=history["training_files"]["LR"], teacher_files=history["training_files"]["teacher"],
        missing=history["missing"], source_files=history["source_files"])
    require(result == history, "Bsync scientific expression differs from historical plan")
    # Every policy/nonidentity field is generated by the pinned constructor.
    # The old dependencies are declared fixture inputs, not re-accepted files.
    return result, dict(
        status="passed_pinned_scientific_constructor_fixture_regression",
        reference=entry(historical_bsync()), constructor=entry(FP / "full_refine_registered_v2.py"),
        full_plan_equal=True, all_fields_differences=[], schedule_policy_equal=True,
        dependencies_schedule_reference_equal=True,
        actual_schedule_rows_regenerated=False,
        parent_or_teacher_dependency_revalidated=False,
        missing_actual_schedule=history["dependencies"]["schedule"],
        reason="actual schedule/coarse-overlap/selection/teacher paths are outside this task's read scope")


def negative_metadata():
    results = []
    mp = PREP / f"flame_salmon/manifest_train_ready_seed{SEED}.json"
    original_read = protocol.read
    mutations = {
        "role_must_remain_held_confirmation_20261102": lambda m: m.update(role="held_confirmation"),
        "phantom_cam03_refused": lambda m: m["cameras"].update(cam03=copy.deepcopy(m["cameras"]["cam02"])),
        "missing_cam10_refused": lambda m: m["splits"]["train"].remove("cam10"),
        "duplicate_train_observation_refused": lambda m: m["observations"].append(
            copy.deepcopy(next(r for r in m["observations"] if r["split"] == "train"))),
        "wrong_frame_time_refused": lambda m: next(r for r in m["observations"] if r["split"] == "train").update(time=0.123),
    }
    for label, mutate in mutations.items():
        value = read(mp)
        mutate(value)
        def changed_read(path):
            return value if local(path) == mp else original_read(path)
        with mock.patch.object(protocol, "read", side_effect=changed_read):
            try:
                protocol.validate_manifest(mp)
            except ValueError as error:
                results.append(dict(name=label, status="passed", evidence=str(error)))
            else:
                raise AssertionError("Invalid metadata accepted: " + label)
    return results


def synthetic_schedule_tests():
    """Algorithm-only fixture. Never written as an accepted experiment schedule."""
    results = []
    for scene in NEW:
        manifest = read(PREP / f"{scene}_area4/manifest_train_ready_seed{SEED}.json")
        cams, poses, scale = bsync.scheduler.pose_rows(manifest)
        def overlap(invert=False):
            frames = {}
            for frame in range(300):
                values = {}
                for i, (pair, pose) in enumerate(poses.items()):
                    ratio = (i % 7) / 7
                    if invert:
                        ratio = 1 - ratio
                    values[pair] = dict(**pose, ratio=ratio)
                frames[frame] = values
            return dict(context=dict(manifest={"fixture": "manifest"}, parent={"fixture": "parent"}),
                        scene=scene, seed=SEED, frames=frames)
        args = (manifest, {"fixture": "manifest"}, {"fixture": "parent"}, SEED)
        table = bsync.scheduler.schedule(*args, overlap(), {"fixture": "overlap"})
        repeated = bsync.scheduler.schedule(*args, overlap(), {"fixture": "overlap"})
        different = bsync.scheduler.schedule(*args, overlap(True), {"fixture": "overlap"})
        bsync.native.validate_schedule(table, manifest, table["identity"], SEED)
        require(table == repeated and table["rows"] != different["rows"],
                "Assignment must be deterministic AND depend on actual overlap")
        require(len(table["rows"]) == 6000 and len(table["record_keys"]) == len(cams) * 300,
                "Wrong full-time schedule population")
        results.append(dict(
            scene=scene, passed=True, synthetic_fixture_not_experiment=True,
            deterministic=True, changed_overlap_changes_rows=True, train_cameras=len(cams),
            keys=len(table["record_keys"]), rows=len(table["rows"]), exposure_audit_passed=True,
            actual_schedule_generated=False))
    return results


def loco_boundary_tests():
    results = []
    for dt, dp, expected in ((0.05, -0.01, "SMX1"), (0.0499, 0.1, "v2"),
                              (0.1, -0.0101, "v2"), (0.1, 0.0, "SMX1")):
        value = dict(kind="loco_eval", camera="cam10", frames=[0, 299, 300],
                     resid={"v2": "fixture/v2", "SMX1": "fixture/SMX1"},
                     checkpoint={"sha256": "synthetic-checkpoint"},
                     summary={"v2": dict(psnr=0., tdiff_psnr=0.),
                              "SMX1": dict(psnr=dp, tdiff_psnr=dt)},
                     decision=dict(delta_tdiff=dt, delta_psnr=dp, choose=expected))
        fit = dict(excluded_cameras=["cam10"], fitting=["cam01", "cam02"],
                   checkpoint_sha256="synthetic-checkpoint", cam00_read=False,
                   args=dict(target="lr", arm="PC", c_reg=True))
        with mock.patch.object(gate, "read", side_effect=lambda p: fit if Path(p).name == "run.json" else value):
            require(gate.choose("fixture/loco.json") == expected, "Frozen LOCO boundary differs")
        results.append(dict(delta_tdiff=dt, delta_psnr=dp, expected=expected, passed=True))
    value["decision"]["choose"] = "v2"
    with mock.patch.object(gate, "read", side_effect=lambda p: fit if Path(p).name == "run.json" else value):
        try:
            gate.choose("fixture/loco.json")
        except ValueError:
            results.append(dict(name="tampered_decision_refused", passed=True))
        else:
            raise AssertionError("Tampered LOCO decision accepted")
    value["decision"]["choose"] = "SMX1"
    fit["excluded_cameras"] = []
    with mock.patch.object(gate, "read", side_effect=lambda p: fit if Path(p).name == "run.json" else value):
        try:
            gate.choose("fixture/loco.json")
        except ValueError:
            results.append(dict(name="in_sample_fit_refused", passed=True))
        else:
            raise AssertionError("LOCO fit without camera exclusion accepted")
    return results


def point_and_geometry_checks(templates):
    """Read only explicitly permitted initial points; never images or models."""
    import numpy as np
    results = []
    for scene in NEW:
        for formation in (None, "area4"):
            _, data, _ = protocol.validate_manifest(
                PREP / (scene if formation is None else scene + "_area4") /
                f"manifest_train_ready_seed{SEED}.json", formation=formation)
            manifest = read(bound(data["manifest"]))
            centers = np.asarray([manifest["cameras"][c]["c2w"] for c in data["train_cameras"]],
                                 dtype=np.float64)[:, :3, 3]
            center = centers.mean(axis=0)
            extent = float(1.1 * np.linalg.norm(centers - center, axis=1).max())
            require(np.allclose(center, data["train_camera_center"], atol=1e-12, rtol=0)
                    and abs(extent-data["cameras_extent"]) <= 1e-12,
                    "Independent NumPy train-only geometry calculation differs")
            with np.load(bound(data["initial_points"]), allow_pickle=False) as points:
                n = points["points"].shape[0]
                require(points["points"].shape == points["colors"].shape == (n, 3)
                        and np.isfinite(points["points"]).all() and np.isfinite(points["colors"]).all()
                        and n == read(bound(data["initialization_receipt"]))["point_count"],
                        "Initial point structure/count/finite check failed")
            results.append(dict(scene=scene, formation=formation or "bicubic", points=n,
                                cameras=len(data["train_cameras"])+1,
                                train_cameras=len(data["train_cameras"]),
                                actual_train_only_numpy_geometry_matches=True))
    return results


def adapter_checks():
    """Prove the bounded path exception and driver process boundary on CPU."""
    root = PREP / "flame_salmon_area4"
    path = OUT / "teacher_cache/flame_salmon_area4/sr_swinir_x4_areaft/cam01/0000.png"
    token = __import__("os").path.relpath(path, root)
    safe = teacher_paths.resolver(bsync.native.safe_child, root, {token: path})
    require(safe(root, token) == path, "Registered external teacher path was rejected")
    results = [dict(name="exact_teacher_inventory_path_allowed", passed=True)]
    for label, dataset, value in (
        ("unregistered_cache_escape_refused", root, token.replace("0000", "0001")),
        ("wrong_dataset_refused", PREP / "sear_steak_area4", token),
        ("arbitrary_project_escape_refused", root, "../../../unknown.png")):
        try:
            safe(dataset, value)
        except ValueError:
            results.append(dict(name=label, passed=True))
        else:
            raise AssertionError(label)
    require(safe(root, "lr/cam01/0000.png") == root / "lr/cam01/0000.png",
            "Original dataset-local safe_child behavior changed")
    kit = teacher_paths.GEO / "modkit6.py"
    original = ["/owner/absolute/python", "-u", str(kit), "--lr-operator", "area",
                "--", "--mode", "fix", "cullretrain", "sr",
                "--out", rel(OUT / "sources/A/sr/fixture")]
    rewritten = teacher_paths.child_command(original, kit, OUT / "teacher_cache/index.json")
    require(rewritten[rewritten.index("--") + 1:] == original[3:],
            "Teacher child adapter changed original SR argv")
    results.append(dict(name="driver_child_preserves_scientific_argv", passed=True))
    residual = OUT / "fit/synthetic_selected"
    require(gate.substitute_residual(["PC=@RESID@", "@RESID@"], residual) ==
            ["PC="+rel(residual), rel(residual)], "LABEL=PATH residual substitution failed")
    results.append(dict(name="selected_residual_both_argument_forms", passed=True))
    # A changed scalar must fail the comparison, even when changed to a path.
    refs = make_stages.historical()
    _, comparisons = make_stages.generate_scene("sear_steak", refs)
    item = copy.deepcopy(next(x for x in comparisons if x["stage"] == "source_A"))
    item["new_argv"][item["new_argv"].index("--lambda-sr")+1] = "output/bad"
    try:
        make_stages.comparison_report([item])
    except ValueError:
        results.append(dict(name="nonpath_recipe_change_refused", passed=True))
    else:
        raise AssertionError("A scientific parameter change escaped argv regression")
    return results


def frozen_view_selection_checks():
    """Run the frozen selection functions on metadata; no PNG/metric access."""
    directory = ROOT / "experiments/dynamic_sr_modules_20261010/area4_baselines_20261024"
    ps = load("_newscene_frozen_view_selection", directory / "frozen/prepare_suite.py")
    source = directory / "score_views_area4.py"
    original = pure_function(source, "original_of",
                             dict(ROOT=ROOT, PS=ps, FORMATION="area4",
                                  json=__import__("json")))
    select = pure_function(source, "select_observations_area4",
                           dict(original_of=original, PS=ps))
    results = []
    for scene in NEW:
        m = read(PREP / f"{scene}_area4/manifest_train_ready_seed{SEED}.json")
        test, train = select(m, "test"), select(m, "train")
        require(len(test) == 300 and len(train) == len(m["splits"]["train"])*30
                and {r["camera_id"] for r in train} == set(m["splits"]["train"]),
                "Frozen newscene train/test view population differs")
        results.append(dict(scene=scene, train_views=len(train), test_views=len(test),
                            original_frozen_selection_passed=True, pixels_read=0,
                            scoring_executed=False))
    return results


def check_initial_files():
    inventory = read(OUT / "entry_inventory.json")
    for name, expected in inventory.items():
        require(sha(ROOT / name) == expected, "Pre-existing file changed: " + name)
    return dict(files=len(inventory), byte_identical=True, files_sha256=inventory)


def run(out):
    out = output_path(out)
    initial_sources = {rel(p): sha(p) for p in sorted(CODE.glob("*.py"))}
    for path in CODE.glob("*.py"):
        ast.parse(path.read_text(), filename=str(path))
    templates = {scene: protocol.build_template(scene) for scene in (*NEW, "flame_steak")}
    template_result = template_regression(templates["flame_steak"])
    template_path = out / f"template_input/flame_steak/seed_{SEED}/config.json"
    write_new(template_path, templates["flame_steak"])
    draft_path = register_new.draft("flame_steak", namespace=out / "registry_regression",
                                    templates=out / "template_input")
    registry_result = registry_regression(draft_path)
    reconstructed, bsync_result = bsync_regression()
    # Avoid duplicating thousands of file-inventory entries on a near-full disk.
    write_new(out / "bsync_fixture_digest.json",
              dict(reference=entry(historical_bsync()), reconstructed_semantic_sha256=digest(reconstructed),
                   historical_semantic_sha256=digest(read(historical_bsync())), equal=True))
    # Source-expression equality includes every one of the historical top-level
    # fields; an explicit field list makes the fixture's extent reviewable.
    bsync_result["compared_top_level_fields"] = list(reconstructed)
    negative = negative_metadata()
    schedules = synthetic_schedule_tests()
    decisions = loco_boundary_tests()
    points = point_and_geometry_checks(templates)
    adapters = adapter_checks()
    selections = frozen_view_selection_checks()
    _, teacher_result = teacher.adapted_source()
    refs = make_stages.historical()
    tasks, comparisons = [], []
    for scene in NEW:
        generated, compared = make_stages.generate_scene(scene, refs)
        tasks += generated
        comparisons += compared
        m = read(PREP / f"{scene}_area4/manifest_train_ready_seed{SEED}.json")
        require("cam10" in m["splits"]["train"], "New LOCO camera does not exist")
        sr = next(row for row in generated if row["name"].endswith("_student"))
        require(len([v for v in sr["argv"] if v.startswith(("A=", "B=", "C="))]) == 3,
                "Student needs exactly three pre-P sources")
        require("--seed" not in sr["argv"] and "--scale" in sr["argv"], "Historical student defaults changed")
        for row in generated:
            for i, token in enumerate(row["argv"][:-1]):
                if token in ("--out", "--out-root", "--torch-home", "--receipt"):
                    require(local(row["argv"][i + 1]).resolve().is_relative_to(OUT.resolve()),
                            "A stage writes outside the new namespace")
    graph = make_stages.validate_graph(tasks)
    tokens, _ = make_stages.comparison_report(comparisons)
    unchanged = check_initial_files()
    require(initial_sources == {rel(p): sha(p) for p in sorted(CODE.glob("*.py"))},
            "Adapter sources changed during the check; use a fresh regression output")
    require("torch" not in sys.modules, "CPU task imported torch")
    result = dict(
        status="passed_CPU_checks_with_explicit_unrun_acceptance",
        interpreter=sys.executable, isolated=sys.flags.isolated,
        numpy=__import__("numpy").__version__, ROOT=rel(ROOT),
        torch_imported=False, GPU_calls=0, training_submitted=False,
        source_files=initial_sources,
        syntax_files=len(list(CODE.glob("*.py"))),
        new_scenes={scene: dict(batch=templates[scene]["planned"]["batch_size"],
                                cameras=len(templates[scene]["data"]["train_cameras"]) + 1,
                                train=len(templates[scene]["data"]["train_cameras"]),
                                center=templates[scene]["data"]["train_camera_center"],
                                extent=templates[scene]["data"]["cameras_extent"],
                                initialization_points=read(bound(templates[scene]["data"]["initialization_receipt"]))["point_count"])
                    for scene in NEW},
        template_regression=template_result, registry_regression=registry_result,
        bsync_regression=bsync_result, negative_metadata=negative,
        synthetic_schedule_tests=schedules, loco_threshold_tests=decisions,
        initial_points_and_geometry=points, bounded_path_and_argv_adapters=adapters,
        frozen_view_selection=selections,
        teacher_path_adapter=teacher_result, task_graph=graph,
        argv_comparisons=dict(stages=len(tokens), tokens=sum(len(x["tokens"]) for x in tokens), all_allowed=True),
        preexisting_files=unchanged,
        not_run=["all-training-LR-PNG hashes", "strict physlr.register",
                 "actual schedule/overlap regression", "checkpoint tensor load",
                 "teacher inference", "strict physlr.sr_register", "training/render/scoring execution"])
    write_new(out / "checks.json", result)
    write_new(out / "template_field_differences.json", template_result)
    write_new(out / "registry_field_differences.json", registry_result)
    write_new(out / "bsync_field_differences.json", bsync_result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default=rel(OUT / "regression/cpu_v1"))
    a = p.parse_args()
    try:
        result = run(a.out)
    except Exception as error:
        write_new(local(a.out) / "failure.json",
                  dict(status="failed_CPU_regression_preserved", error=repr(error),
                       traceback=__import__("traceback").format_exc(),
                       interpreter=sys.executable, isolated=sys.flags.isolated,
                       torch_imported="torch" in sys.modules, training_submitted=False))
        raise
    print(__import__("json").dumps(dict(
        status=result["status"], syntax_files=result["syntax_files"],
        metadata_negative_checks=len(result["negative_metadata"]),
        synthetic_schedule_scenes=len(result["synthetic_schedule_tests"]),
        loco_threshold_checks=len(result["loco_threshold_tests"]),
        task_graph=result["task_graph"], argv=result["argv_comparisons"],
        preexisting_files_unchanged=result["preexisting_files"]["byte_identical"],
        torch_imported=result["torch_imported"], full_acceptance_not_run=result["not_run"])))


if __name__ == "__main__":
    main()
