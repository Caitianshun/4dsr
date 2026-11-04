"""Materialize Bsync policy plans without inventing a parent-dependent schedule.

plan: write an explicitly pending plan from metadata (no torch).
schedule: legacy Flame07 only, with its original confirmation authorization.
New physical parents require an owner-provided compatible overlap producer;
require-schedule validates its actual accepted schedule without inventing rows.
materialize: validate actual parent/teacher/selection/schedule and write a fresh
ready config. This command does not train and never initializes CUDA.
"""
from __future__ import annotations

import argparse
import ast
import copy
from pathlib import Path
import sys
from types import SimpleNamespace

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_new import *
import protocol_new as protocol
import full_refine_registered_v2 as native
import full_held_native_schedule_materializer as scheduler

SOURCE_PIN = "c278928f56b0515adedf15eef9a30850e8bd2a4c41e6f6e0f828068ca774b75d"


def assemble(*, scene_manifest, data, configuration, parent, dependencies, lr_files,
             teacher_files, missing, source_files, seed=SEED):
    """Reuse the EXACT pure plan expression in the pinned scientific constructor.

    Validation is performed separately; this pure expression does not certify any
    supplied dependency. It also enables an honest metadata-fixture regression.
    """
    path = FP / "full_refine_registered_v2.py"
    require(sha(path) == SOURCE_PIN, "Scientific constructor pin differs")
    build = next(n for n in ast.parse(path.read_text()).body
                 if isinstance(n, ast.FunctionDef) and n.name == "build_plan")
    assignment = next(n for n in build.body if isinstance(n, ast.Assign)
                      and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "plan")
    context = dict(manifest=data["manifest"], parent=parent["checkpoint"] if parent else None,
                   schedule=dependencies.get("schedule"))
    env = dict(vars(native))
    env.update(m=scene_manifest, data=data, configuration=configuration, parent=parent,
               dependencies=dependencies, context=context, lr_files=lr_files,
               teacher_files=teacher_files, missing=missing,
               definition=copy.deepcopy(native.ARMS["Bsync"]),
               a=SimpleNamespace(seed=seed, method="Bsync"), upstream=RUNTIME,
               source_files=lambda _: source_files)
    exec(compile(ast.Module(body=[assignment], type_ignores=[]), str(path), "exec"), env)
    return env["plan"]


def declared_sources():
    """Verify authorized source files; keep the external depth-prior SHA declared.

    Its source is irrelevant to Bsync (X=False) but present in historical plans.
    This task has no read permission for that directory, so it is not rehashed.
    """
    sources = copy.deepcopy(read(historical_bsync())["source_files"])
    external = []
    for name, expected in sources["project"].items():
        if name.startswith("experiments/dynamic_sr_confidence_geometry_20261006/"):
            external.append(name)
        else:
            require(sha(ROOT / name) == expected, "Pinned scientific source differs: " + name)
    for name, expected in sources["upstream"].items():
        require(sha(RUNTIME / name) == expected, "Pinned runtime differs: " + name)
    return sources, external


def plan(scene):
    m, data, lr_files = protocol.validate_manifest(
        PREP / f"{scene}_area4/manifest_train_ready_seed{SEED}.json", formation="area4")
    configuration, _ = protocol.configuration(scene)
    sources, external = declared_sources()
    missing = [
        dict(dependency="complete_gateaff_LR_parent", reason="not trained in this task"),
        dict(dependency="areaft_teacher", reason="not generated in this task"),
        dict(dependency="schedule", reason="actual frozen-parent coarse overlap is required"),
        dict(dependency="selection", reason="owner must supply existing frozen recipe evidence"),
        dict(dependency="LR_bytes", reason="PNG paths outside this task's read scope"),
    ]
    value = assemble(scene_manifest=m, data=data, configuration=configuration, parent=None,
                     dependencies={}, lr_files=lr_files, teacher_files=[], missing=missing,
                     source_files=sources)
    value["materialization"] = dict(
        wrapper=entry(__file__), constructor=entry(FP / "full_refine_registered_v2.py"),
        direct_area4_manifest_supported=True, same_scientific_schedule_policy=True,
        actual_schedule_rows_generated=False, GPU_dispatch_allowed=False,
        external_sources_declared_not_rehashed=external,
        reason="assignment depends on actual frozen-parent overlap, not just camera/frame/seed")
    return value


def physical_parent(scene, path, data):
    path = local(path)
    complete_path, sidecar_path = path.parent / "complete.json", path.with_suffix(".json")
    complete, side = read(complete_path), read(sidecar_path)
    actual = entry(path)
    require(complete["status"] == "completed" and complete["kind"] == "lr"
            and complete["scene"] == scene and complete["seed"] == SEED
            and complete["arm"] == "gateaff" and complete["args"]["affine"] is True
            and complete["args"]["prune_gate"] == "none", "Complete same-scene gateaff parent required")
    require(complete["checkpoint"] == actual and
            {k: side[k] for k in ("path", "sha256")} == actual, "Parent/sidecar SHA differs")
    require(side["schema"] == "cullretrain_lr_v1"
            and side["metadata"]["state"] == dict(stage="fine", coarse_iteration=3000,
                                                  fine_iteration=14000, optimizer_resets=2)
            and complete["state"] == side["metadata"]["state"]
            and side["metadata"]["arm"] == "gateaff",
            "Full coarse3000/fine14000 physical parent required")
    plan_path = local(complete["parent"]) / "config.json"
    lr_plan = read(plan_path)
    require(lr_plan["data"]["manifest"] == data["manifest"] and lr_plan["seed"] == SEED
            and lr_plan["scene"] == scene, "Parent registry/manifest/seed differs")
    return dict(checkpoint=actual, complete=entry(complete_path), plan=entry(plan_path),
                sidecar=entry(sidecar_path), state=side["metadata"]["state"]), lr_plan


def teacher_files(index_path, manifest, manifest_path):
    """All physlr.sr_register teacher checks, with an explicit index location."""
    index = read(index_path)
    train = sorted(manifest["splits"]["train"])
    rows = {(r["camera_id"], int(r["frame_index"])): r for r in manifest["observations"]
            if r["split"] == "train"}
    keys = [(r["camera"], int(r["frame"])) for r in index["entries"]]
    require(index["status"] == "completed_teacher_inventory" and index["seed_consumer"] == SEED
            and sorted(index["train_cameras"]) == train and len(keys) == len(set(keys))
            and set(keys) == set(rows), "Incomplete/duplicate teacher keys")
    require(index["manifest_sha256"] == sha(manifest_path), "Teacher manifest SHA differs")
    cfg = index["teacher_config"]
    ft = read(ROOT / "output/dynamic_sr_physlr_20261016/teacher_ft/swinir_area_ft_v1/summary.json")
    require(index["scene"] == manifest["scene"] and index["oracle"] is False
            and index["seed_independent_frozen_prior"] is True
            and index["HR_quality_evaluation"] is False
            and cfg["tag"] == "sr_swinir_x4_areaft" and cfg["batch"] == 6
            and cfg["scale"] == 4 and cfg["mask_cache"] is True
            and cfg["weights"] == ft["weights"],
            "Teacher must use the frozen areaft weights and historical batch6 recipe")
    files = []
    root = manifest_path.parent
    for key, row in zip(keys, index["entries"]):
        obs = rows[key]
        require(row["lr_relative_path"] == obs["lr_path"] and
                Path(row["lr_relative_path"]).parts[0] == "lr" and
                row["lr_sha256"] == obs["lr_sha256"], "Teacher LR input differs")
        path = (root / row["relative_path"]).resolve()
        require(path.resolve().is_relative_to((OUT / "teacher_cache").resolve()),
                "Teacher output outside new namespace")
        require(sha(path) == row["sha256"], "Teacher image SHA differs")
        files.append(dict(path=rel(path.resolve()), sha256=row["sha256"],
                          camera=key[0], frame=key[1]))
    return files


def make_schedule(scene, parent_path, coarse_acceptance, out, confirmation_authorization):
    require(scene == "flame_steak",
            "The pinned overlap producer/authorization only supports original held native parents. "
            "No compatible newscene cullretrain_lr_v1 overlap producer is present. "
            "Use require-schedule only after Claude supplies a genuinely produced, accepted schedule.")
    manifest_path = PREP / scene / f"manifest_train_ready_seed{SEED}.json"
    manifest, data, _ = native.prefix.validate_manifest(manifest_path, SEED)
    parent, _ = native.full_parent(local(parent_path), local(parent_path).parent / "complete.json",
                                   data["manifest"], SEED)
    accepted = entry(coarse_acceptance)
    # These are original strict receipt/Exit0/source/cost/all-output SHA gates.
    # They are NOT monkeypatched for a different checkpoint family. A native-only
    # receipt cannot silently stand in for the physical gateaff parent's receipt.
    require(confirmation_authorization, "Original --confirmation-authorization required")
    scheduler.configure_closed_sources(accepted, entry(confirmation_authorization))
    overlap = scheduler.actual_overlap(accepted, manifest, data["manifest"], parent["checkpoint"], SEED)
    out = output_path(out)
    _, pairs, scale = scheduler.pose_rows(manifest)
    pose = dict(schema="actual_manifest_train_pose_pairs_v1", manifest=data["manifest"],
                pairs=[dict(a=k[0], b=k[1], **v) for k, v in pairs.items()], baseline_scale=scale)
    pose_path = write_new(out / "training_pose.json", pose)
    prepared = scheduler.prepared_overlap(overlap, entry(pose_path))
    prepared_path = write_new(out / "overlap_inputs.json", prepared)
    table = scheduler.schedule(manifest, data["manifest"], parent["checkpoint"], SEED, overlap, entry(prepared_path))
    native.validate_schedule(table, manifest, table["identity"], SEED)
    table_path = write_new(out / "schedule.json", table)
    write_new(out / "complete.json", dict(
        status="completed_CPU_assignment_root_acceptance_still_required",
        schedule=entry(table_path), source=entry(__file__), prepared_overlap=entry(prepared_path),
        GPU_calls=0, training_submitted=False))
    return table_path


def require_schedule(scene, parent, schedule, assignment_acceptance, *, verify_lr_bytes=False):
    """Validate actual identities and original schedule audit; never rebind old rows."""
    mp = PREP / f"{scene}_area4/manifest_train_ready_seed{SEED}.json"
    manifest, data, _ = protocol.validate_manifest(
        mp, formation="area4", verify_lr_bytes=verify_lr_bytes)
    pv, _ = physical_parent(scene, parent, data)
    table = read(schedule)
    native.validate_schedule(table, manifest,
                             dict(manifest=data["manifest"], parent=pv["checkpoint"]), SEED)
    assignment = read(assignment_acceptance)
    require(assignment["status"] == "root_accepted_actual_frozen_parent_coarse_overlap_assignment"
            and assignment["source_registration_accepted"] is True
            and assignment["storage_and_cost_accepted"] is True
            and assignment["full_native_context"] == dict(table["identity"], schedule=entry(schedule))
            and assignment["actual_parent_rendering_completed"] is True
            and assignment["synthetic"] is False
            and assignment["HR_GT_reads"] == assignment["heldout_reads"] == 0,
            "Actual same-scene frozen-parent schedule acceptance missing or mismatched")
    policy = table["assignment_protocol"]
    require(policy["schema"] == scheduler.POLICY
            and policy["source"] == entry(FP / "full_held_native_schedule_materializer.py")
            and policy["HR_or_test_or_quality_cost"] is False,
            "Historical assignment science or training-only boundary changed")
    # Rehash all metadata referenced by the original acceptance; raw payload
    # verification belongs to its immediate producer-return/Exit0 acceptance.
    for key in ("source", "pose_table", "coarse_overlap_table", "assignment_statistics",
                "root_acceptance", "prepared_overlap_inputs", "schedule"):
        bound(assignment[key])
    prepared = read(bound(policy["parent_overlap_inputs"]))
    require(prepared["context"] == table["identity"] and prepared["scene"] == scene
            and prepared["seed"] == SEED and prepared["synthetic"] is False
            and prepared["actual_parent_rendering_completed"] is True,
            "Overlap preparation belongs to a different parent/scene")
    return table


def materialize(scene, *, parent, teacher, schedule, selection, assignment_acceptance):
    manifest_path = PREP / f"{scene}_area4/manifest_train_ready_seed{SEED}.json"
    m, data, lr_files = protocol.validate_manifest(manifest_path, formation="area4", verify_lr_bytes=True)
    parent_value, lr_plan = physical_parent(scene, parent, data)
    configuration, _ = protocol.configuration(scene)
    require(lr_plan["configuration"] == configuration, "LR author configuration differs")
    tf = teacher_files(local(teacher), m, manifest_path)
    native.validate_selection(read(selection), "Bsync")
    table = read(schedule)
    native.validate_schedule(table, m, dict(manifest=data["manifest"], parent=parent_value["checkpoint"]), SEED)
    require_schedule(scene, parent, schedule, assignment_acceptance, verify_lr_bytes=True)
    sources, external = declared_sources()
    # A ready plan must actually verify the external declared source too. This
    # command is for Claude's less restricted metadata environment, not this P2.
    for name in external:
        require(sha(ROOT / name) == sources["project"][name], "External science source differs")
    dependencies = dict(selection=entry(selection), teacher=entry(teacher), schedule=entry(schedule))
    value = assemble(scene_manifest=m, data=data, configuration=configuration, parent=parent_value,
                     dependencies=dependencies, lr_files=lr_files, teacher_files=tf, missing=[],
                     source_files=sources)
    value["materialization"] = dict(
        wrapper=entry(__file__), constructor=entry(FP / "full_refine_registered_v2.py"),
        direct_area4_manifest_supported=True, parent_checkpoint_family="cullretrain_lr_v1_gateaff",
        parent_validated_from_SHA_bound_sidecar=True, torch_checkpoint_loaded=False,
        assignment_acceptance=entry(assignment_acceptance), GPU_dispatch_allowed=False)
    return value


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("plan", "schedule", "require-schedule", "materialize"))
    p.add_argument("--scene", choices=(*NEW, "flame_steak"), required=True)
    p.add_argument("--out")
    for name in ("parent", "teacher", "schedule", "selection", "assignment-acceptance",
                 "coarse-acceptance", "confirmation-authorization"):
        p.add_argument("--" + name)
    a = p.parse_args()
    if a.mode == "schedule":
        require(a.parent and a.coarse_acceptance and a.out, "schedule requires parent, coarse acceptance and fresh out")
        path = make_schedule(a.scene, a.parent, a.coarse_acceptance, a.out,
                             a.confirmation_authorization)
        print(__import__("json").dumps(dict(schedule=entry(path))))
        return
    if a.mode == "require-schedule":
        require(a.parent and a.schedule and a.assignment_acceptance and a.out,
                "require-schedule needs actual parent/schedule/assignment acceptance and fresh --out receipt")
        require_schedule(a.scene, a.parent, a.schedule, a.assignment_acceptance)
        write_new(a.out, dict(status="actual_external_schedule_metadata_validated",
                              schedule=entry(a.schedule), assignment=entry(a.assignment_acceptance),
                              source=entry(__file__), training_submitted=False))
        print(__import__("json").dumps(dict(status="actual_external_schedule_metadata_validated")))
        return
    if a.mode == "plan":
        value = plan(a.scene)
        out = a.out or OUT / f"bsync_pending/{a.scene}/seed{SEED}/Bsync/scientific_plan.json"
    else:
        require(all(getattr(a, k) for k in ("parent", "teacher", "schedule", "selection", "assignment_acceptance")),
                "Actual parent, teacher, schedule, selection, assignment acceptance required")
        value = materialize(a.scene, parent=a.parent, teacher=a.teacher, schedule=a.schedule,
                            selection=a.selection, assignment_acceptance=a.assignment_acceptance)
        out = a.out or OUT / f"clean_bsync/{a.scene}/seed{SEED}/Bsync/scientific_plan.json"
    path = write_new(out, value)
    print(__import__("json").dumps(dict(plan=entry(path), status=value["status"], missing=value["missing"])))


if __name__ == "__main__":
    main()
