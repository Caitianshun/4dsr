"""Original assignment and validators, with explicit new-scene receipt binding."""
from __future__ import annotations

import argparse
import ast
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sched_common import *


def scalar_overlap(index_path, manifest):
    """Run the exact scalar-validation tail of original actual_overlap().

    Its old private systemd acceptance prefix is supplied by acceptance_new.py
    for new runs, or is explicitly outside the historical CPU replay's scope.
    No pair, pose, ratio, population or finite-value validation is removed.
    """
    coarse, scheduler, _ = science_modules()
    index = read(index_path)
    context = index["context"]
    require(context["manifest"] == entry(PREP / (manifest["scene"] +
            ("_area4" if "physical_lr_derivation" in manifest else "")) /
            f"manifest_train_ready_seed{SEED}.json"), "Index manifest differs")
    require(index["scene"] == manifest["scene"] and index["seed"] == SEED,
            "Index scene/seed differs")
    require(index["schema"] == coarse.INDEX_SCHEMA
            and index["status"] == "completed_frozen_native_parent_coarse_overlap"
            and index["policy"] == coarse.POLICY
            and index["policy_sha256"] == digest(coarse.POLICY), "Index policy/status differs")
    cams, poses, scale = scheduler.pose_rows(manifest)
    p = FP / "full_held_native_schedule_materializer.py"
    tree = ast.parse(p.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "actual_overlap")
    start = next(i for i, node in enumerate(fn.body)
                 if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                 and ast.get_source_segment(p.read_text(), node).startswith("require(len(index['frames'])"))
    nodes = fn.body[start:-1]  # original complete frame/pair validation, excluding return
    env = dict(vars(scheduler), manifest=manifest, index=index, context=context,
               cams=cams, poses=poses, scale=scale, root=ROOT)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(p), "exec"), env)
    return dict(context=context, scene=manifest["scene"], seed=SEED, frames=env["frames"],
                frame_entries=index["frames"], poses=poses, baseline_scale=scale,
                index=entry(index_path), registration=index["registration"], cost=index["actual"])


def prepared_inputs(overlap, pose_ref, accepted):
    scheduler = science_modules()[1]
    with attributes(scheduler, COARSE_SOURCE=FP / "full_held_parent_overlap.py"):
        prepared = scheduler.prepared_overlap(overlap, pose_ref)
    prepared["operational_adapter"] = dict(
        source=entry(CODE / "acceptance_new.py"),
        acceptance_schema=accepted["schema"],
        validation="original full-output SHA/closure plus separately supplied fixed-service Exit0",
        parent_family=read(bound(overlap["registration"]))["parent_loading"]["family"])
    return prepared


def generate(index_path, accepted_path, out):
    _, scheduler, native = science_modules()
    import acceptance_new
    accepted = acceptance_new.validate_coarse_acceptance(accepted_path)
    require(accepted["index"] == entry(index_path), "Accepted index differs")
    index = read(index_path)
    mp = bound(index["context"]["manifest"])
    manifest, _, _, _ = manifest_metadata(mp)
    overlap = scalar_overlap(index_path, manifest)
    overlap.update(accepted=entry(accepted_path), postcheck_cost=accepted["postcheck_cost"])
    out = output_path(out)
    require(not out.exists(), "Fresh schedule directory required")
    out.mkdir(parents=True)
    pose = dict(schema="actual_manifest_train_pose_pairs_v1", manifest=index["context"]["manifest"],
                pairs=[dict(a=k[0], b=k[1], **v) for k, v in overlap["poses"].items()],
                baseline_scale=overlap["baseline_scale"])
    pose_ref = entry(write_new(out / "training_pose.json", pose, sorted_keys=True))
    # Reuse the original preparation expression. "root_acceptance" now binds the
    # separately verified fixed-service receipt; no historical Exit0 is relabeled.
    prepared = prepared_inputs(overlap, pose_ref, accepted)
    prepared_ref = entry(write_new(out / "overlap_inputs.json", prepared, sorted_keys=True))
    table = scheduler.schedule(manifest, index["context"]["manifest"], index["context"]["parent"],
                               SEED, overlap, prepared_ref)
    native.validate_schedule(table, manifest, index["context"], SEED)
    schedule_ref = entry(write_new(out / "schedule.json", table, sorted_keys=True))
    stats = dict(schema="actual_parent_overlap_assignment_statistics_v3",
                 schedule=schedule_ref, prepared_overlap_inputs=prepared_ref,
                 selected_pairs=table["assignment_protocol"]["selected_pairs"],
                 pair_frequency=table["assignment_protocol"]["pair_frequency"],
                 overlap_bins=table["assignment_protocol"]["overlap_bins"],
                 full_legal_observations=len(table["record_keys"]),
                 calibration_anchors=len(table["calibration_triplets"]),
                 RGB_column_exposure_equal=True, random_same_RGB_column_exposure_equal=True,
                 teacher_weighted_exposure_equal_B0_Bsync=True, HR_or_quality_used=False)
    stats_ref = entry(write_new(out / "assignment_statistics.json", stats, sorted_keys=True))
    candidate = scheduler.assignment_candidate(table, schedule_ref, prepared_ref, overlap,
                                                pose_ref, stats_ref)
    write_new(out / "assignment_acceptance_candidate.json", candidate, sorted_keys=True)
    result = dict(status="completed_CPU_actual_assignment_pending_owner_acceptance",
                  schedule=schedule_ref, candidate=entry(out / "assignment_acceptance_candidate.json"),
                  source=entry(__file__), overlap_index=entry(index_path), GPU_calls=0,
                  training_submitted=False, algorithm_source=entry(FP / "full_held_native_schedule_materializer.py"))
    no_torch()
    write_new(out / "complete.json", result)
    return result


def regression(out):
    """Real 300-frame index replay; exact original input identity and JSON bytes."""
    _, scheduler, native = science_modules()
    historical = read(HIST_SCHEDULE)
    bsync = read(HIST_BSYNC)
    refresh = read(FPOUT / f"held_confirmation_native_schedules/{TAG}/schedule_B_actual_root_assignment_v1/complete.json")
    require(bsync["dependencies"]["schedule"] == refresh["schedule"] == entry(HIST_SCHEDULE),
            "Bsync historical schedule selection differs")
    require(refresh["schedule_reused_exact_entry"] is True, "B was not an exact reuse")
    manifest, _, _, _ = manifest_metadata(bound(historical["identity"]["manifest"]))
    overlap = scalar_overlap(HIST_INDEX, manifest)
    prepared_ref = historical["assignment_protocol"]["parent_overlap_inputs"]
    prepared = read(bound(prepared_ref))
    require(bound(prepared["coarse_overlap_table"]) == HIST_INDEX
            and prepared["context"] == overlap["context"]
            and prepared["frames"] == overlap["frame_entries"],
            "Historical prepared inputs do not refer to this real index")
    pose = read(bound(prepared["pose_table"]))
    require(pose["manifest"] == historical["identity"]["manifest"]
            and pose["baseline_scale"] == overlap["baseline_scale"], "Pose scale differs")
    tick = time.monotonic()
    table = scheduler.schedule(manifest, historical["identity"]["manifest"],
                               historical["identity"]["parent"], SEED, overlap, prepared_ref)
    native.validate_schedule(table, manifest, historical["identity"], SEED)
    out = output_path(out)
    generated = write_new(out / "schedule.json", table, sorted_keys=True)
    diffs = differences(historical, table)
    write_new(out / "field_differences.json", diffs)
    result = dict(status="passed" if generated.read_bytes() == HIST_SCHEDULE.read_bytes() else "failed",
                  original=entry(HIST_SCHEDULE), regenerated=entry(generated),
                  byte_equal=generated.read_bytes() == HIST_SCHEDULE.read_bytes(),
                  fields_equal=not diffs, rows_equal=table["rows"] == historical["rows"],
                  random_rows_equal=table["random_rows"] == historical["random_rows"],
                  index=entry(HIST_INDEX), frame_SHA_checked=300,
                  unordered_pairs_checked=sum(len(v) for v in overlap["frames"].values()),
                  updates=table["updates"], actual_schedule="schedule_A_original_v1",
                  B_reuses_A=True, seconds=time.monotonic() - tick,
                  scope="Real scalar inputs and original schedule validator; no fresh GPU/Exit0 claim",
                  raw_NPZ_reopened=0, historical_private_root_receipt_revalidated=False)
    write_new(out / "result.json", result)
    require(result["byte_equal"], "Schedule byte regression failed; see field_differences.json")
    no_torch()
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("generate", "regression"))
    p.add_argument("--index")
    p.add_argument("--coarse-acceptance")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if a.mode == "regression":
        value = regression(a.out)
    else:
        require(a.index and a.coarse_acceptance, "Index and real coarse acceptance required")
        value = generate(a.index, a.coarse_acceptance, a.out)
    print(json.dumps(value, ensure_ascii=False))


if __name__ == "__main__":
    main()
