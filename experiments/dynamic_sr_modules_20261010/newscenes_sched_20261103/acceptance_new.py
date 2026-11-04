"""CPU acceptance bridge. Never invent service Exit0 or owner acceptance."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sched_common import *

COARSE_SCHEMA = "fixed_service_actual_coarse_acceptance_v1"


def pointer(value, path):
    require(path.startswith("/"), "Use an absolute JSON pointer")
    for key in path[1:].split("/"):
        key = key.replace("~1", "/").replace("~0", "~")
        value = value[int(key)] if isinstance(value, list) else value[key]
    return value


def validate_admission(admission):
    require(admission["schema"] == "fixed_service_overlap_admission_v1"
            and admission["admitted"] is True and admission["interpreter"] == TRAINPY
            and admission["env_id"] == "mml_bw" and admission["project_id"] == "4dsr",
            "Matching fixed-service admission required")
    raw = read(bound(admission["admission_evidence"]))
    for key in ("project_id", "job_id", "unit", "gpu_uuid"):
        require(pointer(raw, admission["raw_fields"][key]) == admission[key],
                "Raw admission identity differs: " + key)
    require(admission["raw_admitted_status"] in ("admitted", "queued", "ready", "running")
            and pointer(raw, admission["raw_fields"]["status"]) == admission["raw_admitted_status"],
            "Raw service admission absent")
    return admission


def verify_service_exit(path, complete, validation):
    """Normalized owner receipt must point into an unedited service export.

    raw_fields maps keys to JSON pointers, avoiding assumptions about a private
    service API's layout. Values must actually occur in that bound export.
    """
    value = read(path)
    require(value["schema"] == "fixed_service_overlap_exit_v1"
            and value["project_id"] == "4dsr" and value["exited_normally"] is True
            and type(value["exit_code"]) is int and value["exit_code"] == 0,
            "Actual normal fixed-service Exit0 required")
    require(value["producer_complete"] == entry(complete)
            and value["all_output_validation"] == entry(validation),
            "Service exit belongs to other output")
    admission = validate_admission(read(bound(value["admission"])))
    for key in ("project_id", "job_id", "unit", "gpu_uuid", "interpreter"):
        require(value[key] == admission[key], "Service/admission identity differs: " + key)
    require(value["producer_command"] == admission["producer_command"], "Different producer command")
    raw = read(bound(value["raw_service_result"]))
    fields = ("project_id", "job_id", "unit", "gpu_uuid", "exit_code")
    for key in fields:
        require(pointer(raw, value["raw_fields"][key]) == value[key], "Raw service field differs: " + key)
    require(pointer(raw, value["raw_fields"]["status"]) == value["raw_success_status"]
            and value["raw_success_status"] in ("succeeded", "completed", "success"),
            "Raw service result is not a successful terminal result")
    ownership = read(local(complete).parent / "ownership.json")
    require(ownership["authorization"] == admission and ownership["unit"] == value["unit"],
            "Producer observed a different actual owner")
    return value


def accept_coarse(complete, service_exit, out):
    import overlap_new
    coarse, _, _ = science_modules()
    complete = local(complete)
    validation = complete.parent / "all_output_validation.json"
    immediate = read(validation)
    require(immediate["status"] == "passed_original_full_output_SHA_and_native_observer_closure"
            and immediate["producer_complete"] == entry(complete)
            and immediate["wrapper"] == entry(CODE / "overlap_new.py")
            and immediate["process_exit_observed"] is False,
            "Original producer-return full-SHA validation absent")
    execution = verify_service_exit(service_exit, complete, validation)
    done = read(complete)
    index = read(bound(done["index"]))
    plan = read(bound(index["registration"]))
    require(read(bound(read(bound(execution["admission"]))["plan"])) == plan,
            "Admitted plan and actual registration differ")
    tick = time.monotonic()
    with attributes(coarse, validate_identity=overlap_new.identity_validator()):
        proof = coarse.validate_result(plan, complete)
    require(proof == immediate["proof"], "Immediate and post-exit full-SHA closure differ")
    n = len(index["raw_moments"])
    output_bytes = sum(bound(row["raw"]).stat().st_size +
                       bound(row["raw"]).with_suffix(".json").stat().st_size
                       for row in index["raw_moments"])
    output_bytes += sum(bound(ref).stat().st_size for ref in index["frames"])
    value = dict(schema=COARSE_SCHEMA, status="accepted_actual_fixed_service_Exit0_and_all_output_SHA",
                 producer_complete=entry(complete), index=done["index"],
                 registration=index["registration"], context=index["context"],
                 service_exit=entry(service_exit), admission=execution["admission"],
                 immediate_validation=entry(validation), original_validator=entry(FP / "full_held_parent_overlap.py"),
                 wrapper=entry(__file__), original_full_SHA_closure=proof,
                 postcheck_cost=dict(seconds=time.monotonic() - tick, frames=300, native_moments=n,
                                     directed_pairs=plan["planned"]["directed_pairs"],
                                     checked_output_files=2 * n + 300, checked_output_bytes=output_bytes),
                 actual_parent_rendering_completed=True, synthetic=False, HR_GT_reads=0, heldout_reads=0)
    write_new(out, value)
    no_torch()
    return value


def validate_coarse_acceptance(path):
    value = read(path)
    require(value["schema"] == COARSE_SCHEMA
            and value["status"] == "accepted_actual_fixed_service_Exit0_and_all_output_SHA"
            and value["wrapper"] == entry(__file__) and value["synthetic"] is False,
            "Foreign/incomplete coarse acceptance")
    for key in ("producer_complete", "index", "registration", "service_exit", "admission",
                "immediate_validation", "original_validator", "wrapper"):
        bound(value[key])
    verify_service_exit(bound(value["service_exit"]), bound(value["producer_complete"]),
                        bound(value["immediate_validation"]))
    index = read(bound(value["index"]))
    require(index["context"] == value["context"] and value["registration"] == index["registration"]
            and read(bound(value["producer_complete"]))["index"] == value["index"],
            "Accepted coarse identities differ")
    proof = read(bound(value["immediate_validation"]))["proof"]
    require(proof == value["original_full_SHA_closure"] and proof["parent_state_unchanged"] is True
            and proof["index"] == value["index"] and proof["model_imported"] is False,
            "Original all-output validation differs")
    return value


def verify_candidate(candidate_path):
    from schedule_new import scalar_overlap, prepared_inputs
    _, scheduler, native = science_modules()
    candidate = read(candidate_path)
    require(candidate["status"] == "pending_root_actual_assignment_storage_and_cost_acceptance",
            "Actual pending assignment candidate required")
    for key in ("source", "pose_table", "coarse_overlap_table", "assignment_statistics",
                "root_acceptance", "prepared_overlap_inputs", "schedule"):
        bound(candidate[key])
    coarse = validate_coarse_acceptance(bound(candidate["root_acceptance"]))
    require(coarse["index"] == candidate["coarse_overlap_table"], "Candidate coarse acceptance differs")
    table = read(bound(candidate["schedule"]))
    manifest, _, _, _ = manifest_metadata(bound(table["identity"]["manifest"]))
    overlap = scalar_overlap(bound(coarse["index"]), manifest)
    overlap.update(accepted=candidate["root_acceptance"], postcheck_cost=coarse["postcheck_cost"])
    prepared = read(bound(candidate["prepared_overlap_inputs"]))
    require(prepared == prepared_inputs(overlap, candidate["pose_table"], coarse),
            "Exact original prepared-input fields and operational binding differ")
    expected_pose = dict(schema="actual_manifest_train_pose_pairs_v1",
                         manifest=table["identity"]["manifest"],
                         pairs=[dict(a=k[0], b=k[1], **v) for k, v in overlap["poses"].items()],
                         baseline_scale=overlap["baseline_scale"])
    require(read(bound(candidate["pose_table"])) == expected_pose, "Actual training pose table differs")
    require(prepared["context"] == table["identity"] == coarse["context"]
            and prepared["coarse_overlap_table"] == coarse["index"]
            and prepared["root_acceptance"] == candidate["root_acceptance"]
            and prepared["frames"] == overlap["frame_entries"]
            and table["assignment_protocol"]["parent_overlap_inputs"] == candidate["prepared_overlap_inputs"],
            "Prepared actual overlap linkage differs")
    regenerated = scheduler.schedule(manifest, table["identity"]["manifest"],
        table["identity"]["parent"], SEED, overlap, candidate["prepared_overlap_inputs"])
    require(regenerated == table, "Assignment was not produced from actual scalar inputs")
    native.validate_schedule(table, manifest, coarse["context"], SEED)
    stats = read(bound(candidate["assignment_statistics"]))
    require(stats["schedule"] == candidate["schedule"]
            and stats["selected_pairs"] == table["assignment_protocol"]["selected_pairs"]
            and stats["pair_frequency"] == table["assignment_protocol"]["pair_frequency"]
            and stats["overlap_bins"] == table["assignment_protocol"]["overlap_bins"]
            and stats["full_legal_observations"] == len(table["record_keys"])
            and stats["calibration_anchors"] == len(table["calibration_triplets"])
            and all(stats[k] is True for k in ("RGB_column_exposure_equal",
                "random_same_RGB_column_exposure_equal", "teacher_weighted_exposure_equal_B0_Bsync"))
            and stats["HR_or_quality_used"] is False,
            "Candidate statistics differ from actual assignment")
    expected = scheduler.assignment_candidate(table, candidate["schedule"],
        candidate["prepared_overlap_inputs"], overlap, candidate["pose_table"], candidate["assignment_statistics"])
    require(candidate == expected, "Original candidate fields differ")
    return candidate


def accept_assignment(candidate_path, out, source_accepted, cost_accepted):
    require(source_accepted and cost_accepted,
            "Claude must explicitly accept actual source registration and storage/cost")
    candidate = verify_candidate(candidate_path)
    value = copy.deepcopy(candidate)
    value.update(status=candidate["required_final_status"], source_registration_accepted=True,
                 storage_and_cost_accepted=True)
    p = write_new(out, value, sorted_keys=True)
    # Original accept_assignment requires immutable accepted inputs.
    p.chmod(0o444)
    scheduler = science_modules()[1]
    scheduler.accept_assignment(entry(p), candidate)
    no_torch()
    return dict(assignment_acceptance=entry(p), status=value["status"])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("overlap", "assignment"))
    p.add_argument("--complete")
    p.add_argument("--service-exit")
    p.add_argument("--candidate")
    p.add_argument("--source-registration-accepted", action="store_true")
    p.add_argument("--storage-and-cost-accepted", action="store_true")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if a.mode == "overlap":
        require(a.complete and a.service_exit, "Complete and actual service exit required")
        value = accept_coarse(a.complete, a.service_exit, a.out)
        value = dict(status=value["status"], acceptance=entry(a.out), postcheck=value["postcheck_cost"])
    else:
        require(a.candidate, "Actual assignment candidate required")
        value = accept_assignment(a.candidate, a.out, a.source_registration_accepted,
                                  a.storage_and_cost_accepted)
    print(json.dumps(value, ensure_ascii=False))


if __name__ == "__main__":
    main()
