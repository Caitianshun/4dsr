"""Focused stdlib CPU checks; no torch, dispatch, model or image payloads."""
from __future__ import annotations

import argparse
import inspect
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sched_common import *
import overlap_new
import schedule_new
import compare_new
import make_rows
import acceptance_new


def rejected(fn):
    try:
        fn()
    except (ValueError, KeyError, FileNotFoundError):
        return True
    raise AssertionError("Negative case was accepted")


def source_proof():
    coarse, scheduler, native = science_modules()
    functions = {
        coarse: ("prepare", "validate_result", "coarse_size", "scaled_K", "legal_rows",
                 "normalize_raw", "sample_bilinear", "directed_overlap", "pair_overlap", "pose_pair",
                 "coarse_camera", "immutable", "observer_rng_start", "observer_rng_restore"),
        scheduler: ("pose_rows", "assignment", "schedule", "assignment_candidate", "accept_assignment"),
        native: ("validate_schedule", "native_effective_state", "state_covariance", "checkpoint_payload"),
    }
    records = []
    for module, names in functions.items():
        for name in names:
            fn = getattr(module, name)
            source = inspect.getsource(fn).encode()
            require(Path(fn.__code__.co_filename).resolve() == Path(module.__file__).resolve(),
                    "Algorithm function does not come from original module")
            records.append(dict(function=module.__name__ + "." + name,
                                source=entry(module.__file__),
                                function_source_sha256=hashlib.sha256(source).hexdigest()))
    return dict(status="passed", direct_original_functions=records,
                copied_algorithm_functions=0,
                AST_extractions=[
                    "original build_plan plan=dict(...) expression",
                    "original actual_overlap scalar frame/pair validation tail",
                    "original validate_identity except exactly two old held-scope statements"],
                allowed_scientific_adaptations=["new scene registration", "cullretrain_lr_v1 parent loader"],
                operational_adaptation="fixed-service admission/Exit0 bridge, explicit dual-parent SR binding",
                depth_renderer="Historical SHA declared; runtime rehash required outside P2 read scope")


def run(out):
    out = output_path(out)
    coarse, scheduler, native = science_modules()
    preserved = [FP / name for name in PINS] + list(REG.glob("*.py"))
    preserved += [HIST_PLAN, HIST_INDEX, HIST_SCHEDULE, HIST_BSYNC]
    for scene in SCENES:
        for form in ("", "_area4"):
            preserved.extend(PREP / (scene + form) / name for name in
                             ("manifest.json", f"manifest_train_ready_seed{SEED}.json"))
    preserved += [REGOUT / f"lr/registry/{s}_area4/seed_{SEED}/config.json"
                  for s in ("sear_steak", "flame_salmon")]
    before = {rel(p): sha(p) for p in preserved}
    syntax = []
    for p in sorted(CODE.glob("*.py")):
        compile(p.read_text(), str(p), "exec")
        syntax.append(entry(p))
    proof = source_proof()
    write_new(out / "algorithm_source_proof.json", proof)
    scenes = []
    for scene in SCENES:
        m, inputs, center, extent = manifest_metadata(PREP / f"{scene}_area4/manifest_train_ready_seed{SEED}.json")
        scenes.append(dict(scene=scene, cameras=list(m["cameras"]), train_cameras=m["splits"]["train"],
                           observations=len(inputs), center=center, extent=extent,
                           coarse=coarse.coarse_size(m["resolutions"]["lr"]),
                           moments=len(inputs), directed_pairs=300 * len(m["splits"]["train"]) * (len(m["splits"]["train"]) - 1)))

    # Test the parent loader's actual metadata rejection gates using an in-memory
    # sidecar fixture. This does not certify a real checkpoint or write a payload.
    done = read(ROOT / "output/dynamic_lr_area4_20261024/wu/Flame07_area4_wu/complete.json")
    import common_new
    pp = read(common_new.snapshot_file(local(done["parent"]) / "config.json"))
    fake_path = OUT / "unit_fixture_not_materialized/checkpoint_fine_14000.pt"
    fake_ref = dict(path=rel(fake_path), sha256="0" * 64)
    done = copy.deepcopy(done)
    done["checkpoint"] = fake_ref
    side = dict(**fake_ref, schema=overlap_new.CULL,
                metadata=dict(state=overlap_new.STATE, arm="wu_author_area4"))
    real_read, real_entry = overlap_new.read, overlap_new.entry

    def check_parent(d, s):
        def fixture_read(p):
            p = local(p)
            if p == fake_path.parent / "complete.json":
                return d
            if p == fake_path.with_suffix(".json"):
                return s
            if p == local(d["parent"]) / "config.json":
                return pp
            return real_read(p)

        def fixture_entry(p):
            p = local(p)
            if p == fake_path:
                return fake_ref
            if p in (fake_path.parent / "complete.json", fake_path.with_suffix(".json"),
                     local(d["parent"]) / "config.json"):
                return dict(path=rel(p), sha256="1" * 64)
            return real_entry(p)

        with attributes(overlap_new, read=fixture_read, entry=fixture_entry):
            return overlap_new.physical_parent(fake_path, pp["data"]["manifest"])

    check_parent(done, side)
    negatives = []
    for name, mutate in (
        ("wrong_arm", lambda d, s: d.update(arm="gateaff")),
        ("affine", lambda d, s: d["args"].update(affine=True)),
        ("cull", lambda d, s: d["args"].update(cull=True)),
        ("prune_none", lambda d, s: d["args"].update(prune_gate="none")),
        ("wrong_seed", lambda d, s: d.update(seed=1)),
        ("partial_fine", lambda d, s: d["state"].update(fine_iteration=100)),
        ("wrong_family", lambda d, s: s.update(schema=overlap_new.NATIVE)),
        ("sidecar_sha", lambda d, s: s.update(sha256="2" * 64)),
        ("sidecar_arm", lambda d, s: s["metadata"].update(arm="gateaff")),
    ):
        d, s = copy.deepcopy(done), copy.deepcopy(side)
        mutate(d, s)
        rejected(lambda: check_parent(d, s))
        negatives.append(name)

    table = read(HIST_SCHEDULE)
    manifest = read(bound(table["identity"]["manifest"]))
    overlap = schedule_new.scalar_overlap(HIST_INDEX, manifest)
    bad = copy.deepcopy(table)
    bad["rows"][0][1] = bad["rows"][0][0]
    rejected(lambda: native.validate_schedule(bad, manifest, table["identity"], SEED))
    negatives.append("duplicate_camera_schedule")
    bad = copy.deepcopy(table)
    bad["identity"]["parent"]["sha256"] = "3" * 64
    rejected(lambda: native.validate_schedule(bad, manifest, table["identity"], SEED))
    negatives.append("wrong_schedule_parent")
    index = read(HIST_INDEX)
    bad_index = copy.deepcopy(index)
    bad_index["frames"] = bad_index["frames"][:-1]
    sr_read = schedule_new.read
    with attributes(schedule_new, read=lambda p: bad_index if local(p) == HIST_INDEX else sr_read(p)):
        rejected(lambda: schedule_new.scalar_overlap(HIST_INDEX, manifest))
    negatives.append("missing_overlap_frame")
    first = bound(index["frames"][0])
    frame = read(first)
    frame["pairs"][0]["a_to_b"]["ratio"] = 2.
    old_read = scheduler.read
    with attributes(scheduler, read=lambda p: frame if local(p) == first else old_read(p)):
        rejected(lambda: schedule_new.scalar_overlap(HIST_INDEX, manifest))
    negatives.append("overlap_ratio_not_from_counts")
    # Service receipts stay in memory in this unit test: no admission/Exit0/
    # acceptance file is fabricated or published as a real operational result.
    raw_service = dict(project_id="4dsr", job_id="UNIT_TEST_NO_SERVICE",
                       unit="UNIT_TEST_NO_SERVICE.service", gpu_uuid="GPU-UNIT_TEST", status="running")
    admission = dict(schema="fixed_service_overlap_admission_v1", admitted=True,
                     interpreter=TRAINPY, env_id="mml_bw", **{k: raw_service[k] for k in
                         ("project_id", "job_id", "unit", "gpu_uuid")},
                     admission_evidence={"path": "unused_in_memory", "sha256": "0" * 64},
                     raw_fields={k: "/" + k for k in raw_service}, raw_admitted_status="running")
    with attributes(acceptance_new, read=lambda _: raw_service,
                    bound=lambda _: OUT / "unit_fixture_not_materialized/service.json"):
        acceptance_new.validate_admission(admission)
        for key, bad_value in (("job_id", "different"), ("unit", "other.service"),
                               ("gpu_uuid", "GPU-other"), ("raw_admitted_status", "failed")):
            bad_admission = dict(admission, **{key: bad_value})
            rejected(lambda: acceptance_new.validate_admission(bad_admission))
            negatives.append("service_evidence_" + key)
    rejected(lambda: acceptance_new.accept_assignment("unused", "unused", False, False))
    negatives.append("assignment_without_explicit_owner_acceptance")
    summary = compare_new.compare_schedules(table, table, overlap, overlap)
    require(summary["row_agreement"] == 1 and all(x["delta"] == [0, 0, 0] for x in summary["per_camera_exposure"])
            and summary["paired_position_triplet_cost_delta"]["max"] == 0, "Comparison identity case failed")
    changed = copy.deepcopy(table)
    changed["rows"][0], changed["rows"][1] = changed["rows"][1], changed["rows"][0]
    if table["rows"][0] != table["rows"][1]:
        changed_summary = compare_new.compare_schedules(table, changed, overlap, overlap)
        require(changed_summary["exact_ordered_rows"] == 5998
                and all(x["delta"] == [0, 0, 0] for x in changed_summary["per_camera_exposure"]),
                "Comparison missed a row permutation with invariant exposure")
    row_validation = make_rows.validate_rows(make_rows.build()[0])
    after = {rel(p): sha(p) for p in preserved}
    require(before == after, "An existing input changed during checks")
    no_torch()
    result = dict(status="passed", syntax_files=len(syntax), parent_metadata_negative_cases=9,
                  negative_cases=negatives, scene_geometry=scenes, row_validation=row_validation,
                  comparison_identity_and_permutation_passed=True,
                  preserved_input_files=len(before), preserved_input_sha256=before,
                  torch_imported=False, GPU_calls=0, subprocesses_started=0,
                  real_parent_tensor_restore_tested=False, syntax_sources=syntax)
    write_new(out / "checks.json", result)
    return {k: v for k, v in result.items() if k not in ("preserved_input_sha256", "syntax_sources")}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    try:
        value = run(a.out)
    except BaseException as e:
        import traceback
        write_new(local(a.out) / "failure.json", dict(error=repr(e), traceback=traceback.format_exc()))
        raise
    print(json.dumps(value, ensure_ascii=False))


if __name__ == "__main__":
    main()
