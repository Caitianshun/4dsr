"""Bridge author-overlap identity to gateaff SR without rewriting schedule rows."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sched_common import *
import materialize_bsync_new as prior
import sr_register_new as prior_sr
import overlap_new
import acceptance_new


def require_schedule(scene, training_parent, overlap_parent, schedule, assignment, *, verify_lr=False):
    require(scene in ("sear_steak", "flame_salmon"), "New scene required")
    candidate_path = local(schedule).parent / "assignment_acceptance_candidate.json"
    candidate = acceptance_new.verify_candidate(candidate_path)
    _, scheduler, _ = science_modules()
    scheduler.accept_assignment(entry(assignment), candidate)

    def overlap_parent_adapter(actual_scene, parent, data):
        value, pp = overlap_new.physical_parent(parent, data["manifest"])
        require(pp["scene"] == actual_scene, "Overlap parent scene differs")
        return value, pp

    # All original require-schedule checks run unchanged. Only its parent
    # loader is now the separately checked author-overlap parent loader.
    with attributes(prior, physical_parent=overlap_parent_adapter):
        table = prior.require_schedule(scene, overlap_parent, schedule, assignment,
                                       verify_lr_bytes=verify_lr)
    mp = PREP / f"{scene}_area4/manifest_train_ready_seed{SEED}.json"
    m, data, _ = prior.protocol.validate_manifest(mp, formation="area4", verify_lr_bytes=verify_lr)
    train_value, _ = prior.physical_parent(scene, training_parent, data)
    require(table["identity"]["parent"] == entry(overlap_parent)
            and candidate["schedule"] == entry(schedule)
            and candidate["full_native_context"] == dict(table["identity"], schedule=entry(schedule)),
            "Schedule/author-overlap identity differs")
    return dict(schema="separate_overlap_and_training_parent_binding_v1",
                scene=scene, seed=SEED, manifest=entry(mp),
                overlap_parent=entry(overlap_parent),
                training_parent=train_value["checkpoint"], schedule=entry(schedule),
                assignment_acceptance=entry(assignment), source=entry(__file__),
                schedule_relabelled=False, rows_modified=False,
                adaptation="Area4 wu_author parent supplies overlap; shared gateaff parent initializes SR")


def materialize(scene, training_parent, overlap_parent, teacher, schedule, selection, assignment):
    binding = require_schedule(scene, training_parent, overlap_parent, schedule, assignment, verify_lr=True)
    mp = PREP / f"{scene}_area4/manifest_train_ready_seed{SEED}.json"
    manifest, data, lr_files = prior.protocol.validate_manifest(mp, formation="area4", verify_lr_bytes=True)
    parent_value, lr_plan = prior.physical_parent(scene, training_parent, data)
    configuration, _ = prior.protocol.configuration(scene)
    require(lr_plan["configuration"] == configuration, "Training parent configuration differs")
    teachers = prior.teacher_files(local(teacher), manifest, mp)
    prior.native.validate_selection(read(selection), "Bsync")
    sources, external = prior.declared_sources()
    for name in external:
        require(sha(name) == sources["project"][name], "External scientific source differs")
    for name in ("sched_common.py", "bsync_new.py", "overlap_new.py", "schedule_new.py", "acceptance_new.py"):
        sources["project"][rel(CODE / name)] = sha(CODE / name)
    value = prior.assemble(scene_manifest=manifest, data=data, configuration=configuration,
                           parent=parent_value, dependencies=dict(selection=entry(selection),
                               teacher=entry(teacher), schedule=entry(schedule)),
                           lr_files=lr_files, teacher_files=teachers, missing=[],
                           source_files=sources)
    value["schedule_parent_binding"] = binding
    value["materialization"] = dict(wrapper=entry(__file__),
        constructor=entry(FP / "full_refine_registered_v2.py"),
        previous_wrapper=entry(REG / "materialize_bsync_new.py"),
        parent_checkpoint_family="cullretrain_lr_v1_gateaff",
        overlap_checkpoint_family="cullretrain_lr_v1_wu_author_area4",
        parent_validated_from_SHA_bound_sidecar=True, torch_checkpoint_loaded=False,
        assignment_acceptance=entry(assignment), GPU_dispatch_allowed=False)
    no_torch()
    return value


def register(scene, clean, teacher, parent):
    """Use the complete previous registrar, redirect only the new SR output."""
    base = read(clean)
    binding = base["schedule_parent_binding"]
    actual = require_schedule(scene, parent, bound(binding["overlap_parent"]),
                               bound(binding["schedule"]), bound(binding["assignment_acceptance"]),
                               verify_lr=True)
    require(actual == binding and base["parent"]["checkpoint"] == entry(parent),
            "Clean binding/training parent differs")
    import physlr
    # Previous OUT controls both its LR lookup and SR output. Keep LR lookup
    # pinned to the already accepted registry while changing only SR writes.
    def original_lr_registry(s, formation, seed):
        require(s == scene and formation == "area4" and seed == SEED, "Foreign LR registry")
        return REGOUT / f"lr/registry/{s}_area4/seed_{seed}"

    with attributes(prior_sr, OUT=OUT, write_new=write_new):
        with attributes(physlr, plan_dir=original_lr_registry):
            result = prior_sr.register(scene, clean=clean, teacher_index=teacher, parent=parent)
    no_torch()
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("require-schedule", "materialize", "sr-register"))
    p.add_argument("--scene", choices=("sear_steak", "flame_salmon"), required=True)
    p.add_argument("--parent", required=True, help="gateaff shared training parent")
    p.add_argument("--overlap-parent", help="wu_author_area4 overlap parent")
    p.add_argument("--schedule")
    p.add_argument("--assignment-acceptance")
    p.add_argument("--teacher")
    p.add_argument("--selection")
    p.add_argument("--clean")
    p.add_argument("--out")
    a = p.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if a.mode == "sr-register":
        require(a.clean and a.teacher, "Clean Bsync and teacher required")
        value = register(a.scene, a.clean, a.teacher, a.parent)
    else:
        require(a.overlap_parent and a.schedule and a.assignment_acceptance and a.out,
                "Author overlap parent, schedule, assignment acceptance and fresh out required")
        if a.mode == "require-schedule":
            value = require_schedule(a.scene, a.parent, a.overlap_parent, a.schedule,
                                     a.assignment_acceptance)
            value = dict(status="actual_schedule_and_distinct_parent_binding_validated", binding=value,
                         source=entry(__file__), training_submitted=False)
        else:
            require(a.teacher and a.selection, "Actual teacher and frozen selection required")
            value = materialize(a.scene, a.parent, a.overlap_parent, a.teacher, a.schedule,
                                a.selection, a.assignment_acceptance)
        write_new(a.out, value)
        value = dict(output=entry(a.out), status=value["status"])
    print(json.dumps(value, ensure_ascii=False))


if __name__ == "__main__":
    main()
