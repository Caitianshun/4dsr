"""Preserve every original physlr.sr_register check; redirect new paths only."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_new import *
import physlr
from materialize_bsync_new import teacher_files, physical_parent


def register(scene, *, clean, teacher_index, parent, out_tag="areaft_teacher"):
    require(scene in NEW, "Only the two new scenes may be registered here")
    clean = local(clean)
    base = read(clean)
    require(base["status"] == "registered_ready_full_native_SR_refinement" and not base["missing"],
            "Pending clean Bsync plan refuses final registration")
    require(base["scene"] == scene and base["seed"] == SEED and base["method"] == "Bsync",
            "Clean plan scene/seed/method mismatch")
    root = PREP / f"{scene}_area4"
    mp = root / f"manifest_train_ready_seed{SEED}.json"
    files = teacher_files(local(teacher_index), read(mp), mp)
    parent_value, _ = physical_parent(scene, parent, base["data"])
    require(base["parent"]["checkpoint"] == parent_value["checkpoint"]
            and base["full_native_context"]["parent"] == parent_value["checkpoint"]
            and base["full_native_context"]["manifest"] == entry(mp),
            "Clean plan belongs to another physical parent/manifest")
    expected_index = root / f"teacher_index_sr_swinir_x4_areaft_seed{SEED}.json"
    old_read = physlr.RS.read
    old_entry = physlr.RS.entry
    captured = {}

    def redirected_read(path):
        # Only the index location is redirected. Index contents, key coverage,
        # manifest SHA, all LR hashes, image bytes and all original checks remain.
        return old_read(local(teacher_index) if local(path) == expected_index else path)

    def redirected_entry(path):
        # entry() hashes its own file; redirect both the index read and its SHA
        # binding, without fabricating an index in the immutable dataset root.
        return old_entry(local(teacher_index) if local(path) == expected_index else path)

    def new_write(path, value):
        # The original registrar leaves training_files.teacher from its base.
        # Our clean base already contains this SAME areaft teacher file list.
        require(value["training_files"]["teacher"] == files, "Clean teacher inventory differs")
        value["dependencies"]["teacher"] = entry(teacher_index)
        value["newscene_wrapper"] = dict(source=entry(__file__), teacher_index=entry(teacher_index),
                                         original_registrar=entry(PHYS / "physlr.py"),
                                         original_checks_preserved=True, GPU_dispatch_allowed=False)
        write_new(path, value)
        captured["value"] = value

    held = dict(physlr.HELD_BSYNC, **{s: OUT / f"clean_bsync/{s}/seed{SEED}/Bsync/scientific_plan.json" for s in NEW})
    held[scene] = clean
    with attributes(physlr, HELD_BSYNC=held, LR_NS=OUT / "lr", OUT_NS=OUT / "sr",
                    SHORT=dict(physlr.SHORT, **NEW),
                    clean_bsync=lambda s, seed: held[s] if seed == SEED else (_ for _ in ()).throw(ValueError("Seed differs"))):
        with attributes(physlr.RS, read=redirected_read, entry=redirected_entry, write_new=new_write):
            result = physlr.sr_register(scene, "area4", SEED, "sr_swinir_x4_areaft", rel(parent), out_tag)
    require(captured["value"]["dependencies"]["schedule"] == base["dependencies"]["schedule"]
            and captured["value"]["schedule"] == base["schedule"]
            and captured["value"]["configuration"] == base["configuration"],
            "Scientific schedule or configuration changed")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene", choices=tuple(NEW), required=True)
    p.add_argument("--clean")
    p.add_argument("--teacher-index")
    p.add_argument("--parent")
    p.add_argument("--out-tag", default="areaft_teacher")
    a = p.parse_args()
    scene = a.scene
    clean = a.clean or OUT / f"clean_bsync/{scene}/seed{SEED}/Bsync/scientific_plan.json"
    teacher = a.teacher_index or OUT / f"teacher_cache/{scene}_area4/teacher_index_sr_swinir_x4_areaft_seed{SEED}.json"
    parent = a.parent or OUT / f"lr/runs/{NEW[scene]}07_area4_gateaff/checkpoint_fine_14000.pt"
    print(__import__("json").dumps(register(scene, clean=clean, teacher_index=teacher, parent=parent, out_tag=a.out_tag)))


if __name__ == "__main__":
    main()
