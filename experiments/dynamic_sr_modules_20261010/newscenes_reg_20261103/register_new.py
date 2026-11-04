"""Wrap unchanged physlr.register/lr_tasks in the new namespace.

draft writes metadata-only review plans under lr/drafts, never registry.
register runs ALL physlr checks, including every LR PNG SHA, then writes registry.
This restricted worker can run draft/tasks; Claude runs register with pixel access.
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_new import *
import protocol_new as protocol
import physlr

LR_NS = OUT / "lr"
TEMPLATES = OUT / "clean_templates"
LR_SCRIPT = "cullretrain.py"  # pipeline.py resolves script relative to its own HERE


@contextlib.contextmanager
def configured(namespace=LR_NS, templates=TEMPLATES):
    with attributes(physlr, LR_NS=local(namespace), SHORT=dict(physlr.SHORT, **NEW)):
        with attributes(physlr.cp, ORIGINAL_PREFIX=local(templates), write_new=write_new):
            yield physlr


def draft(scene, namespace=LR_NS, templates=TEMPLATES):
    """Metadata constructor, explicitly NOT a successful physlr.register call.

    Rebuild data from the real area4 initialization/manifest. No substituted SHA
    function, fake image file or weakened physlr validation is used.
    """
    original_path = local(templates) / scene / f"seed_{SEED}/config.json"
    original = read(original_path)
    m, actual, inputs = protocol.validate_manifest(
        PREP / f"{scene}_area4/manifest_train_ready_seed{SEED}.json", formation="area4")
    require(m["splits"]["train"] == original["data"]["train_cameras"], "Train split changed")
    require(max(abs(a - b) for a, b in zip(actual["train_camera_center"],
                                         original["data"]["train_camera_center"])) <= 1e-9
            and abs(actual["cameras_extent"] - original["data"]["cameras_extent"]) <= 1e-9,
            "Derived camera geometry changed")
    plan = copy.deepcopy(original)
    # Same data-update fields as physlr.register; keep the bicubic protocol
    # provenance, exactly as the historical registrar does.
    data = copy.deepcopy(original["data"])
    for key in ("manifest", "original_manifest", "initialization_receipt", "initial_points",
                "train_cameras", "training_observations", "training_LR_sources_sha256"):
        data[key] = actual[key]
    data["training_LR_bytes_SHA_verified"] = 0
    degradation = f"physical LR formation area4: {physlr.FORMATIONS['area4']['text']}"
    data["domain_protocol"] = dict(data["domain_protocol"], seed=SEED, degradation=degradation)
    plan["status"] = "draft_area4_LR_parent_metadata_only_physlr_register_LR_bytes_pending"
    plan["data"] = data
    plan["author_adaptations"] = original["author_adaptations"] + [
        "Physical-LR gate (2026-10-16): training LR re-formed as area4 from the registered HR; configuration unchanged."]
    plan["physical_lr"] = dict(
        formation="area4", derivation=entry(PREP / f"{scene}_area4/derivation_receipt.json"),
        original_plan=entry(original_path), registry_source=entry(PHYS / "physlr.py"))
    plan["draft_validation"] = dict(physlr_register_executed=False, LR_bytes_verified=False,
                                    inputs_declared=len(inputs), GPU_dispatch_allowed=False,
                                    wrapper=entry(__file__))
    path = local(namespace) / "drafts" / f"{scene}_area4/seed_{SEED}/config.json"
    write_new(path, plan)
    return path


def register(scene, namespace=LR_NS, templates=TEMPLATES):
    require(scene in (*NEW, "flame_steak"), "Scene outside task")
    # Protocol metadata checks supplement, rather than replace, the original
    # registrar's split, initialization, geometry and complete pixel SHA checks.
    protocol.validate_manifest(PREP / f"{scene}_area4/manifest_train_ready_seed{SEED}.json",
                               formation="area4", verify_lr_bytes=True)
    with configured(namespace, templates):
        result = physlr.register(scene, "area4", SEED)
        path = physlr.plan_dir(scene, "area4", SEED) / "config.json"
    value = read(path)
    require(value["data"]["training_LR_bytes_SHA_verified"] ==
            value["data"]["training_observations"], "LR verification count mismatch")
    receipt = dict(status="all_original_physlr_register_checks_passed", registry=entry(path),
                   original_registrar=entry(PHYS / "physlr.py"), wrapper=entry(__file__),
                   training_submitted=False, GPU_calls=0)
    write_new(local(namespace) / "validation" / f"{scene}_area4_seed{SEED}.json", receipt)
    return result


def tasks(scenes=tuple(NEW), namespace=LR_NS, templates=TEMPLATES):
    rows = []
    with configured(namespace, templates):
        for scene in scenes:
            for row in physlr.lr_tasks(scene, "area4", SEED):
                row["script"] = LR_SCRIPT
                rows.append(row)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("draft", "register", "tasks"))
    p.add_argument("--scene", choices=(*NEW, "flame_steak"), action="append")
    p.add_argument("--namespace", default=rel(LR_NS))
    p.add_argument("--templates", default=rel(TEMPLATES))
    p.add_argument("--out", default=rel(OUT / "tasks/lr_tasks.json"))
    a = p.parse_args()
    scenes = a.scene or tuple(NEW)
    if a.mode == "tasks":
        value = tasks(scenes, a.namespace, a.templates)
        write_new(a.out, value)
        print(__import__("json").dumps(dict(tasks=len(value), file=rel(a.out))))
    else:
        value = []
        for scene in scenes:
            if a.mode == "draft":
                value.append(dict(scene=scene, draft=entry(draft(scene, a.namespace, a.templates))))
            else:
                value.append(register(scene, a.namespace, a.templates))
        print(__import__("json").dumps(value, ensure_ascii=False))


if __name__ == "__main__":
    main()
