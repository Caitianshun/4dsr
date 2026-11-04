"""New N3DV scene protocol and clean LR templates; no torch or training.

Default: metadata/initialization checks only, with LR byte verification explicitly
pending. --verify-lr-bytes additionally opens every legal training LR PNG.
Existing templates are never upgraded in place; use --out-root for a new revision.
"""
from __future__ import annotations

import argparse
import ast
import copy
import math
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_new import *
import full_native_protocol as old
import full_prefix_registered as prefix

_HISTORICAL_DESCRIPTOR = old.descriptor
READINESS = old.READINESS
SCENES = tuple(NEW)
SEEDS = (SEED,)
SCHEMA = old.SCHEMA
CONFIG_FILES = {"sear_steak": "sear_steak.py", "flame_salmon": "flame_salmon_1.py"}
CONFIG_SHA = {
    "arguments/dynerf/sear_steak.py": "9b218d6925d5ba6fc41ed999553a6b298b3797ad820eb51ba81f16386cf2fffa",
    "arguments/dynerf/flame_salmon_1.py": "a73f7405eeceff2503b51d268a22e2ffd04abed947a1002c1a8a880e77355363",
}


def historical_sum(values):
    """Preserve Python 3.10 ordered float addition on Python 3.12+.

    The registered native metadata runtime used 3.10. Builtin sum changed its
    float algorithm in 3.12; an explicit loop reproduces the original formula
    from each scene's own camera values, without reading reference geometry.
    """
    total = 0.0
    for value in values:
        total += value
    return total


def descriptor(scene):
    if scene == "flame_steak":
        return _HISTORICAL_DESCRIPTOR(scene)
    require(scene in NEW, "Unregistered new scene")
    return dict(schema=SCHEMA, scene=scene, role="held_confirmation_20261102",
                native_LR=[336, 252], native_HR=[1344, 1008], frames=[0, 299],
                frame_count=300, time="original_frame_index/300", heldout="cam00",
                training="all actual supplied cameras except cam00, including cam01",
                configuration="Wu843d5_original_N3DV_scene_configuration",
                configuration_paths=["arguments/__init__.py", "arguments/dynerf/default.py",
                                     "arguments/dynerf/" + CONFIG_FILES[scene]],
                source=old.KNOWN_SOURCE["n3dv"], scale=4, HR_training_reads=False,
                teacher_used=False,
                initializer="existing registered full train-only LR sparse initializer; not official supplied COLMAP",
                unchanged_author_MeetRoom_reproduction=None, complete_author_paper_benchmark=False)


def configuration(scene, readiness=READINESS):
    d = descriptor(scene)
    evidence = read(readiness)["original_Wu"]
    require(evidence["git_commit"] == PIN, "Author commit differs")
    sources = {}
    for row in evidence["sources"]:
        require(row["git_commit"] == PIN, "Unpinned author evidence")
        sources[row["git_path"]] = bound(row)
    require(sha(FP / "full_prefix.py") == old.LEGACY_PREFIX_SHA, "Accepted legacy source changed")
    if scene in NEW:
        receipt = read(RUNTIME / "RUNTIME_COPY.json")
        require(receipt["upstream_commit"] == PIN, "Runtime copy commit differs")
        # The recorded tracked diff has no configuration edits. Bind the two
        # added overrides to the copy receipt and independently fixed SHA values.
        require("diff --git a/arguments/" not in receipt["upstream_tracked_diff_of_copied_python"],
                "Runtime receipt declares modified author arguments")
        for name in d["configuration_paths"]:
            path = RUNTIME / name
            row = next(r for r in receipt["files"] if r["copy"] == rel(path))
            require(sha(path) == row["sha256"], "Runtime source changed: " + name)
            if name in sources:
                require(sha(path) == sha(sources[name]), "Configuration differs from pinned snapshot")
            else:
                require(sha(path) == CONFIG_SHA[name], "New scene override SHA differs")
                sources[name] = path
    result = {}
    for cls in ast.parse(sources["arguments/__init__.py"].read_text()).body:
        if not isinstance(cls, ast.ClassDef) or cls.name not in (
                "ModelParams", "ModelHiddenParams", "OptimizationParams", "PipelineParams"):
            continue
        init = next(f for f in cls.body if isinstance(f, ast.FunctionDef) and f.name == "__init__")
        result[cls.name] = {
            n.targets[0].attr: old.pure(n.value) for n in init.body
            if isinstance(n, ast.Assign) and len(n.targets) == 1
            and isinstance(n.targets[0], ast.Attribute)
            and isinstance(n.targets[0].value, ast.Name) and n.targets[0].value.id == "self"}
    for name in d["configuration_paths"][1:]:
        for node in ast.parse(sources[name].read_text()).body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                    and isinstance(node.targets[0], ast.Name) and node.targets[0].id in result:
                result[node.targets[0].id].update(old.pure(node.value))
    expected_batch = 4 if scene == "flame_salmon" else 2
    opt = result["OptimizationParams"]
    require((opt["coarse_iterations"], opt["iterations"], opt["batch_size"], opt["dataloader"],
             opt["custom_sampler"], opt["zerostamp_init"]) ==
            (3000, 14000, expected_batch, True, None, False), "Author recipe changed")
    require((opt["densify_until_iter"], opt["opacity_reset_interval"], opt["densify_from_iter"],
             opt["densification_interval"], opt["pruning_from_iter"], opt["pruning_interval"]) ==
            (10000, 60000, 500, 100, 500, 100), "Author topology changed")
    if scene == "flame_steak":
        require(result == evidence["fully_merged_N3DV_configurations"][scene], "Historical merge differs")
    return result, sources


def actual_cameras(scene):
    return [f"cam{i:02d}" for i in range(21)
            if not (scene == "flame_salmon" and i in (3, 17))]


def validate_grid(manifest, scene):
    with attributes(old, descriptor=descriptor):
        d = old.validate_grid(manifest, scene)
    require(set(manifest["cameras"]) == set(actual_cameras(scene)), "Actual camera inventory differs")
    return d


def validate_manifest(path, seed=SEED, readiness=READINESS, *, verify_lr_bytes=False,
                      formation=None):
    require(seed == SEED, "Only seed20261007 authorized")
    path = local(path)
    m = read(path)
    scene = m["scene"]
    d = validate_grid(m, scene)
    directory = PREP / (scene if formation is None else f"{scene}_{formation}")
    require(path == directory / f"manifest_train_ready_seed{seed}.json", "Unexpected train-ready path")
    require(formation in (None, "area4"), "Only bicubic metadata or area4 accepted")
    require(m["schema"] == "n3dv_dynamic_sr_pilot_v1" and m.get("full_time_decode_verified")
            and m["frame_indices"] == list(range(300)), "Verified full300 required")
    require(m["role"] == d["role"] and m["source"] == d["source"]
            and m.get("official_split") is True and m.get("additional_cam01_dev_holdout") is False,
            "Official source/role/split differs")
    train = m["splits"]["train"]
    require(train == actual_cameras(scene)[1:] and m["splits"]["test"] == ["cam00"]
            and not m["splits"]["dev"], "All actual training cameras required")
    rows = [r for r in m["observations"] if r["split"] == "train"]
    keys = {(r["camera_id"], int(r["frame_index"])) for r in rows}
    require(len(rows) == len(keys) == len(train) * 300 and
            keys == {(c, f) for c in train for f in range(300)}, "Incomplete/duplicate train observations")
    require(all(abs(float(r["time"]) - int(r["frame_index"]) / 300) < 1e-7 for r in rows),
            "Author frame/300 time required")
    init, reg = m["initialization"], m["independent_seed_registration"]
    require(init["seed"] == reg["seed"] == seed and init["train_only_lr"] is True
            and init["official_COLMAP_initialization"] is False
            and reg["LR_model_prefix_trained"] is False, "Pure LR initialization required")
    require(init["npz_path"] == f"initialization_seed_{seed}/points_lr.npz", "Foreign initialization")
    original = bound(m["original_full_manifest"])
    original_value = read(original)
    for key in ("scene", "source", "scale", "role", "frame_indices", "resolutions",
                "cameras", "splits", "observations"):
        require(m[key] == original_value[key], "Derived manifest differs: " + key)
    if formation:
        require(m["physical_lr_derivation"]["formation"] == formation, "Formation mismatch")
        bound(m["physical_lr_derivation"]["original_manifest"])
    receipt_path = directory / f"initialization_seed_{seed}/complete.json"
    receipt = read(receipt_path)
    require(receipt["status"] == "completed_train_ready_manifest_and_initial_points_CPU_acceptance"
            and receipt["derived_manifest"] == entry(path)
            and receipt["identity"]["original_full_manifest"] == entry(original)
            and receipt["identity"]["seed"] == seed, "Initializer receipt differs")
    require(receipt["CUDA_calls"] == 0 and receipt["model_prefix_trained"] is False
            and receipt["load_training_CUDA_camera_creation_executed"] is False,
            "Initializer information boundary differs")
    require(m["full_initialization_source"] == receipt["identity"]["source"], "Initializer source mismatch")
    for key in ("source", "legacy_triangulation", "data_loader"):
        bound(receipt["identity"][key])
    for artifact in receipt["artifacts"]:
        bound(artifact)
    points = directory / init["npz_path"]
    require(sha(points) == init["npz_sha256"], "Initial point SHA mismatch")
    inputs = []
    for row in rows:
        require(Path(row["lr_path"]) == Path("lr") / row["camera_id"] /
                f"{int(row['frame_index']):04d}.png" and row["camera_id"] != "cam00",
                "Illegal LR path")
        p = directory / row["lr_path"]
        if verify_lr_bytes:
            p = old.legal_LR_path(directory, row)
            require(p.is_file() and sha(p) == row["lr_sha256"], "LR byte SHA mismatch")
        inputs.append(dict(camera=row["camera_id"], frame=int(row["frame_index"]),
                           path=rel(p), sha256=row["lr_sha256"]))
    centers = [[float(m["cameras"][c]["c2w"][i][3]) for i in range(3)] for c in train]
    center = [historical_sum(c[i] for c in centers) / len(centers) for i in range(3)]
    extent = 1.1 * max(math.sqrt(historical_sum((c[i] - center[i]) ** 2 for i in range(3)))
                       for c in centers)
    require(math.isfinite(extent) and extent > 0, "Degenerate train-only extent")
    cfg, sources = configuration(scene, readiness)
    domain = dict(**d, seed=seed,
                  configuration_sources={name: entry(sources[name]) for name in d["configuration_paths"]},
                  configuration_sha256=digest(cfg), readiness=entry(readiness),
                  protocol_source=entry(__file__), legacy_accepted_prefix_source=entry(FP / "full_prefix.py"))
    data = dict(manifest=entry(path), original_manifest=entry(original),
                initialization_receipt=entry(receipt_path), initial_points=entry(points),
                train_cameras=train, training_observations=len(rows),
                training_LR_sources_sha256=digest(inputs),
                training_LR_bytes_SHA_verified=len(inputs) if verify_lr_bytes else 0,
                train_camera_center=center, cameras_extent=extent, extent_cam01_included=True,
                heldout_camera="cam00", frames=[0, 299], time="frame_index/300",
                resolution_LR=m["resolutions"]["lr"], resolution_HR=m["resolutions"]["hr"],
                domain_protocol=domain)
    return m, data, inputs


def build_template(scene, seed=SEED, *, verify_lr_bytes=False):
    manifest = PREP / scene / f"manifest_train_ready_seed{seed}.json"
    validator = lambda p, s, r=READINESS: validate_manifest(p, s, r, verify_lr_bytes=verify_lr_bytes)
    # Execute the unchanged historical constructor with only its protocol adapter
    # replaced. Its author-runtime equality checks and planned_counts still run.
    with attributes(prefix, validate_manifest=validator, author_configuration=configuration):
        plan, _ = prefix.build_plan(manifest, seed, READINESS, upstream=RUNTIME)
    plan.pop("CPU_seconds", None)
    plan["status"] = ("clean_LR_template_only_all_training_LR_bytes_verified_no_training"
                      if verify_lr_bytes else "clean_LR_template_only_metadata_checked_LR_bytes_pending_no_training")
    plan["project_sources"].update({rel(__file__): sha(__file__), rel(CODE / "common_new.py"): sha(CODE / "common_new.py")})
    plan["template_validation"] = dict(
        metadata_and_initialization_verified=True, LR_bytes_verified=verify_lr_bytes,
        training_submitted=False, model_initialized=False,
        geometry_policy="actual train cameras, explicit historical Python3.10 ordered floating addition",
        author_binding="pinned readiness SHA + RUNTIME_COPY.json SHA; no network",
        runtime_copy_receipt=entry(RUNTIME / "RUNTIME_COPY.json"))
    return plan


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene", choices=(*NEW, "flame_steak"), action="append")
    p.add_argument("--out-root", default=rel(OUT / "clean_templates"))
    p.add_argument("--verify-lr-bytes", action="store_true")
    a = p.parse_args()
    result = []
    for scene in a.scene or NEW:
        plan = build_template(scene, verify_lr_bytes=a.verify_lr_bytes)
        path = local(a.out_root) / scene / f"seed_{SEED}/config.json"
        write_new(path, plan)
        result.append(dict(scene=scene, template=entry(path), status=plan["status"],
                           batch=plan["planned"]["batch_size"],
                           center=plan["data"]["train_camera_center"], extent=plan["data"]["cameras_extent"]))
    print(__import__("json").dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
