#!/usr/bin/env python3
"""Create new native-grid consumer registrations without training or GPU imports.

The configs retain source-backed fields needed by cullretrain.py. They are not
accepted historical prefixes/endpoints and do not satisfy the older strict
full_native_protocol validator, whose grids and data roots are fixed.
The synchronized schedule uses balanced seeded legal camera assignments; it
does not claim the historical pose + frozen-parent-overlap selection policy.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import struct
import sys

sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
REGISTRY = ROOT / "output/dynamic_sr_native_20261010/registry"
PREPARED = ROOT / "data/dynamic_sr/native_20261010/prepared"
RUNTIME = ROOT / "third_party/4dgs_843d5ac_runtime"
OLD = ROOT / "experiments/dynamic_sr_multiview_footprint_20261007"
CULL = ROOT / "experiments/dynamic_sr_cull_retrain_20261009"
LEGACY = ROOT / "experiments/dynamic_sr_20260918"
SCENES = ("cook_spinach", "cut_roasted_beef")
SEEDS = (20261007, 20261008)
AUTHOR_COMMIT = "843d5ac636c37e4b611242287754f3d4ed150144"
LR_TEMPLATE = (
    ROOT / "output/dynamic_sr_multiview_footprint_20261007/full_LR_prefixes"
    "/cook_spinach/seed_20261007/config.json"
)
SR_TEMPLATE = (
    ROOT / "output/dynamic_sr_multiview_footprint_20261007/full_SR_refinement"
    "/Cook_both_seeds_actual_revisionB_1791421128150050152/cook_spinach"
    "/seed_20261007/Bsync/config.json"
)
FROZEN = dict(
    generator_sha256="d7be23d15473082c57e1e4091bfea9eaafad0de97ef7379db39308ae14c6cbcc",
    network_sha256="1d650d2c1c4519d95db771863691c6b239a0822222f287f6a8342a9e0eccbc6e",
    checkpoint_sha256="4e78e33f22c1aa8a773db0cf4a7381bae97c2362c717f155439ebc690cbd9215",
    scale=4, window=8, tile=0, overlap=32, precision="float32",
    padding="official_mirror_concat_next_window",
)
DEFINITION = dict(
    schedule="rows", lr_weights=[1.0, 0.0, 0.0],
    sr_weights=[0.0, 0.05, 0.05], X=False, E=False,
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()).hexdigest()


def local(path):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def inside(path, namespace):
    # Check spelling before resolve so symlinks cannot disappear from the check.
    spelling = Path(path)
    spelling = spelling if spelling.is_absolute() else ROOT / spelling
    require(not any(p.is_symlink() for p in [spelling, *spelling.parents]),
            "Symlink refused: " + str(spelling))
    path = spelling.resolve()
    require(path.is_relative_to(namespace.resolve()) and path != namespace.resolve(),
            "Path outside the new native namespace: " + str(path))
    return path


def entry(path):
    path = local(path)
    require(path.is_relative_to(ROOT), "Registration inputs must be under project root.")
    return dict(path=str(path.relative_to(ROOT)), sha256=sha(path))


def bound(reference):
    path = local(reference["path"])
    require(sha(path) == reference["sha256"], "Bound native input changed: " + str(path))
    return path


def write(path, value):
    path = inside(path, REGISTRY)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if path.exists():
        require(path.read_text() == text,
                "Existing native registration differs; preserve it and use a new track.")
    else:
        with path.open("x") as handle:
            handle.write(text)
    return path


def png_size(path):
    """Header only, no image decoder or HR pixel tensor."""
    with Path(path).open("rb") as handle:
        header = handle.read(29)
    require(len(header) == 29 and header[:8] == b"\x89PNG\r\n\x1a\n"
            and header[12:16] == b"IHDR" and header[24] == 8 and header[25] == 2,
            "Expected uint8 RGB PNG: " + str(path))
    return list(struct.unpack(">II", header[16:24]))


def validate_preparation(original_path, m):
    """Verify stored evaluation/diagnostic bytes without HR pixel decoding."""
    root = original_path.parent
    complete = read(root / "complete.json")
    require(complete["status"] == "completed_native_full_time_LR_sparse_HR_inputs"
            and complete["manifest"] == entry(original_path)
            and complete["scene"] == m["scene"]
            and complete["GPU_calls"] == 0 and complete["parameter_updates"] == 0,
            "Native preparation completion differs.")
    storage_path = inside(complete["hr_storage"]["path"], PREPARED)
    require(storage_path == root / "hr_storage.json"
            and sha(storage_path) == complete["hr_storage"]["sha256"],
            "Native HR storage sidecar binding differs.")
    spec_path = inside(complete["preparation_spec"]["path"], PREPARED)
    require(spec_path == root / "preparation_spec.json"
            and sha(spec_path) == complete["preparation_spec"]["sha256"],
            "Native preparation spec binding differs.")
    storage = read(storage_path)
    require(storage["schema"] == "native_hr_materialization_sidecar_v1"
            and storage["manifest_schema"] == "n3dv_dynamic_sr_pilot_v1"
            and storage["not_materialized_hr_sha256"] is None,
            "Expected explicit native missing-HR sidecar.")
    expected_stored = {
        f"hr/cam00/{f:04d}.png" for f in range(300)
    } | {
        f"hr/cam01/{f:04d}.png" for f in range(15, 300, 30)
    }
    observations = m["observations"]
    keys = [(r["camera_id"], int(r["frame_index"])) for r in observations]
    require(len(keys) == len(set(keys)) == len(m["cameras"]) * 300
            and set(keys) == {(c, f) for c in m["cameras"] for f in range(300)}
            and complete["all_camera_LR_count"] == len(keys)
            and complete["materialized_HR_count"] == len(expected_stored),
            "Native complete-camera or materialized-HR population differs.")
    all_paths = set()
    for row in observations:
        camera, frame = row["camera_id"], int(row["frame_index"])
        expected_hr = f"hr/{camera}/{frame:04d}.png"
        expected_lr = f"lr/{camera}/{frame:04d}.png"
        require(row["hr_path"] == expected_hr and row["lr_path"] == expected_lr
                and row["split"] == ("test" if camera == "cam00" else "train"),
                "Native observation HR/LR paths or split differ.")
        all_paths.add(expected_hr)
        target = inside(root / expected_hr, PREPARED)
        if expected_hr in expected_stored:
            require(isinstance(row["hr_sha256"], str)
                    and sha(target) == row["hr_sha256"]
                    and png_size(target) == [2704, 2028],
                    "Stored evaluation/diagnostic HR PNG identity/dimensions differ.")
        else:
            require(row["hr_sha256"] is None and not target.exists(),
                    "Unmaterialized training HR must stay absent and marked null.")
        if camera == "cam00":
            lr_path = inside(root / expected_lr, PREPARED)
            require(sha(lr_path) == row["lr_sha256"]
                    and png_size(lr_path) == [676, 507],
                    "Evaluation LR identity/dimensions differ.")
    materialized = storage["materialized_hr_paths"]
    absent = storage["not_materialized_hr_paths"]
    require(len(materialized) == len(set(materialized)) == len(expected_stored)
            and set(materialized) == expected_stored
            and len(absent) == len(set(absent))
            and set(absent) == all_paths - expected_stored,
            "HR materialization sidecar population differs.")
    actual_stored = {
        str(p.relative_to(root)) for p in (root / "hr").rglob("*.png")
    }
    require(actual_stored == expected_stored, "Unexpected expanded HR files in native data.")


def backed_fields(actual, template, context):
    """Every emitted config key must have a counterpart in the supplied example."""
    require(set(actual) <= set(template), f"Unbacked {context} fields: {set(actual) - set(template)}")
    for key, value in actual.items():
        if isinstance(value, dict) and isinstance(template.get(key), dict):
            # Source/file maps have path-valued keys by design; field names above
            # them are checked, while actual paths are new source identities.
            if key not in ("author_sources", "runtime_source_sha256", "project_sources",
                           "project", "upstream"):
                backed_fields(value, template[key], context + "." + key)


def pure(node):
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
            and node.func.id == "dict" and not node.args:
        require(all(k.arg is not None for k in node.keywords), "No dict expansion in config.")
        return {k.arg: pure(k.value) for k in node.keywords}
    return ast.literal_eval(node)


def author_configuration(scene):
    """Literal AST merge, matching full_native_protocol.py:43-50."""
    require(scene in SCENES, "Unsupported native scene.")
    paths = {
        "arguments/__init__.py": RUNTIME / "arguments/__init__.py",
        "arguments/dynerf/default.py": RUNTIME / "arguments/dynerf/default.py",
        f"arguments/dynerf/{scene}.py": RUNTIME / f"arguments/dynerf/{scene}.py",
    }
    result = {}
    for cls in ast.parse(paths["arguments/__init__.py"].read_text()).body:
        if not isinstance(cls, ast.ClassDef) or cls.name not in (
                "ModelParams", "ModelHiddenParams", "OptimizationParams", "PipelineParams"):
            continue
        init = next(f for f in cls.body if isinstance(f, ast.FunctionDef) and f.name == "__init__")
        result[cls.name] = {
            n.targets[0].attr: pure(n.value)
            for n in init.body
            if isinstance(n, ast.Assign) and len(n.targets) == 1
            and isinstance(n.targets[0], ast.Attribute)
            and isinstance(n.targets[0].value, ast.Name)
            and n.targets[0].value.id == "self"
        }
    for name in ("arguments/dynerf/default.py", f"arguments/dynerf/{scene}.py"):
        for node in ast.parse(paths[name].read_text()).body:
            if (isinstance(node, ast.Assign) and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id in result):
                result[node.targets[0].id].update(pure(node.value))
    template = read(LR_TEMPLATE)
    require(template["author_commit"] == AUTHOR_COMMIT, "Historical author pin differs.")
    # Cook and Cut have identical merged literal values; both scene overrides
    # only set batch_size=2, as source arguments/dynerf/{scene}.py:2-4 shows.
    require(result == template["configuration"],
            "Merged author configuration differs from the supplied Cook registration.")
    for name in paths:
        require(sha(paths[name]) == template["author_sources"][name]["sha256"],
                "Pinned author config bytes changed: " + name)
    opt = result["OptimizationParams"]
    require((opt["coarse_iterations"], opt["iterations"], opt["batch_size"])
            == (3000, 14000, 2), "Author stage/batch recipe differs.")
    return result, paths


def validate_manifest(manifest_path, seed):
    require(seed in SEEDS, "Unsupported independent seed.")
    path = inside(manifest_path, PREPARED)
    m = read(path)
    require(m["scene"] in SCENES
            and path == PREPARED / m["scene"] / f"manifest_train_ready_seed{seed}.json",
            "Expected native train-ready manifest location.")
    require(m["schema"] == "n3dv_dynamic_sr_pilot_v1"
            and m.get("full_time_decode_verified") is True and m["scale"] == 4
            and m["frame_indices"] == list(range(300))
            and m["resolutions"] == dict(hr=[2704, 2028], lr=[676, 507]),
            "Expected verified native uncropped full-300 manifest.")
    train = m["splits"]["train"]
    require(m["splits"]["test"] == ["cam00"] and not m["splits"]["dev"]
            and "cam01" in train and train[0] == "cam01" and "cam00" not in train
            and len(train) == len(set(train))
            and set(train) == set(m["cameras"]) - {"cam00"},
            "Expected all legal training cameras and cam00-only test.")
    original_path = inside(bound(m["original_full_manifest"]), PREPARED)
    original = read(original_path)
    for key in ("scene", "source", "scale", "role", "frame_indices", "resolutions",
                "cameras", "splits", "observations"):
        require(m[key] == original[key], "Seed consumer differs from producer: " + key)
    validate_preparation(original_path, original)
    init = m["initialization"]
    require(init["seed"] == seed and init["train_only_lr"] is True
            and init["official_COLMAP_initialization"] is False
            and init["npz_path"] == f"initialization_seed_{seed}/points_lr.npz",
            "Expected independent pure-LR initializer.")
    points = inside(path.parent / init["npz_path"], PREPARED)
    require(sha(points) == init["npz_sha256"], "Initial point SHA differs.")
    receipt_path = path.parent / f"initialization_seed_{seed}/complete.json"
    receipt = read(receipt_path)
    require(receipt["status"] == "completed_train_ready_manifest_and_initial_points_CPU_acceptance"
            and receipt["derived_manifest"] == entry(path)
            and receipt["identity"]["original_full_manifest"] == entry(original_path)
            and receipt["identity"]["seed"] == seed
            and receipt["CUDA_calls"] == 0 and receipt["model_prefix_trained"] is False,
            "Initializer CPU acceptance differs.")
    for c in m["cameras"].values():
        for key in ("K_hr", "K_lr", "w2c", "c2w"):
            require(all(math.isfinite(float(v)) for row in c[key] for v in row),
                    "Nonfinite camera metadata.")
        h, l = c["K_hr"], c["K_lr"]
        require(h[0][0] > 0 and h[1][1] > 0
                and abs(l[0][0] - h[0][0] / 4) < 1e-8
                and abs(l[1][1] - h[1][1] / 4) < 1e-8
                and abs(l[0][2] - ((h[0][2] + .5) / 4 - .5)) < 1e-8
                and abs(l[1][2] - ((h[1][2] + .5) / 4 - .5)) < 1e-8,
                "Pixel-centre x4 intrinsic conversion differs.")
    rows = [r for r in m["observations"] if r["split"] == "train"]
    keys = [(r["camera_id"], int(r["frame_index"])) for r in rows]
    require(len(keys) == len(set(keys)) == len(train) * 300
            and set(keys) == {(c, f) for c in train for f in range(300)},
            "Incomplete/duplicate legal LR observations.")
    inputs = []
    for row in rows:
        require(abs(float(row["time"]) - int(row["frame_index"]) / 300) < 1e-7,
                "Expected frame_index/300 time.")
        expected = f'lr/{row["camera_id"]}/{int(row["frame_index"]):04d}.png'
        require(row["lr_path"] == expected, "Unexpected legal LR path.")
        source = inside(path.parent / expected, PREPARED)
        require(sha(source) == row["lr_sha256"] and png_size(source) == [676, 507],
                "Training LR PNG identity/dimensions differ.")
        inputs.append(dict(camera=row["camera_id"], frame=int(row["frame_index"]),
                           path=str(source.relative_to(ROOT)), sha256=row["lr_sha256"]))
    centers = [[float(m["cameras"][c]["c2w"][i][3]) for i in range(3)] for c in train]
    center = [sum(x[i] for x in centers) / len(centers) for i in range(3)]
    extent = 1.1 * max(math.dist(x, center) for x in centers)
    require(math.isfinite(extent) and extent > 0, "Degenerate train-only camera extent.")
    # Every field below is present in the supplied LR config.data (lines 6-64).
    data = dict(
        manifest=entry(path), original_manifest=entry(original_path),
        initialization_receipt=entry(receipt_path), initial_points=entry(points),
        train_cameras=train, training_observations=len(rows),
        training_LR_sources_sha256=digest(inputs), train_camera_center=center,
        cameras_extent=extent, extent_cam01_included=True, heldout_camera="cam00",
        frames=[0, 299], time="frame_index/300", resolution_LR=[676, 507],
    )
    return path, m, data, inputs


def planned_counts(config):
    o = config["OptimizationParams"]
    coarse, fine, batch = o["coarse_iterations"], o["iterations"], o["batch_size"]
    return dict(
        coarse_iterations=coarse, fine_iterations=fine,
        total_iterations=coarse + fine, coarse_Adam_calls=coarse,
        fine_Adam_calls=fine - 1, total_Adam_calls=coarse + fine - 1,
        RGB_forwards=(coarse + fine) * batch, backward_calls=coarse + fine,
        batch_size=batch, stage_optimizer_resets=2, final_fine_Adam_skipped=True,
    )


def source_maps(configuration_paths):
    """Hash only source directories explicitly included in this task's inputs."""
    template = read(LR_TEMPLATE)
    runtime = {}
    for name, expected in template["runtime_source_sha256"].items():
        if name.startswith(("arguments/", "scene/", "utils/")):
            observed = sha(RUNTIME / name)
            require(observed == expected, "Author runtime source changed: " + name)
            runtime[name] = observed
    project = (Path(__file__), HERE / "teacher_native.py", CULL / "cullretrain.py",
               CULL / "rt.py", LEGACY / "n3dv_data.py", OLD / "full_prefix_registered.py",
               OLD / "full_refine_registered_v2.py", OLD / "losses.py",
               OLD / "full_native_schedule_materializer.py")
    return dict(
        author_sources={name: entry(path) for name, path in configuration_paths.items()},
        runtime_source_sha256=runtime,
        project_sources={str(p.relative_to(ROOT)): sha(p) for p in project},
    )


def lr(manifest_path, seed):
    _, m, data, _ = validate_manifest(manifest_path, seed)
    config, paths = author_configuration(m["scene"])
    sources = source_maps(paths)
    value = dict(
        status="registered_CPU_only_independent_full_LR_parent_pending_GPU_acceptance",
        seed=seed, scene=m["scene"], schema="full_author_LR_prefix_v1", data=data,
        author_commit=AUTHOR_COMMIT, configuration=config, **sources,
        observation="native_lr", white_background=config["ModelParams"]["_white_background"],
        fresh_seed_model=True, representation="native_author_GaussianModel_no_children",
        author_adaptations=[
            "Native uncropped HR2704x2028/LR676x507 pixel-centre camera interface.",
            "New native consumer registration for cullretrain CLI; no accepted historical parent is reused.",
            "Pure-LR train-only known-camera sparse initialization; cam00 is held out.",
            "Author scene configuration is unchanged; the new data root/grid is an explicit interface adaptation.",
        ],
        planned=planned_counts(config), teacher_used=False, HR_used=False,
        heldout_pixels_used=False, CUDA_initialized=False, GPU_forwards=0, Adam_calls=0,
    )
    backed_fields(value, read(LR_TEMPLATE), "LR")
    path = REGISTRY / "full_LR_prefixes" / m["scene"] / f"seed_{seed}" / "config.json"
    return dict(status=value["status"], config=entry(write(path, value)))


def assignment_function():
    """Reuse the existing audited pure-stdlib Hungarian helper, no model imports."""
    source = OLD / "full_native_schedule_materializer.py"
    spec = importlib.util.spec_from_file_location("native_assignment_metadata", source)
    require(spec is not None and spec.loader is not None, "Assignment helper unavailable.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.assignment


def schedule(m, manifest_reference, seed):
    """6000 rows, same real frame, distinct views and identical column exposure."""
    cameras = list(m["splits"]["train"])
    require(3 <= len(cameras) <= 20, "Expected 3..20 legal N3DV training cameras.")
    keys = [(r["camera_id"], int(r["frame_index"]))
            for r in m["observations"] if r["split"] == "train"]
    lookup = {key: i for i, key in enumerate(keys)}
    assign = assignment_function()
    rng = random.Random(seed)
    rows, calibration = [], []
    for frame in range(300):
        anchors = cameras[:] + [
            cameras[(frame + j) % len(cameras)] for j in range(20 - len(cameras))]
        rng.shuffle(anchors)
        found = None
        for _ in range(64):
            costs_b = [[1e9 if a == b else rng.random()
                        for b in anchors] for a in anchors]
            pb = assign(costs_b)
            if any(costs_b[i][j] >= 1e8 for i, j in enumerate(pb)):
                continue
            second = [anchors[j] for j in pb]
            costs_c = [[1e9 if c in (a, b) else rng.random() for c in anchors]
                       for a, b in zip(anchors, second)]
            pc = assign(costs_c)
            if all(costs_c[i][j] < 1e8 for i, j in enumerate(pc)):
                found = [(a, b, anchors[j]) for a, b, j in zip(anchors, second, pc)]
                break
        require(found is not None, "No balanced distinct three-view assignment.")
        block = [[lookup[(c, frame)] for c in triplet] for triplet in found]
        rows.extend(block)
        if frame in (0, 100, 200, 299):
            calibration.extend(next(row[:] for row in block
                                    if keys[row[0]] == (c, frame)) for c in cameras)
    rng.shuffle(rows)
    columns = [[row[i] for row in rows] for i in range(3)]
    for column in columns:
        rng.shuffle(column)
    random_rows = [list(row) for row in zip(*columns)]
    exposure = [Counter(row[i] for row in rows) for i in range(3)]
    require(len(rows) == 6000 and exposure[0] == exposure[1] == exposure[2]
            and set(exposure[0]) == set(range(len(keys))),
            "Full legal exposure or synchronized row count differs.")
    require(all(Counter(r[i] for r in random_rows) == exposure[i] for i in range(3)),
            "Random comparator corresponding-column exposure differs.")
    for row in rows:
        row_keys = [keys[i] for i in row]
        require(len(row) == 3 and len({k[0] for k in row_keys}) == 3
                and len({k[1] for k in row_keys}) == 1
                and "cam00" not in {k[0] for k in row_keys},
                "Invalid synchronized legal triplet.")
    # These fields are the same fields returned by the existing materializer
    # (full_native_schedule_materializer.py:198); identity.parent is omitted
    # because there is no native fine checkpoint at CPU preregistration time.
    return dict(
        schema="full_native_SR_schedule_v1",
        identity=dict(manifest=manifest_reference), seed=seed, updates=6000,
        record_keys=[list(key) for key in keys], rows=rows, random_rows=random_rows,
        calibration_triplets=calibration,
        audit=dict(
            passed=True, full_legal_observations=len(keys), every_legal_key_covered=True,
            three_column_exact_exposure=True,
            random_same_corresponding_column_exact_exposure=True, full6000_block_only=True,
        ),
        assignment_protocol=dict(
            source=entry(__file__),
            cost="Legal distinct-camera pairs have seeded random costs; no pose, overlap, HR or quality cost.",
            HR_or_test_or_quality_cost=False,
            anchor_policy="20slots/realframe; all legal cameras once plus rotating duplicate slots; exact equal column exposure.",
            calibration="all actual train cameras x frames0/100/200/299",
        ),
    )


def validate_teacher(index_path, m, data, seed, dataset_root):
    path = inside(index_path, REGISTRY / "teachers" / m["scene"])
    index = read(path)
    require(index["status"] == "completed_teacher_inventory"
            and index["scene"] == m["scene"] and index["seed_consumer"] == seed
            and index["manifest_sha256"] == data["manifest"]["sha256"]
            and local(index["manifest"]) == local(data["manifest"]["path"])
            and index["train_cameras"] == m["splits"]["train"]
            and index["frame_indices"] == list(range(300))
            and all(index["teacher_config"].get(k) == v for k, v in FROZEN.items()),
            "Teacher index does not bind this complete native seed consumer/frozen recipe.")
    producer_config = bound(index["producer_config"])
    require(index["teacher_config_sha256"] == sha(producer_config)
            and read(producer_config) == index["teacher_config"],
            "Teacher producer config differs.")
    require(index["producer_manifest"] == data["original_manifest"]
            and index["seed_independent_frozen_prior"] is True,
            "Teacher producer or independent-prior binding differs.")
    bound(index["producer_manifest"])
    bound(index["producer_summary"])
    bound(index["plan"])
    expected = {(r["camera_id"], int(r["frame_index"])): r
                for r in m["observations"] if r["split"] == "train"}
    keys = [(r["camera"], int(r["frame"])) for r in index["entries"]]
    require(len(keys) == len(set(keys)) == len(expected) and set(keys) == set(expected),
            "Teacher inventory missing/duplicates/heldout camera.")
    files = []
    for e in index["entries"]:
        camera, frame = e["camera"], int(e["frame"])
        observation = expected[(camera, frame)]
        require(e["relative_path"] == f"sr_swinir_x4/{camera}/{frame:04d}.png"
                and e["lr_relative_path"] == observation["lr_path"]
                and e["lr_sha256"] == observation["lr_sha256"],
                "Teacher entry LR/path binding differs.")
        target = inside(dataset_root / e["relative_path"], PREPARED)
        receipt = target.with_suffix(".json")
        require(sha(target) == e["sha256"] and sha(receipt) == e["receipt_sha256"],
                "Frozen teacher PNG/receipt SHA differs.")
        record = read(receipt)
        require(record["input_hw"] == [507, 676]
                and record["output_hw"] == [2028, 2704]
                and record["input_sha256"] == e["lr_sha256"]
                and record["output_sha256"] == e["sha256"],
                "Native teacher receipt dimensions/identity differ.")
        files.append(dict(camera=camera, frame=frame,
                          path=str(target.relative_to(ROOT)), sha256=e["sha256"]))
    return path, files


def sr(manifest_path, seed, teacher_index):
    path, m, data, lr_files = validate_manifest(manifest_path, seed)
    config, configuration_paths = author_configuration(m["scene"])
    sources = source_maps(configuration_paths)
    lr_config = REGISTRY / "full_LR_prefixes" / m["scene"] / f"seed_{seed}" / "config.json"
    parent_plan = read(lr_config)
    require(parent_plan["configuration"] == config and parent_plan["data"] == data,
            "Native LR registration differs from the SR consumer.")
    teacher_path, teacher_files = validate_teacher(teacher_index, m, data, seed, path.parent)
    table = schedule(m, data["manifest"], seed)
    schedule_path = write(
        REGISTRY / "schedules" / m["scene"] / f"seed_{seed}" / "schedule.json", table)
    sr_template = read(SR_TEMPLATE)
    require(sr_template["definition"] == DEFINITION
            and sr_template["configuration"] == config,
            "Provided registered Bsync definition/configuration changed.")
    sr_data = dict(data, resolution_HR=[2704, 2028])
    value = dict(
        schema="registered_full_native_SR_refinement_v1",
        status="registered_native_cullretrain_Bsync_consumer_configuration",
        missing=[], scene=m["scene"], seed=seed, method="Bsync", definition=DEFINITION,
        data=sr_data,
        dependencies=dict(schedule=entry(schedule_path), teacher=entry(teacher_path)),
        full_native_context=dict(manifest=data["manifest"], schedule=entry(schedule_path)),
        configuration=config,
        source_files=dict(project=sources["project_sources"],
                          upstream=sources["runtime_source_sha256"]),
        representation="native_author_GaussianModel_no_children",
        author_commit=AUTHOR_COMMIT,
        adaptations=[
            "New uncropped native-resolution cullretrain CLI consumer config; no completed historical endpoint is reused.",
            "Frozen SwinIR recipe unchanged; completed teacher index binds this native train-only LR inventory.",
            "New seed-balanced synchronized three-view rows omit historical frozen-parent-overlap assignment.",
            "definition remains registered Bsync lambda0.05 shape required by cullretrain; run --lambda-sr0.2 --dssim0.2 --prune-gate none --affine supplies requested native SR variant.",
            "Parent checkpoint/configuration equality is checked by cullretrain at SR execution; CPU registration does not claim parent training is complete.",
        ],
        training_files=dict(LR=lr_files, teacher=teacher_files),
        schedule=dict(
            steps=6000, LR_clock="fine14000+SRcursor", topology_clock="refinement",
            topology_steps=[1, 6000], topology_rule_source=entry(OLD / "full_prefix_registered.py"),
            parameter_updates=6000, Adam_calls_per_update=1, optimizer_resets=0,
            SR_weights=[0.0, 0.05, 0.05], regularizer_once=True,
            SH_clock="fine14000+SRcursor",
            topology_buffers="clear only max_radii2D/xyz_gradient_accum/denom once at LR-to-HR boundary",
            viewspace_aggregation="same-index sum before norm; max radii; any visibility",
        ),
        planned_cost=dict(
            SR_updates=6000, RGB_forwards=18000, moment_forwards=0,
            backward=6000, Adam=6000,
            prefix_per_seed=dict(iterations=17000, RGB=34000, Adam=16999, batch_size=2),
            prefix_shared_within_seed_only=True,
            teacher_support_calibration_diagnostics_evaluation_separate=True,
        ),
        evaluation_contract=dict(heldout="cam00", frames=list(range(300))),
        HR_training_images_read=False, heldout_pixels_read=False,
        CPU_only_registration=True, GPU_forwards=0, Adam_calls=0,
    )
    backed_fields(value, sr_template, "SR")
    target = REGISTRY / "full_SR_refinement" / m["scene"] / f"seed_{seed}" / "Bsync/config.json"
    return dict(status=value["status"], config=entry(write(target, value)),
                schedule=entry(schedule_path), teacher=entry(teacher_path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("lr", "sr"):
        sp = sub.add_parser(name)
        sp.add_argument("--manifest", type=Path, required=True)
        sp.add_argument("--seed", type=int, required=True)
        if name == "sr":
            sp.add_argument("--teacher-index", type=Path, required=True)
    a = parser.parse_args()
    result = (lr(a.manifest, a.seed) if a.command == "lr"
              else sr(a.manifest, a.seed, a.teacher_index))
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
