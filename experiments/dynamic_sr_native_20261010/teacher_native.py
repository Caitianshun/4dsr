#!/usr/bin/env python3
"""Plan, generate, and accept the frozen training-LR-only native x4 teacher.

This file is a future operator entry: writing it does not start inference.
``plan`` and ``accept`` use only the Python standard library. ``generate`` and
``run`` invoke the pinned historical generator on a separately assigned GPU.
The historical generator source must have the original d7be... SHA; the newer
editable generate_prior.py is deliberately not substituted for it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "output/dynamic_sr_native_20261010/registry"
PREPARED = ROOT / "data/dynamic_sr/native_20261010/prepared"
SCENES = ("cook_spinach", "cut_roasted_beef")
SEEDS = (20261007, 20261008)
DEFAULT_GENERATOR = (
    ROOT / "output/dynamic_sr_20260918/cook_spinach_pilot_v1_lr_integrated"
    "/source_snapshot/06_generate_prior.py"
)
DEFAULT_NETWORK = ROOT / "third_party/swinir_frozen_20261010/network_swinir.py"
DEFAULT_CHECKPOINT = Path(
    "/home/cai_tianshun/Project/mml/outputs/week3_swinir/ckpt"
    "/001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth"
)
FROZEN = dict(
    generator_sha256="d7be23d15473082c57e1e4091bfea9eaafad0de97ef7379db39308ae14c6cbcc",
    network_sha256="1d650d2c1c4519d95db771863691c6b239a0822222f287f6a8342a9e0eccbc6e",
    checkpoint_sha256="4e78e33f22c1aa8a773db0cf4a7381bae97c2362c717f155439ebc690cbd9215",
    scale=4,
    window=8,
    tile=0,
    overlap=32,
    precision="float32",
    padding="official_mirror_concat_next_window",
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


def local(path):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def relative(path):
    path = local(path)
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def entry(path):
    return dict(path=relative(path), sha256=sha(local(path)))


def pinned_reference(path, expected):
    """Record a future pinned input without reading an unavailable host file."""
    return dict(path=relative(path), sha256=expected)


def bound(reference):
    path = local(reference["path"])
    require(path.is_file() and sha(path) == reference["sha256"],
            "Bound file missing or changed: " + str(path))
    return path


def inside(path, namespace):
    spelling = Path(path)
    spelling = spelling if spelling.is_absolute() else ROOT / spelling
    require(not any(part.is_symlink() for part in [spelling, *spelling.parents]),
            "Symlink refused: " + str(spelling))
    path = spelling.resolve()
    require(path.is_relative_to(namespace.resolve()) and path != namespace.resolve(),
            "Path outside the new native namespace: " + str(path))
    return path


def write(path, value):
    path = inside(path, REGISTRY)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if path.exists():
        require(path.read_text() == text, "Existing native artifact differs; use a new track.")
    else:
        with path.open("x") as handle:
            handle.write(text)
    return path


def png_size(path):
    with Path(path).open("rb") as handle:
        header = handle.read(29)
    require(header[:8] == b"\x89PNG\r\n\x1a\n" and header[12:16] == b"IHDR",
            "Expected PNG: " + str(path))
    require(header[24] == 8 and header[25] == 2,
            "Expected uint8 RGB PNG: " + str(path))
    return list(struct.unpack(">II", header[16:24]))


def manifest_inventory(manifest_path):
    manifest_path = inside(manifest_path, PREPARED)
    m = read(manifest_path)
    require(manifest_path == PREPARED / m["scene"] / "manifest.json",
            "Teacher producer must be the original native manifest.")
    require(m["scene"] in SCENES and m["schema"] == "n3dv_dynamic_sr_pilot_v1",
            "Unsupported native scene/schema.")
    require(m.get("full_time_decode_verified") is True and m["scale"] == 4
            and m["frame_indices"] == list(range(300)),
            "Verified full-300 x4 manifest required.")
    require(m["resolutions"] == dict(hr=[2704, 2028], lr=[676, 507]),
            "This track keeps the uncropped native grid.")
    train = m["splits"]["train"]
    require(m["splits"]["test"] == ["cam00"] and not m["splits"]["dev"]
            and "cam01" in train and "cam00" not in train
            and len(train) == len(set(train))
            and set(train) == set(m["cameras"]) - {"cam00"},
            "Expected official cam00 holdout and all other actual cameras.")
    rows = sorted((r for r in m["observations"] if r["split"] == "train"),
                  key=lambda r: (r["camera_id"], int(r["frame_index"])))
    keys = [(r["camera_id"], int(r["frame_index"])) for r in rows]
    require(len(rows) == len(set(keys)) == len(train) * 300
            and set(keys) == {(c, f) for c in train for f in range(300)},
            "Incomplete/duplicate training LR inventory.")
    return manifest_path, m, rows


def consumer(original, producer, seed):
    require(seed in SEEDS, "Unsupported independent seed.")
    path = producer.parent / f"manifest_train_ready_seed{seed}.json"
    m = read(path)
    require(m["original_full_manifest"] == entry(producer),
            "Consumer is bound to another original manifest.")
    for key in ("scene", "splits", "frame_indices", "resolutions", "cameras", "observations"):
        require(m[key] == original[key], "Teacher input metadata differs: " + key)
    require(m["initialization"]["seed"] == seed, "Consumer seed differs.")
    return dict(
        seed=seed,
        status="completed_LR_point_manifest_teacher_generation_pending",
        manifest=entry(path),
        producer_equivalence=(
            "exact scene/splits/frame_indices/resolutions/cameras/observations; "
            "initialization-specific metadata differs"
        ),
        future_teacher_index=relative(
            REGISTRY / "teachers" / original["scene"] / f"teacher_index_seed{seed}.json"),
    )


def plan(manifest_path, seeds, generator, network, checkpoint):
    manifest_path, m, rows = manifest_inventory(manifest_path)
    require(len(seeds) == len(set(seeds)), "Duplicate consumer seed.")
    require(not (manifest_path.parent / "sr_swinir_x4").exists(),
            "Plan requires an absent teacher-cache directory; preserve an earlier attempt.")
    entries = []
    for row in rows:
        camera, frame = row["camera_id"], int(row["frame_index"])
        expected = f"lr/{camera}/{frame:04d}.png"
        require(row["lr_path"] == expected, "Unexpected training LR path.")
        source = inside(manifest_path.parent / expected, PREPARED)
        require(sha(source) == row["lr_sha256"] and png_size(source) == [676, 507],
                "Native LR identity/dimensions changed.")
        target = inside(manifest_path.parent / f"sr_swinir_x4/{camera}/{frame:04d}.png",
                        PREPARED)
        require(not target.exists() and not target.with_suffix(".json").exists(),
                "Plan requires a new teacher cache; preserve an earlier cache/plan.")
        entries.append(dict(
            camera=camera, frame=frame,
            relative_path=f"sr_swinir_x4/{camera}/{frame:04d}.png",
            target_path=relative(target),
            receipt_path=relative(target.with_suffix(".json")),
            lr_relative_path=expected, lr_path=relative(source),
            lr_sha256=row["lr_sha256"], existing_full_cache=False, reuse=None,
        ))
    directory = REGISTRY / "teachers" / m["scene"]
    value = dict(
        schema=1, status="CPU_plan_completed_GPU_teacher_generation_not_started",
        scene=m["scene"], source=entry(__file__), producer_manifest=entry(manifest_path),
        seed_consumers=[consumer(m, manifest_path, int(s)) for s in seeds],
        teacher_generator=pinned_reference(generator, FROZEN["generator_sha256"]),
        original_teacher_generator=pinned_reference(generator, FROZEN["generator_sha256"]),
        network=pinned_reference(network, FROZEN["network_sha256"]),
        checkpoint=pinned_reference(checkpoint, FROZEN["checkpoint_sha256"]),
        frozen_recipe=FROZEN, training_camera_ids=m["splits"]["train"],
        cam01_legal_included=True, frame_indices=list(range(300)),
        train_observation_count=len(rows), expected_LR_dimensions=[676, 507],
        expected_SR_dimensions=[2704, 2028], entries=entries,
        output_directory=relative(manifest_path.parent / "sr_swinir_x4"),
        GPU_command_template=[
            "<root-selected-python>", relative(generator), "--manifest",
            relative(manifest_path), "--checkpoint", str(checkpoint),
            "--network", relative(network), "--dependency-path",
            "experiments/dynamic_sr_20260918/vendor", "--device", "cuda:0",
            "--tile", "0", "--overlap", "32", "--log-every", "60",
        ],
        required_environment={
            "CUDA_VISIBLE_DEVICES": "<root-assigned-free-GPU-identifier>",
            "FOURDSR_ROOT": "host project root; preserve source/data identities",
        },
        no_explicit_camera_subset=True, batch_size=1, full_image_no_tiling=True,
        no_autocast=True, TF32_matmul_and_cudnn=False, cudnn_benchmark=False,
        target_representation=(
            "Frozen original SwinIR output clipped[0,1], multiplied255/roundeduint8 "
            "RGB PNG; consumers load float32/255."
        ),
        input_policy="Exactly all manifest train LR; no cam00/dev/test/HR pixels read",
        external_pretraining="SwinIR-M x4 DF2K (DIV2K+Flickr2K), not target fine-tuned",
        seed_independent_prior=(
            "Generate once from original native manifest; per-seed indexes bind "
            "identical legal LR inventories and independent initialized manifests."
        ),
        completion_requirement=(
            "selection_complete=true; all train PNG/receipt SHA and dimensions "
            "verified before completed_teacher_inventory indexes."
        ),
        future_accept_command=[
            "<CPU-python>", relative(__file__), "accept", "--plan",
            relative(directory / "plan.json"),
        ],
        GPU_inference_executed=False, CUDA_calls=0, HR_quality_evaluation=False,
        confirmation_method_selection=False,
    )
    target = write(directory / "plan.json", value)
    return dict(status=value["status"], plan=entry(target),
                train_images=len(rows), CUDA_calls=0)


def load_plan(plan_path):
    plan_path = inside(plan_path, REGISTRY / "teachers")
    p = read(plan_path)
    require(p["frozen_recipe"] == FROZEN and p["scene"] in SCENES,
            "Frozen recipe/scene differs.")
    bound(p["source"])
    producer = bound(p["producer_manifest"])
    _, m, rows = manifest_inventory(producer)
    require(plan_path == REGISTRY / "teachers" / m["scene"] / "plan.json",
            "Expected canonical native teacher plan location.")
    require(p["expected_LR_dimensions"] == [676, 507]
            and p["expected_SR_dimensions"] == [2704, 2028]
            and p["training_camera_ids"] == m["splits"]["train"],
            "Plan native dimensions/split differ.")
    require(p["train_observation_count"] == len(rows) == len(p["entries"]),
            "Plan legal teacher population differs.")
    for e, row in zip(p["entries"], rows):
        camera, frame = row["camera_id"], int(row["frame_index"])
        target = producer.parent / f"sr_swinir_x4/{camera}/{frame:04d}.png"
        require(e["camera"] == camera and e["frame"] == frame
                and e["relative_path"] == f"sr_swinir_x4/{camera}/{frame:04d}.png"
                and e["target_path"] == relative(target)
                and e["receipt_path"] == relative(target.with_suffix(".json"))
                and e["lr_path"] == relative(producer.parent / row["lr_path"])
                and e["lr_relative_path"] == row["lr_path"]
                and e["lr_sha256"] == row["lr_sha256"],
                "Plan legal LR/teacher entry differs from the producer manifest.")
    for c in p["seed_consumers"]:
        require(consumer(m, producer, c["seed"]) == c,
                "Seed consumer changed after planning.")
    return plan_path, p, producer


def generate(plan_path, checkpoint=None, network=None, generator=None, device="cuda:0"):
    """Future GPU service action; the wrapper itself does not import PyTorch."""
    _, p, producer = load_plan(plan_path)
    generator = local(generator or p["teacher_generator"]["path"])
    network = local(network or p["network"]["path"])
    checkpoint = local(checkpoint or p["checkpoint"]["path"])
    try:
        for path, key in ((generator, "generator_sha256"), (network, "network_sha256"),
                          (checkpoint, "checkpoint_sha256")):
            require(sha(path) == FROZEN[key], "Frozen teacher input changed: " + str(path))
        require(device.startswith("cuda:"), "Generation belongs to the assigned GPU service.")
        # The historical source body was unavailable to this restricted worker.
        # Do not rely on undocumented resume behavior or permit an old partial
        # cache to be overwritten by a later automatically repeated task.
        require(not (producer.parent / "sr_swinir_x4").exists(),
                "Teacher cache already exists; preserve this attempt for explicit operator review.")
        command = [
            sys.executable, "-u", str(generator), "--manifest", str(producer),
            "--checkpoint", str(checkpoint), "--network", str(network),
            "--dependency-path", str(ROOT / "experiments/dynamic_sr_20260918/vendor"),
            "--device", device, "--tile", "0", "--overlap", "32",
            "--log-every", "60",
        ]
        # No --cameras or --limit: every legal training LR must complete.
        environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        subprocess.run(command, check=True, cwd=ROOT, env=environment)
    except BaseException:
        write(local(plan_path).parent / f"generate_failure_{time.time_ns()}.json", dict(
            status="failed_teacher_generation_partial_outputs_preserved",
            plan=entry(local(plan_path)), error=traceback.format_exc(),
            source=entry(__file__), output_directory=p["output_directory"],
            automatic_retry_allowed=False,
        ))
        raise


def accept(plan_path):
    plan_path, p, producer = load_plan(plan_path)
    cache = producer.parent / "sr_swinir_x4"
    config = cache / "prior_config.json"
    cfg = read(config)
    require(cfg["manifest_sha256"] == p["producer_manifest"]["sha256"]
            and all(cfg.get(k) == v for k, v in FROZEN.items())
            and cfg["fine_tuned_on_target"] is False,
            "Frozen producer configuration differs.")
    complete = []
    for q in sorted(cache.glob("summary_*.json")):
        summary = read(q)
        n = p["train_observation_count"]
        if (summary.get("selection_complete") is True and summary.get("selected") == n
                and summary.get("train_inventory_selected") == n
                and summary.get("generated", 0) + summary.get("skipped", 0) == n):
            complete.append(q)
    require(bool(complete), "No successful complete generator summary.")
    existing = plan_path.parent / "acceptance_complete.json"
    if existing.exists():
        previous = read(existing)
        require(previous["producer_manifest"] == p["producer_manifest"],
                "Previous acceptance belongs to another producer.")
        prior_index = read(bound(previous["indexes"][0]))
        summary_path = bound(prior_index["producer_summary"])
        require(summary_path in complete, "Accepted completion summary changed.")
    else:
        summary_path = complete[-1]
    rows = []
    for e in p["entries"]:
        source = inside(e["lr_path"], PREPARED)
        target = inside(e["target_path"], PREPARED)
        receipt = inside(e["receipt_path"], PREPARED)
        record = read(receipt)
        require(sha(source) == e["lr_sha256"]
                and record["input_sha256"] == e["lr_sha256"]
                and sha(target) == record["output_sha256"],
                "Teacher/input SHA identity differs.")
        require(png_size(source) == [676, 507] and png_size(target) == [2704, 2028]
                and record["input_hw"] == [507, 676]
                and record["output_hw"] == [2028, 2704],
                "Teacher native PNG/receipt dimensions differ.")
        rows.append(dict(
            camera=e["camera"], frame=e["frame"], relative_path=e["relative_path"],
            sha256=sha(target), lr_sha256=e["lr_sha256"],
            lr_relative_path=e["lr_relative_path"],
            receipt_sha256=sha(receipt), original_reuse=None,
        ))
    indexes = []
    for c in p["seed_consumers"]:
        cp = bound(c["manifest"])
        value = dict(
            schema=1, status="completed_teacher_inventory", scene=p["scene"],
            seed_consumer=c["seed"], manifest=str(cp), manifest_sha256=sha(cp),
            producer_manifest=p["producer_manifest"], producer_config=entry(config),
            teacher_config=cfg, teacher_config_sha256=sha(config),
            producer_summary=entry(summary_path), plan=entry(plan_path), entries=rows,
            train_cameras=p["training_camera_ids"], frame_indices=p["frame_indices"],
            input_policy=p["input_policy"], seed_independent_frozen_prior=True,
            CUDA_calls_in_acceptance=0, HR_quality_evaluation=False,
        )
        indexes.append(entry(write(c["future_teacher_index"], value)))
    write(existing, dict(
        status="completed_full_teacher_CPU_acceptance",
        producer_manifest=p["producer_manifest"], indexes=indexes,
        teacher_images=len(rows), source=entry(__file__), CUDA_calls=0,
        GPU_inference_executed_by_this_program=False, HR_image_reads=0,
    ))
    return indexes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    pp = sub.add_parser("plan")
    pp.add_argument("--manifest", type=Path, required=True)
    pp.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    pp.add_argument("--generator", type=Path, default=DEFAULT_GENERATOR)
    pp.add_argument("--network", type=Path, default=DEFAULT_NETWORK)
    pp.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    for name in ("generate", "run", "accept"):
        sp = sub.add_parser(name)
        sp.add_argument("--plan", type=Path, required=True)
        if name != "accept":
            sp.add_argument("--checkpoint", type=Path)
            sp.add_argument("--network", type=Path)
            sp.add_argument("--generator", type=Path)
            sp.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.command == "plan":
        result = plan(args.manifest, args.seeds, args.generator, args.network, args.checkpoint)
    elif args.command == "accept":
        result = dict(status="completed_CPU_teacher_acceptance", indices=accept(args.plan))
    else:
        if not (local(args.plan).parent / "acceptance_complete.json").exists():
            generate(args.plan, args.checkpoint, args.network, args.generator, args.device)
        result = (dict(status="completed_teacher_generation_and_CPU_acceptance",
                       indices=accept(args.plan)) if args.command == "run"
                  else dict(status="GPU_generator_returned_CPU_acceptance_required"))
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
