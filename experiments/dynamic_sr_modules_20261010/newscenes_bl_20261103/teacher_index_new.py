"""Verify the completed common teacher and produce sv2c's key->path/SHA index.

For A100 regeneration, --reference-index is the unchanged local producer
inventory: every PNG SHA must agree. No GPU, teacher generation or HR scoring.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_bl import *

TAG = "sr_swinir_x4_areaft"


def producer_path(scene):
    return REG / "teacher_cache" / (scene + "_area4") / f"teacher_index_{TAG}_seed{SEED}.json"


def cache_path(scene, regenerate=False):
    base = OUT if regenerate else REG
    return base / "teacher_cache" / (scene + "_area4")


def index_path(scene):
    return OUT / "teacher_indexes" / scene / "areaft_index.json"


def entries_from_producer(index, m, mp, cache, weights, verify=True):
    require(index["status"] == "completed_teacher_inventory" and index["scene"] == m["scene"]
            and index["seed_consumer"] == SEED and index["manifest_sha256"] == sha(mp),
            "Teacher producer identity mismatch")
    require(index["oracle"] is False and index["seed_independent_frozen_prior"] is True
            and index["HR_quality_evaluation"] is False, "Teacher information boundary mismatch")
    require(sorted(index["train_cameras"]) == sorted(m["splits"]["train"]), "Teacher camera mismatch")
    cfg = index["teacher_config"]
    require(cfg["tag"] == TAG and cfg["batch"] == 6 and cfg["scale"] == 4
            and cfg["mask_cache"] is True and cfg["weights"] == weights,
            "Frozen areaft weights/batch6 recipe mismatch")
    rows = {(r["camera_id"], r["frame_index"]): r for r in m["observations"] if r["split"] == "train"}
    entries, seen = {}, set()
    for row in index["entries"]:
        key = (row["camera"], int(row["frame"]))
        require(key in rows and key not in seen, "Unknown/duplicate teacher key")
        seen.add(key)
        obs = rows[key]
        require(row["lr_relative_path"] == obs["lr_path"] and row["lr_sha256"] == obs["lr_sha256"],
                "Teacher LR identity differs")
        p = (mp.parent / row["relative_path"]).resolve()
        expected = (cache / TAG / key[0] / f"{key[1]:04d}.png").resolve()
        require(p == expected, "Teacher path differs from exact cache inventory")
        require(len(row["sha256"]) == 64 and all(c in "0123456789abcdef" for c in row["sha256"]),
                "Invalid teacher SHA")
        if verify:
            from PIL import Image
            require(sha(p) == row["sha256"], "Teacher PNG SHA mismatch")
            with Image.open(p) as im:
                require(im.format == "PNG" and im.mode == "RGB"
                        and list(im.size) == m["resolutions"]["hr"], "Teacher RGB8 shape mismatch")
                im.verify()
        entries[f"{key[0]}/{key[1]:04d}"] = dict(
            path=rel(p), sha256=row["sha256"], bytes=p.stat().st_size if verify else None)
    require(seen == set(rows), "Incomplete teacher inventory")
    return dict(sorted(entries.items()))


def compare_reference(reference, generated):
    expected = {f"{r['camera']}/{int(r['frame']):04d}": r["sha256"] for r in reference["entries"]}
    require(len(expected) == len(reference["entries"]), "Duplicate reference PNG")
    require(set(expected) == set(generated), "Reference/generated teacher keys differ")
    bad = [k for k in expected if expected[k] != generated[k]["sha256"]]
    require(not bad, "Cross-host teacher PNG SHA differs: " + ", ".join(bad[:8]))
    return dict(images=len(expected), mismatches=0, comparison="every encoded PNG SHA256")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene", choices=SCENES, required=True)
    p.add_argument("--producer-index")
    p.add_argument("--cache-root")
    p.add_argument("--reference-index")
    p.add_argument("--out")
    p.add_argument("--weights-summary", default="output/dynamic_sr_physlr_20261016/teacher_ft/swinir_area_ft_v1/summary.json")
    a = p.parse_args()
    mp, m, _, _, _ = validate_metadata(a.scene)
    source = local(a.producer_index) if a.producer_index else producer_path(a.scene)
    cache = local(a.cache_root) if a.cache_root else cache_path(a.scene)
    require(cache.resolve() in (cache_path(a.scene).resolve(), cache_path(a.scene, True).resolve()),
            "Only the registered local cache or the new regeneration namespace is allowed")
    producer = read(source)
    summary = read(a.weights_summary)
    entries = entries_from_producer(producer, m, mp, cache, summary["weights"])
    # Verify actual weight bytes as well as inventory metadata.
    weights = summary["weights"]
    if isinstance(weights, dict) and "path" in weights and "sha256" in weights:
        require(sha(weights["path"]) == weights["sha256"], "Frozen teacher weight bytes differ")
    record = dict(schema="static_sr_v2_areaft_index", scene=a.scene, manifest=identity(mp),
                  teacher="common frozen SwinIR-M x4 areaft, batch6, exact registered RGB8 PNGs",
                  producer=identity(source), entries=entries, teacher_config=producer["teacher_config"],
                  status="verified_all_teacher_PNGs", no_HR_reference_pixels_read=True)
    if a.reference_index:
        reference = read(a.reference_index)
        # The reference itself must be the same scene/inputs/weights recipe.
        entries_from_producer(reference, m, mp, cache_path(a.scene), summary["weights"], verify=False)
        record["reference"] = identity(a.reference_index)
        record["cross_host_verification"] = compare_reference(reference, entries)
    elif cache.resolve() == cache_path(a.scene, True).resolve():
        raise ValueError("A100 regeneration requires --reference-index; no approximate fallback")
    write_new(a.out or index_path(a.scene), record)
    print(__import__("json").dumps(dict(status=record["status"], images=len(entries),
                                      bytes=sum(e["bytes"] for e in entries.values()),
                                      cross_host=record.get("cross_host_verification"))))


if __name__ == "__main__":
    main()
