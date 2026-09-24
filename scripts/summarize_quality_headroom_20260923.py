"""Recompute existing quality comparisons on CPU; no rendering or training."""
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "output/dynamic_sr_20260923/quality_headroom"
SCENES = ("cook_spinach", "cut_roasted_beef", "meetroom_discussion")
METRICS = ("psnr", "ssim", "lpips_alex", "mse")
sources = {}


def read(path):
    path = ROOT / path
    raw = path.read_bytes()
    sources[str(path.relative_to(ROOT))] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)


def average(rows):
    assert rows
    return {k: mean(r[k] for r in rows) for k in METRICS}


def check(actual, recorded, suffix=""):
    for k in METRICS:
        assert math.isclose(actual[k], recorded[k + suffix], abs_tol=1e-10), k


def delta(before, after):
    return {
        **{k: after[k] - before[k] for k in METRICS},
        "mean_mse_reduction_percent": 100 * (1 - after["mse"] / before["mse"]),
        "lpips_reduction_percent": 100 * (1 - after["lpips_alex"] / before["lpips_alex"]),
    }


prior = read("output/dynamic_sr_20260918/prior_quality_audit/metrics.json")
native = read("output/dynamic_sr_20260921/native_resolution_trajectory/cam00_full60/metrics.json")
result = {"protocol": "Existing fixed 18k continuation checkpoints; full cam00 60 frames at HR resolution. HR oracle uses LR plus four-camera HR at weight 1; SR uses weight 0.1. No new training or image metric inference.", "scenes": {}}
for scene in SCENES:
    base = "output/dynamic_sr_20260919/" + scene
    branches = {}
    identity = None
    for branch in ("lr_long", "sr_w01", "hr_oracle_w10"):
        d = read(base + "_" + branch + "_evaluation/metrics.json")
        cfg = read(base + "_" + branch + "/config.json")
        frames = [r["frame_index"] for r in d["rows"]]
        assert frames == list(range(0, 120, 2))
        assert d["test_camera"] == "cam00"
        current = (d["manifest_sha256"], d["checkpoint_metadata"]["parent_sha"])
        identity = identity or current
        assert current == identity
        assert cfg["parent_sha256"] == identity[1]
        assert cfg["manifest_sha256"] == identity[0]
        assert cfg["steps"] == d["checkpoint_metadata"]["intervention_step"] == 18000
        assert (cfg["teacher"], cfg["weight"]) == {"lr_long": ("none", 0), "sr_w01": ("sr", .1), "hr_oracle_w10": ("hr", 1)}[branch]
        m = average([r["spatial"]["full"] for r in d["rows"]])
        check(m, d["aggregate"]["full"], "_mean")
        branches[branch] = {"mean": m, "checkpoint_sha256": d["checkpoint_sha256"], "teacher": cfg["teacher"], "weight": cfg["weight"], "training_device": cfg.get("device"), "n": len(frames)}
    diag_path = (f"output/dynamic_sr_20260918/{scene}_pilot_v1_sr_reference_diagnostic/metrics.json" if scene != "meetroom_discussion" else base + "_real_lr_reference/metrics.json")
    d = read(diag_path)
    assert d["manifest_sha256"] == identity[0]
    assert d["admissible_novel_view_baseline"] is False and d["used_for_training"] is False
    diagnostic = {}
    for mode in ("bicubic", "swinir"):
        x = d["modes"][mode]
        assert [r["frame_index"] for r in x["rows"]] == list(range(0, 120, 2))
        m = average([r["spatial"]["full"] for r in x["rows"]])
        check(m, x["aggregate"]["full"], "_mean")
        diagnostic[mode] = m
    fit = read(base + "_sr_w01/fit_18000.json")
    keys = [(r["camera_id"], r["frame_index"]) for r in fit["rows"]]
    assert len(keys) == len(set(keys)) == 16
    assert {k[1] for k in keys} == {0, 40, 80, 118}
    if scene == "meetroom_discussion":
        train = read(base + "_train_prior_reference/metrics.json")
        assert train["manifest_sha256"] == identity[0]
        teachers = {mode: {(r["camera_id"], r["frame_index"]): r["spatial"]["full"] for r in train["modes"][mode]["rows"]} for mode in ("bicubic", "swinir")}
    else:
        train = prior["scenes"][scene]
        assert train["manifest_sha256"] == identity[0]
        teachers = {mode: {(r["camera_id"], r["frame_index"]): r[mode]["full"] for r in train["rows"]} for mode in ("bicubic", "swinir")}
    fit_mean = average([r["render_hr"] for r in fit["rows"]])
    check(fit_mean, fit["aggregate"]["render_hr"])
    training_branches = {}
    for branch in ("lr_long", "sr_w01", "hr_oracle_w10"):
        branch_fit = read(base + "_" + branch + "/fit_18000.json")
        assert [(r["camera_id"], r["frame_index"]) for r in branch_fit["rows"]] == keys
        assert branch_fit["step"] == fit["step"]
        m = average([r["render_hr"] for r in branch_fit["rows"]])
        check(m, branch_fit["aggregate"]["render_hr"])
        training_branches[branch] = m
    teacher_mean = {mode: average([rows[k] for k in keys]) for mode, rows in teachers.items()}
    item = {
        "manifest_sha256": identity[0], "parent_sha256": identity[1],
        "reconstruction": branches, "actual_target_view_lr_diagnostic": diagnostic,
        "lr_to_sr": delta(branches["lr_long"]["mean"], branches["sr_w01"]["mean"]),
        "sr_to_hr_oracle": delta(branches["sr_w01"]["mean"], branches["hr_oracle_w10"]["mean"]),
        "same_training_16": {"keys": keys, "reconstruction": training_branches, "sr_to_hr_oracle": delta(fit_mean, training_branches["hr_oracle_w10"]), "sr_reconstruction": fit_mean, **teacher_mean, "reconstruction_to_teacher": delta(fit_mean, teacher_mean["swinir"])},
    }
    if scene in native["scenes"]:
        item["pure_hr_different_training_budget"] = {}
        for step, x in native["scenes"][scene]["full_hr"].items():
            assert x["manifest_sha256"] == identity[0]
            m = average([r["metrics"] for r in x["rows"]])
            check(m, x["mean"])
            item["pure_hr_different_training_budget"][step] = m
    result["scenes"][scene] = item
result["source_sha256"] = sources
result["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
result["validation"] = "PASS: stored means independently recomputed; cam00 frame sets, scene manifests, continuation parents, budgets, weights and 16-view/frame pairing checked. Existing recorded identities checked; original images/checkpoints not rehashed or reevaluated in this summary."
DEST.mkdir(parents=True, exist_ok=True)
(DEST / "summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
for scene, x in result["scenes"].items():
    print(scene)
    for label, m in [(b, v["mean"]) for b, v in x["reconstruction"].items()] + list(x["actual_target_view_lr_diagnostic"].items()) + [("train16_" + k, x["same_training_16"][k]) for k in ("sr_reconstruction", "swinir")]:
        print(label, " / ".join(f"{m[k]:.6f}" for k in METRICS[:3]))
    print("SR to HR:", x["sr_to_hr_oracle"])
    print("train16 teacher gap:", x["same_training_16"]["reconstruction_to_teacher"])
print(result["validation"])
