"""CPU-only, read-only audit of the two existing temporal-sharing assets.

Writes only a new audit JSON. Does not decode HR, construct a CUDA model,
render, train, access a remote host, or mutate historical artifacts.
"""
from __future__ import annotations

import argparse
import collections
import datetime
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ROOT.parent / "4dgs"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text())


def check_file(path, expected=None):
    path = Path(path)
    out = {"path": str(path), "exists": path.is_file()}
    if out["exists"]:
        out.update(bytes=path.stat().st_size, sha256=digest(path))
        if expected is not None:
            out.update(expected_sha256=expected, hash_matches=out["sha256"] == expected)
    elif expected is not None:
        out.update(expected_sha256=expected, hash_matches=False)
    return out


def tensor_info(x):
    return {"shape": list(x.shape), "dtype": str(x.dtype), "device": str(x.device),
            "requires_grad": x.requires_grad,
            "finite": bool(torch.isfinite(x).all()), "numel": x.numel()}


def command(args):
    p = subprocess.run(args, text=True, capture_output=True)
    return {"argv": args, "returncode": p.returncode,
            "stdout": p.stdout.strip(), "stderr": p.stderr.strip()}


def scene_audit(scene, relative_manifest, relative_parent, prior_cameras):
    manifest_path = ROOT / relative_manifest
    parent_path = ROOT / relative_parent
    hist = ROOT / "output/dynamic_sr_20260919" / f"{scene}_sr_w01"
    historical = load_json(hist / "config.json")
    m = load_json(manifest_path)
    base = manifest_path.parent
    ck = torch.load(parent_path, map_location="cpu", weights_only=False)
    model = ck["model"]
    assert len(model) == 14
    opt = model[12]
    deformation = model[2]
    buffers = {"time_poc", "pos_poc", "rotation_scaling_poc", "opacity_poc"}
    deform_names = [k for k in deformation if k not in buffers]
    name_groups = {
        "xyz": ["canonical.xyz"], "f_dc": ["canonical.features_dc"],
        "f_rest": ["canonical.features_rest"], "opacity": ["canonical.opacity"],
        "scaling": ["canonical.scaling"], "rotation": ["canonical.rotation"],
        # get_mlp_parameters() returns deformation_net parameters before timenet,
        # whereas state_dict lists timenet first. Preserve the optimizer order.
        "deformation": ([k for k in deform_names if k.startswith("deformation_net.") and "grid" not in k]
                        + [k for k in deform_names if k.startswith("timenet.")]),
        "grid": [k for k in deform_names if "grid" in k],
    }
    optimizer_groups = []
    for group in opt["param_groups"]:
        names = name_groups[group["name"]]
        assert len(names) == len(group["params"])
        states = []
        for name, pid in zip(names, group["params"]):
            state = opt["state"].get(pid)
            row = {"name": name, "parameter_id": pid, "has_adam_state": state is not None}
            if state is not None:
                if name in deformation:
                    assert state["exp_avg"].shape == deformation[name].shape, name
                row.update(step=float(state["step"]),
                           exp_avg=tensor_info(state["exp_avg"]),
                           exp_avg_sq=tensor_info(state["exp_avg_sq"]),
                           exp_avg_nonzero=int(torch.count_nonzero(state["exp_avg"])),
                           exp_avg_sq_nonzero=int(torch.count_nonzero(state["exp_avg_sq"])))
            states.append(row)
        optimizer_groups.append({"name": group["name"], "lr": group["lr"],
                                 "betas": group["betas"], "eps": group["eps"], "states": states})
    records = [x for x in m["observations"] if x["split"] == "train"]
    prior_ids = [i for i, x in enumerate(records) if x["camera_id"] in prior_cameras]
    assert len(prior_ids) == 240
    assert not set(prior_cameras) & {"cam00", "cam01"}
    training_lr = [check_file(base / x["lr_path"], x["lr_sha256"]) for x in records]
    prior_root = base / "sr_swinir_x4"
    prior_config = load_json(prior_root / "prior_config.json")
    prior_rows = []
    for i in prior_ids:
        x = records[i]
        rel = Path(x["camera_id"]) / Path(x["lr_path"]).name
        receipt = load_json((prior_root / rel).with_suffix(".json"))
        row = check_file(prior_root / rel, receipt["output_sha256"])
        row.update(camera_id=x["camera_id"], frame_index=x["frame_index"],
                   receipt=check_file((prior_root / rel).with_suffix(".json")),
                   receipt_input_matches_manifest=receipt["input_sha256"] == x["lr_sha256"],
                   input_hw=receipt["input_hw"], output_hw=receipt["output_hw"])
        prior_rows.append(row)
    exposure = collections.Counter()
    lr_rng = random.Random(historical["seed"] + 177)
    sr_rng = random.Random(historical["seed"] + 211)
    sample_digest = hashlib.sha256()
    prefix_digests = {}
    first_pairs = []
    for step in range(1, historical["steps"] + 1):
        li, si = lr_rng.randrange(len(records)), sr_rng.choice(prior_ids)
        x = records[si]
        exposure[f'{x["camera_id"]}/{x["frame_index"]}'] += 1
        sample_digest.update(f"{li},{si}\n".encode())
        if step <= 5:
            first_pairs.append({"lr": [records[li]["camera_id"], records[li]["frame_index"]],
                                "sr": [x["camera_id"], x["frame_index"]]})
        if step in [1200, 3000, 6000, 12000, 18000]:
            prefix_digests[str(step)] = sample_digest.hexdigest()
    init = m["initialization"]
    init_report = load_json(base / init["report_path"])
    init_sources = []
    lookup = {(x["camera_id"], x["frame_index"]): x for x in records}
    for row in init_report["source_images"]:
        key = (row["camera_id"], row["frame_index"])
        init_sources.append({"path": row["path"], "train_observation": key in lookup,
                             "receipt_matches_manifest": key in lookup and row["sha256"] == lookup[key]["lr_sha256"],
                             "actual": check_file(base / row["path"], row["sha256"])})
    calibration = Path(m["raw_directory"]) / "poses_bounds.npy"
    camera_checks = []
    scaling = np.array([[0.25, 0, -0.375], [0, 0.25, -0.375], [0, 0, 1.]])
    for camera_id, cam in m["cameras"].items():
        camera_checks.append({"camera": camera_id,
                              "K_hr_to_lr_max_abs": float(np.max(np.abs(scaling @ np.array(cam["K_hr"]) - cam["K_lr"]))),
                              "w2c_c2w_inverse_max_abs": float(np.max(np.abs(np.array(cam["w2c"]) @ np.array(cam["c2w"]) - np.eye(4))))})
    rng = ck["rng"]
    # Validate serializability/restorability on isolated CPU generators only.
    cpu_generator = torch.Generator(device="cpu")
    cpu_generator.set_state(rng["torch"].cpu())
    py_rng = random.Random()
    py_rng.setstate(rng["python"])
    np_rng = np.random.RandomState()
    np_rng.set_state(rng["numpy"])
    source_matches = []
    for source in historical["sources"]:
        current = check_file(source["path"], source["sha256"])
        saved = check_file(source["copy"], source["sha256"])
        source_matches.append({"current": current, "historical_snapshot": saved})
    return {
        "parent": check_file(parent_path, historical["parent_sha256"]),
        "manifest": check_file(manifest_path, historical["manifest_sha256"]),
        "parent_manifest_matches": ck["metadata"]["manifest_sha"] == digest(manifest_path),
        "checkpoint_keys": list(ck), "metadata": ck["metadata"],
        "actual_hidden": ck["hidden"], "actual_optimization_config": ck["optim"],
        "active_sh_degree": model[0], "points": model[1].shape[0],
        "canonical_tensors": {name: tensor_info(model[i]) for name, i in
                              [("xyz", 1), ("features_dc", 4), ("features_rest", 5),
                               ("scaling", 6), ("rotation", 7), ("opacity", 8)]},
        "optimizer_groups": optimizer_groups, "optimizer_state_count": len(opt["state"]),
        "global_rng": {"keys": list(rng), "torch_state": tensor_info(rng["torch"]),
                       "cuda_states_saved": len(rng["cuda"]),
                       "cpu_isolated_rng_restore_pass": True,
                       "cuda_rng_restore_tested": False},
        "samplers_saved_in_parent": "samplers" in ck,
        "scheduler": {"serialized_scheduler_object": False, "functional_schedule_from_optim": True,
                      "offset": ck["metadata"]["step"], "max_steps": ck["optim"]["position_lr_max_steps"],
                      "cumulative_coarse_plus_fine_plus_adapter": 8200},
        "data_protocol": {k: m[k] for k in ["scene", "source", "frame_indices", "time_convention", "resolutions", "degradation", "splits", "pixel_coordinate_convention"]},
        "training_lr": {"count": len(training_lr), "all_hashes_match": all(x["hash_matches"] for x in training_lr), "files": training_lr},
        "prior": {"config": prior_config, "config_identity": check_file(prior_root / "prior_config.json"),
                  "manifest_matches": prior_config["manifest_sha256"] == digest(manifest_path),
                  "checkpoint": check_file(prior_config["checkpoint"], prior_config["checkpoint_sha256"]),
                  "network_source": check_file(ROOT.parent / "mml/scripts/network_swinir.py", prior_config["network_sha256"]),
                  "generator_current": check_file(ROOT / "experiments/dynamic_sr_20260918/generate_prior.py", prior_config["generator_sha256"]),
                  "count": len(prior_rows), "cameras": prior_cameras,
                  "all_output_hashes_match": all(x["hash_matches"] for x in prior_rows),
                  "all_receipt_inputs_match_manifest": all(x["receipt_input_matches_manifest"] for x in prior_rows),
                  "files": prior_rows},
        "historical_A_sampling": {"config": check_file(hist / "config.json"), "seed": historical["seed"],
                                  "record_order": "manifest observations filtered to train, preserving order",
                                  "first_five_pairs": first_pairs, "prefix_sequence_sha256": prefix_digests,
                                  "reconstructed_full_exposure_matches": dict(exposure) == load_json(hist / "exposure.json"),
                                  "reconstruction_limit": "Independent samplers absent from checkpoint; reconstructed sequence is not a saved historical step log."},
        "historical_A_sources": source_matches,
        "initialization": {"npz": check_file(base / init["npz_path"], init["npz_sha256"]),
                           "ply": check_file(base / init["ply_path"]),
                           "report_identity": check_file(base / init["report_path"]),
                           "report": {k: v for k, v in init_report.items() if k not in ["source_images", "pair_statistics"]},
                           "source_files": init_sources,
                           "all_source_train_LR_hashes_match": all(x["train_observation"] and x["receipt_matches_manifest"] and x["actual"]["hash_matches"] for x in init_sources)},
        "calibration": {"file": check_file(calibration, m["poses_bounds_sha256"]),
                        "policy": "Official supplied calibration used; not re-estimated from training LR. Original calibration image sources not independently established by this audit.",
                        "camera_checks": camera_checks,
                        "time_values_match_frame_over_300": all(abs(x["time"] - x["frame_index"] / 300.) < 1e-12 for x in m["observations"])},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "experiments/dynamic_sr_20260923/p0_assets_audit.json")
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("Run with CUDA_VISIBLE_DEVICES='' for this CPU-only audit")
    if args.out.exists():
        raise FileExistsError(args.out)
    torch.set_num_threads(2)
    start = time.monotonic()
    result = {"schema": 2,
              "supersedes": "p0_assets_audit.json: draft optimizer parameter names followed state_dict order; corrected to get_mlp_parameters order and checked state shapes",
              "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "source": check_file(__file__), "python": sys.executable, "torch": torch.__version__,
              "device": "CPU only; map_location=cpu; no CUDA API/model/render calls",
              "hr_image_bytes_read": False,
              "scope": "Existing assets audited; runtime gradient and CUDA reload validation remain separate",
              "upstream_commit": command(["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"]),
              "upstream_status": command(["git", "-C", str(UPSTREAM), "status", "--short"]),
              "scenes": {}}
    specs = [
        ("cook_spinach", "data/dynamic_sr/n3dv_prepared/cook_spinach/manifest.json",
         "output/dynamic_sr_20260918/cook_spinach_pilot_v1_lr_integrated/checkpoint_final.pt",
         ["cam02", "cam06", "cam12", "cam18"]),
        ("meetroom_discussion", "data/dynamic_sr/meetroom_prepared/discussion/manifest.json",
         "output/dynamic_sr_20260919/meetroom_discussion_integrated_parent/checkpoint_final.pt",
         ["cam02", "cam04", "cam08", "cam12"]),
    ]
    for spec in specs:
        result["scenes"][spec[0]] = scene_audit(*spec)
    relevant = [ROOT / "experiments/dynamic_sr_20260918" / name for name in
                ["common.py", "n3dv_data.py", "run_experiment.py", "generate_prior.py", "prepare_n3dv.py"]]
    relevant += [ROOT / "experiments/dynamic_sr_20260919" / name for name in ["controlled_fit.py", "prepare_meetroom.py"]]
    relevant += [UPSTREAM / "scene/gaussian_model.py", UPSTREAM / "scene/deformation.py", UPSTREAM / "gaussian_renderer/__init__.py"]
    result["relevant_current_sources"] = [check_file(p) for p in relevant]
    result["elapsed_seconds"] = time.monotonic() - start
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "sha256": digest(args.out),
                      "elapsed_seconds": result["elapsed_seconds"],
                      "scenes": {k: {"parent_pass": v["parent"]["hash_matches"],
                                       "LR_pass": v["training_lr"]["all_hashes_match"],
                                       "prior_pass": v["prior"]["all_output_hashes_match"],
                                       "sampler_exposure_pass": v["historical_A_sampling"]["reconstructed_full_exposure_matches"]}
                                 for k, v in result["scenes"].items()}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
