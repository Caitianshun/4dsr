"""Sequence Matters on synchronous training-camera images, then a shared Wu4DGS.

The PSRT network and ORB/greedy implementation are loaded from the pinned author
checkout, not vendored. Run this module in an isolated teacher process so its
``archs``/``basicsr`` modules cannot shadow Wu's modules. No CUDA work on import.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

UPSTREAM_COMMIT = "06d9e7fa359f65e6fdb6b96e48786d6028de0caf"
UPSTREAM_URL = "https://github.com/DHPark98/SequenceMatters"


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_upstream(root, commit=UPSTREAM_COMMIT):
    root = Path(root).resolve()
    actual = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if actual != commit:
        raise ValueError(f"Unreviewed upstream revision {actual}; expected {commit}")
    if subprocess.run(["git", "-C", str(root), "diff", "--quiet", "HEAD"]).returncode:
        raise ValueError("Tracked upstream source changed; re-audit before use")
    return root


def source_objects(path, names, namespace):
    """Execute selected definitions verbatim, excluding unrelated import side effects."""
    tree = ast.parse(Path(path).read_text())
    selected = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    if {node.name for node in selected} != set(names):
        raise ValueError(f"Missing upstream definitions in {path}: {names}")
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def validate_records(records, held_out_cameras):
    """Require one scene; reject held-out input rather than silently filtering it."""
    if not records:
        raise ValueError("Empty training input")
    seen = set()
    scenes = set()
    triples = set()
    for record in records:
        required = {"key", "scene", "camera", "time", "split", "lr_path"}
        if not required <= record.keys():
            raise ValueError(f"Missing record fields: {required - record.keys()}")
        if record["split"] != "train" or record["camera"] in held_out_cameras or record["camera"] == "cam00":
            raise ValueError(f"Evaluation camera in teacher input: {record['key']}")
        key_path = Path(record["key"])
        if (record["key"] in seen or not record["key"] or key_path.is_absolute()
                or any(part in (".", "..") for part in record["key"].split("/"))
                or "\\" in record["key"] or "//" in record["key"]):
            raise ValueError("Keys must be unique safe relative camera/frame paths")
        triple = (record["scene"], record["camera"], record["time"])
        if triple in triples:
            raise ValueError(f"Duplicate camera-time record: {triple}")
        seen.add(record["key"])
        triples.add(triple)
        scenes.add(record["scene"])
    if len(scenes) != 1:
        raise ValueError("Teacher preparation is per scene; cross-scene input is prohibited")


def group_synchronous(records, held_out_cameras):
    validate_records(records, held_out_cameras)
    groups = {}
    for record in records:
        groups.setdefault(record["time"], []).append(record)
    for time, group in groups.items():
        if len(group) < 2:
            raise ValueError(f"Time {time}: Sequence Matters requires multiple training cameras")
        if any("camera_center" not in record for record in group):
            raise ValueError("Pose ordering requires calibrated camera_center in a common world frame")
    return groups


def camera_rankings(centers):
    """Author 'pose' metric: angle between origin-to-camera vectors, not distance."""
    import numpy as np
    centers = np.asarray(centers, dtype=np.float64)
    if centers.ndim != 2 or centers.shape[1] != 3 or not np.isfinite(centers).all():
        raise ValueError("Expected finite N by 3 camera centers")
    norms = np.linalg.norm(centers, axis=1)
    if (norms == 0).any():
        raise ValueError("Camera at world origin makes the original angular pose metric undefined")
    cosines = (centers @ centers.T) / (norms[:, None] * norms[None, :])
    angles = np.degrees(np.arccos(np.clip(cosines, -1, 1)))
    np.fill_diagonal(angles, 0)
    order = np.argsort(angles, axis=1)
    ranks = np.argsort(order, axis=1)
    return np.stack([ranks, angles], axis=-1)


def als_plan(position_rankings, feature_rankings, order_function, thresholds=(30, 50), max_sequence_length=8):
    """Author Mip360 ALS scheduling, including first-writer and endpoint rules."""
    count = len(position_rankings)
    if count < 2 or max_sequence_length < 3:
        raise ValueError("Need multiple views and sequence capacity >=3")
    # Upstream appends N to [30,50], even when N < 30; retain this behavior.
    thresholds = list(thresholds) + [count]
    created = set()
    result = []
    for threshold_index, threshold in enumerate(thresholds):
        final = threshold_index == len(thresholds) - 1
        if len(created) == count:
            break
        for reference in range(count):
            indices = order_function(list(range(count)), reference, threshold,
                                     position_rankings, feature_rankings, "pose")
            indices = [int(index) for index in indices]
            if len(indices) <= 2 and not final:
                continue
            indices = indices[:max_sequence_length]
            if all(index in created for index in indices):
                continue
            outputs = [index for offset, index in enumerate(indices)
                       if index not in created and (final or offset not in (0, len(indices) - 1))]
            if outputs:
                result.append({"threshold": threshold, "reference": reference,
                               "indices": indices, "output_indices": outputs})
                created.update(outputs)
    if len(created) != count:
        raise RuntimeError(f"ALS did not cover all views: {created}")
    return result


def sequence_matters_loss(rendered_hr, teacher_hr, observed_lr, ssim_fn, lambda_tex=0.4, lambda_dssim=0.2, lr_operator="bicubic"):
    import torch.nn.functional as F
    if rendered_hr.shape != teacher_hr.shape:
        raise ValueError("Teacher and render size differ")
    if tuple(rendered_hr.shape[-2:]) != tuple(4 * x for x in observed_lr.shape[-2:]):
        raise ValueError("Pinned PSRT protocol requires exactly x4")
    if lr_operator not in ("bicubic", "area"):
        raise ValueError("LR operator must be explicit: native bicubic or area protocol adaptation")
    kwargs = {"align_corners": False, "antialias": True} if lr_operator == "bicubic" else {}
    reduced = F.interpolate(rendered_hr.unsqueeze(0), size=observed_lr.shape[-2:], mode=lr_operator, **kwargs).squeeze(0)
    def photo(left, right):
        return (1 - lambda_dssim) * (left - right).abs().mean() + lambda_dssim * (1 - ssim_fn(left, right))
    hr = photo(rendered_hr, teacher_hr.detach())
    lr = photo(reduced, observed_lr.detach())
    return lambda_tex * hr + (1 - lambda_tex) * lr, {"hr": hr.detach(), "lr": lr.detach()}


def validate_sequence_plan(receipt, manifest, source):
    """Bind a CPU plan to the exact ordered inputs and author ALS output rules."""
    if receipt.get("status") != "plan_only" or receipt.get("targets") or receipt.get("target_sha256"):
        raise ValueError("Expected a completed CPU plan without teacher targets")
    if receipt.get("upstream_commit") != UPSTREAM_COMMIT or receipt.get("source_sha256") != sha256(source):
        raise ValueError("Sequence plan author source identity changed")
    manifest_hash = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    if receipt.get("manifest_sha256") != manifest_hash:
        raise ValueError("Sequence plan manifest identity changed")
    expected_groups = group_synchronous(manifest["records"], manifest["heldout_cameras"])
    saved_groups = receipt.get("groups", [])
    if len(saved_groups) != len(expected_groups):
        raise ValueError("Sequence plan group coverage differs")
    for saved, (time, records) in zip(saved_groups, expected_groups.items()):
        keys = [r["key"] for r in records]
        if saved.get("time") != time or saved.get("keys") != keys:
            raise ValueError("Sequence plan time or ordered camera keys differ")
        hashes = {r["key"]: sha256(r["lr_path"]) for r in records}
        if saved.get("lr_sha256") != hashes:
            raise ValueError("Sequence plan observed LR bytes changed")
        for record in records:
            if "lr_sha256" in record and record["lr_sha256"] != hashes[record["key"]]:
                raise ValueError("Registered observed LR identity changed")
        count = len(keys)
        created = set()
        last_phase = -1
        last_reference = -1
        thresholds = [30, 50, count]
        for sequence in saved.get("sequences", []):
            indices, outputs = sequence.get("indices", []), sequence.get("output_indices", [])
            reference = sequence.get("reference")
            threshold = sequence.get("threshold")
            if (type(reference) is not int or not 0 <= reference < count or not 1 <= len(indices) <= 8
                    or any(type(index) is not int or not 0 <= index < count for index in indices)
                    or len(set(indices)) != len(indices) or indices[0] != reference
                    or any(type(index) is not int for index in outputs)):
                raise ValueError("Sequence plan has invalid camera indices")
            # A threshold may occur twice for N=30/50. Find the first phase
            # consistent with the author's increasing reference scan.
            candidates = [i for i, value in enumerate(thresholds) if value == threshold
                          and (i > last_phase or (i == last_phase and reference > last_reference))]
            if not candidates:
                raise ValueError("Sequence plan threshold/reference order differs")
            phase = candidates[0]
            final = phase == 2
            expected_outputs = [index for offset, index in enumerate(indices)
                                if index not in created and (final or offset not in (0, len(indices)-1))]
            # Disambiguate the repeated N threshold using its final endpoint rule.
            if outputs != expected_outputs and 2 in candidates:
                phase, final = 2, True
                expected_outputs = [index for index in indices if index not in created]
            if outputs != expected_outputs or not outputs or (len(indices) <= 2 and not final):
                raise ValueError("Sequence plan violates first-writer or endpoint output rules")
            created.update(outputs)
            last_phase, last_reference = phase, reference
        if created != set(range(count)):
            raise ValueError("Sequence plan does not cover every training camera exactly once")
    return saved_groups


def prepare_sequences(manifest, upstream, output, psrt_checkpoint=None, spynet_checkpoint=None, plan_only=False, device="cuda", sequence_plan=None):
    """Create a real PSRT target cache; plan_only never claims a teacher was run."""
    import numpy as np
    import cv2
    upstream = verify_upstream(upstream)
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Use a fresh output directory: {output}")
    groups = group_synchronous(manifest["records"], manifest["heldout_cameras"])
    if not plan_only:
        for checkpoint in (psrt_checkpoint, spynet_checkpoint):
            if checkpoint is None or not Path(checkpoint).is_file():
                raise FileNotFoundError(f"Required pretrained PSRT/SpyNet checkpoint missing: {checkpoint}")
    output.mkdir(parents=True, exist_ok=True)
    source = upstream / "vsr/utils_vsr.py"
    cached_groups = None
    if sequence_plan is not None:
        sequence_plan = Path(sequence_plan).resolve()
        cached_groups = validate_sequence_plan(json.loads(sequence_plan.read_text()), manifest, source)
    namespace = None if cached_groups is not None else source_objects(source,
        ["compute_features", "compute_similarity", "compute_feature_distance_ranking", "ordering_sim1_thresholding_sim2"],
        {"np": np, "cv2": cv2})
    receipt = {"status": "plan_only" if plan_only else "running", "upstream_commit": UPSTREAM_COMMIT,
               "upstream_url": UPSTREAM_URL, "source_sha256": sha256(source), "groups": [], "targets": {},
               "target_sha256": {},
               "teacher_generated": False,
               "protocol": "same-time cross-camera ALS; PSRT frozen; BGR author preprocessing",
               "manifest_sha256": hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()}
    if cached_groups is not None:
        receipt["sequence_plan"] = {"path": str(sequence_plan), "sha256": sha256(sequence_plan)}
    model = None
    if not plan_only:
        import torch
        # This function is intended for a separate process, not the Wu process.
        sys.path.insert(0, str(upstream / "vsr/psrt"))
        from archs.psrt_recurrent_arch import BasicRecurrentSwin
        from basicsr.utils import tensor2img
        from PIL import Image
        loader = source_objects(source, ["load_vsr_model"], {"torch": torch, "BasicRecurrentSwin": BasicRecurrentSwin})
        model = loader["load_vsr_model"](str(spynet_checkpoint), str(psrt_checkpoint), device=device)
        checkpoint_state = torch.load(psrt_checkpoint, map_location="cpu", weights_only=False)["params"]
        missing_parameters = set(dict(model.named_parameters())) - set(checkpoint_state)
        if missing_parameters:
            raise ValueError(f"PSRT checkpoint leaves parameters uninitialized: {sorted(missing_parameters)[:12]}")
        receipt["weight_key_audit"] = {"missing_parameters": [],
            "unexpected_keys": sorted(set(checkpoint_state) - set(model.state_dict()))}
        del checkpoint_state
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        receipt["weights"] = {str(Path(p).resolve()): sha256(p) for p in (psrt_checkpoint, spynet_checkpoint)}
    try:
        for group_index, (time, records) in enumerate(groups.items()):
            for record in records:
                if "lr_sha256" in record and sha256(record["lr_path"]) != record["lr_sha256"]:
                    raise ValueError(f"Observed LR identity changed: {record['key']}")
            images = [cv2.imread(str(record["lr_path"])) for record in records]
            if any(image is None for image in images):
                raise FileNotFoundError("Cannot load one or more observed training LR images")
            if len({image.shape for image in images}) != 1:
                raise ValueError("PSRT sequence needs identical LR sizes; do not silently resize calibrated observations")
            if min(images[0].shape[:2]) < 64 and not plan_only:
                raise ValueError("Author PSRT optical-flow branch requires LR sides >=64 pixels")
            if cached_groups is None:
                positions = camera_rankings([record["camera_center"] for record in records])
                features = np.asarray(namespace["compute_feature_distance_ranking"](images))
                plan = als_plan(positions, features, namespace["ordering_sim1_thresholding_sim2"])
            else:
                plan = cached_groups[group_index]["sequences"]
            receipt["groups"].append({"time": time, "keys": [r["key"] for r in records], "sequences": plan,
                                      "lr_sha256": {r["key"]: sha256(r["lr_path"]) for r in records}})
            if model is None:
                continue
            for sequence in plan:
                selected = sequence["indices"]
                batch = torch.from_numpy(np.stack([images[i] for i in selected]).transpose(0, 3, 1, 2)).float()
                with torch.inference_mode():
                    prediction = model(batch.div(255).unsqueeze(0).to(device)).squeeze(0)
                for index in sequence["output_indices"]:
                    image = tensor2img(prediction[selected.index(index)], rgb2bgr=True, min_max=(0, 1))
                    expected = (images[index].shape[0] * 4, images[index].shape[1] * 4)
                    if image.shape[:2] != expected:
                        raise RuntimeError("PSRT did not produce x4 output")
                    target = output / (records[index]["key"] + ".png")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(image).save(target)
                    receipt["targets"][records[index]["key"]] = str(target)
                    receipt["target_sha256"][records[index]["key"]] = sha256(target)
        if model is not None:
            if len(receipt["targets"]) != len(manifest["records"]):
                raise RuntimeError("Teacher cache incomplete")
            receipt["status"] = "teacher_complete"
            receipt["teacher_generated"] = True
    except Exception as error:
        receipt["status"] = "failed"
        receipt["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        (output / "receipt.json").write_text(json.dumps(receipt, indent=2))
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--psrt-checkpoint")
    parser.add_argument("--spynet-checkpoint")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sequence-plan", help="Previously generated CPU plan receipt; verifies all inputs and skips ORB/ALS")
    args = parser.parse_args()
    receipt = prepare_sequences(json.loads(Path(args.manifest).read_text()), args.upstream, args.output,
                                args.psrt_checkpoint, args.spynet_checkpoint, args.plan_only, args.device, args.sequence_plan)
    print(json.dumps({"status": receipt["status"], "targets": len(receipt["targets"])}))


if __name__ == "__main__":
    main()
