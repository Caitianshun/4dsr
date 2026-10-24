"""IE-on-Wu external priors from existing area-matched SwinIR RGB8 PNGs.

Only Depth Anything V2-L is loaded. This is an explicitly labelled shared-RGB
adaptation, not the original float-SwinIR producer. Default CLI operation is a
CPU plan; ``external --run`` belongs inside normal GPU resource admission.
The frozen IE internal producer and training losses are unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.static_sr_shared_teacher_20261024 import shared_cache as cache
from experiments.static_sr_wu_20261022 import common

COMMIT = "74d60d5bb0f49974ac0efa3497dbbd9e246480f4"
DEPTH_SHA = "a7ea19fa0ed99244e67b624c72b8580b7e9553043245905be58796a608eb9345"
DEPTH_NAME = "depth_anything_v2_vitl.pth"
DEPTH_ARCH = dict(encoder="vitl", features=256, out_channels=[256, 512, 1024, 1024])
DEPTH_PREFIX = "submodules/Depth-Anything-V2"
FORMAL_KIND = "iesrgs_shared_swinir_external"
COMPLETE_KIND = "iesrgs_shared_swinir_complete"
FIXTURE_SCHEMA = "iesrgs_shared_swinir_depth_fixture_v1"
CACHE_FORMATS = ("npy", "npz_deflate", "npz_byte_shuffle")
DEFAULT_UPSTREAM = ROOT / "output/static_sr_wu_20261024/next_teacher_gates_cpu_v1/iesrgs_clean_74d60d5"
DEFAULT_WEIGHTS = ROOT / "weight/static_sr_wu_20261022/depth_anything_v2_large" / DEPTH_NAME


def fresh_output(value, project_root=ROOT):
    """Additive task output only; no symlink, traversal, or existing destination."""
    root = Path(project_root).resolve()
    p = cache.path(value, project_root=root)
    if not p.is_relative_to(root / "output") or p == root / "output":
        raise ValueError("IE output must be a new child of this project's output directory")
    if p.exists():
        raise FileExistsError(f"IE output already exists; no replay: {p}")
    return p


def depth_assets(upstream, weights, project_root=ROOT):
    """Verify official Git blobs as well as HEAD/clean status and exact weights.

    Git trust is restricted to this subprocess's exact checkout. Nothing is
    written to a repository or global Git configuration, including tracked pyc.
    """
    root = cache.path(upstream, project_root=project_root)
    weight = cache.path(weights, project_root=project_root)
    git = ["git", "-c", f"safe.directory={root}", "-C", str(root)]
    env = dict(os.environ, GIT_OPTIONAL_LOCKS="0")
    def command(*args):
        return subprocess.check_output([*git, *args], env=env)
    if command("rev-parse", "HEAD").decode().strip() != COMMIT:
        raise ValueError("Depth source is not the pinned IE-SRGS commit")
    if command("status", "--porcelain").strip():
        raise ValueError("Pinned IE-SRGS source must remain completely clean")
    sources = []
    listing = command("ls-tree", "-rz", "HEAD", "--", DEPTH_PREFIX)
    for record in listing.split(b"\0"):
        if not record:
            continue
        meta, name = record.split(b"\t", 1)
        mode, kind, blob = meta.decode().split()
        relative = name.decode()
        if not (relative.startswith(DEPTH_PREFIX + "/depth_anything_v2/") and relative.endswith(".py")
                or relative == DEPTH_PREFIX + "/run.py"):
            continue
        if mode != "100644" or kind != "blob":
            raise ValueError("Depth Python sources must be ordinary tracked files")
        expected = hashlib.sha256(command("cat-file", "blob", blob)).hexdigest()
        item = cache.identity(root / relative, project_root=project_root)
        if item["sha256"] != expected:
            raise ValueError("Depth Python differs from the pinned Git object")
        sources.append(item)
    required = root / DEPTH_PREFIX / "depth_anything_v2/dpt.py"
    if not sources or cache.identity(required, project_root=project_root) not in sources:
        raise ValueError("Pinned Depth Anything source closure is incomplete")
    weight_id = cache.identity(weight, project_root=project_root)
    if weight.name != DEPTH_NAME or weight_id["sha256"] != DEPTH_SHA:
        raise ValueError("Exact official Depth Anything V2-Large vitl weights required")
    return dict(upstream=str(root), upstream_commit=COMMIT, architecture=DEPTH_ARCH,
                weights=weight_id, sources=sources)


def _aligned_rows(shared, manifest):
    """Check complete legal key/time/LR alignment independently of cache import."""
    train = manifest["splits"]["train"]
    if "cam00" in train or set(train) & set(manifest["splits"].get("test", [])):
        raise ValueError("Illegal training/held-out split")
    expected = {(c, f) for c in train for f in manifest["frame_indices"]}
    observations = {(o["camera_id"], o["frame_index"]): o for o in manifest["observations"] if o["split"] == "train"}
    entries = {(e["camera"], e["frame"]): e for e in shared["entries"]}
    if len(entries) != len(shared["entries"]) or set(entries) != expected or set(observations) != expected:
        raise ValueError("Shared RGB must cover every legal key exactly once")
    rows = []
    for k in sorted(expected, key=lambda k: (k[1], k[0])):
        e, o = entries[k], observations[k]
        if (e["key"] != f"{k[0]}/{k[1]:04d}" or e["lr_sha256"] != o["lr_sha256"]
                or e["time"] != o["time"] or abs(e["time"] - k[1] / 300.) > 1e-8
                or e["shape"] != manifest["resolutions"]["hr"]):
            raise ValueError("Shared RGB observation identity/time/shape differs")
        rows.append(e)
    return rows


def build_plan(config_path, shared_cache, upstream, weights, out, *, fixture_key=None,
               cache_format="npy", project_root=ROOT):
    if cache_format not in CACHE_FORMATS:
        raise ValueError("Unknown lossless depth cache format")
    output = fresh_output(out, project_root)
    config = cache.read(config_path, project_root=project_root)
    if config.get("schema") != common.SCHEMA or config.get("method") != "iesrgs":
        raise ValueError("A registered IE-on-Wu configuration is required")
    manifest_path = cache.checked(config["manifest"], project_root=project_root)
    manifest = cache.read(manifest_path, project_root=project_root)
    shared = cache.verify_cache(shared_cache, manifest_path, project_root=project_root)
    if shared.get("tag") != "sr_swinir_x4_areaft" or config["scene"] != manifest["scene"]:
        raise ValueError("This shared IE variant requires the areaft cache for this scene")
    rows = _aligned_rows(shared, manifest)
    if fixture_key is not None:
        selected = [e for e in rows if e["key"] == fixture_key]
        if len(selected) != 1:
            raise ValueError("Fixture must name exactly one legal training key")
    else:
        selected = rows
    assets = depth_assets(upstream, weights, project_root)
    return dict(schema="iesrgs_shared_swinir_depth_plan_v1", status="not_GPU_executed",
                config=cache.identity(config_path, project_root=project_root),
                manifest=config["manifest"], scene=manifest["scene"],
                shared_cache=cache.identity(shared_cache, project_root=project_root),
                source=assets, producer=cache.identity(__file__, project_root=ROOT),
                output=str(output), fixture_key=fixture_key, cache_format=cache_format,
                legal_key_count=len(rows), selected_key_count=len(selected),
                selected_keys=[e["key"] for e in selected], project_root=str(Path(project_root).resolve()),
                RGB_semantics="exact_existing_RGB8_PNG_decoded_float32_div255",
                depth_semantics="same_PNG_BGR_infer_image_518_per_image_minmax_uint8_div65536",
                SwinIR_loads=0, SwinIR_inferences=0, parameter_updates=0)


def load_depth_only(assets, device="cuda", project_root=ROOT):
    """Strict-load only the pinned vitl architecture; no SwinIR import/load."""
    import torch
    verified = depth_assets(assets["upstream"], assets["weights"]["path"], project_root)
    if verified != assets:
        raise ValueError("Depth source or weights changed after planning")
    package_root = Path(assets["upstream"]) / DEPTH_PREFIX
    source_ids = {str(cache.checked(s, project_root=project_root)): s for s in assets["sources"]}
    namespace_dirs = {str(Path(p).parent) for p in source_ids}
    namespace_dirs.add(str(package_root / "depth_anything_v2"))
    def module_path(module):
        filename = getattr(module, "__file__", None)
        if filename is not None:
            p = str(Path(filename).resolve())
            if p in source_ids:
                return p
        else:
            paths = [str(Path(p).resolve()) for p in getattr(module, "__path__", [])]
            if len(paths) == 1 and paths[0] in namespace_dirs:
                return None
        raise RuntimeError("A different Depth Anything package is already loaded")
    for name, module in list(sys.modules.items()):
        if name == "depth_anything_v2" or name.startswith("depth_anything_v2."):
            module_path(module)
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(package_root))
    try:
        module = importlib.import_module("depth_anything_v2.dpt")
        model = module.DepthAnythingV2(**DEPTH_ARCH)
        payload = torch.load(cache.checked(assets["weights"], project_root=project_root),
                             map_location="cpu", weights_only=True)
        model.load_state_dict(payload, strict=True)
        del payload
        model.requires_grad_(False).eval().to(device)
    finally:
        sys.path.remove(str(package_root))
        sys.dont_write_bytecode = previous
    imported = []
    for name, module in list(sys.modules.items()):
        if name == "depth_anything_v2" or name.startswith("depth_anything_v2."):
            path = module_path(module)
            if path is None:
                imported.append(dict(module=name, namespace_paths=list(module.__path__)))
                continue
            if cache.identity(path, project_root=project_root) != source_ids[path]:
                raise RuntimeError("Imported Depth Anything source differs from verified closure")
            imported.append(dict(module=name, **source_ids[path]))
    return model, dict(strict_load=True, architecture=DEPTH_ARCH, weights=assets["weights"],
                       imported_modules=imported, device=device, SwinIR_loaded=False)


def png_inverse_depth(sr_identity, expected_shape, depth_model, project_root=ROOT):
    """Author depth transform, preserving the existing PNG bytes and RGB values."""
    import numpy as np
    from PIL import Image
    path = cache.checked(sr_identity, project_root=project_root)
    with Image.open(path) as image:
        if image.format != "PNG" or image.mode != "RGB" or list(image.size) != expected_shape:
            raise ValueError("Shared RGB must be an exact-size RGB8 PNG")
        rgb = np.array(image, dtype=np.uint8, copy=True)
    inverse = np.asarray(depth_model.infer_image(rgb[:, :, ::-1].copy(), 518))
    if inverse.shape != rgb.shape[:2] or not np.isfinite(inverse).all():
        raise ValueError("Invalid Depth Anything output shape or finite values")
    low, high = float(inverse.min()), float(inverse.max())
    if high <= low:
        raise ValueError("Constant monocular depth cannot provide Pearson guidance")
    inverse_u8 = ((inverse - low) / (high - low) * 255).astype(np.uint8)
    return (inverse_u8.astype(np.float32) / 65536.)[None]


def _write_rows(plan, shared, rows, depth_model, output, model_receipt, runtime_probe=None):
    import torch
    from experiments.static_sr_wu_20261022.methods.iesrgs_priors import _save_array
    project_root = Path(plan["project_root"])
    fixture = plan["fixture_key"] is not None
    result = dict(schema=FIXTURE_SCHEMA if fixture else "static_sr_wu_prior_bundle_v1",
                  status="completed_fixture" if fixture else "completed",
                  kind="iesrgs_shared_swinir_fixture" if fixture else FORMAL_KIND, method="iesrgs",
                  teacher_variant="shared_swinir_areaft_RGB8", fixture=fixture, training_only=True,
                  scene=plan["scene"], manifest=plan["manifest"], upstream_commit=COMMIT,
                  shared_cache=plan["shared_cache"], depth_load=model_receipt,
                  model_provenance=shared["model_provenance"],
                  model_identities=shared["model_identities"] + [plan["source"]["weights"]],
                  producer=shared["producer"] + [cache.identity(__file__, project_root=ROOT),
                      common.identity(ROOT / "experiments/static_sr_wu_20261022/methods/iesrgs_priors.py"),
                      common.identity(ROOT / "experiments/static_sr_wu_20261022/common.py")]
                      + plan["source"]["sources"],
                  cache_formats=dict(external_rgb="existing_RGB8_PNG", external_inverse_depth=plan["cache_format"]),
                  RGB_semantics=plan["RGB_semantics"], depth_semantics=plan["depth_semantics"],
                  parameter_updates=0, SwinIR_loads=0, SwinIR_inferences=0,
                  HR_images_read=0, heldout_images_read=0, entries=[])
    for e in rows:
        inverse = png_inverse_depth(e["sr"], e["shape"], depth_model, project_root)
        depth = _save_array(output, e["key"] + "_external_inverse_depth.npy",
                            torch.from_numpy(inverse), plan["cache_format"])
        result["entries"].append(dict(key=e["key"], camera=e["camera"], frame=e["frame"],
                                      time=e["time"], lr_sha256=e["lr_sha256"],
                                      shared_sr_sha256=e["sr"]["sha256"], external_rgb=e["sr"],
                                      external_inverse_depth=depth))
    if len(result["entries"]) != plan["selected_key_count"]:
        raise ValueError("Incomplete shared depth generation")
    if runtime_probe is not None:
        result["execution"] = runtime_probe()
    filename = "fixture_receipt.json" if fixture else "index.json"
    common.write_new(output / filename, result)
    return output / filename


def _gpu_execution_probe():
    """Called only by explicit --run, after resource admission and CPU validation."""
    import torch
    if torch.cuda.device_count() != 1:
        raise ValueError("Depth generation requires exactly one visible admitted GPU")
    prop = torch.cuda.get_device_properties(0)
    hardware = dict(host=socket.gethostname(), CUDA_VISIBLE_DEVICES=os.environ["CUDA_VISIBLE_DEVICES"],
                    uuid=str(getattr(prop, "uuid", "unavailable")), name=prop.name,
                    total_memory_bytes=prop.total_memory, compute_capability=[prop.major, prop.minor],
                    torch=torch.__version__, cuda=torch.version.cuda, python=sys.executable)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat()
    def snapshot():
        torch.cuda.synchronize()
        return dict(GPU_executed=True, hardware=hardware, started_utc=started_utc,
                    completed_utc=datetime.now(timezone.utc).isoformat(),
                    elapsed_seconds=time.perf_counter() - started,
                    timing_scope="strict_model_load_plus_depth_inference_and_serialization",
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved(), parameter_updates=0)
    return snapshot


def produce(plan):
    """One model instance; one image at a time; failure leaves an unreplayed run."""
    project_root = Path(plan["project_root"])
    current = build_plan(cache.checked(plan["config"], project_root=project_root),
        cache.checked(plan["shared_cache"], project_root=project_root), plan["source"]["upstream"],
        plan["source"]["weights"]["path"], plan["output"], fixture_key=plan["fixture_key"],
        cache_format=plan["cache_format"], project_root=project_root)
    if current != plan:
        raise ValueError("Shared depth plan changed")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible or "," in visible or visible == "-1":
        raise ValueError("GPU production requires one explicitly admitted visible GPU")
    # build_plan above just revalidated the entire cache. Recheck its exact
    # index bytes here; each selected PNG is checked again immediately at use.
    shared = cache.read(cache.checked(plan["shared_cache"], project_root=project_root), project_root=project_root)
    rows = [e for e in shared["entries"] if e["key"] in set(plan["selected_keys"])]
    output = fresh_output(plan["output"], project_root)
    output.mkdir(parents=True, exist_ok=False)
    common.write_new(output / "plan.json", plan)
    probe = None
    try:
        probe = _gpu_execution_probe()
        model, load_receipt = load_depth_only(plan["source"], project_root=project_root)
        return _write_rows(plan, shared, rows, model, output, load_receipt, probe)
    except BaseException as error:
        failed = dict(status="failed_no_replay", error_type=type(error).__name__, error=str(error))
        if probe is not None:
            try:
                failed["execution"] = probe()
            except Exception as probe_error:
                failed["execution_probe_error"] = str(probe_error)
        common.write_new(output / "failure.json", failed)
        raise


def _internal_checkpoint(internal, manifest_id, project_root):
    teacher = internal.get("internal_teacher", {})
    expected = dict(representation="Wu_4DGS_with_Mip_filter", depth="RADE_median_depth",
                    filter_reference_resolution="lr", frozen=True, same_time_training_views_only=True)
    if any(teacher.get(k) != v for k, v in expected.items()):
        raise ValueError("A genuine IE internal Wu+Mip median-depth teacher is required")
    for name in ("iesrgs_priors.py", "iesrgs.py", "iesrgs_renderer.py"):
        p = Path(project_root) / "experiments/static_sr_wu_20261022/methods" / name
        if not any(cache.checked(x, project_root=project_root) == p for x in internal["producer"]):
            raise ValueError("Internal IE source producer identity is missing")
    checkpoint = cache.checked(teacher["checkpoint"], project_root=project_root)
    if not any(cache.checked(x, project_root=project_root) == checkpoint for x in internal["model_identities"]):
        raise ValueError("Internal model identity does not bind the teacher checkpoint")
    import torch
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    meta, cfg = payload.get("metadata", {}), payload.get("adaptation_config", {})
    if (meta.get("stage") != "iesrgs_internal" or meta.get("iterations") != 30000
            or meta.get("optimizer_updates") != 29999 or cfg.get("method") != "iesrgs"
            or cfg.get("manifest") != manifest_id):
        raise ValueError("Internal teacher must complete its own IE 30000 iterations / 29999 updates")


def merge_shared(external_path, internal_path, out, *, project_root=ROOT):
    """Join explicit shared-RGB and authentic IE internal priors, without relabelling either."""
    output = fresh_output(out, project_root)
    external = cache.read(external_path, project_root=project_root)
    internal = cache.read(internal_path, project_root=project_root)
    if (external.get("schema") != "static_sr_wu_prior_bundle_v1" or external.get("kind") != FORMAL_KIND
            or external.get("status") != "completed" or external.get("fixture") is not False
            or internal.get("schema") != "static_sr_wu_prior_bundle_v1"
            or internal.get("kind") != "iesrgs_internal" or internal.get("fixture", False)):
        raise ValueError("Formal merge rejects fixtures and incomplete/non-IE producers")
    for field in ("manifest", "scene", "training_only", "upstream_commit"):
        if external.get(field) != internal.get(field):
            raise ValueError(f"IE sources disagree on {field}")
    if external.get("training_only") is not True or external.get("upstream_commit") != COMMIT:
        raise ValueError("Only pinned train-only IE priors may be merged")
    manifest_path = cache.checked(external["manifest"], project_root=project_root)
    manifest = cache.read(manifest_path, project_root=project_root)
    shared = cache.verify_cache(cache.checked(external["shared_cache"], project_root=project_root),
                                manifest_path, project_root=project_root)
    depth_load = external.get("depth_load", {})
    weight = depth_load.get("weights", {})
    if (depth_load.get("strict_load") is not True or depth_load.get("architecture") != DEPTH_ARCH
            or depth_load.get("SwinIR_loaded") is not False or weight.get("sha256") != DEPTH_SHA
            or Path(weight.get("path", "")).name != DEPTH_NAME
            or external.get("model_identities") != shared["model_identities"] + [weight]):
        raise ValueError("Shared IE depth receipt must bind strict official vitl loading and both real models")
    producer_path = Path(project_root) / "experiments/static_sr_shared_teacher_20261024/prepare_ie_shared.py"
    if not any(cache.checked(x, project_root=project_root) == producer_path for x in external.get("producer", [])):
        raise ValueError("Shared IE external producer identity is missing")
    rows = _aligned_rows(shared, manifest)
    expected = {(e["camera"], e["frame"]): e for e in rows}
    left = {(e["camera"], e["frame"]): e for e in external["entries"]}
    right = {(e["camera"], e["frame"]): e for e in internal["entries"]}
    if (len(left) != len(external["entries"]) or len(right) != len(internal["entries"])
            or set(left) != set(expected) or set(right) != set(expected)):
        raise ValueError("IE prior keys missing, duplicated or held out")
    for index in (external, internal):
        for item in index["producer"] + index["model_identities"]:
            cache.checked(item, project_root=project_root)
    _internal_checkpoint(internal, external["manifest"], project_root)
    import numpy as np
    entries = []
    for key, e in expected.items():
        a, b = left[key], right[key]
        if any(x["time"] != e["time"] or x["lr_sha256"] != e["lr_sha256"] for x in (a, b)):
            raise ValueError("Internal/external priors refer to different LR bytes or times")
        if a["external_rgb"] != e["sr"] or a["shared_sr_sha256"] != e["sr"]["sha256"]:
            raise ValueError("IE external RGB is not the exact shared PNG identity")
        for field, item, channels in (("external_inverse_depth", a, 1), ("internal_rgb", b, 3), ("internal_depth", b, 1)):
            # common accepts absolute paths; its global ROOT need not equal a temporary test root.
            value_id = dict(item[field], path=str(cache.checked(item[field], project_root=project_root)))
            array = common.load_prior_array(value_id, require_float32=True)
            if array.shape != (channels, e["shape"][1], e["shape"][0]) or not np.isfinite(array).all():
                raise ValueError(f"IE {field} array shape/dtype/finite contract differs")
        entries.append({**a, "internal_rgb": b["internal_rgb"], "internal_depth": b["internal_depth"]})
    merged = dict(external, kind=COMPLETE_KIND, method="iesrgs", entries=entries,
                  internal_teacher=internal["internal_teacher"],
                  producer=external["producer"] + internal["producer"],
                  model_identities=external["model_identities"] + internal["model_identities"],
                  source_indices=dict(external=cache.identity(external_path, project_root=project_root),
                                      internal=cache.identity(internal_path, project_root=project_root)))
    merged["cache_formats"] = dict(external["cache_formats"], internal=internal.get("cache_format", "npy"))
    cache.write_new(output, merged, project_root=project_root)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    external = commands.add_parser("external")
    external.add_argument("--config", required=True)
    external.add_argument("--shared-cache", required=True)
    external.add_argument("--upstream", default=str(DEFAULT_UPSTREAM))
    external.add_argument("--depth-weights", default=str(DEFAULT_WEIGHTS))
    external.add_argument("--out", required=True)
    external.add_argument("--fixture-key")
    external.add_argument("--cache-format", choices=CACHE_FORMATS, default="npy")
    external.add_argument("--run", action="store_true", help="Explicit GPU depth inference; use normal broker admission")
    merge = commands.add_parser("merge")
    merge.add_argument("--external-index", required=True)
    merge.add_argument("--internal-index", required=True)
    merge.add_argument("--out", required=True, help="A new JSON file, not a directory")
    args = parser.parse_args(argv)
    if args.command == "merge":
        result = dict(index=str(merge_shared(args.external_index, args.internal_index, args.out)))
    else:
        result = build_plan(args.config, args.shared_cache, args.upstream, args.depth_weights,
                            args.out, fixture_key=args.fixture_key, cache_format=args.cache_format)
        if args.run:
            result = dict(index=str(produce(result)))
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
