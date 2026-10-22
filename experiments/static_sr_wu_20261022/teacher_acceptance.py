"""Actual teacher GPU fixtures; never a formal prior or Wu training result.

Run ONE method per fresh process in its teacher environment, with one explicitly
bound GPU. --plan-only verifies inputs/source/weights on CPU without importing
torch. The dispatcher, not this entry point, owns GPU resource admission.

Sequence uses a complete synchronous training-camera group and its original ALS
plan. StableSR executes the author's standard 512/DDPM200 entry unchanged. 3DSR
executes all four actual latent transitions but echoes the preceding teacher PNG
as feedback: this checks the interface and state machinery, NOT shared-Wu
optimization or 3D consistency. All artifacts are marked fixture-only.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import socket
import sys
import time
import traceback
import weakref

from methods import sequence_matters as sequence

SCHEMA = "static_sr_wu_teacher_acceptance_fixture_v1"
PSRT_SHA = "37b7889aaf79f493f95473af05be43561f3b30d740691a2d4534ba09d653658d"
SPYNET_SHA = "3d2a1287666aa71752ebaedc06999212886ef476f77d691a1b0006107088e714"
TURBO_SHA = "3615aa00dcab2868042dcd1bba4979edbb6051e6e49c147cda6a2d550fb62c58"


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def identity(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sequence.sha256(path), "bytes": path.stat().st_size}


def select_fixture(manifest, frame, method, camera=None):
    """Keep original normalized time/order, and never open held-out/HR images."""
    sequence.validate_records(manifest["records"], manifest["heldout_cameras"])
    selected = [r for r in manifest["records"] if r.get("frame") == frame]
    if not selected or len({r["time"] for r in selected}) != 1:
        raise ValueError("Frame must identify one original synchronous training group")
    expected = {r["camera"] for r in manifest["records"]}
    if {r["camera"] for r in selected} != expected:
        raise ValueError("Selected frame lacks one or more registered training cameras")
    if method == "sequence_matters":
        if camera is not None:
            raise ValueError("Sequence fixture must retain every synchronous training camera")
        sequence.group_synchronous(selected, manifest["heldout_cameras"])
    else:
        camera = camera or selected[0]["camera"]
        selected = [r for r in selected if r["camera"] == camera]
        if len(selected) != 1:
            raise ValueError("Diffusion fixture requires one registered training camera")
    result = copy.deepcopy(manifest)
    result["records"] = copy.deepcopy(selected)
    for record in result["records"]:
        if not Path(record["lr_path"]).is_absolute():
            raise ValueError("Registered LR paths must be absolute; prepare on the execution host")
        if sequence.sha256(record["lr_path"]) != record.get("lr_sha256"):
            raise ValueError("Registered LR identity changed: " + record["key"])
    return result


def derive_sequence_plan(full_plan, manifest, fixture, source):
    sequence.validate_sequence_plan(full_plan, manifest, source)
    time_value = fixture["records"][0]["time"]
    result = copy.deepcopy(full_plan)
    result["groups"] = [g for g in result["groups"] if g["time"] == time_value]
    result["manifest_sha256"] = canonical_hash(fixture)
    sequence.validate_sequence_plan(result, fixture, source)
    return result


def verify_targets(targets, fixture):
    from PIL import Image
    expected = {r["key"]: r for r in fixture["records"]}
    if set(targets) != set(expected):
        raise RuntimeError("Teacher output key coverage differs from the fixture")
    result = []
    for key, record in expected.items():
        with Image.open(record["lr_path"]) as lr:
            size = tuple(4 * x for x in lr.size)
        with Image.open(targets[key]) as sr:
            sr.load()
            if sr.format != "PNG" or sr.mode != "RGB" or sr.size != size:
                raise RuntimeError("Teacher output must be an RGB PNG at original LR x4: " + key)
        result.append({"key": key, **identity(targets[key]), "size_wh": list(size)})
    return result


def finite_tree(value, torch, label):
    """Observe real floating tensors before uint8 conversion; no value changes."""
    if torch.is_tensor(value):
        if not bool(torch.isfinite(value).all().item()):
            raise FloatingPointError("Non-finite teacher tensor: " + label)
        return 1
    if isinstance(value, dict):
        return sum(finite_tree(v, torch, label + "." + str(k)) for k, v in value.items())
    if isinstance(value, (tuple, list)):
        return sum(finite_tree(v, torch, label) for v in value)
    return 0


def synchronize(torch):
    torch.cuda.synchronize()


def memory(torch):
    return {"peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved()}


@contextlib.contextmanager
def runtime_context(asset_file, upstream, task_root):
    from methods.threedsr import prepare_runtime_assets
    plan, environment = prepare_runtime_assets(asset_file, upstream, task_root)
    old_env, old_cwd, old_path = os.environ.copy(), Path.cwd(), sys.path[:]
    os.environ.update(environment)
    sys.path[:0] = [str(upstream), str(Path(upstream) / "third_parties"), str(Path(upstream) / "scripts")]
    os.chdir(plan["execution_workdir"])
    try:
        yield {"registration": plan["runtime_assets_identity"],
               "execution_workdir": plan["execution_workdir"], "offline_cache": True}
    finally:
        os.chdir(old_cwd)
        os.environ.clear()
        os.environ.update(old_env)
        sys.path[:] = old_path


def run_sequence(args, fixture, derived_plan, output, torch, receipt):
    import cv2
    import numpy as np
    sys.path.insert(0, str(Path(args.upstream) / "vsr/psrt"))
    from archs.psrt_recurrent_arch import BasicRecurrentSwin
    calls = receipt["forward_calls"] = []
    group = derived_plan["groups"][0]
    original = BasicRecurrentSwin.forward
    started = time.perf_counter()

    def observed_forward(model, batch, *positional, **keywords):
        if len(calls) >= len(group["sequences"]):
            raise RuntimeError("Unexpected PSRT forward beyond the frozen ALS plan")
        indices = group["sequences"][len(calls)]["indices"]
        rows = [fixture["records"][i] for i in indices]
        arrays = [cv2.imread(r["lr_path"]) for r in rows]
        expected = torch.from_numpy(np.stack(arrays).transpose(0, 3, 1, 2)).float().div(255).unsqueeze(0)
        if batch.shape != expected.shape or not torch.equal(batch.detach().cpu(), expected):
            raise RuntimeError("PSRT input camera order or source BGR preprocessing differs")
        synchronize(torch)
        before = time.perf_counter()
        if not calls:
            receipt["cold_start_to_first_forward_seconds"] = before - started
        prediction = original(model, batch, *positional, **keywords)
        synchronize(torch)
        elapsed = time.perf_counter() - before
        expected_shape = tuple(batch.shape[:-2]) + tuple(4 * x for x in batch.shape[-2:])
        if tuple(prediction.shape) != expected_shape:
            raise RuntimeError("PSRT forward shape differs from B,T,3,4H,4W")
        checked = finite_tree(prediction, torch, "PSRT prediction")
        calls.append({"keys": [r["key"] for r in rows], "input_shape": list(batch.shape),
                      "output_shape": list(prediction.shape), "elapsed_seconds": elapsed,
                      "finite_tensors_checked": checked})
        return prediction

    BasicRecurrentSwin.forward = observed_forward
    try:
        result = sequence.prepare_sequences(fixture, args.upstream, output / "teacher",
            args.psrt_checkpoint, args.spynet_checkpoint, sequence_plan=output / "fixture_plan.json")
    finally:
        BasicRecurrentSwin.forward = original
    if len(calls) != len(group["sequences"]):
        raise RuntimeError("PSRT did not execute every ALS sequence")
    receipt["teacher_receipt"] = identity(output / "teacher/receipt.json")
    receipt["outputs"] = verify_targets(result["targets"], fixture)
    receipt["inference_seconds"] = sum(c["elapsed_seconds"] for c in calls)
    receipt["cost_scope"] = "One complete synchronous camera group; source FP32 and ALS lengths unchanged"


@contextlib.contextmanager
def observe_stablesr_checkpoint_loads(torch, audits):
    """Audit loads without mistaking the author's LPIPS calibration for VGG.

    taming/modules/losses/lpips.py first loads torchvision VGG16 (lines 17,
    79), reuses its feature Parameter objects, then loads only lin0..4 (29).
    Permit that exact sparse load only after observing those same backbone
    parameters successfully loaded. All other missing parameters still fail,
    including the subsequent top-level StableSR and CFW checkpoint loads.
    """
    original = torch.nn.Module.load_state_dict
    loaded_vgg_parameters = {}

    def observed(model, state, *args, **kwargs):
        result = original(model, state, *args, **kwargs)
        parameters = dict(model.named_parameters())
        missing = sorted(set(parameters).intersection(result.missing_keys))
        model_class = type(model).__module__ + "." + type(model).__name__
        permitted = False
        if model_class == "taming.modules.losses.lpips.LPIPS" and missing:
            calibration = {f"lin{i}.model.1.weight" for i in range(5)}
            backbone = {"net." + name for name, _ in model.net.named_parameters()}
            permitted = (set(state) == calibration and set(missing) == backbone
                and not result.unexpected_keys and all(
                    loaded_vgg_parameters.get(id(parameters[name]), lambda: None)() is parameters[name]
                    for name in missing))
        audits.append({"model_class": model_class, "missing_parameters": missing,
                       "missing_keys": list(result.missing_keys), "unexpected_keys": list(result.unexpected_keys),
                       "permitted_sparse_lpips_calibration": permitted,
                       "previously_loaded_vgg_parameters": missing if permitted else []})
        if missing and not permitted:
            raise ValueError("StableSR checkpoint left parameters uninitialized: " + repr(missing[:10]))
        if model_class == "torchvision.models.vgg.VGG" and not result.missing_keys and not result.unexpected_keys:
            loaded_vgg_parameters.update((id(parameter), weakref.ref(parameter)) for parameter in parameters.values())
        return result

    torch.nn.Module.load_state_dict = observed
    try:
        yield
    finally:
        torch.nn.Module.load_state_dict = original


def run_stablesr(args, fixture, output, torch, receipt):
    from methods import splatsure_stablesr as stable
    teacher = output / "teacher"
    (teacher / "inputs").mkdir(parents=True)
    source_image = fixture["records"][0]["lr_path"]
    staged = teacher / "inputs/fixture.png"
    try:
        os.link(source_image, staged)
    except OSError:
        shutil.copyfile(source_image, staged)
    if sequence.sha256(staged) != fixture["records"][0]["lr_sha256"]:
        raise ValueError("Staged LR changed")
    plan = {"python": sys.executable, "source_root": args.upstream,
            "output_root": str(teacher), "seed": args.seed,
            "input_shape_wh": stable.png_shape(staged),
            "model_identities": [{"role": "diffusion", "path": args.stable_checkpoint},
                                 {"role": "cfw", "path": args.vqgan_checkpoint}]}
    plan["tiling"] = stable.stablesr_geometry(plan["input_shape_wh"])
    command = stable.command(plan)
    receipt["author_argv"] = command
    receipt["tiling"] = plan["tiling"]
    receipt["recipe"] = stable.RECIPE
    receipt["model_loads"], receipt["floating_checks"] = [], []
    loads, checks = receipt["model_loads"], receipt["floating_checks"]
    with runtime_context(args.runtime_assets, args.upstream, output / "runtime") as runtime:
        receipt["runtime_assets"] = runtime
        spec = importlib.util.spec_from_file_location("teacher_acceptance_author_stablesr", command[1])
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        original_loader = module.load_model_from_config

        def observed_loader(config, checkpoint, *positional, **keywords):
            synchronize(torch)
            before = time.perf_counter()
            state_audits = []
            with observe_stablesr_checkpoint_loads(torch, state_audits):
                model = original_loader(config, checkpoint, *positional, **keywords)
            synchronize(torch)
            loads.append({"checkpoint": str(checkpoint), "elapsed_seconds": time.perf_counter() - before,
                          "state_audits": state_audits, "parameters": sum(p.numel() for p in model.parameters())})
            return model

        original_color = module.adaptive_instance_normalization

        def observed_color(*positional, **keywords):
            count = finite_tree(positional, torch, "StableSR decoded inputs")
            result = original_color(*positional, **keywords)
            count += finite_tree(result, torch, "StableSR color corrected prediction")
            checks.append({"finite_tensors_checked": count})
            return result

        module.load_model_from_config = observed_loader
        module.adaptive_instance_normalization = observed_color
        argv = sys.argv
        sys.argv = command[1:]
        synchronize(torch)
        before = time.perf_counter()
        try:
            module.main()
        finally:
            sys.argv = argv
        synchronize(torch)
        receipt["author_main_seconds"] = time.perf_counter() - before
    if len(loads) != 2 or not checks:
        raise RuntimeError("Author StableSR path did not load both models and check decoded tensors")
    receipt["inference_and_io_seconds"] = receipt["author_main_seconds"] - sum(x["elapsed_seconds"] for x in loads)
    receipt["outputs"] = verify_targets({fixture["records"][0]["key"]: teacher / "generated/fixture.png"}, fixture)
    receipt["cost_scope"] = "One original LR image; standard 512/DDPM200; includes source tiling and I/O"


def validate_latent(state, record, round_index, model_ids, torch):
    from methods.threedsr import UPSTREAM_COMMIT
    expected = {"key": record["key"], "lr_sha256": record["lr_sha256"], "scene": record["scene"],
                "round_index": round_index, "model_identity": model_ids, "upstream_commit": UPSTREAM_COMMIT}
    if any(state.get(k) != v for k, v in expected.items()) or not state.get("patches"):
        raise RuntimeError("3DSR latent observation/round/source/model identity differs")
    for patch in state["patches"].values():
        if len(patch.get("region", [])) != 4 or not torch.is_tensor(patch.get("x_t")):
            raise RuntimeError("3DSR latent patch is incomplete")
        finite_tree(patch["x_t"], torch, "3DSR persistent noisy latent")
    return {k: v["region"] for k, v in state["patches"].items()}


def run_four_round_fixture(bridge, fixture, output, torch, rounds):
    """Real bridge or test double; persistent latent/feedback barriers are checked."""
    if len(fixture["records"]) != 1:
        raise ValueError("The diffusion acceptance fixture contains exactly one observation")
    targets = None
    previous_regions = None
    record = fixture["records"][0]
    models = [x["sha256"] for x in bridge.weight_loads]
    for index in range(4):
        synchronize(torch)
        torch.cuda.reset_peak_memory_stats()
        before = time.perf_counter()
        feedback = targets
        targets = bridge.generate_targets(fixture, index, output / f"round{index}", feedback)
        synchronize(torch)
        elapsed = time.perf_counter() - before
        metrics = memory(torch)
        outputs = verify_targets(targets, fixture)
        state_path = bridge.state_root / (record["key"] + ".pt")
        state = torch.load(state_path, map_location="cpu", weights_only=True)
        regions = validate_latent(state, record, index, models, torch)
        if previous_regions is not None and regions != previous_regions:
            raise RuntimeError("3DSR patch layout changed between rounds")
        previous_regions = regions
        snapshot = output / "latent_snapshots" / f"round{index}" / (record["key"] + ".pt")
        snapshot.parent.mkdir(parents=True)
        shutil.copyfile(state_path, snapshot)
        rounds.append({"round_index": index, "elapsed_seconds": elapsed, **metrics,
                       "feedback_mode": "none" if index == 0 else "feedback_echo_previous_teacher_png",
                       "feedback_sha256": None if feedback is None else sequence.sha256(feedback[record["key"]]),
                       "outputs": outputs, "latent": identity(snapshot), "patch_regions": regions,
                       "teacher_receipt": identity(output / f"round{index}/receipt.json")})
    return rounds


def run_threedsr(args, fixture, output, torch, receipt):
    from methods.threedsr import StableSRTurboBridge
    receipt["sampler_calls"], receipt["floating_checks"] = [], []
    calls, checks = receipt["sampler_calls"], receipt["floating_checks"]
    with runtime_context(args.runtime_assets, args.upstream, output / "runtime") as runtime:
        receipt["runtime_assets"] = runtime
        synchronize(torch)
        before = time.perf_counter()
        bridge = StableSRTurboBridge(args.upstream, args.stable_checkpoint, args.vqgan_checkpoint,
                                    output / "latent_state", seed=args.seed)
        synchronize(torch)
        receipt["model_load_seconds"] = time.perf_counter() - before
        receipt["model_load_memory"] = memory(torch)
        receipt["weight_loads"] = bridge.weight_loads
        bridge.runtime_assets_context = runtime
        original_sample, original_color = bridge.model.sample_canvas_one_iter, bridge.colorfix

        def observed_sample(*positional, **keywords):
            result = original_sample(*positional, **keywords)
            count = finite_tree(result, torch, "3DSR diffusion step")
            calls.append({"iteration": keywords["iteration"], "return_x0": keywords["return_x0"],
                          "has_x0_input": keywords.get("x0_input") is not None,
                          "finite_tensors_checked": count})
            return result

        def observed_color(*positional, **keywords):
            count = finite_tree(positional, torch, "3DSR decoded inputs")
            result = original_color(*positional, **keywords)
            count += finite_tree(result, torch, "3DSR color corrected prediction")
            checks.append({"finite_tensors_checked": count})
            return result

        bridge.model.sample_canvas_one_iter, bridge.colorfix = observed_sample, observed_color
        rounds = receipt["rounds"] = []
        run_four_round_fixture(bridge, fixture, output, torch, rounds)
    patches = len(rounds[0]["patch_regions"])
    expected = []
    for index in range(4):
        per_patch = ([] if index == 0 else [(4-index, False, True)]) + [(3-index, True, False)]
        expected.extend(per_patch * patches)
    observed = [(c["iteration"], c["return_x0"], c["has_x0_input"]) for c in calls]
    if observed != expected or len(checks) != 4 * patches:
        raise RuntimeError("3DSR sampler did not traverse all four prescribed latent rounds")
    receipt["cost_scope"] = "Four diffusion rounds for one LR; model loaded once; excludes Wu training/rendering"
    receipt["inference_and_io_seconds"] = sum(r["elapsed_seconds"] for r in rounds)
    receipt["feedback_mode"] = "feedback_echo_previous_teacher_png; NO shared-Wu model or parameter update"


def prepare(args, output, receipt):
    manifest = json.loads(Path(args.records).read_text())
    fixture = select_fixture(manifest, args.frame, args.method, args.camera)
    receipt["registered_records"] = identity(args.records)
    receipt["registered_manifest_canonical_sha256"] = canonical_hash(manifest)
    receipt["fixture_manifest_canonical_sha256"] = canonical_hash(fixture)
    receipt["selected_keys"] = [r["key"] for r in fixture["records"]]
    write_json(output / "fixture_records.json", fixture)
    receipt["fixture_records"] = identity(output / "fixture_records.json")
    derived = None
    if args.method == "sequence_matters":
        upstream = sequence.verify_upstream(args.upstream)
        if not args.sequence_plan:
            raise ValueError("Sequence acceptance requires a previously prepared CPU --sequence-plan")
        derived = derive_sequence_plan(json.loads(Path(args.sequence_plan).read_text()), manifest, fixture,
                                       upstream / "vsr/utils_vsr.py")
        write_json(output / "fixture_plan.json", derived)
        receipt["original_sequence_plan"] = identity(args.sequence_plan)
        required = [(args.psrt_checkpoint, PSRT_SHA), (args.spynet_checkpoint, SPYNET_SHA)]
        receipt["upstream_commit"] = sequence.UPSTREAM_COMMIT
    else:
        from methods import splatsure_stablesr as stable
        from methods.threedsr import UPSTREAM_COMMIT
        commit = stable.STABLESR_COMMIT if args.method == "stablesr" else UPSTREAM_COMMIT
        upstream = sequence.verify_upstream(args.upstream, commit)
        required = [(args.stable_checkpoint, stable.WEIGHTS["diffusion"][1] if args.method == "stablesr" else TURBO_SHA),
                    (args.vqgan_checkpoint, stable.WEIGHTS["cfw"][1])]
        if not args.runtime_assets:
            raise ValueError("Diffusion acceptance requires verified --runtime-assets for offline model loading")
        stable.validate_runtime_assets(json.loads(Path(args.runtime_assets).read_text()))
        receipt["runtime_assets_registration"] = identity(args.runtime_assets)
        receipt["upstream_commit"] = commit
    receipt["upstream"] = str(upstream)
    method_root = Path(__file__).parent / "methods"
    adapter_names = ["sequence_matters.py"] if args.method == "sequence_matters" else ["threedsr.py", "splatsure_stablesr.py"]
    receipt["adapter_sources"] = [identity(method_root / name) for name in adapter_names]
    receipt["model_identities"] = []
    for path, expected in required:
        if not path:
            raise ValueError("Missing required official teacher checkpoint")
        item = identity(path)
        if item["sha256"] != expected:
            raise ValueError("Teacher checkpoint differs from pinned official bytes: " + str(path))
        receipt["model_identities"].append(item)
    return fixture, derived


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["sequence_matters", "stablesr", "threedsr"], required=True)
    parser.add_argument("--records", required=True)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--camera", help="Diffusion only; Sequence always keeps the full same-time camera group")
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--out", required=True, help="Fresh fixture directory; never pass a formal prior output")
    parser.add_argument("--sequence-plan")
    parser.add_argument("--psrt-checkpoint")
    parser.add_argument("--spynet-checkpoint")
    parser.add_argument("--stable-checkpoint")
    parser.add_argument("--vqgan-checkpoint")
    parser.add_argument("--runtime-assets")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args(argv)
    for name in ("records", "upstream", "out", "sequence_plan", "psrt_checkpoint", "spynet_checkpoint",
                 "stable_checkpoint", "vqgan_checkpoint", "runtime_assets"):
        if getattr(args, name):
            setattr(args, name, str(Path(getattr(args, name)).resolve()))
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=False)
    receipt = {"schema": SCHEMA, "status": "preparing", "method": args.method,
               "formal_prior_allowed": False, "formal_parent_allowed": False,
               "teacher_generated": False, "gpu_executed": False, "wrapper": identity(__file__),
               "python": sys.executable, "argv": vars(args), "hostname": socket.gethostname(),
               "cost_note": "Acceptance includes finite checks/synchronization overhead; one sample is not a full-scene cost measurement"}
    started, torch = time.perf_counter(), None
    error = None
    write_json(output / "acceptance.json", receipt)
    try:
        fixture, derived = prepare(args, output, receipt)
        receipt["cpu_validation_seconds"] = time.perf_counter() - started
        if args.plan_only:
            receipt["status"] = "plan_only"
        else:
            visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
            if not visible or "," in visible or visible == "-1":
                raise ValueError("Bind exactly one admitted GPU using CUDA_VISIBLE_DEVICES before execution")
            import torch as torch_module
            torch = torch_module
            if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
                raise RuntimeError("Teacher acceptance requires exactly one visible CUDA GPU")
            props = torch.cuda.get_device_properties(0)
            receipt["hardware"] = {"CUDA_VISIBLE_DEVICES": visible, "torch_version": torch.__version__,
                "cuda_runtime": torch.version.cuda, "gpu_name": props.name,
                "gpu_uuid": str(getattr(props, "uuid", "unavailable")), "total_memory_bytes": props.total_memory}
            synchronize(torch)
            torch.cuda.reset_peak_memory_stats()
            receipt["gpu_executed"] = True
            receipt["status"] = "running_GPU_fixture"
            write_json(output / "acceptance.json", receipt)
            before = time.perf_counter()
            if args.method == "sequence_matters":
                run_sequence(args, fixture, derived, output, torch, receipt)
            else:
                {"stablesr": run_stablesr, "threedsr": run_threedsr}[args.method](
                    args, fixture, output, torch, receipt)
            synchronize(torch)
            receipt["gpu_phase_seconds"] = time.perf_counter() - before
            receipt["teacher_generated"] = True
            receipt["status"] = "fixture_passed"
    except Exception as exception:
        error = exception
        receipt["status"] = "failed"
        receipt["error"] = f"{type(exception).__name__}: {exception}"
        receipt["traceback"] = traceback.format_exc()
        receipt["oom"] = "out of memory" in str(exception).lower()
    finally:
        receipt["elapsed_seconds"] = time.perf_counter() - started
        if torch is not None and receipt["gpu_executed"]:
            try:
                metrics = [memory(torch), receipt.get("model_load_memory", {})] + receipt.get("rounds", [])
                receipt["memory"] = {key: max(m.get(key, 0) for m in metrics)
                    for key in ("peak_allocated_bytes", "peak_reserved_bytes")}
            except Exception as metrics_error:
                receipt["memory_error"] = str(metrics_error)
        write_json(output / "acceptance.json", receipt)
    print(json.dumps({"status": receipt["status"], "receipt": str(output / "acceptance.json")}))
    if error is not None:
        raise error
    return receipt


if __name__ == "__main__":
    main()
