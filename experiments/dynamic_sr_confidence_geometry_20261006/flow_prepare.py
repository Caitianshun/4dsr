"""Prepare native-training-LR RAFT-A incrementally after an explicit core decision.

Registration, audit and self-check are CPU-only. Generation is a separate,
explicit action: reuse the 16 hash-bound A files, infer only the 1105 missing
bidirectional pairs, and publish an A-only index after all 1121 pairs pass.
No B/HR flow, teacher image, depth inference or scene update is performed.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = ROOT / "output" / HERE.name
DEFAULT_MANIFEST = ROOT / "data/dynamic_sr/n3dv_prepared/cook_spinach/manifest.json"
DEFAULT_EXISTING = ROOT / "output/dynamic_sr_prior_diagnosis_20260929/image_priors/flow_index.json"
DEFAULT_REGISTRATION = OUT / "phase13/flow_prepare_registration_v5"
DEFAULT_OUTPUT = OUT / "phase13/flow_native_lr_v5"
WEIGHT_SHA = "ff5fadd56d26b40647388883af1547351ea17868b765c05b27231e72dd16a322"
POLICY = dict(branch="A", estimator="torchvision RAFT-large C_T_SKHT_V2",
    num_flow_updates=12, source_shape=[252, 336], network_grid=[256, 336],
    padding=[0, 0, 0, 4], pad_mode="replicate", input_resize=False,
    input="OpenCV BGR decoded uint8, converted RGB; CHW FP32 /127.5-1",
    autocast=False, batch_size_per_direction=1, mode="eval + inference_mode",
    output="last prediction, crop :252,:336, HxWx2 float32 in native LR pixels",
    directions=["forward:first_to_second", "backward:second_to_first"],
    frame_gap=2, normalized_time="original_frame_index/300; difference is not elapsed seconds",
    HR_pixels=False, B_branch=False, teacher_pixels=False, scene_updates=0,
    RAFT_weight_download=False)
CAMERAS = tuple(f"cam{i:02d}" for i in range(2, 21))
FRAMES = tuple(range(0, 120, 2))


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temp.replace(path)


def record(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=sha(path))


def bound(identity):
    path = Path(identity["path"])
    if not path.is_absolute():
        path = ROOT / path
    if sha(path) != identity["sha256"]:
        raise ValueError(f"Identity mismatch: {path}")
    return path.resolve()


def pairs():
    return {(c, f, f + 2) for c in CAMERAS for f in FRAMES[:-1]}


def manifest_records(path, verify_pixels=False):
    manifest = read(path)
    rows = [r for r in manifest["observations"] if r["split"] == "train"]
    observations = {(r["camera_id"], int(r["frame_index"])): r for r in rows}
    if (tuple(manifest["splits"]["train"]) != CAMERAS or len(rows) != 1140
            or set(observations) != {(c, f) for c in CAMERAS for f in FRAMES}
            or manifest["resolutions"]["lr"] != [336, 252]):
        raise ValueError("Registered 19-camera/60-even-frame/native-LR protocol differs")
    inputs = []
    for (camera, frame), row in sorted(observations.items()):
        source = (Path(path).parent / row["lr_path"]).resolve()
        if "hr" in source.parts or "/lr/" not in str(source):
            raise ValueError("Only native registered training LR paths are legal")
        if verify_pixels:
            import cv2
            if sha(source) != row["lr_sha256"]:
                raise ValueError("Training LR file identity changed")
            pixels = cv2.imread(str(source), cv2.IMREAD_COLOR)
            if pixels is None or pixels.shape != (252, 336, 3) or str(pixels.dtype) != "uint8":
                raise ValueError("Native LR file grid/type differs")
        inputs.append(dict(camera=camera, frame=frame, path=str(source), sha256=row["lr_sha256"]))
    return manifest, observations, inputs


def resolve_flow(path, index_path):
    path = Path(path)
    if path.is_absolute():
        return path.resolve()
    root_path = ROOT / path
    return root_path.resolve() if root_path.exists() else (Path(index_path).parent / path).resolve()


def validate_payload(path, expected_sha):
    import numpy as np
    if sha(path) != expected_sha:
        raise ValueError("Frozen flow payload SHA mismatch")
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != {"forward", "backward", "original_grid", "network_grid", "padding"}:
            raise ValueError("Flow schema differs from original RAFT-A payload")
        for name in ("forward", "backward"):
            value = data[name]
            if value.shape != (252, 336, 2) or value.dtype != np.float32 or not np.isfinite(value).all():
                raise ValueError("Native LR flow must be finite float32 HxWx2")
        for name, expected in (("original_grid", [252, 336]),
                               ("network_grid", POLICY["network_grid"]), ("padding", POLICY["padding"])):
            if data[name].tolist() != expected:
                raise ValueError("Original/native grid or replicate padding differs")


def validated_rows(index_path, manifest_path, verify_payload=True, strict_A=False):
    index = read(index_path)
    if not index.get("status", "").startswith("completed"):
        raise ValueError("Only completed source indices accepted")
    if (index["manifest_sha256"] != sha(manifest_path) or index["weight_sha256"] != WEIGHT_SHA
            or index["num_flow_updates"] != 12):
        raise ValueError("Frozen manifest/RAFT weight/update count differs")
    _, observations, _ = manifest_records(manifest_path)
    selected = {}
    for row in index.get("rows", []):
        if row.get("branch") != "A":
            if strict_A:
                raise ValueError("Formal phase13 index must contain A only")
            continue  # Ignore diagnostic B/C metadata; never open their files.
        key = (row["camera"], int(row["first"]), int(row["second"]))
        if key not in pairs() or key in selected or row.get("information") in ("HR", "privileged", "diagnostic_HR_only"):
            raise ValueError("Illegal/duplicate/privileged A pair")
        if row.get("network_grid") != POLICY["network_grid"] or row.get("padding") != POLICY["padding"]:
            raise ValueError("A inference grid/padding differs")
        if len(row.get("sources", [])) != 2:
            raise ValueError("Both exact training LR source identities are required")
        for source, frame in zip(row["sources"], key[1:]):
            observation = observations[key[0], frame]
            text = str(source["path"]).replace("\\", "/")
            if (source["camera"] != key[0] or int(source["frame"]) != frame
                    or source["sha256"] != observation["lr_sha256"] or "/hr/" in text
                    or not text.endswith(observation["lr_path"].replace("\\", "/"))):
                raise ValueError("A flow did not originate from the exact legal LR pair")
        if verify_payload:
            validate_payload(resolve_flow(row["path"], index_path), row["sha256"])
        selected[key] = copy.deepcopy(row)
    return index, selected


def coverage(rows):
    actual = set(rows)
    if not actual <= pairs():
        raise ValueError("Illegal temporal coverage")
    missing = pairs() - actual
    return dict(status="complete_native_training_LR_flow" if not missing else "incomplete_native_training_LR_flow",
        available_pairs=len(actual), required_pairs=1121, directed_predictions=2*len(actual),
        missing_pairs=[dict(camera=c, first=f, second=g) for c, f, g in sorted(missing)],
        missing_pair_count=len(missing), formal_training_ready=not missing,
        RAFT_inference_run=False, scene_parameter_updates=0)


def require_complete(rows):
    report = coverage(rows)
    if not report["formal_training_ready"]:
        raise ValueError(f"Formal phase13 coverage is incomplete: {report['available_pairs']}/1121")
    return report


def software_identity(legacy, allow_existing_CUDA=False):
    # Imports and introspection are CPU operations; never instantiate a network.
    import torch
    import torchvision
    from torchvision.models.optical_flow import raft_large
    if str(torch.__version__) != legacy["torch"] or str(torchvision.__version__) != legacy["torchvision"]:
        raise ValueError("Use exact previously registered torch/torchvision versions")
    wrapper = Path(inspect.getfile(raft_large))
    if sha(wrapper) != legacy["implementation_sha256"]:
        raise ValueError("Previously recorded torchvision wrapper differs")
    directory = Path(inspect.getfile(inspect.unwrap(raft_large))).parent
    files = [wrapper] + sorted(directory.glob("*.py"))
    files += [Path(inspect.getfile(importlib.import_module("torchvision.ops.misc"))),
              Path(inspect.getfile(importlib.import_module("torchvision.models._api")))]
    if torch.cuda.is_initialized() and not allow_existing_CUDA:
        raise RuntimeError("CPU registration must not initialize CUDA")
    return dict(torch=str(torch.__version__), torchvision=str(torchvision.__version__),
        implementation_files=[record(p) for p in dict.fromkeys(files)],
        original_wrapper_only_identity_preserved=True, actual_RAFT_implementation_newly_bound=True,
        identity_inspection_runs_no_network_forward=True)


def register(args):
    if args.registration.exists():
        raise FileExistsError("Choose a new registration directory; never replace an existing plan")
    legacy, existing = validated_rows(args.existing_index, args.manifest)
    if len(existing) != 16:
        raise ValueError("This incremental registration requires the exact existing diagnostic 16 A pairs")
    manifest, _, inputs = manifest_records(args.manifest, verify_pixels=True)
    weight = args.weight or Path(legacy["weight_path"])
    if not weight.exists() or sha(weight) != WEIGHT_SHA:
        raise ValueError("Existing exact RAFT weight required; downloads are forbidden")
    software = software_identity(legacy)
    executed = args.existing_index.parent / "executed_sources/image_prior_cache.py"
    complete = args.existing_index.parent / "cache_complete.json"
    helper = ROOT / "experiments/dynamic_sr_prior_diagnosis_20260929/image_prior_common.py"
    integrity = read(args.existing_index.parent / "integrity.json")
    helper_registered = integrity["source_identifiers"][str(helper.relative_to(ROOT))]
    if sha(helper) != helper_registered:
        raise ValueError("Current auxiliary source differs from original integrity record")
    if sha(executed) != read(complete)["source_sha256"]:
        raise ValueError("Original executed generator snapshot identity mismatch")
    times = [row["seconds"] for row in existing.values()]
    missing = sorted(pairs()-set(existing))
    plan = dict(status="registered_CPU_only_no_flow_generation", source=record(__file__),
        manifest=record(args.manifest), existing_index=record(args.existing_index), weight=record(weight),
        original_executed_generator=record(executed), original_cache_receipt=record(complete),
        original_helper_current_identity=record(helper), original_helper_executed_snapshot_available=False,
        algorithm=POLICY, software=software, output_directory=str(args.output.resolve()),
        training_inputs=inputs, reused_rows=[existing[key] for key in sorted(existing)],
        missing_pairs=[dict(camera=c, first=f, second=g) for c, f, g in missing],
        coverage=coverage(existing), new_bidirectional_pairs=len(missing), new_directed_predictions=2*len(missing),
        recurrent_updates=24*len(missing), original_time_convention=manifest["time_convention"],
        original_pixel_coordinate_convention=manifest["pixel_coordinate_convention"],
        cost_estimate=dict(original_hardware=legacy["gpu"],
            old_A_pair_processing_seconds_mean=statistics.mean(times),
            old_A_pair_processing_seconds_min=min(times), old_A_pair_processing_seconds_max=max(times),
            same_old_hardware_extrapolated_seconds=statistics.mean(times)*len(missing),
            estimate_includes_original_bidirectional_inference_and_file_writing=True,
            new_hardware_timing_unmeasured=True, A_only_peak_GPU_unmeasured=True,
            old_mixed_ABC_and_depth_peak_not_used=True,
            new_uncompressed_flow_bytes=len(missing)*252*336*2*4*2,
            compressed_bytes_extrapolated=sum(resolve_flow(r["path"], args.existing_index).stat().st_size for r in existing.values())/len(existing)*len(missing)),
        decision_gate=dict(required_fields=["proceed_phase13=true", "core_quality_supports_phase13=true",
            "blocking_data_protocol_bug_confirmed=false", "nonempty data_protocol_audit_note",
            "nonempty rationale", "hash-bound core_finalization_state"],
            complete_core_endpoints=12, phase13_training_authorized=False),
        GPU_used=False, inference_performed=False, scene_updates=0)
    args.registration.mkdir(parents=True)
    shutil.copyfile(__file__, args.registration / "flow_prepare.py")
    write(args.registration / "plan.json", plan)
    write(args.registration / "coverage.json", plan["coverage"])
    print(json.dumps({k:plan[k] for k in ["status", "new_bidirectional_pairs", "new_directed_predictions", "cost_estimate"]}))


def decision_gate(path):
    decision = read(path)
    if (decision.get("proceed_phase13") is not True or decision.get("core_quality_supports_phase13") is not True
            or decision.get("blocking_data_protocol_bug_confirmed") is not False
            or not isinstance(decision.get("data_protocol_audit_note"), str)
            or not decision["data_protocol_audit_note"].strip()
            or not isinstance(decision.get("rationale"), str) or not decision["rationale"].strip()):
        raise ValueError("Decision must support phase13, explicitly record no confirmed blocking data-protocol bug, and retain a nonempty audit note")
    state_path = bound(decision["core_finalization_state"])
    if state_path != (OUT / "core_finalization_state.json").resolve():
        raise ValueError("Decision must bind this experiment's exact core finalization state")
    state = read(state_path)
    if state["status"] != "core_complete_pending_research_and_visual_review":
        raise ValueError("Full core consolidation has not completed")
    integrity = read(bound(state["final_integrity"]))
    quality = read(bound(state["quality_summary"]))
    required = {f"r{r}_{arm}" for r in (1, 2) for arm in ("C1", "Jperm", "B2perm", "R", "G", "RG")}
    if (integrity["status"] != "passed_core_endpoint_evidence"
            or {r["task"] for r in integrity["endpoints"]} != required
            or len(integrity["endpoints"]) != 12 or quality["status"] != "completed_core_evidence"):
        raise ValueError("All twelve fixed core endpoints and full quality evidence are required")
    return record(path)


def gpu_snapshot():
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible or "," in visible:
        raise ValueError("Explicitly bind one independently reserved GPU")
    identity = subprocess.check_output(["nvidia-smi", "-i", visible,
        "--query-gpu=uuid,name,memory.used,utilization.gpu", "--format=csv,noheader"], text=True).strip()
    uuid = identity.split(",")[0].strip()
    processes = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name",
        "--format=csv,noheader"], text=True).strip().splitlines()
    if any(p.split(",")[0].strip() == uuid for p in processes):
        raise RuntimeError("Reserved flow GPU has existing compute processes")
    return dict(binding=visible, identity=identity, checked_unix=time.time())


def generate(args):
    plan_path = args.registration / "plan.json"
    plan = read(plan_path)
    if record(__file__) != plan["source"] or POLICY != plan["algorithm"]:
        raise ValueError("Flow preparer changed after CPU registration")
    manifest_path = bound(plan["manifest"]); existing_path = bound(plan["existing_index"])
    weight_path = bound(plan["weight"]); bound(plan["original_executed_generator"])
    legacy, old_rows = validated_rows(existing_path, manifest_path)
    if software_identity(legacy) != plan["software"]:
        raise ValueError("RAFT implementation changed after registration")
    decision = decision_gate(args.decision)
    output = Path(plan["output_directory"])
    if (output / "index.json").exists():
        raise FileExistsError("Completed frozen index already exists; use audit, never regenerate")
    if args.inherited_lock_fd is None:
        # Supervisor owns the shared evaluation lock until the CUDA child exits.
        OUT.mkdir(parents=True, exist_ok=True)
        with (OUT / "evaluation_gpu.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            first = gpu_snapshot()
            # Verify all source bytes between two resource readings.
            manifest_records(manifest_path, verify_pixels=True)
            second = gpu_snapshot()
            command = [sys.executable, "-u", str(Path(__file__).resolve()), "--generate",
                "--registration", str(args.registration), "--decision", str(args.decision),
                "--inherited-lock-fd", str(lock.fileno())]
            child = subprocess.Popen(command, pass_fds=(lock.fileno(),), stdin=subprocess.DEVNULL)
            receipt = dict(status="running", supervisor_pid=os.getpid(), cuda_child_pid=child.pid,
                lock_held_until_child_exit=True, resource_readings=[first, second],
                registration=record(plan_path), decision=decision)
            write(args.registration / "supervisor.json", receipt)
            code = child.wait()
            receipt.update(status="completed" if code == 0 else "failed", returncode=code,
                           cuda_child_fully_exited=True)
            write(args.registration / "supervisor.json", receipt)
        if code != 0:
            raise RuntimeError(f"CUDA flow child failed with returncode {code}; its original failure receipt is retained")
        return
    fd = args.inherited_lock_fd
    if os.fstat(fd).st_ino != (OUT / "evaluation_gpu.lock").stat().st_ino:
        raise ValueError("Inherited GPU lock identity differs")
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    gpu_snapshot()
    import cv2
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torchvision.models.optical_flow import raft_large
    torch.set_num_threads(4); cv2.setNumThreads(4)
    manifest, observations, inputs = manifest_records(manifest_path, verify_pixels=True)
    if inputs != plan["training_inputs"]:
        raise ValueError("Training LR input registry changed after CPU registration")
    output.mkdir(parents=True, exist_ok=True)
    rows = {key:dict(value, origin="reused_original_native_LR_A", information="legal_LR_only",
                     original_index_sha256=sha(existing_path), original_hardware=legacy["gpu"])
            for key, value in old_rows.items()}
    progress_path = output / "progress.json"
    cost = dict(new_bidirectional_pairs=0, RAFT_directed_predictions=0, recurrent_updates=0,
                scene_parameter_updates=0, reused_bidirectional_pairs=len(old_rows),
                ambiguous_interrupted_RAFT_calls=0)
    if progress_path.exists():
        previous = read(progress_path)
        if previous["registration_sha256"] != sha(plan_path):
            raise ValueError("Existing partial output belongs to another immutable registration")
        for row in previous["new_rows"]:
            key = (row["camera"], row["first"], row["second"])
            if key in rows or key not in pairs() or row["branch"] != "A":
                raise ValueError("Illegal/duplicate resumed native-LR pair")
            validate_payload(resolve_flow(row["path"], output / "index.json"), row["sha256"])
            rows[key] = row
        cost = previous["cost"]
        if cost["new_bidirectional_pairs"] != len(rows)-len(old_rows):
            raise ValueError("Resumed new-pair accounting differs from hash-bound rows")
        attempt_path = output / "last_attempt.json"
        if attempt_path.exists():
            attempt = read(attempt_path)
            if attempt["status"] != "committed_pair":
                history = output / "attempt_history" / f"before_resume_{os.getpid()}.json"
                write(history, attempt)
                observed = attempt.get("cost_completed", attempt.get("cost_before", {}))
                extra = max(0, observed.get("RAFT_directed_predictions", 0)-cost["RAFT_directed_predictions"])
                cost["RAFT_directed_predictions"] += extra; cost["recurrent_updates"] += 12*extra
                if attempt.get("interrupted_forward_cost_is_ambiguous"):
                    cost["ambiguous_interrupted_RAFT_calls"] += 1
                # Save this audit before any retry; known repeated calls are charged,
                # while a call interrupted before its receipt is never assumed zero.
                write(progress_path, dict(previous, cost=cost))
    allowed = {str(Path(r["path"]).resolve()) for r in inputs}
    allowed.update(str(resolve_flow(r["path"], existing_path)) for r in rows.values())
    allowed.update(str((output / f"A_{c}_{f:04d}_{g:04d}.npz").resolve()) for c, f, g in pairs()-set(old_rows))
    def guard(event, arguments):
        if event == "open" and isinstance(arguments[0], (str, bytes, os.PathLike)):
            path = Path(os.fsdecode(arguments[0])).resolve()
            if path.suffix.lower() in (".png", ".jpg", ".jpeg", ".npz", ".npy") and str(path) not in allowed:
                raise RuntimeError(f"Flow generation cannot open non-native-LR input: {path}")
    sys.addaudithook(guard)
    started = time.monotonic(); torch.cuda.reset_peak_memory_stats()
    # weights=None avoids torchvision's downloader; strict loading uses only
    # the existing file already bound to the original C_T_SKHT_V2 digest.
    net = raft_large(weights=None, progress=False)
    net.load_state_dict(torch.load(weight_path, map_location="cpu", weights_only=True), strict=True)
    net = net.cuda().eval()
    with torch.inference_mode():
        for camera, first, second in sorted(pairs()-set(rows)):
            dest = output / f"A_{camera}_{first:04d}_{second:04d}.npz"
            if dest.exists():
                raise FileExistsError("Unregistered orphan output exists; preserve it and audit before resuming")
            started_pair = time.monotonic(); sources = []; tensors = []
            attempt_path = output / "last_attempt.json"
            write(attempt_path, dict(status="started_pair", camera=camera, first=first, second=second,
                registration_sha256=sha(plan_path), cost_before=cost.copy(), optimizer_updates=0))
            for frame in (first, second):
                obs = observations[camera, frame]; path = manifest_path.parent / obs["lr_path"]
                if sha(path) != obs["lr_sha256"]:
                    raise ValueError("Training LR bytes changed during generation")
                im = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
                if im.shape != (252, 336, 3):
                    raise ValueError("Native LR grid changed")
                tensor = torch.from_numpy(im.copy()).permute(2, 0, 1).unsqueeze(0).cuda().float()/127.5-1
                tensors.append(F.pad(tensor, (0, 0, 0, 4), mode="replicate"))
                sources.append(dict(camera=camera, frame=frame, path=str(path.resolve()), sha256=obs["lr_sha256"]))
            flow = []
            for direction, (a, b) in enumerate((tensors, tensors[::-1])):
                write(attempt_path, dict(status="before_RAFT_forward", camera=camera, first=first, second=second,
                    direction="forward" if direction == 0 else "backward", cost_completed=cost.copy(),
                    interrupted_forward_cost_is_ambiguous=True, optimizer_updates=0))
                flow.append(net(a, b, num_flow_updates=12)[-1][0, :, :252, :336].permute(1, 2, 0).cpu().numpy().astype(np.float32))
                cost["RAFT_directed_predictions"] += 1; cost["recurrent_updates"] += 12
                write(attempt_path, dict(status="completed_RAFT_forward", camera=camera, first=first, second=second,
                    direction="forward" if direction == 0 else "backward", cost_completed=cost.copy(), optimizer_updates=0))
            with dest.open("xb") as stream:
                np.savez_compressed(stream, forward=flow[0], backward=flow[1], original_grid=np.array([252, 336]),
                    network_grid=np.array([256, 336]), padding=np.array([0, 0, 0, 4]))
            validate_payload(dest, sha(dest))
            row = dict(camera=camera, first=first, second=second, branch="A", frame_gap=2,
                time_gap=observations[camera, second]["time"]-observations[camera, first]["time"],
                path=str(dest.relative_to(ROOT)), sha256=sha(dest), sources=sources,
                network_grid=[256, 336], padding=[0, 0, 0, 4], information="legal_LR_only",
                origin="incremental_registered_native_LR_A", seconds=time.monotonic()-started_pair,
                GPU=torch.cuda.get_device_name(), physical_gpu=os.environ["CUDA_VISIBLE_DEVICES"])
            rows[camera, first, second] = row; cost["new_bidirectional_pairs"] += 1
            write(progress_path, dict(status="partial_generation_not_formal_ready", registration_sha256=sha(plan_path),
                cost=cost, new_rows=[r for r in rows.values() if r["origin"] != "reused_original_native_LR_A"]))
            write(attempt_path, dict(status="committed_pair", camera=camera, first=first, second=second,
                cost_completed=cost.copy(), flow=record(dest), optimizer_updates=0))
            del tensors, flow; torch.cuda.empty_cache()
    completed = require_complete(rows)
    if record(__file__) != plan["source"] or software_identity(legacy, allow_existing_CUDA=True) != plan["software"]:
        raise ValueError("Generation source changed while running")
    torch.cuda.synchronize()
    _, _, final_inputs = manifest_records(manifest_path, verify_pixels=True)
    if final_inputs != inputs or sha(existing_path) != plan["existing_index"]["sha256"]:
        raise ValueError("Original input/index identities changed during generation")
    index = dict(status="completed_phase13_native_LR_flow", rows=[rows[k] for k in sorted(rows)],
        manifest_sha256=sha(manifest_path), weight_sha256=WEIGHT_SHA, num_flow_updates=12,
        estimator=POLICY["estimator"], algorithm=POLICY, registration=record(plan_path), decision=decision,
        coverage=completed, torch=str(torch.__version__), torchvision=legacy["torchvision"],
        GPU=torch.cuda.get_device_name(), physical_gpu=os.environ["CUDA_VISIBLE_DEVICES"],
        reused_original_hardware=legacy["gpu"], reused_original_index=plan["existing_index"],
        cost=dict(**cost, seconds_this_process=time.monotonic()-started,
                  peak_gpu_bytes=torch.cuda.max_memory_allocated()), source=record(__file__),
        original_index_unchanged=sha(existing_path)==plan["existing_index"]["sha256"],
        HR_images_read=False, B_flows_read=False, RAFT_weight_downloaded=False,
        phase13_training_authorized_by_this_generator=False)
    # Main index is published only after a staged candidate passes every payload,
    # metadata, source and full-coverage check. Failed candidates remain evidence.
    candidate = output / "candidate_index.json"
    write(candidate, index)
    _, validated = validated_rows(candidate, manifest_path, strict_A=True)
    require_complete(validated)
    candidate.replace(output / "index.json")
    write(output / "complete.json", dict(status="completed_verified_native_LR_flow", index=record(output / "index.json"),
                                        coverage=completed, cost=index["cost"]))
    print(json.dumps(dict(status=index["status"], cost=index["cost"])))


def self_check(path, manifest_path=DEFAULT_MANIFEST, existing_path=DEFAULT_EXISTING):
    assert len(pairs()) == 1121
    legacy = {(c, f, g): {} for c in ("cam02", "cam06", "cam12", "cam18")
              for f, g in ((38, 40), (40, 42), (78, 80), (80, 82))}
    report = coverage(legacy)
    assert report["available_pairs"] == 16 and report["missing_pair_count"] == 1105 and not report["formal_training_ready"]
    try:
        require_complete(legacy)
    except ValueError:
        incomplete_rejected = True
    else:
        raise AssertionError("Partial diagnostic coverage passed formal gate")
    complete = require_complete({key:{} for key in pairs()})
    assert complete["directed_predictions"] == 2242
    try:
        coverage({("cam00", 0, 2): {}})
    except ValueError:
        heldout_rejected = True
    else:
        raise AssertionError("Held-out camera entered training flow")
    import tempfile
    legacy_index = read(existing_path)
    first_A = copy.deepcopy(next(row for row in legacy_index["rows"] if row["branch"] == "A"))
    fixtures = {}
    fixture = copy.deepcopy(legacy_index); fixture["rows"] = [first_A, copy.deepcopy(first_A)]
    fixtures["duplicate_pair"] = fixture
    fixture = copy.deepcopy(legacy_index); fixture["rows"] = [dict(first_A, branch="B")]
    fixtures["B_in_formal_index"] = fixture
    fixture = copy.deepcopy(legacy_index); fixture["rows"] = [copy.deepcopy(first_A)]
    fixture["rows"][0]["sources"][0]["path"] = fixture["rows"][0]["sources"][0]["path"].replace("/lr/", "/hr/")
    fixtures["HR_source_path"] = fixture
    fixture = copy.deepcopy(legacy_index); fixture["num_flow_updates"] = 11
    fixtures["changed_RAFT_iterations"] = fixture
    fixture = copy.deepcopy(legacy_index); fixture["weight_sha256"] = "0" * 64
    fixtures["changed_weight_identity"] = fixture
    fixture = copy.deepcopy(legacy_index); fixture["rows"] = [dict(first_A, padding=[0, 0, 4, 0])]
    fixtures["changed_padding"] = fixture
    fixture_rejections = {}
    with tempfile.TemporaryDirectory(prefix="flow_prepare_cpu_checks_") as temporary:
        for name, fixture in fixtures.items():
            fixture_path = Path(temporary) / (name + ".json"); write(fixture_path, fixture)
            try:
                validated_rows(fixture_path, manifest_path, verify_payload=False, strict_A=True)
            except ValueError as error:
                fixture_rejections[name] = str(error)
            else:
                raise AssertionError(f"Invalid evidence was accepted: {name}")
        unsupported = Path(temporary) / "unsupported_decision.json"
        write(unsupported, dict(proceed_phase13=False, core_quality_supports_phase13=False))
        try:
            decision_gate(unsupported)
        except ValueError:
            fixture_rejections["unsupported_quality_decision"] = "rejected_before_GPU"
        else:
            raise AssertionError("Unsupported core decision opened generation gate")
    global OUT
    original_out = OUT
    with tempfile.TemporaryDirectory(prefix="flow_prepare_decision_CPU_") as temporary:
        OUT = Path(temporary)
        integrity_path = OUT / "final_integrity.json"
        quality_path = OUT / "quality_summary.json"
        state_path = OUT / "core_finalization_state.json"
        endpoint_rows = [dict(task=f"r{r}_{arm}") for r in (1, 2)
                         for arm in ("C1", "Jperm", "B2perm", "R", "G", "RG")]
        write(integrity_path, dict(status="passed_core_endpoint_evidence", endpoints=endpoint_rows))
        write(quality_path, dict(status="completed_core_evidence"))
        write(state_path, dict(status="core_complete_pending_research_and_visual_review",
                              final_integrity=record(integrity_path), quality_summary=record(quality_path)))
        decision_path = OUT / "synthetic_decision_fixture.json"
        positive = dict(proceed_phase13=True, core_quality_supports_phase13=True,
            blocking_data_protocol_bug_confirmed=False,
            data_protocol_audit_note="CPU synthetic fixture: internal consistency and cause matrix found no confirmed blocking protocol bug; external calibration bias remains unknown.",
            rationale="CPU synthetic support fixture, not a real quality decision.", core_finalization_state=record(state_path))
        try:
            write(decision_path, positive)
            decision_gate(decision_path)
            positive_no_confirmed_bug_accepted = True
            negative_decisions = dict(
                confirmed_blocking_bug=dict(positive, blocking_data_protocol_bug_confirmed=True),
                empty_data_protocol_note=dict(positive, data_protocol_audit_note=""),
                missing_explicit_bug_flag={k:v for k,v in positive.items() if k != "blocking_data_protocol_bug_confirmed"},
                obsolete_data_bug_excluded_claim={k:v for k,v in dict(positive, data_bug_excluded=True).items()
                                                  if k != "blocking_data_protocol_bug_confirmed"})
            for name, negative in negative_decisions.items():
                write(decision_path, negative)
                try:
                    decision_gate(decision_path)
                except ValueError as error:
                    fixture_rejections[name] = str(error)
                else:
                    raise AssertionError(f"Invalid data-protocol decision accepted: {name}")
            # The existing twelve-endpoint requirement must still reject eleven.
            write(integrity_path, dict(status="passed_core_endpoint_evidence", endpoints=endpoint_rows[:-1]))
            write(state_path, dict(status="core_complete_pending_research_and_visual_review",
                                  final_integrity=record(integrity_path), quality_summary=record(quality_path)))
            write(decision_path, dict(positive, core_finalization_state=record(state_path)))
            try:
                decision_gate(decision_path)
            except ValueError as error:
                fixture_rejections["eleven_core_endpoints"] = str(error)
            else:
                raise AssertionError("Incomplete core passed the unchanged endpoint gate")
        finally:
            OUT = original_out
    write(path, dict(status="passed_CPU_protocol_checks", negative_fixture_rejections=fixture_rejections,
        positive_no_confirmed_blocking_bug_decision_accepted=positive_no_confirmed_bug_accepted,
        positive_decision_is_synthetic_CPU_fixture_not_actual_core_quality_evidence=True,
        incomplete_16_of_1121_rejected=incomplete_rejected,
        heldout_camera_rejected=heldout_rejected, complete_1121_and_2242_directed_required=True,
        GPU_used=False, flow_inference_run=False, scene_updates=0, source=record(__file__)))
    print(json.dumps(dict(status="passed_CPU_protocol_checks")))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    modes = p.add_mutually_exclusive_group(required=True)
    modes.add_argument("--register", action="store_true"); modes.add_argument("--audit", action="store_true")
    modes.add_argument("--generate", action="store_true"); modes.add_argument("--self-check", action="store_true")
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--existing-index", type=Path, default=DEFAULT_EXISTING)
    p.add_argument("--registration", type=Path, default=DEFAULT_REGISTRATION)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--weight", type=Path); p.add_argument("--decision", type=Path, default=OUT / "phase13_decision.json")
    p.add_argument("--report", type=Path, default=OUT / "phase13/flow_prepare_CPU_checks_v5.json")
    p.add_argument("--require-complete", action="store_true")
    p.add_argument("--inherited-lock-fd", type=int, default=None, help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.register:
        register(args)
    elif args.audit:
        _, rows = validated_rows(args.existing_index, args.manifest,
                                strict_A=read(args.existing_index).get("status") == "completed_phase13_native_LR_flow")
        result = require_complete(rows) if args.require_complete else coverage(rows)
        write(args.report, dict(**result, index=record(args.existing_index), source=record(__file__)))
        print(json.dumps({k:v for k,v in result.items() if k != "missing_pairs"}))
    elif args.self_check:
        self_check(args.report, args.manifest, args.existing_index)
    else:
        try:
            generate(args)
        except BaseException:
            failure = dict(status="failed", traceback=traceback.format_exc(),
                scene_updates=0, partial_output_preserved=True, pid=os.getpid(),
                original_child_failure_receipts_preserved=True)
            write(args.registration / f"generation_failed_{os.getpid()}.json", failure)
            write(args.registration / "generation_failed.json", failure)
            raise


if __name__ == "__main__":
    main()
