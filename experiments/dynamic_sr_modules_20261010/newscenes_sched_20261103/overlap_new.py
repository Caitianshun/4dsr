"""Scene/parent adapter around the unmodified historical coarse producer.

plan/prepare are for Claude's admitted runtime with complete input access.
regression is the restricted, payload-free historical CPU constructor replay.
prepare never trains: it calls the original full rendering loop and closure.
"""
from __future__ import annotations

import argparse
import ast
from contextlib import ExitStack
import importlib.metadata
from pathlib import Path
import sys
from types import SimpleNamespace

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sched_common import *

NATIVE = "full_author_LR_prefix_checkpoint_v1"
CULL = "cullretrain_lr_v1"
STATE = dict(stage="fine", coarse_iteration=3000, fine_iteration=14000, optimizer_resets=2)


def validate_scope(plan):
    loading = plan["parent_loading"]
    require(plan["scene"] in SCENES and plan["seed"] == SEED, "Scene/seed scope differs")
    require(loading["family"] in (NATIVE, CULL), "Parent family differs")
    if loading["family"] == CULL:
        require(loading["arm"] == "wu_author_area4" and loading["affine"] is False
                and loading["cull"] is False and loading["prune_gate"] == "author",
                "Only area4 author parent allowed for new overlaps")
    else:
        require(plan["scene"] == "flame_steak", "Native replay only Flame07")
        coarse_module.held.validate_authorization(plan["confirmation_authorization"],
                                                  plan["scene"], plan["seed"])
    require(loading["adapter"] == entry(__file__), "Parent adapter changed")


def identity_validator():
    """Replace only the historical held-scene/authorization gate; retain checks."""
    coarse, _, _ = science_modules()
    p = FP / "full_held_parent_overlap.py"
    source = p.read_text()
    tree = ast.parse(source)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "validate_identity")
    removed = [
        n for n in fn.body if isinstance(n, ast.Expr) and
        (ast.get_source_segment(source, n).startswith("held.validate_authorization(")
         or ast.get_source_segment(source, n).startswith("require(plan['scene'] in held.HELD"))
    ]
    require(len(removed) == 2, "Historical scene gate layout changed")
    fn.body = [n for n in fn.body if n not in removed]
    call = ast.parse("validate_scope(plan)").body[0]
    fn.body.insert(0, call)
    tree = ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))
    env = dict(vars(coarse), validate_scope=validate_scope)
    exec(compile(tree, str(p), "exec"), env)
    return env["validate_identity"]


def physical_parent(path, manifest_ref):
    """CPU SHA/sidecar gate; tensor restore is independently checked at prepare."""
    path = local(path)
    done = read(path.parent / "complete.json")
    side = read(path.with_suffix(".json"))
    require(done["kind"] == "lr" and done["status"] == "completed"
            and done["seed"] == SEED and done["arm"] == "wu_author_area4",
            "Completed author area4 LR parent required")
    args = done["args"]
    require(args["affine"] is False and args["cull"] is False
            and args["prune_gate"] == done["prune_gate"] == "author"
            and done["affine"] is None and done["culler"] is None and done["culls"] == [],
            "Affine/culling/changed pruning parent refused")
    require(done["state"] == STATE and side["schema"] == CULL
            and side["metadata"] == dict(state=STATE, arm="wu_author_area4"),
            "Complete cullretrain author state/metadata differs")
    ref = entry(path)
    require(done["checkpoint"] == ref and {k: side[k] for k in ("path", "sha256")} == ref,
            "Checkpoint/sidecar SHA differs")
    pp = local(done["parent"]) / "config.json"
    plan = read(pp)
    require(plan["data"]["manifest"] == manifest_ref and plan["scene"] == done["scene"]
            and plan["seed"] == SEED, "Physical parent registry differs")
    import protocol_new
    cfg, _ = protocol_new.configuration(done["scene"])
    require(plan["configuration"] == cfg, "Author configuration differs")
    for name, value in plan["runtime_source_sha256"].items():
        require(sha(RUNTIME / name) == value, "Parent author runtime changed")
    return dict(checkpoint=ref, complete=entry(path.parent / "complete.json"), plan=entry(pp),
                sidecar=entry(path.with_suffix(".json")), state=STATE), plan


def source_files(upstream):
    coarse, _, _ = science_modules()
    value = coarse_source_files(upstream)
    for name in ("sched_common.py", "overlap_new.py", "acceptance_new.py"):
        value["project"][rel(CODE / name)] = sha(CODE / name)
    return value


# Keep actual original function objects, not duplicated scientific algorithms.
coarse_module, _, native_module = science_modules()
coarse_source_files = coarse_module.source_files
native_checkpoint_payload = native_module.checkpoint_payload
native_restore_model = native_module.restore_native_model


def build_plan(scene, parent, family, plan_path):
    coarse, _, native = science_modules()
    require(scene in SCENES, "Unknown scene")
    require(importlib.metadata.version("numpy") == "1.26.4", "Use NumPy1.26.4")
    os.environ["FOURDSR_UPSTREAM"] = str(RUNTIME)
    os.environ["FOURDSR_ROOT"] = str(ROOT)
    previous_cuda = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    try:
        if family == NATIVE:
            require(scene == "flame_steak", "Native only for Flame07 replay")
            historical = read(HIST_PLAN)
            a = SimpleNamespace(
                upstream=str(RUNTIME), seed=SEED,
                manifest=historical["data"]["manifest"]["path"], parent=rel(parent),
                parent_complete=rel(local(parent).parent / "complete.json"),
                readiness=historical["readiness"]["path"],
                confirmation_authorization=historical["confirmation_authorization"]["path"],
                parent_acceptance=historical["parent_acceptance"]["path"],
                parent_index=rel(FPOUT / "full_LR_prefixes/checkpoint_index.json"))
            with attributes(coarse, source_files=source_files):
                value = coarse.build_plan(a)
        else:
            require(family == CULL, "Unknown parent family")
            import protocol_new
            mp = PREP / f"{scene}_area4/manifest_train_ready_seed{SEED}.json"
            manifest, data, inputs = protocol_new.validate_manifest(
                mp, formation="area4", verify_lr_bytes=True)
            pv, pp = physical_parent(parent, data["manifest"])
            require(pp["scene"] == scene and pp["runtime_source_sha256"] == source_files(RUNTIME)["upstream"],
                    "Actual parent scene/runtime differs")
            registration_path = local(plan_path).parent / "scene_registration.json"
            registration = dict(schema="newscene_area4_overlap_scope_v1", scene=scene, seed=SEED,
                                manifest=data["manifest"], parent=pv["checkpoint"],
                                role_preserved=manifest["role"], source=entry(__file__),
                                source_completion=pv["complete"],
                                operational_admission_granted=False,
                                adaptation="Area4 author LR parent replaces bicubic author LR prefix")
            authorization = entry(write_new(registration_path, registration))
            accepted_entry = dict(scene=scene, seed=SEED, status="completed_wu_author_area4_parent_SHA_checked",
                                  checkpoint=pv["checkpoint"], complete=pv["complete"])
            a = SimpleNamespace(seed=SEED, readiness=protocol_new.READINESS)
            env = dict(vars(coarse), m=manifest, data=data, inputs=inputs,
                       parent=pv, parent_plan=pp, acceptance=pv["complete"],
                       matches=[accepted_entry], authorization=authorization,
                       coarse=coarse.coarse_size(manifest["resolutions"]["lr"]),
                       n=len(manifest["splits"]["train"]), rows=coarse.legal_rows(manifest),
                       upstream=RUNTIME, a=a, prefix=native.prefix, source_files=source_files)
            value = extract_plan_expression(env)
        value["parent_loading"] = dict(family=family, adapter=entry(__file__),
                                      arm="wu_author_area4" if family == CULL else "native_author_prefix",
                                      affine=False, cull=False, prune_gate="author",
                                      payload_format_relabelled=False)
        identity_validator()(value)
        no_torch()
        return value
    finally:
        if previous_cuda is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = previous_cuda


def verify_plan(plan, upstream):
    identity_validator()(plan)
    require(source_files(upstream) == plan["source_files"], "Plan source tree differs")
    mp = bound(plan["data"]["manifest"])
    m, inputs, center, extent = manifest_metadata(mp)
    require(inputs == plan["legal_LR_SHA_inputs"], "Legal LR inventory changed")
    for ref in inputs:
        bound(ref)
    require(plan["data"]["train_camera_center"] == center
            and plan["data"]["cameras_extent"] == extent, "Actual scene geometry differs")
    if plan["parent_loading"]["family"] == NATIVE:
        pv, _ = native_module.full_parent(bound(plan["parent"]["checkpoint"]),
                                         bound(plan["parent"]["complete"]), entry(mp), SEED)
    else:
        pv, _ = physical_parent(bound(plan["parent"]["checkpoint"]), entry(mp))
    require(pv == plan["parent"], "Parent provenance changed")
    coarse = coarse_module
    require(plan["legal_keys"] == [[r["camera_id"], int(r["frame_index"])] for r in coarse.legal_rows(m)]
            and plan["camera_K_coarse"] == {
                c: coarse.scaled_K(m["cameras"][c]["K_lr"], m["resolutions"]["lr"], plan["coarse_size"])
                for c in m["splits"]["train"]}, "Population/intrinsics changed")
    for key in ("parent_acceptance", "readiness", "confirmation_authorization"):
        bound(plan[key])
    require(plan["metadata_runtime"] == dict(python=sys.version.split()[0],
            numpy_distribution=importlib.metadata.version("numpy")),
            "Use the same interpreter for plan and prepare")
    return m


def checkpoint_payload(path, torch, plan, schema, device):
    actual = read(Path(path).with_suffix(".json"))["schema"]
    require(actual in (NATIVE, CULL), "Unsupported real payload family")
    # Original byte/sidecar/embedded-plan/metadata checks apply to both schemas.
    return native_checkpoint_payload(path, torch, plan, actual, device)


def restore_model(g, payload, opt, torch, np, plan, schema):
    if payload["schema"] == NATIVE:
        return native_restore_model(g, payload, opt, torch, np, plan, NATIVE)
    require(payload["schema"] == CULL and payload["plan"] == plan
            and payload["metadata"] == dict(state=STATE, arm="wu_author_area4")
            and payload["affine"] is None and len(payload["model"]) == 14,
            "Not an author cullretrain 14-tuple")
    native = native_module
    prefix = native.prefix
    g.restore(payload["model"], opt)
    g._deformation_accum = payload["deformation_accum"]
    params = prefix.parameter_map(g)
    require(set(params) == set(payload["parameter_gradients"]), "Saved gradient keys differ")
    for name, p in params.items():
        p.grad = payload["parameter_gradients"][name]
    require(native.tree_digest(g.capture()) == native.tree_digest(payload["model"]),
            "Model/one-Adam/topology restore not exact")
    require(native.tree_digest({k: p.grad for k, p in params.items()})
            == native.tree_digest(payload["parameter_gradients"]), "Gradient restore differs")
    require(native.tree_digest(g._deformation_accum) == native.tree_digest(payload["deformation_accum"]),
            "Deformation accumulation differs")
    require(tuple(group["name"] for group in g.optimizer.param_groups) ==
            ("xyz", "deformation", "grid", "f_dc", "f_rest", "opacity", "scaling", "rotation"),
            "Eight-group single Adam required")
    prefix.restore_rng(payload["rng"], torch, np)
    require(native.tree_digest(prefix.rng_state(torch, np)) == native.tree_digest(payload["rng"]),
            "Global RNG restore differs")
    return dict(one_Adam_exact=True, parameters_exact=True, topology_buffers_exact=True,
                saved_gradients_exact=True, global_RNG_exact=True, no_children=True,
                optimizer_conversion=False, checkpoint_family=CULL,
                embedded_audit_available=False, exact_capture_and_gradient_comparison=True)


def admitted_ownership(auth, plan_ref, uuid, lock):
    """Operational bridge only. Claude must supply real fixed-service admission."""
    require(auth["schema"] == "fixed_service_overlap_admission_v1"
            and auth["project_id"] == "4dsr" and auth["admitted"] is True
            and auth["interpreter"] == TRAINPY and auth["env_id"] == "mml_bw",
            "Fixed training-service admission required")
    require(auth["plan"] == plan_ref and auth["producer_source"] == entry(FP / "full_held_parent_overlap.py")
            and auth["manager_source"] == entry(__file__) and auth["resource_adapter"] == entry(__file__),
            "Admitted plan/algorithm/adapter differs")
    from acceptance_new import validate_admission
    validate_admission(auth)
    require(auth["gpu_uuid"] == uuid and os.environ.get("CUDA_VISIBLE_DEVICES") == uuid
            and auth["runtime_root"] == str(ROOT) and auth["physical_gpu_lock"] == str(lock),
            "Admitted GPU/root/lock differs")
    require(Path(sys.executable).resolve() == Path(TRAINPY).resolve(), "Wrong runtime interpreter")
    unit = auth["unit"]
    require(unit.endswith(".service") and "/" not in unit, "Actual service unit required")
    cgroup = Path("/proc/self/cgroup").read_text()
    require(any(unit in Path(line.split(":")[-1]).parts for line in cgroup.splitlines()),
            "Not in admitted service")
    command = [os.fsdecode(v) for v in Path("/proc/self/cmdline").read_bytes().rstrip(b"\0").split(b"\0")]
    declared = auth["producer_command"]
    require(len(command) == len(declared) and len(command) >= 3
            and Path(command[0]).resolve() == Path(declared[0]).resolve()
            and command[1] == declared[1] == "-u"
            and local(command[2]).resolve() == local(declared[2]).resolve() == Path(__file__).resolve()
            and command[3:] == declared[3:], "Admitted producer argv differs")
    return dict(pid=os.getpid(), cmdline=command, cgroup=cgroup, unit=unit,
                authorization=auth, manager_source=entry(__file__),
                resource_adapter=entry(__file__), check_gpu_policy="fixed_service_admission")


def prepare(a):
    coarse, _, native = science_modules()
    plan = read(a.registered_plan)
    auth = read(a.authorization)
    require(a.cpu_threads > 0, "Positive thread count")
    a.upstream = str(RUNTIME)
    a.gpu_uuid = auth["gpu_uuid"]
    a.lock = auth["physical_gpu_lock"]
    output_path(a.out)
    output_path(a.lock)
    a.operator_resource_resolved = True
    os.environ["FOURDSR_UPSTREAM"] = str(RUNTIME)
    os.environ["FOURDSR_ROOT"] = str(ROOT)

    def service_resource(uuid):
        admitted_ownership(auth, entry(a.registered_plan), uuid, local(a.lock))
        return json.dumps(dict(policy="fixed_service_admission", gpu_uuid=uuid,
                               evidence=auth["admission_evidence"]))

    with ExitStack() as stack:
        stack.enter_context(attributes(coarse, OUTPUT_ROOT=OUT / "overlap",
            source_files=source_files, verify_plan=verify_plan,
            validate_identity=identity_validator(), ownership=admitted_ownership))
        stack.enter_context(attributes(native, checkpoint_payload=checkpoint_payload,
                                       restore_native_model=restore_model))
        stack.enter_context(attributes(native.prefix, check_gpu=service_resource))
        # This exact function owns all moment/deformation/pair loops, pixel
        # guarding, counters, failures, native invariance and observer RNG.
        complete = coarse.prepare(a, plan)
        proof = coarse.validate_result(plan, complete)
    write_new(local(a.out) / "all_output_validation.json", dict(
        schema="newscene_overlap_all_output_validation_v1",
        status="passed_original_full_output_SHA_and_native_observer_closure",
        producer_complete=entry(complete), index=read(complete)["index"], proof=proof,
        wrapper=entry(__file__), process_exit_observed=False,
        note="Claude must separately bind fixed-service normal Exit0 before schedule acceptance"))
    return dict(complete=entry(complete), validation=entry(local(a.out) / "all_output_validation.json"))


def regression(out):
    coarse, _, native = science_modules()
    old = read(HIST_PLAN)
    prefix_plan = read(bound(old["parent"]["plan"]))
    completion = read(bound(old["parent"]["complete"]))
    m, inputs, center, extent = manifest_metadata(bound(old["data"]["manifest"]))
    data = copy.deepcopy(prefix_plan["data"])
    data.update(manifest=entry(bound(old["data"]["manifest"])), train_cameras=m["splits"]["train"],
                training_observations=len(inputs), training_LR_sources_sha256=digest(inputs),
                train_camera_center=center, cameras_extent=extent,
                resolution_LR=m["resolutions"]["lr"], resolution_HR=m["resolutions"]["hr"])
    require(completion["checkpoint"] == old["parent"]["checkpoint"]
            and completion["state"] == old["parent"]["state"]
            and completion["topology"] == old["parent"]["audit"], "Historical parent metadata differs")
    verified, declared = [], []
    for path, expected in old["source_files"]["project"].items():
        if path.startswith("experiments/dynamic_sr_confidence_geometry_20261006/"):
            declared.append(dict(path=path, sha256=expected))
        else:
            require(sha(path) == expected, "Historical project source differs")
            verified.append(entry(path))
    for path, expected in old["source_files"]["upstream"].items():
        require(sha(RUNTIME / path) == expected, "Historical upstream source differs")
        verified.append(entry(RUNTIME / path))
    env = dict(vars(coarse), m=m, data=data, inputs=inputs,
               parent=copy.deepcopy(old["parent"]), acceptance=old["parent_acceptance"],
               matches=[old["accepted_parent_entry"]], authorization=old["confirmation_authorization"],
               coarse=coarse.coarse_size(m["resolutions"]["lr"]), n=len(m["splits"]["train"]),
               rows=coarse.legal_rows(m), upstream=RUNTIME, prefix=native.prefix,
               a=SimpleNamespace(seed=SEED, readiness=old["readiness"]["path"]),
               source_files=lambda _: copy.deepcopy(old["source_files"]))
    value = extract_plan_expression(env)
    diff = differences(old, value)
    allowed = {"/metadata_runtime/python", "/metadata_runtime/numpy_distribution"}
    result = dict(status="passed" if all(d["field"] in allowed for d in diff) else "failed",
                  comparison="historical CPU plan constructor replay from available metadata",
                  original=entry(HIST_PLAN), differing_fields=len(diff),
                  all_non_runtime_fields_equal=all(d["field"] in allowed for d in diff),
                  verified_sources=verified, source_declared_not_rehashed=declared,
                  fresh_full_registration=False, parent_payload_read=False,
                  LR_images_read=0, inherited_historical_LR_SHA_count=data["training_LR_bytes_SHA_verified"],
                  inherited_unrevalidated=["initialization receipt/arrays", "parent payload/sidecar",
                      "parent acceptance/index", "confirmation authorization/private freeze chain"],
                  explanation="Stored verification fields are historical input evidence, not new certification")
    p = write_new(local(out) / "reconstructed_plan.json", value)
    result["reconstructed"] = entry(p)
    write_new(local(out) / "field_differences.json", diff)
    write_new(local(out) / "result.json", result)
    require(result["status"] == "passed", "Plan regression failed; see field_differences")
    no_torch()
    return {k: v for k, v in result.items() if k != "verified_sources"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("plan", "prepare", "regression"))
    p.add_argument("--scene", choices=tuple(SCENES))
    p.add_argument("--parent")
    p.add_argument("--family", choices=(NATIVE, CULL), default=CULL)
    p.add_argument("--registered-plan")
    p.add_argument("--out")
    p.add_argument("--authorization")
    p.add_argument("--cpu-threads", type=int, default=2)
    a = p.parse_args()
    if a.mode == "regression":
        require(a.out, "Fresh regression out required")
        value = regression(a.out)
    elif a.mode == "plan":
        require(a.scene and a.parent and a.registered_plan, "Scene/parent/plan required")
        value = build_plan(a.scene, a.parent, a.family, a.registered_plan)
        write_new(a.registered_plan, value)
        value = dict(plan=entry(a.registered_plan), planned=value["planned"], GPU_calls=0)
    else:
        require(a.registered_plan and a.authorization and a.out, "Plan/admission/out required")
        value = prepare(a)
    print(json.dumps(value, ensure_ascii=False))


if __name__ == "__main__":
    main()
