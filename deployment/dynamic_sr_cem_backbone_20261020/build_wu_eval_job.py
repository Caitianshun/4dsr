"""Freeze A100-only existing-Wu render/eval or returned native-HRref eval jobs.

CPU preparation only: no model import, training, SSH or GPU submission. The
broker owns CUDA_VISIBLE_DEVICES. The source stager and run_job recheck frozen
file bytes before execution. Large checkpoint assets already covered by the
full verified asset receipt are not redundantly hashed by this builder.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path("deployment/dynamic_sr_cem_backbone_20261020")
BATCH = Path("output/dynamic_sr_cem_backbone_20261020")
WU_INVENTORY = BATCH / "wu_reference_inventory.json"
EVALUATOR = Path("deployment/dynamic_lr_baselines_20261020/evaluate_images.py")
METRIC_HELPER = Path("experiments/dynamic_sr_20260918/evaluate.py")
GPUS = {0: "GPU-5c08f287-3ffd-edf9-91ed-cd6db2690f3b", 1: "GPU-b79cd3fe-81f0-f449-30be-432e2857e517"}
SCENES = {"cook_spinach": "Cook07", "cut_roasted_beef": "Cut07"}
WU_ARMS = ["areaL0", "naive", "areaT", "areaTm", "w4", "oracle"]


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(8 << 20), b""):
            result.update(data)
    return result.hexdigest()


def relative(path):
    path = Path(path)
    if path.is_absolute():
        path = path.resolve().relative_to(ROOT)
    assert path.parts and ".." not in path.parts and not path.is_absolute()
    assert (ROOT / path).resolve().is_relative_to(ROOT)
    return path


def read(path):
    return json.loads((ROOT / relative(path)).read_text())


def rooted(path):
    return "{ROOT}/" + str(relative(path))


class Required:
    def __init__(self):
        self.files = {}

    def add(self, path, expected=None, size=None, rehash=True):
        path = relative(path)
        full = ROOT / path
        assert full.is_file(), "Required local file absent: " + str(path)
        assert size is None or full.stat().st_size == size, "Required file size differs: " + str(path)
        identity = sha(full) if rehash or expected is None else expected
        assert expected is None or identity == expected, "Frozen SHA differs: " + str(path)
        item = dict(path=str(path), sha256=identity)
        if str(path) in self.files:
            assert self.files[str(path)] == item, "Conflicting required identities: " + str(path)
        self.files[str(path)] = item

    def frozen(self, item):
        # Original checkpoints were fully hashed in prepare + remote asset
        # verification. Keep their exact expected SHA; stage_job still rehashes.
        self.add(item["path"], item["sha256"], item.get("bytes"), rehash=not str(item["path"]).endswith((".pt", ".pth")))


def asset_gate(path):
    path = relative(path)
    receipt = read(path)
    assert receipt["status"] == "verified" and receipt["all_listed_file_bytes_rehashed"] is True
    assert receipt.get("failure_count", 0) == 0
    assert receipt.get("root") == "/home/ubuntu/3DGS/4dsr", "Evaluation requires the verified A100 asset copy"
    inventory_path = relative(receipt["inventory_path"])
    assert sha(ROOT / inventory_path) == receipt["inventory_sha256"]
    inventory = read(inventory_path)
    assert inventory["schema"] == "CEM_area4_closed_input_inventory_v1"
    assert isinstance(inventory["files"], dict)
    assert receipt["file_count"] == receipt["verified_count"] == len(inventory["files"])
    # The full receipt already authenticated the same inventory. Check its
    # per-file identities as well, without reading asset pixels/checkpoints.
    assert set(receipt["files"]) == set(inventory["files"])
    for name, item in inventory["files"].items():
        verified = receipt["files"][name]
        assert verified["status"] == "verified" and verified["stable_during_read"]
        assert (verified["sha256"], verified["bytes"]) == (item["sha256"], item["bytes"])
    return path, inventory_path, receipt, inventory


def asset_covers(inventory, item):
    name = str(relative(item["path"]))
    assert name in inventory["files"], "Asset inventory lacks required evaluation input: " + name
    assert inventory["files"][name]["sha256"] == item["sha256"]
    if "bytes" in item:
        assert inventory["files"][name]["bytes"] == item["bytes"]


def shared_sources(required, receipt, inventory_path):
    for path in [Path(__file__), HERE / "run_job.py", HERE / "stage_job.py", EVALUATOR, METRIC_HELPER, receipt, inventory_path]:
        required.add(path)


def root_command(arguments):
    prefixes = (".venv/", "deployment/", "experiments/", "output/", "data/", "third_party/")
    return [rooted(value) if str(value).startswith(prefixes) else str(value) for value in arguments]


def wu(args, required, assets):
    inventory = read(WU_INVENTORY)
    assert inventory["schema"] == "Wu_area4_fixed_no_posthoc_reference_v1"
    assert len(inventory["tasks"]) == 12
    assert assets["source_inventory"] == dict(path=str(WU_INVENTORY), sha256=sha(ROOT / WU_INVENTORY))
    tasks = [task for task in inventory["tasks"] if task["scene"] == args.scene]
    assert [task["arm"] for task in tasks] == WU_ARMS
    assert len({task["id"] for task in tasks}) == 6
    assert all(task["label"] == SCENES[args.scene] for task in tasks)
    # All common sources are required even though each scene uses only half of
    # the historical checkpoint assets and only its own 300 held-out HR images.
    items = list(inventory["common_required_files"]) + [inventory["evidence"]]
    items += inventory["test_HR_required_files"][SCENES[args.scene]]
    assert len(inventory["test_HR_required_files"][SCENES[args.scene]]) == 300
    for task in tasks:
        assert task["original_training_cull"] is False and task["original_training_culls"] == []
        assert task["irreversible_posthoc_cull_detected"] is False
        items += task["required_files"]
    for item in items:
        asset_covers(assets, item)
        required.frozen(item)
    required.add(WU_INVENTORY)
    # It is also in common_required_files; the duplicate must have the same SHA.
    required.add(HERE / "wu_reference.py")
    out = BATCH / "execution" / f"Wu_{args.scene}_v1"
    commands = []
    outputs = []
    for task in tasks:
        target = relative(task["output_relative"])
        assert not (ROOT / target).exists(), "Preserve existing Wu reference output: " + str(target)
        assert not out.is_relative_to(target) and not target.is_relative_to(out)
        outputs.append(str(target))
        assert task["render_command"] == [".venv/bin/python", str(HERE / "wu_reference.py"), "render", "--inventory", str(WU_INVENTORY), "--task", task["id"]]
        metric = task["metric_command"]
        assert metric[:2] == [".venv/bin/python", str(EVALUATOR)]
        assert metric[metric.index("--manifest") + 1] == task["manifest"]["path"]
        assert metric[metric.index("--render-dir") + 1] == str(target / "renders_hr")
        assert metric[metric.index("--out") + 1] == str(target / "common_A100_metrics")
        commands += [dict(name=f"Wu_{task['arm']}_raw_render", argv=root_command(task["render_command"]), cwd="."),
                     dict(name=f"Wu_{task['arm']}_common_A100_metrics", argv=root_command(metric), cwd=".")]
    return dict(arm="wu_reference", method="Wu_area4_fixed_raw_reference", out=str(out), commands=commands,
        return_paths=[str(out), *outputs], training_HR=False,
        reference_inventory=dict(path=str(WU_INVENTORY), sha256=sha(ROOT / WU_INVENTORY)),
        task_ids=[task["id"] for task in tasks], parameter_updates=0,
        inference_policy=inventory["inference_policy"],
        comparison_limit="existing selected development endpoints; raw native render, no posthoc cull/affine/warp; no new training")


def hrref(args, required, assets):
    run = relative(args.run or BATCH / "runs" / f"{args.scene}_native_HRref_v1")
    manifest = Path(f"data/dynamic_sr/full_time_20261007/prepared/{args.scene}_area4/manifest_train_ready_seed20261007.json")
    data = read(manifest)
    assert data["scene"] == args.scene and data["physical_lr_derivation"]["formation"] == "area4"
    assert data["splits"]["test"] == ["cam00"] and data["frame_indices"] == list(range(300))
    stage_path, render_path = run / "stage2.json", run / "renders_hr/render_receipt.json"
    stage, rendering = read(stage_path), read(render_path)
    assert stage["status"] == "completed" and stage["stage"] == 2 and stage["endpoint"] == 50000
    assert stage["train_level"] == "hr" and stage["train_HR_used"] is True and stage["heldout_pixels_used"] is False
    assert stage["manifest"]["sha256"] == sha(ROOT / manifest) and stage["native_schedule_preserved"] is True
    assert rendering["status"] == "completed" and rendering["kind"] == "native" and rendering["iteration"] == 50000
    assert rendering["camera"] == "cam00" and rendering["scope"] == "heldout_300"
    assert rendering["scene"] == args.scene and rendering["frames"] == rendering["finite_before_clip_quantization_passed_frames"] == 300
    assert rendering["frame_indices"] == list(range(300)) and rendering["raw_prediction_shape"] == [3, 1008, 1344]
    assert rendering["quantization"] == "np.rint(np.clip(CHW_to_HWC_float32,0,1)*255).astype(np.uint8); PIL RGB PNG"
    assert rendering["manifest"]["sha256"] == sha(ROOT / manifest)
    assert rendering["checkpoint"]["sha256"] == stage["final_checkpoint"]["sha256"]
    assert rendering["training_receipt"]["sha256"] == sha(ROOT / stage_path)
    assert rendering["heldout_pixels_read"] is False
    records = rendering["records"]
    assert [item["frame"] for item in records] == list(range(300))
    assert [item["path"] for item in records] == [f"{frame:04d}.png" for frame in range(300)]
    expected = [dict(path=str(manifest), sha256=sha(ROOT / manifest)),
                dict(path=str(stage_path), sha256=sha(ROOT / stage_path)),
                dict(path=str(render_path), sha256=sha(ROOT / render_path))]
    expected += [dict(path=str(run / "renders_hr" / row["path"]), sha256=row["sha256"]) for row in records]
    test = sorted((item for item in data["observations"] if item["split"] == "test"), key=lambda x: x["frame_index"])
    assert [(item["camera_id"], item["frame_index"]) for item in test] == [("cam00", frame) for frame in range(300)]
    expected += [dict(path=str(manifest.parent / item["hr_path"]), sha256=item["hr_sha256"]) for item in test]
    for item in expected:
        asset_covers(assets, item)
        required.frozen(item)
    out = BATCH / "execution" / f"HRref_{args.scene}_A100_v1"
    metrics = BATCH / "shared_A100_metrics/native_HRref" / args.scene
    assert not (ROOT / metrics).exists(), "Preserve existing A100 HRref metrics"
    # The checkpoint identity is frozen through the completed stage + render
    # receipts. Neither local generation nor remote evaluation needs its bytes.
    return dict(arm="native_HRref", method="CEM_native_HRref_common_A100_reevaluation", out=str(out),
        commands=[dict(name="native_HRref_common_A100_metrics", cwd=".", argv=[rooted(".venv/bin/python"), rooted(EVALUATOR),
            "--manifest", rooted(manifest), "--render-dir", rooted(run / "renders_hr"), "--out", rooted(metrics),
            "--method", "CEM_area4_native_HRref"])], return_paths=[str(out), str(metrics)],
        training_HR=True, parameter_updates=0, evaluation_only=True,
        training_stage2=dict(path=str(stage_path), sha256=sha(ROOT / stage_path)),
        source_render_receipt=dict(path=str(render_path), sha256=sha(ROOT / render_path)),
        model_identity=dict(source_path=stage["final_checkpoint"]["path"], sha256=stage["final_checkpoint"]["sha256"],
                            checkpoint_bytes_required_for_reevaluation=False),
        comparison_limit="independent privileged training-HR diagnostic; same returned300 8bitPNG re-evaluated in common A100 environment")


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0].startswith("--"):
        argv.insert(0, "wu")
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="kind", required=True)
    for kind in ["wu", "HRref"]:
        p = sub.add_parser(kind)
        p.add_argument("--scene", choices=list(SCENES), required=True)
        p.add_argument("--gpu", type=int, choices=list(GPUS), required=True)
        p.add_argument("--asset-receipt", type=Path, required=True)
        p.add_argument("--output", type=Path, required=True)
        if kind == "HRref":
            p.add_argument("--run", type=Path, help="Returned nativeHRref run directory; only metadata/300PNGs required")
    args = parser.parse_args(argv)
    output = ROOT / relative(args.output)
    assert not output.exists(), "Never overwrite a frozen job"
    receipt_path, inventory_path, receipt, assets = asset_gate(args.asset_receipt)
    required = Required()
    shared_sources(required, receipt_path, inventory_path)
    details = wu(args, required, assets) if args.kind == "wu" else hrref(args, required, assets)
    assert not (ROOT / details["out"]).exists(), "Preserve existing execution directory"
    short = args.scene.split("_")[0]
    job_id = f"codex-cem-{'wuref' if args.kind == 'wu' else 'hrref-eval'}-{short}-g{args.gpu}-20261020-v1".lower()
    assert re.fullmatch(r"[a-z][a-z0-9-]{0,63}", job_id)
    job = dict(protocol="CEM_area4_backbone_control_v1", input_formation="area4", heldout_used_for_training=False,
        job_id=job_id, scene=args.scene, host="a100-train", gpu_uuid=GPUS[args.gpu],
        builder_source_sha256=sha(Path(__file__)), required_files=[required.files[key] for key in sorted(required.files)],
        input_receipts=[dict(path=str(receipt_path), inventory_sha256=receipt["inventory_sha256"])],
        env=dict(TMPDIR="/tmp", TMP="/tmp", TEMP="/tmp", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                 OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1"), **details)
    assert "CUDA_VISIBLE_DEVICES" not in job["env"]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(job, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(job_id=job_id, path=str(output.relative_to(ROOT)), sha256=sha(output),
                         required_files=len(job["required_files"]), commands=len(job["commands"]), GPU_calls=0)))


if __name__ == "__main__":
    main()
