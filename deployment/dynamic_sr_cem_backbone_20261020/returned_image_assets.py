"""Prepare one returned CTS native-HRref-v2 image set for A100 evaluation.

--probe performs local contracts and one read-only remote SHA inventory.
--stage additionally transfers missing files once, runs frozen verify_assets.py
on A100, and returns its exact receipt. Neither mode imports Torch, queries GPUs,
trains, dispatches, repairs conflicts or retries. Run only AFTER the real HRref
has returned; this tool does not wait for or synthesize an experiment result.

The fixed A100 receipt is assets/a100_{scene}_HRref_v2_images_verified_v1.json.
Pass --run .../{scene}_native_HRref_v2 explicitly to build_wu_eval_job.py HRref;
that frozen builder's default still names v1. Full model assets and cam01
diagnostics remain the responsibility of the original complete collector.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import time
import traceback
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
HERE = Path("deployment/dynamic_sr_cem_backbone_20261020")
BATCH = Path("output/dynamic_sr_cem_backbone_20261020")
REMOTE_ROOT = "/home/ubuntu/3DGS/4dsr"
SCENES = ("cook_spinach", "cut_roasted_beef")
METADATA = ("job.json", "status.json", "complete.json", "stage1.json", "stage2.json", "result_index.json", "renders_hr/render_receipt.json")


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def contracts():
    builder = load_module("returned_HRref_frozen_builder", ROOT / HERE / "build_wu_eval_job.py")
    verifier = load_module("returned_HRref_frozen_verifier", ROOT / HERE / "verify_assets.py")
    builder.ROOT = ROOT
    return builder, verifier


def utc():
    return datetime.now(timezone.utc).isoformat()


def new_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + str(os.getpid()) + ".tmp")
    with temporary.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink()


def paths(scene):
    stem = BATCH / "assets" / ("a100_" + scene + "_HRref_v2_images")
    return {name: Path(str(stem) + suffix) for name, suffix in {
        "inventory": "_inventory_v1.json", "receipt": "_verified_v1.json",
        "probe": "_probe_v1.json", "stage": "_stage_v1.json", "local_verification": "_local_verified_v1.json"}.items()}


def build_inventory(scene, run):
    """Use real manifest identities and returned metadata; no checkpoint IO."""
    from PIL import Image
    builder, _ = contracts()
    run = builder.relative(run)
    assert run == BATCH / "runs" / (scene + "_native_HRref_v2"), "Only the explicitly versioned returned CTS HRref run"
    manifest = Path("data/dynamic_sr/full_time_20261007/prepared") / (scene + "_area4") / "manifest_train_ready_seed20261007.json"
    data = builder.read(manifest)
    assert data["scene"] == scene and data["frame_indices"] == list(range(300))
    assert data["splits"]["test"] == ["cam00"] and data["physical_lr_derivation"]["formation"] == "area4"
    returned = {name: builder.read(run / name) for name in METADATA}
    job, index = returned["job.json"], returned["result_index.json"]
    assert job["protocol"] == "CEM_area4_backbone_control_v1" and job["arm"] == "native_HRref" and job["scene"] == scene
    assert job["out"] == str(run) and job["input_formation"] == "area4" and job["heldout_used_for_training"] is False
    job_sha = builder.sha(ROOT / run / "job.json")
    for name in ("status.json", "complete.json"):
        completed = returned[name]
        assert completed["status"] == "completed" and completed["job_id"] == job["job_id"] and completed["job_sha256"] == job_sha
        assert completed["arm"] == "native_HRref" and completed["scene"] == scene
        assert len(completed["stages"]) == len(job["commands"]) and all(row["returncode"] == 0 for row in completed["stages"])
    assert index["status"] == "native_endpoints_and_300_renders_verified" and index["train_level"] == "hr"
    manifest_sha = builder.sha(ROOT / manifest)
    assert index["manifest_sha256"] == manifest_sha and index["heldout_used_for_training"] is False
    sources = {row["path"]: row["sha256"] for row in job["required_files"]}
    native_path = "experiments/dynamic_sr_cem_area_20261020/native_stage.py"
    render_path = "experiments/dynamic_sr_cem_area_20261020/render_hr.py"
    camera_path = "experiments/dynamic_sr_cem_area_20261020/sr_stage.py"
    for path in (native_path, render_path, camera_path):
        assert builder.sha(ROOT / path) == sources[path], "Returned frozen source identity differs: " + path
    for number, endpoint in ((1, 40000), (2, 50000)):
        stage = returned[f"stage{number}.json"]
        assert stage["status"] == "completed" and stage["stage"] == number and stage["endpoint"] == endpoint
        assert stage["train_level"] == "hr" and stage["train_HR_used"] is True and stage["heldout_pixels_used"] is False
        assert stage["native_schedule_preserved"] is True and not stage["parent_added_affine"] and not stage["parent_added_warp"]
        assert stage["manifest"]["sha256"] == manifest_sha and stage["entry_sha256"] == sources[native_path]
        row = index["stages"][number - 1]
        assert Path(row["path"]).name == f"stage{number}.json" and row["sha256"] == builder.sha(ROOT / run / f"stage{number}.json")
        assert row["checkpoint_sha256"] == stage["final_checkpoint"]["sha256"]
    stage1, stage2 = returned["stage1.json"], returned["stage2.json"]
    assert stage2["author_optimizer_restored"] is True and stage2["start_checkpoint"]["sha256"] == stage1["final_checkpoint"]["sha256"]
    assert stage1["source"]["LR_initialization_pointcloud_sha256"] == stage2["source"]["LR_initialization_pointcloud_sha256"]
    rendering = returned["renders_hr/render_receipt.json"]
    assert rendering["renderer_source_sha256"] == sources[render_path] and rendering["camera_helper_sha256"] == sources[camera_path]
    assert not rendering["training_pixels_read"] and rendering["colour_affine"] == rendering["warp"] == "identity"
    assert index["renders_receipt_sha256"] == builder.sha(ROOT / run / "renders_hr/render_receipt.json")
    files = {}

    def add(path, expected, role, frame=None):
        path = builder.relative(path)
        full = ROOT / path
        before = full.stat()
        actual = builder.sha(full)
        after = full.stat()
        assert (before.st_ino, before.st_size, before.st_mtime_ns) == (after.st_ino, after.st_size, after.st_mtime_ns), "Asset changed during read"
        assert actual == expected, "Returned/input SHA mismatch: " + str(path)
        if full.suffix == ".png":
            with Image.open(full) as image:
                assert image.mode == "RGB" and image.size == (1344, 1008) and image.format == "PNG"
        row = dict(sha256=expected, bytes=before.st_size, role=role, scene=scene)
        if frame is not None:
            row.update(camera="cam00", frame=frame)
        assert str(path) not in files
        files[str(path)] = row

    add(manifest, manifest_sha, "area4_manifest_evaluation_metadata")
    for name in METADATA:
        add(run / name, builder.sha(ROOT / run / name), "returned_native_HRref_metadata")
    assert {p.name for p in (ROOT / run / "renders_hr").glob("*.png")} == {f"{frame:04d}.png" for frame in range(300)}
    for row in rendering["records"]:
        add(run / "renders_hr" / row["path"], row["sha256"], "returned_native_HRref_8bit_prediction", row["frame"])
    test = sorted((row for row in data["observations"] if row["split"] == "test"), key=lambda row: row["frame_index"])
    assert [(row["camera_id"], row["frame_index"]) for row in test] == [("cam00", frame) for frame in range(300)]
    for row in test:
        add(manifest.parent / row["hr_path"], row["hr_sha256"], "heldout_HR_evaluation_only", row["frame_index"])
    inventory = dict(schema="CEM_area4_closed_input_inventory_v1", scene=scene, target="a100", run=str(run),
        scope="Returned CTS native_HRref_v2 300 cam00 PNGs and provenance metadata plus evaluation-only cam00 HR; no model or training inputs",
        train_test_boundary=dict(heldout_camera="cam00", heldout_HR="common evaluation only; never training or geometry", training_HR="not staged; already used only by independent privileged CTS HRref", checkpoint="identity in stage2/render/index metadata; bytes excluded"),
        source_manifest=dict(path=str(manifest), sha256=manifest_sha), model_identity=stage2["final_checkpoint"],
        source_contracts={str(HERE / name): builder.sha(ROOT / HERE / name) for name in ("build_wu_eval_job.py", "verify_assets.py")},
        files=dict(sorted(files.items())), file_count=len(files), total_bytes=sum(row["bytes"] for row in files.values()),
        role_counts=dict(Counter(row["role"] for row in files.values())), all_local_image_bytes_rehashed=True)
    # Reuse the immutable eval builder's exact image/metadata gate, including
    # finite300, timestamps, quantization, manifest and checkpoint cross-links.
    builder.hrref(SimpleNamespace(scene=scene, run=run), builder.Required(), inventory)
    assert len(files) == 608
    return inventory


PROBE = r'''import hashlib,json,sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
d=json.loads(sys.stdin.read());root=Path(d['root']).resolve()
assert str(root)=='/home/ubuntu/3DGS/4dsr'
def check(item):
 name,row=item;relative=Path(name)
 assert not relative.is_absolute() and '..' not in relative.parts
 path=root/relative
 assert path.resolve().is_relative_to(root)
 if not path.exists():return name,dict(status='missing')
 before=path.stat();h=hashlib.sha256()
 with path.open('rb') as stream:
  for block in iter(lambda:stream.read(8<<20),b''):h.update(block)
 after=path.stat();stable=(before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns)
 value=dict(bytes=after.st_size,sha256=h.hexdigest(),stable_during_read=stable)
 value['status']='verified' if stable and value['bytes']==row['bytes'] and value['sha256']==row['sha256'] else 'conflict'
 return name,value
with ThreadPoolExecutor(max_workers=2) as pool:files=dict(pool.map(check,d['files'].items()))
reserved=[name for name in d['must_be_absent'] if (root/name).exists()]
print(json.dumps(dict(status='probed',root=str(root),files=files,missing=[p for p,r in files.items() if r['status']=='missing'],conflicts=[p for p,r in files.items() if r['status']=='conflict'],reserved_existing=reserved)))
'''


def remote_probe(payload):
    result = subprocess.run(["ssh", "-o", "BatchMode=yes", "a100-train", "/usr/bin/python3 -c " + shlex.quote(PROBE)],
                            input=json.dumps(payload), capture_output=True, text=True, check=True, timeout=1800)
    return json.loads(result.stdout)


def stage_missing(missing):
    if missing:
        subprocess.run(["rsync", "-a", "--relative", "--ignore-existing", "--files-from=-", "./", "a100-train:" + REMOTE_ROOT + "/"],
                       cwd=ROOT, input="\n".join(missing) + "\n", text=True, check=True, timeout=7200)


def remote_verify(inventory_path, receipt_path, verifier_sha):
    # The frozen verifier emits the canonical full-SHA receipt. The exclusive
    # guard prevents its replace-based writer from overwriting an old receipt.
    program = """import hashlib,os,runpy,sys
from pathlib import Path
root=Path(ROOT_VALUE);source=root/SOURCE_VALUE;receipt=root/RECEIPT_VALUE
assert not receipt.exists(),'Refusing an existing verified/failed receipt'
guard=receipt.with_name(receipt.name+'.stage.lock')
fd=os.open(guard,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600);os.close(fd)
assert hashlib.sha256(source.read_bytes()).hexdigest()==SOURCE_SHA_VALUE
os.environ['CUDA_VISIBLE_DEVICES']=''
sys.argv=[str(source),'--root',str(root),'--inventory',str(root/INVENTORY_VALUE),'--receipt',str(receipt),'--workers','2']
runpy.run_path(str(source),run_name='__main__')
"""
    for key, value in {"ROOT_VALUE": REMOTE_ROOT, "SOURCE_VALUE": str(HERE / "verify_assets.py"), "RECEIPT_VALUE": str(receipt_path),
                       "SOURCE_SHA_VALUE": verifier_sha, "INVENTORY_VALUE": str(inventory_path)}.items():
        program = program.replace(key, repr(value))
    result = subprocess.run(["ssh", "-o", "BatchMode=yes", "a100-train", "/usr/bin/python3 -c " + shlex.quote(program)],
                            capture_output=True, text=True, timeout=3600)
    return result


def fetch_receipt(path):
    target = ROOT / path
    assert not target.exists(), "Never replace a local verification receipt"
    subprocess.run(["rsync", "-a", "--ignore-existing", "--", "a100-train:" + REMOTE_ROOT + "/" + str(path), str(target)], check=True, timeout=600)


def execute(args):
    assert args.scene in SCENES
    mode = "stage" if args.stage else "probe"
    named = paths(args.scene)
    destination = ROOT / named[mode]
    assert not destination.exists(), "Single attempt only; preserve existing probe/stage outcome"
    destination.parent.mkdir(parents=True, exist_ok=True)
    guard = destination.with_name(destination.name + ".lock")
    descriptor = os.open(guard, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(descriptor)
    began = time.monotonic()
    record = dict(schema="CEM_HRref_returned_images_staging_v1", status="running", mode=mode, scene=args.scene,
                  started_utc=utc(), GPU_calls=0, training_calls=0, retry_policy="none", remote_overwrites=0)
    try:
        builder, verifier = contracts()
        record["source_sha256"] = builder.sha(__file__)
        inventory = build_inventory(args.scene, args.run or BATCH / "runs" / (args.scene + "_native_HRref_v2"))
        inventory_path = ROOT / named["inventory"]
        if inventory_path.exists():
            assert json.loads(inventory_path.read_text()) == inventory, "Preserve a different frozen inventory"
        else:
            new_json(inventory_path, inventory)
        record.update(inventory_path=str(named["inventory"]), inventory_sha256=builder.sha(inventory_path), file_count=inventory["file_count"], total_bytes=inventory["total_bytes"])
        files = dict(inventory["files"])
        for path in (named["inventory"], HERE / "verify_assets.py"):
            files[str(path)] = dict(sha256=builder.sha(ROOT / path), bytes=(ROOT / path).stat().st_size)
        payload = dict(root=REMOTE_ROOT, files=files, must_be_absent=[str(named["receipt"]), str(named["receipt"]) + ".stage.lock"])
        before = remote_probe(payload)
        record["remote_before"] = before
        assert not before["conflicts"] and not before["reserved_existing"], "Remote conflict/existing receipt; no replacement or retry"
        if mode == "probe":
            record["status"] = "probed_missing_only_plan_no_transfer"
        else:
            assert not (ROOT / named["receipt"]).exists(), "Preserve existing local receipt"
            local = verifier.verify(ROOT, inventory_path, 2)
            new_json(ROOT / named["local_verification"], local)
            assert local["status"] == "verified", "Local bytes changed before transfer"
            stage_missing(before["missing"])
            result = remote_verify(named["inventory"], named["receipt"], files[str(HERE / "verify_assets.py")]["sha256"])
            record["remote_verifier"] = dict(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
            fetch_receipt(named["receipt"])
            assert result.returncode == 0, "Full A100 byte verification failed; preserve returned failure receipt"
            builder.asset_gate(named["receipt"])
            record.update(status="returned_HRref_images_A100_full_SHA_verified", receipt_path=str(named["receipt"]), receipt_sha256=builder.sha(ROOT / named["receipt"]),
                          missing_files_transferred=len(before["missing"]), canonical_receipt_status="verified")
    except BaseException as error:
        record.update(status="failed", error=str(error), traceback=traceback.format_exc())
    finally:
        record.update(finished_utc=utc(), elapsed_seconds=time.monotonic() - began)
        new_json(destination, record)
    print(json.dumps({key: record.get(key) for key in ("status", "scene", "mode", "file_count", "receipt_path", "error")}))
    return 1 if record["status"] == "failed" else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=SCENES, required=True)
    parser.add_argument("--run", type=Path, help="Defaults to this scene's returned native_HRref_v2")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--probe", action="store_true")
    mode.add_argument("--stage", action="store_true")
    return execute(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
