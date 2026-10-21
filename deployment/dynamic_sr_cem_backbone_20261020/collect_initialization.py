"""Wait for a CPU COLMAP terminal event, collect once, and record any failure."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import traceback

ROOT = Path(__file__).resolve().parents[2]
REMOTE = "/home/ubuntu/3DGS/4dsr"
BATCH = "output/dynamic_sr_cem_backbone_20261020"
BASE = ROOT / "deployment/dynamic_lr_baselines_20261020/methods/localdygs_cem"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def wait_script(status_path):
    return '''import ctypes,json,os,select,time
from pathlib import Path
p=Path(STATUS_PATH)
deadline=time.monotonic()+3600
libc=ctypes.CDLL('libc.so.6',use_errno=True)
fd=libc.inotify_init1(os.O_CLOEXEC)
if fd<0:raise OSError(ctypes.get_errno())
watched=None
try:
 while True:
  folder=p.parent
  while not folder.exists():folder=folder.parent
  if folder!=watched:
   if watched is not None:libc.inotify_rm_watch(fd,wd)
   wd=libc.inotify_add_watch(fd,os.fsencode(folder),0x100|0x80|0x8)
   if wd<0:raise OSError(ctypes.get_errno())
   watched=folder
  if p.exists():
   value=json.loads(p.read_text())
   if value['status'] in ['completed','failed']:
    print(json.dumps(value));break
  remaining=deadline-time.monotonic()
  if remaining<=0:raise TimeoutError('CPU initializer exit event missing; no automatic replay')
  if select.select([fd],[],[],remaining)[0]:os.read(fd,65536)
finally:os.close(fd)
'''.replace("STATUS_PATH", repr(status_path))


def verify(output, scene, expected):
    import numpy as np
    state = json.loads((output / "complete.json").read_text())
    manifest_path = ROOT / "data/dynamic_sr/full_time_20261007/prepared" / (scene + "_area4") / "manifest_train_ready_seed20261007.json"
    value = json.loads(manifest_path.read_text())
    inputs = ROOT / BATCH / "initialization_inputs" / scene
    source = json.loads((inputs / "input_receipt.json").read_text())
    assert state["manifest_sha256"] == sha(manifest_path) == source["manifest_sha256"]
    assert state["input_receipt_sha256"] == sha(inputs / "input_receipt.json")
    assert state["source_script_sha256"] == expected["initializer"]
    assert state["helper_source_sha256"] == expected["helpers"]
    assert state["native_command_source_sha256"] == expected["helpers"]["run_native_colmap"]
    assert state["input_preparer_source_sha256"] == source["source_script_sha256"]
    assert state["COLMAP_poses_intrinsics_unchanged"] and state["native_frames"] == [0]
    assert state["CUDA_VISIBLE_DEVICES"] == "" and state["CUDA_calls"] == 0
    assert not state["used_HR"] and not state["used_test_pixels"] and not state["used_SR_teacher"]
    assert [x["name"] for x in state["commands"]] == ["features", "matching", "triangulation", "model_txt", "sparse_ply"]
    assert all(x["returncode"] == 0 for x in state["commands"])
    pointcloud = output / "cem_data/colmap_0/sparse/0/points3D.ply"
    assert sha(pointcloud) == state["cem_initialization"]["sha256"]
    spec = importlib.util.spec_from_file_location("calibration_audit_cpu", BASE / "audit_native_initialization.py")
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    sparse = pointcloud.parent
    cameras = {int(r[0]): r for r in helper.rows(sparse / "cameras.txt")}
    images = [r for r in helper.rows(sparse / "images.txt") if len(r) == 10 and r[-1].endswith(".png")]
    assert sorted(Path(r[-1]).stem for r in images) == sorted(value["splits"]["train"])
    errors = []
    for row in images:
        camera = Path(row[-1]).stem
        k = np.asarray(value["cameras"][camera]["K_lr"])
        pose = np.asarray(value["cameras"][camera]["w2c"])
        actual = cameras[int(row[8])]
        assert actual[1:4] == ["PINHOLE", "336", "252"]
        ek = float(np.abs(np.asarray(actual[4:], float) - [k[0, 0], k[1, 1], k[0, 2]+.5, k[1, 2]+.5]).max())
        ep = max(float(np.abs(helper.rotation(np.asarray(row[1:5], float)) - pose[:3, :3]).max()), float(np.abs(np.asarray(row[5:8], float)-pose[:3, 3]).max()))
        assert ek < 1e-9 and ep < 1e-9
        errors.append(dict(camera=camera, K_error=ek, pose_error=ep))
    assert len(helper.rows(sparse / "points3D.txt")) == state["cem_initialization"]["points"] > 0
    acceptance = output / "camera_initialization_acceptance.json"
    assert sha(acceptance) == state["camera_acceptance_sha256"]
    return dict(point_count=state["cem_initialization"]["points"],
                pointcloud_path=str(pointcloud), pointcloud_sha256=sha(pointcloud),
                complete_sha256=sha(output / "complete.json"),
                native_receipt_sha256=sha(output / "cem_data/native_initialization_receipt.json"),
                camera_acceptance_sha256=sha(acceptance), camera_records=errors,
                max_K_error=max(x["K_error"] for x in errors), max_pose_error=max(x["pose_error"] for x in errors),
                initializer_elapsed_seconds=state["elapsed_seconds"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", choices=["cook_spinach", "cut_roasted_beef"], required=True)
    parser.add_argument("--attempt", default="", help="Nonempty attempts get independent scene_ATTEMPT paths")
    args = parser.parse_args()
    assert not args.attempt or re.fullmatch(r"r[1-9][0-9]*", args.attempt), "Invalid attempt"
    label = args.scene + ("_" + args.attempt if args.attempt else "")
    control = ROOT / BATCH / "initialization_control" / label
    control.mkdir(parents=True, exist_ok=True)
    receipt = control / "collection.json"
    assert not receipt.exists(), "No duplicate collection or initializer replay"
    with (control / "collector_claim.json").open("x") as stream:
        stream.write(json.dumps(dict(scene=args.scene, attempt=args.attempt, UTC=datetime.now(timezone.utc).isoformat())))
    result_record = dict(status="failed_collection", scene=args.scene, attempt=args.attempt,
                         started_UTC=datetime.now(timezone.utc).isoformat(), CUDA_calls=0,
                         used_HR=False, used_test_pixels=False, used_SR_teacher=False,
                         collector_source_sha256=sha(__file__), completion_transport="inotify terminal-state event; no replay")
    relative = BATCH + "/initialization/" + label
    output = ROOT / relative
    try:
        expected = dict(initializer=sha(ROOT / "deployment/dynamic_sr_cem_backbone_20261020/initialize_cem.py"),
                        helpers={n: sha(BASE / (n + ".py")) for n in ("prepare_native_colmap", "audit_native_initialization", "run_native_colmap")})
        result_record["expected_sources"] = expected
        result = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ServerAliveInterval=30",
                                 "-o", "ServerAliveCountMax=6", "a100-train", "/usr/bin/python3 -"],
                                input=wait_script(REMOTE + "/" + relative + "/status.json"),
                                text=True, capture_output=True, timeout=3670)
        (control / "terminal_transport.log").write_text(result.stdout + result.stderr)
        result_record["event_transport_returncode"] = result.returncode
        assert result.returncode == 0, "Initializer event transport failed; no automatic replay"
        terminal = json.loads(result.stdout.strip().splitlines()[-1])
        result_record["remote_terminal_status"] = terminal["status"]
        for rel in [relative, BATCH + "/initialization_control/" + label]:
            target = ROOT / rel
            target.mkdir(parents=True, exist_ok=True)
            subprocess.run(["rsync", "-a", "--partial", "a100-train:" + REMOTE + "/" + rel + "/",
                            str(target) + "/"], check=True, timeout=900)
        assert terminal["status"] == "completed", "Failed native initialization returned and preserved locally"
        result_record.update(verify(output, args.scene, expected))
        result_record["status"] = "completed_returned_verified"
    except BaseException:
        result_record["error"] = traceback.format_exc()
        raise
    finally:
        result_record["completed_UTC"] = datetime.now(timezone.utc).isoformat()
        result_record["files"] = [dict(path=str(f.relative_to(ROOT)), sha256=sha(f), bytes=f.stat().st_size)
                                  for f in sorted(output.rglob("*")) if f.is_file()]
        write(receipt, result_record)
        print(json.dumps(dict(scene=args.scene, attempt=args.attempt, status=result_record["status"], receipt=str(receipt))))


if __name__ == "__main__":
    main()
