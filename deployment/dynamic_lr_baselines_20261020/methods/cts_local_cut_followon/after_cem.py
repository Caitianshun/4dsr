"""Wait for the existing CEM Cook collector, then submit LocalDyGS Cut once."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import select
import subprocess
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parents[4]
REMOTE_ROOT = Path("/home/cts/Project/4DSR")
BASE = Path("output/dynamic_lr_baselines_20261020")
CEM_ID = "codex-dynlr-cem-cts0-cook-spinach-train-20261020"


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            result.update(block)
    return result.hexdigest()


def properties(unit):
    result = subprocess.check_output(["systemctl", "--user", "show", unit,
        "--property=MainPID,ActiveState,ExecMainStatus,Result"], text=True)
    return dict(line.split("=", 1) for line in result.splitlines())


def verify_returned_result(index_path):
    """Read and hash returned evidence; never import a model or access CUDA."""
    index_path = Path(index_path)
    value = json.loads(index_path.read_text())
    assert value["status"] == "completed" and value["original_LR"] and value["no_proposed_modules"]
    def mapped(path):
        return ROOT / Path(path).relative_to(REMOTE_ROOT)
    def verify(row):
        path = mapped(row["path"])
        assert digest(path) == row["sha256"], str(path)
        return path
    assets = [verify(row) for row in value["formal_checkpoint"]]
    render_receipt = verify(value["rendered_pngs"]["receipt"])
    rendered = json.loads(render_receipt.read_text())
    assert rendered["status"] == "completed" and rendered["frames"] == 300
    assert rendered["finite_before_clip_quantization_passed_frames"] == 300
    assert rendered["raw_prediction_shape"] == [3, 1008, 1344]
    assert [row["frame"] for row in rendered["records"]] == list(range(300))
    renders = mapped(value["rendered_pngs"]["directory"])
    assert {path.name for path in renders.glob("*.png")} == {f"{frame:04d}.png" for frame in range(300)}
    for row in rendered["records"]:
        assert row["path"] == f"{row['frame']:04d}.png"
        assert digest(renders / row["path"]) == row["sha256"]
    metadata = json.loads(verify(value["source_metadata"]).read_text())
    for row in metadata["files"]:
        verify(row)
    for key in ["manifest", "source_and_patch", "native_initialization", "native_calibration_acceptance", "source_config", "GPU_environment_acceptance"]:
        verify(value[key])
    for row in value["training_receipts"]:
        assert json.loads(verify(row).read_text())["status"] == "completed"
    metrics = value["image_metrics"]
    assert metrics["scope"] == "provisional_CTS" and not metrics["main_table_eligible"]
    verify(metrics["summary"])
    verify(metrics["per_frame"])
    return dict(status="all_returned_hashes_passed", index=str(index_path), index_sha256=digest(index_path),
        checkpoint_assets=len(assets), finite_HR_PNGs=300, source_metadata_files=len(metadata["files"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cem-unit", required=True)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--job-sha256", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    if args.receipt.exists():
        raise SystemExit("Refusing an existing finite follow-on identity; no replay")
    state = dict(status="waiting_for_existing_CEM_Cook_collector_exit", cem_unit=args.cem_unit,
        job=str(args.job), job_sha256=args.job_sha256, automatic_failure_replay=False, began_unix=time.time(),
        source=str(Path(__file__).resolve()), source_sha256=digest(__file__))
    def save():
        args.receipt.write_text(json.dumps(state, indent=2) + "\n")
    save()
    try:
        assert digest(args.job) == args.job_sha256
        before = properties(args.cem_unit)
        state["CEM_service_before_wait"] = before
        save()
        if int(before["MainPID"]):
            try:
                fd = os.pidfd_open(int(before["MainPID"]))
            except ProcessLookupError:
                fd = None
            if fd is not None:
                try:
                    select.select([fd], [], [])
                finally:
                    os.close(fd)
        ended = properties(args.cem_unit)
        assert ended["MainPID"] == "0" and ended["ExecMainStatus"] == "0" and ended["Result"] == "success", ended
        previous = json.loads((ROOT / BASE / "localdygs_cem/cook_cem_after_local_receipt.json").read_text())
        assert previous["status"] == "completed_and_artifacts_returned"
        cem_index = ROOT / BASE / "cts_native_localdygs_cem/cook_spinach/cem/result_index.json"
        current = json.loads(cem_index.read_text())
        assert current["method"] == "cem" and current["scene"] == "cook_spinach" and current["final_iteration"] == 50000
        state["CEM_return_integrity"] = verify_returned_result(cem_index)
        boundary = cem_index.parent / "stage_boundary_acceptance.json"
        proof = json.loads(boundary.read_text())
        assert proof["status"] == "passed" and proof["iteration"] == 40000
        assert not proof["CUDA_initialized"] and not proof["model_or_optimizer_updated"] and proof["checkpoint_unchanged"]
        assert digest(ROOT / Path(proof["checkpoint"]).relative_to(REMOTE_ROOT)) == proof["checkpoint_sha256"]
        state["CEM_native_stage_boundary"] = dict(path=str(boundary), sha256=digest(boundary))
        remote_check = """import fcntl,json,os,subprocess
from pathlib import Path
root=Path('/home/cts/Project/4DSR')
terminal=json.loads((root/'.claude-training/jobs/codex-dynlr-cem-cts0-cook-spinach-train-20261020/status.json').read_text())
assert terminal.get('state')=='ended' and terminal.get('outcome')=='success'
fd=os.open(root/'deployment/cts_20260921/gpu0.lock',os.O_RDONLY|os.O_NOFOLLOW)
try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
finally:os.close(fd)
compute=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True).strip()
assert not compute,compute
print(json.dumps({'status':'passed','existing_CEM_broker_outcome':'success','GPU0_lock_released':True,'CUDA_compute_processes':compute}))
"""
        gate = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", "cts", "/usr/bin/python3 -"],
            input=remote_check, text=True, capture_output=True, timeout=60, check=True)
        state.update(status="CEM_completed_returned_and_GPU_released", pre_Local_Cut_gate=json.loads(gate.stdout))
        save()
        state["status"] = "Local_Cut_broker_execution_and_exit_artifact_return"
        save()
        subprocess.run([sys.executable, str(ROOT / "deployment/dynamic_lr_baselines_20261020/submit_remote.py"),
            "--host", "cts", "--gpu-index", "0", "--job", str(args.job), "--memory-mib", "21000",
            "--timeout-seconds", "172800"], cwd=ROOT, check=True)
        index = ROOT / BASE / "cts_native_localdygs_cem/cut_roasted_beef/localdygs/result_index.json"
        state["Local_Cut_return_integrity"] = verify_returned_result(index)
        result = json.loads(index.read_text())
        assert result["method"] == "localdygs" and result["scene"] == "cut_roasted_beef" and result["final_iteration"] == 30000
        state.update(status="completed_and_artifacts_returned", result_index=str(index), result_index_sha256=digest(index))
    except BaseException:
        state.update(status="failed_no_replay", error=traceback.format_exc())
        raise
    finally:
        state["elapsed_seconds"] = time.time() - state["began_unix"]
        save()


if __name__ == "__main__":
    main()
