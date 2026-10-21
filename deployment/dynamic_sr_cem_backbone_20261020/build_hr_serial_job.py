"""Freeze a fresh native HR run after the proven CTS host-RAM OOM.

Use the author's existing n_job=1 image-reader option through an owned entry
wrapper. Native source, pixels, shuffling, losses, optimizer and schedule stay
unchanged. This builder only freezes metadata; it never submits a GPU task.
"""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile

from build_native_job_v3 import BATCH, HERE, ROOT, sha


ADAPTER = Path("experiments/dynamic_sr_cem_area_20261020/native_stage_serial_io.py")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene", choices=["cook_spinach", "cut_roasted_beef"], required=True)
    p.add_argument("--asset-receipt", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    assert not a.output.exists(), "Preserve all frozen requests"
    evidence = BATCH / "admission/HRref_SIGKILL_host_evidence_v1.json"
    host = json.loads((ROOT / evidence).read_text())
    kernel = host["cts"]["kernel_window"]["stdout"]
    assert "global_oom" in kernel and "Killed process 445330 (python)" in kernel
    assert "anon-rss:63185968kB" in kernel
    failed_run = BATCH / "runs/cook_spinach_native_HRref_v2"
    failed = json.loads((ROOT / failed_run / "status.json").read_text())
    assert failed["status"] == "failed" and failed["stages"][-1]["name"] == "native_stage1"
    assert failed["stages"][-1]["returncode"] == -9
    assert not (ROOT / failed_run / "stage1.json").exists()
    assert not (ROOT / failed_run / "model/chkpnt30000.pth").exists()
    assert not (ROOT / failed_run / "model/chkpnt40000.pth").exists()
    out = BATCH / "runs" / (a.scene + "_native_HRref_v3")
    assert not (ROOT / out).exists(), "Never reuse a failed or scientific output"
    assert (ROOT / ADAPTER).is_file()
    qa_path = BATCH / "admission/serial_io_CPU_QA_CTS_v1.json"
    qa = json.loads((ROOT / qa_path).read_text())
    assert qa["status"] == "passed" and qa["CUDA_initialized"] is False
    assert qa["joblib_version"] == "1.6.0"
    assert qa["wrapper"]["sha256"] == sha(ROOT / ADAPTER)
    assert qa["same_seed_camera_order_equal"] is True
    assert qa["main_process_Python_NumPy_Torch_CPU_RNG_unchanged_by_decoding"] is True
    assert qa["generator_before_next"] == dict(task_iterator_advanced=0, decoded=0)
    assert qa["generator_after_one_next_and_wait"] == dict(task_iterator_advanced=1, decoded=1)
    assert len(qa["image_evidence"]) == 6 and all(x["bitwise_equal"] for x in qa["image_evidence"])
    assert qa["native_model_imported"] is False and qa["native_training_executed"] is False
    with tempfile.TemporaryDirectory(prefix="cem_hr_serial_metadata_") as directory:
        draft = Path(directory) / "draft.json"
        subprocess.run([sys.executable, str(ROOT / HERE / "build_native_job_v3.py"),
            "--scene", a.scene, "--level", "hr", "--host", "cts", "--gpu", "0",
            "--initialization", str(ROOT / BATCH / "initialization" / (a.scene + "_r2")),
            "--asset-receipt", str(a.asset_receipt), "--output", str(draft)], check=True)
        job = json.loads(draft.read_text())
    old_out = job["out"]
    job.update(out=str(out), job_id=f"codex-cem-area4-{a.scene.split('_')[0]}-hrref-serialio-ctsg0-20261020-v6")
    assert re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", job["job_id"])
    job["commands"] = [x for x in job["commands"] if x["name"] != "native_HR_startup_fixture"]
    for command in job["commands"]:
        command["argv"] = [str(x).replace(old_out, str(out)) for x in command["argv"]]
        if command["name"] in ("native_stage1", "native_stage2"):
            assert command["argv"][1] == "{ROOT}/experiments/dynamic_sr_cem_area_20261020/native_stage.py"
            command["argv"][1] = "{ROOT}/" + str(ADAPTER)
    job["return_paths"] = [str(out)]
    manifest = Path("data/dynamic_sr/full_time_20261007/prepared") / (a.scene + "_area4") / "manifest_train_ready_seed20261007.json"
    for path in [ADAPTER, Path(__file__).resolve().relative_to(ROOT), manifest]:
        assert str(path) not in {r["path"] for r in job["required_files"]}
        job["required_files"].append(dict(path=str(path), sha256=sha(ROOT / path)))
    job["host_memory_recovery"] = dict(
        kernel_evidence=dict(path=str(evidence), sha256=sha(ROOT / evidence)),
        failed_native_execution=dict(path=str(failed_run / "status.json"), sha256=sha(ROOT / failed_run / "status.json")),
        io_adapter=dict(path=str(ADAPTER), sha256=sha(ROOT / ADAPTER)),
        actual_CTS_CPU_QA=dict(path=str(qa_path), sha256=sha(ROOT / qa_path)),
        policy="author getTrainCameras n_job=1; synchronous demand-driven joblib generator",
        native_core_source_modified=False, pixels_and_sampling_order_unchanged=True,
        optimizer_and_schedule_unchanged=True, resume_available=False,
        output_reused=False, automatic_replay=False,
        reason="native asynchronous HR image producers exhausted 64GB host RAM before first30k checkpoint")
    job["comparison_limit"] += "; CPU image loading uses the existing author n_job=1 option after proven host OOM; model mathematics unchanged"
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("x") as stream:
        json.dump(job, stream, indent=2)
        stream.write("\n")
    print(json.dumps(dict(job_id=job["job_id"], output=str(a.output), sha256=sha(a.output), GPU_calls=0)))


if __name__ == "__main__":
    main()
