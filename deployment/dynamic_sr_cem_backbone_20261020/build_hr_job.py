"""Freeze a native HR reference with the observed short-save limitation.

The failed five-step startup fixture remains preserved. Native full-length
training is unchanged and begins in a separate, empty model directory.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from build_native_job_v3 import BATCH, HERE, ROOT, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene", choices=["cook_spinach", "cut_roasted_beef"], required=True)
    p.add_argument("--asset-receipt", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    assert not a.output.exists()
    evidence = BATCH / "runs/cook_spinach_native_HRref_v1/startup_fixture/fixture_result.json"
    failed = json.loads((ROOT / evidence).read_text())
    assert failed["status"] == "failed" and failed["native_iterations"] == 5
    log = ROOT / BATCH / "runs/cook_spinach_native_HRref_v1/native_HR_startup_fixture.log"
    assert "self._xyz_motion.detach().flatten(start_dim=1)" in log.read_text()
    out = BATCH / "runs" / (a.scene + "_native_HRref_v2")
    assert not (ROOT / out).exists(), "Never reuse a scientific branch, including failures"
    with tempfile.TemporaryDirectory(prefix="cem_hr_job_metadata_") as directory:
        draft = Path(directory) / "draft.json"
        subprocess.run([sys.executable, str(ROOT / HERE / "build_native_job_v3.py"), "--scene", a.scene,
            "--level", "hr", "--host", "cts", "--gpu", "0", "--initialization",
            str(ROOT / BATCH / "initialization" / (a.scene + "_r2")),
            "--asset-receipt", str(a.asset_receipt), "--output", str(draft)], check=True)
        job = json.loads(draft.read_text())
    old_out = job["out"]
    job.update(out=str(out), job_id=f"codex-cem-area4-{a.scene.split('_')[0]}-native-hrref-ctsg0-20261020-v5")
    job["commands"] = [x for x in job["commands"] if x["name"] != "native_HR_startup_fixture"]
    for command in job["commands"]:
        command["argv"] = [str(x).replace(old_out, str(out)) for x in command["argv"]]
    job["return_paths"] = [str(out)]
    job["required_files"] += [dict(path=str(Path(__file__).resolve().relative_to(ROOT)), sha256=sha(__file__))]
    job["startup_fixture_policy"] = dict(failure_evidence_sha256=sha(ROOT / evidence),
        failure_log_sha256=sha(log), reason="native iteration5 PLY save requires motion arrays formed later in original training",
        artificial_early_save_fixture_removed=True, native_source_modified=False,
        native_original_schedule_preserved=True, scientific_run_reused=False,
        startup_observation="Cook real HR forward/backward reached iteration5 before serialization failure")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(job, indent=2) + "\n")
    print(json.dumps(dict(job_id=job["job_id"], output=str(a.output), sha256=sha(a.output))))


if __name__ == "__main__":
    main()
