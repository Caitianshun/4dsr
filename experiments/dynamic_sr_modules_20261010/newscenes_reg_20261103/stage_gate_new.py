"""LR-only LOCO gate for conditional SMX1 and the actual final model.

This entry never dispatches a job. When Claude executes a task, it optionally
runs its original stage in this same process. A skipped SMX1 stage gets its own
completion receipt so pipeline.py can continue; no missing residual is invented.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import runpy
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_new import *


def choose(path):
    value = read(path)
    require(value["kind"] == "loco_eval" and value["camera"] == "cam10" and value["frames"] == [0, 299, 300],
            "Full cam10 LR-only LOCO readout required")
    require(set(value["resid"]) >= {"v2", "SMX1"}, "Both LOCO variants required")
    for label in ("v2", "SMX1"):
        fit = read(local(value["resid"][label]) / "run.json")
        require("cam10" in fit.get("excluded_cameras", []) and "cam00" not in fit["fitting"]
                and "cam10" not in fit["fitting"]
                and fit["checkpoint_sha256"] == value["checkpoint"]["sha256"],
                "LOCO fit did not exclude cam10 or belongs to another checkpoint")
        require(fit["args"]["target"] == "lr" and fit["args"]["arm"] == "PC"
                and fit["args"].get("c_reg") is True and fit.get("cam00_read") is False,
                "LOCO calibration must use only training LR with the historical PCv2 fit")
    summary = value["summary"]
    dt = summary["SMX1"]["tdiff_psnr"] - summary["v2"]["tdiff_psnr"]
    dp = summary["SMX1"]["psnr"] - summary["v2"]["psnr"]
    require(math.isfinite(dt) and math.isfinite(dp), "Nonfinite LOCO deltas")
    selected = "SMX1" if dt >= 0.05 and dp >= -0.01 else "v2"
    decision = value["decision"]
    require(decision["choose"] == selected and abs(decision["delta_tdiff"] - dt) < 1e-10
            and abs(decision["delta_psnr"] - dp) < 1e-10, "LOCO decision disagrees with frozen rule")
    return selected


def substitute_residual(argv, residual):
    # eval_resid uses LABEL=PATH, while renderers use a separate PATH token.
    return [item.replace("@RESID@", rel(residual)) for item in argv]


def main():
    argv = sys.argv[1:]
    split = argv.index("--") if "--" in argv else len(argv)
    own, downstream = argv[:split], argv[split + 1:]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--decision", required=True)
    p.add_argument("--v2", required=True)
    p.add_argument("--smx1", required=True)
    p.add_argument("--out", required=True, help="fresh wrapper completion/model receipt")
    p.add_argument("--if-smx1", action="store_true")
    p.add_argument("--checkpoint", help="final model inventory, no downstream stage")
    a = p.parse_args(own)
    selected = choose(a.decision)
    residual = local(a.smx1 if selected == "SMX1" else a.v2)
    out = output_path(a.out)
    require(not out.exists(), "Completed wrapper receipt preserved")
    if a.if_smx1 and selected != "SMX1":
        write_new(out, dict(status="skipped_by_frozen_LR_LOCO_rule", choose=selected,
                            decision=entry(a.decision), stage_executed=False, GPU_calls=0))
        return
    if a.checkpoint:
        pvcd = read(local(a.checkpoint).parent / "pvcd.json")
        require((pvcd["K"], pvcd["weight"], pvcd["dssim"], pvcd["scale"], pvcd["seed"],
                 pvcd["calls"], pvcd["sr_steps"], pvcd["calls_equal_steps"], pvcd["complete"]) ==
                (3, 0.5, 0.2, 4, 20261014, 6000, 6000, True, True),
                "Actual final checkpoint lacks complete historical PVCD module")
        warp = read(local(a.checkpoint).parent / "modkit2.json")
        require(warp["switches"]["wf_grid"] == 4 and bool(warp["warp_field_rms_px"]),
                "Actual final checkpoint lacks the trained w4 camera module")
        complete = read(local(a.checkpoint).parent / "complete.json")
        require(complete["status"] == "completed" and complete["seed"] == SEED
                and complete["checkpoint"] == entry(a.checkpoint)
                and complete["lambda_SR"] == 0.2 and complete["dssim"] == 0.2,
                "Student completion/seed/endpoint/recipe differs")
        parents = [row["parent_checkpoint"] for row in
                   [complete, *(read(local(t["checkpoint"]["path"]).parent / "run.json")
                                for t in pvcd["teachers"])]]
        require(len(pvcd["teachers"]) == 3 and all(row == parents[0] for row in parents),
                "All three sources and the student must start from the same LR parent")
        meta = read(residual / "run.json")
        checkpoint = entry(a.checkpoint)
        require(meta.get("checkpoint_sha256") == checkpoint["sha256"],
                "Selected residual does not belong to the final checkpoint")
        require(read(residual / "complete.json")["frames"] == 300, "Full300 residual required")
        require(meta["args"]["target"] == "lr" and meta["args"]["arm"] == "PC"
                and meta["args"].get("c_reg") is True, "Final P+C must remain LR-only PCv2")
        # The renderers independently verify the post-cull xyz/point identity.
        write_new(out, dict(
            status="selected_actual_final_model_requires_uniform_scores",
            checkpoint=checkpoint, residuals=rel(residual), residual_run=entry(residual / "run.json"),
            decision=entry(a.decision), choose=selected,
            modules=["gateaff_LR_parent", "areaft_teacher", "w4_camera_model",
                     "PVCD_K3_scale4", "P+C_v2"] + (["SMX1"] if selected == "SMX1" else []),
            calibration_inputs="cam10 held-out training LR only", HR_used_for_selection=False))
        return
    require(bool(downstream), "Original downstream script and argv required")
    script = local(downstream[0])
    require(script.resolve().is_relative_to((ROOT / "experiments").resolve()), "Stage script outside project")
    args = substitute_residual(downstream[1:], residual)
    sys.path.insert(0, str(script.parent))
    previous = sys.argv
    try:
        sys.argv = [str(script), *args]
        runpy.run_path(str(script), run_name="__main__")
    except SystemExit as error:
        if error.code not in (None, 0):
            raise
    finally:
        sys.argv = previous
    write_new(out, dict(status="completed_selected_original_stage", choose=selected,
                        decision=entry(a.decision), original_script=entry(script),
                        effective_argv=args, stage_executed=True))


if __name__ == "__main__":
    main()
