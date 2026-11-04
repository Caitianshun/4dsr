"""New CEM paths/scene alias; original stage-b, author runner and renderer."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_bl import *
from cem_frame0_new import accepted, paths

RUNTIME = ROOT / "third_party/cem4dgs_calibrated_runtime_v3_20261020"


def recipe(scene):
    return read(CODE / f"configs/{scene}/cem_recipe.json")


def training_argv(scene, stage, gpu):
    """Exact historical stage-b/run_author argv, with new paths and author endpoints."""
    r = recipe(scene)
    output = OUT / "cem" / scene
    argv = [str(OLD / "localdygs_cem/run_author.py"), "--", "--runtime", str(RUNTIME),
            "--receipt", str(output / f"training_stage{stage}_receipt.json"), "--",
            "--configpath", str(CODE / f"configs/{scene}/cem_s{stage}.json"),
            "--source_path", str(paths(scene)[1] / "cem_data"),
            "--model_path", str(output / "model"), "--gpu", gpu]
    if stage == 2:
        argv += ["--start_checkpoint", str(output / "model" / f"chkpnt{r['stage_endpoints'][0]}.pth")]
    return argv


def render_argv(scene, scope, execute):
    output = OUT / "cem" / scene
    return ["--method", "cem", "--manifest", str(manifest_path(scene)),
            "--runtime", str(RUNTIME), "--model", str(output / "model"),
            "--source", str(paths(scene)[1] / "cem_data"), "--scope", scope,
            "--output", str(output / ("renders_" + scope)),
            "--execute" if execute else "--plan-only"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("plan", "train", "render"))
    p.add_argument("--scene", required=True, choices=SCENES)
    p.add_argument("--stage", type=int, choices=(1, 2))
    p.add_argument("--scope", choices=("test", "train"))
    p.add_argument("--execute", action="store_true")
    a = p.parse_args()
    validate_metadata(a.scene)
    if a.mode == "plan":
        print(__import__("json").dumps(dict(
            scene=a.scene, recipe=recipe(a.scene),
            stage1_argv=training_argv(a.scene, 1, "BROKER_PHYSICAL_MASK"),
            stage2_argv=training_argv(a.scene, 2, "BROKER_PHYSICAL_MASK"),
            render={s: render_argv(a.scene, s, True) for s in ("test", "train")}), indent=2))
        return
    require(a.execute, "Owner --execute required")
    accepted(a.scene)
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    require(gpu.isdecimal(), "Single broker physical numeric CUDA mask required by unchanged CEM --gpu")
    os.environ.update(DYNAMIC_LR_PROTOCOL_MANIFEST=str(manifest_path(a.scene)),
                      TMPDIR=str(OUT / "runtime_tmp"), TMP=str(OUT / "runtime_tmp"), TEMP=str(OUT / "runtime_tmp"),
                      OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
    (OUT / "runtime_tmp").mkdir(parents=True, exist_ok=True)
    os.environ.pop("DYNAMIC_LR_RENDER_HR", None)
    overlay = ROOT / "envs/dynamic_lr_baselines_20261020/cem/cuda-toolkit-overlay/lib64"
    os.environ["LD_LIBRARY_PATH"] = ":".join(filter(None, (str(overlay), os.environ.get("LD_LIBRARY_PATH", ""))))
    driver = load("_original_cem_area4_driver", OLD / "localdygs_cem/run_area4.py")
    if a.mode == "train":
        require(a.stage is not None, "--stage required")
        if a.stage == 2:
            previous = read(OUT / "cem" / a.scene / "training_stage1_receipt.json")
            require(previous["status"] == "completed", "Stage1 must complete")
        driver.stage_b(training_argv(a.scene, a.stage, gpu))
    else:
        require(a.scope is not None, "--scope required")
        require(read(OUT / "cem" / a.scene / "training_stage2_receipt.json")["status"] == "completed",
                "Stage2 must complete")
        # Same original renderer, only its recipe lookup resolves the author
        # flame_salmon_1 alias and the new per-scene config metadata.
        sys.path.insert(0, str(OLD / "localdygs_cem"))
        module = load("_original_cem_area4_renderer", OLD / "localdygs_cem/render_local_cem_area4.py")
        module.config_recipe = lambda method, scene: recipe(scene) if method == "cem" and scene in SCENES else driver.recipe(method, scene)
        load("protocol_interface", OLD / "localdygs_cem/protocol_interface_area4.py")
        sys.argv = [str(OLD / "localdygs_cem/render_local_cem_area4.py"), *render_argv(a.scene, a.scope, True)]
        module.main()


if __name__ == "__main__":
    main()
