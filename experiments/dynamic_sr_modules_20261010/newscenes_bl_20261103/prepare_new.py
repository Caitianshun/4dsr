"""Create new-scene configurations from author scene recipes and registered data."""
from __future__ import annotations

import copy
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_bl import *


def build():
    audit = []
    historical_sr4d = read(ROOT / "output/dynamic_sr_modules_20261010/sr4d_area4_20261028/runs/Flame07/sr4d_b/config.json")
    for scene, author in SCENES.items():
        mp, m, rp, registry, geo = validate_metadata(scene)
        destination = CODE / "configs" / scene
        write_new(destination / "geometry.json", dict(scene=scene, manifest=identity(mp), **geo))
        author_sources = []
        endpoints = []
        for stage in (1, 2):
            source = ROOT / f"third_party/cem4dgs_20261020/configs/N3V_s{stage}/{author}.json"
            config = read(source)
            require(config["resolution"] == 2, "Re-audit author resolution change")
            config["resolution"] = 1
            target = destination / f"cem_s{stage}.json"
            write_new(target, config)
            delta = differences(read(source), config)
            require(delta == [{"field": "/resolution", "old": 2, "new": 1}], "CEM scientific config changed")
            author_sources.append(identity(source))
            endpoints.append(config["iterations"])
            audit.append(dict(scene=scene, method="cem", stage=stage, source=identity(source),
                              target=identity(target), differences=delta))
        write_new(destination / "cem_recipe.json", dict(
            schema="newscene_cem_recipe_v1", method="cem", scene=scene, source_scene=author,
            source_files=author_sources, endpoint=endpoints[-1], stage_endpoints=endpoints,
            stage2_additional_iterations=endpoints[1] - endpoints[0],
            configuration_status="official_scene_config_resolution1_for_prepared_area4",
            mini_batch=1, initializer="new-scene training-LR frame0 COLMAP sparse only",
            manifest=identity(mp), geometry=geo))
        for method in ("splatsure", "iesrgs"):
            template = STATIC / f"configs/Flame07/{method}.json"
            cfg = copy.deepcopy(read(template))
            cfg.update(scene=scene, manifest=identity(mp), source_configuration=identity(rp),
                       wu_configuration=copy.deepcopy(registry["configuration"]),
                       cameras_extent=geo["cameras_extent"], train_camera_center=geo["train_camera_center"],
                       label=f"Wu + {method} / area4 / common frozen SwinIR x4 areaft",
                       teacher_variant="sr_swinir_x4_areaft",
                       readiness="CPU_metadata_prepared_real_inputs_and_GPU_acceptance_pending")
            cfg["information"] = dict(train_cameras=geo["train_cameras"], heldout_cameras=["cam00"],
                                      train_observations=len(geo["train_cameras"]) * 300,
                                      frames=list(range(300)), training_HR_read=False,
                                      heldout_pixels_read=False, source_model_scene_transfer=False)
            cfg["specification"] = dict(
                stages=["lr", "weights", "sr"] if method == "splatsure" else ["internal", "iedepth", "sr"],
                schedule="wu_dynerf_coarse3000_fine14000_v2", coarse_iterations=3000, fine_iterations=14000,
                init_sr="fresh", external_rgb="same registered SwinIR x4 areaft RGB8 cache as our method",
                external_depth="DA-V2-L on external RGB" if method == "iesrgs" else None)
            cfg["training"] = dict(
                coarse_iterations=3000, fine_iterations=14000, optimizer_updates=16999,
                batch_views=registry["configuration"]["OptimizationParams"]["batch_size"],
                internal_batch_views=3 if method == "iesrgs" else None,
                internal_view_reduction="sum; same-time author sampler" if method == "iesrgs" else None,
                lr_clock="Wu stage-local; position max_steps20000", topology="Wu 500/100/10000",
                opacity_reset_interval=60000, optimizer_updates_last_fine_iteration=False)
            cfg["dynamic_adaptations"] = [
                "Same main-table sv2c dynamic adaptation: static 3DGS replaced by Wu 843d5ac",
                "Author method losses, renderer, grouping and frozen task-root method sources retained",
                "Shared Wu coarse3000/fine14000 schedule, fresh SR stage; no project LR parent",
                "Common area4 LR observation operator and frozen SwinIR areaft RGB teacher",
                "sv2c per-axis hypot Mip expression prevents float32 underflow; same main-table patch",
                "Geometry recomputed from this scene; real camera IDs including missing streams preserved"]
            write_new(destination / f"{method}.json", cfg)
            audit.append(dict(scene=scene, method=method, template=identity(template),
                              target=identity(destination / f"{method}.json"),
                              differences=differences(read(template), cfg)))
        sr4d_args = copy.deepcopy(historical_sr4d["arguments"])
        sr4d_args.update(manifest=rel(mp), scene=scene,
                          output=rel(OUT / "sr4d" / scene / "model"))
        write_new(destination / "sr4d.json", dict(
            schema="sr4d_author_area4_preparation_v1", scene=scene, manifest=identity(mp),
            arguments=sr4d_args, author_options=historical_sr4d["options"],
            geometry=geo, initial_points=registry["data"]["initial_points"],
            teacher_used=False, batch="one training camera per author iteration; no Wu batch override",
            historical_template=identity(ROOT / "output/dynamic_sr_modules_20261010/sr4d_area4_20261028/runs/Flame07/sr4d_b/config.json"),
            status="recipe_only_not_a_training_completion"))
        write_new(destination / "input_contract.json", dict(
            schema="newscene_baseline_inputs_v1", scene=scene, manifest=identity(mp),
            registry=identity(rp), original_manifest=m["physical_lr_derivation"]["original_manifest"],
            geometry=geo, shared_initializer=m["initialization"],
            LR_images_read_in_P2=False, initialization_bytes_read_in_P2=False,
            teacher_bytes_read_in_P2=False, training_submitted=False,
            score_counts=dict(test=300, train=30 * len(geo["train_cameras"]))))
    write_new(OUT / "comparisons/config_differences.json", audit)
    print(__import__("json").dumps(dict(status="prepared", scenes=list(SCENES),
                                      scientific_CEM_changes=["resolution:2->1"], training=False)))


if __name__ == "__main__":
    build()
