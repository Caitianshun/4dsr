"""Meaningful CPU checks; real-image/COLMAP/GPU execution remains owner work."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import time

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_bl import *
import cem_frame0_new as f0
import teacher_index_new as teacher
from finalize_sync_new import validate_paths, MAX_FILES, MAX_FILE, MAX_TOTAL
from make_lanes_new import compare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=rel(OUT / "checks/cpu_v1"))
    a = parser.parse_args()
    out = bounded(a.out)
    require(not out.exists(), "Use a fresh CPU-check output revision")
    out.mkdir(parents=True)
    started, checks = time.monotonic(), []

    def check(name, ok=True, evidence=None):
        require(ok, name)
        checks.append(dict(name=name, status="passed", evidence=evidence))

    def rejects(name, fun):
        try:
            fun()
        except (ValueError, KeyError):
            check(name, evidence="invalid input rejected")
        else:
            raise AssertionError("Did not reject: " + name)

    # All these roots are within the authorized read_paths. No image/NPZ read.
    readonly = [OLD / "score_views_area4.py", OLD / "frozen/score_views.py",
                OLD / "frozen/prepare_suite.py", STATIC / "sv2c.py",
                STATIC / "sv2_cleanup.py", STATIC / "build_areaft_index.py",
                SR4D / "train_sr4d_area4.py", SR4D / "sr4d_area4.py", SR4D / "sr4d_common.py"]
    readonly += [OLD / "localdygs_cem" / n for n in (
        "prepare_native_colmap_area4.py", "run_native_colmap.py", "run_area4.py",
        "run_author.py", "render_local_cem_area4.py", "prepare_local_cem.py", "protocol_interface_area4.py")]
    readonly += [p for p in (ROOT / "third_party/cem4dgs_20261020/configs").rglob("*.json")]
    readonly += [manifest_path(s) for s in SCENES]
    readonly += [REG / f"lr/registry/{s}_area4/seed_{SEED}/config.json" for s in SCENES]
    before = {rel(p): sha(p) for p in readonly}
    write_new(out / "readonly_before.json", before)
    try:
        for p in CODE.rglob("*.py"):
            compile(p.read_text(), str(p), "exec")
        check("new Python syntax", evidence=len(list(CODE.rglob("*.py"))))
        for p in CODE.rglob("*.json"):
            json.loads(p.read_text())
        check("new configuration JSON syntax")
        original = load("_cpu_original_prepare_frame0", OLD / "localdygs_cem/prepare_native_colmap_area4.py")
        projection = load("_cpu_original_projection", OLD / "localdygs_cem/protocol_interface_area4.py")
        scorer = load("_cpu_shared_scorer", OLD / "score_views_area4.py")
        errors, rotation_errors = [], []
        for scene, author in SCENES.items():
            mp, m, _, registry, geo = validate_metadata(scene)
            check(scene + " actual metadata/splits/registry", evidence=dict(
                cameras=geo["train_cameras"], center=geo["train_camera_center"], extent=geo["cameras_extent"]))
            # A held-out pose perturbation cannot affect training geometry.
            perturbed = copy.deepcopy(m)
            perturbed["cameras"]["cam00"]["w2c"][0][3] += 10000
            check(scene + " heldout excluded from geometry", geometry(perturbed) == geo)
            for stage in (1, 2):
                source = read(ROOT / f"third_party/cem4dgs_20261020/configs/N3V_s{stage}/{author}.json")
                target = read(CODE / f"configs/{scene}/cem_s{stage}.json")
                check(scene + f" CEM stage{stage} only resolution adaptation",
                      differences(source, target) == [dict(field="/resolution", old=2, new=1)])
            for method in ("splatsure", "iesrgs"):
                cfg = read(CODE / f"configs/{scene}/{method}.json")
                check(scene + " " + method + " exact registered Wu recipe",
                      cfg["wu_configuration"] == registry["configuration"])
                check(scene + " " + method + " recomputed geometry",
                      abs(cfg["cameras_extent"] - geo["cameras_extent"]) < 1e-12
                      and cfg["train_camera_center"] == geo["train_camera_center"])
                check(scene + " " + method + " effective schedule",
                      cfg["training"]["coarse_iterations"] == 3000 and cfg["training"]["fine_iterations"] == 14000
                      and cfg["training"]["batch_views"] == (2 if scene == "sear_steak" else 4))
            sr = read(CODE / f"configs/{scene}/sr4d.json")
            hist = read(ROOT / "output/dynamic_sr_modules_20261010/sr4d_area4_20261028/runs/Flame07/sr4d_b/config.json")
            check(scene + " SR4D author options unchanged", sr["author_options"] == hist["options"]
                  and sr["arguments"]["coarse_steps"] == 20000 and sr["arguments"]["fine_steps"] == 40000
                  and sr["teacher_used"] is False and sr["arguments"]["seed"] == 0)
            plan = f0.plan(scene)
            check(scene + " CEM plan only actual train frame0",
                  plan["frames"] == [0] and {r["camera"] for r in plan["inputs"]} == set(m["splits"]["train"])
                  and all(r["frame"] == 0 and "/lr/" in r["path"] for r in plan["inputs"]))
            check(scene + " no dense COLMAP stage",
                  len(plan["commands"]) == 5 and not any(x in str(plan["commands"])
                                                       for x in ("patch_match", "stereo_fusion", "undistorter")))
            for scope, count in (("test", 300), ("train", len(geo["train_cameras"]) * 30)):
                selected = scorer.select_observations_area4(m, scope)
                check(scene + " shared frozen selection " + scope, len(selected) == count,
                      dict(images=count, exact_shared_scorer=True))
            for camera in geo["train_cameras"]:
                spec = m["cameras"][camera]
                w = np.asarray(spec["w2c"])
                q = original.rotation_quaternion(w[:3, :3])
                qw, qx, qy, qz = q
                rotation = np.array([
                    [1 - 2 * (qy*qy + qz*qz), 2 * (qx*qy - qw*qz), 2 * (qx*qz + qw*qy)],
                    [2 * (qx*qy + qw*qz), 1 - 2 * (qx*qx + qz*qz), 2 * (qy*qz - qw*qx)],
                    [2 * (qx*qz - qw*qy), 2 * (qy*qz + qw*qx), 1 - 2 * (qx*qx + qy*qy)]])
                rotation_errors.append(float(np.max(np.abs(rotation - w[:3, :3]))))
                camera_points = np.array([[.13, .21, 2., 1.], [-.3, .2, 3., 1.]])
                world = camera_points @ np.linalg.inv(w).T
                for level in ("lr", "hr"):
                    k = np.asarray(spec["K_" + level])
                    width, height = m["resolutions"][level]
                    p = projection.projection_numpy(k, width, height)
                    clip = world @ w.T @ p.T
                    pixels = ((clip[:, :2] / clip[:, 3:] + 1) * [width, height] - 1) * .5
                    ref = camera_points[:, :3] @ k.T
                    errors.append(float(np.max(np.abs(pixels - ref[:, :2] / ref[:, 2:]))))
        check("actual training camera projection arithmetic", max(errors) < 1e-9,
              dict(cases=len(errors), max_error_pixels=max(errors)))
        check("original COLMAP quaternion arithmetic", max(rotation_errors) < 1e-12,
              dict(cameras=len(rotation_errors), max_error=max(rotation_errors)))
        good = dict(schema="cem_frame0_native_init_v1", status="completed", scene="sear_steak",
                    frames=[0], dense_reconstruction=False, manifest={"sha256": "fixture"},
                    calibrated_poses_and_intrinsics_unchanged=True, used_HR=False, used_test_pixels=False)
        f0.validate_completion(good, "sear_steak", {"sha256": "fixture"})
        for key, val in (("schema", "old_joint_schema"), ("frames", [0, 10]), ("used_HR", True),
                         ("used_test_pixels", True), ("scene", "flame_steak")):
            bad = dict(good, **{key: val})
            rejects("CEM receipt negative " + key, lambda bad=bad: f0.validate_completion(
                bad, "sear_steak", {"sha256": "fixture"}))
        # Synthetic SQLite database tests the nontrivial ID remap; no images.
        fixture = out / "SYNTHETIC_ONLY_database"
        manual, mapped = fixture / "manual", fixture / "mapped"
        manual.mkdir(parents=True)
        (manual / "cameras.txt").write_text("1 PINHOLE 336 252 101 102 167.5 125.5\n2 PINHOLE 336 252 201 202 167.5 125.5\n")
        (manual / "images.txt").write_text("1 1 0 0 0 0 0 0 1 cam01.png\n\n2 1 0 0 0 1 2 3 2 cam02.png\n\n")
        (manual / "points3D.txt").write_text("")
        database = fixture / "database.db"
        with sqlite3.connect(database) as con:
            con.execute("CREATE TABLE images(image_id INTEGER, camera_id INTEGER, name TEXT)")
            con.execute("CREATE TABLE cameras(camera_id INTEGER PRIMARY KEY, model INTEGER, width INTEGER, height INTEGER, params BLOB, prior_focal_length INTEGER)")
            con.executemany("INSERT INTO images VALUES(?,?,?)", [(17, 92, "cam02.png"), (41, 8, "cam01.png")])
            con.executemany("INSERT INTO cameras VALUES(?,0,1,1,NULL,0)", [(92,), (8,)])
        f0.remap_database(database, manual, mapped)
        with sqlite3.connect(database) as con:
            got = {cid: np.frombuffer(params, np.float64).tolist() for cid, params in
                   con.execute("SELECT camera_id,params FROM cameras")}
        check("COLMAP DB arbitrary image/camera ID remap",
              got == {92: [201., 202., 167.5, 125.5], 8: [101., 102., 167.5, 125.5]}
              and "17 1 0 0 0 1 2 3 92 cam02.png" in (mapped / "images.txt").read_text())
        sparse = fixture / "synthetic_sparse"
        sparse.mkdir()
        for name in ("cameras.txt", "images.txt"):
            (sparse / name).write_text((mapped / name).read_text())
        (sparse / "points3D.txt").write_text("1 0.1 0.2 3.0 100 110 120 0.3 17 0 41 0\n")
        check("known-pose finite sparse export acceptance", f0.validate_sparse(sparse, mapped) == 1)
        (sparse / "cameras.txt").write_text((mapped / "cameras.txt").read_text().replace("201", "999"))
        rejects("triangulator camera drift rejected", lambda: f0.validate_sparse(sparse, mapped))
        # Synthetic teacher images and producer metadata exercise actual PNG SHA verification.
        from PIL import Image
        fixture = out / "SYNTHETIC_ONLY_teacher"
        mp = fixture / "dataset/manifest.json"
        mp.parent.mkdir(parents=True)
        mm = dict(scene="sear_steak", splits={"train": ["cam01"], "test": ["cam00"]},
                  observations=[dict(camera_id="cam01", frame_index=0, split="train",
                                     lr_path="lr/cam01/0000.png", lr_sha256="b"*64)],
                  resolutions={"hr": [2, 2]})
        mp.write_text(json.dumps(mm))
        cache = fixture / "cache"
        png = cache / "sr_swinir_x4_areaft/cam01/0000.png"
        png.parent.mkdir(parents=True)
        Image.new("RGB", (2, 2), (30, 70, 90)).save(png)
        producer = dict(status="completed_teacher_inventory", scene="sear_steak", seed_consumer=SEED,
                        manifest_sha256=sha(mp), oracle=False, seed_independent_frozen_prior=True,
                        HR_quality_evaluation=False, train_cameras=["cam01"],
                        teacher_config=dict(tag="sr_swinir_x4_areaft", batch=6, scale=4,
                                            mask_cache=True, weights={"sha256": "a"*64}),
                        entries=[dict(camera="cam01", frame=0, lr_relative_path="lr/cam01/0000.png",
                                      lr_sha256="b"*64, relative_path=__import__("os").path.relpath(png, mp.parent),
                                      sha256=sha(png))])
        weights = producer["teacher_config"]["weights"]
        valid = teacher.entries_from_producer(producer, mm, mp, cache, weights)
        check("teacher actual synthetic PNG SHA/shape conversion", len(valid) == 1)
        check("teacher full reference SHA comparison", teacher.compare_reference(producer, valid)["mismatches"] == 0)
        for key, val in (("sha256", "c"*64), ("lr_sha256", "d"*64), ("camera", "cam00"),
                         ("relative_path", "../wrong.png")):
            bad = copy.deepcopy(producer)
            bad["entries"][0][key] = val
            rejects("teacher negative " + key, lambda bad=bad: teacher.entries_from_producer(
                bad, mm, mp, cache, weights))
        bad = copy.deepcopy(producer)
        bad["entries"] *= 2
        rejects("duplicate teacher keys", lambda: teacher.entries_from_producer(bad, mm, mp, cache, weights))
        bad2 = copy.deepcopy(producer)
        bad2["teacher_config"]["batch"] = 4
        rejects("changed teacher batch", lambda: teacher.entries_from_producer(bad2, mm, mp, cache, weights))
        wrong = copy.deepcopy(valid)
        wrong["cam01/0000"]["sha256"] = "0"*64
        rejects("cross-host encoded PNG SHA mismatch", lambda: teacher.compare_reference(producer, wrong))
        audit = read(OUT / "comparisons/argv_comparison.json")
        check("all historical argv comparisons", len(audit) == 126 and all(r["status"] == "passed" for r in audit),
              dict(groups=len(audit), token_positions=sum(len(r["token_comparison"]) for r in audit)))
        old = ["train.py", "--seed", "0", "--output", "old/output"]
        compare(old, ["train.py", "--seed", "0", "--output", "new/output"], {"old/output": "new/output"}, {})
        rejects("scientific argv scalar mutation", lambda: compare(
            old, ["train.py", "--seed", "1", "--output", "new/output"], {"old/output": "new/output"}, {}))
        rejects("path-shaped scalar cannot bypass argv audit", lambda: compare(
            old, ["train.py", "--seed", "/tmp/fake", "--output", "new/output"], {"old/output": "new/output"}, {}))
        document = {"version": 1, "files": [{"path": rel(CODE / "common_bl.py")}]}
        validate_paths(document, [rel(CODE)])
        for path in ("/absolute.png", "../escape.png", rel(CODE) + "/../escape.png"):
            rejects("unsafe sync path " + path, lambda path=path: validate_paths(
                {"version": 1, "files": [{"path": path}]}, [rel(CODE)]))
        rejects("sync prefix missing", lambda: validate_paths(document, ["data/unrelated"]))
        rejects("sync file-count cap", lambda: validate_paths(
            {"version": 1, "files": [{"path": "x"}] * (MAX_FILES + 1)}, ["x"]))
        batches = read(OUT / "sync/batches.json")["batches"]
        total = 0
        for b in batches:
            d = read(b["manifest"])
            validate_paths(d, ["data", "experiments", "output"])
            require(len(d["files"]) == b["files"] <= MAX_FILES and b["planning_budget_bytes"] <= MAX_TOTAL,
                    "Planned sync batch limits exceeded")
            total += len(d["files"])
        catalog = read(OUT / "sync/catalog.json")
        check("planned sync per-file and batch budgets", all(r["planning_budget_bytes"] <= MAX_FILE for r in catalog["files"]),
              dict(batches=len(batches), catalog_entries=len(catalog["files"]), paths_in_batches=total,
                   actual_bytes_verified=False, actual_prefix_registration_verified=False))
        command_doc = read(OUT / "commands.json")
        for lane in command_doc["lanes"] + command_doc["teacher_alternatives"]:
            result = subprocess.run(["/bin/bash", "-n", str(local(lane["shell"]))], capture_output=True, text=True)
            require(result.returncode == 0, result.stderr)
            for task in lane["tasks"]:
                require(task["interpreter"].startswith(REMOTE[task["host"]] + "/"),
                        "Remote interpreter is not absolute")
                require(task["dispatch_status"] == "prepared_not_submitted", "Unexpected submitted task")
        check("all 12 shell lane syntaxes and absolute interpreter declarations")
        check("Torch not imported / no GPU execution", "torch" not in sys.modules)
        after = {p: sha(p) for p in before}
        check("read-only source preservation during CPU checks", after == before, dict(files=len(before)))
        write_new(out / "checks.json", dict(
            status="passed", checks=checks, interpreter=sys.executable,
            isolated=sys.flags.isolated, seconds=time.monotonic()-started,
            real_LR_or_HR_or_NPZ_read=False, actual_COLMAP_executed=False,
            actual_teacher_verified=False, actual_remote_sync=False, GPU_or_training=False))
        print(json.dumps(dict(status="passed", checks=len(checks), output=rel(out),
                              seconds=round(time.monotonic()-started, 3), torch_imported=False)))
    except BaseException:
        write_new(out / "failure.json", dict(status="failed", checks=checks,
                                            error=__import__("traceback").format_exc()))
        raise


if __name__ == "__main__":
    main()
