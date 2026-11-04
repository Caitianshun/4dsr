"""Concrete path-only sync batches and per-file estimates; no source image reads."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_bl import *
from teacher_index_new import producer_path
from finalize_sync_new import MAX_FILES, MAX_FILE, MAX_TOTAL


def build():
    BATCH_COUNTS.clear()
    rows, batches, dependencies, downloads = [], [], [], []
    own_files = [p for p in CODE.rglob("*") if p.is_file() and p.suffix in (".py", ".json", ".txt", ".sh")]
    # Historical cache PNG bytes are a measured proxy, not measurements of new scenes.
    hist_index = read(STATIC / "configs/Flame07/areaft_index.json")
    teacher_mean = sum(e["bytes"] for e in hist_index["entries"].values()) / len(hist_index["entries"])
    groups = defaultdict(list)

    def add(host, group, path, reason, estimate=None, budget=None, digest=None, measured=False,
            presence="未知", evidence="No remote inventory result for this exact new-scene path"):
        path = local(path)
        if measured:
            estimate = budget = path.stat().st_size
            digest = sha(path)
        require(estimate is not None and budget is not None and 0 <= estimate <= budget <= MAX_FILE,
                "Supply conservative planning bytes")
        record = dict(host=host, path=rel(path), group=group, reason=reason,
                      estimated_bytes=int(estimate), planning_budget_bytes=int(budget),
                      size_basis="actual authorized file" if measured else "estimate; not stat/read in P2",
                      sha256=digest, already_on_remote=presence, remote_evidence=evidence,
                      current_remote_verified=False)
        rows.append(record)
        groups[host, group].append(record)

    historical_common = [OLD / "score_views_area4.py", OLD / "frozen/score_views.py",
                         OLD / "frozen/prepare_suite.py", STATIC / "sv2_cleanup.py"]
    for host in REMOTE:
        for file in own_files:
            if host == "cts" and (file.suffix == ".sh" and file.name.startswith("a100_")
                                  or file.suffix == ".txt" and file.name.startswith("a100_")):
                continue
            add(host, "code", file, "New adapters/configs/lanes; import paths remain project-relative", measured=True)
        sources = historical_common + ([STATIC / "sv2c.py", SR4D / "sr4d_common.py",
                                        SR4D / "sr4d_area4.py", SR4D / "train_sr4d_area4.py"]
                                       if host == "a100" else [
                                           OLD / ("localdygs_cem/" + n) for n in
                                           ("run_area4.py", "run_author.py", "prepare_local_cem.py",
                                            "prepare_native_colmap_area4.py", "protocol_interface_area4.py",
                                            "render_local_cem_area4.py")])
        for file in sources:
            add(host, "historical_code", file, "Original imported entry/scorer; upload only if missing or already identical",
                measured=True, presence="历史已用；当前未知", evidence="Flame07 successful lane/steps; exact hash pinned locally")
        for scene in SCENES:
            mp, m, rp, reg, geo = validate_metadata(scene)
            for file in (mp, mp.parent / "manifest.json", local(m["physical_lr_derivation"]["original_manifest"]["path"]), rp):
                add(host, "metadata", file, "Frozen input/registry identity; scorer must open the original non-area manifest",
                    measured=True)
            for o in m["observations"]:
                if o["split"] == "train":
                    add(host, scene + "_lr", mp.parent / o["lr_path"], "Training LR; all real train cameras, frames0..299",
                        estimate=110000, budget=256 * 1024, digest=o["lr_sha256"])
                if o["split"] == "test" or o["split"] == "train" and o["frame_index"] % 10 == 0:
                    add(host, scene + "_score_hr", mp.parent / o["hr_path"], "Scoring only; no HR model updates",
                        estimate=1800000, budget=4 * 2**20, digest=o["hr_sha256"])
            if host == "a100":
                init = m["initialization"]
                for key, digest_key in (("npz_path", "npz_sha256"), ("ply_path", "ply_sha256"), ("report_path", "report_sha256")):
                    estimate = init["point_count"] * (24 if key == "npz_path" else 48) if key != "report_path" else 16384
                    add(host, "shared_initialization", mp.parent / init[key],
                        "Same-scene LR-only initialization for SR4D/static methods, plus provenance",
                        estimate=estimate, budget=16 * 2**20, digest=init[digest_key])
                add(host, "shared_initialization", local(reg["data"]["initialization_receipt"]["path"]),
                    "Frozen shared initializer completion provenance", estimate=16384, budget=2**20,
                    digest=reg["data"]["initialization_receipt"]["sha256"])
                for o in m["observations"]:
                    if o["split"] == "train":
                        add(host, scene + "_teacher_a", REG / "teacher_cache" / (scene + "_area4") /
                            "sr_swinir_x4_areaft" / o["camera_id"] / f"{o['frame_index']:04d}.png",
                            "Scheme a: upload common frozen teacher RGB8, shared by SplatSuRe/IE-SRGS",
                            estimate=round(teacher_mean), budget=4 * 2**20)
                add(host, "teacher_reference_both_schemes", producer_path(scene),
                    "Scheme a source inventory; scheme b unchanged local per-PNG SHA reference",
                    estimate=2 * 2**20, budget=8 * 2**20)
                dependencies.append(dict(
                    host=host, path=rel(OUT / "teacher_cache" / (scene + "_area4")),
                    type="remote_generated_tree", reason="Scheme b produces here; upload no generated PNGs in scheme b",
                    estimated_bytes=round(teacher_mean * len(geo["train_cameras"]) * 300),
                    planning_budget_bytes=4 * 2**20 * len(geo["train_cameras"]) * 300,
                    already_on_remote="未知", uploaded_in_P2=False))
            else:
                dependencies.append(dict(host=host, path=rel(OUT / "cem" / scene / "native_init"),
                    type="remote_generated_tree", reason="Compute CEM frame0 sparse on CTS from synchronized training LR",
                    estimated_bytes=16 * 2**20, estimate_basis="planning allowance only; actual point count unknown",
                    already_on_remote="未知", uploaded_in_P2=False))

    # Exact known producer/metadata paths; source dependencies outside read_paths
    # are declared, not guessed or recursively scanned.
    for file in (ROOT / "experiments/dynamic_sr_modules_20261010/newscenes_reg_20261103/teacher_cache_new.py",
                 ROOT / "experiments/dynamic_sr_modules_20261010/newscenes_reg_20261103/common_new.py"):
        add("a100", "teacher_b_code", file, "Scheme b uses the same original path-only teacher wrapper", measured=True)
    add("a100", "teacher_recipe", ROOT / "output/dynamic_sr_physlr_20261016/teacher_ft/swinir_area_ft_v1/summary.json",
        "Authoritative frozen teacher weight identity used by index verification; outside P2 read scope",
        estimate=16384, budget=2 * 2**20)
    add("a100", "teacher_b_code", ROOT / "experiments/dynamic_sr_physlr_20261016/teacher_physlr3.py",
        "Pinned original teacher producer, consumed unchanged by teacher_cache_new",
        estimate=32768, budget=2**20, digest="dde031195a6db7f4b64eaa5c921788a9f8f181e7d047363a37052243836acde6")
    # Exact runtime files with historical SHA identities.
    complete = read(ROOT / "output/dynamic_sr_modules_20261010/static_sr_v2_20261026/runs/Flame07/iesrgs_sr/complete.json")
    for file, digest in complete["frozen_sources"].items():
        src = ROOT / "experiments/static_sr_wu_20261022" / file
        require(sha(src) == digest, "Frozen migration source changed")
        dependencies.append(dict(host="a100", path=TASK_ROOT_PATH + "/experiments/static_sr_wu_20261022/" + file,
            type="preinstalled_file", reason="Frozen method source imported from the historical task root",
            estimated_bytes=src.stat().st_size, sha256=digest, already_on_remote="历史成功使用；当前未知",
            remote_evidence="Flame07 complete.json frozen_sources", upload="reuse after remote SHA check; do not replace with another version"))
    dependencies.extend([
        dict(host="a100", path=REMOTE["a100"] + "/.venv/bin/python", type="interpreter",
             reason="SR4D and static-SR/scorer historical interpreter", estimated_bytes=None,
             already_on_remote="历史已用；当前未知", remote_evidence="SR4D config.environment.python; successful A100 job logs"),
        dict(host="cts", path=REMOTE["cts"] + "/envs/dynamic_lr_baselines_20261020/cem/bin/python",
             type="interpreter", reason="CEM leaf, original runtime and COLMAP Python dependencies",
             estimated_bytes=None, already_on_remote="历史已用；当前未知", remote_evidence="Flame07 CEM training receipts"),
        dict(host="cts", path=REMOTE["cts"] + "/.venv/bin/python", type="interpreter",
             reason="Shared scorer, LPIPS environment", estimated_bytes=None,
             already_on_remote="历史已用；当前未知", remote_evidence="Flame07 CEM steps.jsonl score argv"),
        dict(host="cts", path="/usr/local/bin/colmap", type="preinstalled_file",
             reason="Exact frame0 native initializer", estimated_bytes=None,
             sha256=read(ROOT / "output/dynamic_sr_modules_20261010/area4_baselines_20261024/localdygs_cem/flame_steak/native_init/complete.json")["runtime_adaptation"]["colmap_binary_sha256"],
             already_on_remote="历史已用；当前未知", remote_evidence="Flame07 native_init complete.json; af2b8659"),
    ])
    pending = [
        ("a100", "experiments/dynamic_sr_modules_20261010/sr4d_area4_20261028/upstream", "SR4D author source tree"),
        ("a100", "experiments/dynamic_sr_modules_20261010/sr4d_area4_20261028/ext_a100", "Four pinned SR4D extension files"),
        ("a100", "experiments/dynamic_sr_modules_20261010/sr4d_area4_20261028/vendor", "SR4D kornia/thop vendored dependencies"),
        ("a100", TASK_ROOT_PATH + "/output/static_sr_wu_20261022/runtime", "SplatSuRe and IE-SRGS compiled CUDA extensions"),
        ("a100", TASK_ROOT_PATH + "/output/static_sr_wu_20261024/next_teacher_gates_cpu_v1/iesrgs_clean_74d60d5/submodules/Depth-Anything-V2", "Pinned DA-V2-L source (sv2c first candidate)"),
        ("a100", TASK_ROOT_PATH + "/tmp/static_sr_wu_20261022/upstream/iesrgs/submodules/Depth-Anything-V2", "Pinned DA-V2-L source (sv2c fallback candidate; only one candidate needed)"),
        ("a100", TASK_ROOT_PATH + "/weight/static_sr_wu_20261022/depth_anything_v2_large/depth_anything_v2_vitl.pth", "DA-V2-L weights, known exact filename, approx1.34GB"),
        ("cts", "third_party/cem4dgs_calibrated_runtime_v3_20261020", "Original calibrated CEM runtime and installed native extensions"),
        ("a100", "experiments/dynamic_sr_cull_retrain_20261009", "Wu rt/cullretrain/native data interface dependencies already used by static baselines"),
        ("a100", "third_party/4dgs_843d5ac_runtime", "Wu author backbone dependencies"),
        ("a100", "output/dynamic_sr_physlr_20261016/teacher_ft/swinir_area_ft_v1", "Teacher frozen weight file(s); filenames not in allowed material"),
        ("a100", "experiments/dynamic_sr_physlr_20261016", "Scheme b transitive producer/model imports; cannot enumerate without reading original producer"),
    ]
    for host, path, reason in pending:
        dependencies.append(dict(host=host, path=path, type="requires_owner_inventory",
                                 reason=reason, estimated_bytes=1340000000 if path.endswith("vitl.pth") else None,
                                 already_on_remote="未知", historical_evidence="historical lane/runtime dependency; no current inventory result",
                                 complete_file_enumeration=False, upload="only if absent, after exact inventory and fixed prefix acceptance"))
    for host in REMOTE:
        for path, reason, estimate in [
            ("experiments/dynamic_sr_20260918/evaluate.py", "Shared scorer's original metric helper", 32768),
            (rel(OUT / "model_cache/torch_home/hub/checkpoints/alexnet-owt-7be5be79.pth"),
             "Same frozen LPIPS AlexNet backbone in new output cache; materialize from existing host cache, offline", 244408911)]:
            dependencies.append(dict(host=host, path=path, type="requires_owner_inventory",
                                     reason=reason, estimated_bytes=estimate, already_on_remote="未知",
                                     complete_file_enumeration=True, upload="exact path known; source bytes/SHA not available in P2"))
        dependencies.append(dict(host=host, path="installed lpips/weights/v0.1/alex.pth",
                                 type="installed_package_dependency", reason="LPIPS v0.1 head; do not download online",
                                 estimated_bytes=6009, already_on_remote="历史评分已用；当前未知",
                                 upload="verify installed package and scorer's expected state hash"))

    for (host, group), items in sorted(groups.items()):
        chunk, budget = [], 0
        for item in items:
            if chunk and (len(chunk) == MAX_FILES or budget + item["planning_budget_bytes"] > MAX_TOTAL):
                batches.append(save_batch(host, group, chunk, budget))
                chunk, budget = [], 0
            chunk.append(item)
            budget += item["planning_budget_bytes"]
        if chunk:
            batches.append(save_batch(host, group, chunk, budget))
    # Download lists include the three exact surviving samples per camera.
    for scene in SCENES:
        m = read(manifest_path(scene))
        for method in METHOD_NAMES:
            host = "cts" if method == "cem" else "a100"
            base, files = OUT / method / scene, []
            for scope in ("test", "train"):
                for name in ("summary.json", "per_view.csv", "area4_selection.json"):
                    files.append(base / f"scores_{scope}" / name)
                files += [base / f"renders_{scope}/render_index.json", base / f"renders_{scope}/cleanup_receipt.json"]
                if method == "cem":
                    files.append(base / f"renders_{scope}/render_receipt.json")
                frames = (0, 150, 299) if scope == "test" else (0, 150, 290)
                files += [base / f"renders_{scope}/{cam}/{f:04d}.png" for cam in m["splits"][scope] for f in frames]
            if method == "cem":
                files += [base / f"training_stage{s}_receipt.json" for s in (1, 2)]
                files += [base / "native_inputs/frame0_input_receipt.json",
                          base / "native_init/frame0_complete.json", base / "native_init/frame0_status.json"]
            elif method == "sr4d":
                files += [base / "check.json"]
                files += [base / "model" / n for n in ("config.json", "status.json", "input_validation.json",
                           "coarse_complete.json", "fine_complete.json", "extensions_coarse.json", "extensions_fine.json",
                           "timelog/trainlog_stage1.json", "timelog/trainlog_stage2.json",
                           "deform_hr/iteration_40000/stage_metadata.json")]
            else:
                stages = ("lr", "sr") if method == "splatsure" else ("internal", "sr")
                files += [base / stage / n for stage in stages for n in ("run.json", "complete.json", "trajectory.jsonl")]
                files += [base / ("weights/receipt.json" if method == "splatsure" else "depth/index.json")]
            files.append(base / "result_acceptance.json")
            path = OUT / f"sync/download/{host}_{method}_{scene}.json"
            write_new(path, dict(version=1, files=[dict(path=rel(f)) for f in files]))
            downloads.append(dict(host=host, method=method, scene=scene, manifest=rel(path), files=len(files),
                                  reason="scores, per-view CSV, render SHA index, all surviving samples, receipts/config/cost logs",
                                  downloaded=False))
    write_new(OUT / "sync/catalog.json", dict(schema="newscene_sync_catalog_v1", files=rows,
        missing_actual_byte_inventory=True, transferred=False,
        byte_assumptions=dict(LR_estimated_each=110000, LR_planning_budget_each=256 * 1024,
                              score_HR_estimated_each=1800000, HR_planning_budget_each=4 * 2**20,
                              teacher_estimated_each=teacher_mean, teacher_proxy=identity(STATIC / "configs/Flame07/areaft_index.json")),
        limits=dict(max_files=MAX_FILES, max_bytes=MAX_TOTAL, max_file_bytes=MAX_FILE)))
    write_new(OUT / "sync/batches.json", dict(
        schema="newscene_sync_batches_v1", batches=batches, actual_sync=False,
        prefix_registration="未知：任务未提供固定同步入口的前缀登记表；每批先用真实导出核对，再交固定入口验收",
        actual_bytes_gate="Every batch must pass finalize_sync_new.py before transfer",
        scheme_a_groups=["code", "historical_code", "metadata", "shared_initialization",
                         "*_lr", "*_score_hr", "*_teacher_a", "teacher_reference_both_schemes", "teacher_recipe"],
        scheme_b_groups=["code", "historical_code", "metadata", "shared_initialization",
                         "*_lr", "*_score_hr", "teacher_reference_both_schemes", "teacher_recipe", "teacher_b_code"],
        additional_dependencies="dependency_inventory.json must be resolved; no directory placeholder is in a path-only batch"))
    write_new(OUT / "sync/dependency_inventory.json", dict(
        schema="newscene_remote_dependency_inventory_v1", items=dependencies, network_used=False,
        limitation="Unknown tree contents/weight filenames remain explicit inventory gaps; no fabricated complete bootstrap manifest"))
    write_new(OUT / "sync/download_plan.json", dict(version=1, batches=downloads, actual_download=False))
    # CPU input acceptance, separate from GPU lane files.
    preflight = []
    for host in REMOTE:
        for scene in SCENES:
            preflight.append(dict(host=host, cwd=REMOTE[host], interpreter=REMOTE[host] + "/.venv/bin/python",
                                 argv=[rel(CODE / "verify_inputs_new.py"), "--host", host, "--scene", scene],
                                 resource="CPU", done=rel(OUT / f"preflight/{host}/{scene}/inputs.json"),
                                 status="not_run_actual_images_outside_P2_scope"))
    write_new(OUT / "preflight_commands.json", preflight)
    print(__import__("json").dumps(dict(catalog_files=len(rows), upload_batches=len(batches),
                                      download_batches=len(downloads), actual_sync=False)))


def save_batch(host, group, items, budget):
    # Current groups fit one batch. Number suffix is deterministic even if a
    # future larger scene needs a second chunk.
    existing = BATCH_COUNTS[host, group]
    BATCH_COUNTS[host, group] += 1
    # No arbitrary directory expansion: each JSON contains concrete file paths.
    name = f"{host}_{group}_{existing + 1:02d}.json"
    path = OUT / "sync/upload" / name
    write_new(path, dict(version=1, files=[dict(path=r["path"]) for r in items]))
    return dict(host=host, group=group, manifest=rel(path), files=len(items),
                estimated_bytes=sum(r["estimated_bytes"] for r in items), planning_budget_bytes=budget,
                actual_bytes_verified=False, transferred=False)


TASK_ROOT_PATH = "tasks/static_sr_shared_wu_20261024_a100"
BATCH_COUNTS = defaultdict(int)


if __name__ == "__main__":
    build()
