"""Freeze or render the twelve existing Wu area4 development references.

prepare/verify are CPU-only and do not import torch. render requires external
GPU admission. It reloads the original frozen checkpoint without posthoc
culling, training-camera color/warp correction, test-image fitting or updates.
The selected historical checkpoints remain development references, not a new
unselected seed experiment. Score their PNGs with the common evaluate_images.py.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
ARMS = ("areaL0", "naive", "areaT", "areaTm", "w4", "oracle")
SCENES = {"Cook07": "cook_spinach", "Cut07": "cut_roasted_beef"}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def local(path):
    value = Path(path)
    if value.is_absolute():
        raise ValueError("Inventory paths must be project-relative")
    result = (ROOT / value).resolve()
    if not result.is_relative_to(ROOT):
        raise ValueError("Inventory path escaped the project")
    return result


def read(path):
    return json.loads(Path(path).read_text())


def entry(path):
    path = Path(path).resolve()
    return dict(path=str(path.relative_to(ROOT)), sha256=sha(path), bytes=path.stat().st_size)


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def verify_entry(item):
    path = local(item["path"])
    if not path.is_file() or path.stat().st_size != item["bytes"] or sha(path) != item["sha256"]:
        raise ValueError("Required file identity differs: " + str(path))
    return path


def common_sources():
    files = [Path(__file__)]
    files += [ROOT / "experiments/dynamic_sr_cull_retrain_20261009" / n
              for n in ("cullretrain.py", "rt.py")]
    files += [ROOT / "experiments/dynamic_sr_multiview_footprint_20261007" / n
              for n in ("full_prefix_registered.py", "full_refine_registered_v2.py",
                        "full_evaluate_registered_v2.py", "full_native_protocol.py",
                        "full_prefix.py", "fp_common.py", "config.py", "losses.py")]
    files += [ROOT / "experiments/dynamic_sr_20260918" / n
              for n in ("n3dv_data.py", "evaluate.py")]
    files.append(ROOT / "deployment/dynamic_lr_baselines_20261020/evaluate_images.py")
    # These are pinned Python sources, not local-architecture CUDA binaries.
    files += sorted((ROOT / "third_party/4dgs_843d5ac_runtime").rglob("*.py"))
    return [entry(p) for p in sorted(set(files)) if "__pycache__" not in p.parts]


def prepare(args):
    evidence_path = ROOT / "docs/area4_methods_metrics_2026-10-18.evidence.json"
    evidence = read(evidence_path)
    metric_reference_path = ROOT / "output/dynamic_lr_baselines_20261020/a100_wu_original_lr/Cook/shared_PNG_metrics/summary.json"
    metric_reference = read(metric_reference_path)
    tasks, test_assets = [], {}
    for label, scene in SCENES.items():
        manifest_path = ROOT / f"data/dynamic_sr/full_time_20261007/prepared/{scene}_area4/manifest_train_ready_seed20261007.json"
        manifest = read(manifest_path)
        assert manifest["splits"]["test"] == ["cam00"]
        assert manifest["physical_lr_derivation"]["formation"] == "area4"
        observations = sorted((r for r in manifest["observations"] if r["split"] == "test"), key=lambda r: r["frame_index"])
        assert [(r["camera_id"], r["frame_index"]) for r in observations] == [("cam00", i) for i in range(300)]
        images = []
        for row in observations:
            image = entry(manifest_path.parent / row["hr_path"])
            assert image["sha256"] == row["hr_sha256"]
            images.append(image)
        test_assets[label] = images
        plan_path = ROOT / f"output/dynamic_sr_cull_retrain_20261009/wave1/lr/physlr_20261016/registry/{scene}_area4/seed_20261007/config.json"
        plan = read(plan_path)
        assert plan["data"]["manifest"]["sha256"] == sha(manifest_path)
        for arm in ARMS:
            row = (evidence["references"] if arm in ("areaL0", "oracle") else evidence["main"])[label][arm]
            result_path = local(row["path"])
            result = read(result_path)
            if "checkpoint" in result:
                checkpoint = local(result["checkpoint"]["path"])
            else:
                checkpoint = result_path.parent.parent / "sr" / result["label"] / "checkpoint_06000.pt"
            completion_path = checkpoint.parent / "complete.json"
            completion = read(completion_path)
            assert completion["status"] == "completed" and completion["cursor"] == 6000
            assert completion["args"]["cull"] is False and completion["culls"] == []
            assert completion["args"]["affine"] is True
            ck = entry(checkpoint)
            assert ck["sha256"] == completion["checkpoint"]["sha256"]
            if "checkpoint" in result:
                assert ck["sha256"] == result["checkpoint"]["sha256"]
                assert result["points_loaded"] == completion["points"]
            task = dict(id=f"{label}_{arm}", scene=scene, label=label, arm=arm,
                        manifest=entry(manifest_path), plan=entry(plan_path), checkpoint=ck,
                        expected_points_loaded=completion["points"], historical_metrics=row["metrics"],
                        historical_metric_protocol="raw_float_with_posthoc_cull; do not mix with new common PNG results",
                        original_training_cull=False, original_training_culls=[],
                        irreversible_posthoc_cull_detected=False,
                        cull_evidence="Original completion cull=false/culls=[]; original checkpoint SHA verified; historical points_loaded equals saved point count when directly reported. Evaluation culls only the separately loaded in-memory model and does not save it.",
                        historical_selection="fixed identity from the existing best-completed-run development table; not a new random-seed run",
                        source_result=entry(result_path), completion=entry(completion_path),
                        original_run=entry(checkpoint.parent / "run.json"),
                        output_relative=f"output/dynamic_sr_cem_backbone_20261020/wu_reference/{label}/{arm}")
            task["required_files"] = [task[k] for k in ("manifest", "plan", "checkpoint", "source_result", "completion", "original_run")]
            task["render_command"] = [".venv/bin/python", "deployment/dynamic_sr_cem_backbone_20261020/wu_reference.py",
                                      "render", "--inventory", "output/dynamic_sr_cem_backbone_20261020/wu_reference_inventory.json",
                                      "--task", task["id"]]
            task["metric_command"] = [".venv/bin/python", "deployment/dynamic_lr_baselines_20261020/evaluate_images.py",
                                      "--manifest", task["manifest"]["path"], "--render-dir", task["output_relative"] + "/renders_hr",
                                      "--method", "wu_area4_" + arm, "--out", task["output_relative"] + "/common_A100_metrics"]
            tasks.append(task)
    inventory = dict(schema="Wu_area4_fixed_no_posthoc_reference_v1", status="CPU_verified_not_dispatched",
                     prepared_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                     evidence=entry(evidence_path), common_required_files=common_sources(),
                     test_HR_required_files=test_assets, tasks=tasks,
                     inference_policy=dict(posthoc_cull=False, color_affine=False, spatial_warp=False,
                                           test_color_fit=False, parameter_updates=0,
                                           no_training_LR_opened=True, no_training_HR_opened=True,
                                           no_teacher_opened=True, no_test_LR_opened=True),
                     CUDA_binary_policy="A100 must use its existing pinned-runtime sm80 rasterizer and simple_knn extensions. rt.check_runtime verifies module locations; record actual binary SHA at rendering. Local binaries are not portable requirements.",
                     environment_requirements=["A100 base .venv torch2.7.1+cu128", "numpy", "Pillow", "open3d", "plyfile", "torchvision", "tqdm", "tkinter", "scipy", "matplotlib", "lpips", "opencv-python", "pinned runtime CUDA extensions"],
                     expected_metric_environment={k: metric_reference[k] for k in
                         ("torch", "numpy", "opencv", "gpu", "evaluator_sha256", "helper_source_sha256", "lpips_state_dict_sha256")},
                     metric_identity_reference=entry(metric_reference_path),
                     metric_stage="After successful rendering run the frozen common evaluate_images.py in the same admitted A100 base environment; AlexNet LPIPS0.1 and its already installed cache. No separate raw metric average is a main score.")
    write_new(args.out, inventory)
    print(json.dumps(dict(status=inventory["status"], tasks=len(tasks), inventory=entry(Path(args.out)))))


def verify(args):
    inventory = read(args.inventory)
    assert inventory["schema"] == "Wu_area4_fixed_no_posthoc_reference_v1"
    entries = inventory["common_required_files"] + [inventory["evidence"]]
    for task in inventory["tasks"]:
        entries += task["required_files"]
    for images in inventory["test_HR_required_files"].values():
        entries += images
    unique = {x["path"]: x for x in entries}
    for item in unique.values():
        verify_entry(item)
    tree = ast.parse(Path(__file__).read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "render")
    calls = [ast.unparse(n.func) for n in ast.walk(function) if isinstance(n, ast.Call)]
    assert not any(n in calls for n in ("cr.cull", "cr.Culler", "cr.CameraAffine", "model.optimizer.step"))
    assert "cr.load_for_render" in calls and "cr.native.native_rgb" in calls
    print(json.dumps(dict(status="CPU_verified", required_files=len(unique), tasks=len(inventory["tasks"]),
                          inventory_sha256=sha(args.inventory), torch_imported="torch" in sys.modules)))


def render(args):
    inventory = read(args.inventory)
    assert inventory["schema"] == "Wu_area4_fixed_no_posthoc_reference_v1"
    matches = [t for t in inventory["tasks"] if t["id"] == args.task]
    if len(matches) != 1:
        raise ValueError("Task id is absent or duplicated")
    task = matches[0]
    for item in inventory["common_required_files"] + task["required_files"]:
        verify_entry(item)
    for item in inventory["test_HR_required_files"][task["label"]]:
        verify_entry(item)
    out = args.out.resolve() if args.out else local(task["output_relative"])
    if out.exists():
        raise FileExistsError("Render output exists; preserve it: " + str(out))
    # No environment/device mutation here: the external broker owns GPU admission.
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible or len(visible.split(",")) != 1 or visible == "-1":
        raise ValueError("Explicit external single-GPU admission is required")
    import numpy as np
    from PIL import Image
    import torch
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    sys.path.insert(0, str(ROOT / "experiments/dynamic_sr_cull_retrain_20261009"))
    import cullretrain as cr
    runtime = cr.rt.check_runtime()
    plan = read(local(task["plan"]["path"]))
    manifest = read(local(task["manifest"]["path"]))
    model, payload = cr.load_for_render(local(task["checkpoint"]["path"]), plan)
    assert payload["plan"]["configuration"] == plan["configuration"]
    assert payload["plan"]["data"]["manifest"] == plan["data"]["manifest"]
    assert int(model.get_xyz.shape[0]) == task["expected_points_loaded"]
    del payload
    _, _, pipeline, _ = cr.configuration(plan)
    background = torch.ones(3, device="cuda") if cr.white_background(plan) else torch.zeros(3, device="cuda")
    rows = sorted((r for r in manifest["observations"] if r["split"] == "test"), key=lambda r: r["frame_index"])
    assert [(r["camera_id"], r["frame_index"]) for r in rows] == [("cam00", i) for i in range(300)]
    out.mkdir(parents=True, exist_ok=False)
    (out / "renders_hr").mkdir()
    (out / "raw_reference_frames").mkdir()
    record = dict(schema="Wu_area4_fixed_raw_no_posthoc_render_v1", task=task,
                  inventory_sha256=sha(args.inventory), inference_policy=inventory["inference_policy"],
                  points_loaded=task["expected_points_loaded"], points_rendered=int(model.get_xyz.shape[0]),
                  host=socket.gethostname(), gpu=torch.cuda.get_device_name(),
                  CUDA_VISIBLE_DEVICES=os.environ["CUDA_VISIBLE_DEVICES"], runtime=runtime,
                  binary_identity=[entry(Path(module.__file__)) for module in
                                   (sys.modules["diff_gaussian_rasterization._C"], sys.modules["simple_knn._C"])],
                  source=entry(Path(__file__)), PNG_protocol="np.rint(clip(raw,0,1)*255).astype(uint8)")
    write_new(out / "run.json", record)
    started = time.monotonic()
    pngs = []
    try:
        for row in rows:
            with torch.inference_mode():
                camera = cr.native.hr_camera(cr.ev.camera_item(manifest, row, torch), row["frame_index"], manifest)
                state = cr.native.native_effective_state(model, float(row["time"]), torch)
                raw = cr.native.native_rgb(camera, model, state, pipeline, background)["render"].detach()
                width, height = manifest["resolutions"]["hr"]
                assert tuple(raw.shape) == (3, height, width) and bool(torch.isfinite(raw).all())
                arr = raw.cpu().numpy()
            frame = int(row["frame_index"])
            png = out / "renders_hr" / f"{frame:04d}.png"
            Image.fromarray(np.rint(arr.transpose(1, 2, 0).clip(0, 1) * 255).astype(np.uint8)).save(png)
            pngs.append(dict(frame=frame, path=f"renders_hr/{png.name}", sha256=sha(png), gt_sha256=row["hr_sha256"]))
            if frame in (0, 150, 299):
                np.savez_compressed(out / "raw_reference_frames" / f"{frame:04d}.npz", rgb_raw=arr.astype(np.float32))
        assert len(pngs) == 300 and int(model.get_xyz.shape[0]) == task["expected_points_loaded"]
        write_new(out / "render_receipt.json", dict(record, status="completed", frames=300, PNGs=pngs,
                  seconds=time.monotonic() - started, parameter_updates=0,
                  main_metrics_status="pending_common_A100_evaluate_images"))
    except BaseException as error:
        write_new(out / "failure.json", dict(status="failed", exception=repr(error), frames_saved=len(pngs)))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare"); p.add_argument("--out", type=Path, required=True); p.set_defaults(call=prepare)
    p = sub.add_parser("verify"); p.add_argument("--inventory", type=Path, required=True); p.set_defaults(call=verify)
    p = sub.add_parser("render"); p.add_argument("--inventory", type=Path, required=True); p.add_argument("--task", required=True)
    p.add_argument("--out", type=Path); p.set_defaults(call=render)
    args = parser.parse_args(); args.call(args)


if __name__ == "__main__":
    main()
