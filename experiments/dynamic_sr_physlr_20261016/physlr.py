"""Physically sampled LR scenes (Claude, session 4dsr-7c, 2026-10-16).

Question (user, 2026-10-16 evening): can the LR formation itself be designed so that the method has room between LR, SR
and HR, and that room is recoverable? The registered protocol forms LR by torch bicubic with antialias=True, a digital
resize whose prefilter removes almost everything above the LR Nyquist frequency in every view, so multi-view fusion has
nothing beyond a single-image SR prior to recover. A camera instead integrates light over each pixel's area: an LR camera
with 4x4 binned (or 4x larger) pixels behind the same lens produces LR = 4x4 area average of the HR image, with view-
dependent aliasing that carries sub-pixel detail across views (Baker-Kanade box model). This module builds such scenes
for the registered pipeline.

  derive   prepared/<scene>_<formation>/: lr/ written from the registered HR of every observation (HR hard-linked),
           manifest = original with lr_sha256 replaced + physical_lr_derivation block.
           formations: area4 = 4x4 box average in 8-bit sRGB, rounded (pixel binning; no extra optics or noise).
  init     unmodified full_initialization.generate on the derived manifest (pure-LR triangulation, CPU), run with
           /home/cai_tianshun/Project/4dgs/.venv/bin/python and CUDA hidden (same as compress.py init).
  register LR-parent plan = original prefix plan with only `data` rebuilt (same configuration), degradation text set.
  tasks    LR coarse replay + gateaff fine stage task rows for pipeline.py.
  sr-register   Bsync config for the SR stage (clean registration with manifest/parent/teacher/LR files replaced;
           schedule and configuration byte-identical, checked).
compress.py / register_sr.py are imported, never modified. /usr/bin/python3 -I except `init`.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import copy
import importlib.util
import json
import math
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
COMP = ROOT / "experiments/dynamic_sr_compress_20261013"
sys.path.insert(0, str(COMP))
import compress as cp  # noqa: E402

_spec = importlib.util.spec_from_file_location("register_sr_compress", COMP / "register_sr.py")
RS = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(RS)

PREP = cp.PREPARED
LR_NS = ROOT / "output/dynamic_sr_cull_retrain_20261009/wave1/lr/physlr_20261016"
OUT_NS = ROOT / "output/dynamic_sr_physlr_20261016"
FORMATIONS = {
    "area4": dict(scale=4, kind="area", text="LR = 4x4 box (pixel-area) average of the registered HR in 8-bit sRGB, "
                                             "rounded to 8 bit; no optical prefilter, no noise (pixel binning)"),
}
SHORT = cp.SHORT


def scene_dir(scene, formation):
    return PREP / f"{scene}_{formation}"


def run_tag(scene, formation, seed):
    return f"{SHORT[scene]}{str(seed)[-2:]}_{formation}"


def plan_dir(scene, formation, seed):
    return LR_NS / "registry" / f"{scene}_{formation}" / f"seed_{seed}"


def _form(args):
    src, dst, kind, scale = args
    import numpy as np
    from PIL import Image
    hr = np.asarray(Image.open(src).convert("RGB")).astype(np.float64)
    h, w = hr.shape[:2]
    if h % scale or w % scale:
        raise ValueError(f"HR size {w}x{h} not divisible by {scale}")
    if kind != "area":
        raise ValueError(kind)
    lr = hr.reshape(h // scale, scale, w // scale, scale, 3).mean((1, 3))
    lr = np.clip(np.round(lr), 0, 255).astype(np.uint8)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.stem + ".tmp.png")
    Image.fromarray(lr).save(tmp)
    new = cp.sha(tmp)
    if dst.exists():
        same = cp.sha(dst) == new
        tmp.unlink()
        if not same:
            raise ValueError(f"refusing to replace a different LR: {dst}")
    else:
        tmp.replace(dst)
    return new


def derive(scene, formation, workers=12):
    f = FORMATIONS[formation]
    source = PREP / scene / "manifest.json"
    original = cp.read(source)
    if int(original["scale"]) != f["scale"]:
        raise ValueError("scale differs from the registered scene")
    dest = scene_dir(scene, formation)
    if dest.is_symlink():
        raise ValueError("derived directory must not be a symlink")
    manifest = copy.deepcopy(original)
    rows = manifest["observations"]
    for r in rows:  # registered HR must be unchanged
        if not (PREP / scene / r["hr_path"]).exists():
            raise ValueError(f"missing HR {r['hr_path']}")
    jobs = [(PREP / scene / r["hr_path"], dest / r["lr_path"], f["kind"], f["scale"]) for r in rows]
    with cf.ProcessPoolExecutor(workers) as ex:
        shas = list(ex.map(_form, jobs, chunksize=16))
    for r, s in zip(rows, shas):
        r["lr_sha256"] = s
    links = 0
    for r in rows:
        src, dst = PREP / scene / r["hr_path"], dest / r["hr_path"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            if not os.path.samefile(src, dst):
                raise ValueError(f"different HR present: {dst}")
            continue
        os.link(src, dst)
        links += 1
    # HR hashes are re-verified on a sample so a changed source is caught before training
    for r in rows[:: max(1, len(rows) // 60)]:
        if cp.sha(dest / r["hr_path"]) != r["hr_sha256"]:
            raise ValueError(f"HR hash differs: {r['hr_path']}")
    manifest["physical_lr_derivation"] = dict(
        schema="physical_lr_v1", formation=formation, scale=f["scale"], description=f["text"],
        original_manifest=cp.entry(source), generation_source=cp.entry(Path(__file__)),
        modified_original_fields=["observations[*].lr_sha256 (LR pixels re-formed from the registered HR)"],
        hr_layout="hard links to the original HR images", splits_unchanged=True,
        purpose="dataset-design gate: physically sampled LR (aliasing kept); not the registered benchmark protocol")
    cp.write_new(dest / "manifest.json", manifest)
    receipt = dict(scene=scene, formation=formation, manifest=cp.entry(dest / "manifest.json"), hr_links_created=links,
                   observations=len(rows))
    cp.write_new(dest / "derivation_receipt.json", receipt)
    return receipt


def init(scene, formation, seed):
    cp.derived_directory = lambda s, q: scene_dir(scene, formation)
    return cp.init(scene, formation, seed)


def register(scene, formation, seed):
    """compress.register with the physical degradation text (body copied; compress.py untouched)."""
    root = scene_dir(scene, formation)
    original_path = cp.ORIGINAL_PREFIX / scene / f"seed_{seed}" / "config.json"
    original = cp.read(original_path)
    m = cp.read(root / f"manifest_train_ready_seed{seed}.json")
    train = m["splits"]["train"]
    if train != original["data"]["train_cameras"] or m["splits"]["test"] != ["cam00"]:
        raise ValueError("training split differs from the original plan")
    init_ = m["initialization"]
    if init_["seed"] != seed or not init_["train_only_lr"] or cp.sha(root / init_["npz_path"]) != init_["npz_sha256"]:
        raise ValueError("seed-specific pure-LR initialization required")
    rows = [r for r in m["observations"] if r["split"] == "train"]
    inputs = []
    for r in rows:
        p = root / r["lr_path"]
        if p.is_symlink() or not p.resolve().is_relative_to(root.resolve()) or cp.sha(p) != r["lr_sha256"]:
            raise ValueError(f"illegal or changed LR input {p}")
        inputs.append(dict(camera=r["camera_id"], frame=int(r["frame_index"]), path=str(p.relative_to(ROOT)), sha256=r["lr_sha256"]))
    centers = [[float(m["cameras"][c]["c2w"][i][3]) for i in range(3)] for c in train]
    center = [sum(x[i] for x in centers) / len(centers) for i in range(3)]
    extent = 1.1 * max(math.sqrt(sum((x[i] - center[i]) ** 2 for i in range(3))) for x in centers)
    if max(abs(a - b) for a, b in zip(center, original["data"]["train_camera_center"])) > 1e-9 or \
            abs(extent - original["data"]["cameras_extent"]) > 1e-9:
        raise ValueError("camera geometry differs from the original plan")
    data = copy.deepcopy(original["data"])
    data.update(manifest=cp.entry(root / f"manifest_train_ready_seed{seed}.json"), original_manifest=cp.entry(root / "manifest.json"),
                initialization_receipt=cp.entry(root / f"initialization_seed_{seed}/complete.json"),
                initial_points=cp.entry(root / init_["npz_path"]), train_cameras=train, training_observations=len(rows),
                training_LR_sources_sha256=cp.digest(inputs),
                train_camera_center=original["data"]["train_camera_center"], cameras_extent=original["data"]["cameras_extent"])
    if "training_LR_bytes_SHA_verified" in data:
        data["training_LR_bytes_SHA_verified"] = len(inputs)
    degradation = f"physical LR formation {formation}: {FORMATIONS[formation]['text']}"
    if "domain_protocol" in data:
        data["domain_protocol"] = dict(data["domain_protocol"], seed=seed, degradation=degradation)
    else:
        data["degradation"] = degradation
    plan = copy.deepcopy(original)
    plan["status"] = f"registered_{formation}_LR_parent_pending_GPU"
    plan["data"] = data
    plan["author_adaptations"] = original["author_adaptations"] + [
        f"Physical-LR gate (2026-10-16): training LR re-formed as {formation} from the registered HR; configuration unchanged."]
    plan["physical_lr"] = dict(formation=formation, derivation=cp.entry(root / "derivation_receipt.json"),
                               original_plan=cp.entry(original_path), registry_source=cp.entry(Path(__file__)))
    if plan["configuration"] != original["configuration"]:
        raise ValueError("configuration changed")
    cp.write_new(plan_dir(scene, formation, seed) / "config.json", plan)
    return dict(plan=str(plan_dir(scene, formation, seed).relative_to(ROOT)), train_LR=len(inputs))


def lr_tasks(scene, formation, seed):
    t = run_tag(scene, formation, seed)
    plan = str(plan_dir(scene, formation, seed).relative_to(ROOT))
    coarse = LR_NS / "runs" / f"{t}_coarse"
    fine = LR_NS / "runs" / f"{t}_gateaff"
    return [dict(name=f"lr_coarse_{t}", deps=[], done=str((coarse / "complete.json").relative_to(ROOT)),
                 argv=["lr", "--parent", plan, "--coarse-only", "--arm", f"{formation}_author_coarse", "--out", str(coarse.relative_to(ROOT))]),
            dict(name=f"lr_fine_{t}", deps=[f"lr_coarse_{t}"], done=str((fine / "complete.json").relative_to(ROOT)),
                 argv=["lr", "--parent", plan, "--fine0", str((coarse / "checkpoint_fine_00000.pt").relative_to(ROOT)),
                       "--arm", "gateaff", "--prune-gate", "none", "--affine", "--out", str(fine.relative_to(ROOT))])]


def sr_register(scene, formation, seed, teacher_tag, parent_ckpt, out_tag):
    """Same rule as register_sr_stream.py: clean Bsync config, only protocol-dependent fields replaced."""
    clean_cfg_path = RS.CLEAN_BSYNC_SCENE[scene] / f"seed_{seed}" / "Bsync" / "config.json"
    base = RS.read(clean_cfg_path)
    root = scene_dir(scene, formation)
    manifest_path = root / f"manifest_train_ready_seed{seed}.json"
    manifest = RS.read(manifest_path)
    if manifest["physical_lr_derivation"]["formation"] != formation:
        raise ValueError("formation mismatch")
    lr_plan_path = plan_dir(scene, formation, seed) / "config.json"
    lr_plan = RS.read(lr_plan_path)
    if lr_plan["configuration"] != base["configuration"]:
        raise ValueError("LR plan configuration differs from the clean Bsync configuration")
    if lr_plan["data"]["manifest"]["path"] != str(manifest_path.relative_to(ROOT)):
        raise ValueError("LR plan does not use the derived manifest")
    parent_ckpt = ROOT / parent_ckpt
    complete = parent_ckpt.parent / "complete.json"
    teacher_index_path = root / f"teacher_index_{teacher_tag}_seed{seed}.json"
    teacher = RS.read(teacher_index_path)
    train = sorted(manifest["splits"]["train"])
    obs_train = {(o["camera_id"], int(o["frame_index"])): o for o in manifest["observations"] if o["camera_id"] in train}
    keys = [(e["camera"], int(e["frame"])) for e in teacher["entries"]]
    if sorted(teacher["train_cameras"]) != train or len(keys) != len(set(keys)) or set(keys) != set(obs_train):
        raise ValueError("teacher index keys are not exactly the training observations of this manifest")
    if teacher["manifest_sha256"] != RS.sha(manifest_path):
        raise ValueError("teacher index was written for a different manifest")
    if {Path(e["lr_relative_path"]).parts[0] for e in teacher["entries"]} != {"lr"}:
        raise ValueError("teacher inputs must be the manifest's training LR")
    bad = [k for k, e in zip(keys, teacher["entries"]) if e["lr_sha256"] != obs_train[k]["lr_sha256"]]
    if bad:
        raise ValueError(f"teacher inputs differ from the manifest LR at {bad[:3]}")
    for e in teacher["entries"]:
        if RS.sha(root / e["relative_path"]) != e["sha256"]:
            raise ValueError(f"teacher image changed: {e['relative_path']}")
    new = copy.deepcopy(base)
    new["status"] = "registered_ready_physical_lr_SR_refinement"
    new["data"]["manifest"] = RS.entry(manifest_path)
    new["data"]["original_manifest"] = RS.entry(root / "manifest.json")
    new["parent"] = dict(checkpoint=RS.entry(parent_ckpt), complete=RS.entry(complete), plan=RS.entry(lr_plan_path))
    new["full_native_context"]["manifest"] = RS.entry(manifest_path)
    new["full_native_context"]["parent"] = RS.entry(parent_ckpt)
    new["dependencies"]["teacher"] = RS.entry(teacher_index_path)
    obs = {(o["camera_id"], int(o["frame_index"])): o for o in manifest["observations"]}
    new["training_files"]["LR"] = [dict(camera=r["camera"], frame=r["frame"],
                                        path=str((root / obs[(r["camera"], int(r["frame"]))]["lr_path"]).relative_to(ROOT)),
                                        sha256=obs[(r["camera"], int(r["frame"]))]["lr_sha256"])
                                   for r in base["training_files"]["LR"]]
    new["physical_lr"] = dict(formation=formation, derivation_receipt=RS.entry(root / "derivation_receipt.json"),
                              teacher_tag=teacher_tag, teacher_input=teacher.get("input_policy"), derived_from=RS.entry(clean_cfg_path),
                              schedule_policy="schedule kept byte-identical to the clean registration (same cameras and frames)",
                              registrar=RS.entry(Path(__file__)))
    if new["dependencies"]["schedule"] != base["dependencies"]["schedule"]:
        raise AssertionError("schedule must stay identical")
    out = OUT_NS / "registry" / root.name / f"seed_{seed}" / out_tag / "Bsync" / "config.json"
    RS.write_new(out, new)
    return dict(config=str(out.relative_to(ROOT)))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=("derive", "init", "register", "lr-tasks", "sr-register"))
    p.add_argument("--scene", choices=sorted(SHORT), required=True)
    p.add_argument("--formation", choices=sorted(FORMATIONS), default="area4")
    p.add_argument("--seed", type=int, default=20261007)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--teacher-tag", default="sr_swinir_x4")
    p.add_argument("--parent", help="sr-register: LR parent checkpoint (project-relative)")
    p.add_argument("--out-tag", default="swinir_teacher")
    a = p.parse_args()
    if a.mode == "derive":
        r = derive(a.scene, a.formation, a.workers)
    elif a.mode == "init":
        r = init(a.scene, a.formation, a.seed)
    elif a.mode == "register":
        r = register(a.scene, a.formation, a.seed)
    elif a.mode == "lr-tasks":
        r = lr_tasks(a.scene, a.formation, a.seed)
    else:
        parent = a.parent or str((LR_NS / "runs" / f"{run_tag(a.scene, a.formation, a.seed)}_gateaff" / "checkpoint_fine_14000.pt").relative_to(ROOT))
        r = sr_register(a.scene, a.formation, a.seed, a.teacher_tag, parent, a.out_tag)
    print(json.dumps(r, ensure_ascii=False, default=str)[:3000])


if __name__ == "__main__":
    main()
