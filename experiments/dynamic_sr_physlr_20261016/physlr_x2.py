"""Area x2 physical-LR scenes (Claude 4dsr-46, 2026-10-24; user: main table = x4 on the left, x2 on the right).

Wrapper around physlr.py (unchanged; imported) that adds the formation
  area2 = 2x2 box (pixel-area) average of the registered HR in 8-bit sRGB, rounded to 8 bit (pixel binning)
i.e. exactly physlr.py area4 with a block of 2. physlr.derive refuses a formation whose scale differs from the registered
scene (x4), so derive and register are re-implemented here with the scale change of the camera-stream protocol
(experiments/dynamic_sr_stream_20261016/stream.py): manifest = original with scale = 2, resolutions.lr = HR / 2,
cameras.*.K_lr by the pixel-centre rule u' = (u + 0.5) / s - 0.5 (stream.scaled_K, copied below), every observation's
lr_sha256 replaced; LR plan data gets resolution_LR = HR / 2. HR is hard-linked; splits, cameras, configuration and
schedule are unchanged (checked). init, lr-tasks and sr-register are physlr.py's own functions with formation area2.

  physlr_x2.py derive       --scene S            (CPU, system /usr/bin/python3 -I)
  physlr_x2.py init         --scene S            (unmodified full_initialization.generate, the 4dgs .venv interpreter as area4)
  physlr_x2.py register     --scene S
  physlr_x2.py lr-tasks     --scene S            (pipeline.py rows: author coarse replay, then the gateaff fine stage)
  physlr_x2.py sr-register  --scene S --teacher-tag T --out-tag O [--parent CK]
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import copy
import json
import math
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import physlr as PL  # noqa: E402  (unchanged; importable by name so its _form pickles into the worker processes)
cp, ROOT = PL.cp, PL.ROOT

FORMATION = "area2"
SCALE = 2
PL.FORMATIONS[FORMATION] = dict(scale=SCALE, kind="area", text="LR = 2x2 box (pixel-area) average of the registered HR in "
                                                                 "8-bit sRGB, rounded to 8 bit; no optical prefilter, no noise (pixel binning)")


def scaled_K(K_hr, s):
    """stream.py scaled_K: pixel-centre rule u' = (u + 0.5) / s - 0.5 (the convention of the registered K_lr)."""
    fx, fy, cx, cy = K_hr[0][0] / s, K_hr[1][1] / s, (K_hr[0][2] + 0.5) / s - 0.5, (K_hr[1][2] + 0.5) / s - 0.5
    return [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]


def derive(scene, workers=12):
    f = PL.FORMATIONS[FORMATION]
    source = PL.PREP / scene / "manifest.json"
    original = cp.read(source)
    dest = PL.scene_dir(scene, FORMATION)
    if dest.is_symlink():
        raise ValueError("derived directory must not be a symlink")
    m = copy.deepcopy(original)
    W, H = m["resolutions"]["hr"]
    if W % SCALE or H % SCALE:
        raise ValueError("HR size not divisible by the scale")
    # the registered x4 K_lr must follow the same pixel-centre rule (checked on the original before rescaling)
    for c, cam in original["cameras"].items():
        ref = scaled_K(cam["K_hr"], int(original["scale"]))
        if max(abs(a - b) for ra, rb in zip(ref, cam["K_lr"]) for a, b in zip(ra, rb)) > 1e-6:
            raise ValueError(f"{c}: registered K_lr does not follow the pixel-centre rule")
    m["scale"] = SCALE
    m["resolutions"] = dict(hr=[W, H], lr=[W // SCALE, H // SCALE])
    for cam in m["cameras"].values():
        cam["K_lr"] = scaled_K(cam["K_hr"], SCALE)
    rows = m["observations"]
    for r in rows:
        if not (PL.PREP / scene / r["hr_path"]).exists():
            raise ValueError(f"missing HR {r['hr_path']}")
    jobs = [(PL.PREP / scene / r["hr_path"], dest / r["lr_path"], f["kind"], SCALE) for r in rows]
    with cf.ProcessPoolExecutor(workers) as ex:
        shas = list(ex.map(PL._form, jobs, chunksize=16))
    for r, s in zip(rows, shas):
        r["lr_sha256"] = s
    links = 0
    for r in rows:
        src, dst = PL.PREP / scene / r["hr_path"], dest / r["hr_path"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            if not os.path.samefile(src, dst):
                raise ValueError(f"different HR present: {dst}")
            continue
        os.link(src, dst)
        links += 1
    for r in rows[:: max(1, len(rows) // 60)]:
        if cp.sha(dest / r["hr_path"]) != r["hr_sha256"]:
            raise ValueError(f"HR hash differs: {r['hr_path']}")
    m["physical_lr_derivation"] = dict(
        schema="physical_lr_v1", formation=FORMATION, scale=SCALE, description=f["text"],
        original_manifest=cp.entry(source), generation_source=cp.entry(Path(__file__)), wrapped_source=cp.entry(HERE / "physlr.py"),
        modified_original_fields=["scale", "resolutions.lr", "cameras.*.K_lr",
                                  "observations[*].lr_sha256 (LR pixels re-formed from the registered HR)"],
        hr_layout="hard links to the original HR images", splits_unchanged=True,
        purpose="x2 half of the main table (user 2026-10-24): physically sampled LR at scale 2, frozen x4 recipe")
    cp.write_new(dest / "manifest.json", m)
    receipt = dict(scene=scene, formation=FORMATION, scale=SCALE, manifest=cp.entry(dest / "manifest.json"),
                   hr_links_created=links, observations=len(rows))
    cp.write_new(dest / "derivation_receipt.json", receipt)
    return receipt


def register(scene, seed):
    """physlr.register (itself compress.register) with resolution_LR set from the derived manifest (as stream.register)."""
    root = PL.scene_dir(scene, FORMATION)
    original_path = cp.ORIGINAL_PREFIX / scene / f"seed_{seed}" / "config.json"
    original = cp.read(original_path)
    m = cp.read(root / f"manifest_train_ready_seed{seed}.json")
    if int(m["scale"]) != SCALE or m["resolutions"]["lr"] != [m["resolutions"]["hr"][0] // SCALE, m["resolutions"]["hr"][1] // SCALE]:
        raise ValueError("manifest is not the x2 derivation")
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
                train_camera_center=original["data"]["train_camera_center"], cameras_extent=original["data"]["cameras_extent"],
                resolution_LR=m["resolutions"]["lr"])
    if "training_LR_bytes_SHA_verified" in data:
        data["training_LR_bytes_SHA_verified"] = len(inputs)
    degradation = f"physical LR formation {FORMATION}: {PL.FORMATIONS[FORMATION]['text']}"
    if "domain_protocol" in data:
        data["domain_protocol"] = dict(data["domain_protocol"], seed=seed, degradation=degradation)
    else:
        data["degradation"] = degradation
    plan = copy.deepcopy(original)
    plan["status"] = f"registered_{FORMATION}_LR_parent_pending_GPU"
    plan["data"] = data
    plan["author_adaptations"] = original["author_adaptations"] + [
        f"Physical-LR x2 (2026-10-24): training LR re-formed as {FORMATION} from the registered HR; configuration unchanged."]
    plan["physical_lr"] = dict(formation=FORMATION, scale=SCALE, derivation=cp.entry(root / "derivation_receipt.json"),
                               original_plan=cp.entry(original_path), registry_source=cp.entry(Path(__file__)))
    if plan["configuration"] != original["configuration"]:
        raise ValueError("configuration changed")
    cp.write_new(PL.plan_dir(scene, FORMATION, seed) / "config.json", plan)
    return dict(plan=str(PL.plan_dir(scene, FORMATION, seed).relative_to(ROOT)), train_LR=len(inputs))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=("derive", "init", "register", "lr-tasks", "sr-register"))
    p.add_argument("--scene", choices=sorted(PL.SHORT), required=True)
    p.add_argument("--seed", type=int, default=20261007)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--teacher-tag", default="sr_swinir_x2_areaft")
    p.add_argument("--parent", help="sr-register: LR parent checkpoint (project-relative)")
    p.add_argument("--out-tag", default="areaft_teacher")
    a = p.parse_args()
    if a.mode == "derive":
        r = derive(a.scene, a.workers)
    elif a.mode == "init":
        r = PL.init(a.scene, FORMATION, a.seed)
    elif a.mode == "register":
        r = register(a.scene, a.seed)
    elif a.mode == "lr-tasks":
        r = PL.lr_tasks(a.scene, FORMATION, a.seed)
    else:
        parent = a.parent or str((PL.LR_NS / "runs" / f"{PL.run_tag(a.scene, FORMATION, a.seed)}_gateaff" / "checkpoint_fine_14000.pt").relative_to(ROOT))
        r = PL.sr_register(a.scene, FORMATION, a.seed, a.teacher_tag, parent, a.out_tag)
    print(json.dumps(r, ensure_ascii=False, default=str)[:3000])


if __name__ == "__main__":
    main()
