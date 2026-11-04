"""Scoped helpers; no tensor imports, dispatch, or writes to earlier tasks."""
from __future__ import annotations

import ast
from contextlib import contextmanager
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
CODE = Path(__file__).resolve().parent
ROOT = CODE.parents[2]
OUT = ROOT / "output/dynamic_sr_modules_20261010/newscenes_sched_20261103"
REG = ROOT / "experiments/dynamic_sr_modules_20261010/newscenes_reg_20261103"
REGOUT = ROOT / "output/dynamic_sr_modules_20261010/newscenes_reg_20261103"
FP = ROOT / "experiments/dynamic_sr_multiview_footprint_20261007"
FPOUT = ROOT / "output/dynamic_sr_multiview_footprint_20261007"
PREP = ROOT / "data/dynamic_sr/full_time_20261007/prepared"
RUNTIME = ROOT / "third_party/4dgs_843d5ac_runtime"
TRAINPY = "/home/cai_tianshun/miniconda3/envs/mml_bw/bin/python"
SEED = 20261007
SCENES = {"sear_steak": "Sear07", "flame_salmon": "Salmon07", "flame_steak": "Flame07"}
TAG = "flame_steak_seed20261007_local_RTX3090_original_v1_20261009"
HIST_PLAN = FPOUT / f"held_confirmation_parent_coarse_overlap/{TAG}/CPU_registered_plan_v1/registered_plan.json"
HIST_INDEX = FPOUT / f"held_confirmation_parent_coarse_overlap/{TAG}/native_original_v1/index.json"
HIST_SCHEDULE = FPOUT / f"held_confirmation_native_schedules/{TAG}/schedule_A_original_v1/schedule.json"
HIST_BSYNC = FPOUT / "held_confirmation_materialized/flame_steak/seed20261007/Bsync/local_RTX3090_original_v1_20261009/scientific_materialization/scientific_plan.json"
PINS = {
    "full_held_parent_overlap.py": "6352f3f3597abbc54ac20d99e6adcd675eb8f75ed07f63a0893e5ad1927cd643",
    "full_held_native_schedule_materializer.py": "92790ce30ffa3942ca121ad934fc4b3f37de6931061abd8721b333b1a825fa2e",
    "full_refine_registered_v2.py": "c278928f56b0515adedf15eef9a30850e8bd2a4c41e6f6e0f828068ca774b75d",
    "full_prefix_registered.py": "92a19ed94c2bf7dab881341375c4bdc486e6f4032536e977592c7911c3ce6605",
    "full_native_protocol.py": "cb4773700bbf535ebb0d9cc94c51f06ca457e04bf305d65ef9afe911746d6713",
    "footprint.py": "3c2645449d065e032fef38cefd1e269ec971e86735a86ef28aefc1677819c789",
}
for folder in (REG, FP, CODE):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))


def require(ok, message):
    if not ok:
        raise ValueError(message)


def local(path):
    p = Path(path)
    p = p if p.is_absolute() else ROOT / p
    require(p.resolve().is_relative_to(ROOT), "Path outside project")
    require(not any(x.is_symlink() for x in (p, *p.parents)), "Symlink refused")
    return p


def rel(path):
    return str(local(path).relative_to(ROOT))


def read(path):
    return json.loads(local(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with local(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def entry(path):
    return dict(path=rel(path), sha256=sha(path))


def bound(ref):
    require(isinstance(ref, dict) and {"path", "sha256"} <= set(ref), "Missing SHA entry")
    p = local(ref["path"])
    require(sha(p) == ref["sha256"], "Bound SHA changed: " + rel(p))
    if "bytes" in ref:
        require(p.stat().st_size == ref["bytes"], "Bound byte count changed")
    return p


def output_path(path):
    p = local(path)
    require(p.is_relative_to(OUT) or p.is_relative_to(CODE), "Write outside this task")
    return p


def text_new(path, text):
    p = output_path(path)
    require(not p.exists(), "Existing file preserved: " + rel(p))
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("x") as f:
        f.write(text)
    return p


def write_new(path, value, *, sorted_keys=False):
    return text_new(path, json.dumps(value, ensure_ascii=False, sort_keys=sorted_keys,
                                    indent=2, allow_nan=False) + "\n")


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, local(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@contextmanager
def attributes(module, **values):
    previous = {key: getattr(module, key) for key in values}
    try:
        for key, value in values.items():
            setattr(module, key, value)
        yield module
    finally:
        for key, value in previous.items():
            setattr(module, key, value)


def science_modules():
    for name, expected in PINS.items():
        require(sha(FP / name) == expected, "Frozen source changed: " + name)
    import full_held_parent_overlap as overlap
    import full_held_native_schedule_materializer as schedule
    import full_refine_registered_v2 as refine
    return overlap, schedule, refine


def historical_sum(values):
    # Explicit Python3.10 accumulation, used only for historical geometry replay.
    result = 0
    for value in values:
        result += value
    return result


def manifest_metadata(path):
    """Only readable manifest bytes; never follow initialization/image payloads."""
    overlap, _, _ = science_modules()
    m = read(path)
    require(m["scene"] in SCENES and m["scale"] == 4, "Unregistered scene/scale")
    expected = [f"cam{i:02d}" for i in range(21)
                if not (m["scene"] == "flame_salmon" and i in (3, 17))]
    require(set(m["cameras"]) == set(expected) and m["splits"]["train"] == expected[1:],
            "Real camera names/population differ")
    overlap.legal_rows(m)
    require(m["resolutions"] == {"lr": [336, 252], "hr": [1344, 1008]}, "Grid differs")
    original = read(bound(m["original_full_manifest"]))
    for key in ("scene", "source", "scale", "role", "frame_indices", "resolutions",
                "cameras", "splits", "observations"):
        require(m[key] == original[key], "Original manifest differs: " + key)
    for c in m["cameras"].values():
        require(all(math.isfinite(float(v)) for name in ("c2w", "w2c", "K_lr", "K_hr")
                    for row in c[name] for v in row), "Nonfinite camera")
    rows = [row for row in m["observations"] if row["split"] == "train"]
    inputs = []
    for row in rows:
        relative = Path("lr") / row["camera_id"] / f"{int(row['frame_index']):04d}.png"
        require(Path(row["lr_path"]) == relative, "Foreign LR path")
        inputs.append(dict(camera=row["camera_id"], frame=int(row["frame_index"]),
                           path=rel(local(path).parent / relative), sha256=row["lr_sha256"]))
    centers = [[float(m["cameras"][c]["c2w"][i][3]) for i in range(3)]
               for c in m["splits"]["train"]]
    center = [historical_sum(c[i] for c in centers) / len(centers) for i in range(3)]
    extent = 1.1 * max(math.sqrt(historical_sum((c[i] - center[i]) ** 2 for i in range(3)))
                       for c in centers)
    return m, inputs, center, extent


def differences(old, new, prefix=""):
    if isinstance(old, dict) and isinstance(new, dict):
        result = []
        for key in sorted(old.keys() | new.keys()):
            p = prefix + "/" + str(key)
            if key not in old or key not in new:
                result.append(dict(field=p, old=old.get(key), new=new.get(key)))
            else:
                result.extend(differences(old[key], new[key], p))
        return result
    if type(old) is not type(new) or old != new:
        return [dict(field=prefix, old=old, new=new)]
    return []


def extract_plan_expression(environment):
    """Execute the original constructor expression, without cloning its source."""
    p = FP / "full_held_parent_overlap.py"
    require(sha(p) == PINS[p.name], "Plan constructor changed")
    tree = ast.parse(p.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "build_plan")
    node = next(n for n in fn.body if isinstance(n, ast.Assign)
                and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "plan")
    namespace = dict(environment)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(p), "exec"), namespace)
    return namespace["plan"]


def no_torch():
    require(not any(k in sys.modules for k in ("torch", "scene", "gaussian_renderer")),
            "CPU phase imported tensor/model")
