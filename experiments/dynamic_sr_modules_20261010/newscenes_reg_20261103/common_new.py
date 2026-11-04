"""CPU metadata utilities. All writes are new, project-relative and scoped."""
from __future__ import annotations

import ast
import contextlib
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
CODE = Path(__file__).resolve().parent
ROOT = CODE.parents[2]
OUT = ROOT / "output/dynamic_sr_modules_20261010/newscenes_reg_20261103"
FP = ROOT / "experiments/dynamic_sr_multiview_footprint_20261007"
PHYS = ROOT / "experiments/dynamic_sr_physlr_20261016"
PREP = ROOT / "data/dynamic_sr/full_time_20261007/prepared"
RUNTIME = ROOT / "third_party/4dgs_843d5ac_runtime"
SNAPSHOT = ROOT / "tmp/claude_codex_materials_manifest_newscene_recipe_20261102"
SEED = 20261007
NEW = {"sear_steak": "Sear", "flame_salmon": "Salmon"}
PIN = "843d5ac636c37e4b611242287754f3d4ed150144"
assert ROOT.name.lower() == "4dsr" and CODE.parent.name == "dynamic_sr_modules_20261010", (
    "Check ROOT derivation before using a relocated adapter")
for folder in (CODE, FP, PHYS):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def local(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def rel(path):
    return str(local(path).relative_to(ROOT))


def read(path):
    return json.loads(local(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with local(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


def entry(path):
    return {"path": rel(path), "sha256": sha(path)}


def bound(reference):
    path = local(reference["path"])
    require(sha(path) == reference["sha256"], "SHA mismatch: " + rel(path))
    return path


def output_path(path):
    path = local(path)
    require(path.resolve().is_relative_to(OUT.resolve()) or
            path.resolve().is_relative_to(CODE.resolve()), "Write outside authorized namespace")
    require(not any(p.is_symlink() for p in (path, *path.parents)), "Symlink output refused")
    return path


def write_new(path, value):
    path = output_path(path)
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    return text_new(path, text)


def text_new(path, text):
    path = output_path(path)
    if path.exists():
        require(path.read_text() == text, "Existing file preserved: " + rel(path))
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(text)
    return path


def geometry_equal(left, right, tolerance=1e-12):
    """Compare recomputed floating geometry without replacing it with a fixture."""
    return (len(left["train_camera_center"]) == len(right["train_camera_center"]) == 3
            and max(abs(a - b) for a, b in zip(left["train_camera_center"],
                                              right["train_camera_center"])) <= tolerance
            and abs(left["cameras_extent"] - right["cameras_extent"]) <= tolerance)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, local(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@contextlib.contextmanager
def attributes(module, **values):
    previous = {key: getattr(module, key) for key in values}
    try:
        for key, value in values.items():
            setattr(module, key, value)
        yield module
    finally:
        for key, value in previous.items():
            setattr(module, key, value)


def pure_function(path, name, globals_):
    """Extract a named CPU function unchanged, without executing module imports."""
    tree = ast.parse(local(path).read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    namespace = dict(globals_)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


def differences(old, new, prefix=""):
    if isinstance(old, dict) and isinstance(new, dict):
        result = []
        for key in sorted(old.keys() | new.keys()):
            path = prefix + "/" + str(key)
            if key not in old or key not in new:
                result.append(dict(field=path, old=old.get(key), new=new.get(key),
                                   reason="field added or removed"))
            else:
                result.extend(differences(old[key], new[key], path))
        return result
    if old == new:
        return []
    return [dict(field=prefix, old=old, new=new)]


def historical_prefix(scene="flame_steak"):
    return ROOT / f"output/dynamic_sr_multiview_footprint_20261007/full_LR_prefixes/{scene}/seed_{SEED}/config.json"


def historical_bsync(scene="flame_steak"):
    return ROOT / ("output/dynamic_sr_multiview_footprint_20261007/held_confirmation_materialized/"
                   f"{scene}/seed{SEED}/Bsync/local_RTX3090_original_v1_20261009/"
                   "scientific_materialization/scientific_plan.json")


def snapshot_file(original):
    original = local(original)
    manifest = read(SNAPSHOT / "manifest.json")
    matches = [row for row in manifest["files"] if row["source"] == str(original)]
    require(len(matches) == 1, "No unique evidence snapshot: " + rel(original))
    row = matches[0]
    path = SNAPSHOT / row["snapshot_relative"]
    require(sha(path) == row["snapshot_sha256"] and not row["source_changed_during_copy"],
            "Changed evidence snapshot")
    return path
