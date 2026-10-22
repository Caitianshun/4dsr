"""Check the held six-method suite using metadata and byte hashes on CPU.

This is a contract check, not native rendering, metric execution, image-quality
acceptance, CUDA compatibility acceptance, or permission to start future work.
Temporary index fixtures deliberately contain undecoded non-PNG bytes: accepting
their identities proves only the documented metadata/hash validator boundary.
"""
from __future__ import annotations

import argparse
import builtins
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import runpy
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
METHODS = {"wu", "yang", "4c4d", "multi4d", "localdygs", "cem"}
BLOCKED = {"torch", "numpy", "lpips"}
TRAIN = list(range(0, 300, 10))
TEST = list(range(300))


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(2 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    tmp.replace(path)


def path_of(value):
    value = Path(value)
    return value.resolve() if value.is_absolute() else (ROOT / value).resolve()


def load(name, path):
    specification = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


class NoScientificImports:
    def __init__(self):
        self.attempted = []

    def __enter__(self):
        present = sorted(BLOCKED & set(sys.modules))
        if present:
            raise RuntimeError("Preflight must start without scientific libraries: " + str(present))
        self.original = builtins.__import__

        def checked(name, *args, **kwargs):
            if name.split(".", 1)[0] in BLOCKED:
                self.attempted.append(name)
                raise ImportError("CPU preflight blocks " + name)
            return self.original(name, *args, **kwargs)

        builtins.__import__ = checked
        return self

    def __exit__(self, *unused):
        builtins.__import__ = self.original


def cli(path, arguments):
    previous = sys.argv
    stdout = io.StringIO()
    try:
        sys.argv = [str(path), *map(str, arguments)]
        with contextlib.redirect_stdout(stdout):
            runpy.run_path(str(path), run_name="__main__")
    finally:
        sys.argv = previous
    return stdout.getvalue().strip()


def must_reject(name, action):
    try:
        action()
    except (ValueError, FileNotFoundError, FileExistsError, KeyError, RuntimeError, AssertionError) as error:
        return dict(case=name, status="rejected_as_required", exception=type(error).__name__, message=str(error))
    raise AssertionError("Invalid fixture was accepted: " + name)


def source_identities():
    names = ["prepare_suite.py", "score_views.py", "render_wu_multi.py", "render_strict4d.py",
             "render_local_cem.py", "prepare_local_cem.py", "cpu_preflight.py"]
    return {str(HERE / name): sha(HERE / name) for name in names}


def audit_scenes(suite_module, suite):
    if len(suite["scenes"]) != 6:
        raise AssertionError("Exactly six registered scenes required")
    records, manifests = [], {}
    for scene in suite["scenes"]:
        mp = path_of(scene["manifest"])
        manifest = read(mp)
        cameras = manifest["splits"]["train"]
        assert manifest["scene"] == scene["scene"]
        assert not manifest.get("physical_lr_derivation")
        assert manifest["scale"] == 4 and manifest["frame_indices"] == TEST
        assert manifest["splits"]["test"] == ["cam00"] and not manifest["splits"]["dev"]
        assert set(cameras) == set(manifest["cameras"]) - {"cam00"}
        assert len(cameras) == len(set(cameras)) and "cam01" in cameras
        assert "bicubic" in manifest["official_policy"]["degradation"].lower()
        assert sha(mp) == scene["manifest_sha256"]
        keys = [(row["camera_id"], row["frame_index"]) for row in manifest["observations"]]
        assert len(keys) == len(set(keys)) == (len(cameras) + 1) * 300
        assert set(keys) == {(camera, frame) for camera in [*cameras, "cam00"] for frame in TEST}
        for row in manifest["observations"]:
            assert row["split"] == ("test" if row["camera_id"] == "cam00" else "train")
            assert abs(row["time"] - row["frame_index"] / 300) < 1e-12
        for scope in ("train", "test"):
            selected = suite_module.select_observations(manifest, scope)
            assert len(selected) == (len(cameras) * 30 if scope == "train" else 300)
            assert sorted({row["frame_index"] for row in selected}) == (TRAIN if scope == "train" else TEST)
        records.append(dict(scene=manifest["scene"], manifest=str(mp), manifest_sha256=sha(mp),
                            train_cameras=cameras, train_camera_count=len(cameras),
                            training_LR_observations=len(cameras) * 300,
                            fixed_training_HR_observations=len(cameras) * 30, test_HR_observations=300,
                            physical_lr_derivation=False))
        manifests[manifest["scene"]] = (mp, manifest)
    assert sum(row["train_camera_count"] for row in records) == 100
    return records, manifests


def discover_plans():
    found = set(HERE.glob("plans/*/*/*/expected_views.json"))
    strict = read(HERE / "notes/strict4d_CPU_static_acceptance.json")
    found.update(path_of(row["expected_views"]) for row in strict["records"])
    local = read(HERE / "notes/local_cem_cpu_acceptance.json")
    found.update(path_of(row["expected_views"]["path"]) for row in local["plans"])
    return sorted(path.resolve() for path in found)


def audit_plan(path, manifests):
    plan = read(path)
    key = plan["method"], plan["scene"], plan["scope"]
    assert key[0] in METHODS and key[2] in ("train", "test")
    mp, manifest = manifests[key[1]]
    frames = TRAIN if key[2] == "train" else TEST
    cameras = sorted(manifest["splits"][key[2]])
    records = plan["records"]
    expected = [(camera, frame) for camera in cameras for frame in frames]
    assert [(row["camera_id"], row["frame"]) for row in records] == expected
    assert all(row["path"] == f"{row['camera_id']}/{row['frame']:04d}.png" for row in records)
    assert plan["manifest_sha256"] == sha(mp)
    assert plan["resolution"] == manifest["resolutions"]["hr"]
    assert plan["input_representation"] == "uint8_RGB_PNG"
    assert plan["render_camera_transform"] == "identity"
    assert not (path.parent / "render_index.json").exists(), "Expected-only plans cannot claim finished renders"
    if "frames" in plan:
        assert plan["frames"] == frames
    if "cameras" in plan:
        assert sorted(plan["cameras"]) == cameras
    if key[0] in ("yang", "4c4d"):
        assert all(abs(row["native_time"] - row["frame"] / 30) < 1e-12 for row in records)
    else:
        assert not plan.get("HR_reference_pixels_read", False)
    return key, plan, dict(method=key[0], scene=key[1], scope=key[2], views=len(records),
                           plan=str(path), sha256=sha(path), no_completed_render_index=True)


def recreate_plan(path, plan, destination, manifest_path):
    method = plan["method"]
    if method in ("wu", "multi4d"):
        args = ["--spec", plan["spec"]["path"], "--scope", plan["scope"], "--out", destination, "--plan-only"]
        renderer = HERE / "render_wu_multi.py"
    elif method in ("yang", "4c4d"):
        args = ["--project-root", ROOT, "--method", method, "--runtime", plan["runtime"],
                "--dataset", plan["dataset"], "--manifest", manifest_path,
                "--checkpoint", plan["checkpoint"], "--config", plan["configuration"],
                "--scope", plan["scope"], "--output", destination, "--plan-only"]
        if plan.get("expected_checkpoint_sha256"):
            args += ["--checkpoint-sha256", plan["expected_checkpoint_sha256"]]
        renderer = HERE / "render_strict4d.py"
    else:
        args = ["--method", method, "--manifest", manifest_path, "--runtime", plan["runtime"],
                "--model", plan["model"], "--source", plan["source"], "--scope", plan["scope"],
                "--iteration", plan["iteration"], "--output", destination, "--plan-only"]
        if plan.get("config"):
            args += ["--config", path_of(plan["config"]["path"])]
        renderer = HERE / "render_local_cem.py"
    cli(renderer, args)
    fresh = read(destination / "expected_views.json")
    assert [(row["camera_id"], row["frame"], row["path"]) for row in fresh["records"]] == [
        (row["camera_id"], row["frame"], row["path"]) for row in plan["records"]]
    assert fresh["manifest_sha256"] == plan["manifest_sha256"]
    assert not (destination / "render_index.json").exists()


def index_fixtures(directory, scorer, selector):
    manifest = dict(scene="CPU_contract_fixture", frame_indices=TEST, splits=dict(train=["cam01", "cam02"], test=["cam00"], dev=[]),
                    resolutions=dict(hr=[2, 2], lr=[1, 1]), observations=[
                        dict(camera_id=camera, frame_index=frame, time=frame / 300,
                             split="test" if camera == "cam00" else "train")
                        for camera in ("cam00", "cam01", "cam02") for frame in TEST])
    mp = directory / "manifest.json"
    dump(mp, manifest)
    render = directory / "renders"
    payload = b"CPU fixture identity only; deliberately not decoded as an image.\n"
    digest = hashlib.sha256(payload).hexdigest()
    base = dict(schema="dynamic_lr_multiview_render_v1", manifest_sha256=sha(mp), scope="train",
                resolution=[2, 2], input_representation="uint8_RGB_PNG", render_camera_transform="identity", records=[])
    for row in selector.select_observations(manifest, "train"):
        name = f"{row['camera_id']}/{row['frame_index']:04d}.png"
        path = render / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        base["records"].append(dict(camera_id=row["camera_id"], frame=row["frame_index"], path=name, render_sha256=digest))
    index = render / "render_index.json"

    def validate(value):
        dump(index, value)
        return scorer.validate_index(index, mp, "train", render)

    assert len(validate(base)[2]) == 60
    cases = []
    mutations = {
        "wrong_manifest_hash": lambda value: value.update(manifest_sha256="0" * 64),
        "wrong_scope": lambda value: value.update(scope="test"),
        "wrong_schema": lambda value: value.update(schema="expected_views_only"),
        "wrong_resolution": lambda value: value.update(resolution=[4, 4]),
        "floating_point_representation": lambda value: value.update(input_representation="float32"),
        "fitted_camera_transform": lambda value: value.update(render_camera_transform="warp"),
        "duplicate_camera_frame": lambda value: value["records"].append(copy.deepcopy(value["records"][0])),
        "missing_view": lambda value: value["records"].pop(),
        "wrong_camera": lambda value: value["records"][0].update(camera_id="cam99"),
        "modified_index_hash": lambda value: value["records"][0].update(render_sha256="0" * 64),
        "parent_path_escape": lambda value: value["records"][0].update(path="../outside.png"),
        "absolute_path_escape": lambda value: value["records"][0].update(path=str(directory / "outside.png")),
    }
    (directory / "outside.png").write_bytes(payload)
    for name, mutate in mutations.items():
        bad = copy.deepcopy(base)
        mutate(bad)
        cases.append(must_reject(name, lambda bad=bad: validate(bad)))
    link = render / "outside_link.png"
    link.symlink_to(directory / "outside.png")
    bad = copy.deepcopy(base)
    bad["records"][0]["path"] = link.name
    cases.append(must_reject("symlink_path_escape", lambda: validate(bad)))
    first = render / base["records"][0]["path"]
    first.write_bytes(payload + b"changed")
    cases.append(must_reject("modified_file_bytes", lambda: validate(base)))
    first.write_bytes(payload)
    assert len(validate(base)[2]) == 60
    cli(HERE / "score_views.py", ["--manifest", mp, "--scope", "train", "--render-dir", render,
                                  "--out", directory / "unused_score", "--method", "fixture", "--check-index"])
    bad_manifest = copy.deepcopy(manifest)
    bad_manifest["physical_lr_derivation"] = {"formation": "area4"}
    cases.append(must_reject("area4_manifest", lambda: selector.select_observations(bad_manifest, "train")))
    bad_manifest = copy.deepcopy(manifest)
    bad_manifest["observations"].append(copy.deepcopy(bad_manifest["observations"][0]))
    cases.append(must_reject("duplicate_selected_manifest_observation", lambda: selector.select_observations(bad_manifest, "test")))
    bad_manifest = copy.deepcopy(manifest)
    bad_manifest["observations"][0]["time"] = 1.0
    cases.append(must_reject("wrong_registered_time", lambda: selector.select_observations(bad_manifest, "test")))
    return dict(positive_index_views=60, index_check_cli_no_scoring=True,
                fixtures_are_not_decoded_RGB=True, rejected_cases=cases)


def output_fixtures(directory, scorer):
    expected = dict(manifest_sha256="1" * 64, scope="train", method="fixture",
                    render_dir=str(directory / "renders"), images=60, metric_helper_sha256="2" * 64)
    scorer.validate_output_directory(directory / "absent", expected)
    valid = directory / "valid_output"
    dump(valid / "score_plan.json", expected)
    scorer.validate_output_directory(valid, expected)
    cases = []
    for key in expected:
        path = directory / ("bad_" + key)
        value = dict(expected)
        value[key] = "mismatched_value"
        dump(path / "score_plan.json", value)
        cases.append(must_reject("score_plan_wrong_" + key, lambda path=path: scorer.validate_output_directory(path, expected)))
    empty = directory / "empty_existing"
    empty.mkdir()
    cases.append(must_reject("existing_empty_directory", lambda: scorer.validate_output_directory(empty, expected)))
    finished = directory / "finished_output"
    dump(finished / "score_plan.json", expected)
    dump(finished / "summary.json", dict(status="completed"))
    cases.append(must_reject("existing_metric_results", lambda: scorer.validate_output_directory(finished, expected)))
    return dict(absent_output_accepted=True, exact_sole_score_plan_accepted=True, rejected_cases=cases)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=ROOT / "output/dynamic_lr_expansion_20261022/preparation/suite_plan.json")
    parser.add_argument("--out", type=Path, default=HERE / "notes/cpu_preflight.json")
    args = parser.parse_args()
    before = source_identities()
    result = dict(schema="dynamic_lr_CPU_preflight_v1", status="running", GPU_calls=0,
                  remote_calls=0, training_calls=0, parameter_updates=0,
                  scope="metadata_and_byte_hash_contracts_only",
                  limitation="No model/CUDA/image quality/PSNR/SSIM/LPIPS acceptance; no RGB decoding and no execution authorization.")
    sys.path.insert(0, str(HERE))
    try:
        with NoScientificImports() as guard:
            selector = load("prepare_suite", HERE / "prepare_suite.py")
            scorer = load("cpu_preflight_scorer", HERE / "score_views.py")
            suite = read(args.suite)
            assert suite["execution_enabled"] is False and suite["GPU_work_started"] == 0
            assert suite["watchers_created"] is False and suite["remote_jobs_submitted"] == 0
            scenes, manifests = audit_scenes(selector, suite)
            result["scenes"] = scenes
            result["training_camera_instances"] = sum(row["train_camera_count"] for row in scenes)
            paths = discover_plans()
            assert len(paths) == 72, "Expected 6 methods x 6 scenes x 2 scopes"
            audits = [audit_plan(path, manifests) for path in paths]
            keys = [row[0] for row in audits]
            assert len(set(keys)) == 72
            assert set(keys) == {(method, scene, scope) for method in METHODS for scene in manifests for scope in ("train", "test")}
            result["render_plans"] = [row[2] for row in audits]
            result["render_plan_count"] = len(audits)
            result["planned_view_count"] = sum(row[2]["views"] for row in audits)
            temporary_parent = ROOT / "output/dynamic_lr_expansion_20261022"
            temporary_parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix="cpu_preflight_", dir=temporary_parent) as temp:
                directory = Path(temp)
                cli(HERE / "prepare_suite.py", ["--out", directory / "fresh_suite", "--verify", "none"])
                fresh_suite = read(directory / "fresh_suite/suite_plan.json")
                for key, expected in (("task_count", 36), ("reuse_training_count", 16), ("missing_training_count", 20),
                                      ("reuse_test_metrics_count", 12), ("missing_test_metrics_count", 24), ("missing_train_metrics_count", 36)):
                    assert fresh_suite[key] == expected
                for scene, (mp, manifest) in manifests.items():
                    scene_dir = directory / "fresh_suite/scenes" / scene
                    allowlist = read(scene_dir / "training_inputs.json")
                    assert allowlist["HR_pixel_inputs"] is False and allowlist["cam00_pixel_inputs"] is False
                    expected_files = {str((mp.parent / row["lr_path"]).relative_to(ROOT)): row["lr_sha256"]
                                      for row in manifest["observations"] if row["split"] == "train"}
                    assert set(allowlist["files"]) == set(expected_files)
                    for name, metadata in allowlist["files"].items():
                        assert metadata["sha256"] == expected_files[name]
                        assert metadata["role"] == "training_LR_only"
                        assert "/lr/" in name and "/hr/" not in name and "/cam00/" not in name
                    for scope in ("train", "test"):
                        frozen = read(scene_dir / (scope + "_views.json"))
                        chosen = selector.select_observations(manifest, scope)
                        assert frozen["HR_use"] == "scoring_after_prediction_only"
                        assert [(row["camera_id"], row["frame"]) for row in frozen["records"]] == [
                            (row["camera_id"], row["frame_index"]) for row in chosen]
                        assert [row["reference_sha256"] for row in frozen["records"]] == [row["hr_sha256"] for row in chosen]
                result["prepare_suite_CLI_metadata_check"] = dict(
                    passed=True, verification_mode="none; no image bytes decoded", tasks=36,
                    exact_train_LR_allowlist_images=30000, HR_or_cam00_training_files=0,
                    exact_HR_scoring_view_identities=4800)
                for number, (key, plan, unused) in enumerate(audits):
                    mp = manifests[key[1]][0]
                    recreate_plan(paths[number], plan, directory / "render_plans" / str(number), mp)
                    cli(HERE / "score_views.py", ["--manifest", mp, "--scope", key[2], "--render-dir", directory / "future_renders" / str(number),
                                                  "--out", directory / "score_plans" / str(number), "--method", key[0], "--plan-only"])
                    scored_plan = read(directory / "score_plans" / str(number) / "score_plan.json")
                    assert scored_plan["images"] == len(plan["records"])
                    assert scored_plan["frames"] == (TRAIN if key[2] == "train" else TEST)
                result["all_renderer_plan_only_CLI_checks"] = 72
                result["all_scorer_plan_only_CLI_checks"] = 72
                result["index_fixture_checks"] = index_fixtures(directory / "indices", scorer, selector)
                result["score_output_directory_checks"] = output_fixtures(directory / "outputs", scorer)
            result["plan_only_import_guard"] = dict(blocked_modules=sorted(BLOCKED), attempted_imports=guard.attempted,
                                                    imported_forbidden_modules=sorted(BLOCKED & set(sys.modules)))
            assert not guard.attempted and not (BLOCKED & set(sys.modules))
        after = source_identities()
        if before != after:
            raise RuntimeError("A reviewed source changed during preflight; rerun after source updates settle")
        result.update(status="passed_CPU_metadata_hash_contracts_only", source_identities=after)
    except BaseException as error:
        result.update(status="failed_CPU_preflight", error_type=type(error).__name__, error=str(error), source_identities=source_identities())
        dump(args.out, result)
        raise
    dump(args.out, result)
    print(json.dumps({key: result[key] for key in ("status", "render_plan_count", "planned_view_count", "training_camera_instances")}))


if __name__ == "__main__":
    main()
