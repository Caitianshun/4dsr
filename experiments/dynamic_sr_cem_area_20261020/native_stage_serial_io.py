"""Run the frozen native HR entry with its existing one-worker image API.

Only Scene.getTrainCameras(..., n_job=1) is forced. The original method,
reader, camera shuffle, training entry and model code remain byte-identical.
The CPU QA mode extracts these pure functions without importing native Scene.
"""
import argparse
import ast
from collections import defaultdict
from datetime import datetime, timezone
import functools
import hashlib
import inspect
import json
import os
from pathlib import Path
import pickle
import random
import runpy
import sys
import time
import traceback
from types import SimpleNamespace


NATIVE_ENTRY_SHA256 = "645d6a6ec5476814de22a04c5c2e8bf834c81ca2e5b7d49a934e550267c2d0c5"
SCENE_SHA256 = "c5e745bcb6d0a6e89030998ef77553546290def1fe97469ae174f1c7979daa04"
SIGNATURE = ("self", "scale", "shuffle", "return_as", "n_job", "return_path", "get_img", "job_batch_size")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def frozen_functions(runtime):
    """Source identities only; this function imports no CUDA library."""
    runtime = Path(runtime).resolve()
    protocol_path = runtime / "protocol_interface_receipt.json"
    protocol = json.loads(protocol_path.read_text())
    assert protocol["version"] == "native_area4_v1"
    assert protocol["core_losses_sampling_optimization_schedule_unchanged"]
    assert protocol["native_readonly_test_evaluation_disabled"]
    assert sha(runtime / "protocol_interface.py") == protocol["interface_sha256"]
    for item in protocol["core_source_files"] + protocol["configurations"]:
        assert sha(runtime / item["path"]) == item["sha256"], item["path"]
    scene = runtime / "scene/__init__.py"
    assert sha(scene) == SCENE_SHA256
    tree = ast.parse(scene.read_text())
    scene_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Scene")
    method = next(node for node in scene_class.body if isinstance(node, ast.FunctionDef) and node.name == "getTrainCameras")
    reader = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "im_reader")
    return dict(scene_path=str(scene), scene_sha256=sha(scene),
                method_ast_sha256=hashlib.sha256(ast.dump(method, include_attributes=False).encode()).hexdigest(),
                reader_ast_sha256=hashlib.sha256(ast.dump(reader, include_attributes=False).encode()).hexdigest(),
                protocol_receipt_path=str(protocol_path), protocol_receipt_sha256=sha(protocol_path)), method, reader


def serial_method(original, on_call=None):
    """Use the author's existing n_job parameter; do not replace its body."""
    signature = inspect.signature(original)
    assert tuple(signature.parameters) == SIGNATURE, str(signature)
    assert signature.parameters["n_job"].default == 2

    @functools.wraps(original)
    def get_train(self, *args, **kwargs):
        assert os.environ.get("CEM_PROTOCOL_TRAIN_LEVEL") == "hr", "Serial IO adaptation is HR-only"
        assert self.lazy_loader is True, "The frozen lazy-loader is required"
        bound = signature.bind(self, *args, **kwargs)
        bound.apply_defaults()
        requested = bound.arguments["n_job"]
        assert requested in (1, 2), "Unexpected native worker request"
        if bound.arguments["get_img"] and not bound.arguments["return_path"]:
            assert bound.arguments["return_as"] == "generator", "HR image loading must stay on demand"
        bound.arguments["n_job"] = 1
        event = {key: bound.arguments[key] for key in SIGNATURE if key != "self"}
        event.update(requested_n_job=requested, forced_n_job=1, timestamp_utc=datetime.now(timezone.utc).isoformat())
        if on_call:
            on_call(dict(event, phase="before_original_call"))
        result = original(*bound.args, **bound.kwargs)
        if on_call:
            on_call(dict(event, phase="after_original_call", camera_count=len(result[0])))
        return result

    return get_train


def run_native(args):
    native_entry = Path(__file__).with_name("native_stage.py").resolve()
    receipt = args.receipt.resolve()
    io_receipt = args.io_receipt.resolve() if args.io_receipt else receipt.with_name(receipt.stem + ".io.json")
    assert not io_receipt.exists(), "Refusing to overwrite IO execution evidence"
    state = dict(schema="CEM_native_HR_serial_io_v1", status="running", workers=1, n_job=1, no_core_modified=True,
                 scope="HR-only ordered on-demand CPU image decoding through original n_job API",
                 wrapper=dict(path=str(Path(__file__).resolve()), sha256=sha(__file__)),
                 native_entry=dict(path=str(native_entry), sha256=sha(native_entry)),
                 native_receipt=str(receipt), calls=[], started_utc=datetime.now(timezone.utc).isoformat())
    write_json(io_receipt, state)
    began = time.monotonic()
    scene_class = original = None
    try:
        assert args.level == "hr" and args.workers == 1, "This wrapper is HR-only and worker-one only"
        assert args.gpu.isdecimal() and os.environ.get("CUDA_VISIBLE_DEVICES") == args.gpu
        assert not receipt.exists(), "Never overwrite a native stage receipt"
        assert sha(native_entry) == NATIVE_ENTRY_SHA256
        identity, _, _ = frozen_functions(args.runtime)
        state["frozen_original_method"] = identity
        write_json(io_receipt, state)
        runtime = args.runtime.resolve()
        native_options = {key: str(getattr(args, key).resolve()) if key in {"runtime", "manifest", "source", "model", "receipt"}
                          else str(getattr(args, key))
                          for key in ["runtime", "manifest", "source", "model", "stage", "level", "gpu", "receipt"]}
        os.chdir(runtime)
        sys.path.insert(0, str(runtime))
        from scene import Scene
        assert Path(inspect.getsourcefile(Scene)).resolve() == runtime / "scene/__init__.py"
        scene_class, original = Scene, Scene.getTrainCameras

        def record(event):
            state["calls"].append(event)
            write_json(io_receipt, state)

        Scene.getTrainCameras = serial_method(original, record)
        sys.argv = [str(native_entry)]
        for option, value in native_options.items():
            sys.argv.extend(["--" + option, value])
        runpy.run_path(str(native_entry), run_name="__main__")
        assert receipt.is_file()
        native_state = json.loads(receipt.read_text())
        assert native_state["status"] == "completed" and native_state["entry_sha256"] == NATIVE_ENTRY_SHA256
        assert any(call["phase"] == "after_original_call" for call in state["calls"])
        state["status"] = "completed"
    except BaseException:
        state.update(status="failed", error=traceback.format_exc())
        raise
    finally:
        if scene_class is not None and original is not None:
            scene_class.getTrainCameras = original
        state.update(finished_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic() - began)
        write_json(io_receipt, state)
        if receipt.is_file():
            native_state = json.loads(receipt.read_text())
            assert native_state["entry_sha256"] == NATIVE_ENTRY_SHA256
            native_state["io_adaptation"] = dict(
                schema=state["schema"], status=state["status"], workers=1, n_job=1, no_core_modified=True,
                receipt=dict(path=str(io_receipt), sha256=sha(io_receipt)),
                wrapper=state["wrapper"], native_entry=state["native_entry"],
                original_method=state.get("frozen_original_method"),
                completed_calls=sum(call["phase"] == "after_original_call" for call in state["calls"]),
                unchanged="Reader pixels, shuffled camera order, sampling, model, losses, RAdam and native clocks")
            write_json(receipt, native_state)


def cpu_qa(args):
    """Finite real-image QA; no native Scene import, model, GPU or dispatch."""
    assert os.environ.get("CUDA_VISIBLE_DEVICES", "") == "", "CPU QA requires an empty CUDA mask"
    assert not args.qa_output.exists(), "Refusing to overwrite QA evidence"
    import numpy as np
    from PIL import Image, ImageFile
    import torch
    import joblib
    from joblib import Parallel, delayed
    assert not torch.cuda.is_initialized()
    assert torch.get_default_dtype() == torch.float32
    identity, method, reader = frozen_functions(args.runtime)
    general_path = args.runtime.resolve() / "utils/general_utils.py"
    general_tree = ast.parse(general_path.read_text())
    pil = next(node for node in general_tree.body if isinstance(node, ast.FunctionDef) and node.name == "PILtoTorch")
    namespace = dict(np=np, torch=torch, Image=Image, ImageFile=ImageFile,
                     random=random, defaultdict=defaultdict, Parallel=Parallel, delayed=delayed)
    for node in [pil, reader, method]:
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(args.runtime / "scene/__init__.py"), "exec"), namespace)
    original = namespace["getTrainCameras"]
    manifest_path = args.manifest.resolve()
    data = json.loads(manifest_path.read_text())
    assert data["physical_lr_derivation"]["formation"] == "area4"
    train_cameras = sorted(data["splits"]["train"])
    chosen = [item for item in data["observations"] if item["camera_id"] in [train_cameras[0], train_cameras[-1]]
              and item["frame_index"] in [0, 150, 299]]
    chosen.sort(key=lambda item: (item["camera_id"], item["frame_index"]))
    assert len(chosen) == 6 and all(item["split"] == "train" for item in chosen)
    cameras = [SimpleNamespace(image_path=str(manifest_path.parent / item["hr_path"]),
                              resolution=tuple(data["resolutions"]["hr"]), im_scale=1.0,
                              timestamp=item["frame_index"], key=f"{item['camera_id']}/{item['frame_index']:04d}") for item in chosen]
    for item, cam in zip(chosen, cameras):
        assert sha(cam.image_path) == item["hr_sha256"]

    class MetadataOnlyScene:
        lazy_loader = True
        def getTCams(self, scale, split):
            assert scale == 1.0 and split == "train"
            return cameras.copy()

    def seed():
        random.seed(20261021)
        np.random.seed(20261021)
        torch.manual_seed(20261021)

    def rng():
        return hashlib.sha256(pickle.dumps((random.getstate(), np.random.get_state(),
                                           torch.random.get_rng_state().numpy().tobytes()))).hexdigest()

    old_level = os.environ.get("CEM_PROTOCOL_TRAIN_LEVEL")
    os.environ["CEM_PROTOCOL_TRAIN_LEVEL"] = "hr"
    record = []
    try:
        seed()
        before = rng()
        base_cams, base_gen = original(MetadataOnlyScene(), shuffle=True, return_as="generator", job_batch_size=1)
        base_after_shuffle = rng()
        base_pixels = list(base_gen)
        base_after_pixels = rng()
        seed()
        assert rng() == before
        adapted = serial_method(original, record.append)
        adapted_cams, adapted_gen = adapted(MetadataOnlyScene(), shuffle=True, return_as="generator", job_batch_size=1)
        adapted_after_shuffle = rng()
        adapted_pixels = list(adapted_gen)
        adapted_after_pixels = rng()
        assert [cam.key for cam in base_cams] == [cam.key for cam in adapted_cams]
        assert base_after_shuffle == adapted_after_shuffle == base_after_pixels == adapted_after_pixels
        assert len(base_pixels) == len(adapted_pixels) == 6
        image_evidence = []
        for cam, baseline, adapted_pixel in zip(base_cams, base_pixels, adapted_pixels):
            assert baseline.dtype == adapted_pixel.dtype == torch.float32
            assert tuple(baseline.shape) == (3, 1008, 1344)
            assert torch.equal(baseline, adapted_pixel)
            image_evidence.append(dict(key=cam.key, path=cam.image_path, source_sha256=sha(cam.image_path),
                                       float32_CHW_sha256=hashlib.sha256(baseline.contiguous().numpy().tobytes()).hexdigest(),
                                       bitwise_equal=True))
        del base_pixels, adapted_pixels
        counter = dict(task_iterator_advanced=0, decoded=0)
        native_reader = namespace["im_reader"]

        def counted_reader(*values):
            counter["decoded"] += 1
            return native_reader(*values)

        class ObservedParallel:
            def __init__(self, *values, **kwargs):
                assert kwargs["n_jobs"] == 1
                self.parallel = Parallel(*values, **kwargs)
            def __call__(self, iterable):
                def counted():
                    for task in iterable:
                        counter["task_iterator_advanced"] += 1
                        yield task
                return self.parallel(counted())

        namespace["im_reader"], namespace["Parallel"] = counted_reader, ObservedParallel
        _, demand = adapted(MetadataOnlyScene(), shuffle=True, return_as="generator", job_batch_size=1)
        assert counter == dict(task_iterator_advanced=0, decoded=0)
        next(demand)
        assert counter == dict(task_iterator_advanced=1, decoded=1)
        time.sleep(0.05)
        assert counter == dict(task_iterator_advanced=1, decoded=1)
        demand.close()
        assert not torch.cuda.is_initialized()
        qa = dict(schema="CEM_native_HR_serial_io_CPU_QA_v1", status="passed", CUDA_initialized=False,
                  wrapper=dict(path=str(Path(__file__).resolve()), sha256=sha(__file__)), frozen_original_method=identity,
                  native_entry_sha256=sha(Path(__file__).with_name("native_stage.py")),
                  manifest=dict(path=str(manifest_path), sha256=sha(manifest_path)),
                  python_executable=sys.executable, torch_version=torch.__version__, joblib_version=joblib.__version__,
                  joblib_parallel_source=dict(path=str(Path(joblib.__file__).parent / "parallel.py"),
                                             sha256=sha(Path(joblib.__file__).parent / "parallel.py")),
                  seed=20261021, camera_order=[cam.key for cam in base_cams],
                  same_seed_camera_order_equal=True, RNG_after_shuffle=base_after_shuffle,
                  main_process_Python_NumPy_Torch_CPU_RNG_unchanged_by_decoding=True,
                  original_worker_request=2, forced_worker_request=1, image_evidence=image_evidence,
                  generator_before_next=dict(task_iterator_advanced=0, decoded=0),
                  generator_after_one_next_and_wait=counter, finite_fixture_images=6,
                  native_model_imported=False, native_training_executed=False)
        write_json(args.qa_output, qa)
        print(json.dumps(dict(status="passed", output=str(args.qa_output.resolve()), sha256=sha(args.qa_output), joblib=joblib.__version__)))
    finally:
        if old_level is None:
            os.environ.pop("CEM_PROTOCOL_TRAIN_LEVEL", None)
        else:
            os.environ["CEM_PROTOCOL_TRAIN_LEVEL"] = old_level


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--stage", type=int, choices=[1, 2])
    parser.add_argument("--level", choices=["hr"], default="hr")
    parser.add_argument("--gpu")
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--workers", type=int, choices=[1], default=1)
    parser.add_argument("--io-receipt", type=Path)
    parser.add_argument("--cpu-qa", action="store_true")
    parser.add_argument("--qa-output", type=Path)
    args = parser.parse_args()
    if args.cpu_qa:
        assert args.qa_output is not None
        cpu_qa(args)
    else:
        assert all(getattr(args, key) is not None for key in ["source", "model", "stage", "gpu", "receipt"])
        run_native(args)


if __name__ == "__main__":
    main()
