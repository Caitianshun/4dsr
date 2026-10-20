"""Fixed-observation area4 Multi4D runner with complete shared-fine0 state.

GPU admission belongs to the external dispatcher. This program never chooses a
GPU, preempts a job, changes an environment, or modifies the pinned upstream.
"""

import argparse
import ast
from collections import Counter, OrderedDict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np
from PIL import Image
import torch

HERE = Path(__file__).resolve().parent
PROJECT = Path(__file__).resolve().parents[5]
COMMIT = "c483e82cd8daa1b4fdc460ed45882fc2fe7a22b0"
LEGACY = PROJECT / "experiments/dynamic_sr_multi4d_20260924"
METHODS = ("coarse", "nativeLR")


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + "\n")
    temporary.replace(path)


def load_module(name, path):
    specification = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def config(upstream):
    """Return unchanged author dynerf model/hidden/optimizer/pipeline options."""
    from arguments import ModelParams, PipelineParams, OptimizationParams, ModelHiddenParams
    from utils.params_utils import merge_hparams
    import mmcv
    parser = argparse.ArgumentParser()
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    hp = ModelHiddenParams(parser)
    args = merge_hparams(parser.parse_args([]), mmcv.Config.fromfile(str(Path(upstream) / "arguments/dynerf.py")))
    return lp.extract(args), hp.extract(args), op.extract(args), pp.extract(args)


class FinishedStage(Exception):
    """Controlled exit after a complete saved update or common fine0 creation."""


class PausedStage(Exception):
    """A cooperative, complete-state pause, reported with exit status 85."""


class Context:
    def __init__(self, args, spec, scene, hyper, schedule, full_state):
        self.a, self.spec, self.scene = args, spec, scene
        self.hyper, self.schedule, self.fs = hyper, schedule, full_state
        self.out = Path(args.out)
        self.started = time.monotonic()
        self.cache = OrderedDict()
        self.exposure = Counter()
        self.render_calls = self.teacher_samples = self.fine_updates = 0
        self.native_contributions = []
        self.native_scalars = {}
        self.teacher_scalars = {}
        self.current_stage = None
        self.last_completed = None
        self.resume_state = (torch.load(args.resume, map_location="cpu", weights_only=False)
                             if args.resume else None)
        self.resume_used = False
        resume_sha = sha(args.resume) if args.resume else None
        common_fine0_sha = None
        if self.resume_state is not None:
            if self.resume_state["stage"] == "fine" and self.resume_state["completed_updates"] == 0:
                common_fine0_sha = resume_sha
            else:
                common_fine0_sha = self.resume_state["metadata"].get("fine0_sha")
        adapter_identity = json.loads((Path(spec["adapter"]) / "identity.json").read_text())
        self.term_names = {entry["label"]: {node.id for node in ast.walk(ast.parse(entry["expression"]))
                                            if isinstance(node, ast.Name)}
                           for entry in adapter_identity["native_loss_terms"]}
        self.weight = float(spec.get("teacher_weight", .2)) if args.method in {"areaTm", "areaOracle"} else 0.
        self.dssim = float(spec.get("teacher_dssim", .2))
        if self.weight < 0 or not 0 <= self.dssim <= 1:
            raise ValueError("teacher_weight must be nonnegative; teacher_dssim must lie in [0,1]")
        sources = [HERE / "train.py", HERE / "data.py", HERE.parent / "prepare_multi_native.py", LEGACY / "full_state.py",
                   Path(spec["adapter"]) / "adapted_train.py", Path(spec["adapter"]) / "identity.json"]
        upstream = Path(spec["upstream"])
        for name in ("train.py", "arguments/__init__.py", "arguments/dynerf.py", "scene/gaussian_model.py",
                     "scene/deformation.py", "scene/hexplane.py", "gaussian_renderer/__init__.py", "utils/loss_utils.py"):
            sources.append(upstream / name)
        for path in (upstream / "diff_gaussian_rasterization_hybrid").rglob("*"):
            if path.suffix in {".cu", ".h", ".cpp"} and "third_party" not in path.parts:
                sources.append(path)
        self.sources = {str(path): sha(path) for path in sources}
        (self.out / "sources").mkdir()
        for index, path in enumerate(sources):
            shutil.copyfile(path, self.out / "sources" / ("%03d_" % index + path.name))
        self.meta = dict(schema="multi4d_native_lr_run_v1", method=args.method,
                         manifest_sha=sha(spec["manifest"]), init_sha=sha(spec["init"]),
                         schedule_sha=sha(spec["schedule"]), spec_sha=sha(args.spec),
                         upstream_commit=COMMIT, sources=self.sources,
                         gpu=torch.cuda.get_device_name(), torch=str(torch.__version__),
                         visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                         seed=int(spec.get("seed", 0)),
                         fine0_sha=common_fine0_sha, resume_checkpoint_sha256=resume_sha,
                         teacher_weight=self.weight, teacher_dssim=self.dssim,
                         teacher_formula="lambda*((1-dssim)*L1 + dssim*(1-SSIM))",
                         teacher_scope="fine only; existing batch2 cameras; one HR output/camera",
                         native_schedule="coarse2000 + fine20000; phase3 begins at fine10000",
                         precision="native float32; no autocast, AMP or gradient scaler",
                         scaler=None, observation="native original LR RGB; direct native LR losses",
                         adapter_sha256=adapter_identity["adapted_sha256"],
                         singleton_scale_fix=getattr(scene, "singleton_scale_fix", None),
                         oracle_diagnostic_only=args.method == "areaOracle",
                         final_update_repair="None; released author final fine Adam skip preserved")
        self.meta["native_repairs"] = self.check_native_repairs(upstream)
        self.teachers = {}
        if args.method in {"areaTm", "areaOracle"}:
            self.load_teachers()
        write(self.out / "input_audit.json", dict(records=scene.input_identity,
                                                  projection=scene.projection, metadata=self.meta))
        self.log = (self.out / "training.jsonl").open("w", buffering=1)
        torch.cuda.reset_peak_memory_stats()

    def check_native_repairs(self, upstream):
        receipt = Path(self.spec.get("native_repairs_receipt", PROJECT / "output/dynamic_sr_multi4d_20260924/upstream_fixes_v3.json"))
        value = json.loads(receipt.read_text())
        if value["commit"] != COMMIT or len(value["patches"]) != 6:
            raise ValueError("Unexpected native repair receipt")
        for entry in value["patches"]:
            if sha(upstream / entry["path"]) != entry["sha256"]:
                raise ValueError(("Native repair source mismatch", entry["path"]))
        shutil.copyfile(receipt, self.out / "native_repairs.json")
        return dict(receipt_sha256=sha(receipt), patches=value["patches"],
                    disclosure="Inherited compiler headers, hybrid max-contributor statistic, and empty-TR CUDA repairs; no new upstream edits")

    def load_teachers(self):
        index_path = self.spec.get("teacher_index")
        if isinstance(index_path, dict):
            index_path = index_path.get(self.a.method)
        if self.a.method == "areaOracle":
            index_path = self.spec.get("oracle_teacher_index", index_path)
        if not index_path:
            raise ValueError("Teacher method needs the corresponding teacher_index")
        index_path = Path(index_path)
        index = json.loads(index_path.read_text())
        if index["manifest_sha256"] != self.meta["manifest_sha"]:
            raise ValueError("Teacher manifest differs from training manifest")
        expected = {tuple(camera.record_key) for camera in self.scene.train_camera}
        entries = index["entries"]
        if len(entries) != len(expected) or {(e["camera"], e["frame"]) for e in entries} != expected:
            raise ValueError("Teacher index must contain precisely the training cameras and frames")
        kind = index.get("kind", index.get("teacher_kind"))
        if bool(index.get("oracle", False)) != (self.a.method == "areaOracle"):
            raise ValueError("Teacher index must explicitly mark oracle=true only for areaOracle")
        if kind is not None:
            is_oracle = "oracle" in kind.lower() or kind.lower() in {"hr", "clean_hr"}
            if is_oracle != (self.a.method == "areaOracle"):
                raise ValueError(("Teacher/oracle index kind mismatch", kind, self.a.method))
        # relative_path remains relative to the immutable input manifest, as in
        # the original Multi4D audit. Absolute paths are also explicit inputs.
        root = Path(self.spec["manifest"]).parent
        observations = {(record["camera_id"], int(record["frame_index"])): record for record in self.scene.records}
        for entry in entries:
            record = observations[(entry["camera"], entry["frame"])]
            if entry.get("lr_sha256") != record["lr_sha256"]:
                raise ValueError("Teacher record is not bound to the registered training LR image")
            path = root / entry["relative_path"]
            if self.a.method == "areaOracle":
                if path.resolve() != (root / record["hr_path"]).resolve() or entry["sha256"] != record["hr_sha256"]:
                    raise ValueError("Oracle target must be precisely the registered training-view HR image")
            if sha(path) != entry["sha256"]:
                raise ValueError(("Teacher image hash mismatch", path))
            self.teachers[(entry["camera"], entry["frame"])] = path
        self.meta.update(teacher_index_sha=sha(index_path), teacher_kind=kind,
                         teacher_index=str(index_path), teacher_index_entries=len(entries))
        shutil.copyfile(index_path, self.out / "teacher_index.json")

    def sampler_state(self, stage, step):
        return dict(stage=stage, completed_batches=step, schedule_sha=self.meta["schedule_sha"],
                    exposure=dict(self.exposure), render_calls=self.render_calls,
                    teacher_samples=self.teacher_samples, fine_updates=self.fine_updates)

    def stage_start(self, stage, *values):
        *branches, opt = values
        self.current_stage = stage
        step = 0
        if self.resume_state is not None and not self.resume_used:
            state = self.resume_state
            if state["stage"] != stage:
                raise ValueError(("Resume stage differs from the requested native stage", state["stage"], stage))
            for key in ("manifest_sha", "schedule_sha", "init_sha"):
                if state["metadata"][key] != self.meta[key]:
                    raise ValueError(("Resume input identity mismatch", key))
            for key in ("seed", "adapter_sha256", "upstream_commit", "observation"):
                if state["metadata"].get(key) != self.meta[key]:
                    raise ValueError(("Resume method identity mismatch", key))
            if state["hidden_config"] != vars(self.hyper):
                raise ValueError("Resume native hidden configuration changed")
            step = int(state["completed_updates"])
            paused = bool(state.get("cooperative_pause"))
            if paused and (state["metadata"].get("method") != self.a.method
                           or state["metadata"].get("spec_sha") != self.meta["spec_sha"]):
                raise ValueError("A paused run can resume only with identical method and spec bytes")
            if paused:
                for key in ("sources", "teacher_index_sha", "teacher_weight", "teacher_dssim"):
                    if state["metadata"].get(key) != self.meta.get(key):
                        raise ValueError(("Paused-run source or teacher identity changed", key))
            if step == 0 and state["metadata"].get("method") != "coarse":
                raise ValueError("Common fine0 must come from the declared shared coarse preparation")
            if step and not paused and not self.a.stop and not self.spec.get("allow_continuation", False):
                raise ValueError("Formal branches start at common fine0; nonzero continuation must be explicit")
            if step and state["metadata"].get("method") != self.a.method:
                if not (self.spec.get("teacher_start_iteration") == 14001 and step == 14000):
                    raise ValueError("Cross-method continuation only supports the separately declared final6000 diagnostic")
                self.meta["warm_start_diagnostic"] = "areaL0 fine14000 -> final6000; separate from common-fine0 formal comparison"
            self.fs.restore(branches, state, opt)
            self.resume_used = True
            if state.get("scaler") is not None:
                raise ValueError("Native FP32 runner cannot restore a nonempty AMP scaler")
            if "sampler" in state:
                sampled = state["sampler"]
                self.exposure = Counter(sampled.get("exposure", {}))
                self.render_calls = int(sampled.get("render_calls", 0))
                self.teacher_samples = int(sampled.get("teacher_samples", 0))
                self.fine_updates = int(sampled.get("fine_updates", step))
            self.meta["restored_branch_identity"] = self.fs.digest([self.fs.branch_state(g) for g in branches])
            self.meta["resume_completed_updates"] = step
            write(self.out / "reload_audit.json", dict(status="passed", completed_updates=step,
                                                      identity=self.meta["restored_branch_identity"],
                                                      rng_identity=self.fs.digest(self.fs.rng()),
                                                      sampler=self.sampler_state(stage, step)))
        if stage == "fine" and self.a.method == "coarse":
            state = self.snapshot(branches, opt, stage, 0)
            self.fs.save(self.out / "fine0.pt", state)
            self.fs.restore(branches, torch.load(self.out / "fine0.pt", map_location="cpu", weights_only=False), opt)
            write(self.out / "fine0_reload_audit.json", dict(status="passed", sha256=sha(self.out / "fine0.pt"),
                                                            identity=self.fs.digest(state["branches"]),
                                                            rng=self.fs.digest(state["rng"])))
            self.last_completed = dict(stage="fine", completed_updates=0, checkpoint=str(self.out / "fine0.pt"))
            raise FinishedStage()
        if self.a.stop is not None and step >= self.a.stop:
            raise ValueError("--stop must exceed the resumed completed-update counter")
        return step

    def batch(self, stage, iteration):
        self.native_contributions, self.native_scalars, self.teacher_scalars = [], {}, {}
        ids = self.schedule[stage][iteration - 1]
        for index in ids:
            self.exposure[f"{stage}/{self.scene.train_camera[index].image_name}"] += 1
        self.render_calls += len(ids) * (1 if iteration >= 10000 else 4)
        return ids

    def native_term(self, stage, iteration, label, value):
        self.native_contributions.append((label, value.detach()))
        return value

    def teacher(self, stage, cameras, prediction):
        teacher_start = int(self.spec.get("teacher_start_iteration", 1))
        iteration = self.fine_updates + 1
        if stage != "fine" or not self.weight or iteration < teacher_start:
            return prediction.new_zeros(())
        from utils.loss_utils import ssim
        targets = []
        for camera in cameras:
            key = tuple(camera.record_key)
            if key in self.cache:
                image = self.cache.pop(key)
            else:
                with Image.open(self.teachers[key]) as file:
                    image = torch.from_numpy(np.asarray(file.convert("RGB")).copy()).permute(2, 0, 1).float() / 255.
            self.cache[key] = image
            if len(self.cache) > int(self.spec.get("teacher_cache_images", 32)):
                self.cache.popitem(last=False)
            targets.append(image.cuda())
        target = torch.stack(targets)
        if target.shape != prediction.shape:
            raise ValueError(("Teacher HR shape mismatch", target.shape, prediction.shape))
        l1 = (prediction - target).abs().mean()
        dssim = 1.0 - ssim(prediction, target)
        self.teacher_scalars = dict(l1=l1.detach(), dssim=dssim.detach())
        self.teacher_samples += len(cameras)
        return self.weight * ((1.0 - self.dssim) * l1 + self.dssim * dssim)

    def before_backward(self, stage, step, values):
        # These are the original unweighted/native scalar variables where the
        # author named them. Complete weighted contribution traces are separate.
        names = {"loss", "native_loss", "Ll1", "ll2", "psnr_", "ssim_notdense", "ssim_color",
                 "ssim_notdense2", "ssim_loss_temp", "ssim_loss_temp1", "L_soft_dxv",
                 "L_reg_depth_order", "L_reg_tv", "L_sep_fgbg", "L_reg_alpha", "loss_smooth_sta"}
        active = {name for label, _ in self.native_contributions for name in self.term_names[label]}
        names &= active | {"loss", "native_loss", "Ll1", "psnr_"}
        self.native_scalars = {name: values[name].detach() for name in names if name in values
                               and isinstance(values[name], torch.Tensor) and values[name].numel() == 1}

    def after_backward(self, stage, step, *values):
        *branches, teacher = values
        if step in {1, 2, 20, 50, 2000, 3000, 3001, 9999, 10000, 10001, 16000, 20000}:
            norms = {}
            for index, branch in enumerate(branches):
                gradients = [p.grad for group in branch.optimizer.param_groups for p in group["params"] if p.grad is not None]
                if not all(bool(torch.isfinite(gradient).all()) for gradient in gradients):
                    raise FloatingPointError(("Nonfinite gradient", stage, step, index))
                norms[str(index)] = sum(float(gradient.square().sum()) for gradient in gradients) ** .5
            with (self.out / "gradient_audit.jsonl").open("a") as stream:
                stream.write(json.dumps(dict(stage=stage, iteration=step, gradient_norms=norms,
                                             teacher=float(teacher))) + "\n")

    def snapshot(self, branches, opt, stage, step):
        state = self.fs.snapshot(branches, opt, self.hyper, stage, step, self.meta, self.sampler_state(stage, step))
        state.update(scaler=None, precision="float32", adapter_schema="multi4d_native_lr_run_v1")
        return state

    def after_update(self, stage, step, loss, ll1, teacher, *values):
        foreground, background, transient, opt, phase3 = values
        branches = [foreground, background, transient]
        if stage == "fine":
            self.fine_updates = step
        milestones = {2000} if stage == "coarse" else {6000, 10000, 14000, 16000, 20000}
        if self.a.stop is not None:
            milestones.add(self.a.stop)
        row = dict(stage=stage, iteration=step, completed_updates=step, phase=3 if phase3 else 2,
                   loss=float(loss), native_loss=float(self.native_scalars["native_loss"]),
                   lr_l1=float(ll1), teacher=float(teacher),
                   gaussian_counts=[len(branch._xyz) for branch in branches],
                   peak_gb=torch.cuda.max_memory_allocated() / 1e9,
                   peak_reserved_gb=torch.cuda.max_memory_reserved() / 1e9,
                   elapsed_s=time.monotonic() - self.started,
                   render_calls=self.render_calls, teacher_samples=self.teacher_samples,
                   optimizer_updates_per_branch=min(step, opt.iterations - 1), completed_loop_iterations=step)
        if step == 1 or step % 100 == 0 or step in milestones:
            row["native_scalars"] = {key: float(value) for key, value in self.native_scalars.items()}
            row["native_loss_contributions"] = [dict(label=key, value=float(value)) for key, value in self.native_contributions]
            row["teacher_raw"] = {key: float(value) for key, value in self.teacher_scalars.items()}
            row["learning_rates"] = [[dict(name=group.get("name"), lr=group["lr"]) for group in branch.optimizer.param_groups]
                                     for branch in branches]
            self.log.write(json.dumps(row, allow_nan=False) + "\n")
        if step in milestones:
            state = self.snapshot(branches, opt, stage, step)
            checkpoint = self.out / f"{stage}_{step}.pt"
            self.fs.save(checkpoint, state)
            write(self.out / f"{stage}_{step}.json", dict(**row, checkpoint=str(checkpoint), sha256=sha(checkpoint), sampler=state["sampler"]))
            self.last_completed = dict(stage=stage, completed_updates=step, checkpoint=str(checkpoint), sha256=sha(checkpoint))
            if self.a.stop is not None:
                before = self.fs.digest(state["branches"])
                self.fs.restore(branches, torch.load(checkpoint, map_location="cpu", weights_only=False), opt)
                if before != self.fs.digest([self.fs.branch_state(branch) for branch in branches]):
                    raise ValueError("Trained full-state reload identity changed")
                write(self.out / "trained_reload_audit.json", dict(status="passed", stage=stage, step=step, identity=before))
        pause_file = self.spec.get("pause_file")
        terminal_smoke = self.a.stop is not None and step >= self.a.stop
        if (pause_file and not terminal_smoke and not (stage == "fine" and step == 20000)
                and Path(pause_file).exists()):
            state = self.snapshot(branches, opt, stage, step)
            state["cooperative_pause"] = dict(reason="Claude priority / external dispatcher request", pause_file=str(pause_file))
            checkpoint = self.out / "pause.pt"
            self.fs.save(checkpoint, state)
            self.last_completed = dict(stage=stage, completed_updates=step, checkpoint=str(checkpoint), sha256=sha(checkpoint))
            write(self.out / "pause.json", dict(status="paused", **self.last_completed,
                                                sampler=state["sampler"], metadata=self.meta,
                                                resume="Identical spec and method, new --out, --resume this pause.pt; no budget reset"))
            raise PausedStage()
        if self.a.stop is not None and step >= self.a.stop:
            raise FinishedStage()


def validate_schedule(schedule, scene):
    if schedule["record_keys"] != [list(camera.record_key) for camera in scene.train_camera]:
        raise ValueError("Schedule record order differs from the manifest camera order")
    count = len(scene.train_camera)
    for stage, updates in (("coarse", 2000), ("fine", 20000)):
        batches = schedule[stage]
        if len(batches) != updates or any(len(batch) != 2 for batch in batches):
            raise ValueError(("Native batch2 schedule length mismatch", stage))
        flat = [index for batch in batches for index in batch]
        if any(not isinstance(index, int) or not 0 <= index < count for index in flat):
            raise ValueError(("Schedule contains invalid camera index", stage))
        # Native pop-without-replacement resets at the epoch boundary. A final
        # partial epoch must also contain unique indices.
        for offset in range(0, len(flat), count):
            part = flat[offset:offset + count]
            if len(part) != len(set(part)):
                raise ValueError(("Schedule is not native without-replacement", stage, offset))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stop", type=int, help="Engineering smoke endpoint; never a formal native-schedule result")
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    for name in ("manifest", "init", "schedule", "adapter", "upstream"):
        if name not in spec:
            raise ValueError(("Missing spec key", name))
        spec[name] = str(Path(spec[name]).resolve())
    if args.stop is not None and not 1 <= args.stop <= (2000 if args.method == "coarse" else 20000):
        raise ValueError("Smoke --stop must fall within its native stage")
    if args.method != "coarse" and not args.resume:
        raise ValueError("Area methods require the common fine0 --resume checkpoint")
    args.out.mkdir(parents=True, exist_ok=False)
    try:
        if subprocess.check_output(["git", "-C", spec["upstream"], "rev-parse", "HEAD"], text=True).strip() != COMMIT:
            raise ValueError("Upstream commit differs from pinned Multi4D")
        identity = json.loads((Path(spec["adapter"]) / "identity.json").read_text())
        if identity.get("schema") != "multi4d_native_lr_adapter_v1" or identity["commit"] != COMMIT:
            raise ValueError("Wrong generated observation adapter")
        if sha(Path(spec["adapter"]) / "adapted_train.py") != identity["adapted_sha256"]:
            raise ValueError("Generated adapter changed after audit")
        sys.path.insert(0, spec["upstream"])
        torch.set_num_threads(int(spec.get("cpu_threads", 4)))
        native = load_module("multi4d_area4_native", Path(spec["adapter"]) / "adapted_train.py")
        data = load_module("multi4d_area4_data", HERE / "data.py")
        full_state = load_module("multi4d_area4_full_state", LEGACY / "full_state.py")
        dataset, hyper, opt, pipe = config(spec["upstream"])
        if (opt.coarse_iterations, opt.iterations, opt.batch_size) != (2000, 20000, 2):
            raise ValueError("Pinned native schedule/batch size changed")
        seed = int(spec.get("seed", 0))
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        foreground = native.GaussianModel_dynamic(3, hyper)
        background = native.GaussianModel(3)
        transient = native.GaussianModelTransient(3, gaussian_dim=4,
                                                 time_duration=list(getattr(opt, "transient_time_duration", [0., 10.])),
                                                 rot_4d=True, sh_degree_t=2)
        scene = data.ManifestScene(spec["manifest"], spec["init"], foreground, background, transient, args.out)
        schedule = json.loads(Path(spec["schedule"]).read_text())
        validate_schedule(schedule, scene)
        ctx = Context(args, spec, scene, hyper, schedule, full_state)
        native.adapter_context = ctx
        write(args.out / "config.json", dict(args=vars(args), spec=spec, native_optimizer=vars(opt),
                                             native_hidden=vars(hyper), metadata=ctx.meta,
                                             engineering_smoke=args.stop is not None))
        paused = False
        try:
            if args.method == "coarse":
                native.scene_reconstruction(opt, hyper, pipe, [], [], None, foreground, scene, "coarse", 2000, background, transient)
                # Exact author transition: discard only these radius statistics.
                foreground.max_radii2D = torch.zeros_like(foreground.max_radii2D).cuda()
                foreground.mean_radii2D = torch.zeros_like(foreground.mean_radii2D).cuda()
                background.max_radii2D = torch.zeros_like(background.max_radii2D).cuda()
            native.scene_reconstruction(opt, hyper, pipe, [], [], None, foreground, scene, "fine", 20000, background, transient)
        except FinishedStage:
            pass
        except PausedStage:
            paused = True
        for path, digest in ctx.sources.items():
            if sha(path) != digest:
                raise ValueError(("Source changed while running", path))
        status = ("paused" if paused else "smoke_completed" if args.stop is not None
                  else "fine0_prepared" if args.method == "coarse" else "completed")
        write(args.out / ("paused.json" if paused else "complete.json"), dict(status=status, metadata=ctx.meta, last_completed=ctx.last_completed,
                                               elapsed_s=time.monotonic() - ctx.started,
                                               peak_gb=torch.cuda.max_memory_allocated() / 1e9,
                                               render_calls=ctx.render_calls, teacher_samples=ctx.teacher_samples,
                                               fine_updates=ctx.fine_updates))
        ctx.log.close()
        return 85 if paused else 0
    except BaseException:
        write(args.out / "failed.json", dict(status="failed", traceback=traceback.format_exc()))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
