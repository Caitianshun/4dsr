"""CoR-GS 式伪视角共正则（不含 co-pruning） on the PVCD modkit4 SR recipe.

Two independently restored students share registered batches and one PseudoViews
sample per step. Each fits its peer's detached, natively rendered LR image.
The actual modkit4 -> modkit2 -> modkit -> srfix(fix) chain installs the recipe;
only cullretrain.run_sr and the model-specific dispatch of its hooks are replaced.

cotrain_modkit.py --out-b DIR [--weight .5 --dssim .2 --scale 4
    --seed 20261014 --log-every 50] -- <unchanged modkit4 arguments, including -->

See README.md for commands, global-state handling and GPU acceptance checks.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import json
import math
from pathlib import Path
import socket
import sys
import time
import traceback
from types import SimpleNamespace

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
MOD = HERE.parents[1]
CULL = ROOT / "experiments/dynamic_sr_cull_retrain_20261009"
VAR = CULL / "variance_20261014"
GEO = MOD / "geoprobe_20261017"
METHOD = "CoR-GS 式伪视角共正则（不含 co-pruning）"
KEYS = ("a", "b")


def rooted(value):
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def option(argv, name):
    if argv.count(name) != 1 or argv.index(name) + 1 >= len(argv):
        raise ValueError(f"expected exactly one separate {name} VALUE")
    return argv[argv.index(name) + 1]


def replace_option(argv, name, value):
    result = list(argv)
    option(result, name)
    result[result.index(name) + 1] = str(value)
    return result


def parse_cli(argv):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weight", type=float, default=0.5)
    parser.add_argument("--dssim", type=float, default=0.2)
    parser.add_argument("--scale", type=int, choices=(1, 2, 4), default=4)
    parser.add_argument("--seed", type=int, default=20261014)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--out-b", required=True)
    if "--" not in argv:
        if "-h" in argv or "--help" in argv:
            parser.parse_args(argv)
        parser.error("need -- followed by modkit4 arguments (with their own --)")
    cut = argv.index("--")
    own, kit = parser.parse_args(argv[:cut]), argv[cut + 1:]
    if not math.isfinite(own.weight) or own.weight < 0:
        parser.error("--weight must be finite and nonnegative")
    if not 0 <= own.dssim <= 1 or own.seed < 0 or own.log_every < 1:
        parser.error("--dssim must be in [0, 1], --seed >= 0, --log-every >= 1")
    if "--" not in kit:
        parser.error("modkit4 arguments need their own -- before --mode fix")
    kit_options, sr = kit[:kit.index("--")], kit[kit.index("--") + 1:]
    if sr[:4] != ["--mode", "fix", "cullretrain", "sr"]:
        parser.error("expected ... -- --mode fix cullretrain sr <SR arguments>")
    try:
        if option(kit_options, "--lr-operator") != "area":
            parser.error("this area-domain comparison requires --lr-operator area")
        for name in ("--parent-checkpoint", "--baseline-sr", "--arm"):
            option(sr, name)
        outs = dict(a=rooted(option(sr, "--out")), b=rooted(own.out_b))
    except ValueError as error:
        parser.error(str(error))
    if outs["a"] == outs["b"] or any(outs[x] in outs[y].parents for x, y in (("a", "b"), ("b", "a"))):
        parser.error("--out and --out-b must be separate, non-nested directories")
    # The upstream chain creates --out before run_sr; check both paths beforehand.
    for path in outs.values():
        if path.exists() and (not path.is_dir() or any(path.iterdir())):
            raise FileExistsError(f"nonempty output preserved (no resume): {path}")
    return own, kit, outs


def tensor_leaves(value, torch):
    if torch.is_tensor(value):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from tensor_leaves(item, torch)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from tensor_leaves(item, torch)


class PairLoop:
    def __init__(self, own, kit, outs, rt, srfix, torch, np, cd, kit4, gaussian, loss_utils):
        self.own, self.kit, self.outs = own, kit, outs
        self.rt, self.srfix, self.torch, self.np, self.cd = rt, srfix, torch, np, cd
        self.cr, self.kit4, self.GaussianModel = srfix.cr, kit4, gaussian
        self.kit2, self.kit1 = kit4.modkit2, kit4.modkit2.modkit
        self.losses, self.native = self.cr.losses, self.cr.native
        self.loss_utils = loss_utils
        # Capture before any main() installs camera/time/drop/tracking wrappers.
        self.native0 = SimpleNamespace(hr_camera=self.native.hr_camera, native_rgb=self.native.native_rgb,
                                       native_effective_state=self.native.native_effective_state)
        self.ssim0 = loss_utils.ssim
        self.students, self.by_model, self.files, self.completion_written = {}, {}, [], []
        self.active, self.kit_state, self.time_state = None, None, None
        self.drop_options, self.probe_every, self.prefetch = None, 0, None
        self.started = time.time()
        self.record = dict(kind="cotrain_modkit", method=METHOD, co_pruning=False,
                           weight=own.weight, dssim=own.dssim, scale=own.scale, seed=own.seed,
                           log_every=own.log_every, modkit4_argv=kit, out_b=str(outs["b"]),
                           outputs={k: str(v) for k, v in outs.items()}, status="initializing",
                           calls=0, backward_calls=0, started_unix=self.started)

    def open_log(self, path):
        file = path.open("x", buffering=1)
        self.files.append(file)
        return file

    def write_record(self):
        for path in self.outs.values():
            self.rt.write_json(path / "cotrain.json", self.record)

    def student(self, model):
        return self.by_model.get(id(model))

    def require_active(self, student):
        if self.active is not student:
            raise RuntimeError("registered render/update used another student's context")

    @contextmanager
    def activate(self, student):
        previous = self.active
        registries = [(module.REG, dict(module.REG)) for module in (self.kit2, self.kit4)]
        fix = dict(self.srfix.STATE)
        self.active = student
        for registry, _ in registries:
            registry["affine"] = student["affine"]
        # srfix's final-step hook must see this student's out and hit list.
        self.srfix.STATE.update(out=student["out"], hits=student["hits"])
        try:
            yield
        finally:
            self.active = previous
            for registry, saved in registries:
                registry.clear()
                registry.update(saved)
            self.srfix.STATE.clear()
            self.srfix.STATE.update(fix)

    def install_tracking(self, gaussian, cr, native, state):
        """Same tracking as modkit, dispatched by model instead of last restore()."""
        self.kit_state = state
        update, rgb = gaussian.update_learning_rate, native.native_rgb

        def update_learning_rate(model, iteration):
            student = self.student(model)
            if student is not None:
                self.require_active(student)
                student.update(cursor=int(iteration) - cr.LR_OFFSET, terms=dict(lr=[], sr=[]), vs=[])
            return update(model, iteration)

        def native_rgb(camera, model, st, pipe, background, raster_api=None):
            packet = rgb(camera, model, st, pipe, background, raster_api)
            student = self.student(model)
            if student is not None and self.torch.is_grad_enabled():
                self.require_active(student)
                student["vs"].append(packet["viewspace_points"])
                student["registered_renders"] += 1
            return packet

        gaussian.update_learning_rate, native.native_rgb = update_learning_rate, native_rgb

    def install_drop(self, native, gaussian, cr, rate, seed, stop, state):
        """Optional upstream DropGaussian, with a generator/counters per student."""
        self.drop_options = dict(rate=rate, seed=seed, stop=stop)
        rgb, torch = native.native_rgb, self.torch

        def native_rgb(camera, model, st, pipe, background, raster_api=None):
            student = self.student(model)
            if student is not None and torch.is_grad_enabled() and st["opacity"].requires_grad:
                self.require_active(student)
                rate_now = rate * student["cursor"] / stop
                if rate_now > 0:
                    dev = st["opacity"].device
                    if dev not in student["drop_generators"]:
                        generator = torch.Generator(device=dev)
                        generator.manual_seed(seed)
                        student["drop_generators"][dev] = generator
                    keep = (torch.rand(st["opacity"].shape[0], 1, device=dev,
                                       generator=student["drop_generators"][dev]) >= rate_now).to(st["opacity"].dtype)
                    st = dict(st, opacity=st["opacity"] * keep / (1 - rate_now))
                    student["drop"]["renders"] += 1
                    student["drop"]["dropped"] += int((keep == 0).sum())
                    student["drop"]["kept"] += int(keep.sum())
            return rgb(camera, model, st, pipe, background, raster_api)

        native.native_rgb = native_rgb

    def install_time_hooks(self, native, state):
        """modkit2's time hook, looking up the affine by GaussianModel identity."""
        self.time_state = state
        hr, rgb, torch = native.hr_camera, native.native_rgb, self.torch

        def hr_camera(observation, uid, manifest):
            camera = hr(observation, uid, manifest)
            camera.mk_camera_id = observation["camera_id"]
            return camera

        def native_rgb(camera, model, st, pipe, background, raster_api=None):
            student = self.student(model)
            if student is not None and torch.is_grad_enabled() and hasattr(camera, "mk_camera_id"):
                self.require_active(student)
                offset = student["affine"].time_offset(camera.mk_camera_id)
                if offset is not None:
                    tt = (torch.tensor(float(camera.time), dtype=torch.float32, device=model.get_xyz.device)
                          + offset).reshape(1, 1).repeat(model.get_xyz.shape[0], 1)
                    xyz, scale, rotation, opacity, sh = model._deformation(
                        model.get_xyz, model._scaling, model._rotation, model._opacity, model.get_features, tt)
                    st = dict(xyz=xyz, scales=model.scaling_activation(scale), rotation=model.rotation_activation(rotation),
                              opacity=model.opacity_activation(opacity), sh=sh)
                    student["time"]["renders"] += 1
            return rgb(camera, model, st, pipe, background, raster_api)

        native.hr_camera, native.native_rgb = hr_camera, native_rgb

    def install_grad_probe(self, gaussian, losses, state, every, out):
        """The optional modkit diagnostic uses each student's own terms/points/log."""
        self.probe_every = every
        lr_loss, sr_loss, regulation = losses.lr_loss, losses.sr_loss, gaussian.compute_regulation

        def capture(function, key):
            def loss(rgb, target):
                value = function(rgb, target)
                if self.active is not None:
                    self.active["terms"][key].append(value)
                return value
            return loss

        def compute_regulation(model, *args, **kwargs):
            value = regulation(model, *args, **kwargs)
            student = self.student(model)
            if student is not None:
                self.require_active(student)
                self.grad_probe(student)
            return value

        losses.lr_loss, losses.sr_loss = capture(lr_loss, "lr"), capture(sr_loss, "sr")
        gaussian.compute_regulation = compute_regulation

    def grad_probe(self, student):
        probe, cursor = student["probe"], student["cursor"]
        if probe["disabled"] or cursor <= 0 or cursor % self.probe_every:
            return
        torch, model = self.torch, student["g"]
        try:
            terms = student["terms"]
            if len(terms["lr"]) < 1 or len(terms["sr"]) < 2:
                return
            params = dict(xyz=model._xyz, f_dc=model._features_dc, opacity=model._opacity, scaling=model._scaling)
            names = [key for key, param in params.items() if param.requires_grad]
            ps = [params[key] for key in names]
            gl = torch.autograd.grad(terms["lr"][0], ps, retain_graph=True, allow_unused=True)
            gs = torch.autograd.grad(terms["sr"][-2] + terms["sr"][-1], ps, retain_graph=True, allow_unused=True)
            row = dict(cursor=cursor)
            for key, left, right in zip(names, gl, gs):
                if left is None or right is None:
                    continue
                left, right = left.reshape(left.shape[0], -1), right.reshape(right.shape[0], -1)
                dot = (left * right).sum(1)
                both = (left.norm(dim=1) > 0) & (right.norm(dim=1) > 0)
                neg, norm = both & (dot < 0), right.norm(dim=1)
                row[key] = dict(cos=float((left * right).sum() / (left.norm() * right.norm() + 1e-30)),
                                neg_frac=float(neg.sum() / both.sum().clamp_min(1)),
                                neg_sr_norm_share=float(norm[neg].sum() / norm[both].sum().clamp_min(1e-30)),
                                lr_norm=float(left.norm()), sr_norm=float(right.norm()))
            if student["probe_log"] is None:
                student["probe_log"] = self.open_log(student["out"] / "grad_probe.jsonl")
            student["probe_log"].write(json.dumps(row) + "\n")
            probe["rows"] += 1
        except Exception as error:
            probe.update(errors=probe["errors"] + 1, disabled=True, error=repr(error)[:500])
        finally:
            # autograd.grad can populate retain_grad tensors even if it then fails.
            for points in student["vs"]:
                points.grad = None

    def check_independence(self, initial=False):
        """Include optimizer moments and topology buffers, not just Parameter ids."""
        torch, storage_sets, parameter_maps = self.torch, {}, {}
        for key, student in self.students.items():
            model, affine = student["g"], student["affine"]
            params = dict(self.cr.prefix.parameter_map(model))
            params.update({f"deformation.{name}": value for name, value in model._deformation.named_parameters()})
            tensors = list(params.values()) + list(model._deformation.buffers())
            tensors += [getattr(model, name) for name in (
                "max_radii2D", "xyz_gradient_accum", "denom", "_deformation_table", "_deformation_accum")]
            optimizers = [model.optimizer]
            if affine is not None:
                optimizers += [value for value in vars(affine).values() if isinstance(value, torch.optim.Optimizer)]
                params = dict(params)
                params.update({f"affine.{name}": value for name, value in vars(affine).items()
                               if torch.is_tensor(value) and value.requires_grad})
                # Inspect live tensors: state_dict() clones the camera parameters.
                # Comparing clone addresses could hide sharing or report allocator reuse.
                tensors += list(tensor_leaves(vars(affine), torch))
            for optimizer in optimizers:
                tensors += [param for group in optimizer.param_groups for param in group["params"]]
                tensors += list(tensor_leaves(optimizer.state, torch))
            storage_sets[key] = {(str(tensor.device), tensor.untyped_storage().data_ptr())
                                 for tensor in tensors if tensor.numel()}
            parameter_maps[key] = params
        if storage_sets["a"] & storage_sets["b"]:
            raise RuntimeError("students share tensor storage (parameters, optimizer state or topology buffers)")
        if initial:
            left, right = parameter_maps["a"], parameter_maps["b"]
            if left.keys() != right.keys() or any(not torch.equal(left[name], right[name]) for name in left):
                raise RuntimeError("same-parent students did not start with equal parameters")
            self.record["initial_parameters_equal"] = True
        self.record["storage_independence_checks"] = self.record.get("storage_independence_checks", 0) + 1
        self.record["disjoint_student_storage"] = True

    def objective(self, student, cameras, items, lrs, teachers, definition):
        cr, torch = self.cr, self.torch
        model, affine, hidden = student["g"], student["affine"], student["h"]
        # Import after geosr's optional teacher-lowpass has wrapped registered SSIM.
        ssim = self.loss_utils.ssim
        with self.activate(student):
            st = self.native.native_effective_state(model, cameras[0].time, torch)
            packages = [self.native.native_rgb(cam, model, st, student["pipe"], self.background) for cam in cameras]
            rgbs = [packet["render"] for packet in packages]
            if affine is not None:
                rgbs = [affine.apply(rgb, item["camera_id"]) for rgb, item in zip(rgbs, items)]
            lr_values = [self.losses.lr_loss(rgb, lr) for rgb, lr in zip(rgbs, lrs)]
            lr_term = sum(weight * value for weight, value in zip(definition["lr_weights"], lr_values))
            sr_values = [self.losses.sr_loss(rgbs[i], target) for i, target in zip((1, 2), teachers)]
            if self.sr_args.dssim:
                d = self.sr_args.dssim
                down = self.losses.degradation(rgbs[0], lrs[0].shape[-2:])
                lr_term = (1 - d) * lr_term + d * (1 - ssim(down[None], lrs[0][None]))
                sr_mix = [(1 - d) * value + d * (1 - ssim(rgbs[i][None], target[None]))
                          for value, i, target in zip(sr_values, (1, 2), teachers)]
                sr_term = self.sr_args.lambda_sr * sum(sr_mix)
            else:
                sr_term = self.sr_args.lambda_sr * sum(sr_values)
            regularizer = (model.compute_regulation(hidden.time_smoothness_weight, hidden.l1_time_planes, hidden.plane_tv_weight)
                           if hidden.time_smoothness_weight != 0 else rgbs[0].sum() * 0.)
        return dict(lr=lr_term, sr=sr_term, reg=regularizer, lr_values=lr_values, sr_values=sr_values), packages

    def pseudo_render(self, student, camera, pseudo_time):
        model, torch = student["g"], self.torch
        st = self.native0.native_effective_state(model, pseudo_time, torch)
        image = self.native0.native_rgb(camera, model, st, student["pipe"], self.background)["render"]
        if self.own.scale > 1:
            size = (int(image.shape[-2]) // self.own.scale, int(image.shape[-1]) // self.own.scale)
            if min(size) < 1:
                raise ValueError("pseudo-view is smaller than --scale")
            image = self.losses.degradation(image, size)
        return image

    def consistency(self, prediction, peer):
        target = peer.detach()
        if target.requires_grad:
            raise RuntimeError("peer target must be detached")
        l1 = (prediction - target).abs().mean()
        d = self.own.dssim
        return (1 - d) * l1 + d * (1 - self.ssim0(prediction[None], target[None])) if d else l1

    def run_sr(self, args):
        cr, torch, np = self.cr, self.torch, self.np
        self.sr_args = args
        if self.losses.degradation is not self.kit1.calibsr_v2.area_degradation:
            raise RuntimeError("the modkit chain did not install the area LR operator")
        base_dir, parent = rooted(args.baseline_sr), rooted(args.parent_checkpoint)
        base = cr.read(base_dir / "config.json")
        definition = base["definition"]
        if definition != dict(schedule="rows", lr_weights=[1.0, 0.0, 0.0], sr_weights=[0.0, 0.05, 0.05], X=False, E=False):
            raise ValueError("Only the registered Bsync definition is supported")
        self.stop = args.stop if args.stop is not None else cr.SR_STEPS
        if not 1 <= self.stop <= cr.SR_STEPS or args.cpu_threads < 1 or args.cull_interval < 1:
            raise ValueError("require 1 <= stop <= SR_STEPS, positive cpu-threads and cull-interval")
        torch.set_num_threads(args.cpu_threads)
        parent_entry = cr.entry(parent)
        side = parent.with_suffix(".json")
        if side.exists() and cr.read(side).get("sha256") not in (None, parent_entry["sha256"]):
            raise ValueError("parent checkpoint differs from its SHA256 sidecar")
        payload = torch.load(parent, map_location="cuda", weights_only=False)
        self.plan = payload["plan"]
        if self.plan["configuration"] != base["configuration"]:
            raise ValueError("Parent configuration differs from the baseline SR configuration")
        manifest_path, manifest = cr.plan_scene(self.plan)
        if cr.rel(manifest_path) != base["data"]["manifest"]["path"]:
            raise ValueError("Manifest differs from the baseline SR plan")
        if len(manifest["splits"]["train"]) < 3:
            raise ValueError("PseudoViews requires at least three training cameras")
        dataset = cr.N3DVPreparedDataset(manifest, "train", "lr", device="cpu", cache=False)
        table_path, teacher_path = (ROOT / base["dependencies"][key]["path"] for key in ("schedule", "teacher"))
        table, teacher_index = cr.read(table_path), cr.read(teacher_path)
        rows, record_keys = table[definition["schedule"]], [tuple(key) for key in table["record_keys"]]
        if self.stop > len(rows):
            raise ValueError("stop exceeds the registered schedule length")
        teacher_rows = {(entry["camera"], int(entry["frame"])): entry for entry in teacher_index["entries"]}
        teacher_paths = {key: self.native.safe_child(Path(manifest["_root"]), entry["relative_path"])
                         for key, entry in teacher_rows.items()}
        source_files = [Path(__file__), Path(self.cd.__file__), Path(cr.__file__), Path(self.rt.__file__),
                        Path(self.srfix.__file__), Path(self.kit1.__file__), Path(self.kit2.__file__), Path(self.kit4.__file__),
                        Path(self.kit1.calibsr_v2.__file__), Path(self.kit1.geosr.__file__), Path(self.losses.__file__),
                        Path(self.native.__file__), Path(cr.prefix.__file__), Path(cr.ev.__file__),
                        Path(sys.modules[self.GaussianModel.__module__].__file__)]
        common = dict(kind="sr", parent_checkpoint=parent_entry, baseline_sr=cr.rel(base_dir), scene=self.plan["scene"],
                      seed=self.plan["seed"], lambda_SR=args.lambda_sr, dssim=args.dssim, host=socket.gethostname(),
                      gpu=torch.cuda.get_device_name(), runtime=self.rt.check_runtime(),
                      sources={cr.rel(path): self.rt.sha(path) for path in source_files}, started_unix=self.started,
                      resource_scope="paired_process",
                      prune_gate=args.prune_gate, affine=bool(args.affine), cull_interval=args.cull_interval,
                      schedule=cr.entry(table_path), teacher_index=cr.entry(teacher_path))
        self.record.update(parent_checkpoint=parent_entry, baseline_sr=cr.rel(base_dir), scene=self.plan["scene"],
                           stop=self.stop, sr_steps=cr.SR_STEPS, lambda_SR=args.lambda_sr, registered_dssim=args.dssim,
                           host=common["host"], gpu=common["gpu"], runtime=common["runtime"], sources=common["sources"],
                           schedule=common["schedule"], teacher_index=common["teacher_index"],
                           resource_scope="paired_process")
        parent_rng = payload["rng"]
        for key in KEYS:
            # restore() takes ownership of payload tensors and optimizer moments.
            # A second restore of the same object would alias the two students.
            if key == "b":
                payload = torch.load(parent, map_location="cuda", weights_only=False)
                if payload["plan"] != self.plan:
                    raise ValueError("parent changed between the two independent loads")
            hidden, opt, pipe, sh = cr.configuration(self.plan)
            model = self.GaussianModel(sh, hidden)
            model._deformation = model._deformation.cuda()
            model.restore(payload["model"], opt)
            model._deformation_accum = payload["deformation_accum"]
            affine = None
            if args.affine:
                affine = cr.CameraAffine(manifest["splits"]["train"], lr_init=args.affine_sr_lr,
                                         lr_final=args.affine_sr_lr, max_steps=cr.SR_STEPS)
                if payload.get("affine") is not None:
                    affine.load_state_dict(payload["affine"])
                    for group in affine.optimizer.param_groups:
                        group["lr"] = args.affine_sr_lr
            model.optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                for name in ("max_radii2D", "xyz_gradient_accum", "denom"):
                    getattr(model, name).zero_()
            out = self.outs[key]
            out.mkdir(parents=True, exist_ok=True)
            arm = args.arm if key == "a" else args.arm + "_b"
            student_args = dict(vars(args), out=str(out), arm=arm)
            culler = cr.Culler(manifest) if args.cull else None
            record = dict(common, arm=arm, args=student_args, student=key, method=METHOD,
                          peer=str(self.outs["b" if key == "a" else "a"]), culler=culler.describe() if culler else None)
            if args.affine and payload.get("affine") is not None:
                record["affine_inherited"] = True
            student = dict(g=model, affine=affine, h=hidden, opt=opt, pipe=pipe, out=out, arm=arm, record=record,
                           culler=culler, culls=[], cursor=0, steps=0, affine_steps=0, hits=[], registered_renders=0,
                           vs=[], terms=dict(lr=[], sr=[]), drop_generators={}, drop=dict(renders=0, dropped=0, kept=0),
                           time=dict(renders=0), probe=dict(every=self.probe_every, rows=0, errors=0, disabled=False),
                           probe_log=None, consistency_values=[],
                           counts=dict(densify=0, prune=0, pruned_points=0, grow=0, opacity_reset=0,
                                       final_step_topology_skipped=0),
                           log=self.open_log(out / "trajectory.jsonl"), pair_log=self.open_log(out / "cotrain.jsonl"))
            self.students[key], self.by_model[id(model)] = student, student
            self.rt.write_json(out / "run.json", record)
            del payload
        if self.rt.sha(parent) != parent_entry["sha256"]:
            raise ValueError("parent bytes changed while initializing students")
        # Same policy as cotrain_sr: restore once after both initializations; then
        # A and B consume successive draws from the parent's shared topology RNG.
        cr.prefix.restore_rng(parent_rng, torch, np)
        del parent_rng
        self.check_independence(initial=True)
        views = self.cd.PseudoViews(manifest, self.own.seed)
        self.record.update(pseudo_views=dict(cameras=views.cams, neighbours=views.neighbours, alpha=[0.25, 0.75],
                                            times=len(views.times)), status="running",
                           target="peer.detach() after the shared LR degradation; no extra HR clamp",
                           rng_policy="parent RNG restored once; shared A-then-B topology stream; no parameter perturbation")
        self.background = (torch.ones if cr.white_background(self.plan) else torch.zeros)(
            3, dtype=torch.float32, device="cuda")
        self.write_record()

        def fetch(index):
            row = rows[index]
            items = [dataset[item] for item in row]
            keys = [record_keys[item] for item in row]
            teachers = [cr.read_teacher(teacher_paths[keys[j]]) for j in (1, 2)]
            return row, items, keys, teachers

        self.prefetch = cr.Prefetch(fetch, list(range(self.stop)))
        torch.cuda.reset_peak_memory_stats()
        for student in self.students.values():
            if student["culler"] is not None:
                with self.activate(student):
                    student["culls"].append(dict(cr.cull(student["g"], student["culler"], student["log"], "sr_start"), cursor=0))
        self.check_independence()
        for cursor in range(1, self.stop + 1):
            self.step(cursor, self.prefetch.get(cursor - 1), views, manifest, definition)
        self.prefetch.close()
        self.prefetch = None
        for student in self.students.values():
            if student["culler"] is not None and self.stop % args.cull_interval:
                with self.activate(student):
                    student["culls"].append(dict(cr.cull(student["g"], student["culler"], student["log"],
                                                       f"sr_end_{self.stop}"), cursor=self.stop))
        self.check_independence()
        for student in self.students.values():
            if student["hits"] != [self.stop] or student["counts"]["final_step_topology_skipped"] != 1:
                raise RuntimeError("srfix did not skip exactly this student's final topology")
            with self.activate(student):
                path = cr.save(student["out"] / f"checkpoint_{self.stop:05d}.pt", cr.SCHEMA_SR, student["g"], self.plan,
                               dict(affine=student["affine"].state_dict() if student["affine"] else None,
                                    metadata=dict(cursor=self.stop, arm=student["arm"], method=METHOD,
                                                  student=student["record"]["student"], co_pruning=False)))
            student["checkpoint"] = cr.entry(path)
        torch.cuda.synchronize()
        for student in self.students.values():
            student["result"] = dict(status="completed", cursor=self.stop, seconds=time.time() - self.started,
                                     counts=student["counts"], culls=student["culls"], points=int(student["g"].get_xyz.shape[0]),
                                     peak_GB=torch.cuda.max_memory_allocated() / 1e9,
                                     affine=student["affine"].summary() if student["affine"] else None,
                                     checkpoint=student["checkpoint"])
        # Upstream postambles now serialize A's summaries and A's srfix hit.
        first = self.students["a"]
        for module in (self.kit2, self.kit4):
            module.REG["affine"] = first["affine"]
        self.srfix.STATE.update(out=first["out"], hits=first["hits"])
        self.kit_state.update(drop=first["drop"] if self.drop_options else None,
                              probe=first["probe"] if self.probe_every else None)
        if self.time_state is not None:
            self.time_state["time"] = first["time"]
        return first["result"]

    def step(self, cursor, batch, views, manifest, definition):
        cr, torch, args = self.cr, self.torch, self.sr_args
        tick = time.time()
        for student in self.students.values():
            with self.activate(student):
                model = student["g"]
                model.update_learning_rate(cr.LR_OFFSET + cursor)
                if (cr.LR_OFFSET + cursor) % 1000 == 0:
                    model.oneupSHdegree()
                model.optimizer.zero_grad(set_to_none=True)
        row, items, keys, teacher_cpu = batch
        if len(row) != 3 or len({key[0] for key in keys}) != 3 or len({int(key[1]) for key in keys}) != 1:
            raise ValueError("Registered same-time three-view batch differs")
        cameras = [self.native.hr_camera(item, index, manifest) for index, item in zip(row, items)]
        lrs, teachers = [item["image"].cuda() for item in items], [teacher.cuda() for teacher in teacher_cpu]
        parts, packages, pseudo = {}, {}, {}
        for key, student in self.students.items():
            parts[key], packages[key] = self.objective(student, cameras, items, lrs, teachers, definition)
        camera, pseudo_time, info = views.sample(cr.ev, self.native0, torch)
        if hasattr(camera, "mk_camera_id"):
            raise RuntimeError("pseudo camera passed through a modkit time wrapper")
        for key, student in self.students.items():
            pseudo[key] = self.pseudo_render(student, camera, pseudo_time)
            # These are the only screen-space points allowed into batch statistics.
            if len(student["vs"]) != 3 or any(points is not packet["viewspace_points"]
                                             for points, packet in zip(student["vs"], packages[key])):
                raise RuntimeError("tracking/statistics include a foreign or pseudo-view render")
        for key, other in (("a", "b"), ("b", "a")):
            parts[key]["c"] = self.consistency(pseudo[key], pseudo[other])
            parts[key]["loss"] = (parts[key]["lr"] + parts[key]["sr"] + parts[key]["reg"]
                                  + self.own.weight * parts[key]["c"])
        total = parts["a"]["loss"] + parts["b"]["loss"]
        if not bool(torch.isfinite(total)):
            raise FloatingPointError("Nonfinite co-training objective")
        total.backward()
        self.record["backward_calls"] += 1
        statistics = {key: cr.prefix.combine_batch_statistics(packages[key], torch) for key in KEYS}
        encoded, changed = {}, False
        for key, student in self.students.items():
            model, q = student["g"], parts[key]
            before, old_counts = int(model.get_xyz.shape[0]), dict(student["counts"])
            with self.activate(student):
                with torch.no_grad():
                    cr.topology(model, student["opt"], "fine", cursor, self.plan["data"]["cameras_extent"],
                                student["out"], statistics[key], args.prune_gate, student["counts"])
                    model.optimizer.step()
                    model.optimizer.zero_grad(set_to_none=True)
                student["steps"] += 1
                if student["affine"] is not None:
                    student["affine"].step(cursor)
                    student["affine_steps"] += 1
                if student["culler"] is not None and cursor % args.cull_interval == 0:
                    student["culls"].append(dict(cr.cull(model, student["culler"], student["log"], f"sr_{cursor}"),
                                                 cursor=cursor))
            value = float(q["c"].detach())
            student["consistency_values"].append(value)
            encoded[key] = dict(cursor=cursor, loss=float(q["loss"].detach()), LR_term=float(q["lr"].detach()),
                                SR_term=float(q["sr"].detach()), regularizer=float(q["reg"].detach()),
                                consistency=value, C_term=self.own.weight * value,
                                LR_L1=[float(v.detach()) for v in q["lr_values"]],
                                SR_L1=[float(v.detach()) for v in q["sr_values"]],
                                points_before=before, points_after=int(model.get_xyz.shape[0]),
                                keys=[list(key) for key in keys], pseudo=info, registered_views=3,
                                pseudo_views_in_statistics=0, seconds=time.time() - tick)
            student["log"].write(json.dumps(encoded[key], allow_nan=False) + "\n")
            changed |= old_counts != student["counts"] or before != int(model.get_xyz.shape[0])
            # Do not retain a preceding step's autograd graph through diagnostic state.
            student["vs"], student["terms"] = [], dict(lr=[], sr=[])
        if changed or cursor == 1:
            self.check_independence()
        self.record["calls"] += 1
        shared = dict(cursor=cursor, schedule_row=row, keys=[list(key) for key in keys], pseudo=info,
                      pseudo_lr_size=list(pseudo["a"].shape[-2:]), pseudo_gap=float((pseudo["a"] - pseudo["b"]).detach().abs().mean()),
                      students=encoded, seconds=time.time() - self.started)
        for student in self.students.values():
            student["pair_log"].write(json.dumps(shared, allow_nan=False) + "\n")
        if cursor % self.own.log_every == 0 or cursor == self.stop:
            self.record["students"] = self.student_summaries()
            self.write_record()
            print(json.dumps(dict(cotrain=cursor, consistency={key: value["consistency"] for key, value in encoded.items()},
                                  seconds=round(time.time() - self.started, 1))), flush=True)

    def student_summaries(self):
        result = {}
        for key, student in self.students.items():
            values = student["consistency_values"]
            result[key] = dict(steps=student["steps"], affine_steps=student["affine_steps"],
                               registered_renders=student["registered_renders"], consistency_calls=len(values),
                               consistency_first_100=float(self.np.mean(values[:100])) if values else None,
                               consistency_last_100=float(self.np.mean(values[-100:])) if values else None,
                               srfix_hits=student["hits"], counts=student["counts"],
                               drop=student["drop"] if self.drop_options else None,
                               time=student["time"] if self.time_state is not None else None,
                               probe=student["probe"] if self.probe_every else None,
                               checkpoint=student.get("checkpoint"))
        return result

    def finish(self):
        """The upstream chain wrote A's kit records; write B's corresponding records."""
        if set(self.students) != set(KEYS):
            raise RuntimeError("the modkit chain did not invoke the paired SR loop")
        second = self.students["b"]
        summaries = self.student_summaries()
        if self.record["calls"] != self.stop or self.record["backward_calls"] != self.stop or any(
                value["steps"] != self.stop or value["consistency_calls"] != self.stop
                or value["registered_renders"] != 3 * self.stop
                or value["affine_steps"] != (self.stop if self.sr_args.affine else 0)
                for value in summaries.values()):
            raise RuntimeError("lockstep update/render/consistency counts differ")
        affine_summary = second["affine"].summary() if second["affine"] else {}
        records = {}
        for name in ("srfix.json", "modkit.json", "modkit2.json", "modkit4.json"):
            record = copy.deepcopy(self.cr.read(self.outs["a"] / name))
            record["argv"] = replace_option(record["argv"], "--out", second["out"])
            record["argv"] = replace_option(record["argv"], "--arm", second["arm"])
            if name == "srfix.json":
                record["hits"] = second["hits"]
            elif name == "modkit.json":
                record.update(drop=summaries["b"]["drop"], probe=summaries["b"]["probe"])
            elif name == "modkit2.json":
                record.update(time=summaries["b"]["time"],
                              time_offsets_frames={cam: value["time_offset_frames"] for cam, value in affine_summary.items()
                                                   if "time_offset_frames" in value},
                              warp_field_rms_px={cam: value["warp_field_rms_px"] for cam, value in affine_summary.items()
                                                 if "warp_field_rms_px" in value})
            else:
                record["psf_sigma2_px2"] = {cam: value["psf_sigma2_px2"] for cam, value in affine_summary.items()
                                          if "psf_sigma2_px2" in value}
            self.rt.write_json(second["out"] / name, record)
            records[name] = {key: value for key, value in record.items() if key in ("switches", "degradation_function")}
        self.record.update(status="completed", complete=True, calls_equal_steps=True, students=summaries,
                           module_settings=records, seconds=time.time() - self.started)
        self.write_record()
        # Completion markers come last, after BOTH checkpoints and all kit records.
        for student in self.students.values():
            marker = student["out"] / "complete.json"
            self.rt.write_json(marker,
                               dict(student["record"], **student["result"], cotrain=dict(method=METHOD, co_pruning=False,
                                    calls=self.record["calls"], calls_equal_steps=True,
                                    peer=student["record"]["peer"], record=str(student["out"] / "cotrain.json"))))
            self.completion_written.append(marker)

    def fail(self):
        self.record.update(status="failed", complete=False, students=self.student_summaries(),
                           seconds=time.time() - self.started, error=traceback.format_exc())
        # Both output directories were empty at entry. Withdraw only markers
        # this invocation wrote if publishing the second marker failed.
        for path in self.completion_written:
            path.unlink(missing_ok=True)
        for path in self.outs.values():
            path.mkdir(parents=True, exist_ok=True)
            self.rt.write_json(path / f"failure_{time.time_ns()}.json", self.record)
        self.write_record()

    def close(self):
        if self.prefetch is not None:
            self.prefetch.close()
        for file in self.files:
            file.close()


@contextmanager
def patched_chain(pair):
    """Keep every upstream monkeypatch local, including exception paths."""
    cr, gaussian, native, kit1, kit2, kit4 = (
        pair.cr, pair.GaussianModel, pair.native, pair.kit1, pair.kit2, pair.kit4)
    objects = [(cr, ("run_sr", "CameraAffine", "configuration", "read_teacher", "topology")),
               (pair.srfix, ("main",)),
               (gaussian, ("restore", "update_learning_rate", "compute_regulation", "oneupSHdegree")),
               (native, ("hr_camera", "native_rgb", "native_effective_state")),
               (pair.losses, ("degradation", "lr_loss", "sr_loss")), (pair.loss_utils, ("ssim",)),
               (kit1, ("install_tracking", "install_drop", "install_grad_probe")), (kit2, ("install_time_hooks",))]
    attributes = [(obj, name, getattr(obj, name)) for obj, names in objects for name in names]
    dictionaries = [(obj, dict(obj)) for obj in (pair.srfix.STATE, kit2.REG, kit4.REG)]
    argv = sys.argv
    try:
        cr.run_sr = pair.run_sr
        kit1.install_tracking, kit1.install_drop = pair.install_tracking, pair.install_drop
        kit1.install_grad_probe, kit2.install_time_hooks = pair.install_grad_probe, pair.install_time_hooks
        yield
    finally:
        for obj, name, value in reversed(attributes):
            setattr(obj, name, value)
        for obj, value in dictionaries:
            obj.clear()
            obj.update(value)
        sys.argv = argv


def main(argv=None):
    own, kit, outs = parse_cli(sys.argv[1:] if argv is None else argv)
    for path in (CULL, MOD, VAR, GEO):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    import rt
    import srfix
    import torch
    import numpy as np
    import consistency_distill as cd
    import modkit4
    from scene.gaussian_model import GaussianModel
    import utils.loss_utils as loss_utils

    pair = PairLoop(own, kit, outs, rt, srfix, torch, np, cd, modkit4, GaussianModel, loss_utils)
    try:
        with patched_chain(pair):
            sys.argv = [str(GEO / "modkit4.py"), *kit]
            modkit4.main()
            pair.finish()
    except BaseException:
        try:
            pair.fail()
        except BaseException as error:
            print(f"could not save failure evidence: {error!r}", file=sys.stderr, flush=True)
        raise
    finally:
        pair.close()


if __name__ == "__main__":
    main()
