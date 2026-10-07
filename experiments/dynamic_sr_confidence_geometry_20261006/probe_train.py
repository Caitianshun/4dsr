"""Observe the unchanged finite LR-only trainer through Adam step hooks.

No loss, sampler, parameter routing or optimization rule is replaced. Every
existing Adam step delegates once to the original implementation, recording
group gradient RMS, actual parameter displacement, child tanh saturation and
selected exact frozen-state checks. Only the three 500-update cause probes use
this wrapper; source identity includes this file in checkpoint snapshots.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
import time
import traceback

import torch

from depth_prior import ROOT, sha256, write_json


def clone_value(value):
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: clone_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(clone_value(item) for item in value)
    return value


def exactly_equal(first, second):
    if torch.is_tensor(first):
        return torch.is_tensor(second) and torch.equal(first, second)
    if isinstance(first, dict):
        return isinstance(second, dict) and first.keys() == second.keys() and all(exactly_equal(first[key], second[key]) for key in first)
    if isinstance(first, (tuple, list)):
        return type(first) is type(second) and len(first) == len(second) and all(exactly_equal(a, b) for a, b in zip(first, second))
    return first == second


def saturation(parameter):
    value = parameter.detach().tanh().abs()
    return dict(abs_tanh_mean=float(value.mean()), abs_tanh_p95=float(torch.quantile(value.flatten(), .95)),
                fraction_abs_tanh_ge095=float((value >= .95).float().mean()),
                mean_tanh_derivative=float((1 - value.square()).mean()))


class AdamProbeRecorder:
    def __init__(self, output, method, parent_identity=None):
        self.output = Path(output)
        self.method = method
        self.parent_identity = parent_identity
        self.optimizers = {}
        self.counts = {}
        self.records = 0
        self.hook_seconds = 0.
        self.config_written = False

    def observe(self, optimizer, original_step, closure=None):
        if closure is not None:
            raise ValueError("Registered cause probes do not use optimizer closures")
        started = time.monotonic()
        key = id(optimizer)
        if key not in self.optimizers:
            self.optimizers[key] = len(self.optimizers)
            self.counts[key] = 0
        self.counts[key] += 1
        round_index = self.counts[key]
        exact_check = round_index in [1, 100, 500]
        groups = []
        before = {}
        frozen = {}
        raw_offset_before = None
        for group in optimizer.param_groups:
            parameters = group["params"]
            count = sum(p.numel() for p in parameters)
            gradient_square = 0.
            gradient_max = 0.
            active_count = 0
            for parameter in parameters:
                if parameter.grad is not None:
                    gradient = parameter.grad.detach()
                    if not torch.isfinite(gradient).all():
                        raise ValueError("Nonfinite active gradient in finite cause probe")
                    gradient_square += float(gradient.double().square().sum())
                    gradient_max = max(gradient_max, float(gradient.abs().max()))
                    active_count += parameter.numel()
                    before[id(parameter)] = parameter.detach().clone()
                elif exact_check:
                    frozen[id(parameter)] = (parameter.detach().clone(), clone_value(optimizer.state.get(parameter, {})))
            if group.get("name") == "offset_raw":
                raw_offset_before = saturation(parameters[0])
            groups.append(dict(name=group.get("name", "unnamed"), parameter_elements=count,
                active_gradient_elements=active_count, grad_rms=math.sqrt(gradient_square / max(count, 1)),
                grad_maxabs=gradient_max, lr=float(group["lr"])))
        before_hook = time.monotonic() - started
        step_started = time.monotonic()
        result = original_step(optimizer, closure)
        step_seconds = time.monotonic() - step_started
        after_started = time.monotonic()
        for group, row in zip(optimizer.param_groups, groups):
            displacement_square = 0.
            displacement_max = 0.
            baseline_square = 0.
            frozen_count = 0
            for parameter in group["params"]:
                if id(parameter) in before:
                    old = before[id(parameter)]
                    displacement = parameter.detach() - old
                    displacement_square += float(displacement.double().square().sum())
                    displacement_max = max(displacement_max, float(displacement.abs().max()))
                    baseline_square += float(old.double().square().sum())
                elif id(parameter) in frozen:
                    old_parameter, old_state = frozen[id(parameter)]
                    if not torch.equal(old_parameter, parameter.detach()) or not exactly_equal(old_state, optimizer.state.get(parameter, {})):
                        raise AssertionError("Frozen parameter/Adam state advanced in LR-only intervention")
                    frozen_count += 1
            row.update(actual_delta_rms=math.sqrt(displacement_square / max(row["parameter_elements"], 1)),
                actual_delta_maxabs=displacement_max,
                relative_delta_rms=math.sqrt(displacement_square / max(baseline_square, 1e-24)) if baseline_square else 0.,
                exact_frozen_parameters_checked=frozen_count if exact_check else None)
            if group.get("name") == "offset_raw":
                row["children_tanh_before"] = raw_offset_before
                row["children_tanh_after"] = saturation(group["params"][0])
        hook_seconds = before_hook + time.monotonic() - after_started
        self.hook_seconds += hook_seconds
        self.records += 1
        if not self.config_written:
            write_json(self.output / "probe_hook_config.json", dict(method=self.method,
                source_sha256=sha256(__file__), trainer_sha256=sha256(Path(__file__).with_name("train.py")),
                parent=self.parent_identity, loss_sampler_or_routing_changed=False,
                optimizer_contract="original Adam.step invoked exactly once per existing optimizer call; closure absent",
                exact_frozen_checks=[1, 100, 500], group_gradient_RMS="sqrt(sum grad² / all elements in group; absent grads count zero)",
                actual_displacement_RMS="sqrt(sum (parameter_after-step - parameter_before-step)² / all elements in group)",
                observation_overhead_in_training_wall_time=True))
            self.config_written = True
        record = dict(method=self.method, suffix_step=round_index, optimizer_index=self.optimizers[key],
            groups=groups, original_adam_step_seconds=step_seconds, observation_hook_seconds=hook_seconds,
            exact_frozen_state_checked=exact_check)
        with (self.output / "probe_optimizer_observations.jsonl").open("a", buffering=1) as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        return result

    def finalize(self, expected_steps):
        if len(self.optimizers) != 2 or sorted(self.counts.values()) != [expected_steps, expected_steps]:
            raise AssertionError("Cause probe must execute exactly two existing Adam calls per round")
        write_json(self.output / "probe_hook_complete.json", dict(status="passed", method=self.method,
            optimizer_calls=self.records, counts=list(self.counts.values()), parameter_updates=expected_steps,
            source_sha256=sha256(__file__), observation_hook_seconds=self.hook_seconds,
            exact_frozen_parameter_and_state_checks_passed=True, formulas_or_routing_changed=False))


def self_check(directory):
    """A CPU scalar Adam check, never a scene-training or GPU experiment."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    active = torch.nn.Parameter(torch.tensor([1., 2.]))
    frozen = torch.nn.Parameter(torch.tensor([3., 4.]))
    offset = torch.nn.Parameter(torch.tensor([[3., 0., -3.]]))
    optimizer = torch.optim.Adam([dict(params=[active], name="active"), dict(params=[frozen], name="frozen"),
                                  dict(params=[offset], name="offset_raw")], lr=.01)
    # Inherit a real populated Adam state, then freeze one tensor without reset.
    (active.square().sum() + frozen.square().sum() + offset.square().sum()).backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    frozen.requires_grad_(False)
    before_frozen = frozen.detach().clone()
    before_state = clone_value(optimizer.state[frozen])
    before_active = active.detach().clone()
    (active.square().sum() + offset.square().sum()).backward()
    recorder = AdamProbeRecorder(directory, "CPU_mechanism_check")
    recorder.observe(optimizer, torch.optim.Adam.step)
    assert torch.equal(frozen.detach(), before_frozen) and exactly_equal(before_state, optimizer.state[frozen])
    row = json.loads((directory / "probe_optimizer_observations.jsonl").read_text().splitlines()[-1])
    active_report = next(g for g in row["groups"] if g["name"] == "active")
    measured = float((active.detach() - before_active).double().square().mean().sqrt())
    assert abs(active_report["actual_delta_rms"] - measured) < 1e-12
    write_json(directory / "cpu_hook_check.json", dict(status="passed", inherited_frozen_state_unchanged=True,
        active_parameter_displacement_rms=measured, recorded_rms=active_report["actual_delta_rms"],
        gpu_used=False, scene_training_updates=0, source_sha256=sha256(__file__)))


def main(args):
    if args.self_check is not None:
        self_check(args.self_check)
        return
    if args.method not in ["P_joint", "P_xyz", "P_SH"] or args.stop != 500 or str(args.repeat) != "1":
        raise ValueError("This wrapper only supports the three preregistered repeat1 500-update LR-only probes")
    path = Path(__file__).with_name("train.py")
    spec = importlib.util.spec_from_file_location("cg_observed_cause_trainer", path)
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    source_identity = trainer.source_identity
    # Registration-only extension: original trainer checkpoint stores/snapshots
    # the wrapper, so no additional code is omitted from exact restore audits.
    def observed_identity():
        identity = source_identity()
        identity[str(Path(__file__).relative_to(ROOT))] = sha256(__file__)
        return identity
    trainer.source_identity = observed_identity
    protocol = trainer.read(trainer.OUT / "protocol.json")
    recorder = AdamProbeRecorder(args.out, args.method, protocol["parent"])
    original_step = torch.optim.Adam.step
    def observed_step(optimizer, closure=None):
        return recorder.observe(optimizer, original_step, closure)
    torch.optim.Adam.step = observed_step
    try:
        trainer.main(args)
        recorder.finalize(500)
    finally:
        torch.optim.Adam.step = original_step


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["P_joint", "P_xyz", "P_SH"])
    parser.add_argument("--repeat", default="1")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--stop", type=int, default=500)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--self-check", type=Path)
    args = parser.parse_args()
    if args.self_check is None and (args.method is None or args.out is None):
        parser.error("--method and --out required for scene probes")
    try:
        main(args)
    except BaseException:
        if args.out is not None and args.out.exists():
            write_json(args.out / "probe_hook_failed.json", dict(status="failed", traceback=traceback.format_exc()))
        raise
