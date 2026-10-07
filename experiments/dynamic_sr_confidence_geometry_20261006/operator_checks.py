"""Numerical acceptance for the actual-degradation null-space operator.

Run on a reserved GPU to record its cost, e.g.:
python -m experiments.dynamic_sr_confidence_geometry_20261006.operator_checks \
  --device cuda --hr-size 512 640 --lr-size 128 160 --out operator_checks.json
CPU output does not claim CUDA acceptance or GPU timings.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import torch
from torch.nn import functional as F

try:
    from .degradation_operator import ActualDegradation
except ImportError:
    # Avoid collision with Python's standard-library module named operator.
    import importlib.util
    spec = importlib.util.spec_from_file_location("sr_actual_degradation", Path(__file__).with_name("degradation_operator.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    ActualDegradation = module.ActualDegradation


def _rms(x):
    return float(x.double().square().mean().sqrt())


def _relative(x, reference):
    return _rms(x) / max(_rms(reference), 1e-30)


def _sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def _check_shape(hr_size, lr_size, dtype, device, seed):
    op = ActualDegradation(hr_size, lr_size, device=device, dtype=dtype)
    generator = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn((3, *hr_size), generator=generator, device=device, dtype=dtype)
    y = torch.randn((3, *lr_size), generator=generator, device=device, dtype=dtype)
    z = torch.randn(x.shape, generator=generator, device=device, dtype=dtype)
    exact = F.interpolate(x[None], size=lr_size, mode="bicubic",
                          align_corners=False, antialias=True)[0]
    d = op.D0(x)
    q = op.Q(x)
    dot_left = (d * y).double().sum()
    dot_right = (x * op.transpose(y)).double().sum()
    denominator = max(float(d.double().norm() * y.double().norm()), 1e-30)
    symmetry_denominator = max(float(q.double().norm() * z.double().norm()), 1e-30)
    cond = math.prod(axis["gram_condition"] for axis in op.metadata()["axes"].values())
    eps = torch.finfo(dtype).eps
    tol = 64 * eps * max(cond, 1.)
    border = torch.zeros_like(x)
    border[:, 0, 0] = 1
    border[:, -1, -1] = -1
    border[:, 0, -1] = 2
    border[:, -1, 0] = -2
    border_exact = F.interpolate(border[None], size=lr_size, mode="bicubic",
                                 align_corners=False, antialias=True)[0]
    c = torch.rand((1, *hr_size), generator=generator, device=device, dtype=dtype)
    train_x = x.detach().clone().requires_grad_(True)
    teacher = torch.randn(x.shape, generator=generator, device=device, dtype=dtype)
    detail_loss = op.detail_loss(train_x, teacher, c)
    g, = torch.autograd.grad(detail_loss, train_x)
    manual = op.Q(c * op.Q(train_x.detach() - teacher).sign()) / train_x.numel()
    weighted_inside_x = x.detach().clone().requires_grad_(True)
    wrong_loss = op.Q(c * (weighted_inside_x - teacher)).abs().mean()
    wrong_g, = torch.autograd.grad(wrong_loss, weighted_inside_x)
    b = x[None].repeat(2, 1, 1, 1)
    measurements = {
        "interpolate_relative_rms": _relative(d - exact, exact),
        "interpolate_max_absolute": float((d - exact).abs().max()),
        "border_pulses_relative_rms": _relative(op.D0(border) - border_exact, border_exact),
        "adjoint_normalized_inner_error": float((dot_left - dot_right).abs()) / denominator,
        "Q_self_adjoint_normalized_inner_error": float(((q * z).double().sum()
                - (x * op.Q(z)).double().sum()).abs()) / symmetry_denominator,
        "Q_idempotence_relative_rms": _relative(op.Q(q) - q, q),
        "D0Q_relative_rms_to_D0x": _relative(op.D0(q), d),
        "D0pinv_identity_relative_rms": _relative(op.D0(op.pinv(y)) - y, y),
        "detail_gradient_D0_leakage_to_gradient_rms": _relative(op.D0(g), g),
        "detail_gradient_manual_vjp_relative_rms": _relative(g - manual, manual),
        "NCHW_equivalence_relative_rms": _relative(op.D0(b)[0] - d, d),
        "D_clamp_matches_reference_relative_rms": _relative(op.D(x) - exact.clamp(0, 1), exact),
    }
    # The negative control is recorded as evidence, not a criterion for Q quality.
    negative_control = {"weighted_inside_Q_gradient_leakage_to_gradient_rms":
                        _relative(op.D0(wrong_g), wrong_g)}
    passed = all(math.isfinite(value) and value <= tol for value in measurements.values())
    return {"operator": op.metadata(), "tolerance": {
                "formula": "64 * dtype_epsilon * max(1, cond(Gram_h)*cond(Gram_w))",
                "registered_before_run": True, "value": tol},
            "measurements": measurements, "negative_control": negative_control,
            "passed": passed}


def _finite_difference(device):
    # f64 directional differences isolate correctness from f32 cancellation.
    op = ActualDegradation((19, 27), (5, 7), device=device, dtype=torch.float64)
    gen = torch.Generator(device=device).manual_seed(2026100644)
    x = torch.randn((3, 19, 27), generator=gen, device=device,
                    dtype=torch.float64, requires_grad=True)
    teacher = torch.randn(x.shape, generator=gen, device=device, dtype=x.dtype)
    c = torch.rand((1, 19, 27), generator=gen, device=device, dtype=x.dtype)
    direction = torch.randn(x.shape, generator=gen, device=device, dtype=x.dtype)
    direction = direction / direction.square().mean().sqrt()
    step = 1e-6
    loss = op.detail_loss(x, teacher, c)
    g, = torch.autograd.grad(loss, x)
    analytical = float((g * direction).sum())
    with torch.no_grad():
        numerical = float((op.detail_loss(x + step * direction, teacher, c)
                           - op.detail_loss(x - step * direction, teacher, c)) / (2 * step))
    abs_error = abs(numerical - analytical)
    # Absolute error criterion handles a directional derivative accidentally near 0.
    return {"dtype": "float64", "hr_size": [19, 27], "lr_size": [5, 7],
            "step": step, "analytical": analytical, "numerical": numerical,
            "absolute_error": abs_error, "absolute_tolerance": 2e-7,
            "passed": math.isfinite(abs_error) and abs_error < 2e-7}


def _benchmark(hr_size, lr_size, dtype, device, repeats):
    _sync(device)
    start = time.perf_counter()
    op = ActualDegradation(hr_size, lr_size, device=device, dtype=dtype)
    _sync(device)
    construction_seconds = time.perf_counter() - start
    x = torch.rand((3, *hr_size), device=device, dtype=dtype, requires_grad=True)
    if torch.device(device).type == "cuda":
        baseline = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
    else:
        baseline = None
    for _ in range(2):
        op.Q(x).square().mean().backward()
        x.grad = None
    _sync(device)
    start = time.perf_counter()
    for _ in range(repeats):
        op.Q(x)
    _sync(device)
    forward_ms = 1000 * (time.perf_counter() - start) / repeats
    start = time.perf_counter()
    for _ in range(repeats):
        op.Q(x).square().mean().backward()
        x.grad = None
    _sync(device)
    combined_ms = 1000 * (time.perf_counter() - start) / repeats
    storage_bytes = 0
    for buf in op.buffers():
        if buf.layout == torch.sparse_coo:
            storage_bytes += buf.values().numel() * buf.values().element_size()
            storage_bytes += buf.indices().numel() * buf.indices().element_size()
        else:
            storage_bytes += buf.numel() * buf.element_size()
    return {"hr_size": list(hr_size), "lr_size": list(lr_size),
            "dtype": str(dtype), "device": str(device), "repeats": repeats,
            "construction_seconds": construction_seconds,
            "Q_forward_ms": forward_ms, "Q_forward_backward_ms": combined_ms,
            "module_storage_bytes": storage_bytes,
            "cuda_peak_incremental_allocated_bytes": None if baseline is None else
                torch.cuda.max_memory_allocated(device) - baseline,
            "gpu_cost_verified": baseline is not None,
            "excludes_rgb_render_cost": True}


def run_checks(hr_size=(128, 192), lr_size=(32, 48), *, device="cpu", repeats=5):
    cases = []
    sizes = [((7, 11), (2, 3)), ((32, 48), (8, 12)),
             ((35, 51), (9, 13)), (tuple(hr_size), tuple(lr_size))]
    for dtype in (torch.float64, torch.float32):
        for index, (hr, lr) in enumerate(sizes):
            cases.append(_check_shape(hr, lr, dtype, device, 2026100600 + index))
    finite_difference = _finite_difference(device)
    benchmark = _benchmark(hr_size, lr_size, torch.float32, device, repeats)
    return {"schema": 1, "torch_version": torch.__version__, "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device) if torch.device(device).type == "cuda" else None,
            "random_generators": "local generators; global training RNG untouched",
            "cases": cases, "finite_difference": finite_difference, "benchmark": benchmark,
            "passed": all(item["passed"] for item in cases) and finite_difference["passed"],
            "scope": "synthetic engineering acceptance; no method-quality claim"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--hr-size", nargs=2, type=int, default=(128, 192))
    parser.add_argument("--lr-size", nargs=2, type=int, default=(32, 48))
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    report = run_checks(args.hr_size, args.lr_size, device=args.device, repeats=args.repeats)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"passed": report["passed"], "out": str(args.out),
                      "benchmark": report["benchmark"]}, ensure_ascii=False))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
