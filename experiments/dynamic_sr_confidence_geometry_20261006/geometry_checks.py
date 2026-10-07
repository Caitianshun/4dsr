"""Mechanism checks for G; CPU checks do not claim CUDA renderer validation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import torch

from depth_prior import (NUMERICS, geometry_loss, normalize_moments,
                         prepare_geometry_target, render_moments,
                         robust_normalize, sample_pairs, sha256, write_json)


def cpu_checks():
    torch.set_num_threads(4)
    checks = {}
    # Equal-pixel depth averaging would give 3, but alpha-weighted depth is 3.5.
    alpha = torch.tensor([[.25, .75]], dtype=torch.float64)
    depth = torch.tensor([[2., 4.]], dtype=torch.float64)
    moments = torch.stack((alpha, alpha * depth, alpha * depth.square()))
    result = normalize_moments(moments, (1, 1))
    assert float(result["expected_z"]) == 3.5
    assert float(result["variance_z_raw"]) == .75
    assert float(result["moments"][2]) == 6.5  # no RGB-range clamp
    assert float(result["alpha"]) == .5
    checks["area_before_division"] = dict(status="passed", z=3.5, variance=.75,
        wrong_depth_average=3., unclamped_second_moment=6.5)

    # Explicit cancellation check preserves raw negative variance for reporting.
    numerical = normalize_moments(torch.tensor([[[1.]], [[2.]], [[3.999999]]]))
    assert float(numerical["variance_z_raw"]) < 0
    assert float(numerical["variance_z"]) == 0
    checks["raw_variance_preserved"] = dict(status="passed", raw=float(numerical["variance_z_raw"]))

    q = torch.arange(1., 17., dtype=torch.float64).reshape(4, 4).requires_grad_()
    normalized, stats = robust_normalize(q)
    assert all(not v.requires_grad for v in stats.values())
    normalized[1, 1].backward()
    expected = torch.zeros_like(q)
    expected[1, 1] = 1 / stats["scale"]
    assert torch.equal(q.grad, expected)
    constant = torch.full((4, 4), 2000.)
    _, constant_stats = robust_normalize(constant)
    assert float(constant_stats["scale"]) == 2.
    checks["detached_median_iqr"] = dict(status="passed", scale=float(stats["scale"]),
        constant_scale=float(constant_stats["scale"]), off_pixel_gradients=0.)

    global_rng = torch.get_rng_state().clone()
    pairs = sample_pairs((96, 128), seed=2026100603)
    assert torch.equal(global_rng, torch.get_rng_state())
    assert torch.equal(pairs, sample_pairs((96, 128), seed=2026100603))
    first, second = pairs.unbind(1)
    dx = first % 128 - second % 128
    dy = first // 128 - second // 128
    distance = (dx.square() + dy.square()).float().sqrt()
    assert len(pairs) == 4096 and float(distance.min()) >= 4. and float(distance.max()) <= 80.
    checks["fixed_pairs"] = dict(status="passed", n=4096, min_distance=float(distance.min()),
        max_distance=float(distance.max()), training_rng_unchanged=True)

    teacher = torch.tensor([[4., 3., 2., 1.], [4., 3., 2., 1.],
                            [4., 3., 2., 1.], [4., 3., 2., 1.]])
    parent_alpha = torch.ones_like(teacher)
    parent_alpha.flatten()[3] = 0.
    table = torch.tensor([[0, 2], [0, 4], [0, 3]])
    target = prepare_geometry_target(teacher, [teacher.clone(), -teacher.clone()],
                                     parent_alpha, pairs=table)
    assert target["weights"][0] > 0 and target["weights"][0] < 1
    assert target["weights"][1] == 0  # teacher tie
    assert target["weights"][2] == 0  # frozen parent alpha unsupported
    inverse = (-teacher).clone().requires_grad_()
    packet = dict(inverse_depth=inverse, alpha=torch.zeros_like(teacher))
    loss, _ = geometry_loss(packet, target)
    n, _ = robust_normalize(inverse, target["frozen_support"])
    expected_loss = target["weights"][0] * torch.relu(.1 - target["signs"][0] * (n.flatten()[0] - n.flatten()[2])) / 3.
    assert torch.allclose(loss, expected_loss)
    assert float(loss) > 0  # decreasing current alpha cannot remove fixed supervision
    loss.backward()
    assert inverse.grad is not None and float(inverse.grad.abs().max()) > 0
    checks["ranking_and_fixed_support"] = dict(status="passed", loss=float(loss),
        weight=float(target["weights"][0]), denominator=3, ties_zero=True,
        frozen_alpha_zero_pairs_excluded=True, current_alpha_does_not_gate=True,
        gradient_nonzero=True)
    return checks


def gpu_checks(manifest_path, checkpoint):
    """Small real-rasterizer fixture; no training and no parameter updates."""
    import copy
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "experiments/dynamic_sr_prior_guidance_20260927"))
    from shared import evaluator, load_manifest
    manifest = load_manifest(manifest_path)
    observation = next(o for o in manifest["observations"] if o["split"] == "train")
    original = evaluator().render_camera(manifest, observation, 0)
    camera = copy.copy(original)
    camera.image_height = camera.image_width = 64
    inv = torch.linalg.inv(camera.world_view_transform.cuda())
    local = torch.tensor([[.04, .025, 2., 1.], [.02, -.01, 4., 1.]], device="cuda")
    xyz = (local @ inv)[:, :3].detach().requires_grad_()
    covariance = (torch.eye(3, device="cuda")[None].repeat(2, 1, 1) * .04).requires_grad_()
    opacity = torch.tensor([[.5], [.4]], device="cuda", requires_grad=True)
    output = render_moments(camera, xyz, covariance, opacity, (16, 16))
    assert output["axial_z"].requires_grad and output["moments"].requires_grad
    assert torch.allclose(output["hr_moments"][1], output["native_depth"].squeeze(), atol=2e-5, rtol=0)
    # The original 5:11 LR fixture crossed native alpha cutoffs (including empty
    # rays), so ±1e-3 finite differences changed support and were not a smooth
    # derivative test. Preserve that failure; use a supported central footprint.
    window = slice(7, 9)
    mesh = torch.linspace(-.4, .4, 2, device="cuda")
    weight = 1 + mesh[None, :]
    assert float(output["alpha"][window, window].min()) > .1
    loss = (output["expected_z"][window, window] * weight).mean()
    gradients = torch.autograd.grad(loss, (xyz, covariance, opacity), allow_unused=True, retain_graph=True)
    assert gradients[0] is not None and torch.isfinite(gradients[0]).all()
    assert float(gradients[0].norm()) > 0
    assert gradients[1] is None and gradients[2] is None
    # Compare the center coordinate derivative via smooth finite differences.
    fd_rows = []
    for point in [0]:
        for axis in range(3):
            plus = xyz.detach().clone()
            minus = xyz.detach().clone()
            plus[point, axis] += .001
            minus[point, axis] -= .001
            plus_result = render_moments(camera, plus, covariance, opacity, (16, 16))
            minus_result = render_moments(camera, minus, covariance, opacity, (16, 16))
            support_changes = int(((plus_result["hr_moments"][0, 28:36, 28:36] > 0) !=
                                   (minus_result["hr_moments"][0, 28:36, 28:36] > 0)).sum())
            assert support_changes == 0, (axis, support_changes)
            positive = (plus_result["expected_z"][window, window] * weight).mean()
            negative = (minus_result["expected_z"][window, window] * weight).mean()
            fd = float((positive - negative) / .002)
            analytic = float(gradients[0][point, axis])
            error = abs(fd - analytic)
            assert error <= max(.01, .03 * abs(fd)), (axis, analytic, fd)
            fd_rows.append(dict(point=point, axis=axis, analytic=analytic, finite_difference=fd,
                                absolute_error=error, hr_support_changes_in_test_window=support_changes))
    # Verify the actual ranking loss uses that same live xyz while frozen parent
    # support/teacher ordering and detached statistics do not open other paths.
    support_alpha = torch.zeros_like(output["alpha"])
    support_alpha[window, window] = 1.
    target = prepare_geometry_target(-output["inverse_depth"].detach(), [], support_alpha,
                                     pairs=torch.tensor([[7 * 16 + 7, 8 * 16 + 8]]))
    ranking, _ = geometry_loss(output, target)
    ranking_gradients = torch.autograd.grad(ranking, (xyz, covariance, opacity), allow_unused=True)
    assert ranking_gradients[0] is not None and torch.isfinite(ranking_gradients[0]).all()
    assert float(ranking_gradients[0].norm()) > 0
    assert ranking_gradients[1] is None and ranking_gradients[2] is None
    return dict(status="passed", actual_cuda_renderer=True, gpu=torch.cuda.get_device_name(),
        xyz_gradient_norm=float(gradients[0].norm()), covariance_gradient=None, opacity_gradient=None,
        native_matches_unclamped_M1=True, finite_differences=fd_rows,
        ranking_loss=float(ranking), ranking_xyz_gradient_norm=float(ranking_gradients[0].norm()),
        fixture_window_lr=[7, 9], fixture_alpha_min=float(output["alpha"][window, window].min()),
        historical_failure="5:11 window included empty LR ray; eps1e-3 changed LR support and z jumped. See preserved failure and geometry_fixture_failure_analysis.json; tolerance unchanged.",
        checkpoint_sha256=sha256(checkpoint) if checkpoint is not None else None,
        note="CUDA synthetic moments only; parent-specific 32-observation G calibration is separate.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    if args.gpu and args.manifest is None:
        parser.error("--gpu requires --manifest")
    started = time.monotonic()
    report = dict(status="passed", cpu=cpu_checks(), numerics=NUMERICS,
        gpu=gpu_checks(args.manifest, args.checkpoint) if args.gpu else dict(status="not_run"),
        torch=str(torch.__version__), source_sha256=sha256(__file__),
        depth_prior_sha256=sha256(Path(__file__).with_name("depth_prior.py")), seconds=time.monotonic() - started)
    write_json(args.out, report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
