"""CPU acceptance of source semantics and protocol edge cases, not GPU approval.

Run from project root:
  CUDA_VISIBLE_DEVICES='' /home/cai_tianshun/Project/4dgs/.venv/bin/python -m \
    unittest experiments.static_sr_wu_20261022.methods.iesrgs_cpu_tests -v
"""
from __future__ import annotations

import importlib.util
import math
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import torch
import torch.nn.functional as F

from .iesrgs import (IESRGSConfig, IESRGSPriors, SameTimeViewSampler,
                     discrepancy_mask, fusion_loss, internal_multiview_loss,
                     pearson_loss, ssim)
from .iesrgs_renderer import apply_3d_filter, compute_3d_filter, render_state
from .iesrgs_priors import external_tensors, prepare_internal
from ..common import identity


class IESRGSCPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(131)

    def priors(self):
        return IESRGSPriors(torch.rand(3, 17, 19) + .01, torch.rand(1, 17, 19) + 2,
                            torch.rand(3, 17, 19), torch.rand(1, 17, 19))

    def test_released_mask_semantics_and_zero(self):
        internal = torch.tensor([.5, .1, 0., 0.]).expand(3, 2, 4).clone()
        external = torch.tensor([.6, .3, 0., .5]).expand(3, 2, 4).clone()
        expected = (torch.abs((external - internal) / internal) >= .9).float()
        actual = discrepancy_mask(internal, external)
        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(torch.isfinite(actual).all())
        self.assertEqual(actual[0, 0].tolist(), [0., 1., 0., 1.])

    def test_complete_loss_and_gradient_match_released_equations(self):
        prior = self.priors()
        rendered = torch.rand(3, 17, 19, requires_grad=True)
        depth = torch.rand(1, 17, 19, requires_grad=True) + 2
        output = fusion_loss(rendered, depth, prior)
        mask = (torch.abs((prior.external_rgb - prior.internal_rgb) / prior.internal_rgb) >= .9).float()
        l1 = ((rendered - prior.internal_rgb).abs() * mask).mean()
        l1 += ((rendered - prior.external_rgb).abs() * (1 - mask)).mean()
        texture = .8 * l1 + .2 * (1 - ssim(rendered, prior.external_rgb))
        texture += .2 * (1 - ssim(rendered, prior.internal_rgb))
        a = depth.flatten() - depth.mean()
        b = -prior.external_inverse_depth.flatten() + prior.external_inverse_depth.mean()
        pearson = 1 - (a * b).sum() / (a.square().sum() * b.square().sum()).sqrt()
        expected = texture + .001 * (depth - prior.internal_depth).abs().mean() + .0001 * pearson
        torch.testing.assert_close(output["total"], expected)
        expected_grads = torch.autograd.grad(expected, (rendered, depth), retain_graph=True)
        actual_grads = torch.autograd.grad(output["total"], (rendered, depth))
        for actual, expected in zip(actual_grads, expected_grads):
            torch.testing.assert_close(actual, expected)
            self.assertTrue(torch.isfinite(actual).all())
            self.assertGreater(actual.abs().sum().item(), 0)

    def test_ssim_matches_pinned_source(self):
        source = Path(__file__).resolve().parents[3] / "tmp/static_sr_wu_20261022/upstream/iesrgs/utils/loss_utils.py"
        self.assertTrue(source.exists(), "pinned source checkout required for parity acceptance")
        spec = importlib.util.spec_from_file_location("iesrgs_reference_losses", source)
        reference = importlib.util.module_from_spec(spec)
        previous = sys.dont_write_bytecode
        sys.dont_write_bytecode = True
        try:
            spec.loader.exec_module(reference)
        finally:
            sys.dont_write_bytecode = previous
        a, b = torch.rand(3, 17, 19), torch.rand(3, 17, 19)
        torch.testing.assert_close(ssim(a, b), reference.ssim(a, b), atol=2e-6, rtol=2e-5)

    def test_teacher_gradients_are_frozen(self):
        prior = self.priors()
        for value in vars(prior).values():
            value.requires_grad_(True)
        output = fusion_loss(torch.rand(3, 17, 19, requires_grad=True),
                             torch.rand(1, 17, 19, requires_grad=True), prior)
        output["total"].backward()
        self.assertTrue(all(value.grad is None for value in vars(prior).values()))

    def test_negative_inverse_depth_orientation(self):
        depth = torch.arange(1., 9.)
        inverse = 10 - depth
        self.assertAlmostEqual(float(pearson_loss(depth, -inverse)), 0., places=6)
        self.assertAlmostEqual(float(pearson_loss(depth, inverse)), 2., places=6)
        with self.assertRaisesRegex(ValueError, "constant"):
            pearson_loss(depth, torch.ones_like(depth))

    def test_unmasked_dssim_and_three_view_sum(self):
        target = torch.rand(3, 13, 15)
        rendered = torch.rand(3, 13, 15, requires_grad=True)
        single = .8 * (target - rendered).abs().mean() + .2 * (1 - ssim(rendered, target))
        torch.testing.assert_close(internal_multiview_loss([rendered] * 3, [target] * 3), 3 * single)
        with self.assertRaises(ValueError):
            internal_multiview_loss([rendered], [target])
        depth = torch.arange(13 * 15.).reshape(1, 13, 15)
        # Internal and external equal -> mask zero; source still adds both DSSIM.
        value = fusion_loss(rendered, depth, IESRGSPriors(target, depth, target, 500 - depth))
        torch.testing.assert_close(value["texture"], .8 * (target - rendered).abs().mean() + .4 * (1 - ssim(rendered, target)))

    def test_sampler_scene_and_time_isolation(self):
        rows = [dict(scene_id="scene", frame_id=frame, camera_id=f"cam{cam:02}", split="train")
                for frame in (0, 1) for cam in range(1, 8)]
        a, b = SameTimeViewSampler(rows), SameTimeViewSampler(rows)
        for _ in range(40):
            selected = a.sample()
            self.assertEqual(selected, b.sample())
            self.assertEqual(len({row["frame_id"] for row in selected}), 1)
            self.assertEqual(len({row["camera_id"] for row in selected}), 3)
        for illegal in [dict(rows[0], split="test"), dict(rows[0], scene_id="foreign")]:
            with self.assertRaises(ValueError):
                SameTimeViewSampler(rows + [illegal])

    def camera(self, time=0.):
        return SimpleNamespace(image_width=100, image_height=80, FoVx=math.pi / 2,
                               FoVy=math.pi / 2, world_view_transform=torch.eye(4),
                               full_proj_transform=torch.eye(4), camera_center=torch.zeros(3), time=time)

    def test_mip_filter_and_mass_compensation(self):
        xyz = torch.tensor([[0., 0., 2.], [0., 0., 4.], [100., 0., 3.]])
        filt = compute_3d_filter(xyz, [self.camera()])
        torch.testing.assert_close(filt[:, 0], torch.tensor([2., 4., 4.]) / 50 * math.sqrt(.2))
        scales, opacity = torch.ones(3, 3) * .2, torch.ones(3, 1) * .8
        filtered, alpha = apply_3d_filter(scales, opacity, filt)
        torch.testing.assert_close(filtered.prod(-1) * alpha[:, 0], scales.prod(-1) * opacity[:, 0])
        self.assertTrue((filtered > scales).all())
        with self.assertRaisesRegex(ValueError, "no Gaussian visible"):
            compute_3d_filter(torch.tensor([[0., 0., -2.]]), [self.camera()])

    def test_rade_adapter_contract_and_timestamp_rejection(self):
        capture = {}
        class Raster:
            def __init__(self, raster_settings):
                capture["settings"] = raster_settings
            def __call__(self, **kw):
                capture["kw"] = kw
                return (torch.ones(3, 80, 100), torch.ones(2), *(torch.ones(1, 80, 100) for _ in range(6)))
        api = SimpleNamespace(GaussianRasterizationSettings=lambda **kw: kw, GaussianRasterizer=Raster)
        state = dict(xyz=torch.tensor([[0., 0., 2.], [0., 0., 3.]], requires_grad=True),
                     scales=torch.ones(2, 3), rotation=torch.ones(2, 4),
                     opacity=torch.ones(2, 1), sh=torch.ones(2, 16, 3))
        packet = render_state(self.camera(), state, torch.zeros(3), 3, [self.camera()], api=api)
        self.assertIn("median_depth", packet)
        self.assertTrue(capture["settings"]["require_depth"])
        self.assertEqual(capture["settings"]["kernel_size"], .1)
        self.assertIs(capture["kw"]["means3D"], state["xyz"])
        with self.assertRaisesRegex(ValueError, "timestamp"):
            render_state(self.camera(1.), state, torch.zeros(3), 3, [self.camera()], api=api)

    def test_external_preprocessing_path(self):
        capture = {}
        class SwinStub:
            def __call__(self, x):
                capture["padded"] = x.shape
                return F.interpolate(x, scale_factor=4, mode="nearest")
        class DepthStub:
            def infer_image(self, bgr, input_size):
                capture["input_size"] = input_size
                capture["bgr"] = bgr
                return np.arange(bgr.shape[0] * bgr.shape[1], dtype=np.float32).reshape(bgr.shape[:2])
        lr = torch.rand(3, 8, 9)
        sr, depth = external_tensors(lr, SwinStub(), DepthStub())
        self.assertEqual(capture["padded"], (1, 3, 16, 16))
        self.assertEqual(tuple(sr.shape), (3, 32, 36))
        self.assertEqual(capture["input_size"], 518)
        self.assertTrue(np.array_equal(capture["bgr"], (sr.permute(1, 2, 0).numpy() * 255).astype(np.uint8)[:, :, ::-1]))
        self.assertAlmostEqual(float(depth.max()), 255 / 65536, places=8)

    def test_fixture_checkpoint_cannot_produce_internal_priors(self):
        with tempfile.TemporaryDirectory(prefix="iesrgs_cpu_") as directory:
            checkpoint = Path(directory) / "fixture.pt"
            torch.save(dict(metadata=dict(stage="iesrgs_internal", iterations=3, optimizer_updates=3)), checkpoint)
            item = identity(checkpoint)
            backend = SimpleNamespace(loaded_checkpoint_identity=item)
            with self.assertRaisesRegex(ValueError, "wiring fixtures"):
                prepare_internal(backend, item, Path(directory) / "must_not_exist")
            self.assertFalse((Path(directory) / "must_not_exist").exists())


if __name__ == "__main__":
    unittest.main()
