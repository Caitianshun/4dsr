"""CPU scientific parity / isolation checks; no CUDA imports or model training.

Run from the project root:
 CUDA_VISIBLE_DEVICES='' <torch-python> -m unittest \
   experiments.static_sr_wu_20261022.methods.splatsure_test -v
Set SPLATSURE_UPSTREAM to use another checkout of the pinned official source.
"""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch
import torch.nn.functional as F

from .splatsure import (
    FidelityAccumulator, RASTERIZER_COMMIT, SplatSuReConfig, fidelity_from_views,
    project_covariance, selection_weight_map, splatsure_loss, true_radii_from_covariance,
)
from .splatsure_prepare import prepare_splatsure_cache, same_time_training_groups

# Only CPU kernels are used; this also makes tests quick on many-core hosts.
torch.set_num_threads(2)


class SplatSuReTests(unittest.TestCase):
    def test_true_radius_matches_pinned_cuda_and_excludes_dilation(self):
        cov = torch.tensor([[[4., 0.], [0., 1.]], [[0.01, 0.], [0., 0.01]]])
        radius = true_radii_from_covariance(cov)
        expected = torch.tensor([6., 3 * (0.01 + 0.1**0.5)**0.5])
        torch.testing.assert_close(radius, expected)
        self.assertFalse(torch.allclose(radius, true_radii_from_covariance(cov + 0.3 * torch.eye(2))))

    def test_projection_matches_ewa_jacobian_clamping(self):
        xyz = torch.tensor([[0., 0., 2.], [100., -100., 2.]])
        cov = torch.eye(3).expand(2, 3, 3).clone()
        projected = project_covariance(xyz, cov, torch.eye(4), 10., 20., 1., 0.5)
        jac = torch.tensor([[5., 0., -6.5], [0., 10., 6.5]])
        torch.testing.assert_close(projected[0], torch.diag(torch.tensor([25., 100.])))
        torch.testing.assert_close(projected[1], jac @ jac.T)

    def test_actual_contribution_and_three_views(self):
        radii = torch.tensor([[1., 1., 1., 100.], [2., 1., 2., 100.], [4., 1., 4., 1.]])
        contrib = torch.tensor([[1., 1., 1., 0.], [1., 1., 0., 0.], [1., 1., 0., 1.]])
        score, acc = fidelity_from_views(radii, contrib)
        self.assertEqual(acc.max_camera_index.tolist(), [2, 0, 0, 2])
        self.assertEqual(acc.seen_count.tolist(), [3, 3, 1, 1])
        self.assertEqual(score[2:].tolist(), [0., 0.])
        torch.testing.assert_close(score[1], torch.sigmoid(torch.tensor(-2.)))
        self.assertGreater(score[0], 0.999)

    def test_streaming_matches_official_weight_maps_algorithm(self):
        torch.manual_seed(91)
        radii = torch.rand(5, 11) * 10
        visible = torch.randint(0, 2, (5, 11)).bool()
        minimum, maximum = torch.full((11,), 1e6), torch.full((11,), -1.)
        index, count = torch.full((11,), -1), torch.zeros(11, dtype=torch.int64)
        for camera in range(5):
            mask = (radii[camera] > 0) & visible[camera]
            count[mask] += 1
            lo, hi = mask & (radii[camera] < minimum), mask & (radii[camera] > maximum)
            minimum[lo], maximum[hi], index[hi] = radii[camera][lo], radii[camera][hi], camera
        expected = torch.sigmoid(((maximum / minimum).clamp_min(1) - 1.1) / 0.05)
        expected[count < 3] = 0
        actual, acc = fidelity_from_views(radii, visible)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(acc.max_camera_index, index)

    def test_weight_normalization_has_upstream_conditional_and_constant_edge(self):
        a = torch.tensor([[0.5, 0.7], [0.9, 0.8]])
        b = torch.zeros_like(a)
        expected = ((1 - a) - (1 - a).min()) / ((1 - a).max() - (1 - a).min() + 1e-6)
        torch.testing.assert_close(selection_weight_map(a, b), expected)
        torch.testing.assert_close(selection_weight_map(torch.full((2, 2), 0.5), b), b)
        a[0, 0] = 0
        torch.testing.assert_close(selection_weight_map(a, b), 1 - a)

    def test_zero_weights_keep_upstream_constant_without_sr_gradient(self):
        render = torch.rand(3, 12, 16, requires_grad=True)
        target_lr = torch.rand(3, 3, 4, requires_grad=True)
        target_sr = torch.rand_like(render, requires_grad=True)
        out = splatsure_loss(render, target_lr, target_sr, torch.zeros(3, 4))
        torch.testing.assert_close(out["sr_loss"], torch.tensor(0.2))
        out["loss"].backward()
        self.assertGreater(render.grad.abs().sum(), 0)
        self.assertIsNone(target_lr.grad)
        self.assertIsNone(target_sr.grad)

    def test_weighted_loss_value_and_gradient_match_upstream(self):
        root = Path(os.environ.get("SPLATSURE_UPSTREAM", "tmp/static_sr_wu_20261022/upstream/splatsure"))
        file = root / "utils/loss_utils.py"
        if not file.exists():
            self.skipTest("pinned upstream checkout not present; external numerical parity unavailable")
        spec = importlib.util.spec_from_file_location("official_splatsure_loss_cpu", file)
        upstream = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(upstream)
        torch.manual_seed(20261022)
        render = torch.rand(3, 24, 28, requires_grad=True)
        render_official = render.detach().clone().requires_grad_(True)
        lr = torch.rand(3, 6, 7)
        sr = torch.rand(3, 24, 28)
        weight = torch.rand(6, 7)
        weight_hr = F.interpolate(weight[None, None], (24, 28))[0, 0]
        down = F.interpolate(render_official[None], (6, 7), mode='bicubic', antialias=True, align_corners=False)[0]
        l1 = 0.4 * ((render_official - sr).abs() * weight_hr[None]).mean() + 0.6 * (down - lr).abs().mean()
        ssim = 0.4 * upstream.weighted_ssim(render_official, sr, weight_hr) + 0.6 * upstream.ssim(down, lr)
        official_loss = 0.8 * l1 + 0.2 * (1 - ssim)
        result = splatsure_loss(render, lr, sr, weight)
        torch.testing.assert_close(result['loss'], official_loss, atol=2e-7, rtol=1e-6)
        official_loss.backward()
        result['loss'].backward()
        torch.testing.assert_close(render.grad, render_official.grad, atol=3e-8, rtol=2e-5)

    @staticmethod
    def manifest():
        return dict(scene='fixture', splits=dict(train=['cam01','cam02','cam03'],test=['cam00']),
                    observations=[dict(camera_id=c, frame_index=f, time=f/300.,
                       split='test' if c == 'cam00' else 'train', hr_path='must_not_open.png')
                       for f in (0, 1) for c in ('cam00','cam01','cam02','cam03')])

    def test_grouping_rejects_cross_time_and_heldout_mislabelling(self):
        manifest = self.manifest()
        groups = same_time_training_groups(manifest)
        self.assertEqual([len(g[2]) for g in groups], [3, 3])
        manifest['observations'][2]['time'] = 2.
        with self.assertRaises(ValueError):
            same_time_training_groups(manifest)
        manifest = self.manifest()
        manifest['observations'][0]['split'] = 'train'
        with self.assertRaises(ValueError):
            same_time_training_groups(manifest)

    def test_mock_cache_exercises_time_and_split_isolation_without_parameter_updates(self):
        class Backend:
            def __init__(self):
                self.calls = []
            def effective_state(self, time):
                return time
            def render(self, obs, state=None, resolution='lr', colors=None, raster_api=None):
                assert obs['split'] == 'train' and obs['camera_id'] != 'cam00'
                assert obs['time'] == state and resolution == 'lr'
                assert not torch.is_grad_enabled()
                self.calls.append((obs['camera_id'], obs['frame_index']))
                if colors is None:
                    radius = torch.tensor([float(int(obs['camera_id'][-1])), 1.])
                    return dict(render=torch.zeros(3,2,2), true_radii=radius, is_contributing=torch.ones(2))
                return dict(render=colors.mean(0)[:,None,None].expand(3,2,2))
        backend = Backend()
        with tempfile.TemporaryDirectory() as tmp:
            receipt = prepare_splatsure_cache(backend, self.manifest(), Path(tmp)/'cache',
                    SimpleNamespace(static_sr_source_commit=RASTERIZER_COMMIT),
                    dict(parent_checkpoint_sha256='f'*64))
            self.assertEqual(receipt['status'], 'prepared')
            self.assertEqual(len(receipt['records']), 6)
            self.assertEqual(len(backend.calls), 18)
            self.assertEqual(receipt['parameter_updates'], 0)
            self.assertEqual(receipt['HR_image_reads'], 0)
            self.assertEqual(len(json.loads((Path(tmp)/'cache/receipt.json').read_text())['records']), 6)

    def test_native_radius_only_fails_closed(self):
        class Backend:
            def effective_state(self, time):
                return None
            def render(self, *args, **kwargs):
                return dict(render=torch.zeros(3,2,2), radii=torch.ones(3))
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/'cache'
            with self.assertRaises(RuntimeError):
                prepare_splatsure_cache(Backend(), self.manifest(), out,
                    SimpleNamespace(static_sr_source_commit=RASTERIZER_COMMIT),
                    dict(parent_checkpoint_sha256='f'*64))
            self.assertEqual(json.loads((out/'receipt.json').read_text())['status'], 'failed')


if __name__ == '__main__':
    unittest.main()
