"""CPU protocol and source-parity tests. These are NOT pretrained/GPU acceptance."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parents[1]
sys.path.insert(0, str(BASE))
from methods.sequence_matters import (
    UPSTREAM_COMMIT, als_plan, camera_rankings, group_synchronous, sequence_matters_loss,
    source_objects, validate_records, verify_upstream,
    prepare_sequences, sha256, validate_sequence_plan,
)
from methods.threedsr import require_round_state, run_alternating, threedsr_loss
from methods.threedsr_renderer import compute_3d_filter, apply_3d_filter


class SequenceDiffusionTests(unittest.TestCase):
    def records(self):
        return [dict(key=f"s_c{camera}_t{time}", scene="s", camera=f"cam{camera:02d}", time=time,
                     split="train", lr_path="unused.png", camera_center=[1., camera, 0.])
                for time in [0., 0.1] for camera in [1, 2, 3]]

    def test_fixed_time_grouping_and_information_boundary(self):
        groups = group_synchronous(self.records(), ["cam00"])
        self.assertEqual(len(groups), 2)
        for time, records in groups.items():
            self.assertEqual({r["time"] for r in records}, {time})
            self.assertEqual(len({r["camera"] for r in records}), 3)
        leaked = self.records()
        leaked[0]["camera"] = "cam00"
        with self.assertRaises(ValueError):
            validate_records(leaked, ["cam00"])
        mixed = self.records()
        mixed[-1]["scene"] = "other_scene"
        with self.assertRaises(ValueError):
            validate_records(mixed, ["cam00"])

    def test_pose_is_angular_and_origin_degenerate_fails(self):
        import numpy as np
        ranks = camera_rankings([[1, 0, 0], [100, 1, 0], [1, 1, 0]])
        # Euclidean distance would prefer index 2. Author angle prefers index 1.
        self.assertLess(ranks[0, 1, 0], ranks[0, 2, 0])
        self.assertAlmostEqual(ranks[0, 2, 1], 45.)
        with self.assertRaises(ValueError):
            camera_rankings([[0, 0, 0], [1, 1, 1]])

    def test_real_producer_control_flow_nested_keys_and_color(self):
        import numpy as np
        import torch
        import types
        from PIL import Image
        class FakePSRT(torch.nn.Module):
            def __init__(self, **kwargs):
                super().__init__()
            def forward(self, images):
                batch, frames, channels, h, w = images.shape
                return torch.nn.functional.interpolate(images.reshape(batch*frames, channels, h, w),
                    scale_factor=4, mode="nearest").reshape(batch, frames, channels, h*4, w*4)
        def source_tensor2img(tensor, **kwargs):
            # Author BasicSR call swaps BGR-network tensor back to RGB for PIL.
            image = (tensor.permute(1, 2, 0).cpu().numpy()*255).round().astype(np.uint8)
            return image[:, :, ::-1].copy() if kwargs["rgb2bgr"] else image
        arch = types.ModuleType("archs.psrt_recurrent_arch")
        arch.BasicRecurrentSwin = FakePSRT
        basic = types.ModuleType("basicsr.utils")
        basic.tensor2img = source_tensor2img
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            checkpoint = folder / "fake_test_checkpoint.pt"
            torch.save({"params": {}}, checkpoint)
            rows = []
            for camera in range(1, 4):
                filename = folder / f"cam{camera}.png"
                image = np.zeros((64, 64, 3), dtype=np.uint8)
                image[:] = [200, 30 + camera, 10]
                Image.fromarray(image).save(filename)
                rows.append(dict(key=f"cam{camera:02d}/0000", scene="s", camera=f"cam{camera:02d}",
                                 time=0, split="train", lr_path=str(filename), camera_center=[1, camera, 0]))
            manifest = {"records": rows, "heldout_cameras": ["cam00"]}
            upstream = ROOT / "tmp/static_sr_wu_20261022/upstream/sequence_matters"
            plan = prepare_sequences(manifest, upstream, folder / "plan", plan_only=True, device="cpu")
            plan_path = folder / "plan/receipt.json"
            with mock.patch.dict(sys.modules, {"archs.psrt_recurrent_arch": arch, "basicsr.utils": basic}), \
                    mock.patch("methods.sequence_matters.camera_rankings", side_effect=AssertionError("Recomputed camera ranks")), \
                    mock.patch("methods.sequence_matters.als_plan", side_effect=AssertionError("Recomputed ALS")), \
                    mock.patch("cv2.ORB_create", side_effect=AssertionError("Recomputed ORB")):
                receipt = prepare_sequences(manifest, upstream, folder / "cache", checkpoint, checkpoint,
                                            device="cpu", sequence_plan=plan_path)
            self.assertEqual(receipt["status"], "teacher_complete")
            self.assertTrue(receipt["teacher_generated"])
            self.assertEqual(receipt["sequence_plan"]["sha256"], sha256(plan_path))
            self.assertEqual(receipt["groups"], plan["groups"])
            for camera in range(1, 4):
                image = np.asarray(Image.open(receipt["targets"][f"cam{camera:02d}/0000"]))
                self.assertEqual(image.shape, (256, 256, 3))
                self.assertEqual(image[100, 100].tolist(), [200, 30+camera, 10])

    def test_sequence_plan_rejects_changed_inputs_and_invalid_outputs(self):
        import numpy as np
        from PIL import Image
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            rows = self.records()[:3]
            for index, record in enumerate(rows):
                image_path = folder / f"{index}.png"
                Image.fromarray(np.full((64, 64, 3), index, np.uint8)).save(image_path)
                record["lr_path"] = str(image_path)
                record["lr_sha256"] = sha256(image_path)
            manifest = {"records": rows, "heldout_cameras": ["cam00"]}
            upstream = ROOT / "tmp/static_sr_wu_20261022/upstream/sequence_matters"
            source = upstream / "vsr/utils_vsr.py"
            plan = prepare_sequences(manifest, upstream, folder / "plan", plan_only=True, device="cpu")
            self.assertFalse(plan["teacher_generated"])
            validate_sequence_plan(plan, manifest, source)
            for field, value in [("upstream_commit", "wrong"), ("source_sha256", "wrong"),
                                 ("manifest_sha256", "wrong")]:
                changed = copy.deepcopy(plan)
                changed[field] = value
                with self.assertRaises(ValueError):
                    validate_sequence_plan(changed, manifest, source)
            changed_manifest = copy.deepcopy(manifest)
            changed_manifest["records"][0]["camera_center"] = [2, 3, 4]
            with self.assertRaisesRegex(ValueError, "manifest"):
                validate_sequence_plan(plan, changed_manifest, source)
            changed = copy.deepcopy(plan)
            changed["groups"][0]["sequences"][0]["output_indices"].append(0)
            with self.assertRaisesRegex(ValueError, "first-writer"):
                validate_sequence_plan(changed, manifest, source)
            changed = copy.deepcopy(plan)
            changed["groups"][0]["sequences"][0]["indices"][0] = len(rows)
            with self.assertRaisesRegex(ValueError, "indices"):
                validate_sequence_plan(changed, manifest, source)
            Path(rows[0]["lr_path"]).write_bytes(b"replaced observation")
            with self.assertRaisesRegex(ValueError, "LR bytes"):
                validate_sequence_plan(plan, manifest, source)

    def test_mip_filter_matches_author_source_and_preserves_mass(self):
        import ast
        import math
        import torch
        from types import SimpleNamespace
        source = ROOT / "tmp/static_sr_wu_20261022/upstream/threedsr/scene/gaussian_model.py"
        tree = ast.parse(source.read_text())
        klass = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "GaussianModel")
        original = next(n for n in klass.body if isinstance(n, ast.FunctionDef) and n.name == "compute_3D_filter")
        namespace = {"torch": torch}
        exec(compile(ast.Module(body=[original], type_ignores=[]), str(source), "exec"), namespace)
        camera = SimpleNamespace(R=torch.eye(3).numpy(), T=torch.zeros(3).numpy(),
                                 focal_x=50., focal_y=50., image_width=100, image_height=100,
                                 FoVx=math.pi/2, FoVy=math.pi/2, world_view_transform=torch.eye(4))
        xyz = torch.tensor([[0., 0., 2.], [.1, .2, 4.], [10., 10., -1.]])
        original_self = SimpleNamespace(get_xyz=xyz)
        namespace["compute_3D_filter"](original_self, [camera])
        actual = compute_3d_filter(xyz, [camera])
        self.assertTrue(torch.allclose(actual, original_self.filter_3D))
        scales = torch.full((3, 3), .1, requires_grad=True)
        opacity = torch.full((3, 1), .5, requires_grad=True)
        filtered, compensated = apply_3d_filter(scales, opacity, actual)
        self.assertTrue(torch.allclose(scales.prod(-1) * opacity[:, 0], filtered.prod(-1) * compensated[:, 0]))
        (filtered.sum() + compensated.sum()).backward()
        self.assertTrue(torch.isfinite(scales.grad).all())

    def test_als_first_writer_matches_author_process(self):
        import numpy as np
        import torch
        from PIL import Image
        root = verify_upstream(ROOT / "tmp/static_sr_wu_20261022/upstream/sequence_matters")
        source = root / "vsr/utils_vsr.py"
        count = 12
        angles = np.linspace(0, 1.1, count)
        positions = camera_rankings(np.stack([np.cos(angles), np.sin(angles), np.zeros(count)], axis=1))
        rng = np.random.default_rng(4)
        feature_dist = rng.uniform(0, 50, (count, count))
        np.fill_diagonal(feature_dist, 0)
        features = np.stack([np.argsort(np.argsort(feature_dist, axis=1), axis=1), feature_dist], axis=-1)
        order = source_objects(source, ["ordering_sim1_thresholding_sim2"], {"np": np})["ordering_sim1_thresholding_sim2"]
        plan = als_plan(positions, features, order, thresholds=(2, 4), max_sequence_length=8)
        assigned = [i for sequence in plan for i in sequence["output_indices"]]
        self.assertEqual(sorted(assigned), list(range(count)))
        self.assertEqual(len(set(assigned)), count)
        # Run the author's untouched ALS control flow on a marker model. Markers
        # encode the chosen sequence and output position, checking first-writer
        # and endpoint choices against an independently executed implementation.
        images = [np.full((4, 4, 3), i, dtype=np.uint8) for i in range(count)]
        class MarkerModel:
            def __call__(self, batch):
                ids = (batch[0, :, 0, 0, 0] * 255).round().long().tolist()
                result = torch.zeros_like(batch)
                for offset in range(len(ids)):
                    result[0, offset] = ((sum((j+1)*v for j, v in enumerate(ids)) + offset) % 255) / 255
                return result
        def tensor2img(tensor, **kwargs):
            return (tensor.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
        import os
        namespace = dict(np=np, torch=torch, os=os, Image=Image, tensor2img=tensor2img,
                         compute_feature_distance_ranking=lambda _: features,
                         compute_position_distance_ranking=lambda *_: positions,
                         ordering_sim1_thresholding_sim2=order)
        function = source_objects(source, ["process_ALS"], namespace)["process_ALS"]
        with tempfile.TemporaryDirectory() as folder:
            function(MarkerModel(), "pose", images, [f"{i}.png" for i in range(count)],
                     folder, "unused", [2, 4, count], 8, "cpu")
            for sequence in plan:
                indices = sequence["indices"]
                for index in sequence["output_indices"]:
                    expected = (sum((j+1)*v for j, v in enumerate(indices)) + indices.index(index)) % 255
                    self.assertEqual(int(np.asarray(Image.open(Path(folder) / f"{index}.png"))[0, 0, 0]), expected)

    def test_native_losses_and_explicit_area_adaptation_have_gradients(self):
        import torch
        import torch.nn.functional as F
        torch.manual_seed(0)
        render = torch.rand(3, 32, 32, requires_grad=True)
        teacher = torch.rand_like(render, requires_grad=True)
        observed = torch.rand(3, 8, 8, requires_grad=True)
        # Set SSIM coefficient to zero to validate independent analytic losses.
        no_ssim = lambda a, b: torch.tensor(1.)
        for function, hr_weight, lr_weight in [(sequence_matters_loss, .4, .6), (threedsr_loss, 1., 1.)]:
            for mode in ("bicubic", "area"):
                kwargs = {"align_corners": False, "antialias": True} if mode == "bicubic" else {}
                reduced = F.interpolate(render[None], size=(8, 8), mode=mode, **kwargs)[0]
                expected = hr_weight * (render - teacher).abs().mean() + lr_weight * (reduced - observed).abs().mean()
                loss, _ = function(render, teacher, observed, no_ssim, lambda_dssim=0, lr_operator=mode)
                self.assertTrue(torch.allclose(loss, expected))
                loss.backward()
                self.assertTrue(torch.isfinite(render.grad).all())
                self.assertIsNone(teacher.grad)
                self.assertIsNone(observed.grad)
        with self.assertRaises(ValueError):
            sequence_matters_loss(render, teacher, observed, no_ssim, lr_operator="nearest")

    def test_per_observation_latent_identity_and_sequential_rounds(self):
        state = dict(key="scene_cam02_t3", lr_sha256="abc", round_index=1)
        require_round_state(2, state, "scene_cam02_t3", "abc")
        for round_index, key, input_hash in [(1, state["key"], "abc"), (2, "scene_cam01_t3", "abc"),
                                             (2, state["key"], "changed")]:
            with self.assertRaises(ValueError):
                require_round_state(round_index, state, key, input_hash)

    def test_diffusion_shared_optimization_feedback_barriers(self):
        keys = ["cam1_t0", "cam2_t0", "cam1_t1", "cam2_t1"]
        events = []
        with tempfile.TemporaryDirectory() as folder:
            files = {key: str(Path(folder) / (key + ".png")) for key in keys}
            for file in files.values():
                Path(file).write_bytes(b"orchestration-test-only")
            def denoise_all(round_index, rendered_paths):
                self.assertEqual(rendered_paths is None, round_index == 0)
                events.append((round_index, "denoise"))
                return files
            def optimize_shared(targets, iterations, round_index):
                self.assertEqual(iterations, 5000)
                self.assertEqual(set(targets), set(keys))
                events.append((round_index, "optimize_shared_all_times"))
                return "synthetic_checkpoint"
            def render_training(keys, round_index):
                events.append((round_index, "render_shared_all_times"))
                return files
            receipts = run_alternating(keys, denoise_all, optimize_shared, render_training)
            self.assertEqual(len(receipts), 4)
            self.assertEqual(events, [(r, action) for r in range(4) for action in
                (["denoise", "optimize_shared_all_times", "render_shared_all_times"] if r < 3 else ["denoise", "optimize_shared_all_times"])])
            with self.assertRaises(RuntimeError):
                run_alternating(keys, denoise_all, optimize_shared, lambda **kwargs: {})


if __name__ == "__main__":
    unittest.main()
