"""CPU contracts for real-teacher fixtures; no model construction or CUDA call."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
import teacher_acceptance as acceptance
from methods import sequence_matters as sequence
from methods import threedsr


class TeacherAcceptanceContracts(unittest.TestCase):
    def setUp(self):
        from PIL import Image
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.rows = []
        for frame in (0, 149):
            for camera in ("cam03", "cam01", "cam02"):
                path = self.root / f"{camera}_{frame}.png"
                Image.new("RGB", (16, 8), (int(camera[-1]) * 20, frame, 100)).save(path)
                self.rows.append({"key": f"{camera}/{frame:04d}", "scene": "scene",
                    "camera": camera, "frame": frame, "time": frame / 300, "split": "train",
                    "lr_path": str(path), "lr_sha256": sequence.sha256(path),
                    "camera_center": [int(camera[-1]), 1, 1]})
        self.manifest = {"records": self.rows, "heldout_cameras": ["cam00"]}
        self.source = self.root / "utils_vsr.py"
        self.source.write_text("# hashed author source fixture\n")
        self.plan = {"status": "plan_only", "targets": {}, "target_sha256": {},
            "upstream_commit": sequence.UPSTREAM_COMMIT, "source_sha256": sequence.sha256(self.source),
            "manifest_sha256": acceptance.canonical_hash(self.manifest), "groups": []}
        for time_value, rows in sequence.group_synchronous(self.rows, ["cam00"]).items():
            self.plan["groups"].append({"time": time_value, "keys": [r["key"] for r in rows],
                "lr_sha256": {r["key"]: r["lr_sha256"] for r in rows},
                "sequences": [{"threshold": 3, "reference": 0, "indices": [0, 1, 2],
                               "output_indices": [0, 1, 2]}]})

    def cpu_torch(self):
        import torch
        cuda = SimpleNamespace(synchronize=mock.Mock(), reset_peak_memory_stats=mock.Mock(),
            max_memory_allocated=lambda: 123, max_memory_reserved=lambda: 456)
        return SimpleNamespace(is_tensor=torch.is_tensor, isfinite=torch.isfinite,
            load=torch.load, save=torch.save, cuda=cuda)

    def test_complete_camera_group_retains_original_order_and_time(self):
        selected = acceptance.select_fixture(self.manifest, 149, "sequence_matters")
        self.assertEqual([r["camera"] for r in selected["records"]], ["cam03", "cam01", "cam02"])
        self.assertEqual({r["time"] for r in selected["records"]}, {149 / 300})
        with self.assertRaisesRegex(ValueError, "every synchronous"):
            acceptance.select_fixture(self.manifest, 149, "sequence_matters", "cam01")
        incomplete = copy.deepcopy(self.manifest)
        incomplete["records"].pop()
        with self.assertRaisesRegex(ValueError, "lacks"):
            acceptance.select_fixture(incomplete, 149, "sequence_matters")

    def test_single_camera_selection_and_heldout_rejection(self):
        selected = acceptance.select_fixture(self.manifest, 0, "threedsr", "cam01")
        self.assertEqual([r["key"] for r in selected["records"]], ["cam01/0000"])
        with self.assertRaises(ValueError):
            acceptance.select_fixture(self.manifest, 0, "stablesr", "cam00")
        poisoned = copy.deepcopy(self.manifest)
        poisoned["records"][0]["camera"] = "cam00"
        with self.assertRaisesRegex(ValueError, "Evaluation"):
            acceptance.select_fixture(poisoned, 0, "sequence_matters")

    def test_subset_plan_is_bound_to_full_input_and_unchanged_als(self):
        fixture = acceptance.select_fixture(self.manifest, 149, "sequence_matters")
        subset = acceptance.derive_sequence_plan(self.plan, self.manifest, fixture, self.source)
        self.assertEqual(subset["groups"], [self.plan["groups"][1]])
        self.assertEqual(subset["manifest_sha256"], acceptance.canonical_hash(fixture))
        self.assertEqual(len(self.plan["groups"]), 2)
        changed = copy.deepcopy(self.plan)
        changed["groups"][0]["sequences"][0]["indices"] = [0, 2, 1]
        changed["groups"][0]["sequences"][0]["output_indices"] = [0, 1, 2]
        with self.assertRaisesRegex(ValueError, "first-writer"):
            acceptance.derive_sequence_plan(changed, self.manifest, fixture, self.source)

    def test_changed_lr_bytes_are_rejected_before_model_execution(self):
        Path(self.rows[0]["lr_path"]).write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "identity changed"):
            acceptance.select_fixture(self.manifest, 0, "sequence_matters")
        fixture = copy.deepcopy(self.manifest)
        fixture["records"] = fixture["records"][3:]
        with self.assertRaisesRegex(ValueError, "bytes changed"):
            acceptance.derive_sequence_plan(self.plan, self.manifest, fixture, self.source)

    def test_floating_check_catches_nan_before_png_can_hide_it(self):
        import torch
        self.assertEqual(acceptance.finite_tree((torch.ones(2), {"x": torch.zeros(1)}), torch, "test"), 2)
        with self.assertRaises(FloatingPointError):
            acceptance.finite_tree(torch.tensor([float("nan")]), torch, "decoder")

    def small_lpips_modules(self):
        """Tiny CPU modules preserve the author's same-Parameter reuse/load order."""
        import torch
        class VGG(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.features = torch.nn.Sequential(torch.nn.Conv2d(1, 1, 1))
        class LPIPS(torch.nn.Module):
            def __init__(self, vgg):
                super().__init__()
                self.net = torch.nn.Module()
                self.net.slice1 = torch.nn.Sequential(vgg.features[0])
                self.scaling_layer = torch.nn.Module()
                self.scaling_layer.register_buffer("shift", torch.zeros(1))
                for index in range(5):
                    layer = torch.nn.Module()
                    layer.model = torch.nn.Sequential(torch.nn.Dropout(), torch.nn.Conv2d(1, 1, 1, bias=False))
                    self.add_module(f"lin{index}", layer)
        VGG.__module__ = "torchvision.models.vgg"
        LPIPS.__module__ = "taming.modules.losses.lpips"
        return VGG, LPIPS

    def test_sparse_author_lpips_load_requires_previous_same_vgg_parameters(self):
        import torch
        VGG, LPIPS = self.small_lpips_modules()
        original = torch.nn.Module.load_state_dict
        audits = []
        with acceptance.observe_stablesr_checkpoint_loads(torch, audits):
            vgg = VGG()
            vgg.load_state_dict({k: torch.full_like(v, .25) for k, v in vgg.state_dict().items()})
            metric = LPIPS(vgg)
            previous_backbone = {k: v.clone() for k, v in metric.net.state_dict().items()}
            calibration = {f"lin{i}.model.1.weight": torch.full_like(getattr(metric, f"lin{i}").model[1].weight, .75)
                           for i in range(5)}
            result = metric.load_state_dict(calibration, strict=False)
            self.assertTrue(result.missing_keys)
            self.assertTrue(audits[-1]["permitted_sparse_lpips_calibration"])
            self.assertEqual(set(audits[-1]["previously_loaded_vgg_parameters"]),
                             {"net.slice1.0.weight", "net.slice1.0.bias"})
            for name, value in metric.net.state_dict().items():
                self.assertTrue(torch.equal(value, previous_backbone[name]))
            root = torch.nn.Module()
            root.loss = metric
            root.decoder = torch.nn.Linear(1, 1)
            root.load_state_dict(root.state_dict(), strict=False)
            self.assertEqual(audits[-1]["missing_parameters"], [])
            self.assertFalse(audits[-1]["permitted_sparse_lpips_calibration"])
        self.assertIs(torch.nn.Module.load_state_dict, original)

    def test_lpips_rejects_unloaded_or_different_vgg_and_missing_calibration(self):
        import torch
        VGG, LPIPS = self.small_lpips_modules()
        original = torch.nn.Module.load_state_dict
        for case in ("unloaded", "different_parameter_objects", "missing_lin", "unexpected_key"):
            with self.subTest(case=case):
                audits = []
                with self.assertRaisesRegex(ValueError, "uninitialized"):
                    with acceptance.observe_stablesr_checkpoint_loads(torch, audits):
                        vgg = VGG()
                        if case != "unloaded":
                            vgg.load_state_dict(vgg.state_dict())
                        metric = LPIPS(VGG() if case == "different_parameter_objects" else vgg)
                        calibration = {name: value for name, value in metric.state_dict().items() if name.startswith("lin")}
                        if case == "missing_lin":
                            calibration.pop("lin3.model.1.weight")
                        if case == "unexpected_key":
                            calibration["foreign.weight"] = torch.ones(1)
                        metric.load_state_dict(calibration, strict=False)
                self.assertFalse(audits[-1]["permitted_sparse_lpips_calibration"])
                self.assertIs(torch.nn.Module.load_state_dict, original)

    def test_lpips_shape_error_restores_hook_without_relaxing_torch_load(self):
        import torch
        VGG, LPIPS = self.small_lpips_modules()
        original = torch.nn.Module.load_state_dict
        with self.assertRaisesRegex(RuntimeError, "size mismatch"):
            with acceptance.observe_stablesr_checkpoint_loads(torch, []):
                vgg = VGG()
                vgg.load_state_dict(vgg.state_dict())
                metric = LPIPS(vgg)
                calibration = {name: value for name, value in metric.state_dict().items() if name.startswith("lin")}
                calibration["lin0.model.1.weight"] = torch.ones(2, 1, 1, 1)
                metric.load_state_dict(calibration, strict=False)
        self.assertIs(torch.nn.Module.load_state_dict, original)

    def test_stablesr_structural_and_vq_missing_parameters_remain_errors(self):
        import torch
        original = torch.nn.Module.load_state_dict
        for component in ("diffusion_model", "structcond_stage_model", "decoder"):
            with self.subTest(component=component):
                model = torch.nn.Module()
                model.add_module(component, torch.nn.Linear(1, 1))
                full = model.state_dict()
                with self.assertRaisesRegex(ValueError, "uninitialized"):
                    with acceptance.observe_stablesr_checkpoint_loads(torch, []):
                        model.load_state_dict(full)
                        model.load_state_dict({k: v for k, v in full.items() if k != component + ".weight"}, strict=False)
                self.assertIs(torch.nn.Module.load_state_dict, original)

    def test_output_coverage_and_actual_x4_shape_are_required(self):
        from PIL import Image
        fixture = acceptance.select_fixture(self.manifest, 0, "stablesr", "cam01")
        target = self.root / "target.png"
        Image.new("RGB", (64, 32)).save(target)
        targets = {"cam01/0000": str(target)}
        self.assertEqual(acceptance.verify_targets(targets, fixture)[0]["size_wh"], [64, 32])
        with self.assertRaisesRegex(RuntimeError, "coverage"):
            acceptance.verify_targets({}, fixture)
        Image.new("RGB", (32, 64)).save(target)
        with self.assertRaisesRegex(RuntimeError, "LR x4"):
            acceptance.verify_targets(targets, fixture)

    def fake_bridge(self, fixture, tamper=False):
        import torch
        from PIL import Image
        owner = self

        class Bridge:
            def __init__(self):
                self.state_root = owner.root / "state"
                self.weight_loads = [{"sha256": "main"}, {"sha256": "cfw"}]
                self.calls = []

            def generate_targets(self, manifest, index, output, feedback=None):
                row = manifest["records"][0]
                self.calls.append((index, copy.deepcopy(feedback)))
                state_path = self.state_root / (row["key"] + ".pt")
                previous = torch.load(state_path, weights_only=True) if state_path.exists() else None
                threedsr.require_round_state(index, previous, row["key"], row["lr_sha256"])
                if index == 0:
                    owner.assertIsNone(feedback)
                else:
                    owner.assertEqual(set(feedback), {row["key"]})
                    owner.assertEqual(Path(feedback[row["key"]]).parent.name, f"round{index-1}")
                output.mkdir()
                png = output / "target.png"
                Image.new("RGB", (64, 32), (index * 40, 0, 0)).save(png)
                state_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"key": row["key"], "lr_sha256": row["lr_sha256"], "scene": row["scene"],
                    "round_index": index, "upstream_commit": threedsr.UPSTREAM_COMMIT,
                    "model_identity": ["wrong" if tamper else "main", "cfw"],
                    "patches": {"0": {"region": [0, 32, 0, 64], "x_t": torch.ones(1, 4, 4, 8) * index}}}, state_path)
                (output / "receipt.json").write_text(json.dumps({"round_index": index}))
                return {row["key"]: str(png)}

        return Bridge()

    def test_four_real_bridge_boundaries_echo_feedback_and_snapshot_each_state(self):
        fixture = acceptance.select_fixture(self.manifest, 0, "threedsr", "cam01")
        bridge, torch = self.fake_bridge(fixture), self.cpu_torch()
        rounds = []
        acceptance.run_four_round_fixture(bridge, fixture, self.root, torch, rounds)
        self.assertEqual([r["round_index"] for r in rounds], list(range(4)))
        self.assertEqual([c[0] for c in bridge.calls], list(range(4)))
        self.assertEqual(torch.cuda.reset_peak_memory_stats.call_count, 4)
        self.assertEqual(torch.cuda.synchronize.call_count, 8)
        self.assertEqual(rounds[0]["feedback_mode"], "none")
        self.assertTrue(all(r["feedback_mode"] == "feedback_echo_previous_teacher_png" for r in rounds[1:]))
        for index, record in enumerate(rounds):
            self.assertEqual(torch.load(record["latent"]["path"], weights_only=True)["round_index"], index)
            if index:
                self.assertEqual(record["feedback_sha256"], rounds[index-1]["outputs"][0]["sha256"])
        self.assertEqual(len({r["latent"]["sha256"] for r in rounds}), 4)

    def test_changed_latent_identity_stops_before_next_round(self):
        fixture = acceptance.select_fixture(self.manifest, 0, "threedsr", "cam01")
        bridge = self.fake_bridge(fixture, tamper=True)
        with self.assertRaisesRegex(RuntimeError, "identity differs"):
            acceptance.run_four_round_fixture(bridge, fixture, self.root, self.cpu_torch(), [])
        self.assertEqual(len(bridge.calls), 1)

    def test_sequence_observer_checks_actual_camera_tensor_order(self):
        import cv2
        import numpy as np
        import torch
        from PIL import Image
        fixture = acceptance.select_fixture(self.manifest, 0, "sequence_matters")
        plan = acceptance.derive_sequence_plan(self.plan, self.manifest, fixture, self.source)
        author_module = ModuleType("archs.psrt_recurrent_arch")

        class Model:
            def forward(self, batch):
                return batch.repeat_interleave(4, dim=-2).repeat_interleave(4, dim=-1)

        author_module.BasicRecurrentSwin = Model
        args = SimpleNamespace(upstream=str(self.root), psrt_checkpoint="unused", spynet_checkpoint="unused")
        altered_order = False

        def fake_prepare(manifest, upstream, out, *a, **kw):
            out.mkdir()
            arrays = [cv2.imread(r["lr_path"]) for r in manifest["records"]]
            if altered_order:
                arrays.reverse()
            batch = torch.from_numpy(np.stack(arrays).transpose(0, 3, 1, 2)).float().div(255).unsqueeze(0)
            Model().forward(batch)
            targets = {}
            for row in manifest["records"]:
                dest = out / (row["camera"] + ".png")
                Image.new("RGB", (64, 32)).save(dest)
                targets[row["key"]] = str(dest)
            (out / "receipt.json").write_text("{}")
            return {"targets": targets}

        with mock.patch.dict(sys.modules, {"archs.psrt_recurrent_arch": author_module}), \
                mock.patch.object(sequence, "prepare_sequences", side_effect=fake_prepare), \
                mock.patch.object(torch.cuda, "synchronize"):
            out = self.root / "valid_sequence"
            out.mkdir()
            receipt = {}
            acceptance.run_sequence(args, fixture, plan, out, torch, receipt)
            self.assertEqual(receipt["forward_calls"][0]["keys"], [r["key"] for r in fixture["records"]])
            altered_order = True
            out = self.root / "wrong_sequence"
            out.mkdir()
            with self.assertRaisesRegex(RuntimeError, "camera order"):
                acceptance.run_sequence(args, fixture, plan, out, torch, {})
        self.assertIs(Model.forward, author_module.BasicRecurrentSwin.forward)

    def test_import_and_help_never_import_torch_or_select_cuda(self):
        code = "import sys;sys.path.insert(0,sys.argv[1]);import teacher_acceptance;print('torch' in sys.modules)"
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1"}
        run = subprocess.run([sys.executable, "-c", code, str(BASE)], capture_output=True, text=True, env=env, check=True)
        self.assertEqual(run.stdout.strip(), "False")
        help_run = subprocess.run([sys.executable, str(BASE / "teacher_acceptance.py"), "--help"],
            capture_output=True, text=True, env=env, check=True)
        self.assertIn("--sequence-plan", help_run.stdout)
        self.assertIn("--runtime-assets", help_run.stdout)

    def test_plan_receipt_cannot_claim_formal_or_gpu_completion(self):
        output = self.root / "plan_output"
        args = ["--method", "stablesr", "--records", str(self.root / "records.json"),
                "--upstream", str(self.root), "--out", str(output), "--plan-only"]
        with mock.patch.object(acceptance, "prepare", return_value=(self.manifest, None)):
            receipt = acceptance.main(args)
        self.assertEqual(receipt["schema"], acceptance.SCHEMA)
        self.assertEqual(receipt["status"], "plan_only")
        for field in ("formal_prior_allowed", "formal_parent_allowed", "teacher_generated", "gpu_executed"):
            self.assertFalse(receipt[field])
        self.assertEqual(json.loads((output / "acceptance.json").read_text())["status"], "plan_only")

    def test_missing_binding_writes_failed_receipt_without_gpu_execution(self):
        output = self.root / "failed_output"
        args = ["--method", "threedsr", "--records", str(self.root / "records.json"),
                "--upstream", str(self.root), "--out", str(output)]
        with mock.patch.object(acceptance, "prepare", return_value=(self.manifest, None)), \
                mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}):
            with self.assertRaisesRegex(ValueError, "Bind exactly one"):
                acceptance.main(args)
        saved = json.loads((output / "acceptance.json").read_text())
        self.assertEqual(saved["status"], "failed")
        self.assertFalse(saved["gpu_executed"])
        with self.assertRaises(FileExistsError):
            acceptance.main(args)


if __name__ == "__main__":
    unittest.main()
