"""CPU contract tests; no metric model construction or GPU execution."""
from pathlib import Path
import subprocess
import sys
import unittest

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
import metric_acceptance as acceptance


class MetricAcceptanceContracts(unittest.TestCase):
    def setUp(self):
        self.self_metrics = dict(PSNR=120., SSIM=1., LPIPS=0.)
        self.different = dict(PSNR=22., SSIM=.6, LPIPS=.2)

    def test_fixed_metrics_and_strict_absolute_lpips_parity(self):
        result = acceptance.require_metric_contract(self.self_metrics, self.different, .2, .2 + 5e-7)
        self.assertTrue(result['passed'])
        self.assertEqual(result['relative_tolerance'], 0)
        with self.assertRaisesRegex(ValueError, 'strict absolute'):
            acceptance.require_metric_contract(self.self_metrics, self.different, .2, .2 + 1.1e-6)

    def test_self_nondistinct_and_nonfinite_rejected(self):
        for self_values, other_values in [({**self.self_metrics, 'SSIM': .99}, self.different),
                                        (self.self_metrics, {**self.different, 'LPIPS': 0.}),
                                        (self.self_metrics, {**self.different, 'PSNR': float('nan')})]:
            with self.assertRaises(ValueError):
                acceptance.require_metric_contract(self_values, other_values, .2, .2)

    def test_pair_preserves_timestamp_and_train_only_boundary(self):
        manifest = dict(frame_indices=[0, 149, 299], splits=dict(train=['cam01', 'cam02'], test=['cam00']))
        rows = [dict(camera_id=c, frame_index=149, time=149/300, split='train') for c in ['cam01', 'cam02']]
        self.assertEqual(acceptance.select_pair(manifest, rows, 149), rows)
        for cameras in [('cam00', 'cam01'), ('cam01', 'cam01'), ('cam01', 'cam03')]:
            with self.assertRaises(ValueError):
                acceptance.select_pair(manifest, rows, 149, cameras)
        with self.assertRaises(ValueError):
            acceptance.select_pair(manifest, [{**r, 'time': 0.5} for r in rows], 149)

    def test_hash_mismatch_rejected_before_runtime(self):
        import tempfile
        with tempfile.TemporaryDirectory() as root:
            file = Path(root) / 'incorrect.py'
            file.write_text('incorrect source')
            with self.assertRaisesRegex(ValueError, 'Identity changed'):
                acceptance.checked(dict(path=str(file), sha256=acceptance.SSIM_SHA))

    def test_import_does_not_import_torch_or_backend(self):
        code = "import sys;sys.path.insert(0,sys.argv[1]);import metric_acceptance;print('torch' in sys.modules, 'backend' in sys.modules)"
        result = subprocess.run([sys.executable, '-c', code, str(BASE)], capture_output=True, text=True, check=True)
        self.assertEqual(result.stdout.strip(), 'False False')


if __name__ == '__main__':
    unittest.main()
