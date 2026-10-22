"""CPU contracts; synthetic arrays are tests, never empirical cache estimates."""
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import numpy as np

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
import lossless_cache_probe as probe


class LosslessCacheContracts(unittest.TestCase):
    def test_exact_uint8_encoding_and_rejection_of_other_depths(self):
        original = np.arange(256, dtype=np.float32).reshape(1, 16, 16) / np.float32(65536.)
        encoded = probe.inverse_depth_uint8(original)
        self.assertEqual(encoded.dtype, np.uint8)
        np.testing.assert_array_equal(encoded.flatten(), np.arange(256, dtype=np.uint8))
        with self.assertRaisesRegex(ValueError, 'exactly'):
            probe.inverse_depth_uint8(original + np.float32(1 / 131072))
        with self.assertRaisesRegex(ValueError, 'float32'):
            probe.inverse_depth_uint8(original.astype(np.float64))

    def test_bitwise_contract_catches_signed_zero(self):
        with self.assertRaisesRegex(ValueError, 'bit patterns'):
            probe.assert_exact(np.array([0.], np.float32), np.array([-0.], np.float32))

    def test_real_codecs_roundtrip_and_refuse_overwrite(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            original = np.random.default_rng(3).normal(size=(3, 12, 16)).astype(np.float32)
            source = root / 'input.npy'
            np.save(source, original, allow_pickle=False)
            for codec, suffix in [('npz_deflate', 'npz'), ('zstd_level3', 'npy.zst')]:
                zstd = shutil.which('zstd')
                if codec.startswith('zstd') and not zstd:
                    continue
                output = root / ('compressed.' + suffix)
                result = probe.benchmark_codec(source, original, output, codec, zstd)
                self.assertTrue(result['roundtrip']['bitwise_equal'])
                self.assertGreater(result['compressed_bytes'], 0)
                with self.assertRaises(FileExistsError):
                    probe.benchmark_codec(source, original, output, codec, zstd)

    def test_estimates_use_worst_domain_field_and_fallback(self):
        scenes = [dict(short='A', domain='N3DV', observations=2,
                       field_payload_bytes={'external_rgb': 1200, 'internal_rgb': 1200})]
        results = [dict(domain='N3DV', field='external_rgb', benchmarks=[
            dict(codec='npz_deflate', stored_fraction_of_payload=ratio)]) for ratio in [.4, .8]]
        estimate = probe.extrapolate(scenes, results, 1.1)[0]
        self.assertEqual(estimate['components'][0]['estimated_bytes'], 1056)
        self.assertEqual(estimate['components'][1]['estimated_bytes'], 1200 + 2 * 4096)
        self.assertFalse(estimate['all_domain_fields_sampled'])
        self.assertFalse(estimate['capacity_guaranteed'])
        for factor in [.9, float('nan')]:
            with self.assertRaises(ValueError):
                probe.extrapolate(scenes, results, factor)

    def test_incomplete_gpu_receipts_are_rejected(self):
        import json
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'receipt.json'
            path.write_text(json.dumps({'schema': 'iesrgs_external_teacher_GPU_acceptance_v1',
                                       'status': 'pending_GPU_execution', 'GPU_executed': False}))
            with self.assertRaisesRegex(ValueError, 'completed real GPU'):
                probe.collect_samples([path], [], [])
            with self.assertRaisesRegex(ValueError, 'finished'):
                probe.collect_samples([], [path], [])

    def test_import_does_not_import_torch(self):
        code = "import sys;sys.path.insert(0,sys.argv[1]);import lossless_cache_probe;print('torch' in sys.modules)"
        result = subprocess.run([sys.executable, '-c', code, str(BASE)], capture_output=True, text=True, check=True)
        self.assertEqual(result.stdout.strip(), 'False')


if __name__ == '__main__':
    unittest.main()
