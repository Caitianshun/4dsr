"""CPU build-boundary checks; never invokes nvcc or CUDA runtime operations."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
import cts_native_build as native


class NativeBuildContracts(unittest.TestCase):
    def test_numeric_abi_ordering(self):
        text = 'GLIBCXX_3.4.9 GLIBCXX_3.4.32 GLIBC_2.9 GLIBC_2.34 CXXABI_1.3.13'
        with mock.patch.object(native, 'command', return_value=text):
            result = native.version_requirements(Path('fixture.so'))
        self.assertEqual(result['max_GLIBCXX'], '3.4.32')
        self.assertEqual(result['max_GLIBC'], '2.34')

    def test_tampered_source_rejected_and_existing_output_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'sources/simple_knn'
            source.mkdir(parents=True)
            file = source / 'ext.cpp'
            file.write_text('source')
            report = dict(schema='static_sr_wu_CTS_exact_native_source_bundle_v1', source_Wu_commit=native.WU,
                packages=[dict(name='simple_knn', upstream_commit=native.KNN, source_relative='sources/simple_knn',
                    files=[dict(relative_path='ext.cpp', sha256=native.sha(file))])], patches=[])
            (root / 'source_inventory.json').write_text(json.dumps(report))
            native.verify_bundle(root)
            existing = root / 'old_output'; existing.mkdir()
            with self.assertRaises(FileExistsError):
                native.build(root, existing, root/'cuda', 'gcc', 'g++', 2)
            file.write_text('tampered')
            with self.assertRaisesRegex(ValueError, 'Bundled source changed'):
                native.verify_bundle(root)

    def test_import_does_not_import_torch(self):
        code="import sys;sys.path.insert(0,sys.argv[1]);import cts_native_build;print('torch' in sys.modules)"
        result = subprocess.run([sys.executable, '-c', code, str(BASE)], capture_output=True, text=True, check=True)
        self.assertEqual(result.stdout.strip(), 'False')


if __name__ == '__main__':
    unittest.main()
