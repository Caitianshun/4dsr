"""CPU-only boundaries for the target-host method builder; no compiler calls."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
import cts_method_build as build


class CTSMethodBuildContracts(unittest.TestCase):
    def test_existing_output_cannot_be_replaced(self):
        with tempfile.TemporaryDirectory() as root, mock.patch.object(build, 'verify_plan', return_value={}):
            with self.assertRaises(FileExistsError):
                build.build('verified.json', root, root, 'gcc', 'g++', 2)

    def test_source_inventory_excludes_prior_binaries_and_git_metadata(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            for name in ['setup.py', 'cuda_rasterizer/forward.cu', 'build/old.o', 'pkg/_C.so', '.git/config', 'pkg/__pycache__/old.pyc']:
                path = root/name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text('fixture')
            self.assertEqual({str(p.relative_to(root)) for p in build.source_files(root)},
                             {'setup.py', 'cuda_rasterizer/forward.cu'})

    def test_ie_patch_rejects_already_modified_source(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root); path = root/'cuda_rasterizer/rasterizer_impl.h'
            path.parent.mkdir(); path.write_text('#pragma once\n#include <cstdint>\n')
            with self.assertRaisesRegex(ValueError, 'original CUDA compatibility patch'):
                build.expected_ie_patch(root)

    def test_import_does_not_import_torch(self):
        code="import sys;sys.path.insert(0,sys.argv[1]);import cts_method_build;print('torch' in sys.modules)"
        result=subprocess.run([sys.executable,'-c',code,str(BASE)],capture_output=True,text=True,check=True)
        self.assertEqual(result.stdout.strip(),'False')


if __name__=='__main__': unittest.main()
