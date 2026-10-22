"""CPU-only boundary checks for the exact IE external-teacher acceptance path."""
import os
from pathlib import Path
import subprocess
import sys
import unittest
import numpy as np

BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE))
import external_teacher_acceptance as acceptance


class ExternalTeacherAcceptanceContracts(unittest.TestCase):
    def setUp(self):
        self.resolutions={'lr':[4,3],'hr':[16,12]}
        self.sr=np.full((3,12,16),.5,dtype=np.float32)
        self.depth=(np.arange(192,dtype=np.float32).reshape(1,12,16)%256)/65536.

    def test_source_depth_representation_and_exact_shape(self):
        result=acceptance.validate_arrays(self.sr,self.depth,self.resolutions)
        self.assertEqual(result['external_rgb']['shape'],[3,12,16])
        self.assertEqual(result['external_inverse_depth']['dtype'],'float32')
        with self.assertRaisesRegex(ValueError,'shape'):
            acceptance.validate_arrays(self.sr[:,:10],self.depth,self.resolutions)

    def test_wrong_depth_scaling_or_quantization_rejected(self):
        with self.assertRaisesRegex(ValueError,'representation'):
            acceptance.validate_arrays(self.sr,self.depth*256,self.resolutions)
        with self.assertRaisesRegex(ValueError,'quantized'):
            acceptance.validate_arrays(self.sr,self.depth+1/131072,self.resolutions)

    def test_constant_nonfinite_and_float64_outputs_rejected(self):
        with self.assertRaisesRegex(ValueError,'Constant'):
            acceptance.validate_arrays(self.sr,np.zeros_like(self.depth),self.resolutions)
        broken=self.sr.copy();broken[0,0,0]=float('nan')
        with self.assertRaisesRegex(ValueError,'Nonfinite'):
            acceptance.validate_arrays(broken,self.depth,self.resolutions)
        with self.assertRaisesRegex(ValueError,'float32'):
            acceptance.validate_arrays(self.sr.astype(np.float64),self.depth,self.resolutions)

    def test_import_does_not_import_torch(self):
        code="import sys;sys.path.insert(0,sys.argv[1]);import external_teacher_acceptance;print('torch' in sys.modules)"
        completed=subprocess.run([sys.executable,'-c',code,str(BASE)],capture_output=True,text=True,check=True,
            env={**os.environ,'CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1'})
        self.assertEqual(completed.stdout.strip(),'False')


if __name__=='__main__':unittest.main()
