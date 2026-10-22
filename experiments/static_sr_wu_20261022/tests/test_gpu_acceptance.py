"""CPU contracts for GPU acceptance scope and non-formal artifact barriers."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE))
import gpu_acceptance as acceptance
from common import validate_lr_parent_payload


class AcceptanceContracts(unittest.TestCase):
    def setUp(self):
        self.manifest=dict(frame_indices=list(range(300)),splits={'train':['cam01','cam02','cam03'],'test':['cam00']})
        self.rows=[dict(camera_id=c,frame_index=f,time=f/300.,split='train')
            for f in [0,149,299] for c in ['cam01','cam02','cam03']]

    def test_full_original_time_and_three_training_views(self):
        groups=acceptance.select_samples(self.manifest,self.rows,[0,149,299])
        self.assertEqual([g[0]['time'] for g in groups],[0,149/300.,299/300.])
        self.assertEqual([len(g) for g in groups],[3,3,3])
        with self.assertRaisesRegex(ValueError,'distinct original'):
            acceptance.select_samples(self.manifest,self.rows,[0,0])

    def test_heldout_missing_camera_and_renormalized_time_rejected(self):
        for field,value in [('camera_id','cam00'),('split','test'),('time',1.)]:
            altered=[dict(r) for r in self.rows];altered[0][field]=value
            with self.assertRaises(ValueError):acceptance.select_samples(self.manifest,altered,[0,149,299])
        with self.assertRaisesRegex(ValueError,'three distinct'):
            acceptance.select_samples(self.manifest,self.rows[1:],[0,149,299])

    def test_fixture_can_never_satisfy_formal_parent_schema_or_metadata(self):
        source={'schema':'static_sr_wu_checkpoint_v1','model':'identity fixture',
            'metadata':{'stage':'lr','iterations':30000,'optimizer_updates':29999},
            'adaptation_config':{'method':'splatsure'}}
        sealed=acceptance.seal_checkpoint(source)
        self.assertEqual(source['schema'],'static_sr_wu_checkpoint_v1')
        self.assertEqual(sealed['schema'],acceptance.FIXTURE_SCHEMA)
        self.assertEqual(sealed['metadata']['stage'],acceptance.FIXTURE_STAGE)
        self.assertFalse(sealed['metadata']['formal_parent_allowed'])
        self.assertFalse(sealed['metadata']['formal_prior_allowed'])
        self.assertNotIn(sealed['schema'],('static_sr_wu_checkpoint_v1','full_author_LR_prefix_checkpoint_v1'))
        with self.assertRaises(ValueError):validate_lr_parent_payload(sealed,'splatsure')

    def test_import_does_not_import_torch_or_select_gpu(self):
        code="import sys;sys.path.insert(0,sys.argv[1]);import gpu_acceptance;print('torch' in sys.modules)"
        completed=subprocess.run([sys.executable,'-c',code,str(BASE)],capture_output=True,text=True,
            env={**os.environ,'CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1'},check=True)
        self.assertEqual(completed.stdout.strip(),'False')

    def test_raw_equivalence_gate_does_not_hide_clipping_difference(self):
        import torch
        a=torch.full((3,4,4),1.2);b=torch.full((3,4,4),1.3)
        self.assertFalse(acceptance.compare_images(a,b)['passed'])
        self.assertTrue(acceptance.compare_images(a.clamp(0,1),b.clamp(0,1))['passed'])
        self.assertFalse(acceptance.compare_images(torch.zeros(2,3),torch.ones(2,3))['passed'])

    def test_roundtrip_digest_covers_optimizer_moments_and_tensor_values(self):
        import torch
        source={'model':torch.tensor([1.,2.]),'adam':{'step':torch.tensor(1.),'exp_avg':torch.tensor([.1,.2])}}
        duplicate={'model':source['model'].clone(),'adam':{k:v.clone() for k,v in source['adam'].items()}}
        self.assertEqual(acceptance.tree_digest(source),acceptance.tree_digest(duplicate))
        duplicate['adam']['exp_avg'][0]=.3
        self.assertNotEqual(acceptance.tree_digest(source),acceptance.tree_digest(duplicate))

    def test_ie_fixture_uses_production_lossless_writer_and_reader(self):
        import numpy as np
        import torch
        from common import load_prior_array
        value=torch.linspace(-.125,1.25,96,dtype=torch.float32).reshape(3,4,8)
        for codec in ('npz_deflate','npz_byte_shuffle'):
            with tempfile.TemporaryDirectory() as directory:
                entry=acceptance.save_array(Path(directory),'NONFORMAL_ie',value,codec)
                self.assertEqual(entry['storage']['format'],codec)
                recovered=load_prior_array(entry)
                self.assertEqual(recovered.tobytes(),value.numpy().tobytes())
                self.assertEqual(recovered.dtype,np.dtype('float32'))
                self.assertNotIn('schema',entry)


if __name__=='__main__':unittest.main()
