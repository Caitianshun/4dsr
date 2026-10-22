"""CPU tests for fixed scope, scene identity, and immutable registration reuse."""
from __future__ import annotations
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
import common
import prepare_suite as suite


class SixSceneSuiteTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.lr = self.root / 'lr/image.png'
        self.lr.parent.mkdir()
        self.lr.write_bytes(b'training LR identity fixture; no image inference')
        init = self.root / 'points_lr.npz'
        init.write_bytes(b'train LR initializer identity fixture')
        self.manifest = dict(schema='n3dv_dynamic_sr_pilot_v1', scene='cook_spinach', scale=4,
            physical_lr_derivation={'formation':'area4'}, frame_indices=list(range(300)),
            splits={'train':['cam01','cam02'], 'test':['cam00']},
            cameras={camera: {'c2w':[[1,0,0,i],[0,1,0,0],[0,0,1,1],[0,0,0,1]]}
                for i,camera in enumerate(['cam00','cam01','cam02'])},
            observations=[dict(camera_id=camera,frame_index=frame,time=frame/300.,split='train',
                lr_path='lr/image.png',lr_sha256=common.sha(self.lr))
                for frame in range(300) for camera in ['cam01','cam02']],
            initialization=dict(train_only_lr=True,npz_path=init.name,npz_sha256=common.sha(init),point_count=2))
        self.manifest_path = self.root / 'manifest.json'
        self.manifest_path.write_text(json.dumps(self.manifest))
        self.source = self.root / 'source.json'
        self.source.write_text(json.dumps(dict(scene='cook_spinach', configuration={'model':'Wu'},
            data={'manifest':common.identity(self.manifest_path),'cameras_extent':1.0})))

    def payloads(self):
        return suite.registration_payloads('Cook07', self.manifest_path, self.source)[2]

    def test_exact_six_scene_scope_and_300_original_frames(self):
        self.assertEqual(suite.SCENES, (('Cook07','cook_spinach'),('Cut07','cut_roasted_beef'),
            ('Coffee07','coffee_martini'),('Flame07','flame_steak'),
            ('Discussion07','meetroom_discussion'),('VRHeadset07','meetroom_vrheadset')))
        suite.validate_scope('Cook07', self.manifest)
        for frames in [list(range(299)), list(range(1,301)), [0,1,1]]:
            bad=copy.deepcopy(self.manifest);bad['frame_indices']=frames
            with self.assertRaisesRegex(ValueError,'original frames'):
                suite.validate_scope('Cook07',bad)
        with self.assertRaisesRegex(ValueError,'identity differs'):
            suite.validate_scope('Coffee07',self.manifest)

    def test_all_four_methods_bind_same_scene_and_input_identity(self):
        payloads=self.payloads()
        self.assertEqual(len(payloads),6)
        for method in common.METHODS:
            cfg=payloads[method+'.json']
            self.assertEqual(cfg['scene'],'cook_spinach')
            self.assertEqual(cfg['manifest'],common.identity(self.manifest_path))
            self.assertEqual(cfg['source_configuration'],common.identity(self.source))
            self.assertEqual(cfg['information']['train_observations'],600)
            self.assertFalse(cfg['information']['training_HR_read'])
        self.assertTrue(payloads['registration.json']['LR_bytes_verified'])
        self.assertTrue(payloads['splatsure.json']['label'].endswith('/ stablesr'))

    def test_existing_registration_reused_without_writes(self):
        out=self.root/'registered';payloads=self.payloads()
        self.assertEqual(suite.materialize_registration(out,payloads),'created')
        original={p.name:(p.read_bytes(),p.stat().st_mtime_ns) for p in out.iterdir()}
        self.assertEqual(suite.materialize_registration(out,payloads),'reused_verified')
        self.assertEqual(original,{p.name:(p.read_bytes(),p.stat().st_mtime_ns) for p in out.iterdir()})

    def test_mismatch_and_partial_outputs_rejected_without_repair(self):
        out=self.root/'registered';payloads=self.payloads()
        suite.materialize_registration(out,payloads)
        broken=out/'iesrgs.json';broken.write_text('{"scene":"foreign_scene"}')
        before={p.name:p.read_bytes() for p in out.iterdir()}
        with self.assertRaisesRegex(ValueError,'preserved without changes'):
            suite.materialize_registration(out,payloads)
        self.assertEqual(before,{p.name:p.read_bytes() for p in out.iterdir()})
        broken.unlink()
        with self.assertRaisesRegex(ValueError,'preserved without changes'):
            suite.materialize_registration(out,payloads)
        self.assertFalse(broken.exists())

    def test_lr_bytes_changed_rejects_registration(self):
        self.lr.write_bytes(b'changed after manifest registration')
        with self.assertRaisesRegex(ValueError,'LR bytes changed'):
            self.payloads()

    def test_source_from_foreign_scene_rejected(self):
        source=json.loads(self.source.read_text());source['scene']='coffee_martini'
        self.source.write_text(json.dumps(source))
        with self.assertRaisesRegex(ValueError,'different scene'):
            self.payloads()

    def test_plan_worker_requires_cpu_only_single_thread_environment(self):
        with mock.patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES':'0'},clear=True):
            with self.assertRaisesRegex(RuntimeError,'CPU-only single-thread'):
                suite.plan_worker('Cook07')
        with self.assertRaisesRegex(ValueError,'independent single-thread'):
            suite.plan_suite(workers=1)


if __name__ == '__main__':
    unittest.main()
