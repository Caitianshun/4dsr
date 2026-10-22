"""Integration boundaries not covered by individual paper loss parity tests."""
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
from common import ROOT, identity, training_manifest, verify_prior_index
from prepare import build
from train_stage import preflight, topology

MANIFEST=ROOT/'data/dynamic_sr/full_time_20261007/prepared/cook_spinach_area4/manifest_train_ready_seed20261007.json'
PLAN=ROOT/'output/dynamic_sr_physlr_20261016/registry/cook_spinach_area4/seed_20261007/swinir_teacher/Bsync/config.json'

class Integration(unittest.TestCase):
    def test_real_train_only_records(self):
        m,rows=training_manifest(MANIFEST)
        self.assertEqual(len(rows),6000)
        self.assertEqual(len({r['frame_index'] for r in rows}),300)
        self.assertNotIn('cam00',{r['camera_id'] for r in rows})
        self.assertEqual(m['physical_lr_derivation']['formation'],'area4')

    def test_no_hr_image_reads(self):
        original=Path.open
        def open_guard(path,*a,**kw):
            if path.suffix=='.png' and ('hr' in path.parts or 'cam00' in path.parts):
                raise AssertionError('Training contract tried to open held-out or HR image')
            return original(path,*a,**kw)
        with patch.object(Path,'open',open_guard):
            training_manifest(MANIFEST,verify_pixels=True)

    def test_missing_prior_cannot_start(self):
        cfg=build(MANIFEST,PLAN,'splatsure')
        with self.assertRaisesRegex(ValueError,'prior bundle'):
            preflight(cfg,'sr')

    def test_foreign_manifest_rejected(self):
        cfg=build(MANIFEST,PLAN,'iesrgs')
        cfg['manifest']['sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'Identity changed'):
            preflight(cfg,'internal')

    def test_fresh_and_continued_source_recipes(self):
        for method in ('splatsure','iesrgs','sequence_matters'):
            self.assertEqual(build(MANIFEST,PLAN,method)['specification']['init_sr'],'fresh')
        cfg=build(MANIFEST,PLAN,'threedsr')
        self.assertEqual(cfg['specification']['init_sr'],'lr_parent')
        self.assertEqual(cfg['specification']['lr_steps'],30000)

    def test_camera_metadata_geometry(self):
        # Cameras are CPU tensors in Wu and can be tested without creating a model or CUDA context.
        import torch
        import numpy as np
        sys.path.insert(0,str(ROOT/'third_party/4dgs_843d5ac_runtime'))
        sys.path.insert(0,str(ROOT/'experiments/dynamic_sr_20260918'))
        from backend import WuBackend
        m,rows=training_manifest(MANIFEST)
        b=WuBackend.__new__(WuBackend);b.manifest=m;b.torch=torch
        b.indices={(o['camera_id'],o['frame_index']):i for i,o in enumerate(rows)}
        cam=b.camera(rows[0],'hr')
        w,h=m['resolutions']['hr'];k=np.asarray(m['cameras'][rows[0]['camera_id']]['K_hr'])
        self.assertEqual((cam.image_width,cam.image_height),(w,h))
        self.assertAlmostEqual(float(cam.projection_matrix[2,0]),(2*k[0,2]+1-w)/w,places=6)
        self.assertAlmostEqual(float(cam.projection_matrix[2,1]),(2*k[1,2]+1-h)/h,places=6)
        held=next(o for o in m['observations'] if o['camera_id']=='cam00')
        with self.assertRaisesRegex(ValueError,'Held-out'):
            b.camera(held)
        self.assertEqual(b.camera(held,allow_heldout=True).image_width,w)

    def test_source_topology_final_and_round_boundaries(self):
        # A trap proves no densification is dispatched after the source cutoffs.
        b=SimpleNamespace(config={'method':'threedsr'},g=None,torch=None)
        topology(b,[],2500,5000,Path('/unused'))
        topology(b,[],5000,5000,Path('/unused'))
        b.config['method']='splatsure'
        topology(b,[],15000,30000,Path('/unused'))

if __name__=='__main__':unittest.main()
