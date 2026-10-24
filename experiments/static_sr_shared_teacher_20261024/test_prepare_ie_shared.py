"""CPU-only shared-RGB IE producer contracts; synthetic fixtures are not teachers."""
import copy
import importlib
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from experiments.static_sr_shared_teacher_20261024 import prepare_ie_shared as ie


class FakeDepth:
    def __init__(self, value=None):
        self.value, self.inputs = value, []

    def infer_image(self, bgr, size):
        self.inputs.append((bgr.copy(), size))
        if self.value is not None:
            return self.value.copy()
        return np.arange(bgr.shape[0] * bgr.shape[1], dtype=np.float32).reshape(bgr.shape[:2])


class SharedIE(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.rgb = np.array([[[4, 39, 250], [10, 80, 140], [255, 0, 17]],
                             [[35, 48, 57], [71, 96, 101], [123, 145, 222]]], dtype=np.uint8)
        self.png = self.root / 'data/sr.png'
        self.png.parent.mkdir()
        Image.fromarray(self.rgb).save(self.png)
        self.sid = self.ident(self.png)
        self.manifest = dict(scene='unit_scene', splits=dict(train=['cam01'], test=['cam00']),
            frame_indices=[0, 149], resolutions=dict(hr=[3, 2]),
            observations=[dict(camera_id='cam01', frame_index=f, time=f / 300.,
                               split='train', lr_sha256='a' * 64) for f in (0, 149)])
        self.mp = self.write('data/manifest.json', self.manifest)
        self.mid = self.ident(self.mp)
        self.config = dict(schema=ie.common.SCHEMA, method='iesrgs', scene='unit_scene', manifest=self.mid)
        self.cp = self.write('config.json', self.config)
        self.source = self.root / 'source.py'
        self.source.write_text('# synthetic unit source\n')
        self.weight = self.root / ie.DEPTH_NAME
        self.weight.write_bytes(b'unit-model-not-real-weights')
        self.assets = dict(upstream=str(self.root / 'upstream'), upstream_commit=ie.COMMIT,
                           architecture=ie.DEPTH_ARCH, weights=self.ident(self.weight), sources=[self.ident(self.source)])
        self.shared = dict(tag='sr_swinir_x4_areaft', model_identities=[self.ident(self.weight)],
            producer=[self.ident(self.source)], model_provenance={'synthetic_unit_test': True},
            entries=[dict(key=f'cam01/{f:04d}', camera='cam01', frame=f, time=f / 300.,
                          lr_sha256='a' * 64, sr=self.sid, shape=[3, 2]) for f in (0, 149)])
        self.cache_file = self.write('shared.json', self.shared)

    def tearDown(self):
        self.tmp.cleanup()

    def ident(self, p):
        return ie.cache.identity(p, project_root=self.root)

    def write(self, name, obj):
        p = self.root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(obj))
        return p

    def plan(self, **kw):
        with patch.object(ie.cache, 'verify_cache', return_value=copy.deepcopy(self.shared)), \
             patch.object(ie, 'depth_assets', return_value=self.assets):
            return ie.build_plan(self.cp, self.cache_file, self.root / 'upstream', self.weight,
                                 self.root / 'output/new', project_root=self.root, **kw)

    def test_rgb8_bgr_and_exact_source_depth_quantization(self):
        prediction = np.array([[10, 11, 12], [13, 14, 17]], dtype=np.float32)
        model = FakeDepth(prediction)
        got = ie.png_inverse_depth(self.sid, [3, 2], model, self.root)
        expected = (((prediction - 10) / 7 * 255).astype(np.uint8).astype(np.float32) / 65536.)[None]
        self.assertEqual(got.dtype, np.float32)
        self.assertEqual(got.tobytes(), expected.tobytes())
        self.assertEqual(model.inputs[0][1], 518)
        self.assertEqual(model.inputs[0][0].dtype, np.uint8)
        self.assertEqual(model.inputs[0][0].tobytes(), self.rgb[:, :, ::-1].copy().tobytes())
        self.assertTrue(model.inputs[0][0].flags.c_contiguous)

    def test_original_training_reader_decodes_same_png(self):
        path = str(ie.ROOT / 'experiments/static_sr_wu_20261022')
        with patch.dict(sys.modules, {'common': ie.common}):
            sys.path.insert(0, path)
            try:
                train = importlib.import_module('train_stage')
            finally:
                sys.path.remove(path)
        item = dict(self.sid, path=str(self.png))
        got = train.load_tensor(item, device='cpu')
        expected = torch.from_numpy(self.rgb.copy()).permute(2, 0, 1).float() / 255
        self.assertTrue(torch.equal(got, expected))

    def test_invalid_depth_shape_nan_constant_rejected(self):
        for prediction in (np.zeros((2, 3), np.float32), np.zeros((3, 2), np.float32),
                           np.full((2, 3), np.nan, np.float32)):
            with self.subTest(shape=prediction.shape), self.assertRaises(ValueError):
                ie.png_inverse_depth(self.sid, [3, 2], FakeDepth(prediction), self.root)

    def test_png_hash_size_palette_and_rgba_rejected(self):
        with self.assertRaises(ValueError):
            ie.png_inverse_depth(dict(self.sid, sha256='b' * 64), [3, 2], FakeDepth(), self.root)
        with self.assertRaises(ValueError):
            ie.png_inverse_depth(self.sid, [2, 3], FakeDepth(), self.root)
        for mode in ('P', 'RGBA'):
            p = self.root / f'{mode}.png'
            Image.fromarray(self.rgb).convert(mode).save(p)
            with self.assertRaises(ValueError):
                ie.png_inverse_depth(self.ident(p), [3, 2], FakeDepth(), self.root)

    def test_plan_full_or_single_legal_fixture_no_model_load(self):
        with patch.object(ie, 'load_depth_only', side_effect=AssertionError('must not load')):
            full = self.plan()
            fixture = self.plan(fixture_key='cam01/0149')
        self.assertEqual(full['selected_key_count'], 2)
        self.assertEqual(fixture['selected_keys'], ['cam01/0149'])
        self.assertEqual(fixture['legal_key_count'], 2)
        self.assertEqual(full['status'], 'not_GPU_executed')
        self.assertFalse((self.root / 'output').exists())
        for key in ('cam00/0149', '../cam01/0149', 'cam01/0001'):
            with self.assertRaises(ValueError):
                self.plan(fixture_key=key)

    def test_missing_duplicate_heldout_time_lr_rejected(self):
        bad = []
        missing = copy.deepcopy(self.shared); missing['entries'].pop(); bad.append(missing)
        duplicate = copy.deepcopy(self.shared); duplicate['entries'].append(duplicate['entries'][0]); bad.append(duplicate)
        for field, value in [('camera', 'cam00'), ('time', 0.3), ('lr_sha256', 'f' * 64), ('shape', [9, 9])]:
            changed = copy.deepcopy(self.shared); changed['entries'][0][field] = value; bad.append(changed)
        for shared in bad:
            with self.subTest(shared=shared), self.assertRaises(ValueError):
                ie._aligned_rows(shared, self.manifest)

    def test_output_is_fresh_project_output_and_no_symlinks(self):
        self.assertEqual(ie.fresh_output(self.root / 'output/new', self.root), self.root / 'output/new')
        (self.root / 'output').mkdir()
        (self.root / 'output/old').mkdir()
        (self.root / 'output/link').symlink_to(self.root / 'data', target_is_directory=True)
        for p in (self.root / 'output/old', self.root / 'data/new', self.root / 'output',
                  self.root / 'output/link/new', self.root / 'output/../data/new', Path('/tmp/outside-ie')):
            with self.subTest(path=p), self.assertRaises((ValueError, FileExistsError)):
                ie.fresh_output(p, self.root)

    def test_writer_references_png_without_rgb_reencoding(self):
        for fmt in ie.CACHE_FORMATS:
            plan = self.plan(cache_format=fmt)
            out = self.root / 'output' / fmt
            out.mkdir(parents=True)
            result = ie._write_rows(plan, self.shared, self.shared['entries'], FakeDepth(), out, {'synthetic_unit_test': True})
            doc = json.loads(result.read_text())
            self.assertEqual(doc['kind'], ie.FORMAL_KIND)
            self.assertEqual(doc['method'], 'iesrgs')
            self.assertFalse(doc['fixture'])
            self.assertEqual(doc['entries'][0]['external_rgb'], self.sid)
            self.assertEqual(len(list(out.rglob('*.png'))), 0)
            self.assertFalse(any('external_rgb' in p.name for p in out.rglob('*')))
            a = ie.common.load_prior_array(doc['entries'][0]['external_inverse_depth'], require_float32=True)
            expected = ie.png_inverse_depth(self.sid, [3, 2], FakeDepth(), self.root)
            self.assertEqual(a.tobytes(), expected.tobytes())
            self.assertEqual(self.ident(self.png), self.sid)

    def test_fixture_written_as_separate_schema(self):
        plan = self.plan(fixture_key='cam01/0149')
        out = self.root / 'output/fixture'; out.mkdir(parents=True)
        stats = dict(GPU_executed=True, hardware={'name': 'mock-only'}, elapsed_seconds=1.2,
                     peak_allocated_bytes=123, peak_reserved_bytes=456)
        p = ie._write_rows(plan, self.shared, self.shared['entries'][1:], FakeDepth(), out, {}, lambda: stats)
        doc = json.loads(p.read_text())
        self.assertEqual(p.name, 'fixture_receipt.json')
        self.assertEqual(doc['schema'], ie.FIXTURE_SCHEMA)
        self.assertTrue(doc['fixture'])
        self.assertEqual(doc['execution'], stats)
        self.assertFalse((out / 'index.json').exists())

    def merge_inputs(self, meta=None):
        producers = []
        for name in ('iesrgs_priors.py', 'iesrgs.py', 'iesrgs_renderer.py'):
            p = self.root / 'experiments/static_sr_wu_20261022/methods' / name
            p.parent.mkdir(parents=True, exist_ok=True); p.write_text('# synthetic unit source\n')
            producers.append(self.ident(p))
        checkpoint = self.root / 'unit_checkpoint.pt'
        torch.save(dict(metadata=meta or dict(stage='iesrgs_internal', iterations=30000, optimizer_updates=29999),
                        adaptation_config=dict(method='iesrgs', manifest=self.mid)), checkpoint)
        def array(name, shape):
            p = self.root / name
            np.save(p, np.arange(np.prod(shape), dtype=np.float32).reshape(shape), allow_pickle=False)
            return self.ident(p)
        depth = array('depth.npy', (1, 2, 3)); rgb = array('rgb.npy', (3, 2, 3))
        base = dict(schema='static_sr_wu_prior_bundle_v1', manifest=self.mid, scene='unit_scene',
                    upstream_commit=ie.COMMIT, training_only=True)
        producer = self.root / 'experiments/static_sr_shared_teacher_20261024/prepare_ie_shared.py'
        producer.parent.mkdir(parents=True, exist_ok=True); producer.write_text('# synthetic unit producer\n')
        external = dict(base, status='completed', kind=ie.FORMAL_KIND, fixture=False,
            shared_cache=self.ident(self.cache_file), producer=[self.ident(self.source), self.ident(producer)],
            model_identities=self.shared['model_identities'] + [self.ident(self.weight)],
            depth_load=dict(strict_load=True, architecture=ie.DEPTH_ARCH, SwinIR_loaded=False, weights=self.ident(self.weight)),
            cache_formats={'external_rgb': 'existing_RGB8_PNG'}, entries=[])
        internal = dict(base, kind='iesrgs_internal', producer=producers, model_identities=[self.ident(checkpoint)],
            internal_teacher=dict(checkpoint=self.ident(checkpoint), representation='Wu_4DGS_with_Mip_filter',
                depth='RADE_median_depth', filter_reference_resolution='lr', frozen=True,
                same_time_training_views_only=True), entries=[])
        for e in self.shared['entries']:
            entry = {k: e[k] for k in ('key', 'camera', 'frame', 'time', 'lr_sha256')}
            external['entries'].append(dict(entry, external_rgb=e['sr'], shared_sr_sha256=e['sr']['sha256'], external_inverse_depth=depth))
            internal['entries'].append(dict(entry, internal_rgb=rgb, internal_depth=depth))
        return external, internal

    def merge(self, external, internal):
        ep = self.write('external.json', external); ip = self.write('internal.json', internal)
        with patch.object(ie.cache, 'verify_cache', return_value=self.shared), \
             patch.object(ie, 'DEPTH_SHA', self.ident(self.weight)['sha256']):
            return ie.merge_shared(ep, ip, self.root / 'output/merged.json', project_root=self.root)

    def test_merge_new_kind_compatible_fields(self):
        external, internal = self.merge_inputs()
        result = json.loads(self.merge(external, internal).read_text())
        self.assertEqual(result['kind'], ie.COMPLETE_KIND)
        self.assertEqual(result['method'], 'iesrgs')
        self.assertEqual(len(result['entries']), 2)
        self.assertEqual(result['entries'][0]['external_rgb'], self.sid)
        self.assertIn('internal_depth', result['entries'][0])
        self.assertIn('source_indices', result)

    def test_merge_rejects_fixture_missing_heldout_wrong_rgb_time(self):
        external, internal = self.merge_inputs()
        mutations = [lambda x: x.update(schema=ie.FIXTURE_SCHEMA, fixture=True),
                     lambda x: x.update(status='running'), lambda x: x['entries'].pop(),
                     lambda x: x['entries'][0].update(camera='cam00'),
                     lambda x: x['entries'][0].update(time=0.4),
                     lambda x: x['entries'][0].update(external_rgb=self.ident(self.source))]
        for mutate in mutations:
            e = copy.deepcopy(external); mutate(e)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                self.merge(e, internal)

    def test_merge_rejects_ordinary_lr_and_short_internal_checkpoint(self):
        for meta in (dict(stage='lr', iterations=30000, optimizer_updates=29999),
                     dict(stage='iesrgs_internal', iterations=3, optimizer_updates=2),
                     dict(stage='iesrgs_internal', iterations=30000, optimizer_updates=30000)):
            external, internal = self.merge_inputs(meta)
            with self.subTest(meta=meta), self.assertRaises(ValueError):
                self.merge(external, internal)

    def test_merge_rejects_wrong_depth_representation_or_missing_producer(self):
        for mode in ('depth', 'producer', 'shape'):
            external, internal = self.merge_inputs()
            if mode == 'depth': internal['internal_teacher']['depth'] = 'Wu_accumulated_depth'
            elif mode == 'producer': internal['producer'].pop()
            else: internal['entries'][0]['internal_depth'] = internal['entries'][0]['internal_rgb']
            with self.subTest(mode=mode), self.assertRaises(ValueError): self.merge(external, internal)

    def test_merge_requires_strict_vitl_load_weights_and_producer_identity(self):
        external, internal = self.merge_inputs()
        changes = [lambda e: e['depth_load'].update(strict_load=False),
                   lambda e: e['depth_load'].update(architecture=dict(encoder='vits')),
                   lambda e: e['depth_load']['weights'].update(sha256='d' * 64),
                   lambda e: e['model_identities'].pop(),
                   lambda e: e['producer'].pop()]
        for change in changes:
            e = copy.deepcopy(external); change(e)
            with self.subTest(change=change), self.assertRaises(ValueError): self.merge(e, internal)

    def test_source_check_refuses_wrong_git_commit_before_weight_loading(self):
        with patch.object(ie.subprocess, 'check_output', return_value=b'wrong\n'), self.assertRaises(ValueError):
            ie.depth_assets(self.root, self.weight, self.root)

    def test_strict_depth_only_loader_uses_exact_architecture_and_cpu_load(self):
        package = self.root / 'upstream' / ie.DEPTH_PREFIX / 'depth_anything_v2'
        package.mkdir(parents=True)
        dpt = package / 'dpt.py'; dpt.write_text('# synthetic unit module\n')
        assets = dict(self.assets, sources=[self.ident(dpt)])
        calls = []
        class Model:
            def __init__(self, **kw): calls.append(('architecture', kw))
            def load_state_dict(self, value, strict): calls.append(('strict', strict, value))
            def requires_grad_(self, value): calls.append(('grad', value)); return self
            def eval(self): return self
            def to(self, device): calls.append(('device', device)); return self
        namespace = types.ModuleType('depth_anything_v2'); namespace.__path__ = [str(package)]
        module = types.ModuleType('depth_anything_v2.dpt'); module.__file__ = str(dpt); module.DepthAnythingV2 = Model
        with patch.object(ie, 'depth_assets', return_value=assets), \
             patch.dict(sys.modules, {'depth_anything_v2': namespace, 'depth_anything_v2.dpt': module}), \
             patch.object(torch, 'load', return_value={'unit': 'state'}) as load:
            _, receipt = ie.load_depth_only(assets, device='cpu', project_root=self.root)
        self.assertEqual(calls[0], ('architecture', ie.DEPTH_ARCH))
        self.assertIn(('strict', True, {'unit': 'state'}), calls)
        self.assertIn(('grad', False), calls)
        self.assertEqual(load.call_args.kwargs, dict(map_location='cpu', weights_only=True))
        self.assertFalse(receipt['SwinIR_loaded'])
        self.assertTrue(receipt['strict_load'])


if __name__ == '__main__':
    unittest.main()
