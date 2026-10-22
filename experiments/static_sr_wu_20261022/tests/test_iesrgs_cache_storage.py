"""CPU contracts for optional IE containers; no teacher inference or CUDA."""
from copy import deepcopy
from pathlib import Path
import io
import json
import sys
import tempfile
import unittest
import warnings
import zipfile
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from common import identity, load_prior_array, read, write_new
from methods.iesrgs import IESRGSPriors, fusion_loss
from methods.iesrgs_priors import CACHE_FORMATS, _save_array, load_priors, merge_priors
from train_stage import load_tensor


class CacheStorageContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_both_formats_restore_signed_zero_subnormal_float32_bits(self):
        array = np.array([0, 0x80000000, 1, 0x3f800001, 0x7f7fffff], dtype=np.uint32).view(np.float32)
        with tempfile.TemporaryDirectory() as directory:
            for fmt in CACHE_FORMATS:
                item = _save_array(Path(directory) / fmt, 'a.npy', torch.from_numpy(array), fmt)
                actual = load_prior_array(item, require_float32=True)
                self.assertEqual(actual.dtype, array.dtype)
                self.assertEqual(actual.shape, array.shape)
                self.assertEqual(actual.tobytes(), array.tobytes())
                self.assertEqual(load_tensor(item, 'cpu').numpy().tobytes(), array.tobytes())
                if fmt == 'npz_deflate':
                    with zipfile.ZipFile(item['path']) as archive:
                        self.assertEqual(archive.namelist(), ['array.npy'])
                        self.assertEqual(archive.infolist()[0].compress_type, zipfile.ZIP_DEFLATED)

    def test_legacy_npy_and_metadata_free_npz_are_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            array = torch.arange(24, dtype=torch.float32).reshape(3, 2, 4)
            for fmt in CACHE_FORMATS:
                item = _save_array(Path(directory) / fmt, 'a.npy', array, fmt)
                item.pop('storage')
                self.assertTrue(torch.equal(load_tensor(item, 'cpu'), array))
                entry = {key: item for key in ('internal_rgb', 'internal_depth', 'external_rgb', 'external_inverse_depth')}
                self.assertTrue(torch.equal(load_priors(entry, 'cpu').external_rgb, array))
            # Existing other-method numeric NPY caches retain their float() conversion.
            path = Path(directory) / 'old_int.npy'
            np.save(path, np.arange(6, dtype=np.uint8), allow_pickle=False)
            self.assertEqual(load_tensor(identity(path), 'cpu').dtype, torch.float32)

    def test_wrong_file_sha_and_supplied_array_metadata_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            item = _save_array(Path(directory), 'a.npy', torch.ones(1, 2, 3), 'npz_deflate')
            changed = deepcopy(item)
            changed['sha256'] = '0' * 64
            with self.assertRaisesRegex(ValueError, 'Identity changed'):
                load_prior_array(changed)
            for field, value in [('format', 'npy'), ('dtype', 'float16'), ('shape', [6]),
                                 ('array_order', 'F'), ('array_sha256', '0' * 64)]:
                changed = deepcopy(item)
                changed['storage'][field] = value
                with self.assertRaisesRegex(ValueError, 'metadata differs'):
                    load_prior_array(changed)

    def test_extra_wrong_duplicate_and_uncompressed_zip_members_are_rejected(self):
        buffer = io.BytesIO()
        np.save(buffer, np.ones(2, dtype=np.float32), allow_pickle=False)
        with tempfile.TemporaryDirectory() as directory:
            for i, (names, compression) in enumerate([
                    (['array.npy', 'extra.npy'], zipfile.ZIP_DEFLATED),
                    (['wrong.npy'], zipfile.ZIP_DEFLATED),
                    (['array.npy', 'array.npy'], zipfile.ZIP_DEFLATED),
                    (['array.npy'], zipfile.ZIP_STORED)]):
                path = Path(directory) / f'{i}.npz'
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', UserWarning)
                    with zipfile.ZipFile(path, 'w', compression=compression) as archive:
                        for name in names:
                            archive.writestr(name, buffer.getvalue())
                with self.assertRaisesRegex(ValueError, 'Prior NPZ'):
                    load_prior_array(identity(path))

    def test_pickle_nonfloat32_and_nonfinite_npz_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            for i, array in enumerate([np.array([{'bad': 1}], dtype=object),
                    np.ones(2, dtype=np.float16), np.array([np.nan], dtype=np.float32)]):
                path = Path(directory) / f'{i}.npz'
                np.savez_compressed(path, array=array)
                with self.assertRaises(ValueError):
                    load_prior_array(identity(path))

    def test_producer_default_format_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            item = _save_array(root, 'a.npy', torch.ones(3))
            self.assertEqual(item['storage']['format'], 'npy')
            self.assertTrue(item['path'].endswith('.npy'))
            with self.assertRaises(FileExistsError):
                _save_array(root, 'a.npy', torch.zeros(3))
            with self.assertRaisesRegex(ValueError, 'Unsupported'):
                _save_array(root, 'bad.npy', torch.ones(3), 'float16')
            self.assertFalse((root / 'bad.npy').exists())

    def test_complete_loss_and_render_gradients_match_both_readers(self):
        generator = torch.Generator().manual_seed(40)
        arrays = dict(internal_rgb=torch.rand(3, 16, 20, generator=generator),
                      internal_depth=torch.rand(1, 16, 20, generator=generator) + 1,
                      external_rgb=torch.rand(3, 16, 20, generator=generator),
                      external_inverse_depth=torch.rand(1, 16, 20, generator=generator))
        rendered = torch.rand(3, 16, 20, generator=generator)
        depth = torch.rand(1, 16, 20, generator=generator) + 1
        results = []
        with tempfile.TemporaryDirectory() as directory:
            for fmt in CACHE_FORMATS:
                entry = {key: _save_array(Path(directory) / fmt, key + '.npy', value, fmt)
                         for key, value in arrays.items()}
                for loader in (lambda e: load_priors(e, 'cpu'),
                               lambda e: IESRGSPriors(**{key: load_tensor(value, 'cpu') for key, value in e.items()})):
                    rgb, median = rendered.clone().requires_grad_(), depth.clone().requires_grad_()
                    priors = loader(entry)
                    terms = fusion_loss(rgb, median, priors)
                    terms['total'].backward()
                    results.append(({key: value.detach().numpy().tobytes() for key, value in terms.items()},
                                    rgb.grad.numpy().tobytes(), median.grad.numpy().tobytes()))
                    self.assertGreater(rgb.grad.abs().sum().item(), 0)
                    self.assertGreater(median.grad.abs().sum().item(), 0)
                    self.assertTrue(all(not x.requires_grad for x in vars(priors).values()))
        self.assertTrue(all(result == results[0] for result in results[1:]))
        self.assertFalse(torch.cuda.is_initialized())

    def test_shuffle_rejects_unknown_malformed_and_oversized_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            item = _save_array(root, 'original.npy', torch.arange(24, dtype=torch.float32).reshape(3, 2, 4), 'npz_byte_shuffle')
            with np.load(item['path'], allow_pickle=False) as archive:
                payload, encoded = archive['array'], archive['metadata']
            metadata = json.loads(encoded.tobytes())
            changes = [dict(codec='unknown'), dict(codec_version=2), dict(codec_version=True),
                dict(dtype='float16'), dict(dtype_str='>f4'), dict(shape=[3, -2, 4]),
                dict(shape=[True, 2, 4]), dict(shape=[65536, 65536]), dict(shape=[25]),
                dict(array_order='F'), dict(array_sha256='0' * 64), dict(extra='unrecognized')]
            for i, change in enumerate(changes):
                path = root / f'bad_{i}.npz'
                modified = dict(metadata, **change)
                np.savez_compressed(path, array=payload, metadata=np.frombuffer(json.dumps(modified).encode(), np.uint8))
                with self.assertRaises(ValueError):
                    load_prior_array(identity(path))
            for name, altered in [('wrong_payload_dtype', payload.astype(np.uint16)),
                                   ('wrong_payload_shape', payload.T.copy())]:
                path = root / (name + '.npz')
                np.savez_compressed(path, array=altered, metadata=encoded)
                with self.assertRaises(ValueError):
                    load_prior_array(identity(path))
            duplicate = encoded.tobytes().decode().replace('{', '{"codec_version":1,', 1)
            path = root / 'duplicate_json_key.npz'
            np.savez_compressed(path, array=payload, metadata=np.frombuffer(duplicate.encode(), np.uint8))
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                load_prior_array(identity(path))
            for dtype in (torch.float16, torch.float64):
                with self.assertRaisesRegex(ValueError, 'without numeric conversion'):
                    _save_array(root, 'must_not_exist.npy', torch.ones(3, dtype=dtype), 'npz_byte_shuffle')
            self.assertFalse((root / 'must_not_exist.npz').exists())

    def test_shuffle_rejects_object_headers_trailing_payload_and_zip_layouts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            item = _save_array(root, 'source.npy', torch.ones(3, 2, 4), 'npz_byte_shuffle')
            with zipfile.ZipFile(item['path']) as archive:
                blobs = {name: archive.read(name) for name in archive.namelist()}
            object_bytes = io.BytesIO()
            np.save(object_bytes, np.array([{'never_unpickle': True}], dtype=object))
            huge_header = io.BytesIO()
            np.lib.format.write_array_header_1_0(huge_header, {'descr': '|u1', 'fortran_order': False, 'shape': (4, 2**60)})
            variants = [(dict(blobs, **{'metadata.npy': object_bytes.getvalue()}), zipfile.ZIP_DEFLATED),
                (dict(blobs, **{'array.npy': huge_header.getvalue()}), zipfile.ZIP_DEFLATED),
                (dict(blobs, **{'array.npy': blobs['array.npy'] + b'x'}), zipfile.ZIP_DEFLATED),
                (dict(blobs, **{'foreign.npy': blobs['array.npy']}), zipfile.ZIP_DEFLATED),
                (blobs, zipfile.ZIP_STORED)]
            for i, (members, compression) in enumerate(variants):
                path = root / f'hostile_{i}.npz'
                with zipfile.ZipFile(path, 'w', compression=compression) as archive:
                    for name, blob in members.items():
                        archive.writestr(name, blob)
                with self.assertRaises(ValueError):
                    load_prior_array(identity(path))

    def test_both_producer_entrypoints_serialize_shuffle_with_mocked_cpu_teachers(self):
        # Temporary fixtures exercise the production storage calls only. They
        # are not actual trained checkpoints, teacher evidence or formal priors.
        import methods.iesrgs_priors as producers
        from PIL import Image
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = root / 'unit_manifest.json'; manifest_path.write_text('{}')
            checkpoint = root / 'unit_not_a_real_checkpoint'; checkpoint.write_text('mock only')
            for relative in ('resolution.py', 'submodules/Depth-Anything-V2/run.py', 'timm.py'):
                p = root / relative; p.parent.mkdir(parents=True, exist_ok=True); p.write_text('# mock dependency\n')
            Image.fromarray(np.zeros((4, 5, 3), dtype=np.uint8)).save(root / 'lr.png')
            cfg = {'manifest': identity(manifest_path)}
            manifest = {'_root': str(root), 'scene': 'UNIT_TEST_ONLY'}
            rows = [dict(camera_id='cam01', frame_index=149, time=149/300., lr_path='lr.png', lr_sha256='mock')]
            rgb = torch.arange(3 * 16 * 20, dtype=torch.float32).reshape(3, 16, 20) / 1000
            depth = torch.ones(1, 16, 20)
            with patch.object(producers, 'training_manifest', return_value=(manifest, rows)), patch.object(producers, '_upstream', return_value=root), patch.object(producers, 'load_external_models', return_value=(None, None)), patch.object(producers, 'external_tensors', return_value=(rgb, depth)), patch.dict(sys.modules, {'timm': SimpleNamespace(__file__=str(root / 'timm.py'))}):
                external = read(producers.prepare_external(cfg, root, identity(checkpoint), identity(checkpoint), root / 'external', 'cpu', cache_format='npz_byte_shuffle'))
            backend = SimpleNamespace(config=cfg, loaded_checkpoint_identity=identity(checkpoint),
                effective_state=lambda t: {'xyz': torch.zeros(1, 3)}, camera=lambda row, res: (row, res),
                background=torch.zeros(3), g=SimpleNamespace(active_sh_degree=0))
            payload = {'metadata': {'stage': 'iesrgs_internal', 'iterations': 30000, 'optimizer_updates': 29999}, 'adaptation_config': cfg}
            with patch.object(producers, 'training_manifest', return_value=(manifest, rows)), patch.object(producers.torch, 'load', return_value=payload), patch.object(producers, 'compute_3d_filter', return_value=None), patch.object(producers, 'render_state', return_value={'render': rgb, 'median_depth': depth}):
                internal = read(producers.prepare_internal(backend, identity(checkpoint), root / 'internal', cache_format='npz_byte_shuffle'))
            for bundle in (external, internal):
                self.assertEqual(bundle['cache_format'], 'npz_byte_shuffle')
                for name, item in bundle['entries'][0].items():
                    if isinstance(item, dict):
                        self.assertEqual(item['storage']['codec_version'], 1)
                        expected = rgb if name.endswith('rgb') else depth
                        self.assertEqual(load_tensor(item, 'cpu').numpy().tobytes(), expected.numpy().tobytes())

    def test_merge_preserves_mixed_containers_and_legacy_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            common = dict(manifest={'path': 'fixture', 'sha256': 'fixture'}, scene='fixture',
                          training_only=True, upstream_commit='fixture', producer=[identity(__file__)],
                          model_identities=[identity(__file__)])
            base_row = dict(camera='cam01', frame=0, time=0., lr_sha256='fixture')
            ext = dict(common, kind='iesrgs_external', cache_format='npz_deflate', entries=[dict(base_row)])
            internal = dict(common, kind='iesrgs_internal', internal_teacher={'fixture': True}, entries=[dict(base_row)])
            for key in ('external_rgb', 'external_inverse_depth'):
                ext['entries'][0][key] = _save_array(root, key + '.npy', torch.ones(3), 'npz_deflate')
            for key in ('internal_rgb', 'internal_depth'):
                item = _save_array(root, key + '.npy', torch.ones(3))
                item.pop('storage')  # Simulate a real legacy index.
                internal['entries'][0][key] = item
            write_new(root / 'ext.json', ext)
            write_new(root / 'int.json', internal)
            merged = read(merge_priors(root / 'ext.json', root / 'int.json', root / 'merged.json'))
            self.assertEqual(merged['cache_formats'], {'external': 'npz_deflate', 'internal': 'npy'})
            self.assertNotIn('cache_format', merged)
            self.assertEqual(merged['entries'][0]['internal_rgb'], internal['entries'][0]['internal_rgb'])


if __name__ == '__main__':
    unittest.main()
