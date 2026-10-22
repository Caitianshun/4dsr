"""CPU contracts at the SplatSuRe/public-runner boundary.

Real producer receipt writers and the real bundle/loss entry points are used.
All images, weights and successful inference execution are synthetic fixtures;
no model inference, CUDA kernel, training or scientific acceptance occurs.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

BASE = Path(__file__).resolve().parents[1]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

import bundle_priors
import common
import train_stage
from methods import splatsure_stablesr, splatsure_teacher


class SplatSuRePublicLossTests(unittest.TestCase):
    def check_loss_and_gradient(self, stage):
        import torch
        from methods.iesrgs import ssim
        from methods.splatsure import SplatSuReConfig, splatsure_loss

        # There are pixels below zero, inside [0,1], and above one in every
        # channel. Both saturated regions must have zero derivative to raw RGB.
        raw = torch.linspace(-.4, 1.4, 3 * 16 * 16).reshape(3, 16, 16).requires_grad_()
        lr = torch.full((3, 16, 16) if stage == 'lr' else (3, 4, 4), .23)
        teacher = torch.full((3, 16, 16), .71)
        weight = torch.linspace(.1, .9, 16 * 16).reshape(1, 16, 16)
        obs = dict(camera_id='cam01', frame_index=0, time=0.)
        regularizer = torch.tensor(.031)
        backend = SimpleNamespace(torch=torch, effective_state=lambda t: {'time': t},
            render=lambda *a, **k: {'render': raw}, lr=lambda o: lr,
            regularizer=lambda: regularizer)
        priors = {('cam01', 0): {'sr': 'teacher', 'weight': 'weight'}}
        with mock.patch.object(train_stage, 'load_tensor', side_effect=lambda x: {
                'teacher': teacher, 'weight': weight}[x]):
            actual, packets = train_stage.batch_loss(backend, [obs], stage, priors,
                {'method': 'splatsure', 'degradation': 'area4'})

        reference = raw.detach().clone().requires_grad_()
        rgb = reference.clamp(0, 1)
        if stage == 'lr':
            expected = .8 * (rgb - lr).abs().mean() + .2 * (1 - ssim(rgb, lr))
        else:
            expected = splatsure_loss(rgb, lr, teacher, weight,
                SplatSuReConfig(lr_operator='area'))['loss']
        expected = expected + regularizer
        actual_gradient, = torch.autograd.grad(actual, raw)
        expected_gradient, = torch.autograd.grad(expected, reference)
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=1e-5, atol=1e-7)
        saturated = (raw.detach() < 0) | (raw.detach() > 1)
        self.assertEqual(int(torch.count_nonzero(actual_gradient[saturated])), 0)
        self.assertGreater(int(torch.count_nonzero(actual_gradient[~saturated])), 0)
        self.assertEqual(len(packets), 1)

    def test_lr_loss_clamps_rgb_and_blocks_saturated_gradient(self):
        self.check_loss_and_gradient('lr')

    def test_sr_loss_clamps_rgb_and_blocks_saturated_gradient(self):
        self.check_loss_and_gradient('sr')


class SplatSuReProducerBundleTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        from PIL import Image

        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        # Public identity() intentionally uses project-relative paths, while
        # both actual Splat producers intentionally emit absolute paths.
        patcher = mock.patch.object(common, 'ROOT', self.root)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.lr_array = np.full((8, 8, 3), 70, dtype=np.uint8)
        self.sr_array = np.full((32, 32, 3), 90, dtype=np.uint8)
        observations = []
        for camera in ('cam01', 'cam02'):
            path = self.root / 'lr' / camera / '0000.png'
            path.parent.mkdir(parents=True)
            Image.fromarray(self.lr_array).save(path)
            observations.append(dict(camera_id=camera, frame_index=0, time=0., split='train',
                lr_path=str(path.relative_to(self.root)), lr_sha256=common.sha(path)))
        init = self.file('initial.npz', b'contract-only initializer; not a loadable model')
        self.manifest = dict(schema='n3dv_dynamic_sr_pilot_v1', scene='cpu_contract', scale=4,
            physical_lr_derivation={'formation': 'area4'}, resolutions={'lr': [8, 8], 'hr': [32, 32]},
            splits={'train': ['cam01', 'cam02'], 'test': ['cam00']}, frame_indices=[0],
            observations=observations, initialization={'train_only_lr': True,
                'npz_path': init.name, 'npz_sha256': common.sha(init)})
        self.manifest_file = self.write('manifest.json', self.manifest)
        self.config = dict(method='splatsure', scene='cpu_contract', teacher_variant='stablesr',
                           manifest=common.identity(self.manifest_file))
        self.config_file = self.write('config.json', self.config)
        self.assertFalse(Path(self.config['manifest']['path']).is_absolute())

        parent = self.file('lr_parent.pt', b'CPU contract LR checkpoint identity')
        parent_id = splatsure_stablesr.identity(parent)
        map_records = []
        for row in observations:
            path = self.file(f"maps/{row['camera_id']}.pt", b'CPU contract map identity')
            map_records.append(dict(camera_id=row['camera_id'], frame_index=0,
                path=path.name, sha256=common.sha(path)))
        self.maps_receipt = self.write('maps/receipt.json', dict(status='prepared',
            manifest_sha256=common.sha(self.manifest_file), records=map_records,
            provenance={'parent_checkpoint': parent_id, 'parent_checkpoint_sha256': parent_id['sha256']}))

        self.source = self.root / 'upstream'
        for path in (splatsure_stablesr.SCRIPT, splatsure_stablesr.CONFIG,
                     splatsure_stablesr.CFW_CONFIG, 'environment.yaml', 'README.md'):
            self.file('upstream/' + path, b'CPU contract source identity')
        self.stable_models = [self.file('weights/stablesr_000117.ckpt', b'diffusion fixture'),
                              self.file('weights/vqgan_cfw_00011.ckpt', b'cfw fixture')]
        assets = []
        for role in ('openclip', 'lpips', 'vgg16'):
            path = self.file('runtime/' + role + '.pt', role.encode())
            assets.append(dict(splatsure_stablesr.identity(path), role=role))
        hashes={item['role']:item['sha256'] for item in assets}
        patcher=mock.patch.object(splatsure_stablesr, 'RUNTIME_SHA256', hashes)
        patcher.start()
        self.addCleanup(patcher.stop)
        hf=self.root/'runtime/hf/hub'
        torch_home=self.root/'runtime/torch'
        hf_repo=hf/splatsure_stablesr.OPENCLIP_CACHE_REPO
        mappings=[]
        targets={'openclip':hf_repo/'snapshots'/splatsure_stablesr.OPENCLIP_REVISION/'open_clip_pytorch_model.bin',
                 'vgg16':torch_home/'hub/checkpoints/vgg16-397923af.pth',
                 'lpips':self.root/'runtime/lpips/vgg.pth'}
        for role,target in targets.items():
            self.file(str(target.relative_to(self.root)),role.encode())
            mappings.append(dict(splatsure_stablesr.identity(target),role=role))
        self.file(str((hf_repo/'refs/main').relative_to(self.root)),splatsure_stablesr.OPENCLIP_REVISION.encode())
        self.assets_file = self.write('runtime/assets.json', {'files': assets,'cache_mappings':mappings,
            'cache_environment':{'TORCH_HOME':str(torch_home),'HUGGINGFACE_HUB_CACHE':str(hf)}})

    def file(self, name, payload):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        return path

    def write(self, name, payload):
        return self.file(name, (json.dumps(payload) + '\n').encode())

    def stable_receipt(self, *, run_bridge=False):
        """Run the real plan/accept writers around synthetic inference output."""
        from PIL import Image
        weights = {role: (path.name, common.sha(path)) for role, path in
                   zip(('diffusion', 'cfw'), self.stable_models)}
        out = self.root / 'stablesr_teacher'
        with mock.patch.object(splatsure_stablesr, 'source_check', return_value=self.source), \
             mock.patch.object(splatsure_stablesr, 'WEIGHTS', weights):
            plan = splatsure_stablesr.plan(self.manifest_file, out, source=self.source,
                weight_root=self.stable_models[0].parent, python=sys.executable, runtime_assets=self.assets_file)
            self.assertEqual(plan['missing_dependencies'], [])
            def fake_inference(command=None, **kwargs):
                if command is not None:
                    self.assertEqual(kwargs['cwd'], str(out/'execution_workdir'))
                    self.assertEqual(kwargs['env']['HF_HUB_OFFLINE'], '1')
                    self.assertEqual(kwargs['env']['TORCH_HOME'], str(self.root/'runtime/torch'))
                    splatsure_stablesr.verify_workdir(plan)
                generated = out / 'generated'
                generated.mkdir()
                for row in plan['targets']:
                    Image.fromarray(self.sr_array).save(generated / row['staged_name'])
                return SimpleNamespace(returncode=0)
            if run_bridge:
                with mock.patch.object(splatsure_stablesr.subprocess, 'run', side_effect=fake_inference), \
                     mock.patch.dict('os.environ', {'CUDA_VISIBLE_DEVICES': 'contract-no-CUDA-execution'}):
                    receipt=splatsure_stablesr.run(out/'plan.json')
            else:
                splatsure_stablesr.prepare_workdir(plan)
                fake_inference()
                self.write('stablesr_teacher/execution.json', dict(status='completed', returncode=0,
                    plan=splatsure_stablesr.identity(out / 'plan.json'),
                    execution_workdir=plan['execution_workdir'], runtime_assets_identity=plan['runtime_assets_identity'],
                    fixture='No actual inference execution'))
                receipt = splatsure_stablesr.accept(out / 'plan.json')
        return out / 'receipt.json', receipt, self.stable_models

    def swinir_receipt(self):
        """Run the real completion writer, replacing only learned inference."""
        network = self.file('weights/network_swinir.py', b'CPU contract network identity')
        checkpoint = self.file('weights/swinir.pth', b'CPU contract generic weight identity')
        out = self.root / 'swinir_teacher'
        with mock.patch.object(splatsure_teacher, 'NETWORK_SHA256', common.sha(network)), \
             mock.patch.object(splatsure_teacher, 'CHECKPOINT_SHA256', common.sha(checkpoint)), \
             mock.patch.object(splatsure_teacher, 'build_model', return_value=object()), \
             mock.patch.object(splatsure_teacher, 'infer', return_value=self.sr_array), \
             mock.patch.dict('os.environ', {'CUDA_VISIBLE_DEVICES': 'contract-no-CUDA-execution'}):
            receipt = splatsure_teacher.produce(self.manifest_file, out, network=network,
                checkpoint=checkpoint, device='cuda:0')
        return out / 'receipt.json', receipt, [network, checkpoint]

    def bundle(self, receipt_path, models, *, variant='stablesr', name='bundle.json'):
        self.config['teacher_variant'] = variant
        self.write('config.json', self.config)
        args = ['bundle_priors.py', '--config', str(self.config_file), '--sr-receipt', str(receipt_path),
                '--weights-receipt', str(self.maps_receipt), '--output', str(self.root / name)]
        for model in models:
            args += ['--model', str(model)]
        with mock.patch.object(sys, 'argv', args):
            bundle_priors.main()
        return common.read(self.root / name)

    def test_stablesr_real_receipt_absolute_paths_and_role_metadata_bundle(self):
        path, receipt, models = self.stable_receipt()
        self.assertTrue(Path(receipt['manifest']['path']).is_absolute())
        self.assertNotEqual(receipt['manifest'], self.config['manifest'])
        self.assertEqual({m['role'] for m in receipt['model_identities']}, {'diffusion', 'cfw'})
        self.assertTrue(all('official_url' in m for m in receipt['model_identities']))
        result = self.bundle(path, models)
        self.assertEqual(result['manifest'], self.config['manifest'])
        self.assertEqual(result['teacher_recipe'], splatsure_stablesr.RECIPE)
        self.assertEqual(len(result['entries']), 2)
        self.assertEqual({m['role'] for m in result['model_identities'] if 'role' in m}, {'diffusion', 'cfw'})

    def test_swinir_real_receipt_absolute_paths_bundle(self):
        path, receipt, models = self.swinir_receipt()
        self.assertTrue(Path(receipt['manifest']['path']).is_absolute())
        result = self.bundle(path, models, variant='swinir')
        self.assertEqual(result['teacher_variant'], 'swinir')
        self.assertEqual(result['teacher_recipe'], 'SplatSuRe+SwinIR-M_DF2K_x4_generic')
        self.assertEqual(len(result['entries']), 2)

    def test_rejects_stablesr_wrong_recipe_even_with_correct_schema_and_models(self):
        path, receipt, models = self.stable_receipt()
        receipt['recipe'] = 'StableSR-Turbo-or-undocumented-substitution'
        path.write_text(json.dumps(receipt))
        with self.assertRaisesRegex(ValueError, 'recipe'):
            self.bundle(path, models)

    def test_rejects_swinir_scene_teacher_recipe(self):
        path, receipt, models = self.swinir_receipt()
        receipt['recipe'] = 'SwinIR-scene-finetuned-area4'
        path.write_text(json.dumps(receipt))
        with self.assertRaisesRegex(ValueError, 'generic SwinIR'):
            self.bundle(path, models, variant='swinir')

    def test_rejects_missing_extra_and_different_path_models(self):
        path, receipt, models = self.stable_receipt()
        extra = self.file('weights/extra.pt', b'foreign model')
        alias = self.file('weights/copied_diffusion.ckpt', models[0].read_bytes())
        cases = [models[:1], models + [extra], [alias, models[1]]]
        for index, supplied in enumerate(cases):
            with self.subTest(case=index), self.assertRaisesRegex(ValueError, 'actual SplatSuRe teacher'):
                self.bundle(path, supplied, name=f'wrong_models_{index}.json')

    def test_rejects_same_hash_manifest_at_a_different_file(self):
        path, receipt, models = self.stable_receipt()
        foreign = self.file('different_manifest.json', self.manifest_file.read_bytes())
        receipt['manifest'] = splatsure_stablesr.identity(foreign)
        self.assertEqual(receipt['manifest']['sha256'], self.config['manifest']['sha256'])
        path.write_text(json.dumps(receipt))
        with self.assertRaisesRegex(ValueError, 'manifest'):
            self.bundle(path, models)

    def test_rejects_stale_manifest_or_model_hash(self):
        path, receipt, models = self.stable_receipt()
        for index, field in enumerate(('manifest', 'model_identities')):
            changed = copy.deepcopy(receipt)
            identity = changed['manifest'] if field == 'manifest' else changed['model_identities'][0]
            identity['sha256'] = '0' * 64
            changed_path = self.write(f'stale_{index}.json', changed)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'Identity changed'):
                self.bundle(changed_path, models, name=f'stale_bundle_{index}.json')

    def test_rejects_unfinished_producer_receipt(self):
        path, receipt, models = self.stable_receipt()
        receipt['status'] = 'planned_runtime_unverified'
        path.write_text(json.dumps(receipt))
        with self.assertRaisesRegex(ValueError, 'successful actual teacher'):
            self.bundle(path, models)

    def test_real_bridge_uses_isolated_workdir_and_bound_cache_environment(self):
        path,receipt,models=self.stable_receipt(run_bridge=True)
        work=Path(receipt['execution_workdir'])
        self.assertEqual((work/'configs').resolve(), (self.source/'configs').resolve())
        self.assertEqual(common.sha(work/splatsure_stablesr.LPIPS_WORKDIR_PATH),
                         splatsure_stablesr.RUNTIME_SHA256['lpips'])
        self.assertFalse((self.source/'taming').exists(), 'No cache should be created inside upstream source')
        execution=common.read(path.parent/'execution.json')
        self.assertEqual(execution['execution_workdir'], str(work))
        self.assertEqual(execution['runtime_assets_identity'], splatsure_stablesr.identity(self.assets_file))

    def test_runtime_cache_rejects_same_bytes_at_an_unused_loader_path(self):
        assets=common.read(self.assets_file)
        openclip=next(x for x in assets['cache_mappings'] if x['role']=='openclip')
        openclip['path']=next(x['path'] for x in assets['files'] if x['role']=='openclip')
        with self.assertRaisesRegex(ValueError, 'upstream loader path'):
            splatsure_stablesr.validate_runtime_assets(assets)

    def test_plan_preserves_virtualenv_interpreter_symlink(self):
        launcher=self.root/'teacher_env/bin/python'
        launcher.parent.mkdir(parents=True)
        launcher.symlink_to(sys.executable)
        weights={role:(path.name,common.sha(path)) for role,path in
                 zip(('diffusion','cfw'),self.stable_models)}
        with mock.patch.object(splatsure_stablesr,'source_check',return_value=self.source), \
             mock.patch.object(splatsure_stablesr,'WEIGHTS',weights):
            plan=splatsure_stablesr.plan(self.manifest_file,self.root/'venv_plan',source=self.source,
                weight_root=self.stable_models[0].parent,python=launcher,runtime_assets=self.assets_file)
        self.assertEqual(plan['python'],str(launcher.absolute()))
        self.assertNotEqual(plan['python'],str(launcher.resolve()))
        self.assertEqual(splatsure_stablesr.command(plan)[0],str(launcher.absolute()))

    def test_runtime_cache_rejects_newline_in_offline_hf_revision(self):
        ref=self.root/'runtime/hf/hub'/splatsure_stablesr.OPENCLIP_CACHE_REPO/'refs/main'
        ref.write_text(splatsure_stablesr.OPENCLIP_REVISION+'\n')
        with self.assertRaisesRegex(ValueError, 'revision ref'):
            splatsure_stablesr.validate_runtime_assets(common.read(self.assets_file))

    def test_runtime_cache_rejects_wrong_offline_hf_revision(self):
        ref=self.root/'runtime/hf/hub'/splatsure_stablesr.OPENCLIP_CACHE_REPO/'refs/main'
        ref.write_text('a-different-model-revision')
        with self.assertRaisesRegex(ValueError, 'revision ref'):
            splatsure_stablesr.validate_runtime_assets(common.read(self.assets_file))

    def test_run_checks_cache_bytes_before_launching_external_program(self):
        path,receipt,models=self.stable_receipt()
        mapping=next(x for x in receipt['runtime_assets']['cache_mappings'] if x['role']=='vgg16')
        Path(mapping['path']).write_bytes(b'changed after planning')
        with mock.patch.object(splatsure_stablesr, 'source_check', return_value=self.source), \
             mock.patch.object(splatsure_stablesr.subprocess, 'run') as child:
            with self.assertRaisesRegex(ValueError, 'cache mapping hash'):
                splatsure_stablesr.run(path.parent/'plan.json')
            child.assert_not_called()
        self.assertFalse((path.parent/'inputs').exists())


if __name__ == '__main__':
    unittest.main()
