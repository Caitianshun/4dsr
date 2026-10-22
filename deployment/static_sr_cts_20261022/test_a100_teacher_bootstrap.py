"""CPU contract tests; no SSH, GPU imports, downloads, or environment deployment."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('a100_bootstrap_tested', Path(__file__).with_name('a100_teacher_bootstrap.py'))
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)


class BootstrapContracts(unittest.TestCase):
    def synthetic_manifest(self, root, rows, *, kind='overlays', **changes):
        source = root/'inventory.json'
        value = dict(schema='static_sr_cts_transport_inventory_v1', stage=kind,
                     files=rows, symlinks=[], source_root='/original/CTS', destination_root='/old/CTS')
        value.update(changes)
        source.write_text(json.dumps(value))
        return source

    def inventory_call(self, source, count=1, size=3):
        with patch.dict(b.MANIFESTS, overlays=(Path('unused'), b.sha(source))), \
             patch.dict(b.EXPECTED_COUNTS, overlays=(count, size)):
            return b.selected_inventory('overlays', source)

    def entry(self, rel, data=b'abc'):
        import hashlib
        return dict(path=str(rel), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())

    def closure(self, root):
        rows = []
        for rel in (b.SITE_REL/'sitecustomize.py', b.UPSTREAM/'stablesr/.git/HEAD',
                    b.UPSTREAM/'taming-transformers/taming/__init__.py'):
            path = root/rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'abc')
            rows.append(self.entry(rel))
        return dict(files=rows)

    def models(self, root):
        shared = root/'shared-cache'
        definitions, models = {}, {}
        for role, (name, _, _) in b.MODELS.items():
            path = (shared if role in ('vgg16', 'lpips') else root/b.WEIGHT_REL)/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(('synthetic-'+role).encode())
            definitions[role] = (name, path.stat().st_size, b.sha(path))
            models[role] = dict(b.identity(path), role=role)
        return shared, definitions, models

    def test_real_pinned_plan_is_closed_teacher_subset(self):
        value = b.plan()
        self.assertEqual(len(value['files']), 6022)
        self.assertEqual(value['total_bytes'], 471476754)
        self.assertEqual(value['source_commits'], b.COMMITS)
        allowed = sum(b.SELECTED_PREFIXES.values(), ())
        self.assertTrue(all(any(Path(x['path']).is_relative_to(p) for p in allowed) for x in value['files']))
        self.assertEqual([x['path'] for x in value['files'] if x['path'].endswith('.pth')],
                         [str(b.SITE_REL/'distutils-precedence.pth')])
        self.assertEqual(value['GPU_calls'], 0)
        self.assertEqual(len(value['required_task_weights']), 3)
        self.assertEqual(len(value['reuse_shared_models']), 2)

    def test_bridge_model_and_source_constants_match(self):
        path = b.ROOT/'experiments/static_sr_wu_20261022/methods/splatsure_stablesr.py'
        spec = importlib.util.spec_from_file_location('stablesr_bridge_constants_test', path)
        bridge = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bridge)
        self.assertEqual(bridge.STABLESR_COMMIT, b.COMMITS['stablesr'])
        self.assertEqual(bridge.OPENCLIP_REVISION, b.OPENCLIP_REVISION)
        self.assertEqual(bridge.OPENCLIP_CACHE_REPO, b.OPENCLIP_REPO)
        for role, expected in bridge.RUNTIME_SHA256.items():
            self.assertEqual(b.MODELS[role][2], expected)
        for role, expected in bridge.WEIGHTS.items():
            self.assertEqual((b.MODELS[role][0], b.MODELS[role][2]), expected)

    def test_pinned_manifest_sha_cannot_be_overridden_by_content(self):
        with tempfile.TemporaryDirectory() as temp:
            path = self.synthetic_manifest(Path(temp), [])
            with self.assertRaisesRegex(ValueError, 'inventory SHA differs'):
                b.selected_inventory('overlays', path)

    def test_inventory_filters_other_methods_and_preserves_original(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            expected = self.entry(b.SITE_REL/'sitecustomize.py')
            source = self.synthetic_manifest(root, [expected, self.entry('unselected/other.py')])
            before = source.read_bytes()
            rows, origin = self.inventory_call(source)
            self.assertEqual(rows, [expected])
            self.assertEqual(source.read_bytes(), before)
            self.assertEqual(origin['sha256'], b.sha(source))

    def test_invalid_inventory_member_duplicates_paths_sizes_and_symlinks_fail(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            row = self.entry(b.SITE_REL/'sitecustomize.py')
            cases = [([row, row], {}), ([dict(row, path='../escape')], {}),
                     ([dict(row, path='/absolute')], {}), ([dict(row, path='a//b')], {}),
                     ([dict(row, bytes=True)], {}), ([dict(row, sha256='x'*64)], {}),
                     ([row], {'symlinks': [{'path': 'link'}]}), ([row], {'stage': 'other'})]
            for rows, changes in cases:
                with self.subTest(rows=rows, changes=changes):
                    source = self.synthetic_manifest(root, rows, **changes)
                    with self.assertRaises(ValueError):
                        self.inventory_call(source)

    def test_duplicate_json_keys_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'bad.json'
            path.write_text('{"same":1,"same":2}')
            with self.assertRaisesRegex(ValueError, 'Duplicate JSON key'):
                b.strict_json(path)

    def test_actual_tree_must_equal_full_closed_subset(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plan = self.closure(root)
            with patch.object(b, 'ROOT', root):
                self.assertEqual(b.verify_files(plan)['files'], 3)
                extra = root/b.SITE_REL/'injected.pth'
                extra.write_text('import unintended\n')
                with self.assertRaisesRegex(ValueError, 'closure differs'):
                    b.verify_files(plan)
                extra.unlink()
                victim = root/plan['files'][0]['path']
                victim.write_bytes(b'xyz')
                with self.assertRaisesRegex(ValueError, 'identity differs'):
                    b.verify_files(plan)

    def test_foreign_launcher_and_symlink_directory_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plan = self.closure(root)
            with patch.object(b, 'ROOT', root):
                foreign = root/b.ENV_REL/'pyvenv.cfg'
                foreign.write_text('/local/environment')
                with self.assertRaisesRegex(ValueError, 'Unexpected preexisting venv'):
                    b.verify_files(plan)
                foreign.unlink()
                (root/b.SITE_REL/'link').symlink_to(root, target_is_directory=True)
                with self.assertRaisesRegex(ValueError, 'Symlink'):
                    b.verify_files(plan)

    def test_cpu_environment_removes_import_injection_and_preserves_parent(self):
        inherited = dict(PYTHONPATH='/foreign', PYTHONHOME='/foreign', VIRTUAL_ENV='/foreign',
                         CUDA_VISIBLE_DEVICES='GPU-never-use', LD_LIBRARY_PATH='/base/private/lib')
        with patch.dict(os.environ, inherited, clear=True):
            before = dict(os.environ)
            env = b.cpu_environment()
            self.assertEqual(dict(os.environ), before)
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '')
        self.assertEqual(env['HF_HUB_OFFLINE'], '1')
        self.assertEqual(env['TRANSFORMERS_OFFLINE'], '1')
        self.assertEqual(env['LD_LIBRARY_PATH'], inherited['LD_LIBRARY_PATH'])
        self.assertFalse(any(x in env for x in ('PYTHONPATH', 'PYTHONHOME', 'VIRTUAL_ENV')))

    def test_base_identity_requires_exact_versions_and_venv_launcher(self):
        value = dict(python=str(b.BASE_PYTHON), prefix=str(b.BASE_PYTHON.parent.parent),
            purelib=str(b.BASE_PYTHON.parent.parent/'lib/python3.10/site-packages'),
            python_version=[3, 10, 12], versions=dict(b.BASE_VERSIONS), CUDA_initialized=False)
        self.assertEqual(b.validate_base(value), value)
        for changed in (dict(python='/usr/bin/python3.10'), dict(CUDA_initialized=True),
                        dict(versions=dict(b.BASE_VERSIONS, torch='2.8.0')),
                        dict(purelib='/foreign/site-packages')):
            with self.subTest(changed=changed), self.assertRaises(RuntimeError):
                b.validate_base(dict(value, **changed))

    def test_overlay_makes_destination_pth_and_never_pip(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            site = root/b.SITE_REL
            site.mkdir(parents=True)
            (site/'sitecustomize.py').write_text('# verified compatibility file')
            base = dict(purelib='/a100/base/lib/python3.10/site-packages')
            commands = []
            def run(argv, **kwargs):
                commands.append((argv, kwargs))
                (root/b.ENV_REL/'pyvenv.cfg').write_text('include-system-site-packages = false\n')
            with patch.object(b, 'ROOT', root), patch.object(b.subprocess, 'run', side_effect=run):
                value = b.overlay(base, b.cpu_environment())
                with self.assertRaises(FileExistsError):
                    b.overlay(base, b.cpu_environment())
            self.assertEqual(len(commands), 1)
            self.assertEqual(commands[0][0][1:5], ['-m', 'venv', '--copies', '--without-pip'])
            self.assertEqual(commands[0][1]['env']['CUDA_VISIBLE_DEVICES'], '')
            self.assertEqual((site/'reuse_wu_torch.pth').read_text(), base['purelib']+'\n')
            self.assertEqual((site/'taming_pinned_namespace.pth').read_text(), str(root/b.UPSTREAM/'taming-transformers')+'\n')
            self.assertIn(str(site/'sitecustomize.py'), (site/'static_sr_torchvision_compat.pth').read_text())
            self.assertEqual(value['python'], str(root/b.ENV_REL/'bin/python'))

    def test_fixed_cache_aliases_pass_existing_bridge_and_preserve_shared_bytes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            shared, definitions, models = self.models(root)
            before = {p.name: p.read_bytes() for p in shared.iterdir()}
            out = root/'receipt'
            out.mkdir()
            with patch.object(b, 'ROOT', root), patch.object(b, 'SHARED_CACHE', shared), patch.object(b, 'MODELS', definitions):
                path = b.runtime_assets(models, out)
                value = json.loads(path.read_text())
                mappings = b.validate_assets(value)
                self.assertEqual(set(mappings), {'openclip', 'vgg16', 'lpips'})
                self.assertTrue(all(Path(item['path']).is_relative_to(root/b.CACHE_REL)
                                    for item in value['files']+value['cache_mappings']))
                self.assertEqual({x['role']:x['original_source'] for x in value['cache_mappings']},
                                 {x:models[x] for x in mappings})
                bridge_path = Path(__file__).resolve().parents[2]/'experiments/static_sr_wu_20261022/methods/splatsure_stablesr.py'
                spec = importlib.util.spec_from_file_location('stablesr_cache_contract_test', bridge_path)
                bridge = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(bridge)
                with patch.object(bridge, 'RUNTIME_SHA256', {r: definitions[r][2] for r in mappings}):
                    self.assertEqual(bridge.validate_runtime_assets(value), mappings)
                ref = root/b.CACHE_REL/'hf/hub'/b.OPENCLIP_REPO/'refs/main'
                self.assertEqual(ref.read_bytes(), b.OPENCLIP_REVISION.encode())
                self.assertEqual({p.name: p.read_bytes() for p in shared.iterdir()}, before)
                original = list(value['files'])
                value['files'] = [models[x] for x in ('openclip', 'vgg16', 'lpips')]
                with self.assertRaisesRegex(ValueError, 'task-local alias'):
                    b.validate_assets(value)
                value['files'] = original
                value['cache_environment']['TORCH_HOME'] = '/foreign/cache'
                with self.assertRaisesRegex(ValueError, 'escaped'):
                    b.validate_assets(value)

    def test_alias_identical_retry_conflict_and_copy_fallback(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, target = root/'source', root/'nested/alias'
            source.write_bytes(b'official synthetic bytes')
            with patch.object(b, 'ROOT', root), patch.object(b.os, 'link', side_effect=OSError('different volume')):
                self.assertEqual(b.alias_file(source, target, b.sha(source)), 'copy')
                self.assertEqual(b.alias_file(source, target, b.sha(source)), 'existing_verified')
                target.write_bytes(b'keep me')
                with self.assertRaises(FileExistsError):
                    b.alias_file(source, target, b.sha(source))
                self.assertEqual(target.read_bytes(), b'keep me')

    def test_destination_symlink_and_outside_write_fail(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)/'task'
            root.mkdir()
            source = Path(temp)/'shared'
            source.write_bytes(b'abc')
            (root/'link').symlink_to(source)
            with patch.object(b, 'ROOT', root):
                with self.assertRaisesRegex(ValueError, 'Symlink'):
                    b.alias_file(source, root/'link', b.sha(source))
                with self.assertRaisesRegex(ValueError, 'escaped'):
                    b.write_exact(Path(temp)/'outside', 'denied')
                with self.assertRaisesRegex(ValueError, 'escaped'):
                    b.write_exact(root/'../outside', 'denied')
            self.assertEqual(source.read_bytes(), b'abc')

    def test_apply_forbidden_at_local_workspace(self):
        with self.assertRaisesRegex(RuntimeError, 'fixed A100 task root'):
            b.require_a100()

    def apply_fixture(self, root, returncode=0, child_failure=False):
        args = SimpleNamespace(out=root/'receipt', overlay_inventory=None, upstream_inventory=None)
        calls = []
        def assets(models, out):
            path = out/'assets.json'
            path.write_text('{}')
            return path
        def run(argv, **kwargs):
            calls.append((argv, kwargs['env']))
            kwargs['stdout'].write('synthetic CPU check, no environment installed\n')
            result = Path(argv[argv.index('--result')+1])
            result.write_text(json.dumps(dict(
                status='failed_CPU_check_GPU_not_run' if child_failure else 'teacher_only_CPU_imports_config_and_offline_resolver_passed_GPU_unverified',
                checked=[{'module': 'synthetic_completed_import'}], forbidden_attempts=[],
                CUDA_initialized=False, GPU_calls=0, checkpoint_loads=0,
                teacher_models_instantiated=0, teacher_images_generated=0)))
            return SimpleNamespace(returncode=returncode)
        from contextlib import ExitStack
        with ExitStack() as stack:
            for name, value in [('ROOT', root), ('require_a100', lambda: None),
                ('plan', lambda *a: {'files': []}), ('verify_files', lambda p: {'files': 0}),
                ('verify_sources', lambda e: {'stablesr': {'tracked_clean': True}}),
                ('query_base', lambda e: {'purelib': '/base/site'}), ('verified_models', lambda: {}),
                ('overlay', lambda base, env: {'python': '/synthetic/new/venv/bin/python'}),
                ('runtime_assets', assets)]:
                stack.enter_context(patch.object(b, name, value))
            stack.enter_context(patch.object(b.subprocess, 'run', side_effect=run))
            if returncode or child_failure:
                with self.assertRaises(RuntimeError):
                    b.apply(args)
            else:
                b.apply(args)
        return calls, json.loads((args.out/'receipt.json').read_text())

    def test_apply_only_invokes_teacher_cpu_check(self):
        with tempfile.TemporaryDirectory() as temp:
            calls, record = self.apply_fixture(Path(temp))
            self.assertEqual(record['status'], 'teacher_only_CPU_bootstrap_passed_GPU_unverified')
            self.assertEqual(list(record['checks']), ['stablesr'])
            self.assertEqual(len(calls), 1)
            argv, env = calls[0]
            self.assertEqual(argv[2], 'check')
            self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '')
            self.assertEqual(env['HF_HUB_OFFLINE'], '1')
            self.assertFalse(record['training_started'])

    def test_failed_check_preserves_completed_evidence_and_log(self):
        with tempfile.TemporaryDirectory() as temp:
            calls, record = self.apply_fixture(Path(temp), returncode=1, child_failure=True)
            self.assertEqual(record['status'], 'failed_CPU_bootstrap_GPU_not_run')
            self.assertEqual(record['checks']['stablesr']['returncode'], 1)
            item = record['checks']['stablesr']['result']
            self.assertEqual(item['sha256'], b.sha(item['path']))
            self.assertEqual(json.loads(Path(item['path']).read_text())['checked'],
                             [{'module': 'synthetic_completed_import'}])

    def test_zero_exit_without_passing_child_contract_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            calls, record = self.apply_fixture(Path(temp), child_failure=True)
            self.assertEqual(record['status'], 'failed_CPU_bootstrap_GPU_not_run')

    def test_source_check_verifies_both_pins_and_clean_worktrees(self):
        responses = [SimpleNamespace(stdout=b.COMMITS['stablesr']+'\n', returncode=0),
                     SimpleNamespace(returncode=0),
                     SimpleNamespace(stdout=b.COMMITS['taming-transformers']+'\n', returncode=0),
                     SimpleNamespace(returncode=0)]
        with patch.object(b.subprocess, 'run', side_effect=responses) as run:
            self.assertEqual(set(b.verify_sources(b.cpu_environment())), set(b.COMMITS))
        self.assertEqual(run.call_count, 4)
        self.assertTrue(all(call.kwargs['env']['GIT_OPTIONAL_LOCKS'] == '0' for call in run.call_args_list))
        with patch.object(b.subprocess, 'run', side_effect=[SimpleNamespace(stdout='wrong', returncode=0),
                                                          SimpleNamespace(returncode=0)]):
            with self.assertRaisesRegex(ValueError, 'HEAD/content'):
                b.verify_sources(b.cpu_environment())


if __name__ == '__main__':
    unittest.main()
