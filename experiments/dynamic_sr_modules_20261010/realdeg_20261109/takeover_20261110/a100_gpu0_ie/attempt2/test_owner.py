"""CPU-only synthetic owner admission/fail-stop tests; no service or CUDA calls."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('ie_gpu0_owner_test', HERE / 'owner.py')
O = importlib.util.module_from_spec(spec)
spec.loader.exec_module(O)


class OwnerTests(unittest.TestCase):
    def test_original_plan_and_only_host_adaptation(self):
        plan = O.read(HERE / 'plan.json')
        O.validate_plan(plan)
        changed = copy.deepcopy(plan)
        changed['adapted_tasks'][1]['argv'].extend(['--fine', '1'])
        with self.assertRaisesRegex(RuntimeError, 'Host adaptation'):
            O.validate_plan(changed)
        changed = copy.deepcopy(plan)
        changed['source_bindings'][0]['sha256'] = '0' * 64
        with self.assertRaisesRegex(RuntimeError, 'Bound input changed'):
            O.validate_plan(changed)
        changed = copy.deepcopy(plan)
        changed['gpu_uuid'] = 'GPU-wrong'
        with self.assertRaisesRegex(RuntimeError, 'Changed host'):
            O.validate_plan(changed)

    def test_default_path_never_imports_torch_or_executes(self):
        result = subprocess.run([sys.executable, '-B', str(HERE / 'owner.py')],
                                check=True, text=True, capture_output=True)
        self.assertEqual(json.loads(result.stdout)['GPU_calls'], 0)
        self.assertNotIn('torch', O.__dict__)
        with self.assertRaisesRegex(RuntimeError, 'Wrong execution root'):
            O.execution_context(O.read(HERE / 'plan.json'))

    def test_readonly_alias_bound_inside_root_and_output_symlink_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'target').write_text('same bytes')
            (root / 'alias').symlink_to(root / 'target')
            value = dict(path='alias', sha256=O.sha(root / 'target'))
            self.assertEqual(O.check_ref(root, value).resolve(), root / 'target')
            with self.assertRaisesRegex(RuntimeError, 'Symlink'):
                O.safe_path(root, 'alias')
            (root / 'outside').symlink_to('/etc/passwd')
            with self.assertRaisesRegex(RuntimeError, 'escapes'):
                O.check_ref(root, dict(path='outside', sha256='0' * 64))

    def test_activation_binds_final_plan_and_delegation_without_hash_cycle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            here = root / 'source'
            here.mkdir()
            (here / 'owner.py').write_text('synthetic owner source')
            plan = dict(handoff=dict(delegation_path='handoff/delegation.json', activation_path='handoff/activation.json'),
                original_lane=dict(sha256='1' * 64), config=dict(sha256='2' * 64),
                original_request=dict(sha256='3' * 64), original_source_lock=dict(sha256='4' * 64))
            O.write_new(here / 'plan.json', plan)
            delegation = dict(schema='rd4_local_baseline_owner_delegation_v1', original_job_id=O.ORIGINAL_JOB,
                method=O.METHOD, scene=O.SCENE, remote_owner_job_id=O.JOB, target_host='a100-train',
                target_gpu_uuid=O.GPU, state='ownership_assigned_not_completion',
                remote_completion_claimed=False, auto_local_fallback=False,
                remote_runner=O.ref(root, here / 'owner.py'), remote_plan=O.ref(root, here / 'plan.json'),
                original=dict(lane=plan['original_lane'], config=plan['config'], request=plan['original_request'],
                              source_lock=plan['original_source_lock']))
            O.write_new(root / plan['handoff']['delegation_path'], delegation)
            activation = dict(status='active', original_job_id=O.ORIGINAL_JOB, remote_owner_job_id=O.JOB,
                old_waiting_controller_stopped=True, replacement_excludes_original_job=True,
                delegation_sha256=O.sha(root / plan['handoff']['delegation_path']),
                owner_plan_sha256=O.sha(here / 'plan.json'), owner_runner_sha256=O.sha(here / 'owner.py'))
            with patch.object(O, 'ROOT', root), patch.object(O, 'HERE', here):
                with self.assertRaises(FileNotFoundError):
                    O.handoff(plan)
                O.write_new(root / plan['handoff']['activation_path'], activation)
                self.assertEqual(O.handoff(plan)['activation']['sha256'], O.sha(root / plan['handoff']['activation_path']))
                (here / 'plan.json').write_text('{"changed":true}')
                with self.assertRaisesRegex(RuntimeError, 'final runner/plan'):
                    O.handoff(plan)

    def test_child_failure_is_recorded_and_next_step_never_starts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'owner.py').write_text('fixture')
            (root / 'plan.json').write_text('{}')
            plan = dict(output_root='cell', ownership_file='cell/owner_a100_gpu0.json',
                original_lane={}, config={}, python=sys.executable,
                rade_runtime='unused', adapted_tasks=[dict(id='teacher'), dict(id='internal')])
            calls = []
            def fake_child(plan, flag, value, log_root, env):
                calls.append((flag, value))
                if flag == '--preflight':
                    O.write_new(log_root / 'gpu_preflight.json', dict(status='passed'))
                    return dict(task='preflight', returncode=0)
                raise RuntimeError('synthetic teacher failure')
            import disk_guard
            with patch.object(O, 'ROOT', root), patch.object(O, 'HERE', root), \
                    patch.object(O, 'execution_context', return_value={}), \
                    patch.object(O, 'handoff', return_value={}), \
                    patch.object(O, 'asset_checks', return_value=[]), \
                    patch.object(O, 'run_child', side_effect=fake_child), \
                    patch.object(disk_guard, 'check', return_value={}):
                with self.assertRaisesRegex(RuntimeError, 'synthetic teacher failure'):
                    O.run(plan)
                self.assertEqual(calls, [('--preflight', ''), ('--task', 'teacher')])
                state = O.read(root / 'cell/jobs' / O.JOB / 'status.json')
                self.assertEqual(state['status'], 'failed')
                self.assertFalse((root / 'cell/jobs' / O.JOB / 'complete.json').exists())
                with self.assertRaisesRegex(RuntimeError, 'never adopt/rerun'):
                    O.fresh_outputs(plan)

    def test_child_nonzero_exit_gets_exact_exit_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def failed(argv, **kwargs):
                return subprocess.CompletedProcess(argv, 7)
            with patch.object(O, 'ROOT', root), patch.object(O, 'HERE', root):
                with self.assertRaisesRegex(RuntimeError, 'Child failed'):
                    O.run_child(dict(python=sys.executable), '--task', 'fixture', root, {}, invoke=failed)
                receipt = O.read(root / 'fixture.exit.json')
                self.assertEqual(receipt['returncode'], 7)
                self.assertEqual(receipt['task'], 'fixture')


if __name__ == '__main__':
    unittest.main()
