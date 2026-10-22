"""Return contracts: mixed artifacts and failed jobs must not report success."""
import importlib.util
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('submit_tested', Path(__file__).with_name('submit_remote.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
job_spec = importlib.util.spec_from_file_location('run_job_tested', Path(__file__).with_name('run_job.py'))
job_module = importlib.util.module_from_spec(job_spec)
job_spec.loader.exec_module(job_module)


class ReturnContracts(unittest.TestCase):
    def test_reject_escape_and_source_tree(self):
        for name in ('/tmp/out','../output','output/../../data','data/input','.',''):
            with self.subTest(name=name), self.assertRaises(ValueError):
                module.checked_return_path(name)

    def exercise(self, kinds, outcome):
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        root = Path(temp.name); dispatch = root/'dispatch'; dispatch.mkdir()
        calls = []
        def run(argv, **kwargs):
            calls.append(argv)
            return SimpleNamespace(stdout=json.dumps(kinds) if argv[0]=='ssh' else '', returncode=0)
        with patch.object(module,'ROOT',root), patch.object(module.subprocess,'run',side_effect=run):
            if outcome != 'success' or 'missing' in kinds.values():
                with self.assertRaises(RuntimeError):
                    module.collect_outputs('cts','/snapshot',list(kinds),{'outcome':outcome},dispatch)
            else:
                module.collect_outputs('cts','/snapshot',list(kinds),{'outcome':outcome},dispatch)
        return root, calls, json.loads((dispatch/'collected.json').read_text())

    def test_mixed_files_and_directories(self):
        root,calls,receipt = self.exercise({'output/run/compact':'directory','output/run/status.json':'file'},'success')
        copies=[a for a in calls if a[0]=='rsync']
        self.assertEqual(copies[0][-2],'cts:/snapshot/output/run/compact/')
        self.assertEqual(copies[1][-2],'cts:/snapshot/output/run/status.json')
        self.assertFalse((root/'output/run/status.json').is_dir())
        self.assertEqual(receipt['status'],'collected')
        self.assertFalse(receipt['local_SHA_verification'])

    def test_failed_job_collects_existing_and_records_missing(self):
        _,calls,receipt = self.exercise({'output/run/status.json':'file','output/run/later':'missing'},'failure')
        self.assertEqual(len([a for a in calls if a[0]=='rsync']),1)
        self.assertEqual(receipt['missing_paths'],['output/run/later'])
        self.assertEqual(receipt['status'],'partial_failure_evidence')

    def test_failure_with_all_outputs_still_raises(self):
        _, _, receipt = self.exercise({'output/run/status.json':'file'},'failure')
        self.assertEqual(receipt['status'], 'failure_evidence_collected')

    def test_failed_or_timed_out_transfer_does_not_hide_later_diagnostics(self):
        for exception in (subprocess.CalledProcessError(23, ['rsync']),
                          subprocess.TimeoutExpired(['rsync'], 7200)):
            with self.subTest(error=type(exception).__name__), tempfile.TemporaryDirectory() as temp:
                root=Path(temp); dispatch=root/'dispatch'; dispatch.mkdir(); calls=[]
                kinds={'output/run/compact':'directory', 'output/run/diagnostics.json':'file'}
                def run(argv, **kwargs):
                    calls.append(argv)
                    if argv[0]=='ssh':
                        return SimpleNamespace(stdout=json.dumps(kinds))
                    if argv[-2].endswith('/compact/'):
                        raise exception
                    return SimpleNamespace(stdout='')
                with patch.object(module,'ROOT',root), patch.object(module.subprocess,'run',side_effect=run):
                    with self.assertRaises(RuntimeError):
                        module.collect_outputs('cts','/snapshot',list(kinds),{'outcome':'failure'},dispatch)
                receipt=json.loads((dispatch/'collected.json').read_text())
                self.assertEqual(len([a for a in calls if a[0]=='rsync']),2)
                self.assertEqual(receipt['returned_paths'],[{'path':'output/run/diagnostics.json','kind':'file'}])
                self.assertEqual(receipt['transfer_errors'][0]['error_type'],type(exception).__name__)
                self.assertEqual(receipt['status'],'partial_failure_evidence')
                self.assertFalse(receipt['local_SHA_verification'])

    def test_local_path_type_conflicts_are_refused_before_copy(self):
        for kind in ('file','directory'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temp:
                root=Path(temp); target=root/'output/run/artifact'; target.parent.mkdir(parents=True)
                if kind=='file': target.mkdir()
                else: target.write_text('existing file stays')
                with patch.object(module,'ROOT',root), self.assertRaises(ValueError):
                    module.checked_return_destination('output/run/artifact',kind)
                if kind=='file': self.assertTrue(target.is_dir())
                else: self.assertEqual(target.read_text(),'existing file stays')

    def test_local_symlink_ancestors_and_broken_targets_are_refused(self):
        for position in ('ancestor','target','broken','root'):
            with self.subTest(position=position), tempfile.TemporaryDirectory() as temp:
                outer=Path(temp); root=outer/'snapshot'; root.mkdir()
                outside=outer/'outside'; outside.mkdir()
                (root/'output').mkdir()
                if position=='ancestor': (root/'output/run').symlink_to(outside, target_is_directory=True)
                elif position=='root':
                    alias=outer/'alias'; alias.symlink_to(root,target_is_directory=True); root=alias
                else:
                    (root/'output/run').mkdir()
                    (root/'output/run/status.json').symlink_to(outside/('missing' if position=='broken' else 'existing'))
                with patch.object(module,'ROOT',root), self.assertRaises(ValueError):
                    module.checked_return_destination('output/run/status.json','file')
                self.assertEqual(list(outside.iterdir()),[])

    def test_destination_conflict_still_returns_other_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); dispatch=root/'dispatch'; dispatch.mkdir()
            (root/'output/run/status.json').mkdir(parents=True)
            kinds={'output/run/status.json':'file','output/run/diagnostics.json':'file'}; calls=[]
            def run(argv,**kwargs):
                calls.append(argv)
                return SimpleNamespace(stdout=json.dumps(kinds) if argv[0]=='ssh' else '')
            with patch.object(module,'ROOT',root), patch.object(module.subprocess,'run',side_effect=run):
                with self.assertRaises(RuntimeError):
                    module.collect_outputs('cts','/snapshot',list(kinds),{'outcome':'success'},dispatch)
            receipt=json.loads((dispatch/'collected.json').read_text())
            self.assertEqual(len([a for a in calls if a[0]=='rsync']),1)
            self.assertEqual(receipt['transfer_errors'][0]['path'],'output/run/status.json')
            self.assertEqual(receipt['returned_paths'][0]['path'],'output/run/diagnostics.json')


class FailureDiagnostics(unittest.TestCase):
    def record(self):
        return dict(job_id='test',stages=[dict(name='teacher',returncode=2,seconds=1.25)])

    def test_failure_tail_is_bounded_with_full_log_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp); payload=b'prefix-not-in-tail\n'+b'x'*65536+b'\nreal failure\n'
            (out/'teacher.log').write_bytes(payload)
            result=job_module.diagnostics(self.record(),out,'RuntimeError: teacher failed')
            log=result['failed_stage']['log']
            self.assertEqual(result['status'],'failed')
            self.assertEqual(log['bytes'],len(payload))
            self.assertEqual(log['sha256'],hashlib.sha256(payload).hexdigest())
            self.assertTrue(log['tail_truncated'])
            self.assertEqual(log['tail_bytes'],65536)
            self.assertEqual(log['tail_text'],payload[-65536:].decode())

    def test_success_only_returns_short_stage_summary(self):
        with tempfile.TemporaryDirectory() as temp:
            result=job_module.diagnostics(self.record(),Path(temp))
            self.assertEqual(result['status'],'completed')
            self.assertNotIn('failed_stage',result)
            self.assertEqual(result['stages'],self.record()['stages'])

    def test_short_or_missing_failure_log_keeps_exception(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp)
            missing=job_module.diagnostics(self.record(),out,'launch failed')
            self.assertIn('read_error',missing['failed_stage']['log'])
            self.assertEqual(missing['exception'],'launch failed')
            (out/'teacher.log').write_bytes(b'bad input\n')
            short=job_module.diagnostics(self.record(),out,'bad input')
            self.assertFalse(short['failed_stage']['log']['tail_truncated'])
            self.assertEqual(short['failed_stage']['log']['tail_bytes'],10)

    def test_real_cpu_stage_exit_always_writes_diagnostics(self):
        for returncode in (0, 2):
            with self.subTest(returncode=returncode), tempfile.TemporaryDirectory() as temp:
                root=Path(temp); broker=root/'broker'; broker.mkdir()
                (broker/'execution_context.json').write_text(json.dumps(dict(
                    project_id='4dsr',selected_gpu_uuids=['CPU-test-no-CUDA'])))
                source=root/'job.json'
                source.write_text(json.dumps(dict(job_id='cpu-diagnostics-test',method='test',scene='test',
                    gpu_uuid='CPU-test-no-CUDA',training_only_inputs=True,no_proposed_modules=True,
                    out='output/job',shared_project_root='',env={'CUDA_VISIBLE_DEVICES':''},
                    commands=[dict(name='cpu_stage',argv=[sys.executable,'-c',
                        "import sys; print('CPU stage stderr evidence', file=sys.stderr); sys.exit("+str(returncode)+")"])])))
                with patch.object(job_module,'ROOT',root), patch.object(sys,'argv',['run_job.py','--job',str(source)]), \
                        patch.dict(os.environ,{'CLAUDE_REMOTE_JOB_DIR':str(broker),'CUDA_VISIBLE_DEVICES':''}):
                    if returncode:
                        with self.assertRaises(RuntimeError): job_module.main()
                    else: job_module.main()
                result=json.loads((root/'output/job/diagnostics.json').read_text())
                self.assertEqual(result['status'],'failed' if returncode else 'completed')
                if returncode:
                    self.assertIn('CPU stage stderr evidence',result['failed_stage']['log']['tail_text'])
                    self.assertIn('Stage failed: cpu_stage rc=2',result['exception'])
                else:
                    self.assertNotIn('failed_stage',result)


if __name__=='__main__': unittest.main()
