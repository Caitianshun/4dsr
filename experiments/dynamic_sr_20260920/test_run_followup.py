"""Failure handling and real child-exit checks; no GPU or experiment launch."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

spec = importlib.util.spec_from_file_location('followup', Path(__file__).with_name('run_followup.py'))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class CompletionTests(unittest.TestCase):
    def test_real_child_failure_keeps_log_and_raises(self):
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / 'child.log'
            with self.assertRaisesRegex(RuntimeError, 'exit 7'):
                m.checked_run([sys.executable, '-c', 'print("saved evidence", flush=True); raise SystemExit(7)'], log, os.environ)
            self.assertEqual(log.read_text().strip(), 'saved evidence')

    def test_return_happens_after_child_finished_writing(self):
        with tempfile.TemporaryDirectory() as d:
            log, marker = Path(d) / 'child.log', Path(d) / 'finished'
            command = [sys.executable, '-c', 'import pathlib,sys;pathlib.Path(sys.argv[1]).write_text("done")', str(marker)]
            event = m.checked_run(command, log, os.environ)
            self.assertEqual(marker.read_text(), 'done')
            self.assertGreaterEqual(event['seconds'], 0)

    def test_failed_training_never_launches_evaluation(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            source = root / 'source/cook_spinach_sr_w01'
            source.mkdir(parents=True)
            (source / 'config.json').write_text(json.dumps({'manifest': '/unused/manifest.json'}))
            with mock.patch.object(m, 'OUT', root / 'out'), mock.patch.object(m, 'ORIGINAL', root / 'source'), \
                 mock.patch.object(m, 'checked_run', side_effect=RuntimeError('synthetic training failure')) as run, \
                 mock.patch.object(m, 'verify_run') as verify:
                with self.assertRaisesRegex(RuntimeError, 'synthetic'):
                    m.scene_worker('cook_spinach', 0)
                self.assertEqual(run.call_count, 1)
                verify.assert_not_called()
            state = json.loads((root / 'out/cook_spinach_state.json').read_text())
            self.assertEqual(state['status'], 'failed')
            self.assertEqual(state['completed'], [])
            self.assertEqual(state['current']['stage'], 'training')


if __name__ == '__main__':
    unittest.main()
