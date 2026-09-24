"""No GPU, real Codex invocation, notification, or experiment mutations."""
import os
from pathlib import Path
import select
import subprocess
import sys
import tempfile
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import completion_trigger as trigger


class CompletionTriggerTests(unittest.TestCase):
    def test_real_exit_events_success_and_failure(self):
        for code in (0,7):
            with self.subTest(code=code):
                child=subprocess.Popen([sys.executable,'-c',
                    f'import sys;sys.stdin.read();sys.exit({code})'],stdin=subprocess.PIPE)
                fd=os.pidfd_open(child.pid)
                try:
                    child.stdin.close()
                    ready,_,_=select.select([fd],[],[])
                    self.assertEqual(ready,[fd]);self.assertEqual(child.wait(),code)
                finally:os.close(fd)

    def test_review_process_exit_and_timeout(self):
        with tempfile.TemporaryFile(mode='w+') as log:
            for code in (0,9):
                result=trigger.run_review_process([sys.executable,'-c',f'raise SystemExit({code})'],'',log,2)
                self.assertEqual(result['returncode'],code)
                self.assertFalse(result['timed_out'])
            result=trigger.run_review_process([sys.executable,'-c','import signal;signal.pause()'],'',log,.1)
            self.assertTrue(result['timed_out']);self.assertLess(result['elapsed_s'],2)
            self.assertEqual(result['returncode'],-9)

    def test_subscription_exit_races(self):
        with patch.object(trigger,'read',return_value={'status':'running','pid':123}), \
             patch.object(trigger.os,'pidfd_open',side_effect=ProcessLookupError):
            self.assertIsNone(trigger.watch_state('unused'))
        states=[{'status':'running','pid':123},{'status':'complete','pid':123}]
        with patch.object(trigger,'read',side_effect=states), \
             patch.object(trigger.os,'pidfd_open',return_value=81),patch.object(trigger.os,'close') as close:
            self.assertIsNone(trigger.watch_state('unused'));close.assert_called_once_with(81)
        states=[{'status':'running','pid':123},{'status':'running','pid':456}]
        with patch.object(trigger,'read',side_effect=states), \
             patch.object(trigger.os,'pidfd_open',return_value=82),patch.object(trigger.os,'close') as close:
            with self.assertRaises(AssertionError):trigger.watch_state('unused')
            close.assert_called_once_with(82)

    def test_review_markers_and_timeout_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest=Path(tmp)
            with patch.object(trigger,'run_review_process',return_value={
                    'returncode':-9,'timed_out':True,'elapsed_s':.1}) as run:
                self.assertEqual(trigger.review(dest,'completed',.1)['status'],'timeout')
                self.assertEqual(trigger.review(dest,'completed',.1)['status'],'already_started')
                run.assert_called_once()
                self.assertEqual(trigger.read(dest/'completed_review_result.json')['status'],'timeout')

    def test_review_success_requires_nonempty_report_and_zero_exit(self):
        for code,report_text,expected in ((0,'verified','complete'),(0,'','failed'),(3,'partial','failed')):
            with self.subTest(code=code,report_text=report_text),tempfile.TemporaryDirectory() as tmp:
                def run(cmd,prompt,log,timeout):
                    Path(cmd[cmd.index('-o')+1]).write_text(report_text)
                    return {'returncode':code,'timed_out':False,'elapsed_s':.01}
                with patch.object(trigger,'run_review_process',side_effect=run):
                    self.assertEqual(trigger.review(Path(tmp),'completed',1)['status'],expected)

    def test_notification_is_at_most_once_and_failure_is_nonfatal(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(trigger,'notify',return_value={'status':'sent'}) as notify:
            self.assertEqual(trigger.notify_once(Path(tmp),'failure','title','body')['status'],'sent')
            self.assertEqual(trigger.notify_once(Path(tmp),'failure','title','body')['status'],'already_attempted')
            notify.assert_called_once()
        with patch.object(trigger.shutil,'which',return_value='/mock/notify-send'), \
             patch.object(trigger.subprocess,'run',side_effect=subprocess.TimeoutExpired('mock',10)):
            self.assertEqual(trigger.notify('title','body')['status'],'failed')

    def test_local_success_notification_precedes_review(self):
        calls=[]
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            trigger.write(out/'scene_queue.json',{'status':'complete','pid':1})
            trigger.write(out/'generalization_trajectory/state.json',{'status':'complete','completed':list(range(18))})
            trigger.write(out/'generalization_trajectory/metrics.json',{'no_checkpoint_selection':True})
            def fake_review(*args):
                self.assertIn('notify',calls)
                self.assertEqual(trigger.read(out/'events/trigger_state.json')['status'],'verified')
                calls.append('review');return {'status':'complete'}
            with patch.object(trigger,'OUT',out),patch.object(trigger,'SCENES',['scene']), \
                 patch.object(trigger,'verify_scene',return_value={'scene':'scene','status':'passed'}), \
                 patch.object(trigger,'summary',return_value=SimpleNamespace(returncode=0,stderr='')), \
                 patch.object(trigger,'notify',side_effect=lambda *args:calls.append('notify') or {'status':'sent'}), \
                 patch.object(trigger,'review',side_effect=fake_review):
                result=trigger.listen(out/'events',1)
            self.assertEqual(result['status'],'verified');self.assertEqual(calls,['notify','review'])

    def test_blocked_failure_review_does_not_block_next_scene(self):
        review_started=threading.Event();second_checked=threading.Event();calls=[]
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            trigger.write(out/'bad_queue.json',{'status':'failed','pid':1,'error':'intentional'})
            trigger.write(out/'good_queue.json',{'status':'complete','pid':2})
            def fake_review(*args):
                self.assertEqual(calls,['notify'])
                review_started.set()
                self.assertTrue(second_checked.wait(2),'Model review blocked the event handler')
                return {'status':'failed'}
            def fake_verify(scene):
                self.assertTrue(review_started.wait(2),'Review worker did not start')
                second_checked.set();return {'scene':scene,'status':'passed'}
            with patch.object(trigger,'OUT',out),patch.object(trigger,'SCENES',['bad','good']), \
                 patch.object(trigger,'verify_scene',side_effect=fake_verify), \
                 patch.object(trigger,'summary',return_value=SimpleNamespace(returncode=0,stderr='')), \
                 patch.object(trigger,'notify',side_effect=lambda *args:calls.append('notify') or {'status':'sent'}), \
                 patch.object(trigger,'review',side_effect=fake_review) as review:
                result=trigger.listen(out/'events',1)
            self.assertEqual(result['status'],'failed')
            self.assertEqual(result['scenes']['good']['status'],'passed');review.assert_called_once()

    def test_failed_trajectory_notifies_before_failure_review(self):
        calls=[]
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            trigger.write(out/'scene_queue.json',{'status':'complete','pid':1})
            trigger.write(out/'generalization_trajectory/state.json',{'status':'failed','completed':[]})
            def fake_review(*args):
                self.assertEqual(calls,['notify']);return {'status':'failed'}
            with patch.object(trigger,'OUT',out),patch.object(trigger,'SCENES',['scene']), \
                 patch.object(trigger,'verify_scene',return_value={'scene':'scene','status':'passed'}), \
                 patch.object(trigger,'summary',return_value=SimpleNamespace(returncode=0,stderr='')), \
                 patch.object(trigger,'notify',side_effect=lambda *args:calls.append('notify') or {'status':'sent'}), \
                 patch.object(trigger,'review',side_effect=fake_review):
                result=trigger.listen(out/'events',1)
            self.assertEqual(result['trajectory']['status'],'failed')


if __name__=='__main__':unittest.main(verbosity=2)
