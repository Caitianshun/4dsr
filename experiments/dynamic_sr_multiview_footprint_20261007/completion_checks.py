"""CPU-only event callback checks. No SSH, systemd service or GPU dispatch."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch
import completion as C
from fp_common import ROOT, HERE, OUT, read, write, entry, sha


def asynchronous(function):
    results = queue.Queue()
    def target():
        try: results.put(('ok', function()))
        except BaseException as error: results.put(('error', repr(error)))
    thread = threading.Thread(target=target, daemon=True); thread.start()
    return thread, results


def receive(results):
    status, value = results.get(timeout=10)
    if status != 'ok': raise AssertionError(value)
    return value


def fixture_endpoint(out, task):
    directory = out/'runs'/task; directory.mkdir(parents=True, exist_ok=True)
    cp = directory/'checkpoint_12000.pt'; cp.write_text('SYNTHETIC CPU CHECK ONLY; NOT MODEL WEIGHTS\n'+task)
    write(directory/'complete.json', dict(status='SYNTHETIC_CPU_completed_training_fixture', checkpoint=entry(cp)))
    request = dict(status='pending_uniform_evaluation', task=task, checkpoint=entry(cp))
    write(out/'evaluation_queue'/(task+'.json'), request)
    return request


def fixture_evaluation(out, task, request):
    evaluation = out/'evaluation'/task/'extra/adapter_complete.json'
    diagnostic = out/'diagnostics'/task/'diagnostics.json'
    write(evaluation, dict(status='SYNTHETIC_CPU_evaluation_fixture'))
    write(diagnostic, dict(status='SYNTHETIC_CPU_diagnostic_fixture'))
    write(out/'tasks'/(task+'.json'), dict(status='completed_training_evaluation_fixed_diagnostics',
        task=task, training=entry(out/'runs'/task/'complete.json'), checkpoint=request['checkpoint'],
        evaluation=entry(evaluation), diagnostics=entry(diagnostic)))


class FakeTransport:
    def __init__(self, events=(), request=None):
        self.events, self.requests = events, request; self.calls = []
    def stream(self, mode, *arguments):
        self.calls.append(('stream', mode, arguments)); yield from self.events
    def bundle(self, purpose, task='', terminal=None):
        self.calls.append(('bundle', purpose, task)); return dict(status='SYNTHETIC_CPU_transfer_fixture', terminal=terminal)
    def request(self, task):
        self.calls.append(('request', task)); return self.requests


def checks(out):
    started = time.perf_counter(); results = {}
    fixture_root = out/'fixtures'; fixture_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='callback_', dir=fixture_root) as temporary:
        directory = Path(temporary)
        # A real CPU child is controlled by stdin. Bind its pidfd before the
        # release message, then prove POLLIN and independently recover Exit7.
        child = subprocess.Popen([sys.executable, '-u', '-c',
            "import sys;print('ready',flush=True);sys.stdin.readline();sys.exit(7)"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        assert child.stdout.readline().strip() == 'ready'
        original = os.pidfd_open; bound = threading.Event()
        def bind(pid):
            descriptor = original(pid); bound.set(); return descriptor
        with patch.object(C.os, 'pidfd_open', bind):
            thread, delivered = asynchronous(lambda: C.pidfd_wait(child.pid))
            assert bound.wait(5); child.stdin.write('exit\n'); child.stdin.flush()
            proof = receive(delivered); thread.join(5)
        assert proof['exit_proved'] and child.wait(timeout=5) == 7
        results['real_CPU_child_pidfd_exit_event_does_not_infer_Exit0'] = True

        events = C.DirectoryEvents(directory/'queue', directory/'events.fifo')
        thread, delivered = asynchronous(events.next)
        write(directory/'queue/r1_B0.json', dict(task='r1_B0'))
        message = receive(delivered); assert any(row['name'] == 'r1_B0.json' and row['mechanism'] == 'inotify' for row in message)
        thread.join(5); results['real_atomic_close_rename_inotify_trigger'] = True
        thread, delivered = asynchronous(events.next)
        with (directory/'events.fifo').open('w') as fifo: fifo.write(json.dumps(dict(task='r1_X'))+'\n')
        message = receive(delivered); assert any(row['name'] == 'r1_X.json' and row['mechanism'] == 'FIFO' for row in message)
        thread.join(5); events.close(); results['real_FIFO_message_event_trigger'] = True

        dependencies = {}
        for key in ('protocol', 'fixture', 'parent', 'support', 'calibration'):
            path = directory/'dependencies'/(key+'.json'); write(path, dict(status='passed' if key == 'calibration' else 'SYNTHETIC_CPU_dependency'))
            dependencies[key] = entry(path)
        prep = directory/'prepared'; watcher = C.DirectoryEvents(prep/'preparation')
        thread, delivered = asynchronous(lambda: C.wait_for_preparation(prep, watcher))
        assert thread.is_alive()
        write(prep/'preparation/complete.json', dict(status='completed_native_support_single_calibration', formal_updates=0, **dependencies))
        assert receive(delivered) is None; thread.join(5); watcher.close()
        results['preparation_gate_blocks_and_wakes_only_on_actual_complete_event'] = True
        watcher = C.DirectoryEvents(prep/'preparation'); C.wait_for_preparation(prep, watcher); watcher.close()
        results['preparation_completed_before_subscribe_recovers_once'] = True

        successful = dict(kind='terminal', invocation_id='fixture-invocation', successful=True,
            exit_proved=True, exit_kind=1, exit_code=0)
        assert C.validate_preparation(successful, prep) == entry(prep/'preparation/complete.json')
        for changed in (dict(successful=False, exit_code=7), dict(exit_proved=False), dict(exit_kind=2), dict(exit_code=None)):
            bad = dict(successful, **changed)
            try: C.validate_preparation(bad, prep)
            except ValueError: pass
            else: raise AssertionError('Failed/unknown/signaled exit accepted')
        results['strict_success_and_failure_unknown_signal_exit_separation'] = True

        args = SimpleNamespace(host='SYNTHETIC_CPU_HOST', workspace='SYNTHETIC_CPU_WORKSPACE', out=prep,
            remote_python=sys.executable, remote_activate=None, unit='fixture-unit', system_scope=False)
        stream = [dict(kind='bound', invocation_id='fixture-invocation'), successful]
        fake = FakeTransport(stream)
        assert C.prepare_return(args, fake)
        assert read(prep/'completion/prepare_return_complete.json')['status'].startswith('completed_')
        failure = dict(successful, successful=False, exit_code=9)
        fake = FakeTransport([dict(kind='bound', invocation_id='fixture-invocation'), failure])
        assert not C.prepare_return(args, fake)
        assert any(row[:2] == ('bundle', 'prepare') for row in fake.calls)
        assert read(prep/'completion/prepare_return_complete.json')['status'] == 'failed_preparation_returned_artifacts'
        results['prepare_return_success_and_failure_both_return_artifacts'] = True
        fake = FakeTransport([dict(kind='bound', invocation_id='different-invocation')])
        try: C.prepare_return(args, fake)
        except ValueError: pass
        else: raise AssertionError('Different restarted unit accepted')
        results['reconnect_cannot_cross_unit_invocation_identity'] = True
        fake = FakeTransport(stream)
        fake_count = {'attempt':0, 'backoff':0}
        def reconnect(mode, *arguments):
            fake_count['attempt'] += 1
            if fake_count['attempt'] == 1:
                yield stream[0]; raise ConnectionError('SYNTHETIC_CPU_disconnect')
            yield from stream
        fake.stream = reconnect
        with patch.object(C, 'backoff', lambda attempt: fake_count.__setitem__('backoff', fake_count['backoff']+1)):
            assert C.prepare_return(args, fake)
        assert fake_count == dict(attempt=2, backoff=1)
        results['network_reconnect_restores_same_bound_invocation_without_model_polling'] = True

        endpoint_out = directory/'endpoint'
        request = fixture_endpoint(endpoint_out, 'r1_B0'); calls=[]
        def evaluate(task):
            calls.append(task); fixture_evaluation(endpoint_out, task, request)
            return 0, ['SYNTHETIC_CPU_evaluator'], 'SYNTHETIC_CPU_log'
        consumer = C.EvaluationConsumer(SimpleNamespace(out=endpoint_out, gpu='GPU-SYNTHETIC_CPU'),
            FakeTransport(request=request), evaluate)
        first = consumer.consume(dict(name='r1_B0.json', origin='local', mechanism='startup_recovery'))
        second = consumer.consume(dict(name='r1_B0.json', origin='local', mechanism='inotify'))
        third = consumer.consume(dict(name='r1_B0.json', origin='remote', mechanism='inotify'))
        assert first['status'] == second['status'] == third['status'] == 'completed_uniform_evaluation_event_callback' and calls == ['r1_B0']
        assert consumer.transport.calls == [('request', 'r1_B0')]
        results['existing_queue_one_startup_recovery_local_remote_duplicates_are_idempotent'] = True
        consumer2 = C.EvaluationConsumer(SimpleNamespace(out=endpoint_out, gpu='GPU-SYNTHETIC_CPU'), FakeTransport(request=request), evaluate)
        consumer2.consume(dict(name='r1_B0.json', origin='local', mechanism='startup_recovery'))
        assert calls == ['r1_B0']; results['consumer_restart_durable_receipt_does_not_repeat_evaluation'] = True

        failed_request = fixture_endpoint(endpoint_out, 'r1_X'); failed_calls=[]
        def failed(task): failed_calls.append(task); return 7, ['SYNTHETIC_CPU_failure'], 'SYNTHETIC_CPU_log'
        bad_consumer = C.EvaluationConsumer(SimpleNamespace(out=endpoint_out, gpu='GPU-SYNTHETIC_CPU'), FakeTransport(), failed)
        result = bad_consumer.consume(dict(name='r1_X.json', origin='local', mechanism='startup_recovery'))
        assert result['status'] == 'failed_uniform_evaluation_event_callback'
        bad_consumer.consume(dict(name='r1_X.json', origin='local', mechanism='inotify'))
        assert failed_calls == ['r1_X']; results['failed_evaluation_receipt_preserved_duplicate_does_not_spin'] = True
        fixture_endpoint(endpoint_out, 'r1_M')
        zero_but_missing = C.EvaluationConsumer(SimpleNamespace(out=endpoint_out, gpu='GPU-SYNTHETIC_CPU'), FakeTransport(),
            lambda task: (0, ['SYNTHETIC_CPU_missing_receipt'], 'SYNTHETIC_CPU_log'))
        result = zero_but_missing.consume(dict(name='r1_M.json', origin='local', mechanism='inotify'))
        assert result['status'] == 'failed_uniform_evaluation_event_callback' and 'error' in result
        results['Exit0_without_bound_evaluation_and_diagnostics_is_not_completion'] = True
        try: C.valid_task('r3_EXTRA')
        except ValueError: pass
        else: raise AssertionError('Expanded method/seed admitted')
        results['exact_registered_12_tasks_no_expansion'] = len(C.TASKS) == 12
        # Exercise the real remote manifest branch locally, with a synthetic
        # workspace and an injected rsync copier. No SSH command is executed.
        remote_root = directory/'remote'; remote_out = remote_root/C.REL_OUT
        remote_request = fixture_endpoint(remote_out, 'r2_B0')
        source_relative = HERE.joinpath('completion.py').relative_to(ROOT)
        remote_source = remote_root/source_relative; remote_source.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(HERE/'completion.py',remote_source)
        write(remote_out/'runs/r2_B0/config.json',dict(source_identity={str(source_relative):sha(remote_source)}))
        (remote_out/'logs').mkdir(parents=True,exist_ok=True)
        (remote_out/'logs/r2_B0.log').write_text('SYNTHETIC_CPU_remote_endpoint_log\n')
        command=[sys.executable,'-u','-c',C.REMOTE_HELPER,'manifest',str(remote_root),C.REL_OUT,'task','r2_B0']
        returned=subprocess.run(command,capture_output=True,text=True,check=True)
        manifest=json.loads(returned.stdout)
        assert manifest['status']=='frozen_remote_file_SHA_inventory'
        destination=directory/'received'; transport=C.Transport('SYNTHETIC_CPU_HOST',remote_root,destination)
        transport.stream=lambda *arguments: iter([manifest])
        fake_rsync_calls=[]
        def fake_rsync(command,**kwargs):
            fake_rsync_calls.append(command)
            assert command[0]=='rsync'
            source=Path(manifest['transfer_root']);target=Path(command[-1]);target.mkdir(parents=True,exist_ok=True)
            for record in manifest['entries']:
                src=source/record['path'];dst=target/record['path'];dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
            return SimpleNamespace(returncode=0)
        with patch.object(C.subprocess,'run',fake_rsync):
            imported=transport.bundle('task','r2_B0')
        assert imported['status']=='completed_received_all_file_SHA_verified'
        assert sha(destination/'runs/r2_B0/checkpoint_12000.pt')==remote_request['checkpoint']['sha256']
        results['real_local_manifest_snapshot_and_injected_rsync_verify_every_SHA']=True
        incomplete=directory/'incomplete_received'; incomplete_transport=C.Transport('SYNTHETIC_CPU_HOST',remote_root,incomplete)
        incomplete_transport.stream=lambda *arguments:iter([manifest])
        def partial(command,**kwargs):
            target=Path(command[-1]);target.mkdir(parents=True,exist_ok=True)
            record=manifest['entries'][0];dst=target/record['path'];dst.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(Path(manifest['transfer_root'])/record['path'],dst)
            return SimpleNamespace(returncode=23)
        with patch.object(C.subprocess,'run',partial):
            try:incomplete_transport.bundle('task','r2_B0')
            except ConnectionError:pass
            else:raise AssertionError('Partial rsync accepted')
        assert not (incomplete/'runs/r2_B0/complete.json').exists()
        assert any((incomplete/'completion/incoming').iterdir())
        results['partial_failed_transfer_retains_staging_never_publishes_complete']=True
        corrupt=dict(manifest,entries=[dict(row) for row in manifest['entries']])
        corrupt['entries'][0]['sha256']='0'*64
        corrupt_out=directory/'corrupt_received'; corrupt_transport=C.Transport('SYNTHETIC_CPU_HOST',remote_root,corrupt_out)
        corrupt_transport.stream=lambda *arguments:iter([corrupt])
        with patch.object(C.subprocess,'run',fake_rsync):
            try:corrupt_transport.bundle('task','r2_B0')
            except ValueError:pass
            else:raise AssertionError('Corrupt byte SHA accepted')
        assert not (corrupt_out/'runs/r2_B0/complete.json').exists()
        results['received_SHA_conflict_keeps_staging_and_never_publishes_complete']=True
        source_bad=dict(manifest,source_identities=[dict(row) for row in manifest['source_identities']])
        source_bad['source_identities'][0]['sha256']='0'*64
        conflict=directory/'source_conflict_received'; conflict_transport=C.Transport('SYNTHETIC_CPU_HOST',remote_root,conflict)
        conflict_transport.stream=lambda *arguments:iter([source_bad])
        with patch.object(C.subprocess,'run',fake_rsync):
            try:conflict_transport.bundle('task','r2_B0')
            except ValueError:pass
            else:raise AssertionError('Different trained source accepted')
        assert not (conflict/'runs/r2_B0/complete.json').exists()
        assert sha(HERE/'completion.py')==sha(remote_source)
        results['source_identity_conflict_retains_bytes_does_not_overwrite_live_source']=True
        # The helper termination predicate accepts retained Exit0 only after
        # MainPID=0, SubState=exited and a genuine monotonic exit timestamp.
        begin=C.REMOTE_HELPER.index(' def finished(properties):')
        end=C.REMOTE_HELPER.index(' while not finished(current):',begin)
        namespace={};exec(__import__('textwrap').dedent(C.REMOTE_HELPER[begin:end]),namespace)
        finished=namespace['finished']
        assert finished(dict(ActiveState='active',SubState='exited',MainPID='0',ExecMainExitTimestampMonotonic='1'))
        assert not finished(dict(ActiveState='active',SubState='running',MainPID='123',ExecMainExitTimestampMonotonic='0'))
        assert not finished(dict(ActiveState='active',SubState='exited',MainPID='0',ExecMainExitTimestampMonotonic='0'))
        results['RemainAfterExit_active_exited_supported_active_running_not_success']=True
        compile(C.REMOTE_HELPER, '<remote_CPU_helper>', 'exec')
        assert '--phase' in C.HERE.joinpath('run_suite.py').read_text()
        results['remote_helper_compile_and_run_suite_interface_import_CPU_only'] = True
    receipt = dict(status='passed_CPU_event_completion_callback_checks', checks=results,
        source={name: entry(HERE/name) for name in ('completion.py', 'completion_checks.py')},
        fixture_scope='Synthetic CPU-only artifacts below operator_checks/completion/fixtures; none are real training or quality results.',
        seconds=time.perf_counter()-started,
        cost=dict(formal_updates=0, Adam_calls=0, RGB_renderer_forwards=0, moment_renderer_forwards=0, GPU_calls=0,
            seconds=time.perf_counter()-started), SSH_calls=0, services_started=0,
        live_remote_DBus_RefUnit_and_rsync_not_executed=True)
    out.mkdir(parents=True, exist_ok=True); path=out/('receipt_'+str(time.time_ns())+'.json'); write(path,receipt)
    print(json.dumps(dict(status=receipt['status'], checks=len(results), receipt=entry(path)),ensure_ascii=False))
    return receipt


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=OUT/'operator_checks/completion')
    checks(parser.parse_args().out)
