"""Continue the independent B0 arm to its registered curve checkpoint.

This preparation-overlap worker does not need X support or calibration. It
does not select a quality endpoint, move a suffix to different hardware, or
add training updates. The normal suffix worker later resumes this checkpoint.
"""
import argparse
import fcntl
import socket
import time
import traceback
from types import SimpleNamespace
from fp_common import OUT, HERE, entry, read, write
import run_suite as suite


def main(a):
    assert a.operator_resource_resolved and a.gpu.startswith('GPU-')
    suite.plan()
    directory = OUT / 'runs/r1_B0'
    checkpoint = suite.continuation(directory)
    assert checkpoint is not None, 'An accepted first100 checkpoint is required'
    cursor = read(checkpoint.with_suffix('.json'))['metadata']['suffix_step']
    assert cursor == 100, 'Refuse an unregistered segment or duplicate work'
    initial = read(directory / 'attempt_complete_0000_0100.json')
    assert initial['physical_gpu'] == a.gpu and initial['updates'] == 100
    status = OUT / 'workers/B0_preparation_overlap.json'
    (OUT / 'locks').mkdir(parents=True, exist_ok=True)
    sources = suite.stage_files()
    with (OUT / 'locks' / f'{a.gpu}.controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        resource = suite.check_resource(a.gpu, status)
        assert resource['name'] == initial['gpu'], 'Same suffix hardware changed'
        identity = dict(host=socket.gethostname(), GPU_model=resource['name'],
                        physical_GPU=resource['uuid'], driver=resource['driver'],
                        python=a.python, protocol=entry(OUT / 'protocol.json'))
        platform_path = OUT / 'workers/suffix_1_platform.json'
        if platform_path.exists():
            assert read(platform_path) == identity
        else:
            write(platform_path, identity)
        suite.train(SimpleNamespace(repeat='1', python=a.python, gpu=a.gpu),
                    'B0', 3000, suite.environment(a.gpu), status)
        assert suite.stage_files() == sources, 'Active preparation sources changed'
        receipt = directory / 'attempt_complete_0100_3000.json'
        result = read(receipt)
        assert result['suffix_endpoint'] == 3000 and result['updates'] == 2900
        write(status, dict(status='completed_registered_B0_curve_checkpoint',
                           formal_suffix_updates=3000, added_updates=2900,
                           checkpoint=result['checkpoint'], segment=entry(receipt),
                           no_X_dependency=True, quality_not_used_for_selection=True,
                           next='normal suffix worker after X first100 timing'))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu', required=True)
    p.add_argument('--python', default=suite.PYTHON)
    p.add_argument('--operator-resource-resolved', action='store_true')
    a = p.parse_args()
    try:
        main(a)
    except suite.ResourceBusy:
        write(OUT / 'workers/B0_preparation_overlap.json',
              dict(status='waiting_for_free_GPU_no_work_launched',
                   traceback=traceback.format_exc()))
        raise SystemExit(75)
    except BaseException:
        write(OUT / 'workers' / f'B0_overlap_failure_{time.time_ns()}.json',
              dict(status='failed_preserved', traceback=traceback.format_exc()))
        raise
