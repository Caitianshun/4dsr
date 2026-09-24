"""Resume completed models after a dispatcher-only compatibility failure.

No training and no pidfd dependency. Preserve the original failed receipts.
"""
import json, os, signal, time, traceback
from datetime import datetime, timezone
from pathlib import Path
from resume_parallel_meeting import OUT, GPU, evaluate, write


def main():
    start = time.monotonic()
    previous = json.loads((OUT / 'status.json').read_text())
    assert previous['status'] == 'failed'
    assert "no attribute 'pidfd_open'" in previous['traceback']
    write(OUT / 'status_parallel_failure_snapshot.json', previous)
    handoff = json.loads((OUT / 'parallel_handoff.json').read_text())
    for branch in 'ABCD':
        done = json.loads((OUT / branch / 'complete.json').read_text())
        assert done['parameter_updates'] == 6000 and done['source_unchanged']
    # The C trainer is already a zombie; stop only its paused old dispatcher.
    child = int(next(iter(handoff['children'])))
    proc = Path(f'/proc/{child}/stat')
    if proc.exists():
        assert proc.read_text().split(') ', 1)[1].split()[0] == 'Z'
    serial = handoff['serial_pid']
    cmdline = Path(f'/proc/{serial}/cmdline')
    if cmdline.exists():
        assert b'run_meeting_controls.py' in cmdline.read_bytes()
        os.kill(serial, signal.SIGKILL)
    state = dict(status='evaluation_recovery_running', pid=os.getpid(),
                 started_utc=datetime.now(timezone.utc).isoformat(),
                 reason='runtime Python lacks os.pidfd_open; all four training runs completed',
                 retraining=False, interrupted_training=False, final_evaluation_gpu=GPU,
                 prior_failure='status_parallel_failure_snapshot.json')
    write(OUT / 'recovery.json', state)
    write(OUT / 'status.json', state)
    try:
        evaluate('C', GPU)
        evaluate('D', GPU)
        draws, rows = [], []
        for branch in 'ABCD':
            folder = OUT / branch
            done = json.loads((folder / 'complete.json').read_text())
            draws.append(done['draw_sha256'])
            for step in [1200, 3000, 6000]:
                m = json.loads((folder / f'eval_{step}/metrics.json').read_text())
                assert len(m['rows']) == 60
                assert m['frame_indices'] == list(range(0, 120, 2))
                assert len(list((folder / f'eval_{step}/predictions').glob('*.png'))) == 60
                assert (folder / f'own_target_fit_{step}.json').exists()
                rows.append(dict(branch=branch, step=step, aggregate=m['aggregate'],
                                 temporal=m['temporal_aggregate'], checkpoint_sha256=m['checkpoint_sha256'],
                                 evaluation_seconds=m['elapsed_seconds']))
        assert len(set(draws)) == 1
        write(OUT / 'summary.json', dict(rows=rows, all_draws_equal=True,
              source='fixed endpoints, unified GPU0 evaluation; dispatcher failure recovered without retraining',
              training_hardware='A/B/C RTX PRO6000; D RTX3090; unified evaluation does not remove training hardware effects'))
        state.update(status='completed_and_evaluated', finished_utc=datetime.now(timezone.utc).isoformat(),
                     elapsed_seconds=time.monotonic()-start, all_draws_equal=True)
        write(OUT / 'recovery.json', state)
        write(OUT / 'status.json', state)
        write(OUT / 'complete.json', state)
    except BaseException:
        state.update(status='failed', traceback=traceback.format_exc())
        write(OUT / 'recovery.json', state)
        write(OUT / 'status.json', state)
        raise


if __name__ == '__main__':
    main()
