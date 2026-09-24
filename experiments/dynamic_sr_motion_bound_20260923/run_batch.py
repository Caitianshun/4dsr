"""Two new refinement arms; reuse old A training and reevaluate all three uniformly."""
import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
import traceback


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2))
    temp.replace(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--selection', required=True)
    p.add_argument('--baseline', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--train-gpu', required=True)
    p.add_argument('--eval-gpu', required=True)
    args = p.parse_args()
    root = Path(__file__).resolve().parents[2]
    scripts = Path(__file__).resolve().parent
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    assert args.train_gpu != args.eval_gpu
    started = time.monotonic()
    state = dict(status='running', pid=os.getpid(), started_utc=datetime.now(timezone.utc).isoformat(),
                 train_gpu=args.train_gpu, eval_gpu=args.eval_gpu, steps=6000,
                 branches=['joint', 'ordinary_split', 'bound_split'],
                 note='A training reused; B/C trained on same GPU as A; all reevaluated on GPU1; no remote jobs.')
    write(out / 'status.json', state)
    jobs, errors = queue.Queue(), []

    def child(label, command, gpu):
        snapshot = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name,used_memory',
                                            '--format=csv,noheader'], text=True)
        for row in csv.reader(snapshot.splitlines()):
            if row[0].strip() == gpu and row[2].strip() not in ['/opt/todesk/bin/ToDesk_Session', '/usr/libexec/gnome-remote-desktop-daemon']:
                raise RuntimeError(f'GPU occupied: {row}')
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=gpu, CUDA_DEVICE_ORDER='PCI_BUS_ID', OMP_NUM_THREADS='4', MPLBACKEND='Agg')
        record = dict(status='running', command=command, gpu=gpu, gpu_before=snapshot,
                      started_utc=datetime.now(timezone.utc).isoformat())
        tick = time.monotonic()
        event = out / 'events' / f'{label}.json'
        with (out / f'{label}.log').open('w') as log:
            process = subprocess.Popen(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
            record['pid'] = process.pid
            write(event, record)
            rc = process.wait()
        record.update(status='completed' if rc == 0 else 'failed', returncode=rc, elapsed_s=time.monotonic()-tick)
        write(event, record)
        if rc:
            raise RuntimeError(f'{label} exited {rc}')

    def evaluate():
        while True:
            branch = jobs.get()
            if branch is None:
                return
            try:
                for split, step in [('dev', 1200), ('dev', 6000), ('test', 6000)]:
                    folder = out / branch
                    child(f'{branch}_eval_{split}_{step}', [sys.executable, str(scripts / 'evaluate.py'),
                          '--manifest', args.manifest, '--checkpoint', str(folder / f'checkpoint_{step}.pt'),
                          '--out', str(folder / f'eval_{split}_{step}'), '--split', split], args.eval_gpu)
            except BaseException:
                failure = dict(stage='evaluation', branch=branch, traceback=traceback.format_exc())
                errors.append(failure)
                write(out / f'failed_eval_{branch}.json', failure)

    baseline=Path(args.baseline).resolve()
    afolder=out/'joint'; afolder.mkdir()
    for name in ['config.json','complete.json','exposure.json','training.jsonl','checkpoint_1200.pt','checkpoint_6000.pt']:
        (afolder/name).symlink_to(baseline/name)
    write(afolder/'reuse.json',dict(training_reused=True,source=str(baseline),new_evaluation_only=True))
    jobs.put('joint')
    worker = threading.Thread(target=evaluate)
    worker.start()
    try:
        for branch in ['ordinary_split','bound_split']:
            child(f'{branch}_train', [sys.executable, str(scripts / 'train.py'), '--manifest', args.manifest,
                  '--checkpoint', args.checkpoint, '--selection', args.selection, '--out', str(out / branch), '--branch', branch,
                  '--steps', '6000', '--milestones', '1200,6000', '--seed', '20260923',
                  '--prior-cameras', 'cam02,cam04,cam08,cam12'], args.train_gpu)
            jobs.put(branch)
    except BaseException:
        failure = dict(stage='training', traceback=traceback.format_exc())
        errors.append(failure)
        write(out / 'failed_training.json', failure)
    finally:
        jobs.put(None)
        worker.join()
    if errors:
        state.update(status='failed', errors=errors, elapsed_s=time.monotonic()-started)
        write(out / 'status.json', state)
        raise RuntimeError(errors)
    child('existing_evidence', [sys.executable, str(scripts/'export_existing_examples.py'), '--out',
          str(out.parent/'evidence_existing'/'gpu_views_v1')], args.eval_gpu)
    draws = []
    for branch in state['branches']:
        folder = out / branch
        complete = json.loads((folder / 'complete.json').read_text())
        assert complete['parameter_updates'] == 6000 and not complete['smoke']
        draws.append(complete['draw_sha256'])
        for split, step in [('dev', 1200), ('dev', 6000), ('test', 6000)]:
            evaluation = json.loads((folder / f'eval_{split}_{step}' / 'complete.json').read_text())
            assert evaluation['status'] == 'completed_evaluation'
    assert len(set(draws)) == 1
    state.update(status='completed_and_evaluated', elapsed_s=time.monotonic()-started,
                 finished_utc=datetime.now(timezone.utc).isoformat(), draw_sha256=draws[0])
    write(out / 'status.json', state)
    write(out / 'complete.json', state)
    # Evaluation completion invokes the CPU summary; no timer discovers success.
    with (out / 'summary.log').open('w') as log:
        rc = subprocess.run([sys.executable, str(scripts / 'summarize.py'), '--root', str(out)],
                            cwd=root, stdout=log, stderr=subprocess.STDOUT).returncode
    if rc:
        write(out / 'summary_failed.json', dict(returncode=rc))
        raise RuntimeError('Training/evaluation completed, summary failed; see summary.log')


if __name__ == '__main__':
    main()
