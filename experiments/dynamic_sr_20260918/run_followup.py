"""Finish fixed diagnostic jobs after a scene's existing GPU suite exits.

Local process waiting does not invoke the language model. Result interpretation
is left to the scheduled three-hour check, not inferred from job success.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scene', required=True, choices=['cook_spinach', 'cut_roasted_beef'])
    p.add_argument('--suite-pid', required=True, type=int)
    p.add_argument('--checkpoint', required=True)
    a = p.parse_args()
    project = Path(__file__).resolve().parents[2]
    scripts = Path(__file__).parent
    root = project / 'output/dynamic_sr_20260918'
    manifest = project / 'data/dynamic_sr/n3dv_prepared' / a.scene / 'manifest.json'
    receipt = root / f'{a.scene}_pilot_v1_followup.json'
    if receipt.exists():
        raise FileExistsError(receipt)
    state = dict(scene=a.scene, status='waiting_for_suite', pid=os.getpid(),
                 suite_pid=a.suite_pid, gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
                 args=vars(a), events=[])

    def save():
        temporary = receipt.with_suffix('.tmp')
        temporary.write_text(json.dumps(state, indent=2))
        temporary.replace(receipt)

    def call(label, command):
        log = root / 'logs' / f'{a.scene}_pilot_v1_followup_{label}.log'
        state['status'] = label
        save()
        started = time.monotonic()
        with log.open('x') as f:
            subprocess.run(command, stdout=f, stderr=subprocess.STDOUT,
                           check=True, cwd=project)
        state['events'].append(dict(label=label, elapsed_s=time.monotonic()-started,
                                    log=str(log), command=command))
        save()

    save()
    try:
        sentinel = root / f'{a.scene}_pilot_v1_suite.json'
        started = time.monotonic()
        while not sentinel.exists():
            proc = Path(f'/proc/{a.suite_pid}/cmdline')
            command = proc.read_bytes().replace(b'\0', b' ').decode() if proc.exists() else ''
            if 'run_suite.py' not in command or a.scene not in command:
                raise RuntimeError('Original suite exited without success receipt; inspect its logs.')
            if time.monotonic() - started > 12 * 3600:
                raise TimeoutError('Original pilot exceeded 12 hours; no new GPU job started.')
            time.sleep(30)
        # Corrected diagnostic matches training: reduce raw radiance before clipping.
        call('sampling_linear', [sys.executable, str(scripts/'evaluate.py'),
             '--manifest', str(manifest), '--checkpoint', a.checkpoint,
             '--out', str(root/f'{a.scene}_pilot_v1_sampling_warmup_linear'), '--sampling-only'])
        # Extra-HR diagnostic: same LR-only initialization and iteration budget.
        # It is never a parent of any LR/SR branch or an admissible LR method.
        hr_out = root / f'{a.scene}_pilot_v1_hr_reference'
        call('hr_reference_train', [sys.executable, str(scripts/'run_experiment.py'),
             '--manifest', str(manifest), '--out', str(hr_out), '--task', 'warmup',
             '--observation', 'hr_reference', '--coarse-steps', '1000', '--fine-steps', '6000'])
        call('hr_reference_eval', [sys.executable, str(scripts/'evaluate.py'),
             '--manifest', str(manifest), '--checkpoint', str(hr_out/'checkpoint_final.pt'),
             '--out', str(hr_out/'evaluation')])
        state['status'] = 'complete'
        save()
    except Exception:
        state['status'] = 'failed'
        state['error'] = traceback.format_exc()
        save()
        raise


if __name__ == '__main__':
    main()
