"""One-GPU STG pipeline; process exits immediately trigger evaluation.

Launch only inside the registered remote GPU broker. This CPU supervisor never
starts simultaneous CUDA children, retries a failed training run, or resumes
from STG's incomplete optimizer checkpoint. Successful stage events are files
that the inotify collector can receive without model polling.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def write(p, value):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(p)


def execute(cmd, log, env, deadline):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError('Registered pipeline time budget exhausted')
    with Path(log).open('xb') as f:
        child = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
        try:
            rc = child.wait(timeout=remaining)
        except BaseException:
            child.terminate()
            try:
                child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
            raise
    if rc:
        raise RuntimeError(f'Child exit {rc}: {log}')


def run(plan_path):
    plan = json.loads(plan_path.read_text())
    if plan['host'] != socket.gethostname():
        raise ValueError('Plan host differs from current host')
    context = os.environ.get('CLAUDE_REMOTE_JOB_DIR')
    if not context:
        raise RuntimeError('Pipeline requires existing registered remote GPU broker')
    ctx = json.loads((Path(context) / 'execution_context.json').read_text())
    if ctx['selected_gpu_uuids'] != [plan['gpu_uuid']]:
        raise ValueError('Broker GPU identity differs from plan')
    for item in plan['source_files']:
        if sha(ROOT / item['path']) != item['sha256']:
            raise ValueError('Frozen executable source changed: ' + item['path'])
    receipt = json.loads((ROOT / plan['input_verification']).read_text())
    if receipt['status'] != 'verified':
        raise ValueError('Remote input identity verification did not pass')
    out = ROOT / plan['output']
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'status.json').exists():
        raise FileExistsError('This pipeline output already has an execution record; no replay')
    env = dict(os.environ)
    env.update(PYTHONPATH=os.pathsep.join(str(ROOT / p) for p in plan['python_paths']),
               TORCH_HOME=str(ROOT / plan['torch_home']), PYTHONNOUSERSITE='1',
               OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4')
    env['CUDA_HOME'] = str(ROOT / plan['cuda_home'])
    env['LD_LIBRARY_PATH'] = str(ROOT / plan['cuda_home'] / 'lib64')
    begun = time.monotonic()
    deadline = begun + plan['max_seconds']
    state = dict(schema='stg_pipeline_v1', status='running', mode=plan['mode'],
                 plan_sha256=sha(plan_path), host=socket.gethostname(), gpu_uuid=plan['gpu_uuid'],
                 broker_job=ctx['job_id'], completed=[], active=None, started_unix=time.time())
    write(out / 'status.json', state)
    try:
        sampling_by_segment = {}
        for stage in plan['stages']:
            if shutil.disk_usage(ROOT).free < plan.get('minimum_free_bytes', 12 << 30):
                raise RuntimeError('Insufficient free disk for the next stage')
            spec_path = ROOT / stage['spec']
            if sha(spec_path) != stage['spec_sha256']:
                raise ValueError('Registered stage spec changed')
            stage_out = out / stage['relative_output']
            stage_out.mkdir(parents=True, exist_ok=False)
            state['active'] = stage['name']
            write(out / 'status.json', state)
            stage_started = time.monotonic()
            train = [sys.executable, '-u', str(HERE / 'train.py'), '--spec', str(spec_path),
                     '--out', str(stage_out / 'train'), '--iterations', str(stage['iterations'])]
            execute(train, stage_out / 'train.log', env, min(deadline, stage_started + plan['max_stage_seconds']))
            ck = stage_out / 'train/point_cloud' / ('iteration_' + str(stage['iterations'])) / 'point_cloud.ply'
            if not ck.is_file() or not ck.with_suffix('.pt').is_file():
                raise RuntimeError('Training returned without complete PLY and decoder')
            training_receipt = json.loads((stage_out / 'train/complete.json').read_text())
            key = (stage['segment_start'], stage['iterations'])
            sample_hash = training_receipt['sampling_sha256']
            if key in sampling_by_segment and sampling_by_segment[key] != sample_hash:
                raise ValueError('Three-arm same-segment sampling identity diverged')
            sampling_by_segment[key] = sample_hash
            evaluate = [sys.executable, '-u', str(HERE / 'evaluate.py'), '--spec', str(spec_path),
                        '--checkpoint', str(ck), '--out', str(stage_out / 'eval'),
                        '--training-receipt', str(stage_out / 'train/complete.json')]
            if plan['mode'] == 'smoke':
                start = stage['segment_start']
                evaluate += ['--frames', str(start), str(start+1), str(start+49), '--skip-lpips', '--test-only']
            execute(evaluate, stage_out / 'eval.log', env, deadline)
            result = json.loads((stage_out / 'eval/summary.json').read_text())
            if result.get('status') == 'failed':
                raise RuntimeError('Evaluator reported failure')
            if plan['mode'] == 'formal' and result.get('status') != 'completed':
                raise RuntimeError('Formal evaluation is incomplete; do not continue')
            done = dict(name=stage['name'], relative_output=stage['relative_output'],
                        seconds=time.monotonic()-stage_started, checkpoint_sha256=sha(ck),
                        decoder_sha256=sha(ck.with_suffix('.pt')), evaluation_status=result['status'])
            state['completed'].append(done)
            state['active'] = None
            write(out / 'status.json', state)
            write(out / 'events' / (stage['name'] + '.json'), done)
        state.update(status='completed', seconds=time.monotonic()-begun, active=None,
                     scientific_status='engineering_smoke_only' if plan['mode']=='smoke' else 'requires_complete_300_frame_aggregation')
        write(out / 'status.json', state)
        write(out / 'events/terminal.json', state)
    except BaseException as e:
        state.update(status='failed', error_type=type(e).__name__, error=str(e),
                     traceback=traceback.format_exc(), seconds=time.monotonic()-begun)
        write(out / 'status.json', state)
        write(out / 'events/terminal.json', state)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    run(args.plan)
