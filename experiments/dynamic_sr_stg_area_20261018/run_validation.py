"""Run registered smoke and formal STG plans under one external broker lock.

Wait for data and GPU availability outside this process. A failed stage is
terminal. No training state is resumed or retried, and no nested job is submitted.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def run(registration):
    config = json.loads(registration.read_text())
    plans = {}
    for mode in ('smoke', 'formal'):
        item = config[mode]
        path = ROOT / item['path']
        if sha(path) != item['sha256']:
            raise ValueError('Registered plan changed: ' + mode)
        plans[mode] = json.loads(path.read_text())
        if plans[mode]['mode'] != mode or plans[mode]['gpu_uuid'] != config['gpu_uuid']:
            raise ValueError('Plan mode or GPU differs from registration')
    verification = json.loads((ROOT / config['input_verification']).read_text())
    if verification['status'] != 'verified' or verification['receipt_sha256'] != sha(ROOT / config['input_inventory']):
        raise ValueError('Full input identity is not verified')
    if not os.environ.get('CLAUDE_REMOTE_JOB_DIR'):
        raise RuntimeError('External registered GPU broker is required')
    context = json.loads((Path(os.environ['CLAUDE_REMOTE_JOB_DIR']) / 'execution_context.json').read_text())
    if context['selected_gpu_uuids'] != [config['gpu_uuid']]:
        raise ValueError('Broker GPU differs from registration')
    out = ROOT / config['output']
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'started.json').open('x') as stream:
        json.dump(dict(registration_sha256=sha(registration), started_unix=time.time()), stream)
    state = dict(status='running', active='smoke', job_id=context['job_id'],
                 registration_sha256=sha(registration), gpu_uuid=config['gpu_uuid'])
    write(out / 'status.json', state)
    started = time.monotonic()
    try:
        for mode in ('smoke', 'formal'):
            state['active'] = mode
            write(out / 'status.json', state)
            subprocess.run([sys.executable, '-u', str(HERE/'run_pipeline.py'), '--plan',
                            str(ROOT/config[mode]['path'])], cwd=ROOT, check=True)
            result = json.loads((ROOT/plans[mode]['output']/'status.json').read_text())
            if result['status'] != 'completed' or result['plan_sha256'] != config[mode]['sha256']:
                raise RuntimeError('Registered stage did not complete: ' + mode)
            if mode == 'smoke':
                expected = {(item['name'], item['relative_output']): item for item in plans[mode]['stages']}
                actual = [(item['name'], item['relative_output']) for item in result['completed']]
                if (len(actual) != 3 or len(set(actual)) != 3 or set(actual) != set(expected)
                        or {item['method'] for item in expected.values()} != {'areaL0', 'areaTm', 'areaOracle'}):
                    raise RuntimeError('Three-arm smoke is incomplete')
                sample_hashes = set()
                for item in result['completed']:
                    stage = expected[(item['name'], item['relative_output'])]
                    stage_out = ROOT/plans[mode]['output']/item['relative_output']
                    receipt = json.loads((stage_out/'train/complete.json').read_text())
                    if (receipt['iterations'] != 40 or receipt['checkpoint_reload_check']['status'] != 'passed'
                            or receipt['method'] != stage['method'] or receipt['segment_start'] != 0):
                        raise RuntimeError('Native checkpoint save-load validation did not pass')
                    evaluation = json.loads((stage_out/'eval/summary.json').read_text())
                    if (evaluation['status'] != 'incomplete' or evaluation['frame_indices'] != [0, 1, 49]
                            or evaluation['method'] != stage['method']
                            or evaluation['segment_start'] != 0
                            or not evaluation['test_only_engineering_smoke']
                            or evaluation['lpips_status'] != 'explicitly_skipped_smoke_only'):
                        raise RuntimeError('Unexpected engineering smoke evaluation protocol')
                    sample_hashes.add(receipt['sampling_sha256'])
                if len(sample_hashes) != 1:
                    raise RuntimeError('Paired smoke sampling differs')
                # All CUDA children have exited. A foreign process must stop
                # this job; never evict it or weaken broker admission rules.
                query = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name',
                                        '--format=csv,noheader,nounits'], capture_output=True, text=True,
                                       check=True, timeout=15)
                rows, seen = [], set()
                for row in csv.reader(query.stdout.splitlines()):
                    if (len(row) != 3 or not re.fullmatch(r'GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}', row[0].strip())
                            or not row[1].strip().isdigit() or int(row[1].strip()) <= 0 or not row[2].strip()
                            or (row[0].strip(), int(row[1].strip())) in seen):
                        raise RuntimeError('Unrecognised GPU occupancy output')
                    seen.add((row[0].strip(), int(row[1].strip())))
                    rows.append(row)
                foreign = [row for row in rows if row[0].strip() == config['gpu_uuid']]
                if foreign:
                    raise RuntimeError('GPU occupancy changed after smoke: ' + json.dumps(foreign))
        state.update(status='completed', active=None, seconds=time.monotonic()-started,
                     scientific_status='requires_complete_300_frame_aggregation')
    except BaseException as error:
        state.update(status='failed', error_type=type(error).__name__, error=str(error),
                     traceback=traceback.format_exc(), seconds=time.monotonic()-started)
        write(out/'status.json', state)
        raise
    write(out/'status.json', state)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration', type=Path, required=True)
    run(parser.parse_args().registration.resolve())
