"""Persistent CTS CPU-only orchestration of the exact five CUDA extensions.

Does not deploy binaries or use a GPU. Each existing builder owns an isolated,
fresh output. The controller performs resource checks and deployment separately.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
CUDA = Path('/home/cts/Project/4DSR/tools/cuda-12.8-minimal')


def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):value.update(block)
    return value.hexdigest()


def save(path, value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n')
    temporary.replace(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',required=True,type=Path)
    args=parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CPU service must explicitly hide all GPUs')
    if ROOT != Path('/home/cts/Project/4DSR/tasks/static_sr_wu_20261022'):
        raise ValueError('This CPU service is scoped to the isolated CTS task snapshot')
    output=(ROOT/args.out).resolve()
    if not output.is_relative_to(ROOT):raise ValueError('Output escapes task snapshot')
    output.mkdir(parents=True,exist_ok=False)
    base=ROOT/'experiments/static_sr_wu_20261022'
    commands=[dict(name='native',argv=[sys.executable,'-u',str(base/'cts_native_build.py'),'build',
        '--cuda-home',str(CUDA),'--cc','gcc','--cxx','g++','--jobs','2']),
        dict(name='methods',argv=[sys.executable,'-u',str(base/'cts_method_build.py'),'build',
        '--plan',str(ROOT/'output/static_sr_wu_20261022/runtime/cts_method_build_plan_v1.json'),
        '--cuda-home',str(CUDA),'--cc','gcc','--cxx','g++','--jobs','2'])]
    record=dict(schema='static_sr_CTS_five_extension_CPU_service_v1',status='running',host=socket.gethostname(),
        root=str(ROOT),script_sha256=sha(__file__),python=sys.executable,GPU_executed=False,
        binary_deployment_performed=False,started_utc=datetime.now(timezone.utc).isoformat(),
        environment={k:os.environ.get(k) for k in ['CUDA_VISIBLE_DEVICES','LD_LIBRARY_PATH','PYTHONDONTWRITEBYTECODE']},stages=[])
    started=time.perf_counter();save(output/'status.json',record)
    try:
        for item in commands:
            print('START CPU BUILD '+item['name'],flush=True)
            before=time.perf_counter();log=output/(item['name']+'.log')
            with log.open('x') as stream:
                result=subprocess.run(item['argv'],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
            stage=dict(**item,returncode=result.returncode,elapsed_seconds=time.perf_counter()-before,
                       log=str(log.relative_to(ROOT)),log_sha256=sha(log))
            record['stages'].append(stage);save(output/'status.json',record)
            print('END CPU BUILD '+item['name']+' exit='+str(result.returncode),flush=True)
            if result.returncode:raise RuntimeError('CPU build failed: '+item['name'])
        record['status']='five_extensions_CPU_build_import_passed_GPU_pending'
    except BaseException:
        record.update(status='failed',error=traceback.format_exc())
        raise
    finally:
        record.update(elapsed_seconds=time.perf_counter()-started,finished_utc=datetime.now(timezone.utc).isoformat())
        save(output/'status.json',record)
        print(json.dumps(dict(status=record['status'],receipt=str(output/'status.json'))),flush=True)


if __name__=='__main__':main()
