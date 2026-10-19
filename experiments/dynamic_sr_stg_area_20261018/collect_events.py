"""Collect STG stage artifacts on remote inotify completion events.

Only this run's output is copied. Event replay after reconnect is idempotent.
There is no model polling; the hourly chat heartbeat is an exception fallback.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
HOSTS = {'a100-train': '/home/ubuntu/3DGS/4dsr', 'cts': '/home/cts/Project/4DSR'}


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def relative(value):
    p = PurePosixPath(value)
    if p.is_absolute() or '..' in p.parts or not p.parts:
        raise ValueError('Invalid event path')
    return p


def collect(host, run):
    run = relative(run)
    if run.parts[:2] != ('output', 'dynamic_sr_stg_area_20261018'):
        raise ValueError('Collector is restricted to this STG experiment output')
    remote_root = HOSTS[host]
    out = ROOT / str(run)
    out.mkdir(parents=True, exist_ok=True)
    log = (out / 'collection.jsonl').open('a', buffering=1)
    ssh = ['ssh', '-o', 'BatchMode=yes', '-o', 'ServerAliveInterval=30', '-o', 'ServerAliveCountMax=3', host]
    command = ['/usr/bin/python3', remote_root + '/experiments/dynamic_sr_stg_area_20261018/watch_events.py',
               remote_root + '/' + str(run) + '/events']
    def sync(subpath):
        target = out / str(subpath)
        target.mkdir(parents=True, exist_ok=True)
        src = host + ':' + remote_root + '/' + str(run / subpath) + '/'
        subprocess.run(['rsync', '-a', '--checksum', '--timeout=120', '-e',
                        'ssh -o BatchMode=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3',
                        src, str(target) + '/'], check=True)
        return target
    for attempt in range(48):
        proc = subprocess.Popen(ssh + [shlex.join(command)], stdout=subprocess.PIPE, text=True)
        try:
            for line in proc.stdout:
                event = json.loads(line)
                value = event['value']
                if event['event'] == 'terminal.json':
                    # Own run only. Includes failed-stage logs and any partial checkpoint.
                    subprocess.run(['rsync', '-a', '--checksum', '--exclude=collection.jsonl', '--timeout=120',
                        '-e', 'ssh -o BatchMode=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3',
                        host + ':' + remote_root + '/' + str(run) + '/', str(out) + '/'], check=True)
                    log.write(json.dumps(dict(event='terminal_collected', remote_state=value['status'], time=time.time()))+'\n')
                    return
                subpath = relative(value['relative_output'])
                target = sync(subpath)
                checkpoints = list((target / 'train/point_cloud').glob('iteration_*/point_cloud.ply'))
                if len(checkpoints) != 1 or sha(checkpoints[0]) != value['checkpoint_sha256'] or sha(checkpoints[0].with_suffix('.pt')) != value['decoder_sha256']:
                    raise ValueError('Collected native checkpoint/decoder identity mismatch')
                log.write(json.dumps(dict(event='stage_collected', stage=value['name'], local=str(target),
                    checkpoint_sha256=value['checkpoint_sha256'], decoder_sha256=value['decoder_sha256'],
                    time=time.time()))+'\n')
            raise ConnectionError('Remote event stream ended before terminal event')
        except Exception as error:
            log.write(json.dumps(dict(event='collector_retry', attempt=attempt, error=str(error), time=time.time()))+'\n')
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        time.sleep(60)
    raise RuntimeError('Collector reconnect budget exhausted; inspect remote status without replaying training')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--host', choices=HOSTS, required=True)
    p.add_argument('--run', required=True)
    a = p.parse_args()
    collect(a.host, a.run)
