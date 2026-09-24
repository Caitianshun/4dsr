"""One SR4D trajectory, two fixed endpoints, existing Wu checkpoints only.

No Wu training, teacher generation, or input preparation. Training return starts
the fixed evaluation chain immediately; batch exit emits the collection event.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = Path(__file__).resolve().parent


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(4 << 20), b''):
            h.update(b)
    return h.hexdigest()


def write(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--scene', required=True, choices=['cook_spinach', 'meetroom_discussion'])
    ap.add_argument('--gpu', required=True)
    ap.add_argument('--sr4d-root', type=Path, required=True)
    ap.add_argument('--sr4d-python', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    assets_path = ROOT / 'deployment/sr4d_20260922/minimal_assets.json'
    assets = json.loads(assets_path.read_text())['scenes'][a.scene]
    for rel, expected in assets['files'].items():
        if sha(ROOT / rel) != expected:
            raise RuntimeError('Frozen comparison asset mismatch: ' + rel)
    manifest = ROOT / assets['manifest_path']
    sr = a.sr4d_python.expanduser().absolute()  # Keep venv symlink unresolved.
    wu = ROOT / '.venv/bin/python'
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    logs = out / 'logs'; logs.mkdir()
    snapshot = out / 'source_snapshot'; snapshot.mkdir()
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=a.gpu, OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
               FOURDSR_UPSTREAM=str(ROOT / 'vendor/4dgs'), TORCH_HOME=str(ROOT / '.cache/torch'),
               MPLBACKEND='Agg', PYTHONFAULTHANDLER='1')
    state = dict(status='running', started_at=now(), pid=os.getpid(), scene=a.scene, gpu=a.gpu,
                 manifest_path=assets['manifest_path'], manifest_sha256=sha(manifest),
                 protocol='existing_bicubic_minimal_v1', steps=[], sources={},
                 new_training='Only SR4D coarse 20000 + fine 18000; one trajectory',
                 evaluations='SR4D fine 6000/18000; existing Wu SR weight0.1 6000/18000',
                 assets=assets, smoke=False)
    for p in SCRIPTS.glob('*.py'):
        shutil.copy2(p, snapshot / p.name); state['sources'][p.name] = sha(p)
    shutil.copy2(assets_path, snapshot / assets_path.name)
    protocol = ROOT / 'docs/sr4d_comparison_protocol_2026-09-22.md'
    shutil.copy2(protocol, snapshot / protocol.name)
    state['protocol_sha256'] = sha(protocol)
    write(out / 'status.json', state)

    def run(name, command):
        record = dict(name=name, command=list(map(str, command)), status='running',
                      started_at=now(), log=str(logs / (name + '.log')))
        state['steps'].append(record); state['current'] = name
        write(out / 'status.json', state)
        started = time.monotonic()
        with open(record['log'], 'x') as log:
            child = subprocess.Popen(record['command'], cwd=ROOT, env=env,
                                     stdout=log, stderr=subprocess.STDOUT)
            record['pid'] = child.pid; write(out / 'status.json', state)
            rc = child.wait()
        record.update(status='complete' if rc == 0 else 'failed', returncode=rc,
                      seconds=time.monotonic() - started, finished_at=now())
        write(out / 'status.json', state)
        if rc:
            raise RuntimeError(name + ' failed; see ' + record['log'])

    try:
        srout = out / 'sr4d'
        run('sr4d_train', [sr, SCRIPTS / 'train_sr4d.py', '--upstream', a.sr4d_root,
            '--manifest', manifest, '--output', srout, '--coarse-steps', '20000',
            '--fine-steps', '18000', '--save-iterations', '6000', '18000'])
        for method in ['sr4d', 'wu']:
            for iteration in [6000, 18000]:
                name = f'{method}_{iteration}_evaluation'
                command = [sr if method == 'sr4d' else wu, SCRIPTS / 'evaluate_comparison.py',
                    '--method', method, '--manifest', manifest, '--out', out / name,
                    '--prior-cameras', assets['prior_cameras'], '--input-protocol', 'existing-bicubic']
                if method == 'sr4d':
                    command += ['--upstream', a.sr4d_root, '--run', srout,
                                '--stage', 'fine', '--iteration', str(iteration)]
                else:
                    command += ['--upstream', ROOT / 'vendor/4dgs', '--checkpoint',
                                ROOT / assets['wu_run'] / f'checkpoint_{iteration}.pt']
                run(name, command)
        state.update(status='complete', current=None, finished_at=now())
        write(out / 'complete.json', dict(status='complete', scene=a.scene,
              finished_at=now(), steps=len(state['steps']), new_training_runs=1))
    except BaseException:
        state.update(status='failed', finished_at=now(), traceback=traceback.format_exc())
        raise
    finally:
        write(out / 'status.json', state)
        event = dict(status=state['status'], finished_at=now(), out=str(out))
        write(out / 'completion_event.json', event)
        fifo = out.parent / (out.name + '.completion.fifo')
        if fifo.exists():
            try:
                fd = os.open(fifo, os.O_WRONLY | os.O_NONBLOCK)
                os.write(fd, (json.dumps(event) + '\n').encode()); os.close(fd)
            except OSError:
                pass


if __name__ == '__main__':
    main()
