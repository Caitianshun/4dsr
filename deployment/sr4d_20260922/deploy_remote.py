#!/usr/bin/env python3
"""Build SR4D in an independent cloned environment; never modify the base venv."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--base', required=True)
    ap.add_argument('--gpu', required=True)
    ap.add_argument('--arch', required=True)
    a = ap.parse_args()
    root, base = Path(a.root).resolve(), Path(a.base).resolve()
    out = root / 'remote_deployment'
    out.mkdir(exist_ok=True)
    envdir = root / '.venv_sr4d_20260922'
    python = envdir / 'bin/python'
    started = time.time()
    status = {'status': 'running', 'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
              'root': str(root), 'base': str(base), 'env': str(envdir),
              'gpu': a.gpu, 'arch': a.arch, 'steps': []}
    def save():
        (out / 'status.json').write_text(json.dumps(status, indent=2) + '\n')
    def run(cmd, name, env=None, cwd=None):
        print('RUN', name, cmd, flush=True)
        started_step = time.time()
        with (out / (name + '.log')).open('w') as log:
            p = subprocess.run(list(map(str, cmd)), cwd=cwd or root, env=env,
                               stdout=log, stderr=subprocess.STDOUT)
        status['steps'].append({'name': name, 'returncode': p.returncode, 'elapsed_s': time.time() - started_step})
        save()
        if p.returncode:
            print((out / (name + '.log')).read_text()[-12000:], flush=True)
            raise RuntimeError(f'{name} failed with {p.returncode}')
    identity_code = '''import torch, importlib, json, hashlib, sys
r={'python':sys.executable,'prefix':sys.prefix,'torch':torch.__version__,'modules':{}}
for n in ['diff_gaussian_rasterization','diff_gaussian_rasterization._C','simple_knn._C']:
 m=importlib.import_module(n); p=m.__file__;r['modules'][n]={'file':p,'sha256':hashlib.sha256(open(p,'rb').read()).hexdigest()}
print(json.dumps(r,indent=2))'''
    baseenv = os.environ.copy()
    baseenv['CUDA_VISIBLE_DEVICES'] = ''
    baseenv['PYTHONNOUSERSITE'] = '1'
    baseenv.pop('PYTHONPATH', None)
    def base_identity():
        return json.loads(subprocess.check_output([str(base / '.venv/bin/python'), '-c', identity_code],
                           env=baseenv, cwd='/tmp', text=True))
    before = base_identity()
    (out / 'original_environment_before.json').write_text(json.dumps(before, indent=2) + '\n')
    save()
    try:
        manifest = json.loads((root / 'source_manifest.json').read_text())
        bad = [f['path'] for f in manifest['files'] if digest(root / f['path']) != f['sha256']]
        if bad:
            raise RuntimeError(f'source hash mismatch: {bad[:8]}')
        (out / 'source_verified.json').write_text(json.dumps({'files': len(manifest['files']), 'manifest_sha256': digest(root / 'source_manifest.json'), 'bad': bad}, indent=2) + '\n')
        if envdir.exists():
            raise RuntimeError('Refusing to overwrite existing destination environment')
        run(['cp', '-a', '--reflink=auto', base / '.venv', envdir], 'clone_environment')
        # Copies have independent file contents (reflink COW or normal copy), no hard links.
        old, new = str(base / '.venv'), str(envdir)
        rewritten = []
        for path in list((envdir / 'bin').iterdir()) + [envdir / 'pyvenv.cfg']:
            if path.is_file() and not path.is_symlink():
                try:
                    text = path.read_text()
                except (UnicodeDecodeError, OSError):
                    continue
                if old in text:
                    path.write_text(text.replace(old, new))
                    rewritten.append(str(path.relative_to(envdir)))
        site = envdir / 'lib/python3.10/site-packages'
        quarantine = out / 'cloned_old_extensions_quarantine'
        quarantine.mkdir()
        removed = []
        pth_before = {}
        for path in site.glob('*.pth'):
            pth_before[path.name] = path.read_text()
        for path in list(site.iterdir()):
            if ('diff_gaussian_rasterization' in path.name or 'simple_knn' in path.name):
                removed.append(path.name)
                shutil.move(str(path), str(quarantine / path.name))
        (out / 'clone_relocation.json').write_text(json.dumps({'rewritten': rewritten, 'quarantined': removed, 'original_pth': pth_before}, indent=2) + '\n')
        env = baseenv.copy()
        env['CUDA_HOME'] = str(base / 'tools/cuda-12.8-minimal')
        env['PATH'] = str(envdir / 'bin') + ':' + env['CUDA_HOME'] + '/bin:' + env['PATH']
        env['LD_LIBRARY_PATH'] = str(base / 'tools/runtime-libs') + ':' + env['CUDA_HOME'] + '/lib64:' + env.get('LD_LIBRARY_PATH', '')
        env.update(TORCH_CUDA_ARCH_LIST=a.arch, MAX_JOBS='4', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MPLBACKEND='Agg', TORCH_HOME=str(base / '.cache/torch'))
        run([python, '-c', 'import sys; print(sys.executable);print(sys.prefix); assert sys.prefix == '+repr(str(envdir))], 'new_environment_identity', env)
        missing_code = '''import importlib.metadata as m,json,pathlib
r=[]
for line in pathlib.Path('deployment/requirements-runtime.txt').read_text().splitlines():
 if not line or line.startswith('#'):continue
 name=line.split('==')[0]
 try:m.version(name)
 except m.PackageNotFoundError:r.append(line)
print(json.dumps(r))'''
        missing = json.loads(subprocess.check_output([str(python), '-c', missing_code], env=env, cwd=root, text=True))
        (out / 'missing_requirements.json').write_text(json.dumps(missing, indent=2) + '\n')
        if missing:
            run([python, '-m', 'pip', 'install', '--disable-pip-version-check', *missing], 'install_missing_runtime', env)
        for extension in ['depth-diff-gaussian-rasterization', 'simple-knn']:
            run([python, '-m', 'pip', 'install', '--disable-pip-version-check', '--no-build-isolation', '--no-deps', root / 'submodules' / extension], 'build_' + extension, env)
        run([python, '-m', 'pip', 'check'], 'pip_check', env)
        run([python, '-m', 'pip', 'freeze'], 'requirements_frozen', env)
        run([python, '-c', identity_code], 'sr4d_extension_identity', env)
        run([python, '-c', 'import train, train_2, render; print("SR4D train/train_2/render imports passed")'], 'sr4d_import', env)
        run([python, 'deployment/test_tasa.py'], 'test_tasa_cpu', env)
        gpu_status = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total', '--format=csv,noheader,nounits'], text=True)
        apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name,used_memory', '--format=csv,noheader'], text=True)
        (out / 'gpu_before_smoke.txt').write_text(gpu_status + '\n' + apps)
        row = next(x.strip().split(', ') for x in gpu_status.splitlines() if x.strip().split(', ')[0] == a.gpu)
        if row[1] in apps or int(row[3]) > 5 or int(row[4]) > 2048:
            raise RuntimeError('Target GPU no longer idle; not starting smoke')
        env['CUDA_VISIBLE_DEVICES'] = a.gpu
        run([python, 'deployment/test_cuda_core.py', '--json-out', out / 'cuda_core.json'], 'test_cuda_core', env)
        (root / 'activate_sr4d.sh').write_text('''# Source this file before running SR4D; the original 4dsr environment is unchanged.
source "''' + str(envdir) + '''/bin/activate"
export SR4D_ROOT="''' + str(root) + '''"
export CUDA_HOME="''' + env['CUDA_HOME'] + '''"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="''' + env['LD_LIBRARY_PATH'] + '''"
export TORCH_CUDA_ARCH_LIST="''' + a.arch + '''"
export TORCH_HOME="''' + env['TORCH_HOME'] + '''"
export CUDA_VISIBLE_DEVICES="''' + a.gpu + '''"
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MAX_JOBS=4
export MPLBACKEND=Agg
unset PYTHONPATH
''')
        status['status'] = 'complete'
    except Exception:
        status['status'] = 'failed'
        status['error'] = traceback.format_exc()
        print(status['error'], flush=True)
    finally:
        after = base_identity()
        (out / 'original_environment_after.json').write_text(json.dumps(after, indent=2) + '\n')
        status['original_environment_unchanged'] = before == after
        if before != after:
            status['status'] = 'failed'
            status['identity_error'] = 'Original environment extensions changed'
        status['elapsed_s'] = time.time() - started
        status['finished_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        status['receipt_hashes'] = {p.name: digest(p) for p in out.iterdir() if p.is_file() and p.name != 'status.json'}
        save()
        print(json.dumps(status, indent=2), flush=True)
    return 0 if status['status'] == 'complete' else 1


if __name__ == '__main__':
    sys.exit(main())
