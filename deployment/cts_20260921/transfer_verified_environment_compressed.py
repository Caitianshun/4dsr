"""Reconstruct A100's general-SR runtime using byte-identical local files first.

Run locally, after a fresh cts venv and the A100 source SHA256 inventory exist.
Only hashes that match the source inventory are reused; remaining files and
symlinks are transferred from A100. No package outside that inventory is added.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
DEPLOY = ROOT / 'deployment/cts_20260921'
V = DEPLOY / 'verification'
REMOTE = '/home/cts/Project/4DSR'
PREFIX = '.venv/lib/python3.10/site-packages/'

def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()

def run(cmd, **kw):
    print('RUN', cmd, flush=True)
    subprocess.run(cmd, check=True, **kw)

def main():
    state = dict(status='running', started_unix=time.time(), local_reuse_files=0,
                 local_reuse_bytes=0, remote_files=0, remote_bytes=None)
    try:
        matches = {'site': [], 'cuda': [], 'cache': []}
        roots = {'site': Path('/home/cai_tianshun/Project/4dgs/.venv/lib/python3.1/site-packages'),
                 'cuda': Path('/usr/local/cuda-12.8'),
                 'cache': Path('/home/cai_tianshun/.cache/torch')}
        targets = {'site': PREFIX.rstrip('/'), 'cuda': 'tools/cuda-12.8-minimal', 'cache': '.cache/torch'}
        missing = []
        for line in (V/'a100_environment_source.sha256').read_text().splitlines():
            digest, rel = line.split('  ', 1)
            if rel.startswith(PREFIX):
                group, sub = 'site', rel[len(PREFIX):]
            elif rel.startswith('tools/cuda-12.8-minimal/'):
                group, sub = 'cuda', rel[len('tools/cuda-12.8-minimal/'):]
            elif rel.startswith('.cache/torch/'):
                group, sub = 'cache', rel[len('.cache/torch/'):]
            else:
                raise ValueError(rel)
            local = roots[group] / sub
            if local.is_file() and sha(local) == digest:
                matches[group].append(sub)
                state['local_reuse_files'] += 1
                state['local_reuse_bytes'] += local.stat().st_size
            else:
                missing.append(rel)
        state['remote_files'] = len(missing)
        (V/'environment_reuse.json').write_text(json.dumps(state, indent=2))
        print(json.dumps(state), flush=True)
        for group, entries in matches.items():
            filelist = V / f'environment_local_{group}.txt'
            filelist.write_text('\n'.join(entries)+'\n')
            run(['ssh', 'cts', f'mkdir -p {REMOTE}/{targets[group]}'])
            # Follow local symlinks: the inventory identifies regular file bytes.
            run(['rsync', '-aLz', '--compress-choice=zstd', '--compress-level=3', '--partial', '--stats', '--files-from='+str(filelist), str(roots[group])+'/',
                 f'cts:{REMOTE}/{targets[group]}/'])
        links = [line.split('\t',1)[0] for line in (V/'a100_environment_symlinks.tsv').read_text().splitlines()]
        todo = '\n'.join(missing + links) + '\n'
        (V/'environment_from_a100.txt').write_text(todo)
        source = subprocess.Popen(['ssh', 'a100-train',
            'cd /home/ubuntu/3DGS/4dsr && tar -czf - --no-recursion -T -'],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        destination = subprocess.Popen(['ssh', 'cts', f'cd {REMOTE} && tar -xzf -'], stdin=source.stdout)
        source.stdout.close()
        source.stdin.write(todo.encode()); source.stdin.close()
        src_rc = source.wait(); dst_rc = destination.wait()
        if src_rc or dst_rc:
            raise RuntimeError(f'transfer exit codes: {src_rc}, {dst_rc}')
        run(['rsync', '-a', str(V/'a100_environment_source.sha256'),
             f'cts:{REMOTE}/deployment/cts_20260921/verification/'])
        run(['ssh', 'cts', f'cd {REMOTE} && sha256sum --quiet -c deployment/cts_20260921/verification/a100_environment_source.sha256'])
        state.update(status='complete', all_source_regular_files_verified=True, symlinks=len(links))
    except BaseException as exc:
        state.update(status='failed', error=repr(exc))
        raise
    finally:
        state['finished_unix'] = time.time()
        (V/'environment_reuse.json').write_text(json.dumps(state, indent=2))

if __name__ == '__main__':
    main()
