"""Install a private CUDA 12.8 compiler from NVIDIA's hashed redistributions."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

ROOT = Path('/home/ubuntu/3DGS/4dsr')
BASE = 'https://developer.download.nvidia.com/compute/cuda/redist/'
CACHE = ROOT / 'tools/cuda_downloads'
CACHE.mkdir(parents=True, exist_ok=True)
spec = CACHE / 'redistrib_12.8.1.json'
subprocess.run(['curl', '-fLsS', '--retry', '3', '--max-time', '180', BASE + spec.name, '-o', str(spec)], check=True)
meta = json.loads(spec.read_text())
dest = ROOT / 'tools/cuda-12.8-minimal'
dest.mkdir(exist_ok=True)
receipts = []
for name in ['cuda_nvcc', 'cuda_cudart', 'cuda_cccl']:
    entry = meta[name]['linux-x86_64']
    archive = CACHE / Path(entry['relative_path']).name
    subprocess.run(['curl', '-fLsS', '--retry', '3', '--max-time', '1800', BASE + entry['relative_path'], '-o', str(archive)], check=True)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    assert digest == entry['sha256'], (name, digest)
    unpacked = CACHE / (name + '_unpacked')
    unpacked.mkdir(exist_ok=True)
    with tarfile.open(archive) as tf:
        # Archives come from NVIDIA and are verified against their manifest.
        for member in tf.getmembers():
            assert not member.name.startswith('/') and '..' not in Path(member.name).parts
        tf.extractall(unpacked)
    source = next(p for p in unpacked.iterdir() if p.is_dir())
    shutil.copytree(source, dest, dirs_exist_ok=True, symlinks=True)
    receipts.append(dict(component=name, version=meta[name]['version'], **entry))
    print(name, meta[name]['version'], 'installed', flush=True)
if not (dest / 'lib64').exists():
    (dest / 'lib64').symlink_to('lib')
(ROOT / 'deployment/a100_20260920/cuda_components.json').write_text(json.dumps(receipts, indent=2))
subprocess.run([str(dest / 'bin/nvcc'), '--version'], check=True)
