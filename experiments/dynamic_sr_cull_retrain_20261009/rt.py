"""Runtime/path setup shared by the 2026-10-09 culling retrain entry.

The claude-training service mounts only this project and one env, so the baseline
4DGaussians runtime is loaded from the byte-identical copy in third_party/.
"""
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT / 'third_party/4dgs_843d5ac_runtime'
EXP = ROOT / 'experiments/dynamic_sr_multiview_footprint_20261007'
LEGACY = ROOT / 'experiments/dynamic_sr_20260918'
DATA = ROOT / 'data/dynamic_sr/full_time_20261007/prepared'
BASE = ROOT / 'output/dynamic_sr_multiview_footprint_20261007'

for p in (str(Path(__file__).resolve().parent), str(LEGACY), str(EXP), str(RUNTIME)):
    if p in sys.path:
        sys.path.remove(p)
    sys.path.insert(0, p)
os.environ.setdefault('TORCH_HOME', str(RUNTIME / 'torch_hub'))


def check_runtime():
    """Fail unless the rasterizer/knn/model code resolve to the pinned copy."""
    import diff_gaussian_rasterization
    import simple_knn._C
    import scene.gaussian_model
    import torch
    torch.hub.set_dir(str(RUNTIME / 'torch_hub'))
    for mod in (diff_gaussian_rasterization, simple_knn._C, scene.gaussian_model):
        if not Path(mod.__file__).resolve().is_relative_to(RUNTIME):
            raise RuntimeError(f'{mod.__name__} resolved outside the pinned runtime: {mod.__file__}')
    return dict(torch=torch.__version__, torch_git=torch.version.git_version, cuda=torch.version.cuda,
                rasterizer=str(Path(diff_gaussian_rasterization.__file__).relative_to(ROOT)))


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, indent=1, ensure_ascii=False, allow_nan=False))
    os.replace(tmp, path)
