"""Verify pinned evaluator files and construct LPIPS offline on CPU; no forward."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    result = hashlib.sha256()
    with path.open('rb') as handle:
        for data in iter(lambda: handle.read(8 << 20), b''):
            result.update(data)
    return result.hexdigest()


def check_file(record):
    path = (ROOT / record['path']).resolve()
    if not path.is_relative_to(ROOT) or not path.is_file():
        raise ValueError('Missing/outside snapshot: ' + str(path))
    if path.stat().st_size != record['bytes'] or sha(path) != record['sha256']:
        raise ValueError('Evaluator identity mismatch: ' + str(path))
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(out)
    started = time.monotonic()
    receipt_path = ROOT / 'third_party/static_sr_eval_20261022/source_receipt.json'
    receipt = json.loads(receipt_path.read_text())
    if receipt['schema'] != 'static_sr_lpips_evaluator_assets_v1':
        raise ValueError('Unknown evaluator receipt schema')
    for item in receipt['files'] + receipt['models']:
        check_file(item)
    os.environ.update(CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
                      OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                      TORCH_HOME=str(ROOT / receipt['TORCH_HOME']))
    import torch
    import torchvision
    import urllib.request

    def blocked(*args, **kwargs):
        raise AssertionError('CPU evaluator acceptance forbids CUDA initialization and downloads')

    torch.cuda.init = blocked
    torch.cuda._lazy_init = blocked
    torch._C._cuda_init = blocked
    torch.hub.download_url_to_file = blocked
    urllib.request.urlopen = blocked
    sys.path.insert(0, str(ROOT / receipt['python_path']))
    import lpipsPyTorch
    from lpipsPyTorch import LPIPS
    if Path(lpipsPyTorch.__file__).resolve() != ROOT / receipt['python_path'] / 'lpipsPyTorch/__init__.py':
        raise ValueError('Wrong evaluator implementation imported')
    criterion = LPIPS(net_type='alex', version='0.1').eval()
    if any(value.device.type != 'cpu' for value in list(criterion.parameters()) + list(criterion.buffers())):
        raise ValueError('Evaluator model was not constructed on CPU')
    if torch.cuda.is_initialized():
        raise ValueError('CUDA initialization detected')
    result = dict(schema='static_sr_lpips_evaluator_cpu_acceptance_v1',
                  status='offline_cpu_constructor_passed',
                  source_receipt=dict(path=str(receipt_path.relative_to(ROOT)), sha256=sha(receipt_path)),
                  python=sys.executable, torch=torch.__version__, torchvision=torchvision.__version__,
                  TORCH_HOME=os.environ['TORCH_HOME'], imported_source=lpipsPyTorch.__file__,
                  weight_loads='AlexNet feature backbone and LPIPS v0.1 learned head from verified cache',
                  network_type='alex', version='0.1', feature_layers=criterion.net.target_layers,
                  channels=criterion.net.n_channels_list, CUDA_initialized=False,
                  forward_calls=0, GPU_acceptance=False, metric_acceptance=False,
                  elapsed_seconds=time.monotonic()-started)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
