"""Exercise both A100s, native kNN, LPIPS and frozen SwinIR without training."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments/dynamic_sr_20260918'))
import numpy as np
import torch
import torchvision
import cv2
import lpips
from simple_knn._C import distCUDA2
import diff_gaussian_rasterization
from generate_prior import build_model, infer, DEFAULT_NETWORK, DEFAULT_CHECKPOINT, NETWORK_SHA256, CHECKPOINT_SHA256

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    torch.set_num_threads(4)
    torch.manual_seed(20260920)
    assert torch.__version__ == '2.7.1+cu128', torch.__version__
    assert np.__version__ == '1.26.4' and cv2.__version__ == '4.10.0'
    assert torch.cuda.device_count() == 2, torch.cuda.device_count()
    result = dict(torch=torch.__version__, torchvision=torchvision.__version__,
                  cuda=torch.version.cuda, numpy=np.__version__, opencv=cv2.__version__,
                  python=sys.version, cuda_home=os.environ['CUDA_HOME'],
                  rasterizer_module=diff_gaussian_rasterization.__file__, devices=[])
    for i in range(2):
        with torch.cuda.device(i):
            torch.manual_seed(20260920)
            x = torch.randn(256, 256, device='cuda', requires_grad=True)
            loss = (x @ x.T).square().mean()
            loss.backward()
            assert torch.isfinite(x.grad).all()
            points = torch.rand(64, 3, device='cuda')
            actual = distCUDA2(points)
            reference = torch.cdist(points, points).square()
            reference.fill_diagonal_(float('inf'))
            reference = reference.topk(3, largest=False).values.mean(1)
            error = float((actual-reference).abs().max())
            assert error < 2e-5, error
            result['devices'].append(dict(index=i, name=torch.cuda.get_device_name(i),
                capability=list(torch.cuda.get_device_capability(i)), knn_max_abs_error=error,
                matrix_backward_finite=True))
    torch.cuda.set_device(0)
    metric = lpips.LPIPS(net='alex').cuda().eval().requires_grad_(False)
    x = torch.rand(1, 3, 64, 64, device='cuda')
    with torch.no_grad():
        identical = float(metric(x, x, normalize=True))
        different = float(metric(x, torch.zeros_like(x), normalize=True))
    assert abs(identical) < 1e-7 and np.isfinite(different) and different > 0
    result['lpips_alex'] = dict(identical=identical, different=different)
    assert sha(DEFAULT_NETWORK) == NETWORK_SHA256
    assert sha(DEFAULT_CHECKPOINT) == CHECKPOINT_SHA256
    net = build_model(DEFAULT_NETWORK, DEFAULT_CHECKPOINT, 'cuda:0')
    small = np.random.default_rng(20260920).integers(0, 256, (32, 32, 3), dtype=np.uint8)
    predicted = infer(net, small, 'cuda:0', 0, 32)
    assert predicted.shape == (128, 128, 3) and predicted.dtype == np.uint8
    result['swinir'] = dict(network_sha256=NETWORK_SHA256, checkpoint_sha256=CHECKPOINT_SHA256,
                            input_shape=list(small.shape), output_shape=list(predicted.shape))
    result.update(status='passed', finished_unix=time.time(),
                  scope='Environment only; renderer backward and real-data training checked by separate smoke runs.')
    path = ROOT / 'deployment/a100_20260920/environment_smoke.json'
    path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
