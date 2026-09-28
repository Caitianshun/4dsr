"""Read-only software and compiled-extension identity; no model or GPU work."""
import hashlib
import importlib
import json
import platform
from pathlib import Path
import sys


def identity():
    import torch
    import numpy
    result=dict(python=sys.version,executable=sys.executable,platform=platform.platform(),
        torch=str(torch.__version__),cuda=torch.version.cuda,numpy=numpy.__version__,extensions={},
        precision=dict(matmul_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_tf32=torch.backends.cudnn.allow_tf32,
            cudnn_benchmark=torch.backends.cudnn.benchmark,deterministic_algorithms=torch.are_deterministic_algorithms_enabled()))
    for name in ['diff_gaussian_rasterization._C','simple_knn._C']:
        m=importlib.import_module(name);p=Path(m.__file__)
        result['extensions'][name]=dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    return result


if __name__=='__main__':print(json.dumps(identity()))
