"""Compile private Multi4D sm80 extensions with CUDA devices hidden.

Run with the private multi_env Python. Only that environment and the declared
private runtime directory are written. No model or CUDA kernel is executed.
"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[4]
RUNTIME = ROOT / "output/dynamic_lr_baselines_20261020/runtime"
UPSTREAM = RUNTIME / "Multi4D-c483e82"


def status(state, **kw):
    p = RUNTIME / "multi_build_status.json"
    temporary = p.with_suffix(".tmp")
    temporary.write_text(json.dumps(dict(status=state, pid=os.getpid(), **kw), indent=2) + "\n")
    temporary.replace(p)


def main():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    assert os.environ.get("TORCH_CUDA_ARCH_LIST") == "8.0"
    assert Path(sys.prefix).resolve() == RUNTIME / "multi_env"
    started = time.time()
    try:
        jobs = [
            ("python_dependencies", ["natsort==8.4.0", "scikit-image==0.24.0"]),
            ("simple_knn_sm80", [str(UPSTREAM / "submodules/simple-knn"), "--no-deps", "--no-build-isolation"]),
            ("rasterizer_sm80", [str(UPSTREAM / "submodules/diff-gaussian-rasterization"), "--no-deps", "--no-build-isolation"]),
        ]
        for stage, packages in jobs:
            status(stage, started_unix=started)
            with (RUNTIME / (stage + ".log")).open("w") as stream:
                subprocess.run([sys.executable, "-m", "pip", "install", *packages], stdin=subprocess.DEVNULL,
                               stdout=stream, stderr=subprocess.STDOUT, check=True)
        status("hybrid_JIT_sm80", started_unix=started)
        import torch
        sys.path.insert(0, str(UPSTREAM))
        p = UPSTREAM / "gaussian_renderer/diff_gaussian_rasterization_hybrid.py"
        s = importlib.util.spec_from_file_location("native_multi4d_hybrid_CPU_build", p)
        m = importlib.util.module_from_spec(s)
        s.loader.exec_module(m)
        import simple_knn._C
        import diff_gaussian_rasterization._C
        import mmcv
        import numpy
        import natsort
        import skimage
        assert not torch.cuda.is_initialized()
        status("completed_CPU_build_no_CUDA_initialized", seconds=time.time() - started,
               torch=str(torch.__version__), mmcv=mmcv.__version__, numpy=numpy.__version__,
               simple_knn=simple_knn._C.__file__, rasterizer=diff_gaussian_rasterization._C.__file__,
               hybrid=m._C.__file__, CUDA_initialized=False)
    except BaseException:
        status("failed", seconds=time.time() - started, error=traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
