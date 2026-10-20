"""Check real native CUDA dependencies inside an admitted broker process.

These fixed operator fixtures are environment diagnostics, not scene models,
training, method modifications, or substitutes for native initialization.
"""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=["localdygs", "cem"], required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise SystemExit("Refusing an existing GPU acceptance result")
    import torch
    assert torch.cuda.is_available()
    started = time.monotonic()
    package_paths = []
    if a.method == "localdygs":
        import tinycudann
        import torch_scatter
        import diff_gaussian_rasterization
        import simple_knn._C
        package_paths += [Path(tinycudann.__file__), Path(importlib.import_module("tinycudann.modules")._C.__file__),
                          Path(diff_gaussian_rasterization.__file__), Path(diff_gaussian_rasterization._C.__file__), Path(simple_knn._C.__file__)]
        x = torch.tensor([1., 3., 2.], device="cuda")
        index = torch.tensor([0, 0, 1], dtype=torch.int64, device="cuda")
        result, indices = torch_scatter.scatter_max(x, index)
        assert torch.equal(result, torch.tensor([3., 2.], device="cuda"))
        dependency = dict(tinycudann_import="passed with actual GPU capability query", torch_scatter_GPU="passed",
                          native_rasterizer_import="passed", native_simple_knn_import="passed")
    else:
        import numpy as np
        import faiss
        import diff_gaussian_rasterization_df
        import simple_knn._C
        from pytorch3d.transforms import quaternion_apply, quaternion_multiply, quaternion_invert
        import pytorch3d
        package_paths += [Path(faiss.__file__), *Path(faiss.__file__).parent.glob("*.so"), Path(pytorch3d.__file__),
                          Path(diff_gaussian_rasterization_df.__file__), Path(diff_gaussian_rasterization_df._C.__file__), Path(simple_knn._C.__file__)]
        resource = faiss.StandardGpuResources()
        index = faiss.GpuIndexFlatL2(resource, 3)
        vectors = np.array([[0., 0., 0.], [1., 0., 0.], [0., 2., 0.]], dtype=np.float32)
        index.add(vectors)
        distances, indices = index.search(vectors, 1)
        assert np.array_equal(indices[:, 0], [0, 1, 2]) and np.max(np.abs(distances)) < 1e-6
        quaternion = torch.tensor([[1., 0., 0., 0.]], device="cuda")
        point = torch.tensor([[1., 2., 3.]], device="cuda")
        assert torch.equal(quaternion_apply(quaternion, point), point)
        assert torch.equal(quaternion_multiply(quaternion, quaternion_invert(quaternion)), quaternion)
        dependency = dict(faiss_version=faiss.__version__, native_faiss_GPU_L2="passed fixed 3-point identity fixture",
                          native_pytorch3d_GPU_quaternion="passed fixed identity fixture",
                          native_rasterizer_import="passed", native_simple_knn_import="passed")
    torch.cuda.synchronize()
    properties = torch.cuda.get_device_properties(0)
    imported = []
    for file in package_paths:
        assert file.resolve().is_relative_to(Path(sys.prefix).resolve()), f"Borrowed the wrong method extension: {file}"
        imported.append(dict(path=str(file), sha256=hashlib.sha256(file.read_bytes()).hexdigest()))
    value = dict(status="passed", method=a.method, scope=__doc__, torch_version=torch.__version__,
                 CUDA_version=torch.version.cuda, gpu_name=properties.name,
                 compute_capability=[properties.major, properties.minor],
                 elapsed_seconds=time.monotonic() - started, dependencies=dependency,
                 method_packages_resolve_inside_isolated_leaf=True, imported_package_and_binary_identity=imported,
                 scene_model_created=False, optimizer_steps=0)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(value, indent=2) + "\n")
    print(json.dumps(value, indent=2))


if __name__ == "__main__":
    main()
