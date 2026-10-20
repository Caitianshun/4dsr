"""Create leaf-local CUDA header/library symlinks; never edit the base toolkit.

The private NVCC toolkit intentionally omits cuSPARSE/cuBLAS development
headers, while torch's version-matched NVIDIA packages already provide them.
"""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--compiler-root", type=Path, required=True)
    p.add_argument("--base-site", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    base, out = a.compiler_root.resolve(), a.output.resolve()
    links = []

    def link(source, destination):
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.is_symlink() and destination.resolve() == source.resolve():
            return
        if destination.exists() or destination.is_symlink():
            raise RuntimeError(f"Conflicting private CUDA overlay entry: {destination}")
        destination.symlink_to(source.resolve(), target_is_directory=source.is_dir())
        links.append(dict(path=str(destination), source=str(source.resolve())))

    for source in base.iterdir():
        if source.name not in ["include", "lib64", "lib"]:
            link(source, out / source.name)
    for prefix in [base, *sorted((a.base_site / "nvidia").glob("*"))]:
        for name, dest in [("include", "include"), ("lib64", "lib64"), ("lib", "lib64")]:
            folder = prefix / name
            if not folder.is_dir():
                continue
            for source in folder.iterdir():
                destination = out / dest / source.name
                # The compiler toolkit has priority for overlapping runtime
                # headers. Missing cuBLAS/cuSPARSE/solver entries are added.
                if destination.exists() or destination.is_symlink():
                    continue
                link(source, destination)
    for name in ["cublas", "cublasLt", "cusparse", "cusolver", "curand", "cudart"]:
        destination = out / "lib64" / f"lib{name}.so"
        candidates = sorted((out / "lib64").glob(f"lib{name}.so.*"))
        if not destination.exists() and candidates:
            link(candidates[0], destination)
    receipt = dict(compiler_root=str(base), base_nvidia_packages=str(a.base_site / "nvidia"),
                   output=str(out), scope="read-only symlink overlay inside this isolated method environment", added_links=links)
    (out / "overlay_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(out)


if __name__ == "__main__":
    main()
