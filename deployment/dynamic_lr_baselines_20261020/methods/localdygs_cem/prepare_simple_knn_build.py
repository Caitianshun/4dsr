"""Prepare an isolated native simple-knn build with a standard header fix.

CUDA 12.8 no longer supplies FLT_MAX through incidental transitive includes.
Only the required <cfloat> include is added; all kernel bodies are identical.
"""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import shutil


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    original = a.source / "simple_knn.cu"
    text = original.read_text()
    assert "FLT_MAX" in text
    patched = text if "#include <cfloat>" in text else text.replace("#define BOX_SIZE 1024", "#include <cfloat>\n\n#define BOX_SIZE 1024", 1)
    if a.output.exists():
        old = json.loads((a.output / "header_compatibility_receipt.json").read_text())
        assert old["upstream_sha256"] == sha(original)
        assert (a.output / "simple_knn.cu").read_text() == patched
        print("Existing isolated header-only build source verified")
        return
    shutil.copytree(a.source, a.output, ignore=shutil.ignore_patterns(".git", "build", "dist", "*.egg-info", "__pycache__", "*.so"))
    (a.output / "simple_knn.cu").write_text(patched)
    patch = "".join(difflib.unified_diff(text.splitlines(True), patched.splitlines(True), fromfile="a/simple_knn.cu", tofile="b/simple_knn.cu"))
    (a.output / "cfloat_compatibility.patch").write_text(patch)
    # Removing the one include must recover the exact original source.
    assert patched == text or patched.replace("#include <cfloat>\n\n", "", 1) == text
    receipt = dict(upstream=str(a.source.resolve()), upstream_sha256=sha(original),
                   build_source_sha256=sha(a.output / "simple_knn.cu"),
                   patch_sha256=sha(a.output / "cfloat_compatibility.patch"),
                   change="include <cfloat> only" if patch else "upstream already includes <cfloat>",
                   kernel_bodies_unchanged=True)
    (a.output / "header_compatibility_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
