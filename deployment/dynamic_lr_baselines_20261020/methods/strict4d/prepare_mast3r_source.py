#!/usr/bin/env python3
"""Copy pinned MAtCha source and record PyTorch checkpoint compatibility only."""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import shutil

p=argparse.ArgumentParser()
p.add_argument("--source",type=Path,required=True)
p.add_argument("--target",type=Path,required=True)
a=p.parse_args()
if a.target.exists():
    raise FileExistsError(a.target)
shutil.copytree(a.source,a.target,ignore=shutil.ignore_patterns(".git","__pycache__","*.pyc","*.so","build","*.egg-info"))
patch,changes=[],{}
for rel in ["mast3r/mast3r/model.py","mast3r/dust3r/dust3r/model.py","mast3r/mast3r/retrieval/processor.py"]:
    f=a.target/rel
    before=f.read_text()
    after=before.replace("torch.load(model_path, map_location='cpu')","torch.load(model_path, map_location='cpu', weights_only=False)")
    after=after.replace("torch.load(modelname, 'cpu')","torch.load(modelname, 'cpu', weights_only=False)")
    if before==after:
        raise ValueError(rel)
    f.write_text(after)
    patch.extend(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile="a/"+rel,tofile="b/"+rel))
    changes[rel]={"before_sha256":hashlib.sha256(before.encode()).hexdigest(),"after_sha256":hashlib.sha256(after.encode()).hexdigest()}
(a.target/"adapter.patch").write_text("".join(patch))
(a.target/"adapter_identity.json").write_text(json.dumps({"MAtCha_commit":"b119fd96e484fc81eb40623c1ea92ad3dbd3c21e", "changes":changes,
    "reason":"Explicit loading of official MASt3R trusted checkpoints under PyTorch2.6+; no prior model or optimization change",
    "native_RoPE":"Author's built-in PyTorch fallback, if CUDA extension is unavailable; no custom approximation",
    "native_FAISS":"Author's CPU FAISS fallback when faiss-gpu unavailable"},indent=2)+"\n")
