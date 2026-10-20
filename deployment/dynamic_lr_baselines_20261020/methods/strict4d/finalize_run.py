#!/usr/bin/env python3
"""Create a machine-readable endpoint index after all 300 HR renders exist."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
from PIL import Image


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(8<<20),b""):h.update(b)
    return h.hexdigest()


p=argparse.ArgumentParser()
p.add_argument("--method",choices=["yang","4c4d"],required=True)
p.add_argument("--run",type=Path,required=True)
p.add_argument("--manifest",type=Path,required=True)
p.add_argument("--dataset",type=Path,required=True)
p.add_argument("--config",type=Path,required=True)
p.add_argument("--runtime",type=Path,required=True)
a=p.parse_args()
manifest=json.loads(a.manifest.read_text())
stage=json.loads((a.dataset/"stage_receipt.json").read_text())
resources=json.loads((a.run/"train_resources.json").read_text())
if resources["status"]!="complete" or sha(a.manifest)!=stage["manifest_sha256"]:
    raise ValueError("Native training/stage identity is incomplete")
checkpoint=a.run/"model/chkpnt30000.pth"
if not checkpoint.is_file():
    raise FileNotFoundError(checkpoint)
render_root=a.run/"render/renders_hr"
names={f"{f:04d}.png" for f in range(300)}
if {p.name for p in render_root.glob("*.png")}!=names:
    raise ValueError("Requires exactly all 300 endpoint HR renders")
for f in range(300):
    with Image.open(render_root/f"{f:04d}.png") as im:
        if list(im.size)!=manifest["resolutions"]["hr"] or im.mode!="RGB":
            raise ValueError("Invalid endpoint render dimensions/mode")
training_log=a.run/"native_train.log"
initial=re.search(r"Number of points at initialisation\s*:\s*(\d+)",training_log.read_text(errors="replace"))
configuration_copy=a.run/"native_config.yaml"
shutil.copy2(a.config,configuration_copy)
metadata=a.run/"native_source_metadata"
metadata.mkdir()
for name in ["adapter_identity.json","adapter.patch"]:
    shutil.copy2(a.runtime/name,metadata/name)
result={"schema":"dynamic_lr_method_result_index_v1","status":"complete_native_training_and_300_HR_PNGs",
        "method":a.method,"scene":manifest["scene"],"original_LR":True,"no_proposed_modules":True,
        "manifest":{"path":str(a.manifest),"sha256":sha(a.manifest)},
        "checkpoint":{"path":str(checkpoint),"sha256":sha(checkpoint),"iteration":30000,"selection":"author full-schedule endpoint; no heldout-based selection"},
        "renders":{"directory":str(render_root),"pattern":"{frame:04d}.png","count":300,"camera":"cam00"},
        "config":{"path":str(configuration_copy),"submitted_path":str(a.config),"sha256":sha(configuration_copy)},
        "source_patch_identity":json.loads((a.runtime/"adapter_identity.json").read_text()),
        "resources":resources,"initialization_point_count":None if initial is None else int(initial[1]),
        "native_time_mapping":"native_time = 10 * manifest_time = frame_index / 30; model duration [0,10]",
        "native_initial_time":"Uniform[-1,11]" if a.method=="yang" else "Uniform[0,10] at author's redundant_ratio=0",
        "native_initial_temporal_variance":2.,
        "metrics_status":"pending shared A100 evaluator; method-internal scores not main-table numbers"}
receipt_path=a.dataset/("frame0_init/dense_initialization_receipt.json" if a.method=="yang" else "mast3r_initialization_receipt.json")
result["native_initialization_receipt"]=json.loads(receipt_path.read_text())
result["stage_receipt_path"]=str(a.dataset/"stage_receipt.json")
result["source_metadata_copy"]=str(metadata)
if (a.run/"job.json").exists():
    job=json.loads((a.run/"job.json").read_text())
    result["gpu_uuid"]=job["gpu_uuid"]
(a.run/"result_index.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps({k:result[k] for k in ["status","method","scene","initialization_point_count"]}))
