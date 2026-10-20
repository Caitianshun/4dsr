#!/usr/bin/env python3
"""Bind native camera, byte, complete DataLoader and environment CPU evidence."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


p=argparse.ArgumentParser()
p.add_argument("--dataset",type=Path,required=True)
p.add_argument("--runtime",type=Path,required=True)
p.add_argument("--environment-receipt",type=Path,required=True)
p.add_argument("--output",type=Path,required=True)
p.add_argument("--method",choices=["yang","4c4d"],default="yang")
a=p.parse_args()
paths={"stage":a.dataset/"stage_receipt.json","projection":a.dataset/f"projection_audit_{'yang_pillowfix' if a.method=='yang' else '4c4d'}.json",
       "full_loader":a.dataset/"full_native_loader_cpu_audit.json",
       "environment":a.environment_receipt,"source":a.runtime/"adapter_identity.json"}
if a.method=="yang":paths["dtype"]=a.dataset/"dtype_byte_equivalence.json"
r={k:json.loads(v.read_text()) for k,v in paths.items()}
identity=r["stage"]["manifest_sha256"]
assert not r["stage"]["used_HR_pixels"] and not r["stage"]["used_heldout_pixels"]
assert r["environment"]["returncode"]==0
for key in ("projection","full_loader",*(["dtype"] if a.method=="yang" else [])):
    assert r[key]["status"]=="passed" and r[key]["manifest_sha256"]==identity,key
assert r["projection"]["max_projection_error_pixels"]<.002
if a.method=="yang":assert all(x["dtype_bytes_equal"] and x["native_loader_tensor_equal"] for x in r["dtype"]["checked"])
assert r["full_loader"]["native_training_samples"]==r["stage"]["training_observations"]
assert r["full_loader"]["workers"]==12 and r["full_loader"]["batch_size"]==4
assert r["full_loader"]["RLIMIT_NOFILE_after"][0]>=65536
assert r["full_loader"]["native_loader_sha256"]==sha(a.runtime/"utils/data_utils.py")
if a.method=="yang":assert r["dtype"]["native_loader_sha256"]==sha(a.runtime/"utils/data_utils.py")
contract={"schema":"strict4d_cpu_deployment_contract_v1","status":"passed","manifest_sha256":identity,
          "used_HR_pixels":False,"used_heldout_pixels":False,"GPU_calls":0,
          "runtime":str(a.runtime.resolve()),"receipt_files":[{"kind":k,"path":str(v.resolve()),"sha256":sha(v)} for k,v in paths.items()],
          "training_wrapper_sha256":sha(Path(__file__).parent/"train_exec.py"),
          "native_training_samples":r["stage"]["training_observations"],"projection_max_error_pixels":r["projection"]["max_projection_error_pixels"],
          "compatibility_contract":{"RGB":"byte-identical uint8 Pillow adapter" if a.method=="yang" else "unchanged author's OpenCV loader BGR-to-RGB operation","worker_count":12,"batch_size":4,
                                    "IPC":"own short 0700 TMPDIR created per GPU job","nofile":"process-local soft=min(65536,hard), original hard/system unchanged"},
          "scientific_parameters_modified":False}
a.output.write_text(json.dumps(contract,indent=2)+"\n")
print(json.dumps({k:contract[k] for k in ["status","native_training_samples","projection_max_error_pixels"]}))
