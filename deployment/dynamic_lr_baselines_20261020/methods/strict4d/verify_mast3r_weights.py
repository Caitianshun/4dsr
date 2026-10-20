#!/usr/bin/env python3
"""Wait on CPU preparation events, then actually load official prior weights."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from queue_after import wait_file


def sha(path):
    digest=hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(8<<20),b""):
            digest.update(block)
    return digest.hexdigest()


def main(a):
    os.environ["CUDA_VISIBLE_DEVICES"]=""
    environment=wait_file(a.environment_receipt)
    if environment["returncode"]!=0:
        raise ValueError("MASt3R isolated environment did not pass CPU import")
    download=wait_file(a.weights/"download_receipt.json")
    if download["status"]!="downloaded_and_hashed":
        raise ValueError("Official MASt3R weight download did not complete")
    files={Path(r["path"]).name:r for r in download["files"]}
    for name,row in files.items():
        if sha(a.weights/name)!=row["sha256"]:
            raise ValueError("MASt3R official file hash changed before CPU load")
    began=time.monotonic()
    report={"status":"running","GPU_calls":0,"pixels_used":False,"weights":download,
            "source_patch_identity":json.loads((a.source/"adapter_identity.json").read_text())}
    try:
        sys.path.insert(0,str((a.source/"mast3r").resolve()))
        import torch
        from mast3r.model import AsymmetricMASt3R
        from mast3r.retrieval.processor import Retriever
        prefix="MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
        model=AsymmetricMASt3R.from_pretrained(str(a.weights/(prefix+".pth")))
        retriever=Retriever(str(a.weights/(prefix+"_retrieval_trainingfree.pth")),backbone=model,device="cpu")
        assert not torch.cuda.is_initialized()
        assert next(model.parameters()).device.type=="cpu"
        report.update(status="passed",native_MASt3R_model_parameter_count=sum(p.numel() for p in model.parameters()),
                      native_retrieval_model_parameter_count=sum(p.numel() for p in retriever.model.parameters()),
                      CUDA_initialized=False,torch_version=torch.__version__,
                      official_checkpoint_loading="native from_pretrained and Retriever constructors, including args Namespace and ASMK codebook")
    except BaseException as error:
        report.update(status="failed",error=repr(error))
        raise
    finally:
        report["CPU_load_seconds"]=time.monotonic()-began
        report["process_memory_high_watermark"]=[s for s in Path("/proc/self/status").read_text().splitlines() if s.startswith("VmHWM:")]
        a.output.write_text(json.dumps(report,indent=2)+"\n")
        print(json.dumps({k:report[k] for k in ["status","GPU_calls","CPU_load_seconds"]}),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--weights",type=Path,required=True)
    p.add_argument("--environment-receipt",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    main(p.parse_args())
