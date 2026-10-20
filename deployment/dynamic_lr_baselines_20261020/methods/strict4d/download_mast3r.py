#!/usr/bin/env python3
"""Download official MASt3R geometry-prior weights with resumable curl and hashes."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from datetime import datetime,timezone

p=argparse.ArgumentParser()
p.add_argument("--output",type=Path,required=True)
a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=True)
base="https://download.europe.naverlabs.com/ComputerVision/MASt3R/"
names=["MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth", "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth", "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl"]
records=[]
for name in names:
    target=a.output/name
    if not target.exists():
        temporary=target.with_suffix(target.suffix+".part")
        subprocess.run(["curl","-fL","--retry","3","--retry-delay","5","--connect-timeout","30","--continue-at","-", "--output",str(temporary),base+name],check=True)
        if temporary.stat().st_size < 100000:
            raise ValueError("Unexpected tiny weight response")
        temporary.replace(target)
    h=hashlib.sha256()
    with target.open("rb") as f:
        for b in iter(lambda:f.read(8<<20),b""):h.update(b)
    records.append({"url":base+name,"path":str(target),"bytes":target.stat().st_size,"sha256":h.hexdigest()})
(a.output/"download_receipt.json").write_text(json.dumps({"status":"downloaded_and_hashed", "utc":datetime.now(timezone.utc).isoformat(),"publisher_hash_available":False,"files":records},indent=2)+"\n")
print(json.dumps(records,indent=2))
