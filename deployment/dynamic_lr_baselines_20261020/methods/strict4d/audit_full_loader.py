#!/usr/bin/env python3
"""CPU exercise of all native training LR samples with original 12 workers."""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import resource
import sys
import tempfile
import time
from types import ModuleType,SimpleNamespace


def main(a):
    os.environ["CUDA_VISIBLE_DEVICES"]=""
    short=tempfile.mkdtemp(prefix="4dsr-s4d-qa-",dir="/tmp")
    for key in ("TMPDIR","TMP","TEMP"):os.environ[key]=short
    low,high=resource.getrlimit(resource.RLIMIT_NOFILE)
    assert high>=65536
    resource.setrlimit(resource.RLIMIT_NOFILE,(max(low,65536),high))
    import torch
    from torch.utils.data import DataLoader
    sys.path.insert(0,str(a.runtime.resolve()))
    package=ModuleType("scene");package.__path__=[str(a.runtime.resolve()/"scene")];sys.modules["scene"]=package
    reader=importlib.import_module("scene.dataset_readers")
    camera_utils=importlib.import_module("utils.camera_utils")
    data_utils=importlib.import_module("utils.data_utils")
    stage=json.loads((a.dataset/"stage_receipt.json").read_text())
    started=time.monotonic()
    infos=reader.readCamerasFromTransforms(str(a.dataset.resolve()),"transforms_train.json",False,
                                         time_duration=[0.,10.],frame_ratio=1,dataloader=True)
    args=SimpleNamespace(resolution=1,dataloader=True,data_device="cpu")
    cameras=camera_utils.cameraList_from_camInfos(infos,1.,args)
    loader=DataLoader(data_utils.CameraDataset(cameras,False),batch_size=4,shuffle=True,
                      num_workers=12,collate_fn=lambda x:x,drop_last=True)
    count=0;maximum_fd=0
    for batch in loader:
        for image,camera in batch:
            assert tuple(image.shape)==(3,252,336)
            assert image.device.type=="cpu"
            count+=1
        maximum_fd=max(maximum_fd,len(os.listdir("/proc/self/fd")))
    assert count==stage["training_observations"] and not torch.cuda.is_initialized()
    r={"status":"passed","GPU_calls":0,"used_HR_pixels":False,"used_heldout_pixels":False,
       "manifest_sha256":stage["manifest_sha256"],"native_training_samples":count,"workers":12,"batch_size":4,
       "maximum_main_process_file_descriptors":maximum_fd,"short_private_TMPDIR":short,
       "RLIMIT_NOFILE_before":[low,high],"RLIMIT_NOFILE_after":list(resource.getrlimit(resource.RLIMIT_NOFILE)),
       "seconds":time.monotonic()-started,"native_loader_sha256":hashlib.sha256((a.runtime/"utils/data_utils.py").read_bytes()).hexdigest()}
    a.output.write_text(json.dumps(r,indent=2)+"\n")
    print(json.dumps(r))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--runtime",type=Path,required=True)
    p.add_argument("--dataset",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    main(p.parse_args())
