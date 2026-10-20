#!/usr/bin/env python3
"""Measure the unmodified native training script in-process; no model changes."""
import argparse
from functools import wraps
import json
import os
from pathlib import Path
import runpy
import resource
import sys
import time
import traceback

p=argparse.ArgumentParser()
p.add_argument("--script",type=Path,required=True)
p.add_argument("--record",type=Path,required=True)
p.add_argument("argv",nargs=argparse.REMAINDER)
a=p.parse_args()
import torch
import numpy as np
arguments=a.argv[1:] if a.argv[:1]==["--"] else a.argv
sys.argv=[str(a.script),*arguments]
sys.path.insert(0,str(a.script.resolve().parent))
record={"status":"running","native_argv":sys.argv,"CUDA_VISIBLE_DEVICES":os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_version":torch.__version__,"method_parameters_modified_by_wrapper":False,
        "observation_hooks":"Read native initialization and before-iteration 1/10/100 point counts and memory; no extra RNG or parameter update",
        "startup_observations":[]}
soft_files,hard_files=resource.getrlimit(resource.RLIMIT_NOFILE)
required_files=65536 if hard_files==resource.RLIM_INFINITY else min(65536,hard_files)
if soft_files<required_files:
    resource.setrlimit(resource.RLIMIT_NOFILE,(required_files,hard_files))
record["RLIMIT_NOFILE"]={"before":[soft_files,hard_files],"after":list(resource.getrlimit(resource.RLIMIT_NOFILE)),
                         "scope":"this training process and its inherited native DataLoader workers only; no system/hard-limit change"}
began=time.monotonic()


def save_record():
    a.record.parent.mkdir(parents=True,exist_ok=True)
    a.record.write_text(json.dumps(record,indent=2)+"\n")


def observe(model, phase, iteration=None):
    row={"phase":phase,"before_iteration":iteration,"elapsed_seconds":time.monotonic()-began,
         "points":int(model.get_xyz.shape[0]),
         "torch_allocated_bytes":torch.cuda.memory_allocated(),
         "torch_reserved_bytes":torch.cuda.memory_reserved(),
         "peak_torch_allocated_bytes":torch.cuda.max_memory_allocated(),
         "peak_torch_reserved_bytes":torch.cuda.max_memory_reserved()}
    if phase=="native_initialization_returned" and model.gaussian_dim==4:
        # Copy to CPU for read-only statistics without allocating extra CUDA
        # work tensors or consuming random numbers in the native training.
        times=model._t.detach().cpu().double().numpy()
        variances=np.exp(2*model._scaling_t.detach().cpu().double().numpy())
        row.update(time_duration=list(model.time_duration),
                   initial_time_min=float(times.min()),initial_time_max=float(times.max()),
                   initial_time_mean=float(times.mean()),
                   initial_temporal_variance_min=float(variances.min()),
                   initial_temporal_variance_max=float(variances.max()))
    record["startup_observations"].append(row)
    save_record()
    print("STRICT4D_STARTUP_QA "+json.dumps(row),flush=True)


try:
    torch.cuda.reset_peak_memory_stats()
    from scene.gaussian_model import GaussianModel
    native_create=GaussianModel.create_from_pcd
    native_update_lr=GaussianModel.update_learning_rate

    @wraps(native_create)
    def observed_create(self,*args,**kwargs):
        result=native_create(self,*args,**kwargs)
        observe(self,"native_initialization_returned")
        return result

    @wraps(native_update_lr)
    def observed_update_lr(self,iteration,*args,**kwargs):
        result=native_update_lr(self,iteration,*args,**kwargs)
        if iteration in (1,10,100):
            observe(self,"native_learning_rate_returned",iteration)
        return result

    GaussianModel.create_from_pcd=observed_create
    GaussianModel.update_learning_rate=observed_update_lr
    runpy.run_path(str(a.script),run_name="__main__")
    record["status"]="complete"
except BaseException:
    record.update(status="failed",error=traceback.format_exc())
    raise
finally:
    record["training_seconds"]=time.monotonic()-began
    if torch.cuda.is_initialized():
        record.update(peak_torch_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_torch_reserved_bytes=torch.cuda.max_memory_reserved(),
                      GPU_name=torch.cuda.get_device_name(0),
                      GPU_total_memory_bytes=torch.cuda.get_device_properties(0).total_memory)
    save_record()
