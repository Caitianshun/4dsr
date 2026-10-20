#!/usr/bin/env python3
"""Freeze one scene's admitted native init -> full training -> HR render chain."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[4]
HERE=Path(__file__).resolve().parent
p=argparse.ArgumentParser()
p.add_argument("--method",choices=["yang","4c4d"],required=True)
p.add_argument("--scene",choices=["cook_spinach","cut_roasted_beef"],required=True)
p.add_argument("--gpu-uuid",required=True)
p.add_argument("--input-receipt",required=True)
p.add_argument("--input-inventory-sha256",required=True)
p.add_argument("--parent-input-inventory-sha256",default="8e4edaf8de7ee2a106b1519c091d76a7a6086b274a35d7b68b544132bdc130fc")
p.add_argument("--output",type=Path,required=True)
p.add_argument("--job-id",help="Explicit new broker identity after a confirmed pre-execution deployment rejection")
p.add_argument("--runtime-name",help="Versioned isolated runtime directory, e.g. yang_pillowfix")
p.add_argument("--run-name",help="Versioned new output basename; must not overwrite failed/scientific runs")
p.add_argument("--reuse-complete-initialization",action="store_true",help="Reuse only a completed, SHA-verified native geometry stage")
a=p.parse_args()
root="{ROOT}"
method=a.method
job_id=a.job_id or f"dynamic-lr-{method}-{a.scene.replace('_','-')}-20261020"
scratch_tempdir="/tmp/4dsr-s4d-"+hashlib.sha256(job_id.encode()).hexdigest()[:12]
common="deployment/dynamic_lr_baselines_20261020/methods/strict4d"
base="output/dynamic_lr_baselines_20261020/strict4d"
dataset=f"{base}/inputs/{a.scene}" if method=="yang" else f"{base}/inputs4c4d/{a.scene}"
runtime=f"{base}/runtime/{a.runtime_name or method}"
venv=f"envs/dynamic_lr_baselines_20261020/{method}/bin/python"
run_name=a.run_name or f"{method}_{a.scene}"
out=f"{base}/runs/{run_name}"
manifest=f"data/dynamic_sr/full_time_20261007/prepared/{a.scene}/manifest_train_ready_seed20261007.json"
config=f"{common}/configs/{run_name}.yaml"


def command(name,argv,cwd="."):
    return {"name":name,"argv":argv,"cwd":cwd}


def path(relative):
    return root+"/"+relative


commands=[command("native_gate",[path(venv),path(common+"/gate_native.py"),"--dataset",path(dataset),"--manifest",path(manifest),
              "--environment-receipt",path(common+f"/logs/{a.runtime_name or method}_env_exit.json"),"--projection-audit",path(dataset+f"/projection_audit_{a.runtime_name or method}.json"),
              "--scratch-tempdir",scratch_tempdir])]
if method=="yang" and a.runtime_name=="yang_pillowfix":
    commands[0]["argv"] += ["--loader-audit",path(dataset+"/dtype_byte_equivalence.json"),"--deployment-contract",path(dataset+"/deployment_contract_yang_pillowfix.json")]
if method=="4c4d":
    commands[0]["argv"] += ["--deployment-contract",path(dataset+"/deployment_contract_4c4d.json")]
if a.reuse_complete_initialization:
    receipt="frame0_init/dense_initialization_receipt.json" if method=="yang" else "mast3r_initialization_receipt.json"
    commands[0]["argv"] += ["--initialization-receipt",path(dataset+"/"+receipt)]
elif method=="yang":
    commands.append(command("native_colmap_dense",[path(venv),path(common+"/initialize_colmap.py"),"--dataset",path(dataset),"--phase","dense","--colmap","/usr/local/bin/colmap"]))
else:
    weight="weights/4c4d_20261020/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
    commands.append(command("native_mast3r",[path(venv),path(common+"/initialize_mast3r.py"),"--dataset",path(dataset),
        "--source",path(base+"/runtime/mast3r"),"--python",path("envs/dynamic_lr_baselines_20261020/mast3r/bin/python"),
        "--weights",path(weight+".pth"),"--retrieval",path(weight+"_retrieval_trainingfree.pth"),
        "--environment-receipt",path(common+"/logs/mast3r_env_exit.json")]))
commands.append(command("native_config",[path(venv),path(common+"/make_config.py"),"--runtime",path(runtime),"--dataset",path(dataset),"--output",path(out+"/model"),"--config",path(config)]))
train=[path(venv),path(common+"/train_exec.py"),"--script",path(runtime+"/train.py"),"--record",path(out+"/train_resources.json"),"--","--config",path(config)]
if method=="4c4d":
    training_ids=json.loads((ROOT/manifest).read_text())["splits"]["train"]
    train += ["--training_view",",".join(str(int(c[3:])) for c in training_ids),"--testing_view","0"]
commands.append(command("native_train",train,cwd=runtime))
commands.append(command("native_hr_render",[path(venv),path(common+"/render_heldout.py"),"--method",method,"--runtime",path(runtime),"--dataset",path(dataset),
        "--manifest",path(manifest),"--checkpoint",path(out+"/model/chkpnt30000.pth"),"--output",path(out+"/render"),
        "--metric-source",path("experiments/dynamic_sr_20260918/evaluate.py"),"--render-only"]))
commands.append(command("final_index",[path(venv),path(common+"/finalize_run.py"),"--method",method,"--run",path(out),"--manifest",path(manifest),
        "--dataset",path(dataset),"--config",path(config),"--runtime",path(runtime)]))
files=[]
script_names={"gate_native.py","make_config.py","train_exec.py","render_heldout.py","finalize_run.py","prepare_native.py"}
if not a.reuse_complete_initialization:
    script_names.add("initialize_colmap.py" if method=="yang" else "initialize_mast3r.py")
for local in sorted(HERE/n for n in script_names):
    files.append({"path":str(local.relative_to(ROOT)),"sha256":hashlib.sha256(local.read_bytes()).hexdigest()})
for relative in [runtime+"/adapter_identity.json",runtime+"/adapter.patch",runtime+"/scene/gaussian_model.py",runtime+"/utils/data_utils.py",runtime+"/train.py",runtime+"/configs/dynerf/"+a.scene+".yaml"]:
    local=ROOT/relative
    files.append({"path":relative,"sha256":hashlib.sha256(local.read_bytes()).hexdigest()})
job={"job_id":job_id, "method":method,"scene":a.scene,"gpu_uuid":a.gpu_uuid,
     "original_LR":True,"no_proposed_modules":True,"out":out,"input_receipt":a.input_receipt,"input_inventory_sha256":a.input_inventory_sha256,
     "parent_input_inventory_sha256":a.parent_input_inventory_sha256,
     "commands":commands,"required_files":files,
     "env":{"PYTHONNOUSERSITE":"1","TORCH_CUDA_ARCH_LIST":"8.0","TORCH_EXTENSIONS_DIR":path(".cache/strict4d_extensions_20261020/"+method),"OMP_NUM_THREADS":"4","OPENBLAS_NUM_THREADS":"4","MAX_JOBS":"2","QT_QPA_PLATFORM":"offscreen", "TMPDIR":scratch_tempdir,"TMP":scratch_tempdir,"TEMP":scratch_tempdir},
     "return_paths":[out,dataset,common+"/logs"],"no_automatic_training_replay":True,"native_time_mapping":"10 * manifest_time = frame_index /30"}
a.output.parent.mkdir(parents=True,exist_ok=True)
if a.output.exists():
    raise FileExistsError(a.output)
a.output.write_text(json.dumps(job,indent=2)+"\n")
print(a.output)
