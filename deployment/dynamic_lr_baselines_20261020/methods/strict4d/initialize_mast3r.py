#!/usr/bin/env python3
"""Run the exact MAtCha posed MASt3R-SfM stage on registered training LR frame0.

This is a CUDA preprocessing job and must enter the root GPU broker.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def run(a):
    dataset = a.dataset.resolve()
    receipt = json.loads((dataset/"stage_receipt.json").read_text())
    rows = {r["name"]: r for r in receipt["training_image_records"] if r["frame"] == 0}
    folder = dataset/"frame0_init"
    if set(p.name for p in (folder/"images").iterdir()) != set(rows):
        raise ValueError("MASt3R inputs must be exactly training LR frame0")
    for name,row in rows.items():
        if sha(folder/"images"/name) != row["sha256"]:
            raise ValueError("MASt3R source LR changed")
    out = dataset/"mast3r_sfm"
    destination = dataset/"points3d.ply"
    if out.exists() or destination.exists():
        raise FileExistsError("MASt3R outputs already exist; do not replay/overwrite")
    checkpoint = a.weights.resolve()
    retrieval = a.retrieval.resolve()
    environment=json.loads(a.environment_receipt.read_text())
    downloads=json.loads((checkpoint.parent/"download_receipt.json").read_text())
    loaded=json.loads((checkpoint.parent/"weight_load_receipt.json").read_text())
    if environment["returncode"]!=0 or downloads["status"]!="downloaded_and_hashed" or loaded["status"]!="passed":
        raise ValueError("MASt3R CPU environment or official checkpoint preparation did not pass")
    files={Path(r["path"]).name:r for r in downloads["files"]}
    codebook=retrieval.with_name(retrieval.name.replace("_trainingfree.pth","_codebook.pkl"))
    for file in [checkpoint,retrieval,codebook]:
        if sha(file)!=files[file.name]["sha256"]:
            raise ValueError("MASt3R checkpoint/codebook bytes differ from download receipt")
    argv = [str(a.python), "mast3r/run_mast3r.py", "--scene_path", str(folder), "--output_dir", str(out),
            "--weights_path", str(checkpoint), "--retrieval_model", str(retrieval),
            "--min_conf_thr", "0", "--matching_conf_thr", "0", "--n_coarse_iterations", "1000",
            "--n_refinement_iterations", "1000", "--TSDF_thresh", "0", "--fix_focal", "--fix_principal_point",
            "--fix_rotation", "--fix_translation", "--n_images", "-1", "--use_all_images", "--image_size", "512",
            "--max_window_size", "20", "--max_refid", "10", "--use_calibrated_poses", "--output_conf_thr", "0.1",
            "--align_camera_locations", "--gpu", "0"]
    record = {"status": "running", "argv": argv, "cwd": str(a.source.resolve()), "manifest_sha256": receipt["manifest_sha256"],
              "checkpoint_sha256": sha(checkpoint), "retrieval_sha256": sha(retrieval), "source_inputs": list(rows.values()),
              "codebook_sha256":sha(codebook),"official_weight_download_identity":downloads,
              "preprocessor_source_identity":json.loads((a.source/"adapter_identity.json").read_text()),
              "environment_CPU_import_receipt":environment,
              "official_checkpoint_CPU_load_receipt":loaded,
              "used_HR_pixels": False, "used_heldout_pixels": False, "pretrained_MASt3R_prior": True}
    began = time.monotonic()
    try:
        subprocess.run(argv, cwd=a.source, check=True)
        source = out/"points.ply"
        if not source.is_file() or source.stat().st_size < 1024:
            raise ValueError("MASt3R produced no nonempty dense point cloud")
        shutil.copy2(source, destination)
        record.update(status="complete", point_cloud_sha256=sha(destination), point_cloud=str(destination))
    except BaseException as e:
        record.update(status="failed", error=repr(e))
        raise
    finally:
        record["seconds"] = time.monotonic()-began
        (dataset/"mast3r_initialization_receipt.json").write_text(json.dumps(record,indent=2)+"\n")


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--dataset",type=Path,required=True)
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--python",type=Path,required=True)
    p.add_argument("--weights",type=Path,required=True)
    p.add_argument("--retrieval",type=Path,required=True)
    p.add_argument("--environment-receipt",type=Path,required=True)
    run(p.parse_args())
