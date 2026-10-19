"""HAT-L x4 area-matched teacher cache on the training LR of a physical-LR scene (Claude 4dsr-7c, 2026-10-18; wave-14 stage 2).

  teacher_physlr_hat.py --manifest MANIFEST --weights FT_DIR [--batch 3] [--max-hours H]

Same output layout and index schema as teacher_physlr3.py (tag sr_hat_l_x4_areaft), so physlr.py sr-register accepts it; only the
network differs: the HAT-L fine-tune of finetune_swinir_area.py --arch hat_l (FT_DIR/summary.json gives the weights path and sha256,
checked before use), run with the official HAT test pre-processing (reflect pad to a multiple of the window size 16, crop;
hat_arch_local.infer). FP32, TF32 off. Training cameras only; no HR or test-camera pixels are read.

Budget rule (README 第十四波, amendment 1): the first camera is generated first; with its measured seconds per frame the remaining
frames are projected, and if elapsed + projection exceeds --max-hours the run stops, writes <TAG>_budget_stop.json next to the
manifest and exits with code 3 (no index is written, so nothing can be registered from a partial cache).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TAG = "sr_hat_l_x4_areaft"


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(path):
    return str(Path(path).resolve().relative_to(ROOT))


def save_png(img, path):
    from PIL import Image
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(path, compress_level=1)
    return sha(path)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", required=True, help="manifest_train_ready_seed<seed>.json of the scene (project-relative)")
    p.add_argument("--weights", required=True, help="HAT-L fine-tune output directory with summary.json")
    p.add_argument("--batch", type=int, default=3, help="frames per forward pass (same-size images)")
    p.add_argument("--max-hours", type=float, default=0.0, help="stop after the first camera if the projected total exceeds this")
    p.add_argument("--device", default="cuda")
    p.add_argument("--debug-frames", type=int, default=0, help="debug: only the first N frames of the first camera, no index")
    a = p.parse_args()
    import numpy as np
    import torch
    from PIL import Image
    sys.path.insert(0, str(HERE))
    import hat_arch_local as H
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    ft = json.loads((ROOT / a.weights / "summary.json").read_text())
    if ft.get("arch") != "hat_l":
        raise SystemExit("not a HAT-L fine-tune")
    wpath, wsha = ROOT / ft["weights"]["path"], ft["weights"]["sha256"]
    if sha(wpath) != wsha:
        raise SystemExit("HAT-L weights differ")
    net = H.build_hat_l()
    H.load_state(net, wpath, "params")
    net = net.eval().to(a.device)
    mpath = ROOT / a.manifest
    m = json.loads(mpath.read_text())
    root = mpath.parent
    if int(m["scale"]) != 4:
        raise SystemExit("x4 scenes only")
    seed = int(m["initialization"]["seed"])
    train = sorted(m["splits"]["train"])
    obs = {(o["camera_id"], int(o["frame_index"])): o for o in m["observations"] if o["camera_id"] in train}
    t0 = time.time()
    entries, per_cam = {}, []
    for ci, cam in enumerate(train):
        tc, done_before = time.time(), 0
        todo = []
        for f in range(300):
            o = obs[(cam, f)]
            lr_path = root / o["lr_path"]
            out = root / TAG / cam / f"{f:04d}.png"
            if out.exists():
                entries[(cam, f)] = dict(camera=cam, frame=f, relative_path=str(out.relative_to(root)), sha256=sha(out),
                                         lr_sha256=o["lr_sha256"], lr_relative_path=str(lr_path.relative_to(root)))
                done_before += 1
                continue
            if sha(lr_path) != o["lr_sha256"]:
                raise ValueError(f"teacher input changed: {lr_path}")
            todo.append((f, o, lr_path, out))
        if a.debug_frames:
            todo = todo[:a.debug_frames]
        for i in range(0, len(todo), a.batch):
            chunk = todo[i:i + a.batch]
            x = torch.from_numpy(np.stack([np.asarray(Image.open(c[2]).convert("RGB")) for c in chunk]).astype(np.float32) / 255.0)
            with torch.no_grad():
                y = H.infer(net, x.permute(0, 3, 1, 2).to(a.device)).clamp(0, 1)
            for j, (f, o, lr_path, out) in enumerate(chunk):
                img = (y[j].permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)
                entries[(cam, f)] = dict(camera=cam, frame=f, relative_path=str(out.relative_to(root)), sha256=save_png(img, out),
                                         lr_sha256=o["lr_sha256"], lr_relative_path=str(lr_path.relative_to(root)))
            del x, y
        sec = time.time() - tc
        per_cam.append(dict(camera=cam, new_frames=len(todo), seconds=sec))
        print(f"{time.time() - t0:8.0f}s {TAG} {cam} done ({len(todo)} new, {sec:.0f}s); peak "
              f"{torch.cuda.max_memory_allocated() / 2**30 if a.device == 'cuda' else 0:.1f} GiB", flush=True)
        if a.debug_frames:
            print(json.dumps(dict(debug=per_cam)), flush=True)
            return
        if ci == 0 and a.max_hours > 0 and todo:
            spf = sec / len(todo)
            left = sum(1 for c in train[1:] for f in range(300) if not (root / TAG / c / f"{f:04d}.png").exists())
            proj = (time.time() - t0) + spf * left
            rec = dict(camera=cam, s_per_frame=spf, frames_left=left, projected_hours=proj / 3600, max_hours=a.max_hours,
                       gpu=torch.cuda.get_device_name() if a.device == "cuda" else "cpu", batch=a.batch)
            print(json.dumps(dict(budget_check=rec)), flush=True)
            if proj > a.max_hours * 3600:
                (root / f"{TAG}_budget_stop.json").write_text(json.dumps(dict(rec, status="stopped_over_budget"), indent=1) + "\n")
                sys.exit(3)
    index = dict(schema=1, status="completed_teacher_inventory", scene=m["scene"], seed_consumer=seed, manifest=rel(mpath),
                 manifest_sha256=sha(mpath), oracle=False,
                 teacher_config=dict(kind="physlr_teacher", tag=TAG, teacher="hat_l_area_finetuned_x4", batch=a.batch, png_compress_level=1,
                                     weights=dict(path=rel(wpath), sha256=wsha), scale=4, preprocess="reflect pad to 16, crop",
                                     code=dict(path=rel(__file__), sha256=sha(__file__)),
                                     arch_code=dict(path=rel(HERE / "hat_arch_local.py"), sha256=sha(HERE / "hat_arch_local.py")),
                                     gpu=torch.cuda.get_device_name(), torch=torch.__version__, seconds=time.time() - t0,
                                     per_camera=per_cam),
                 entries=[entries[k] for k in sorted(entries)], train_cameras=train, frame_indices=list(range(300)),
                 input_policy="training LR of this manifest (physically sampled LR)", seed_independent_frozen_prior=True,
                 HR_quality_evaluation=False)
    out = root / f"teacher_index_{TAG}_seed{seed}.json"
    text = json.dumps(index, indent=1) + "\n"
    if out.exists() and out.read_text() != text:
        raise ValueError(f"refusing to replace a different index {out}")
    out.write_text(text)
    print("wrote", rel(out))


if __name__ == "__main__":
    main()
