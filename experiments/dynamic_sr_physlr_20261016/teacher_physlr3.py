"""SwinIR-M x4 teacher with selectable weights on the training LR of a physical-LR scene (Claude, 4dsr-7c, 2026-10-17).

teacher_physlr3.py = teacher_physlr2.py (cached shifted-window masks, PNG compress level 1) with --weights / --tag, so the
area-fine-tuned SwinIR of finetune_swinir_area.py can be applied: --weights takes the fine-tune directory (its summary.json
gives the weights path and sha256, checked before use). Without --weights it is teacher_physlr2.py (classical DF2K weights,
tag sr_swinir_x4). Same output layout and index schema; training cameras only; no HR or test-camera pixels are read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SWINIR_DIR = ROOT / "third_party/swinir_frozen_20261010"
WEIGHTS = (ROOT / "third_party/swinir_frozen_20261010/001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth",
           "4e78e33f22c1aa8a773db0cf4a7381bae97c2362c717f155439ebc690cbd9215")
TAG = "sr_swinir_x4"  # same tag and index schema as teacher_physlr.py


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
    tmp = path.with_name(path.stem + ".tmp.png")
    Image.fromarray(img).save(tmp, compress_level=1)
    new = sha(tmp)
    if path.exists():
        same = sha(path) == new
        tmp.unlink()
        if not same:
            raise ValueError(f"refusing to replace a different teacher image: {path}")
    else:
        tmp.replace(path)
    return new


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", required=True, help="manifest_train_ready_seed<seed>.json of the scene (project-relative)")
    p.add_argument("--batch", type=int, default=4, help="frames per forward pass (same-size images; per-image results equal up to GPU arithmetic)")
    p.add_argument("--weights", help="fine-tune output directory with summary.json (finetune_swinir_area.py)")
    p.add_argument("--tag", default=None, help="output tag (default sr_swinir_x4, or sr_swinir_x4_areaft with --weights)")
    a = p.parse_args()
    global TAG, WEIGHTS
    if a.weights:
        ft = json.loads((ROOT / a.weights / "summary.json").read_text())
        WEIGHTS = (ROOT / ft["weights"]["path"], ft["weights"]["sha256"])
        TAG = a.tag or "sr_swinir_x4_areaft"
    elif a.tag:
        TAG = a.tag
    import numpy as np
    import torch
    from PIL import Image
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    sys.path.insert(0, str(SWINIR_DIR))
    import network_swinir
    from network_swinir import SwinIR
    cache = {}
    original_mask = network_swinir.SwinTransformerBlock.calculate_mask

    def cached_mask(self, x_size):
        key = (self.window_size, self.shift_size, tuple(x_size))
        if key not in cache:
            cache[key] = original_mask(self, x_size).cuda()
        return cache[key]
    network_swinir.SwinTransformerBlock.calculate_mask = cached_mask
    if sha(WEIGHTS[0]) != WEIGHTS[1]:
        raise SystemExit("SwinIR weights differ")
    net = SwinIR(upscale=4, in_chans=3, img_size=64, window_size=8, img_range=1.0, depths=[6] * 6, embed_dim=180,
                 num_heads=[6] * 6, mlp_ratio=2, upsampler="pixelshuffle", resi_connection="1conv")
    state = torch.load(WEIGHTS[0], map_location="cpu", weights_only=True)
    net.load_state_dict(state["params"] if "params" in state else state, strict=True)
    net = net.eval().cuda()
    mpath = ROOT / a.manifest
    m = json.loads(mpath.read_text())
    root = mpath.parent
    if int(m["scale"]) != 4:
        raise SystemExit("x4 scenes only")
    seed = int(m["initialization"]["seed"])
    train = sorted(m["splits"]["train"])
    obs = {(o["camera_id"], int(o["frame_index"])): o for o in m["observations"] if o["camera_id"] in train}
    t0 = time.time()
    entries = {}
    for cam in train:
        todo = []
        for f in range(300):
            o = obs[(cam, f)]
            lr_path = root / o["lr_path"]
            out = root / TAG / cam / f"{f:04d}.png"
            if out.exists():
                entries[(cam, f)] = dict(camera=cam, frame=f, relative_path=str(out.relative_to(root)), sha256=sha(out),
                                         lr_sha256=o["lr_sha256"], lr_relative_path=str(lr_path.relative_to(root)))
                continue
            if sha(lr_path) != o["lr_sha256"]:
                raise ValueError(f"teacher input changed: {lr_path}")
            todo.append((f, o, lr_path, out))
        for i in range(0, len(todo), a.batch):
            chunk = todo[i:i + a.batch]
            x = torch.from_numpy(np.stack([np.asarray(Image.open(c[2]).convert("RGB")) for c in chunk]).astype(np.float32) / 255.0)
            x = x.permute(0, 3, 1, 2).cuda()
            h, w = x.shape[2:]
            hp, wp = (h // 8 + 1) * 8 - h, (w // 8 + 1) * 8 - w
            x = torch.cat([x, torch.flip(x, [2])], 2)[:, :, :h + hp, :]
            x = torch.cat([x, torch.flip(x, [3])], 3)[:, :, :, :w + wp]
            with torch.no_grad():
                y = net(x)[..., :h * 4, :w * 4].clamp(0, 1)
            for j, (f, o, lr_path, out) in enumerate(chunk):
                img = (y[j].permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)
                entries[(cam, f)] = dict(camera=cam, frame=f, relative_path=str(out.relative_to(root)), sha256=save_png(img, out),
                                         lr_sha256=o["lr_sha256"], lr_relative_path=str(lr_path.relative_to(root)))
            del x, y
        print(f"{time.time() - t0:8.0f}s {TAG} {cam} done; peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
    index = dict(schema=1, status="completed_teacher_inventory", scene=m["scene"], seed_consumer=seed, manifest=rel(mpath),
                 manifest_sha256=sha(mpath), oracle=False,
                 teacher_config=dict(kind="physlr_teacher", tag=TAG, teacher="swinir_area_finetuned_x4" if a.weights else "swinir_classical_x4", batch=a.batch, mask_cache=True, png_compress_level=1,
                                     weights=dict(path=rel(WEIGHTS[0]), sha256=WEIGHTS[1]), scale=4,
                                     code=dict(path=rel(__file__), sha256=sha(__file__)), gpu=torch.cuda.get_device_name(),
                                     torch=torch.__version__, seconds=time.time() - t0),
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
