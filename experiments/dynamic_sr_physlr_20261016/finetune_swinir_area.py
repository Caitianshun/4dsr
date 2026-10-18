"""Fine-tune SwinIR-M x4 (classical, DF2K) for the pixel-area (4x4 box) degradation (Claude, 4dsr-7c, 2026-10-17).
Interpreter mml_bw, GPU through the training service.

Data: data/dynamic_sr/area_teacher_20261017/index.json (HR frames of NON-evaluation multi-view sources written by
area_teacher_data.py; no N3DV kitchen scene, no MeetRoom). Frames with time index 5 (<cam>_5.png) are validation, the rest
training. LR = round(mean of each 4x4 HR block) in 8-bit sRGB, i.e. exactly physlr.py area4 (block grid aligned: crop offsets
are multiples of 4). Random 256x256 HR crops (64x64 LR, the network's window-mask size), random flips and transposes, L1
loss, Adam (lr --lr, cosine to lr/100 over --iters), batch --batch, FP32 with TF32 off.
Validation every --val-every steps: centre 256x256 HR crop of every validation frame, PSNR (RGB, 8-bit rounded output) of the
current model; the initial (bicubic-trained) model is scored once on the same crops. Bicubic-up of the LR is reported too.
Outputs in --out: swinir_area_ft.pth (params), log.jsonl, summary.json (sha256 of weights, data index, code).
--degradation arealin (2026-10-18): the same 4x4 block mean taken in linear light (sRGB decode, mean, encode, rounded), i.e.
physlr.py area4lin, for the wave-12 colour-space control; the default `area` is unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SWINIR_DIR = ROOT / "third_party/swinir_frozen_20261010"
WEIGHTS = (ROOT / "third_party/swinir_frozen_20261010/001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth",
           "4e78e33f22c1aa8a773db0cf4a7381bae97c2362c717f155439ebc690cbd9215")
DATA = ROOT / "data/dynamic_sr/area_teacher_20261017"


def relpath(path):
    path = Path(path).resolve()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--iters", type=int, default=6000)
    p.add_argument("--batch", type=int, default=6)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--val-every", type=int, default=1000)
    p.add_argument("--seed", type=int, default=20261017)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-frames", type=int, default=0, help="debug: limit training frames")
    p.add_argument("--degradation", choices=("area", "arealin"), default="area")
    a = p.parse_args()
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "summary.json").exists():
        raise FileExistsError(f"completed fine-tune preserved: {out}")
    sys.path.insert(0, str(SWINIR_DIR))
    from network_swinir import SwinIR
    if sha(WEIGHTS[0]) != WEIGHTS[1]:
        raise SystemExit("SwinIR weights differ")

    def build():
        net = SwinIR(upscale=4, in_chans=3, img_size=64, window_size=8, img_range=1.0, depths=[6] * 6, embed_dim=180,
                     num_heads=[6] * 6, mlp_ratio=2, upsampler="pixelshuffle", resi_connection="1conv")
        state = torch.load(WEIGHTS[0], map_location="cpu", weights_only=True)
        net.load_state_dict(state["params"] if "params" in state else state, strict=True)
        return net.to(a.device)

    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    index = json.loads((DATA / "index.json").read_text())
    train, val = [], []
    for rel, e in sorted(index["frames"].items()):
        img = np.asarray(Image.open(DATA / rel).convert("RGB"))
        if hashlib.sha256((DATA / rel).read_bytes()).hexdigest() != e["sha256"]:
            raise ValueError(f"frame changed: {rel}")
        (val if rel.endswith("_5.png") else train).append((rel, img))
    if a.max_frames:
        train = train[:a.max_frames]
        val = val[:max(2, a.max_frames // 4)]
    print(f"train frames {len(train)}, val frames {len(val)}", flush=True)

    def box_lr(hr_u8):  # (B,3,H,W) uint8-valued float -> rounded 4x4 mean, same as physlr.py (round half to even)
        if a.degradation == "area":
            return torch.round(F.avg_pool2d(hr_u8, 4))
        x = hr_u8.double() / 255.0  # area4lin: decode, mean, encode in float64 as physlr.py
        lin = torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)
        m = F.avg_pool2d(lin, 4)
        enc = torch.where(m <= 0.0031308, 12.92 * m, 1.055 * m.clamp(min=0) ** (1 / 2.4) - 0.055)
        return torch.round(255.0 * enc.clamp(0, 1)).float()

    def sample_batch():
        hrs = []
        for _ in range(a.batch):
            _, img = train[rng.integers(len(train))]
            h, w = img.shape[:2]
            y = int(rng.integers(0, (h - 256) // 4 + 1)) * 4
            x = int(rng.integers(0, (w - 256) // 4 + 1)) * 4
            c = img[y:y + 256, x:x + 256]
            if rng.random() < 0.5:
                c = c[:, ::-1]
            if rng.random() < 0.5:
                c = c[::-1]
            if rng.random() < 0.5:
                c = c.transpose(1, 0, 2)
            hrs.append(np.ascontiguousarray(c))
        hr = torch.from_numpy(np.stack(hrs)).permute(0, 3, 1, 2).float().to(a.device)
        lr = box_lr(hr) / 255.0
        return lr, hr / 255.0

    vcrops = []
    for rel, img in val:
        h, w = img.shape[:2]
        y, x = ((h - 256) // 2) // 4 * 4, ((w - 256) // 2) // 4 * 4
        vcrops.append((rel, np.ascontiguousarray(img[y:y + 256, x:x + 256])))
    vhr = torch.from_numpy(np.stack([c for _, c in vcrops])).permute(0, 3, 1, 2).float().to(a.device)
    vlr = box_lr(vhr) / 255.0
    vhr = vhr / 255.0

    def score(net):
        net.eval()
        ps = []
        with torch.no_grad():
            for i in range(0, len(vlr), 8):
                y = net(vlr[i:i + 8]).clamp(0, 1)
                y = torch.round(y * 255) / 255
                mse = ((y - vhr[i:i + 8]) ** 2).mean((1, 2, 3))
                ps += (-10 * torch.log10(mse.clamp_min(1e-12))).tolist()
        net.train()
        return float(np.mean(ps)), ps

    bic = F.interpolate(vlr, scale_factor=4, mode="bicubic", align_corners=False).clamp(0, 1)
    bic_psnr = float((-10 * torch.log10(((torch.round(bic * 255) / 255 - vhr) ** 2).mean((1, 2, 3)))).mean())
    net = build()
    init_psnr, init_list = score(net)
    print(f"val: bicubic-up {bic_psnr:.3f}  initial SwinIR {init_psnr:.3f}", flush=True)
    log = (out / "log.jsonl").open("a", buffering=1)
    log.write(json.dumps(dict(step=0, val_psnr=init_psnr, bicubic_up_psnr=bic_psnr)) + "\n")
    opt = torch.optim.Adam(net.parameters(), lr=a.lr, betas=(0.9, 0.99))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * min(s, a.iters) / a.iters)))
    net.train()
    t0 = time.time()
    run = 0.0
    curve = [dict(step=0, val_psnr=init_psnr)]
    for step in range(1, a.iters + 1):
        lr_in, hr = sample_batch()
        pred = net(lr_in)
        loss = (pred - hr).abs().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        run = 0.98 * run + 0.02 * float(loss) if step > 1 else float(loss)
        if step % 100 == 0:
            log.write(json.dumps(dict(step=step, loss=run, lr=sched.get_last_lr()[0], seconds=time.time() - t0)) + "\n")
        if step % a.val_every == 0 or step == a.iters:
            v, _ = score(net)
            curve.append(dict(step=step, val_psnr=v))
            log.write(json.dumps(dict(step=step, val_psnr=v)) + "\n")
            print(f"step {step} loss {run:.4f} val {v:.3f} ({time.time() - t0:.0f}s, peak "
                  f"{torch.cuda.max_memory_allocated() / 2**30 if a.device == 'cuda' else 0:.1f} GiB)", flush=True)
    final_psnr, final_list = score(net)
    wpath = out / "swinir_area_ft.pth"
    torch.save(dict(params=net.state_dict()), wpath)
    summary = dict(status="completed", weights=dict(path=relpath(wpath), sha256=sha(wpath)),
                   base_weights=dict(path=relpath(WEIGHTS[0]), sha256=WEIGHTS[1]),
                   data_index=dict(path=relpath(DATA / "index.json"), sha256=sha(DATA / "index.json")),
                   train_frames=len(train), val_frames=len(val), args=vars(a), degradation="round(4x4 box mean), 8-bit sRGB" if a.degradation == "area" else "round(4x4 box mean in linear light), 8-bit sRGB encoded",
                   val=dict(bicubic_up=bic_psnr, initial=init_psnr, final=final_psnr,
                            per_frame=[dict(frame=r, initial=i, final=f) for (r, _), i, f in zip(vcrops, init_list, final_list)]),
                   curve=curve, seconds=time.time() - t0, code=dict(path=relpath(Path(__file__)),
                                                                     sha256=sha(Path(__file__))),
                   gpu=torch.cuda.get_device_name() if a.device == "cuda" else "cpu", torch=torch.__version__)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(dict(initial=init_psnr, final=final_psnr, bicubic_up=bic_psnr)), flush=True)


if __name__ == "__main__":
    main()
