"""Bounded HAT-L teacher screening for the area4 protocol (Claude 4dsr-7c, 2026-10-18; pre-registration in README "第十四波").

  hat_screen.py check  --out OUT.json [--max-frames N]          CPU: strict load + functional bicubic x4 check vs SwinIR-M
  hat_screen.py bench  --out OUT.json [--batch 6] [--iters 30]   GPU: training step time / peak memory at 64x64 LR crops, and
                                                                  full-frame (336x252 LR) inference time
  hat_screen.py teach  --out OUT.json --hat-ft DIR --swinir-ft DIR [--device cuda] [--faces FACES.json] [--smoke]
                                                                  teacher-level comparison of the two area-matched teachers on
                                                                  fixed training-view frames (Cut07 gating, Cook07 reported),
                                                                  the external validation crops, and a consecutive-frame check;
                                                                  fixed regions (HR-decided detail windows + registered face
                                                                  windows) with crops saved to OUT_crops/ for blind review

Weights: the official HAT-L x4 ImageNet-pretrain file placed by Codex (weight/HAT/official_20261018, sha256 5992bd38...), loaded with
weights_only=True and strict=True through hat_arch_local.py (a local re-implementation of the official module). SwinIR-M classical x4
(DF2K) is the reference. Training cameras only; no test-camera pixels are read. TF32 off.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
HAT_W = (ROOT / "weight/HAT/official_20261018/HAT-L_SRx4_ImageNet-pretrain.pth",
         "5992bd38522f2b8faf11ea4bd8ee08de92465bb66892166576999afc36d60043")
SWINIR_DIR = ROOT / "third_party/swinir_frozen_20261010"
SWINIR_W = (SWINIR_DIR / "001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth",
            "4e78e33f22c1aa8a773db0cf4a7381bae97c2362c717f155439ebc690cbd9215")
DATA = ROOT / "data/dynamic_sr/area_teacher_20261017"
PREP = ROOT / "data/dynamic_sr/full_time_20261007/prepared"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def psnr(mse):
    return float(-10 * np.log10(max(float(mse), 1e-12)))


def bands(err):
    h, w = err.shape[:2]
    r = np.sqrt(np.fft.fftfreq(h)[:, None] ** 2 + np.fft.fftfreq(w)[None] ** 2)
    spec = np.fft.fft2(err.astype(np.float64), axes=(0, 1))
    return [np.real(np.fft.ifft2(spec * m[..., None], axes=(0, 1))) for m in (r < .125, (r >= .125) & (r < .25), r >= .25)]


def scores(img, hr):
    e = img - hr
    out = dict(psnr=psnr((e ** 2).mean()))
    for k, b in zip(("low", "mid", "high"), bands(e)):
        out[f"{k}_psnr"] = psnr((b ** 2).mean())
    return out


def hat_net(device, use_checkpoint=False, weights=None):
    import hat_arch_local as H
    w = weights or HAT_W
    if sha(w[0]) != w[1]:
        raise SystemExit("HAT weights differ")
    net = H.build_hat_l(use_checkpoint)
    H.load_state(net, w[0], "params_ema" if weights is None else "params")
    return net.to(device).eval()


def swinir_net(device, weights=None):
    sys.path.insert(0, str(SWINIR_DIR))
    import torch
    from network_swinir import SwinIR
    w = weights or SWINIR_W
    if sha(w[0]) != w[1]:
        raise SystemExit("SwinIR weights differ")
    net = SwinIR(upscale=4, in_chans=3, img_size=64, window_size=8, img_range=1.0, depths=[6] * 6, embed_dim=180,
                 num_heads=[6] * 6, mlp_ratio=2, upsampler="pixelshuffle", resi_connection="1conv")
    st = torch.load(w[0], map_location="cpu", weights_only=True)
    net.load_state_dict(st["params"] if "params" in st else st, strict=True)
    return net.to(device).eval()


def run_swinir(net, x):
    """teacher_physlr3.py padding (flip-pad to the next multiple of 8 beyond the size), crop."""
    import torch
    h, w = x.shape[2:]
    hp, wp = (h // 8 + 1) * 8 - h, (w // 8 + 1) * 8 - w
    x = torch.cat([x, torch.flip(x, [2])], 2)[:, :, :h + hp, :]
    x = torch.cat([x, torch.flip(x, [3])], 3)[:, :, :, :w + wp]
    return net(x)[..., :h * 4, :w * 4]


def run_hat(net, x):
    import hat_arch_local as H
    return H.infer(net, x)


def val_crops(max_frames=0):
    from PIL import Image
    index = json.loads((DATA / "index.json").read_text())
    crops = []
    for rel, e in sorted(index["frames"].items()):
        if not rel.endswith("_5.png"):
            continue
        img = np.asarray(Image.open(DATA / rel).convert("RGB"))
        if hashlib.sha256((DATA / rel).read_bytes()).hexdigest() != e["sha256"]:
            raise SystemExit(f"frame changed: {rel}")
        h, w = img.shape[:2]
        y, x = ((h - 256) // 2) // 4 * 4, ((w - 256) // 2) // 4 * 4
        crops.append((rel, np.ascontiguousarray(img[y:y + 256, x:x + 256])))
    return crops[:max_frames] if max_frames else crops


def area_lr(hr_u8):
    """hr_u8: (B,3,H,W) uint8-valued float -> round(4x4 mean) / 255, as finetune_swinir_area.py / physlr.py area4."""
    import torch.nn.functional as F
    import torch
    return torch.round(F.avg_pool2d(hr_u8, 4)) / 255.0


def q8(y):
    import torch
    return torch.round(y.clamp(0, 1) * 255) / 255


def detail_windows(hr, k=3, size=128, stride=64):
    """Pre-registered detail regions: the k non-overlapping size x size windows (top-left on a stride grid) with the highest HR luma
    Laplacian energy. Decided by the HR frame only."""
    y = hr @ np.array([0.299, 0.587, 0.114])
    lap = np.zeros_like(y)
    lap[1:-1, 1:-1] = 4 * y[1:-1, 1:-1] - y[:-2, 1:-1] - y[2:, 1:-1] - y[1:-1, :-2] - y[1:-1, 2:]
    ii = np.pad((lap ** 2).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    h, w = y.shape
    cand = sorted(((ii[y0 + size, x0 + size] - ii[y0, x0 + size] - ii[y0 + size, x0] + ii[y0, x0], x0, y0)
                   for y0 in range(0, h - size + 1, stride) for x0 in range(0, w - size + 1, stride)), reverse=True)
    out = []
    for _, x0, y0 in cand:
        if all(abs(x0 - b[0]) >= size or abs(y0 - b[1]) >= size for b in out):
            out.append((int(x0), int(y0), int(x0 + size), int(y0 + size)))
        if len(out) == k:
            break
    return out


def save_png(path, img):
    from PIL import Image
    Image.fromarray(np.clip(np.round(img * 255), 0, 255).astype(np.uint8)).save(path)


def cmd_check(a):
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(16)
    t0 = time.time()
    hat = hat_net("cpu")
    import hat_arch_local as H
    oca = hat.relative_position_index_OCA
    rec = dict(strict_load="passed", params=int(sum(p.numel() for p in hat.parameters())),
               rpi_oca_range=[int(oca.min()), int(oca.max())],
               rpi_oca_matches_local_formula=bool(torch.equal(oca, H.build_hat_l().calculate_rpi_oca())),
               rpi_sa_matches_local_formula=bool(torch.equal(hat.relative_position_index_SA, H.build_hat_l().calculate_rpi_sa())))
    sw = swinir_net("cpu")
    rows = []
    for rel, c in val_crops(a.max_frames):
        hr = torch.from_numpy(c).permute(2, 0, 1)[None].float() / 255.0
        lr = F.interpolate(hr, scale_factor=0.25, mode="bicubic", align_corners=False, antialias=True).clamp(0, 1)
        lr = torch.round(lr * 255) / 255
        with torch.no_grad():
            yh, ys = q8(run_hat(hat, lr)), q8(run_swinir(sw, lr))
        bic = q8(F.interpolate(lr, scale_factor=4, mode="bicubic", align_corners=False))
        rows.append(dict(frame=rel, hat=psnr(((yh - hr) ** 2).mean()), swinir=psnr(((ys - hr) ** 2).mean()),
                         bicubic=psnr(((bic - hr) ** 2).mean())))
        print(f"{time.time() - t0:6.0f}s {rel}: HAT-L {rows[-1]['hat']:.3f} SwinIR-M {rows[-1]['swinir']:.3f} bicubic {rows[-1]['bicubic']:.3f}",
              flush=True)
    d = [r["hat"] - r["swinir"] for r in rows]
    rec.update(functional_bicubic=dict(frames=len(rows), hat=float(np.mean([r["hat"] for r in rows])),
                                       swinir=float(np.mean([r["swinir"] for r in rows])),
                                       bicubic=float(np.mean([r["bicubic"] for r in rows])), hat_minus_swinir_mean=float(np.mean(d)),
                                       hat_better_share=float(np.mean([x > 0 for x in d]))), rows=rows, seconds=time.time() - t0)
    return rec


def cmd_bench(a):
    """Training step time / peak memory at batch --batch with 64x64 LR crops, without and with gradient checkpointing (an
    out-of-memory setting is recorded, not fatal), then full-frame inference time (252x336 LR) for HAT-L and SwinIR-M."""
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # 16 GiB cap (README 第十四波): an over-budget setting fails inside this process instead of starving the concurrent LR worker
    frac = min(1.0, 16 * 2 ** 30 / torch.cuda.get_device_properties(0).total_memory)
    torch.cuda.set_per_process_memory_fraction(frac)
    out = dict(gpu=torch.cuda.get_device_name(), batch=a.batch, iters=a.iters, memory_fraction=frac, train={})
    for ck in (False, True):
        net = opt = x = y = loss = None
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            net = hat_net("cuda", use_checkpoint=ck)
            net.train()
            opt = torch.optim.Adam(net.parameters(), lr=1e-7, betas=(0.9, 0.99))
            x = torch.rand(a.batch, 3, 64, 64, device="cuda")
            y = torch.rand(a.batch, 3, 256, 256, device="cuda")
            times = []
            for i in range(a.iters):
                torch.cuda.synchronize()
                t = time.time()
                loss = (net(x) - y).abs().mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                torch.cuda.synchronize()
                times.append(time.time() - t)
            out["train"]["checkpoint" if ck else "plain"] = dict(s_per_iter=float(np.median(times[5:])),
                                                                 peak_GB=torch.cuda.max_memory_allocated() / 2 ** 30)
        except torch.cuda.OutOfMemoryError as e:
            out["train"]["checkpoint" if ck else "plain"] = dict(oom=str(e)[:200])
        del net, opt, x, y, loss
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    net = hat_net("cuda")
    lr = torch.rand(1, 3, 252, 336, device="cuda")
    ti = []
    with torch.no_grad():
        for i in range(6):
            torch.cuda.synchronize()
            t = time.time()
            run_hat(net, lr)
            torch.cuda.synchronize()
            ti.append(time.time() - t)
    out.update(infer_s_per_frame=float(np.median(ti[1:])), infer_peak_GB=torch.cuda.max_memory_allocated() / 2 ** 30)
    del net
    sw = swinir_net("cuda")
    ts = []
    with torch.no_grad():
        for i in range(6):
            torch.cuda.synchronize()
            t = time.time()
            run_swinir(sw, lr)
            torch.cuda.synchronize()
            ts.append(time.time() - t)
    out.update(swinir_infer_s_per_frame=float(np.median(ts[1:])),
               cut07_cache_h_estimate=out["infer_s_per_frame"] * 6000 / 3600)
    print(json.dumps(out, indent=1), flush=True)
    return out


def cmd_teach(a):
    import torch
    import lpips
    from PIL import Image
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dev = a.device
    hs = json.loads((ROOT / a.hat_ft / "summary.json").read_text())
    ss = json.loads((ROOT / a.swinir_ft / "summary.json").read_text())
    hat = hat_net(dev, weights=(ROOT / hs["weights"]["path"], hs["weights"]["sha256"]))
    sw = swinir_net(dev, weights=(ROOT / ss["weights"]["path"], ss["weights"]["sha256"]))
    # AlexNet backbone of LPIPS from the runtime copy, as rt.check_runtime() (the training service has no network; 10-18 16:18 the
    # first teach run failed trying to download it)
    torch.hub.set_dir(str(ROOT / "third_party/4dgs_843d5ac_runtime/torch_hub"))
    lp = lpips.LPIPS(net="alex", verbose=False).to(dev).eval()

    def lpips_of(y, hr):
        with torch.no_grad():
            return float(lp(y * 2 - 1, hr * 2 - 1).mean())

    out = dict(hat_ft=dict(dir=a.hat_ft, weights=hs["weights"], val=hs.get("val")), swinir_ft=dict(dir=a.swinir_ft, weights=ss["weights"],
                                                                                                   val=ss.get("val")))
    faces = json.loads(Path(a.faces).read_text()) if a.faces else {}
    if a.faces:
        out["faces"] = dict(path=a.faces, sha256=sha(a.faces))
    crop_dir = (Path(a.out) if Path(a.out).is_absolute() else ROOT / a.out).with_suffix("")
    crop_dir = crop_dir.parent / (crop_dir.name + "_crops")
    crop_dir.mkdir(parents=True, exist_ok=True)
    out["crops_dir"] = str(crop_dir.relative_to(ROOT))
    # 1) external validation crops (same crops and LR for both)
    vr = []
    for rel, c in val_crops(2 if a.smoke else 0):
        hr = torch.from_numpy(c).permute(2, 0, 1)[None].float().to(dev)
        lr = area_lr(hr)
        hr = hr / 255.0
        with torch.no_grad():
            yh, ys = q8(run_hat(hat, lr)), q8(run_swinir(sw, lr))
        vr.append(dict(frame=rel, hat=psnr(((yh - hr) ** 2).mean().item()), swinir=psnr(((ys - hr) ** 2).mean().item())))
    out["external_val"] = dict(frames=len(vr), hat=float(np.mean([r["hat"] for r in vr])), swinir=float(np.mean([r["swinir"] for r in vr])),
                               rows=vr)
    # 2) fixed training-view frames of dev scenes (area4 LR as registered), full frames
    t_inf = {"hat": [], "swinir": []}
    for scene, gate in (("cut_roasted_beef", True), ("cook_spinach", False)):
        sdir = PREP / f"{scene}_area4"
        m = json.loads((sdir / "manifest.json").read_text())
        train = sorted(m["splits"]["train"])
        if "cam00" in train:
            raise SystemExit("test camera in train split")
        cams = train[:: max(1, len(train) // 6)][:6]
        cams, fr = (cams[:1], (0,)) if a.smoke else (cams, (0, 299))
        obs = {(o["camera_id"], int(o["frame_index"])): o for o in m["observations"]}
        rows = []
        for cam in cams:
            for f in fr:
                o = obs[(cam, f)]
                if sha(sdir / o["lr_path"]) != o["lr_sha256"]:
                    raise SystemExit("LR changed")
                lr = torch.from_numpy(np.asarray(Image.open(sdir / o["lr_path"]).convert("RGB")).astype(np.float32) / 255.0)
                lr = lr.permute(2, 0, 1)[None].to(dev)
                hr_np = np.asarray(Image.open(sdir / o["hr_path"]).convert("RGB")).astype(np.float64) / 255.0
                hr = torch.from_numpy(hr_np.astype(np.float32)).permute(2, 0, 1)[None].to(dev)
                r = dict(camera=cam, frame=f)
                yt, yn = {}, {}
                for name, net, fn in (("hat", hat, run_hat), ("swinir", sw, run_swinir)):
                    torch.cuda.synchronize() if dev == "cuda" else None
                    t = time.time()
                    with torch.no_grad():
                        yt[name] = q8(fn(net, lr))
                    torch.cuda.synchronize() if dev == "cuda" else None
                    t_inf[name].append(time.time() - t)
                    yn[name] = yt[name][0].permute(1, 2, 0).double().cpu().numpy()
                    r[name] = scores(yn[name], hr_np)
                    r[name]["lpips"] = lpips_of(yt[name], hr)
                # fixed regions: HR-decided detail windows, plus the face window registered before teach ran (if any)
                rois = [(f"detail{i}", b) for i, b in enumerate(detail_windows(hr_np))]
                fb = faces.get(scene, {}).get(cam, {}).get(str(f))
                if fb:
                    if min(fb[2] - fb[0], fb[3] - fb[1]) < 64:
                        raise SystemExit(f"face window below 64 px: {scene} {cam} {f}")
                    rois.append(("face", tuple(int(v) for v in fb)))
                r["rois"] = []
                for rn, (x0, y0, x1, y1) in rois:
                    rr = dict(roi=rn, box=[x0, y0, x1, y1])
                    stem = f"{scene}_{cam}_{f:04d}_{rn}"
                    save_png(crop_dir / f"{stem}_hr.png", hr_np[y0:y1, x0:x1])
                    for name in ("hat", "swinir"):
                        rr[name] = dict(psnr=psnr(((yn[name][y0:y1, x0:x1] - hr_np[y0:y1, x0:x1]) ** 2).mean()),
                                        lpips=lpips_of(yt[name][..., y0:y1, x0:x1], hr[..., y0:y1, x0:x1]))
                        save_png(crop_dir / f"{stem}_{name}.png", yn[name][y0:y1, x0:x1])
                    r["rois"].append(rr)
                rows.append(r)
                print(f"{scene} {cam} {f}: HAT {r['hat']['psnr']:.3f} SwinIR {r['swinir']['psnr']:.3f}", flush=True)
        keys = ("psnr", "low_psnr", "mid_psnr", "high_psnr", "lpips")
        summ = {k: dict(hat=float(np.mean([r["hat"][k] for r in rows])), swinir=float(np.mean([r["swinir"][k] for r in rows])),
                        hat_minus_swinir=float(np.mean([r["hat"][k] - r["swinir"][k] for r in rows]))) for k in keys}
        for kind in ("detail", "face"):
            v = [x for r in rows for x in r["rois"] if x["roi"].startswith(kind)]
            summ[f"roi_{kind}"] = dict(n=len(v), **{m: dict(hat_minus_swinir=float(np.mean([x["hat"][m] - x["swinir"][m] for x in v])))
                                                    for m in ("psnr", "lpips") if v})
        out[scene] = dict(gating=gate, cams=cams, frames=list(fr), rows=rows, summary=summ)
    # 3) consecutive frames (temporal): error of the frame-to-frame difference, Cut07 two cameras, frames 149 / 150
    sdir = PREP / "cut_roasted_beef_area4"
    m = json.loads((sdir / "manifest.json").read_text())
    obs = {(o["camera_id"], int(o["frame_index"])): o for o in m["observations"]}
    tr = []
    for cam in sorted(m["splits"]["train"])[:1 if a.smoke else 2]:
        ys, hs_ = {}, {}
        for f in (149, 150):
            o = obs[(cam, f)]
            lr = torch.from_numpy(np.asarray(Image.open(sdir / o["lr_path"]).convert("RGB")).astype(np.float32) / 255.0).permute(2, 0, 1)[None].to(dev)
            hs_[f] = np.asarray(Image.open(sdir / o["hr_path"]).convert("RGB")).astype(np.float64) / 255.0
            with torch.no_grad():
                ys[f] = {n: q8(fn(net, lr))[0].permute(1, 2, 0).double().cpu().numpy() for n, net, fn in
                         (("hat", hat, run_hat), ("swinir", sw, run_swinir))}
        dh = hs_[150] - hs_[149]
        tr.append(dict(camera=cam, **{n: psnr((((ys[150][n] - ys[149][n]) - dh) ** 2).mean()) for n in ("hat", "swinir")}))
    out["temporal_diff_psnr"] = tr
    out["inference_s_per_frame"] = {k: float(np.median(v[1:])) for k, v in t_inf.items()}
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=("check", "bench", "teach"))
    p.add_argument("--out", required=True)
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--batch", type=int, default=6)
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--checkpoint", action="store_true", help="bench: gradient checkpointing per residual group")
    p.add_argument("--hat-ft")
    p.add_argument("--swinir-ft")
    p.add_argument("--device", default="cuda")
    p.add_argument("--faces", help="teach: JSON {scene: {camera: {frame: [x0, y0, x1, y1]}}} in HR pixels, fixed before teach ran")
    p.add_argument("--smoke", action="store_true", help="teach: 2 external crops, 1 camera x frame 0 per scene, 1 temporal camera")
    a = p.parse_args()
    rec = dict(check=cmd_check, bench=cmd_bench, teach=cmd_teach)[a.mode](a)
    rec["source"] = {str(Path(f).resolve().relative_to(ROOT)): sha(f) for f in (__file__, HERE / "hat_arch_local.py")}
    out = Path(a.out)
    out = out if out.is_absolute() else ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=1))
    print(json.dumps({k: v for k, v in rec.items() if k not in ("rows",)}, indent=1, default=str)[:3000], flush=True)


if __name__ == "__main__":
    main()
