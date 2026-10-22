"""v2 (10-21 ~09:25 PDT, before any result; addenda of 4dsr-25: dynamic-count-normalised photometric term, per-camera
read-outs for every arm). Per-frame absorbability upper bound of the main-recipe parent's dynamic misfit to its own training LR
(Claude 4dsr-f3, 2026-10-21; joint design with 4dsr-70; pre-registration tmp/claude_f3_joint_review_20261021/prereg_absorb.md).

Question: the w4 parent fits its training LR 7-9.5 dB worse in dynamic regions than in static ones
(tmp/claude_f3_joint_review_20261021/out/lr_residual_structure_*.json). Can extra per-frame degrees of freedom absorb that
legal information at all, and which kind (position / colour / all)? If even free per-frame residuals do not lift held-out
training views or cam00 in dynamic regions, no motion or appearance module can harvest this pool.

For each selected frame t the parent's deformed state S(t) (author deformation, post-hoc training-camera cull as the main
table) is frozen. A free set F_t = Gaussians whose projected centre falls in the dilated LR dynamic mask of >= 2 fitting
cameras receives per-frame residuals; each arm starts from the parent:
  P     position (px units of the free set's median depth, kNN-Laplacian smoothness + small shrink)
  C     SH-DC colour
  PCSO  position + colour + log-scale + opacity logit
Loss: L1 between D(fwd_c(R_c)) and y_c (the training LR; fwd_c = the checkpoint's camera model: warp field then colour
affine; D = 4x4 area average, clamp [0, 1]) over the fitting cameras = all training cameras except two held-out ones,
6 random fitting cameras per step, Adam, --steps steps.
Read-outs (before = parent, after = each arm): LR closure PSNR of the canonical residual A_c^-1 (y_c - D fwd_c(R_c)) on the
full image and in the dynamic mask, pooled separately over fitting and held-out cameras; cam00 HR PSNR on the full image and
in the evaluate.py dynamic region (|HR - B| > 0.06 max over RGB, B = median of cam00 HR over frames 0, 5, ..., elliptical
dilation 5 px). cam00 is DIAGNOSTIC ONLY: never used for fitting, the free set, held-out choice or any stopping rule.
LR dynamic mask (training cameras): max_ch |y~(t) - median_t y~| > 0.06, 3x3 dilation, y~ = A^-1 (y - b) (median over frames
0, 5, ..., 295); this is the definition of the CPU diagnostic. Moving proxy (added at 4dsr-70's request after its motion premise:
position at t is already within ~0.3-0.8 px, so report fast-changing content separately): training cameras
dyn & dilate3(max_ch |y~(t+2) - y~(t-2)| > 0.08); cam00 dyn0 & dilate5(max_ch |HR(t+2) - HR(t-2)| > 0.08).
Held-out rule (fixed before any result): rank training cameras by camera-centre distance to cam00, drop the 3 nearest, take
the cameras at ranks floor(m/2) and floor(m/2)+2 of the remaining m.

  absorb_bound.py --checkpoint CK --out DIR [--frames 30 65 ...] [--arms P C PCSO] [--steps 300] [--self-check-cache DIR]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CULL = ROOT / "experiments/dynamic_sr_cull_retrain_20261009"
GEO = ROOT / "experiments/dynamic_sr_modules_20261010/geoprobe_20261017"
if str(CULL) not in sys.path:
    sys.path.insert(0, str(CULL))

import cullretrain as cr  # noqa: E402  (runtime paths, torch, author code)
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402

native, ev, rt = cr.native, cr.ev, cr.rt
FRAMES = [30, 65, 100, 135, 170, 205, 240, 275]
ARMS = ("P", "C", "PCSO")


def camera_model(raw, device):
    """The checkpoint's per-camera model, rebuilt exactly as experiments/dynamic_sr_mvfb_20261020/cache_dump.py does."""
    if raw is None or "wfield" not in raw:
        raise SystemExit("checkpoint has no warp-field camera model (main recipe required)")
    src = (GEO / "modkit2.py").read_text()
    ns = dict(math=math, json=json, Path=Path, REG={})
    exec(src[src.index("def make_ext_affine("):src.index("def manifest_dt(")], ns)
    wf = raw["wfield"]
    m = ns["make_ext_affine"](cr.CameraAffine, None, int(wf.shape[-1]), 2e-3, 0.0)(list(raw["names"]), device=device)
    with torch.no_grad():
        m.wfield.copy_(wf.to(device=device, dtype=torch.float32))
        m.delta.copy_(raw["delta"].to(device=device, dtype=torch.float32))
    m.wfield.requires_grad_(False)
    m.delta.requires_grad_(False)
    return m


def load_model(path, plan, device):
    """cullretrain.load_for_render with an explicit device (CPU dry runs)."""
    from scene.gaussian_model import GaussianModel
    h, opt, pipe, sh = cr.configuration(plan)
    payload = torch.load(path, map_location=device, weights_only=False)
    g = GaussianModel(sh, h)
    (g.active_sh_degree, xyz, deform_state, g._deformation_table, f_dc, f_rest, scaling, rotation, opacity,
     g.max_radii2D, g.xyz_gradient_accum, g.denom, _opt, g.spatial_lr_scale) = payload["model"]
    g._deformation.load_state_dict(deform_state)
    g._deformation = g._deformation.to(device).eval()
    for name, t in (("_xyz", xyz), ("_features_dc", f_dc), ("_features_rest", f_rest), ("_scaling", scaling),
                    ("_rotation", rotation), ("_opacity", opacity)):
        setattr(g, name, torch.nn.Parameter(t.detach().clone().float().to(device), requires_grad=False))
    for p in g._deformation.parameters():
        p.requires_grad_(False)
    g._deformation_accum = payload.get("deformation_accum")
    return g


def area(img, size):
    h, w = size
    k = img.shape[-1] // w
    if img.shape[-2] != h * k or img.shape[-1] != w * k:
        raise ValueError("HR size is not an integer multiple of the LR size")
    return F.avg_pool2d(img[None], k)[0].clamp(0, 1)


def ellipse(r, device):
    y, x = torch.meshgrid(torch.arange(-r, r + 1, device=device), torch.arange(-r, r + 1, device=device), indexing="ij")
    return ((x.float() / (r + 0.5)) ** 2 + (y.float() / (r + 0.5)) ** 2 <= 1.0).float()


def dilate(mask, kernel):
    r = kernel.shape[-1] // 2
    return F.conv2d(mask[None, None].float(), kernel[None, None], padding=r)[0, 0] > 0


def read_rgb(path, device):
    with Image.open(path) as im:
        return torch.from_numpy(np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0).permute(2, 0, 1).to(device)


def knn(x, k=8, chunk=4096):
    idx = []
    for s in range(0, x.shape[0], chunk):
        d = torch.cdist(x[s:s + chunk], x)
        idx.append(d.topk(k + 1, largest=False).indices[:, 1:])
    return torch.cat(idx)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--frames", type=int, nargs="*", default=FRAMES)
    p.add_argument("--arms", nargs="*", default=list(ARMS))
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--batch", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dry", action="store_true", help="CPU-safe setup check: load, cull, masks, free sets; no rendering")
    p.add_argument("--self-check-cache", default=None, help="mvfb cache dir of the same checkpoint: parent residuals must match")
    a = p.parse_args()
    dev = a.device
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    if (out / "summary.json").exists():
        raise FileExistsError(f"completed run preserved: {out}")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    started = time.time()
    ck = ROOT / a.checkpoint
    payload = torch.load(ck, map_location="cpu", weights_only=False)
    plan, raw_aff = payload["plan"], payload.get("affine")
    del payload
    manifest_path, manifest = cr.plan_scene(plan)
    root = Path(manifest["_root"])
    train = list(manifest["splits"]["train"])
    model = camera_model(raw_aff, dev)
    g = load_model(ck, plan, dev)
    h, opt, pipe, sh = cr.configuration(plan)
    record = dict(kind="absorb_bound", checkpoint=cr.entry(ck), manifest=cr.entry(manifest_path), args=vars(a),
                  points_loaded=int(g.get_xyz.shape[0]), host=__import__("socket").gethostname(),
                  gpu=torch.cuda.get_device_name() if dev == "cuda" else "cpu",
                  sources_code={cr.rel(q): rt.sha(q) for q in (Path(__file__), Path(cr.__file__), GEO / "modkit2.py")})
    record["cull_record"] = cr.cull(g, cr.Culler(manifest, device=dev), None, "posthoc", optimizer_backed=False)
    cams = manifest["cameras"]
    centre = lambda c: np.asarray(cams[c]["c2w"], np.float64)[:3, 3]
    ranked = [c for _, c in sorted((float(np.linalg.norm(centre(c) - centre("cam00"))), c) for c in train)]
    rest = ranked[3:]
    m = len(rest)
    held = [rest[m // 2], rest[m // 2 + 2]]
    fit = [c for c in train if c not in held]
    record.update(held_out=held, fitting=fit, ranked_by_cam00_distance=ranked)
    lr_w, lr_h = manifest["resolutions"]["lr"]
    hr_w, hr_h = manifest["resolutions"]["hr"]
    obs = {(o["camera_id"], int(o["frame_index"])): o for o in manifest["observations"]}
    index = {c: i for i, c in enumerate(train)}
    names = list(raw_aff["names"])
    delta = raw_aff["delta"].double() - raw_aff["delta"].double().mean(0)
    Ainv = {c: torch.linalg.inv(torch.eye(3, dtype=torch.float64) + delta[names.index(c), :, :3]).float().to(dev) for c in train}
    bvec = {c: delta[names.index(c), :, 3].float().to(dev) for c in train}
    canon = lambda c, y: torch.einsum("ij,jhw->ihw", Ainv[c], y - bvec[c][:, None, None])
    k3 = torch.ones(3, 3, device=dev)
    med = {c: torch.stack([canon(c, read_rgb(root / obs[(c, f)]["lr_path"], dev)) for f in range(0, 300, 5)]).median(0).values
           for c in train}
    hr_bg = torch.stack([read_rgb(root / obs[("cam00", f)]["hr_path"], dev) for f in range(0, 300, 5)]).median(0).values
    k5 = ellipse(5, dev)
    background = torch.ones(3, device=dev) if cr.white_background(plan) else torch.zeros(3, device=dev)
    fx = float(np.asarray(cams[fit[0]]["K_hr"])[0][0])
    rows = []

    def camera(c, f):
        return native.hr_camera(ev.camera_item(manifest, obs[(c, f)], torch), index.get(c, 0), manifest)

    def render(cam, state):
        return native.native_rgb(cam, g, state, pipe, background)["render"]

    def readout(state, f, cams_, ys, masks, movs, hr0, dyn0, mov0, per_camera=False):
        res = {"per_camera_psnr": {}} if per_camera else {}
        ps = lambda se_, n_: -10 * math.log10(max(se_, 1e-12) / max(n_, 1))
        with torch.no_grad():
            for group, cs in (("fit", fit), ("held", held)):
                se = se_d = se_m = n = n_d = n_m = 0.0
                for c in cs:
                    e = canon(c, ys[c]) - canon(c, area(model.apply(render(cams_[c], state), c), (lr_h, lr_w)))
                    e2 = (e ** 2).sum(0)
                    se += float(e2.sum()); n += e2.numel() * 3
                    se_d += float(e2[masks[c]].sum()); n_d += int(masks[c].sum()) * 3
                    se_m += float(e2[movs[c]].sum()); n_m += int(movs[c].sum()) * 3
                    if per_camera:
                        res["per_camera_psnr"][c] = dict(psnr=-10 * math.log10(float(e2.mean()) / 3),
                                                         dyn_psnr=-10 * math.log10(max(float(e2[masks[c]].sum()), 1e-12) / (3 * max(int(masks[c].sum()), 1))),
                                                         mov_psnr=-10 * math.log10(max(float(e2[movs[c]].sum()), 1e-12) / (3 * max(int(movs[c].sum()), 1))))
                res[group] = dict(psnr=ps(se, n), dyn_psnr=ps(se_d, n_d), mov_psnr=ps(se_m, n_m), mov_pixels=n_m / 3)
            x = render(cams_["cam00"], state).clamp(0, 1)
            e2 = ((x - hr0) ** 2).sum(0)
            res["cam00"] = dict(psnr=ps(float(e2.sum()), e2.numel() * 3), dyn_psnr=ps(float(e2[dyn0].sum()), 3 * int(dyn0.sum())),
                                mov_psnr=ps(float(e2[mov0].sum()), 3 * int(mov0.sum())), mov_pixels=int(mov0.sum()))
        return res

    def build(S, free, prm, px):
        st = dict(S)
        if "mu" in prm:
            st["xyz"] = S["xyz"].index_add(0, free, prm["mu"] * px)
        if "c" in prm:
            sh_ = S["sh"].clone()
            sh_[free, 0, :] = sh_[free, 0, :] + prm["c"]
            st["sh"] = sh_
        if "s" in prm:
            st["scales"] = S["scales"].index_put((free,), S["scales"][free] * torch.exp(prm["s"]))
        if "o" in prm:
            o = S["opacity"][free].clamp(1e-6, 1 - 1e-6)
            st["opacity"] = S["opacity"].index_put((free,), torch.sigmoid(torch.log(o / (1 - o)) + prm["o"]))
        return st

    if a.self_check_cache and not a.dry:
        cdir = ROOT / a.self_check_cache
        f = 30
        with np.load(cdir / f"f{f:04d}.npz") as z:
            srcs, eps_ref = list(z["sources"]), torch.from_numpy(z["src_eps"].astype(np.float32)).to(dev)
        with torch.no_grad():
            S = native.native_effective_state(g, float(obs[(srcs[0], f)]["time"]), torch)
            diffs = []
            for i, c in enumerate(srcs):
                y = read_rgb(root / obs[(c, f)]["lr_path"], dev)
                e = canon(c, y) - canon(c, area(model.apply(render(camera(c, f), S), c), (lr_h, lr_w)))
                diffs.append(float((e - eps_ref[i]).abs().max()))
        record["self_check"] = dict(cache=a.self_check_cache, frame=f, max_abs_diff=diffs)
        if max(diffs) > 5e-3:
            raise SystemExit(f"self-check failed: parent residuals differ from the cache by {max(diffs):.4f}")
        print("self-check ok", diffs, flush=True)

    for f in a.frames:
        t0 = time.time()
        time_value = float(obs[(train[0], f)]["time"])
        with torch.no_grad():
            S = {k: v.detach() for k, v in native.native_effective_state(g, time_value, torch).items()}
        ys = {c: read_rgb(root / obs[(c, f)]["lr_path"], dev) for c in train}
        masks = {c: dilate((canon(c, ys[c]) - med[c]).abs().amax(0) > 0.06, k3) for c in train}
        movs = {c: masks[c] & dilate((canon(c, read_rgb(root / obs[(c, min(f + 2, 299))]["lr_path"], dev))
                                      - canon(c, read_rgb(root / obs[(c, max(f - 2, 0))]["lr_path"], dev))).abs().amax(0) > 0.08, k3)
                for c in train}
        cams_ = {c: camera(c, f) for c in train + ["cam00"]}
        count = torch.zeros(S["xyz"].shape[0], dtype=torch.int32, device=dev)
        depth = []
        for c in fit:
            P = cams_[c].full_proj_transform.to(dev)
            ph = torch.cat([S["xyz"], torch.ones_like(S["xyz"][:, :1])], 1) @ P
            wv = torch.cat([S["xyz"], torch.ones_like(S["xyz"][:, :1])], 1) @ cams_[c].world_view_transform.to(dev)
            ndc = ph[:, :2] / ph[:, 3:4].clamp_min(1e-6)
            px_ = ((ndc[:, 0] + 1) * hr_w - 1) * 0.5
            py_ = ((ndc[:, 1] + 1) * hr_h - 1) * 0.5
            lx = torch.floor((px_ + 0.5) / (hr_w // lr_w)).long()
            ly = torch.floor((py_ + 0.5) / (hr_h // lr_h)).long()
            ok = (wv[:, 2] > 0.2) & (lx >= 0) & (lx < lr_w) & (ly >= 0) & (ly < lr_h)
            hit = torch.zeros_like(ok)
            hit[ok] = masks[c][ly[ok], lx[ok]]
            count += hit.int()
            depth.append(wv[hit, 2])
        free = torch.nonzero(count >= 2).squeeze(1)
        px = float(torch.cat(depth).median()) / fx if free.numel() else 0.0
        hr0 = read_rgb(root / obs[("cam00", f)]["hr_path"], dev)
        dyn0 = dilate((hr0 - hr_bg).abs().amax(0) > 0.06, k5)
        mov0 = dyn0 & dilate((read_rgb(root / obs[("cam00", min(f + 2, 299))]["hr_path"], dev)
                              - read_rgb(root / obs[("cam00", max(f - 2, 0))]["hr_path"], dev)).abs().amax(0) > 0.08, k5)
        row = dict(frame=f, time=time_value, free=int(free.numel()), px_world=px,
                   lr_dyn_fraction={c: float(masks[c].float().mean()) for c in train},
                   lr_mov_fraction={c: float(movs[c].float().mean()) for c in train},
                   cam00_dyn_fraction=float(dyn0.float().mean()), cam00_mov_fraction=float(mov0.float().mean()))
        if a.dry:
            print(json.dumps(row)[:400], flush=True)
            rows.append(row)
            continue
        row["parent"] = readout(S, f, cams_, ys, masks, movs, hr0, dyn0, mov0, per_camera=True)
        nb = knn(S["xyz"][free]) if free.numel() > 16 else None
        for arm in a.arms:
            ta = time.time()
            prm, groups = {}, []
            if "P" in arm:
                prm["mu"] = torch.zeros(free.numel(), 3, device=dev, requires_grad=True)
                groups.append(dict(params=[prm["mu"]], lr=0.05))
            if "C" in arm:
                prm["c"] = torch.zeros(free.numel(), 3, device=dev, requires_grad=True)
                groups.append(dict(params=[prm["c"]], lr=2.5e-3))
            if "S" in arm:
                prm["s"] = torch.zeros(free.numel(), 3, device=dev, requires_grad=True)
                groups.append(dict(params=[prm["s"]], lr=5e-3))
            if "O" in arm:
                prm["o"] = torch.zeros(free.numel(), 1, device=dev, requires_grad=True)
                groups.append(dict(params=[prm["o"]], lr=5e-2))
            optim = torch.optim.Adam(groups, eps=1e-15)
            trace = []
            for it in range(a.steps):
                st = build(S, free, prm, px)
                batch = rng.choice(len(fit), size=min(a.batch, len(fit)), replace=False)
                loss = 0.0
                for j in batch:
                    c = fit[j]
                    # full-frame L1 SUM normalised by the camera's dynamic-pixel count (v2, 4dsr-25 addendum 1): static pixels still
                    # count, but the dynamic gradient is no longer diluted ~1/dyn_fraction against the regularisers
                    loss = loss + (area(model.apply(render(cams_[c], st), c), (lr_h, lr_w)) - ys[c]).abs().sum() / (3 * max(int(masks[c].sum()), 1))
                loss = loss / len(batch)
                reg = 0.0
                if "mu" in prm:
                    u = prm["mu"]
                    if nb is not None:
                        reg = reg + 1e-3 * ((u - u[nb].mean(1)) ** 2).sum(1).mean()
                    reg = reg + 1e-5 * (u ** 2).sum(1).mean()
                if "s" in prm:
                    reg = reg + 1e-4 * (prm["s"] ** 2).sum(1).mean()
                if "o" in prm:
                    reg = reg + 1e-4 * (prm["o"] ** 2).mean()
                total = loss + reg
                optim.zero_grad(set_to_none=True)
                total.backward()
                optim.step()
                if it % 50 == 0 or it == a.steps - 1:
                    trace.append((it, float(loss), float(reg) if torch.is_tensor(reg) else reg))
            with torch.no_grad():
                st = build(S, free, {k: v.detach() for k, v in prm.items()}, px)
                res = readout(st, f, cams_, ys, masks, movs, hr0, dyn0, mov0, per_camera=True)
                stats = {}
                if "mu" in prm:
                    mag = prm["mu"].norm(dim=1)
                    stats["mu_px"] = dict(median=float(mag.median()), p90=float(mag.quantile(0.9)), max=float(mag.max()))
                if "c" in prm:
                    stats["c_abs_mean"] = float(prm["c"].abs().mean())
            row[arm] = dict(res, trace=trace, stats=stats, seconds=time.time() - ta)
            print(f"frame {f} arm {arm}: fit dyn {row['parent']['fit']['dyn_psnr']:.2f}->{res['fit']['dyn_psnr']:.2f} "
                  f"held mov {row['parent']['held']['mov_psnr']:.2f}->{res['held']['mov_psnr']:.2f} "
                  f"held dyn {row['parent']['held']['dyn_psnr']:.2f}->{res['held']['dyn_psnr']:.2f} "
                  f"cam00 dyn {row['parent']['cam00']['dyn_psnr']:.2f}->{res['cam00']['dyn_psnr']:.2f} "
                  f"({time.time() - ta:.0f}s)", flush=True)
        row["seconds"] = time.time() - t0
        rows.append(row)
        (out / "rows.jsonl").open("a").write(json.dumps(row) + "\n")
    summary = dict(record, rows=rows, seconds=time.time() - started,
                   peak_GB=torch.cuda.max_memory_allocated() / 1e9 if dev == "cuda" else None)
    if not a.dry:
        agg = {}
        for arm in ["parent"] + list(a.arms):
            for key in ("fit", "held", "cam00"):
                for mtr in ("psnr", "dyn_psnr", "mov_psnr"):
                    v = [r[arm][key][mtr] for r in rows if arm in r]
                    agg.setdefault(arm, {}).setdefault(key, {})[mtr] = float(np.mean(v)) if v else None
        summary["mean_over_frames"] = agg
    tmp = out / ("summary.dry.json" if a.dry else "summary.json.tmp")
    tmp.write_text(json.dumps(summary, indent=1))
    if not a.dry:
        tmp.replace(out / "summary.json")
    print(f"done in {time.time() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
