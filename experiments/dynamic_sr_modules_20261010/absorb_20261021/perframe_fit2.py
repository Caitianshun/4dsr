"""LR-anchored per-frame dynamic refinement of a main-recipe (w4) parent (Claude 4dsr-f3, 2026-10-21; joint design with 4dsr-25;
pre-registration tmp/claude_f3_joint_review_20261021/prereg_perframe.md). Full-frame version of the P1 bound (absorb_bound2.py).

For every frame t (sequentially) the parent's deformed state S(t) (author deformation, post-hoc training-camera cull as the main
table) is frozen and a free set F_t (Gaussians whose projected centre falls in the dilated LR dynamic mask of >= 2 training
cameras, the P1 rule) receives per-frame residuals:
  P   position (fitted in px units of the free set's median depth, saved in world units)
  C   SH-DC colour
  PC  both
Target (--target):
  lr          the training LR through the checkpoint's camera model and the area operator (P1; the method)
  teacher     the matched SR teacher (sr_swinir_x4_areaft) through warp field and colour, no area (ablation)
  lr+teacher  lr + lambda_T x teacher (the w4 SR-stage objective; ablation)
All training cameras are fitted (no held-out camera); cam00 is never read. Losses, regularisers and step sizes are those of P1 v2:
full-frame L1 sum normalised by the camera's LR dynamic-pixel count (teacher term: by 16 x that count), 6 random cameras per step,
Adam (lr 0.05 px / 2.5e-3), kNN-Laplacian 1e-3 and shrink 1e-5 on the position residual. Warm start (default): frame t starts from
frame t-1's residual of the Gaussians present in both free sets (same post-cull index). Optional --temporal lambda_t penalises the
change against that start on the shared Gaussians (default 0).

Outputs in --out: run.json (provenance, post-cull xyz SHA256 for index verification), f####.npz per frame (idx int32 post-cull
index, dxyz float32 world units added after native_effective_state(t), dsh0 float32 added to the deformed SH-DC), rows.jsonl
(per-frame training-camera LR closure before/after, per camera; free count; seconds), converge.json (with --converge: all-camera
fixed-batch losses at LOG_STEPS for the first --converge-frames frames), complete.json at the end. Resumable: existing frames are
skipped and the next frame warm-starts from the last saved one.
apply_residual() / load_residual() are the reference application for the evaluation wrapper.

  perframe_fit.py --checkpoint CK --out DIR [--arm P|C|PC] [--target lr|teacher|lr+teacher] [--frames 0-299] [--steps 300]
                  [--steps-first N] [--temporal L] [--no-warm] [--converge] [--dry]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import socket
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import absorb_bound2 as ab  # noqa: E402  (helpers and the cullretrain runtime paths)
import numpy as np  # noqa: E402
import torch  # noqa: E402

cr, native, ev, rt = ab.cr, ab.native, ab.ev, ab.rt
ROOT = ab.ROOT
TEACHER_DIR = "sr_swinir_x4_areaft"
LOG_STEPS = (0, 25, 50, 100, 150, 200, 300)


def apply_residual(state, res):
    """state = native.native_effective_state(g, t) of the culled parent; res = load_residual(...) of the same frame."""
    st = dict(state)
    idx = res["idx"]
    if idx.numel() == 0:
        return st
    if res.get("dxyz") is not None:
        st["xyz"] = state["xyz"].index_add(0, idx, res["dxyz"].to(state["xyz"].dtype))
    if res.get("dsh0") is not None:
        sh = state["sh"].clone()
        sh[idx, 0, :] = sh[idx, 0, :] + res["dsh0"].to(sh.dtype)
        st["sh"] = sh
    return st


def load_residual(path, device):
    with np.load(path) as z:
        out = dict(idx=torch.from_numpy(z["idx"].astype(np.int64)).to(device))
        out["dxyz"] = torch.from_numpy(z["dxyz"].astype(np.float32)).to(device) if "dxyz" in z.files else None
        out["dsh0"] = torch.from_numpy(z["dsh0"].astype(np.float32)).to(device) if "dsh0" in z.files else None
    return out


def frames_arg(s):
    out = []
    for part in s.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b) + 1)) if b else [int(a)]
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--arm", default="P", choices=["P", "C", "PC"])
    p.add_argument("--target", default="lr", choices=["lr", "teacher", "lr+teacher"])
    p.add_argument("--lambda-teacher", type=float, default=0.2)
    p.add_argument("--frames", default="0-299")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--steps-first", type=int, default=None, help="steps of the first (cold) frame of the run; default --steps")
    p.add_argument("--batch", type=int, default=6)
    p.add_argument("--knn", type=float, default=1e-3)
    p.add_argument("--temporal", type=float, default=0.0)
    p.add_argument("--c-cold", action="store_true",
                   help="v3 (10-21 ~23:20, prereg_perframe.md Addendum 8, contingent on v2 failing its drift test): warm-start only the "
                        "position residual; the colour residual starts from 0 every frame (no accumulation by construction, no new weight)")
    p.add_argument("--c-reg", action="store_true",
                   help="v2 (10-21 ~22:50, prereg_perframe.md Addendum 7): give the colour residual the same kNN-Laplacian and shrink as "
                        "the position residual (v1 left it unregularised; with warm start it drifted over the sequence)")
    p.add_argument("--no-warm", action="store_true")
    p.add_argument("--converge", action="store_true")
    p.add_argument("--converge-frames", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dry", action="store_true", help="CPU-safe check: load, cull, masks, free sets of two frames; no rendering")
    a = p.parse_args()
    dev = a.device
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    if (out / "complete.json").exists():
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
    if "cam00" in train:
        raise SystemExit("cam00 must not be a training camera")
    model = ab.camera_model(raw_aff, dev)
    g = ab.load_model(ck, plan, dev)
    h, opt, pipe, sh = cr.configuration(plan)
    cull_record = cr.cull(g, cr.Culler(manifest, device=dev), None, "posthoc", optimizer_backed=False)
    xyz_sha = hashlib.sha256(g._xyz.detach().float().cpu().numpy().tobytes()).hexdigest()
    ck_entry = cr.entry(ck)
    record = dict(kind="perframe_fit", checkpoint=ck_entry, checkpoint_sha256=ck_entry["sha256"], manifest=cr.entry(manifest_path),
                  args=vars(a),
                  points_after_cull=int(g._xyz.shape[0]), post_cull_xyz_sha256=xyz_sha, cull_record=cull_record,
                  host=socket.gethostname(), gpu=torch.cuda.get_device_name() if dev == "cuda" else "cpu",
                  fitting=train, cam00_read=False,
                  sources_code={cr.rel(q): rt.sha(q) for q in (Path(__file__), HERE / "absorb_bound2.py", Path(cr.__file__),
                                                               ab.GEO / "modkit2.py")})
    old = out / "run.json"
    if old.exists():
        prev_rec = json.loads(old.read_text())
        for k in ("checkpoint", "post_cull_xyz_sha256", "args"):
            pa, na = prev_rec.get(k), record.get(k)
            if k == "args":
                pa = {x: y for x, y in (pa or {}).items() if x not in ("frames",)}
                na = {x: y for x, y in na.items() if x not in ("frames",)}
            if pa != na:
                raise SystemExit(f"resume refused: run.json {k} differs")
    else:
        old.write_text(json.dumps(record, indent=1))
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
    med = {c: torch.stack([canon(c, ab.read_rgb(root / obs[(c, f)]["lr_path"], dev)) for f in range(0, 300, 5)]).median(0).values
           for c in train}
    background = torch.ones(3, device=dev) if cr.white_background(plan) else torch.zeros(3, device=dev)
    fx = float(np.asarray(manifest["cameras"][train[0]]["K_hr"])[0][0])
    use_lr, use_tea = a.target in ("lr", "lr+teacher"), a.target in ("teacher", "lr+teacher")
    w_tea = 1.0 if a.target == "teacher" else a.lambda_teacher

    def camera(c, f):
        return native.hr_camera(ev.camera_item(manifest, obs[(c, f)], torch), index.get(c, 0), manifest)

    def render(cam, state):
        return native.native_rgb(cam, g, state, pipe, background)["render"]

    def build(S, free, prm, px):
        st = dict(S)
        if "mu" in prm:
            st["xyz"] = S["xyz"].index_add(0, free, prm["mu"] * px)
        if "c" in prm:
            sh_ = S["sh"].clone()
            sh_[free, 0, :] = sh_[free, 0, :] + prm["c"]
            st["sh"] = sh_
        return st

    def cam_loss(c, st, ys, masks, teas, cams_):
        x = model.apply(render(cams_[c], st), c)
        n = 3 * max(int(masks[c].sum()), 1)
        loss = 0.0
        if use_lr:
            loss = loss + (ab.area(x, (lr_h, lr_w)) - ys[c]).abs().sum() / n
        if use_tea:
            loss = loss + w_tea * (x.clamp(0, 1) - teas[c]).abs().sum() / (16 * n)
        return loss

    def closure(st, ys, masks, cams_):
        res = {}
        with torch.no_grad():
            for c in train:
                e = canon(c, ys[c]) - canon(c, ab.area(model.apply(render(cams_[c], st), c), (lr_h, lr_w)))
                e2 = (e ** 2).sum(0)
                res[c] = dict(psnr=-10 * math.log10(max(float(e2.mean()) / 3, 1e-12)),
                              dyn_psnr=-10 * math.log10(max(float(e2[masks[c]].sum()), 1e-12) / (3 * max(int(masks[c].sum()), 1))))
        return res

    frames = frames_arg(a.frames)
    prev = None
    done_frames = [f for f in frames if (out / f"f{f:04d}.npz").exists()]
    converge = json.loads((out / "converge.json").read_text()) if (out / "converge.json").exists() else {}
    first_cold = True
    for k, f in enumerate(frames):
        npz = out / f"f{f:04d}.npz"
        if npz.exists():
            prev = load_residual(npz, dev)
            first_cold = False
            continue
        t0 = time.time()
        time_value = float(obs[(train[0], f)]["time"])
        with torch.no_grad():
            S = {kk: v.detach() for kk, v in native.native_effective_state(g, time_value, torch).items()}
        ys = {c: ab.read_rgb(root / obs[(c, f)]["lr_path"], dev) for c in train}
        masks = {c: ab.dilate((canon(c, ys[c]) - med[c]).abs().amax(0) > 0.06, k3) for c in train}
        cams_ = {c: camera(c, f) for c in train}
        count = torch.zeros(S["xyz"].shape[0], dtype=torch.int32, device=dev)
        depth = []
        for c in train:
            P = cams_[c].full_proj_transform.to(dev)
            hom = torch.cat([S["xyz"], torch.ones_like(S["xyz"][:, :1])], 1)
            ph = hom @ P
            wv = hom @ cams_[c].world_view_transform.to(dev)
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
        if a.dry:
            print(json.dumps(dict(frame=f, free=int(free.numel()), px_world=px, xyz_sha=xyz_sha[:12],
                                  dyn_fraction=float(np.mean([float(masks[c].float().mean()) for c in train])))), flush=True)
            if k >= 1:
                break
            continue
        teas = {}
        if use_tea:
            teas = {c: ab.read_rgb(root / TEACHER_DIR / c / f"{f:04d}.png", dev) for c in train}
        n = free.numel()
        mu0 = torch.zeros(n, 3, device=dev)
        c0 = torch.zeros(n, 3, device=dev)
        shared = torch.zeros(n, dtype=torch.bool, device=dev)
        if prev is not None and not a.no_warm and n and prev["idx"].numel():
            pos = torch.full((S["xyz"].shape[0],), -1, dtype=torch.long, device=dev)
            pos[prev["idx"]] = torch.arange(prev["idx"].numel(), device=dev)
            j = pos[free]
            shared = j >= 0
            if prev.get("dxyz") is not None and px > 0:
                mu0[shared] = prev["dxyz"][j[shared]] / px
            if prev.get("dsh0") is not None and not a.c_cold:
                c0[shared] = prev["dsh0"][j[shared]]
        prm, groups = {}, []
        if "P" in a.arm:
            prm["mu"] = mu0.clone().requires_grad_(True)
            groups.append(dict(params=[prm["mu"]], lr=0.05))
        if "C" in a.arm:
            prm["c"] = c0.clone().requires_grad_(True)
            groups.append(dict(params=[prm["c"]], lr=2.5e-3))
        steps = a.steps_first if (first_cold and a.steps_first) else a.steps
        nb = ab.knn(S["xyz"][free]) if n > 16 else None
        log_this = a.converge and len([x for x in converge if x != "doc"]) < a.converge_frames and str(f) not in converge
        curve = []

        def full_loss(prm_):
            with torch.no_grad():
                st_ = build(S, free, {kk: v.detach() for kk, v in prm_.items()}, px)
                return float(sum(cam_loss(c, st_, ys, masks, teas, cams_) for c in train) / len(train))

        optim = torch.optim.Adam(groups, eps=1e-15) if (groups and n) else None
        for it in range(steps if optim else 0):
            if log_this and it in LOG_STEPS:
                curve.append((it, full_loss(prm)))
            st = build(S, free, prm, px)
            batch = rng.choice(len(train), size=min(a.batch, len(train)), replace=False)
            loss = sum(cam_loss(train[jj], st, ys, masks, teas, cams_) for jj in batch) / len(batch)
            reg = 0.0
            if "mu" in prm:
                u = prm["mu"]
                if nb is not None and a.knn > 0:
                    reg = reg + a.knn * ((u - u[nb].mean(1)) ** 2).sum(1).mean()
                reg = reg + 1e-5 * (u ** 2).sum(1).mean()
                if a.temporal > 0 and shared.any():
                    reg = reg + a.temporal * ((u[shared] - mu0[shared]) ** 2).sum(1).mean()
            if "c" in prm and a.c_reg:
                v = prm["c"]
                if nb is not None and a.knn > 0:
                    reg = reg + a.knn * ((v - v[nb].mean(1)) ** 2).sum(1).mean()
                reg = reg + 1e-5 * (v ** 2).sum(1).mean()
            if "c" in prm and a.temporal > 0 and shared.any() and not a.c_cold:
                reg = reg + a.temporal * ((prm["c"][shared] - c0[shared]) ** 2).sum(1).mean()
            optim.zero_grad(set_to_none=True)
            (loss + reg).backward()
            optim.step()
        if log_this:
            curve.append((steps, full_loss(prm)))
            converge.setdefault("doc", "all-training-camera mean loss (the fitting objective without regularisers) at LOG_STEPS; "
                                       "first frame of the run is cold, later frames warm-started")
            converge[str(f)] = dict(curve=curve, cold=bool(first_cold or a.no_warm), steps=steps, free=int(n))
            (out / "converge.json").write_text(json.dumps(converge, indent=1))
        with torch.no_grad():
            final = {kk: v.detach() for kk, v in prm.items()}
            st = build(S, free, final, px)
            before = closure(S, ys, masks, cams_)
            after = closure(st, ys, masks, cams_)
        save = dict(idx=free.cpu().numpy().astype(np.int32), frame=np.int32(f), px_world=np.float64(px))
        if "mu" in final:
            save["dxyz"] = (final["mu"] * px).cpu().numpy().astype(np.float32)
        if "c" in final:
            save["dsh0"] = final["c"].cpu().numpy().astype(np.float32)
        tmp = out / f"f{f:04d}.tmp.npz"
        np.savez(tmp, **save)
        tmp.replace(npz)
        prev = load_residual(npz, dev)
        mean = lambda d, key: float(np.mean([v[key] for v in d.values()]))
        row = dict(frame=f, time=time_value, free=int(n), warm_shared=int(shared.sum()), px_world=px, steps=steps,
                   cold=bool(first_cold or a.no_warm), closure_before=before, closure_after=after,
                   mean_dyn_before=mean(before, "dyn_psnr"), mean_dyn_after=mean(after, "dyn_psnr"),
                   mean_full_before=mean(before, "psnr"), mean_full_after=mean(after, "psnr"),
                   mu_px_median=float(final["mu"].norm(dim=1).median()) if "mu" in final and n else None,
                   seconds=time.time() - t0)
        with (out / "rows.jsonl").open("a") as fh:
            fh.write(json.dumps(row) + "\n")
        print(f"frame {f}: free {n} shared {int(shared.sum())} steps {steps} train dyn {row['mean_dyn_before']:.2f}->"
              f"{row['mean_dyn_after']:.2f} full {row['mean_full_before']:.2f}->{row['mean_full_after']:.2f} ({row['seconds']:.0f}s)",
              flush=True)
        first_cold = False
    if a.dry:
        print("dry run ok", flush=True)
        return
    missing = [f for f in frames if not (out / f"f{f:04d}.npz").exists()]
    if not missing and frames_arg("0-299") == frames:
        sizes = sum((out / f"f{f:04d}.npz").stat().st_size for f in frames)
        (out / "complete.json").write_text(json.dumps(dict(frames=len(frames), seconds_this_session=time.time() - started,
                                                           npz_bytes=sizes, resumed_frames=len(done_frames),
                                                           peak_GB=torch.cuda.max_memory_allocated() / 1e9 if dev == "cuda" else None),
                                                      indent=1))
    print(f"done in {time.time() - started:.0f}s; missing {len(missing)}", flush=True)


if __name__ == "__main__":
    main()
