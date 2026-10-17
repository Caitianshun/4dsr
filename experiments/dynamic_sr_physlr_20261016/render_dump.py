"""Dump HR-resolution renders and depth of a trained model for given cameras and frames (Claude, 4dsr-7c, 2026-10-17).
Interpreter mml_bw, GPU through the training service. Imports cullretrain; nothing is modified.

Same render path as cullretrain.run_eval: load_for_render, optional post-hoc training-camera cull (Culler, as eval --cull),
native.hr_camera + native_effective_state + native_rgb, raw render (no colour affine). The rasterizer's depth output is saved
as float32. No HR or LR image is read. Per frame one file <out>/f<frame>.npz with
  cameras (names), rgb uint8 (C,3,H,W) = round(255 * clamp(render)), depth float32 (C,H,W),
and <out>/meta.json with the checkpoint/manifest sha256, the colour affine of the checkpoint (if any: names, delta),
point counts and the cull record.

  render_dump.py --manifest M --checkpoint CK --frames 0 150 299 [--cameras cam00 cam01 ...] [--cull] --out DIR
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CULL = ROOT / "experiments/dynamic_sr_cull_retrain_20261009"
if str(CULL) not in sys.path:
    sys.path.insert(0, str(CULL))

import cullretrain as cr  # noqa: E402  (sets up runtime paths, imports torch and the author code)
import numpy as np  # noqa: E402
import torch  # noqa: E402

native, ev, rt = cr.native, cr.ev, cr.rt


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", required=True, help="project-relative manifest with the cameras")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--frames", type=int, nargs="+", required=True)
    p.add_argument("--cameras", nargs="*", default=None, help="default: cam00 + all training cameras")
    p.add_argument("--cull", action="store_true")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    if (out / "meta.json").exists():
        raise FileExistsError(f"completed dump preserved: {out}")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    started = time.time()
    mpath = ROOT / a.manifest
    manifest = cr.load_manifest(mpath)
    ck = ROOT / a.checkpoint
    payload = torch.load(ck, map_location="cpu", weights_only=False)
    plan = payload["plan"]
    affine = payload.get("affine")
    affine = dict(names=list(affine["names"]), delta=affine["delta"].cpu().numpy().tolist()) if affine else None
    del payload
    g, _ = cr.load_for_render(ck, plan)
    h, opt, pipe, sh = cr.configuration(plan)
    record = dict(kind="render_dump", checkpoint=cr.entry(ck), manifest=cr.entry(mpath), frames=a.frames, cull=bool(a.cull),
                  points_loaded=int(g.get_xyz.shape[0]), gpu=torch.cuda.get_device_name(), runtime=rt.check_runtime(),
                  sources={cr.rel(q): rt.sha(q) for q in (Path(__file__), Path(cr.__file__))}, affine=affine)
    if a.cull:
        record["cull_record"] = cr.cull(g, cr.Culler(manifest), None, "posthoc", optimizer_backed=False)
    record["points_rendered"] = int(g.get_xyz.shape[0])
    background = torch.ones(3, dtype=torch.float32, device="cuda") if cr.white_background(plan) else \
        torch.zeros(3, dtype=torch.float32, device="cuda")
    cams = a.cameras or (["cam00"] + list(manifest["splits"]["train"]))
    obs = {(o["camera_id"], int(o["frame_index"])): o for o in manifest["observations"]}
    for f in a.frames:
        rgbs, depths = [], []
        for i, c in enumerate(cams):
            o = obs[(c, f)]
            with torch.no_grad():
                cam = native.hr_camera(ev.camera_item(manifest, o, torch), i, manifest)
                state = native.native_effective_state(g, float(o["time"]), torch)
                pkg = native.native_rgb(cam, g, state, pipe, background)
            rgbs.append((pkg["render"].detach().clamp(0, 1) * 255 + 0.5).byte().cpu().numpy())
            depths.append(pkg["depth"].detach().float().reshape(pkg["render"].shape[-2:]).cpu().numpy())
        np.savez_compressed(out / f"f{f:04d}.npz", cameras=np.array(cams), rgb=np.stack(rgbs), depth=np.stack(depths))
        print(f"frame {f}: {len(cams)} cameras, {time.time() - started:.0f}s", flush=True)
    record["cameras"] = cams
    record["seconds"] = time.time() - started
    (out / "meta.json").write_text(json.dumps(record, indent=1))


if __name__ == "__main__":
    main()
