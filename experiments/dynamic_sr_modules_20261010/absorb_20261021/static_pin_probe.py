"""Zero-training probe: how much could an anti-jitter term on static Gaussians gain at all? (Claude 4dsr-46, 2026-10-22; design by
4dsr-f3 in the joint review of the fT drop, prereg_motion.md Addendum G'.) Gate for the F2 training arm: if pinning every static
Gaussian to its time-mean position does not lift the cam00 static region, a static anti-jitter loss has no room.

Static set S: Gaussians of the post-hoc-culled checkpoint whose projected deformed centre never falls in the dilated LR dynamic mask of
any training camera at any of the 300 frames (mask as perframe_fit.py: max_ch |y~(t) - median_t y~| > 0.06, 3x3 dilation, y~ = the
camera model's colour-canonical LR, median over frames 0, 5, ..., 295). Training-camera LR only; cam00 is not read.
For every frame t the residual pins S: dxyz_i(t) = mean_t' x_i(t') - x_i(t) (world units; rotation / scale / opacity / colour
deformation untouched). Output = the per-frame residual format of perframe_fit.py (fNNNN.npz: idx, dxyz, frame, px_world) and a
run.json with the checkpoint / cull identity, so eval_resid.py --resid PIN=OUT --require-all scores it against the parent in one pass.
Also records the jitter amplitude of S (HR px at each Gaussian's depth in the first training camera).
Usage: static_pin_probe.py --checkpoint CK --out DIR [--device cuda] [--smoke N  (CPU check: first N frames, nothing written)]"""
from __future__ import annotations

import argparse
import hashlib
import json
import socket
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import absorb_bound2 as ab  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

cr, native, ev, rt = ab.cr, ab.native, ab.ev, ab.rt
ROOT = ab.ROOT


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--smoke', type=int, default=0)
    a = p.parse_args()
    dev = a.device
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'complete.json').exists():
        raise FileExistsError(f'completed run preserved: {out}')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    started = time.time()
    ck = ROOT / a.checkpoint
    payload = torch.load(ck, map_location='cpu', weights_only=False)
    plan, raw_aff = payload['plan'], payload.get('affine')
    del payload
    manifest_path, manifest = cr.plan_scene(plan)
    root = Path(manifest['_root'])
    train = list(manifest['splits']['train'])
    if 'cam00' in train:
        raise SystemExit('cam00 must not be a training camera')
    g = ab.load_model(ck, plan, dev)
    cull_record = cr.cull(g, cr.Culler(manifest, device=dev), None, 'posthoc', optimizer_backed=False)
    xyz_sha = hashlib.sha256(g._xyz.detach().float().cpu().numpy().tobytes()).hexdigest()
    ck_entry = cr.entry(ck)
    lr_w, lr_h = manifest['resolutions']['lr']
    hr_w, hr_h = manifest['resolutions']['hr']
    obs = {(o['camera_id'], int(o['frame_index'])): o for o in manifest['observations']}
    index = {c: i for i, c in enumerate(train)}
    names = list(raw_aff['names'])
    delta = raw_aff['delta'].double() - raw_aff['delta'].double().mean(0)
    Ainv = {c: torch.linalg.inv(torch.eye(3, dtype=torch.float64) + delta[names.index(c), :, :3]).float().to(dev) for c in train}
    bvec = {c: delta[names.index(c), :, 3].float().to(dev) for c in train}
    canon = lambda c, y: torch.einsum('ij,jhw->ihw', Ainv[c], y - bvec[c][:, None, None])
    k3 = torch.ones(3, 3, device=dev)
    med = {c: torch.stack([canon(c, ab.read_rgb(root / obs[(c, f)]['lr_path'], dev)) for f in range(0, 300, 5)]).median(0).values
           for c in train}
    fx = float(np.asarray(manifest['cameras'][train[0]]['K_hr'])[0][0])
    frames = list(range(a.smoke)) if a.smoke else list(range(300))
    n = int(g._xyz.shape[0])
    dyn_any = torch.zeros(n, dtype=torch.bool, device=dev)
    xyz_sum = torch.zeros(n, 3, dtype=torch.float64, device=dev)

    def camera(c, f):
        return native.hr_camera(ev.camera_item(manifest, obs[(c, f)], torch), index.get(c, 0), manifest)

    def state(f):
        with torch.no_grad():
            return native.native_effective_state(g, float(obs[(train[0], f)]['time']), torch)['xyz'].detach()

    for f in frames:
        x = state(f)
        xyz_sum += x.double()
        hom = torch.cat([x, torch.ones_like(x[:, :1])], 1)
        for c in train:
            y = ab.read_rgb(root / obs[(c, f)]['lr_path'], dev)
            mask = ab.dilate((canon(c, y) - med[c]).abs().amax(0) > 0.06, k3)
            cam = camera(c, f)
            ph = hom @ cam.full_proj_transform.to(dev)
            wv = hom @ cam.world_view_transform.to(dev)
            ndc = ph[:, :2] / ph[:, 3:4].clamp_min(1e-6)
            lx = torch.floor((((ndc[:, 0] + 1) * hr_w - 1) * 0.5 + 0.5) / (hr_w // lr_w)).long()
            ly = torch.floor((((ndc[:, 1] + 1) * hr_h - 1) * 0.5 + 0.5) / (hr_h // lr_h)).long()
            ok = (wv[:, 2] > 0.2) & (lx >= 0) & (lx < lr_w) & (ly >= 0) & (ly < lr_h)
            hit = torch.zeros_like(ok)
            hit[ok] = mask[ly[ok], lx[ok]]
            dyn_any |= hit
        if f % 50 == 0:
            print(json.dumps(dict(pass1_frame=f, dynamic_so_far=int(dyn_any.sum()), elapsed=round(time.time() - started, 1))), flush=True)
    static = torch.nonzero(~dyn_any).squeeze(1)
    if a.smoke:
        print(json.dumps(dict(smoke=a.smoke, points=n, static=int(static.numel()), seconds=round(time.time() - started, 1))), flush=True)
        return
    mean = (xyz_sum / len(frames)).float()
    cam0 = camera(train[0], 0)
    depth = (torch.cat([mean[static], torch.ones_like(mean[static][:, :1])], 1) @ cam0.world_view_transform.to(dev))[:, 2].clamp_min(0.2)
    amp_max = torch.zeros(static.numel(), device=dev)
    amp_sum = torch.zeros(static.numel(), device=dev)
    for f in frames:
        x = state(f)
        d = mean[static] - x[static]
        px = d.norm(dim=1) * fx / depth
        amp_max = torch.maximum(amp_max, px)
        amp_sum += px
        tmp = out / f'f{f:04d}.tmp.npz'
        np.savez(tmp, idx=static.cpu().numpy().astype(np.int32), dxyz=d.cpu().numpy().astype(np.float32), frame=np.int32(f),
                 px_world=np.float64(float(depth.median()) / fx))
        tmp.replace(out / f'f{f:04d}.npz')
    q = lambda t, v: float(torch.quantile(t[torch.randperm(t.numel(), device=dev)[:200000]], v)) if t.numel() else None
    amp_mean = amp_sum / len(frames)
    record = dict(kind='static_pin_probe', doc=__doc__, checkpoint=ck_entry, checkpoint_sha256=ck_entry['sha256'],
                  manifest=cr.entry(manifest_path), points_after_cull=n, post_cull_xyz_sha256=xyz_sha, cull_record=cull_record,
                  static=int(static.numel()), static_fraction=float(static.numel() / n), args=vars(a),
                  jitter_hr_px=dict(mean_of_mean=float(amp_mean.mean()) if static.numel() else None, median_of_mean=q(amp_mean, 0.5),
                                    p90_of_mean=q(amp_mean, 0.9), median_of_max=q(amp_max, 0.5), p90_of_max=q(amp_max, 0.9)),
                  host=socket.gethostname(), gpu=torch.cuda.get_device_name() if dev == 'cuda' else 'cpu', seconds=time.time() - started,
                  sources_code={cr.rel(q_): rt.sha(q_) for q_ in (Path(__file__), HERE / 'absorb_bound2.py', Path(cr.__file__))})
    (out / 'run.json').write_text(json.dumps(record, indent=1))
    (out / 'complete.json').write_text(json.dumps(dict(frames=len(frames), npz_bytes=sum((out / f'f{f:04d}.npz').stat().st_size
                                                                                          for f in frames)), indent=1))
    print(json.dumps({k: v for k, v in record.items() if k not in ('doc', 'cull_record', 'sources_code', 'manifest')}), flush=True)


if __name__ == '__main__':
    main()
