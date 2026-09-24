"""LR-only correspondence/phase screening, with HR used solely for evaluation.

This is not a reconstruction benchmark or an identifiability proof. All anchors,
cameras, grid points and thresholds are fixed before reading HR. CPU only.
"""
from pathlib import Path
import hashlib
import json
import platform
import socket
import time
from datetime import datetime, timezone

import cv2
import lpips
import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'output/dynamic_sr_20260920/observability_real_v1'
SCENES = {'cook_spinach': 'n3dv_prepared/cook_spinach',
          'meetroom_discussion': 'meetroom_prepared/discussion'}
CAMERAS = ['cam02', 'cam06']
ANCHORS = [24, 56, 88]
OFFSETS = [-8, -2, 2, 8]
FLOW_ARGS = (0.5, 4, 21, 5, 7, 1.5, 0)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor(x):
    return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))).unsqueeze(0)


def resize(x, size, aa=False):
    return F.interpolate(tensor(x), size=size, mode='bicubic', align_corners=False,
                         antialias=aa)[0].permute(1, 2, 0).numpy()


def warp(x, flow):
    h, w = flow.shape[:2]
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    return cv2.remap(x, xx + flow[..., 0], yy + flow[..., 1], cv2.INTER_CUBIC,
                     borderMode=cv2.BORDER_REFLECT_101)


def scalar_stats(x):
    a = np.asarray(x, dtype=np.float64).ravel()
    return None if not len(a) else dict(n=len(a), mean=float(a.mean()),
        p10=float(np.quantile(a, .1)), median=float(np.median(a)), p90=float(np.quantile(a, .9)))


def metrics(pred, gt, lp):
    pred = np.clip(pred, 0, 1)
    mse = float(np.mean((pred - gt) ** 2))
    with torch.inference_mode():
        perceptual = float(lp(tensor(pred) * 2 - 1, tensor(gt) * 2 - 1))
    blur = lambda a: cv2.GaussianBlur(a.astype(np.float64), (11,11), 1.5)
    mx, my = blur(pred), blur(gt)
    vx, vy = blur(pred**2)-mx**2, blur(gt**2)-my**2
    cov = blur(pred*gt)-mx*my
    smap = ((2*mx*my+.01**2)*(2*cov+.03**2))/((mx**2+my**2+.01**2)*(vx+vy+.03**2))
    return dict(mse=mse, psnr=-10 * np.log10(max(mse, 1e-20)),
        ssim=float(smap[5:-5,5:-5].mean()),
        lpips_alex=perceptual)


def main():
    started = time.monotonic()
    started_utc = datetime.now(timezone.utc).isoformat()
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    OUT.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).read_text()
    (OUT / 'source.py').write_text(source)
    lp = lpips.LPIPS(net='alex').eval().cpu()
    inputs = {}
    results = []
    pairs = []
    cells = []
    protocol = dict(cameras=CAMERAS, anchors=ANCHORS, neighbor_offsets=OFFSETS,
        scale=4, grid_step_lr=16, grid_margin_lr=12, patch_radius_lr=6,
        texture='LR min structure-tensor eigenvalue > 1e-5; no HR selection',
        reliable='forward-backward error < 0.15 LR px, Farneback/DIS disagreement < 0.25 LR px, RGB photometric MAE < 0.03, inside margin 4',
        relaxed='forward-backward < 0.5 and disagreement < 0.5, same photometric/in-bounds checks',
        patch_acceptance='center passes strict mask and >=75% of the fixed 12x12 LR patch passes strict mask',
        moving='estimated median accepted-neighbor displacement >= 0.25 LR px; not physical motion ground truth',
        phase_rich='accepted observations including reference >= 3; circular phase dispersion mean of x/y >= 0.10',
        hr_use='evaluation only, after LR correspondences and masks have been fixed',
        fusion='average bicubic LR anchor with confidence-gated bicubic LR neighbors warped at HR; not inverse multi-frame SR',
        high_residual='HR minus exact project bicubic-AA-downsample then bicubic-upsample; not orthogonal frequency decomposition',
        cpu_only=True)
    for scene, subpath in SCENES.items():
        data = ROOT / 'data/dynamic_sr' / subpath
        manifest_path = data / 'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        inputs[str(manifest_path)] = sha(manifest_path)
        observations = {(o['camera_id'], o['frame_index']): o for o in manifest['observations']}
        for cam in CAMERAS:
            assert cam in manifest['splits']['train']
            for anchor in ANCHORS:
                lr = {}
                # Read only LR until all correspondence/mask decisions have been made.
                for frame in [anchor] + [anchor + o for o in OFFSETS]:
                    ob = observations[(cam, frame)]
                    path = data / ob['lr_path']
                    got = sha(path)
                    assert got == ob['lr_sha256']
                    inputs[str(path)] = got
                    lr[frame] = np.asarray(Image.open(path).convert('RGB'), dtype=np.float32) / 255.
                ref = lr[anchor]
                h, w = ref.shape[:2]
                gray = {k: cv2.cvtColor(np.round(v * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY) for k, v in lr.items()}
                min_eigen = cv2.cornerMinEigenVal(gray[anchor].astype(np.float32) / 255., 5, ksize=3)
                textured = min_eigen > 1e-5
                yy, xx = np.mgrid[:h, :w].astype(np.float32)
                interior = (xx >= 8) & (xx < w - 8) & (yy >= 8) & (yy < h - 8)
                textured &= interior
                states = []
                for off in OFFSETS:
                    frame = anchor + off
                    flow = cv2.calcOpticalFlowFarneback(gray[anchor], gray[frame], None, *FLOW_ARGS)
                    back = cv2.calcOpticalFlowFarneback(gray[frame], gray[anchor], None, *FLOW_ARGS)
                    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
                    disflow = dis.calc(gray[anchor], gray[frame], None)
                    fb = np.linalg.norm(flow + warp(back, flow), axis=2)
                    disagree = np.linalg.norm(flow - disflow, axis=2)
                    phot = np.mean(np.abs(warp(lr[frame], flow) - ref), axis=2)
                    inside = ((xx + flow[..., 0] >= 4) & (xx + flow[..., 0] < w - 4) &
                              (yy + flow[..., 1] >= 4) & (yy + flow[..., 1] < h - 4) & interior)
                    valid = inside & (fb < .15) & (disagree < .25) & (phot < .03)
                    relaxed = inside & (fb < .5) & (disagree < .5) & (phot < .03)
                    displacement = np.linalg.norm(flow, axis=2)
                    states.append(dict(frame=frame, flow=flow, fb=fb, disagree=disagree,
                                       phot=phot, valid=valid, relaxed=relaxed,
                                       displacement=displacement))
                    pairs.append(dict(scene=scene, camera=cam, anchor=anchor, neighbor=frame,
                        offset=off, textured_pixels=int(textured.sum()),
                        strict_fraction=float(valid[textured].mean()),
                        relaxed_fraction=float(relaxed[textured].mean()),
                        fb_all_texture=scalar_stats(fb[textured]),
                        disagreement_all_texture=scalar_stats(disagree[textured]),
                        displacement_reliable_texture=scalar_stats(displacement[valid & textured])))
                # Freeze LR-only records before touching HR reference images.
                stem = f'{scene}_{cam}_{anchor:04d}'
                np.savez_compressed(OUT / f'{stem}_lr_correspondence.npz',
                    flow=np.stack([s['flow'] for s in states]),
                    fb=np.stack([s['fb'] for s in states]),
                    estimator_disagreement=np.stack([s['disagree'] for s in states]),
                    valid=np.stack([s['valid'] for s in states]), textured=textured)
                local_cells = []
                for y in range(12, h - 12, 16):
                    for x in range(12, w - 12, 16):
                        if not textured[y, x]:
                            continue
                        accepted = [i for i, s in enumerate(states) if s['valid'][y, x]
                                    and s['valid'][y-6:y+6,x-6:x+6].mean() >= .75]
                        flows = np.array([[0., 0.]] + [states[i]['flow'][y, x].tolist() for i in accepted])
                        disp = 1 - np.abs(np.exp(2j * np.pi * flows).mean(axis=0))
                        motion = float(np.median([states[i]['displacement'][y, x] for i in accepted])) if accepted else 0.
                        c = dict(scene=scene, camera=cam, anchor=anchor, x_lr=x, y_lr=y,
                            accepted_neighbors=accepted, n_observations=len(flows),
                            dispersion_x=float(disp[0]), dispersion_y=float(disp[1]),
                            phase_dispersion=float(disp.mean()),
                            estimated_displacement_median=motion, moving_proxy=motion >= .25,
                            phase_rich=bool(len(flows) >= 3 and disp.mean() >= .10))
                        local_cells.append(c)
                hrs = {}
                for frame in [anchor] + [anchor + o for o in OFFSETS]:
                    ob = observations[(cam, frame)]
                    path = data / ob['hr_path']
                    got = sha(path)
                    assert got == ob['hr_sha256']
                    inputs[str(path)] = got
                    hrs[frame] = np.asarray(Image.open(path).convert('RGB'), dtype=np.float32) / 255.
                hr = hrs[anchor]
                hh, ww = hr.shape[:2]
                base = np.clip(resize(ref, (hh, ww)), 0, 1)
                high = {k: v - resize(resize(v, (h, w), True), (hh, ww)) for k, v in hrs.items()}
                numer = base.copy()
                denom = np.ones((hh, ww, 1), dtype=np.float32)
                noalign = base.copy()
                weighted_high = np.zeros_like(hr)
                high_den = np.zeros((hh, ww, 1), dtype=np.float32)
                transported_high = []
                for state in states:
                    flow_hr = cv2.resize(state['flow'], (ww, hh), interpolation=cv2.INTER_LINEAR) * 4
                    valid_hr = cv2.resize(state['valid'].astype(np.float32), (ww, hh), interpolation=cv2.INTER_NEAREST)[..., None]
                    up = np.clip(resize(lr[state['frame']], (hh, ww)), 0, 1)
                    numer += warp(up, flow_hr) * valid_hr
                    denom += valid_hr
                    noalign += up * valid_hr
                    wh = warp(high[state['frame']], flow_hr)
                    transported_high.append(wh)
                    weighted_high += wh * valid_hr
                    high_den += valid_hr
                predictions = dict(single=base, aligned_fusion=numer / denom,
                                   noalign_same_weights=noalign / denom)
                repeat_error = float(np.max(np.abs(np.stack([base]*5).mean(axis=0)-base)))
                assert repeat_error < 2e-7
                observation = dict(scene=scene, camera=cam, anchor=anchor,
                    methods={name: metrics(im, hr, lp) for name, im in predictions.items()},
                    texture_fraction=float(textured[interior].mean()),
                    grid_cells=len(local_cells), repeated_reference_fusion_max_abs=repeat_error)
                results.append(observation)
                for cell in local_cells:
                    x, y = cell['x_lr'] * 4, cell['y_lr'] * 4
                    sl = np.s_[y - 24:y + 24, x - 24:x + 24]
                    true = high[anchor][sl]
                    zero_error = float(np.mean(true ** 2))
                    cell['reference_high_energy'] = zero_error
                    cell['hr_detail_neighbor_evaluation'] = []
                    for i in cell['accepted_neighbors']:
                        aligned_error = float(np.mean((transported_high[i][sl] - true) ** 2))
                        raw_error = float(np.mean((high[states[i]['frame']][sl] - true) ** 2))
                        cell['hr_detail_neighbor_evaluation'].append(dict(neighbor=states[i]['frame'],
                            aligned_mse=aligned_error, unaligned_mse=raw_error,
                            aligned_to_zero_ratio=aligned_error / max(zero_error, 1e-15),
                            unaligned_to_zero_ratio=raw_error / max(zero_error, 1e-15),
                            better_than_zero=aligned_error < zero_error,
                            better_than_unaligned=aligned_error < raw_error))
                    cell['prediction_patch'] = {name: dict(mse=float(np.mean((im[sl] - hr[sl]) ** 2)))
                                                for name, im in predictions.items()}
                    cells.append(cell)
                # Deterministic LR-selected moving/high-texture cell for a visual example.
                cand = [c for c in local_cells if c['n_observations'] >= 3 and c['moving_proxy']]
                chosen = max(cand, key=lambda c: c['phase_dispersion']) if cand else local_cells[len(local_cells) // 2]
                x, y = chosen['x_lr'] * 4, chosen['y_lr'] * 4
                sl = np.s_[max(0,y-64):min(hh,y+64), max(0,x-64):min(ww,x+64)]
                canv = Image.new('RGB', (4 * 256, 300), 'white')
                draw = ImageDraw.Draw(canv)
                for i, (name, im) in enumerate([('HR evaluation only', hr)] + list(predictions.items())):
                    draw.text((i*256+4, 5), name, fill='black')
                    patch = Image.fromarray(np.round(np.clip(im[sl], 0, 1)*255).astype(np.uint8)).resize((256,256), Image.Resampling.NEAREST)
                    canv.paste(patch, (i*256, 25))
                draw.text((4, 282), f'{stem}; LR grid center {chosen["x_lr"]},{chosen["y_lr"]}; LR-selected, not HR-ranked', fill='black')
                canv.save(OUT / f'{stem}_panel.png')
                print(stem, observation['methods'], flush=True)
    summary = {}
    for scene in SCENES:
        rr = [r for r in results if r['scene'] == scene]
        pp = [p for p in pairs if p['scene'] == scene]
        cc = [c for c in cells if c['scene'] == scene]
        groups = {}
        for name, subset in [('all_textured_grid', cc),
                             ('eligible_3observations', [c for c in cc if c['n_observations'] >= 3]),
                             ('moving_phase_rich', [c for c in cc if c['moving_proxy'] and c['phase_rich']]),
                             ('moving_phase_poor', [c for c in cc if c['moving_proxy'] and not c['phase_rich'] and c['n_observations'] >= 3]),
                             ('small_motion', [c for c in cc if not c['moving_proxy'] and c['n_observations'] >= 3])]:
            ee = [e for c in subset for e in c['hr_detail_neighbor_evaluation']]
            groups[name] = dict(cells=len(subset), accepted_neighbor_evaluations=len(ee),
                phase_dispersion=scalar_stats([c['phase_dispersion'] for c in subset]),
                aligned_high_to_zero_ratio=scalar_stats([e['aligned_to_zero_ratio'] for e in ee]),
                unaligned_high_to_zero_ratio=scalar_stats([e['unaligned_to_zero_ratio'] for e in ee]),
                detail_better_than_zero_fraction=float(np.mean([e['better_than_zero'] for e in ee])) if ee else None,
                detail_better_than_unaligned_fraction=float(np.mean([e['better_than_unaligned'] for e in ee])) if ee else None,
                mean_patch_mse={m: float(np.mean([c['prediction_patch'][m]['mse'] for c in subset])) if subset else None for m in ['single', 'aligned_fusion', 'noalign_same_weights']})
        summary[scene] = dict(n_anchors=len(rr), n_pairs=len(pp),
            strict_texture_fraction=float(np.mean([p['strict_fraction'] for p in pp])),
            relaxed_texture_fraction=float(np.mean([p['relaxed_fraction'] for p in pp])),
            methods={m: {k:float(np.mean([r['methods'][m][k] for r in rr])) for k in ['psnr','ssim','lpips_alex']} for m in ['single','aligned_fusion','noalign_same_weights']},
            groups=groups)
    identity = dict(started_utc=started_utc, finished_utc=datetime.now(timezone.utc).isoformat(),
        host=socket.gethostname(), platform=platform.platform(), python=platform.python_version(),
        torch=torch.__version__, opencv=cv2.__version__, numpy=np.__version__,
        source_sha256=sha(__file__), elapsed_seconds=time.monotonic()-started,
        gpu_used=False, cpu_torch_threads=4, cpu_opencv_threads=2)
    payload = dict(protocol=protocol, identity=identity, input_sha256=inputs,
                   summary=summary, observations=results, pairs=pairs, cells=cells)
    (OUT/'metrics.json').write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False))
    (OUT/'complete.json').write_text(json.dumps(dict(identity=identity, metrics_sha256=sha(OUT/'metrics.json')), indent=2))
    print(json.dumps(dict(identity=identity,summary=summary),indent=2), flush=True)


if __name__ == '__main__':
    main()
