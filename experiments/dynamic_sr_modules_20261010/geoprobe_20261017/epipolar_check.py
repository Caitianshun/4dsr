"""Independent consistency check of the dataset camera calibration (Claude, session 4dsr-e9, 2026-10-18). CPU only, no model, no
training; reads HR frames of every camera including the test camera (diagnostic only; nothing here selects a method).

Question (Codex feedback 10-18, item 2): a smooth registration of the test view cam00 reverses the warp-field ranking on Discussion07,
but that probe fits a transform per reconstruction and cannot tell a cam00 calibration inconsistency from a drift of the shared
scene. This check uses only the dataset's own calibration (manifest K_hr and w2c; OpenCV pinhole: x_cam = R x + t, z forward,
u = K [x/z, y/z]) and image features, so it is independent of every reconstruction:
  1. per camera, a static image = per-pixel median of HR frames --frames (moving content removed), converted to 8-bit grey;
  2. SIFT features (OpenCV, --nfeatures), mutual nearest neighbours with Lowe's ratio test (--ratio) for every camera pair;
  3. for each match the symmetric point-to-epipolar-line distance in HR px under F_ij = K_j^-T [t_ij]x R_ij K_i^-1 with
     R_ij = R_j R_i^T, t_ij = t_j - R_ij t_i (no fitting of F); matches above --outlier px are counted as outliers, not averaged.
Reports per pair the inlier count and the median distance; per camera the median of its pair medians (cam00 against the training
cameras); per scene the medians of pairs with and without cam00, and for cam00's pairs the median distance of cam00's matches in the
inner and the outer half of the image radius (lens distortion grows outward). Writes summary.json in --out.

  epipolar_check.py --manifest SCENE=rel/manifest.json [...] --out rel/dir [--frames 0 30 ...] [--nfeatures 6000] [--ratio 0.7]
"""
from __future__ import annotations

import argparse
import itertools
import json
import socket
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]


def skew(t):
    return np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])


def fundamental(ci, cj):
    Ri, ti = np.array(ci['w2c'])[:3, :3], np.array(ci['w2c'])[:3, 3]
    Rj, tj = np.array(cj['w2c'])[:3, :3], np.array(cj['w2c'])[:3, 3]
    R = Rj @ Ri.T
    t = tj - R @ ti
    Ki, Kj = np.array(ci['K_hr']), np.array(cj['K_hr'])
    return np.linalg.inv(Kj).T @ skew(t) @ R @ np.linalg.inv(Ki)


def sym_dist(F, xi, xj):
    """symmetric point-to-line distance (px) for matched points xi (image i) and xj (image j), N x 2"""
    hi = np.c_[xi, np.ones(len(xi))]
    hj = np.c_[xj, np.ones(len(xj))]
    lj = hi @ F.T          # epipolar lines in image j
    li = hj @ F            # epipolar lines in image i
    dj = np.abs((lj * hj).sum(1)) / np.hypot(lj[:, 0], lj[:, 1])
    di = np.abs((li * hi).sum(1)) / np.hypot(li[:, 0], li[:, 1])
    return 0.5 * (di + dj)


def static_image(root, obs, frames):
    imgs = [cv2.imread(str(root / obs[f]['hr_path']), cv2.IMREAD_GRAYSCALE) for f in frames]
    return np.median(np.stack(imgs), axis=0).astype(np.uint8)


def mutual_matches(da, db, ratio):
    index = dict(algorithm=1, trees=4)
    fl = cv2.FlannBasedMatcher(index, dict(checks=64))
    ab = fl.knnMatch(da, db, k=2)
    ba = fl.knnMatch(db, da, k=2)
    best_ba = {m[0].queryIdx: m[0].trainIdx for m in ba if len(m) == 2 and m[0].distance < ratio * m[1].distance}
    out = []
    for m in ab:
        if len(m) == 2 and m[0].distance < ratio * m[1].distance and best_ba.get(m[0].trainIdx) == m[0].queryIdx:
            out.append((m[0].queryIdx, m[0].trainIdx))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--manifest', action='append', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--frames', type=int, nargs='*', default=list(range(0, 300, 30)))
    p.add_argument('--nfeatures', type=int, default=6000)
    p.add_argument('--ratio', type=float, default=0.7)
    p.add_argument('--outlier', type=float, default=20.0)
    p.add_argument('--min-matches', type=int, default=40)
    a = p.parse_args()
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'summary.json').exists():
        raise FileExistsError('Completed check preserved: ' + str(out))
    cv2.setNumThreads(8)
    started = time.time()
    result = dict(kind='epipolar_check', args=vars(a), host=socket.gethostname(), opencv=cv2.__version__, scenes={})
    sift = cv2.SIFT_create(nfeatures=a.nfeatures)
    for spec in a.manifest:
        scene, rel = spec.split('=', 1)
        mpath = ROOT / rel
        m = json.loads(mpath.read_text())
        root = mpath.parent
        cams = sorted(m['cameras'])
        test = [c for c in cams if c not in m['splits']['train']]
        obs = {}
        for o in m['observations']:
            obs.setdefault(o['camera_id'], {})[int(o['frame_index'])] = o
        feats = {}
        for c in cams:
            img = static_image(root, obs[c], a.frames)
            kp, de = sift.detectAndCompute(img, None)
            feats[c] = (np.array([k.pt for k in kp], np.float64), de, img.shape)
        pairs = {}
        cam00_inner, cam00_outer = [], []
        for ci, cj in itertools.combinations(cams, 2):
            (pi, di, shp), (pj, dj, _) = feats[ci], feats[cj]
            if di is None or dj is None:
                continue
            mm = mutual_matches(di, dj, a.ratio)
            if len(mm) < a.min_matches:
                continue
            xi = pi[[q for q, _ in mm]]
            xj = pj[[t for _, t in mm]]
            d = sym_dist(fundamental(m['cameras'][ci], m['cameras'][cj]), xi, xj)
            ok = d < a.outlier
            if ok.sum() < a.min_matches:
                pairs[f'{ci}-{cj}'] = dict(matches=len(mm), inliers=int(ok.sum()), median_px=None)
                continue
            pairs[f'{ci}-{cj}'] = dict(matches=len(mm), inliers=int(ok.sum()), median_px=float(np.median(d[ok])),
                                       p75_px=float(np.percentile(d[ok], 75)), outlier_share=float(1 - ok.mean()))
            for c, x in ((ci, xi), (cj, xj)):
                if c in test:
                    h, w = shp
                    r = np.hypot(x[:, 0] - w / 2, x[:, 1] - h / 2) / np.hypot(w / 2, h / 2)
                    cam00_inner += list(d[ok & (r < 0.5)])
                    cam00_outer += list(d[ok & (r >= 0.5)])
        per_cam = {}
        for c in cams:
            meds = [v['median_px'] for k, v in pairs.items() if c in k.split('-') and v['median_px'] is not None]
            per_cam[c] = dict(pairs=len(meds), median_of_pair_medians_px=float(np.median(meds)) if meds else None)
        with_test = [v['median_px'] for k, v in pairs.items() if set(k.split('-')) & set(test) and v['median_px'] is not None]
        without = [v['median_px'] for k, v in pairs.items() if not set(k.split('-')) & set(test) and v['median_px'] is not None]
        train_cams = [per_cam[c]['median_of_pair_medians_px'] for c in cams if c not in test and per_cam[c]['median_of_pair_medians_px'] is not None]
        summ = dict(test_cameras=test, pairs_with_test=len(with_test), pairs_without_test=len(without),
                    median_pairs_with_test_px=float(np.median(with_test)) if with_test else None,
                    median_pairs_without_test_px=float(np.median(without)) if without else None,
                    test_camera_value_px={c: per_cam[c]['median_of_pair_medians_px'] for c in test},
                    train_camera_values_px=dict(median=float(np.median(train_cams)) if train_cams else None,
                                                max=float(max(train_cams)) if train_cams else None,
                                                rank_of_test=[int(sum(v >= per_cam[c]['median_of_pair_medians_px'] for v in train_cams)) + 1
                                                              for c in test if per_cam[c]['median_of_pair_medians_px'] is not None]),
                    test_inner_half_median_px=float(np.median(cam00_inner)) if cam00_inner else None,
                    test_outer_half_median_px=float(np.median(cam00_outer)) if cam00_outer else None)
        result['scenes'][scene] = dict(manifest=rel, cameras=cams, pairs=pairs, per_camera=per_cam, summary=summ)
        print(json.dumps({scene: summ, 's': round(time.time() - started)}), flush=True)
    result['seconds'] = time.time() - started
    (out / 'summary.json').write_text(json.dumps(result, indent=1))


if __name__ == '__main__':
    main()
