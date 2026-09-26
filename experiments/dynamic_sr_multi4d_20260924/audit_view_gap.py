"""CPU-only view-gap audit of frozen Multi4D/Wu evaluations.

No fitting, checkpoint selection, or changes to official training. PNG residual
statistics are quantized diagnostics, kept separate from float-render metrics.
Requires numpy, Pillow, OpenCV, matplotlib; source outputs remain private.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
from statistics import mean

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
FRAMES = [0, 40, 80, 118]
METHODS = ['M0', 'M1', 'U40000']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def avg(rows, region='full'):
    keys = ['psnr', 'ssim', 'mse', 'lpips_alex' if region == 'full' else 'lpips_alex_spatial_mask']
    result = {k: mean(r['spatial'][region][k] for r in rows) for k in keys}
    if 'lr_reprojection' in rows[0]:
        result['lr_psnr'] = mean(r['lr_reprojection'][region]['psnr'] for r in rows)
    return result


def pixel_diagnostic(task):
    method, split, evaluation, manifest_path, out = task
    cv2.setNumThreads(1)
    manifest_path, evaluation, out = Path(manifest_path), Path(evaluation), Path(out)
    m = json.loads(manifest_path.read_text())
    d = json.loads((evaluation / 'metrics.json').read_text())
    camera = d['evaluation_cameras'][0]
    obs = {(o['camera_id'], o['frame_index']): o for o in m['observations']}
    totals, identities = {}, []
    error_sum = None
    checks = []
    for row in d['rows']:
        frame = row['frame_index']
        ref = manifest_path.parent / obs[(camera, frame)]['hr_path']
        assert sha(ref) == obs[(camera, frame)]['hr_sha256']
        pred = evaluation / 'predictions' / camera / f'{frame:04d}.png'
        with Image.open(ref) as f:
            gt = np.asarray(f.convert('RGB'), dtype=np.float32) / 255
        with Image.open(pred) as f:
            im = np.asarray(f.convert('RGB'), dtype=np.float32) / 255
        assert gt.shape == im.shape and np.isfinite(im).all()
        err = np.mean((im.astype(np.float64)-gt.astype(np.float64))**2, axis=2)
        h, w = err.shape
        if error_sum is None:
            error_sum = np.zeros_like(err)
            yy, xx = np.indices((h, w))
            masks = {
                'full': np.ones((h, w), dtype=bool),
                'border10': (xx < w*.1) | (xx >= w*.9) | (yy < h*.1) | (yy >= h*.9),
                'left10': xx < w*.1,
                'top10': yy < h*.1,
                'center80': (xx >= w*.1) & (xx < w*.9) & (yy >= h*.1) & (yy < h*.9),
            }
            totals = {k: dict(pixel_fraction=float(mask.mean()), squared_error_sum=0., psnr=[]) for k, mask in masks.items()}
        error_sum += err
        for k, mask in masks.items():
            v = err[mask]
            totals[k]['squared_error_sum'] += float(v.sum())
            totals[k]['psnr'].append(float(-10*np.log10(v.mean())))
        checks.append(float(-10*np.log10(err.mean()))-row['spatial']['full']['psnr'])
        identities.append(dict(camera=camera, frame=frame, prediction_sha256=sha(pred), hr_sha256=sha(ref)))
    for k, v in totals.items():
        v['psnr_mean'] = mean(v.pop('psnr'))
        v['fraction_of_full_squared_error'] = v['squared_error_sum']/totals['full']['squared_error_sum']
    error_map = error_sum / len(d['rows'])
    path = out / f'{method}_{camera}_mean_mse.npy'
    np.save(path, error_map.astype(np.float32))
    return dict(method=method, camera=camera, n=len(d['rows']), regions=totals,
                quantized_minus_recorded_psnr_maxabs=max(abs(v) for v in checks),
                mean_mse_map=str(path), source_images=identities)


def epipolar_diagnostic(manifest_path):
    """Check LR correspondences against GIVEN calibration, never adjust cameras."""
    cv2.setNumThreads(2)
    cv2.setRNGSeed(20260926)
    manifest_path = Path(manifest_path)
    m = json.loads(manifest_path.read_text())
    observations = {(o['camera_id'], o['frame_index']): o for o in m['observations']}
    centers = {k: np.array(v['c2w'])[:3, 3] for k, v in m['cameras'].items()}
    targets = ['cam00', 'cam01', 'cam02', 'cam06', 'cam12', 'cam18']
    sift = cv2.SIFT_create(nfeatures=2500)
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    cache, sources = {}, {}

    def features(camera, frame):
        key = camera, frame
        if key not in cache:
            o = observations[key]
            p = manifest_path.parent / o['lr_path']
            assert sha(p) == o['lr_sha256']
            image = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            kp, des = sift.detectAndCompute(image, None)
            cache[key] = (np.array([k.pt for k in kp]), des)
            sources[str(p)] = sha(p)
        return cache[key]

    def sampson(f, a, b):
        ah = np.c_[a, np.ones(len(a))]
        bh = np.c_[b, np.ones(len(b))]
        fa, ftb = ah @ f.T, bh @ f
        return np.abs(np.sum(bh*fa, axis=1)) / np.sqrt((fa[:, :2]**2).sum(1)+(ftb[:, :2]**2).sum(1)+1e-30)

    rows = []
    for target in targets:
        neighbours = sorted((c for c in m['splits']['train'] if c != target), key=lambda c: (np.linalg.norm(centers[c]-centers[target]), c))[:3]
        for neighbour in neighbours:
            ca, cb = m['cameras'][target], m['cameras'][neighbour]
            ra, rb = np.array(ca['w2c']), np.array(cb['w2c'])
            transform = rb @ np.linalg.inv(ra)
            tx, ty, tz = transform[:3, 3]
            skew = np.array([[0, -tz, ty], [tz, 0, -tx], [-ty, tx, 0]])
            f = np.linalg.inv(np.array(cb['K_lr'])).T @ skew @ transform[:3, :3] @ np.linalg.inv(np.array(ca['K_lr']))
            for frame in FRAMES:
                pa, da = features(target, frame)
                pb, db = features(neighbour, frame)
                if da is None or db is None:
                    rows.append(dict(camera=target, neighbour=neighbour, frame=frame, status='no_features'))
                    continue
                ab = {a.queryIdx: a.trainIdx for pair in matcher.knnMatch(da, db, k=2) if len(pair)==2 for a, b in [pair] if a.distance < .75*b.distance}
                ba = {a.queryIdx: a.trainIdx for pair in matcher.knnMatch(db, da, k=2) if len(pair)==2 for a, b in [pair] if a.distance < .75*b.distance}
                matches = [(i,j) for i,j in ab.items() if ba.get(j)==i]
                if len(matches)<8:
                    rows.append(dict(camera=target, neighbour=neighbour, frame=frame, status='insufficient_matches', matches=len(matches)))
                    continue
                a, b = np.array([pa[i] for i,j in matches]), np.array([pb[j] for i,j in matches])
                errors = sampson(f, a, b)
                estimated, mask = cv2.findFundamentalMat(a, b, cv2.FM_RANSAC, 1., .999)
                inliers = mask.ravel().astype(bool) if mask is not None else np.zeros(len(a),dtype=bool)
                rows.append(dict(camera=target, neighbour=neighbour, frame=frame, status='ok', matches=len(matches),
                    given_calibration_median_sampson_lr_pixels=float(np.median(errors)),
                    given_calibration_below1px_fraction=float((errors<1).mean()),
                    estimated_F_inliers=int(inliers.sum()),
                    given_calibration_median_on_estimated_F_inliers=float(np.median(errors[inliers])) if inliers.any() else None))
    summary = {}
    for c in targets:
        selected = [r for r in rows if r['camera']==c and r['status']=='ok']
        summary[c] = dict(pairs_frames=len(selected), median_pair_error= float(np.median([r['given_calibration_median_sampson_lr_pixels'] for r in selected])),
                          mean_pair_below1px_fraction=mean(r['given_calibration_below1px_fraction'] for r in selected))
    return dict(protocol='SIFT reciprocal ratio .75; three nearest training camera centers; fixed four times; given K_lr/w2c Sampson distance, no pose fitting. RANSAC is diagnostic only. Not a dense visibility or calibration accuracy proof.', summary=summary, rows=rows, source_sha256=sources)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    base = ROOT/'output/dynamic_sr_multi4d_20260924'
    manifest_path = ROOT/'data/dynamic_sr/n3dv_prepared/cook_spinach/manifest.json'
    m = json.loads(manifest_path.read_text())
    methods = json.loads((base/'final_v2/methods.json').read_text())['methods']
    audit = json.loads((base/'completion_audit_v1/complete.json').read_text())
    expected = {(x['method'],x['split']):x['metrics_sha256'] for x in audit['evaluations']}
    sources, results, pixel_tasks = {}, {}, []
    for method in METHODS:
        entry = methods[method]
        assert sha(entry['checkpoint']) == next(x['sha256'] for x in audit['checkpoints'] if x['method']==method)
        results[method] = {}
        for split, directory in entry['evaluations'].items():
            path = Path(directory)/'metrics.json'
            d = json.loads(path.read_text())
            sources[str(path)] = sha(path)
            assert sha(path) == expected[(method,split)]
            assert d['manifest_sha256'] == sha(manifest_path)
            rows = d['rows']
            assert len(rows)==(16 if split=='train_fixed' else 60)
            assert d['evaluation_cameras']==({'train_fixed':['cam02','cam06','cam12','cam18'],'dev':['cam01'],'test':['cam00']}[split])
            values = avg(rows)
            for key in ['psnr','ssim','mse','lpips_alex']:
                assert abs(values[key]-d['aggregate']['full'][key+'_mean'])<1e-9
            fixed = [r for r in rows if r['frame_index'] in FRAMES]
            results[method][split] = dict(full_window=values, matched_times=avg(fixed),
                matched_dynamic=avg(fixed,'dynamic'),matched_static=avg(fixed,'static'),
                per_camera={c:avg([r for r in fixed if r['camera_id']==c]) for c in d['evaluation_cameras']})
            if split!='train_fixed':
                pixel_tasks.append((method, split, directory, str(manifest_path), str(args.out)))
        train = results[method]['train_fixed']['matched_times']
        results[method]['train_minus_novel_psnr'] = {s:train['psnr']-results[method][s]['matched_times']['psnr'] for s in ['test','dev']}
    protocol = dict(frames=FRAMES, methods=METHODS, source='Existing fixed endpoints, no best selection; same time IDs only, not same surface visibility.',
                    pixel_diagnostic='All 60 frames, fixed outer10%, center80%, left10%, top10%; descriptive post-hoc diagnostics, never replace full-frame score.',
                    training='No new training or GPU use; frozen source metric and checkpoint hashes rechecked.')
    (args.out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    with ProcessPoolExecutor(max_workers=4) as pool:
        epi_future = pool.submit(epipolar_diagnostic, str(manifest_path))
        pixels = list(pool.map(pixel_diagnostic,pixel_tasks))
        epi = epi_future.result()
    result = dict(protocol=protocol, metrics=results,png_diagnostics=pixels,epipolar=epi,
                  source_sha256=sources,script_sha256=sha(__file__),manifest_sha256=sha(manifest_path))
    (args.out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(3,2,figsize=(12,12),constrained_layout=True)
    for i,method in enumerate(METHODS):
        for j,camera in enumerate(['cam00','cam01']):
            entry = next(p for p in pixels if p['method']==method and p['camera']==camera)
            errors = np.load(entry['mean_mse_map'])
            im = ax[i,j].imshow(errors,vmin=0,vmax=.015,cmap='inferno')
            ax[i,j].set_title(f'{method} {camera}: mean squared RGB error (60 frames)')
            ax[i,j].axis('off')
    fig.colorbar(im,ax=ax.ravel().tolist(),shrink=.5,label='Mean squared error; shared scale, clipped at 0.015')
    fig.savefig(args.out/'residual_maps.png',dpi=140)
    plt.close(fig)
    print(json.dumps({'status':'complete','metrics':{k:v['train_minus_novel_psnr'] for k,v in results.items()},'epipolar':epi['summary'],
        'png_regions':[dict(method=p['method'],camera=p['camera'],border_error_fraction=p['regions']['border10']['fraction_of_full_squared_error'],center_psnr=p['regions']['center80']['psnr_mean']) for p in pixels]},indent=2))


if __name__=='__main__':
    main()
