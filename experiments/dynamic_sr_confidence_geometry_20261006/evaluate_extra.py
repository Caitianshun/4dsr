"""Frozen endpoint float RGB, actual-LR, radial FFT, ROI and moments audit.

HR and held-out LR are read only after rendering for evaluation. This process
does no optimizer updates and must be scheduled under evaluation_gpu.lock by
the orchestrator. All registered frames are retained, rather than picking best.
"""
from __future__ import annotations
import argparse
import ast
import csv
import os
from pathlib import Path
import time
import traceback

from cg_common import ROOT, HERE, OUT, read, sha, write, entry, bound, module, setup


def csvwrite(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    tmp = path.with_suffix('.csv.tmp')
    with tmp.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    tmp.replace(path)


def roi_function():
    # Import exactly the metric function without importing a historical run's
    # checkpoint validator or changing that historical evaluation module.
    source = ROOT/'experiments/dynamic_sr_prior_diagnosis_20260929/evaluation_adapter.py'
    node = next(x for x in ast.parse(source.read_text()).body if isinstance(x, ast.FunctionDef) and x.name == 'roi_metrics')
    scope = {}; exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), scope)
    return scope['roi_metrics'], source


def load_existing_main(label):
    values = {}
    for split in ('test', 'dev', 'train_fixed'):
        path = OUT/'evaluation'/label/split/'metrics.json'
        if path.exists():
            data = read(path)
            for row in data['rows']:
                values[row['camera_id'], int(row['frame_index'])] = row['spatial']['full']
    return values


def confidence_comparison():
    """One fixed CPU evidence audit; HR is isolated from cache construction."""
    import cv2
    import numpy as np
    import torch
    from scipy.stats import spearmanr
    from confidence_cache import ConfidenceCache, LegalInputs, pool3, resize_chw, local_std
    from degradation_operator import ActualDegradation
    torch.set_num_threads(4); cv2.setNumThreads(4)
    protocol = read(OUT/'protocol.json')
    manifest_path, teacher_index = bound(protocol['manifest']), bound(protocol['teacher'])
    inputs = LegalInputs(manifest_path, teacher_index)
    new_index = OUT/'cache/confidence_view/cache_manifest.json'
    old_index = ROOT/'output/dynamic_sr_prior_diagnosis_20260929/diagnostic_maps/prior_index.json'
    old_rows = {(r['camera'], int(r['frame'])):r for r in read(old_index)['entries']}
    keys = [(c, f) for c in inputs.manifest['splits']['train'] for f in (0, 40, 80, 118)]
    registration = dict(status='registered_before_HR_reads', observations=[dict(camera=c, frame=f) for c, f in keys],
        source_sha256=sha(__file__), manifest_sha256=sha(manifest_path), teacher_index_sha256=sha(teacher_index),
        new_confidence_index_sha256=sha(new_index), old_T_index_sha256=sha(old_index), parent=protocol['parent'],
        formulas=dict(old_T='original frozen demand map, bilinear resized; not reliability',
                      CLEAR_style='exp(-RGB closure error 3x3 LR mean / 0.05), bilinear HR; diagnostic adaptation only',
                      IE_style='exp(-mean_RGB(abs(U6000-T)/(clamp(U6000,0,1)+1e-6))); unit exponential scale; original selection threshold is not reproduced',
                      new='frozen c_LR*c_view, unchanged cache'),
        error='mean_RGB absolute teacher-to-HR, and mean_RGB absolute Q(teacher-HR), original float32 PNG/255',
        strata=dict(confidence_bins=[0, .1, .25, .5, .75, 1], texture='5x5 real-LR local std <1/255 versus >=1/255',
                    dynamic='adjacent real-LR coarse RGB MAE >=.02, static complement; proxy only',
                    boundary='real-LR grayscale Sobel magnitude >=.05, proxy only',
                    unknown='cached zero valid neighbours; nearest upsampled',
                    correlation='one deterministic per-observation HR pixel stride4 in both axes, no quality-based subsampling'),
        policy='HR evaluates fixed maps only; no weight, threshold, ordering or confidence cache changes from this audit',
        hr_derived_training_items=False)
    plan_path = OUT/'confidence_comparison_protocol.json'
    if plan_path.exists():
        assert read(plan_path) == registration, 'Confidence comparison registration changed'
    else:
        write(plan_path, registration)
    complete_path = OUT/'confidence_comparison_complete.json'
    if complete_path.exists():
        assert read(complete_path)['protocol_sha256'] == sha(plan_path)
        return
    cache = ConfidenceCache(new_index)
    operator = ActualDegradation((1008, 1344), (252, 336), device='cpu', dtype=torch.float32)
    records = []; started = time.monotonic()
    for n, key in enumerate(keys):
        camera, frame = key
        teacher = inputs.pixels(key, teacher=True); lr = inputs.pixels(key)
        # Load and validate the parent float reference before reading HR.
        raw_path = ROOT/'output/dynamic_sr_prior_diagnosis_20260929/observables/U6000'/f'{camera}_{frame:04d}.npz'
        receipt = read(raw_path.with_suffix('.json'))
        assert receipt['checkpoint']['sha256'] == protocol['parent']['sha256'] and receipt['npz_sha256'] == sha(raw_path)
        with np.load(raw_path, allow_pickle=False) as z:
            internal = torch.from_numpy(z['rgb_raw']).clamp(0, 1)
        old = bound(old_rows[key])
        demand = np.load(old, allow_pickle=False)
        if demand.ndim == 3:
            demand = demand.squeeze()
        demand = resize_chw(torch.from_numpy(demand.copy())[None].float(), (1008, 1344))[0].numpy()
        closure = pool3((resize_chw(teacher, (252, 336), mode='bicubic', antialias=True).clamp(0, 1)-lr).abs().mean(0))
        clear = resize_chw(torch.exp(-closure/.05)[None], (1008, 1344))[0].numpy()
        ie = torch.exp(-(internal-teacher).abs().div(internal+1e-6).mean(0)).numpy()
        maps = dict(old_T=demand, CLEAR_style=clear, IE_style=ie, new=cache.get(camera, frame, device='cpu')[0].numpy())
        with np.load(new_index.parent/f'{camera}_{frame:04d}.npz') as z:
            unknown = cv2.resize((z['valid_neighbours'] == 0).astype(np.uint8), (1344, 1008), interpolation=cv2.INTER_NEAREST).astype(bool)
        feature_path = new_index.parent/'features'/f'{camera}_{frame:04d}.npz'
        adjacent = frame+2 if frame < 118 else frame-2
        with np.load(feature_path) as z, np.load(new_index.parent/'features'/f'{camera}_{adjacent:04d}.npz') as zz:
            coarse_lr = z['lr']; dynamic = np.abs(coarse_lr-zz['lr']).mean(-1) >= .02
        dynamic = cv2.resize(dynamic.astype(np.uint8), (1344, 1008), interpolation=cv2.INTER_NEAREST).astype(bool)
        gray = lr.mean(0).numpy(); textured = local_std(gray) >= 1/255
        textured = cv2.resize(textured.astype(np.uint8), (1344, 1008), interpolation=cv2.INTER_NEAREST).astype(bool)
        gradient = np.sqrt(cv2.Sobel(gray, cv2.CV_32F, 1, 0)**2 + cv2.Sobel(gray, cv2.CV_32F, 0, 1)**2)
        boundary = cv2.resize((gradient >= .05).astype(np.uint8), (1344, 1008), interpolation=cv2.INTER_NEAREST).astype(bool)
        observation = inputs.records[key]; hr_path = inputs.root/observation['hr_path']
        assert sha(hr_path) == observation['hr_sha256']
        from PIL import Image
        with Image.open(hr_path) as im:
            hr = torch.from_numpy(np.array(im.convert('RGB'), copy=True)).permute(2, 0, 1).float()/255
        with torch.inference_mode():
            error = (teacher-hr).abs().mean(0).numpy()
            q_error = operator.Q(teacher-hr).abs().mean(0).numpy()
        masks = dict(full=np.ones(error.shape, bool), unknown=unknown, supported=~unknown,
                     dynamic_proxy=dynamic, static_proxy=~dynamic, low_texture=~textured, textured=textured,
                     boundary_proxy=boundary, interior_proxy=~boundary)
        for method, weight in maps.items():
            assert np.isfinite(weight).all() and (weight >= 0).all() and (weight <= 1).all()
            strata = dict(masks)
            bins = [0, .1, .25, .5, .75, 1]
            for lo, hi in zip(bins[:-1], bins[1:]):
                strata[f'confidence_{lo:g}_{hi:g}'] = (weight >= lo) & ((weight < hi) if hi < 1 else (weight <= hi))
            for name, mask in strata.items():
                count = int(mask.sum()); row = dict(camera=camera, frame=frame, method=method, scope=name, pixels=count,
                                                   fraction=count/error.size, interpretation='diagnostic heuristic, HR never feeds training weights')
                if count:
                    w = weight[mask].astype(np.float64); e = error[mask].astype(np.float64); qe = q_error[mask].astype(np.float64)
                    mass, square = float(w.sum()), float((w*w).sum())
                    row.update(mean_weight=float(w.mean()), kish_effective_fraction=mass*mass/(count*square) if square else 0,
                               teacher_abs_error=float(e.mean()), q_teacher_abs_error=float(qe.mean()),
                               weighted_teacher_abs_error=float((w*e).sum()/mass) if mass else None,
                               weighted_q_teacher_abs_error=float((w*qe).sum()/mass) if mass else None)
                    sample = mask[::4, ::4]; xs = weight[::4, ::4][sample]; ys = error[::4, ::4][sample]; qys = q_error[::4, ::4][sample]
                    row.update(spearman_abs_error=float(spearmanr(xs, ys).statistic) if len(xs) >= 16 and np.std(xs) > 1e-10 and np.std(ys) > 1e-10 else None,
                               spearman_q_error=float(spearmanr(xs, qys).statistic) if len(xs) >= 16 and np.std(xs) > 1e-10 and np.std(qys) > 1e-10 else None)
                records.append(row)
        if (n+1) % 19 == 0:
            print(dict(stage='fixed_confidence_comparison', observations=n+1, seconds=time.monotonic()-started), flush=True)
    csvwrite(OUT/'confidence_comparison.csv', records)
    write(complete_path, dict(status='completed_zero_update_confidence_diagnostic', observations=76, maps=4,
                              protocol_sha256=sha(plan_path), result=entry(OUT/'confidence_comparison.csv'),
                              parameter_updates=0, HR_read_by='this isolated evaluator only', GPU_used=False,
                              quality_threshold_search=False, confidence_cache_modified=False, seconds=time.monotonic()-started,
                              source_sha256=sha(__file__)))


def main(args):
    import numpy as np
    import torch
    import lpips
    shared, _ = setup()
    from motion_model import load_model, render_model
    from n3dv_data import load_manifest
    from detail_loss import highpass
    from depth_prior import render_moments
    torch.set_num_threads(4)
    protocol = read(OUT/'protocol.json'); checkpoint = args.checkpoint.resolve()
    manifest_path = bound(protocol['manifest']); manifest = load_manifest(manifest_path)
    output = args.out or OUT/'evaluation'/args.label/'extra'
    output.mkdir(parents=True, exist_ok=True)
    identity = dict(checkpoint_sha256=sha(checkpoint), manifest_sha256=sha(manifest_path),
                    source_sha256=sha(__file__), label=args.label, protocol_sha256=sha(OUT/'protocol.json'))
    if (output/'complete.json').exists():
        previous = read(output/'complete.json')
        if previous['identity'] != identity:
            raise ValueError('Completed extra evaluation identity changed')
        return
    if (output/'config.json').exists() and read(output/'config.json') != identity:
        raise ValueError('Partial extra evaluation identity changed; retain failure and use a new output')
    write(output/'config.json', identity)
    fft_source = ROOT/'experiments/dynamic_sr_same_observation_20260930/frequency.py'
    fft = module('confidence_geometry_frozen_fft', fft_source)
    roi_metrics, roi_source = roi_function()
    roi_path = ROOT/'output/dynamic_sr_prior_diagnosis_20260929/spectrum/roi_protocol.json'
    roi = read(roi_path); assert roi['manifest_sha256'] == sha(manifest_path)
    evaluator = shared.evaluator(); legacy = evaluator.legacy
    observations = sorted([o for o in manifest['observations'] if o['camera_id'] in ('cam00', 'cam01') or
                           (o['split'] == 'train' and o['frame_index'] in (0, 40, 80, 118))],
                          key=lambda o: (o['camera_id'], o['frame_index']))
    assert len(observations) == 196 and sum(o['split'] == 'train' for o in observations) == 76
    model = load_model(checkpoint, manifest); model.g._deformation.eval()
    watched = [model.g._xyz, model.g._scaling, model.g._rotation, model.g._opacity,
               model.g._features_dc, model.g._features_rest, *model.g._deformation.parameters()]
    if model.children is not None:
        watched += list(model.children.parameters())
    versions = [p._version for p in watched]
    metric = lpips.LPIPS(net='alex', spatial=False).cuda().eval().requires_grad_(False)
    teachers = {(r['camera'], int(r['frame'])): r for r in read(bound(protocol['teacher']))['entries']}
    main_values = load_existing_main(args.label)
    rows, freq_rows, region_rows, float_rows = [], [], [], []
    started = time.monotonic(); rgb_new = moment_new = 0
    for i, observation in enumerate(observations):
        camera, frame = observation['camera_id'], int(observation['frame_index'])
        split = 'train76' if observation['split'] == 'train' else 'development'
        stem = f'{camera}_{frame:04d}'; float_path = output/'floats'/f'{stem}.npz'
        receipt_path = float_path.with_suffix('.json')
        if float_path.exists() and receipt_path.exists():
            saved = read(receipt_path)
            assert saved['checkpoint_sha256'] == identity['checkpoint_sha256'] and saved['sha256'] == sha(float_path)
            with np.load(float_path, allow_pickle=False) as data:
                arrays = {k: data[k] for k in data.files}
            raw = torch.from_numpy(arrays['rgb_raw']).cuda()
        else:
            with torch.inference_mode():
                cam = evaluator.render_camera(manifest, observation, i)
                raw = render_model(model, cam)['render']; rgb_new += 1
                state = shared.effective_state(model, observation['time'])
                moments = render_moments(cam, state['xyz'], state['cov'], state['opacity'], (252, 336)); moment_new += 1
                assert torch.isfinite(raw).all() and torch.isfinite(moments['moments']).all()
                arrays = dict(rgb_raw=raw.cpu().numpy().astype(np.float32),
                              moments_lr=moments['moments'].cpu().numpy().astype(np.float32),
                              alpha_lr=moments['alpha'].cpu().numpy().astype(np.float32),
                              depth_lr=moments['expected_z'].cpu().numpy().astype(np.float32),
                              raw_variance_lr=moments['variance_z_raw'].cpu().numpy().astype(np.float32))
            float_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(float_path, **arrays)
            saved = dict(camera=camera, frame=frame, path=str(float_path.relative_to(output)), sha256=sha(float_path),
                         checkpoint_sha256=identity['checkpoint_sha256'], information='evaluation_only_raw_float32_RGB_and_area_moments')
            write(receipt_path, saved)
        float_rows.append(saved)
        images = {}
        for role in ('hr', 'lr'):
            path = Path(manifest['_root'])/observation[role+'_path']
            assert sha(path) == observation[role+'_sha256']
            images[role] = legacy.read_rgb(path)
        prediction = legacy.image_array(raw); gt = images['hr']
        empty = np.zeros(prediction.shape[:2], bool)
        if (camera, frame) in main_values:
            quality = main_values[camera, frame]
            observed_psnr = legacy.metric_psnr(float(np.square(prediction-gt).mean()))
            assert abs(observed_psnr-quality['psnr']) < 1e-4, (camera, frame, observed_psnr, quality['psnr'])
        else:
            quality = legacy.spatial_metrics(prediction, gt, empty, metric)['full']
        closure = evaluator.lr_reprojection_metrics(raw, images['lr'], empty)['full']
        alpha, variance, depth = arrays['alpha_lr'], arrays['raw_variance_lr'], arrays['depth_lr']
        supported = (alpha >= 1e-3) & np.isfinite(depth) & (depth > 0)
        row = dict(endpoint=args.label, camera=camera, frame=frame, split=split,
                   psnr=quality['psnr'], ssim=quality['ssim'], lpips=quality['lpips_alex'],
                   lr_l1=closure['l1'], lr_mse=closure['mse'], lr_psnr=closure['psnr'],
                   alpha_mean=float(alpha.mean()), alpha_support_fraction=float(supported.mean()),
                   variance_raw_minimum=float(variance.min()), variance_negative_fraction=float((variance < 0).mean()),
                   variance_supported_mean=float(variance[supported].mean()) if supported.any() else None,
                   expected_z_supported_mean=float(depth[supported].mean()) if supported.any() else None,
                   rgb_below_zero_fraction=float((arrays['rgb_raw'] < 0).mean()),
                   rgb_above_one_fraction=float((arrays['rgb_raw'] > 1).mean()),
                   hr_sha256=observation['hr_sha256'], lr_sha256=observation['lr_sha256'])
        if split == 'train76':
            teacher_path = Path(manifest['_root'])/teachers[camera, frame]['relative_path']
            assert sha(teacher_path) == teachers[camera, frame]['sha256']
            teacher = torch.from_numpy(legacy.read_rgb(teacher_path).transpose(2, 0, 1).copy()).to(raw)
            with torch.inference_mode():
                row['teacher_rgb_l1'] = float((raw-teacher).abs().double().mean())
                row['teacher_H_l1'] = float((highpass(raw, (252, 336))-highpass(teacher, (252, 336))).abs().double().mean())
        rows.append(row)
        freq_rows.append(dict(endpoint=args.label, repeat=args.label[1] if args.label.startswith('r') else 0,
                              camera=camera, frame=frame, split=split, **fft.decompose(prediction, gt)))
        if frame in (40, 80) and camera in roi['regions_by_camera_xyxy_exclusive']:
            region_values = roi_metrics(legacy, prediction, gt, roi['regions_by_camera_xyxy_exclusive'][camera], metric, 'cuda')
            for name, value in region_values.items():
                region_rows.append(dict(endpoint=args.label, camera=camera, frame=frame, split=split, region=name, **value))
        if (i+1) % 20 == 0:
            print(dict(stage='endpoint_extra', endpoint=args.label, observations=i+1, seconds=time.monotonic()-started), flush=True)
    assert versions == [p._version for p in watched], 'Frozen endpoint evaluation modified model tensors'
    csvwrite(output/'metrics_per_frame.csv', rows)
    csvwrite(output/'frequency_budget.csv', freq_rows)
    csvwrite(output/'regional_quality.csv', region_rows)
    write(output/'float_index.json', dict(status='completed', identity=identity, entries=float_rows, grid='RGB CHW1344x1008; area moments CHW336x252',
                                        parameter_updates=0, not_training_cache=True))
    train_rows = sorted((r for r in rows if r['split'] == 'train76'), key=lambda r: (r['camera'], r['frame']))[:32]
    parent_dir = OUT/'evaluation/U6000/extra'
    parent_metric = parent_dir/'metrics_per_frame.csv'
    probe = dict(status='parent_is_current' if args.label == 'U6000' else 'pending_parent_extra',
                 observations=[dict(camera=r['camera'], frame=r['frame']) for r in train_rows],
                 selection='first32 fixed train76 camera/frame lexicographic; same calibration ordering', parameter_updates=0)
    if args.label != 'U6000' and parent_metric.exists():
        with parent_metric.open() as f:
            baseline = {(r['camera'], int(r['frame'])): r for r in csv.DictReader(f)}
        changes = []
        for r in train_rows:
            b = baseline[r['camera'], r['frame']]
            changes.append(dict(camera=r['camera'], frame=r['frame'],
                                **{k+'_delta': float(r[k])-float(b[k]) for k in ('psnr', 'ssim', 'lpips', 'lr_l1', 'lr_mse', 'alpha_mean', 'alpha_support_fraction')}))
        probe.update(status='completed_fixed_probe_difference', changes=changes,
                     mean_changes={k:float(np.mean([r[k] for r in changes])) for k in changes[0] if k not in ('camera', 'frame')},
                     parent_metrics_sha256=sha(parent_metric))
    write(output/'probe32_difference.json', probe)
    import diff_gaussian_rasterization._C as extension
    receipt = dict(status='completed_extra_evaluation', identity=identity, observations=196, train76=76,
                   rgb_forwards_new=rgb_new, moment_forwards_new=moment_new, cached_observations=196-rgb_new,
                   parameter_updates=0, seconds=time.monotonic()-started,
                   gpu=torch.cuda.get_device_name(), physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
                   torch=torch.__version__, cuda=torch.version.cuda, extension_sha256=sha(extension.__file__),
                   sources={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__), HERE/'depth_prior.py', fft_source, roi_source,
                             ROOT/'experiments/dynamic_sr_detail_supervision_20260924/evaluate.py', ROOT/'experiments/dynamic_sr_20260918/evaluate.py']},
                   roi_protocol_sha256=sha(roi_path), results={p.name:entry(p) for p in [output/'metrics_per_frame.csv', output/'frequency_budget.csv',
                       output/'regional_quality.csv', output/'float_index.json', output/'probe32_difference.json']},
                   precision='Quality and FFT: clamp(raw float32,0,1); LR: actual bicubic(raw) then clamp, no PNG quantization',
                   information='HR and held-out LR evaluation only; camera/time are the only rendering inputs; moments are diagnostic, not geometry truth')
    write(output/'complete.json', receipt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--label'); parser.add_argument('--out', type=Path)
    parser.add_argument('--confidence-comparison', action='store_true')
    args = parser.parse_args()
    try:
        if args.confidence_comparison:
            confidence_comparison()
        elif args.checkpoint is None or args.label is None:
            parser.error('--checkpoint and --label are required for endpoint extras')
        else:
            main(args)
    except BaseException:
        output = args.out or (OUT if args.confidence_comparison else OUT/'evaluation'/str(args.label)/'extra')
        write(output/('confidence_comparison_failed.json' if args.confidence_comparison else 'failed.json'),
              dict(status='failed_extra_evaluation', checkpoint=str(args.checkpoint), traceback=traceback.format_exc()))
        raise
