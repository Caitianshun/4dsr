"""Evaluate two frozen Wu checkpoints on the user-amended STG screening window.

Run only inside the existing remote GPU broker. No optimizer is created and no
model parameter is updated. Native training-camera visibility culling is kept;
test images, fixed masks and ROIs enter metrics only, after native rendering.
This completes a registered100-frame evaluation, never a300-frame benchmark.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import socket
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SHARED = ROOT / 'experiments/dynamic_sr_multi4d_area_20261017/evaluate.py'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def path(value):
    p = Path(value)
    p = p.resolve() if p.is_absolute() else (ROOT / p).resolve()
    if not p.is_relative_to(ROOT):
        raise ValueError('Registered path escapes the project')
    return p


def identity(p):
    p = Path(p).resolve()
    return dict(path=str(p), sha256=sha(p))


def checked(entry):
    p = path(entry['path'])
    if sha(p) != entry['sha256']:
        raise ValueError('Registered file identity changed: ' + str(p))
    return p


def read(p):
    return json.loads(Path(p).read_text())


def write(p, value):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    temporary = p.with_name(p.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + '\n')
    temporary.replace(p)


def module(name, p):
    spec = importlib.util.spec_from_file_location(name, p)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def preflight(registration):
    import numpy as np
    from PIL import Image
    config = read(registration)
    if config['schema'] != 'stg_wu_screening_evaluation_v1':
        raise ValueError('Unknown evaluation registration')
    if config['frames'] != list(range(100)) or [m['method'] for m in config['models']] != ['areaTm', 'complete']:
        raise ValueError('Only the frozen two-model cam00 frames0..99 screening is supported')
    for item in config['input_files'] + config['source_files']:
        checked(item)
    manifest_path = checked(config['manifest'])
    manifest = read(manifest_path)
    if manifest['splits']['test'] != ['cam00'] or 'cam00' in manifest['splits']['train']:
        raise ValueError('Held-out camera information boundary differs')
    tests = sorted((r for r in manifest['observations'] if r['split'] == 'test'), key=lambda r: r['frame_index'])
    if [(r['camera_id'], r['frame_index']) for r in tests] != [('cam00', f) for f in range(300)]:
        raise ValueError('Requires original complete300-frame manifest')
    if manifest['resolutions']['hr'] != [1344, 1008] or manifest['resolutions']['lr'] != [336, 252]:
        raise ValueError('Unexpected resolution')
    for row in tests[:100]:
        if abs(row['time'] - row['frame_index'] / 300) > 1e-7:
            raise ValueError('Native global time changed')
        for kind in ('hr', 'lr'):
            p = (manifest_path.parent / row[kind + '_path']).resolve()
            if not p.is_relative_to(manifest_path.parent) or sha(p) != row[kind + '_sha256']:
                raise ValueError('Evaluation reference differs: ' + str(p))
    mask = np.asarray(Image.open(checked(config['mask'])).convert('L')) > 0
    if mask.shape != (1008,1344) or hashlib.sha256(mask.tobytes()).hexdigest() != config['mask']['bool_pixel_sha256']:
        raise ValueError('Registered fixed mask pixels differ')
    roi = read(checked(config['roi_protocol']))
    if roi['status'] != 'fixed_before_prediction_reads':
        raise ValueError('ROI was not fixed before prediction reads')
    for box in roi['regions_by_camera_xyxy_exclusive']['cam00'].values():
        if (len(box) != 4 or not all(isinstance(x,int) for x in box)
                or not 0 <= box[0] < box[2] <= 1344 or not 0 <= box[1] < box[3] <= 1008):
            raise ValueError('Invalid registered ROI bounds')
    lookup = {(r['camera_id'],r['frame_index']):r for r in manifest['observations']}
    for ref in roi['reference_images']:
        row = lookup[ref['camera'],ref['frame']]
        if row['hr_sha256'] != ref['sha256'] or sha(manifest_path.parent/row['hr_path']) != ref['sha256']:
            raise ValueError('ROI reference pixels changed')
    for item in config['models']:
        checked(item['checkpoint'])
        checked(item['legacy_raw_frame0'])
        legacy = read(checked(item['legacy_summary']))
        if (legacy['status'] != 'completed' or legacy['frames'] != 300 or not legacy['posthoc_cull']
                or legacy['checkpoint']['sha256'] != item['checkpoint']['sha256']
                or legacy['manifest']['sha256'] != config['manifest']['sha256']):
            raise ValueError('Frozen legacy evaluation does not bind this complete checkpoint/manifest')
        training = read(checked(item['training_complete']))
        if training['status'] != 'completed' or training['checkpoint']['sha256'] != item['checkpoint']['sha256']:
            raise ValueError('Checkpoint training completion identity differs')
        checked(item['legacy_csv'])
        with checked(item['legacy_csv']).open() as stream:
            rows = [r for r in csv.DictReader(stream) if r['scope']=='test']
        if ([(r['camera'],int(r['frame'])) for r in rows] != [('cam00',f) for f in range(300)]
                or not all(math.isfinite(float(r[k])) for r in rows for k in ('psnr','ssim','lpips'))):
            raise ValueError('Invalid legacy CSV coverage or metric')
        with np.load(checked(item['legacy_raw_frame0']),allow_pickle=False) as archive:
            raw=archive['rgb_raw']
            if raw.shape != (3,1008,1344) or raw.dtype != np.float32 or not np.isfinite(raw).all():
                raise ValueError('Invalid frozen frame0 raw')
    return config


def evaluate_model(config, item, shared, cr, torch, np, Image):
    out = path(config['output']) / item['method']
    out.mkdir(exist_ok=False)
    previews = out / 'previews'
    previews.mkdir()
    raw_dir = out / 'raw'
    raw_dir.mkdir()
    checkpoint = checked(item['checkpoint'])
    manifest_path = checked(config['manifest'])
    manifest = cr.load_manifest(manifest_path)
    reference_root = manifest_path.parent
    tests = sorted((r for r in manifest['observations'] if r['split'] == 'test'
                    and r['frame_index'] in config['frames']), key=lambda r: r['frame_index'])
    g, payload = cr.load_for_render(checkpoint, torch.load(checkpoint, map_location='cpu', weights_only=False)['plan'])
    plan = payload['plan']
    if path(plan['data']['manifest']['path']) != manifest_path:
        raise ValueError('Embedded checkpoint plan points at a different manifest')
    del payload
    _, _, pipe, _ = cr.configuration(plan)
    loaded = int(g.get_xyz.shape[0])
    culler = cr.Culler(manifest)
    cull_record = cr.cull(g, culler, None, 'posthoc', optimizer_backed=False)
    rendered = int(g.get_xyz.shape[0])
    if [loaded, rendered, cull_record['culled']] != item['expected_points']:
        raise ValueError('Posthoc visibility cull did not reproduce frozen point counts')
    background = torch.ones(3, dtype=torch.float32, device='cuda') if cr.white_background(plan) else torch.zeros(3, dtype=torch.float32, device='cuda')
    helpers = shared.metric_helpers()
    fft = shared.module('wu_screening_frequency', shared.FFT)
    mask_path = checked(config['mask'])
    dynamic = np.asarray(Image.open(mask_path).convert('L')) > 0
    mask_hash = hashlib.sha256(dynamic.tobytes()).hexdigest()
    if dynamic.shape != (1008, 1344) or mask_hash != config['mask']['bool_pixel_sha256']:
        raise ValueError('Fixed full300 dynamic mask differs')
    # Explicit inherited ROI reference checks match the existing Multi4D helper.
    roi = read(config['roi_protocol']['path'] if Path(config['roi_protocol']['path']).is_absolute()
               else path(config['roi_protocol']['path']))
    lookup = {(r['camera_id'], r['frame_index']): r for r in manifest['observations']}
    for ref in roi['reference_images']:
        if lookup[ref['camera'], ref['frame']]['hr_sha256'] != ref['sha256']:
            raise ValueError('Inherited ROI reference differs from current manifest')
    rois = roi['regions_by_camera_xyxy_exclusive']['cam00']
    roi_info = dict(status='fixed', source=identity(checked(config['roi_protocol'])), definition=roi,
                    current_manifest_sha256=sha(manifest_path), inherited_from_short_window=True,
                    inheritance_check='Original HR reference hashes agree; original fixed coordinates retained',
                    blind_review_status='not_performed; evidence only')
    import lpips
    torch.hub.set_dir(str(path(config['torch_home'])))
    metric = lpips.LPIPS(net='alex', version='0.1', spatial=False, pnet_rand=False,
                        pretrained=True, model_path=str(checked(config['lpips_linear_weights'])),
                        verbose=False).cuda().eval().requires_grad_(False)
    with checked(item['legacy_csv']).open() as stream:
        legacy_rows = {int(r['frame']): r for r in csv.DictReader(stream) if r['scope'] == 'test' and r['camera'] == 'cam00'}
    base = dict(schema='stg_wu_screening_evaluation_v1', model='Wu', method=item['method'],
                manifest=identity(manifest_path), checkpoint=identity(checkpoint), posthoc_cull=True,
                frame_indices=config['frames'], checkpoint_training_status='completed',
                lpips_status='alex_v0.1', gpu=torch.cuda.get_device_name(), host=socket.gethostname(),
                runtime=cr.rt.check_runtime(), source=identity(__file__),
                metric_sources=[identity(shared.METRIC), identity(shared.FFT), identity(SHARED)],
                mask_receipts={'cam00': dict(mask=identity(mask_path), mask_pixel_sha256=mask_hash,
                                            full_reference_frames=300)}, roi_protocol=roi_info,
                points_loaded=loaded, capacity_points=rendered, cull=cull_record, culler=culler.describe(),
                information_boundary='cam00 calibration, global time and zero RGB only enter native rendering; reference pixels/mask/ROI enter metrics after rendering. No colour or spatial correction for cam00.',
                scientific_status='registered100_frame_development_screening_not_full300',
                legacy_summary=identity(checked(item['legacy_summary'])),
                LR_diagnostic_definition='area4 avg_pool; differs from bicubic closure diagnostic in legacy Wu CSV')
    write(out / 'run.json', base)
    # check_runtime resets hub location; retain the already-checked offline cache.
    torch.hub.set_dir(str(path(config['torch_home'])))
    begun = time.monotonic()
    rows, temporal, previous, raw_files, gaps = [], [], None, [], []
    max_by_metric = dict(psnr=0., ssim=0., lpips=0.)
    with (out / 'per_observation.jsonl').open('x') as log:
        for index, record in enumerate(tests):
            frame = record['frame_index']
            with torch.no_grad():
                cam = cr.native.hr_camera(cr.ev.camera_item(manifest, record, torch), index, manifest)
                st = cr.native.native_effective_state(g, float(record['time']), torch)
                raw = cr.native.native_rgb(cam, g, st, pipe, background)['render'].detach()
            if tuple(raw.shape) != (3, 1008, 1344) or not bool(torch.isfinite(raw).all()):
                raise ValueError('Nonfinite or wrong-shaped raw render')
            raw_array = raw.cpu().numpy().astype(np.float32)
            if frame == 0:
                with np.load(checked(item['legacy_raw_frame0'])) as archive:
                    old_raw = archive['rgb_raw']
                error = np.abs(raw_array - old_raw)
                frame0 = dict(max_abs=float(error.max()), mean_abs=float(error.mean()))
                if frame0['max_abs'] > config['consistency']['raw_max_abs'] or frame0['mean_abs'] > config['consistency']['raw_mean_abs']:
                    raise ValueError('Frame0 old/new floating render consistency failed: ' + json.dumps(frame0))
            raw_file = raw_dir / f'cam00_{frame:04d}.npz'
            np.savez_compressed(raw_file, rgb_raw=raw_array, camera=np.asarray('cam00'), frame=np.asarray(frame),
                                checkpoint_sha256=np.asarray(item['checkpoint']['sha256']),
                                manifest_sha256=np.asarray(config['manifest']['sha256']))
            raw_files.append(dict(frame=frame, **identity(raw_file)))
            pred = helpers.image_array(raw)
            gt = shared.checked_rgb(reference_root, record, 'hr', helpers)
            lr = shared.checked_rgb(reference_root, record, 'lr', helpers)
            spatial = helpers.spatial_metrics(pred, gt, dynamic, metric, 'cuda')
            full = spatial['full']
            projected = torch.nn.functional.avg_pool2d(raw.float()[None], 4)[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
            row = dict(scope='test', camera='cam00', frame=frame, psnr=full['psnr'], ssim=full['ssim'],
                       lpips=full['lpips_alex'], mse=full['mse'], regions=spatial, fixed_rois={},
                       LR_diagnostic_psnr=helpers.metric_psnr(float(((projected-lr)**2).mean())))
            band = fft.decompose(pred, gt)
            row.update({key: band[key] for key in ('low_mse', 'mid_mse', 'high_mse', 'low_psnr', 'mid_psnr', 'high_psnr')})
            for key in max_by_metric:
                delta = abs(row[key] - float(legacy_rows[frame][key]))
                max_by_metric[key] = max(max_by_metric[key], delta)
                if delta > config['consistency'][key]:
                    gaps.append(dict(frame=frame, metric=key, absolute_difference=delta))
            if gaps:
                raise ValueError('Legacy whole-image metric consistency failed: ' + json.dumps(gaps[-3:]))
            for name, (x0, y0, x1, y1) in rois.items():
                roi_mask = np.zeros(dynamic.shape, bool)
                roi_mask[y0:y1, x0:x1] = True
                row['fixed_rois'][name] = helpers.spatial_metrics(pred, gt, roi_mask, None)['dynamic']
                if frame in config['preview_frames']:
                    for label, pixels in [('render', pred), ('HR', gt)]:
                        Image.fromarray(np.rint(pixels[y0:y1, x0:x1].clip(0,1)*255).astype(np.uint8)).save(
                            previews / f'cam00_{frame:04d}_{name}_{label}.png')
            if previous is not None:
                if frame != previous[0] + 1:
                    raise ValueError('Temporal sequence is not consecutive')
                residual = (pred-previous[1]) - (gt-previous[2])
                temporal.append(dict(previous_frame=previous[0], frame=frame,
                    frame_difference_residual_l1=float(np.abs(residual).mean()),
                    frame_difference_residual_mse=float((residual**2).mean())))
            previous = (frame, pred, gt)
            rows.append(row)
            log.write(json.dumps(row, allow_nan=False)+'\n'); log.flush()
    if len(rows) != 100 or len(temporal) != 99:
        raise ValueError('Screening frame/pair coverage is incomplete')
    write(out/'temporal_per_pair.json', temporal)
    write(out/'raw_inventory.json', raw_files)
    result = dict(base, status='completed', frames=100, test=shared.aggregate(rows),
                  region_aggregate={name: {key: shared.mean_metric([r['regions'][name] for r in rows], key)
                         for key in ('psnr','ssim','mse','lpips_alex_spatial_mask')} for name in ('dynamic','static')},
                  fixed_roi_aggregate={name: {key: shared.mean_metric([r['fixed_rois'][name] for r in rows], key)
                         for key in ('psnr','ssim','mse')} for name in rois},
                  temporal=dict(pairs=99, **{key: shared.mean_metric(temporal,key) for key in
                          ('frame_difference_residual_l1','frame_difference_residual_mse')}),
                  per_observation=identity(out/'per_observation.jsonl'), temporal_per_pair=identity(out/'temporal_per_pair.json'),
                  raw_inventory=identity(out/'raw_inventory.json'), frame0_raw_consistency=frame0,
                  legacy_metric_max_absolute_difference=max_by_metric, consistency_tolerances=config['consistency'],
                  seconds=time.monotonic()-begun, visual_acceptance='not_performed')
    write(out/'summary.json', result)
    del g, metric, previous, raw, st, cam
    gc.collect(); torch.cuda.empty_cache()
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration', type=Path, required=True)
    parser.add_argument('--preflight-only', action='store_true')
    args=parser.parse_args()
    config=preflight(args.registration)
    if args.preflight_only:
        print(json.dumps(dict(status='preflight_passed', registration=identity(args.registration),
                              torch_imported='torch' in sys.modules, cuda_initialized=False)))
        return
    context=os.environ.get('CLAUDE_REMOTE_JOB_DIR')
    if not context or read(Path(context)/'execution_context.json')['selected_gpu_uuids'] != [config['gpu_uuid']]:
        raise RuntimeError('Registered remote GPU broker and exact GPU binding required')
    if socket.gethostname() != config['host']:
        raise ValueError('Registration host differs')
    out=path(config['output']);out.mkdir(exist_ok=False, parents=True)
    state=dict(status='running', completed=[], active=None, registration=identity(args.registration), started_unix=time.time())
    write(out/'status.json',state)
    try:
        import numpy as np
        import torch
        from PIL import Image
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False
        torch.set_num_threads(4)
        sys.path.insert(0,str(ROOT/'experiments/dynamic_sr_cull_retrain_20261009'))
        import cullretrain as cr
        shared=module('wu_screening_shared_metrics', SHARED)
        for item in config['models']:
            state['active']=item['method'];write(out/'status.json',state)
            result=evaluate_model(config,item,shared,cr,torch,np,Image)
            state['completed'].append(dict(method=item['method'], summary=identity(out/item['method']/'summary.json')))
            write(out/'events'/f"{item['method']}.json",state['completed'][-1])
        state.update(status='completed',active=None,seconds=time.time()-state['started_unix'],scientific_status='100_frame_screening_only')
    except BaseException as error:
        state.update(status='failed',error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc())
        raise
    finally:
        write(out/'status.json',state);write(out/'events/terminal.json',state)


if __name__=='__main__':
    main()
