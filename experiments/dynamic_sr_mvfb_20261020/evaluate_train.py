"""Frozen-checkpoint, whole-image S1 training-view evaluation (no parameter updates).

The manifest is fixed BEFORE inference. Example (paths relative to project root)::

  {"scene": "Cook07", "cache_dir": "output/.../Cook07_w4_even",
   "selection_note": "Preselected cameras and uniform times; development subset",
   "methods": {"B2": "output/.../s1_render_only_wide2_Cook07/final.pt",
               "D": "output/.../s1_full_Cook07/final.pt"},
   "samples": [{"camera": "cam01", "frame": 150, "preview": true,
                "rois": {"face": [816, 452, 944, 580]}},
               {"camera": "cam01", "frame": 152, "preview": true,
                "sequence": "cam01_motion"}]}

Each method is mapped from canonical RGB into the query camera's observation
domain by the SAME query_forward warp/colour model, clamped as float, and scored
on the SAME valid rectangular crop. HR is decoded only after all predictions;
it never enters FrameSources, Crops or a network. Query LR loaded by SameScene's
adapter is removed before forward construction; the query is excluded from all
source cameras. Teacher images are never loaded.

Metrics are the original legacy.spatial_metrics functions, AST-loaded without
importing unrelated GPU runtime configuration. Mean scores are arithmetic means
of per-image scores, with separate per-camera and ROI means. A subset mean is
explicitly labelled as such. This is a fitting diagnostic, not a novel-view gain.

Usage: evaluate_train.py --manifest selection.json --out NEW_DIR --device cpu
       evaluate_train.py --manifest selection.json --out NEW_DIR --validate-only

CUDA evaluation must be dispatched only after the caller has admitted resources
under project rules. This script does not acquire or override GPU reservations.
--max-new-samples permits a bounded timing pilot; rerun without it to resume.
Only manifest-selected previews/ROIs are stored, never every full output by default.
"""
from __future__ import annotations

import argparse
import ast
import csv
import datetime as dt
import hashlib
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LEGACY_PATH = ROOT / 'experiments/dynamic_sr_20260918/evaluate.py'
HUB = ROOT / 'third_party/4dgs_843d5ac_runtime/torch_hub'
METRIC_KEYS = ('psnr', 'ssim', 'lpips_alex', 'mse')


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def resolve(value):
    p = Path(value)
    return p.resolve() if p.is_absolute() else (ROOT / p).resolve()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def identity(path):
    p = Path(path).resolve()
    return dict(path=str(p), sha256=sha(p), bytes=p.stat().st_size)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def write(path, value):
    p = Path(path)
    tmp = p.with_name(p.name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(p)


def legacy_helpers(np, cv2, torch, Image):
    names = ('read_rgb', 'image_array', 'ssim_map_rgb', 'masked_mean', 'metric_psnr', 'spatial_metrics')
    nodes = [n for n in ast.parse(LEGACY_PATH.read_text()).body
             if isinstance(n, ast.FunctionDef) and n.name in names]
    if len(nodes) != len(names):
        raise ValueError('legacy evaluator function inventory changed')
    scope = dict(np=np, cv2=cv2, torch=torch, Image=Image, Path=Path)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(LEGACY_PATH), 'exec'), scope)
    return SimpleNamespace(**{name: scope[name] for name in names})


def checkpoint_path(spec):
    return resolve(spec['checkpoint'] if isinstance(spec, dict) else spec)


def validate_manifest(man, scene):
    """Check selection and training membership without decoding HR or running a model."""
    if not man.get('selection_note'):
        raise ValueError('manifest must explain its selection in selection_note')
    if set(man.get('methods', {})) != {'B2', 'D'}:
        raise ValueError('methods must specify exactly B2 and D; parent is always included')
    samples = man.get('samples', [])
    if not samples:
        raise ValueError('manifest has no samples')
    if man.get('expected_samples', len(samples)) != len(samples):
        raise ValueError('expected_samples differs from the manifest sample list')
    if man.get('cache_meta_sha256') and sha(scene.cache / 'meta.json') != man['cache_meta_sha256']:
        raise ValueError('cache metadata differs from the frozen evaluation manifest')
    seen = set()
    for sample in samples:
        q, f = sample['camera'], sample['frame']
        if not isinstance(f, int) or isinstance(f, bool):
            raise ValueError('frame must be an integer')
        key = (q, f)
        if key in seen:
            raise ValueError(f'duplicate sample: {key}')
        seen.add(key)
        if q not in scene.names or f not in scene.frames:
            raise ValueError(f'{key} is outside the S1 training-camera/frame candidate pool')
        if q in scene.cms[q].sources:
            raise ValueError(f'query appears in sources: {q}')
        for label, box in sample.get('rois', {}).items():
            if not re.fullmatch(r'[A-Za-z0-9_-]+', label):
                raise ValueError(f'unsafe ROI label {label}')
            if len(box) != 4 or any(not isinstance(v, int) for v in box):
                raise ValueError(f'invalid ROI box: {box}')
            x0, y0, x1, y1 = box
            if not (0 <= x0 < x1 <= scene.W and 0 <= y0 < y1 <= scene.H):
                raise ValueError(f'ROI out of bounds: {box}')
            if min(x1 - x0, y1 - y0) < 64:
                raise ValueError('ROI must be at least 64 pixels per side for scalar AlexNet LPIPS')
        if 'sequence' in sample and not re.fullmatch(r'[A-Za-z0-9_-]+', sample['sequence']):
            raise ValueError('sequence label must use letters, digits, underscore or hyphen')
    return seen


def prepare(man_path, man, scene, torch):
    models, runs = {}, {}
    for name, spec in man['methods'].items():
        cp = checkpoint_path(spec)
        if man.get('checkpoint_sha256', {}).get(name) and sha(cp) != man['checkpoint_sha256'][name]:
            raise ValueError(f'{name} checkpoint differs from the frozen evaluation manifest')
        run_path = cp.parent / 'run.json'
        completed = cp.parent / 'complete.json'
        run = json.loads(run_path.read_text())
        if not completed.is_file():
            raise ValueError(f'checkpoint run has no complete.json: {cp.parent}')
        if run.get('kind') != 'mvfb_train_samescene' or run.get('scene') != man['scene']:
            raise ValueError(f'not a same-scene checkpoint of {man["scene"]}: {cp}')
        expected_variant = {'B2': 'render_only_wide2', 'D': 'full'}[name]
        if run['args']['variant'] != expected_variant:
            raise ValueError(f'{name} must use {expected_variant}')
        if resolve(run['cache']) != scene.cache.resolve():
            raise ValueError(f'training cache differs for {name}')
        if run.get('query_calib') or run['args'].get('query_calib'):
            raise ValueError('this S1 evaluator does not accept S1-G/S2 query calibration')
        if set(run['queries']) != set(scene.train):
            raise ValueError(f'{name} training query pool differs from evaluation scene')
        if run['sources'] != {q: scene.cms[q].sources for q in scene.train}:
            raise ValueError(f'{name} source lists differ from training')
        if run['args'].get('k', 6) != man.get('k', 6):
            raise ValueError(f'{name} source count differs')
        code_differences = {}
        for training_path, training_hash in run.get('sources_code', {}).items():
            relative = Path(training_path)
            try:
                short = relative.relative_to('experiments/dynamic_sr_mvfb_20261020')
            except ValueError:
                continue
            runtime_path = HERE / short
            runtime_hash = sha(runtime_path)
            if training_hash != runtime_hash:
                difference = dict(training_sha256=training_hash, runtime_sha256=runtime_hash)
                code_differences[training_path] = difference
                # A changed forward may not be silently applied to an old checkpoint.
                # Use a byte-identical frozen training source copy instead.
                if short.parts[0] == 'mvfb':
                    raise ValueError(f'{name} model/adapter source changed; restore its training version: {training_path}')
        ck = torch.load(cp, map_location='cpu', weights_only=False)
        if ck.get('step') != run['args']['steps'] or ck.get('variant') != expected_variant:
            raise ValueError(f'incomplete or unexpected checkpoint: {cp}')
        if ck.get('model_class', 'MVFB') != {'B2': 'MVFBWide2', 'D': 'MVFB'}[name]:
            raise ValueError(f'unexpected model class for {name}')
        models[name] = ck
        runs[name] = dict(checkpoint=identity(cp), run=identity(run_path), complete=identity(completed),
                          variant=ck['variant'], model_class=ck.get('model_class', 'MVFB'), step=ck['step'],
                          training_queries=run['queries'], training_frames=scene.frames,
                          training_candidate_images=len(run['queries']) * len(scene.frames),
                          supervision=run.get('supervision'), training_code_hashes=run.get('sources_code'),
                          training_to_runtime_code_differences=code_differences,
                          corr_query_warp=run['args'].get('corr_query_warp', 0),
                          note='Candidate-pool membership; random crop exposure per pixel is not recorded.')
    files = {}

    def add(path, expected=None):
        key = str(Path(path).resolve())
        if key not in files:
            files[key] = identity(path)
        if expected and files[key]['sha256'] != expected:
            raise ValueError(f'input hash differs from prepared manifest: {key}')
        return files[key]

    prepared_path = scene.prep / 'manifest_train_ready_seed20261007.json'
    add(prepared_path, scene.meta['manifest']['sha256'])
    add(scene.cache / 'meta.json')
    for sample in man['samples']:
        q, f = sample['camera'], sample['frame']
        add(scene.cache / f'f{f:04d}.npz')
        observation = scene.obs[q, f]
        add(scene.prep / observation['hr_path'], observation.get('hr_sha256'))
        # The adapter also reads LRq; it is removed before any forward object.
        for camera in [q] + scene.cms[q].sources:
            obs = scene.obs[camera, f]
            add(scene.prep / obs['lr_path'], obs.get('lr_sha256'))
    source_paths = [Path(__file__), LEGACY_PATH] + sorted((HERE / 'mvfb').glob('*.py'))
    record = dict(kind='frozen_s1_training_view_evaluation', manifest=identity(man_path),
                  scene=man['scene'], cache=identity(scene.cache / 'meta.json'),
                  prepared_manifest=identity(prepared_path), parent_checkpoint=scene.meta.get('checkpoint'),
                  parent_cull_record=scene.meta.get('cull_record'), methods=runs,
                  sources={q: scene.cms[q].sources for q in scene.train},
                  code={str(p): sha(p) for p in source_paths}, inputs=files,
                  metric_protocol=dict(function='legacy.spatial_metrics', colour_space='RGB [0,1] float clamp after query_forward',
                    observation_domain='q camera warp + colour affine; HR used only for scoring',
                    evaluation_crop='symmetric ceil(max(abs(query_w4_displacement))) + 3; identical across methods',
                    PSNR='mean of per-image RGB PSNR', SSIM='11x11 Gaussian sigma 1.5; omit 5-pixel crop border',
                    LPIPS='AlexNet v0.1 standard scalar, plus legacy spatial diagnostic',
                    affine_fit_on_evaluation_HR=False, teacher_input=False, query_LR_forward_input=False),
                  selection_note=man['selection_note'], parameter_updates=0,
                  limitations=['Training-candidate images measure fitting, not novel-view generalization.',
                    'The cached parent RGB is uint8, while network output and scoring use clamped float.',
                    'Parent camera membership comes from the parent cache; cache frame selection is not the parent training schedule.',
                    'Fixed ROIs are a selected-region diagnostic; no claim of human visual acceptance.'])
    return models, record


def recovered_rows(path):
    if not path.exists():
        return []
    blob = path.read_bytes()
    parts, rows, used = blob.splitlines(keepends=True), [], 0
    for i, part in enumerate(parts):
        try:
            rows.append(json.loads(part))
            used += len(part)
        except json.JSONDecodeError:
            if i != len(parts) - 1 or part.endswith(b'\n'):
                raise ValueError(f'malformed non-trailing JSONL record in {path}')
            saved = path.with_name(path.name + f'.partial_{time.time_ns()}')
            saved.write_bytes(blob[used:])
            path.write_bytes(blob[:used])
    return rows


def average(rows, name, region=None):
    metrics = [r['methods'][name]['metrics'] if region is None else r['methods'][name]['rois'][region]
               for r in rows if region is None or region in r['methods'][name]['rois']]
    return dict(count=len(metrics), **{key: sum(m[key] for m in metrics) / len(metrics)
                                     for key in METRIC_KEYS}) if metrics else dict(count=0)


def aggregate(rows, record, man, scene, finished):
    methods = ('parent', 'B2', 'D')
    per_camera = {q: {m: average([r for r in rows if r['camera'] == q], m) for m in methods}
                  for q in sorted({r['camera'] for r in rows})}
    roi_names = sorted({k for r in rows for k in r.get('rois', {})})
    means = {m: average(rows, m) for m in methods}
    full_pool = {(q, f) for q in scene.train for f in scene.frames}
    selected = {(s['camera'], s['frame']) for s in man['samples']}
    delta = {}
    if rows:
        for left, right in [('B2', 'parent'), ('D', 'parent'), ('D', 'B2')]:
            delta[f'{left}_minus_{right}'] = {key: means[left][key] - means[right][key] for key in METRIC_KEYS}
    return dict(status='complete' if finished else 'partial', updated_utc=utc(),
                identity_sha256=digest(record), scene=man['scene'], completed_images=len(rows),
                selected_images=len(selected), full_training_candidate_images=len(full_pool),
                selection_is_full_training_pool=selected == full_pool,
                interpretation='full training candidate-pool mean' if selected == full_pool and finished else 'fixed selected training-view subset; not a full training-set mean',
                per_method=means, deltas=delta, per_camera=per_camera,
                per_roi={roi: {m: average(rows, m, roi) for m in methods} for roi in roi_names},
                frame_lists={q: sorted(s['frame'] for s in man['samples'] if s['camera'] == q)
                             for q in sorted({s['camera'] for s in man['samples']})},
                metric_protocol=record['metric_protocol'], limitations=record['limitations'])


def export_csv(path, rows):
    keys = ('camera', 'frame', 'method', 'parent_training_camera', 'network_training_candidate_pool',
            'eval_x0', 'eval_y0', 'eval_x1', 'eval_y1') + METRIC_KEYS
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            for method, result in row['methods'].items():
                writer.writerow(dict(camera=row['camera'], frame=row['frame'], method=method,
                                     parent_training_camera=row['training_identity']['parent_training_camera'],
                                     network_training_candidate_pool=row['training_identity']['network_training_candidate_pool'],
                                     **dict(zip(('eval_x0', 'eval_y0', 'eval_x1', 'eval_y1'), row['evaluation_box_xyxy'])),
                                     **{k: result['metrics'][k] for k in METRIC_KEYS}))


def main():
    global ROOT, LEGACY_PATH, HUB
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--project-root', default=os.environ.get('MVFB_EVAL_PROJECT_ROOT', str(ROOT)),
                        help='data/output root; permits running this script and mvfb from a frozen source directory')
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('--max-new-samples', type=int, default=None)
    args = parser.parse_args()
    if args.threads < 1 or args.max_new_samples is not None and args.max_new_samples < 1:
        parser.error('threads and max-new-samples must be positive')
    if args.device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['OMP_NUM_THREADS'] = str(args.threads)
    os.environ['MKL_NUM_THREADS'] = str(args.threads)
    ROOT = Path(args.project_root).resolve()
    LEGACY_PATH = ROOT / 'experiments/dynamic_sr_20260918/evaluate.py'
    HUB = ROOT / 'third_party/4dgs_843d5ac_runtime/torch_hub'
    sys.path.insert(0, str(HERE))
    import cv2
    import inspect
    import numpy as np
    import torch
    from PIL import Image
    from mvfb.controls import MVFBWide2
    from mvfb.model import MVFB, Crops, FrameSources
    import mvfb.data as data_module
    import mvfb.samescene as same_module
    from mvfb.samescene import SameScene, query_forward
    # Only relocate IO roots when executing a byte-identical frozen source copy.
    # Do not edit frozen module files or change model/geometry operations.
    data_module.ROOT = ROOT
    data_module.PREP = ROOT / 'data/dynamic_sr/full_time_20261007/prepared'
    data_module.CACHE = ROOT / 'output/dynamic_sr_mvfb_20261020/cache'
    same_module.PREP = data_module.PREP
    same_module.CACHE_ALL = ROOT / 'output/dynamic_sr_mvfb_20261020/cache_all'
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    man_path, out = resolve(args.manifest), resolve(args.out)
    man = json.loads(man_path.read_text())
    cache_dir = resolve(man['cache_dir']) if man.get('cache_dir') else same_module.CACHE_ALL / same_module.CACHE_ALL_DIRS[man['scene']]
    if 'cache_dir' in inspect.signature(SameScene).parameters:
        scene = SameScene(man['scene'], k=man.get('k', 6), cache_dir=cache_dir)
    else:
        if cache_dir.name != same_module.CACHE_ALL_DIRS[man['scene']]:
            raise ValueError('frozen S1 SameScene requires the original cache folder name under cache_root')
        scene = SameScene(man['scene'], k=man.get('k', 6), cache_root=cache_dir.parent)
    if not hasattr(scene, 'names'):
        scene.names = list(scene.meta['train'])
    validate_manifest(man, scene)
    models, record = prepare(man_path, man, scene, torch)
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'identity.json').exists():
        if json.loads((out / 'identity.json').read_text()) != record:
            raise ValueError('manifest, checkpoint, code or input identity changed; choose a new output directory')
    else:
        write(out / 'identity.json', record)
    write(out / 'validation.json', dict(status='validated', utc=utc(), samples=len(man['samples']),
                                        identity_sha256=digest(record), parameter_updates=0, image_inference_performed=False))
    if args.validate_only:
        print(json.dumps(dict(status='validated', out=str(out), samples=len(man['samples']), identity_sha256=digest(record))), flush=True)
        return
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no CPU fallback was requested')
    import lpips
    torch.hub.set_dir(str(HUB))

    def no_download(*_args, **_kwargs):
        raise RuntimeError('Downloads disabled; the project-local AlexNet cache must already exist')

    torch.hub.download_url_to_file = no_download
    alex = HUB / 'checkpoints/alexnet-owt-7be5be79.pth'
    if not alex.is_file():
        raise FileNotFoundError(alex)
    metric = lpips.LPIPS(net='alex', version='0.1', spatial=False, pnet_rand=False, pretrained=True, verbose=False).to(args.device).eval().requires_grad_(False)
    legacy = legacy_helpers(np, cv2, torch, Image)
    runtime = dict(started_utc=utc(), device=args.device, threads=args.threads, torch=torch.__version__,
                   numpy=np.__version__, opencv=cv2.__version__, python=sys.version, executable=sys.executable,
                   CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),
                   hardware=torch.cuda.get_device_name() if args.device == 'cuda' else 'CPU',
                   lpips_alexnet=identity(alex), lpips_calibration=identity(Path(lpips.__file__).parent / 'weights/v0.1/alex.pth'),
                   lpips_package=lpips.__file__, parameter_updates=0)
    runtime_path = out / 'runtime_identity.json'
    comparable = {k: v for k, v in runtime.items() if k != 'started_utc'}
    if runtime_path.exists() and json.loads(runtime_path.read_text()) != comparable:
        raise ValueError('runtime changed while resuming; use the same device/environment or a new output directory')
    write(runtime_path, comparable)
    write(out / f'runtime_{time.time_ns()}.json', runtime)
    nets = {}
    for name, ck in models.items():
        net = {'B2': MVFBWide2, 'D': MVFB}[name](**ck['kwargs']).to(args.device).eval().requires_grad_(False)
        net.load_state_dict(ck['state'])
        nets[name] = net
    del models
    progress_path = out / 'images.jsonl'
    rows = recovered_rows(progress_path)
    expected = {(s['camera'], s['frame']) for s in man['samples']}
    done = {(r['camera'], r['frame']) for r in rows}
    if len(done) != len(rows) or not done <= expected or any(set(r['methods']) != {'parent', 'B2', 'D'} for r in rows):
        raise ValueError('resume rows have duplicates, unexpected samples or incomplete methods')
    origins = torch.tensor([[0, 0]])
    new_samples = 0

    def score(pred, hr):
        return legacy.spatial_metrics(pred, hr, np.zeros(pred.shape[:2], bool), metric, args.device)['full']

    def save_rgb(path, array):
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.rint(np.clip(array, 0, 1) * 255).astype(np.uint8)).save(path)

    try:
        with progress_path.open('a') as stream:
            for sample in man['samples']:
                q, f = sample['camera'], sample['frame']
                if (q, f) in done:
                    continue
                began = time.monotonic()
                cm, qm = scene.cms[q].clone_to(args.device), scene.qms[q]
                item = scene.load(f, q, teacher=False)
                item.pop('LRq')
                if any(k in item for k in ('HR', 'LR', 'LRq', 'T', 'Tq')) or q in cm.sources:
                    raise ValueError('forbidden query supervision in forward inputs')
                margin = int(np.ceil(float(qm.U.abs().max()))) + 3
                x0 = y0 = margin
                x1, y1 = scene.W - margin, scene.H - margin
                if min(x1 - x0, y1 - y0) < 64:
                    raise ValueError('valid evaluation crop is too small')
                for box in sample.get('rois', {}).values():
                    if not (x0 <= box[0] < box[2] <= x1 and y0 <= box[1] < box[3] <= y1):
                        raise ValueError(f'ROI {box} is outside common valid crop')
                predictions, inference_times = {}, {}
                with torch.inference_mode():
                    empty_sources = SimpleNamespace(cm=cm, K=0)
                    plain = Crops(item, empty_sources, origins, (scene.H, scene.W), args.device, transport_sources=False)
                    for name in ('parent', 'B2', 'D'):
                        if args.device == 'cuda':
                            torch.cuda.synchronize()
                        begin_forward = time.monotonic()
                        if name == 'parent':
                            x = plain.R
                        elif name == 'B2':
                            x, _ = nets[name](empty_sources, plain)
                        else:
                            src = FrameSources(item, cm, args.device)
                            crops = Crops(item, src, origins, (scene.H, scene.W), args.device, transport_sources=True)
                            x, _ = nets[name](src, crops)
                        observed, mask = query_forward(x, qm, origins, (scene.H, scene.W), scene.H, scene.W)
                        if not bool(mask[0, 0, y0:y1, x0:x1].all()):
                            raise ValueError('common evaluation rectangle includes invalid bicubic samples')
                        predictions[name] = legacy.image_array(observed[0])
                        if args.device == 'cuda':
                            torch.cuda.synchronize()
                        inference_times[name] = time.monotonic() - begin_forward
                        if name == 'D':
                            del src, crops
                        del observed, mask, x
                # Only now decode the query HR. It cannot enter any model object.
                obs = scene.obs[q, f]
                hr = legacy.read_rgb(scene.prep / obs['hr_path'])
                if hr.shape != (scene.H, scene.W, 3):
                    raise ValueError('query HR size differs from the render')
                row = dict(camera=q, frame=f, rois=sample.get('rois', {}), sequence=sample.get('sequence'),
                           training_identity=dict(parent_training_camera=q in scene.names,
                             network_training_candidate_pool=q in scene.train and f in scene.frames,
                             network_training_frame_selection='explicit frames in parent cache metadata',
                             HR_used_for_training=False, HR_role='scoring only'),
                           sources=cm.sources, forward_input_keys=sorted(item), query_excluded_from_sources=True,
                           evaluation_box_xyxy=[x0, y0, x1, y1], query_max_displacement_hr=float(qm.U.abs().max()),
                           input_inventory_refs=[str(scene.cache / f'f{f:04d}.npz'), str(scene.prep / obs['hr_path'])],
                           methods={})
                preview = sample.get('preview', False) or bool(sample.get('sequence'))
                dest = out / 'previews' / (sample.get('sequence') or q) / f'{q}_f{f:04d}'
                if preview:
                    save_rgb(dest / 'HR.png', hr)
                for roi, (rx0, ry0, rx1, ry1) in sample.get('rois', {}).items():
                    save_rgb(dest / f'HR_{roi}.png', hr[ry0:ry1, rx0:rx1])
                for name, pred in predictions.items():
                    measured = score(pred[y0:y1, x0:x1].copy(), hr[y0:y1, x0:x1].copy())
                    regional = {}
                    for roi, (rx0, ry0, rx1, ry1) in sample.get('rois', {}).items():
                        regional[roi] = score(pred[ry0:ry1, rx0:rx1].copy(), hr[ry0:ry1, rx0:rx1].copy())
                        save_rgb(dest / f'{name}_{roi}.png', pred[ry0:ry1, rx0:rx1])
                    if preview:
                        save_rgb(dest / f'{name}.png', pred)
                    row['methods'][name] = dict(metrics=measured, rois=regional,
                      inference_and_observation_seconds=inference_times[name],
                      checkpoint=record['parent_checkpoint'] if name == 'parent' else record['methods'][name]['checkpoint'])
                row.update(seconds=time.monotonic() - began, completed_utc=utc(),
                           preview_directory=str(dest) if preview or sample.get('rois') else None)
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
                stream.flush()
                os.fsync(stream.fileno())
                rows.append(row)
                done.add((q, f))
                new_samples += 1
                write(out / 'progress.json', aggregate(rows, record, man, scene, False))
                print(json.dumps(dict(camera=q, frame=f, completed=len(rows), total=len(expected),
                                      metrics={m: {k: row['methods'][m]['metrics'][k] for k in METRIC_KEYS} for m in row['methods']},
                                      seconds=row['seconds'])), flush=True)
                del item, plain, predictions, hr, cm
                if args.max_new_samples and new_samples >= args.max_new_samples:
                    break
        for path, expected_hash in record['code'].items():
            if sha(path) != expected_hash:
                raise RuntimeError(f'runtime source changed during evaluation: {path}')
        finished = done == expected
        summary = aggregate(rows, record, man, scene, finished)
        export_csv(out / 'images.csv', rows)
        write(out / ('summary.json' if finished else 'progress.json'), summary)
        write(out / 'status.json', dict(status='complete' if finished else 'partial', utc=utc(),
                                        completed=len(rows), total=len(expected), parameter_updates=0,
                                        code_hashes_unchanged=True))
        print(json.dumps(dict(status='complete' if finished else 'partial', completed=len(rows), total=len(expected), out=str(out))), flush=True)
    except BaseException as exc:
        write(out / 'status.json', dict(status='failed', utc=utc(), error=repr(exc), traceback=traceback.format_exc(),
                                        completed=len(rows), total=len(expected), parameter_updates=0))
        raise


if __name__ == '__main__':
    main()
