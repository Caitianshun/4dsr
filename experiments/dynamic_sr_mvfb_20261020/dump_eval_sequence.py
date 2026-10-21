"""Export the fixed cam00 frames 144..151 as a visual temporal probe only.

No training or main metric calculation. Existing 300-frame scores remain the
quantitative reference. Frozen S1 B2/D checkpoints are used with their exact
training-version mvfb source files. Query LR and teachers never enter inference;
cam00 HR is decoded after all three predictions and saved only as a reference.

  dump_eval_sequence.py --scene Cook07 --b2 RUN_B2/final.pt --d RUN_D/final.pt \
      --out NEW_DIR --device cpu [--cache-dir CACHE_CAM00] [--project-root ROOT]

Requires cam00 cache meta.json and f0144.npz..f0151.npz, their six source LR
images, cam00 HR references, the prepared manifest, B2/D checkpoint/run/complete
files and the parent all-camera cache metadata referenced by each run.
CUDA resource admission is the caller's responsibility under project rules.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
FRAMES = list(range(144, 152))


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            value.update(chunk)
    return value.hexdigest()


def identity(path):
    p = Path(path).resolve()
    return dict(path=str(p), sha256=sha(p), bytes=p.stat().st_size)


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def write(path, value):
    p = Path(path)
    temp = p.with_name(p.name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.replace(p)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--scene', choices=('Cook07', 'Discussion07'), required=True)
    parser.add_argument('--b2', required=True)
    parser.add_argument('--d', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--cache-dir')
    parser.add_argument('--metrics-reference', help='optional existing full-300-frame summary/CSV, recorded by hash only')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--project-root', default=os.environ.get('MVFB_EVAL_PROJECT_ROOT', str(HERE.parents[1])))
    parser.add_argument('--threads', type=int, default=8)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('threads must be positive')
    if args.device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['OMP_NUM_THREADS'] = str(args.threads)
    os.environ['MKL_NUM_THREADS'] = str(args.threads)
    root = Path(args.project_root).resolve()

    def path(value):
        p = Path(value)
        return p.resolve() if p.is_absolute() else (root / p).resolve()

    sys.path.insert(0, str(HERE))
    import numpy as np
    import torch
    from PIL import Image
    import mvfb.data as data_module
    from mvfb.data import Scene
    from mvfb.controls import MVFBWide2
    from mvfb.model import MVFB, Crops, FrameSources
    data_module.ROOT = root
    data_module.PREP = root / 'data/dynamic_sr/full_time_20261007/prepared'
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cache_name = data_module.SCENES[args.scene][1]
    cache = path(args.cache_dir) if args.cache_dir else root / 'output/dynamic_sr_mvfb_20261020/cache' / cache_name
    if cache.name != cache_name:
        raise ValueError('Scene requires the original cam00 cache folder name')
    scene = Scene(args.scene, cache_root=cache.parent, frames=FRAMES)
    if scene.cm.target != 'cam00' or 'cam00' in scene.cm.sources:
        raise ValueError('cam00 must be held out and excluded from all sources')
    if len(scene.cm.sources) != 6:
        raise ValueError('this fixed probe expects six source cameras')
    models, checkpoints = {}, {}
    for name, filename, variant, cls in [('B2', args.b2, 'render_only_wide2', MVFBWide2), ('D', args.d, 'full', MVFB)]:
        cp = path(filename)
        run_path, complete_path = cp.parent / 'run.json', cp.parent / 'complete.json'
        run = json.loads(run_path.read_text())
        if not complete_path.is_file() or run.get('kind') != 'mvfb_train_samescene' or run.get('scene') != args.scene:
            raise ValueError(f'{name}: not a completed S1 run of {args.scene}')
        if run['args']['variant'] != variant or 'cam00' in run['queries']:
            raise ValueError(f'{name}: unexpected variant or query-camera pool')
        if not set(scene.cm.sources) <= set(run['queries']):
            raise ValueError(f'{name}: novel-view sources are outside the S1 training-camera pool')
        for filename_, expected in run['sources_code'].items():
            relative = Path(filename_)
            try:
                short = relative.relative_to('experiments/dynamic_sr_mvfb_20261020')
            except ValueError:
                continue
            if short.parts[0] == 'mvfb' and sha(HERE / short) != expected:
                raise ValueError(f'{name}: model/adapter source differs from its training version: {filename_}')
        parent_meta = path(run['cache']) / 'meta.json'
        parent = json.loads(parent_meta.read_text())
        if parent['checkpoint']['sha256'] != scene.meta['checkpoint']['sha256'] or parent.get('cull_record') != scene.meta.get('cull_record'):
            raise ValueError(f'{name}: cam00 cache has a different parent checkpoint or culling state')
        ck = torch.load(cp, map_location='cpu', weights_only=False)
        if ck.get('step') != run['args']['steps'] or ck.get('variant') != variant:
            raise ValueError(f'{name}: incomplete checkpoint')
        expected_class = {'B2': 'MVFBWide2', 'D': 'MVFB'}[name]
        if ck.get('model_class', 'MVFB') != expected_class:
            raise ValueError(f'{name}: unexpected model class')
        model = cls(**ck['kwargs']).eval().requires_grad_(False)
        model.load_state_dict(ck['state'])
        models[name] = model
        checkpoints[name] = dict(checkpoint=identity(cp), run=identity(run_path), complete=identity(complete_path),
                                 parent_cache=identity(parent_meta), training_code_hashes=run['sources_code'])
    prepared_path = scene.prep / 'manifest_train_ready_seed20261007.json'
    if sha(prepared_path) != scene.meta['manifest']['sha256']:
        raise ValueError('prepared manifest changed since cache creation')
    inputs = {}
    for frame in FRAMES:
        inputs[str(scene.cache / f'f{frame:04d}.npz')] = identity(scene.cache / f'f{frame:04d}.npz')
        for camera in ['cam00'] + scene.cm.sources:
            obs = scene.obs[camera, frame]
            role = 'hr' if camera == 'cam00' else 'lr'
            filename_ = scene.prep / obs[f'{role}_path']
            found = identity(filename_)
            if obs.get(f'{role}_sha256') and found['sha256'] != obs[f'{role}_sha256']:
                raise ValueError(f'input image hash changed: {filename_}')
            inputs[str(filename_)] = found
    code = {str(p): sha(p) for p in [Path(__file__)] + sorted((HERE / 'mvfb').glob('*.py'))}
    fixed = dict(kind='cam00_visual_temporal_probe', scene=args.scene, camera='cam00', frames=FRAMES,
                 sources=scene.cm.sources, query_excluded_from_sources=True, cache_meta=identity(scene.cache / 'meta.json'),
                 prepared_manifest=identity(prepared_path), checkpoints=checkpoints, inputs=inputs, code=code,
                 output_rule='same original canonical cam00 output as evaluate.py: float clamp [0,1], PNG round(value*255)',
                 HR_role='reference only, decoded after all method predictions', parameter_updates=0,
                 existing_quantitative_reference=identity(path(args.metrics_reference)) if args.metrics_reference else None,
                 interpretation='8 consecutive held-out-view frames for visual temporal review; not a new-time test or training-fitting mean; quantitative metrics reuse existing full-300-frame evaluation')
    out = path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'identity.json').exists() and json.loads((out / 'identity.json').read_text()) != fixed:
        raise ValueError('probe identity changed; use a new output directory')
    if (out / 'frames.jsonl').exists() or (out / 'summary.json').exists():
        raise FileExistsError('existing probe frames preserved; use a new output directory')
    write(out / 'identity.json', fixed)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; caller must admit resources before dispatch')
    for model in models.values():
        model.to(args.device)
    cm = scene.cm.clone_to(args.device)
    origins = torch.tensor([[0, 0]])
    rows = []
    runtime = dict(utc=utc(), device=args.device, threads=args.threads, torch=torch.__version__, numpy=np.__version__,
                   executable=sys.executable, CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),
                   hardware=torch.cuda.get_device_name() if args.device == 'cuda' else 'CPU')
    write(out / 'runtime.json', runtime)
    try:
        for frame in FRAMES:
            began = time.monotonic()
            item = scene.load(frame, supervision=False)
            if any(k in item for k in ('HR', 'LR', 'LRq', 'T', 'Tq')):
                raise ValueError('query supervision entered the forward inputs')
            outputs = {}
            with torch.inference_mode():
                empty = SimpleNamespace(cm=cm, K=0)
                plain = Crops(item, empty, origins, (scene.H, scene.W), args.device, transport_sources=False)
                for name in ('parent', 'B2', 'D'):
                    if name == 'parent':
                        prediction = plain.R
                    elif name == 'B2':
                        prediction, _ = models[name](empty, plain)
                    else:
                        src = FrameSources(item, cm, args.device)
                        crops = Crops(item, src, origins, (scene.H, scene.W), args.device, transport_sources=True)
                        prediction, _ = models[name](src, crops)
                    outputs[name] = prediction[0].detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
                    if name == 'D':
                        del src, crops
                    del prediction
            obs = scene.obs['cam00', frame]
            with Image.open(scene.prep / obs['hr_path']) as reference:
                outputs['HR'] = np.asarray(reference.convert('RGB'), dtype=np.float32) / 255.0
            image_ids = {}
            for name, array in outputs.items():
                if array.shape != (scene.H, scene.W, 3) or not np.isfinite(array).all():
                    raise ValueError(f'{name}: invalid output dimensions or non-finite pixels')
                filename_ = out / name / f'{frame:04d}.png'
                filename_.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(np.rint(array * 255).astype(np.uint8)).save(filename_)
                image_ids[name] = identity(filename_)
            row = dict(camera='cam00', frame=frame, dataset_time=obs['time'], physical_seconds_at_30fps=frame / 30,
                       completed_utc=utc(), seconds=time.monotonic() - began, forward_input_keys=sorted(item), images=image_ids)
            rows.append(row)
            with (out / 'frames.jsonl').open('a') as stream:
                stream.write(json.dumps(row, allow_nan=False) + '\n')
                stream.flush()
                os.fsync(stream.fileno())
            write(out / 'status.json', dict(status='running', completed_frames=len(rows), expected_frames=8, utc=utc()))
            print(json.dumps(dict(frame=frame, seconds=row['seconds'], completed=len(rows))), flush=True)
            del outputs, item, plain
        if any(sha(p) != expected for p, expected in code.items()):
            raise RuntimeError('frozen runtime code changed during probe')
        write(out / 'summary.json', dict(fixed, status='complete', runtime=runtime, per_frame=rows,
                                          code_hashes_unchanged=True, completed_utc=utc()))
        write(out / 'status.json', dict(status='complete', completed_frames=8, expected_frames=8, utc=utc()))
    except BaseException as exc:
        write(out / 'status.json', dict(status='failed', completed_frames=len(rows), expected_frames=8, utc=utc(), error=repr(exc)))
        raise


if __name__ == '__main__':
    main()
