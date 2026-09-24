#!/usr/bin/env python3
"""ZJU split adapter of frozen official SwinIR-M DF2K x4 prior.

Only change: cam00 may be training when the explicit disjoint manifest says so.
Network, weights, tiling and image operations remain the existing frozen prior.

Train-LR inputs only.

No HR image is opened or used by this program. Native prepared manifest uses
observations=[{camera_id, frame_index, split, lr_path, ...}], with paths relative
to its directory. The compact splits.train + frames + lr_dir schema is also
accepted. Test/dev observations are never selected.

The existing official weights and network source are read-only inputs. Their
SHA256 values are pinned and verified. Tiled inference follows the official
uniform-overlap averaging implementation; tile configuration is recorded and
must not change when resuming. Batch size is one.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

SCRIPT_ROOT = Path(__file__).resolve().parent
DEFAULT_NETWORK = Path(os.environ.get(
    'FOURDSR_SWINIR_NETWORK',
    '/home/cai_tianshun/Project/mml/scripts/network_swinir.py')).expanduser()
DEFAULT_CHECKPOINT = Path(os.environ.get(
    'FOURDSR_SWINIR_CHECKPOINT',
    '/home/cai_tianshun/Project/mml/outputs/week3_swinir/ckpt/001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth')).expanduser()
NETWORK_SHA256 = '1d650d2c1c4519d95db771863691c6b239a0822222f287f6a8342a9e0eccbc6e'
CHECKPOINT_SHA256 = '4e78e33f22c1aa8a773db0cf4a7381bae97c2362c717f155439ebc690cbd9215'
CHECKPOINT_URL = 'https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth'
WINDOW = 8
SCALE = 4


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value) -> None:
    temp = path.with_suffix(path.suffix + f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temp.replace(path)


def child_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f'Path leaves prepared scene directory: {relative}')
    return path


def load_inventory(manifest_path: Path, cameras: str | None):
    manifest = json.loads(manifest_path.read_text())
    if int(manifest.get('scale', 4)) != SCALE:
        raise ValueError('This checkpoint and script support x4 only')
    splits = manifest['splits']
    train = list(splits['train'])
    forbidden = set(splits.get('test', [])) | set(splits.get('dev', [])) | set(splits.get('val', []))
    if len(train) != len(set(train)) or set(train) & forbidden :
        raise ValueError('Invalid train split: duplicate or held-out camera present')
    selected = train if cameras is None else cameras.split(',')
    if not selected or not set(selected) <= set(train) or len(selected) != len(set(selected)):
        raise ValueError('Requested cameras must be a unique subset of manifest train cameras')
    if 'observations' in manifest:
        inventory = []
        seen = set()
        for observation in manifest['observations']:
            camera = observation['camera_id']
            if observation['split'] != 'train' or camera not in selected:
                continue
            relative_lr = Path(observation['lr_path'])
            if (len(relative_lr.parts) != 3 or relative_lr.parts[0] not in ('lr', 'lr_x4')
                    or relative_lr.parts[1] != camera or relative_lr.suffix.lower() != '.png'):
                raise ValueError(f'Unexpected train LR path: {relative_lr}')
            source = child_path(manifest_path.parent, str(relative_lr))
            relative = f'{camera}/{relative_lr.name}'
            if relative in seen:
                raise ValueError(f'Duplicate train observation: {relative}')
            if not source.is_file():
                raise FileNotFoundError(source)
            seen.add(relative)
            inventory.append((relative, source))
        if not inventory or {Path(rel).parts[0] for rel, _ in inventory} != set(selected):
            raise ValueError('Manifest has no train observations for one or more requested cameras')
        return manifest, sorted(inventory)
    lr_root = child_path(manifest_path.parent, manifest.get('lr_dir', 'lr_x4'))
    if 'lr' not in lr_root.name.lower():
        raise ValueError('Input directory must explicitly identify LR data')
    frame_names = []
    for record in manifest['frames']:
        if isinstance(record, dict):
            name = record.get('file', record.get('filename'))
            if name is None:
                frame_id = record.get('frame_id', record.get('frame'))
                name = manifest.get('frame_pattern', '{frame:06d}.png').format(frame=int(frame_id))
        else:
            name = manifest.get('frame_pattern', '{frame:06d}.png').format(frame=int(record))
        if Path(name).name != name or Path(name).suffix.lower() != '.png':
            raise ValueError(f'Expected plain PNG frame filename, got {name}')
        frame_names.append(name)
    if not frame_names or len(set(frame_names)) != len(frame_names):
        raise ValueError('Frame inventory is empty or contains duplicates')
    inventory = []
    for camera in selected:
        if Path(camera).name != camera or not camera.startswith('cam'):
            raise ValueError(f'Invalid camera name: {camera}')
        for frame_name in frame_names:
            relative = f'{camera}/{frame_name}'
            source = child_path(lr_root, relative)
            if not source.is_file():
                raise FileNotFoundError(source)
            inventory.append((relative, source))
    return manifest, inventory


def build_model(network: Path, checkpoint: Path, device: str):
    import torch
    spec = importlib.util.spec_from_file_location('official_swinir_network', network)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    model = module.SwinIR(upscale=SCALE, in_chans=3, img_size=64,
                         window_size=WINDOW, img_range=1., depths=[6] * 6,
                         embed_dim=180, num_heads=[6] * 6, mlp_ratio=2,
                         upsampler='pixelshuffle', resi_connection='1conv')
    weights = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model.load_state_dict(weights['params'] if 'params' in weights else weights, strict=True)
    model.requires_grad_(False)
    return model.to(device=device, dtype=torch.float32).eval()


def infer(model, array, device: str, tile: int, overlap: int):
    import numpy as np
    import torch
    x = torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1))).unsqueeze(0).to(device=device, dtype=torch.float32) / 255.
    h_original, w_original = x.shape[-2:]
    # Match official SwinIR testing: mirror-concatenate to the next window,
    # including a full window when the input is already divisible by eight.
    h_pad = (h_original // WINDOW + 1) * WINDOW - h_original
    w_pad = (w_original // WINDOW + 1) * WINDOW - w_original
    x = torch.cat([x, torch.flip(x, [2])], dim=2)[:, :, :h_original + h_pad, :]
    x = torch.cat([x, torch.flip(x, [3])], dim=3)[:, :, :, :w_original + w_pad]
    h, w = x.shape[-2:]
    with torch.inference_mode():
        if not tile:
            output = model(x)
        else:
            actual_tile = min(tile, h, w)
            if actual_tile % WINDOW or overlap >= actual_tile:
                raise ValueError('Tile must be divisible by eight and larger than overlap')
            stride = actual_tile - overlap
            hs = list(range(0, h - actual_tile, stride)) + [h - actual_tile]
            ws = list(range(0, w - actual_tile, stride)) + [w - actual_tile]
            output = x.new_zeros((1, 3, h * SCALE, w * SCALE))
            counts = x.new_zeros((1, 1, h * SCALE, w * SCALE))
            for hi in hs:
                for wi in ws:
                    patch = model(x[..., hi:hi + actual_tile, wi:wi + actual_tile])
                    output[..., hi*SCALE:(hi+actual_tile)*SCALE, wi*SCALE:(wi+actual_tile)*SCALE].add_(patch)
                    counts[..., hi*SCALE:(hi+actual_tile)*SCALE, wi*SCALE:(wi+actual_tile)*SCALE].add_(1.)
            output.div_(counts)
        output = output[..., :h_original * SCALE, :w_original * SCALE]
        return output.squeeze(0).clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument('--network', type=Path, default=DEFAULT_NETWORK)
    parser.add_argument('--dependency-path', type=Path, default=SCRIPT_ROOT.parent / 'dynamic_sr_20260918/vendor')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--tile', type=int, default=0, help='LR tile size; 0 means full image')
    parser.add_argument('--overlap', type=int, default=32)
    parser.add_argument('--cameras', help='Comma-separated subset of train cameras')
    parser.add_argument('--limit', type=int, help='Smoke-test limit; same output is resumable')
    parser.add_argument('--dry-run', action='store_true', help='Validate inventory and SHA without loading model')
    parser.add_argument('--log-every', type=int, default=20)
    args = parser.parse_args()
    if args.tile < 0 or args.tile % WINDOW or args.overlap < 0:
        parser.error('tile must be zero or a positive multiple of eight; overlap must be nonnegative')
    if args.limit is not None and args.limit <= 0:
        parser.error('limit must be positive')
    manifest_path = args.manifest.resolve()
    manifest, inventory = load_inventory(manifest_path, args.cameras)
    for path, expected in ((args.network, NETWORK_SHA256), (args.checkpoint, CHECKPOINT_SHA256)):
        observed = sha256(path)
        if observed != expected:
            raise ValueError(f'Pinned source mismatch: {path}; actual SHA256 {observed}')
    print(json.dumps({'scene': manifest.get('scene'), 'train_images_selected': len(inventory),
                      'checkpoint_sha256': CHECKPOINT_SHA256, 'dry_run': args.dry_run}), flush=True)
    if args.dry_run:
        return
    sys.path.insert(0, str(args.dependency_path.resolve()))
    import numpy as np
    from PIL import Image
    import torch
    import timm
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    out_root = manifest_path.parent / 'sr_swinir_x4'
    out_root.mkdir(parents=True, exist_ok=True)
    config = {'schema': 1, 'manifest': str(manifest_path), 'manifest_sha256': sha256(manifest_path),
              'checkpoint': str(args.checkpoint.resolve()), 'checkpoint_sha256': CHECKPOINT_SHA256,
              'checkpoint_official_url': CHECKPOINT_URL, 'network_sha256': NETWORK_SHA256,
              'generator_sha256': sha256(Path(__file__)),
              'scale': SCALE, 'window': WINDOW, 'precision': 'float32', 'tile': args.tile,
              'overlap': args.overlap, 'padding': 'official_mirror_concat_next_window',
              'external_training_data': 'DIV2K training + Flickr2K (DF2K)',
              'fine_tuned_on_target': False, 'input_policy': 'manifest train LR only; no HR image reads',
              'torch_version': torch.__version__, 'timm_version': timm.__version__,
              'numpy_version': np.__version__, 'pillow_version': Image.__version__}
    config_path = out_root / 'prior_config.json'
    if config_path.exists():
        previous = json.loads(config_path.read_text())
        if previous != config:
            raise ValueError('Existing prior configuration differs; refusing to mix results')
    else:
        atomic_json(config_path, config)
    model = build_model(args.network, args.checkpoint, args.device)
    job_id = f'{int(time.time())}_{os.getpid()}'
    log_path = out_root / f'generation_{job_id}.jsonl'
    selected_inventory = inventory if args.limit is None else inventory[:args.limit]
    generated = skipped = 0
    started = time.monotonic()
    with log_path.open('w', buffering=1) as log:
        for relative, source in selected_inventory:
            input_sha = sha256(source)
            target = out_root / relative
            receipt = target.with_suffix('.json')
            if target.exists() and receipt.exists():
                record = json.loads(receipt.read_text())
                if record['input_sha256'] != input_sha or record['output_sha256'] != sha256(target):
                    raise ValueError(f'Changed input or output during resume: {relative}')
                skipped += 1
                continue
            if target.exists() or receipt.exists():
                raise ValueError(f'Incomplete output/receipt pair; inspect before retry: {target}')
            t0 = time.monotonic()
            with Image.open(source) as image:
                array = np.asarray(image.convert('RGB'))
            output = infer(model, array, args.device, args.tile, args.overlap)
            if output.shape != (array.shape[0] * SCALE, array.shape[1] * SCALE, 3):
                raise ValueError(f'Wrong x4 output size for {relative}: {output.shape}')
            target.parent.mkdir(parents=True, exist_ok=True)
            temp = target.with_suffix(f'.{os.getpid()}.tmp')
            Image.fromarray(output).save(temp, format='PNG')
            temp.replace(target)
            record = {'relative_path': relative, 'input_sha256': input_sha,
                      'output_sha256': sha256(target), 'input_hw': list(array.shape[:2]),
                      'output_hw': list(output.shape[:2]), 'seconds': time.monotonic() - t0}
            atomic_json(receipt, record)
            log.write(json.dumps(record) + '\n')
            generated += 1
            if generated == 1 or generated % args.log_every == 0:
                print(json.dumps({'generated': generated, 'skipped': skipped,
                                  'selected': len(selected_inventory), 'seconds': round(time.monotonic()-started, 2),
                                  'latest': relative, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(args.device) if args.device.startswith('cuda') else None}), flush=True)
    summary = {'job_id': job_id, 'scene': manifest.get('scene'), 'generated': generated,
               'skipped': skipped, 'selected': len(selected_inventory), 'train_inventory_selected': len(inventory),
               'selection_complete': len(selected_inventory) == len(inventory),
               'seconds': time.monotonic() - started, 'device': args.device,
               'gpu_name': torch.cuda.get_device_name(args.device) if args.device.startswith('cuda') else None}
    atomic_json(out_root / f'summary_{job_id}.json', summary)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
