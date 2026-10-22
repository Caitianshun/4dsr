"""Frozen official SwinIR-M x4 teacher for the explicit SplatSuRe+SwinIR variant.

The paper's main tables use StableSR; this is its documented SISR substitution,
registered for comparison with the common generic pretrained SwinIR checkpoint.
No area4 scene fine-tuning is accepted. Import and --check-only do not load torch.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DEPENDENCIES = ROOT / 'experiments/dynamic_sr_20260918/vendor'
DEFAULT_NETWORK = ROOT / 'third_party/swinir_frozen_20261010/network_swinir.py'
DEFAULT_CHECKPOINT = ROOT / 'third_party/swinir_frozen_20261010/001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth'
NETWORK_SHA256 = '1d650d2c1c4519d95db771863691c6b239a0822222f287f6a8342a9e0eccbc6e'
OFFICIAL_NETWORK_SHA256 = '9e143898679ebeebc5d2fc94ad1b89c38aa4a4d43da4e0fcba0f93e476994913'
CHECKPOINT_SHA256 = '4e78e33f22c1aa8a773db0cf4a7381bae97c2362c717f155439ebc690cbd9215'
SWINIR_COMMIT = '6545850fbf8df298df73d81f3e8cba638787c8bd'
CHECKPOINT_URL = 'https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def identity(path):
    return dict(path=str(Path(path).resolve()), sha256=sha(path))


def inventory(manifest_path, network=DEFAULT_NETWORK, checkpoint=DEFAULT_CHECKPOINT):
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('scale') != 4:
        raise ValueError('SwinIR model is x4')
    train = set(manifest['splits']['train'])
    held = set(manifest['splits'].get('test', [])) | set(manifest['splits'].get('dev', []))
    if not train or train & held or 'cam00' in train:
        raise ValueError('invalid training camera split')
    targets = []
    seen = set()
    for row in manifest['observations']:
        if row['split'] != 'train':
            continue
        camera, frame = row['camera_id'], row['frame_index']
        if camera not in train or (camera, frame) in seen:
            raise ValueError('foreign or duplicate training observation')
        seen.add((camera, frame))
        source = (manifest_path.parent / row['lr_path']).resolve()
        expected = Path('lr') / camera / f'{frame:04d}.png'
        if Path(row['lr_path']) != expected or not source.is_relative_to(manifest_path.parent):
            raise ValueError('expected the registered scene-local LR PNG path')
        if sha(source) != row['lr_sha256']:
            raise ValueError('registered training LR bytes changed')
        targets.append(dict(camera=camera, frame=frame, time=row['time'], split='train',
                            lr_path=str(source), lr_sha256=row['lr_sha256']))
    expected = {(camera, frame) for camera in train for frame in manifest['frame_indices']}
    if seen != expected:
        raise ValueError('teacher inputs must cover the complete registered train split')
    network_hash = sha(network)
    if network_hash not in (NETWORK_SHA256, OFFICIAL_NETWORK_SHA256) or sha(checkpoint) != CHECKPOINT_SHA256:
        raise ValueError('SwinIR code or generic checkpoint differs; scene-finetuned priors are not this variant')
    targets.sort(key=lambda row: (row['frame'], row['camera']))
    return manifest, targets


def build_model(network=DEFAULT_NETWORK, checkpoint=DEFAULT_CHECKPOINT, device='cuda:0', dependency_path=DEFAULT_DEPENDENCIES):
    import torch
    if dependency_path is not None:
        sys.path.insert(0, str(Path(dependency_path).resolve()))
    import timm
    if timm.__version__ != "1.0.15":
        raise RuntimeError("registered CPU compatibility uses timm 1.0.15; record and validate a different version explicitly")
    if sha(network) not in (NETWORK_SHA256, OFFICIAL_NETWORK_SHA256) or sha(checkpoint) != CHECKPOINT_SHA256:
        raise ValueError('unrecognized SwinIR model identity')
    spec = importlib.util.spec_from_file_location('frozen_swinir_splatsure_' + SWINIR_COMMIT[:8], network)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.SwinIR(upscale=4, in_chans=3, img_size=64, window_size=8,
          img_range=1., depths=[6]*6, embed_dim=180, num_heads=[6]*6, mlp_ratio=2,
          upsampler='pixelshuffle', resi_connection='1conv')
    payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model.load_state_dict(payload['params'] if 'params' in payload else payload, strict=True)
    return model.requires_grad_(False).to(device=device, dtype=torch.float32).eval()


def infer(model, image_rgb, *, device='cuda:0', tile=0, overlap=32):
    """Return uint8 RGB exactly following official padding/crop/round semantics.

    Pure inference only. tile=0 is whole-image inference, official default.
    Tiling uses official uniform overlap averaging; changing it changes recipe.
    """
    import numpy as np
    import torch
    if tile < 0 or tile % 8 or overlap < 0:
        raise ValueError('tile must be zero/multiple of 8 and overlap nonnegative')
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3 or image_rgb.dtype != np.uint8:
        raise ValueError('teacher input must be uint8 RGB')
    tensor = torch.from_numpy(np.ascontiguousarray(image_rgb.transpose(2,0,1)))[None].to(device=device, dtype=torch.float32) / 255
    h, w = tensor.shape[-2:]
    if min(h, w) < 8:
        raise ValueError('official one-mirror padding requires image dimensions >= 8')
    hp, wp = (h//8+1)*8-h, (w//8+1)*8-w
    tensor = torch.cat([tensor, tensor.flip([2])], dim=2)[:, :, :h+hp, :]
    tensor = torch.cat([tensor, tensor.flip([3])], dim=3)[:, :, :, :w+wp]
    with torch.inference_mode():
        if not tile:
            output = model(tensor)
        else:
            hh, ww = tensor.shape[-2:]
            tile = min(tile, hh, ww)
            if tile % 8 or overlap >= tile:
                raise ValueError('actual tile must be multiple of 8 and exceed overlap')
            stride = tile - overlap
            ys = list(range(0, hh-tile, stride)) + [hh-tile]
            xs = list(range(0, ww-tile, stride)) + [ww-tile]
            output = tensor.new_zeros(1,3,hh*4,ww*4)
            counts = torch.zeros_like(output)
            for y in ys:
                for x in xs:
                    patch = model(tensor[...,y:y+tile,x:x+tile])
                    output[...,y*4:(y+tile)*4,x*4:(x+tile)*4] += patch
                    counts[...,y*4:(y+tile)*4,x*4:(x+tile)*4] += torch.ones_like(patch)
            output /= counts
        output = output[0,:,:h*4,:w*4].float().clamp(0,1).cpu().numpy()
        return (output.transpose(1,2,0)*255.).round().astype(np.uint8)


def produce(manifest_path, output_dir, *, network=DEFAULT_NETWORK, checkpoint=DEFAULT_CHECKPOINT,
            check_only=False, device='cuda:0', tile=0, overlap=32):
    manifest, targets = inventory(manifest_path, network, checkpoint)
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    receipt = dict(schema='wu_splatsure_swinir_teacher_v1', status='planned_CPU_only',
        scene=manifest['scene'], method='splatsure', recipe='SplatSuRe+SwinIR-M_DF2K_x4_generic',
        original_main_teacher='StableSR; this uses the paper-supported SwinIR variant',
        manifest=identity(manifest_path), training_only=True, HR_images_read=0, heldout_images_read=0,
        parameter_updates=0, fine_tuned_on_target=False, external_pretraining_data='DIV2K+Flickr2K',
        producer=[identity(__file__)], model_identities=[identity(network),identity(checkpoint)],
        upstream_commit=SWINIR_COMMIT, checkpoint_official_url=CHECKPOINT_URL,
        precision='float32', padding='official_mirror_concat_next_window', tile=tile, overlap=overlap,
        outputs='uint8_RGB_PNG_official_rounding',
        dependency_path=str(DEFAULT_DEPENDENCIES), timm_version='1.0.15', targets=targets)
    path = out/'receipt.json'
    path.write_text(json.dumps(receipt,indent=2)+'\n')
    if check_only:
        return receipt
    if not device.startswith('cuda') or not os.environ.get('CUDA_VISIBLE_DEVICES') or ',' in os.environ['CUDA_VISIBLE_DEVICES']:
        raise ValueError('full teacher generation requires one admitted visible GPU')
    import torch
    import numpy as np
    from PIL import Image
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    started=time.monotonic()
    try:
        model=build_model(network,checkpoint,device)
        receipt.update(status='generating',torch=torch.__version__,device=device)
        for row in targets:
            if sha(row['lr_path']) != row['lr_sha256']:
                raise ValueError('training LR changed after inventory')
            with Image.open(row['lr_path']) as image:
                array=np.asarray(image.convert('RGB')).copy()
            result=infer(model,array,device=device,tile=tile,overlap=overlap)
            destination=out/'sr'/row['camera']/f"{row['frame']:04d}.png"
            destination.parent.mkdir(parents=True,exist_ok=True)
            Image.fromarray(result,'RGB').save(destination)
            row['sr']=identity(destination)
        receipt.update(status='completed',seconds=time.monotonic()-started,training_LR_reads=len(targets))
    except BaseException as exc:
        receipt.update(status='failed',error_type=type(exc).__name__,error=str(exc),seconds=time.monotonic()-started)
        path.write_text(json.dumps(receipt,indent=2)+'\n')
        raise
    path.write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',required=True)
    parser.add_argument('--out',required=True)
    parser.add_argument('--network',type=Path,default=DEFAULT_NETWORK)
    parser.add_argument('--checkpoint',type=Path,default=DEFAULT_CHECKPOINT)
    parser.add_argument('--check-only',action='store_true')
    parser.add_argument('--tile',type=int,default=0)
    parser.add_argument('--overlap',type=int,default=32)
    args=parser.parse_args()
    result=produce(args.manifest,args.out,network=args.network,checkpoint=args.checkpoint,
                   check_only=args.check_only,tile=args.tile,overlap=args.overlap)
    print(json.dumps(dict(status=result['status'],targets=len(result['targets']))))

if __name__=='__main__':
    main()
