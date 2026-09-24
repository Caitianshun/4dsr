"""Same frozen SwinIR teacher, consuming unquantized author-code LR only."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / 'dynamic_sr_20260918'
sys.path.insert(0, str(OLD))
sys.path.insert(0, str(OLD / 'vendor'))
from generate_prior import (build_model, infer, NETWORK_SHA256, CHECKPOINT_SHA256,
                            sha256, atomic_json)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--network', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--cameras', required=True)
    p.add_argument('--tile', type=int, default=128)
    p.add_argument('--limit', type=int)
    a = p.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    m = json.loads(a.manifest.read_text())
    assert m['comparison_protocol'] == 'sr4d_author_lr_v1'
    cams = set(a.cameras.split(','))
    assert cams <= set(m['splits']['train']) and not cams & {'cam00', 'cam01'}
    assert sha256(a.network) == NETWORK_SHA256
    assert sha256(a.checkpoint) == CHECKPOINT_SHA256
    selected = [x for x in m['observations'] if x['split']=='train' and x['camera_id'] in cams]
    if a.limit:
        selected = selected[:a.limit]
    root = a.manifest.parent
    out = root / 'sr_swinir_x4'
    out.mkdir(exist_ok=True)
    config = dict(manifest_sha256=sha256(a.manifest), network_sha256=NETWORK_SHA256,
                  checkpoint_sha256=CHECKPOINT_SHA256, cameras=sorted(cams), tile=a.tile,
                  overlap=32, source_sha256=sha256(Path(__file__)), input='float LR; no quantization',
                  output='PNG 8-bit frozen teacher, same as existing baseline',
                  teacher_training_degradation='Classical x4 bicubic DF2K; input is now bilinear',
                  extra_training_data='DIV2K and Flickr2K', training_HR_decodes=0)
    cp = out / 'prior_config.json'
    if cp.exists():
        assert json.loads(cp.read_text()) == config
    else:
        atomic_json(cp, config)
    model = build_model(a.network, a.checkpoint, 'cuda')
    start = time.monotonic()
    rows = []
    for i, obs in enumerate(selected):
        path = root / obs['lr_float_path']
        assert sha256(path) == obs['lr_float_sha256']
        lr = np.load(path, allow_pickle=False)
        assert lr.dtype == np.float32 and lr.shape[0] == 3
        target = out / obs['camera_id'] / Path(obs['lr_path']).name
        target.parent.mkdir(exist_ok=True)
        if not target.exists():
            # infer's normalization divides float input by 255. Multiplication
            # here is floating point; there is no conversion to uint8.
            pred = infer(model, lr.transpose(1,2,0) * np.float32(255), 'cuda', a.tile, 32)
            Image.fromarray(pred).save(target)
        rows.append(dict(camera_id=obs['camera_id'], frame=obs['frame_index'],
                         lr_sha256=obs['lr_float_sha256'], prior_sha256=sha256(target)))
        if (i+1)%20 == 0:
            print(json.dumps(dict(done=i+1,total=len(selected),seconds=time.monotonic()-start)),flush=True)
    atomic_json(out / ('smoke_complete.json' if a.limit else 'complete.json'),
                dict(status='complete', rows=rows, seconds=time.monotonic()-start,
                     device=torch.cuda.get_device_name(), config=config))


if __name__ == '__main__':
    main()
