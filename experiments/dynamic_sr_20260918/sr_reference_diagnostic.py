"""Diagnostic only: real held-out LR -> frozen SR, not novel-view synthesis.

No 3D model is loaded or updated. Outputs stay outside the train-prior directory.
This measures single-image SR quality under an easier information condition;
multi-view reconstruction is not mathematically bounded by this reference.
"""
import argparse
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from common import sha256
from n3dv_data import load_manifest
from evaluate import (aggregate, prepare_cache, read_rgb, spatial_metrics,
                      temporal_metrics, full_flow, write_json, write_rgb)
from generate_prior import (build_model, infer, DEFAULT_CHECKPOINT, DEFAULT_NETWORK,
                            CHECKPOINT_SHA256, NETWORK_SHA256)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scene', required=True, choices=['cook_spinach','cut_roasted_beef'])
    a = p.parse_args()
    project = Path(__file__).resolve().parents[2]
    out = project/'output/dynamic_sr_20260918'/f'{a.scene}_pilot_v1_sr_reference_diagnostic'
    if (out/'metrics.json').exists():
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    cv2.setNumThreads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    manifest = load_manifest(project/'data/dynamic_sr/n3dv_prepared'/a.scene/'manifest.json')
    observations = sorted([o for o in manifest['observations'] if o['split']=='test'],key=lambda x:x['frame_index'])
    assert len(observations)==60 and {o['camera_id'] for o in observations}=={'cam00'}
    cache, info = prepare_cache(manifest, observations,
         SimpleNamespace(dynamic_threshold=.025, flow_scale=.5, cache_dir=None))
    mask = np.asarray(Image.open(cache/'dynamic_mask.png')) > 0
    assert sha256(DEFAULT_NETWORK) == NETWORK_SHA256
    assert sha256(DEFAULT_CHECKPOINT) == CHECKPOINT_SHA256
    sys.path.insert(0, str(Path(__file__).parent/'vendor'))
    sr_model = build_model(DEFAULT_NETWORK, DEFAULT_CHECKPOINT, 'cuda')
    import lpips
    metric = lpips.LPIPS(net='alex').cuda().eval()
    rows = {'bicubic': [], 'swinir': []}
    last, previous_gt, sources = {}, None, []
    started = time.monotonic()
    width, height = manifest['resolutions']['hr']
    with torch.inference_mode():
        for i, observation in enumerate(observations):
            lr_path = Path(manifest['_root'])/observation['lr_path']
            hr_path = Path(manifest['_root'])/observation['hr_path']
            source = np.asarray(Image.open(lr_path).convert('RGB'))
            sources.append(dict(frame=observation['frame_index'],lr_path=str(lr_path),
                                lr_sha256=sha256(lr_path),hr_path=str(hr_path),hr_sha256=sha256(hr_path)))
            gt = read_rgb(hr_path)
            x = torch.from_numpy(source.copy()).permute(2,0,1).float().cuda()/255
            up = F.interpolate(x[None],size=(height,width),mode='bicubic',
                               align_corners=False,antialias=True)[0].clamp(0,1)
            predictions = {'bicubic': up.permute(1,2,0).cpu().numpy(),
                           'swinir': infer(sr_model,source,'cuda',0,32).astype(np.float32)/255}
            if i:
                with np.load(cache/info['flow_pairs'][i-1]['file']) as f:
                    flow, valid = full_flow(f['backward'],f['valid'],height,width)
            for mode, pred in predictions.items():
                row = dict(frame_index=observation['frame_index'],
                           spatial=spatial_metrics(pred,gt,mask,metric))
                if i:
                    row['temporal'] = temporal_metrics(last[mode],pred,previous_gt,gt,flow,valid,mask)
                rows[mode].append(row)
                last[mode] = pred
                write_rgb(out/mode/f'{observation["frame_index"]:04d}.png',pred)
            previous_gt = gt
    result = dict(scene=a.scene,manifest_sha256=sha256(manifest['_manifest_path']),
       cache_key=info['cache_key'],source='ACTUAL held-out cam00 LR images, diagnostic-only access',
       admissible_novel_view_baseline=False,used_for_training=False,
       interpretation='Reference capability of the frozen single-image SR model, not a mathematical upper bound on multi-view reconstruction.',
       prior_checkpoint_sha256=CHECKPOINT_SHA256,script_sha256=sha256(__file__),sources=sources,
       frame_indices=[o['frame_index'] for o in observations],elapsed_seconds=time.monotonic()-started,
       modes={mode:dict(rows=r,aggregate=aggregate(r,'spatial'),
                       temporal_aggregate=aggregate(r,'temporal')) for mode,r in rows.items()})
    write_json(out/'metrics.json',result)
    print({mode:value['aggregate']['full'] for mode,value in result['modes'].items()},flush=True)


if __name__ == '__main__': main()
