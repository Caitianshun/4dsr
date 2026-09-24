"""Inference-only baselines: held-out LR render -> bicubic or frozen SwinIR.

No held-out LR/HR pixels enter the model input; references are evaluator-only.
Both baselines receive the same rounded uint8 rendering, stored for audit.
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

from common import load_checkpoint, render_image, sha256
from n3dv_data import load_manifest, N3DVPreparedDataset, observation_to_4dgs_camera
from evaluate import (aggregate, prepare_cache, read_rgb, spatial_metrics,
                      temporal_metrics, full_flow, write_json, write_rgb)
from generate_prior import (build_model, infer, DEFAULT_CHECKPOINT, DEFAULT_NETWORK,
                            CHECKPOINT_SHA256, NETWORK_SHA256)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scene', required=True)
    a = p.parse_args()
    project = Path(__file__).resolve().parents[2]
    root = project/'output/dynamic_sr_20260918'
    out = root/f'{a.scene}_pilot_v1_postrender_controls'
    if (out/'metrics.json').exists():
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    cv2.setNumThreads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    manifest = load_manifest(project/'data/dynamic_sr/n3dv_prepared'/a.scene/'manifest.json')
    data = N3DVPreparedDataset(manifest, 'test', 'lr')
    data.observations.sort(key=lambda x:x['frame_index'])
    cache, cache_info = prepare_cache(manifest, data.observations,
         SimpleNamespace(dynamic_threshold=.025, flow_scale=.5, cache_dir=None))
    mask = np.asarray(Image.open(cache/'dynamic_mask.png')) > 0
    checkpoint = root/f'{a.scene}_pilot_v1_lr_native/checkpoint_final.pt'
    g, _, _, _ = load_checkpoint(checkpoint)
    g._deformation.eval()
    assert sha256(DEFAULT_NETWORK) == NETWORK_SHA256
    assert sha256(DEFAULT_CHECKPOINT) == CHECKPOINT_SHA256
    sys.path.insert(0, str(Path(__file__).parent/'vendor'))
    sr_model = build_model(DEFAULT_NETWORK, DEFAULT_CHECKPOINT, 'cuda')
    import lpips
    metric = lpips.LPIPS(net='alex').cuda().eval()
    rows = {'bicubic': [], 'swinir': []}
    last, previous_gt = {}, None
    started = time.monotonic()
    height, width = reversed(manifest['resolutions']['hr'])
    with torch.inference_mode():
        for i in range(len(data)):
            item = data[i]
            camera = observation_to_4dgs_camera(item, i)
            native = render_image(g,camera)['render'].clamp(0,1)
            source = native.mul(255).round().byte().permute(1,2,0).cpu().numpy()
            name = f'{item["frame_index"]:04d}.png'
            (out/'lr_render_inputs').mkdir(exist_ok=True)
            Image.fromarray(source).save(out/'lr_render_inputs'/name)
            # The evaluator opens HR only after the model input has been fixed.
            gt = read_rgb(Path(manifest['_root'])/data.observations[i]['hr_path'])
            x = torch.from_numpy(source.copy()).permute(2,0,1).float().cuda()/255
            up = F.interpolate(x[None],size=(height,width),mode='bicubic',
                               align_corners=False,antialias=True)[0].clamp(0,1)
            predictions = {'bicubic': up.permute(1,2,0).cpu().numpy(),
                           'swinir': infer(sr_model,source,'cuda',0,32).astype(np.float32)/255}
            if i:
                with np.load(cache/cache_info['flow_pairs'][i-1]['file']) as f:
                    flow, valid = full_flow(f['backward'],f['valid'],height,width)
            for mode, pred in predictions.items():
                row = dict(frame_index=item['frame_index'],
                           spatial=spatial_metrics(pred,gt,mask,metric))
                if i:
                    row['temporal'] = temporal_metrics(last[mode],pred,previous_gt,gt,flow,valid,mask)
                rows[mode].append(row)
                last[mode] = pred
                write_rgb(out/mode/name,pred)
            previous_gt = gt
    result = dict(scene=a.scene,checkpoint=str(checkpoint),checkpoint_sha256=sha256(checkpoint),
       manifest_sha256=sha256(manifest['_manifest_path']),cache_key=cache_info['cache_key'],
       source='native LR novel-view render, rounded uint8, no held-out image pixels as input',
       prior_checkpoint_sha256=CHECKPOINT_SHA256,script_sha256=sha256(__file__),
       frame_indices=[x['frame_index'] for x in data.observations],elapsed_seconds=time.monotonic()-started,
       modes={mode:dict(rows=r,aggregate=aggregate(r,'spatial'),
                       temporal_aggregate=aggregate(r,'temporal')) for mode,r in rows.items()})
    write_json(out/'metrics.json',result)
    print({mode: value['aggregate']['full'] for mode,value in result['modes'].items()},flush=True)


if __name__ == '__main__': main()
