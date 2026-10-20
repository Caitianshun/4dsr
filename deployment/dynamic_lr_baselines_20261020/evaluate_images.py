"""One evaluator for all full-frame, held-out HR baseline renderings.

The main table uses 8-bit rendered RGB PNGs for every method. This is distinct
from historical raw-float render metrics and must not be silently mixed with
them. No model is loaded or updated by this program.
"""
import argparse
import ast
import csv
import hashlib
import json
import os
from pathlib import Path
import socket
import time
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT/'experiments/dynamic_sr_20260918/evaluate.py'


def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):
            value.update(block)
    return value.hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--manifest',required=True)
    parser.add_argument('--render-dir',required=True)
    parser.add_argument('--pattern',default='{frame:04d}.png')
    parser.add_argument('--render-index',help='Optional JSON list of {frame,path} relative to render-dir')
    parser.add_argument('--out',required=True)
    parser.add_argument('--method',required=True)
    parser.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    args=parser.parse_args()
    import cv2
    import lpips
    import numpy as np
    from PIL import Image
    import torch
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    os.environ.setdefault('TORCH_HOME',str(ROOT/'.cache/torch'))
    torch.hub.set_dir(str(ROOT/'.cache/torch/hub'))
    names={'read_rgb','ssim_map_rgb','masked_mean','metric_psnr','spatial_metrics'}
    nodes=[node for node in ast.parse(SOURCE.read_text()).body if isinstance(node,ast.FunctionDef) and node.name in names]
    assert {node.name for node in nodes}==names
    scope=dict(np=np,cv2=cv2,torch=torch,Image=Image)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(SOURCE),'exec'),scope)
    metric=SimpleNamespace(**{name:scope[name] for name in names})
    manifest_path=Path(args.manifest).resolve()
    manifest=json.loads(manifest_path.read_text())
    observations=sorted([o for o in manifest['observations'] if o['split']=='test'],key=lambda o:o['frame_index'])
    assert [o['frame_index'] for o in observations]==list(range(300))
    assert {o['camera_id'] for o in observations}=={'cam00'}
    directory=Path(args.render_dir).resolve()
    if args.render_index:
        index=json.loads(Path(args.render_index).read_text())
        paths={int(row['frame']):directory/row['path'] for row in index}
        assert set(paths)==set(range(300)) and len(index)==300
    else:
        paths={i:directory/args.pattern.format(frame=i) for i in range(300)}
    assert all(p.is_file() for p in paths.values())
    out=Path(args.out)
    out.mkdir(parents=True,exist_ok=False)
    model=lpips.LPIPS(net='alex',version='0.1',spatial=False,pnet_rand=False,pretrained=True,verbose=False).to(args.device).eval().requires_grad_(False)
    weights_hash=hashlib.sha256()
    for key,value in sorted(model.state_dict().items()):
        weights_hash.update(key.encode());weights_hash.update(value.detach().cpu().numpy().tobytes())
    rows=[]
    start=time.monotonic()
    for observation in observations:
        frame=observation['frame_index']
        pred_path=paths[frame]
        reference=manifest_path.parent/observation['hr_path']
        assert sha(reference)==observation['hr_sha256']
        with Image.open(pred_path) as im:
            assert im.size==tuple(manifest['resolutions']['hr']) and im.mode=='RGB'
        pred,gt=metric.read_rgb(pred_path),metric.read_rgb(reference)
        value=metric.spatial_metrics(pred,gt,np.zeros(pred.shape[:2],bool),model,args.device)['full']
        rows.append(dict(frame=frame,psnr=value['psnr'],ssim=value['ssim'],lpips=value['lpips_alex'],
                         mse=value['mse'],render_sha256=sha(pred_path),reference_sha256=observation['hr_sha256']))
    with (out/'per_frame.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    result=dict(schema='dynamic_lr_shared_HR_image_metrics_v1',status='completed',method=args.method,scene=manifest['scene'],
                camera='cam00',frames=300,resolution=manifest['resolutions']['hr'],input_representation='8-bit rendered RGB PNG for every method',
                test={key:float(np.mean([row[key] for row in rows])) for key in ['psnr','ssim','lpips']},
                psnr_aggregation='arithmetic mean of per-frame full-image PSNR',
                ssim_definition='RGB 11x11 Gaussian sigma1.5 population covariance; exclude outer5px',
                lpips_definition='AlexNet v0.1 input[-1,1] full-image scalar',
                manifest_sha256=sha(manifest_path),helper_source_sha256=sha(SOURCE),evaluator_sha256=sha(Path(__file__)),
                lpips_state_dict_sha256=weights_hash.hexdigest(),host=socket.gethostname(),torch=torch.__version__,
                numpy=np.__version__,opencv=cv2.__version__,device=args.device,
                gpu=torch.cuda.get_device_name() if args.device=='cuda' else None,seconds=time.monotonic()-start,
                render_directory=str(directory))
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
