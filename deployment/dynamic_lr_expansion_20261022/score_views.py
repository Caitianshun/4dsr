"""Plan, validate, or explicitly score frozen RGB renders for train/test views.

Default operation writes an evaluation plan using standard-library metadata only.
The model is never loaded. HR pixels are opened only in explicit scoring mode.
"""
from __future__ import annotations
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
from prepare_suite import ROOT, LPIPS_STATE_SHA256, select_observations, sha, dump

HELPER = ROOT/'experiments/dynamic_sr_20260918/evaluate.py'


def expected_views(manifest_path: Path, scope: str):
    manifest=json.loads(manifest_path.read_text())
    rows=select_observations(manifest,scope)
    return manifest, rows, {(o['camera_id'],o['frame_index']) for o in rows}


def validate_index(path: Path, manifest_path: Path, scope: str, render_dir: Path, check_hashes=True):
    manifest, observations, expected=expected_views(manifest_path,scope)
    index=json.loads(path.read_text())
    if index['schema']!='dynamic_lr_multiview_render_v1':raise ValueError('Wrong render index schema')
    if index['manifest_sha256']!=sha(manifest_path) or index['scope']!=scope:raise ValueError('Render/manifest/scope mismatch')
    if index['resolution']!=manifest['resolutions']['hr']:raise ValueError('Wrong HR render resolution')
    if index['input_representation']!='uint8_RGB_PNG' or index['render_camera_transform']!='identity':
        raise ValueError('Expected uncorrected native uint8 RGB renders')
    paths={}
    for record in index['records']:
        key=(record['camera_id'],int(record['frame']))
        if key in paths:raise ValueError('Duplicate camera/frame render')
        p=(render_dir/record['path']).resolve()
        if not p.is_relative_to(render_dir.resolve()):raise ValueError('Render path escapes render directory')
        if not p.is_file():raise FileNotFoundError(p)
        if check_hashes and sha(p)!=record['render_sha256']:raise ValueError('Render hash changed: '+str(p))
        paths[key]=p
    if set(paths)!=expected:raise ValueError('Missing, extra, or wrong-camera renders')
    return manifest,observations,paths,index


def validate_output_directory(out: Path, expected: dict):
    """Accept an exact plan-only directory while preserving any prior result."""
    if not out.exists():return
    if not out.is_dir() or {p.name for p in out.iterdir()}!={'score_plan.json'}:
        raise FileExistsError('Preserve existing scoring results: '+str(out))
    planned=json.loads((out/'score_plan.json').read_text())
    if any(planned.get(k)!=v for k,v in expected.items()):
        raise ValueError('Existing score plan does not match this execution')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest',required=True,type=Path)
    ap.add_argument('--scope',required=True,choices=['train','test'])
    ap.add_argument('--render-dir',required=True,type=Path)
    ap.add_argument('--render-index',type=Path)
    ap.add_argument('--out',required=True,type=Path)
    ap.add_argument('--method',required=True)
    ap.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    ap.add_argument('--torch-home',type=Path,default=ROOT/'.cache/torch')
    mode=ap.add_mutually_exclusive_group()
    mode.add_argument('--plan-only',action='store_true',help='Default; no image inference or CUDA imports')
    mode.add_argument('--check-index',action='store_true',help='Only validate indexed files and hashes; no scoring')
    mode.add_argument('--execute',action='store_true',help='Explicit future execution; never used by preparation')
    args=ap.parse_args()
    mp=args.manifest.resolve();out=args.out.resolve();rd=args.render_dir.resolve()
    manifest,obs,expected=expected_views(mp,args.scope)
    if not args.execute and not args.check_index:
        plan={'schema':'dynamic_lr_view_score_plan_v1','status':'HELD_NOT_SCORED','execution_enabled':False,'method':args.method,
              'manifest':str(mp),'manifest_sha256':sha(mp),'scope':args.scope,'images':len(obs),
              'cameras':sorted({o['camera_id'] for o in obs}),'frames':sorted({o['frame_index'] for o in obs}),
              'resolution':manifest['resolutions']['hr'],'render_dir':str(rd),'render_index':str(args.render_index or rd/'render_index.json'),
              'metrics':['PSNR','SSIM','LPIPS'],'metric_helper_sha256':sha(HELPER),'input_representation':'uint8_RGB_PNG',
              'HR_use':'scoring_only_after_native_prediction','requires_cached_LPIPS_weights':True,
              'records':[{'camera_id':o['camera_id'],'frame':o['frame_index'],'path':f"{o['camera_id']}/{o['frame_index']:04d}.png"} for o in obs]}
        dump(out/'score_plan.json',plan)
        print(json.dumps({k:plan[k] for k in ['status','scope','images','execution_enabled']}));return
    manifest,obs,paths,index=validate_index(args.render_index or rd/'render_index.json',mp,args.scope,rd)
    if args.check_index:
        print(json.dumps({'status':'render_index_verified_no_scoring','images':len(paths),'scope':args.scope}));return
    # Fail before creating results if cached AlexNet backbone is absent. No network download is permitted.
    backbone=args.torch_home/'hub/checkpoints/alexnet-owt-7be5be79.pth'
    if not backbone.is_file():raise FileNotFoundError('Cached LPIPS AlexNet backbone required: '+str(backbone))
    if args.device=='cuda' and not os.environ.get('CUDA_VISIBLE_DEVICES'):
        raise ValueError('Explicit admitted CUDA_VISIBLE_DEVICES required')
    validate_output_directory(out,{'manifest_sha256':sha(mp),'scope':args.scope,
                 'method':args.method,'render_dir':str(rd),'images':len(obs),
                 'metric_helper_sha256':sha(HELPER)})
    os.environ['TORCH_HOME']=str(args.torch_home)
    import cv2
    import lpips
    import numpy as np
    from PIL import Image
    import torch
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    torch.hub.set_dir(str(args.torch_home/'hub'))
    names={'read_rgb','ssim_map_rgb','masked_mean','metric_psnr','spatial_metrics'}
    nodes=[n for n in ast.parse(HELPER.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in names]
    if {n.name for n in nodes}!=names:raise ValueError('Frozen evaluator helper contract changed')
    context={'np':np,'cv2':cv2,'torch':torch,'Image':Image}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(HELPER),'exec'),context)
    metric=SimpleNamespace(**{name:context[name] for name in names})
    model=lpips.LPIPS(net='alex',version='0.1',spatial=False,pnet_rand=False,pretrained=True,verbose=False).to(args.device).eval().requires_grad_(False)
    weight_hash=hashlib.sha256()
    for k,v in sorted(model.state_dict().items()):weight_hash.update(k.encode());weight_hash.update(v.detach().cpu().numpy().tobytes())
    if weight_hash.hexdigest()!=LPIPS_STATE_SHA256:
        raise ValueError('LPIPS weights differ from the twelve reused baseline evaluations')
    began=time.monotonic();rows=[];out.mkdir(parents=True,exist_ok=True)
    with torch.inference_mode():
        for o in obs:
            pred=paths[(o['camera_id'],o['frame_index'])];gt_path=(mp.parent/o['hr_path']).resolve()
            if sha(gt_path)!=o['hr_sha256']:raise ValueError('HR reference changed: '+str(gt_path))
            with Image.open(pred) as im:
                if im.size!=tuple(manifest['resolutions']['hr']) or im.mode!='RGB':raise ValueError('Wrong RGB PNG shape or mode')
            arr,gt=metric.read_rgb(pred),metric.read_rgb(gt_path)
            if arr.shape!=gt.shape:raise ValueError('Render/GT dimensions differ')
            v=metric.spatial_metrics(arr,gt,np.zeros(arr.shape[:2],bool),model,args.device)['full']
            rows.append({'camera_id':o['camera_id'],'frame':o['frame_index'],'psnr':v['psnr'],'ssim':v['ssim'],
                         'lpips':v['lpips_alex'],'render_sha256':sha(pred),'reference_sha256':o['hr_sha256']})
    with (out/'per_view.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    mean={k:float(np.mean([r[k] for r in rows])) for k in ['psnr','ssim','lpips']}
    result={'schema':'dynamic_lr_multiview_HR_metrics_v1','status':'completed','method':args.method,'scene':manifest['scene'],
       'scope':args.scope,'images':len(rows),'frames':len({r['frame'] for r in rows}),'cameras':sorted({r['camera_id'] for r in rows}),
       args.scope:mean,'per_camera':{c:{k:float(np.mean([r[k] for r in rows if r['camera_id']==c])) for k in ['psnr','ssim','lpips']} for c in manifest['splits'][args.scope]},
       'manifest_sha256':sha(mp),'render_index_sha256':sha(args.render_index or rd/'render_index.json'),
       'evaluator_sha256':sha(Path(__file__)),'helper_source_sha256':sha(HELPER),'lpips_state_dict_sha256':weight_hash.hexdigest(),
       'lpips_backbone_sha256':sha(backbone),'input_representation':'8-bit rendered RGB PNG for every method',
       'psnr_aggregation':'arithmetic mean of per-image full-image PSNR','ssim_definition':'RGB 11x11 Gaussian sigma1.5 population covariance; exclude outer5px',
       'lpips_definition':'AlexNet v0.1 input[-1,1] full-image scalar','training_HR_used_for_model_updates':False,
       'render_camera_transform':'identity','host':socket.gethostname(),'device':args.device,'torch':torch.__version__,'numpy':np.__version__,
       'opencv':cv2.__version__,'seconds':time.monotonic()-began}
    dump(out/'summary.json',result);print(json.dumps(result))

if __name__=='__main__':main()
