"""Evaluation-only original-teacher/HR and own-target fit at fixed train views."""
import argparse
import json
from pathlib import Path
import sys
import torch
import lpips
import numpy as np

from train_sharing_control import setup, OLD
sys.path.insert(0,str(OLD))
from common import render_image
from controlled_fit import fit_diagnostic,write_json
from evaluate import image_array,spatial_metrics

p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--step',type=int,required=True)
cli=p.parse_args();cfg=json.loads((cli.run/'config.json').read_text())
args=argparse.Namespace(**{k:cfg[k] for k in ['manifest','checkpoint','cache_manifest','prior_cameras']})
args.checkpoint=str(cli.run/f'checkpoint_{cli.step}.pt');torch.set_num_threads(4)
m,g,h,o,ck,records,cams,ids,target=setup(args)
g._deformation.eval();metric=lpips.LPIPS(net='alex').cuda().eval().requires_grad_(False)
with torch.inference_mode():
    original=fit_diagnostic(g,m,set(cfg['prior_cameras'].split(',')),cli.run,cli.step,metric)
    rows=[]
    for i in ids:
        if records[i]['frame_index'] not in [0,40,80,118]:continue
        pred=image_array(render_image(g,cams[i])['render'])
        own=image_array(target(i,cfg['branch']))
        mask=np.zeros(pred.shape[:2],bool)
        rows.append(dict(camera_id=records[i]['camera_id'],frame_index=records[i]['frame_index'],
                         metrics=spatial_metrics(pred,own,mask,metric)['full']))
    avg={k:float(np.mean([r['metrics'][k] for r in rows])) for k in ['psnr','ssim','lpips_alex','mse']}
    write_json(cli.run/f'own_target_fit_{cli.step}.json',dict(rows=rows,aggregate=avg,
        target_clipped_for_metrics=True,training_uses_raw_float32=True,parameter_updates=0))
