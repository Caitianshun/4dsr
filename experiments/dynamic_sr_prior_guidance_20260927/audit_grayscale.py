"""Train76 privileged operator diagnostic; never returns training targets."""
import argparse
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn.functional as F


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4);m=json.loads(a.manifest.read_text());rows=[];weights=torch.tensor([.2126,.7152,.0722])[:,None,None]
    def d(x):return F.interpolate(x[None],size=(252,336),mode='bicubic',align_corners=False,antialias=True)[0]
    for o in m['observations']:
        if o['split']!='train' or o['frame_index'] not in [0,40,80,118]:continue
        hrpath=a.manifest.parent/o['hr_path'];lrpath=a.manifest.parent/o['lr_path']
        assert hashlib.sha256(hrpath.read_bytes()).hexdigest()==o['hr_sha256'];assert hashlib.sha256(lrpath.read_bytes()).hexdigest()==o['lr_sha256']
        hr=torch.from_numpy(cv2.cvtColor(cv2.imread(str(hrpath)),cv2.COLOR_BGR2RGB)).permute(2,0,1).float()/255
        lr=torch.from_numpy(cv2.cvtColor(cv2.imread(str(lrpath)),cv2.COLOR_BGR2RGB)).permute(2,0,1).float()/255
        a0=(d(hr)*weights).sum(0);b0=d((hr*weights).sum(0,keepdim=True))[0]
        ac=(d(hr).clamp(0,1)*weights).sum(0);bc=b0.clamp(0,1);quant=(d(hr).clamp(0,1)*255).round()/255
        assert torch.equal(quant,lr)
        rows.append(dict(camera=o['camera_id'],frame=o['frame_index'],linear_commute_maxabs=float((a0-b0).abs().max()),clamped_commute_maxabs=float((ac-bc).abs().max()),clamped_commute_meanabs=float((ac-bc).abs().mean()),stored_gray_vs_float_meanabs=float(((lr*weights).sum(0)-ac).abs().mean()),quantized_rgb_exact=True))
    (a.out/'complete.json').write_text(json.dumps(dict(status='completed',rows=rows,aggregate={k:float(np.mean([r[k] for r in rows])) for k in ['linear_commute_maxabs','clamped_commute_maxabs','clamped_commute_meanabs','stored_gray_vs_float_meanabs']},linear_max=max(r['linear_commute_maxabs'] for r in rows),gray_definition='Fixed encoded-RGB linear projection Y=.2126R+.7152G+.0722B; no claim of linear radiance. Same bicubic AA D, clamp and PNG rounding audited separately.',gray_loss_trained=False),indent=2))


if __name__=='__main__':main()
