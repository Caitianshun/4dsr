"""Fixed 19 cameras x four frames LR reprojection; no HR or teacher reads."""
import argparse
import os
import time
import numpy as np
from shared import *
_ev=evaluator();render_camera,lr_reprojection_metrics,legacy=_ev.render_camera,_ev.lr_reprojection_metrics,_ev.legacy

def main():
    p=argparse.ArgumentParser()
    for name in ['manifest','checkpoint','out']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4);start=time.monotonic()
    m=load_manifest(a.manifest);model=load_model(a.checkpoint,m);model.g._deformation.eval()
    rows=[];observations=sorted((o for o in m['observations'] if o['split']=='train' and o['frame_index'] in [0,40,80,118]),key=lambda o:(o['camera_id'],o['frame_index']))
    assert len(observations)==76
    with torch.inference_mode():
        for i,o in enumerate(observations):
            cam=render_camera(m,o,i);raw=render_model(model,cam)['render']
            path=Path(m['_root'])/o['lr_path'];assert sha256(path)==o['lr_sha256']
            lr=legacy.read_rgb(path);metric=lr_reprojection_metrics(raw,lr,np.zeros(raw.shape[-2:],bool))['full']
            rows.append(dict(camera_id=o['camera_id'],frame_index=o['frame_index'],lr_sha256=o['lr_sha256'],**metric))
    def avg(rr):return {k:float(np.mean([r[k] for r in rr])) for k in ['mse','psnr','l1']}
    write_json(a.out/'metrics.json',dict(status='completed',checkpoint=str(a.checkpoint),checkpoint_sha256=sha256(a.checkpoint),manifest_sha256=sha256(a.manifest),
        rows=rows,aggregate=avg(rows),by_camera={c:avg([r for r in rows if r['camera_id']==c]) for c in m['splits']['train']},
        train16_subset=avg([r for r in rows if r['camera_id'] in ['cam02','cam06','cam12','cam18']]),
        protocol='19 train cameras x frames0/40/80/118; raw float HR render -> unchanged D -> actual LR; mean per-frame PSNR; not all1140',
        gpu=torch.cuda.get_device_name(),torch=str(torch.__version__),seconds=time.monotonic()-start,parameter_updates=0))
    print(json.dumps(dict(status='completed',aggregate=avg(rows))))

if __name__=='__main__':main()
