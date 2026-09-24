"""Evaluate HR detail transport only at LR-selected strict valid pixels.

Supplement to the original fixed-patch report; no flow, threshold or selection
is refit using HR, and original results are not overwritten.
"""
from pathlib import Path
import json
import time
import cv2
import numpy as np
import torch
import observability_real as main


def run():
    start = time.monotonic()
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    root = main.OUT
    dest = root / 'strict_pixel_recheck'
    dest.mkdir(exist_ok=False)
    parent = json.loads((root/'metrics.json').read_text())
    cells = []
    for obs in parent['observations']:
        scene, cam, frame = obs['scene'], obs['camera'], obs['anchor']
        stem = f'{scene}_{cam}_{frame:04d}'
        npz = np.load(root/f'{stem}_lr_correspondence.npz')
        data = main.ROOT/'data/dynamic_sr'/main.SCENES[scene]
        hrs = {}
        for ff in [frame]+[frame+o for o in main.OFFSETS]:
            path = data/f'hr/{cam}/{ff:04d}.png'
            assert main.sha(path) == parent['input_sha256'][str(path)]
            from PIL import Image
            hrs[ff] = np.asarray(Image.open(path).convert('RGB'), dtype=np.float32)/255.
        hh, ww = hrs[frame].shape[:2]
        h, w = npz['flow'].shape[1:3]
        highs = {ff: im-main.resize(main.resize(im,(h,w),True),(hh,ww)) for ff,im in hrs.items()}
        warped = [main.warp(highs[frame+off], cv2.resize(npz['flow'][i],(ww,hh))*4)
                  for i,off in enumerate(main.OFFSETS)]
        valid_hr = [cv2.resize(npz['valid'][i].astype(np.uint8),(ww,hh),interpolation=cv2.INTER_NEAREST)>0
                    for i in range(4)]
        for c in parent['cells']:
            if (c['scene'], c['camera'], c['anchor']) != (scene,cam,frame):
                continue
            x,y=c['x_lr']*4,c['y_lr']*4
            sl=np.s_[y-24:y+24,x-24:x+24]
            out={k:c[k] for k in ['scene','camera','anchor','x_lr','y_lr','n_observations','moving_proxy','phase_rich']}
            evals=[]
            for i in c['accepted_neighbors']:
                m=valid_hr[i][sl]
                true=highs[frame][sl][m]
                z=float(np.mean(true**2))
                a=float(np.mean((warped[i][sl][m]-true)**2))
                u=float(np.mean((highs[frame+main.OFFSETS[i]][sl][m]-true)**2))
                evals.append(dict(neighbor=frame+main.OFFSETS[i],valid_fraction=float(m.mean()),
                    zero_detail_mse=z, aligned_to_zero_ratio=a/max(z,1e-15),
                    unaligned_to_zero_ratio=u/max(z,1e-15),
                    better_than_zero=a<z,better_than_unaligned=a<u))
            out['evaluations']=evals
            cells.append(out)
    summary={}
    for scene in main.SCENES:
        cc=[c for c in cells if c['scene']==scene]
        summary[scene]={}
        for name, selected in [('all',cc),('moving_phase_rich',[c for c in cc if c['moving_proxy'] and c['phase_rich']])]:
            ee=[e for c in selected for e in c['evaluations']]
            summary[scene][name]=dict(cells=len(selected),evaluations=len(ee),
                aligned_ratio=main.scalar_stats([e['aligned_to_zero_ratio'] for e in ee]),
                unaligned_ratio=main.scalar_stats([e['unaligned_to_zero_ratio'] for e in ee]),
                better_than_zero_fraction=float(np.mean([e['better_than_zero'] for e in ee])),
                better_than_unaligned_fraction=float(np.mean([e['better_than_unaligned'] for e in ee])))
    result=dict(parent_metrics_sha256=main.sha(root/'metrics.json'),
        source_sha256=main.sha(__file__), shared_helper_sha256=main.sha(main.__file__),
        elapsed_seconds=time.monotonic()-start,summary=summary,cells=cells)
    (dest/'source.py').write_text(Path(__file__).read_text())
    (dest/'metrics.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    run()
