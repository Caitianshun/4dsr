"""Native LR/HR checkpoints, all scored at HR with the unchanged metric helpers.

Reference evaluation is independent of prior preparation and does not train.
The native-LR bicubic output never consumes held-out LR pixels.
"""
import argparse
import gc
import importlib.util
import time
import traceback
from dv_common import *
import numpy as np
import torch
import torch.nn.functional as F

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=OUT/'references');a=ap.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    p=read(OLD/'protocol.json')
    assert os.environ['CUDA_VISIBLE_DEVICES']==p['evaluation']['physical_gpu']
    shared=setup();ev=shared.evaluator()
    from common import load_checkpoint,render_image,resized_camera
    from n3dv_data import load_manifest
    from runtime_identity import identity
    assert identity()==p['runtime']['evaluation']
    manifest=load_manifest(bound(p['manifest']))
    train_cams=sorted({x['camera_id'] for x in manifest['observations'] if x['split']=='train'})
    assert train_cams==p['evaluation']['train_cameras'] and not {'cam00','cam01'}&set(train_cams)
    obs=sorted([x for x in manifest['observations'] if x['camera_id'] in ['cam00','cam01']],key=lambda x:(x['camera_id'],x['frame_index']))
    assert [(x['camera_id'],x['frame_index']) for x in obs]==[(c,f) for c in ['cam00','cam01'] for f in range(0,120,2)]
    torch.set_num_threads(4)
    import lpips
    metric=lpips.LPIPS(net='alex',spatial=False).cuda().eval().requires_grad_(False)
    choices=[('LR-direct-HRrender','cook_warmup_s20260918',6000,'native_lr'),
             ('HR-direct-6k','cook_spinach_pilot_v1_hr_reference',6000,'hr_reference'),
             ('HR-direct-4k','cook_spinach_pilot_v1_hr_reference',4000,'hr_reference')]
    registry={};allrows=[];start=time.monotonic()
    for label,run,step,supervision in choices:
        path=ROOT/'output/dynamic_sr_20260918'/run/f'checkpoint_{step}.pt'
        if not path.exists():
            registry[label]=dict(status='unavailable',missing_path=str(path),reason='checkpoint missing; not substituted')
            continue
        digest=sha(path);resultpath=a.out/(label+'.json')
        if resultpath.exists():
            result=read(resultpath);assert result['checkpoint']['sha256']==digest
            registry.update(result['registry']);allrows.extend(result['rows']);continue
        tick=time.monotonic();g,_,_,ck=load_checkpoint(path);g._deformation.eval()
        cfg=read(path.parent/'config.json');meta=ck['metadata']
        assert meta['manifest_sha']==p['manifest']['sha256'] and meta['stage']=='fine' and meta['step']==step
        assert meta['observation']==supervision and cfg['observation']==supervision
        assert cfg['task']=='warmup' and cfg['checkpoint'] is None and cfg['coarse_steps']==1000
        points=len(g.get_xyz);rows=[];renders=0
        labels=[label,'LR-direct-Bicubic'] if supervision=='native_lr' else [label]
        entries={l:dict(status='evaluated',checkpoint=entry(path),config=entry(path.parent/'config.json'),metadata=meta,
            supervision=supervision,coarse_updates=1000,fine_parent_updates=step,sr_prefix_updates=0,suffix_updates=0,
            total_updates=1000+step,training_rgb_forwards=1000+step,points=points,heldout_cameras_verified=['cam00','cam01'],
            training_cameras=train_cams,equal_training_budget=False,
            output_definition='native float LR render clamp then torch bicubic align_corners=False antialias=True then clamp' if l.endswith('Bicubic') else 'native model rendered directly on HR canvas',
            intermediate_uint8=False) for l in labels}
        with torch.inference_mode():
            for i,o in enumerate(obs):
                cam=ev.render_camera(manifest,o,i)
                predictions={label:render_image(g,cam)['render']};renders+=1
                if supervision=='native_lr':
                    w,h=manifest['resolutions']['lr'];lr=render_image(g,resized_camera(cam,h,w))['render'];renders+=1
                    predictions['LR-direct-Bicubic']=F.interpolate(lr.clamp(0,1)[None].float(),size=(1008,1344),mode='bicubic',align_corners=False,antialias=True)[0].clamp(0,1)
                hrpath=Path(manifest['_root'])/o['hr_path'];assert sha(hrpath)==o['hr_sha256']
                gt=ev.legacy.read_rgb(hrpath)
                for name,raw in predictions.items():
                    assert torch.isfinite(raw).all()
                    pred=ev.legacy.image_array(raw)
                    q=ev.legacy.spatial_metrics(pred,gt,np.zeros(gt.shape[:2],bool),metric)['full']
                    rows.append(dict(endpoint=name,camera=o['camera_id'],frame=o['frame_index'],psnr=q['psnr'],ssim=q['ssim'],lpips=q['lpips_alex'],hr_sha256=o['hr_sha256']))
                    if o['frame_index'] in [40,80]:ev.legacy.write_rgb(a.out/'predictions'/name/o['camera_id']/f"{o['frame_index']:04d}.png",pred)
                if (i+1)%20==0:print(json.dumps(dict(reference=label,observations=i+1,seconds=time.monotonic()-tick)),flush=True)
        result=dict(checkpoint=entry(path),registry=entries,rows=rows,rgb_forwards=renders,seconds=time.monotonic()-tick,
            runtime=identity(),metric_source=entry(ROOT/'experiments/dynamic_sr_20260918/evaluate.py'),script=entry(Path(__file__)),manifest=p['manifest'])
        write(resultpath,result);registry.update(entries);allrows.extend(rows)
        del g,ck;gc.collect();torch.cuda.empty_cache()
    write(OUT/'baseline_registry.json',dict(references=registry,protocol='HR output vs identical HR references; 60 frames per camera; native loaders',
        manifest=p['manifest'],training_budget_caution='Evaluation matched only; optimization budget and capacity differ',metric_source=entry(ROOT/'experiments/dynamic_sr_20260918/evaluate.py')))
    write(a.out/'complete.json',dict(status='completed' if len(registry)==4 and all(v['status']=='evaluated' for v in registry.values()) else 'incomplete_missing_assets',
        registry=entry(OUT/'baseline_registry.json'),rows=allrows,seconds=time.monotonic()-start,
        rgb_forwards=sum(read(f)['rgb_forwards'] for f in a.out.glob('*direct*.json'))))

if __name__=='__main__':
    try:main()
    except BaseException:
        write(OUT/'references/failed.json',dict(error=traceback.format_exc()));raise
