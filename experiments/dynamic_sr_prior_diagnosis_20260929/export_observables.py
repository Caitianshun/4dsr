"""Export 196 raw floating RGB observations for each of five registered models.

PNG is never a renderer cache. rgb_eval := clip(rgb_raw,0,1), float32 CHW.
"""
import argparse,gc,time,traceback
from structure_common import *
import numpy as np
import torch

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--models',nargs='+',default=list(registry()));a=ap.parse_args()
 motion,ev=setup();from n3dv_data import load_manifest
 from common import UPSTREAM
 torch.set_num_threads(4);p=read(OLD/'protocol.json');mp=local(p['manifest']['path']);assert sha(mp)==p['manifest']['sha256'];manifest=load_manifest(mp)
 obs=sorted([o for o in manifest['observations'] if o['camera_id'] in ['cam00','cam01'] or (o['split']=='train' and o['frame_index'] in p['evaluation']['train_frames'])],key=lambda o:(o['camera_id'],o['frame_index']))
 assert len(obs)==196 and sum(o['split']=='train' for o in obs)==76
 sources={str(f):sha(f) for f in [Path(__file__),ROOT/'experiments/dynamic_sr_prior_diagnosis_20260929/structure_common.py',ROOT/'experiments/dynamic_sr_motion_bound_20260923/motion_model.py',ROOT/'experiments/dynamic_sr_20260918/common.py',Path(UPSTREAM)/'gaussian_renderer/__init__.py',Path(UPSTREAM)/'submodules/depth-diff-gaussian-rasterization/cuda_rasterizer/forward.cu']}
 import diff_gaussian_rasterization._C as ext
 renderer=dict(sources=sources,extension=dict(path=ext.__file__,sha256=sha(ext.__file__)),torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),visible_cuda=os.environ.get('CUDA_VISIBLE_DEVICES'))
 allrows=[];t0=time.time()
 for name in a.models:
  ident=registry()[name];cp=local(ident['checkpoint']['path']);assert sha(cp)==ident['checkpoint']['sha256'];model=motion.load_model(cp,manifest);model.g._deformation.eval();before=[x._version for x in model.g._deformation.parameters()];rows=[];begin=time.time();renders=0;out=OUT/'observables'/name;out.mkdir(parents=True,exist_ok=True)
  pts=len(model.g._xyz)+(len(model.children.xyz()) if model.children else 0);assert pts==ident['expected_points'];assert model.checkpoint['metadata']['manifest_sha']==p['manifest']['sha256']
  for i,o in enumerate(obs):
   stem=f"{o['camera_id']}_{o['frame_index']:04d}";npz=out/(stem+'.npz');js=out/(stem+'.json')
   if js.exists() and npz.exists():
    r=read(js);assert r['checkpoint']['sha256']==ident['checkpoint']['sha256'] and r['npz_sha256']==sha(npz);rows.append(r);continue
   with torch.no_grad():
    cam=ev.render_camera(manifest,o,i);raw=motion.render_model(model,cam)['render'];assert torch.isfinite(raw).all();raw=raw.detach().cpu().numpy().astype(np.float32,copy=False)
   save_npz(npz,rgb_raw=raw);renders+=1
   for role in ['hr','lr']:assert sha(Path(manifest['_root'])/o[role+'_path'])==o[role+'_sha256']
   cal=manifest['cameras'][o['camera_id']]
   r=dict(schema=1,model=name,scene=manifest['scene'],camera=o['camera_id'],frame=o['frame_index'],timestamp=o['time'],timestamp_definition=manifest['time_convention'],split='train' if o['split']=='train' else 'development',original_manifest_split=o['split'],asset_stream='diagnostic_only',checkpoint=ident['checkpoint'],training_supervision=ident['supervision'],total_updates=ident['total_updates'],points=pts,rgb_dtype='float32',rgb_layout='CHW',rgb_range=dict(min=float(raw.min()),max=float(raw.max()),below_zero_fraction=float((raw<0).mean()),above_one_fraction=float((raw>1).mean())),evaluation_definition='np.clip(rgb_raw,0,1), no quantization',clamp_position='evaluation only; raw tensor unchanged',resolution_wh=manifest['resolutions']['hr'],camera_calibration={k:cal[k] for k in ['K_hr','K_lr','w2c','c2w']},camera_pixel_centers=manifest['pixel_coordinate_convention'],downsample_kernel=manifest['degradation'],hr_path=str(Path(manifest['_root'])/o['hr_path']),hr_sha256=o['hr_sha256'],lr_path=str(Path(manifest['_root'])/o['lr_path']),lr_sha256=o['lr_sha256'],npz_path=str(npz),npz_sha256=sha(npz),renderer=renderer)
   write(js,r);rows.append(r)
   if (i+1)%40==0:print(name,i+1,round(time.time()-begin,1),flush=True)
  assert before==[x._version for x in model.g._deformation.parameters()]
  write(out/'index.json',dict(schema=1,rows=rows,checkpoint=ident['checkpoint'],manifest=p['manifest'],renderer=renderer))
  write(out/'complete.json',dict(status='completed',observations=len(rows),rgb_forwards_new=renders,cached_observations=len(rows)-renders,seconds=time.time()-begin,parameter_updates=0,index_sha256=sha(out/'index.json')));allrows.extend(rows)
  del model;gc.collect();torch.cuda.empty_cache()
 write(OUT/'observables/complete.json',dict(status='completed',models=a.models,observations=len(allrows),seconds=time.time()-t0,parameter_updates=0,interface=dict(path='<model>/<camera>_<frame:04d>.npz',raw='rgb_raw float32 CHW',evaluation='np.clip(rgb_raw,0,1)',ground_truth='row.hr_path RGB PNG /255.0 float32',identity='same stem .json; index.json rows')))
if __name__=='__main__':
 try:main()
 except BaseException:
  write(OUT/'observables/failed.json',dict(error=traceback.format_exc()));raise
