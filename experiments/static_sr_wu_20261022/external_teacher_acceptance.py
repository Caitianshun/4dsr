"""IE-SRGS external-teacher GPU acceptance on one train LR per image-size domain.

Uses the exact production load_external_models/external_tensors/_save_array
functions. No model parameters are updated. This small acceptance output is not
a formal prior bundle. Default is CPU planning; --run requires one admitted GPU.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import os
from pathlib import Path
import socket
import sys
import time
import traceback
import json
from common import ROOT,checked,identity,read,resolve,sha,training_manifest
from prepare_suite import SCENES,DEFAULT_PREPARED
from gpu_acceptance import save_receipt


def plan_external(prepared,scenes,frame,asset_index,upstream):
    if len(set(scenes))!=len(scenes) or set(scenes)-dict(SCENES).keys():
        raise ValueError('Use unique registered six-scene names')
    assets=read(asset_index)
    if assets.get('method')!='iesrgs' or assets.get('weights_ready') is not True:
        raise ValueError('IE-SRGS verified external-asset index is required')
    weights={name:assets['models'][name]['identity'] for name in ('swinir','depth_anything_v2_large')}
    for item in weights.values():checked(item)
    if Path(weights['swinir']['path']).name!='001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth':
        raise ValueError('Require the source SwinIR-M DF2K x4 checkpoint')
    if Path(weights['depth_anything_v2_large']['path']).name!='depth_anything_v2_vitl.pth':
        raise ValueError('Require Depth Anything V2-Large vitl, never vits substitution')
    samples=[]
    for short in scenes:
        config_path=Path(prepared)/short/'iesrgs.json';cfg=read(config_path)
        m,rows=training_manifest(checked(cfg['manifest']),verify_pixels=False)
        if cfg['method']!='iesrgs' or cfg['scene']!=dict(SCENES)[short] or m['scene']!=cfg['scene']:
            raise ValueError('Registered IE-SRGS scene identity differs')
        if frame not in m['frame_indices']:raise ValueError('Requested frame absent')
        selected=sorted((r for r in rows if r['frame_index']==frame),key=lambda r:r['camera_id'])
        if not selected:raise ValueError('No training LR at requested timestamp')
        row=selected[0];lr=Path(m['_root'])/row['lr_path']
        if row['camera_id']=='cam00' or row['camera_id'] not in m['splits']['train'] or row['split']!='train':
            raise ValueError('Teacher acceptance accepts training inputs only')
        if sha(lr)!=row['lr_sha256']:raise ValueError('LR bytes changed')
        samples.append(dict(short=short,scene=m['scene'],config=identity(config_path),manifest=cfg['manifest'],
            camera=row['camera_id'],frame=frame,time=row['time'],lr=identity(lr),resolutions=m['resolutions']))
    if not any(s['scene'].startswith('meetroom_') for s in samples) or not any(not s['scene'].startswith('meetroom_') for s in samples):
        raise ValueError('Cover both N3DV and MeetRoom original image sizes')
    producer=ROOT/'experiments/static_sr_wu_20261022/methods/iesrgs_priors.py'
    return dict(schema='iesrgs_external_teacher_GPU_acceptance_v1',status='pending_GPU_execution',GPU_executed=False,
        teacher_networks_executed=False,parameter_updates=0,formal_prior_bundle=False,formal_training_ready_not_implied=True,
        source=identity(__file__),production_producer=identity(producer),asset_index=identity(asset_index),weights=weights,
        upstream=str(resolve(upstream)),samples=samples,training_HR_read=False,heldout_pixels_read=False,
        operation='exact production SwinIR mirror-padding and float SR; DepthAnythingV2 vitl on truncated uint8 SR at input_size=518; uint8 normalized inverse depth /65536',
        validation_scope='Strict pretrained load, real CUDA inference, shape/range/finite checks, exact CPU float32 cache serialization; no SR quality claim')


def validate_arrays(sr,inverse,resolutions):
    import numpy as np
    sr=np.asarray(sr);inverse=np.asarray(inverse)
    w,h=resolutions['hr']
    if sr.shape!=(3,h,w) or inverse.shape!=(1,h,w):raise ValueError('Exact calibrated x4 teacher shape mismatch')
    if sr.dtype!=np.float32 or inverse.dtype!=np.float32:raise ValueError('Production cache must be float32')
    if not np.isfinite(sr).all() or not np.isfinite(inverse).all():raise ValueError('Nonfinite external teacher values')
    if sr.min()<0 or sr.max()>1:raise ValueError('SwinIR output outside source clamped range')
    if inverse.min()<0 or inverse.max()>255/65536:raise ValueError('Depth representation differs from source uint8 /65536')
    if not np.array_equal(inverse*65536,np.rint(inverse*65536)):raise ValueError('Depth cache is not source quantized inverse depth')
    if float(inverse.max()-inverse.min())<=0:raise ValueError('Constant inverse depth cannot supply Pearson guidance')
    return {name:dict(shape=list(value.shape),dtype=str(value.dtype),minimum=float(value.min()),maximum=float(value.max()),
        mean=float(value.mean()),standard_deviation=float(value.std())) for name,value in [('external_rgb',sr),('external_inverse_depth',inverse)]}


def run_external(plan,out,upstream,dependency_root=None):
    visible=os.environ.get('CUDA_VISIBLE_DEVICES','')
    if not visible or ',' in visible or visible=='-1':raise ValueError('Explicit one-GPU resource admission required')
    import numpy as np
    import torch
    from PIL import Image
    from methods.iesrgs_priors import load_external_models,external_tensors,_save_array
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise RuntimeError('Exactly one admitted GPU must be visible')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.manual_seed(20261007)
    begun=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    plan.update(status='running',GPU_executed=True,host=socket.gethostname(),device=visible,
        gpu_name=torch.cuda.get_device_name(),gpu_uuid=str(torch.cuda.get_device_properties(0).uuid),
        torch=torch.__version__,cuda=torch.version.cuda,started_utc=datetime.now(timezone.utc).isoformat(),results=[])
    save_receipt(out,plan)
    loading=time.perf_counter()
    swinir,depth=load_external_models(upstream,plan['weights']['swinir'],plan['weights']['depth_anything_v2_large'],
        device='cuda',dependency_root=dependency_root)
    torch.cuda.synchronize()
    for model in (swinir,depth):
        model.eval()
        for param in model.parameters():param.requires_grad_(False)
    plan['strict_loading']=dict(passed=True,source_function='load_external_models: strict=True for both state dictionaries',
        elapsed_seconds=time.perf_counter()-loading,
        swinir_parameters=sum(p.numel() for p in swinir.parameters()),depth_parameters=sum(p.numel() for p in depth.parameters()),
        swinir_state_entries=len(swinir.state_dict()),depth_state_entries=len(depth.state_dict()),
        missing_keys=[],unexpected_keys=[])
    peak=torch.cuda.max_memory_allocated()
    for item in plan['samples']:
        root=out/'NONFORMAL_single_image_fixtures'/item['short'];root.mkdir(parents=True,exist_ok=False)
        torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
        lr_path=checked(item['lr'])
        with Image.open(lr_path) as im:
            if list(im.size)!=item['resolutions']['lr']:raise ValueError('Observed LR resolution differs')
            image=np.array(im.convert('RGB'),copy=True)
        lr=torch.from_numpy(image).permute(2,0,1).float()/255
        with torch.no_grad():sr,inverse=external_tensors(lr.to('cuda'),swinir,depth)
        torch.cuda.synchronize()
        inference=time.perf_counter()-start
        cpu_arrays={'external_rgb':sr.detach().cpu().numpy(),'external_inverse_depth':inverse.detach().cpu().numpy()}
        stats=validate_arrays(cpu_arrays['external_rgb'],cpu_arrays['external_inverse_depth'],item['resolutions'])
        serialized={name:_save_array(root,name+'.npy',tensor) for name,tensor in [('external_rgb',sr),('external_inverse_depth',inverse)]}
        for name,entry in serialized.items():
            recovered=np.load(checked(entry),allow_pickle=False)
            if not np.array_equal(recovered,cpu_arrays[name]):raise RuntimeError('CPU float32 serialization changed teacher values')
        sample_peak=torch.cuda.max_memory_allocated();peak=max(peak,sample_peak)
        plan['results'].append(dict(short=item['short'],camera=item['camera'],frame=item['frame'],time=item['time'],
            lr=item['lr'],status='passed',statistics=stats,outputs=serialized,
            elapsed_inference_seconds=inference,elapsed_with_serialization_seconds=time.perf_counter()-start,
            peak_allocated_bytes=sample_peak,source_serialization='iesrgs_priors._save_array',
            cpu_serialization_exact_roundtrip=True,formal_prior_bundle=False))
        plan['teacher_networks_executed']=True;save_receipt(out,plan)
        del lr,sr,inverse,cpu_arrays;torch.cuda.empty_cache()
    plan.update(status='passed_external_teacher_GPU_inference_only',elapsed_seconds=time.perf_counter()-begun,
        peak_allocated_bytes=peak,finished_utc=datetime.now(timezone.utc).isoformat(),
        training_HR_read=False,heldout_pixels_read=False,parameter_updates=0)
    return plan


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared',type=Path,default=DEFAULT_PREPARED)
    p.add_argument('--scenes',nargs='+',choices=[s for s,_ in SCENES],default=['Cook07','Discussion07'])
    p.add_argument('--frame',type=int,default=149)
    p.add_argument('--asset-index',type=Path,default=ROOT/'output/static_sr_wu_20261022/priors/iesrgs_external_asset_index.json')
    p.add_argument('--upstream',type=Path,default=ROOT/'tmp/static_sr_wu_20261022/upstream/iesrgs')
    p.add_argument('--dependency-root',type=Path)
    p.add_argument('--out',required=True,type=Path)
    mode=p.add_mutually_exclusive_group();mode.add_argument('--run',action='store_true');mode.add_argument('--check-only',action='store_true')
    a=p.parse_args()
    plan=plan_external(resolve(a.prepared),a.scenes,a.frame,resolve(a.asset_index),resolve(a.upstream))
    out=resolve(a.out);out.mkdir(parents=True,exist_ok=False);save_receipt(out,plan)
    if a.run:
        try:run_external(plan,out,resolve(a.upstream),resolve(a.dependency_root) if a.dependency_root else None)
        except BaseException:
            plan.update(status='failed',error=traceback.format_exc());save_receipt(out,plan);raise
        save_receipt(out,plan)
    else:
        plan['torch_imported']='torch' in sys.modules;save_receipt(out,plan)
    print(json.dumps({'status':plan['status'],'receipt':identity(out/'receipt.json')}))


if __name__=='__main__':main()
