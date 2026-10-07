"""One authorized native-CUDA X gradient fixture, with zero optimizer steps.

Run only on a GPU explicitly allocated by the root controller. The script
acquires the shared GPU lock and checks compute occupancy before importing or
initializing CUDA. Full HR camera intrinsics are preserved. Detached mip LOD and
stable interior support are held fixed in the finite-difference oracle.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback

from fp_common import ROOT, HERE, OUT, read, write, sha, bound, entry, setup, module


def check_gpu(uuid):
    selected=os.environ.get('CUDA_VISIBLE_DEVICES')
    if selected != uuid:
        raise ValueError('Bind CUDA_VISIBLE_DEVICES to the explicitly allocated UUID')
    result=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory',
                           '--format=csv,noheader'],capture_output=True,text=True,check=True)
    busy=[row for row in result.stdout.splitlines() if row.startswith(uuid+',') and int(row.split(',')[1])!=os.getpid()]
    if busy:
        raise RuntimeError('Allocated GPU is occupied; existing jobs are protected: '+'; '.join(busy))
    return result.stdout


def native_checks(protocol, counters):
    import torch
    import footprint as fp
    from support_cache import parent_camera, normalize_hr_moments
    from losses import charbonnier
    shared,policy=setup()
    from n3dv_data import load_manifest
    from motion_model import load_model
    from common import image_tensor
    p=read(protocol);manifest=load_manifest(bound(p['manifest']))
    observations=[next(o for o in manifest['observations'] if o['camera_id']==c and o['frame_index']==40)
                  for c in ('cam03','cam04')]
    assert all(o['split']=='train' for o in observations)
    lrpath=Path(manifest['_root'])/observations[1]['lr_path']
    assert sha(lrpath)==observations[1]['lr_sha256']
    reads=[]
    def guard(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        path=Path(os.fsdecode(args[0]))
        if path.suffix.lower() not in ('.png','.jpg','.jpeg','.npy','.npz'):return
        if path.resolve()!=lrpath.resolve():raise AssertionError(('Fixture image read outside one legal target LR',str(path)))
        reads.append(str(path))
    sys.addaudithook(guard)
    target_lr=image_tensor(lrpath)
    model=load_model(bound(p['parent']),manifest)
    assert model.checkpoint['metadata']['intervention_step']==6000
    cameras=[parent_camera(manifest,o,n) for n,o in enumerate(observations)]
    assert all((c.image_height,c.image_width)==(1008,1344) for c in cameras)
    gm=module('fp_native_check_moments',ROOT/'experiments/dynamic_sr_confidence_geometry_20261006/depth_prior.py')
    torch.set_num_threads(4);torch.cuda.reset_peak_memory_stats()
    started=time.monotonic()
    packets=[];moments=[]
    for camera in cameras:
        counters['rgb_forwards']+=1
        packet=policy.render_model(model,camera,trace=True)
        state=packet['effective_state']
        counters['moment_forwards']+=1
        moment=gm.render_moments(camera,state['xyz'],state['cov'],state['opacity'])
        packets.append(packet);moments.append(moment)
    source_rgb=packets[0]['render'];source_state=packets[0]['effective_state'];target_state=packets[1]['effective_state']
    source_normal=normalize_hr_moments(moments[0]['hr_moments'])
    target_normal=normalize_hr_moments(moments[1]['hr_moments'])
    projected=fp.project(cameras[0],cameras[1],target_normal['z'])
    fixed_footprint=fp.projection_footprint(projected['xy'],projected['valid'])
    source_valid=source_normal['valid_hr'].detach()
    frozen_hr=target_normal['valid_hr'].detach()
    base_edge=fp.edge_prediction(source_rgb,target_normal['z'],cameras[0],cameras[1],target_lr.shape[-2:],
                                frozen_mask_hr=frozen_hr,source_valid_hr=source_valid)
    # Select a deterministic valid block closest to the image centre using
    # geometry only. No HR/dev reference or quality score chooses the fixture.
    valid=base_edge['valid_lr'].detach()
    h,w=valid.shape
    centre=(h//2,w//2)
    choices=[]
    for row in range(h//4,3*h//4,8):
        for col in range(w//4,3*w//4,8):
            if bool(valid[row:row+8,col:col+8].all()):
                choices.append(((row-centre[0])**2+(col-centre[1])**2,row,col))
    if not choices:raise RuntimeError('No stable 8x8 interior supported X footprint in registered training fixture')
    _,row,col=min(choices)
    selection=torch.zeros_like(valid);selection[row:row+8,col:col+8]=True
    def objective(rgb,z):
        projection=fp.project(cameras[0],cameras[1],z)
        warped=fp.mip_sample(rgb,projection['xy'],fixed_footprint,source_valid)
        stable=fp.degradation_support(projection['valid'] & warped['valid'] & frozen_hr,target_lr.shape[-2:])
        if not bool(stable[selection].all()):raise RuntimeError('Finite difference crossed valid sampling support')
        prediction=fp.downsample(warped['rgb'],target_lr.shape[-2:])[0]
        return charbonnier(prediction[:,selection]-target_lr[:,selection]).mean()
    value=objective(source_rgb,target_normal['z'])
    counters['autograd_calls']+=1
    gradients=torch.autograd.grad(value,(source_rgb,source_state['xyz'],target_state['xyz'],
                                        target_state['cov'],target_state['opacity'],
                                        source_state['cov'],source_state['opacity']),retain_graph=True,allow_unused=True)
    counters['native_RGB_backwards']+=1
    counters['native_moment_backwards']+=1
    rgbgrad,srcgrad,tgtgrad,covgrad,opacitygrad,src_covgrad,src_opacitygrad=gradients
    assert rgbgrad is not None and srcgrad is not None and tgtgrad is not None
    assert all(bool(torch.isfinite(g).all()) and float(g.abs().max())>0 for g in (rgbgrad,srcgrad,tgtgrad))
    assert covgrad is None and opacitygrad is None
    assert src_covgrad is not None and src_opacitygrad is not None
    assert all(bool(torch.isfinite(g).all()) and float(g.abs().max())>0 for g in (src_covgrad,src_opacitygrad))
    # Native position directional FD for the single highest-derivative target
    # Gaussian. Same covariance/opacity and fixed LOD/mask remain unchanged.
    index=int(tgtgrad.norm(dim=1).argmax())
    direction=tgtgrad[index].detach()/tgtgrad[index].detach().norm()
    analytic=float((tgtgrad[index]*direction).sum())
    finite=[]
    with torch.no_grad():
        frozen_rgb=source_rgb.detach()
        for step in (1e-3,5e-4):
            values=[]
            for sign in (1.,-1.):
                xyz=target_state['xyz'].detach().clone();xyz[index]+=sign*step*direction
                counters['moment_forwards']+=1
                m=gm.render_moments(cameras[1],xyz,target_state['cov'],target_state['opacity'])
                z=normalize_hr_moments(m['hr_moments'])['z']
                values.append(float(objective(frozen_rgb,z)))
            finite.append((values[0]-values[1])/(2*step))
    # Float32 native rasterization is not bitwise smooth. Require both scales to
    # agree in sign and magnitude with a predeclared 15% relative envelope; a
    # failure is preserved and is not repaired by silently loosening tolerance.
    relative=[abs(fd-analytic)/max(abs(analytic),1e-8) for fd in finite]
    assert abs(analytic)>1e-7, ('No useful native target xyz signal',analytic)
    assert all(fd*analytic>0 and error<=.15 for fd,error in zip(finite,relative)), (analytic,finite,relative)
    # Source RGB FD avoids re-rendering and isolates the live area/grid/D path.
    rgbindex=tuple(int(v) for v in torch.unravel_index(rgbgrad.abs().argmax(),rgbgrad.shape))
    step=1e-3
    with torch.no_grad():
        hi=source_rgb.detach().clone();lo=source_rgb.detach().clone();hi[rgbindex]+=step;lo[rgbindex]-=step
        rgbfd=float((objective(hi,target_normal['z'].detach())-objective(lo,target_normal['z'].detach()))/(2*step))
    rgbanalytic=float(rgbgrad[rgbindex]);rgbrelative=abs(rgbfd-rgbanalytic)/max(abs(rgbanalytic),1e-8)
    assert rgbfd*rgbanalytic>0 and rgbrelative<=.03,(rgbanalytic,rgbfd,rgbrelative)
    torch.cuda.synchronize()
    return dict(status='passed_native_CUDA_X_gradient_fixture',loss=float(value.detach()),
        keys=[[o['camera_id'],o['frame_index']] for o in observations],parent=p['parent'],manifest=p['manifest'],
        target_LR=dict(path=str(lrpath.relative_to(ROOT)),sha256=sha(lrpath)),image_reads=reads,
        original_HR_read=False,dev_read=False,HR_camera_sizes=[[c.image_height,c.image_width] for c in cameras],
        xyz_count=int(target_state['xyz'].shape[0]),selection_LR=[row,col,8,8],
        source_RGB_gradient_norm=float(rgbgrad.norm()),source_xyz_gradient_norm=float(srcgrad.norm()),
        target_xyz_gradient_norm=float(tgtgrad.norm()),moment_cov_gradient=None,moment_opacity_gradient=None,
        source_RGB_cov_gradient_norm=float(src_covgrad.norm()),
        source_RGB_opacity_gradient_norm=float(src_opacitygrad.norm()),
        target_xyz_FD=dict(index=index,direction=direction.tolist(),analytic=analytic,steps=[1e-3,5e-4],
                          finite_differences=finite,relative_errors=relative,max_allowed_relative_error=.15),
        source_RGB_FD=dict(index=list(rgbindex),analytic=rgbanalytic,finite_difference=rgbfd,
                           relative_error=rgbrelative,max_allowed_relative_error=.03),
        seconds=time.monotonic()-started,peak_gb=torch.cuda.max_memory_allocated()/1e9,
        gpu=torch.cuda.get_device_name(),torch=str(torch.__version__),cuda=torch.version.cuda,
        scope='Gradient fixture only; no quality, exact geometry or visibility-truth claim. Detached LOD and stable support fixed in FD oracle.')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--gpu-uuid',required=True)
    parser.add_argument('--lock',type=Path,required=True)
    parser.add_argument('--protocol',type=Path,default=OUT/'protocol.json')
    parser.add_argument('--out',type=Path,default=OUT/'operator_checks'/'footprint_cuda')
    a=parser.parse_args();a.lock.parent.mkdir(parents=True,exist_ok=True);a.out.mkdir(parents=True,exist_ok=True)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ');target=a.out/f'{stamp}_{os.getpid()}.json'
    started=time.monotonic();counters=dict(rgb_forwards=0,moment_forwards=0,autograd_calls=0,
        native_RGB_backwards=0,native_moment_backwards=0,Adam_calls=0,parameter_updates=0)
    base=dict(started_utc=stamp,hostname=socket.gethostname(),gpu_uuid=a.gpu_uuid,lock=str(a.lock),
        source={n:entry(HERE/n) for n in ('footprint.py','footprint_cuda_checks.py')})
    with a.lock.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        try:
            base['startup_compute_inventory']=check_gpu(a.gpu_uuid)
            result=dict(base,**native_checks(a.protocol,counters),counters=counters,wall_seconds=time.monotonic()-started)
            write(target,result);write(a.out/'latest.json',dict(status=result['status'],receipt=entry(target)))
            print(json.dumps(dict(status=result['status'],receipt=str(target),counters=counters,seconds=result['wall_seconds'])),flush=True)
        except BaseException:
            result=dict(base,status='failed_or_not_dispatched_native_CUDA_X_gradient_fixture',
                        counters=counters,wall_seconds=time.monotonic()-started,error=traceback.format_exc())
            write(target,result);print(json.dumps(result,ensure_ascii=False),flush=True);raise


if __name__=='__main__':main()
