"""Missing native-file depth and one fixed RAFT-family estimator, no training."""
import os, sys, time, inspect
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from image_prior_common import *

def run_cache():
    m,obs,keys=setup();torch.set_num_threads(4);start=time.time()
    assert torch.cuda.is_available()
    source=ROOT/'tmp/prior_guidance_20260927/Depth-Anything-V2';sys.path.insert(0,str(source))
    from depth_anything_v2.dpt import DepthAnythingV2
    weight=ROOT/'weight/depth_anything_v2/depth_anything_v2_vits.pth';assert sha(weight)=='715fade13be8f229f8a70cc02066f656f2423a59effd0579197bbf57860e1378'
    common=dict(time_convention=m['time_convention'],pixel_coordinate_convention=m['pixel_coordinate_convention'],manifest_sha256=sha(MANIFEST),gpu=torch.cuda.get_device_name(),cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),torch=torch.__version__,torchvision=torchvision.__version__)
    directory=OUT/'depth_A';directory.mkdir(exist_ok=True);rows=[]
    with torch.inference_mode():
        net=DepthAnythingV2(encoder='vits',features=64,out_channels=[48,96,192,384]).cuda().eval();net.load_state_dict(torch.load(weight,map_location='cpu',weights_only=True))
        for c,f in keys:
            dest=directory/f'{c}_{f:04d}.npz';o=obs[c,f]
            if not dest.exists():
                im=image(obs,c,f);tensor,_=net.image2tensor(cv2.cvtColor(im,cv2.COLOR_RGB2BGR),518);grid=list(tensor.shape[-2:]);pred=F.interpolate(net(tensor)[:,None],size=im.shape[:2],mode='bilinear',align_corners=True)[0,0].cpu().numpy().astype(np.float32);np.savez_compressed(dest,depth=pred,network_grid=np.array(grid))
            z=np.load(dest);rows.append(dict(camera=c,frame=f,path=str(dest.relative_to(ROOT)),sha256=sha(dest),input_sha256=o['lr_sha256'],source_file_grid=[252,336],network_grid=z['network_grid'].tolist(),information='legal_LR_only'))
        write(directory/'complete.json',dict(status='completed',rows=rows,**common,weight_sha256=sha(weight),source_sha256=sha(source/'depth_anything_v2/dpt.py'),preprocess='Native LR file fed to official image2tensor(input_size=518); no pre-upsample. Network lower-bound resize produces 518x686, the same tensor grid as B/C. A vs B measures resampling route, NOT different network inference resolution.',output='raw nonnegative relative inverse-depth-like score on LR grid; no per-image normalization',seconds=time.time()-start));del net;torch.cuda.empty_cache()
        print('native_depth_complete',len(rows),time.time()-start,flush=True)
        from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
        weights=Raft_Large_Weights.C_T_SKHT_V2
        weight_path=Path(torch.hub.get_dir())/'checkpoints/raft_large_C_T_SKHT_V2-ff5fadd5.pth';assert weight_path.exists()
        net=raft_large(weights=weights,progress=False).cuda().eval();flowrows=[];start_flow=time.time()
        for c,f,g in PAIRS:
            for branch in ['A','B','C']:
                directory=OUT/('flow_legal' if branch!='C' else 'flow_diagnostic_only');directory.mkdir(exist_ok=True)
                dest=directory/f'{branch}_{c}_{f:04d}_{g:04d}.npz';ti=time.time()
                sources=[]
                for frame in [f,g]:
                    kind='hr' if branch=='C' else 'lr';sources.append(dict(camera=c,frame=frame,path=str(MANIFEST.parent/obs[c,frame][kind+'_path']),sha256=obs[c,frame][kind+'_sha256']))
                if not dest.exists():
                    ims=[image(obs,c,t,'hr' if branch=='C' else 'lr') for t in [f,g]]
                    if branch=='B':ims=[cv2.resize(im,(1344,1008),interpolation=cv2.INTER_CUBIC) for im in ims]
                    tensors=[torch.from_numpy(im.copy()).permute(2,0,1).unsqueeze(0).cuda().float()/127.5-1 for im in ims];h,w=ims[0].shape[:2];ph=(-h)%8;pw=(-w)%8;tensors=[F.pad(t,(0,pw,0,ph),mode='replicate') for t in tensors]
                    out=[]
                    for first,second in [tensors,tensors[::-1]]:
                        pred=net(first,second,num_flow_updates=12)[-1][0,:,:h,:w].permute(1,2,0).cpu().numpy().astype(np.float32);out.append(pred)
                    np.savez_compressed(dest,forward=out[0],backward=out[1],original_grid=np.array([h,w]),network_grid=np.array([h+ph,w+pw]),padding=np.array([0,pw,0,ph]));del tensors;torch.cuda.empty_cache()
                z=np.load(dest);flowrows.append(dict(camera=c,first=f,second=g,frame_gap=g-f,time_gap=obs[c,g]['time']-obs[c,f]['time'],branch=branch,path=str(dest.relative_to(ROOT)),sha256=sha(dest),sources=sources,network_grid=z['network_grid'].tolist(),padding=z['padding'].tolist(),seconds=time.time()-ti))
            print('RAFT_pair_complete',c,f,g,'elapsed',time.time()-start_flow,flush=True)
        write(OUT/'flow_index.json',dict(status='completed',rows=flowrows,**common,estimator='torchvision RAFT-large C_T_SKHT_V2 (existing cached RAFT family)',weight_path=str(weight_path),weight_sha256=sha(weight_path),weight_url=weights.url,implementation_path=inspect.getfile(raft_large),implementation_sha256=sha(inspect.getfile(raft_large)),num_flow_updates=12,preprocess='RGB uint8 -> FP32 [-1,1]; A native 252x336; B cv2 cubic to1008x1344; C true HR1008x1344. Replicate pad right/bottom to multiple8, unpad exactly. No resize for A/C.',flow_coordinates='Pixel-center index coordinates; cv2 resize uses half-pixel centers. Flow resized then dx and dy multiplied by new_width/old_width and new_height/old_height. Forward and backward kept on original grids.',seconds=time.time()-start_flow,peak_gpu_gb=torch.cuda.max_memory_allocated()/2**30,source_reference='https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.optical_flow.raft_large.html',used_for_training=False));del net;torch.cuda.empty_cache()
    write(OUT/'cache_complete.json',dict(status='completed',**common,total_seconds=time.time()-start,depth_entries=len(keys),flow_directed_predictions=len(flowrows)*2,source_sha256=sha(__file__)))
if __name__=='__main__':run_cache()
