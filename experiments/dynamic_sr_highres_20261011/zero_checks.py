"""Calibrated train-view image screening before costly reconstruction.

sample: source video -> fixed projection/crop/rung -> exact x4 LR and held-out
evaluation HR samples. infer: frozen untiled SwinIR and bicubic, same metrics.
HR sample pixels are evaluation-only and never enter a legal training dataset.
"""
from __future__ import annotations
import argparse, importlib.util, json, os, sys, time
from pathlib import Path
from common import ROOT, OUT, read, write, sha, entry, mkdir, disk_guard

CHECKPOINT_SHA = '4e78e33f22c1aa8a773db0cf4a7381bae97c2362c717f155439ebc690cbd9215'
NETWORK_SHA = '1d650d2c1c4519d95db771863691c6b239a0822222f287f6a8342a9e0eccbc6e'


def sample(a):
    import prepare_colmap_dataset as prep
    import cv2, torch
    torch.set_num_threads(2); cv2.setNumThreads(1)
    cfg=read(a.dataset); cameras,mapping,diag,_=prep.metadata(cfg)
    root=mkdir(a.out); disk_guard(root,2)
    rows=[]
    for n,c in sorted(cameras.items()):
        if n in cfg['test_original']: continue
        maps,_=prep.remap_maps(c)
        for f in (15,150,285):
            other=f+30 if f+30<300 else f-30
            hr=prep.transform_frame(prep.read_frame(c['video'],a.start+f),c,a.rung,maps)
            companion=prep.transform_frame(prep.read_frame(c['video'],a.start+other),c,a.rung,maps)
            lr=prep.downsample(hr)
            paths={k:f'{mapping[n]}_{f:04d}_{k}.png' for k in ('hr','lr','companion')}
            digests={k:prep.png(root/paths[k],v) for k,v in (('hr',hr),('lr',lr),('companion',companion))}
            rows.append(dict(camera=mapping[n],original_camera=n,frame=f,source_frame=a.start+f,
                             companion_frame=other,paths=paths,sha256=digests))
    write(root/'samples.json',dict(dataset=cfg['id'],rung=a.rung,start=a.start,entries=rows,
        dataset_config=entry(a.dataset),source=entry(__file__),role='train-view diagnostic HR only; never legal supervision',
        test_views_used=False,teacher_cache=False))


def build_model(device):
    import torch
    network=ROOT/'third_party/swinir_frozen_20261010/network_swinir.py'
    checkpoint=network.parent/'001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth'
    if sha(network)!=NETWORK_SHA or sha(checkpoint)!=CHECKPOINT_SHA:
        raise ValueError('Frozen SwinIR source/weights changed')
    s=importlib.util.spec_from_file_location('highres_pinned_swinir',network)
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
    model=m.SwinIR(upscale=4,in_chans=3,img_size=64,window_size=8,img_range=1.,depths=[6]*6,
        embed_dim=180,num_heads=[6]*6,mlp_ratio=2,upsampler='pixelshuffle',resi_connection='1conv')
    weights=torch.load(checkpoint,map_location='cpu',weights_only=True)
    model.load_state_dict(weights.get('params',weights),strict=True)
    return model.requires_grad_(False).float().eval().to(device)


def infer_swinir(model,lr,device):
    import numpy as np, torch
    x=torch.from_numpy(np.ascontiguousarray(lr.transpose(2,0,1)))[None].float().to(device)/255.
    h,w=x.shape[-2:]; hp=(h//8+1)*8-h;wp=(w//8+1)*8-w
    x=torch.cat([x,torch.flip(x,[2])],2)[:,:,:h+hp,:]
    x=torch.cat([x,torch.flip(x,[3])],3)[:,:,:,:w+wp]
    with torch.inference_mode(): y=model(x)[...,:h*4,:w*4]
    return y[0].clamp(0,1).mul(255).round().byte().permute(1,2,0).cpu().numpy()


def band_metrics(error,mask=None):
    import numpy as np
    h,w=error.shape[:2];r=np.hypot(np.fft.fftfreq(h)[:,None],np.fft.fftfreq(w)[None,:])
    s=np.fft.fft2(error,axes=(0,1));result={}
    for name,m in (('low',r<.125),('mid',(r>=.125)&(r<.25)),('high',r>=.25)):
        image=np.fft.ifft2(s*m[:,:,None],axes=(0,1)).real
        result[name+'_mse']=float(np.mean(image**2) if mask is None else np.mean(image[mask]**2))
    return result


def infer(a):
    import cv2,numpy as np,torch,torch.nn.functional as F,lpips
    from PIL import Image
    torch.set_num_threads(2);cv2.setNumThreads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    samples=read(a.samples);root=a.samples.parent; device=a.device
    os.environ['TORCH_HOME']=str(ROOT/'third_party/4dgs_843d5ac_runtime/torch_hub')
    torch.hub.set_dir(os.environ['TORCH_HOME'])
    spec=importlib.util.spec_from_file_location('highres_image_metrics',ROOT/'experiments/dynamic_sr_20260918/evaluate.py')
    sys.path.insert(0,str(ROOT/'experiments/dynamic_sr_20260918'))
    ev=importlib.util.module_from_spec(spec);spec.loader.exec_module(ev)
    metric=lpips.LPIPS(net='alex',version='0.1',verbose=False).to(device).eval().requires_grad_(False)
    model=build_model(device);rows=[]
    def rgb(p): return np.asarray(Image.open(p).convert('RGB')).copy()
    for e in samples['entries']:
        for k,p in e['paths'].items():
            if sha(root/p)!=e['sha256'][k]:raise ValueError('Changed diagnostic sample')
        hr=rgb(root/e['paths']['hr']);lr=rgb(root/e['paths']['lr']);other=rgb(root/e['paths']['companion'])
        x=torch.from_numpy(lr).permute(2,0,1)[None].float()/255.
        b=F.interpolate(x,size=hr.shape[:2],mode='bicubic',align_corners=False).clamp(0,1)[0].permute(1,2,0).numpy()
        if device.startswith('cuda'):torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
        begin=time.monotonic();t=infer_swinir(model,lr,device).astype(np.float32)/255.
        if device.startswith('cuda'):torch.cuda.synchronize()
        cost=dict(seconds=time.monotonic()-begin,peak_allocated_bytes=torch.cuda.max_memory_allocated() if device.startswith('cuda') else None)
        truth=hr.astype(np.float32)/255.;dynamic=(np.max(np.abs(hr.astype(np.int16)-other.astype(np.int16)),axis=2)>8)
        scores={}
        for name,pred in (('bicubic',b),('SwinIR',t)):
            scores[name]=ev.spatial_metrics(pred,truth,dynamic,metric,device)
            for region,mask in (('full',None),('dynamic',dynamic),('static',~dynamic)):
                scores[name][region]['bands']=band_metrics(pred-truth,mask) if mask is None or mask.any() else None
        centered=truth-truth.mean(axis=(0,1),keepdims=True);energy=band_metrics(centered)
        total_energy=sum(energy.values())
        row=dict(camera=e['camera'],frame=e['frame'],metrics=scores,cost=cost,
            high_frequency_energy_share=(energy['mid_mse']+energy['high_mse'])/total_energy if total_energy else None)
        rows.append(row);print(json.dumps(dict(stage='Z5_image',camera=e['camera'],frame=e['frame'],seconds=cost['seconds'])),flush=True)
    mean=lambda f:float(np.mean([f(x) for x in rows]))
    bpsnr=mean(lambda x:x['metrics']['bicubic']['full']['psnr'])
    band_gain={k:mean(lambda x:10*np.log10(x['metrics']['bicubic']['full']['bands'][k+'_mse']/max(x['metrics']['SwinIR']['full']['bands'][k+'_mse'],1e-20))) for k in ('mid','high')}
    result=dict(status='awaiting_review',dataset=samples['dataset'],rung=samples['rung'],rows=rows,
        samples_identity=dict(path=str(a.samples),sha256=sha(a.samples)),source=entry(__file__),device=device,
        gpu=torch.cuda.get_device_name() if device.startswith('cuda') else None,
        frozen=dict(network_SHA256=NETWORK_SHA,checkpoint_SHA256=CHECKPOINT_SHA,tile=0,precision='float32',padding='official_mirror_concat_next_window'),
        aggregate=dict(bicubic_PSNR=bpsnr,SwinIR_PSNR=mean(lambda x:x['metrics']['SwinIR']['full']['psnr']),band_PSNR_gain=band_gain),
        Z5_pass=bool(bpsnr<=34. and min(band_gain.values())>=1.),
        gate_definition='bicubic full PSNR <=34 dB AND both radial mid and high band gains >=1 dB, train cameras frames15/150/285',
        limitations=['Radial FFT bands are convention, not exact square LR passband.','High frequencies may contain noise, compression and resampling.','Image diagnostics do not establish novel-view reconstruction headroom.'])
    write(a.out,result)


def main():
    p=argparse.ArgumentParser();s=p.add_subparsers(dest='command',required=True)
    a=s.add_parser('sample');a.add_argument('--dataset',type=Path,required=True);a.add_argument('--rung',choices=['half','full'],required=True);a.add_argument('--start',type=int,required=True);a.add_argument('--out',type=Path,required=True)
    b=s.add_parser('infer');b.add_argument('--samples',type=Path,required=True);b.add_argument('--device',default='cuda');b.add_argument('--out',type=Path,required=True)
    a=p.parse_args(); (sample if a.command=='sample' else infer)(a)
if __name__=='__main__':main()
