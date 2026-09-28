"""Official frozen BasicVSR++ REDS BI x4; registered train-LR windows only.

No model algorithm is substituted. Generator state loading strips exactly
the checkpoint's generator. prefix; mirror state is reset before every call.
"""
import argparse
import subprocess
import platform
import time
import traceback
from collections import Counter
from dv_common import *

MODES=('repeat7','video7')
QUANTIZATION='clamp(0,1) * 255 -> round -> uint8 RGB PNG; loader divides by 255'

def window_indices(j,n=60):return [min(n-1,max(0,j+k)) for k in range(-3,4)]

def source_windows(manifest,mode):
    assert mode in MODES
    obs=[x for x in manifest['observations'] if x['split']=='train']
    cams=sorted({x['camera_id'] for x in obs})
    assert cams==[f'cam{i:02d}' for i in range(2,21)] and len(obs)==1140
    result=[]
    for cam in cams:
        rows=sorted([x for x in obs if x['camera_id']==cam],key=lambda x:x['frame_index'])
        assert [x['frame_index'] for x in rows]==list(range(0,120,2))
        for j,o in enumerate(rows):
            ids=window_indices(j) if mode=='video7' else [j]*7
            assert ids[3]==j
            assert all(abs(x['time']-x['frame_index']/300)<1e-12 for x in rows)
            result.append((o,[rows[i] for i in ids],7-len(set(ids))))
    return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=MODES,required=True)
    ap.add_argument('--repo',type=Path,required=True);ap.add_argument('--weights',type=Path,required=True)
    ap.add_argument('--manifest',type=Path,default=ROOT/'data/dynamic_sr/n3dv_prepared/cook_spinach/manifest.json')
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--limit',type=int,default=0)
    a=ap.parse_args();a.repo=a.repo.resolve();a.weights=a.weights.resolve();a.out=a.out.resolve()
    a.out.mkdir(parents=True,exist_ok=True)
    lock=(a.out/'generation.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    import numpy as np
    import torch
    from PIL import Image
    import mmcv
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    sys.path.insert(0,str(a.repo))
    from mmedit.models.backbones.sr_backbones.basicvsr_pp import BasicVSRPlusPlus
    from mmcv.ops import ModulatedDeformConv2d
    manifest=read(a.manifest);root=a.manifest.resolve().parent;windows=source_windows(manifest,a.mode)
    config=a.repo/'configs/basicvsr_plusplus_reds4.py'
    identity=dict(mode=a.mode,manifest=entry(a.manifest.resolve()),repository='https://github.com/ckkelvinchan/BasicVSR_PlusPlus',
        commit=subprocess.check_output(['git','-C',str(a.repo),'rev-parse','HEAD'],text=True).strip(),
        configuration=dict(path=str(config),sha256=sha(config)),weights=dict(path=str(a.weights),sha256=sha(a.weights)),
        upstream_python_sha256={str(q.relative_to(a.repo)):sha(q) for q in sorted((a.repo/'mmedit').rglob('*.py'))},
        precision='FP32; no autocast; TF32 disabled',quantization=QUANTIZATION,window_length=7,center_index=3,
        source_stride=2,padding='endpoint replication',weight_prefix_adaptation='exclude wrapper step_counter buffer only; strip generator. prefix; strict=True',
        wrapper_adaptations=['spynet_pretrained=None because full checkpoint contains SPyNet','is_mirror_extended=False before each generator call'],
        pretrained_domain='REDS BI x4; official 30-frame training; project bicubic antialias/clamp/PNG may differ',
        generator_shape=[1,7,3,1008,1344],source_shape=[1,7,3,252,336],batch=1,
        code=entry(Path(__file__).resolve()),python=sys.version,torch=str(torch.__version__),cuda=torch.version.cuda,
        mmcv=mmcv.__version__,numpy=np.__version__,host=platform.node(),gpu=torch.cuda.get_device_name(),physical_gpu=os.environ['CUDA_VISIBLE_DEVICES'])
    identity_path=a.out/'identity.json'
    if identity_path.exists():assert read(identity_path)==identity,'Cannot mix cache identities'
    else:write(identity_path,identity)
    # Read files are bound to the train-LR manifest; HR and evaluation caches are not read.
    allowed={str((root/o['lr_path']).resolve()):o['lr_sha256'] for o,_,_ in windows}
    for path,digest in allowed.items():assert sha(path)==digest,path
    source_reads=Counter()
    def guard(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        path=Path(os.fsdecode(args[0])).resolve()
        if path.suffix.lower() not in ['.png','.jpg','.jpeg','.npy','.npz']:return
        if path.is_relative_to(a.out):return
        assert str(path) in allowed,('illegal prior input',str(path));source_reads[str(path)]+=1
    sys.addaudithook(guard)
    state=torch.load(a.weights,map_location='cpu')['state_dict']
    assert state and {k for k in state if not k.startswith('generator.')}=={'step_counter'}
    wrapper_step_counter=state.pop('step_counter')
    assert wrapper_step_counter.numel()==1
    model=BasicVSRPlusPlus(mid_channels=64,num_blocks=7,is_low_res_input=True,spynet_pretrained=None)
    model.load_state_dict({k.removeprefix('generator.'):v for k,v in state.items()},strict=True)
    assert any(isinstance(x,ModulatedDeformConv2d) for x in model.modules()),'Official DCN alignment required'
    model=model.cuda().eval().requires_grad_(False)
    assert not model.training and not any(x.requires_grad for x in model.parameters())
    write(a.out/'load_audit.json',dict(strict=True,missing_keys=[],unexpected_keys=[],parameters=sum(x.numel() for x in model.parameters()),
        frozen=True,eval=True,official_DCN_present=True,excluded_wrapper_step_counter=int(wrapper_step_counter.item()),identity_sha256=sha(identity_path)))
    start=time.monotonic();entries=[];infer_s=0;new_calls=0;disk_s=0;torch.cuda.reset_peak_memory_stats()
    image_cache={};max_cache=60;cache_hits=cache_misses=0
    def tensor(o):
        nonlocal cache_hits,cache_misses
        key=(o['camera_id'],o['frame_index'])
        if key in image_cache:cache_hits+=1;return image_cache[key]
        cache_misses+=1
        with Image.open(root/o['lr_path']) as im:
            assert im.mode=='RGB' and im.size==(336,252)
            value=torch.from_numpy(np.asarray(im).copy()).permute(2,0,1).float().div_(255)
        if len(image_cache)>=max_cache:image_cache.pop(next(iter(image_cache)))
        image_cache[key]=value;return value
    for count,(o,src,duplicates) in enumerate(windows):
        if a.limit and count>=a.limit:break
        dest=a.out/'targets'/o['camera_id']/f"{o['frame_index']:04d}.png";receipt=dest.with_suffix('.json')
        expected=dict(camera=o['camera_id'],frame=o['frame_index'],mode=a.mode,
            sources=[dict(camera=v['camera_id'],frame=v['frame_index'],path=str((root/v['lr_path']).relative_to(ROOT)),sha256=v['lr_sha256']) for v in src],
            repeated_frame_instances=duplicates,center_index=3,identity_sha256=sha(identity_path),
            shape=[3,1008,1344],dtype='uint8 PNG decoded as FP32 RGB/255',quantization=QUANTIZATION)
        if receipt.exists():
            e=read(receipt);assert all(e[k]==v for k,v in expected.items());assert sha(dest)==e['sha256'];entries.append(e);continue
        with torch.inference_mode():
            x=torch.stack([tensor(v) for v in src])[None].cuda()
            assert x.shape==(1,7,3,252,336) and x.dtype==torch.float32
            model.is_mirror_extended=False
            torch.cuda.synchronize();tick=time.monotonic();y=model(x);torch.cuda.synchronize();elapsed=time.monotonic()-tick
            assert tuple(y.shape)==(1,7,3,1008,1344) and y.dtype==torch.float32 and torch.isfinite(y).all()
            center=y[0,3].cpu();assert not y.requires_grad
            new_calls+=1;infer_s+=elapsed;tick=time.monotonic();dest.parent.mkdir(parents=True,exist_ok=True)
            if o['camera_id']=='cam02' and o['frame_index'] in [0,40,80,118]:
                sample=a.out/'float_samples'/f"cam02_{o['frame_index']:04d}.npy";sample.parent.mkdir(exist_ok=True)
                np.save(sample,center.numpy())
            pixels=center.clamp(0,1).mul(255).round().byte().permute(1,2,0).numpy()
            temp=dest.with_suffix('.tmp.png');Image.fromarray(pixels,'RGB').save(temp);temp.replace(dest)
            e=dict(**expected,path=str(dest.relative_to(ROOT)),sha256=sha(dest),inference_seconds=elapsed)
            write(receipt,e);entries.append(e);disk_s+=time.monotonic()-tick
            del x,y,center
        if count==0 or (count+1)%60==0:
            progress=dict(mode=a.mode,completed=len(entries),new_calls=new_calls,inference_seconds=infer_s,
                wall_seconds=time.monotonic()-start,peak_gpu_gb=torch.cuda.max_memory_allocated()/1e9)
            write(a.out/'progress.json',progress);print(json.dumps(progress),flush=True)
    complete=len(entries)==1140
    result=dict(status='completed' if complete else 'engineering_subset',mode=a.mode,identity=identity,identity_sha256=sha(identity_path),entries=entries,
        inference_seconds_total=sum(e['inference_seconds'] for e in entries),new_calls=new_calls,new_inference_seconds=infer_s,
        source_frame_instances=len(entries)*7,unique_source_images=1140,wall_seconds=time.monotonic()-start,disk_write_seconds=disk_s,
        peak_gpu_gb=torch.cuda.max_memory_allocated()/1e9,disk_bytes=sum(local(e['path']).stat().st_size for e in entries),
        source_reads=dict(source_reads),image_cache_hits=cache_hits,image_cache_misses=cache_misses)
    write(a.out/('prior_index.json' if complete else 'engineering_index.json'),result)
    write(a.out/'complete.json',dict(status=result['status'],index=entry(a.out/('prior_index.json' if complete else 'engineering_index.json'))))

if __name__=='__main__':
    try:main()
    except BaseException:
        traceback.print_exc();raise
