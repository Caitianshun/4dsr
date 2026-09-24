"""Frozen SwinIR same-input references missing from the historical cache.
No historical cache is modified. Uses only actual LR, never reads HR.
"""
from pathlib import Path
import sys,json,time,hashlib,socket
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/dynamic_sr_20260918/vendor'))
sys.path.insert(0,str(ROOT/'experiments/dynamic_sr_20260918'))
import generate_prior as gp
import torch
import numpy as np
from PIL import Image
OUT=ROOT/'output/dynamic_sr_20260921/local_inverse_teacher_v1'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    start=time.monotonic();OUT.mkdir(parents=True,exist_ok=False)
    (OUT/'source.py').write_text(Path(__file__).read_text())
    assert sha(gp.DEFAULT_NETWORK)==gp.NETWORK_SHA256
    assert sha(gp.DEFAULT_CHECKPOINT)==gp.CHECKPOINT_SHA256
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    model=gp.build_model(gp.DEFAULT_NETWORK,gp.DEFAULT_CHECKPOINT,'cuda:0')
    data=ROOT/'data/dynamic_sr/meetroom_prepared/discussion';manifest=json.loads((data/'manifest.json').read_text());records=[]
    # Also repeat one old cached frame to check cross-GPU numerics.
    for cam,f in [('cam06',24),('cam06',56),('cam06',88),('cam02',24)]:
        ob=next(o for o in manifest['observations'] if o['camera_id']==cam and o['frame_index']==f)
        p=data/ob['lr_path'];assert sha(p)==ob['lr_sha256'];im=np.asarray(Image.open(p).convert('RGB'))
        result=gp.infer(model,im,'cuda:0',0,32);dest=OUT/f'{cam}_{f:04d}.png';Image.fromarray(result).save(dest)
        r=dict(camera=cam,frame=f,input_path=str(p),input_sha256=sha(p),output_path=str(dest),output_sha256=sha(dest))
        old=data/f'sr_swinir_x4/{cam}/{f:04d}.png'
        if old.exists():
            a=np.asarray(Image.open(old).convert('RGB'));diff=result.astype(int)-a.astype(int)
            r.update(old_sha256=sha(old),old_repeat_max_abs_uint8=int(abs(diff).max()),old_repeat_fraction_changed=float((diff!=0).mean()),old_repeat_mse_01=float(np.mean((diff/255.)**2)))
        records.append(r);print(json.dumps(r),flush=True)
    d=dict(source_sha256=sha(__file__),network_sha256=gp.NETWORK_SHA256,checkpoint_sha256=gp.CHECKPOINT_SHA256,generator_sha256=sha(gp.__file__),host=socket.gethostname(),gpu=torch.cuda.get_device_name(0),precision='fp32, tf32 disabled',tile=0,overlap=32,torch=torch.__version__,peak_bytes=torch.cuda.max_memory_allocated(),elapsed_seconds=time.monotonic()-start,records=records)
    (OUT/'receipt.json').write_text(json.dumps(d,indent=2));(OUT/'complete.json').write_text(json.dumps(dict(receipt_sha256=sha(OUT/'receipt.json'))))
if __name__=='__main__':main()
