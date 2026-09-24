"""Independent flattening/regularizer and float32-quantization audit."""
import json
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import observability_natural_control as ref


def main():
    start=time.monotonic()
    torch.set_num_threads(4)
    p=ref.OUT
    m=json.loads((p/'metrics.json').read_text())
    o=np.load(p/'operators.npz')
    saved=np.load(p/'predictions.npz')
    target=saved['target']
    # Construct finite differences from explicit (row,column) neighbors.
    gx=np.zeros((1024,1024));gy=gx.copy()
    for y in range(32):
        for x in range(32):
            i=y*32+x
            gx[i,i]=gy[i,i]=1
            gx[i,y*32+(x+1)%32]=-1
            gy[i,((y+1)%32)*32+x]=-1
    reg=torch.from_numpy(gx.T@gx+gy.T@gy)
    audit={}
    for name,shifts in [('repeated',[(0,0)]*16),('complementary',[(y,x) for y in range(4) for x in range(4)])]:
        y32=[];y64=[]
        for dy,dx in shifts:
            batch=np.stack([np.roll(im,(dy,dx),(0,1)) for im in target])
            # Direct image degradation independently using full spatial tensors.
            ll=[]
            for dtype in [torch.float32,torch.float64]:
                a=torch.tensor(batch.transpose(0,3,1,2).copy(),dtype=dtype)
                a=a.repeat(1,1,3,3)
                low=F.interpolate(a,size=(24,24),mode='bicubic',antialias=True,align_corners=False)[:,:,8:16,8:16]
                ll.append((low.clamp(0,1)*255).round().numpy()/255)
            # Flatten explicitly one LR site and one patch/color at a time.
            for yy in range(8):
                for xx in range(8):
                    y32.append(ll[0][:,:,yy,xx].reshape(-1))
                    y64.append(ll[1][:,:,yy,xx].reshape(-1))
        y32=np.stack(y32).astype(np.float64);y64=np.stack(y64)
        a=torch.from_numpy(o[name])
        aa=a.T@a
        rhs32=a.T@torch.from_numpy(y32)
        rhs64=a.T@torch.from_numpy(y64)
        audit[name]=dict(float32_uint8_quantization_changed_fraction=float(np.mean(np.round(y32*255)!=np.round(y64*255))),
                        float32_numeric_representation_changed_fraction=float(np.mean(y32!=y64)),
                        float32_quantized_max_abs=float(np.max(np.abs(y32-y64))),solvers={})
        for lam in ref.LAMBDAS:
            chol=torch.linalg.cholesky(aa+lam*reg+1e-10*torch.eye(1024,dtype=torch.float64))
            pred32=torch.cholesky_solve(rhs32,chol).numpy().reshape(1024,16,3).transpose(1,0,2).reshape(16,32,32,3)
            pred64=torch.cholesky_solve(rhs64,chol).numpy().reshape(1024,16,3).transpose(1,0,2).reshape(16,32,32,3)
            key=f'{name}_lambda_{lam:g}'
            parity=float(np.max(np.abs(pred64-saved[key])))
            assert parity<1e-9
            diffs=[]
            for i,gt in enumerate(target):
                pp=np.clip(pred32[i],0,1)[4:28,4:28];old=np.clip(saved[key][i],0,1)[4:28,4:28];gg=gt[4:28,4:28]
                diff=float(-10*np.log10(np.mean((pp-gg)**2))+10*np.log10(np.mean((old-gg)**2)))
                diffs.append(diff)
            audit[name]['solvers'][str(lam)]=dict(independent_flattening_regularizer_max_abs=parity,
                float32_psnr_delta_mean=float(np.mean(diffs)),float32_psnr_delta_max_abs=float(np.max(np.abs(diffs))))
    out=dict(parent_metrics_sha256=ref.sha(p/'metrics.json'),source_sha256=ref.sha(__file__),
             elapsed_seconds=time.monotonic()-start,audit=audit)
    (p/'audit_source.py').write_text(Path(__file__).read_text())
    (p/'audit.json').write_text(json.dumps(out,indent=2))
    print(json.dumps(out,indent=2))


if __name__=='__main__':
    main()
