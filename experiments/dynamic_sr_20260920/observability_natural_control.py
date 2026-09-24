"""Known-motion natural-texture pixel inverse problem; no learned network.

32x32 HR patches are selected by an earlier LR-only grid protocol. HR supplies
the synthetic latent texture and evaluation target, never regularizer tuning.
Periodic integer HR shifts are controlled motions, not real scene motion.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import platform
import socket
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'output/dynamic_sr_20260920/observability_natural_control_v1'
PARENT = ROOT/'output/dynamic_sr_20260920/observability_real_v1/metrics.json'
SCENES = {'cook_spinach':'n3dv_prepared/cook_spinach',
          'meetroom_discussion':'meetroom_prepared/discussion'}
LAMBDAS = [1e-4,1e-3,1e-2]
N = 32


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def degrade(x):
    """Input N,C,32,32. Tile 3x3, exact project AA, center 8x8 LR."""
    tiled=x.repeat(1,1,3,3)
    return F.interpolate(tiled,size=(24,24),mode='bicubic',align_corners=False,antialias=True)[...,8:16,8:16]


def up(x):
    return F.interpolate(x,size=(32,32),mode='bicubic',align_corners=False)


def measure(pred,gt):
    p=np.clip(pred,0,1)[4:28,4:28]
    g=gt[4:28,4:28]
    mse=float(np.mean((p-g)**2))
    blur=lambda x:cv2.GaussianBlur(x.astype(np.float64),(11,11),1.5)
    a,b=blur(p),blur(g)
    va,vb,cov=blur(p*p)-a*a,blur(g*g)-b*b,blur(p*g)-a*b
    ss=((2*a*b+.01**2)*(2*cov+.03**2))/((a*a+b*b+.01**2)*(va+vb+.03**2))
    pt=torch.from_numpy(np.clip(pred,0,1).transpose(2,0,1).copy()).unsqueeze(0)
    gtens=torch.from_numpy(gt.transpose(2,0,1).copy()).unsqueeze(0)
    hp=(pt-up(degrade(pt)))[0,:,4:28,4:28].numpy()
    hg=(gtens-up(degrade(gtens)))[0,:,4:28,4:28].numpy()
    he=float(np.mean(hg**2))
    hm=float(np.mean((hp-hg)**2))
    return dict(psnr=float(-10*np.log10(max(mse,1e-30))),mse=mse,ssim=float(ss[5:-5,5:-5].mean()),
        high_residual_mse=hm,reference_high_energy=he,high_residual_error_ratio=hm/max(he,1e-30),
        clip_fraction=float(((pred<0)|(pred>1)).mean()),
        unclipped_center_mse=float(np.mean((pred[4:28,4:28]-g)**2)))


def main():
    start=time.monotonic()
    start_utc=datetime.now(timezone.utc).isoformat()
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    torch.set_default_dtype(torch.float64)
    OUT.mkdir(parents=True,exist_ok=False)
    (OUT/'source.py').write_text(Path(__file__).read_text())
    parent=json.loads(PARENT.read_text())
    chosen=[]
    inputs={str(PARENT):sha(PARENT)}
    patches=[]
    for scene,sub in SCENES.items():
        cs=sorted([c for c in parent['cells'] if c['scene']==scene and c['phase_rich'] and c['moving_proxy']],
                  key=lambda c:(c['camera'],c['anchor'],c['y_lr'],c['x_lr']))[:8]
        assert len(cs)==8
        for c in cs:
            path=ROOT/'data/dynamic_sr'/sub/f'hr/{c["camera"]}/{c["anchor"]:04d}.png'
            h=sha(path)
            assert h==parent['input_sha256'][str(path)]
            inputs[str(path)]=h
            im=np.asarray(Image.open(path).convert('RGB'),dtype=np.float64)/255.
            x,y=c['x_lr']*4,c['y_lr']*4
            patch=im[y-16:y+16,x-16:x+16].copy()
            assert patch.shape==(32,32,3)
            patches.append(patch)
            chosen.append({k:c[k] for k in ['scene','camera','anchor','x_lr','y_lr']})
    target=torch.from_numpy(np.stack(patches).transpose(0,3,1,2))
    # Each basis is a single HR pixel. Degrading it constructs rows without
    # assuming a Fourier representation or an analytic kernel approximation.
    basis=torch.eye(N*N).reshape(N*N,1,N,N)
    a0=degrade(basis).reshape(N*N,64).T.contiguous()
    shifts=[(dy,dx) for dy in range(4) for dx in range(4)]
    operators={}
    measurements={}
    parity={}
    for name,motions in [('repeated',[(0,0)]*16),('complementary',shifts)]:
        blocks=[]
        yy=[]
        diffs=[]
        for dy,dx in motions:
            # y=A0 roll(x,+d); its coefficient row is roll(A0,-d).
            ai=torch.roll(a0.reshape(64,N,N),shifts=(-dy,-dx),dims=(1,2)).reshape(64,N*N)
            exact=degrade(torch.roll(target,shifts=(dy,dx),dims=(2,3)))
            linear=(ai@target.reshape(len(patches),3,N*N).permute(2,0,1).reshape(N*N,-1)).reshape(64,len(patches),3).permute(1,2,0).reshape(len(patches),3,8,8)
            diffs.append(float(torch.max(torch.abs(exact-linear))))
            assert diffs[-1]<1e-10
            blocks.append(ai)
            yy.append(torch.round(exact.clamp(0,1)*255)/255)
        operators[name]=torch.cat(blocks,dim=0)
        measurements[name]=torch.stack(yy,dim=0).permute(0,3,4,1,2).reshape(1024,-1)
        parity[name]=dict(max_abs=max(diffs),per_shift_max_abs=diffs)
    # Additional random texture probe catches accidental natural-patch symmetry.
    random=torch.rand((2,3,N,N),generator=torch.Generator().manual_seed(4920))
    random_err=[]
    for dy,dx in shifts:
        ai=torch.roll(a0.reshape(64,N,N),shifts=(-dy,-dx),dims=(1,2)).reshape(64,N*N)
        direct=degrade(torch.roll(random,(dy,dx),(2,3))).flatten(2)
        pred=torch.einsum('kh,nch->nck',ai,random.flatten(2))
        random_err.append(float(torch.max(torch.abs(direct-pred))))
    assert max(random_err)<1e-10
    parity['random_max_abs']=max(random_err)
    # Periodic first forward differences: x[i]-x[i+1] on each axis.
    eye=torch.eye(N*N)
    gx=eye-torch.roll(eye.reshape(N*N,N,N),shifts=-1,dims=2).reshape(N*N,N*N)
    gy=eye-torch.roll(eye.reshape(N*N,N,N),shifts=-1,dims=1).reshape(N*N,N*N)
    regularizer=gx.T@gx+gy.T@gy
    assert float(torch.max(torch.abs(regularizer@torch.ones(N*N))))<1e-12
    predictions={}
    # Single-image baseline uses the same quantized zero-motion observation.
    zero=torch.round(degrade(target).clamp(0,1)*255)/255
    predictions['single_bicubic']=up(zero).permute(0,2,3,1).numpy()
    solver_checks={}
    spectra={}
    for name,a in operators.items():
        gram=a.T@a
        eig=torch.linalg.eigvalsh(gram)
        spectra[name]=dict(max=float(eig[-1]),min=float(eig[0]),
            rank_at_relative_1e_8=int((eig>eig[-1]*1e-8).sum()),
            rank_at_relative_1e_4=int((eig>eig[-1]*1e-4).sum()),
            trace=float(eig.sum()))
        rhs=a.T@measurements[name]
        for lam in LAMBDAS:
            system=gram+lam*regularizer+1e-10*eye
            chol=torch.linalg.cholesky(system)
            solution=torch.cholesky_solve(rhs,chol)
            key=f'{name}_lambda_{lam:g}'
            predictions[key]=solution.reshape(N*N,len(patches),3).permute(1,0,2).reshape(len(patches),N,N,3).numpy()
            solver_checks[key]=dict(relative_linear_residual=float(torch.linalg.norm(system@solution-rhs)/torch.linalg.norm(rhs)),
                quantized_observation_mse=float(torch.mean((a@solution-measurements[name])**2)))
            assert solver_checks[key]['relative_linear_residual']<1e-8
    rows=[]
    for i,c in enumerate(chosen):
        rows.append(dict(**c,metrics={k:measure(v[i],patches[i]) for k,v in predictions.items()}))
    summary={}
    for scene in SCENES:
        rr=[r for r in rows if r['scene']==scene]
        summary[scene]={k:{m:float(np.mean([r['metrics'][k][m] for r in rr]))
                           for m in ['psnr','ssim','high_residual_mse','high_residual_error_ratio','clip_fraction']}
                        for k in predictions}
        summary[scene]['paired_deltas']={str(lam):dict(
            psnr=float(np.mean([r['metrics'][f'complementary_lambda_{lam:g}']['psnr']-r['metrics'][f'repeated_lambda_{lam:g}']['psnr'] for r in rr])),
            ssim=float(np.mean([r['metrics'][f'complementary_lambda_{lam:g}']['ssim']-r['metrics'][f'repeated_lambda_{lam:g}']['ssim'] for r in rr])),
            psnr_win_count=sum(r['metrics'][f'complementary_lambda_{lam:g}']['psnr']>r['metrics'][f'repeated_lambda_{lam:g}']['psnr'] for r in rr))
            for lam in LAMBDAS}
    # Use the predeclared middle lambda for display, never choose it by HR score.
    canvas=Image.new('RGB',(640,16*180),'white')
    draw=ImageDraw.Draw(canvas)
    for i,c in enumerate(chosen):
        ims=[('HR',patches[i]),('Single',predictions['single_bicubic'][i]),
             ('Repeat 1e-3',predictions['repeated_lambda_0.001'][i]),
             ('Phase 1e-3',predictions['complementary_lambda_0.001'][i])]
        for j,(name,im) in enumerate(ims):
            draw.text((j*160+2,i*180+2),name,fill='black')
            canvas.paste(Image.fromarray(np.round(np.clip(im,0,1)*255).astype(np.uint8)).resize((160,160),Image.Resampling.NEAREST),(j*160,i*180+18))
    canvas.save(OUT/'all_patches_middle_lambda.png')
    np.savez_compressed(OUT/'predictions.npz',target=np.stack(patches),**predictions)
    np.savez_compressed(OUT/'operators.npz',a0=a0.numpy(),**{k:v.numpy() for k,v in operators.items()})
    protocol=dict(representation='1024 pixel unknowns per color, no texture dictionary',
        selected_from_lr_only_parent=True,selection_order='scene, camera, anchor, y_lr, x_lr; first 8 per scene',
        hr_use='controlled latent source and evaluation; never choose lambda by HR',
        periodic_hr_tile=[32,32],tiled_shape=[96,96],center_lr=[8,8],evaluation_center=[24,24],
        motions_hr_pixels=shifts,n_frames_each=16,quantization='clamp [0,1], round to uint8 /255, no extra noise',
        degradation='torch bicubic antialias=True align_corners=False exactly x4 on 3x3 periodic tile',
        regularization='||A x-y||^2 + lambda sum((periodic dx x)^2+(periodic dy x)^2) +1e-10 ||x||^2',
        lambdas=LAMBDAS,postprocess='clip reconstructed RGB to [0,1] for primary metrics; unclipped MSE and clip fractions saved',
        ssim='11x11 sigma1.5 Gaussian, population covariance, mean over valid centers inside evaluation crop',
        lpips='omitted for 24x24 evaluation; no perceptual claim',
        limitations=['controlled periodic rigid translation, not measured nonrigid real motion',
          'known exact warp and blur, no correspondence error or occlusion',
          'few correlated LR-selected patches, not independent benchmark',
          'real photographs do not remove the synthetic-motion assumption'])
    result=dict(protocol=protocol,identity=dict(host=socket.gethostname(),platform=platform.platform(),
        started_utc=start_utc,finished_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-start,
        python=platform.python_version(),torch=torch.__version__,opencv=cv2.__version__,source_sha256=sha(__file__),gpu_used=False),
        input_sha256=inputs,operator_parity=parity,solver_checks=solver_checks,spectra=spectra,
        rows=rows,summary=summary)
    (OUT/'metrics.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    (OUT/'complete.json').write_text(json.dumps(dict(metrics_sha256=sha(OUT/'metrics.json'),source_sha256=sha(__file__)),indent=2))
    print(json.dumps({k:result[k] for k in ['identity','operator_parity','spectra','summary']},indent=2))


if __name__=='__main__':
    main()
