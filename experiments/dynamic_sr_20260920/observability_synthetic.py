#!/usr/bin/env python3
"""Controlled local dynamic SR inverse problems. No GS training or HR leakage.

Analytic periodic texture avoids interpolation artifacts. Central crop after
the project's bicubic-AA operator excludes resize boundary clues. Local
Schur calibration is deliberately separated from fixed wrong-motion fitting.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '2')
import argparse, hashlib, json, platform, time
from pathlib import Path
import numpy as np
import scipy.linalg as la
from scipy.ndimage import gaussian_filter
import torch
import torch.nn.functional as F
from threadpoolctl import threadpool_limits

N, SCALE, FRAMES = 32, 4, 16
LOW = N // SCALE
BASE = [(1,0),(0,1),(1,1),(1,-1),(2,0),(0,2),(2,1),(1,2)]
FREQUENCIES = [(x+dx,y+dy) for x,y in BASE for dx,dy in [(0,0),(8,0),(0,8),(8,8)]]
EXTRA = [(3,2),(3,-2),(5,3),(6,-3),(7,4),(11,2),(12,3),(13,-2)]

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,d):
    Path(p).write_text(json.dumps(d,indent=2,ensure_ascii=False,allow_nan=False))

def basis(freq, extent=N, shift=(0.,0.), bend=0.):
    a=np.arange(extent,dtype=np.float64)+.5-(extent-N)//2
    x,y=np.meshgrid(a,a)
    xx=x+shift[0]+bend*np.sin(2*np.pi*y/N)
    yy=y+shift[1]
    f=np.asarray(freq)
    phase=2*np.pi/N*(xx[...,None]*f[:,0]+yy[...,None]*f[:,1])
    z=np.stack([np.cos(phase),np.sin(phase)],axis=-1)*np.sqrt(2.)
    return z.reshape(extent,extent,-1)

def base_operator(freq,kernel):
    f=np.asarray(freq,dtype=float)
    x,y=np.meshgrid((np.arange(LOW)+.5)*SCALE,(np.arange(LOW)+.5)*SCALE)
    ph=2*np.pi/N*(x[...,None]*f[:,0]+y[...,None]*f[:,1])
    z=(np.stack([np.cos(ph),np.sin(ph)],axis=-1)*np.sqrt(2)).reshape(LOW*LOW,-1)
    if kernel=='project_bicubic_aa':
        # No clamp for signed basis. Actual synthetic images stay in [0,1].
        b=basis(freq,3*N)
        x=torch.from_numpy(b.transpose(2,0,1).copy())[None]
        a=F.interpolate(x,size=(3*LOW,3*LOW),mode='bicubic',align_corners=False,antialias=True)
        return a[0,:,LOW:2*LOW,LOW:2*LOW].numpy().reshape(len(freq)*2,-1).T.copy()
    if kernel=='pixel_box': gain=np.sinc(f[:,0]/LOW)*np.sinc(f[:,1]/LOW)
    elif kernel=='ideal_lowpass': gain=(np.max(np.abs(f),axis=1)<LOW/2).astype(float)
    else: raise ValueError(kernel)
    return z*np.repeat(gain,2)

def shifted(a,shifts,freq):
    f=np.asarray(freq,dtype=float)
    ph=2*np.pi/N*(np.asarray(shifts)@f.T)
    c,s=np.cos(ph)[:,None,:],np.sin(ph)[:,None,:]
    u,v=a[:,0::2][None],a[:,1::2][None]
    z=np.stack([u*c-v*s,v*c+u*s],axis=-1)
    return z.reshape(len(shifts),len(a),a.shape[1])

def bent_operator(shifts,amps):
    blocks=[]
    for sh,amp in zip(shifts,amps):
        b=basis(FREQUENCIES,3*N,sh,amp)
        x=torch.from_numpy(b.transpose(2,0,1).copy())[None]
        z=F.interpolate(x,size=(3*LOW,3*LOW),mode='bicubic',align_corners=False,antialias=True)
        blocks.append(z[0,:,LOW:2*LOW,LOW:2*LOW].numpy().reshape(len(FREQUENCIES)*2,-1).T.copy())
    return np.stack(blocks)

def mask(kind):
    m=np.ones((FRAMES,LOW,LOW),bool)
    if kind=='half_fixed': m[:,:,:LOW//2]=False
    elif kind=='half_moving':
        for i in range(FRAMES):
            m[i,:,[(j+i)%LOW for j in range(LOW//2)]]=False
    elif kind!='none': raise ValueError(kind)
    return m.reshape(FRAMES,LOW*LOW)

def spectrum(a,noise,prior):
    s=la.svdvals(a)
    ev=s*s
    return dict(rank=int(np.sum(s>s[0]*1e-10)),columns=a.shape[1],
        singular_min=float(s[-1]),singular_max=float(s[0]),
        modes_snr_above_one=int(np.sum(ev*prior**2>noise**2)),
        effective_dimension=float(np.sum(ev/(ev+noise**2/prior**2))),
        expected_posterior_mse=float(np.mean(noise**2/(ev+noise**2/prior**2))),
        eigenvalues=ev.tolist())

def estimator(a,noise,prior):
    lam=(noise/prior)**2
    return la.solve(a.T@a+lam*np.eye(a.shape[1]),a.T,assume_a='pos')

def ssim(g,p):
    ux,uy=gaussian_filter(g,1.5,truncate=3.3334),gaussian_filter(p,1.5,truncate=3.3334)
    vx=gaussian_filter(g*g,1.5,truncate=3.3334)-ux*ux
    vy=gaussian_filter(p*p,1.5,truncate=3.3334)-uy*uy
    cov=gaussian_filter(g*p,1.5,truncate=3.3334)-ux*uy
    z=((2*ux*uy+.01**2)*(2*cov+.03**2))/((ux*ux+uy*uy+.01**2)*(vx+vy+.03**2))
    return float(z[5:-5,5:-5].mean())

def quality(theta,truth,observed,fit,hold,truehold,bhr,extra=None):
    pred=(bhr@theta).reshape(N,N)+.5
    ref=(bhr@truth).reshape(N,N)+.5
    if extra is not None: ref+=extra
    mse=float(np.mean((pred-ref)**2))
    err=theta-truth
    high=np.repeat([max(abs(x),abs(y))>=LOW/2 for x,y in FREQUENCIES],2)
    return dict(hr_mse=mse,hr_psnr=float(-10*np.log10(max(mse,1e-16))),hr_ssim=ssim(ref,pred),
                coefficient_mse=float(np.mean(err**2)),high_coefficient_mse=float(np.mean(err[high]**2)),
                fitted_lr_mse=float(np.mean((fit@theta-observed)**2)),
                heldout_lr_mse=float(np.mean((hold@theta-truehold)**2)))

def aggregate(rows):
    fields=['hr_mse','hr_psnr','hr_ssim','coefficient_mse','high_coefficient_mse','fitted_lr_mse','heldout_lr_mse']
    out={k:float(np.mean([r[k] for r in rows])) for k in fields}
    out['hr_rmse']=float(np.sqrt(out['hr_mse']))
    out['count']=len(rows)
    return out

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--seeds',type=int,default=32)
    args=p.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    start=time.monotonic();torch.set_num_threads(2)
    prior=.0125;noise=1/255;repeats=2
    grid=np.asarray([(x,y) for y in range(4) for x in range(4)],float)
    schedules={'repeated':np.zeros_like(grid),'integer_motion':grid*4,'complementary':grid}
    config=dict(script_sha256=sha(__file__),seed_count=args.seeds,noise_repeats=repeats,
        hr_size=N,lr_size=LOW,frames=FRAMES,frequencies=FREQUENCIES,extra_frequencies=EXTRA,
        noise_std=noise,coefficient_prior_std=prior,estimated_motion_error_lr=[0,.05,.15,.3],
        schedules_hr_pixels={k:v.tolist() for k,v in schedules.items()},
        degradation_note='Central tile of 3x tiled texture; exact project bicubic antialias=True; signed basis unclamped, image range audited.',
        boundary='No interpolation warp; analytic continuous periodic texture; shift frame 0 anchored.',
        scope='Oracle low-dimensional inverse-problem diagnostic, not a learned method or 4DGS benchmark.',
        versions=dict(torch=torch.__version__,numpy=np.__version__,python=platform.python_version()),
        machine=platform.node(),device='CPU',kernels=['project_bicubic_aa','pixel_box','ideal_lowpass'])
    save(out/'protocol.json',config)
    bhr=basis(FREQUENCIES).reshape(N*N,-1);d=bhr.shape[1]
    assert np.max(np.abs(bhr.T@bhr/(N*N)-np.eye(d)))<1e-12
    lowcols=np.repeat([max(abs(x),abs(y))<LOW/2 for x,y in FREQUENCIES],2)
    draws=[]
    for seed in range(args.seeds):
        rng=np.random.default_rng(8100+seed)
        theta=rng.normal(0,prior,d)
        # Fixed draws paired across all candidate layouts/kernels.
        draws.append((theta,rng.normal(size=(repeats,FRAMES,LOW*LOW)),rng.normal(size=(FRAMES,2))))
    allrows=[];summaries=[];checks=[];gains=[];condition_operators={}
    for kernel in config['kernels']:
        base=base_operator(FREQUENCIES,kernel)
        for fi,f in enumerate(FREQUENCIES):
            gains.append(dict(kernel=kernel,frequency=f,
                lr_response_rms=float(np.sqrt(np.mean(base[:,2*fi:2*fi+2]**2)))))
        for schedule,shifts in schedules.items():
            block=shifted(base,shifts,FREQUENCIES)
            # Known off-grid observations are used only for evaluation.
            testsh=np.array([[.37,1.19],[2.29,.63],[3.47,2.41],[1.73,3.19]])
            hold=shifted(base,testsh,FREQUENCIES).reshape(-1,d)
            for missing in ['none','half_fixed','half_moving']:
                visible=mask(missing);a=block[visible]
                meta=dict(kernel=kernel,schedule=schedule,missing=missing,visible_rows=int(visible.sum()))
                checks.append({**meta,**spectrum(a,noise,prior)})
                for sigma in config['estimated_motion_error_lr']:
                    rows=[]
                    for seed,(theta,eps,uniterr) in enumerate(draws):
                        err=uniterr*sigma*SCALE;err[0]=0
                        fitted=shifted(base,shifts+err,FREQUENCIES)[visible]
                        k=estimator(fitted,noise,prior)
                        clean=a@theta;held=hold@theta
                        for rep in range(repeats):
                            obs=clean+noise*eps[rep][visible]
                            est=k@obs
                            r={**meta,'motion_error_lr':sigma,'seed':seed,'noise_rep':rep,
                               **quality(est,theta,obs,fitted,hold,held,bhr)}
                            rows.append(r)
                    allrows.extend(rows);summaries.append({**meta,'motion_error_lr':sigma,**aggregate(rows)})
            # Noiseless pseudo-inverse and a low-frequency negative control.
            full=block.reshape(-1,d)
            exact_truth=draws[0][0]
            recovered=la.pinv(full,rtol=1e-10)@(full@exact_truth)
            alow=full[:,lowcols];k=estimator(alow,noise,prior)
            lowerrors=[]
            for theta,eps,_ in draws:
                for rep in range(repeats):
                    y=alow@theta[lowcols]+noise*eps[rep].reshape(-1)
                    lowerrors.append(float(np.mean((k@y-theta[lowcols])**2)))
            condition_operators[(kernel,schedule)]=dict(a=full,base=base,hold=hold)
            checks.append(dict(kernel=kernel,schedule=schedule,diagnostic='noiseless_and_lowpass_control',
                noiseless_coefficient_rmse=float(np.sqrt(np.mean((recovered-exact_truth)**2))),
                low_only_mean_coefficient_mse=float(np.mean(lowerrors)),low_only_rank=int(np.linalg.matrix_rank(alow)),low_only_columns=int(lowcols.sum())))
    # Explicitly distinguish fixed wrong motion above from locally unknown motion.
    local=[];base=base_operator(FREQUENCIES,'project_bicubic_aa')
    theta0=np.random.default_rng(513).normal(0,prior,d)
    localprior=.00625;mc=512
    for schedule in ['repeated','complementary']:
        shifts=schedules[schedule];blocks=shifted(base,shifts,FREQUENCIES);a=blocks.reshape(-1,d)
        jn=np.zeros((FRAMES*LOW*LOW,2*(FRAMES-1)))
        f=np.asarray(FREQUENCIES)
        for t in range(1,FRAMES):
            for axis in range(2):
                deriv=blocks[t].copy();freq=2*np.pi/N*f[:,axis]
                deriv[:,0::2]=-blocks[t,:,1::2]*freq
                deriv[:,1::2]=blocks[t,:,0::2]*freq
                jn[t*LOW*LOW:(t+1)*LOW*LOW,2*(t-1)+axis]=deriv@theta0
        hk=a.T@a/noise**2
        for sigma in [.025,.10,.25]:
            sigman=sigma*SCALE
            hnn=jn.T@jn/noise**2+np.eye(jn.shape[1])/sigman**2
            cross=a.T@jn/noise**2
            effective=hk-cross@la.solve(hnn,cross.T,assume_a='pos')
            effective=(effective+effective.T)/2
            cov=la.inv(effective+np.eye(d)/localprior**2)
            effrhs=a.T/noise**2-cross@la.solve(hnn,jn.T/noise**2,assume_a='pos')
            ke=cov@effrhs
            covk=la.inv(hk+np.eye(d)/localprior**2);kk=covk@a.T/noise**2
            rng=np.random.default_rng(915)
            dd=rng.normal(0,localprior,(d,mc));nn=rng.normal(0,sigman,(jn.shape[1],mc));eps=rng.normal(0,noise,(len(a),mc))
            lin=a@dd+jn@nn+eps
            nonlinear=[]
            for i in range(mc):
                perturb=np.vstack([np.zeros((1,2)),nn[:,i].reshape(-1,2)])
                actual=shifted(base,shifts+perturb,FREQUENCIES).reshape(-1,d)
                nonlinear.append(actual@(theta0+dd[:,i])-a@theta0+eps[:,i])
            nonlinear=np.stack(nonlinear,axis=1)
            for estimator_name,kmat,pred in [('motion_assumed_known',kk,float(np.trace(covk)/d)),('nuisance_aware',ke,float(np.trace(cov)/d))]:
                for generator,y in [('linearized',lin),('nonlinear_exact_shift',nonlinear)]:
                    mse=float(np.mean((kmat@y-dd)**2))
                    local.append(dict(schedule=schedule,motion_uncertainty_lr=sigma,estimator=estimator_name,generator=generator,
                        empirical_coefficient_mse=mse,predicted_mse=pred,empirical_over_predicted=mse/pred,
                        trials=mc,minimum_effective_eigenvalue=float(la.eigvalsh(effective)[0])))
    # Model mismatch: unmodelled frequencies. No claims of realism/novelty from in-basis algebra.
    mismatch=[]
    bex=basis(EXTRA).reshape(N*N,-1)
    for kernel in config['kernels']:
        exbase=base_operator(EXTRA,kernel)
        for schedule in ['repeated','complementary']:
            mats=condition_operators[(kernel,schedule)];a=mats['a'];hold=mats['hold'];k=estimator(a,noise,prior)
            aex=shifted(exbase,schedules[schedule],EXTRA).reshape(-1,len(EXTRA)*2)
            hextra=shifted(exbase,testsh,EXTRA).reshape(-1,len(EXTRA)*2)
            rows=[]
            for seed,(theta,eps,_) in enumerate(draws):
                phi=np.random.default_rng(4561+seed).normal(0,.04/np.sqrt(len(EXTRA)*2),len(EXTRA)*2)
                for rep in range(repeats):
                    obs=a@theta+aex@phi+noise*eps[rep].reshape(-1)
                    r=quality(k@obs,theta,obs,a,hold,hold@theta+hextra@phi,bhr,(bex@phi).reshape(N,N))
                    rows.append(r)
            mismatch.append(dict(kernel=kernel,schedule=schedule,unmodelled_texture_rms=.04,**aggregate(rows)))
    # Known nonrigid deformation vs pretending its only effect was translation.
    nonrigid=[]
    shifts=schedules['complementary'];nominal=shifted(base,shifts,FREQUENCIES).reshape(-1,d)
    for amplitude in [.5,1.5]:
        amps=amplitude*np.sin(np.arange(FRAMES)*2*np.pi/FRAMES)
        actual=bent_operator(shifts,amps).reshape(-1,d)
        for model,fit in [('known_nonrigid',actual),('translation_only_mismatch',nominal)]:
            k=estimator(fit,noise,prior);rows=[]
            hold=condition_operators[('project_bicubic_aa','complementary')]['hold']
            for theta,eps,_ in draws:
                for rep in range(repeats):
                    obs=actual@theta+noise*eps[rep].reshape(-1)
                    rows.append(quality(k@obs,theta,obs,fit,hold,hold@theta,bhr))
            nonrigid.append(dict(bend_amplitude_hr=amplitude,model=model,**aggregate(rows)))
    # Validate shift operator against directly evaluated padded analytic texture.
    sh=np.array([.37,1.19]);direct=basis(FREQUENCIES,3*N,sh)
    x=torch.from_numpy(direct.transpose(2,0,1).copy())[None]
    z=F.interpolate(x,size=(3*LOW,3*LOW),mode='bicubic',align_corners=False,antialias=True)
    direct=z[0,:,LOW:2*LOW,LOW:2*LOW].numpy().reshape(d,-1).T
    parity=float(np.max(np.abs(direct-shifted(base,sh[None],FREQUENCIES)[0])))
    assert parity<1e-12,parity
    images=np.array([bhr@q[0]+.5 for q in draws])
    assert images.min()>0 and images.max()<1,(images.min(),images.max())
    save(out/'rows.json',allrows)
    result=dict(completed=True,protocol_sha256=sha(out/'protocol.json'),script_sha256=sha(__file__),
        summaries=summaries,operator_checks=checks,frequency_response=gains,local_nuisance=local,
        out_of_basis=mismatch,nonrigid=nonrigid,analytic_shift_parity_max=parity,
        reference_image_range=[float(images.min()),float(images.max())],elapsed_seconds=time.monotonic()-start,
        limitations=['Known small Fourier dictionary and calibrated coefficient prior are privileged synthetic assumptions.',
            'Linear Schur calibration is an implementation sanity check; nonlinear generator and out-of-basis checks probe its limits.',
            'Patch metrics are grayscale PSNR/SSIM; LPIPS is inappropriate as primary score for 32-pixel Fourier patterns.',
            'Masks remove rows; they do not simulate alpha-composited occluders or validate real visibility estimation.',
            'Wrong-fixed-motion cases measure model bias and must not be treated as empirical Schur covariance calibration.'])
    save(out/'metrics.json',result)
    save(out/'completion.json',dict(completed=True,metrics_sha256=sha(out/'metrics.json'),rows_sha256=sha(out/'rows.json'),elapsed_seconds=result['elapsed_seconds']))
    print(json.dumps(dict(completed=True,elapsed_seconds=result['elapsed_seconds'],cases=len(summaries),fits=len(allrows),shift_parity=parity)))

if __name__=='__main__':
    with threadpool_limits(limits=1): main()
