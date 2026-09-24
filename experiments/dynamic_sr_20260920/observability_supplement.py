#!/usr/bin/env python3
"""Prespecified robustness checks after v1 audit; never overwrite v1."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse,json,time
from pathlib import Path
import numpy as np
import scipy.linalg as la
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from threadpoolctl import threadpool_limits
from observability_synthetic import (basis,base_operator,shifted,estimator,quality,aggregate,
    FREQUENCIES,N,LOW,SCALE,FRAMES,sha,save)

def paired(rows,left,right,field='hr_psnr'):
    a={r['seed']:r[field] for r in rows if r['schedule']==left}
    b={r['seed']:r[field] for r in rows if r['schedule']==right}
    diff=np.array([b[i]-a[i] for i in sorted(a)])
    rng=np.random.default_rng(431)
    boot=np.mean(diff[rng.integers(0,len(diff),(4000,len(diff)))],axis=1)
    return dict(mean=float(diff.mean()),seed_std=float(diff.std(ddof=1)),
        paired_bootstrap95=[float(x) for x in np.quantile(boot,[.025,.975])],
        fraction_positive=float(np.mean(diff>0)),independent_unit='texture seed, paired layouts')

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);args=p.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    torch.set_num_threads(2)
    config=dict(script_sha256=sha(__file__),base_script_sha256=sha(Path(__file__).with_name('observability_synthetic.py')),
        seeds=32,quantization='round(255*x)/255, no extra sensor noise; float before quantization audited in [0,1]',
        fit_prior_std=.0125,quant_fit_noise_std=1/(255*np.sqrt(12)),
        alias_extra=[[-7,0],[-6,1],[1,-7],[2,-7],[-7,2],[-8,1],[-6,0],[0,-7]],
        local_bases=list(range(1200,1208)),scope='supplement after detecting orthogonal v1 EXTRA; not hidden replacement')
    save(out/'protocol.json',config)
    grid=np.array([(x,y) for y in range(4) for x in range(4)],float)
    shifts={'repeated':np.zeros_like(grid),'complementary':grid}
    bhr=basis(FREQUENCIES).reshape(N*N,-1);d=bhr.shape[1];prior=.0125
    tests=np.array([[.37,1.19],[2.29,.63],[3.47,2.41],[1.73,3.19]])
    quantrows=[];qsum=[];noise_rows=[];noise_sum=[];aliasrows=[];alias_summary=[];normalization=[]
    for kernel in ['project_bicubic_aa','pixel_box','ideal_lowpass']:
        base=base_operator(FREQUENCIES,kernel);hold=shifted(base,tests,FREQUENCIES).reshape(-1,d)
        for regularizer_scale in [.5,1.,2.]:
            for schedule,sh in shifts.items():
                a=shifted(base,sh,FREQUENCIES).reshape(-1,d)
                noise=config['quant_fit_noise_std'];k=estimator(a,noise,prior*regularizer_scale)
                for seed in range(config['seeds']):
                    theta=np.random.default_rng(8100+seed).normal(0,prior,d)
                    clean=a@theta+.5
                    assert clean.min()>0 and clean.max()<1
                    obs=np.rint(clean*255)/255-.5
                    quantrows.append(dict(kernel=kernel,schedule=schedule,seed=seed,regularizer_scale=regularizer_scale,
                        **quality(k@obs,theta,obs,a,hold,hold@theta,bhr),actual_quant_error_mse=float(np.mean((obs-(clean-.5))**2))))
            rows=[r for r in quantrows if r['kernel']==kernel and r['regularizer_scale']==regularizer_scale]
            qsum.append(dict(kernel=kernel,regularizer_scale=regularizer_scale,
                means={s:aggregate([r for r in rows if r['schedule']==s]) for s in shifts},paired_gain=paired(rows,'repeated','complementary')))
        if kernel=='project_bicubic_aa':
            for noise_factor in [.25,.5,1.,2.]:
                ns=noise_factor/255
                for schedule,sh in shifts.items():
                    a=shifted(base,sh,FREQUENCIES).reshape(-1,d);k=estimator(a,ns,prior)
                    for seed in range(config['seeds']):
                        rng=np.random.default_rng(8100+seed);theta=rng.normal(0,prior,d);eps=rng.normal(size=len(a))
                        obs=a@theta+ns*eps
                        noise_rows.append(dict(noise_std=ns,noise_factor=noise_factor,schedule=schedule,seed=seed,
                            **quality(k@obs,theta,obs,a,hold,hold@theta,bhr)))
                rows=[r for r in noise_rows if r['noise_factor']==noise_factor]
                noise_sum.append(dict(noise_factor=noise_factor,paired_gain=paired(rows,'repeated','complementary'),
                    means={s:aggregate([r for r in rows if r['schedule']==s]) for s in shifts}))
        # Shared LR alias groups deliberately overlap fitted modes.
        extra=config['alias_extra'];bextra=basis(extra).reshape(N*N,-1);ax=base_operator(extra,kernel)
        for schedule,sh in shifts.items():
            a=shifted(base,sh,FREQUENCIES).reshape(-1,d);ae=shifted(ax,sh,extra).reshape(-1,len(extra)*2)
            he=shifted(ax,tests,extra).reshape(-1,len(extra)*2);k=estimator(a,1/255,prior)
            col_interaction=float(np.linalg.norm(a.T@ae))
            for seed in range(config['seeds']):
                rng=np.random.default_rng(8100+seed);theta=rng.normal(0,prior,d);eps=rng.normal(size=len(a))
                phi=np.random.default_rng(4561+seed).normal(0,.04/np.sqrt(16),16)
                obs=a@theta+ae@phi+eps/255
                aliasrows.append(dict(kernel=kernel,schedule=schedule,seed=seed,
                    **quality(k@obs,theta,obs,a,hold,hold@theta+he@phi,bhr,(bextra@phi).reshape(N,N))))
            rows=[r for r in aliasrows if r['kernel']==kernel and r['schedule']==schedule]
            alias_summary.append(dict(kernel=kernel,schedule=schedule,column_interaction_frobenius=col_interaction,**aggregate(rows)))
    # Multiple independent linearization points, nonlinear exact-shift generator only.
    local=[];base=base_operator(FREQUENCIES,'project_bicubic_aa');sh=grid
    blocks=shifted(base,sh,FREQUENCIES);a=blocks.reshape(-1,d);ns=1/255;lp=.00625
    hk=a.T@a/ns**2;covk=la.inv(hk+np.eye(d)/lp**2);kk=covk@a.T/ns**2
    f=np.array(FREQUENCIES);mc=256
    for baseseed in config['local_bases']:
        theta0=np.random.default_rng(baseseed).normal(0,prior,d)
        jn=np.zeros((len(a),30))
        for t in range(1,FRAMES):
            for ax in range(2):
                deriv=blocks[t].copy();freq=2*np.pi/N*f[:,ax]
                deriv[:,0::2]=-blocks[t,:,1::2]*freq;deriv[:,1::2]=blocks[t,:,0::2]*freq
                jn[t*LOW**2:(t+1)*LOW**2,2*(t-1)+ax]=deriv@theta0
        for sigma in [.025,.1,.25]:
            sigman=sigma*SCALE;hnn=jn.T@jn/ns**2+np.eye(30)/sigman**2;cross=a.T@jn/ns**2
            eff=hk-cross@la.solve(hnn,cross.T,assume_a='pos');eff=(eff+eff.T)/2
            cov=la.inv(eff+np.eye(d)/lp**2);ke=cov@(a.T/ns**2-cross@la.solve(hnn,jn.T/ns**2,assume_a='pos'))
            rng=np.random.default_rng(120000+baseseed);dd=rng.normal(0,lp,(d,mc));nn=rng.normal(0,sigman,(30,mc));eps=rng.normal(0,ns,(len(a),mc))
            obs=[]
            for i in range(mc):
                pert=np.vstack([np.zeros((1,2)),nn[:,i].reshape(-1,2)])
                obs.append(shifted(base,sh+pert,FREQUENCIES).reshape(-1,d)@(theta0+dd[:,i])-a@theta0+eps[:,i])
            obs=np.stack(obs,axis=1)
            for name,k,c in [('motion_assumed_known',kk,covk),('nuisance_aware',ke,cov)]:
                mse=float(np.mean((k@obs-dd)**2));pred=float(np.trace(c)/d)
                local.append(dict(base_seed=baseseed,motion_uncertainty_lr=sigma,estimator=name,mse=mse,predicted_mse=pred,ratio=mse/pred,trials=mc))
    # Exact counterexample: distinct HR candidates with identical LR observations.
    a=shifted(base,grid,FREQUENCIES).reshape(-1,d)
    _,s,vh=la.svd(a,full_matrices=False);v=vh[-1];delta=.04*v
    theta=np.random.default_rng(7003).normal(0,.01,d)
    hr1=(.5+bhr@(theta-delta)).reshape(N,N);hr2=(.5+bhr@(theta+delta)).reshape(N,N)
    lr1=(.5+a@(theta-delta)).reshape(FRAMES,LOW,LOW);lr2=(.5+a@(theta+delta)).reshape(FRAMES,LOW,LOW)
    assert np.min(hr1)>0 and np.max(hr1)<1 and np.min(hr2)>0 and np.max(hr2)<1
    null=dict(hr_rmse=float(np.sqrt(np.mean((hr1-hr2)**2))),lr_max_difference=float(np.max(np.abs(lr1-lr2))),
        quantized_lr_equal=bool(np.array_equal(np.rint(lr1*255),np.rint(lr2*255))),smallest_singular_value=float(s[-1]))
    fig,axes=plt.subplots(1,3,figsize=(9,3),constrained_layout=True)
    for ax,img,title in zip(axes,[hr1,hr2,lr1[0]],['HR candidate A','HR candidate B','Identical LR (all 16 phases)']):
        ax.imshow(img,cmap='gray',vmin=0,vmax=1,interpolation='nearest');ax.set_title(title,fontsize=10);ax.axis('off')
    fig.savefig(out/'nullspace_counterexample.png',dpi=180);plt.close(fig)
    np.savez_compressed(out/'nullspace_arrays.npz',hr_a=hr1,hr_b=hr2,lr_a=lr1,lr_b=lr2,delta=delta)
    # Bootstrap main experiment by seed, averaging the paired noise replicates first.
    main_path=out.parent/'observability_synthetic_v1/rows.json';mainrows=json.loads(main_path.read_text());main_paired=[]
    for kernel in ['project_bicubic_aa','pixel_box','ideal_lowpass']:
        xs=[r for r in mainrows if r['kernel']==kernel and r['missing']=='none' and r['motion_error_lr']==0 and r['schedule']!='integer_motion']
        rows=[]
        for schedule in shifts:
            for seed in range(32):
                sub=[r for r in xs if r['schedule']==schedule and r['seed']==seed]
                rows.append(dict(schedule=schedule,seed=seed,hr_psnr=float(np.mean([r['hr_psnr'] for r in sub]))))
        main_paired.append(dict(kernel=kernel,**paired(rows,'repeated','complementary')))
    result=dict(completed=True,script_sha256=sha(__file__),protocol_sha256=sha(out/'protocol.json'),
        quantized=qsum,quantized_rows=quantrows,noise_sweep=noise_sum,alias_mismatch=alias_summary,alias_rows=aliasrows,
        local_multiple_points=local,nullspace=null,main_paired_bootstrap=main_paired,elapsed_seconds=time.monotonic()-start,
        limitations=['32 generated texture seeds do not represent 32 real scenes.',
        'Quantization residual is signal-dependent and temporally correlated; uniform-noise fitting is only a fixed regularizer.',
        'Shared-alias EXTRA correction is supplementary; original orthogonal EXTRA control remains unchanged.',
        'Local nuisance estimator uses privileged texture Jacobians and known uncertainty distribution.'])
    save(out/'metrics.json',result);save(out/'completion.json',dict(completed=True,metrics_sha256=sha(out/'metrics.json'),elapsed_seconds=result['elapsed_seconds']))
    print(json.dumps(dict(completed=True,elapsed_seconds=result['elapsed_seconds'],nullspace=null)))

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
