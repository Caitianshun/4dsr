"""Actual multi-frame LR inverse problems; no HR is used until evaluation.

Fixed old LR-only cells, sparse dense-flow image formation, full AA support,
nonperiodic enlarged HR latent, normalized reference/repeat/multiframe controls.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import platform
import socket
import time
import argparse

import cv2
import numpy as np
from PIL import Image, ImageDraw
import scipy
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, cg
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[2]
PARENT=ROOT/'output/dynamic_sr_20260920/observability_real_v1'
DEFAULT_OUT=ROOT/'output/dynamic_sr_20260921/real_lr_local_inverse_v1'
SCENES={'cook_spinach':'n3dv_prepared/cook_spinach','meetroom_discussion':'meetroom_prepared/discussion'}
OFFSETS=[-8,-2,2,8]
FLOW_ARGS=(.5,4,21,5,7,1.5,0)
N=96
SOURCE_N=80
LAMBDAS=[.01,.1,1.]
RIDGE=1e-5


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False))
def ten(x):return torch.from_numpy(np.ascontiguousarray(x.transpose(2,0,1))).unsqueeze(0)
def rgb(p):return np.asarray(Image.open(p).convert('RGB'),dtype=np.float64)/255.
def resize(x,size,aa=False):return F.interpolate(ten(x),size=size,mode='bicubic',align_corners=False,antialias=aa)[0].permute(1,2,0).numpy()


def choose(parent):
    chosen=[]
    for scene in SCENES:
        candidates=sorted([{k:c[k] for k in ['scene','camera','anchor','x_lr','y_lr','accepted_neighbors','n_observations','phase_dispersion','estimated_displacement_median']} for c in parent['cells'] if c['scene']==scene and c['moving_proxy'] and c['phase_rich']],key=lambda c:(c['camera'],c['anchor'],c['y_lr'],c['x_lr']))
        # Image dimensions and latent boundary checks use only LR metadata.
        dims=(252,336) if scene=='cook_spinach' else (180,320)
        candidates=[c for c in candidates if 14<=c['x_lr']<dims[1]-14 and 14<=c['y_lr']<dims[0]-14]
        groups=sorted(set((c['camera'],c['anchor']) for c in candidates))
        got=[]
        while len(got)<8:
            progress=False
            for group in groups:
                pool=[c for c in candidates if (c['camera'],c['anchor'])==group and c not in got]
                # Within a camera, repeat rounds must be >=24 LR pixels from
                # previously chosen positions, including other times.
                if not pool:continue
                spatial=[c for c in pool if all(c['camera']!=p['camera'] or np.hypot(c['x_lr']-p['x_lr'],c['y_lr']-p['y_lr'])>=24 for p in got)]
                if not spatial:
                    # First pass covers every camera/time even if the object
                    # remains at a nearby position; report these correlations.
                    spatial=pool if not any((p['camera'],p['anchor'])==group for p in got) else []
                if not spatial:continue
                got.append(spatial[0]);progress=True
                if len(got)==8:break
            if not progress:break
        assert len(got)==8,(scene,len(got))
        chosen+=got
    return chosen


def aa_matrix():
    # Exactly torch's one-dimensional antialiased bicubic weights, then kron.
    eye=torch.eye(SOURCE_N,dtype=torch.float64).reshape(SOURCE_N,1,1,SOURCE_N)
    weights=F.interpolate(eye,size=(1,SOURCE_N//4),mode='bicubic',align_corners=False,antialias=True)[:,0,0,:].T.numpy()
    one=sp.csr_matrix(weights[2:-2])
    return sp.kron(one,one,format='csr')


def warp_matrix(coords):
    x,y=coords[...,0].ravel(),coords[...,1].ravel()
    ix,iy=np.floor(x).astype(int),np.floor(y).astype(int)
    dx,dy=x-ix,y-iy
    rows=[];cols=[];vals=[]
    valid=np.ones(len(x),dtype=bool)
    for ox,oy,w in [(0,0,(1-dx)*(1-dy)),(1,0,dx*(1-dy)),(0,1,(1-dx)*dy),(1,1,dx*dy)]:
        good=(ix+ox>=0)&(ix+ox<N)&(iy+oy>=0)&(iy+oy<N)
        valid&=good|(np.abs(w)<1e-14)
        take=good&(np.abs(w)>1e-14)
        rows.append(np.flatnonzero(take));cols.append((iy[take]+oy)*N+ix[take]+ox);vals.append(w[take])
    return sp.csr_matrix((np.concatenate(vals),(np.concatenate(rows),np.concatenate(cols))),shape=(SOURCE_N**2,N*N)),valid


def direct_forward(x,coords):
    grid=torch.from_numpy(2*(coords+.5)/N-1).unsqueeze(0)
    warped=F.grid_sample(ten(x),grid,mode='bilinear',padding_mode='zeros',align_corners=False)
    return F.interpolate(warped,size=(SOURCE_N//4,SOURCE_N//4),mode='bicubic',align_corners=False,antialias=True)[0,:,2:-2,2:-2].permute(1,2,0).numpy().reshape(-1,3)


def self_tests(d):
    rng=np.random.default_rng(2112)
    random=rng.normal(size=(N,N,3))
    yy,xx=np.mgrid[:SOURCE_N,:SOURCE_N]
    checks={}
    for name,(sx,sy) in {'identity':(0.,0.),'fractional':(.37,-.21),'integer':(4.,-8.)}.items():
        coords=np.stack([xx+8-sx,yy+8-sy],axis=-1).astype(float)
        w,v=warp_matrix(coords);a=d@w
        valid=np.asarray(abs(d)@(~v).astype(float)).ravel()<1e-14
        pred=a@random.reshape(-1,3);exact=direct_forward(random,coords)
        err=float(np.max(np.abs(pred[valid]-exact[valid])))
        assert err<2e-12,(name,err)
        ramp=np.stack(np.meshgrid(np.arange(N),np.arange(N)),axis=-1)
        ramp=np.concatenate([ramp,np.ones((N,N,1))],axis=-1).astype(float)
        z=a@ramp.reshape(-1,3)
        outy,outx=np.mgrid[2:18,2:18]
        expected=np.stack([outx*4+1.5+8-sx,outy*4+1.5+8-sy,np.ones_like(outx)],axis=-1).reshape(-1,3)
        signerr=float(np.max(np.abs(z[valid]-expected[valid])))
        assert signerr<2e-12,(name,signerr)
        checks[name]=dict(random_rgb_forward_max_abs=err,ramp_shift_sign_max_abs=signerr)
    return checks


def reg_matrix():
    idx=np.arange(N*N).reshape(N,N)
    p=np.concatenate([idx[:,:-1].ravel(),idx[:-1,:].ravel()])
    q=np.concatenate([idx[:,1:].ravel(),idx[1:,:].ravel()])
    rows=np.arange(len(p))
    l=sp.csr_matrix((np.concatenate([np.ones(len(p)),-np.ones(len(p))]),(np.concatenate([rows,rows]),np.concatenate([p,q]))),shape=(len(p),N*N))
    return l,(l.T@l).tocsr()/len(p)


def solve(a,y,b,reg,lam):
    m=a.shape[0]
    # All valid scalar pixel observations have equal weight, total one.
    at=a.T.tocsr()
    diag=np.asarray(a.multiply(a).sum(axis=0)).ravel()/m+lam*reg.diagonal()+RIDGE/(N*N)
    op=LinearOperator((N*N,N*N),matvec=lambda x:at@(a@x)/m+lam*(reg@x)+RIDGE*x/(N*N))
    pre=LinearOperator((N*N,N*N),matvec=lambda x:x/diag)
    rhs=at@y/m+RIDGE*b.reshape(-1,3)/(N*N)
    out=[];checks=[]
    for c in range(3):
        it=[0]
        def callback(x):it[0]+=1
        x,info=cg(op,rhs[:,c],x0=b.reshape(-1,3)[:,c],rtol=1e-11,atol=1e-15,maxiter=4000,M=pre,callback=callback)
        rel=float(np.linalg.norm(op@x-rhs[:,c])/max(np.linalg.norm(rhs[:,c]),1e-30))
        assert info==0 and rel<2e-11,(info,rel,it)
        out.append(x);checks.append(dict(iterations=it[0],relative_normal_equation_residual=rel,info=int(info)))
    return np.stack(out,axis=-1).reshape(N,N,3),checks


def make_operator(d,back,old_valid,origin_ref,center_neigh,shape,strict_footprint=False):
    h,w=shape
    nx,ny=center_neigh
    ox,oy=nx-10,ny-10
    assert ox>=0 and oy>=0 and ox+20<=w and oy+20<=h
    # LR index u maps to HR index 4*(u+.5)-.5. Displacements scale by 4.
    if back is None:
        bh=np.zeros((SOURCE_N,SOURCE_N,2),dtype=np.float64)
    else:
        tb=torch.from_numpy(back.astype(np.float64).transpose(2,0,1)).unsqueeze(0)
        bh=F.interpolate(tb,size=(h*4,w*4),mode='bilinear',align_corners=False)[0,:,oy*4:(oy+20)*4,ox*4:(ox+20)*4].permute(1,2,0).numpy()*4
    yy,xx=np.mgrid[:SOURCE_N,:SOURCE_N]
    global_src=np.stack([xx+ox*4,yy+oy*4],axis=-1)
    coords=global_src+bh-np.array(origin_ref)[None,None,:]
    warp,inside=warp_matrix(coords)
    a=(d@warp).tocsr()
    keep=np.asarray(abs(d)@(~inside).astype(float)).ravel()<1e-14
    ys,xs=np.mgrid[oy+2:oy+18,ox+2:ox+18]
    if back is not None:
        bl=back[ys,xs]
        rx=(xs+bl[...,0]).astype(np.float32);ry=(ys+bl[...,1]).astype(np.float32)
        validity=cv2.remap(old_valid.astype(np.float32),rx,ry,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
        keep&=(validity.ravel()>.999)
        if strict_footprint:
            ref_lr=(coords+np.array(origin_ref)[None,None,:]+.5)/4-.5
            support_valid=cv2.remap(old_valid.astype(np.float32),ref_lr[...,0].astype(np.float32),ref_lr[...,1].astype(np.float32),cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)>.999
            keep&=np.asarray(abs(d)@(~support_valid.ravel()).astype(float)).ravel()<1e-14
    rng=np.random.default_rng(4931)
    probe=rng.normal(size=(N,N,3))
    parity=float(np.max(np.abs((a@probe.reshape(-1,3)-direct_forward(probe,coords))[keep])))
    assert parity<3e-12,parity
    rowsum=np.asarray(a.sum(axis=1)).ravel()
    assert np.max(np.abs(rowsum[keep]-1))<2e-12
    return a[keep],ys.ravel()[keep],xs.ravel()[keep],dict(parity_max_abs=parity,source_origin_lr=[ox,oy],total_candidate_rows=256,accepted_rows=int(keep.sum()),outside_or_invalid_rows=int((~keep).sum())),coords


def measure(p,g):
    raw=p.copy();p=np.clip(p,0,1)
    sl=np.s_[24:72,24:72]
    x,y=p[sl],g[sl]
    mse=float(np.mean((x-y)**2));blur=lambda z:cv2.GaussianBlur(z,(11,11),1.5)
    mx,my=blur(x),blur(y);vx,vy=blur(x*x)-mx*mx,blur(y*y)-my*my;cov=blur(x*y)-mx*my
    ss=((2*mx*my+.01**2)*(2*cov+.03**2))/((mx*mx+my*my+.01**2)*(vx+vy+.03**2))
    hp=(p-resize(resize(p,(24,24),True),(96,96)))[sl]
    hg=(g-resize(resize(g,(24,24),True),(96,96)))[sl]
    hm=float(np.mean((hp-hg)**2));he=float(np.mean(hg*hg))
    return dict(psnr=float(-10*np.log10(max(mse,1e-30))),mse=mse,ssim=float(ss[5:-5,5:-5].mean()),high_residual_mse=hm,reference_high_energy=he,high_residual_error_ratio=hm/max(he,1e-30),clip_fraction=float(((raw[sl]<0)|(raw[sl]>1)).mean()),unclipped_mse=float(np.mean((raw[sl]-y)**2)))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=DEFAULT_OUT);parser.add_argument('--test-only',action='store_true');parser.add_argument('--strict-footprint',action='store_true');args=parser.parse_args()
    start=time.monotonic();started=datetime.now(timezone.utc).isoformat()
    torch.set_num_threads(4);cv2.setNumThreads(2)
    d=aa_matrix();checks=self_tests(d)
    if args.test_only:print(json.dumps(checks,indent=2));return
    out=args.out;out.mkdir(parents=True,exist_ok=False);(out/'source.py').write_text(Path(__file__).read_text())
    parent=json.loads((PARENT/'metrics.json').read_text());chosen=choose(parent)
    inputs={str(PARENT/'metrics.json'):sha(PARENT/'metrics.json')}
    protocol=dict(selection='fixed parent LR-only moving and phase-rich cells; round robin camera/time; scan y,x; >=24px camera-level spatial separation for additional rounds',chosen=chosen,lambdas=LAMBDAS,ridge=RIDGE,latent_hr=[N,N],source_hr=[SOURCE_N,SOURCE_N],evaluate_hr=[48,48],observed_lr_region=[16,16],warp='LR-estimated Farneback backward flow, torch bilinear upsample align_corners=False, HR bilinear grid_sample equivalent',degradation='torch bicubic antialias=True align_corners=False x4, exact extracted weights, no periodic boundary',invalid='discard LR rows whose complete AA+warp support exits latent patch or whose reverse-mapped old strict LR validity <.999',objective='mean(all valid RGB-pixel squared residuals per color) + lambda mean(nonperiodic adjacent HR differences squared) + 1e-5 mean((x-bicubic)^2)',repeat='repeat full reference rows as many times as actual observation count; total data normalized by valid row count',strict_all_aa_support=args.strict_footprint,evaluation_only_hr=True,lpips='not reported for 48x48 tiny crops; no full-frame perceptual claim',stopping='all three fixed lambdas reported; no HR-based tuning',backbone='unchanged 4DGaussians; this is image-local input diagnostic, not new-view reconstruction')
    dump(out/'protocol_before_hr.json',protocol)
    _,reg=reg_matrix();allpred=[];records=[];frame_cache={}
    for ci,c in enumerate(chosen):
        data=ROOT/'data/dynamic_sr'/SCENES[c['scene']]
        manifestp=data/'manifest.json';manifest=json.loads(manifestp.read_text());inputs[str(manifestp)]=sha(manifestp)
        obs={(o['camera_id'],o['frame_index']):o for o in manifest['observations']}
        key=(c['scene'],c['camera'],c['anchor'])
        if key not in frame_cache:
            frames=[c['anchor']]+[c['anchor']+k for k in OFFSETS];lrs={}
            for f in frames:
                ob=obs[(c['camera'],f)];p=data/ob['lr_path'];got=sha(p);assert got==ob['lr_sha256'];inputs[str(p)]=got;lrs[f]=rgb(p)
            gray={f:cv2.cvtColor(np.round(x*255).astype(np.uint8),cv2.COLOR_RGB2GRAY) for f,x in lrs.items()}
            backs=[cv2.calcOpticalFlowFarneback(gray[c['anchor']+off],gray[c['anchor']],None,*FLOW_ARGS) for off in OFFSETS]
            statep=PARENT/f'{c["scene"]}_{c["camera"]}_{c["anchor"]:04d}_lr_correspondence.npz';inputs[str(statep)]=sha(statep);state=np.load(statep)
            frame_cache[key]=(lrs,backs,state)
        lrs,backs,state=frame_cache[key];ref=lrs[c['anchor']];h,w=ref.shape[:2]
        x,y=c['x_lr'],c['y_lr'];origin=(x*4-48,y*4-48)
        b=resize(ref,(h*4,w*4))[origin[1]:origin[1]+N,origin[0]:origin[0]+N]
        aa=[];yy=[];opmeta=[];allcoords=[]
        indexes=[None]+c['accepted_neighbors']
        for ni in indexes:
            f=c['anchor'] if ni is None else c['anchor']+OFFSETS[ni]
            fl=np.zeros(2) if ni is None else state['flow'][ni,y,x]
            center=(int(round(x+fl[0])),int(round(y+fl[1])))
            a,ys,xs,meta,coords=make_operator(d,None if ni is None else backs[ni],None if ni is None else state['valid'][ni],origin,center,(h,w),args.strict_footprint)
            assert a.shape[0]>=1,(ci,f,a.shape)
            aa.append(a);yy.append(lrs[f][ys,xs]);meta.update(frame=f,estimated_center_forward_lr=fl.tolist());opmeta.append(meta);allcoords.append(coords)
        multi=sp.vstack(aa,format='csr');target=np.concatenate(yy);a0,y0=aa[0],yy[0]
        repeat=sp.vstack([a0]*len(aa),format='csr');repeat_y=np.tile(y0,(len(aa),1))
        # Algebraic and solved repeated-input negative control at all lambdas.
        gramdiff=(repeat.T@repeat/repeat.shape[0]-a0.T@a0/a0.shape[0]).tocsr()
        grammax=float(np.max(np.abs(gramdiff.data))) if gramdiff.nnz else 0.
        rhsmax=float(np.max(np.abs(repeat.T@repeat_y/repeat.shape[0]-a0.T@y0/a0.shape[0])))
        assert grammax<1e-15 and rhsmax<1e-15
        predictions={'bicubic':b};solver={};repeat_checks={}
        for lam in LAMBDAS:
            p0,s0=solve(a0,y0,b,reg,lam);pm,sm=solve(multi,target,b,reg,lam);pr,sr=solve(repeat,repeat_y,b,reg,lam)
            rdiff=float(np.max(np.abs(p0-pr)));assert rdiff<3e-7,(ci,lam,rdiff)
            predictions[f'reference_lambda_{lam:g}']=p0;predictions[f'multiframe_lambda_{lam:g}']=pm
            predictions[f'repeat_lambda_{lam:g}']=pr
            solver[str(lam)]={'reference':s0,'multiframe':sm,'repeat':sr};repeat_checks[str(lam)]=rdiff
        rec=dict(index=ci,cell=c,operators=opmeta,source_frames=[o['frame'] for o in opmeta],total_actual_rows=int(multi.shape[0]),reference_rows=int(a0.shape[0]),unique_actual_frames=len(aa),repeat_normalized_gram_max_abs=grammax,repeat_normalized_rhs_max_abs=rhsmax,repeat_solution_max_abs=repeat_checks,solver=solver,lr_fit={m:dict(reference_mse=float(np.mean((a0@p.reshape(-1,3)-y0)**2)),multiframe_mse=float(np.mean((multi@p.reshape(-1,3)-target)**2))) for m,p in predictions.items()})
        path=out/f'patch_{ci:02d}_solved_before_hr.npz';np.savez_compressed(path,**predictions);rec['prediction_sha256_before_hr']=sha(path)
        # Preserve sparse operators for independent audits and HR-only later oracle residuals.
        sp.save_npz(out/f'patch_{ci:02d}_operator.npz',multi);np.savez_compressed(out/f'patch_{ci:02d}_observations.npz',y=target)
        dump(out/f'patch_{ci:02d}_record_before_hr.json',rec);allpred.append(predictions);records.append(rec)
        print(json.dumps(dict(stage='solved_before_hr',index=ci,cell=c,valid_rows=multi.shape[0],frames=len(aa),elapsed=time.monotonic()-start)),flush=True)
    dump(out/'all_predictions_frozen_before_hr.json',dict(utc=datetime.now(timezone.utc).isoformat(),predictions=[r['prediction_sha256_before_hr'] for r in records],protocol_sha256=sha(out/'protocol_before_hr.json')))
    # ONLY NOW read HR. No parameter, mask, position, stopping, or method choices follow.
    targets=[];rows=[]
    for ci,(c,preds,rec) in enumerate(zip(chosen,allpred,records)):
        data=ROOT/'data/dynamic_sr'/SCENES[c['scene']];man=json.loads((data/'manifest.json').read_text());ob=next(o for o in man['observations'] if o['camera_id']==c['camera'] and o['frame_index']==c['anchor'])
        hp=data/ob['hr_path'];assert sha(hp)==ob['hr_sha256'];inputs[str(hp)]=sha(hp)
        x,y=c['x_lr']*4,c['y_lr']*4;gt=rgb(hp)[y-48:y+48,x-48:x+48];targets.append(gt)
        teacher=data/f'sr_swinir_x4/{c["camera"]}/{c["anchor"]:04d}.png'
        if teacher.exists():inputs[str(teacher)]=sha(teacher);preds['swinir_existing']=rgb(teacher)[y-48:y+48,x-48:x+48]
        a=sp.load_npz(out/f'patch_{ci:02d}_operator.npz');yy=np.load(out/f'patch_{ci:02d}_observations.npz')['y'];err=a@gt.reshape(-1,3)-yy
        startrow=0;oracle=[]
        for meta in rec['operators']:
            e=err[startrow:startrow+meta['accepted_rows']];oracle.append(dict(frame=meta['frame'],rmse=float(np.sqrt(np.mean(e*e))),mae=float(np.mean(np.abs(e)))));startrow+=meta['accepted_rows']
        rows.append(dict(index=ci,cell=c,methods={m:measure(p,gt) for m,p in preds.items()},reference_hr_forward_only_evaluation=oracle))
        np.savez_compressed(out/f'patch_{ci:02d}_evaluation.npz',hr_evaluation_only=gt,**preds)
    summary={}
    for scene in SCENES:
        rr=[r for r in rows if r['cell']['scene']==scene];names=sorted(set(m for r in rr for m in r['methods']))
        summary[scene]={m:dict(n=sum(m in r['methods'] for r in rr),**{k:float(np.mean([r['methods'][m][k] for r in rr if m in r['methods']])) for k in ['psnr','ssim','mse','high_residual_mse','high_residual_error_ratio','clip_fraction']}) for m in names}
    panels=['bicubic','reference_lambda_0.1','multiframe_lambda_0.1','swinir_existing']
    canvas=Image.new('RGB',(5*192,len(chosen)*212),'white');draw=ImageDraw.Draw(canvas)
    for ci,(gt,preds,c) in enumerate(zip(targets,allpred,chosen)):
        vals=[('HR evaluation',gt)]+[(m,preds.get(m)) for m in panels]
        for j,(name,img) in enumerate(vals):
            draw.text((j*192+2,ci*212+1),name,fill='black')
            if img is not None:canvas.paste(Image.fromarray(np.round(np.clip(img[24:72,24:72],0,1)*255).astype(np.uint8)).resize((192,192),Image.Resampling.NEAREST),(j*192,ci*212+17))
        draw.text((0,ci*212+198),f'{ci}: {c["scene"]} {c["camera"]} t={c["anchor"]} ({c["x_lr"]},{c["y_lr"]})',fill='red')
    canvas.save(out/'all_fixed_patches_middle_lambda.png')
    identity=dict(started_utc=started,finished_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-start,host=socket.gethostname(),platform=platform.platform(),python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,torch=torch.__version__,opencv=cv2.__version__,gpu_used=False,source_sha256=sha(__file__))
    result=dict(protocol=protocol,identity=identity,self_tests=checks,input_sha256=inputs,records=records,rows=rows,summary=summary)
    dump(out/'metrics.json',result);dump(out/'complete.json',dict(metrics_sha256=sha(out/'metrics.json'),source_sha256=sha(__file__),finished_utc=identity['finished_utc']))
    print(json.dumps(dict(identity=identity,summary=summary),indent=2),flush=True)

if __name__=='__main__':main()
