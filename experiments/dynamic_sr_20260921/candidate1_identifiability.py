"""Bounded local detail/nuisance identifiability diagnostic; HR only at end."""
from pathlib import Path
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.environ.setdefault('OMP_NUM_THREADS','2')
import hashlib,json,time,socket,platform,importlib.util
from datetime import datetime,timezone
import numpy as np
import scipy.sparse as sp
from scipy.stats import spearmanr
import cv2,torch

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('inv',Path(__file__).with_name('real_lr_local_inverse.py')); inv=importlib.util.module_from_spec(spec);spec.loader.exec_module(inv)
PARENT=ROOT/'output/dynamic_sr_20260921/real_lr_local_inverse_strict_footprint_v1'
OUT=ROOT/'output/dynamic_sr_20260921/candidate1_identifiability_v1'
PROTOCOL=ROOT/'docs/dynamic_sr_candidate1_protocol_2026-09-21.md'
MOTION=[.05,.15,.30];NOISE=[1/255,2/255,4/255];DETAIL=.01;EPS=1e-4
LUMA=np.array([.2126,.7152,.0722])

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False,default=lambda v:v.item() if isinstance(v,np.generic) else v))
def gray(x):return x@LUMA

def basis():
    y,x=np.mgrid[:48,:48];cols=[];labels=[]
    for k in [4,8,12,16,20,24,28,32]:
        for kx,ky in [(k,0),(0,k),(k,k),(k,2)]:
            a=np.cos(np.pi*kx*(x+.5)/48)*np.cos(np.pi*ky*(y+.5)/48);a/=np.sqrt(np.mean(a*a))
            b=np.zeros((96,96));b[24:72,24:72]=a;cols.append(b.ravel());labels.append([kx,ky])
    b=np.stack(cols,axis=1);assert np.max(abs(b.T@b/(48*48)-np.eye(32)))<1e-14
    return b,labels


def forward(x,coords,d,rows):return (d@inv.warp_matrix(coords)[0]@x.ravel())[rows]
def jacn(x,coords,d,rows):
    yy,xx=np.mgrid[:80,:80];xx=(xx-39.5)/40;yy=(yy-39.5)/40
    fields=[]
    for axis,f in [(0,np.ones_like(xx)),(1,np.ones_like(xx)),(0,xx),(0,yy),(1,xx),(1,yy)]:
        a=np.zeros_like(coords);a[...,axis]=4*f;fields.append(a)
    cols=[];errs=[]
    for f in fields:
        plus=forward(x,coords+EPS*f,d,rows);minus=forward(x,coords-EPS*f,d,rows);v=(plus-minus)/(2*EPS)
        small=(forward(x,coords+EPS*.5*f,d,rows)-forward(x,coords-EPS*.5*f,d,rows))/EPS
        cols.append(v);errs.append(float(np.linalg.norm(v-small)/max(np.linalg.norm(v),1e-14)))
    value=forward(x,coords,d,rows)
    cols.extend([value,np.ones(len(rows))])
    # A finite perturbation checks the local linear model, separately from derivative convergence.
    direction=np.array([.31,-.23,.17,-.13,.11,.19])
    field=sum(a*v for a,v in zip(fields,direction));step=.01
    pred=np.stack(cols[:6],1)@direction*step
    exact=forward(x,coords+step*field,d,rows)-value
    relative=float(np.linalg.norm(exact-pred)/max(np.linalg.norm(exact),1e-14))
    return np.stack(cols,1),dict(fd_half_step_relative=errs,finite_motion_001_lr_relative=relative),fields


def schur(jd,jn,sigma,motion):
    scales=np.tile([motion]*6+[motion*.1,motion*.05],jn.shape[1]//8)
    a=jd/sigma;n=jn*scales[None,:]/sigma
    h=a.T@a;c=a.T@n;g=np.eye(n.shape[1])+n.T@n
    removed=c@np.linalg.solve(g,c.T);eff=(h-removed);eff=(eff+eff.T)/2
    eig=np.linalg.eigvalsh(eff);orig=np.linalg.eigvalsh(h)
    assert eig.min()>-1e-8 and np.linalg.eigvalsh(h-eff).min()>-1e-8
    # Reparameterization: Jn -> Jn*S, prior std -> std/S must preserve information.
    reparam=np.geomspace(.2,5,jn.shape[1]) if jn.shape[1] else np.zeros(0)
    nn=(jn*reparam[None,:])*(scales/reparam)[None,:]/sigma
    hh=h-a.T@nn@np.linalg.solve(np.eye(nn.shape[1])+nn.T@nn,nn.T@a)
    scaleerr=float(np.max(np.abs(hh-eff)));assert scaleerr<1e-8
    summary=dict(trace_known=float(np.trace(h)),trace_effective=float(np.trace(eff)),trace_retained=float(np.trace(eff)/max(np.trace(h),1e-30)),df_known=float(np.sum(orig/(1+orig))),df_effective=float(np.sum(eig/(1+eig))),modes_gt_one_known=int((orig>1).sum()),modes_gt_one_effective=int((eig>1).sum()),eigen_min=float(eig.min()),scale_invariance_max_abs=scaleerr,diagonal_known=np.diag(h).tolist(),diagonal_effective=np.diag(eff).tolist(),eigenvalues_known=orig.tolist(),eigenvalues_effective=eig.tolist())
    return h,eff,n,g,summary


def profiles(residual,n,g,sigma):
    r=residual/sigma;fit=np.linalg.solve(g,n.T@r);remaining=r-n@fit
    return dict(raw_mse=float(np.mean(residual**2)),profile_penalized_per_row=float((remaining@remaining+fit@fit)/len(r)),profile_residual_mse=float(np.mean((remaining*sigma)**2)),nuisance_prior_squared=float(fit@fit))


def corr(x,y):
    if np.std(x)<1e-15 or np.std(y)<1e-15:return None
    return float(spearmanr(x,y).statistic)


def controlled(b,d,B):
    rng=np.random.default_rng(21901);coefs=rng.normal(size=(32,32));coefs/=np.linalg.norm(coefs,axis=0,keepdims=True)
    noise=rng.normal(size=(16*256,32))*NOISE[0]
    yy,xx=np.mgrid[:80,:80];base=np.stack([xx+8,yy+8],-1).astype(float)
    shifts=[(sx,sy) for sy in range(4) for sx in range(4)]
    results=[];matrices={}
    for name in ['repeat','complement','complement_wrong_015','complement_static_half','complement_moving_half']:
        aa=[];at=[];ns=[];ys=[];noise_rows=[]
        for fi,(sx,sy) in enumerate(shifts):
            true=base.copy() if name=='repeat' else base+np.array([sx,sy])
            assumed=true.copy()
            if name=='complement_wrong_015' and fi:
                assumed+=.6*np.array([np.cos(fi),np.sin(fi)])
            rows=np.arange(256)
            if 'half' in name:
                iy,ix=np.mgrid[:16,:16];mask=(ix+iy+(fi%2 if 'moving' in name else 0))%2==0;rows=rows[mask.ravel()]
            a=(d@inv.warp_matrix(assumed)[0])[rows];t=(d@inv.warp_matrix(true)[0])[rows]
            aa.append(a);at.append(t);noise_rows.extend((fi*256+rows).tolist())
            if fi:
                j,_,_=jacn(b,assumed,d,rows);ns.append(sp.csr_matrix(j))
        a=sp.vstack(aa);t=sp.vstack(at);jd=np.asarray(a@B)*DETAIL
        jn=sp.block_diag([sp.csr_matrix((len(aa[0].toarray()),0))]+ns,format='csr').toarray()
        y=np.asarray(t@b.ravel())[:,None]+(t@B*DETAIL)@coefs+noise[noise_rows]
        r=y-np.asarray(a@b.ravel())[:,None]
        cases=[]
        for m in MOTION:
            h,e,n,g,s=schur(jd,jn,NOISE[0],m)
            raw=jd.T@r/NOISE[0]**2;effrhs=raw-(jd/NOISE[0]).T@n@np.linalg.solve(g,n.T@(r/NOISE[0]))
            known=np.linalg.solve(h+np.eye(32),raw);unknown=np.linalg.solve(e+np.eye(32),effrhs)
            cases.append(dict(motion_std_lr=m,**s,known_coeff_mse=float(np.mean((known-coefs)**2)),effective_coeff_mse=float(np.mean((unknown-coefs)**2))))
        # Counterexample holds Jd/Jn fixed. Correct vs sign-flipped known detail.
        gt=y[:,0]-noise[noise_rows,0]
        correct=np.asarray(t@b.ravel())+(t@B*DETAIL)@coefs[:,0]
        wrong=np.asarray(t@b.ravel())-(t@B*DETAIL)@coefs[:,0]
        results.append(dict(name=name,frames=16,rows=int(a.shape[0]),cases=cases,same_support_teacher_counterexample=dict(information_identical_by_fixed_linearization=True,correct_lr_mse=float(np.mean((correct-gt)**2)),wrong_lr_mse=float(np.mean((wrong-gt)**2)),hr_detail_wrong_mse=float(np.mean((DETAIL*B@(2*coefs[:,0]))[np.flatnonzero(np.any(B!=0,axis=1))]**2)))))
        matrices[name]=dict(Jd=jd,Jn=jn)
    return results


def main():
    started=time.monotonic();torch.set_num_threads(2);cv2.setNumThreads(2)
    OUT.mkdir(parents=True,exist_ok=False);(OUT/'source.py').write_text(Path(__file__).read_text());(OUT/'protocol_before_hr.md').write_text(PROTOCOL.read_text())
    B,labels=basis();d=inv.aa_matrix();checks=inv.self_tests(d);records=[];cache={};inputs={str(PROTOCOL):sha(PROTOCOL),str(PARENT/'protocol_before_hr.json'):sha(PARENT/'protocol_before_hr.json')}
    chosen=json.loads((PARENT/'protocol_before_hr.json').read_text())['chosen'];allfrozen=[]
    for i,c in enumerate(chosen):
        data=ROOT/'data/dynamic_sr'/inv.SCENES[c['scene']];mp=data/'manifest.json';man=json.loads(mp.read_text());inputs[str(mp)]=sha(mp);obs={(o['camera_id'],o['frame_index']):o for o in man['observations']}
        key=(c['scene'],c['camera'],c['anchor'])
        if key not in cache:
            frames=[c['anchor']]+[c['anchor']+k for k in inv.OFFSETS];lrs={}
            for f in frames:
                p=data/obs[c['camera'],f]['lr_path'];inputs[str(p)]=sha(p);assert inputs[str(p)]==obs[c['camera'],f]['lr_sha256'];lrs[f]=inv.rgb(p)
            grays={f:cv2.cvtColor(np.round(v*255).astype(np.uint8),cv2.COLOR_RGB2GRAY) for f,v in lrs.items()}
            back=[cv2.calcOpticalFlowFarneback(grays[c['anchor']+off],grays[c['anchor']],None,*inv.FLOW_ARGS) for off in inv.OFFSETS]
            p=inv.PARENT/f'{c["scene"]}_{c["camera"]}_{c["anchor"]:04d}_lr_correspondence.npz';inputs[str(p)]=sha(p);state=np.load(p);cache[key]=(lrs,back,state)
        lrs,backs,state=cache[key];ref=lrs[c['anchor']];h,w=ref.shape[:2];x,y=c['x_lr'],c['y_lr'];origin=(x*4-48,y*4-48)
        b=gray(inv.resize(ref,(h*4,w*4))[origin[1]:origin[1]+96,origin[0]:origin[0]+96])
        if c['scene']=='meetroom_discussion' and c['camera']=='cam06':tp=ROOT/f'output/dynamic_sr_20260921/local_inverse_teacher_v1/cam06_{c["anchor"]:04d}.png'
        else:tp=data/f'sr_swinir_x4/{c["camera"]}/{c["anchor"]:04d}.png'
        inputs[str(tp)]=sha(tp);teacher=gray(inv.rgb(tp)[origin[1]:origin[1]+96,origin[0]:origin[0]+96])
        aa=[];target=[];nblocks=[];ntblocks=[];fd=[];photob=[];fb=[];dis=[];opmeta=[];coordstore={}
        for ni in [None]+c['accepted_neighbors']:
            f=c['anchor'] if ni is None else c['anchor']+inv.OFFSETS[ni];fl=np.zeros(2) if ni is None else state['flow'][ni,y,x];center=tuple(np.round(np.array([x,y])+fl).astype(int))
            a,ys,xs,meta,coords=inv.make_operator(d,None if ni is None else backs[ni],None if ni is None else state['valid'][ni],origin,center,(h,w),True)
            ox,oy=meta['source_origin_lr'];rows=(ys-oy-2)*16+(xs-ox-2)
            aa.append(a);target.append(gray(lrs[f][ys,xs]));meta.update(frame=f);opmeta.append(meta);coordstore[f'coords_{len(aa)-1}']=coords;coordstore[f'rows_{len(aa)-1}']=rows
            if ni is None:
                nblocks.append(sp.csr_matrix((a.shape[0],0)));ntblocks.append(sp.csr_matrix((a.shape[0],0)))
            else:
                j,test,_=jacn(b,coords,d,rows);nblocks.append(sp.csr_matrix(j));fd.append(test)
                jt,_,_=jacn(teacher,coords,d,rows);ntblocks.append(sp.csr_matrix(jt))
                sl=np.s_[y-6:y+6,x-6:x+6];valid=state['valid'][ni][sl];fb.extend(state['fb'][ni][sl][valid].tolist());dis.extend(state['estimator_disagreement'][ni][sl][valid].tolist())
        A=sp.vstack(aa,format='csr');Y=np.concatenate(target);Jn=sp.block_diag(nblocks,format='csr').toarray();Jn_teacher=sp.block_diag(ntblocks,format='csr').toarray();Jd=np.asarray(A@B)*DETAIL
        old=sp.load_npz(PARENT/f'patch_{i:02d}_operator.npz');delta=(A-old).tocsr();assert delta.nnz==0 or max(abs(delta.data))<3e-12
        oldy=np.load(PARENT/f'patch_{i:02d}_observations.npz')['y'];assert np.max(abs(Y-gray(oldy)))<1e-14
        deriv=B.T@(teacher-b).ravel()/(48*48)/DETAIL
        residual_b=A@b.ravel()-Y;residual_t=A@teacher.ravel()-Y
        schurs=[];teacher_schurs=[];stored=dict(B=B,Jd=Jd,Jn=Jn,Jn_teacher=Jn_teacher,base=b,teacher=teacher,target=Y,teacher_coeff=deriv,**coordstore)
        for sigma in NOISE:
            for m in MOTION:
                hk,he,n,g,s=schur(Jd,Jn,sigma,m)
                sb=profiles(residual_b,n,g,sigma);st=profiles(residual_t,n,g,sigma)
                t=deriv;detail_support_known=float(t@hk@t/max(t@t,1e-30));detail_support_eff=float(t@he@t/max(t@t,1e-30))
                schurs.append(dict(noise_std=sigma,motion_std_lr=m,**s,bicubic_profile=sb,teacher_profile=st,teacher_direction_known=detail_support_known,teacher_direction_effective=detail_support_eff))
                stored[f'H_{sigma:.8f}_{m:.2f}']=he
                _,et,_,_,ts=schur(Jd,Jn_teacher,sigma,m);teacher_schurs.append(dict(noise_std=sigma,motion_std_lr=m,**ts));stored[f'H_teacher_{sigma:.8f}_{m:.2f}']=et
        patch_gray=gray(ref)[y-6:y+6,x-6:x+6];gx=cv2.Sobel(patch_gray,cv2.CV_64F,1,0,ksize=3);gy=cv2.Sobel(patch_gray,cv2.CV_64F,0,1,ksize=3)
        structure=float(np.linalg.eigvalsh(np.stack([gx.ravel(),gy.ravel()])@np.stack([gx.ravel(),gy.ravel()]).T/len(gx.ravel())).min())
        record=dict(index=i,cell=c,n_rows=int(A.shape[0]),n_nuisance=Jn.shape[1],basis_labels=labels,operators=opmeta,fd_checks=fd,baselines=dict(fb_mean=float(np.mean(fb)),disagreement_mean=float(np.mean(dis)),flow_confidence=-float(np.mean(fb))/.15-float(np.mean(dis))/.25,lr_structure_min_eigen=structure,bicubic_lr_mse=float(np.mean(residual_b**2)),teacher_lr_mse=float(np.mean(residual_t**2))),schurs=schurs,teacher_linearization_schurs=teacher_schurs)
        np.savez_compressed(OUT/f'patch_{i:02d}_frozen_before_hr.npz',**stored);dump(OUT/f'patch_{i:02d}_scores_before_hr.json',record);records.append(record);allfrozen.append(sha(OUT/f'patch_{i:02d}_frozen_before_hr.npz'))
        if i==0:control_base=b.copy()
        print(json.dumps(dict(stage='frozen',patch=i,rows=A.shape[0],elapsed=time.monotonic()-started)),flush=True)
    controls=controlled(control_base,d,B);dump(OUT/'controls_before_real_hr.json',controls)
    dump(OUT/'all_scores_frozen_before_hr.json',dict(utc=datetime.now(timezone.utc).isoformat(),source_sha256=sha(__file__),protocol_sha256=sha(PROTOCOL),files=allfrozen,controls_sha256=sha(OUT/'controls_before_real_hr.json')))
    # HR is first read below, after all real-data scores, nuisance settings and controls are frozen.
    for i,r in enumerate(records):
        c=r['cell'];z=np.load(OUT/f'patch_{i:02d}_frozen_before_hr.npz');p=PARENT/f'patch_{i:02d}_evaluation.npz';inputs[str(p)]=sha(p);gt=gray(np.load(p)['hr_evaluation_only']);base=z['base'];t=z['teacher']
        err=B.T@(t-gt).ravel()/(48*48)/DETAIL;gtcoef=B.T@(gt-base).ravel()/(48*48)/DETAIL
        sol=np.load(PARENT/f'patch_{i:02d}_solved_before_hr.npz');one=gray(sol['reference_lambda_0.1']);multi=gray(sol['multiframe_lambda_0.1']);oe=B.T@(one-gt).ravel()/(48*48)/DETAIL;me=B.T@(multi-gt).ravel()/(48*48)/DETAIL
        r['hr_evaluation_only']=dict(teacher_luma_mse=float(np.mean((t[24:72,24:72]-gt[24:72,24:72])**2)),bicubic_luma_mse=float(np.mean((base[24:72,24:72]-gt[24:72,24:72])**2)),teacher_direction_squared_error=(err**2).tolist(),teacher_direction_error_mean=float(np.mean(err**2)),gt_detail_energy=float(np.sum(gtcoef**2)),single_direction_mse=float(np.mean(oe**2)),multi_direction_mse=float(np.mean(me**2)),multi_minus_single_direction_mse=float(np.mean(me**2-oe**2)))
        for s in r['schurs']+r['teacher_linearization_schurs']:
            s['within_patch_rho_known_vs_teacher_error']=corr(s['diagonal_known'],err**2);s['within_patch_rho_effective_vs_teacher_error']=corr(s['diagonal_effective'],err**2)
    summaries={}
    for scene in list(inv.SCENES)+['all']:
        rr=[r for r in records if scene=='all' or r['cell']['scene']==scene];baseline={}
        targets={'teacher_luma_mse':[r['hr_evaluation_only']['teacher_luma_mse'] for r in rr],'teacher_direction_error_mean':[r['hr_evaluation_only']['teacher_direction_error_mean'] for r in rr],'multi_minus_single_direction_mse':[r['hr_evaluation_only']['multi_minus_single_direction_mse'] for r in rr]}
        for key in rr[0]['baselines']:baseline[key]={k:corr([r['baselines'][key] for r in rr],v) for k,v in targets.items()}
        cases=[]
        for si in range(9):
            ss=[r['schurs'][si] for r in rr];keys=['trace_known','trace_effective','trace_retained','df_known','df_effective','teacher_direction_known','teacher_direction_effective']
            correlations={k:{tk:corr([s[k] for s in ss],tv) for tk,tv in targets.items()} for k in keys}
            for k in ['raw_mse','profile_penalized_per_row','profile_residual_mse']:correlations['teacher_'+k]={tk:corr([s['teacher_profile'][k] for s in ss],tv) for tk,tv in targets.items()}
            cases.append(dict(noise_std=ss[0]['noise_std'],motion_std_lr=ss[0]['motion_std_lr'],means={k:float(np.mean([s[k] for s in ss])) for k in keys},within_patch_rho_known_mean=float(np.mean([s['within_patch_rho_known_vs_teacher_error'] for s in ss])),within_patch_rho_effective_mean=float(np.mean([s['within_patch_rho_effective_vs_teacher_error'] for s in ss])),rho=correlations))
        # Descriptive group-size aware aggregation; no p-values for correlated patch modes.
        clusters={}
        for r in rr:clusters.setdefault((r['cell']['scene'],r['cell']['camera'],r['cell']['anchor']),[]).append(r)
        cluster_rows=[dict(group=list(g),n=len(v),error=float(np.mean([x['hr_evaluation_only']['teacher_direction_error_mean'] for x in v])),known=float(np.mean([x['schurs'][1]['trace_known'] for x in v])),effective=float(np.mean([x['schurs'][1]['trace_effective'] for x in v])),flow=float(np.mean([x['baselines']['flow_confidence'] for x in v]))) for g,v in clusters.items()]
        teacher_cases=[]
        for si in range(9):
            ss=[r['teacher_linearization_schurs'][si] for r in rr];teacher_cases.append(dict(noise_std=ss[0]['noise_std'],motion_std_lr=ss[0]['motion_std_lr'],means={k:float(np.mean([s[k] for s in ss])) for k in ['trace_known','trace_effective','trace_retained','df_known','df_effective']},rho_teacher_coeff_error=corr([s['trace_effective'] for s in ss],targets['teacher_direction_error_mean'])))
        summaries[scene]=dict(teacher_linearization_cases=teacher_cases,n=len(rr),clusters=cluster_rows,cluster_rho={k:corr([z[k] for z in cluster_rows],[z['error'] for z in cluster_rows]) for k in ['known','effective','flow']},baselines_rho=baseline,cases=cases)
    payload=dict(identity=dict(host=socket.gethostname(),platform=platform.platform(),elapsed_seconds=time.monotonic()-started,source_sha256=sha(__file__),protocol_sha256=sha(PROTOCOL),device='CPU'),input_sha256=inputs,checks=checks,controls=controls,records=records,summary=summaries,limitations=['Local image-space diagnostic, not GS parameter projection or training improvement','Fixed LR-correspondence geometry, visibility and bicubic base; uncalibrated Gaussian nuisance/noise sensitivity','16 correlated chosen moving phase-rich patches, 12 scene-camera-time clusters','Information does not certify frozen teacher correctness; HR only endpoint evaluation','32 luminance modes are an explicit subspace, not full recoverability'])
    dump(OUT/'metrics.json',payload);dump(OUT/'complete.json',dict(status='complete',metrics_sha256=sha(OUT/'metrics.json'),source_sha256=sha(__file__),elapsed_seconds=time.monotonic()-started));print(json.dumps(dict(stage='complete',elapsed=time.monotonic()-started)),flush=True)
if __name__=='__main__':main()
