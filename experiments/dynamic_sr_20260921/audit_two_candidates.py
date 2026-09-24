"""Independent CPU audit of saved candidate diagnostics; never edits inputs."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.environ.setdefault('OMP_NUM_THREADS','2')
import argparse, hashlib, json, math, time
from pathlib import Path
import numpy as np
import scipy.linalg as la
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[2]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def close(a,b,tol=1e-10):
    e=float(np.max(np.abs(np.asarray(a)-np.asarray(b))))
    assert e<tol,(e,tol)
    return e

def direct_images(images,coords,rows):
    x=torch.as_tensor(np.ascontiguousarray(images),dtype=torch.float64)[:,None]
    grid=torch.as_tensor(2*(coords+.5)/96-1,dtype=torch.float64)
    if grid.ndim==3:grid=grid[None].expand(len(x),-1,-1,-1)
    y=F.grid_sample(x,grid,align_corners=False,mode='bilinear',padding_mode='zeros')
    y=F.interpolate(y,size=(20,20),mode='bicubic',align_corners=False,antialias=True)
    return y[:,0,2:18,2:18].reshape(len(x),-1)[:,rows].numpy()

def candidate1(path):
    metrics=read(path/'metrics.json');receipt=read(path/'all_scores_frozen_before_hr.json')
    assert sha(path/'source.py')==receipt['source_sha256']
    assert sha(path/'protocol_before_hr.md')==receipt['protocol_sha256']
    assert sha(path/'controls_before_real_hr.json')==receipt['controls_sha256']
    gram_error=0.;energy_error=0.;basis_error=0.;project_error=0.;min_eig=1.;n=0
    jd_error=0.;jn_error=0.;extra_hashes={}
    rng=np.random.default_rng(298121)
    for i,r in enumerate(metrics['records']):
        p=path/f'patch_{i:02d}_frozen_before_hr.npz';assert sha(p)==receipt['files'][i]
        z=np.load(p);b=z['B'];a=z['Jd'];known=a.T@a
        extra_hashes[str(path/f'patch_{i:02d}_scores_before_hr.json')]=sha(path/f'patch_{i:02d}_scores_before_hr.json')
        basis_error=max(basis_error,close(b.T@b/2304,np.eye(32),1e-12))
        cf=b.T@(z['teacher']-z['base']).ravel()/2304/.01
        project_error=max(project_error,close(cf,z['teacher_coeff'],1e-12))
        cursor=0
        for k in range(len(r['operators'])):
            coords=z[f'coords_{k}'];rows=z[f'rows_{k}'].astype(int);length=len(rows)
            pred=direct_images(b.T.reshape(32,96,96),coords,rows).T*.01
            jd_error=max(jd_error,close(pred,a[cursor:cursor+length],1e-11))
            if k:
                yy,xx=np.mgrid[:80,:80];u=(xx-39.5)/40;v=(yy-39.5)/40
                fields=np.zeros((6,80,80,2))
                fields[0,:,:,0]=4;fields[1,:,:,1]=4
                fields[2,:,:,0]=4*u;fields[3,:,:,0]=4*v
                fields[4,:,:,1]=4*u;fields[5,:,:,1]=4*v
                for base,key in [('base','Jn'),('teacher','Jn_teacher')]:
                    images=np.repeat(z[base][None],6,axis=0)
                    plus=direct_images(images,coords[None]+1e-4*fields,rows)
                    minus=direct_images(images,coords[None]-1e-4*fields,rows)
                    columns=np.c_[((plus-minus)/2e-4).T,direct_images(z[base][None],coords,rows)[0],np.ones(length)]
                    recorded=z[key][cursor:cursor+length,(k-1)*8:k*8]
                    jn_error=max(jn_error,close(columns,recorded,2e-10))
            cursor+=length
        for jkey,prefix,records in [('Jn','H_',r['schurs']),('Jn_teacher','H_teacher_',r['teacher_linearization_schurs'])]:
            jn=z[jkey]
            for s in records:
                sigma=s['noise_std'];motion=s['motion_std_lr']
                std=np.tile([motion]*6+[.1*motion,.05*motion],jn.shape[1]//8)
                # Eliminate nuisance through an augmented least-squares QR,
                # independently of the implementation's normal-equation Schur.
                nwhite=jn*std/sigma
                augn=np.vstack([nwhite,np.eye(jn.shape[1])])
                augd=np.vstack([a/sigma,np.zeros((jn.shape[1],a.shape[1]))])
                coeff=la.lstsq(augn,augd,lapack_driver='gelsy')[0]
                residual=augd-augn@coeff;hh=residual.T@residual
                saved=z[f'{prefix}{sigma:.8f}_{motion:.2f}']
                gram_error=max(gram_error,close(hh,saved,2e-8))
                v=rng.normal(size=32);v/=np.linalg.norm(v)
                profile=la.lstsq(augn,augd@v,lapack_driver='gelsy')[0]
                objective=float(np.sum((augd@v-augn@profile)**2))
                energy_error=max(energy_error,abs(objective-float(v@saved@v)))
                assert abs(objective-float(v@saved@v))<2e-8
                eig=np.linalg.eigvalsh(saved);removed=np.linalg.eigvalsh(known/sigma**2-saved)
                min_eig=min(min_eig,float(eig.min()),float(removed.min()));assert min_eig>-2e-8
                close(np.trace(saved),s['trace_effective'],2e-8)
                n+=1
    dependency=ROOT/'experiments/dynamic_sr_20260921/real_lr_local_inverse.py'
    extra_hashes[str(dependency)]=sha(dependency)
    return dict(n_patches=len(metrics['records']),n_matrices=n,
      augmented_qr_gram_max_abs=gram_error,profile_energy_max_abs=energy_error,
      basis_orthonormal_max_abs=basis_error,teacher_projection_max_abs=project_error,
      independent_torch_Jd_max_abs=jd_error,independent_torch_Jn_max_abs=jn_error,
      minimum_effective_or_removed_eigenvalue=min_eig,receipt_hashes_pass=True,
      posthoc_audit_hash_receipt=extra_hashes,
      metrics_sha256=sha(path/'metrics.json'))

def candidate2(path):
    m=read(path/'metrics.json');lock=read(path/'lr_scores_locked.json');cfg=m['config'];protocol=m['protocol']
    assert sha(path/'lr_scores_locked.json')==m['locked_lr_sha256']
    assert sha(path/'source.py')==cfg['source_sha256']
    assert sha(path/'protocol.json')==lock['protocol_sha256']
    assert lock['optimizer_unchanged'] and m['optimizer_unchanged']
    assert m['final_parameter_hash']==cfg['initial_parameter_hash']
    assert all(v['role'] in ['actual_lr','frozen_sr_teacher'] for v in lock['inputs'].values())
    for file,value in read(path/'inputs.json').items():assert sha(file)==value['sha256']
    maxscore=0.;maxmetric=0.;maxnorm=0.;maxgroup=0.
    byid={}
    for r,old in zip(m['rows'],lock['rows']):
        assert all(r[k]==v for k,v in old.items())
        byid.setdefault(r['proposal_id'],[]).append(r)
        for name,keys in r['B'].items():
            assert len(keys)==2 and r['A'] not in keys
            assert all(k.split('/')[0] not in ['cam00','cam01'] for k in keys)
            s=sum(r['lr_before'][k]['mse']-r['lr_after'][k]['mse'] for k in keys)/2
            maxscore=max(maxscore,close(s,r['scores'][name],1e-12))
        maxscore=max(maxscore,close(-r['scale']*r['A_LR_derivative'],r['scores']['A_derivative'],1e-12))
        maxscore=max(maxscore,close(r['lr_before'][r['A']]['mse']-r['lr_after'][r['A']]['mse'],r['scores']['A_actual'],1e-12))
        if 'A_train_L1_actual' in r['scores']:
            maxscore=max(maxscore,close(r['lr_before'][r['A']]['l1']-r['lr_after'][r['A']]['l1'],r['scores']['A_train_L1_actual'],1e-12))
            maxscore=max(maxscore,close(-r['scale']*r['A_LR_train_L1_derivative'],r['scores']['A_train_L1_derivative'],1e-12))
        for name,score in r['scores'].items():
            tol=r.get('score_thresholds',{}).get(name,r['derivative_threshold'] if name=='A_derivative' else r['score_threshold'])
            state='improve' if score>tol else 'worsen' if score < -tol else 'unresolved'
            assert r['gate_state'][name]==state
        proposal=next(p for p in protocol['proposals'] if p['id']==r['proposal_id'])
        for kind,ks in [('novel',proposal['evaluation_novel']),('train_cross',proposal['evaluation_train'])]:
            for metric,value in r['eval_delta'][kind].items():
                v=sum(r['hr_after'][k][metric]-r['hr_before'][k][metric] for k in ks)/len(ks)
                maxmetric=max(maxmetric,close(v,value,1e-12))
    for pid,rows in byid.items():
        p=path/'directions'/f'{pid}.pt';assert sha(p)==lock['direction_hashes'][pid]
        values=torch.load(p,map_location='cpu',weights_only=True)
        n2=sum(float(v.double().square().sum()) for v in values.values())
        norm=math.sqrt(n2);maxnorm=max(maxnorm,close(norm,rows[0]['direction_l2'],1e-8))
        for gr,stat in zip(cfg['groups'],rows[0]['groupstats']):
            assert gr['name']==stat['name']
            count=sum(values[k].numel() for k in gr['names'])
            rms=math.sqrt(sum(float(values[k].double().square().sum()) for k in gr['names'])/count)
            maxgroup=max(maxgroup,close(rms,stat['direction_rms'],1e-10))
        for r in rows:maxnorm=max(maxnorm,close(r['scale']*norm,r['actual_step_l2'],1e-8))
    for scale,summary in m['summary'].items():
        rows=[r for r in m['rows'] if str(r['scale'])==scale]
        for target in ['novel','train_cross']:
            for gate,statistics in summary[target]['gates'].items():
                take=[r for r in rows if r['scores'][gate]>r['score_thresholds'][gate]]
                assert len(take)==statistics['accepted']
                if take:
                    close(sum(r['eval_delta'][target]['psnr']<-.001 for r in take)/len(take),statistics['psnr_harm_rate'],1e-12)
                for metric,value in statistics['gated_mean_delta_all_proposals'].items():
                    close(sum(r['eval_delta'][target][metric] for r in take)/len(rows),value,1e-12)
    return dict(n_proposals=len(byid),n_steps=len(m['rows']),
      saved_direction_l2_max_abs=maxnorm,saved_group_rms_max_abs=maxgroup,
      lr_scores_max_abs=maxscore,hr_metric_delta_max_abs=maxmetric,
      frozen_lr_rows_unchanged=True,parameter_rollback_hash_pass=True,
      optimizer_state_hash_pass=True,directions_hash_pass=True,
      no_hr_file_in_lr_receipt=True,all_input_hashes_pass=True,summary_gate_counts_and_means_pass=True,
      source_hash_pass=True,metrics_sha256=sha(path/'metrics.json'))

def supplement(path):
    m=read(path/'metrics.json');lock=read(path/'lr_locked.json');p=m['protocol']
    assert sha(path/'lr_locked.json')==m['lock_sha256']
    assert sha(path/'source.py')==lock['source_sha256']
    assert all(v['role'] in ['actual_lr','frozen_sr_teacher'] for v in lock['inputs'].values())
    assert m['rollback_exact'] and m['optimizer_unchanged']
    for file,value in m['inputs'].items():assert sha(file)==value['sha256']
    rows=[r for r in m['rows'] if r['scale']==.005];n=len(rows)
    assert all(r['accepted'][k]==(r['scores'][k]>1e-8) for r in m['rows'] for k in ['spread8','random8'])
    for q in p['proposals']:
        assert len(set(q['extra_B']['spread8']))==8 and len(set(q['extra_B']['random8']))==8
        assert q['A'] not in q['extra_B']['spread8']+q['extra_B']['random8']
        assert all(x.split('/')[0] not in ['cam00','cam01'] for x in q['extra_B']['spread8']+q['extra_B']['random8'])
    for gate,b in m['budget'].items():
        selection=lock['gates'][gate]
        total=sum(r['direction_l2'] for r in rows)
        accepted=sum(r['direction_l2'] for r in rows if selection[r['proposal_id']])
        alpha=.005*accepted/total
        close(alpha,b['alpha'],1e-14);close(.005*accepted,b['gate_total_l2'],1e-12)
        close(alpha*total,b['uniform_total_l2'],1e-12)
        for group in ['novel','train_cross']:
            for metric in ['psnr','ssim','lpips','hf_mse']:
                v=sum(r['variants']['primary']['delta'][group][metric] for r in m['hr_rows'] if selection[r['proposal_id']])/n
                close(v,b['results'][group]['gate'][metric],1e-12)
                v=sum(r['variants'][f'uniform_{gate}']['delta'][group][metric] for r in m['hr_rows'])/n
                close(v,b['results'][group]['uniform'][metric],1e-12)
    return dict(n_proposals=n,source_and_lock_hashes_pass=True,no_hr_in_lr_lock=True,all_input_hashes_pass=True,
      distinct_train_observations_pass=True,lr_acceptance_threshold_pass=True,
      total_parameter_path_length_budget_pass=True,budget_hr_summary_pass=True,
      primary_cross2_lr_replay_max_abs=m['parent_cross_score_replay_max_abs_error'],
      limitation='Supplement does not save every LR before/after scalar; audit verifies thresholds and budgets, not independently rerender all extra-B predictions.',
      metrics_sha256=sha(path/'metrics.json'))

def main():
    p=argparse.ArgumentParser();p.add_argument('--candidate1',type=Path);p.add_argument('--candidate2',type=Path,nargs='*',default=[]);p.add_argument('--supplement',type=Path,nargs='*',default=[]);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(2);start=time.monotonic();result=dict(device='CPU',source_sha256=sha(__file__))
    if a.candidate1:result['candidate1']=candidate1(a.candidate1)
    result['candidate2']={str(x):candidate2(x) for x in a.candidate2}
    result['supplement']={str(x):supplement(x) for x in a.supplement}
    result['elapsed_seconds']=time.monotonic()-start
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
