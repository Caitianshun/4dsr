"""Three-metric paired endpoints; no quality threshold controls dispatch."""
import csv
import statistics
from dv_common import *

METRICS=('psnr','ssim','lpips')
DIRECTION={'psnr':1,'ssim':1,'lpips':-1}
ARMS=('J_joint','A_sh','S_cov')
PAIRS=(('A_sh','J_joint'),('S_cov','J_joint'),('S_cov','A_sh'))

def csvwrite(path,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

def vector(endpoint):
    return {k:statistics.mean(endpoint['cameras'][c][k] for c in ['cam00','cam01']) for k in METRICS}

def dominates(a,b):
    diff=[DIRECTION[k]*(a[k]-b[k]) for k in METRICS]
    return all(x>=0 for x in diff) and any(x>0 for x in diff)

def pareto(vectors):
    return [a for a,v in vectors.items() if not any(a!=b and dominates(w,v) for b,w in vectors.items())]

def classify(deltas):
    values=[[DIRECTION[k]*r[k] for k in METRICS] for r in deltas]
    consistent={k:all(DIRECTION[k]*r[k]>=0 for r in deltas) or all(DIRECTION[k]*r[k]<=0 for r in deltas) for k in METRICS}
    if all(all(x>=0 for x in r) and any(x>0 for x in r) for r in values):name='repeat_supported_three_metric_noninferiority'
    elif all(all(x<=0 for x in r) for r in values):name='no_three_metric_benefit'
    elif all(consistent.values()):name='repeat_supported_quality_tradeoff'
    else:name='unstable_or_insufficient_evidence'
    if len(deltas)<2:name='insufficient_completed_repeats'
    return dict(classification=name,direction_consistent=consistent,
        practical_value_or_significance_not_established=True)

def summarize():
    p=require_run_root(OUT);base=read(bound(p['baseline_reuse']['endpoint']))
    data={'U6000':base};missing=[]
    for t in p['task_plan']:
        label=t['task_id'];f=OUT/'evaluation'/label/'complete.json'
        if not f.exists():missing.append(label);continue
        c=read(f);ep=OUT/'evaluation'/label/'endpoint.json';assert sha(ep)==c['endpoint_sha256']
        data[label]=read(ep)
    main=[];frame=[];camera=[];paired=[];aux=[]
    baseline_dir=local(p['baseline_reuse']['directory'])
    for label,e in data.items():
        repeat='baseline' if label=='U6000' else label.split('_',1)[0][1:]
        arm='U6000' if label=='U6000' else label.split('_',1)[1]
        for scope,v in list(e['cameras'].items())+[('equal_camera_mean',vector(e))]:
            main.append(dict(endpoint=label,repeat=repeat,arm=arm,scope=scope,**{k:v[k] for k in METRICS}))
        directory=baseline_dir if label=='U6000' else OUT/'evaluation'/label
        for filename,dest in [('metrics_per_frame.csv',frame),('metrics_per_camera.csv',camera)]:
            with (directory/filename).open() as f:dest.extend(csv.DictReader(f))
    vectors={};paired_repeat={}
    for repeat in ['1','2']:
        labels=[f'r{repeat}_{a}' for a in ARMS]
        if not all(a in data for a in labels):continue
        vectors[repeat]={a:vector(data[f'r{repeat}_{a}']) for a in ARMS}
        for candidate,reference in [*PAIRS,*[(a,'U6000') for a in ARMS]]:
            a=data[f'r{repeat}_{candidate}'];b=base if reference=='U6000' else data[f'r{repeat}_{reference}']
            for scope in ['cam00','cam01','equal_camera_mean']:
                x=vector(a) if scope=='equal_camera_mean' else a['cameras'][scope]
                y=vector(b) if scope=='equal_camera_mean' else b['cameras'][scope]
                row=dict(repeat=repeat,candidate=candidate,reference=reference,scope=scope)
                for k in METRICS:
                    row[k+'_candidate']=x[k];row[k+'_reference']=y[k];row[k+'_delta']=x[k]-y[k]
                row['lpips_denominator']=y['lpips'];row['lpips_relative']=None if y['lpips']==0 else (x['lpips']-y['lpips'])/y['lpips'];paired.append(row)
                if scope=='equal_camera_mean':paired_repeat.setdefault(candidate+'-minus-'+reference,[]).append({k:x[k]-y[k] for k in METRICS})
            scopes={'train76_lr':(a['train76_lr'],b['train76_lr'])}
            for c in ['cam00','cam01']:
                for k in ['dynamic_lpips','temporal','temporal_full']:scopes[c+'.'+k]=(a['cameras'][c][k],b['cameras'][c][k])
            for c in p['evaluation']['train_cameras']:
                for k in ['psnr','ssim','lpips','lr_l1']:scopes[c+'.'+k]=(a['train'][c][k],b['train'][c][k])
            for k,(x,y) in scopes.items():aux.append(dict(repeat=repeat,candidate=candidate,reference=reference,metric=k,candidate_value=x,reference_value=y,delta=x-y,denominator=y,relative=None if y==0 or k.endswith('.psnr') else (x-y)/y,role='descriptive'))
    statistics_by_pair={}
    for pair,diffs in paired_repeat.items():
        statistics_by_pair[pair]=dict(repeats=len(diffs),**classify(diffs),
            delta={k:dict(mean=statistics.mean(x[k] for x in diffs),minimum=min(x[k] for x in diffs),maximum=max(x[k] for x in diffs)) for k in METRICS})
    mean_vectors={a:{k:statistics.mean(vectors[r][a][k] for r in ['1','2']) for k in METRICS} for a in ARMS} if len(vectors)==2 else {}
    per_set={r:dict(vectors=v,pareto=pareto(v),dominance={a:[b for b in v if a!=b and dominates(v[a],v[b])] for a in v}) for r,v in vectors.items()}
    summary=dict(execution=dict(completed_formal_endpoints=len(data)-1,expected=6,missing=missing,complete=not missing),
        primary_metrics=list(METRICS),baseline=vector(base),paired=statistics_by_pair,per_repeat=per_set,
        mean_vectors=mean_vectors,pareto_mean=pareto(mean_vectors) if mean_vectors else None,
        numerical_replay=dict(status='warning',does_not_gate_quality=True,source=p['inherited_evidence']['replay_followup']),
        interpretation='Two paired suffixes share U6000; no significance test; raw direction and amplitude, not thresholded pass/fail.',
        quality_controls_execution=False,protocol_sha256=sha(OUT/'protocol.json'))
    for file,rows in [('main_quality.csv',main),('paired_quality.csv',paired),('auxiliary_changes.csv',aux),('metrics_per_frame.csv',frame),('metrics_per_camera.csv',camera)]:csvwrite(OUT/file,rows)
    write(OUT/'quality_summary.json',summary)
    return summary

if __name__=='__main__':print(json.dumps(summarize(),ensure_ascii=False))
