"""Endpoint-only three-metric comparison, signed reference gaps and real costs."""
from dv_common import *
import csv
import statistics

METRICS=['psnr','ssim','lpips']
def csvwrite(path,rows):
    path=Path(path);keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(rows)

def main():
    p=require_run_root(OUT)
    old=list(csv.DictReader((ROOT/'output/dynamic_sr_temporal_prior_20260928/main_quality.csv').open()))
    for r in old:
        for k in METRICS:r[k]=float(r[k])
    rows=old.copy();costs=[]
    for task in p['task_plan']:
        label=task['task_id'];folder=OUT/'evaluation'/label
        receipt=read(folder/'complete.json');assert receipt['checkpoint_sha256']==sha(OUT/'runs'/label/'attempt_01/train/checkpoint_12000.pt')
        endpoint=read(folder/'endpoint.json')
        for camera,q in endpoint['cameras'].items():
            rows.append(dict(endpoint=label,repeat=task['repeat'],arm='T',scope=camera,**{k:q[k] for k in METRICS}))
        rows.append(dict(endpoint=label,repeat=task['repeat'],arm='T',scope='equal_camera_mean',
            **{k:statistics.mean(endpoint['cameras'][c][k] for c in ['cam00','cam01']) for k in METRICS}))
        run=read(OUT/'runs'/label/'attempt_01/train/complete.json')
        costs.append(dict(task_id=label,training_gpu=task['physical_gpu'],training_host=p['training']['host'],
            **{k:run[k] for k in ['actual_updates','adam_calls','training_rgb_forwards','train_s','wall_s','peak_gb']},
            evaluation_seconds=receipt['seconds']+receipt.get('reused_completed_seconds',0),
            recovery_evaluation_seconds=receipt['seconds'],
            completed_evaluation_rgb_forwards=receipt['rgb_forwards'],
            recovery_new_rgb_forwards=receipt.get('new_rgb_forwards',receipt['rgb_forwards']),
            reused_rgb_forwards=receipt.get('reused_rgb_forwards',0),
            points=p['training']['points'],extra_parameters=0))
    def select(arm,scope):return [r for r in rows if r['arm']==arm and r['scope']==scope]
    averages=[]
    for arm in dict.fromkeys(r['arm'] for r in rows):
        for scope in ['cam00','cam01','equal_camera_mean']:
            rr=select(arm,scope)
            if rr:averages.append(dict(arm=arm,scope=scope,repeat_count=len(rr),**{k:statistics.mean(r[k] for r in rr) for k in METRICS}))
    comparisons=[]
    references=['LR-direct-HRrender','LR-direct-Bicubic','HR-direct-6k','J1','Async2','Sync2']
    for target in [r for r in averages if r['arm']=='T']:
        for ref in references:
            rr=[r for r in averages if r['arm']==ref and r['scope']==target['scope']]
            if rr:comparisons.append(dict(method='T',endpoint='T_mean',repeat='mean',reference=ref,scope=target['scope'],**{'delta_'+k:target[k]-rr[0][k] for k in METRICS}))
    for target in [r for r in rows if r['arm']=='T']:
        for ref in references:
            rr=[r for r in rows if r['arm']==ref and r['scope']==target['scope'] and r['repeat'] in [target['repeat'],'reference']]
            if len(rr)==1:comparisons.append(dict(method='T',endpoint=target['endpoint'],repeat=target['repeat'],reference=ref,scope=target['scope'],**{'delta_'+k:target[k]-rr[0][k] for k in METRICS}))
    paired=[]
    for rep in ['1','2']:
        for scope in ['cam00','cam01','equal_camera_mean']:
            a=next(r for r in rows if r['arm']=='T' and r['repeat']==rep and r['scope']==scope)
            b=next(r for r in rows if r['arm']=='J1' and r['repeat']==rep and r['scope']==scope)
            paired.append(dict(repeat=rep,scope=scope,**{'delta_'+k:a[k]-b[k] for k in METRICS}))
    meanT=next(r for r in averages if r['arm']=='T' and r['scope']=='equal_camera_mean')
    dominated=[]
    for r in averages:
        if r['scope']!='equal_camera_mean' or r['arm']=='T':continue
        if r['psnr']>=meanT['psnr'] and r['ssim']>=meanT['ssim'] and r['lpips']<=meanT['lpips'] and any(r[k]!=meanT[k] for k in METRICS):dominated.append(r['arm'])
    csvwrite(OUT/'main_quality.csv',rows);csvwrite(OUT/'mean_quality.csv',averages)
    csvwrite(OUT/'quality_gaps.csv',comparisons);csvwrite(OUT/'paired_deltas.csv',paired);csvwrite(OUT/'costs.csv',costs)
    budget=read(OUT/'budget.json');assert budget['effective_updates']==12000
    result=dict(status='completed',metrics=METRICS,averages=averages,reference_gaps=comparisons,
        paired_deltas=paired,costs=costs,dominated_by=dominated,budget=budget,
        protocol=entry(OUT/'protocol.json'),decision=entry(OUT/'diagnosis_decision.json'),
        evidence_level='two paired suffixes share U6000; development cameras; no independent-seed significance',
        diagnostic_evidence=read(OUT/'diagnosis_decision.json')['evidence'],
        primary_aggregation='mean of per-frame metric per camera; cameras equally then suffixes equally',
        baseline_budget_caution='LR/HR direct 7000 total, U6000 14200 total, J1/T endpoint 20200 total; counts differ for native LR/HR',
        training_hardware='A100-SXM4-40GB both suffixes, possibly different physical units; historical J1 A100, final evaluation same local RTX3090')
    write(OUT/'final_summary.json',result)
    write(OUT/'final_integrity.json',dict(status='completed',effective_updates=12000,
        actual_updates=budget['actual_updates'],checkpoint_index=entry(OUT/'checkpoint_index.json'),
        summary=entry(OUT/'final_summary.json'),all_results_returned_locally=True,chat_notification_sent=False))
    print(json.dumps(dict(status='completed',T=meanT,dominated_by=dominated)),flush=True)

if __name__=='__main__':main()
