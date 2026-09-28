"""Complete camera/suffix tables and fixed-reference gaps, with no automatic quality veto."""
import csv
import statistics
from dv_common import *

METRICS=('psnr','ssim','lpips')
ARMS=('J1','Async2','Sync2','Repeat7','Video7')

def csvwrite(path,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

def vector(e):return {k:statistics.mean(e['cameras'][c][k] for c in ['cam00','cam01']) for k in METRICS}

def compare(data):
    rows=[]
    for label,e in data.items():
        if 'direct' in label:continue
        repeat=label[1] if label.startswith('r') else None
        references=['LR-direct-HRrender','HR-direct-6k']
        if repeat:references.extend([f'r{repeat}_J1',f'r{repeat}_Async2'])
        if label.endswith('Video7'):references.append(f'r{repeat}_Repeat7')
        for ref in references:
            if ref==label:continue
            for scope in ['cam00','cam01','equal_camera_mean']:
                row=dict(endpoint=label,repeat=repeat,reference=ref,scope=scope,definition='method minus reference',status='available' if ref in data else 'unavailable')
                x=vector(e) if scope=='equal_camera_mean' else e['cameras'][scope]
                y=None if ref not in data else vector(data[ref]) if scope=='equal_camera_mean' else data[ref]['cameras'][scope]
                for k in METRICS:
                    row[k]=x[k];row[k+'_reference']=None if y is None else y[k];row[k+'_delta']=None if y is None else x[k]-y[k]
                row['lpips_relative_reduction_percent']=None if y is None or y['lpips']==0 else 100*(y['lpips']-x['lpips'])/y['lpips']
                rows.append(row)
    return rows

def summarize():
    p=require_run_root(OUT);data={'U6000':read(bound(p['baseline_reuse']['endpoint']))}
    directories={'U6000':local(p['baseline_reuse']['directory'])};frame=[];missing=[]
    for r,ref in p['historical_J1'].items():
        label=f'r{r}_J1';data[label]=read(bound(ref['endpoint']));directories[label]=local(ref['directory'])
    for label,ref in p['historical_multiview'].items():
        data[label]=read(bound(ref['endpoint']));directories[label]=local(ref['directory'])
    for t in p['task_plan']:
        label=t['task_id'];directory=OUT/'evaluation'/label
        if not (directory/'complete.json').exists():missing.append(label);continue
        receipt=read(directory/'complete.json');assert receipt['endpoint_sha256']==sha(directory/'endpoint.json')
        assert receipt['protocol_sha256']==sha(OUT/'protocol.json')
        data[label]=read(directory/'endpoint.json');directories[label]=directory
    refpath=OUT/'references/complete.json'
    if refpath.exists():
        refs=read(refpath)['rows'];frame.extend(refs)
        for label in sorted({r['endpoint'] for r in refs}):
            data[label]=dict(cameras={c:{k:statistics.mean(r[k] for r in refs if r['endpoint']==label and r['camera']==c) for k in METRICS} for c in ['cam00','cam01']})
    else:missing.append('direct_references')
    main=[];aux=[]
    for label,e in data.items():
        repeat=label[1] if label.startswith('r') else 'reference';arm=label.split('_',1)[1] if repeat!='reference' else label
        for scope,v in list(e['cameras'].items())+[('equal_camera_mean',vector(e))]:
            main.append(dict(endpoint=label,repeat=repeat,arm=arm,scope=scope,**{k:v[k] for k in METRICS}))
            for key,value in v.items():
                if key not in METRICS:aux.append(dict(endpoint=label,scope=scope,metric=key,value=value,role='diagnostic'))
        for camera,v in e.get('train',{}).items():
            for k,value in v.items():aux.append(dict(endpoint=label,scope=camera,metric=k,value=value,role='train76_diagnostic'))
        if label in directories:
            with (directories[label]/'metrics_per_frame.csv').open() as f:
                for row in csv.DictReader(f):row['endpoint']=label;frame.append(row)
    gaps=compare(data);pairs={}
    for reference in ['Repeat7','J1','Async2']:
        rr=[r for r in gaps if r['endpoint'] in ['r1_Video7','r2_Video7'] and r['reference'].endswith('_'+reference) and r['scope']=='equal_camera_mean' and r['status']=='available']
        pairs['Video7-minus-'+reference]=dict(repeats=len(rr),delta={k:dict(mean=statistics.mean(r[k+'_delta'] for r in rr),minimum=min(r[k+'_delta'] for r in rr),maximum=max(r[k+'_delta'] for r in rr),direction_consistent=all(r[k+'_delta']>=0 for r in rr) or all(r[k+'_delta']<=0 for r in rr)) for k in METRICS} if rr else {})
    means={arm:{k:statistics.mean(vector(data[f'r{r}_{arm}'])[k] for r in ['1','2']) for k in METRICS} for arm in ARMS if all(f'r{r}_{arm}' in data for r in ['1','2'])}
    summary=dict(execution=dict(completed_formal_endpoints=sum(t['task_id'] in data for t in p['task_plan']),expected=4,missing=missing,complete=not missing),
        primary_metrics=METRICS,mean_vectors=means,paired=pairs,baseline=vector(data['U6000']),data=data,
        interpretation='Two paired suffixes share U6000. Video7 minus Repeat7 isolates input observations at fixed teacher weights, not removal of the optical-flow operator. Historical direct references have unmatched training budget/capacity. cam00/01 are development views.',
        quality_controls_execution=False,protocol_sha256=sha(OUT/'protocol.json'))
    for name,rows in [('main_quality.csv',main),('quality_gaps.csv',gaps),('metrics_per_frame.csv',frame),('auxiliary_metrics.csv',aux)]:csvwrite(OUT/name,rows)
    write(OUT/'quality_summary.json',summary);return summary

if __name__=='__main__':print(json.dumps(summarize(),ensure_ascii=False))
