"""Assemble row-level metrics, deltas, physical-shape summaries and actual cost."""
import csv,hashlib,json,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=ROOT/'output/dynamic_sr_covariance_probe_20260927'
def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def csvwrite(p,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with Path(p).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def main():
    p=read(OUT/'protocol.json');d=read(OUT/'decision.json');rows=[];train=[];checkpoints=[];manifests=[]
    baseline=read(OUT/'parent_train/complete.json')['rows']
    rows.extend(baseline);rows.extend(x for x in read(OUT/'parent_hr/complete.json')['rows'] if x['reference']=='HR')
    for pair in d['training']['pairs']:
        repeat=pair['repeat']
        for arm in p['order'][str(repeat)]:
            run=OUT/f'r{repeat}_{arm}';manifest=read(run/'train/run_manifest.json');complete=read(run/'train/complete.json');manifests.append(manifest);train.append(complete)
            logs=[json.loads(line) for line in (run/'train/training.jsonl').read_text().splitlines()];assert len(logs)==600 and [r['camera'] for r in logs]==p['sequences'][str(repeat)] and sum(r['forwards'] for r in logs)==1800
            for step in [0,300,600]:
                ck=read(run/'train'/f'checkpoint_{step}.json');assert sha(ck['path'])==ck['sha256'];checkpoints.append(ck)
                if step==0:
                    assert ck['model_hash']==p['parent']['model_hash'];rows.extend(dict(x,arm=arm,repeat=repeat,checkpoint_sha256=ck['sha256']) for x in baseline)
                else:rows.extend(read(run/f'eval_{step}/complete.json')['rows'])
            rows.extend(x for x in read(run/'hr/complete.json')['rows'] if x['reference']=='HR')
    keys=lambda r:(r['camera'],r['region'],r['reference'],r['metric'])
    base={keys(r):r['value'] for r in rows if r['arm']=='AB600'}
    controls={(r['repeat'],r['step'],*keys(r)):r['value'] for r in rows if r['arm']=='C'}
    for r in rows:
        v=r['value'];parent=base.get(keys(r));control=controls.get((r['repeat'],r['step'],*keys(r)));r['delta_vs_AB600']=v-parent if parent is not None else None;r['relative_change_vs_AB600']=v/parent-1 if parent is not None and abs(parent)>=1e-6 else None;r['delta_vs_C']=v-control if control is not None else None;r['relative_change_vs_C']=v/control-1 if control is not None and abs(control)>=1e-6 else None
    ids=[(r['stage'],r['repeat'],r['arm'],r['step'],*keys(r)) for r in rows];assert len(ids)==len(set(ids));csvwrite(OUT/'metrics.csv',rows)
    shape=[];aux=0;aux_seconds=0
    for f in sorted((OUT/'footprints').glob('*/complete.json')):
        x=read(f);assert max(a['sum_maxabs'] for a in x['audits'])<=1e-5;shape.extend(x['rows']);shape.extend(x['shape_statistics']);aux+=x['counts']['auxiliary_forwards'];aux_seconds+=x['seconds']
    csvwrite(OUT/'shape.csv',shape)
    evals=[read(f) for f in OUT.glob('*/complete.json') if f.parent.name in ['parent_train','parent_hr']]+[read(f) for f in OUT.glob('r*/*/complete.json') if f.parent.name.startswith('eval_') or f.parent.name=='hr']
    engineer=read(OUT/'engineering_v2/complete.json');failed=read(OUT/'engineering/failure.json');updates=sum(x['updates'] for x in train);assert updates+engineer['updates']+failed['updates']<=p['budget']['total_updates_max']
    cost=dict(formal_updates=updates,engineering_successful_updates=engineer['updates'],failed_engineering_updates=failed['updates'],engineering_total=engineer['updates']+failed['updates'],failed_formal_updates=0,total_update_rounds=updates+engineer['updates']+failed['updates'],actual_Adam_step_calls=updates+engineer['adam_calls']+failed['adam_calls'],formal_training_RGB_forwards=sum(x['training_forwards'] for x in train),engineering_training_RGB_forwards=engineer['training_forwards']+failed['training_forwards'],readonly_RGB_forwards=sum(x['counts']['rgb_forwards'] for x in evals)+engineer['readonly_forwards']+failed['readonly_forwards'],auxiliary_forwards=aux,training_GPU_synchronized_wall_seconds=sum(x['train_seconds'] for x in train),evaluation_seconds=sum(x['seconds'] for x in evals),footprint_seconds=aux_seconds,engineering_success_seconds=engineer['seconds'],failed_engineering_command_seconds_approx=4.91,pipeline_wall_seconds=read(OUT/'execution_index.json')['wall_seconds'],peak_training_allocated_bytes=max(x['peak_allocated_bytes'] for x in train),engineering_retries=1,formal_retries=0,limits=p['budget'],hardware=p['physical_gpu'],note='GPU synchronized training wall time includes host loss bookkeeping and ledger I/O; not kernel-only timing')
    assert cost['engineering_total']<=32;write(OUT/'cost.json',cost);write(OUT/'run_manifest.json',dict(stage='A',runs=manifests,checkpoint_index='checkpoint_index.json'));write(OUT/'checkpoint_index.json',dict(status='completed',parent=p['parent'],checkpoints=checkpoints))
    write(OUT/'aggregation_audit.json',dict(status='passed',metrics_rows=len(rows),shape_rows=len(shape),unique_metric_keys=True,checkpoint_hashes_verified=True,actual_logs_and_sequences_verified=True,denominators_below_1e_6_marked_null=True,source_metrics='unrounded float tensors; display PNGs never used for metrics',decision=d['stop_reason'],created_unix=time.time()))
    print(json.dumps(dict(cost=cost,metrics_rows=len(rows),shape_rows=len(shape)),indent=2))
if __name__=='__main__':main()
