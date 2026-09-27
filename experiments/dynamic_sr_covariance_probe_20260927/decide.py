"""Predeclared Stage A investment gates; no ROI selection or endpoint selection."""
import json
from pathlib import Path

def read(p):return json.loads(Path(p).read_text())
def mean(x):return sum(x)/len(x)
def pair(root,repeat):
    root=Path(root);p=read(root/'protocol.json');b=read(root/'parent_train/complete.json')['values'];v={arm:read(root/f'r{repeat}_{arm}/eval_600/complete.json')['values'] for arm in ['C','S']};tests=[]
    def check(name,s,c,mode,limit):
        denominator=c if mode!='absolute' else None;valid=denominator is None or denominator>=p['thresholds']['min_denominator'];value=(1-s/c if mode=='gain' else s/c-1) if mode!='absolute' and valid else s-c if mode=='absolute' else None
        passed=valid and (value>=limit if mode=='gain' else value<=limit)
        tests.append(dict(name=name,S=s,reference=c,denominator=denominator,quantity=mode,value=value,threshold=limit,passed=bool(passed),judgeable=valid))
    h=lambda arm,c:mean([v[arm][c][r+'_H_L1'] for r in p['rois'][c]])
    person={a:mean([h(a,c) for c in p['rois']]) for a in v};check('person_four_ROI_H_gain',person['S'],person['C'],'gain',.05)
    for c in p['rois']:
        check(c+'_person_H_no_worse',h('S',c),h('C',c),'harm',0.)
        check(c+'_full_H',v['S'][c]['H'],v['C'][c]['H'],'harm',.02)
    low={a:mean([r['LR'] for r in vv.values()]) for a,vv in v.items()}
    check('train19_LR_vs_C',low['S'],low['C'],'harm',.02);check('train19_LR_vs_AB600',low['S'],mean([r['LR'] for r in b.values()]),'harm',.02)
    other={a:mean([r['H'] for c,r in vv.items() if c not in p['rois']]) for a,vv in v.items()};check('remaining17_H',other['S'],other['C'],'harm',.02)
    manifests=[read(root/f'r{repeat}_{a}/train/run_manifest.json') for a in ['C','S']];done=[read(root/f'r{repeat}_{a}/train/complete.json') for a in ['C','S']]
    contracts=all(d['freeze_pass'] and d['finite_pass'] and d['source_unchanged'] and not d['blocked'] and d['updates']==600 and d['training_forwards']==1800 for d in done) and manifests[0]['sequence']==manifests[1]['sequence'] and manifests[0]['initial_model_hash']==manifests[1]['initial_model_hash']==p['parent']['model_hash'] and all(m['gpu_uuid']==p['physical_gpu'] and not m['optimizer_initial']['state'] for m in manifests)
    tests.append(dict(name='contracts',passed=contracts));return dict(repeat=repeat,tests=tests,passed=all(t['passed'] for t in tests),person=person,train19_LR=low,remaining17_H=other)

def hr(root):
    root=Path(root);p=read(root/'protocol.json');training=read(root/'train_decision.json');pairs=[];directions=[]
    for r in training['pairs']:
        repeat=r['repeat'];rows={a:read(root/f'r{repeat}_{a}/hr/complete.json')['rows'] for a in ['C','S']};tests=[]
        def get(a,c,region,metric):return next(x['value'] for x in rows[a] if x['camera']==c and x['region']==region and x['reference']=='HR' and x['metric']==metric)
        av={a:{m:mean([get(a,c,reg,m) for c,rr in p['rois'].items() for reg in rr]) for m in ['psnr','ssim','lpips_alex']} for a in rows}
        def check(name,s,c,mode,limit):
            val=c-s if mode=='psnr_loss' else s/max(c,1e-6)-1;tests.append(dict(name=name,S=s,reference=c,denominator=c if mode=='harm' else None,value=val,threshold=limit,passed=bool((mode=='psnr_loss' or c>=1e-6) and val<=limit)))
        check('four_ROI_LPIPS_harm',av['S']['lpips_alex'],av['C']['lpips_alex'],'harm',.02);check('four_ROI_PSNR_loss',av['S']['psnr'],av['C']['psnr'],'psnr_loss',.10)
        directions.append({m:(av['S'][m]>av['C'][m] if m!='lpips_alex' else av['S'][m]<av['C'][m]) for m in av['C']})
        for c in ['cam00','cam01']:
            check(c+'_PSNR_loss',get('S',c,'full','psnr'),get('C',c,'full','psnr'),'psnr_loss',.20);check(c+'_LPIPS_harm',get('S',c,'full','lpips_alex'),get('C',c,'full','lpips_alex'),'harm',.05)
        pairs.append(dict(repeat=repeat,means=av,tests=tests,passed=all(x['passed'] for x in tests)))
    consistent={m:len(directions)==2 and all(d[m] for d in directions) for m in ['psnr','ssim','lpips_alex']};eligible=training['both_pairs_pass'] and all(r['passed'] for r in pairs) and any(consistent.values())
    return dict(status='frozen',stage='A',training=training,HR_pairs=pairs,consistent_HR_improvement=consistent,consistent_HR_status='assessed' if len(pairs)==2 else 'not assessed because repeat was not admitted',run_stage_B=eligible,stop_reason=None if eligible else 'Stage A training gate failed' if not training['both_pairs_pass'] else 'Stage A independent HR quality gate failed',interpretation='eligible for dynamic validation' if eligible else 'covariance adaptation did not meet the predefined investment gate')
