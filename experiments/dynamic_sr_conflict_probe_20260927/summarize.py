"""Verify actual training records and collect bounded results without selecting endpoints."""
import argparse
from collections import Counter
from probe_common import *

def main():
    pa=argparse.ArgumentParser();pa.add_argument('--protocol',type=Path,required=True);pa.add_argument('--out',type=Path,required=True);a=pa.parse_args();p=load_protocol(a.protocol);root=Path(p['_root']);a.out.mkdir(parents=True,exist_ok=True)
    decision=read(root/'decision.json');assert decision['status']=='frozen';assert read(root/'pipeline/status.json')['status']=='completed'
    initial=torch.load(root/'initial_baked.pt',map_location='cpu',weights_only=False)['baked'];runs=[];checkpoints=[];metrics=[];transfers=[];costs=[]
    for rid in range(1,decision['round_count']+1):
        for arm in p['order']:
            d=root/f'repeat{rid}/{arm}/train';c=read(d/'complete.json');cfg=read(d/'config.json');logs=[json.loads(x) for x in (d/'training.jsonl').read_text().splitlines()]
            assert len(logs)==600 and [x['step'] for x in logs]==list(range(1,601));assert [x['camera'] for x in logs]==p['lr_sequence'];assert sum(x['forwards'] for x in logs)==1800
            assert all([x['a_coefficient'],x['b_coefficient']]==p['arms'][arm] for x in logs);assert c['geometry_unchanged'] and c['SH_changed'] and c['HR_training_reads']==0 and not c['blocked'];assert cfg['protocol_sha256']==sha(a.protocol);assert all(sha(f)==h for f,h in cfg['source_sha256'].items() if Path(f).name!='summarize.py')
            assert Counter(x['camera'] for x in logs)==Counter(p['lr_sequence']);assert c['gpu_uuid']==p['physical_gpu']
            for step in [0,300,600]:
                f=d/f'checkpoint_{step}.pt';record=read(f.with_suffix('.json'));assert sha(f)==record['sha256'];obj=torch.load(f,map_location='cpu',weights_only=False);s=obj['baked'];assert s['base_count']==initial['base_count'] and s['degree']==initial['degree']
                for name in p['frozen']:assert torch.equal(s['values'][name],initial['values'][name]),(arm,step,name)
                if step==0:assert all(torch.equal(s['values'][n],initial['values'][n]) for n in initial['values']);assert not obj['optimizer']['state']
                else:assert any(not torch.equal(s['values'][n],initial['values'][n]) for n in p['trainable']);assert all(int(v['step'])==step for v in obj['optimizer']['state'].values())
                assert [g['name'] for g in obj['optimizer']['param_groups']]==p['trainable'];checkpoints.append(dict(**record,arm=arm,repeat_id=rid,raw_frozen_exact=True,optimizer_SH_only=True))
            runs.append(dict(arm=arm,repeat_id=rid,steps=len(logs),forwards=1800,LR_sequence_exact=True,HR_training_reads=0,geometry_hash=c['geometry_hash'],exposure=c['exposure'],SH_update_statistics=c['SH_update_statistics']))
            costs.append(dict(stage='training',arm=arm,repeat_id=rid,**{k:c[k] for k in ['preload_seconds','train_seconds','loop_wall_seconds','total_wall_seconds','peak_allocated_bytes','gpu_uuid']}))
    legal={str(Path(f['path']).resolve()) for f in p['training_files']}
    for f in root.glob('**/metrics.csv'):
        if 'failures' in f.parts or f.parent==a.out:continue
        with f.open() as h:metrics.extend(csv.DictReader(h))
    for d in sorted(root.glob('updates_*')):
        if not d.is_dir():continue
        c=read(d/'complete.json');assert c['actual_adam_candidates']==4 and c['source_unchanged'] and c['contracts_passed']==20 and c['rows']==840;assert set(c['image_reads'])<=legal and not c['blocked'];assert read(d/'update_ledger.json')['actual_adam_candidates']==4
        with (d/'transfer.csv').open() as h:transfers.extend(csv.DictReader(h))
        costs.append(dict(stage=d.name,seconds=c['seconds'],counts=c['counts'],optimizer_steps=4))
    commands=read(root/'pipeline/commands.json');hr=[c for c in commands if c['label'].endswith('_HR')];assert hr and min(c['started'] for c in hr)>decision['frozen_unix']
    assert sha(p['manifest']['path'])==p['manifest']['sha256'];assert sha(p['teacher_index']['path'])==p['teacher_index']['sha256'];assert all(sha(x['path'])==x['sha256'] for x in p['training_files'])
    for f in root.glob('**/complete.json'):
        if any(x in f.parts for x in ['failures','summary']):continue
        c=read(f)
        if c.get('privileged') is False:assert set(c['image_reads'])<=legal and not c['blocked']
        if 'eval' in str(f) or 'privileged' in str(f):costs.append(dict(stage=str(f.parent.relative_to(root)),seconds=c.get('seconds'),counts=c.get('counts')))
    train_steps=600*len(runs);diag_steps=4*(decision['round_count']+1);engineering=read(root/'engineering/update_ledger.json')['actual_updates'];assert train_steps<=4800 and diag_steps<=12 and engineering<=8
    integrity=dict(status='passed',runs=runs,checkpoints=checkpoints,training_updates=train_steps,training_forwards=1800*len(runs),diagnostic_updates=diag_steps,engineering_updates=engineering,total_updates=train_steps+diag_steps+engineering,decision_before_HR=True,first_HR_started=min(c['started'] for c in hr),decision_frozen=decision['frozen_unix'],source_and_data_hashes_passed=True,cameras_unchanged_via_manifest=True)
    write(root/'integrity.json',integrity);csvwrite(a.out/'metrics.csv',metrics);csvwrite(a.out/'transfer.csv',transfers)
    costs.append(dict(stage='footprints',**{k:read(root/'footprints/complete.json')[k] for k in ['seconds','counts']}))
    cost=dict(training_updates=train_steps,diagnostic_updates=diag_steps,engineering_updates=engineering,total_updates=train_steps+diag_steps+engineering,training_forwards=1800*len(runs),stages=costs,pipeline_attempts_wall_seconds=sum(x['seconds'] for x in commands),elapsed_since_first_launch_seconds=time.time()-read(root/'launch.json')['started'],note='Elapsed includes implementation and failure repair; resumed diagnostic successful-work counters exclude preserved failed-attempt work, whose wall time is separately retained.')
    write(root/'cost.json',cost);write(a.out/'index.json',dict(status='completed',metrics_rows=len(metrics),transfer_rows=len(transfers),decision=str(root/'decision.json'),integrity=str(root/'integrity.json'),cost=str(root/'cost.json')));print(json.dumps(dict(status='passed',updates=cost['total_updates'],metrics=len(metrics),transfers=len(transfers))))

if __name__=='__main__':main()
