"""Close a bounded engineering failure without inventing formal candidate metrics."""
import csv
import time
import shutil
from collections import defaultdict
from dv_common import *

def main():
    p=read(OUT/'protocol.json');budget=read(OUT/'budget.json');extra=read(OUT/'engineering_followup/result.json');replay=read(OUT/'engineering/replay.json')
    assert budget['engineering_updates']==32 and budget['formal_updates']==0 and extra['status']=='failed'
    assert sha(local(p['old_decision']['path']))==p['old_decision']['sha256']
    assert not (OUT/'engineering/complete.json').exists()
    controller=read(OUT/'controller/status.json');assert controller['status']=='failed'
    assert not any(x.get('label','').endswith('_remote') for x in read(OUT/'controller/commands.json'))
    decisions=dict(status='stopped_engineering_failure',formal_training_started=False,formal_updates=0,quality_decision='not_evaluated',eligible_candidates=None,repeat_allowed=False,old_static_decision='fail_preserved',reason='Original independent-suffix S restart exceeded the 3x ordinary replay envelope. Two predeclared additional ordinary controls exhausted the 32-round budget; one exceeded the unchanged absolute RMSE limit.',blocking_measurement=dict(metric='raw float RGB RMSE at cam02 frame40 after 2 updates',observed=extra['controls'][1]['rmse'],limit=p['engineering']['render_rmse'],source='engineering_followup/result.json',identity_exact=True),protocol_sha256=sha(OUT/'protocol.json'),followup_result_sha256=sha(OUT/'engineering_followup/result.json'),no_more_updates_under_current_budget=True)
    write(OUT/'decision.json',decisions)
    for name in ['metrics_per_frame.csv','metrics_per_camera.csv']:shutil.copyfile(OUT/'evaluation/U6000'/name,OUT/name)
    index=[]
    for jf in sorted(OUT.glob('engineering*/**/checkpoint_*.json')):
        v=read(jf);path=jf.with_suffix('.pt');exists=path.exists()
        if exists:assert sha(path)==v['sha256']
        index.append(dict(kind='engineering_only',path=str(path),sha256=v['sha256'],local=exists,remote_path='/home/ubuntu/3DGS/4dsr/'+str(path.relative_to(ROOT)),host='a100-train',GPU=p['training']['physical_gpu'],task=HERE.name,metadata=v['metadata'],status='retained_for_diagnosis_not_formal_endpoint'))
    returned=(OUT/'return_checkpoint_files.txt').read_text().splitlines()
    assert all(local(x).exists() for x in returned)
    write(OUT/'checkpoint_index.json',dict(status='engineering_failure_archived',parent=p['parent'],engineering=index,critical_returned_count=sum(x['local'] for x in index),formal_checkpoints=[]))
    complete=[read(f) for f in OUT.glob('engineering*/**/complete.json')]
    train=[x for x in complete if x.get('actual_updates') is not None]
    assert sum(x['actual_updates'] for x in train)==28
    journal=defaultdict(list)
    for line in (OUT/'engineering_service_journal.jsonl').read_text().splitlines():
        try:d=json.loads(line)
        except ValueError:continue
        unit=d.get('USER_UNIT') or d.get('_SYSTEMD_USER_UNIT')
        if unit:journal[unit].append(d)
    services={}
    for unit,rr in journal.items():
        times=[int(x['__REALTIME_TIMESTAMP'])/1e6 for x in rr]
        services[unit]=dict(first_unix=min(times),last_unix=max(times),logged_wall_seconds=max(times)-min(times),messages=[x['MESSAGE'] for x in rr])
    # Classify actual forwards from the ledger and preserved attempt progress.
    # routing aux: 2 failed zero-step RGBs + 2 successful passes * (3 zero-step RGBs + 4 effective-state calls) =16.
    # replay_render: 16 RGBs; extra control comparisons:4 RGBs.
    assert budget['by_run']['routing']['auxiliary_forwards']==16
    readonly_engineering_rgb=8+budget['by_run']['replay_render']['auxiliary_forwards']+budget['by_run']['additional_control_readonly_render']['auxiliary_forwards']
    cost=dict(status='final',formal_updates=0,engineering_updates=32,adam_update_rounds=28,disposable_SGD_rounds=4,actual_adam_calls=budget['adam_calls'],formal_training_rgb_forwards=0,engineering_gradient_rgb_forwards=budget['training_rgb_forwards'],engineering_readonly_rgb_forwards=readonly_engineering_rgb,auxiliary_effective_state_forwards=8,legacy_readonly_rgb_forwards=63,baseline_evaluation_rgb_forwards=196,total_rgb_forwards=budget['training_rgb_forwards']+readonly_engineering_rgb+63+196,successful_fixture_train_seconds=sum(x['train_s'] for x in train),fixture_peak_allocated_GB=max(x['peak_gb'] for x in train),services=services,service_logged_wall_seconds=sum(x['logged_wall_seconds'] for x in services.values()),legacy_evaluation_seconds=read(OUT/'legacy_summary.json')['seconds'],baseline_evaluation_seconds=read(OUT/'evaluation/U6000/complete.json')['seconds'],historical_updates_excluded=1214,budget=p['budget'],note='32 rounds = 28 two-Adam rounds + four disposable SR-only SGD rounds. Gradient-only and read-only passes do not update parameters. Cost categories are reclassified from the immutable debit ledger, not GPU theoretical estimates.')
    assert cost['actual_adam_calls']==56 and readonly_engineering_rgb==28
    write(OUT/'cost.json',cost)
    write(OUT/'execution_index.json',dict(status='stopped_at_engineering_gate',protocol_sha256=sha(OUT/'protocol.json'),decision_sha256=sha(OUT/'decision.json'),formal_updates=0,engineering_updates=32,old_decision_unchanged=True,controller_fail_closed=True,all_required_evaluation_assets_present=True,baseline_only_evaluation=True,critical_checkpoints_returned=True,ended_unix=time.time()))
    source={str(f.relative_to(ROOT)):sha(f) for f in HERE.glob('*.py')};write(OUT/'final_source_identity.json',dict(files=source,base_public_commit='b9c35bc2769a6494f975577cbebe13620f5a52e9',research_directory_is_not_git=True))
    print(json.dumps(dict(status='stopped_at_engineering_gate',cost=cost),ensure_ascii=False))
if __name__=='__main__':main()
