"""Freeze four endpoints, two exact-exposure tables and inherited identity assets."""
import copy
import csv
import shutil
from dv_common import *
from pair_schedule import generate, exposure_rows
from sr_view_batch import ARM_TO_MODE


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT/'protocol.json').exists()
    old = read(OLD/'protocol.json')
    assert sha(OLD/'protocol.json') == '92171dd81329a4ed329501fdb0ad55bb22c96126834e869778ecc31e9467a250'
    p = copy.deepcopy(old)
    schedules, pairs, exposures = {}, {}, []
    for repeat, e in old['schedules'].items():
        dest = OUT/f'schedule_{repeat}.json'; shutil.copyfile(bound(e), dest)
        schedules[repeat] = entry(dest)
        table = generate(read(dest), repeat)
        table.update(original_schedule=entry(dest))
        path = OUT/f'pair_schedule_{repeat}.json'; write(path, table)
        pairs[repeat] = entry(path); exposures.extend(exposure_rows(table))
    write(OUT/'pair_schedule.json', dict(tables=pairs, schedules=schedules,
        construction='fixed camera cycle; within-100-step second-frame cyclic permutation',
        lr_weighted_contribution_matched=False))
    with (OUT/'exposure.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(exposures[0]));w.writeheader();w.writerows(exposures)
    order={'1':['Async2','Sync2'],'2':['Sync2','Async2']}
    p['training'].update(order=order,objective='LR L1 + original regularization + .05 teacher_a L1 + .05 teacher_b L1',
        rgb_per_update=3,sr_slots=2,sr_coefficients=[.05,.05],module='SRViewBatch',default_mode='single')
    p.update(schema=3,experiment=RUN_ID,run_id=RUN_ID,run_root=str(OUT.relative_to(ROOT)),
        registered_unix=time.time(),adopted_plan=entry(OUT/'adopted_plan.codex.md'),
        schedules=schedules,pair_schedules=pairs,sources=sources(),runtime=read(OUT/'runtime.json'),
        task_plan=[dict(task_id=f'r{r}_{arm}',repeat=r,arm=arm,mode=ARM_TO_MODE[arm],start=6000,stop=12000,
            effective_updates=6000,rgb_per_update=3,sr_slots=2,adam_per_update=2) for r in ['1','2'] for arm in order[r]],
        budget=dict(effective_formal_max=24000,actual_formal_max=30000,retry_max=6000,
            effective_RGB=72000,effective_adam_calls=48000,new_evaluation_RGB=784,
            execution_wall_seconds_max=172800,historical_engineering_updates=32,new_engineering_updates=0),
        inherited_evidence=dict(prior_protocol=entry(OLD/'protocol.json'),prior_final_integrity=entry(OLD/'final_integrity.json'),
            prior_quality=entry(OLD/'quality_summary.json'),prior_readiness=entry(OLD/'research_readiness.json'),
            prior_checkpoint_index=entry(OLD/'checkpoint_index.json')),
        deadline='Four endpoints only; no third repeat, scene, loss, module or threshold search')
    p.pop('inherited_reload_audits',None)
    p['historical_J1']={}
    index=read(OLD/'checkpoint_index.json')['checkpoints']
    for repeat in ['1','2']:
        label=f'r{repeat}_J_joint';v=index[label];cp=local(v['checkpoint']);assert sha(cp)==v['sha256']
        directory=OLD/'evaluation'/label
        p['historical_J1'][repeat]=dict(checkpoint=entry(cp),receipt=entry(directory/'complete.json'),
            endpoint=entry(directory/'endpoint.json'),directory=str(directory.relative_to(ROOT)),rgb_per_update=2,
            effective_updates=6000,training_rgb_forwards=12000,adam_calls=12000,reused=True)
    for e in p['training_files']+p['evaluation_files']:bound(e)
    for a in p['temporal_caches'].values():
        for e in a['assets']:bound(e)
    assert len(p['training_files'])==2280 and len(p['task_plan'])==4
    write(OUT/'protocol.json',p)
    write(OUT/'module_config.json',dict(module='SRViewBatch',default='single',modes={'single':[.1],'async2':[.05,.05],'sync2':[.05,.05]},
        new_parameters=0,inference_operations=0,policy='joint',sources=p['sources']))
    print(json.dumps(dict(protocol_sha256=sha(OUT/'protocol.json'),tasks=4,pair_schedules=pairs)))


if __name__=='__main__':main()
