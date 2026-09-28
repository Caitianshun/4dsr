"""Freeze the four-arm protocol after both complete teacher indices exist."""
import copy
import shutil
from dv_common import *
from temporal_prior import ARM_TO_MODE

def main():
    assert not (OUT/'protocol.json').exists()
    p=copy.deepcopy(read(OLD/'protocol.json'))
    for key in ['pair_schedules','old_decision','legacy_summary','numerical_replay','replaced_rules']:p.pop(key,None)
    schedules={}
    for r,e in p['schedules'].items():
        dest=OUT/f'schedule_{r}.json';shutil.copyfile(bound(e),dest);schedules[r]=entry(dest)
    order={'1':['Repeat7','Video7'],'2':['Video7','Repeat7']}
    priors={mode:entry(OUT/'priors'/mode/'prior_index.json') for mode in ARM_TO_MODE.values()}
    files=[x for x in p['training_files'] if x['role']=='LR']
    keys={(x['camera'],x['frame']) for x in files};assert len(files)==len(keys)==1140
    identities=[]
    for mode,e in priors.items():
        index=read(bound(e));assert index['mode']==mode and index['status']=='completed'
        assert len(index['entries'])==1140 and {(x['camera'],x['frame']) for x in index['entries']}==keys
        identities.append(index['identity'])
        for item in index['entries']:
            bound(item);files.append(dict(role='teacher',mode=mode,camera=item['camera'],frame=item['frame'],path=item['path'],sha256=item['sha256']))
    for key in ['commit','upstream_python_sha256','configuration','weights','precision','quantization','window_length','center_index','code','python','torch','cuda','mmcv','numpy']:
        assert identities[0][key]==identities[1][key],('Teacher identity mismatch',key)
    p['training'].update(order=order,objective='LR L1 + original regularization + 0.1 frozen target RGB L1',
        rgb_per_update=2,sr_slots=1,sr_coefficients=[.1],module='TemporalPrior',default_mode='single_image',policy='joint',
        transport_host='a100-via-5090',remote_root='/home/ubuntu/3DGS/4dsr')
    p.update(schema=4,experiment=RUN_ID,run_id=RUN_ID,run_root=str(OUT.relative_to(ROOT)),registered_unix=time.time(),
        adopted_plan=entry(OUT/'source_plan/dynamic_sr_temporal_prior_execution_2026-09-28.docx'),
        schedules=schedules,priors=priors,training_files=files,sources=sources(),
        task_plan=[dict(task_id=f'r{r}_{arm}',repeat=r,arm=arm,mode=ARM_TO_MODE[arm],start=6000,stop=12000,
            effective_updates=6000,rgb_per_update=2,adam_per_update=2) for r in ['1','2'] for arm in order[r]],
        budget=dict(effective_formal_max=24000,actual_formal_max=30000,retry_max=6000,effective_RGB=48000,effective_adam_calls=48000,
            new_evaluation_RGB=784,reference_evaluation_RGB=480,execution_wall_seconds_max=172800,new_engineering_updates=0),
        inherited_evidence=dict(prior_protocol=entry(OLD/'protocol.json'),prior_final_integrity=entry(OLD/'final_integrity.json'),
            prior_readiness=entry(OLD/'research_readiness.json')),
        historical_multiview={},deadline='Four endpoints, no quality gate after repeat1, no teacher/window/loss scan')
    for r in ['1','2']:
        for arm in ['Async2','Sync2']:
            label=f'r{r}_{arm}';directory=OLD/'evaluation'/label
            p['historical_multiview'][label]=dict(repeat=r,arm=arm,directory=str(directory.relative_to(ROOT)),
                receipt=entry(directory/'complete.json'),endpoint=entry(directory/'endpoint.json'))
    write(OUT/'protocol.json',p)
    write(OUT/'prior_index.json',dict(priors=priors,entries=2280,unique_training_LR=1140,source_frame_instances=15960,
        input_boundary='Only 19 registered train cameras; no HR or held-out LR; 7 registered frames at original stride2'))
    registry=read(OUT/'baseline_registry.json')
    original_registry=OUT/'references/baseline_registry_at_evaluation.json'
    if not original_registry.exists():shutil.copyfile(OUT/'baseline_registry.json',original_registry)
    parent_config_path=bound(p['parent']).parent/'config.json';parent_config=read(parent_config_path)
    inherited=parent_config['parent_metadata'];assert inherited['args']['mode']=='lr_integrated' and inherited['args']['steps']==1200
    assert inherited['args']['coarse_steps']==1000 and inherited['args']['fine_steps']==6000 and parent_config['steps']==6000
    assert parent_config['method']=='U' and parent_config['branch']=='ordinary_split'
    base=dict(coarse_updates=1000,fine_parent_updates=6000,lr_integration_updates=1200,sr_prefix_updates=6000,
        points=p['training']['points'],parent_config=entry(parent_config_path),heldout_cameras_verified=['cam00','cam01'],
        equal_training_budget=False,lr_integration_source=entry(ROOT/'experiments/dynamic_sr_20260918/run_experiment.py'))
    registry['continuations']={'U6000':dict(base,suffix_updates=0,total_updates=14200,training_rgb_forwards=20200,supervision='train LR and frozen SwinIR')}
    for r in ['1','2']:
        for arm in ['J1','Async2','Sync2','Repeat7','Video7']:
            count=3 if arm in ['Async2','Sync2'] else 2
            registry['continuations'][f'r{r}_{arm}']=dict(base,suffix_updates=6000,total_updates=20200,
                suffix_rgb_forwards=count*6000,training_rgb_forwards=20200+count*6000,
                supervision='train LR and frozen '+('BasicVSR++ '+arm if arm in ['Repeat7','Video7'] else 'SwinIR'),
                step_label='SR12000 = 6000 prefix + 6000 suffix, excludes preceding stages')
    write(OUT/'baseline_registry.json',registry)
    reference_receipt=read(OUT/'references/complete.json')
    reference_receipt.update(registry_at_evaluation=entry(original_registry),registry=entry(OUT/'baseline_registry.json'),
        registry_augmentation='Continuation lineage added; original direct-reference metrics unchanged')
    write(OUT/'references/complete.json',reference_receipt)
    print(json.dumps(dict(protocol_sha256=sha(OUT/'protocol.json'),tasks=4,training_files=len(files))))

if __name__=='__main__':main()
