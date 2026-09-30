"""Freeze the single evidence-supported selection module, once after diagnosis."""
from dv_common import *
import argparse
import copy

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--audit-root',type=Path,default=OUT)
    ap.add_argument('--out',type=Path,default=OUT/'diagnosis_decision.json');ap.add_argument('--freeze-protocol',action='store_true');a=ap.parse_args()
    assert a.audit_root.resolve()==OUT
    names=['diagnosis_complete.json','spectrum/summary.json','spectrum/arithmetic_audit.json',
        'spectrum/anchor_decision.json','spectrum/anchor_detail_review.json',
        'spectrum/demand_coverage_summary.json','image_priors/summary.json',
        'structure/sampling_summary.json','structure/postprocess.json','texture_calibration.json','control_tests.json']
    evidence={n:entry(OUT/n) for n in names}
    assert read(OUT/'spectrum/arithmetic_audit.json')['rows']==980
    assert read(OUT/'control_tests.json')['status']=='passed'
    decision=dict(status='diagnosis_completed_choice_frozen',created_unix=time.time(),evidence=evidence,
        selected_modules=['T'],structure_branch=None,teacher_target='original frozen SwinIR PNG',
        primary_observation='Large static cam01 low-frequency error is concentrated in lamp-wall ROI; cam00 also has texture error.',
        supported_causes=['current 2D output has spatially concentrated structural/appearance failure',
            'SwinIR is better than the fixed Video7 alternative on registered original training images',
            'sampling demand varies with same-time contributing training views'],
        unresolved=['geometry vs coverage vs view-dependent radiance of cam01 wall',
            'whether selective texture supervision can improve the three quality metrics',
            'how much reconstruction error is causally attributable to cross-view teacher conflict',
            'depth and flow reliability away from independently supported sparse LR locations'],
        choices=dict(T='SplatSuRe same-time demand weighting; no new network or capacity; strongest implemented limited candidate',
            anchoring='not selected: teacher low-frequency benefit ~0.23%, fixed ROI MSE slightly worse; LPIPS improvement retained as a diagnostic tradeoff',
            depth='not selected: LR/HR prediction stability high but third-view support sparse and does not cover dominant cam01 error; ray means have substantial mixture thickness',
            flow='not selected: locally supported motion correspondence covers few pixels; no support for dominant static failure',
            sampling='not selected: fixed 8 observations of current models lose all three metrics under 2HR area sampling',
            capacity='not selected: no controlled evidence that extra points repair the observed failure'),
        limitations=['T is a texture-selection test, not an asserted solution to the dominant low-frequency failure',
            'cam00/01 are development evidence','two suffixes share U6000, not independent from-scratch seeds'],
        training=dict(arms=['T'],repeats=2,suffix_updates_each=6000,total_effective_updates=12000,
            main_endpoint=12000,recovery_only_endpoint=9000,base='historical J1 exact-compatible arithmetic reused',
            no_interim_quality_gate=True,no_hyperparameter_search=True))
    if a.out.exists():
        existing=read(a.out);assert existing['evidence']==evidence and existing['selected_modules']==['T'];decision=existing
    else:write(a.out,decision)
    if not a.freeze_protocol:return
    mp=OUT/'legal/texture_maps/prior_index.json';maps=read(mp)
    assert maps['status']=='completed' and len(maps['entries'])==1140
    old=read(ROOT/'output/dynamic_sr_temporal_prior_20260928/protocol.json')
    base=read(ROOT/'output/dynamic_sr_attribute_routing_20260928/protocol.json')
    p={k:copy.deepcopy(old[k]) for k in ['manifest','parent','teacher','lr_curve','old_schedule','schedules','roi',
        'evaluation','evaluation_files','temporal_caches','runtime','historical_J1','historical_multiview','baseline_reuse','quality']}
    p.update(schema=1,run_id=RUN_ID,run_root=str(OUT.relative_to(ROOT)),status='frozen_after_diagnosis',
        registered_unix=time.time(),decision=entry(a.out),sources=sources(),control_tests=entry(OUT/'control_tests.json'))
    p['texture']=dict(teacher=p['teacher'],maps=entry(mp),anchored_targets=None,
        loss_scale=read(OUT/'texture_calibration.json')['loss_scale'],calibration=entry(OUT/'texture_calibration.json'),
        confidence=None,fixed_map_strategy=maps['fixed_strategy'],selection=maps['selection'])
    p['training_files']=copy.deepcopy(base['training_files'])
    for e in maps['entries']:p['training_files'].append(dict(role='texture_weight',**e))
    gpus=['GPU-5c08f287-3ffd-edf9-91ed-cd6db2690f3b','GPU-b79cd3fe-81f0-f449-30be-432e2857e517']
    p['training']=dict(host='a100-train',remote_root='/home/ubuntu/3DGS/4dsr',physical_gpus=gpus,
        parent_step=6000,stop=12000,save=[9000,12000],points=132972,lr_offset=7200,sr_weight=.1,
        objective='original full LR L1 + inherited regularization + 0.1 * fixed calibrated demand-weighted SwinIR L1',
        policy='joint',sampling_module=False,trainable_parameters='unchanged base, children, shared deformation')
    p['task_plan']=[dict(task_id=f'r{i}_T',repeat=str(i),arm='T',start=6000,stop=12000,
        effective_updates=6000,rgb_per_update=2,adam_per_update=2,physical_gpu=gpu) for i,gpu in enumerate(gpus,1)]
    p['budget']=dict(effective_formal_max=12000,actual_formal_max=18000,retry_max=6000,
        effective_RGB=24000,effective_adam_calls=24000,engineering_updates=0)
    p['information_boundary']='Train image/array opens whitelist only legal training LR, frozen SwinIR and frozen U6000 demand maps. HR/dev priors diagnostic only.'
    p['arithmetic_compatibility']='Original J1 LR/reg backward then weighted SR backward then both original Adams; when disabled exact loss/render/pixel gradients, native-backward numerical variance reported.'
    for name in ['protocol.json','frozen_protocol.json']:
        path=OUT/name
        if path.exists():raise RuntimeError('Frozen protocol already exists; never overwrite '+str(path))
        write(path,p)
    print(json.dumps(dict(status=p['status'],tasks=p['task_plan'],protocol_sha256=sha(OUT/'protocol.json'))))

if __name__=='__main__':main()
