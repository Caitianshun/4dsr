"""Freeze two inherited schedules and six endpoints without post-hoc quality gates."""
import copy
import shutil
from dv_common import *

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT/'protocol.json').exists()
    old = read(OLD/'protocol.json'); p = copy.deepcopy(old)
    for k in ['engineering', 'decision', 'previous_observed_evidence']: p.pop(k, None)
    schedules = {}
    for n, e in old['schedules'].items():
        src = bound(e); dst = OUT/f'schedule_{n}.json'; shutil.copyfile(src, dst)
        assert sha(dst) == e['sha256']; schedules[n] = entry(dst)
    p.update(schema=2, experiment=RUN_ID, run_id=RUN_ID, run_root=str(OUT.relative_to(ROOT)),
        registered_unix=time.time(), adopted_plan=entry(OUT/'adopted_plan.codex.md'), schedules=schedules,
        legacy_summary=entry(OLD/'legacy_summary.json'),
        budget=dict(effective_formal_max=36000, actual_formal_max=42000, retry_max=6000, execution_wall_seconds_max=172800,
                    effective_RGB=72000, effective_adam_calls=72000, historical_engineering_updates=32),
        quality=dict(primary_metrics=['psnr','ssim','lpips'], primary_scope='full image cam00/cam01 equal weighted',
            diagnostics_only=['LR_L1','temporal','teacher_H','H_HR','dynamic_regions','ROI'],
            automatic_quality_veto=False, weighted_score=False, statistical_significance_claim=False,
            pareto_definition='non-dominated three-metric vectors after equal camera and then equal repeat averaging'),
        numerical_replay=dict(role='warning_only', old_failure_preserved=True,
                              original_rmse_limit=0.0001, observed_max=0.00013784242037218064),
        replaced_rules=['RMSE, meanabs and tensor replay envelopes are descriptive',
                        'LR/temporal/teacher/individual-camera thresholds do not gate execution',
                        'both repeats always execute; SSIM is primary'],
        task_plan=[dict(task_id=f'r{r}_{arm}',repeat=r,arm=arm,start=6000,stop=12000,effective_updates=6000)
                   for r in ['1','2'] for arm in old['training']['order'][r]],
        baseline_reuse=dict(directory=str((OLD/'evaluation/U6000').relative_to(ROOT)),
                            receipt=entry(OLD/'evaluation/U6000/complete.json'),
                            endpoint=entry(OLD/'evaluation/U6000/endpoint.json'), rgb_forwards_new=0,
                            original_wrapper=entry(OUT/'baseline_original_evaluate_endpoint.py'),
                            wrapper_difference='Only ffmpeg executable fallback added; primary metrics and render unchanged'),
        inherited_evidence={name:entry(OLD/path) for name,path in {
            'protocol':'protocol.json','routing':'routing_audit.json','replay':'engineering/replay.json',
            'replay_followup':'engineering_followup/result.json','old_decision':'decision.json',
            'old_budget':'budget.json','old_deployment':'deployment_verified.json',
            'completed_fixture_config':'engineering/r2_S_cov_resume/config.json'}.items()},
        inherited_reload_audits=[entry(f) for f in sorted((OLD/'engineering').glob('*/reload_audit.json'))],
        deadline='Two working days for execution and report; no new seed/module/scene or quality-threshold search')
    p['training']['fixed_repeats']=2
    for e in p['training_files'] + p['evaluation_files']: bound(e)
    for asset in p['temporal_caches'].values():
        for e in asset['assets']: bound(e)
    assert len(p['task_plan'])==6 and len(p['training_files'])==2280
    write(OUT/'protocol.json',p)
    write(OUT/'schedule_manifest.json',dict(schedules=schedules,inherited_from=entry(OLD/'schedule_manifest.json'),protocol_sha256=sha(OUT/'protocol.json')))
    audit=read(OLD/'routing_audit.json')
    write(OUT/'module_config.json',dict(module='SRAttributeRouter',arithmetic_source=entry(ROOT/'experiments/dynamic_sr_dynamic_validation_20260927/routing.py'),
        policies={'J_joint':'joint','A_sh':'appearance','S_cov':'appearance_covariance'},
        inherited_parameter_descriptions={r['arm']:r['policy'] for r in audit['rows']},
        no_new_parameters=True,no_inference_operations=True,live_parameter_ids='Recorded in each attempt config and first_update_audit',
        actual_policy_bindings_pending=True))
    print(json.dumps(dict(protocol_sha256=sha(OUT/'protocol.json'),tasks=6,training_files=2280)))

if __name__=='__main__':main()
