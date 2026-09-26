"""Fixed engineering investment gates, not statistical significance tests."""
import json
from pathlib import Path
import importlib.util

spec=importlib.util.spec_from_file_location('accepted_values',Path(__file__).parents[1]/'dynamic_sr_view_recovery_20260926/summarize.py')
accepted=importlib.util.module_from_spec(spec);spec.loader.exec_module(accepted)
values=accepted.values


def delta(a,b):
    return dict(psnr=a['psnr']-b['psnr'],ssim=a['ssim']-b['ssim'],
                **{k:a[k]/b[k]-1 for k in ['lpips','dynamic_lpips','temporal']})


def early_gate(early,control):
    d={c:delta(early[c],control[c]) for c in control}
    harm=any(v['psnr']<-.20 for v in d.values()) or d['cam00']['dynamic_lpips']>.05 or d['cam00']['temporal']>.05
    return dict(step=12000,continue_to_18000=not harm,early_minus_control=d,
                reason='pre_registered_safety_stop' if harm else 'passed_early_safety',limits_unchanged=True)


def decide(candidate,control,u6):
    d={c:delta(candidate[c],control[c]) for c in control};b={c:delta(candidate[c],u6[c]) for c in u6}
    x=d['cam00'];y=d['cam01'];z=b['cam00']
    harm=x['psnr']<-.20 or x['dynamic_lpips']>.05 or x['temporal']>.05 or y['psnr']<-.20 or y['lpips']>.05
    control_signal=(x['psnr']>=.20 and x['dynamic_lpips']<=.02 and x['temporal']<=.02) or (x['dynamic_lpips']<=-.05 and x['psnr']>=-.10 and x['temporal']<=.02)
    early_signal=z['psnr']>=-.10 and z['dynamic_lpips']<=-.03 and z['temporal']<=.02
    return dict(status='candidate_needs_independent_confirmation' if not harm and control_signal and early_signal else 'stop_current_recipe',
                harm=harm,control_signal=control_signal,u6000_signal=early_signal,vs_control=d,vs_u6000=b,
                endpoint=18000,extend_40000=False,statistical_claim=False)
