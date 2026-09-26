"""One targeted repeat distinguishes ordinary CUDA variation from resume drift.

Retains the failed strict maximum-element test. The prospective engineering
supplement uses image mean error plus a repeat envelope, never quality scores.
"""
import subprocess
import sys
import numpy as np
from shared import *
from freeze_policy import apply_policy, frozen_state
from check_engineering import tensor_compare

def main():
    torch.set_num_threads(4);out=OUT/'engineering_v1';p=paths();m=load_manifest(p['manifest']);ev=evaluator()
    supplement=out/'replay_protocol.json'
    if supplement.exists():raise FileExistsError(supplement)
    write_json(supplement,dict(status='registered_before_targeted_repeat',
        original_max_element_failure_preserved=True,repeat_envelope_multiplier=3.,
        render_meanabs_limit=1e-5,render_rmse_limit=1e-4,
        rationale='Sparse rasterization/Adam updates can have large max-element CUDA variation; require resume difference within ordinary same-start replay envelope, exact restored Adam/samplers/frozen state, and small whole-image mean/rms differences.',
        no_change_to_scientific_gates=True))
    start=Path(read(OUT/'protocol.json')['start'])
    def run(arm,resume,stop,name):
        dest=out/name;cmd=[sys.executable,'-u',str(Path(__file__).with_name('train.py')),'--method',arm,'--manifest',str(p['manifest']),
            '--resume',str(resume),'--schedule',str(p['schedule']),'--protocol',str(OUT/'protocol.json'),'--lr-curve',str(p['lr_curve']),
            '--target-index',str(p['teacher']),'--stop',str(stop),'--out',str(dest),'--smoke']
        with dest.with_suffix('.log').open('w') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
    run('C_joint',start,6004,'C_joint_repeat4')
    run('F_app',start,6004,'F_app_direct4');run('F_app',start,6002,'F_app_prefix2')
    run('F_app',out/'F_app_prefix2/checkpoint_6002.pt',6004,'F_app_resume2')
    run('F_app',start,6004,'F_app_repeat4')
    results=[]
    for arm in ['C_joint','F_app']:
        models={n:load_model(out/f'{arm}_{n}/checkpoint_6004.pt',m) for n in ['direct4','resume2','repeat4']}
        for model in models.values():apply_policy(model,arm)
        x=models['direct4'];comp={}
        for label in ['resume2','repeat4']:
            y=models[label];assert x.checkpoint['samplers']==y.checkpoint['samplers']
            assert x.checkpoint['metadata']['draw_sha256']==y.checkpoint['metadata']['draw_sha256']
            diffs=tensor_compare((x.g.capture(),x.children.state_dict(),x.child_optimizer.state_dict()),(y.g.capture(),y.children.state_dict(),y.child_optimizer.state_dict()))
            pixels=[]
            with torch.no_grad():
                for c in ['cam00','cam01','cam02']:
                    o=next(o for o in m['observations'] if o['camera_id']==c and o['frame_index']==40);cam=ev.render_camera(m,o,0)
                    delta=render_model(x,cam)['render']-render_model(y,cam)['render']
                    pixels.append(dict(camera=c,maxabs=float(delta.abs().max()),meanabs=float(delta.abs().mean()),rmse=float(delta.square().mean().sqrt())))
            comp[label]=dict(tensor_maxabs=max(v['maxabs'] for v in diffs),tensor_deltas=diffs,render=pixels)
            if arm=='F_app':
                assert frozen_state(x)==frozen_state(y)
                assert gamma_digest(x)==gamma_digest(y)==read(out/'zero_step.json')['gamma']
        resume,repeat=comp['resume2'],comp['repeat4']
        tests=dict(parameter_envelope=resume['tensor_maxabs']<=max(1e-4,3*repeat['tensor_maxabs']),
            image_mean_limit=all(v['meanabs']<=1e-5 and v['rmse']<=1e-4 for v in resume['render']),
            image_repeat_envelope=all(v['rmse']<=max(1e-6,3*r['rmse']) for v,r in zip(resume['render'],repeat['render'])))
        results.append(dict(arm=arm,comparisons=comp,tests=tests,passed=all(tests.values())))
        del models,x,y;torch.cuda.empty_cache()
    passed=all(r['passed'] for r in results)
    write_json(out/'replay_audit.json',dict(status='passed_with_documented_cuda_nondeterminism' if passed else 'failed',results=results,
        original_strict_test='failed and retained; not reclassified as a pass',
        sampling_exact=True,load_adam_exact=True,freeze_and_gamma_exact=True,
        quality_decision_use=False,gpu=torch.cuda.get_device_name(),actual_fixture_updates=dict(C_joint=12,F_app=12)))
    assert passed,'Resume exceeds empirically checked replay envelope'
    write_json(out/'complete.json',dict(status='passed_with_documented_cuda_nondeterminism',audit='replay_audit.json',
        protocol='replay_protocol.json',original_failure='initial_tolerance_failure.json',quality_decision_use=False))

if __name__=='__main__':main()
