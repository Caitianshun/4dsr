"""Two predeclared 2-step non-resume controls within the remaining four-round budget.

The original one-control failure remains immutable. No numerical limit changes:
use all three ordinary controls to measure the existing 3x CUDA replay envelope;
each control must independently satisfy both existing absolute render bounds.
"""
import subprocess
import traceback
from dv_common import *
shared=setup()
import torch
from engineering import maxdiff
from training_support import digest_state
from motion_model import load_model

def main():
    torch.set_num_threads(4);out=OUT/'engineering_followup';out.mkdir(exist_ok=False);p=read(OUT/'protocol.json');old=read(OUT/'engineering/replay.json');assert len(old['rows'])==4
    assert all(all(r['tests'].values()) for r in old['rows'][:3]);assert old['rows'][3]['tests']==dict(tensor=True,mean=True,rms=True,rms_envelope=False)
    assert read(OUT/'budget.json')['engineering_updates']==28
    registration=dict(status='registered_before_extra_control_updates',protocol_sha256=sha(OUT/'protocol.json'),old_replay_sha256=sha(OUT/'engineering/replay.json'),script_sha256=sha(__file__),new_updates=4,controls=2,updates_per_control=2,limits=p['engineering'],rule='Both new controls are always executed. All three ordinary controls retained; the existing restored trajectory must fit 3x maximum ordinary-control deviations. No restored run is rerun or selected. Absolute bounds must hold for every control and restored run. Otherwise stop with no formal training.')
    write(out/'registration.json',registration)
    parent=bound(p['parent']);m=shared.load_manifest(bound(p['manifest']));ev=shared.evaluator();o=next(o for o in m['observations'] if (o['camera_id'],o['frame_index'])==('cam02',40));cam=ev.render_camera(m,o,0)
    x=load_model(OUT/'engineering/r2_S_cov_direct/checkpoint_6002.pt',m);controls=[old['rows'][3]['comparisons']['repeat']];identities=[]
    for i in [1,2]:
        dest=out/f'ordinary_control_{i}';cmd=[sys.executable,'-u',str(HERE/'train.py'),'--method','S_cov','--repeat','2','--resume',str(parent),'--stop','6002','--out',str(dest),'--engineering']
        with (out/f'ordinary_control_{i}.log').open('w') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
        path=dest/'checkpoint_6002.pt';y=load_model(path,m)
        assert digest_state(x.checkpoint['rng'])==digest_state(y.checkpoint['rng']);assert digest_state(x.checkpoint['samplers'])==digest_state(y.checkpoint['samplers'])
        maximum=maxdiff((x.g.capture(),x.children.state_dict(),x.child_optimizer.state_dict()),(y.g.capture(),y.children.state_dict(),y.child_optimizer.state_dict()))
        with torch.no_grad():
            debit('engineering','additional_control_readonly_render',aux=2);delta=shared.render_model(x,cam)['render']-shared.render_model(y,cam)['render']
        controls.append(dict(tensor_maxabs=maximum,meanabs=float(delta.abs().mean()),rmse=float(delta.square().mean().sqrt()),samplers_exact=True,global_rng_exact=True))
        identities.append(dict(path=str(path.relative_to(ROOT)),sha256=sha(path)))
        del y;torch.cuda.empty_cache()
    limits=p['engineering'];r=old['rows'][3]['comparisons']['resume'];envelope={k:max(v[k] for v in controls) for k in ['tensor_maxabs','meanabs','rmse']}
    tests=dict(identity=True,all_absolute_bounds=all(v['meanabs']<=limits['render_meanabs'] and v['rmse']<=limits['render_rmse'] for v in [r,*controls]),tensor=r['tensor_maxabs']<=max(limits['tensor_floor'],limits['replay_envelope']*envelope['tensor_maxabs']),rms_envelope=r['rmse']<=max(limits['render_rmse_floor'],limits['replay_envelope']*envelope['rmse']))
    write(out/'result.json',dict(status='passed' if all(tests.values()) else 'failed',controls=controls,restored=r,envelope=envelope,tests=tests,registration_sha256=sha(out/'registration.json'),control_checkpoints=identities,original_single_control_failed=True))
    assert read(OUT/'budget.json')['engineering_updates']==32
    assert all(tests.values()),tests
    source=sources();route=read(OUT/'routing_audit.json');route.update(status='passed',protocol_sha256=sha(OUT/'protocol.json'),sources=source);write(OUT/'routing_audit.json',route)
    write(OUT/'engineering/complete.json',dict(status='passed',protocol_sha256=sha(OUT/'protocol.json'),parent_sha256=p['parent']['sha256'],sources=source,fixtures=read(OUT/'budget.json'),gpu=torch.cuda.get_device_name(),physical_gpu=os.environ['CUDA_VISIBLE_DEVICES'],replays=old['rows'],ordinary_control_extension=dict(registration_sha256=sha(out/'registration.json'),result_sha256=sha(out/'result.json'),script_sha256=sha(__file__)),earlier_single_control_failure_preserved=True,scope='Exact restoration identities plus unchanged absolute limits and existing 3x envelope measured with three ordinary controls for suffix S. No formal-quality selection.'))
if __name__=='__main__':
    try:main()
    except BaseException:write(OUT/'engineering_followup/failed.json',dict(status='failed',traceback=traceback.format_exc()));raise
