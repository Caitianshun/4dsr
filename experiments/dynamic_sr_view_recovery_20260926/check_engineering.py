"""Zero-step parity and continuous-vs-resumed four-update integration only."""
import os
import subprocess
import sys
import time
import numpy as np
from shared import *
from freeze_policy import apply_policy, frozen_state

def tensor_compare(a,b):
    diffs=[]
    def rec(x,y,path):
        assert type(x)==type(y),(path,type(x),type(y))
        if torch.is_tensor(x):
            assert x.shape==y.shape and x.dtype==y.dtype
            d=float((x.double()-y.double()).abs().max()) if x.numel() else 0.
            diffs.append(dict(name=path,maxabs=d))
        elif isinstance(x,dict):
            assert x.keys()==y.keys()
            for k in x:rec(x[k],y[k],path+'/'+str(k))
        elif isinstance(x,(list,tuple)):
            assert len(x)==len(y)
            for i,(u,v) in enumerate(zip(x,y)):rec(u,v,path+'/'+str(i))
        elif isinstance(x,np.ndarray):assert np.array_equal(x,y)
        else:assert x==y,(path,x,y)
    rec(a,b,'root');return diffs

def main():
    torch.set_num_threads(4);p=paths();m=load_manifest(p['manifest']);ev=evaluator()
    out=OUT/'engineering_v1';out.mkdir(exist_ok=False);protocol=read(OUT/'protocol.json')
    start=Path(protocol['start']);reference=load_model(start,m);gam=gamma_digest(reference)
    zero=[]
    for arm in ['C_joint','F_app']:
        model=load_model(start,m);policy=apply_policy(model,arm)
        assert gamma_digest(model)==gam
        with torch.no_grad():
            for c in ['cam00','cam01','cam02','cam18']:
                for f in [40,80]:
                    o=next(o for o in m['observations'] if o['camera_id']==c and o['frame_index']==f)
                    cam=ev.render_camera(m,o,0);delta=float((render_model(model,cam)['render']-render_model(reference,cam)['render']).abs().max())
                    assert delta==0;zero.append(dict(arm=arm,camera=c,frame=f,maxabs=delta))
        del model
    del reference;torch.cuda.empty_cache()
    write_json(out/'zero_step.json',dict(status='passed',rows=zero,gamma=gam))
    def run(arm,resume,stop,directory):
        cmd=[sys.executable,'-u',str(Path(__file__).with_name('train.py')),'--method',arm,'--manifest',str(p['manifest']),
            '--resume',str(resume),'--schedule',str(p['schedule']),'--protocol',str(OUT/'protocol.json'),'--lr-curve',str(p['lr_curve']),
            '--target-index',str(p['teacher']),'--stop',str(stop),'--out',str(directory),'--smoke']
        with directory.with_suffix('.log').open('w') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
    result=[]
    for arm in ['C_joint','F_app']:
        direct=out/(arm+'_direct4');prefix=out/(arm+'_prefix2');resume=out/(arm+'_resume2')
        run(arm,start,6004,direct);run(arm,start,6002,prefix);run(arm,prefix/'checkpoint_6002.pt',6004,resume)
        x=load_model(direct/'checkpoint_6004.pt',m);y=load_model(resume/'checkpoint_6004.pt',m)
        apply_policy(x,arm);apply_policy(y,arm)
        assert x.checkpoint['samplers']==y.checkpoint['samplers']
        diffs=tensor_compare((x.g.capture(),x.children.state_dict(),x.child_optimizer.state_dict()),(y.g.capture(),y.children.state_dict(),y.child_optimizer.state_dict()))
        maximum=max(v['maxabs'] for v in diffs);assert maximum<=protocol['numerical_test_tolerance']['short_resume_parameter_maxabs'],maximum
        if arm=='F_app':assert frozen_state(x)==frozen_state(y) and gamma_digest(x)==gam and gamma_digest(y)==gam
        render_diffs=[]
        with torch.no_grad():
            for c in ['cam00','cam01','cam02']:
                o=next(o for o in m['observations'] if o['camera_id']==c and o['frame_index']==40);cam=ev.render_camera(m,o,0)
                render_diffs.append(float((render_model(x,cam)['render']-render_model(y,cam)['render']).abs().max()))
        assert max(render_diffs)<=protocol['numerical_test_tolerance']['short_resume_render_maxabs'],render_diffs
        result.append(dict(arm=arm,model_and_adam_maxabs=maximum,all_tensor_deltas=diffs,render_maxabs=max(render_diffs),samplers_exact=True,policy_reapplied=True))
        del x,y;torch.cuda.empty_cache()
    write_json(out/'complete.json',dict(status='passed',results=result,zero_step=zero,gpu=torch.cuda.get_device_name(),
        quality_decision_use=False,updates_per_arm=8,note='Four continuous and two+two resumed are engineering fixtures only'))

if __name__=='__main__':main()
