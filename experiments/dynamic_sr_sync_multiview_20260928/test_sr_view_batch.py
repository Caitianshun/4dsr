"""One CPU semantic suite; zero real renderer calls or training updates."""
import copy
import random
import tempfile
from unittest.mock import patch
import torch
from dv_common import *
from pair_schedule import generate,audit,counts
from sr_view_batch import SRViewBatch
from task_state import next_action,fixed_tasks
from evaluation_adapter import checkpoint_family
import summarize


def main():
    checks=[]
    def check(name,value):
        assert value,name;checks.append(name)
    # The old one-view mode returns exactly the historical scalar coefficient.
    single=SRViewBatch()
    check('default single has one exact .1 slot',single.observations(6001,3,7)==((7,.1),))
    check('module owns no model optimizer or trainable state',set(vars(single))=={'mode','table','table_sha','cursor'})
    prior=read(OLD/'protocol.json');manifest=read(bound(prior['manifest']))
    observations=[o for o in manifest['observations'] if o['split']=='train']
    legal={(o['camera_id'],o['frame_index']) for o in observations}
    check('1140 legal LR and frozen teacher pairs only',len(legal)==1140 and
        all({(e['camera'],e['frame']) for e in prior['training_files'] if e['role']==role}==legal for role in ['LR','teacher']))
    check('original camera times and resolution preserved',all(abs(o['time']-o['frame_index']/300)<1e-12 for o in observations)
        and manifest['resolutions']['lr']==[336,252] and manifest['resolutions']['hr']==[1344,1008])
    for repeat in ['1','2']:
        original=read(OLD/f'schedule_{repeat}.json');before=random.getstate();table=generate(original,repeat)
        check(f'r{repeat} no global RNG use',random.getstate()==before)
        check(f'r{repeat} deterministic table',table==generate(original,repeat))
        check(f'r{repeat} real temporal intervention and exact windows',audit(table,original['record_keys'])['passed'])
        for mode in ['sync2','async2']:
            stream=SRViewBatch(mode,table,'frozen_sha')
            for row in table['rows'][mode][:3000]:stream.observations(row['step'],row['lr_index'],row['sr_a_index'])
            resumed=SRViewBatch(mode,table,'frozen_sha',cursor=9000);resumed.validate_resume(stream.state())
            for row in table['rows'][mode][3000:]:
                args=(row['step'],row['lr_index'],row['sr_a_index'])
                assert stream.observations(*args)==resumed.observations(*args)
            check(f'r{repeat} {mode} full suffix matches 9000 resume',stream.state()==resumed.state())
            try:SRViewBatch(mode,table,'wrong',cursor=12000).validate_resume(stream.state())
            except AssertionError:checks.append(f'r{repeat} {mode} rejects changed pair SHA')
            else:raise AssertionError('changed schedule accepted')
    # Shared-parameter nonlinear toy model, two independent Adam optimizers.
    # Same LR/reg derivatives, two independent forwards, one update per optimizer.
    def build():
        a=torch.nn.Parameter(torch.tensor([.4,-.2],dtype=torch.float64));b=torch.nn.Parameter(torch.tensor([.3],dtype=torch.float64))
        return a,b,[torch.optim.Adam([a],lr=.001),torch.optim.Adam([b],lr=.002)]
    def output(a,b,x):return (a*x).sum().sin()+b.square().sum()
    a,b,opts=build();c,d,refs=build()
    xs=[torch.tensor([1.,2.],dtype=torch.float64),torch.tensor([-.5,3.],dtype=torch.float64)]
    target=[.2,.7]
    lr=(a.square().sum()+b.square().sum());lr.backward()
    for x,y in zip(xs,target):(.05*(output(a,b,x)-y).abs()).backward()
    objective=c.square().sum()+d.square().sum()+.05*sum((output(c,d,x)-y).abs() for x,y in zip(xs,target))
    objective.backward()
    check('two sequential SR backwards equal weighted combined gradient',all(torch.allclose(x.grad,y.grad,rtol=0,atol=1e-15) for x,y in [(a,c),(b,d)]))
    for opt in opts+refs:opt.step()
    check('one step in each Adam matches combined objective',torch.equal(a,c) and torch.equal(b,d) and all(int(next(iter(o.state.values()))['step'])==1 for o in opts))
    check('active task waits, endpoint reuses, 9000 resumes',
        [next_action(True,*state) for state in [(True,False,False),(False,True,True),(False,False,True)]]==['wait_existing','evaluate_endpoint','resume_9000'])
    check('identity mismatch blocks',next_action(False,False,False,False)=='blocked_identity')
    plan=[dict(task_id=f'r{r}_{arm}',repeat=r,arm=arm,start=6000,stop=12000,sr_slots=2,rgb_per_update=3,effective_updates=6000)
        for r,arms in [('1',['Async2','Sync2']),('2',['Sync2','Async2'])] for arm in arms]
    check('four-task formal budget and fixed order',len(fixed_tasks(dict(task_plan=plan,budget=dict(effective_formal_max=24000))))==4)
    with tempfile.TemporaryDirectory() as td:
        out=Path(td);p=dict(task_plan=plan,baseline_reuse={},historical_J1={},evaluation=dict(train_cameras=[f'cam{i:02d}' for i in range(2,21)]),inherited_evidence={'prior_protocol':'fixture'})
        endpoint=read(OLD/'evaluation/r1_J_joint/endpoint.json')
        labels=['U6000','r1_J_joint','r2_J_joint']+[t['task_id'] for t in plan]
        for label in labels:
            directory=out/'evaluation'/label;directory.mkdir(parents=True)
            ep=directory/'endpoint.json';write(ep,endpoint)
            cp=directory/'complete.json';write(cp,dict(endpoint_sha256=sha(ep),status='completed_evaluation'))
            for name in ['metrics_per_frame.csv','metrics_per_camera.csv']:
                (directory/name).write_text('endpoint,camera,psnr,ssim,lpips\n'+label+',cam00,1,1,1\n')
            ref=dict(directory=str(directory),endpoint=dict(path=str(ep),sha256=sha(ep)),receipt=dict(path=str(cp),sha256=sha(cp)))
            if label=='U6000':p['baseline_reuse']=ref
            elif label.endswith('J_joint'):p['historical_J1'][label[1]]=ref
        write(out/'protocol.json',p)
        with patch.object(summarize,'OUT',out),patch.object(summarize,'require_run_root',return_value=p):
            result=summarize.summarize()
        check('four simulated complete receipts reach full paired summary',result['execution']['complete'] and result['execution']['completed_formal_endpoints']==4 and len(result['paired'])==6)
        check('mock summary writes primary and paired tables',all((out/name).exists() for name in ['main_quality.csv','paired_quality.csv','quality_summary.json']))
    # Explicit metadata dispatch, without model label rewriting.
    meta=dict(stage='sync_multiview',run_id=RUN_ID,method='Sync2',repeat='1',task_id='r1_Sync2',intervention_step=12000,pair_sha256='pair')
    ck=dict(metadata=meta,sr_view_batch=dict(mode='sync2',slots=2,coefficients=[.05,.05],cursor=12000,pair_sha256='pair'))
    check('explicit new endpoint family',checkpoint_family(ck)=='sync_multiview')
    check('historical J and U identities preserved',checkpoint_family(dict(metadata=dict(method='J_joint'),attribute_routing={}))=='historical_J1' and checkpoint_family(dict(metadata=dict(method='U',intervention_step=6000)))=='historical_U6000')
    ck['sr_view_batch']['cursor']=9000
    try:checkpoint_family(ck)
    except AssertionError:checks.append('stale pair cursor rejects evaluation')
    else:raise AssertionError('bad pair cursor accepted')
    write(OUT/'control_tests.json',dict(status='passed',checks=checks,real_training_updates=0,RGB_forwards=0,
        cpu_toy_optimizer_steps=4,script_sha256=sha(__file__)))
    print(json.dumps(dict(status='passed',checks=len(checks))))


if __name__=='__main__':main()
