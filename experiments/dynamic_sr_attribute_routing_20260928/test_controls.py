"""CPU scheduling, immutable artifact recovery, and unchanged module delegation tests."""
import tempfile
from unittest.mock import patch
from dv_common import *
from task_state import next_action, fixed_tasks
from quality import dominates, pareto, classify

def main():
    checks=[]
    def check(name, condition):
        assert condition,name;checks.append(name)
    plan=[dict(task_id=f'r{r}_{a}',repeat=r,arm=a,start=6000,stop=12000) for r,arms in [('1',['J_joint','A_sh','S_cov']),('2',['S_cov','A_sh','J_joint'])] for a in arms]
    check('numerical warning still registers all six tasks',len(fixed_tasks(dict(task_plan=plan,numerical_replay='warning')))==6)
    check('identity failure never dispatches',next_action(False,False,False,False)=='blocked_identity')
    check('running service is not dispatched twice',next_action(True,True,False,False)=='wait_existing')
    check('valid 9000 checkpoint selects resume',next_action(True,False,False,True)=='resume_9000')
    check('valid 12000 endpoint selects evaluation',next_action(True,False,True,True)=='evaluate_endpoint')
    check('no checkpoint uses original parent',next_action(True,False,False,False)=='start_parent')
    import remote_task
    with tempfile.TemporaryDirectory() as td:
        root=Path(td);out=root/'output'/RUN_ID;attempt=out/'runs/r1_J_joint/attempt_01';payload=attempt/'train';payload.mkdir(parents=True)
        protocol=dict(task_plan=plan,schedules={'1':dict(sha256='schedule')});write(out/'protocol.json',protocol)
        write(out/'research_readiness.json',dict(sources={'a':'hash'}));write(attempt/'dispatch.json',dict(unit='fixture.service'))
        write(payload/'config.json',dict(sources={'a':'hash'}))
        for step in [9000,12000]:
            cp=payload/f'checkpoint_{step}.pt';cp.write_bytes(f'CPU fixture {step}'.encode())
            write(cp.with_suffix('.json'),dict(sha256=sha(cp),metadata=dict(task_id='r1_J_joint',method='J_joint',repeat='1',protocol_sha256=sha(out/'protocol.json'),schedule_sha256='schedule')))
        with patch.object(remote_task,'ROOT',root),patch.object(remote_task,'OUT',out),patch.object(remote_task,'require_run_root',return_value=protocol),patch.object(remote_task,'info',return_value={'ActiveState':'inactive'}):
            result=remote_task.inspect('r1_J_joint')
            check('unacknowledged 12000 is not accepted as endpoint',result['endpoint'] is None and result['resume']['step']==9000)
            write(payload/'complete.json',dict(status='completed'))
            result=remote_task.inspect('r1_J_joint')
            check('real artifact inspector reuses completed 12000',result['endpoint']['step']==12000)
            (payload/'checkpoint_12000.pt').write_bytes(b'tampered')
            try:remote_task.inspect('r1_J_joint')
            except AssertionError:checks.append('checkpoint identity mismatch raises before dispatch')
            else:raise AssertionError('tampered checkpoint accepted')
    from readiness import validate
    import readiness
    with patch.object(readiness,'read',return_value=dict(status='ready_for_quality_experiment',protocol_sha256='right',parent_sha256='parent',sources={'code':'wrong'},numerical_replay='warning')),patch.object(readiness,'sha',return_value='right'):
        try:validate(dict(parent=dict(sha256='parent')),{'code':'right'})
        except AssertionError:checks.append('readiness source mismatch is not bypassed by warning')
        else:raise AssertionError('bad source accepted')
    # Facade delegation is tested with exact object identity and one audited arithmetic call.
    import sr_attribute_router as module
    model=object();loss=object();gradient_result=object()
    with patch.object(module._validated,'selected',return_value={'sh':object()}),patch.object(module._validated,'backward_prior',return_value=gradient_result) as backward:
        for policy,arm in module.POLICIES.items():
            router=module.SRAttributeRouter(policy,model)
            assert router.backward_sr(loss,audit=True) is gradient_result
            backward.assert_called_with(loss,model,arm,audit=True)
        check('three policies delegate original arithmetic exactly once each',backward.call_count==3)
    check('three metrics all participate in Pareto',pareto({'A':dict(psnr=31,ssim=.9,lpips=.1),'S':dict(psnr=32,ssim=.89,lpips=.09)})==['A','S'])
    check('repeated tradeoff remains a tradeoff',classify([dict(psnr=.1,ssim=-.001,lpips=-.002)]*2)['classification']=='repeat_supported_quality_tradeoff')
    check('opposite repeat signs are insufficient',classify([dict(psnr=.1,ssim=.001,lpips=-.002),dict(psnr=-.1,ssim=.001,lpips=-.002)])['classification']=='unstable_or_insufficient_evidence')
    check('one repeat cannot claim repeat support',classify([dict(psnr=.1,ssim=.001,lpips=-.002)])['classification']=='insufficient_completed_repeats')
    write(OUT/'control_tests.json',dict(status='passed',checks=checks,parameter_updates=0,RGB_forwards=0,script_sha256=sha(__file__)))
    print(json.dumps(dict(status='passed',checks=len(checks))))

if __name__=='__main__':main()
