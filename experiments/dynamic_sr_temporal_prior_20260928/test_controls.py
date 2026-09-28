"""One concentrated CPU audit of the protocol semantics and four-receipt summary."""
from dv_common import *
from prepare_prior import source_windows,window_indices
from task_state import fixed_tasks,next_action
from summarize import compare
import subprocess
import shutil
import csv

def main():
    m=read(ROOT/'data/dynamic_sr/n3dv_prepared/cook_spinach/manifest.json')
    a=source_windows(m,'repeat7');b=source_windows(m,'video7')
    assert window_indices(0)==[0,0,0,0,1,2,3] and window_indices(59)==[56,57,58,59,59,59,59]
    assert len(a)==len(b)==1140
    for (x,aa,_),(y,bb,_) in zip(a,b):
        assert x==y and aa==[x]*7 and bb[3]==x
        assert all(z['camera_id']==x['camera_id'] and z['split']=='train' for z in bb)
    tasks=[dict(repeat=r,arm=arm,start=6000,stop=12000) for r,arm in [('1','Repeat7'),('1','Video7'),('2','Video7'),('2','Repeat7')]]
    assert len(fixed_tasks(dict(task_plan=tasks)))==4
    assert next_action(True,False,False,False)=='start_parent'
    assert next_action(True,True,False,True)=='wait_existing'
    assert next_action(True,False,False,True)=='resume_9000'
    assert next_action(True,False,True,True)=='evaluate_endpoint'
    def endpoint(v):return dict(cameras={c:dict(psnr=v,ssim=v/100,lpips=1-v/100) for c in ['cam00','cam01']})
    data={f'r{t["repeat"]}_{t["arm"]}':endpoint(31 if t['arm']=='Video7' else 30) for t in tasks}
    for r in ['1','2']:
        data[f'r{r}_J1']=endpoint(29);data[f'r{r}_Async2']=endpoint(29.5)
    data['LR-direct-HRrender']=endpoint(27);data['HR-direct-6k']=endpoint(33)
    rows=compare(data);v=[r for r in rows if r['endpoint']=='r2_Video7' and r['reference']=='r2_Repeat7' and r['scope']=='equal_camera_mean'][0]
    assert v['psnr_delta']==1 and v['lpips_delta']<0 and len(data)==10
    del data['HR-direct-6k'];assert any(r['status']=='unavailable' for r in compare(data))
    fixture=OUT/'engineering/completion_fixture'
    if fixture.exists():shutil.rmtree(fixture)
    fixture.mkdir(parents=True)
    def asset(label,value):
        dest=fixture/label;write(dest,value);return entry(dest)
    def history(label):
        directory=fixture/'history'/label;directory.mkdir(parents=True)
        e=asset('history/'+label+'/endpoint.json',endpoint(29))
        (directory/'metrics_per_frame.csv').write_text('endpoint,camera,frame,psnr,ssim,lpips\n'+label+',cam00,0,29,.29,.71\n')
        return dict(endpoint=e,directory=str(directory.relative_to(ROOT)))
    base=history('U6000');j={r:history(f'r{r}_J1') for r in ['1','2']}
    old={f'r{r}_{a}':history(f'r{r}_{a}') for r in ['1','2'] for a in ['Async2','Sync2']}
    plan=[dict(t,task_id=f'r{t["repeat"]}_{t["arm"]}') for t in tasks]
    pp=dict(run_id=RUN_ID,run_root=str(fixture.relative_to(ROOT)),baseline_reuse=base,historical_J1=j,historical_multiview=old,task_plan=plan)
    write(fixture/'protocol.json',pp)
    for t in plan:
        directory=fixture/'evaluation'/t['task_id'];directory.mkdir(parents=True)
        write(directory/'endpoint.json',endpoint(31 if t['arm']=='Video7' else 30))
        write(directory/'complete.json',dict(endpoint_sha256=sha(directory/'endpoint.json'),protocol_sha256=sha(fixture/'protocol.json')))
        (directory/'metrics_per_frame.csv').write_text('endpoint,camera,frame,psnr,ssim,lpips\n'+t['task_id']+',cam00,0,30,.30,.70\n')
    write(fixture/'references/complete.json',dict(rows=[dict(endpoint=l,camera=c,frame=0,psnr=v,ssim=v/100,lpips=1-v/100)
        for l,v in [('LR-direct-HRrender',27),('HR-direct-6k',33)] for c in ['cam00','cam01']]))
    result=subprocess.run([sys.executable,str(HERE/'summarize.py')],env={**os.environ,'FOURDSR_RUN_ROOT':str(fixture)},capture_output=True,text=True,check=True)
    fixture_summary=read(fixture/'quality_summary.json')
    assert fixture_summary['execution']['complete'] and fixture_summary['execution']['completed_formal_endpoints']==4
    assert fixture_summary['paired']['Video7-minus-Repeat7']['delta']['psnr']['mean']==1
    write(OUT/'control_tests.json',dict(status='passed',input_windows=2280,four_receipt_fixture=True,actual_summary_entrypoint_exercised=True,missing_baseline_is_nonblocking=True,
        resume_decisions=True,formal_parameter_updates=0,script=entry(Path(__file__))))

if __name__=='__main__':main()
