"""Persistent sequential execution, immediate return-triggered evaluation and immutable HR gate."""
import os,sys,json,subprocess,time,traceback
from pathlib import Path
from decide import pair,hr
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=ROOT/'output/dynamic_sr_covariance_probe_20260927'
def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False))
def run(script,args):
    dest=OUT/'logs';dest.mkdir(exist_ok=True);tag=f'{time.time_ns()}_{script}'
    command=[sys.executable,'-u',str(HERE/(script+'.py')),'--protocol',str(OUT/'protocol.json')]+[str(x) for x in args]
    with (dest/(tag+'.log')).open('w') as f:r=subprocess.run(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
    with (OUT/'commands.jsonl').open('a') as f:f.write(json.dumps(dict(command=command,returncode=r.returncode,log=str(dest/(tag+'.log')),completed=time.time()))+'\n')
    assert r.returncode==0,(script,r.returncode,tag)
def evaluate(ck,out,arm,repeat,step,HR=False):run('evaluate',['--checkpoint',ck,'--out',out,'--arm',arm,'--repeat',repeat,'--step',step]+(['--hr'] if HR else []))
def main():
    started=time.time();p=read(OUT/'protocol.json');assert read(OUT/'engineering_v2/complete.json')['status']=='completed';write(OUT/'execution_index.json',dict(status='running',stage='A',pid=os.getpid(),started_unix=started,gpu=p['physical_gpu']))
    evaluate(p['parent']['path'],OUT/'parent_train','AB600',0,0)
    pairs=[]
    for repeat in [1,2]:
        for arm in p['order'][str(repeat)]:
            out=OUT/f'r{repeat}_{arm}';out.mkdir();run('train',['--out',out/'train','--arm',arm,'--repeat',repeat])
            for step in [300,600]:evaluate(out/'train'/f'checkpoint_{step}.pt',out/f'eval_{step}',arm,repeat,step)
        result=pair(OUT,repeat);pairs.append(result);write(OUT/f'pair{repeat}_train_decision.json',result)
        if not result['passed']:break
    decision=dict(status='frozen',frozen_unix=time.time(),pairs=pairs,both_pairs_pass=len(pairs)==2 and all(x['passed'] for x in pairs),HR_pixels_used=False)
    write(OUT/'train_decision.json',decision)
    evaluate(p['parent']['path'],OUT/'parent_hr','AB600',0,0,True)
    endpoints=[('parent',p['parent']['path'])]
    for item in pairs:
        repeat=item['repeat']
        for arm in p['order'][str(repeat)]:
            out=OUT/f'r{repeat}_{arm}';ck=out/'train/checkpoint_600.pt';evaluate(ck,out/'hr',arm,repeat,600,True);endpoints.append((f'r{repeat}_{arm}',ck))
    final=hr(OUT);write(OUT/'decision.json',final)
    for label,ck in endpoints:run('footprint',['--checkpoint',ck,'--out',OUT/'footprints'/label])
    write(OUT/'execution_index.json',dict(status='stage_A_complete',stage='A',run_stage_B=final['run_stage_B'],stop_reason=final['stop_reason'],started_unix=started,completed_unix=time.time(),wall_seconds=time.time()-started,pairs=len(pairs),formal_updates=len(pairs)*1200))
if __name__=='__main__':
    try:main()
    except BaseException:
        write(OUT/'pipeline_failure.json',dict(status='failed',time=time.time(),traceback=traceback.format_exc()));raise
