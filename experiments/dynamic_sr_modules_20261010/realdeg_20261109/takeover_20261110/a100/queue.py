"""Run only the frozen A100 RD lanes, sequentially, under one fixed-service GPU lock."""
from pathlib import Path
import json, os, subprocess, sys, time, traceback, hashlib
ROOT=Path(__file__).resolve().parents[5]
BASE=ROOT/'experiments/dynamic_sr_modules_20261010/realdeg_20261109/baselines'
sys.path.insert(0,str(BASE))
from rd_common import metadata, output, identity
from disk_guard import check as disk_guard
STATE=ROOT/'output/dynamic_sr_modules_20261010/realdeg_20261109/takeover_20261110/a100/queue'
def write(p,j):
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(j,indent=2)+'\n');os.replace(t,p)
def event(j):
 j=dict(unix=time.time(),**j)
 with (STATE/'events.jsonl').open('a') as f:f.write(json.dumps(j)+'\n')
 print(json.dumps(j),flush=True)
def main():
 plan=json.loads(Path(__file__).with_name('queue_plan.json').read_text())
 if sys.argv[1:]!=['--execute']:
  print(json.dumps(plan,indent=2));return
 if str(ROOT)!='/home/ubuntu/3DGS/4dsr' or os.environ.get('CUDA_VISIBLE_DEVICES')!='1':raise ValueError('Wrong host/GPU')
 admission=json.loads((STATE.parent/'inputs_admission.json').read_text())
 if admission['status']!='passed':raise ValueError('Input admission not passed')
 STATE.mkdir(parents=True,exist_ok=True)
 status=dict(schema='rd4_a100_queue_v1',status='running',began_unix=time.time(),completed=[],current_job=None,input_admission=identity(STATE.parent/'inputs_admission.json'),queue_plan=identity(Path(__file__).with_name('queue_plan.json')))
 try:
  for row in plan['jobs']:
   jid=row['job_id'];lane=BASE/'lanes'/f'{jid}.json'
   if identity(lane)['sha256']!=row['lane_sha256']:raise ValueError('Frozen lane changed: '+jid)
   l=json.loads(lane.read_text());done=ROOT/output(l['method'],l['scene'])/'jobs'/jid/'complete.json'
   if done.exists():raise ValueError('Existing completion needs explicit owner review; no automatic rerun/skip: '+jid)
   disk_guard(ROOT,20)
   if l['method']=='splatsure':
    from teacher_rd import verify_consumer
    verify_consumer(l['scene'])
   status.update(current_job=jid);write(STATE/'status.json',status);event(dict(event='lane_started',job_id=jid))
   result=subprocess.run([sys.executable,'-u',str(BASE/'run_lane.py'),'--lane',str(lane.relative_to(ROOT)),'--execute'],cwd=ROOT)
   if result.returncode:raise RuntimeError(f'{jid} exited {result.returncode}')
   receipt=json.loads(done.read_text())
   if receipt.get('status')!='completed' or receipt.get('job_id')!=jid or receipt.get('lane')!=identity(lane):raise ValueError('Completion receipt mismatch')
   status['completed'].append(dict(job_id=jid,receipt=identity(done)));write(STATE/'status.json',status);event(dict(event='lane_completed',job_id=jid,receipt=identity(done)))
   all_cell=[x['job_id'] for x in plan['jobs'] if x['method']==l['method'] and x['scene']==l['scene']]
   if jid==all_cell[-1]:
    from returns_rd import verify_render, return_files
    checked=verify_render(l['method'],l['scene']);files=return_files(l['method'],l['scene'])
    write(STATE/f'{l["method"]}-{l["scene"]}-return-full.json',dict(version=1,files=files))
    write(STATE/f'{l["method"]}-{l["scene"]}-download.json',dict(version=1,files=[dict(path=f['path']) for f in files]))
    event(dict(event='render_verified_return_ready',method=l['method'],scene=l['scene'],verification=checked))
  status.update(status='completed',ended_unix=time.time(),current_job=None);write(STATE/'status.json',status);event(dict(event='queue_completed'))
 except BaseException:
  status.update(status='failed',ended_unix=time.time(),error=traceback.format_exc());write(STATE/'status.json',status);event(dict(event='queue_failed',job_id=status['current_job'],error=status['error']));raise
if __name__=='__main__':main()
