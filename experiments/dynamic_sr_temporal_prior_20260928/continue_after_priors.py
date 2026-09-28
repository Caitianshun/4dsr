"""Process-exit driven cache return, registration, identity verification and batch execution."""
import concurrent.futures
import subprocess
import shlex
import traceback
from dv_common import *

HOST='a100-via-5090'
REMOTE='/home/ubuntu/3DGS/4dsr'

def main():
    folder=OUT/'preparation';folder.mkdir(parents=True,exist_ok=True)
    lock=(folder/'continuation.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    started=time.time();state={}
    def status(**kw):
        state.update(kw);state.update(updated_unix=time.time(),seconds=time.time()-started);write(folder/'continuation.json',state)
    def run(cmd,name,env=None):
        with (folder/(name+'.log')).open('a') as f:subprocess.run([str(x) for x in cmd],stdout=f,stderr=subprocess.STDOUT,env=env,check=True)
    def collect(mode):
        unit=f'4dsr-temporal-prior-{mode}-20260928.service'
        remote='cd '+shlex.quote(REMOTE)+' && '+shlex.join(['python3',f'experiments/{RUN_ID}/service_wait.py','--unit',unit])
        run(['ssh',HOST,remote],mode+'_wait_exit')
        dest=OUT/'priors'/mode;dest.mkdir(parents=True,exist_ok=True)
        run(['rsync','-az',f'{HOST}:{REMOTE}/output/{RUN_ID}/priors/{mode}/',str(dest)+'/'],mode+'_return')
        index=read(dest/'prior_index.json');assert index['status']=='completed' and len(index['entries'])==1140
        for e in index['entries']:bound(e)
        return entry(dest/'prior_index.json')
    try:
        status(status='running',phase='waiting_for_prior_process_exit')
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures={pool.submit(collect,mode):mode for mode in ['repeat7','video7']}
            indices={}
            for future in concurrent.futures.as_completed(futures):
                mode=futures[future];indices[mode]=future.result();status(returned_priors=indices)
        run(['rsync','-az',f'{HOST}:{REMOTE}/output/{RUN_ID}/preparation/',str(folder/'remote')+'/'],'preparation_receipts_return')
        status(phase='registering_four_arm_protocol')
        if not (OUT/'protocol.json').exists():run([sys.executable,HERE/'register.py'],'register')
        run([sys.executable,HERE/'test_controls.py'],'control_tests')
        run([sys.executable,HERE/'readiness.py'],'readiness')
        p=require_run_root(OUT)
        paths={str(f.relative_to(ROOT)) for f in HERE.glob('*.py')}
        assets=[p[k] for k in ['parent','manifest','teacher','lr_curve','old_schedule']]+list(p['schedules'].values())+list(p['priors'].values())+p['training_files']
        paths.update(e['path'] for e in assets)
        paths.update(x for x in p['sources'] if not x.startswith('upstream/'))
        for name in ['protocol.json','research_readiness.json','control_tests.json']:paths.add(str((OUT/name).relative_to(ROOT)))
        # New code and new run assets are copied; pre-existing shared files must match.
        listing=folder/'training_files.txt';listing.write_text('\n'.join(sorted(paths))+'\n')
        protected=[dict(path=rel,sha256=sha(ROOT/rel)) for rel in paths
                   if not rel.startswith('experiments/'+RUN_ID+'/') and not rel.startswith('output/'+RUN_ID+'/')]
        protected_path=folder/'protected_assets.json';write(protected_path,protected)
        run(['rsync','-az',str(protected_path),f'{HOST}:{REMOTE}/output/{RUN_ID}/preparation/protected_assets.json'],'protected_assets_manifest')
        guard='''import json,hashlib
from pathlib import Path
bad=[]
for e in json.loads(Path(INPUT).read_text()):
 p=Path(e['path'])
 if p.exists() and hashlib.sha256(p.read_bytes()).hexdigest()!=e['sha256']:bad.append(str(p))
assert not bad,('Existing shared files differ; no overwriting',bad)
'''.replace('INPUT',repr(str(protected_path.relative_to(ROOT))))
        run(['ssh',HOST,'cd '+shlex.quote(REMOTE)+' && python3 -c '+shlex.quote(guard)],'protect_shared_assets')
        run(['rsync','-az','--files-from='+str(listing),str(ROOT)+'/',f'{HOST}:{REMOTE}/'],'training_incremental_sync')
        verify_code='''import json,hashlib,os
from pathlib import Path
p=json.loads(Path("output/RUN/protocol.json").read_text());bad=[]
entries=[p[k] for k in ["parent","manifest","teacher","lr_curve","old_schedule"]]+list(p["schedules"].values())+list(p["priors"].values())+p["training_files"]
entries += [dict(path=("vendor/4dgs/"+k[9:]) if k.startswith("upstream/") else k,sha256=v) for k,v in p["sources"].items()]
for e in entries:
 f=Path(e["path"])
 if not f.is_file() or hashlib.sha256(f.read_bytes()).hexdigest()!=e["sha256"]:bad.append(str(f))
assert not bad,bad
from runtime_identity import identity
assert identity()==p["runtime"]["training"]
print(json.dumps(dict(status="passed",files=len(entries),runtime=identity())))
'''.replace('RUN',RUN_ID)
        shell='cd '+shlex.quote(REMOTE)+' && source activate_a100.sh && PYTHONPATH='+shlex.quote(str(Path('experiments')/RUN_ID))+' python -c '+shlex.quote(verify_code)
        run(['ssh',HOST,shell],'training_remote_verify')
        receipt=json.loads((folder/'training_remote_verify.log').read_text().splitlines()[-1]);write(OUT/'deployment_verified.json',receipt)
        status(phase='four_arm_training_and_evaluation')
        run([sys.executable,'-u',HERE/'run_batch.py'],'run_batch')
        run([sys.executable,'-u',HERE/'finalize.py'],'final_integrity')
        status(status='completed',phase='results_ready_for_visual_review',quality_summary=entry(OUT/'quality_summary.json'))
    except BaseException:
        status(status='failed',error=traceback.format_exc());raise

if __name__=='__main__':main()
