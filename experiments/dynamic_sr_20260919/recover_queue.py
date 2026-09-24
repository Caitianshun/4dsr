"""Resume missing units, preserving failed attempts and completed experiments.

Partial training is rerun from the common parent, never mislabelled as exact
checkpoint continuation. Recovery changes durability/diagnostics only.
"""
import argparse,json,os,shutil,subprocess,zipfile
from pathlib import Path
from run_control_queue import ROOT,OLD,NEW,OUT,PY,now

CASES=[('lr_long','none',0,False),('sr_w01','sr',.1,False),('sr_w10','sr',1.,False),
       ('hr_oracle_w10','hr',1.,False),('sr_w10_dense','sr',1.,True),('lr_dense','none',0,True)]
def read(path):return json.loads(Path(path).read_text())
def valid_json(path):
    try:return read(path)
    except (OSError,ValueError):return None
def durable(path):
    with path.open('rb') as f:os.fsync(f.fileno())
def write(path,obj):
    tmp=path.with_suffix('.tmp')
    with tmp.open('w') as f:
        json.dump(obj,f,indent=2);f.flush();os.fsync(f.fileno())
    tmp.replace(path)
    fd=os.open(str(path.parent),os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)
def archive(path,dest):
    if path.exists() or path.is_symlink():
        dest.mkdir(parents=True,exist_ok=True)
        target=dest/path.name
        if target.exists():raise FileExistsError(target)
        shutil.move(str(path),str(target))

def main():
    p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--gpu',required=True)
    a=p.parse_args();scene=a.scene
    os.environ.update(CUDA_VISIBLE_DEVICES=a.gpu,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',
                      PYTHONFAULTHANDLER='1',TORCH_SHOW_CPP_STACKTRACES='1')
    attempts=OUT/'recovery_archive_20260920'/scene
    state=OUT/f'{scene}_queue.json';old=read(state)
    archive(state,attempts)
    status=dict(scene=scene,gpu=a.gpu,pid=os.getpid(),started=now(),status='running',completed=[],
                recovery_from=str(attempts/state.name),reused=[],prior_status=old['status'])
    write(state,status)
    meeting=scene=='meetroom_discussion'
    manifest=ROOT/('data/dynamic_sr/meetroom_prepared/discussion/manifest.json' if meeting else f'data/dynamic_sr/n3dv_prepared/{scene}/manifest.json')
    parent=OUT/'meetroom_discussion_integrated_parent/checkpoint_final.pt' if meeting else ROOT/f'output/dynamic_sr_20260918/{scene}_pilot_v1_lr_integrated/checkpoint_final.pt'
    cams='cam02,cam04,cam08,cam12' if meeting else 'cam02,cam06,cam12,cam18'
    def execute(name,args,marker,directory):
        if valid_json(marker):
            status['reused'].append(name);status['completed'].append(name);write(state,status);return
        archive(directory,attempts);archive(OUT/(name+'.log'),attempts)
        status.update(current=name,current_started=now());write(state,status)
        with (OUT/(name+'.log')).open('x') as log:
            result=subprocess.run([str(PY),*map(str,args)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError(f'{name}: exit {result.returncode}; no automatic retry')
        if not valid_json(marker):raise RuntimeError(f'{name}: missing/invalid completion artifact')
        # Flush all already closed artifacts once before marking the unit done.
        for f in directory.rglob('*'):
            if f.is_file() and not f.is_symlink():durable(f)
        status['completed'].append(name);write(state,status)
    try:
        for case,teacher,weight,dense in CASES:
            name=scene+'_'+case;out=OUT/name
            marker=out/'complete.json';done=valid_json(marker)
            if done:
                if done['intervention_step']!=18000:raise RuntimeError(name+' wrong endpoint')
                with zipfile.ZipFile(out/'checkpoint_final.pt') as z:
                    bad=z.testzip()
                    if bad:raise RuntimeError(name+' corrupt checkpoint '+bad)
            args=[NEW/'controlled_fit.py','--manifest',manifest,'--checkpoint',parent,'--out',out,
                  '--teacher',teacher,'--weight',weight,'--steps',18000,'--milestones','1200,6000,12000,18000',
                  '--prior-cameras',cams]
            if dense:args+=['--dense']
            execute(name,args,marker,out)
            ev=OUT/(name+'_evaluation')
            execute(name+'_eval',[OLD/'evaluate.py','--manifest',manifest,'--checkpoint',out/'checkpoint_final.pt',
                '--out',ev,'--no-video'],ev/'metrics.json',ev)
            sampling=OUT/(name+'_sampling')
            execute(name+'_sampling',[OLD/'evaluate.py','--manifest',manifest,'--checkpoint',out/'checkpoint_final.pt',
                '--out',sampling,'--sampling-only','--no-video'],sampling/'sampling.json',sampling)
        status.update(status='complete',finished=now(),current=None);write(state,status)
    except BaseException as e:
        status.update(status='failed',finished=now(),error=repr(e));write(state,status);raise

if __name__=='__main__':main()
