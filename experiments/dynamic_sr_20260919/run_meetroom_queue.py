"""Wait for GPU0 kitchen queue, then run the preregistered independent domain."""
import argparse,json,os,select,sys,time
from pathlib import Path
from run_control_queue import ROOT,OLD,NEW,OUT,PY,now,write,run

def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',default='0');a=p.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']=a.gpu
    os.environ['OMP_NUM_THREADS']='4';os.environ['OPENBLAS_NUM_THREADS']='4'
    scene='meetroom_discussion';state=OUT/f'{scene}_queue.json'
    if state.exists():raise FileExistsError(state)
    status=dict(scene=scene,gpu=a.gpu,pid=os.getpid(),started=now(),status='waiting_gpu',completed=[])
    write(state,status)
    manifest=ROOT/'data/dynamic_sr/meetroom_prepared/discussion/manifest.json'
    dependency=OUT/'cook_spinach_queue.json'
    try:
        parent_state=json.loads(dependency.read_text())
        if parent_state['status'] not in ['complete','failed']:
            # Kernel process-exit event, no interval/timeout/file polling.
            fd=os.pidfd_open(parent_state['pid'])
            try:select.select([fd],[],[])
            finally:os.close(fd)
            parent_state=json.loads(dependency.read_text())
        if parent_state['status']!='complete':
            raise RuntimeError('GPU predecessor did not complete; inspect before continuing')
        m=json.loads(manifest.read_text());assert m['scene']==scene
        assert m['splits']['test']==['cam00'] and m['splits']['dev']==['cam01']
        status['status']='running';status['gpu_acquired']=now();write(state,status)
        cameras='cam02,cam04,cam08,cam12'
        run(scene+'_prior',[OLD/'generate_prior.py','--manifest',manifest,'--cameras',cameras,'--device','cuda:0'],state,status)
        warm=OUT/(scene+'_native_warmup')
        run(scene+'_native_warmup',[OLD/'run_experiment.py','--task','warmup','--manifest',manifest,
            '--out',warm,'--observation','native_lr','--coarse-steps',1000,'--fine-steps',6000],state,status)
        parent=OUT/(scene+'_integrated_parent')
        run(scene+'_integrated_parent',[OLD/'run_experiment.py','--task','branch','--manifest',manifest,
            '--checkpoint',warm/'checkpoint_final.pt','--out',parent,'--mode','lr_integrated','--steps',1200,
            '--prior-cameras',cameras],state,status)
        for name,ck in [('native_warmup',warm),('integrated_parent',parent)]:
            prefix=scene+'_'+name
            run(prefix+'_eval',[OLD/'evaluate.py','--manifest',manifest,'--checkpoint',ck/'checkpoint_final.pt',
                '--out',OUT/(prefix+'_evaluation'),'--no-video'],state,status)
            run(prefix+'_sampling',[OLD/'evaluate.py','--manifest',manifest,'--checkpoint',ck/'checkpoint_final.pt',
                '--out',OUT/(prefix+'_sampling'),'--sampling-only','--no-video'],state,status)
        for mode in ['train_prior','real_lr','render_lr']:
            argv=[NEW/'reference_controls.py','--manifest',manifest,'--out',OUT/(scene+'_'+mode+'_reference'),
                  '--mode',mode,'--prior-cameras',cameras]
            if mode=='render_lr':argv+=['--checkpoint',warm/'checkpoint_final.pt']
            run(scene+'_'+mode+'_reference',argv,state,status)
        cases=[('lr_long','none',0,False),('sr_w01','sr',.1,False),('sr_w10','sr',1.,False),
               ('hr_oracle_w10','hr',1.,False),('sr_w10_dense','sr',1.,True),('lr_dense','none',0,True)]
        for case,teacher,weight,dense in cases:
            name=scene+'_'+case;out=OUT/name
            args=[NEW/'controlled_fit.py','--manifest',manifest,'--checkpoint',parent/'checkpoint_final.pt',
                  '--out',out,'--teacher',teacher,'--weight',weight,'--steps',18000,
                  '--milestones','1200,6000,12000,18000','--prior-cameras',cameras]
            if dense:args+=['--dense']
            run(name,args,state,status)
            run(name+'_eval',[OLD/'evaluate.py','--manifest',manifest,'--checkpoint',out/'checkpoint_final.pt',
                '--out',OUT/(name+'_evaluation'),'--no-video'],state,status)
            run(name+'_sampling',[OLD/'evaluate.py','--manifest',manifest,'--checkpoint',out/'checkpoint_final.pt',
                '--out',OUT/(name+'_sampling'),'--sampling-only','--no-video'],state,status)
        status.update(status='complete',finished=now(),current=None);write(state,status)
    except BaseException as e:
        status.update(status='failed',finished=now(),error=repr(e));write(state,status);raise

if __name__=='__main__':main()
