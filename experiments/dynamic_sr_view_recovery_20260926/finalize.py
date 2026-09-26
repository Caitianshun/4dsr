"""Wait once on controller process exit, verify artifacts, export fixed visuals."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import select
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'output/dynamic_sr_view_recovery_20260926'

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(v,indent=2));tmp.replace(p)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--pid',type=int,required=True);a=parser.parse_args()
    out=OUT/'final_v1';out.mkdir(exist_ok=False);started=time.time()
    write(out/'status.json',dict(status='waiting_for_controller_exit',pid=os.getpid(),controller_pid=a.pid))
    try:
        try:fd=os.pidfd_open(a.pid)
        except ProcessLookupError:fd=None
        if fd is not None:
            try:poll=select.poll();poll.register(fd,select.POLLIN);poll.poll()
            finally:os.close(fd)
        controller=read(OUT/'controller_v1/status.json');assert controller['status']=='completed',controller
        methods=read(OUT/'methods.json')['methods'];new_names=list(methods)
        old=read(ROOT/'output/dynamic_sr_controlled_headroom_20260926/final_v1/methods.json')['methods']
        methods={**{k:old[k] for k in ['U6000','U18000','U40000']},**methods}
        write(out/'methods.json',dict(methods=methods));audits=[]
        schedule=read(OUT/'protocol.json')['schedule'];schedule=read(schedule['path']);frozen=[]
        for name in new_names:
            item=methods[name];cp=Path(item['checkpoint']);train=Path(item['train_dir']);config=read(train/'config.json');complete=read(train/'complete.json');step=item['updates']
            assert sha(cp)==item['sha256']==read(train/f'checkpoint_{step}.json')['sha256']
            h=hashlib.sha256()
            for li,si,_ in schedule['rows'][:step]:h.update(f'{li},{si}\n'.encode())
            assert complete['metadata']['draw_sha256']==h.hexdigest()
            assert complete['actual_updates']==(12000 if step==18000 else 22000)
            for source in config['sources']:assert sha(source['path'])==sha(source['snapshot'])==source['sha256']
            reads=read(train/'image_reads.json')['actual_high_resolution_opens']
            allowed={str((ROOT/'data/dynamic_sr/n3dv_prepared/cook_spinach'/e['relative_path']).resolve()) for e in config['target_inputs']}
            assert set(reads)<=allowed and all('/sr_swinir_x4/' in p for p in reads)
            check=read(train/'freeze_audit.json');assert len(check['checks'])>=2
            if item['arm']=='F_app':
                for q in check['checks']:
                    assert q['frozen_state']==check['frozen_reference'] and q['gamma']==check['gamma_reference'] and q['color_changed'] and q['all_frozen_grad_none']
                frozen.append(dict(method=name,actual_trainable=check['policy']['actual_trainable'],frozen_reference=check['frozen_reference'],gamma=check['gamma_reference']))
            for split,path in item['evaluations'].items():
                folder=Path(path);receipt=read(folder/'complete.json');metric=read(folder/'metrics.json')
                assert receipt['metrics_sha256']==sha(folder/'metrics.json')
                assert metric['checkpoint_sha256']==sha(cp) and len(metric['rows'])==(16 if split=='train_fixed' else 60)
                assert metric['gpu']==config['gpu']
                audits.append(dict(method=name,split=split,metrics_sha256=sha(folder/'metrics.json'),observations=len(metric['rows'])))
            metric=read(Path(item['train76'])/'metrics.json');assert len(metric['rows'])==76 and metric['checkpoint_sha256']==sha(cp)
            audits.append(dict(method=name,split='train76',observations=76,metrics_sha256=sha(Path(item['train76'])/'metrics.json')))
        write(OUT/'artifact_audit.json',dict(status='passed',evaluations=audits,frozen=frozen,completed_unix=time.time(),
            sampling_suffix_equal=True,original_source_unchanged=True,image_read_boundary_passed=True))
        write(out/'status.json',dict(status='exporting_fixed_views',pid=os.getpid()))
        p=read(OUT/'protocol.json');roi=ROOT/'output/dynamic_sr_soft_motion_20260924/roi_registry/cook_spinach/roi_protocol.json'
        command=[sys.executable,'-u',str(Path(__file__).with_name('export_views.py')),'--manifest',p['manifest']['path'],'--methods',str(out/'methods.json'),
            '--teacher-index',p['teacher']['path'],'--roi-protocol',str(roi),'--out',str(out/'views')]
        with (out/'export.log').open('w') as log:subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        write(out/'status.json',dict(status='completed_pending_visual_scientific_review',seconds=time.time()-started,completed_unix=time.time(),
            auto_report='metrics and fixed images ready; no claim of chat notification or human visual review'))
    except BaseException:
        write(out/'status.json',dict(status='failed',traceback=traceback.format_exc()));raise

if __name__=='__main__':main()
