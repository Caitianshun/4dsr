"""Incremental legal-input A100 deployment with mandatory remote SHA validation."""
from dv_common import *
import argparse
import subprocess
import shlex

def remote_verify():
    p=require_run_root(OUT)
    files=read(OUT/'deployment_files.json')['files']
    for e in files:bound(e)
    from readiness import validate
    validate(p)
    from runtime_identity import identity
    assert identity()==p['runtime']['training']
    write(OUT/'remote_verification.json',dict(status='passed',file_count=len(files),
        protocol=entry(OUT/'protocol.json'),runtime=identity(),host='a100-train',time_unix=time.time()))
    print(json.dumps(dict(status='passed',files=len(files))))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--remote-verify',action='store_true');a=ap.parse_args()
    if a.remote_verify:return remote_verify()
    p=require_run_root(OUT);paths={}
    def add(e):paths[e['path']]=dict(path=e['path'],sha256=e['sha256'])
    for e in p['training_files']:add(e)
    for k in ['manifest','parent','teacher','lr_curve','old_schedule','control_tests']:add(p[k])
    for e in p['schedules'].values():add(e)
    add(p['texture']['maps']);add(p['texture']['calibration'])
    for rel,h in p['sources'].items():
        if not rel.startswith('upstream/'):add(dict(path=rel,sha256=h))
    for file in HERE.glob('*.py'):add(entry(file))
    for name in ['protocol.json','frozen_protocol.json','audit_manifest.json','diagnosis_decision.json']:add(entry(OUT/name))
    # Imported generic adapters are small; include their verified source, no output/data expansion.
    for folder in ['dynamic_sr_detail_supervision_20260924','dynamic_sr_motion_bound_20260923',
        'dynamic_sr_soft_motion_20260924','dynamic_sr_20260918','dynamic_sr_20260920',
        'dynamic_sr_prior_guidance_20260927','dynamic_sr_dynamic_validation_20260927',
        'dynamic_sr_controlled_headroom_20260926']:
        for file in (ROOT/'experiments'/folder).glob('*.py'):add(entry(file))
    write(OUT/'deployment_files.json',dict(files=list(paths.values()),scope='current generic SR code, training LR/SwinIR, frozen legal maps and parent only'))
    files=OUT/'deployment_files.txt';files.write_text('\n'.join(sorted(paths))+'\n'+str((OUT/'deployment_files.json').relative_to(ROOT))+'\n')
    log=OUT/'deployment';log.mkdir(exist_ok=True)
    remote='/home/ubuntu/3DGS/4dsr';host='a100-train'
    cmd=['rsync','-az','--files-from='+str(files),str(ROOT)+'/',host+':'+remote+'/']
    with (log/'sync.log').open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
    shell='cd '+shlex.quote(remote)+' && source activate_a100.sh && '+shlex.join(['python',f'experiments/{RUN_ID}/deploy.py','--remote-verify'])
    with (log/'verify.log').open('w') as f:subprocess.run(['ssh','-o','BatchMode=yes',host,shell],stdout=f,stderr=subprocess.STDOUT,check=True)
    subprocess.run(['rsync','-az',f'{host}:{remote}/output/{RUN_ID}/remote_verification.json',str(OUT/'remote_verification.json')],check=True)
    write(log/'complete.json',dict(status='verified',files=len(paths),commands=cmd,transport=host,
        remote_root=remote,returned_verification=entry(OUT/'remote_verification.json')))

if __name__=='__main__':main()
