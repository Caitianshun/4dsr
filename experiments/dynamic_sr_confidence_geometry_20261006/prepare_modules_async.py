"""Validated coefficient exit triggers source-bound async module workers."""
import argparse, subprocess, traceback
from cg_common import *
from worker_async import wait_pid

def launch(name, command):
    subprocess.run(['systemd-run','--user','--unit='+name,'--property=Type=exec',
      '--property=WorkingDirectory='+str(ROOT),'/usr/bin/python3','-u',str(HERE/'worker_async.py'),*command],check=True)
    return int(subprocess.check_output(['systemctl','--user','show',name,'--property=MainPID','--value'],text=True))

def main(a):
    wait_pid(a.calibration_pid)
    cal=read(OUT/'calibration.json'); assert cal['status']=='passed'
    assert read(OUT/'geometry_cuda_checks_v2.json')['gpu']['status']=='passed'
    for path in [OUT/'cache/geometry/index.json',OUT/'cache/confidence_view/cache_manifest.json']:
        value=read(path); assert len(value['entries'])==1140
        for item in value['entries']: assert sha(path.parent/item['path'])==item['sha256']
    wait_pid(a.prefetch_pid)
    write(OUT/'module_dispatch_state.json',dict(status='syncing_complete_frozen_modules_async',calibration=entry(OUT/'calibration.json')))
    with (OUT/'module_deployment.log').open('a') as log:
        subprocess.run(['/usr/bin/python3',str(HERE/'deploy.py')],stdout=log,stderr=subprocess.STDOUT,check=True)
    local_unit='4dsr-cg-local-modules-async-r1-20261006'
    remote_unit='4dsr-cg-a100-modules-async-r2-20261006'
    lp=launch(local_unit,['--host','local','--gpu','0','--repeat','1','--methods','R','G','RG',
                         '--training-dependency-pid',str(a.local_base_pid)])
    rp=launch(remote_unit,['--host','a100-train','--gpu','1','--repeat','2','--methods','R','G','RG',
                          '--training-dependency-pid',str(a.remote_base_pid)])
    write(OUT/'module_dispatch_state.json',dict(status='dispatched_dependency_safe_async',methods=['R','G','RG'],
      repeats=['1','2'],coefficients=entry(OUT/'calibration.json'),remote_verification=entry(OUT/'remote_verification.json'),
      local_unit=local_unit,remote_unit=remote_unit,local_worker_pid=lp,remote_worker_pid=rp,
      base_local_pid=a.local_base_pid,base_remote_pid=a.remote_base_pid,
      fixed_training_unchanged=True,evaluation_decoupled=True))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ['calibration-pid','prefetch-pid','local-base-pid','remote-base-pid']:
        p.add_argument('--'+key,type=int,required=True)
    a=p.parse_args()
    try: main(a)
    except BaseException:
        write(OUT/'module_dispatch_state.json',dict(status='failed',traceback=traceback.format_exc()));raise
