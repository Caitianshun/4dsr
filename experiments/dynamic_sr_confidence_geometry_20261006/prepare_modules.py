"""Dependency exit triggers verified module deployment and the six remaining endpoints."""
import argparse,subprocess,select,traceback
from cg_common import *
def wait_pid(pid):
    try:fd=os.pidfd_open(pid)
    except ProcessLookupError:return
    select.select([fd],[],[]);os.close(fd)
def service(name,command):
    subprocess.run(['systemd-run','--user','--unit='+name,'--property=WorkingDirectory='+str(ROOT),
      '/usr/bin/python3','-u',str(HERE/'worker.py'),*command],check=True)
def main(a):
    wait_pid(a.calibration_pid)
    cal=read(OUT/'calibration.json');assert cal['status']=='passed'
    assert read(OUT/'geometry_cuda_checks_v2.json')['gpu']['status']=='passed'
    for path in [OUT/'cache/geometry/index.json',OUT/'cache/confidence_view/cache_manifest.json']:
        value=read(path);assert len(value['entries'])==1140
        for e in value['entries']:assert sha(path.parent/e['path'])==e['sha256']
    # Prefetch is a separate process; finish it before reusing its inventory path.
    if a.prefetch_pid:wait_pid(a.prefetch_pid)
    write(OUT/'module_dispatch_state.json',dict(status='syncing_complete_frozen_modules',k_R=cal['k_R'],lambda_G=cal['lambda_G']))
    with (OUT/'module_deployment.log').open('w') as log:
        subprocess.run(['/usr/bin/python3',str(HERE/'deploy.py')],stdout=log,stderr=subprocess.STDOUT,check=True)
    service('4dsr-cg-local-modules-r1-20261006',['--host','local','--gpu','0','--repeat','1','--methods','R','G','RG','--training-dependency-pid',str(a.local_base_pid)])
    service('4dsr-cg-a100-modules-r2-20261006',['--host','a100-train','--gpu','1','--repeat','2','--methods','R','G','RG','--training-dependency-pid',str(a.remote_base_pid)])
    write(OUT/'module_dispatch_state.json',dict(status='dispatched_dependency_safe',methods=['R','G','RG'],
      repeats=['1','2'],coefficients=entry(OUT/'calibration.json'),remote_verification=entry(OUT/'remote_verification.json')))
if __name__=='__main__':
    ap=argparse.ArgumentParser()
    for name in ['calibration-pid','prefetch-pid','local-base-pid','remote-base-pid']:ap.add_argument('--'+name,type=int,required=True)
    a=ap.parse_args()
    try:main(a)
    except BaseException:write(OUT/'module_dispatch_state.json',dict(status='failed',traceback=traceback.format_exc()));raise
