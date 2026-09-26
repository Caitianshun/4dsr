"""Local PRO6000 prefixes; dispatch verified 3090 continuations before eval."""
import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys
import traceback
from run_segment import ROOT,leases,train,evaluate


def main():
    p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True);a=p.parse_args();plan=leases.read(a.spec)
    root=ROOT/plan['out'];root.mkdir(parents=True,exist_ok=False)
    state=dict(status='starting',pid=os.getpid(),started_utc=leases.stamp());leases.write(root/'status.json',state)
    try:
        for method in ['Z','O']:
            s=plan['jobs'][method];out=ROOT/s['out'];out.mkdir(parents=True,exist_ok=False)
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=s['gpu'],OMP_NUM_THREADS='4')
            with leases.lock_file(f'/tmp/4dsr_soft_gpu_{s["gpu"]}.lock'):
                state['status']=method+'_prefix';leases.write(root/'status.json',state);train(s,out,env)
                checkpoint=out/'train/checkpoint_6000.pt';identity=leases.identity(checkpoint)
                remote=plan['remote_root'];relative=checkpoint.relative_to(ROOT);host=s.get('tail_host','cts')
                unit=f'headroom-{method.lower()}-tail-20260926-v1';script=Path(remote)/'experiments/dynamic_sr_controlled_headroom_20260926/run_segment.py'
                if host=='cts':
                    subprocess.run(['ssh','cts','mkdir -p '+shlex.quote(str(Path(remote)/relative.parent))],check=True)
                    subprocess.run(['rsync','-a',str(checkpoint),'cts:'+str(Path(remote)/relative)],check=True)
                    check=subprocess.check_output(['ssh','cts','sha256sum '+shlex.quote(str(Path(remote)/relative))],text=True).split()[0];assert check==identity['sha256']
                    command='source /home/cts/Project/4DSR/activate_cts.sh; export FOURDSR_UPSTREAM=/home/cts/Project/4DSR/vendor/4dgs; exec /home/cts/Project/4DSR/.venv/bin/python -u '+shlex.quote(str(script))+' --spec '+shlex.quote(str(Path(remote)/s['remote_spec']))
                    subprocess.run(['ssh','cts','systemd-run --user --unit='+unit+' --property=WorkingDirectory='+shlex.quote(remote)+' --property=StandardOutput=append:'+shlex.quote(remote+'/output/dynamic_sr_controlled_headroom_20260926/'+method+'_tail_service.log')+' --property=StandardError=inherit /bin/bash -c '+shlex.quote(command)],check=True)
                    pid=int(subprocess.check_output(['ssh','cts','systemctl --user show '+unit+' -p MainPID --value'],text=True).strip())
                else:
                    assert host=='local'
                    subprocess.run(['systemd-run','--user','--unit='+unit,'--property=WorkingDirectory='+str(ROOT),
                        '--property=StandardOutput=append:'+str(root/(method+'_tail_service.log')),'--property=StandardError=inherit',
                        '--setenv=FOURDSR_UPSTREAM='+plan['wu_upstream'],plan['wu_python'],'-u',str(Path(__file__).with_name('run_segment.py')),
                        '--spec',str(ROOT/s['remote_spec'])],check=True)
                    pid=int(subprocess.check_output(['systemctl','--user','show',unit,'-p','MainPID','--value'],text=True).strip())
                assert pid>0
                return_unit=f'headroom-{method.lower()}-return-20260926-v1'
                return_command=['systemd-run','--user','--unit='+return_unit,'--property=WorkingDirectory='+str(ROOT),
                    '--property=StandardOutput=append:'+str(root/(method+'_return_service.log')),'--property=StandardError=inherit',
                    '/usr/bin/python3','-u',str(Path(__file__).with_name('return_tail.py')),'--spec',str(a.spec.resolve()),'--method',method,'--pid',str(pid)]
                subprocess.run(return_command,check=True)
                return_pid=int(subprocess.check_output(['systemctl','--user','show',return_unit,'-p','MainPID','--value'],text=True).strip());assert return_pid>0
                leases.write(root/(method+'_remote.json'),dict(unit=unit,pid=pid,host=host,checkpoint=identity,remote_root=remote,return_unit=return_unit,return_pid=return_pid))
                evaluate(s,out,checkpoint,6000,env)
                leases.write(out/'complete.json',dict(status='completed_and_evaluated',method=method,updates=6000,checkpoint=identity))
        state.update(status='prefixes_evaluated_tails_dispatched',finished_utc=leases.stamp());leases.write(root/'complete.json',state)
    except BaseException:
        state.update(status='failed',traceback=traceback.format_exc());leases.write(root/'failed.json',state);raise
    finally:leases.write(root/'status.json',state)


if __name__=='__main__':main()
