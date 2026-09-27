"""Use an exit event, then finish raw ROI evaluation and machine receipts."""
import ctypes
import errno
import fcntl
import select
import subprocess
import traceback
import time
from context import *
def main():
 out=OUT/'completion_chain';out.mkdir(exist_ok=False);pid=int(sys.argv[1]);started=time.time()
 def state(**k):write_json(out/'status.json',dict(pid=os.getpid(),seconds=time.time()-started,**k))
 try:
  state(status='waiting_for_process_exit',after_pid=pid);lib=ctypes.CDLL(None,use_errno=True);fd=lib.pidfd_open(pid,0)
  if fd>=0:select.select([fd],[],[]);os.close(fd)
  elif ctypes.get_errno()!=errno.ESRCH:raise OSError(ctypes.get_errno())
  assert read(OUT/'remaining/status.json')['status']=='completed'
  lock=(OUT/'pro6000.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);env={**os.environ,'CUDA_VISIBLE_DEVICES':read(OUT/'protocol.json')['physical_gpu']}
  for script in ['audit_roi.py','audit_boundaries.py','supplement_identity.py','summarize_evidence.py']:
   state(status='running',phase=script)
   with (out/(script+'.log')).open('w') as f:r=subprocess.run([sys.executable,'-u',HERE/script],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
   assert r.returncode==0,script
  state(status='completed',phase='machine_receipts_complete')
 except BaseException:state(status='failed',error=traceback.format_exc());raise
if __name__=='__main__':main()
