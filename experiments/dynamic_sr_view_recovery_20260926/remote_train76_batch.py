"""Verified, fixed remote diagnostic batch, immediate exit status, no polling."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[2]
def main():
    out=ROOT/'recovery_train76';out.mkdir(exist_ok=False);start=time.time()
    try:
        hashes=json.loads((ROOT/'cts_sha256.json').read_text())
        for p,h in hashes.items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,p
        (out/'input_hashes.json').write_text(json.dumps(hashes,indent=2))
        before=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv'],text=True)
        assert len(before.strip().splitlines())==1,before
        (out/'resource_start.txt').write_text(before+subprocess.check_output(['nvidia-smi'],text=True))
        spec=json.loads((ROOT/'cts_train76_spec.json').read_text())
        for label,checkpoint in spec.items():
            cmd=[sys.executable,'-u',str(ROOT/'experiments/dynamic_sr_view_recovery_20260926/train76.py'),
                '--manifest',str(ROOT/'data/dynamic_sr/n3dv_prepared/cook_spinach/manifest.json'),
                '--checkpoint',str(ROOT/checkpoint),'--out',str(out/label)]
            with (out/f'{label}.log').open('w') as log:subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        result=dict(status='completed',seconds=time.time()-start,host='cts',gpu=os.environ['CUDA_VISIBLE_DEVICES'],input_hashes_verified=True)
    except BaseException:
        result=dict(status='failed',seconds=time.time()-start,traceback=traceback.format_exc());raise
    finally:
        (out/'status.json').write_text(json.dumps(result,indent=2))

if __name__=='__main__':main()
