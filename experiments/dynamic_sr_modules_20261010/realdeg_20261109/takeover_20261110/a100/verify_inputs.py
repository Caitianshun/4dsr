"""CPU byte identity admission for the frozen RD A100 queue; no scientific changes."""
from pathlib import Path
import json, hashlib, sys, time
ROOT=Path(__file__).resolve().parents[5]
BASE=ROOT/'experiments/dynamic_sr_modules_20261010/realdeg_20261109/baselines'
sys.path.insert(0,str(BASE))
from rd_common import metadata, SCENES

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def main():
 expect=json.loads(Path(__file__).with_name('expected_assets.json').read_text())
 result={'schema':'rd4_a100_inputs_admission_v1','created_unix':time.time(),'status':'checking','root':str(ROOT),'HR_pixels_read':False,'cam00_pixels_read':False,'assets':[],'scenes':[]}
 for r in expect['assets']:
  p=ROOT/r['path']; actual=sha(p)
  if actual!=r['sha256']: raise ValueError('Reference SHA mismatch: '+str(p))
  result['assets'].append(dict(path=r['path'],sha256=actual,bytes=p.stat().st_size))
 for scene in SCENES:
  mp,m=metadata(scene,verify_init=True);count=0
  for row in m['observations']:
   if row['split']!='train':continue
   p=mp.parent/row['lr_path']
   if sha(p)!=row['lr_sha256']:raise ValueError('LR mismatch: '+str(p))
   count+=1
  result['scenes'].append(dict(scene=scene,manifest_sha256=sha(mp),train_LR_verified=count,initialization_verified=True))
 result['status']='passed'
 out=ROOT/'output/dynamic_sr_modules_20261010/realdeg_20261109/takeover_20261110/a100/inputs_admission.json'
 out.parent.mkdir(parents=True,exist_ok=True)
 with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps(result),flush=True)
if __name__=='__main__':main()
