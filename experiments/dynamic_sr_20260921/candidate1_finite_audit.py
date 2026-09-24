"""Post-hoc full-scale forward linearity audit; no HR input or fitting."""
from pathlib import Path
import importlib.util,json,hashlib,time
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'output/dynamic_sr_20260921/candidate1_identifiability_v1'
spec=importlib.util.spec_from_file_location('c',Path(__file__).with_name('candidate1_identifiability.py'));c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 start=time.monotonic();d=c.inv.aa_matrix();rng=np.random.default_rng(211904);records=[]
 for i in range(16):
  z=np.load(OUT/f'patch_{i:02d}_frozen_before_hr.npz');r=json.loads((OUT/f'patch_{i:02d}_scores_before_hr.json').read_text());n=z['Jn'].shape[1]//8
  directions=rng.normal(size=(3,n,8));yy,xx=np.mgrid[:80,:80];xx=(xx-39.5)/40;yy=(yy-39.5)/40
  for name,x,jn in [('bicubic',z['base'],z['Jn']),('swinir',z['teacher'],z['Jn_teacher'])]:
   for m in c.MOTION:
    scales=np.array([m]*6+[m*.1,m*.05]);errors=[];norms=[]
    for direction in directions:
     delta=direction*scales[None,:];pred=jn@delta.ravel();true=[np.zeros(r['operators'][0]['accepted_rows'])]
     for k,dk in enumerate(delta,1):
      co=z[f'coords_{k}'];rows=z[f'rows_{k}'];offset=np.stack([4*(dk[0]+dk[2]*xx+dk[3]*yy),4*(dk[1]+dk[4]*xx+dk[5]*yy)],axis=-1)
      old=c.forward(x,co,d,rows);new=(1+dk[6])*c.forward(x,co+offset,d,rows)+dk[7];true.append(new-old)
     true=np.concatenate(true);errors.append(float(np.linalg.norm(pred-true)/max(np.linalg.norm(true),1e-20)));norms.append(float(np.sqrt(np.mean(true**2))))
    records.append(dict(index=i,scene=r['cell']['scene'],linearization=name,motion_std_lr=m,relative_errors=errors,lr_rmse_changes=norms))
 summary=[]
 for name in ['bicubic','swinir']:
  for m in c.MOTION:
   rr=[r for r in records if r['linearization']==name and r['motion_std_lr']==m];e=np.array([v for r in rr for v in r['relative_errors']]);summary.append(dict(linearization=name,motion_std_lr=m,n_patch_direction=len(e),relative_error_median=float(np.median(e)),relative_error_p90=float(np.quantile(e,.9)),relative_error_max=float(e.max())))
 data=dict(source_sha256=sha(__file__),input_sha256={p.name:sha(p) for p in sorted(OUT.glob('patch_*_frozen_before_hr.npz'))},protocol='Post-hoc linearity audit after viewing main results; fixed seed211904, three standard-normal 8D nuisance directions per neighbor, all16patches/two linearization points/three prespecified scales, exact bilinearwarp+AA, linear gain, no HR used; fixed visibility only.',summary=summary,records=records,elapsed_seconds=time.monotonic()-start)
 (OUT/'finite_fullscale_audit.json').write_text(json.dumps(data,indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
