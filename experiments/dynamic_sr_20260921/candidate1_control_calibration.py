"""Control-only covariance calibration, with no changes to real-data scores."""
from pathlib import Path
import importlib.util,inspect,json,hashlib,time
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'output/dynamic_sr_20260921/candidate1_identifiability_v1'
spec=importlib.util.spec_from_file_location('c',Path(__file__).with_name('candidate1_identifiability.py'));c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 start=time.monotonic();source=inspect.getsource(c.controlled)
 # d is uniform on the 32D unit sphere: Cov(d)=I/32, hence precision 32I.
 # Retain identical data, noise, nuisance priors and all information scores.
 assert source.count('h+np.eye(32)')==1 and source.count('e+np.eye(32)')==1
 source=source.replace('h+np.eye(32)','h+32*np.eye(32)').replace('e+np.eye(32)','e+32*np.eye(32)')
 (OUT/'control_calibration_generated_function.py').write_text(source)
 namespace=dict(c.__dict__);exec(compile(source,'control_calibration_generated_function.py','exec'),namespace)
 z=np.load(OUT/'patch_00_frozen_before_hr.npz');res=namespace['controlled'](z['base'],c.inv.aa_matrix(),z['B']);old=json.loads((OUT/'controls_before_real_hr.json').read_text())
 checks=[]
 for a,b in zip(old,res):
  assert a['name']==b['name']
  err=max(abs(x[k]-y[k]) for x,y in zip(a['cases'],b['cases']) for k in ['df_known','df_effective','trace_known','trace_effective']);assert err<1e-10
  checks.append(dict(name=a['name'],information_unchanged_max_abs=err,original_known_mse=a['cases'][1]['known_coeff_mse'],calibrated_known_mse=b['cases'][1]['known_coeff_mse'],original_effective_mse=a['cases'][1]['effective_coeff_mse'],calibrated_effective_mse=b['cases'][1]['effective_coeff_mse']))
 data=dict(source_sha256=sha(__file__),generated_source_sha256=sha(OUT/'control_calibration_generated_function.py'),parent_source_sha256=sha(c.__file__),protocol='Post-hoc technical calibration only after seeing unexpected control estimator error. Derive 32I precision analytically from fixed unit-sphere coefficient covariance I/32. No HR tuning, no actual-LR score changes, same seed/texture/noise/masks/perturbations. Unit covariance is a second-moment match, not Gaussian ground-truth claim.',controls=res,checks=checks,elapsed_seconds=time.monotonic()-start)
 (OUT/'control_covariance_calibration.json').write_text(json.dumps(data,indent=2));print(json.dumps(checks,indent=2))
if __name__=='__main__':main()
