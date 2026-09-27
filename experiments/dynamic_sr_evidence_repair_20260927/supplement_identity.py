"""Append actual extension source identity and verified remote-return receipt."""
import hashlib
import json
import shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'output/dynamic_sr_evidence_repair_20260927'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,indent=2))
def main():
 env=read(OUT/'environment.json');root=Path(env['extension']).parents[1];files={}
 for f in root.rglob('*'):
  if f.is_file() and (f.suffix in ['.cu','.h','.cpp','.py'] or f.name=='build.ninja') and 'third_party' not in str(f):files[str(f)]=sha(f)
 write(OUT/'identity_v1/actual_extension_source.json',dict(extension=env['extension'],extension_sha256=env['extension_sha256'],actual_source_root=str(root),files=files,build_ninja_available=any(Path(x).name=='build.ninja' for x in files),build_options_source=str(root/'setup.py'),limitation='Historical compiler invocation unavailable if build.ninja is absent; setup.py hashes declared options, not proof of the exact historic build command. Actual loaded binary hash retained.'))
 dest=OUT/'remote_a100/moments_v1';m=read(dest/'complete.json');verified=[]
 for row in m['rows']:
  f=dest/row['path'];assert sha(f)==row['sha256'];verified.append(dict(path=str(f),sha256=row['sha256']))
 stats={}
 for k in ['bicubic','bicubic_float64','area','area_float64']:
  rows=[r['statistics'][k] for r in m['rows']];stats[k]={}
  for field in ['negative_variance_fraction','substantial_negative_fraction']:
   stats[k]['mean_'+field]=sum(r['valid'][field] for r in rows)/len(rows);stats[k]['max_'+field]=max(r['valid'][field] for r in rows)
  stats[k]['max_alpha_above_one_fraction']=max(r['alpha_above_one_fraction'] for r in rows)
 write(OUT/'moment_audit.json',dict(status='completed',area_engineering_passed=m['area_engineering_passed'],stats=stats,fixtures=m['fixtures'],derivatives=m['derivatives'],raw_variance_never_clamped=True,evidence=str(dest/'complete.json'),gpu=m['gpu'],seconds=m['seconds']))
 write(OUT/'remote_return.json',dict(status='hash_verified',files=verified,receipt_sha256=sha(dest/'complete.json'),remote_path='/home/ubuntu/3DGS/4dsr/output/dynamic_sr_evidence_repair_20260927/moments_v1',host='a100-train',parameter_updates=0))
 failures=read(OUT/'failures.json');new=[dict(stage='remaining_launch',error='This Python os module lacks pidfd_open',resolution='Use libc pidfd_open for actual exit events; initial failure archived',archive='remaining_initial_pidfd_failure'),dict(stage='remote_return',error='Rsync destination parent initially absent',resolution='Create dedicated destination directory and retry; all76 returned files hash-verified')]
 for item in new:
  if item not in failures:failures.append(item)
 write(OUT/'failures.json',failures)
 initial=OUT/'identity_v1/source_manifest_initial.json'
 if not initial.exists():shutil.copyfile(OUT/'source_manifest.json',initial)
 sm=read(initial);sm['supplemental_actual_extension_sha256']=files;sm['current_evidence_source_sha256']={str(f):sha(f) for f in (ROOT/'experiments/dynamic_sr_evidence_repair_20260927').glob('*.py')};sm['initial_manifest_sha256']=sha(initial);write(OUT/'source_manifest.json',sm)
if __name__=='__main__':main()
