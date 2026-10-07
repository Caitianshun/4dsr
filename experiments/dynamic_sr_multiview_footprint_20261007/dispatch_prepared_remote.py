"""Start the closed remote suffix while its immutable preparation is returned.

No local preparation gate is published early. The original owned-worker
identity is unchanged, so the registered paired continuation later attaches
the same invocation instead of duplicating work. This launcher is CPU-only.
"""
import argparse,json,os,shlex,subprocess,sys,time
from pathlib import Path
import advance_after_preparation as advance
import completion
from fp_common import ROOT,HERE,OUT,read,write,entry,bound

REMOTE_METADATA = r"""
import hashlib,json,os,pathlib,sys
root=pathlib.Path(sys.argv[1]);out=root/sys.argv[2]
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def load(p):return json.loads(p.read_text())
def check(e):
 p=root/e['path'];assert p.is_relative_to(root) and digest(p)==e['sha256'];return load(p)
p=out/'preparation/complete.json';prep=load(p);assert prep['status']=='completed_native_support_single_calibration' and prep['formal_updates']==0
values={k:check(prep[k]) for k in ('protocol','fixture','parent','support','calibration')}
assert values['calibration']['status']=='passed'
for relative,expected in prep['sources'].items():
 source=pathlib.Path(os.environ['FOURDSR_UPSTREAM'])/relative[9:] if relative.startswith('upstream/') else root/relative
 assert digest(source)==expected,relative
print(json.dumps(dict(status='closed_remote_preparation_metadata_SHA_verified',workspace=str(root),preparation=prep,preparation_sha256=digest(p),calibration=values['calibration'],formal_updates=0,GPU_calls=0)))
"""

def main(a):
 a.registration=a.registration.resolve()
 cfg=advance.validate_config(read(a.registration));advance.validate_evaluator(cfg,advance.LocalBackend(cfg).properties(cfg['units']['evaluation']['name']))
 transport=completion.Transport(cfg['remote_host'],cfg['workspace'],OUT,cfg['remote_cpu_python'])
 unit='4dsr-footprint-native-preparation-accelerated-a100-gpu1-20261007.service';invocation='025242a6002c4d7780f530687907f1cf'
 try:
  terminal=None
  for event in transport.stream('wait-unit',unit,invocation,'user'):
   if event.get('kind')=='terminal':terminal=event
  advance.terminal_ok(terminal,dict(name=unit,invocation_id=invocation))
  command=[cfg['remote_cpu_python'],'-c',REMOTE_METADATA,cfg['workspace'],completion.REL_OUT]
  shell='source '+shlex.quote(cfg['remote_activate'])+' && '+shlex.join(command)
  result=subprocess.run(transport.ssh+['bash -lc '+shlex.quote(shell)],capture_output=True,text=True)
  if result.returncode:raise RuntimeError('Remote immutable preparation metadata failed: '+result.stderr)
  metadata=json.loads(result.stdout)
  assert metadata['workspace']==cfg['workspace'] and metadata['preparation']['protocol']==entry(OUT/'protocol.json')
  snapshot=OUT/'completion/remote_closed_preparation_metadata.json';write(snapshot,dict(**metadata,terminal=terminal,launcher_source=entry(Path(__file__))))
  # Read only this exact closed remote metadata for dispatch identity. Never
  # fabricate the local preparation complete marker or skip asset return.
  original_read=advance.read
  def exact_closed_metadata(path):
   return metadata['preparation'] if Path(path)==OUT/'preparation/complete.json' else original_read(path)
  advance.read=exact_closed_metadata
  engine=advance.Advance(cfg,entry(a.registration))
  try:worker=engine.dispatch('2')
  finally:advance.read=original_read;engine.remote.close()
  write(OUT/'completion/prepared_remote_early_dispatch.json',dict(status='registered_owned_remote_suffix_before_local_immutable_return',worker=worker,remote_preparation=entry(snapshot),local_preparation_gate_published=False,uniform_evaluator=entry(OUT/'completion/registration.json'),source=entry(Path(__file__)),formal_updates=0,finished_unix=time.time()))
  print(json.dumps(dict(status='registered_owned_remote_suffix_before_local_immutable_return',unit=worker['name'],invocation_id=worker['invocation_id'],pid=worker['main_pid'],GPU=cfg['hardware']['r2']['GPU'])))
 finally:transport.close()
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--registration',type=Path,required=True);a=p.parse_args()
 try:main(a)
 except BaseException as error:
  import traceback
  write(OUT/'completion/prepared_remote_early_dispatch_failure.json',dict(status='failed_saved_no_success_report',error=repr(error),traceback=traceback.format_exc(),source=entry(Path(__file__)),formal_updates_claimed=0));raise
