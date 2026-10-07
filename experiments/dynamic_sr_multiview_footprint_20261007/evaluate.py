"""Reuse the unchanged float/FFT/ROI/LPIPS evaluator on all 196 observations.

The historical evaluator source is imported unchanged. Only its output/protocol
root is redirected to this isolated experiment. No training source or prior
results are modified, and no development pixel enters training or calibration.
"""
import argparse,traceback,time
from fp_common import ROOT,HERE,OUT,read,write,sha,entry,bound,module,setup,Path

def verify_extra(a,source):
 complete=read(a.out/'complete.json');p=read(OUT/'protocol.json')
 expected=dict(checkpoint_sha256=sha(a.checkpoint),manifest_sha256=p['manifest']['sha256'],
  source_sha256=sha(source),label=a.label,protocol_sha256=sha(OUT/'protocol.json'))
 assert complete['status']=='completed_extra_evaluation' and complete['identity']==expected
 assert complete['observations']==196 and complete['parameter_updates']==0
 for identity in complete['results'].values():bound(identity)
 for relative,identity in complete['sources'].items():assert sha(ROOT/relative)==identity
 index=read(a.out/'float_index.json');assert index['identity']==expected and len(index['entries'])==196
 expected_keys={(c,f) for c in ('cam00','cam01') for f in range(0,120,2)}
 expected_keys|={(f'cam{c:02d}',f) for c in range(2,21) for f in (0,40,80,118)}
 assert {(e['camera'],int(e['frame'])) for e in index['entries']}==expected_keys
 for observation in index['entries']:
  assert sha(a.out/observation['path'])==observation['sha256']
  assert observation['checkpoint_sha256']==expected['checkpoint_sha256']
 return complete

def main(a):
 setup()
 source=ROOT/'experiments/dynamic_sr_confidence_geometry_20261006/evaluate_extra.py'
 evaluator=module('fp_unchanged_float_evaluator',source)
 evaluator.OUT=OUT
 a.out=a.out or OUT/'evaluation'/a.label/'extra'
 if (a.out/'adapter_complete.json').exists():
  receipt=read(a.out/'adapter_complete.json')
  assert receipt['adapter']==entry(Path(__file__)) and receipt['original_evaluator']==entry(source)
  assert receipt['protocol']==entry(OUT/'protocol.json')
  bound(receipt['extra_complete']);assert receipt['checkpoint']==entry(a.checkpoint)
  verify_extra(a,source)
  return
 started=time.monotonic()
 evaluator.main(a)
 complete=verify_extra(a,source)
 write(a.out/'adapter_complete.json',dict(status='completed_unchanged_uniform_float_evaluation',
  checkpoint=entry(a.checkpoint),protocol=entry(OUT/'protocol.json'),adapter=entry(Path(__file__)),
  original_evaluator=entry(source),extra_complete=entry(a.out/'complete.json'),
  observations=196,development_observations=120,train76=76,parameter_updates=0,
  sources_modified=False,original_measurement_semantics_unchanged=True,
  gpu=complete['gpu'],physical_gpu=complete['physical_gpu'],seconds=time.monotonic()-started,
  historical_U6000_reference=entry(ROOT/'output/dynamic_sr_confidence_geometry_20261006/evaluation/U6000/extra/complete.json'),
  note='Parent reference reused by hash; no duplicate parent rendering. New endpoints use the same evaluation platform recorded in suite plan.'))

if __name__=='__main__':
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--checkpoint',type=Path,required=True)
 ap.add_argument('--label',required=True);ap.add_argument('--out',type=Path);a=ap.parse_args()
 try:main(a)
 except BaseException:
  write((a.out or OUT/'evaluation'/a.label/'extra')/f'adapter_failed_{time.time_ns()}.json',
   dict(status='failed',traceback=traceback.format_exc(),parameter_updates=0));raise
