"""Finish the already accepted B0 prefix during independent preparation.

This uses only the registered remaining formal updates. Uniform evaluation
waits for the separately registered preparation/diagnostic prerequisites.
"""
import argparse,fcntl,socket,time,traceback
from types import SimpleNamespace
from fp_common import OUT,HERE,entry,read,write
import run_suite as suite

def main(a):
 assert a.operator_resource_resolved and a.gpu.startswith('GPU-')
 suite.plan();directory=OUT/'runs/r1_B0';checkpoint=suite.continuation(directory)
 assert checkpoint is not None and read(checkpoint.with_suffix('.json'))['metadata']['suffix_step']==3000
 initial=read(directory/'attempt_complete_0000_0100.json');assert initial['physical_gpu']==a.gpu
 status=OUT/'workers/B0_endpoint_preparation_overlap.json';sources=suite.stage_files()
 with (OUT/'locks'/f'{a.gpu}.controller.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);resource=suite.check_resource(a.gpu,status)
  identity=dict(host=socket.gethostname(),GPU_model=resource['name'],physical_GPU=resource['uuid'],driver=resource['driver'],python=a.python,protocol=entry(OUT/'protocol.json'))
  assert read(OUT/'workers/suffix_1_platform.json')==identity
  suite.train(SimpleNamespace(repeat='1',python=a.python,gpu=a.gpu),'B0',6000,suite.environment(a.gpu),status)
  assert suite.stage_files()==sources
  complete=read(directory/'complete.json');assert complete['status']=='completed_training' and complete['updates']==6000
  checkpoint=suite.endpoint('r1_B0')
  write(OUT/'evaluation_queue/r1_B0.json',dict(status='pending_uniform_evaluation',checkpoint=entry(checkpoint),task='r1_B0',trigger='successful registered remaining B0 training child return',preparation_gate_preserved=True))
  write(status,dict(status='completed_B0_formal_endpoint_queued_uniform_evaluation',complete=entry(directory/'complete.json'),checkpoint=entry(checkpoint),added_formal_updates=3000,total_B0_formal_updates=6000,no_X_dependency=True,quality_not_used_for_selection=True,source=entry(__file__)))
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--gpu',required=True);p.add_argument('--python',default=suite.PYTHON);p.add_argument('--operator-resource-resolved',action='store_true');a=p.parse_args()
 try:main(a)
 except BaseException:
  write(OUT/'workers'/f'B0_endpoint_overlap_failure_{time.time_ns()}.json',dict(status='failed_preserved',traceback=traceback.format_exc(),source=entry(__file__)));raise
