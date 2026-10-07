"""CPU-only contracts for registered full native parent domain interfaces.

Only stdlib, metadata and already legal train LR/initializer SHA are read. CUDA,
model imports, external processes and complete parent tensor loads are forbidden.
"""
from __future__ import annotations
import argparse,ast,copy,io,json,os,random,sys,time
from contextlib import redirect_stdout,redirect_stderr
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import full_prefix_registered as prefix
import full_native_protocol as protocol
from fp_common import ROOT,HERE,OUT,entry,sha,read,write

UNCHANGED=('digest','pure','planned_counts','AuthorShuffle','parameter_map','rng_state','restore_rng',
 'rng_transport_source_contract','validate_resume_amendment','create_resume_amendment',
 'combine_batch_statistics','operation','author_topology','author_iteration','topology_audit',
 'Journal','save_checkpoint','restore_checkpoint','latest_checkpoint','historical_cost')

def reject(call):
 try:call()
 except (ValueError,AssertionError,KeyError,FileNotFoundError,RuntimeError):return
 raise AssertionError('Required rejection did not occur')

def stateful_rng_fixture():
 class Byte:
  def __init__(self,device='cuda'):self.device=device;self.dtype='uint8'
  def cpu(self):return Byte('cpu')
 class Cuda:
  def set_rng_state_all(self,values):
   assert all(x.device=='cpu' and x.dtype=='uint8' for x in values);self.received=values
 class Torch:
  cuda=Cuda()
  def set_rng_state(self,value):assert value.device=='cpu' and value.dtype=='uint8';self.received=value
 class NP:
  random=SimpleNamespace(set_state=lambda value:None)
 t=Torch();saved=random.getstate();prefix.restore_rng(dict(python=saved,numpy=None,torch=Byte(),cuda=[Byte(),Byte()]),t,NP())
 assert len(t.cuda.received)==2

class FakeTorch:
 int64='int64'
 def __init__(self,seed=9):self.rng=random.Random(seed)
 def empty(self,*args,**kwargs):
  owner=self
  class Draw:
   def random_(self):self.value=owner.rng.randrange(2**63);return self
   def item(self):return self.value
  return Draw()
 class Generator:
  def manual_seed(self,seed):self.rng=random.Random(seed)
 def randperm(self,length,generator):
  order=list(range(length));generator.rng.shuffle(order);return SimpleNamespace(tolist=lambda:order)

def sampler_fixture():
 t=FakeTorch();a=prefix.AuthorShuffle(12,4,t)
 first=[a.next() for _ in range(3)];saved=a.state_dict();rng=t.rng.getstate()
 expected=[a.next() for _ in range(8)];fresh=FakeTorch();fresh.rng.setstate(rng);b=prefix.AuthorShuffle(12,4,fresh,saved)
 assert [b.next() for _ in range(8)]==expected and expected[0]==first[-1]
 assert b.state_dict()==a.state_dict() and b.repeated>0

def computation_contract():
 original=ast.parse((HERE/'full_prefix.py').read_text());candidate=ast.parse((HERE/'full_prefix_registered.py').read_text())
 nodes=lambda t:{n.name:n for n in t.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
 a,b=nodes(original),nodes(candidate)
 assert sha(HERE/'full_prefix.py')==protocol.LEGACY_PREFIX_SHA
 for name in UNCHANGED:assert ast.dump(a[name],include_attributes=False)==ast.dump(b[name],include_attributes=False),name
 old,new=copy.deepcopy(a['train']),copy.deepcopy(b['train'])
 oldtry=next(n for n in old.body if isinstance(n,ast.Try));newtry=next(n for n in new.body if isinstance(n,ast.Try))
 # Sole changes inside the author execution are the registered image shape and
 # complete scene-specific count validation. All rendering/loss/update code is exact.
 oldnodes=[n for n in ast.walk(oldtry) if isinstance(n,ast.If) and ast.unparse(n.test).startswith('tuple(camera.original_image.shape)')]
 newnodes=[n for n in ast.walk(newtry) if isinstance(n,ast.If) and ast.unparse(n.test).startswith('tuple(camera.original_image.shape)')]
 assert len(oldnodes)==len(newnodes)==1;newnodes[0].test=copy.deepcopy(oldnodes[0].test)
 for field in ('body','handlers','orelse','finalbody'):
  values=getattr(newtry,field)
  for i,n in enumerate(values):
   if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and ast.unparse(n.value.func)=='domain.validate_parent_state':
    values[i]=copy.deepcopy(next(v for v in oldtry.body if isinstance(v,ast.If) and "state['accepted_Adam'] != 16999" in ast.unparse(v)))
 assert ast.dump(oldtry,include_attributes=False)==ast.dump(newtry,include_attributes=False),'Author train execution changed beyond dimension/final-count guards'
 config,sources=protocol.configuration('cook_spinach');tree=ast.parse(sources['scene/gaussian_model.py'].read_text())
 g=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='GaussianModel')
 methods={n.name:n for n in g.body if isinstance(n,ast.FunctionDef)}
 capture=next(n for n in ast.walk(methods['capture']) if isinstance(n,ast.Return)).value
 assert isinstance(capture,ast.Tuple) and len(capture.elts)==14 and ast.unparse(capture.elts[12])=='self.optimizer.state_dict()'
 adam=[n for n in ast.walk(methods['training_setup']) if isinstance(n,ast.Call) and ast.unparse(n.func)=='torch.optim.Adam']
 assert len(adam)==1
 groups=next(n.value for n in ast.walk(methods['training_setup']) if isinstance(n,ast.Assign) and isinstance(n.value,ast.List) and len(n.value.elts)==8)
 names=[next(ast.literal_eval(v) for k,v in zip(x.keys,x.values) if ast.literal_eval(k)=='name') for x in groups.elts]
 assert names==['xyz','deformation','grid','f_dc','f_rest','opacity','scaling','rotation']
 return dict(exact_AST_definitions=list(UNCHANGED),native_capture_tuple_length=14,optimizer_capture_index=12,one_Adam_groups=names,
  permitted_train_body_adaptations=['manifest native camera shape','scene-config exact final RGB/backward/Adam state'],Gaussian_source=entry(sources['scene/gaussian_model.py']))

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path);a=parser.parse_args()
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU contracts require hard CUDA_VISIBLE_DEVICES empty'
 started=time.monotonic();directory=a.out or OUT/'operator_checks/full_prefix_registered'/str(time.time_ns());directory.mkdir(parents=True,exist_ok=False)
 results=[]
 def check(name,fn):
  before=time.monotonic();fn();results.append(dict(name=name,status='passed',CPU_seconds=time.monotonic()-before))
 try:
  contract={};check('pinned_native_computation_AST14tuple_oneAdam8',lambda:contract.update(computation_contract()))
  check('mapped_CPU_and_CUDA_saved_rng_inputs_are_CPU_ByteTensor',stateful_rng_fixture)
  check('author_shuffle_exact_mock_resume_and_extra_previous_batch',sampler_fixture)
  contexts={}
  for scene in ('meetroom_discussion','meetroom_vrheadset','coffee_martini','cook_spinach','cut_roasted_beef'):
   for seed in protocol.SEEDS:
    def actual(scene=scene,seed=seed):
     path=ROOT/f'data/dynamic_sr/full_time_20261007/prepared/{scene}/manifest_train_ready_seed{seed}.json'
     plan,inputs=prefix.build_plan(path,seed);plan.pop('CPU_seconds');cfg,_=protocol.configuration(scene)
     d=protocol.descriptor(scene);assert plan['data']['resolution_LR']==d['native_LR'] and plan['data']['resolution_HR']==d['native_HR']
     expected=protocol.expected_parent_state(cfg);assert plan['planned']['RGB_forwards']==expected['accepted_RGB']
     assert len(inputs)==plan['data']['training_LR_bytes_SHA_verified']==plan['data']['training_observations']
     assert all('/lr/' in x['path'] and '/cam00/' not in x['path'] for x in inputs)
     contexts[(scene,seed)]=(plan,inputs)
    check(f'actual_{scene}_seed{seed}_legal_LR_source_recipe_grid_counts',actual)
  plan=contexts[('meetroom_discussion',20261007)][0]
  check('new_registered_plan_complete_source_and_train_input_acceptance',lambda:protocol.validate_plan(plan))
  def CLI_registration():
   argv=['full_prefix_registered.py','--mode','plan','--manifest',str(protocol.bound(plan['data']['manifest'])),'--seed','20261007']
   registry=directory/'mock_CLI';first=copy.deepcopy(plan);first['CPU_seconds']=99.
   with patch.object(prefix,'OUT',registry),patch.object(prefix,'build_plan',return_value=(first,[])),patch.object(protocol,'completed_prefix',return_value=None),patch.object(sys,'argv',argv),redirect_stdout(io.StringIO()):
    prefix.main();path=registry/'full_prefix_registered_plans/meetroom_discussion/seed_20261007/registration.json';before=entry(path)
    first['CPU_seconds']=123.;prefix.main();assert entry(path)==before and read(path)==plan
    first['configuration']['OptimizationParams']['batch_size']=2;reject(prefix.main)
  check('CPU_CLI_registration_immutable_idempotent_source_recipe_conflict_refused',CLI_registration)
  def train_preconditions(kind):
   out=directory/('mock_resume_'+kind);out.mkdir()
   arguments=SimpleNamespace(out=out,resume='latest' if kind=='no_registered_plan' else None)
   if kind=='closed':write(out/'complete.json',dict(status='existing_result'))
   if kind=='source_conflict':write(out/'config.json',dict(scene='foreign_plan'))
   if kind=='explicit_resume_required':write(out/'config.json',plan)
   with patch.object(protocol,'validate_plan',return_value=True),patch.object(protocol,'forbid_completed_training',return_value=None):
    reject(lambda:prefix.train(arguments,plan,[]))
   assert 'torch' not in sys.modules and 'numpy' not in sys.modules
  for kind in ('closed','source_conflict','explicit_resume_required','no_registered_plan'):
   check('pre_model_resume_'+kind+'_refused',lambda k=kind:train_preconditions(k))
  cfg=plan['configuration'];good=protocol.expected_parent_state(cfg)
  check('batch4_complete68000RGB_state_acceptance',lambda:protocol.validate_parent_state(good,cfg))
  for key in ('accepted_RGB','accepted_backward','accepted_Adam','optimizer_resets','fine_iteration','coarse_iteration'):
   wrong=copy.deepcopy(good);wrong[key]-=1
   check('incomplete_parent_'+key+'_refused',lambda w=wrong:reject(lambda:protocol.validate_parent_state(w,cfg)))
  check('unknown_scene_refused',lambda:reject(lambda:protocol.configuration('unregistered_scene')))
  check('unknown_seed_refused',lambda:reject(lambda:protocol.validate_manifest(ROOT/'missing.json',20261009)))
  m=read(protocol.bound(plan['data']['manifest']))
  for field,change in [('resolutions',dict(lr=[336,252],hr=[1344,1008])),('scale',2)]:
   wrong=copy.deepcopy(m);wrong[field]=change
   check('wrong_'+field+'_refused',lambda w=wrong:reject(lambda:protocol.validate_grid(w,'meetroom_discussion')))
  wrong=copy.deepcopy(m);wrong['cameras']['cam01']['K_lr'][0][2]+=.125
  check('pixelcentre_principal_point_mismatch_refused',lambda:reject(lambda:protocol.validate_grid(wrong,'meetroom_discussion')))
  row=next(r for r in m['observations'] if r['split']=='train');root=protocol.bound(plan['data']['manifest']).parent
  for name,changes in [('HR',dict(lr_path='hr/cam01/0000.png')),('test_LR',dict(camera_id='cam00',split='test',lr_path='lr/cam00/0000.png')),('traversal',dict(lr_path='../lr/cam01/0000.png'))]:
   bad=dict(row,**changes);check(name+'_image_input_refused',lambda b=bad:reject(lambda:protocol.legal_LR_path(root,b)))
  for label,mutate in [('recipe',lambda p:p['configuration']['OptimizationParams'].update(batch_size=2)),('plan_source',lambda p:p['project_sources'].update({str(HERE.relative_to(ROOT)/'full_prefix_registered.py'):'0'*64})),('budget',lambda p:p['planned'].update(RGB_forwards=34000)),('HR_used',lambda p:p.update(HR_used=True)),('manifestSHA',lambda p:p['data']['manifest'].update(sha256='0'*64))]:
   bad=copy.deepcopy(plan);mutate(bad);check(label+'_plan_mismatch_refused',lambda b=bad:reject(lambda:protocol.validate_plan(b)))
  for scene in ('cook_spinach','cut_roasted_beef'):
   for seed in protocol.SEEDS:
    with patch.object(protocol,'completed_prefix',return_value=dict(scene=scene,seed=seed)):
     check(f'accepted_{scene}_seed{seed}_retraining_refused',lambda s=scene,k=seed:reject(lambda:protocol.forbid_completed_training(s,k)))
  with patch.object(protocol,'sha',side_effect=lambda path:'0'*64 if Path(path)==HERE/'full_prefix.py' else sha(path)):
   check('legacy_source_mismatch_refused',lambda:reject(lambda:protocol.configuration('meetroom_discussion')))
  uuid='GPU-fake';commands=[]
  def resource_fixture(output,hardware,should_reject=False):
   commands.clear()
   def fake(cmd,**kwargs):commands.append(cmd);return output if '--query-compute-apps=gpu_uuid,pid,process_name,used_memory' in cmd else hardware
   with patch.dict(os.environ,CUDA_VISIBLE_DEVICES=uuid),patch.object(prefix.subprocess,'check_output',side_effect=fake):
    if should_reject:reject(lambda:prefix.check_gpu(uuid))
    else:assert prefix.check_gpu(uuid)==hardware
  check('mock_clean_3090_resource_pass',lambda:resource_fixture('',uuid+', NVIDIA GeForce RTX 3090, 24576, 0, 550'))
  check('mock_any_foreign_compute_refused',lambda:resource_fixture(uuid+', 987654, /opt/todesk/a, 2', '',True))
  check('mock_RTX5090_training_refused',lambda:resource_fixture('',uuid+', NVIDIA GeForce RTX 5090, 32000, 0, 550',True))
  assert not any(x in sys.modules for x in ('torch','numpy','PIL','cv2','scene','gaussian_renderer'))
  write(directory/'receipt.json',dict(status='passed_CPU_only_registered_native_full_prefix_contracts',checks=results,count=len(results),
   sources={n:entry(HERE/n) for n in ('full_native_protocol.py','full_prefix_registered.py','full_prefix_registered_checks.py','full_prefix.py')},
   computation_contract=contract,contexts=[dict(scene=s,seed=k,manifest=p['data']['manifest'],initializer=p['data']['initial_points'],grid=p['data']['resolution_LR'],train_LR_SHA_verified=len(i),planned=p['planned']) for (s,k),(p,i) in contexts.items()],
   CPU_seconds=time.monotonic()-started,formal_Adam_calls=0,actual_GPU_calls=0,native_raster_calls=0,external_process_calls=0,
   model_imports=0,torch_imported=False,HR_images_read=0,test_images_read=0,actual_tensor_roundtrip_or_GPU_quality_validated=False,
   scope='Metadata/source/LR-byte SHA and mock contracts only; GPU native initialization, training, checkpoint restore and quality remain root-owned acceptance gates.'))
  print(json.dumps(dict(status='passed',checks=len(results),receipt=entry(directory/'receipt.json'),CPU_seconds=time.monotonic()-started)))
 except BaseException:
  import traceback
  write(directory/'failure.json',dict(status='failed_CPU_only_contract_preserved',checks=results,error=traceback.format_exc(),CPU_seconds=time.monotonic()-started,actual_GPU_calls=0,formal_Adam_calls=0,external_process_calls=0))
  raise
if __name__=='__main__':main()
