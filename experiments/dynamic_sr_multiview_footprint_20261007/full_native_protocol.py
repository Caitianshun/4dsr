"""Registered native full-time scene interfaces; CPU metadata and SHA checks.

MeetRoom uses the pinned Wu generic N3DV configuration as an explicit domain
adaptation. It is not an original Wu MeetRoom experiment or StreamRF optimizer.
No model, tensor, image decoder, GPU query or process dispatcher is imported.
"""
from __future__ import annotations
import ast,hashlib,json,math
from pathlib import Path
from fp_common import ROOT,HERE,OUT,local,read,sha,entry,bound

SCHEMA='registered_native_full_LR_domain_v1'
PIN='843d5ac636c37e4b611242287754f3d4ed150144'
LEGACY_PREFIX_SHA='ab5fd0313d37bbfa31c240f3c38f8ba8bec37cf5022217f1d3166f18ed822a6f'
READINESS=OUT/'full_data_readiness/full_prefix_readiness.json'
SEEDS=(20261007,20261008)
SCENES=('cook_spinach','cut_roasted_beef','meetroom_discussion','meetroom_vrheadset','coffee_martini','flame_steak')
MEETROOM=('meetroom_discussion','meetroom_vrheadset')
KNOWN_SOURCE={'n3dv':'https://github.com/facebookresearch/Neural_3D_Video','meetroom':'https://github.com/AlgoHunt/StreamRF'}

def require(ok,message):
 if not ok:raise ValueError(message)
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
def pure(node):
 if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='dict' and not node.args:return {k.arg:pure(k.value) for k in node.keywords}
 return ast.literal_eval(node)
def descriptor(scene):
 require(scene in SCENES,'Unregistered full native scene')
 meet=scene in MEETROOM
 return dict(schema=SCHEMA,scene=scene,role='already_used_development' if scene in SCENES[:4] else 'held_confirmation',
  native_LR=[320,180] if meet else [336,252],native_HR=[1280,720] if meet else [1344,1008],frames=[0,299],frame_count=300,
  time='original_frame_index/300',heldout='cam00',training='all actual supplied cameras except cam00, including cam01',
  configuration='Wu843d5_generic_N3DV_explicit_MeetRoom_adaptation' if meet else 'Wu843d5_original_N3DV_scene_configuration',
  configuration_paths=['arguments/__init__.py','arguments/dynerf/default.py']+([] if meet else [f'arguments/dynerf/{scene}.py']),
  source=KNOWN_SOURCE['meetroom' if meet else 'n3dv'],scale=4,HR_training_reads=False,teacher_used=False,
  initializer='existing registered full train-only LR sparse initializer; not official supplied COLMAP',
  unchanged_author_MeetRoom_reproduction=False if meet else None,complete_author_paper_benchmark=False)
def configuration(scene,readiness=READINESS):
 d=descriptor(scene);evidence=read(local(readiness))['original_Wu'];require(evidence['git_commit']==PIN,'Author pin differs')
 sources={r['git_path']:bound(r) for r in evidence['sources']}
 require(sha(HERE/'full_prefix.py')==LEGACY_PREFIX_SHA,'Original accepted prefix source changed')
 result={}
 for cls in ast.parse(sources['arguments/__init__.py'].read_text()).body:
  if not isinstance(cls,ast.ClassDef) or cls.name not in ('ModelParams','ModelHiddenParams','OptimizationParams','PipelineParams'):continue
  init=next(f for f in cls.body if isinstance(f,ast.FunctionDef) and f.name=='__init__')
  result[cls.name]={n.targets[0].attr:pure(n.value) for n in init.body if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Attribute) and isinstance(n.targets[0].value,ast.Name) and n.targets[0].value.id=='self'}
 for name in d['configuration_paths'][1:]:
  require(name in sources,'Missing pinned configuration source')
  for n in ast.parse(sources[name].read_text()).body:
   if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id in result:result[n.targets[0].id].update(pure(n.value))
 if scene not in MEETROOM:require(result==evidence['fully_merged_N3DV_configurations'][scene],'Original merged configuration differs')
 else:require(result==evidence['fully_merged_N3DV_configurations']['coffee_martini'],'Generic configuration differs from pinned default/empty-override evidence')
 o=result['OptimizationParams'];expected_batch=4 if scene in MEETROOM+('coffee_martini',) else 2
 require((o['coarse_iterations'],o['iterations'],o['batch_size'],o['dataloader'],o['custom_sampler'],o['zerostamp_init'])==(3000,14000,expected_batch,True,None,False),'Pinned stage/batch/sampler recipe differs')
 require((o['densify_until_iter'],o['opacity_reset_interval'],o['densify_from_iter'],o['densification_interval'],o['pruning_from_iter'],o['pruning_interval'])==(10000,60000,500,100,500,100),'Pinned topology schedule differs')
 return result,sources
def expected_parent_state(config):
 o=config['OptimizationParams'];c=o['coarse_iterations'];f=o['iterations'];b=o['batch_size']
 require((c,f)==(3000,14000) and b in (2,4),'Registered native full schedule differs')
 return dict(stage='fine',coarse_iteration=c,fine_iteration=f,optimizer_resets=2,accepted_RGB=(c+f)*b,accepted_backward=c+f,accepted_Adam=c+f-1)
def validate_parent_state(state,config):require(state==expected_parent_state(config),'Complete prefix state differs from frozen scene configuration')
def validate_grid(m,scene):
 d=descriptor(scene);require(m['scene']==scene and m['scale']==4 and m['resolutions']==dict(lr=d['native_LR'],hr=d['native_HR']),'Registered native scene/grid differs')
 for camera in m['cameras'].values():
  for name in ('K_lr','K_hr','w2c','c2w'):require(all(math.isfinite(float(x)) for row in camera[name] for x in row),'Nonfinite camera metadata')
  h,l=camera['K_hr'],camera['K_lr'];require(h[0][0]>0 and h[1][1]>0 and abs(l[0][0]-h[0][0]/4)<1e-8 and abs(l[1][1]-h[1][1]/4)<1e-8,'LR focal scale differs')
  require(abs(l[0][2]-((h[0][2]+.5)/4-.5))<1e-8 and abs(l[1][2]-((h[1][2]+.5)/4-.5))<1e-8,'Pixel-centre principal point differs')
 return d
def legal_LR_path(root,row):
 relative=Path(row['lr_path']);expected=Path('lr')/row['camera_id']/f"{int(row['frame_index']):04d}.png"
 require(relative==expected and row['split']=='train' and row['camera_id']!='cam00','HR, test or unregistered LR path refused')
 p=root/relative;require(not p.is_symlink() and p.resolve().is_relative_to(root.resolve()),'LR path escaped registered dataset');return p
def validate_manifest(path,seed,readiness=READINESS):
 require(seed in SEEDS,'Unregistered independent seed');path=local(path).resolve();m=read(path);d=validate_grid(m,m['scene'])
 require(path==ROOT/'data/dynamic_sr/full_time_20261007/prepared'/m['scene']/f'manifest_train_ready_seed{seed}.json','Only registered train-ready manifest location allowed')
 require(m['schema']=='n3dv_dynamic_sr_pilot_v1' and m.get('full_time_decode_verified') and m['frame_indices']==list(range(300)),'Verified full300 interface required')
 require(m['role']==d['role'] and m['source']==d['source'] and m.get('official_split') is True and m.get('additional_cam01_dev_holdout') is False,'Official scene/source/role/split interface differs')
 train=m['splits']['train'];require(m['splits']['test']==['cam00'] and not m['splits']['dev'] and set(train)==set(m['cameras'])-{'cam00'} and 'cam01' in train and len(train)==len(set(train)),'Official cam00 holdout and all actual remaining cameras required')
 if m['scene'] in MEETROOM:require(train==[f'cam{i:02d}' for i in range(1,13)],'MeetRoom registered camera set differs')
 rows=[r for r in m['observations'] if r['split']=='train'];keys={(r['camera_id'],int(r['frame_index'])) for r in rows}
 require(len(rows)==len(keys)==300*len(train) and keys=={(c,f) for c in train for f in range(300)},'Incomplete/duplicate legal train observations')
 require(all(abs(float(r['time'])-int(r['frame_index'])/300)<1e-7 for r in rows),'Author frame/300 time required')
 init=m['initialization'];reg=m['independent_seed_registration'];require(init['seed']==reg['seed']==seed and init['train_only_lr'] is True and init['official_COLMAP_initialization'] is False and reg['LR_model_prefix_trained'] is False,'Fresh seed-specific pure-LR initializer required')
 require(init['npz_path']==f'initialization_seed_{seed}/points_lr.npz','Unregistered initial-point input refused')
 original=bound(m['original_full_manifest']);o=read(original)
 for k in ('scene','source','scale','role','frame_indices','resolutions','cameras','splits','observations'):require(m[k]==o[k],'Derived manifest differs from original full data: '+k)
 receipt_path=path.parent/f'initialization_seed_{seed}/complete.json';receipt=read(receipt_path)
 require(receipt['status']=='completed_train_ready_manifest_and_initial_points_CPU_acceptance' and receipt['derived_manifest']==entry(path) and receipt['identity']['original_full_manifest']==entry(original) and receipt['identity']['seed']==seed,'Initializer receipt/source identity differs')
 require(receipt['CUDA_calls']==0 and receipt['model_prefix_trained'] is False and receipt['load_training_CUDA_camera_creation_executed'] is False,'Initializer boundary differs')
 require(m['full_initialization_source']==receipt['identity']['source'],'Initializer implementation identity differs')
 for key in ('source','legacy_triangulation','data_loader'):bound(receipt['identity'][key])
 for artifact in receipt['artifacts']:bound(artifact)
 points=path.parent/init['npz_path'];require(sha(points)==init['npz_sha256'],'Initial point SHA differs')
 inputs=[]
 for row in rows:
  p=legal_LR_path(path.parent,row);require(p.is_file() and sha(p)==row['lr_sha256'],'Legal full LR source SHA differs')
  inputs.append(dict(camera=row['camera_id'],frame=int(row['frame_index']),path=str(p.relative_to(ROOT)),sha256=row['lr_sha256']))
 centers=[[float(m['cameras'][c]['c2w'][i][3]) for i in range(3)] for c in train];center=[sum(x[i] for x in centers)/len(centers) for i in range(3)];extent=1.1*max(math.sqrt(sum((x[i]-center[i])**2 for i in range(3))) for x in centers)
 require(math.isfinite(extent) and extent>0,'Degenerate train-only camera extent')
 cfg,sources=configuration(m['scene'],readiness)
 domain=dict(**d,seed=seed,configuration_sources={name:entry(sources[name]) for name in d['configuration_paths']},configuration_sha256=digest(cfg),readiness=entry(local(readiness)),protocol_source=entry(Path(__file__)),legacy_accepted_prefix_source=entry(HERE/'full_prefix.py'))
 return m,dict(manifest=entry(path),original_manifest=entry(original),initialization_receipt=entry(receipt_path),initial_points=entry(points),train_cameras=train,training_observations=len(rows),training_LR_sources_sha256=digest(inputs),training_LR_bytes_SHA_verified=len(inputs),train_camera_center=center,cameras_extent=extent,extent_cam01_included=True,heldout_camera='cam00',frames=[0,299],time='frame_index/300',resolution_LR=m['resolutions']['lr'],resolution_HR=m['resolutions']['hr'],domain_protocol=domain),inputs
def completed_prefix(scene,seed):
 descriptor(scene);require(seed in SEEDS,'Unregistered independent seed');index=OUT/'full_LR_prefixes/checkpoint_index.json'
 if not index.exists():return None
 matches=[r for r in read(index)['entries'] if r['scene']==scene and int(r['seed'])==seed and r['status']=='completed_independent_full_author_LR_prefix']
 require(len(matches)<=1,'Duplicate completed prefix index identities')
 if not matches:return None
 for k in ('checkpoint','complete','exit_acceptance'):bound(matches[0][k])
 return dict(index=entry(index),entry=matches[0])
def forbid_completed_training(scene,seed):require(completed_prefix(scene,seed) is None,'Same scene/seed native parent already accepted; preserve it and refuse retraining')
def validate_plan(plan):
 d=descriptor(plan['scene']);config,sources=configuration(plan['scene'],plan['readiness']['path']);require(plan['configuration']==config and plan['author_commit']==PIN and plan['schema']=='registered_native_full_LR_prefix_v1','Registered native plan recipe/schema differs')
 require(plan['data']['domain_protocol']['schema']==SCHEMA and plan['data']['domain_protocol']['scene']==d['scene'] and plan['seed'] in SEEDS,'Registered domain identity differs')
 require(plan['data']['resolution_LR']==d['native_LR'] and plan['data']['resolution_HR']==d['native_HR'],'Registered plan grid differs')
 for name,expected in plan['project_sources'].items():require(sha(ROOT/name)==expected,'Frozen registered prefix source differs: '+name)
 for name,value in plan['author_sources'].items():require(entry(sources[name])==value,'Pinned author source differs')
 require(plan['data']['domain_protocol']['configuration_sha256']==digest(config),'Frozen configuration digest differs')
 expected=expected_parent_state(config);require(plan['planned']['RGB_forwards']==expected['accepted_RGB'] and plan['planned']['total_Adam_calls']==expected['accepted_Adam'] and plan['planned']['total_iterations']==expected['accepted_backward'],'Native planned budget differs')
 manifest=bound(plan['data']['manifest']);m,actual,inputs=validate_manifest(manifest,plan['seed'],plan['readiness']['path'])
 require(actual==plan['data'] and plan['observation']=='native_lr' and plan['HR_used'] is False and plan['teacher_used'] is False and plan['heldout_pixels_used'] is False,'Registered legal data/input boundary differs')
 return True
