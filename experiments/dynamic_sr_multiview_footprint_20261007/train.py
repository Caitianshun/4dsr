"""Fixed-budget native-state continuation for the six registered M/X arms.

Three RGB graphs are accumulated at one parameter state. Same-time arms share
one effective state and each optimizer steps once. All support is training-only.
The first 100 steps are part of 6000; segment continuation retains both Adams,
global RNG, schedule position, hardware identity, and every prior log/checkpoint.
"""
import argparse,os,sys,time,math,random,traceback,shutil
from collections import Counter
from fp_common import *
shared,policy=setup()
import numpy as np
import torch
from motion_model import load_model,refinement_state,capacity_summary,rasterize
from common import resized_camera
from n3dv_data import load_manifest
from training_support import TeacherCache,initial_identity,digest_state
from config import ARMS,METHODS
from schedules import select_rows
import losses
adapter=module('fp_training_adapter',ROOT/'experiments/dynamic_sr_20260918/run_experiment.py')
resume_support=module('fp_resume_support',ROOT/'experiments/dynamic_sr_20260920/resume_control.py')
old_sampler=module('fp_native_parent_sampler',ROOT/'experiments/dynamic_sr_confidence_geometry_20261006/schedules.py')

def rng():
 return dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all(),numpy=np.random.get_state(),python=random.getstate())

def json_value(v):
 if isinstance(v,torch.Tensor):
  v=v.detach().cpu()
  return v.item() if v.numel()==1 else v.tolist()
 if isinstance(v,dict):return {str(k):json_value(x) for k,x in v.items()}
 if isinstance(v,(list,tuple)):return [json_value(x) for x in v]
 if isinstance(v,np.generic):return v.item()
 return v

def rms(tensor):
 return float(tensor.detach().double().square().mean().sqrt()) if tensor is not None else 0.

def save_checkpoint(model,a,p,table,cursor,sources,start,train_s,started,calibration_sha):
 step=6000+cursor;ck=model.checkpoint
 meta=dict(scene='cook_spinach',stage='multiview_footprint',run_id=HERE.name,branch='ordinary_split',
  method=a.method,repeat=str(a.repeat),step=7200+step,intervention_step=step,suffix_step=cursor,
  manifest_sha=p['manifest']['sha256'],protocol_sha256=sha(OUT/'protocol.json'),
  schedule_sha256=sha(local(p['schedules'][str(a.repeat)]['path'])),calibration_sha256=calibration_sha,
  extent=ck['metadata']['extent'],points=132972,segment_start=start,train_s=train_s,
  elapsed_s=time.monotonic()-started,gpu=torch.cuda.get_device_name())
 payload=dict(model=model.g.capture(),hidden=vars(model.h),optim=vars(model.o),metadata=meta,rng=rng(),
  samplers=dict(kind='registered_multiview_footprint',cursor=cursor,schedule_sha256=meta['schedule_sha256']),
  motion_refinement=refinement_state(model),multiview_footprint=dict(sources=sources,method=a.method,repeat=str(a.repeat)))
 path=a.out/f'checkpoint_{step}.pt';tmp=path.with_suffix('.tmp')
 with tmp.open('wb') as f:torch.save(payload,f);f.flush();os.fsync(f.fileno())
 tmp.replace(path);write(path.with_suffix('.json'),dict(path=str(path.relative_to(ROOT)),sha256=sha(path),metadata=meta))
 return path

def optimizer_audit(model):
 groups=[]
 for optimizer in (model.g.optimizer,model.child_optimizer):
  ids=[id(p) for group in optimizer.param_groups for p in group['params']]
  assert len(ids)==len(set(ids)),'Optimizer contains a duplicate parameter'
  groups.append(set(ids))
 assert not groups[0]&groups[1],'Both optimizers update the same parameter'
 return dict(disjoint=True,parameter_tensors=[len(g) for g in groups],calls_per_round=2)

def recover_log_tail(directory,start):
 """Keep failed-work cost but exclude its uncommitted path from final curves."""
 path=directory/'training.jsonl'
 lines=path.read_text().splitlines() if path.exists() else [];kept=[];tail=[]
 for line in lines:
  try:cursor=int(json.loads(line)['suffix_step'])
  except (ValueError,KeyError,TypeError):tail.append(line);continue
  (kept if cursor<=start else tail).append(line)
 last_attempt=directory/'last_attempt.json'
 incomplete=read(last_attempt) if last_attempt.exists() else {}
 if not tail and incomplete.get('suffix_step',0)<=start:return
 incident=directory/'recovery_incidents'/str(time.time_ns());incident.mkdir(parents=True)
 if path.exists():shutil.copyfile(path,incident/'training_before_recovery.jsonl')
 else:(incident/'training_before_recovery.jsonl').write_text('')
 (incident/'uncommitted_tail.jsonl').write_text('\n'.join(tail)+'\n')
 if last_attempt.exists():shutil.copyfile(last_attempt,incident/'last_attempt.json')
 confirmed=[]
 for line in tail:
  try:confirmed.append(json.loads(line))
  except ValueError:pass
 write(incident/'receipt.json',dict(status='archived_failed_work_not_selected_endpoint_trajectory',
  resume_suffix_step=start,confirmed_completed_updates=len(confirmed),
  confirmed_RGB_forwards=sum(r.get('rgb_forwards',0) for r in confirmed),
  confirmed_moment_forwards=sum(r.get('moment_forwards',0) for r in confirmed),
  confirmed_Adam_calls=sum(r.get('adam_calls',0) for r in confirmed),
  incomplete_attempt=incomplete if incomplete.get('suffix_step',0)>max([start]+[r['suffix_step'] for r in confirmed]) else None,
  trailing_incomplete_attempt='see preserved last_attempt; active operation cost is unknown until returned',
  preserved_trajectory=entry(incident/'training_before_recovery.jsonl'),uncommitted_tail=entry(incident/'uncommitted_tail.jsonl')))
 temporary=path.with_suffix('.recovered.tmp');temporary.write_text('\n'.join(kept)+('\n' if kept else ''));temporary.replace(path)

def committed_segments(directory,stop,checkpoint=None):
 """Repair only a receipt lost after an atomic checkpoint, from its log rows."""
 segments=[read(f) for f in sorted(directory.glob('segment_*.json'))]
 position=0
 for segment in segments:
  assert segment['start']==position and segment['suffix_endpoint']<=stop
  assert segment['updates']==segment['suffix_endpoint']-position
  bound(segment['checkpoint']);position=segment['suffix_endpoint']
 if position==stop:return
 assert checkpoint is not None and checkpoint['metadata']['suffix_step']==stop
 rows=[json.loads(line) for line in (directory/'training.jsonl').read_text().splitlines()]
 selected=[row for row in rows if position<row['suffix_step']<=stop]
 assert [r['suffix_step'] for r in selected]==list(range(position+1,stop+1))
 earlier=next((row for row in rows if row['suffix_step']==position),None)
 path=directory/f'checkpoint_{6000+stop}.pt'
 metadata=checkpoint['metadata']
 receipt=dict(status='recovered_committed_segment_receipt',method=metadata['method'],repeat=metadata['repeat'],
  start=position,suffix_endpoint=stop,updates=stop-position,
  training_rgb_forwards=sum(r['rgb_forwards'] for r in selected),moment_forwards=sum(r['moment_forwards'] for r in selected),
  adam_calls=sum(r['adam_calls'] for r in selected),
  train_s=selected[-1]['train_s']-(earlier['train_s'] if earlier and earlier['suffix_step']>metadata['segment_start'] else 0.),
  wall_s=selected[-1]['wall_s']-(earlier['wall_s'] if earlier and earlier['suffix_step']>metadata['segment_start'] else 0.),
  peak_gb=max(row['peak_gb'] for row in selected),checkpoint=entry(path),
  gpu=metadata['gpu'],physical_gpu='see original config and attempt receipts',topology_unchanged=True,
  source_identity=checkpoint['multiview_footprint']['sources'],calibration_sha256=metadata['calibration_sha256'],
  receipt_recovered_from_atomic_checkpoint_and_exact_contiguous_committed_log=True)
 write(directory/f'segment_{position:04d}_{stop:04d}.json',receipt)

def main(a):
 p=read(OUT/'protocol.json');definition=ARMS[a.method]
 assert p['arms'][a.method]==definition
 assert a.out.resolve().is_relative_to((OUT/'runs').resolve())
 if a.resume:
  assert a.out.is_dir() and (a.out/'config.json').is_file()
 else:a.out.mkdir(parents=True,exist_ok=False)
 torch.set_num_threads(4)
 sources=source_identity(a.method);cal={};calibration_sha=None;support=None;gm=None
 allowed={str(local(e['path']).resolve()):e for e in p['training_files']};reads=Counter()
 if definition['X'] or definition['E']:
  cal=read(OUT/'calibration.json');assert cal['status']=='passed'
  calibration_sha=sha(OUT/'calibration.json')
  assert cal['identity']['protocol']==entry(OUT/'protocol.json')
  assert cal['identity']['parent']==p['parent'] and cal['identity']['manifest']==p['manifest']
  assert cal['identity']['schedule']==entry(local(p['schedules']['1']['path']))
  assert cal['identity']['support_index']==entry(OUT/'support/frozen/index.json')
  for identity in cal['source_files'].values():bound(identity)
 if definition['X']:
  from support_cache import SupportCache
  support=SupportCache(OUT/'support/frozen/index.json')
  for e in support.training_entries():allowed[str(local(e['path']).resolve())]=e
  assert support.index['identity']['source_sha256']==sha(HERE/'support_cache.py')
  assert support.index['identity']['footprint_source_sha256']==sha(HERE/'footprint.py')
  for identity in support.index['identity']['schedules']:bound(identity)
  gm=module('fp_live_depth_moments',ROOT/'experiments/dynamic_sr_confidence_geometry_20261006/depth_prior.py')
 def guard(event,arg):
  if event!='open' or not isinstance(arg[0],(str,bytes,os.PathLike)):return
  path=Path(os.fsdecode(arg[0]))
  if path.suffix.lower() not in ('.png','.jpg','.jpeg','.npy','.npz'):return
  name=str(path.resolve());assert name in allowed,('Training read outside legal whitelist',name);reads[name]+=1
 sys.addaudithook(guard)
 # Original LR/teacher bytes are checked once at startup. Frozen support NPZs
 # are independently byte-verified by SupportCache on each first actual read.
 for e in p['training_files']:bound(e)
 m=load_manifest(bound(p['manifest']));records,cameras=adapter.load_training(m)
 assert len(records)==1140
 cameras=[resized_camera(c,1008,1344) for c in cameras]
 schedule_path=local(p['schedules'][str(a.repeat)]['path']);table=read(schedule_path)
 schedule_index=read(OUT/'schedules/index.json');assert schedule_index['status']=='completed'
 registered=next(e for e in schedule_index['schedules'] if local(e['path']).resolve()==schedule_path.resolve())
 bound(registered)
 assert table['record_keys']==[[r['camera_id'],r['frame_index']] for r in records]
 rows=select_rows(table,a.method);assert len(rows)==6000 and table['audit']['passed']
 model=load_model(a.resume or bound(p['parent']),m);g=model.g;ck=model.checkpoint
 assert model.branch=='ordinary_split' and capacity_summary(model)['active_gaussians']==132972
 resume_support.assert_state_equal(ck['model'][12],g.optimizer.state_dict())
 resume_support.assert_state_equal(ck['motion_refinement']['child_optimizer'],model.child_optimizer.state_dict())
 optimizers=optimizer_audit(model)
 if a.resume:
  start=ck['metadata']['suffix_step']
  assert ck['metadata']['method']==a.method and ck['metadata']['repeat']==str(a.repeat)
  assert ck['metadata']['protocol_sha256']==sha(OUT/'protocol.json')
  assert ck['metadata']['calibration_sha256']==calibration_sha
  assert ck['multiview_footprint']['sources']==sources
  assert ck['samplers']==dict(kind='registered_multiview_footprint',cursor=start,schedule_sha256=sha(schedule_path))
  assert ck['metadata']['gpu']==torch.cuda.get_device_name(),'Continuation training hardware changed'
  parent=torch.load(bound(p['parent']),map_location='cpu',weights_only=False)
  prefix=old_sampler.validate_parent(read(bound(p['old_schedule'])),parent);del parent
 else:
  start=0;assert ck['metadata']['method']=='U' and ck['metadata']['intervention_step']==6000
  prefix=old_sampler.validate_parent(read(bound(p['old_schedule'])),ck)
 assert start<a.stop<=6000
 if a.resume:
  recover_log_tail(a.out,start);committed_segments(a.out,start,ck)
 names=shared.all_named(model);topology=shared.topology(model);initial=initial_identity(model)
 teacher_index=read(bound(p['teacher']));by={(e['camera'],e['frame']):e for e in teacher_index['entries']}
 paths={i:Path(m['_root'])/by[r['camera_id'],r['frame_index']]['relative_path'] for i,r in enumerate(records)}
 cache=TeacherCache(paths,(3,1008,1344),8);curve=read(bound(p['lr_curve']))['rows']
 config=dict(method=a.method,repeat=str(a.repeat),parent=p['parent'],protocol_sha256=sha(OUT/'protocol.json'),
  schedule_sha256=sha(schedule_path),source_identity=sources,initial_identity=initial,calibration=cal,
  calibration_sha256=calibration_sha,original_HR_training=False,training_files=list(allowed.values()),
  torch=str(torch.__version__),cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),
  physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),optimizer_audit=optimizers,
  RGB_per_update=3,moment_per_update=3 if definition['X'] else 0)
 if a.resume:
  previous=read(a.out/'config.json')
  for key in ('method','repeat','protocol_sha256','schedule_sha256','source_identity','calibration_sha256','gpu','torch','cuda'):
   assert previous[key]==config[key],key
 else:
  write(a.out/'config.json',config)
  for rel in sources:
   path=Path(os.environ.get('FOURDSR_UPSTREAM','/home/cai_tianshun/Project/4dgs'))/rel[9:] if rel.startswith('upstream/') else ROOT/rel
   dest=a.out/'sources'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dest)
 resume_support.restore_global_rng(ck['rng']);assert digest_state(ck['rng'])==digest_state(rng())
 write(a.out/f'reload_audit_{start:04d}.json',dict(status='passed',prefix=prefix,global_rng_exact=True,two_adam_exact=True,
  identity=initial,topology=topology,fork_cursor=start,optimizer_audit=optimizers,new_suffix_is_native_continuation=False))
 started=time.monotonic();train_s=0.;torch.cuda.reset_peak_memory_stats();last=None
 chunk_start=start;chunk_train=0.;chunk_wall=0.
 with (a.out/'training.jsonl').open('a',buffering=1) as log:
  for cursor in range(start+1,a.stop+1):
   attempt=dict(suffix_step=cursor,rgb=0,moments=0,autograd_calls=0,adam_calls=0,
    status='registered_attempt',active_operation=None)
   def mark(operation=None,returned=None):
    if returned:attempt[returned]+=1
    attempt['active_operation']=operation
    attempt['status']='active_operation_cost_unknown_until_return' if operation else 'confirmed_returned_operations'
    write(a.out/'last_attempt.json',attempt)
   mark()
   tick=time.monotonic();step=6000+cursor;g.update_learning_rate(7200+step)
   rates=curve[step-1];assert rates[0]==step
   for opt,rr in ((g.optimizer,rates[1]),(model.child_optimizer,rates[2])):
    for group in opt.param_groups:
     assert math.isclose(group['lr'],rr[group['name']],rel_tol=1e-14,abs_tol=0)
     group['lr']=rr[group['name']]
   g.optimizer.zero_grad(set_to_none=True);model.child_optimizer.zero_grad(set_to_none=True)
   row=rows[cursor-1];keys=[table['record_keys'][i] for i in row];cams=[cameras[i] for i in row]
   same_time=definition['schedule']=='rows'
   if same_time:
    assert len({k[0] for k in keys})==3 and len({k[1] for k in keys})==1
    state=policy.effective_state(model,cams[0].time);states=[state]*3
   else:states=[policy.effective_state(model,c.time) for c in cams]
   rgbs=[]
   for c,s in zip(cams,states):
    mark('RGB_forward');rgbs.append(rasterize(g,c,s['xyz'],s['cov'],s['opacity'],s['sh'])['render']);mark(returned='rgb')
   lrs=[records[i]['image'].cuda() for i in row]
   degraded_raw=[torch.nn.functional.interpolate(rgb[None],size=tuple(lr.shape[-2:]),mode='bicubic',
    align_corners=False,antialias=True)[0] for rgb,lr in zip(rgbs,lrs)]
   clamp_fractions=[((v.detach()<0)|(v.detach()>1)).float().mean() for v in degraded_raw]
   errors=[value.clamp(0,1)-lr for value,lr in zip(degraded_raw,lrs)]
   lr_values=[e.abs().mean() for e in errors]
   lr_term=sum(w*v for w,v in zip(definition['lr_weights'],lr_values))
   if definition['E']:lr_term=.8*errors[0].abs().mean()+.2*cal['kappa']*errors[0].square().mean()
   sr_values=[losses.sr_loss(rgbs[i],cache.get(row[i])) for i in (1,2)]
   reg=g.compute_regulation(model.h.time_smoothness_weight,model.h.l1_time_planes,model.h.plane_tv_weight)
   total=lr_term+.05*(sr_values[0]+sr_values[1])+reg
   x=rgbs[0].sum()*0.;xdiag=None;moments=[]
   if definition['X']:
    for c,s in zip(cams,states):
     mark('moment_forward');moments.append(gm.render_moments(c,s['xyz'],s['cov'],s['opacity']));mark(returned='moments')
    mark('six_edge_X_graph')
    x,xdiag=losses.x_loss(rgbs,moments,cams,lrs,keys,support)
    mark()
    total=total+cal['lambda_X']*min(cursor/500,1.)*x
   assert torch.isfinite(total),'Nonfinite training loss'
   check=cursor==start+1 or cursor in (1,100,500,3000,6000)
   gdiag=None
   if check:
    mark('gdiag_RGB_autograd')
    rgb_gradients=torch.autograd.grad(total,rgbs,retain_graph=True,allow_unused=True)
    mark(returned='autograd_calls');mark('gdiag_meanLR_xyz_autograd')
    _,lr_xyz=losses.summed_xyz_gradient_rms(sum(lr_values)/3,[s['xyz'] for s in states])
    mark(returned='autograd_calls');mark('gdiag_total_xyz_autograd')
    total_xyz_rms,total_xyz=losses.summed_xyz_gradient_rms(total,[s['xyz'] for s in states])
    mark(returned='autograd_calls');x_xyz_rms=0.
    if definition['X']:
     mark('gdiag_X_xyz_autograd')
     x_xyz_rms=losses.summed_xyz_gradient_rms(x,[s['xyz'] for s in states])[0];mark(returned='autograd_calls')
    gdiag=dict(rgb_total_RMS=[rms(v) for v in rgb_gradients],xyz_LR_mean_RMS=rms(lr_xyz),
     xyz_X_RMS=x_xyz_rms,xyz_total_RMS=total_xyz_rms,
     xyz_convention='same-index sum before RMS; identical shared tensors queried once',
     X_includes_source_RGB_and_target_moment_paths=bool(definition['X']))
    del lr_xyz,total_xyz,rgb_gradients
   mark('main_accumulated_backward');total.backward();mark(returned='autograd_calls')
   if check:
    assert all(torch.isfinite(x.grad).all() for x in names.values() if x.grad is not None)
   mark('base_Adam_step');g.optimizer.step();mark(returned='adam_calls')
   mark('children_Adam_step');model.child_optimizer.step();mark(returned='adam_calls')
   torch.cuda.synchronize();train_s+=time.monotonic()-tick
   if check:
    assert shared.topology(model)==topology
    assert all(torch.isfinite(x).all() for x in names.values())
   rec=json_value(dict(suffix_step=cursor,intervention_step=step,observations=keys,
    lr_objective=lr_term,lr_l1=[v.detach() for v in lr_values],lr_mse=[e.detach().square().mean() for e in errors],
    sr_l1=[v.detach() for v in sr_values],reg=reg.detach(),x_loss=x.detach(),x_diagnostics=xdiag,
    gdiag=gdiag,LR_degradation_clamp_fraction=clamp_fractions,
    RGB_out_of_unit_interval_fraction=[((r.detach()<0)|(r.detach()>1)).float().mean() for r in rgbs],
    rgb_forwards=3,moment_forwards=len(moments),adam_calls=2,autograd_calls=attempt['autograd_calls'],regularizer_calls=1,
    train_s=train_s,wall_s=time.monotonic()-started,peak_gb=torch.cuda.max_memory_allocated()/1e9))
   log.write(json.dumps(rec,allow_nan=False)+'\n')
   if cursor==start+1:
    write(a.out/f'first_update_audit_{start:04d}.json',dict(status='passed',suffix_step=cursor,definition=definition,
     observations=keys,rgb_forwards=3,moment_forwards=len(moments),all_graphs_before_two_Adam=True,
     shared_effective_state=same_time,regularizer_calls=1,original_HR_training=False,gdiag=gdiag))
   if check or cursor%100==0:
    print(json.dumps({k:v for k,v in rec.items() if k!='x_diagnostics'},allow_nan=False),flush=True)
   if cursor in (100,3000,6000) or cursor==a.stop:
    last=save_checkpoint(model,a,p,table,cursor,sources,start,train_s,started,calibration_sha)
    chunk_end_wall=time.monotonic()-started
    write(a.out/f'segment_{chunk_start:04d}_{cursor:04d}.json',dict(status='completed_committed_formal_segment',
     method=a.method,repeat=str(a.repeat),start=chunk_start,suffix_endpoint=cursor,updates=cursor-chunk_start,
     training_rgb_forwards=3*(cursor-chunk_start),moment_forwards=(3 if definition['X'] else 0)*(cursor-chunk_start),
     adam_calls=2*(cursor-chunk_start),train_s=train_s-chunk_train,wall_s=chunk_end_wall-chunk_wall,
     peak_gb=torch.cuda.max_memory_allocated()/1e9,checkpoint=entry(last),gpu=torch.cuda.get_device_name(),
     physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),topology_unchanged=True,source_identity=sources,
     calibration_sha256=calibration_sha))
    chunk_start=cursor;chunk_train=train_s;chunk_wall=chunk_end_wall
   del total,rgbs,states,errors,lr_values,sr_values,moments,x,lr_term,reg,degraded_raw,lrs,s
   if same_time:del state
 for rel,identity_hash in sources.items():
  path=Path(os.environ.get('FOURDSR_UPSTREAM','/home/cai_tianshun/Project/4dgs'))/rel[9:] if rel.startswith('upstream/') else ROOT/rel
  assert sha(path)==identity_hash,('Training source changed',rel)
 if calibration_sha:assert sha(OUT/'calibration.json')==calibration_sha,'Frozen coefficient file changed'
 write(a.out/f'image_reads_{start:04d}_{a.stop:04d}.json',dict(actual_reads=dict(reads),all_legal=True,HR_derived_training=False))
 lr_exposure=Counter();sr_exposure=Counter()
 for row in rows[start:a.stop]:
  for index,w in zip(row,definition['lr_weights']):lr_exposure[str(table['record_keys'][index])]+=w
  for index,w in zip(row,definition['sr_weights']):sr_exposure[str(table['record_keys'][index])]+=w
 write(a.out/f'exposure_{start:04d}_{a.stop:04d}.json',dict(schedule_sha256=sha(schedule_path),start=start,stop=a.stop,
  coefficient_weighted_LR=dict(lr_exposure),coefficient_weighted_SR=dict(sr_exposure),
  registered_balance_block=table['exposure_block_size'],no_100_step_balance_claim=True,definition=definition))
 receipt=dict(status='completed_training' if a.stop==6000 else 'completed_formal_segment',method=a.method,repeat=str(a.repeat),
  updates=a.stop-start,suffix_endpoint=a.stop,start=start,training_rgb_forwards=3*(a.stop-start),
  moment_forwards=(3 if definition['X'] else 0)*(a.stop-start),adam_calls=2*(a.stop-start),
  train_s=train_s,wall_s=time.monotonic()-started,peak_gb=torch.cuda.max_memory_allocated()/1e9,
  checkpoint=entry(last),gpu=torch.cuda.get_device_name(),physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
  topology_unchanged=True,source_identity=sources,calibration_sha256=calibration_sha)
 write(a.out/f'attempt_complete_{start:04d}_{a.stop:04d}.json',receipt)
 if a.stop==6000:
  segments=[read(f) for f in sorted(a.out.glob('segment_*.json'))]
  position=0
  for segment in segments:
   assert segment['start']==position and segment['updates']==segment['suffix_endpoint']-segment['start']
   position=segment['suffix_endpoint']
  assert position==6000 and sum(s['updates'] for s in segments)==6000
  receipt.update(updates=6000,training_rgb_forwards=sum(s['training_rgb_forwards'] for s in segments),
   moment_forwards=sum(s['moment_forwards'] for s in segments),adam_calls=sum(s['adam_calls'] for s in segments),
   train_s=sum(s['train_s'] for s in segments),wall_s=sum(s['wall_s'] for s in segments),
   peak_gb=max(s['peak_gb'] for s in segments),segments=[entry(f) for f in sorted(a.out.glob('segment_*.json'))])
  write(a.out/'complete.json',receipt)

if __name__=='__main__':
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--method',choices=METHODS,required=True)
 ap.add_argument('--repeat',choices=['1','2'],default='1');ap.add_argument('--out',type=Path,required=True)
 ap.add_argument('--stop',type=int,default=6000);ap.add_argument('--resume',type=Path);a=ap.parse_args()
 try:main(a)
 except BaseException:
  a.out.mkdir(parents=True,exist_ok=True)
  write(a.out/f'failed_{time.time_ns()}.json',dict(status='failed',traceback=traceback.format_exc(),stop=a.stop,
   resume=str(a.resume) if a.resume else None));raise
