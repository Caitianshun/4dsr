"""Portable CPU full-time pose-plus-frozen-parent-overlap schedule entry.

This module preserves the private assignment and strict coarse acceptance
functions. It materializes schedules and statistics, not training plans or
machine service registrations. Data, parents, actual coarse exit/all-SHA
receipts and final assignment acceptance are supplied separately. No candidate
selection, HR quality inspection or GPU launch is performed here.
"""
from __future__ import annotations
import argparse, copy, hashlib, importlib.metadata, json, math, os, random, sys, time
import full_held_confirmation_contract_v2 as held
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
OUT=ROOT/'output/dynamic_sr_multiview_footprint_20261007'
POLICY='full6000_actual_frozen_parent_overlap_pose_assignment_v3'
ORIGINAL_COARSE_SHA='92ea5023e6696b5e32e86d6358b4c3f060b28924eb533631c90fdca93d754a36'
COARSE_OBSERVER_SHA='7daf97ca67982a80a3149e492dd84d345ff6d8901e654acf5d59d8517f2dfdd4'
COARSE_MANAGER_SHA='76faa033d2ab2ebb53c5d73f1138035fff7a58a7ef4e84149a44f7d42c4f7957'
COARSE_SOURCE=COARSE_OBSERVER=COARSE_MANAGER=None
COARSE_SHA=None
NATIVE_PINS={'full_refine_registered_v2.py':'c278928f56b0515adedf15eef9a30850e8bd2a4c41e6f6e0f828068ca774b75d','full_native_protocol.py':'cb4773700bbf535ebb0d9cc94c51f06ca457e04bf305d65ef9afe911746d6713'}

def require(ok,msg):
 if not ok:raise ValueError(msg)

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()

def read(p):return json.loads(Path(p).read_text())

def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()

def path(p,root=ROOT):
 p=Path(p);p=p if p.is_absolute() else root/p
 require(not any(v.is_symlink() for v in [p,*p.parents]) and p.resolve().is_relative_to(root.resolve()),'Path outside registered ROOT or symlink')
 return p.resolve()

def ref(p,root=ROOT):
 p=path(p,root);return dict(path=str(p.relative_to(root)),sha256=sha(p))

def bound(e,root=ROOT):
 require(isinstance(e,dict) and {'path','sha256'}<=set(e),'Missing bound file entry')
 p=path(e['path'],root);require(p.is_file() and sha(p)==e['sha256'],'Bound file SHA differs: '+str(p))
 if 'bytes' in e:require(type(e['bytes']) is int and e['bytes']==p.stat().st_size,'Bound optional file bytes differs: '+str(p))
 return p

def same_reference(a,b,root=ROOT):
 """Path/SHA bind file identity; byte metadata must match actual size."""
 pa=bound(a,root);pb=bound(b,root)
 return pa==pb and a['sha256']==b['sha256']

def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 text=json.dumps(v,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n'
 if p.exists():require(p.read_text()==text,'Immutable materialization differs; use new revision')
 else:
  with p.open('x') as f:f.write(text)
  p.chmod(0o444)
 return p

def immutable_bound(e,root=ROOT):
 p=bound(e,root);require(not p.stat().st_mode & 0o222,'Immutable input must be frozen before registration');return p

def pose_rows(manifest):
 cameras=list(manifest['splits']['train']);require(3<=len(cameras)<=20 and 'cam00' not in cameras and 'cam01' in cameras and not manifest['splits']['dev'],'Only complete official legal training cameras')
 require(manifest['frame_indices']==list(range(300)),'Full300 frames required')
 centers={c:[float(manifest['cameras'][c]['c2w'][i][3]) for i in range(3)] for c in cameras}
 axes={c:[float(manifest['cameras'][c]['c2w'][i][2]) for i in range(3)] for c in cameras}
 require(all(math.isfinite(x) for q in [*centers.values(),*axes.values()] for x in q),'Nonfinite training pose')
 pairs={}
 for i,a in enumerate(sorted(cameras)):
  for b in sorted(cameras)[i+1:]:
   den=math.sqrt(sum(x*x for x in axes[a])*sum(x*x for x in axes[b]));require(den>0,'Degenerate optical axis')
   cosine=max(-1.,min(1.,sum(x*y for x,y in zip(axes[a],axes[b]))/den))
   pairs[(a,b)]=dict(camera_center_distance=math.dist(centers[a],centers[b]),optical_axis_angle_radians=math.acos(cosine))
 return cameras,pairs,max([r['camera_center_distance'] for r in pairs.values()]+[1e-9])

def assignment(cost):
 """Hungarian exact assignment, small20x20 pose-only cost, no dependencies."""
 n=len(cost);u=[0.]*(n+1);v=[0.]*(n+1);p=[0]*(n+1);way=[0]*(n+1)
 for i in range(1,n+1):
  p[0]=i;j0=0;minv=[float('inf')]*(n+1);used=[False]*(n+1)
  while True:
   used[j0]=True;i0=p[j0];delta=float('inf');j1=0
   for j in range(1,n+1):
    if not used[j]:
     cur=cost[i0-1][j-1]-u[i0]-v[j]
     if cur<minv[j]:minv[j]=cur;way[j]=j0
     if minv[j]<delta:delta=minv[j];j1=j
   require(math.isfinite(delta),'No finite exact camera assignment')
   for j in range(n+1):
    if used[j]:u[p[j]]+=delta;v[j]-=delta
    else:minv[j]-=delta
   j0=j1
   if p[j0]==0:break
  while True:
   j1=way[j0];p[j0]=p[j1];j0=j1
   if j0==0:break
 out=[0]*n
 for j in range(1,n+1):out[p[j]-1]=j-1
 require(set(out)==set(range(n)),'Incomplete exact assignment');return out

def actual_overlap(accepted_ref,manifest,manifest_ref,parent_ref,seed,root=ROOT):
 """Consume original exact Exit0 + full raw-SHA acceptance. Do not reopen NPZ.

 The accepted source already hashes all raw packets. Here we verify all300
 scalar frame JSONs and their populations/ratios against the immutable source
 receipt, plus the full cost and unchanged native model/Adam/RNG boundary.
 """
 require(accepted_ref is not None,'Actual accepted parent coarse overlap remains pending')
 accepted=read(immutable_bound(accepted_ref,root));require(accepted.get('status')=='accepted_actual_owned_coarse_manager_exact_Exit0_and_immediate_all_output_SHA','Only actual precise manager Exit0/all-output SHA acceptance supported')
 require(accepted['observer_source']['sha256']==COARSE_OBSERVER_SHA and accepted['observer_source']['path']==str(COARSE_OBSERVER.relative_to(ROOT)),'Coarse exact observer source differs');bound(accepted['observer_source'],root)
 registration=read(bound(accepted['registration'],root));t=accepted['terminal'];s=t['systemd']
 require(registration['status']=='registered_actual_root_coarse_owned_service' and t['kind']=='terminal' and t['unit']==registration['unit'] and t['invocation_id']==registration['invocation_id'],'Different actual coarse owner/invocation')
 require(t['successful'] is True and t['exit_proved'] is True and t['exit_kind']==1 and t['exit_code']==0 and s['MainPID']=='0' and s['ExecMainPID']==str(registration['main_pid']) and s['ExecMainStartTimestampMonotonic']==registration['systemd']['ExecMainStartTimestampMonotonic'] and s['ExecMainCode']=='1' and s['ExecMainStatus']=='0' and s['Result']=='success' and int(s['ExecMainExitTimestampMonotonic'])>0,'Coarse exact normal Exit0 absent')
 require(s.get('LoadState')!='not-found' and (s['InvocationID']==registration['invocation_id'] or s['InvocationID']=='' and t.get('retained_ExecMainPID_and_start_match') is True),'Coarse exact terminal identity unavailable')
 chain=read(bound(accepted['exit_chain'],root));require(chain['status']=='accepted_actual_producer_normal_return_full_output_SHA_and_complete_population' and chain['all_output_raw_SHA_verified'] is True,'No producer all-SHA immediate return check')
 done=read(bound(chain['producer_complete'],root));index=read(bound(accepted['index'],root));plan=read(bound(index['registration'],root))
 require(chain['index']['path']==accepted['index']['path'] and chain['index']['sha256']==accepted['index']['sha256'] and {k:done['index'][k] for k in ('path','sha256')}=={k:accepted['index'][k] for k in ('path','sha256')},'Coarse index completion identities differ')
 require(plan['schema']=='registered_frozen_native_parent_coarse_overlap_plan_v2','Failed v1 observer lifecycle cannot supply V3 assignment')
 require(index['schema']=='frozen_native_parent_coarse_overlap_v1' and done['status']==index['status']=='completed_frozen_native_parent_coarse_overlap','Foreign/pending coarse result')
 context=dict(manifest=manifest_ref,parent=parent_ref)
 require(done['context']==index['context']==plan['context']==context and index['scene']==manifest['scene'] and index['seed']==seed and plan['scene']==manifest['scene'] and plan['seed']==seed,'Coarse parent/manifest/seed differs')
 require(index['source']['sha256']==COARSE_SHA and index['source']['path']==str(COARSE_SOURCE.relative_to(ROOT)) and index['source']==done['source'],'Frozen actual coarse producer source differs');bound(index['source'],root)
 require(index['policy']==plan['policy'] and index['policy_sha256']==digest(index['policy']) and index['source_files']==plan['source_files'],'Coarse policy/source lineage differs')
 auth=read(bound(registration['authorization'],root));require(auth['producer_source']==index['source'] and auth['manager_source']['sha256']==COARSE_MANAGER_SHA and auth['manager_source']['path']==str(COARSE_MANAGER.relative_to(ROOT)),'Actual producer v2/manager typed closure lineage differs');bound(auth['manager_source'],root)
 typed=chain['producer_typed_observer_and_native_closure'];require(typed['status']=='CPU_all_SHA_coarse_parent_overlap_accepted_pending_exact_unit_Exit0' and typed['parent_state_unchanged'] is True and typed['model_imported'] is False and typed['GPU_queries']==0,'Source native/observer typed closure absent')
 invariant=index['invariants'];require(invariant['before']==invariant['after'] and invariant['parent_state_unchanged'] is True and invariant['observer_RNG_restored'] is True,'Parent/observer model/Adam/RNG changed')
 startup=read(bound(invariant['observer_RNG_checks']['startup'],root));restore_observer=read(bound(invariant['observer_RNG_checks']['restoration'],root));nativeproof=read(bound(invariant['native_invariance'],root));bound(startup['snapshot'],root)
 require(nativeproof==dict(before=invariant['before'],after=invariant['after'],exact_equal=True,scope='native_parent_before_observer_restore'),'Native model invariant proof differs')
 require(startup['status']=='passed_exact_observer_RNG_startup_selfcheck' and startup['policy']==plan['observer_RNG_policy'] and startup['before']==startup['after'] and startup['exact_equal'] is True and startup['native_model_restore_started'] is False and startup['native_moment_forwards']==0 and startup['before']['CUDA_state_count']==1,'Actual startup CUDA RNG selfcheck absent')
 require(restore_observer['status']=='passed_exact_observer_RNG_restoration' and restore_observer['policy']==plan['observer_RNG_policy'] and restore_observer['scope']=='after_native_parent_check' and restore_observer['before']==restore_observer['after']==startup['before'] and restore_observer['exact_equal'] is True,'Observer post-native restoration proof absent')
 require(all(invariant['restore'].get(k) is True for k in ('one_Adam_exact','parameters_exact','topology_buffers_exact','saved_gradients_exact','global_RNG_exact','no_children')) and invariant['restore']['optimizer_conversion'] is False,'Native14tuple/oneAdam restore contract absent')
 require(index['RGB_forwards']==index['Adam_calls']==index['formal_updates']==index['autograd_calls']==index['training_pixel_decodes']==index['teacher_reads']==0 and index['HR_pixels_read'] is False and index['heldout_pixels_read'] is False,'Forbidden coarse pixel or update operation')
 cams,poses,scale=pose_rows(manifest);wanted={(c,f) for c in cams for f in range(300)}
 require(index['total_native_moment_forwards']==plan['planned']['native_moment_forwards']==len(wanted) and index['prior_reused_moments']==0 and len(index['raw_moments'])==len(wanted) and {(r['camera'],r['frame']) for r in index['raw_moments']}==wanted,'Actual coarse full legal moment population differs')
 journal=read(bound(done['attempt'],root));require(journal==done['actual_cost'] and journal['status']=='completed_frozen_parent_coarse_overlap' and journal['active_operation'] is None and journal['attempted_moments']==journal['completed_moments']==len(wanted) and journal['attempted_shared_states']==journal['completed_shared_states']==300 and journal['completed_directed_pairs']==300*len(cams)*(len(cams)-1),'Actual complete cost/counter population differs')
 require(all(type(journal[k]) in (float,int) and math.isfinite(journal[k]) and journal[k]>0 for k in ('seconds','moment_seconds','pair_seconds')),'Actual nonzero producer cost absent')
 require(chain['frames']==300 and chain['native_moments']==len(wanted) and chain['directed_pairs']==journal['completed_directed_pairs'] and chain['checked_output_files']==len(wanted)*2+300 and chain['checked_output_bytes']>0,'Immediate all-output receipt population differs')
 require(len(index['frames'])==300 and len({r['path'] for r in index['frames']})==300,'Actual fullframe scalar table incomplete')
 frames={}
 for reference in index['frames']:
  value=read(bound(reference,root));f=value['frame'];require(type(f) is int and f in range(300) and f not in frames and value['context']==context and value['policy_sha256']==index['policy_sha256'] and value['time']==f/300,'Frame scalar identity differs')
  rows=value['pairs'];require(len(rows)==len(poses) and {(r['a'],r['b']) for r in rows}==set(poses),'Actual frame pair population differs')
  lookup={}
  for r in rows:
   scores=[]
   for direction in ('a_to_b','b_to_a'):
    q=r[direction];num,den=q['visible_target_pixels'],q['supported_target_pixels'];require(type(num) is int and type(den) is int and 0<=num<=den and q['ratio']==(num/den if den else None),'Directional overlap counts/ratio differ');scores.append(q['ratio'])
   require(r['ratio']==(min(scores) if all(q is not None for q in scores) else None),'Conservative min overlap ratio differs')
   pose=poses[(r['a'],r['b'])]
   for k,v in pose.items():require(math.isfinite(r[k]) and math.isclose(r[k],v,rel_tol=1e-11,abs_tol=1e-11),'Coarse baseline/angle differs from actual training pose')
   lookup[(r['a'],r['b'])]=r
  frames[f]=lookup
 return dict(accepted=accepted_ref,index=accepted['index'],registration=index['registration'],context=context,scene=manifest['scene'],seed=seed,frames=frames,frame_entries=index['frames'],poses=poses,baseline_scale=scale,cost=journal,postcheck_cost=chain['cost'],checked_raw_payloads_reopened=0)

def prepared_overlap(overlap,pose_ref):
 return dict(schema='registered_actual_native_parent_overlap_schedule_inputs_v1',status='registered_actual_parent_overlap_inputs_from_exact_Exit0_full_SHA',context=overlap['context'],scene=overlap['scene'],seed=overlap['seed'],source=ref(COARSE_SOURCE),pose_table=pose_ref,coarse_overlap_table=overlap['index'],root_acceptance=overlap['accepted'],coarse_registration=overlap['registration'],frames=overlap['frame_entries'],cost=dict(wall_seconds=overlap['cost']['seconds'],native_moment_forwards=overlap['cost']['completed_moments'],CPU_postcheck=overlap['postcheck_cost']),actual_parent_rendering_completed=True,synthetic=False,HR_GT_reads=0,heldout_reads=0,raw_payload_not_rehashed=True)

def schedule(manifest,manifest_ref,parent_ref,seed,overlap,prepared_ref):
 cams,poses,scale=pose_rows(manifest);require(overlap['context']==dict(manifest=manifest_ref,parent=parent_ref) and overlap['scene']==manifest['scene'] and overlap['seed']==seed,'Actual overlap context differs before assignment')
 keys=[(r['camera_id'],int(r['frame_index'])) for r in manifest['observations'] if r['split']=='train'];lookup={k:i for i,k in enumerate(keys)}
 require(len(keys)==len(lookup)==len(cams)*300 and set(keys)=={(c,f) for c in cams for f in range(300)},'Incomplete/foreign legal LR observations')
 rng=random.Random(seed);rows=[];cal=[];pair_freq={};selected=[];unknown=0
 for frame in range(300):
  camera_order=cams[:];rng.shuffle(camera_order);anchors=camera_order+[camera_order[(frame+j)%len(cams)] for j in range(20-len(cams))];rng.shuffle(anchors)
  def score(a,b):
   q=overlap['frames'][frame][tuple(sorted((a,b)))];ratio=q['ratio']
   return q['camera_center_distance']/scale+q['optical_axis_angle_radians']/math.pi+(1-ratio if ratio is not None else 1.)
  def match(exclusions,other=None):
   costs=[[1e9 if b in exclusions[i] else score(a,b)+(score(other[i],b) if other is not None else 0)+1e-10*rng.random() for b in anchors] for i,a in enumerate(anchors)]
   perm=assignment(costs);require(all(costs[i][j]<1e8 for i,j in enumerate(perm)),'No legal disjoint exact camera assignment');return [anchors[j] for j in perm]
  second=match([{c} for c in anchors]);third=match([{a,b} for a,b in zip(anchors,second)],second)
  for a,b,c in zip(anchors,second,third):
   row=[lookup[(x,frame)] for x in (a,b,c)];rows.append(row)
   pairrows=[]
   for x,y in ((a,b),(a,c),(b,c)):
    q=overlap['frames'][frame][tuple(sorted((x,y)))];unknown+=q['ratio'] is None;pairrows.append(dict(a=x,b=y,ratio=q['ratio'],baseline=q['camera_center_distance'],angle_radians=q['optical_axis_angle_radians']))
    for u,v in ((x,y),(y,x)):pair_freq[u+'->'+v]=pair_freq.get(u+'->'+v,0)+1
   selected.append(dict(frame=frame,indices=row,pairs=pairrows))
  if frame in (0,100,200,299):
   for c in cams:cal.append(next(row[:] for row in rows[-20:] if keys[row[0]]==(c,frame)))
 rng.shuffle(rows);columns=[[r[i] for r in rows] for i in range(3)]
 for column in columns:rng.shuffle(column)
 random_rows=[list(row) for row in zip(*columns)]
 from collections import Counter
 exposure=[Counter(r[i] for r in rows) for i in range(3)]
 require(exposure[0]==exposure[1]==exposure[2] and all(Counter(r[i] for r in random_rows)==exposure[i] for i in range(3)),'Exposure match failed')
 bands={'null':unknown,'[0,.25)':0,'[.25,.5)':0,'[.5,.75)':0,'[.75,1]':0}
 for row in selected:
  for q in row['pairs']:
   v=q['ratio']
   if v is not None:bands['[0,.25)' if v<.25 else '[.25,.5)' if v<.5 else '[.5,.75)' if v<.75 else '[.75,1]']+=1
 require(sum(bands.values())==18000,'Selected three-pair count differs')
 return dict(schema='full_native_SR_schedule_v1',identity=dict(manifest=manifest_ref,parent=parent_ref),seed=seed,updates=6000,record_keys=[list(k) for k in keys],rows=rows,random_rows=random_rows,calibration_triplets=cal,audit=dict(passed=True,full_legal_observations=len(keys),every_legal_key_covered=True,three_column_exact_exposure=True,random_same_corresponding_column_exact_exposure=True,full6000_block_only=True),assignment_protocol=dict(schema=POLICY,source=ref(Path(__file__)),cost='unit-weight baseline/max_train_baseline + optical_axis_angle/pi + (1-conservative_actual_parent_overlap); unknown ratio uses fixed penalty1 and stays null. Third column minimizes sum of costs against anchor and second. Exact assignment per real frame; seeded1e-10 tie only.',parent_overlap='actual_frozen_parent_coarse_overlap',parent_overlap_inputs=prepared_ref,baseline_angle_overlap_proportions_registered=True,HR_or_test_or_quality_cost=False,anchor_policy='20slots/realframe; all legal cameras once plus rotating duplicate slots; each column matches exact exposure, no claim uniform once except Cook6000',calibration='all actual train cameras×frames0/100/200/299',pair_frequency=pair_freq,baseline_distance_scale=scale,selected_pairs=selected,overlap_bins=dict(counts=bands,proportions={k:v/18000 for k,v in bands.items()}),null_overlap_not_ground_truth=True))

def resolve_schedule(sr,manifest,manifest_ref,parent_ref,seed,overlap,prepared_ref,existing=None,root=ROOT):
 expected=schedule(manifest,manifest_ref,parent_ref,seed,overlap,prepared_ref);sr.validate_schedule(expected,manifest,dict(manifest=manifest_ref,parent=parent_ref),seed)
 if existing is None:return expected,None
 p=immutable_bound(existing,root);table=read(p);require(table==expected,'Exact reused V3 schedule/context/source/policy/frame scalar assignment differs');sr.validate_schedule(table,manifest,dict(manifest=manifest_ref,parent=parent_ref),seed)
 return table,p

def assignment_candidate(table,schedule_ref,prepared_ref,overlap,pose_ref,statistics_ref):
 return dict(status='pending_root_actual_assignment_storage_and_cost_acceptance',required_final_status='root_accepted_actual_frozen_parent_coarse_overlap_assignment',full_native_context=dict(table['identity'],schedule=schedule_ref),parent=table['identity']['parent'],manifest=table['identity']['manifest'],schedule=schedule_ref,prepared_overlap_inputs=prepared_ref,source=ref(Path(__file__)),pose_table=pose_ref,coarse_overlap_table=overlap['index'],assignment_statistics=statistics_ref,root_acceptance=overlap['accepted'],actual_parent_rendering_completed=True,synthetic=False,pose_plus_parent_overlap=True,HR_GT_reads=0,heldout_reads=0,source_registration_accepted=False,storage_and_cost_accepted=False,cost=dict(wall_seconds=overlap['cost']['seconds'],native_moment_forwards=overlap['cost']['completed_moments'],shared_once_between_B0_and_Bsync=True,not_recharged_on_revision=True),GPU_dispatch_allowed=False)

def accept_assignment(reference,candidate,root=ROOT):
 if reference is None:return None
 value=read(immutable_bound(reference,root));require(value['status']=='root_accepted_actual_frozen_parent_coarse_overlap_assignment' and value['source_registration_accepted'] is True and value['storage_and_cost_accepted'] is True,'Root actual assignment/storage/cost acceptance pending')
 expected={k:v for k,v in candidate.items() if k not in ('status','required_final_status','source_registration_accepted','storage_and_cost_accepted','GPU_dispatch_allowed')}
 require(all(value.get(k)==v for k,v in expected.items()),'Root assignment acceptance changed actual source/context/statistics/cost')
 for name in ('source','pose_table','coarse_overlap_table','assignment_statistics','root_acceptance','prepared_overlap_inputs','schedule'):bound(value[name],root)
 return reference

def fixed_train_diagnostics(manifest):
 cameras=list(manifest['splits']['train']);require('cam01' in cameras and 'cam00' not in cameras,'Full official train camera01 required')
 keys=[['cam01',f] for f in range(300)]+[[c,f] for c in cameras if c!='cam01' for f in (40,80,120,160)]
 legal={(r['camera_id'],int(r['frame_index'])) for r in manifest['observations'] if r['split']=='train'}
 require(len(keys)==len({tuple(k) for k in keys})==300+4*(len(cameras)-1) and {tuple(k) for k in keys}<=legal,'Fixed training-only diagnostics population differs')
 return keys

def configure_closed_sources(accepted_ref,confirmation_authorization):
 """Bind new declared held lineage; never relabel an old dev receipt."""
 global COARSE_SOURCE,COARSE_OBSERVER,COARSE_MANAGER,COARSE_SHA,COARSE_OBSERVER_SHA,COARSE_MANAGER_SHA
 authorization=held.validate_authorization(confirmation_authorization)
 accepted=read(immutable_bound(accepted_ref));index=read(bound(accepted['index']))
 require(index['scene'] in held.HELD and index['seed']==20261007,'Only exact held parent overlap accepted')
 lineage=authorization['coarse_operational_sources'][index['scene']]
 source=index['source'];portable=HERE/'full_held_parent_overlap.py'
 require(source==lineage['producer'] and source['sha256']==sha(portable) and source['path']==str(portable.relative_to(ROOT)),'Held scientific producer source differs')
 COARSE_SOURCE=bound(source);COARSE_SHA=source['sha256']
 require(accepted['observer_source']==lineage['observer'],'Declared actual held observer source differs')
 COARSE_OBSERVER=bound(accepted['observer_source']);COARSE_OBSERVER_SHA=lineage['observer']['sha256']
 registration=read(bound(accepted['registration']));operational=read(bound(registration['authorization']))
 require(operational['manager_source']==lineage['manager'] and operational['producer_source']==source,'Declared actual held manager/producer source differs')
 COARSE_MANAGER=bound(operational['manager_source']);COARSE_MANAGER_SHA=lineage['manager']['sha256']
 return accepted

def native_metadata():
 require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU scheduling explicitly hides CUDA')
 require(sys.version_info[:2]==(3,10) and importlib.metadata.version('numpy')=='1.26.4','Use native Python3.10/NumPy1.26.4 metadata runtime')
 require(not any(n in sys.modules for n in ('torch','numpy','scene','gaussian_renderer')),'No tensor/model import during scheduling')
 for name,pin in NATIVE_PINS.items():require(sha(HERE/name)==pin,'Frozen native metadata source differs: '+name)
 if str(HERE) not in sys.path:sys.path.insert(0,str(HERE))
 import full_refine_registered_v2 as sr
 return sr

def materialize_schedule(a):
 """Fresh schedule A or exact-entry refresh B; never authorize training."""
 started=time.monotonic();confirmation_authorization=ref(path(a.confirmation_authorization));held.validate_authorization(confirmation_authorization);sr=native_metadata();upstream=Path(a.upstream).resolve()
 require(os.environ.get('FOURDSR_UPSTREAM')==str(upstream),'Explicit upstream environment mismatch')
 manifestp=path(a.manifest);manifest,data,_=sr.prefix.validate_manifest(manifestp,a.seed)
 require(manifest['role']=='held_confirmation' and manifest['scene'] in held.HELD and a.seed==20261007,'Only two exact held parents after actual single-method freeze')
 parent,_=sr.full_parent(path(a.parent),path(a.parent_complete),data['manifest'],a.seed)
 held.held_parent_scope(manifest,parent,a.seed)
 accepted_ref=ref(path(a.coarse_acceptance));configure_closed_sources(accepted_ref,confirmation_authorization)
 overlap=actual_overlap(accepted_ref,manifest,data['manifest'],parent['checkpoint'],a.seed)
 out=path(a.out);namespace=OUT/'held_confirmation_native_schedules'
 require(out.is_relative_to(namespace) and out!=namespace and not out.exists(),'Fresh independent schedule-only output required')
 existing=ref(path(a.reuse_schedule)) if a.reuse_schedule else None
 out.mkdir(parents=True,exist_ok=False)
 try:
  preparedp=bound(read(immutable_bound(existing))['assignment_protocol']['parent_overlap_inputs']) if existing else out/'overlap_inputs.json'
  posep=bound(read(preparedp)['pose_table']) if existing else out/'training_pose.json'
  posevalue=dict(schema='actual_manifest_train_pose_pairs_v1',manifest=data['manifest'],pairs=[dict(a=k[0],b=k[1],**v) for k,v in overlap['poses'].items()],baseline_scale=overlap['baseline_scale'])
  if not existing:write(posep,posevalue)
  poseref=ref(posep);preparedvalue=prepared_overlap(overlap,poseref)
  if existing:require(read(preparedp)==preparedvalue and read(posep)==posevalue,'Exact reused preparation and pose entries differ')
  else:write(preparedp,preparedvalue)
  preparedref=ref(preparedp);table,reused=resolve_schedule(sr,manifest,data['manifest'],parent['checkpoint'],a.seed,overlap,preparedref,existing)
  sp=reused or out/'schedule.json'
  if reused is None:write(sp,table)
  schedule_ref=copy.deepcopy(existing) if reused is not None else ref(sp)
  stats=dict(schema='actual_parent_overlap_assignment_statistics_v3',schedule=schedule_ref,prepared_overlap_inputs=preparedref,selected_pairs=table['assignment_protocol']['selected_pairs'],pair_frequency=table['assignment_protocol']['pair_frequency'],overlap_bins=table['assignment_protocol']['overlap_bins'],full_legal_observations=len(table['record_keys']),calibration_anchors=len(table['calibration_triplets']),RGB_column_exposure_equal=True,random_same_RGB_column_exposure_equal=True,teacher_weighted_exposure_equal_B0_Bsync=True,HR_or_quality_used=False)
  stats_path=reused.parent/'assignment_statistics.json' if reused else out/'assignment_statistics.json';write(stats_path,stats)
  candidate=assignment_candidate(table,schedule_ref,preparedref,overlap,poseref,ref(stats_path))
  assignment_ref=ref(path(a.assignment_acceptance)) if a.assignment_acceptance else None
  accepted_assignment=accept_assignment(assignment_ref,candidate)
  write((reused.parent if reused else out)/'assignment_acceptance_candidate.json',candidate)
  result=dict(schema='portable_full_native_schedule_materialization_v1',status='completed_CPU_schedule_only_exact_reuse' if existing else 'completed_CPU_schedule_only_assignment_candidate',source=ref(Path(__file__)),confirmation_authorization=confirmation_authorization,manifest=data['manifest'],parent=parent['checkpoint'],schedule=schedule_ref,prepared_overlap_inputs=preparedref,pose_table=poseref,assignment_statistics=ref(stats_path),assignment_candidate=candidate,assignment_acceptance=accepted_assignment,schedule_reused_exact_entry=bool(existing),full_native_context=dict(table['identity'],schedule=schedule_ref),train_diagnostic_keys=fixed_train_diagnostics(manifest),parent_preparation_cost_shared_once=dict(wall_seconds=overlap['cost']['seconds'],native_moment_forwards=overlap['cost']['completed_moments'],not_recharged_on_revision=True),GPU_dispatch_allowed=False,native_training_plan_registered=False,formal_updates=0,Adam_calls=0,model_forwards=0,GPU_queries=0,HR_or_quality_read=False,CPU_wall_seconds=time.monotonic()-started)
  require(not any(n in sys.modules for n in ('torch','numpy','scene','gaussian_renderer')),'Scheduling imported model/tensor')
  write(out/'complete.json',result);return result
 except BaseException as error:
  write(out/'failure.json',dict(status='failed_CPU_schedule_only_partial_preserved',error=repr(error),source=ref(Path(__file__)),GPU_dispatch_allowed=False,formal_updates=0,Adam_calls=0));raise

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ('manifest','parent','parent-complete','coarse-acceptance','upstream','out'):p.add_argument('--'+name,required=True)
 p.add_argument('--confirmation-authorization',required=True)
 p.add_argument('--seed',type=int,required=True);p.add_argument('--reuse-schedule');p.add_argument('--assignment-acceptance')
 a=p.parse_args();value=materialize_schedule(a);print(json.dumps(dict(status=value['status'],schedule=value['schedule'],GPU_dispatch_allowed=False,CPU_wall_seconds=value['CPU_wall_seconds'])))

if __name__=='__main__':main()
