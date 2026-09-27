"""Finalize checkpoint/source indices and exact sampling exposure; no GPU work."""
import collections
import csv
import json
from pathlib import Path
from supplement_identity import sha,write,read,ROOT,OUT

def main():
 protocol=read(OUT/'protocol.json');schedule=read(protocol['schedule']['path']);indices=read(OUT/'checkpoint_manifest.json');indices['new_checkpoints']={};exposure=[];all_status={}
 for phase in ['replay_tails','replay_long']:
  all_status[phase]=read(OUT/phase/'status.json');assert all_status[phase]['status']=='completed'
  jobs=read(OUT/phase/'methods.json')['methods'];seen=set()
  for name,item in jobs.items():
   ck=Path(item['checkpoint']);assert sha(ck)==item['checkpoint_sha256'];indices['new_checkpoints'][name]=dict(path=str(ck),sha256=item['checkpoint_sha256'],run_id=item['run_id'],parent_run_id=item['parent_run_id'],status='completed_evaluated',renderer_id=item['renderer_id'])
   if 'train_dir' not in item or item['train_dir'] in seen:continue
   seen.add(item['train_dir']);config=read(Path(item['train_dir'])/'config.json');count=collections.Counter()
   for li,si,old in schedule['rows'][config['segment_start']:config['segment_stop']]:
    for stream,index in [('LR',li),('SR',si)]:
     camera,frame=schedule['record_keys'][index];count[stream,camera,frame]+=1
   assert sum(count.values())==2*(config['segment_stop']-config['segment_start'])
   for (stream,camera,frame),n in sorted(count.items()):exposure.append(dict(run_id=item['run_id'],stream=stream,camera=camera,frame=frame,count=n,segment_start=config['segment_start'],segment_stop=config['segment_stop'],schedule_sha256=protocol['schedule']['sha256']))
 for arm in ['Shared40','Baked40']:
  all_status[arm]=read(OUT/'fixed_time'/arm/'complete.json')
  for step in [600,1200]:
   ck=OUT/'fixed_time'/arm/f'checkpoint_{step}.pt';identity=read(ck.with_suffix('.json'));assert sha(ck)==identity['sha256'];indices['new_checkpoints'][arm+str(step)]=dict(path=str(ck),sha256=identity['sha256'],run_id=arm,parent_run_id='U6000',status='completed_evaluated')
 write(OUT/'checkpoint_manifest.json',indices)
 old=OUT/'footprint_summary_float32.json'
 if not old.exists():write(old,read(OUT/'footprint_summary.json'))
 summary=read(old);refs=read(OUT/'footprint_reference_v1/complete.json')['rows']
 for row in summary:
  ref=next(x for x in refs if x['run_id']==row['model'] and x['camera']=='cam02' and x['frame']==40)
  stat=next(x for x in ref['rows'] if x['region']==row['region'])
  row.update(base_axes_median=stat['base_axes_median'],child_axes_median=stat['child_axes_median'],base_axes_quantiles=stat['base_axes_quantiles_p10_p50_p90_p99'],child_axes_quantiles=stat['child_axes_quantiles_p10_p50_p90_p99'],projection_precision='float64_svd_actual_float32_factor',reference_path=ref['path'])
  assert row['base_count']==stat['base_count'] and row['child_count']==stat['child_count']
 write(OUT/'footprint_summary.json',summary)
 with (OUT/'sampling_exposure.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(exposure[0]));w.writeheader();w.writerows(exposure)
 # These historical sources are descriptive context, not pooled metric records.
 historical=['docs/dynamic_sr_detail_supervision_2026-09-24.md','docs/dynamic_sr_geometry_residual_2026-09-24.md','docs/dynamic_sr_motion_bound_refinement_2026-09-23.md','docs/dynamic_sr_soft_motion_2026-09-24.md']
 write(OUT/'history_reuse.json',dict(sources=[dict(path=str(ROOT/x),sha256=sha(ROOT/x)) for x in historical],note='No FAS full-scene formal experiment located; FAS/MASI remain DySRGS literature context. Existing ordinary/bound split, S, G/L evidence retained; no reruns or point growth.'))
 write(OUT/'execution_index.json',dict(status='experiments_complete_document_pending',latest_authoritative=['replay_tails/status.json','replay_long/status.json','remaining/status.json','completion_chain/status.json','decision.json'],phases=read(OUT/'decision.json')['phase_status'],superseded_failed_controllers=['p0_engineering_status.json','remaining_initial_pidfd_failure'],checkpoint_count=len(indices['new_checkpoints']),P3='not_run_admission_failed',controllers={k:v.get('status') for k,v in all_status.items()}))
if __name__=='__main__':main()
