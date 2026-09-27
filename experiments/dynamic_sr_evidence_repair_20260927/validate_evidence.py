"""Integrity assertions for the completed evidence contract and paired probes."""
import csv
import json
from pathlib import Path
from supplement_identity import sha,read,write,OUT,ROOT

def main():
 protocol=read(OUT/'protocol.json');decision=read(OUT/'decision.json');cost=read(OUT/'cost.json');assert decision['status']=='completed';assert cost['major_updates']==33750
 s=OUT/'fixed_time/Shared40';b=OUT/'fixed_time/Baked40';assert read(s/'sequence.json')==read(b/'sequence.json')
 for p in [s,b]:
  cfg=read(p/'config.json');assert cfg['checkpoint_sha256']==protocol['start_sha256'];assert cfg['point_count']==132972;assert cfg['initial_adam_states']==0 and not cfg['regularization'];assert read(p/'initial_parity.json')['passed'];assert read(p/'complete.json')['source_unchanged']
 files=[]
 for mode in ['Shared40','Baked40','references']:
  p=OUT/'probe_evaluation'/mode/'complete.json';j=read(p)
  for n,item in j['methods'].items():
   assert sha(item['checkpoint'])==item['checkpoint_sha256']
   if mode!='references':assert len(item['rows'])==21
   for fp in item['footprints']:assert sha(fp['path'])==fp['sha256'];files.append(fp['path'])
 sampled_updates=0;schedule=read(protocol['schedule']['path'])['rows']
 for p in (OUT/'replay_long').glob('*/train/complete.json'):
  complete=read(p);assert complete['source_unchanged'];cfg=read(p.parent/'config.json')
  rows=[json.loads(line) for line in (p.parent/'sampling.jsonl').read_text().splitlines()]
  assert len(rows)==cfg['segment_stop']-cfg['segment_start']==complete['actual_updates']
  for step,row in enumerate(rows,cfg['segment_start']+1):
   assert row['step']==step
   assert [row['lr_index'],row['sr_index']]==schedule[step-1][:2]
  assert rows[-1]['prefix_sha256']==complete['metadata']['draw_sha256'];sampled_updates+=len(rows)
 assert sampled_updates==30450
 references=read(OUT/'footprint_reference_v1/complete.json')['rows']
 for item in references:
  assert sha(item['path'])==item['sha256']
  assert sha(item['old_path'])==item['old_sha256']
  assert item['candidates']>0
 with (OUT/'per_frame_metrics.csv').open() as f:
  reader=csv.DictReader(f);required={'run_id','parent_run_id','checkpoint_sha256','renderer_id','metric_version','camera','frame','split','value','source_path'};assert required<=set(reader.fieldnames)
  count=0
  for r in reader:
   assert all(r[k] not in ('',None) for k in required),r;float(r['value']);assert Path(r['source_path']).exists();count+=1
 d=read(OUT/'depth_admission.json');assert not d['passed'];assert d['nonzero_depth_steps']==69;assert read(OUT/'rank_support_check.json')['minimum_participating_tracks']>=10
 write(OUT/'integrity.json',dict(status='passed',major_updates=cost['major_updates'],metric_rows=count,footprints_checked=len(files),stable_footprint_references_checked=len(references),actual_long_sampling_rows_checked=sampled_updates,same_fixed_sequence=True,HR_training_reads='Rejected by training audit hooks',source='validate_evidence.py'))
 print('passed',count,'metric rows',len(files),'footprints')
if __name__=='__main__':main()
