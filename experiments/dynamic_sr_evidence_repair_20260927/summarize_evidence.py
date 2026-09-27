"""Aggregate immutable receipts; never pool historical renderer versions."""
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'output/dynamic_sr_evidence_repair_20260927';PRIOR=ROOT/'output/dynamic_sr_prior_guidance_20260927'
read=lambda p:json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False))
def csvwrite(p,rows):
 keys=list(dict.fromkeys(k for r in rows for k in r))
 with Path(p).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def flatten(x,prefix=''):
 if isinstance(x,dict):
  for k,v in x.items():yield from flatten(v,prefix+'.'+k if prefix else k)
 elif isinstance(x,(float,int)) and not isinstance(x,bool):yield prefix,x

def fullmetric(j):
 a=j['aggregate'];t=j.get('temporal_aggregate',{});l=j['lr_reprojection_aggregate'];return dict(psnr=a['full']['psnr_mean'],ssim=a['full']['ssim_mean'],lpips=a['full']['lpips_alex_mean'],dynamic_lpips=a['dynamic']['lpips_alex_spatial_mask_mean'],temporal=t.get('dynamic',{}).get('gt_relative_warp_l1_mean'),lr_l1=l['full']['l1_mean'])
def contrast(a,b):
 return {k:(a[k]-b[k] if k in ['psnr','ssim'] else a[k]/b[k]-1) for k in a if a[k] is not None and b[k] is not None}
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--partial',action='store_true');a=parser.parse_args();protocol=read(OUT/'protocol.json');replay={};metricrows=[];runs=[];cost=[]
 def record(rows,identity):
  for row in rows:
   ident={**identity,'camera':row.get('camera_id',row.get('camera')),'frame':row.get('frame_index',row.get('frame'))}
   for k,v in row.items():
    if k in ['spatial','temporal','h_hr','h_teacher','lr_reprojection','lr','hr','teacher_rgb','teacher_H','hr_H','teacher_hr_H','teacher_hr','roi']:
     for key,val in flatten(v,k):metricrows.append(dict(**ident,metric=key,value=val))
 for phase in ['replay_tails','replay_long']:
  if not (OUT/phase/'methods.json').exists():continue
  methods=read(OUT/phase/'methods.json')['methods']
  if not a.partial:assert read(OUT/phase/'status.json')['status']=='completed'
  seen=set()
  for name,item in methods.items():
   value={};value_sources={};ident=dict(run_id=name,parent_run_id=item['parent_run_id'],checkpoint_sha256=item['checkpoint_sha256'],renderer_id=item['renderer_id'])
   for split,folder in item['evaluations'].items():
    f=Path(folder)/'metrics.json';j=read(f);assert j['manifest_sha256']==protocol['manifest']['sha256'];assert j['checkpoint_sha256']==item['checkpoint_sha256']
    assert j['version']=='detail_supervision_eval_v1',j['version']
    label={'test':'cam00','dev':'cam01','train_fixed':'train16'}[split];value[label]=fullmetric(j);value_sources[label]=str(f)
    record(j['rows'],dict(**ident,metric_version=j['version'],split=split,source_path=str(f)))
   if 'train76' in item:
    value_sources['train76']=str(Path(item['train76'])/'metrics.json');t76=read(value_sources['train76']);value['train76']=t76['aggregate'];assert t76['manifest_sha256']==protocol['manifest']['sha256']
    for r76 in t76['rows']:
     for key in ['mse','l1','psnr']:metricrows.append(dict(**ident,camera=r76['camera_id'],frame=r76['frame_index'],split='train76',metric_version='train76_raw_lr_v1',metric=key,value=r76[key],source_path=value_sources['train76']))
   replay[name]=value
   for split,v in value.items():runs.append(dict(**ident,split=split,camera=split if split.startswith('cam') else 'train_group',frame='fixed_aggregate',metric_version='detail_supervision_eval_v1' if split!='train76' else 'train76_raw_lr_v1',source_path=value_sources[split],**v))
   if 'train_dir' in item and item['train_dir'] not in seen:
    seen.add(item['train_dir']);train=Path(item['train_dir']);complete=read(train/'complete.json');config=read(train/'config.json');assert complete['source_unchanged'];assert config['visible_cuda']==protocol['physical_gpu']
    cost.append(dict(run_id=item['run_id'],phase='P0',actual_updates=complete['actual_updates'],train_s=complete['train_s'],elapsed_s=complete['elapsed_s'],peak_gb=complete['peak_gb'],gpu=config['gpu']))
 comparisons={}
 def compare(label,x,y):
  if x in replay and y in replay:comparisons[label]={split:contrast(replay[x][split],replay[y][split]) for split in ['cam00','cam01','train16']}
 for s in [17600,17700,17850,18000]:compare('Tail2_vs_Tail1_'+str(s),'Tail2_'+str(s),'Tail1_'+str(s))
 for s in [12000,17550,18000]:compare('LiveB_vs_LiveA_'+str(s),'LiveB_'+str(s),'LiveA_'+str(s))
 for s in [17550,18000]:compare('Restart12_vs_LiveA_'+str(s),'Restart12_'+str(s),'LiveA_'+str(s))
 compare('Restart17550_vs_LiveA_18000','Restart17550_18000','LiveA_18000')
 compare('readonly_repeat','readonly2','readonly1')
 sensitivity={k:{cam:(abs(v[cam]['psnr'])>.05 or abs(v[cam]['dynamic_lpips'])>.01 or (v[cam]['temporal'] is not None and abs(v[cam]['temporal'])>.01)) for cam in ['cam00','cam01']} for k,v in comparisons.items() if k!='readonly_repeat'}
 probe={};fixed=[];teacher=[]
 for mode in ['Shared40','Baked40','references']:
  file=OUT/'probe_evaluation'/mode/'complete.json'
  if not file.exists():continue
  j=read(file)
  for name,item in j['methods'].items():
   groups={split:[r for r in item['rows'] if r['frame']==40 and (r['split']=='train' if split=='train19' else r['camera']==split)] for split in ['train19','cam00','cam01']}
   if mode=='references':groups['train16']=[r for r in item['rows'] if r['split']=='train' and r['camera'] in protocol['probe_cameras'] and r['frame'] in protocol['frames']]
   probe[name]={}
   for split,rr in groups.items():
    if not rr:continue
    keys=['lr.l1','teacher_rgb.l1','teacher_rgb.mse','teacher_H.l1','teacher_H.mse','hr.psnr','hr.ssim','hr.lpips_alex','hr_H.l1','teacher_hr_H.l1','teacher_hr.psnr','teacher_hr.ssim','teacher_hr.lpips_alex'];v={}
    flat=[dict(flatten(r)) for r in rr]
    for k in keys:
     vals=[r[k] for r in flat if k in r]
     if vals:v[k]=sum(vals)/len(vals)
    probe[name][split]=dict(count=len(rr),**v)
    row=dict(run_id=name,parent_run_id=rr[0]['parent_run_id'],renderer_id=rr[0]['renderer_id'],metric_version=rr[0]['metric_version'],camera=split if split.startswith('cam') else 'train_group',frame=40 if split!='train16' else '0_40_80_118_aggregate',split=split,observation_count=len(rr),**v,checkpoint_sha256=item['checkpoint_sha256'],source_path=str(file))
    if mode=='references':teacher.append(row)
    if split!='train16' and (mode!='references' or name=='U6000'):fixed.append(row)
   for r in item['rows']:record([r],{k:r[k] for k in ['run_id','parent_run_id','checkpoint_sha256','renderer_id','metric_version','split','source_path']})
  if mode!='references':
   d=OUT/'fixed_time'/mode;c=read(d/'complete.json');co=c['config'];assert c['source_unchanged'];assert read(d/'initial_parity.json')['passed'];cost.append(dict(run_id=mode,phase='P2',actual_updates=c['actual_updates'],train_s=c['train_s'],elapsed_s=c['seconds'],peak_gb=c['peak_gb'],gpu=co['gpu'],trainable_scalars=co['trainable_scalars']))
 fixed_decision={}
 if all(k in probe for k in ['Shared401200','Baked401200','U6000']):
  s,b=probe['Shared401200']['train19'],probe['Baked401200']['train19'];hgain=1-b['teacher_H.l1']/s['teacher_H.l1'];lrharm=b['lr.l1']/s['lr.l1']-1
  fixed_decision=dict(teacher_H_relative_gain=hgain,lr_relative_harm=lrharm,fit_signal=hgain>=.1 and lrharm<=.02,train_hr_psnr_gain=b['hr.psnr']-s['hr.psnr'],train_hr_lpips_relative=b['hr.lpips_alex']/s['hr.lpips_alex']-1,cam00_psnr_gain=probe['Baked401200']['cam00']['hr.psnr']-probe['Shared401200']['cam00']['hr.psnr'])
 depth=read(OUT/'depth_admission.json');parity=read(OUT/'renderer_parity.json')
 result=dict(status='partial' if a.partial else 'completed',phase_status=dict(P0='completed_numerical_envelope_not_certified' if not a.partial else 'in_progress',P1='completed_admission_failed',P2='completed' if fixed_decision else 'in_progress',P3='not_run'),replay=replay,comparisons=comparisons,sensitivity=sensitivity,probe=probe,fixed_time_decision=fixed_decision,P3_unrun_reason=depth['unrun_reason'],P3_nonzero_support_steps=depth['nonzero_depth_steps'],evidence_ids=['identity_v1','renderer_backward_v1','renderer_delegated_v1','replay_tails','replay_long','support_v1','remote_a100/moments_v1','depth_evidence_v1','fixed_time','probe_evaluation'])
 if not a.partial:
  roi=read(OUT/'roi_fit_v1/complete.json')
  for r in roi['rows']:
   ident={k:r[k] for k in ['run_id','parent_run_id','checkpoint_sha256','renderer_id','metric_version','camera','frame','split']}
   for k in ['lr','hr_H','teacher_rgb','teacher_H','teacher_hr_H']:
    for metric,val in flatten(r.get(k,{}),k):metricrows.append(dict(**ident,source_path=str(OUT/'roi_fit_v1/complete.json'),metric=r['region']+'.'+metric,value=val))
  assert len(replay)==19,len(replay)  # 2 read-only + 8 tails + 9 long endpoints.
  assert fixed_decision and sum(v['actual_updates'] for v in cost)==33750
  assert not depth['passed'];result['next_single_question']='当前剩余高频误差更受多视角目标不相容还是投影足迹与优化限制影响？'
  result['pass_fail']=dict(identity=True,renderer_numerical_envelope=False,depth_gates=depth['gates'],baked_fit_signal=fixed_decision['fit_signal'],P3=False)
  result['unrun_reason']=dict(P3=depth['unrun_reason'],extra_search='Outside the bounded plan; no positive fixed-time fitting signal')
  meta=['run_id','parent_run_id','checkpoint_sha256','renderer_id','metric_version','camera','frame','split','source_path']
  for r in runs+fixed+teacher:
   identity={k:r[k] for k in meta}
   for key,v in r.items():
    if key not in meta and isinstance(v,(float,int)) and key!='observation_count':metricrows.append(dict(**identity,metric='aggregate.'+key,value=v))
  csvwrite(OUT/'replay_runs.csv',runs);csvwrite(OUT/'teacher_fit.csv',teacher);csvwrite(OUT/'fixed_time_fit.csv',fixed);csvwrite(OUT/'per_frame_metrics.csv',metricrows);write(OUT/'cost.json',dict(preparation=dict(sift_seconds=read(OUT/'support_v1/complete.json')['seconds'],depth_gate_seconds=depth['seconds'],roi_and_footprint_seconds=read(OUT/'roi_fit_v1/complete.json')['seconds'],teacher_generation='reused_existing_no_new_inference'),controller_commands={phase:read(OUT/phase/'commands.json') for phase in ['replay_tails','replay_long','remaining']},major_updates=sum(v['actual_updates'] for v in cost),major_budget=protocol['max_major_updates'],engineering_updates=68,runs=cost,moment_audit=read(OUT/'moment_audit.json'),note='Training intervals exclude separate evaluation. P0 comparison endpoints do not duplicate update counts.'))
  write(OUT/'decision.json',result)
 else:write(OUT/'interim_summary.json',result)
 print(json.dumps(dict(status=result['status'],replay_endpoints=len(replay),probe_models=list(probe),fixed_time_decision=fixed_decision),ensure_ascii=False))
if __name__=='__main__':main()
