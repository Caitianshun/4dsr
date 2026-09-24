#!/usr/bin/env python3
"""Frozen, resettable one-step SR proposals; LR scoring precedes all HR access."""
from __future__ import annotations
import argparse, hashlib, json, math, os, random, shutil, socket, sys, time, traceback
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
OLD=ROOT/'experiments/dynamic_sr_20260918'
sys.path.insert(0,str(OLD))
from common import (load_checkpoint, render_image, resized_camera, downsample,
                    image_tensor, named_parameters, sha256, write_json, seed_all)
from n3dv_data import load_manifest, N3DVPreparedDataset, observation_to_4dgs_camera
from evaluate import ssim_map_rgb

SCENES={
 'cook_spinach':('data/dynamic_sr/n3dv_prepared/cook_spinach/manifest.json',['cam02','cam06','cam12','cam18']),
 'meetroom_discussion':('data/dynamic_sr/meetroom_prepared/discussion/manifest.json',['cam02','cam04','cam08','cam12'])}
SCALES=[.025,.1,.4]
FRAMES=[24,56,88]

def key(c,f):return f'{c}/{f:04d}'
def parse(k):c,f=k.split('/');return c,int(f)
def hash_tensors(values):
 h=hashlib.sha256()
 for k,v in sorted(values.items()):
  h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def opt_hash(opt):
 h=hashlib.sha256()
 for i,g in enumerate(opt.param_groups):
  h.update(json.dumps({k:v for k,v in g.items() if k!='params'},sort_keys=True).encode())
  for j,p in enumerate(g['params']):
   for k,v in sorted(opt.state.get(p,{}).items()):
    h.update(f'{i}/{j}/{k}'.encode())
    h.update(v.detach().cpu().contiguous().numpy().tobytes() if torch.is_tensor(v) else str(v).encode())
 return h.hexdigest()

def make_protocol(m,prior):
 train=sorted(key(x['camera_id'],x['frame_index']) for x in m['observations'] if x['split']=='train')
 cameras=sorted({parse(k)[0] for k in train})
 proposals=[]
 for c in prior:
  for f in FRAMES:
   others=sorted((x for x in cameras if x!=c),key=lambda x:(abs(int(x[3:])-int(c[3:])),x))
   cross=[key(x,f) for x in others[:2]]
   far=[key(c,(f+32)%120),key(c,(f+64)%120)]
   near=[key(c,f-2),key(c,f+2)]
   excluded=set([key(c,f)]+near+far+cross)
   rand=random.Random(20260921+len(proposals)).sample([x for x in train if x not in excluded],2)
   a=key(c,f)
   p=dict(id=f'{c}_{f:04d}',A=a,B=dict(near=near,far=far,cross=cross,random=rand),
          evaluation_train=cross,evaluation_novel=[key('cam00',f)])
   for ks in p['B'].values():
    assert len(ks)==2 and all(x in train and x!=a and parse(x)[0] not in ['cam00','cam01'] for x in ks)
   proposals.append(p)
 return dict(version='candidate2_v1',scales=SCALES,proposals=proposals,
  update='p <- p - scale * checkpoint_group_lr * current_A_SR_gradient / group_RMS(current_A_SR_gradient); gradient uses SR weight 0.1 which cancels in RMS normalization. This is a diagnostic direction, not the historic Adam step.',
  thresholds=dict(lr_absolute_mse=1e-8,hr_psnr_db=.001,hr_ssim=1e-5,hr_lpips=1e-5),
  scoring='B MSE decrease averaged over exactly two actual LR images; accept only decrease > max(1e-8,10*repeat-render MSE noise). A LR derivative and actual LR decrease are comparison baselines.',
  design='Fixed pre-result camera/time IDs, all scales reported. Primary scale 0.1. Secondary scales test local finite-step behavior, not independent samples. No HR used in proposal, observation selection, step normalization, gate thresholds, or rankings.',
  scope='B was historically trained but excluded from this proposal; cam00 is previously examined development novel view. No final holdout claim. Far time is not asserted to have complementary subpixel phase. No density/topology/optimizer updates.')

class Data:
 def __init__(self,m):
  self.m=m;self.root=Path(m['_root']);self.lookup={};self.cache={};self.inputs={}
  for split in ['train','dev','test']:
   ds=N3DVPreparedDataset(m,split,'lr')
   for i,o in enumerate(ds.observations):self.lookup[key(o['camera_id'],o['frame_index'])]=(ds,i,o)
 def record(self,path,role):
  path=Path(path)
  if str(path) not in self.inputs:self.inputs[str(path)]=dict(role=role,sha256=sha256(path),bytes=path.stat().st_size)
 def get(self,k):
  if k not in self.cache:
   ds,i,o=self.lookup[k];self.record(self.root/o['lr_path'],'actual_lr');r=ds[i];w,h=self.m['resolutions']['hr']
   self.cache[k]=(r['image'],resized_camera(observation_to_4dgs_camera(r,i),h,w),o)
  return self.cache[k]
 def lr(self,k):return self.get(k)[0].cuda()
 def cam(self,k):return self.get(k)[1]
 def teacher(self,k):
  _,_,o=self.get(k)
  path=self.root/'sr_swinir_x4'/o['camera_id']/Path(o['lr_path']).name
  self.record(path,'frozen_sr_teacher');return image_tensor(path)
 def hr(self,k):
  path=self.root/self.get(k)[2]['hr_path'];self.record(path,'evaluation_only_hr');return image_tensor(path)

@torch.no_grad()
def measure_lr(g,data,k):
 raw=render_image(g,data.cam(k))['render'];t=data.lr(k);v=downsample(raw,t.shape[-2:])
 return dict(mse=float((v-t).square().mean()),l1=float((v-t).abs().mean()),
             clamp_fraction=float(((raw<0)|(raw>1)).float().mean()))
@torch.no_grad()
def set_state(params,base,direction=None,scale=0.):
 for k,p in params.items():p.copy_(base[k] if direction is None else base[k]+scale*direction[k])
@torch.no_grad()
def metric_hr(g,data,k,lpips):
 x=render_image(g,data.cam(k))['render'].clamp(0,1);y=data.hr(k)
 mse=float((x-y).square().mean());size=data.get(k)[0].shape[-2:]
 hx=x-torch.nn.functional.interpolate(downsample(x,size)[None],size=x.shape[-2:],mode='bicubic',align_corners=False)[0]
 hy=y-torch.nn.functional.interpolate(downsample(y,size)[None],size=y.shape[-2:],mode='bicubic',align_corners=False)[0]
 hf_mse=float((hx-hy).square().mean());xx=x.permute(1,2,0).cpu().numpy();yy=y.permute(1,2,0).cpu().numpy()
 return dict(mse=mse,hf_mse=hf_mse,psnr=-10*math.log10(max(mse,1e-12)),
   ssim=float(ssim_map_rgb(xx,yy)[5:-5,5:-5].mean()),
   lpips=float(lpips(x[None]*2-1,y[None]*2-1)))

def corr(a,b):
 from scipy.stats import spearmanr
 if len(a)<3 or np.ptp(a)==0 or np.ptp(b)==0:return None
 return float(spearmanr(a,b).statistic)
def summarize(rows,protocol):
 out={}
 for scale in SCALES:
  rs=[r for r in rows if r['scale']==scale];res={}
  if not rs:out[str(scale)]=dict(n=0);continue
  for target in ['novel','train_cross']:
   ms={}
   for gate in ['near','far','cross','random','A_actual','A_derivative']:
    scores=np.array([r['scores'][gate] for r in rs]);d=np.array([r['eval_delta'][target]['psnr'] for r in rs])
    tol=np.array([r['score_threshold'] if gate!='A_derivative' else r['derivative_threshold'] for r in rs])
    accept=scores>tol
    ms[gate]=dict(n=len(rs),accepted=int(accept.sum()),neutral=int((np.abs(scores)<=tol).sum()),rejected=int((scores < -tol).sum()),
      spearman_psnr=corr(scores,d),psnr_improvement_precision=float((d[accept]>.001).mean()) if accept.any() else None,
      psnr_harm_rate=float((d[accept]<-.001).mean()) if accept.any() else None,
      accepted_mean_delta={k:float(np.mean([r['eval_delta'][target][k] for r,a in zip(rs,accept) if a])) if accept.any() else None for k in ['psnr','ssim','lpips','hf_mse']},
      gated_mean_delta_all_proposals={k:float(np.mean([r['eval_delta'][target][k] if a else 0. for r,a in zip(rs,accept)])) for k in ['psnr','ssim','lpips','hf_mse']})
   res[target]=dict(unfiltered_mean_delta={k:float(np.mean([r['eval_delta'][target][k] for r in rs])) for k in ['psnr','ssim','lpips','hf_mse']},gates=ms,
      improved=sum(r['eval_delta'][target]['psnr']>.001 for r in rs),neutral=sum(abs(r['eval_delta'][target]['psnr'])<=.001 for r in rs),harmed=sum(r['eval_delta'][target]['psnr']<-.001 for r in rs))
  res['teacher_improved']=sum(r['A_teacher_after']<r['A_teacher_before'] for r in rs)
  res['teacher_relative_change_mean']=float(np.mean([(r['A_teacher_after']/r['A_teacher_before']-1) for r in rs]))
  out[str(scale)]=res
 return out

def run(args):
 out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
 shutil.copy2(__file__,out/'source.py')
 started=time.monotonic();torch.set_num_threads(4);seed_all(20260921)
 manifest=ROOT/SCENES[args.scene][0];m=load_manifest(manifest);protocol=make_protocol(m,SCENES[args.scene][1])
 if args.limit:protocol['proposals']=protocol['proposals'][:args.limit]
 write_json(out/'protocol.json',protocol)
 g,h,o,ck=load_checkpoint(args.checkpoint);params=named_parameters(g);base={k:p.detach().clone() for k,p in params.items()}
 parameter_hash=hash_tensors(base);optimizer_hash=opt_hash(g.optimizer);data=Data(m)
 names={id(p):k for k,p in params.items()};groups=[]
 for gr in g.optimizer.param_groups:
  ns=[names[id(p)] for p in gr['params']];groups.append(dict(name=gr['name'],lr=gr['lr'],names=ns))
 assert set(k for gr in groups for k in gr['names'])==set(params)
 config=dict(scene=args.scene,checkpoint=str(args.checkpoint),checkpoint_sha256=sha256(args.checkpoint),manifest_sha256=sha256(manifest),
 source_sha256=sha256(__file__),host=socket.gethostname(),gpu=torch.cuda.get_device_name(),torch=torch.__version__,parent_metadata=ck['metadata'],groups=groups,initial_parameter_hash=parameter_hash,initial_optimizer_hash=optimizer_hash)
 write_json(out/'config.json',config)
 (out/'directions').mkdir();lrbase={};rows=[];audits=[]
 # No method below this phase reads an HR path. Teacher is frozen preexisting SR.
 for i,p in enumerate(protocol['proposals']):
  set_state(params,base);assert hash_tensors(params)==parameter_hash
  a=p['A'];teacher=data.teacher(a);g.optimizer.zero_grad(set_to_none=True)
  raw=render_image(g,data.cam(a))['render'];loss=.1*(raw-teacher).abs().mean();loss.backward()
  grads={k:(v.grad.detach().clone() if v.grad is not None else torch.zeros_like(v)) for k,v in params.items()}
  direction={};groupstats=[]
  for gr in groups:
   n=sum(grads[k].numel() for k in gr['names']);rms=math.sqrt(sum(float(grads[k].double().square().sum()) for k in gr['names'])/n)
   for k in gr['names']:direction[k]=-gr['lr']*grads[k]/max(rms,1e-12)
   groupstats.append(dict(name=gr['name'],gradient_rms=rms,direction_rms=math.sqrt(sum(float(direction[k].double().square().sum()) for k in gr['names'])/n),lr=gr['lr']))
  teacher_before=float(loss.detach())/.1
  sr_derivative=sum(float((grads[k].double()*direction[k]).sum()) for k in params)/.1
  del grads,raw,loss
  g.optimizer.zero_grad(set_to_none=True)
  raw=render_image(g,data.cam(a))['render'];t=data.lr(a);lr_mse=(downsample(raw,t.shape[-2:])-t).square().mean();lr_mse.backward()
  lr_deriv=sum(float((v.grad.detach().double()*direction[k]).sum()) for k,v in params.items() if v.grad is not None)
  gradnorm=math.sqrt(sum(float(v.grad.detach().double().square().sum()) for v in params.values() if v.grad is not None))
  dirnorm=math.sqrt(sum(float(v.double().square().sum()) for v in direction.values()))
  alignment=-lr_deriv/max(gradnorm*dirnorm,1e-30)
  g.optimizer.zero_grad(set_to_none=True);del raw,lr_mse
  needed=sorted(set([a]+[x for vv in p['B'].values() for x in vv]))
  for k in needed:
   if k not in lrbase:lrbase[k]=measure_lr(g,data,k)
  repeated=measure_lr(g,data,a);noise=abs(repeated['mse']-lrbase[a]['mse']);tol=max(1e-8,10*noise)
  torch.save({k:v.cpu() for k,v in direction.items()},out/'directions'/f"{p['id']}.pt")
  finite_difference=[]
  for epsilon in [.001,.005]:
   observations=[]
   for sign in [-1,1]:
    set_state(params,base,direction,sign*epsilon)
    with torch.no_grad(): ts=float((render_image(g,data.cam(a))['render']-teacher).abs().mean())
    observations.append((ts,measure_lr(g,data,a)['mse']))
   fd_sr=(observations[1][0]-observations[0][0])/(2*epsilon);fd_lr=(observations[1][1]-observations[0][1])/(2*epsilon)
   finite_difference.append(dict(epsilon=epsilon,sr_analytic=sr_derivative,sr_fd=fd_sr,sr_relative_error=abs(fd_sr-sr_derivative)/max(abs(sr_derivative),1e-12),lr_analytic=lr_deriv,lr_fd=fd_lr,lr_relative_error=abs(fd_lr-lr_deriv)/max(abs(lr_deriv),1e-12)))
  set_state(params,base)
  for scale in SCALES:
   set_state(params,base,direction,scale)
   after={k:measure_lr(g,data,k) for k in needed}
   with torch.no_grad():teacher_after=float((render_image(g,data.cam(a))['render']-teacher).abs().mean())
   scores={b:float(np.mean([lrbase[k]['mse']-after[k]['mse'] for k in ks])) for b,ks in p['B'].items()}
   scores['A_actual']=lrbase[a]['mse']-after[a]['mse'];scores['A_derivative']=-scale*lr_deriv
   row=dict(proposal_id=p['id'],A=a,B=p['B'],scale=scale,scores=scores,score_threshold=tol,derivative_threshold=tol,
      gate_state={b:('improve' if v>tol else 'worsen' if v < -tol else 'unresolved') for b,v in scores.items()},
      A_teacher_before=teacher_before,A_teacher_after=teacher_after,A_teacher_predicted_delta=scale*sr_derivative,
      A_teacher_actual_delta=teacher_after-teacher_before,A_LR_derivative=lr_deriv,descent_alignment=alignment,
      direction_l2=dirnorm,actual_step_l2=scale*dirnorm,parameter_rms_fraction=scale*dirnorm/math.sqrt(sum(float(v.double().square().sum()) for v in base.values())),
      lr_before={k:lrbase[k] for k in needed},lr_after=after,repeat_render_mse_noise=noise,groupstats=groupstats,finite_difference=finite_difference)
   rows.append(row)
  set_state(params,base);assert hash_tensors(params)==parameter_hash
  audit=dict(proposal_id=p['id'],rollback_exact=True,sr_direction_descent=sr_derivative<0,repeat_render_mse_noise=noise)
  assert sr_derivative<0 and all(math.isfinite(v) for r in rows[-3:] for v in r['scores'].values())
  audits.append(audit);write_json(out/'lr_scores_partial.json',rows)
  print(json.dumps(dict(event='lr_proposal_done',id=p['id'],elapsed_s=time.monotonic()-started)),flush=True)
  del direction,teacher
 write_json(out/'lr_scores_locked.json',dict(rows=rows,audits=audits,protocol_sha256=sha256(out/'protocol.json'),optimizer_unchanged=opt_hash(g.optimizer)==optimizer_hash,direction_hashes={p['id']:sha256(out/'directions'/f"{p['id']}.pt") for p in protocol['proposals']},inputs=data.inputs))
 locked_hash=sha256(out/'lr_scores_locked.json');locked=json.loads((out/'lr_scores_locked.json').read_text())
 # HR reading begins ONLY after the complete LR decision file has been locked.
 import lpips
 metric=lpips.LPIPS(net='alex').cuda().eval().requires_grad_(False)
 before={}
 for p in protocol['proposals']:
  for k in p['evaluation_train']+p['evaluation_novel']:
   if k not in before:before[k]=metric_hr(g,data,k,metric)
 for p in protocol['proposals']:
  assert sha256(out/'directions'/f"{p['id']}.pt")==locked['direction_hashes'][p['id']]
  direction={k:v.cuda() for k,v in torch.load(out/'directions'/f"{p['id']}.pt",weights_only=True).items()}
  for r in [r for r in rows if r['proposal_id']==p['id']]:
   set_state(params,base,direction,r['scale'])
   scores={k:metric_hr(g,data,k,metric) for k in p['evaluation_train']+p['evaluation_novel']}
   r['hr_before']={k:before[k] for k in scores};r['hr_after']=scores;r['eval_delta']={}
   for name,ks in [('train_cross',p['evaluation_train']),('novel',p['evaluation_novel'])]:
    r['eval_delta'][name]={metric:float(np.mean([scores[k][metric]-before[k][metric] for k in ks])) for metric in ['mse','psnr','ssim','lpips','hf_mse']}
  set_state(params,base);assert hash_tensors(params)==parameter_hash
  write_json(out/'evaluated_partial.json',rows)
  print(json.dumps(dict(event='hr_proposal_done',id=p['id'],elapsed_s=time.monotonic()-started)),flush=True)
 assert sha256(out/'lr_scores_locked.json')==locked_hash
 assert opt_hash(g.optimizer)==optimizer_hash
 result=dict(config=config,protocol=protocol,rows=rows,summary=summarize(rows,protocol),audits=audits,
  summary_teacher_descent=summarize([r for r in rows if r['A_teacher_before']-r['A_teacher_after']>1e-7],protocol),completed=True,elapsed_s=time.monotonic()-started,peak_memory_gb=torch.cuda.max_memory_allocated()/1e9,
  final_parameter_hash=hash_tensors(params),optimizer_unchanged=True,locked_lr_sha256=locked_hash)
 write_json(out/'inputs.json',data.inputs)
 write_json(out/'metrics.json',result)
 write_json(out/'complete.json',dict(status='complete',elapsed_s=result['elapsed_s'],metrics_sha256=sha256(out/'metrics.json'),source_sha256=sha256(__file__)))
 print(json.dumps(dict(event='complete',elapsed_s=result['elapsed_s'])),flush=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('--scene',choices=list(SCENES),required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--out',required=True);p.add_argument('--limit',type=int,default=0)
 a=p.parse_args()
 try:run(a)
 except BaseException:
  out=Path(a.out)
  if out.exists():write_json(out/'failure.json',dict(traceback=traceback.format_exc(),status='failed'))
  raise
if __name__=='__main__':main()
