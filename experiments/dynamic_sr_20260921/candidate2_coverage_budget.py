#!/usr/bin/env python3
"""Exploratory B coverage and LR-chosen, exactly matched total-step budget control."""
import argparse, json, math, random, shutil, sys, time
from pathlib import Path
import numpy as np
import torch
from scipy.stats import spearmanr
from candidate2_update_validation_v2 import (ROOT, SCENES, SCALES, Data, key, parse, load_manifest, load_checkpoint,
 named_parameters, hash_tensors, opt_hash, set_state, measure_lr, metric_hr, sha256, write_json)

def run(args):
 out=Path(args.out);out.mkdir(parents=True,exist_ok=False);shutil.copy2(__file__,out/'source.py');start=time.monotonic();torch.set_num_threads(4)
 parent=Path(args.parent);cfg=json.loads((parent/'config.json').read_text());prior_protocol=json.loads((parent/'protocol.json').read_text());prior_lr=json.loads((parent/'lr_scores_locked.json').read_text())
 m=load_manifest(ROOT/SCENES[cfg['scene']][0]);cams=sorted({o['camera_id'] for o in m['observations'] if o['split']=='train'});train=sorted(key(o['camera_id'],o['frame_index']) for o in m['observations'] if o['split']=='train')
 centers={c:np.asarray(m['cameras'][c]['c2w'])[:3,3] for c in cams};proposals=[]
 for i,p in enumerate(prior_protocol['proposals']):
  c,f=parse(p['A']);chosen=[]
  for j in range(4):
   pool=[x for x in cams if x!=c and x not in chosen]
   chosen.append(max(pool,key=lambda x:(min(np.linalg.norm(centers[x]-centers[y]) for y in [c]+chosen),x)))
  spread=[key(x,t) for x in chosen for t in [f,(f+40)%120]]
  rand=random.Random(932091+i).sample([x for x in train if x!=p['A'] and x not in spread],8)
  assert len(set(spread))==8 and all(x in train and x!=p['A'] for x in spread+rand)
  proposals.append(dict(**p,extra_B=dict(spread8=spread,random8=rand)))
 protocol=dict(version='candidate2_exploratory_coverage_budget_v1',scene=cfg['scene'],parent_lr_sha256=sha256(parent/'lr_scores_locked.json'),
  design='Post-primary-results exploratory check, not confirmatory holdout. Fixed calibrated-camera-center maximin selects four training cameras excluding A, beginning farthest from A; each at A time and +40 modulo 120. Compare 8 fixed random other observations. No HR-dependent selection.',
  budget='Primary scale .005. For each cross2/spread8 LR gate, uniform no-gate alpha=.005*sum(accepted direction L2)/sum(all direction L2). Match total parameter L2 path length across 12 independent-reset proposals. This is coordinate dependent, not equality of physical/rendered change. Reevaluate original .005 and all budget controls on the same current GPU.',
  proposals=proposals,scales=SCALES,threshold=1e-8,primary_scale=.005)
 write_json(out/'protocol.json',protocol)
 g,_,_,_=load_checkpoint(cfg['checkpoint']);params=named_parameters(g);base={k:p.detach().clone() for k,p in params.items()};phash=hash_tensors(base);ohash=opt_hash(g.optimizer);data=Data(m);cache={};rows=[];parity=[]
 for p in proposals:
  path=parent/'directions'/f"{p['id']}.pt";assert sha256(path)==prior_lr['direction_hashes'][p['id']]
  direction={k:v.cuda() for k,v in torch.load(path,weights_only=True).items()}
  needed=set(x for ks in p['extra_B'].values() for x in ks)|set(p['B']['cross'])
  set_state(params,base)
  for k in needed:
   if k not in cache:cache[k]=measure_lr(g,data,k)
  for scale in SCALES:
   set_state(params,base,direction,scale);after={k:measure_lr(g,data,k) for k in needed}
   scores={gate:float(np.mean([cache[k]['mse']-after[k]['mse'] for k in ks])) for gate,ks in p['extra_B'].items()}
   original=next(r for r in prior_lr['rows'] if r['proposal_id']==p['id'] and r['scale']==scale)
   old_replay=float(np.mean([cache[k]['mse']-after[k]['mse'] for k in p['B']['cross']]));parity.append(abs(old_replay-original['scores']['cross']))
   rows.append(dict(proposal_id=p['id'],scale=scale,scores=scores,accepted={k:v>1e-8 for k,v in scores.items()},direction_l2=original['direction_l2'],original_cross2_accept=original['gate_state']['cross']=='improve'))
  set_state(params,base);assert hash_tensors(params)==phash
 primary=[r for r in rows if r['scale']==.005];normsum=sum(r['direction_l2'] for r in primary)
 gates={'cross2':{r['proposal_id']:r['original_cross2_accept'] for r in primary},'spread8':{r['proposal_id']:r['accepted']['spread8'] for r in primary}}
 alphas={gate:.005*sum(r['direction_l2'] for r in primary if selection[r['proposal_id']])/normsum for gate,selection in gates.items()}
 lock=dict(rows=rows,gates=gates,budget_alphas=alphas,inputs=data.inputs,source_sha256=sha256(__file__),gpu=torch.cuda.get_device_name(),parent_cross_score_replay_max_abs_error=max(parity),parameter_hash=phash,optimizer_hash=ohash)
 write_json(out/'lr_locked.json',lock);lockhash=sha256(out/'lr_locked.json')
 print(json.dumps(dict(event='lr_locked',elapsed_s=time.monotonic()-start,budget_alphas=alphas)),flush=True)
 # New LR choices and budget alphas are fixed before reading/re-evaluating HR.
 prior_metrics=json.loads((parent/'metrics.json').read_text());summary={}
 for scale in SCALES:
  sr=[r for r in rows if r['scale']==scale];section={}
  for gate in ['spread8','random8']:
   gs=[r['scores'][gate] for r in sr];deltas=[next(q['eval_delta']['novel'] for q in prior_metrics['rows'] if q['proposal_id']==r['proposal_id'] and q['scale']==scale) for r in sr];accept=[r['accepted'][gate] for r in sr];n=sum(accept)
   section[gate]=dict(accepted=n,neutral=sum(abs(x)<=1e-8 for x in gs),harmed_accepted=sum(v['psnr']<-.001 for v,a in zip(deltas,accept) if a),improved_accepted=sum(v['psnr']>.001 for v,a in zip(deltas,accept) if a),
    spearman_psnr=float(spearmanr(gs,[d['psnr'] for d in deltas]).statistic),gated_mean_delta={k:float(np.mean([v[k] if a else 0 for v,a in zip(deltas,accept)])) for k in deltas[0]})
  summary[str(scale)]=section
 import lpips
 metric=lpips.LPIPS(net='alex').cuda().eval().requires_grad_(False);before={};hrrows=[]
 for p in proposals:
  for k in p['evaluation_train']+p['evaluation_novel']:
   if k not in before:before[k]=metric_hr(g,data,k,metric)
 for p in proposals:
  direction={k:v.cuda() for k,v in torch.load(parent/'directions'/f"{p['id']}.pt",weights_only=True).items()}
  variants={'primary':.005,**{f'uniform_{gate}':alpha for gate,alpha in alphas.items()}}
  record=dict(proposal_id=p['id'],variants={})
  for name,alpha in variants.items():
   set_state(params,base,direction,alpha);scores={k:metric_hr(g,data,k,metric) for k in p['evaluation_train']+p['evaluation_novel']};d={}
   for group,ks in [('train_cross',p['evaluation_train']),('novel',p['evaluation_novel'])]:d[group]={metric:float(np.mean([scores[k][metric]-before[k][metric] for k in ks])) for metric in ['psnr','ssim','lpips','hf_mse']}
   record['variants'][name]=dict(alpha=alpha,delta=d)
  hrrows.append(record);set_state(params,base);assert hash_tensors(params)==phash
 budget={}
 for gate,selection in gates.items():
  budget[gate]=dict(alpha=alphas[gate],accepted=sum(selection.values()),gate_total_l2=.005*sum(r['direction_l2'] for r in primary if selection[r['proposal_id']]),uniform_total_l2=alphas[gate]*normsum,
    results={group:{name:{k:float(np.mean([(r['variants']['primary']['delta'][group][k] if selection[r['proposal_id']] else 0) if name=='gate' else r['variants'][f'uniform_{gate}']['delta'][group][k] for r in hrrows])) for k in ['psnr','ssim','lpips','hf_mse']} for name in ['gate','uniform']} for group in ['novel','train_cross']})
 assert sha256(out/'lr_locked.json')==lockhash;assert opt_hash(g.optimizer)==ohash
 result=dict(protocol=protocol,rows=rows,summary=summary,budget=budget,hr_rows=hrrows,lock_sha256=lockhash,inputs=data.inputs,elapsed_s=time.monotonic()-start,peak_gb=torch.cuda.max_memory_allocated()/1e9,gpu=torch.cuda.get_device_name(),parent_cross_score_replay_max_abs_error=max(parity),rollback_exact=True,optimizer_unchanged=True)
 write_json(out/'metrics.json',result);write_json(out/'complete.json',dict(status='complete',elapsed_s=result['elapsed_s'],metrics_sha256=sha256(out/'metrics.json')));print(json.dumps(dict(event='complete',elapsed_s=result['elapsed_s'])),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--parent',required=True);p.add_argument('--out',required=True);run(p.parse_args())
