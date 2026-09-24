"""Post-run descriptive analyses, no refits or parameter selection."""
from pathlib import Path
import json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'output/dynamic_sr_20260921/candidate1_identifiability_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 d=json.loads((OUT/'metrics.json').read_text());records=[]
 for r in d['records']:
  i=r['index'];z=np.load(OUT/f'patch_{i:02d}_frozen_before_hr.npz');jd=z['Jd'];h=jd.T@jd/(1/255)**2
  e=z['H_0.00392157_0.15'];ev,u=np.linalg.eigh(h+np.eye(32));wh=(u/np.sqrt(ev)[None,:])@u.T
  loss=np.linalg.eigvalsh(wh@(h-e)@wh)
  s=r['schurs'][1];known=np.diag(h);effective=np.diag(e);low=np.array([max(x)<=12 for x in r['basis_labels']])
  records.append(dict(index=i,scene=r['cell']['scene'],maximum_relative_loss_regularized=float(loss.max()),median_diagonal_loss=float(np.median(1-effective/np.maximum(known,1e-30))),maximum_diagonal_loss=float(np.max(1-effective/np.maximum(known,1e-30))),teacher_direction_fraction_explained=1-s['teacher_direction_effective']/s['teacher_direction_known'],low_nominal_mean_curve=float(known[low].mean()),high_nominal_mean_curve=float(known[~low].mean()),teacher_raw_rmse=float(np.sqrt(r['baselines']['teacher_lr_mse'])),teacher_profile_rmse=float(np.sqrt(s['teacher_profile']['profile_residual_mse'])),teacher_profile_penalized_per_row=s['teacher_profile']['profile_penalized_per_row']))
 summary={}
 for scene in ['cook_spinach','meetroom_discussion','all']:
  rr=[r for r in records if scene=='all' or r['scene']==scene]
  summary[scene]={k:dict(mean=float(np.mean([r[k] for r in rr])),median=float(np.median([r[k] for r in rr])),min=float(min(r[k] for r in rr)),max=float(max(r[k] for r in rr))) for k in rr[0] if k not in ['index','scene']}
 receipt=dict(source_sha256=sha(__file__),main_source_matches_snapshot=sha(OUT/'source.py')==d['identity']['source_sha256'],dependencies={str(ROOT/'experiments/dynamic_sr_20260921/real_lr_local_inverse.py'):sha(ROOT/'experiments/dynamic_sr_20260921/real_lr_local_inverse.py')},score_json_sha256={p.name:sha(p) for p in sorted(OUT.glob('patch_*_scores_before_hr.json'))},frozen_npz_sha256={p.name:sha(p) for p in sorted(OUT.glob('patch_*_frozen_before_hr.npz'))},main_metrics_sha256=sha(OUT/'metrics.json'),summary=summary,records=records,scope='Post-run algebraic/descriptive audit; no rerun or parameter tuning. Dependency hash captured at audit time, not pre-HR receipt.')
 (OUT/'descriptive_audit.json').write_text(json.dumps(receipt,indent=2))
 fig,axes=plt.subplots(1,3,figsize=(13,3.8),constrained_layout=True)
 for scene,color,label in [('cook_spinach','#2E6F40','Cook spinach'),('meetroom_discussion','#8A4F9A','MeetRoom')]:
  s=d['summary'][scene];x=np.array([1,2,4]);known=[s['cases'][j]['means']['df_known'] for j in [1,4,7]];eff=[s['cases'][j]['means']['df_effective'] for j in [1,4,7]]
  axes[0].plot(x,known,'o--',color=color,label=label+' known');axes[0].plot(x,eff,'x-',color=color,label=label+' nuisance')
  rr=[r for r in d['records'] if r['cell']['scene']==scene]
  axes[1].scatter([r['schurs'][1]['trace_known'] for r in rr],[r['schurs'][1]['trace_effective'] for r in rr],color=color,label=label)
  axes[2].scatter([r['schurs'][1]['df_effective'] for r in rr],[r['hr_evaluation_only']['teacher_direction_error_mean'] for r in rr],color=color,label=label)
 axes[0].set(xlabel='Assumed LR noise std (1/255)',ylabel='Effective dimensions (out of 32)',title='Noise prior sensitivity');axes[0].legend(fontsize=7)
 axes[1].plot([7000,23000],[7000,23000],'k:',linewidth=1);axes[1].set(xlabel='Known-motion information trace',ylabel='After nuisance subtraction',title='Small change in aggregate support');axes[1].legend(fontsize=8)
 axes[2].set(xlabel='Effective dimensions',ylabel='Frozen-teacher coefficient error',title='Support is not teacher correctness');axes[2].legend(fontsize=8)
 fig.savefig(OUT/'candidate1_summary.png',dpi=180);plt.close(fig)
 print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
