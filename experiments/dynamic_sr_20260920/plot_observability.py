#!/usr/bin/env python3
"""Plots saved numerical observations; no fitting or result selection."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'output/dynamic_sr_20260920/observability_supplement_v1'
main=json.loads((OUT.parent/'observability_synthetic_v1/metrics.json').read_text())
supp=json.loads((OUT/'metrics.json').read_text())
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(2,2,figsize=(10.8,7.2),constrained_layout=True)
r=supp['main_paired_bootstrap'];means=np.array([x['mean'] for x in r]);cis=np.array([x['paired_bootstrap95'] for x in r])
axes[0,0].bar(np.arange(3),means,color=['#4477AA','#228833','#999999'])
axes[0,0].errorbar(np.arange(3),means,yerr=np.maximum(0,np.stack([means-cis[:,0],cis[:,1]-means])),fmt='none',color='black',capsize=4)
axes[0,0].set_xticks(range(3),['Project bicubic AA','Pixel box','Ideal low-pass'])
axes[0,0].set_ylabel('Complementary minus repeated PSNR (dB)')
axes[0,0].set_title('A  Benefit depends on the imaging kernel')
rs=[x for x in main['summaries'] if x['kernel']=='project_bicubic_aa' and x['schedule']=='complementary' and x['missing']=='none']
axes[0,1].plot([x['motion_error_lr'] for x in rs],[x['hr_psnr'] for x in rs],'-o',color='#CC6677')
baseline=next(x['hr_psnr'] for x in main['summaries'] if x['kernel']=='project_bicubic_aa' and x['schedule']=='repeated' and x['missing']=='none' and x['motion_error_lr']==0)
axes[0,1].axhline(baseline,linestyle='--',color='#777777',label='Repeated, correct correspondence')
axes[0,1].set_xlabel('Fixed correspondence error std. (LR pixels)')
axes[0,1].set_ylabel('HR reconstruction PSNR (dB)');axes[0,1].legend(fontsize=8)
axes[0,1].set_title('B  Incorrect alignment can erase the benefit')
rs=supp['noise_sweep'];axes[1,0].plot([x['noise_factor'] for x in rs],[x['paired_gain']['mean'] for x in rs],'-o',color='#4477AA')
axes[1,0].set_xlabel('Independent LR noise std. (multiples of 1/255)')
axes[1,0].set_ylabel('Complementary phase gain (dB)')
axes[1,0].set_title('C  Numerical rank is insufficient at low SNR')
for method,color,label in [('motion_assumed_known','#CC6677','Assume motion known'),('nuisance_aware','#228833','Account for motion uncertainty')]:
    xs=[.025,.1,.25];ys=[];lo=[];hi=[]
    for x in xs:
        vals=[r['ratio'] for r in supp['local_multiple_points'] if r['motion_uncertainty_lr']==x and r['estimator']==method]
        ys.append(np.mean(vals));lo.append(min(vals));hi.append(max(vals))
    axes[1,1].plot(xs,ys,'-o',color=color,label=label)
    axes[1,1].fill_between(xs,lo,hi,color=color,alpha=.13)
axes[1,1].axhline(1,color='#777777',linestyle='--')
axes[1,1].set_xlabel('Motion uncertainty std. (LR pixels)')
axes[1,1].set_ylabel('Empirical / predicted coefficient MSE')
axes[1,1].set_title('D  Local uncertainty model fails for larger shifts')
axes[1,1].legend(fontsize=8)
fig.suptitle('Controlled dynamic detail observability (not 4DGS benchmark scores)',fontsize=13)
fig.savefig(OUT/'observability_summary.png',dpi=180)
fig.savefig(OUT/'observability_summary.pdf')
print(OUT/'observability_summary.png')
