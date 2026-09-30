"""Static publication-inspection figures from raw frozen-model observations."""
from structure_common import *
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def main():
 out=OUT/'structure';models=list(registry());paths=[]
 for camera in ['cam00','cam01']:
  data=[np.load(out/name/f'{camera}_0040_moments.npz') for name in models];pool=np.concatenate([d['hr_z_normalized'][d['hr_alpha']>.5][::100] for d in data]);lo,hi=np.quantile(pool,[.02,.98]);fig,axes=plt.subplots(3,5,figsize=(16,8),layout='constrained')
  for j,(name,d) in enumerate(zip(models,data)):
   alpha=d['hr_alpha'];z=d['hr_z_normalized'];thick=np.sqrt(np.maximum(d['hr_variance_raw'],0))/np.maximum(z,1e-6);valid=alpha>.5
   for i,(v,cmap,vmin,vmax) in enumerate([(alpha,'gray',0,1),(np.where(valid,z,np.nan),'viridis',lo,hi),(np.where(valid,thick,np.nan),'magma',0,.6)]):
    axes[i,j].imshow(v,cmap=cmap,vmin=vmin,vmax=vmax);axes[i,j].set_xticks([]);axes[i,j].set_yticks([])
    if i==0:axes[i,j].set_title(name)
    if j==0:axes[i,j].set_ylabel(['Accumulated alpha','Normalized axial depth','Relative ray thickness'][i])
    if camera=='cam01':axes[i,j].add_patch(Rectangle((0,0),560,420,fill=False,edgecolor='red',lw=.9))
  fig.suptitle(f'{camera}, frame 40: same time / camera / color scales; alpha > 0.5 for depth display\nDepth range [{lo:.2f}, {hi:.2f}]; thickness range [0, 0.6]. Depth is a ray-weighted average, not surface truth.',fontsize=12)
  path=out/(camera+'_moments_frame40.png');fig.savefig(path,dpi=130);plt.close(fig);paths.append(dict(path=str(path),sha256=sha(path)))
 d=read(out/'sampling_summary.json')['aggregate'];fig,axes=plt.subplots(1,3,figsize=(13,4.2),layout='constrained');x=np.arange(len(models))
 for ax,k,label in zip(axes,['psnr','ssim','lpips'],['PSNR change (higher is better)','SSIM change (higher is better)','LPIPS change (lower is better)']):
  for offset,mode,title in [(-.18,'lr_bicubic','LR render + bicubic'),(.18,'two_hr_area','2HR render + area')]:
   vals=[d[name]['both'][mode][k]-d[name]['both']['direct_hr'][k] for name in models];ax.bar(x+offset,vals,width=.36,label=title)
  ax.set_xticks(x,models,rotation=25,ha='right');ax.axhline(0,color='k',lw=.7);ax.set_title(label);ax.grid(axis='y',alpha=.2)
 axes[0].legend(fontsize=8);fig.suptitle('Frozen models: output sampling sensitivity on the same 8 development observations')
 path=out/'sampling_delta.png';fig.savefig(path,dpi=150);plt.close(fig);paths.append(dict(path=str(path),sha256=sha(path)));write(out/'figures.json',dict(status='generated',figures=paths,source_sha256=sha(__file__)))
if __name__=='__main__':main()
