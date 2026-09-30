#!/usr/bin/env python3
"""Fixed-frame scientific frequency figures; no image metrics use these PNGs."""
from spectral_common import *
from audit_spectrum_color import OBS, dataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def main():
    figdir=OUT/'figures'; figdir.mkdir(parents=True,exist_ok=True)
    _,m,dr=dataset(); observations={(o['camera_id'],o['frame_index']):o for o in m['observations']}
    rois=read(OUT/'roi_protocol.json')['regions_by_camera_xyxy_exclusive']; entries=[]
    for camera in ['cam00','cam01']:
        frame=40; p=OBS/'r1_Async2'/f'{camera}_{frame:04d}.npz'; o=observations[camera,frame]
        raw=np.load(p)['rgb_raw'].transpose(1,2,0); r=np.clip(raw,0,1); g=rgb(dr/o['hr_path'])
        budget,(cr,cg,b)=spectral_budget(r,g,raw); ce=cr-cg
        full=np.sqrt(np.square(r.astype(float)-g.astype(float)).mean(axis=2))
        residuals=[]
        for label,mask in [('DC',b['dc']),('Low (non-DC)',b['low']),('Mid + high',b['mid']|b['high'])]:
            c=np.zeros_like(ce);c[mask]=ce[mask];e=inverse(c)
            residuals.append((label,np.sqrt(np.square(e).mean(axis=2))))
        fig,axes=plt.subplots(2,3,figsize=(13.5,7.8)); fig.subplots_adjust(left=.015,right=.945,bottom=.055,top=.925,wspace=.015,hspace=.08)
        for ax,x,label in zip(axes.flat,[g,r,full]+[x[1] for x in residuals],['Genuine HR','Async2, repeat 1','Full RGB error (RMS)']+[x[0]+' residual RMS' for x in residuals]):
            if x.ndim==3: ax.imshow(x)
            else: im=ax.imshow(x,cmap='magma',vmin=0,vmax=.20)
            ax.set_title(label,fontsize=11); ax.axis('off')
        for ax in axes[0,:2]:
            for box in rois.get(camera,{}).values():
                x0,y0,x1,y1=box;ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,edgecolor='#22ffcc',linewidth=.65))
        ca=fig.add_axes([.953,.17,.015,.58]);fig.colorbar(im,cax=ca,label='Per-pixel RMS, fixed scale 0 to 0.20')
        fig.suptitle(f'cook_spinach / {camera} / frame 40 — fixed full-image DCT decomposition',fontsize=14)
        fig.text(.015,.018,'Spatial residual maps are visual diagnostics. Bands are globally orthogonal; squared maps do not add pointwise.',fontsize=9)
        dest=figdir/f'{camera}_0040_frequency_residuals.png';fig.savefig(dest,dpi=160,bbox_inches='tight',pad_inches=.12);plt.close(fig)
        entries.append(dict(path=str(dest.relative_to(ROOT)),sha256=sha(dest),camera=camera,frame=40,
                            float_asset_sha256=sha(p),hr_sha256=sha(dr/o['hr_path']),budget=budget))
    write(figdir/'index.json',dict(status='generated',entries=entries,visual_inspection='pending',
          definition='Fixed frame40 and prior ROIs; full-image DCT residual bands reconstructed without local windows; common display RMS range [0,.20]. Display clipping only.',
          conclusion_boundary='No pointwise or ROI-by-band additivity claim. Numbers from float assets, never figure PNG.'))

if __name__=='__main__':main()
