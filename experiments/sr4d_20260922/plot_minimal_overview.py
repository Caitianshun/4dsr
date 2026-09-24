import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np,json
from PIL import Image
from pathlib import Path
r=Path('/home/cai_tianshun/Project/4dsr');o=r/'output/sr4d_20260922'
fig,ax=plt.subplots(2,5,figsize=(18,5.5),layout='constrained')
for i,(run,mp) in enumerate([('cook_existinglr_v1','data/dynamic_sr/n3dv_prepared/cook_spinach'),('meeting_existinglr_v1','data/dynamic_sr/meetroom_prepared/discussion')]):
 data=json.loads((r/mp/'manifest.json').read_text());row=next(x for x in data['observations'] if x['camera_id']=='cam00' and x['frame_index']==0)
 ims=[np.asarray(Image.open(r/mp/row['hr_path']))]
 labs=['HR reference','SR4D 6k','SR4D 18k','Wu + SR0.1 6k','Wu + SR0.1 18k']
 for met,it in [('sr4d',6000),('sr4d',18000),('wu',6000),('wu',18000)]:
  ims.append(np.clip(np.load(o/run/f'{met}_{it}_evaluation/predictions/cam00_0000.npy').transpose(1,2,0),0,1))
 for j,im in enumerate(ims):
  ax[i,j].imshow(im,interpolation='nearest');ax[i,j].set_title(labs[j],fontsize=11);ax[i,j].axis('off')
 ax[i,0].text(-.03,.5,'Cook spinach' if i==0 else 'MeetRoom',rotation=90,va='center',ha='right',transform=ax[i,0].transAxes,fontsize=12)
fig.suptitle('Fixed novel view: cam00, frame 0 | same input data; stage counts are not total budget',fontsize=13)
fig.savefig(o/'final_assessment/novel_overview.png',dpi=150);plt.close(fig)
