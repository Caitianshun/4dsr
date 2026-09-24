"""CPU-only fixed-endpoint summaries and figures from saved evidence; no training."""
from pathlib import Path
import hashlib,json
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'output/sr4d_20260922';OUT=BASE/'final_assessment';OUT.mkdir(exist_ok=True)
SCENES=[('cook_spinach','cook_existinglr_v1','data/dynamic_sr/n3dv_prepared/cook_spinach','Cook spinach'),('meetroom_discussion','meeting_existinglr_v1','data/dynamic_sr/meetroom_prepared/discussion','MeetRoom')]
KEYS=['psnr','ssim','lpips_alex','scale_high_residual_mse']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
summary={'scope':'Two fixed endpoints per system; CPU-only aggregation of saved metrics/images; no new rendering/training','sources':{},'scenes':{}}
novel=['| 场景 | 系统 | 阶段步数 | PSNR ↑ | SSIM ↑ | LPIPS ↓ |','| --- | --- | ---: | ---: | ---: | ---: |']
deltas=['| 场景／系统 | 新视角 ΔPSNR | ΔSSIM | ΔLPIPS | 教师相机组 ΔPSNR | 其余训练相机组 ΔPSNR |','| --- | ---: | ---: | ---: | ---: | ---: |']
prior_table=['| 场景／同一16张训练观察 | PSNR ↑ | SSIM ↑ | LPIPS ↓ |','| --- | ---: | ---: | ---: |']
for scene,run,mp,label in SCENES:
 manifest=read(ROOT/mp/'manifest.json');data={};s={'endpoints':{},'deltas':{},'cost':{},'sources':{}}
 for method in ['sr4d','wu']:
  for it in [6000,18000]:
   p=BASE/run/f'{method}_{it}_evaluation/metrics.json';d=read(p);summary['sources'][str(p.relative_to(ROOT))]=sha(p);data[method,it]=d
   s['endpoints'][f'{method}_{it}']={'groups':{g:{k:v for k,v in x.items() if k!='per_camera'} for g,x in d['groups'].items()},'point_count':d['identity']['point_count']}
   v=d['groups']['novel_cam00']['hr'];novel.append(f'| {label} | {"SR4D" if method=="sr4d" else "Wu＋SR0.1"} | {it} | {v["psnr"]:.4f} | {v["ssim"]:.5f} | {v["lpips_alex"]:.5f} |')
  ds={g:{k:data[method,18000]['groups'][g]['hr'][k]-data[method,6000]['groups'][g]['hr'][k] for k in KEYS} for g in data[method,6000]['groups']}
  pairs=[(a,b) for a,b in zip(data[method,6000]['rows'],data[method,18000]['rows']) if a['group']=='novel_cam00']
  assert len(pairs)==60 and all((a['camera_id'],a['frame_index'])==(b['camera_id'],b['frame_index']) for a,b in pairs)
  ds['novel_per_frame_improvement_counts']={k:sum((b['hr'][k]-a['hr'][k])*(1 if k in ['psnr','ssim'] else -1)>0 for a,b in pairs) for k in KEYS}
  s['deltas'][method]=ds;v=ds['novel_cam00'];deltas.append(f'| {label}／{method} | {v["psnr"]:+.4f} | {v["ssim"]:+.5f} | {v["lpips_alex"]:+.5f} | {ds["teacher_camera_train"]["psnr"]:+.4f} | {ds["lr_only_camera_train"]["psnr"]:+.4f} |')
 for stage in ['coarse','fine']:s['cost'][stage]=read(BASE/run/'sr4d'/f'{stage}_complete.json')
 selected={(x['camera_id'],x['frame_index']) for x in data['wu',18000]['rows'] if x['group']=='teacher_camera_train'}
 if scene=='cook_spinach':
  pp=ROOT/'output/dynamic_sr_20260918/prior_quality_audit/metrics.json';pdoc=read(pp);pd=pdoc['scenes'][scene];assert pd['manifest_sha256']==sha(ROOT/mp/'manifest.json')
  chosen=[x for x in pd['rows'] if (x['camera_id'],x['frame_index']) in selected];assert len(chosen)==16
  priors={name:{k:float(np.mean([x[name]['full'][k] for x in chosen])) for k in KEYS[:3]} for name in ['bicubic','swinir']}
  source_rows=chosen
 else:
  pp=ROOT/'output/dynamic_sr_20260919/meetroom_discussion_train_prior_reference/metrics.json';pd=read(pp);assert pd['manifest_sha256']==sha(ROOT/mp/'manifest.json')
  priors={}
  for name in ['bicubic','swinir']:
   chosen=[x for x in pd['modes'][name]['rows'] if (x['camera_id'],x['frame_index']) in selected];assert len(chosen)==16
   priors[name]={k:float(np.mean([x['spatial']['full'][k] for x in chosen])) for k in KEYS[:3]}
  source_rows=[x for x in pd['sources'] if (x['camera_id'],x['frame_index']) in selected]
 obs={(x['camera_id'],x['frame_index']):x for x in manifest['observations']}
 for x in source_rows:
  row=obs[x['camera_id'],x['frame_index']]
  assert row['lr_sha256']==x['lr_sha256'] and row['hr_sha256']==x['hr_sha256']
  prior=ROOT/mp/'sr_swinir_x4'/x['camera_id']/Path(row['lr_path']).name
  assert sha(prior)==x['prior_sha256']
 summary['sources'][str(pp.relative_to(ROOT))]=sha(pp);s['teacher_quality_16']={'source':str(pp.relative_to(ROOT)),'scope':'same 16 training observations; historical diagnostics; no new teacher or evaluation','scores':priors}
 for name,vs in priors.items():prior_table.append(f'| {label}／{name} | {vs["psnr"]:.4f} | {vs["ssim"]:.5f} | {vs["lpips_alex"]:.5f} |')
 for method in ['sr4d','wu']:
  vs=data[method,18000]['groups']['teacher_camera_train']['hr'];prior_table.append(f'| {label}／{method} 18k | {vs["psnr"]:.4f} | {vs["ssim"]:.5f} | {vs["lpips_alex"]:.5f} |')
 summary['scenes'][scene]=s
 # Fixed training view/frame and normalized centre crop, not selected by error.
 row=obs['cam02',0];hr=np.asarray(Image.open(ROOT/mp/row['hr_path']),np.float32)/255
 lr=np.asarray(Image.open(ROOT/mp/row['lr_path']),np.float32)/255
 up=F.interpolate(torch.from_numpy(lr).permute(2,0,1)[None],size=hr.shape[:2],mode='bicubic',align_corners=False,antialias=True)[0].clamp(0,1).permute(1,2,0).numpy()
 teacher=np.asarray(Image.open(ROOT/mp/'sr_swinir_x4/cam02/0000.png'),np.float32)/255
 ims=[hr,up,teacher];labs=['HR reference','Observed LR, bicubic up','Frozen SwinIR']
 for met,it in [('sr4d',6000),('sr4d',18000),('wu',6000),('wu',18000)]:
  ims.append(np.clip(np.load(BASE/run/f'{met}_{it}_evaluation/predictions/cam02_0000.npy').transpose(1,2,0),0,1));labs.append(f'{met.upper()} {it//1000}k')
 h,w=hr.shape[:2];box=[int(.34*w),int(.38*h),int(.70*w),int(.90*h)];x1,y1,x2,y2=box
 fig,axes=plt.subplots(2,7,figsize=(23,6.5),layout='constrained')
 for j,im in enumerate(ims):
  axes[0,j].imshow(im,interpolation='nearest');axes[0,j].set_title(labs[j],fontsize=11)
  axes[1,j].imshow(im[y1:y2,x1:x2],interpolation='nearest')
  for i in [0,1]:axes[i,j].axis('off')
 fig.suptitle(label+' | fixed training cam02, frame 0 | same centre crop, no metric-based selection',fontsize=14)
 fig.savefig(OUT/(scene+'_training_comparison.png'),dpi=150);plt.close(fig)
 s['training_figure']={'camera':'cam02','frame':0,'crop_xyxy':box,'selection':'normalized centre: x .34-.70, y .38-.90; illustrative, not extra regional statistical evidence'}
save(OUT/'summary.json',summary)
(OUT/'tables.md').write_text('# 新视角 cam00 全60帧\n\n'+'\n'.join(novel)+'\n\n# 18k 相对6k\n\n'+'\n'.join(deltas)+'\n\n# 历史教师参照，同一16训练观察\n\n'+'\n'.join(prior_table)+'\n')
print((OUT/'tables.md').read_text())
