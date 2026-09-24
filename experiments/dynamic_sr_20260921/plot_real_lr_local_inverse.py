"""Lay out all preselected patches without covering image content with labels."""
from pathlib import Path
import json,hashlib
import numpy as np
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'output/dynamic_sr_20260921'
OUT=BASE/'real_lr_local_inverse_assessment_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
d=json.loads((OUT/'metrics.json').read_text())
records=[]
for scene in ['cook_spinach','meetroom_discussion']:
 rows=[r for r in d['results']['whole_support_valid']['rows'] if r['cell']['scene']==scene]
 canvas=Image.new('RGB',(5*192,8*234),'white');draw=ImageDraw.Draw(canvas)
 for ri,r in enumerate(rows):
  stored=np.load(BASE/f'real_lr_local_inverse_strict_footprint_v1/patch_{r["index"]:02d}_evaluation.npz');c=r['cell']
  if scene=='meetroom_discussion' and c['camera']=='cam06':
   full=np.asarray(Image.open(BASE/f'local_inverse_teacher_v1/cam06_{c["anchor"]:04d}.png').convert('RGB'),dtype=float)/255.;x,y=c['x_lr']*4,c['y_lr']*4;teacher=full[y-48:y+48,x-48:x+48]
  else:teacher=stored['swinir_existing']
  ims=[('HR evaluation',stored['hr_evaluation_only']),('Bicubic',stored['bicubic']),('Single inverse .1',stored['reference_lambda_0.1']),('Multi inverse .1',stored['multiframe_lambda_0.1']),('Frozen SwinIR',teacher)]
  for j,(name,im) in enumerate(ims):
   draw.text((j*192+2,ri*234+2),name,fill='black');canvas.paste(Image.fromarray(np.round(np.clip(im[24:72,24:72],0,1)*255).astype(np.uint8)).resize((192,192),Image.Resampling.NEAREST),(j*192,ri*234+20))
  draw.text((4,ri*234+217),f'Fixed region {r["index"]}: {c["camera"]}, frame {c["anchor"]}, LR center ({c["x_lr"]}, {c["y_lr"]})',fill='black')
 p=OUT/f'{scene}_all_fixed_patches.png';canvas.save(p);records.append(dict(path=str(p),sha256=sha(p)))
(OUT/'panel_receipt.json').write_text(json.dumps(dict(source_sha256=sha(__file__),metrics_sha256=sha(OUT/'metrics.json'),images=records),indent=2))
