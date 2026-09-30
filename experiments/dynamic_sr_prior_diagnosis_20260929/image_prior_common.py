"""Frozen image-prior diagnostics: identities and coordinate-safe operations."""
import csv, hashlib, json, time
from pathlib import Path
import cv2
import numpy as np
from scipy.stats import spearmanr
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'output/dynamic_sr_prior_diagnosis_20260929/image_priors'
MANIFEST=ROOT/'data/dynamic_sr/n3dv_prepared/cook_spinach/manifest.json'
OLD=ROOT/'output/dynamic_sr_prior_guidance_20260927/remote_a100'
TEACHER=ROOT/'output/dynamic_sr_detail_supervision_20260924/teacher_inventory_v1/cook_spinach/teacher_index.json'
VIDEO=ROOT/'output/dynamic_sr_temporal_prior_20260928/priors/video7/prior_index.json'
PAIRS=[(c,f,g) for c in ['cam02','cam06','cam12','cam18'] for f,g in [(38,40),(40,42),(78,80),(80,82)]]
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2,ensure_ascii=False,default=lambda x:x.item() if isinstance(x,np.generic) else str(x),allow_nan=False));tmp.replace(p)
def csvwrite(p,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r));
    with Path(p).open('w',newline='') as f:
        w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)
def setup():
    cv2.setNumThreads(4);cv2.setRNGSeed(20260929);OUT.mkdir(parents=True,exist_ok=True)
    m=read(MANIFEST);obs={(o['camera_id'],o['frame_index']):o for o in m['observations']}
    records=read(OLD/'depth_legal/complete.json')['rows'];keys=[(r['camera'],r['frame']) for r in records]
    return m,obs,keys

def image(obs,c,f,kind='lr'):
    o=obs[c,f];p=MANIFEST.parent/o[kind+'_path'];assert sha(p)==o[kind+'_sha256'];im=cv2.imread(str(p));assert im is not None
    return cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
def imfloat(im):return im.astype(np.float32)/255

def stats(a):
    a=np.asarray(a);a=a[np.isfinite(a)]
    return dict(n=int(a.size),mean=float(a.mean()) if a.size else None,median=float(np.median(a)) if a.size else None,p10=float(np.quantile(a,.1)) if a.size else None,p90=float(np.quantile(a,.9)) if a.size else None)
def corr(a,b):
    ok=np.isfinite(a)&np.isfinite(b);a=np.asarray(a)[ok];b=np.asarray(b)[ok]
    return float(spearmanr(a,b).statistic) if a.size>=16 and a.var()>1e-10 and b.var()>1e-10 else None

def sample(im,xy):return cv2.remap(im.astype(np.float32),np.asarray(xy[:,0],np.float32)[:,None],np.asarray(xy[:,1],np.float32)[:,None],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=float('nan'))[:,0]

def warp(im,flow):
    h,w=flow.shape[:2];yy,xx=np.indices((h,w),dtype=np.float32);x=xx+flow[...,0];y=yy+flow[...,1]
    return cv2.remap(im.astype(np.float32),x,y,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0),(x>=0)&(x<=w-1)&(y>=0)&(y<=h-1)
def resize_flow(f,shape):
    h,w=shape;oh,ow=f.shape[:2];out=cv2.resize(f,(w,h),interpolation=cv2.INTER_LINEAR);out[...,0]*=w/ow;out[...,1]*=h/oh;return out

def project(xyz,cal):
    xyz=np.asarray(xyz).reshape(-1,3);q=np.c_[xyz,np.ones(len(xyz))]@np.asarray(cal['w2c'])[:3].T;p=q@np.asarray(cal['K_lr']).T
    return p[:,:2]/p[:,2:],q[:,2]

def depths(keys):
    data={};identities={}
    for kind,fields in [('depth_legal',{'lr':'B','scale':'scale','crop':'crop'}),('depth_privileged',{'hr':'C','sr':'SwinIR'})]:
        directory=OLD/kind;j=read(directory/'complete.json');identities[kind]={'path':str(directory),'receipt_sha256':sha(directory/'complete.json'),**{k:v for k,v in j.items() if k!='rows'}}
        for r in j['rows']:
            p=directory/r['path'];assert sha(p)==r['sha256'];z=np.load(p);d=data.setdefault((r['camera'],r['frame']),{});d.update({name:z[field] for field,name in fields.items()})
    native=read(OUT/'depth_A/complete.json');native_rows={(r['camera'],r['frame']):r for r in native['rows']}
    identities['depth_A']={'path':str(OUT/'depth_A'),'receipt_sha256':sha(OUT/'depth_A/complete.json'),**{k:v for k,v in native.items() if k!='rows'}}
    for c,f in keys:
        p=OUT/'depth_A'/f'{c}_{f:04d}.npz'
        assert p.exists() and sha(p)==native_rows[c,f]['sha256']
        data[c,f]['A']=np.load(p)['depth']
    return data,identities
