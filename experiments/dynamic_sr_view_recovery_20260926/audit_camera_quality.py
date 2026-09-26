"""CPU-only, post-hoc input audit. Never excludes a camera from scores automatically.

Full held-out 60-frame windows plus four fixed frames per training camera.
Texture/exposure proxies are descriptive and depend on scene content.
"""
import argparse
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def quality(image):
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    dx=cv2.Sobel(gray,cv2.CV_32F,1,0,ksize=3)
    dy=cv2.Sobel(gray,cv2.CV_32F,0,1,ksize=3)
    return dict(mean_luma=float(gray.mean()),std_luma=float(gray.std()),
        laplacian_variance=float(cv2.Laplacian(gray,cv2.CV_64F).var()),
        gradient_rms=float(np.sqrt(np.mean(dx*dx+dy*dy))),
        bright_fraction=float((gray>=250).mean()),dark_fraction=float((gray<=5).mean()),
        any_channel_ge250_fraction=float((image.max(axis=2)>=250).mean()))


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4);cv2.setNumThreads(4)
    m=json.loads(a.manifest.read_text());root=a.manifest.parent;rows=[];frames=[0,40,80,118]
    targets=['cam00','cam01'];previous={};raw_checks=[]
    for obs in m['observations']:
        cam=obs['camera_id'];frame=obs['frame_index']
        if cam not in targets and frame not in frames:continue
        paths={k:root/obs[k+'_path'] for k in ['hr','lr']}
        for k,path in paths.items():assert sha(path)==obs[k+'_sha256'],str(path)
        hr,lr=(cv2.imread(str(paths[k])) for k in ['hr','lr'])
        assert hr.shape==(1008,1344,3) and lr.shape==(252,336,3)
        h,w=hr.shape[:2];center=hr[h//10:h-h//10,w//10:w-w//10]
        row=dict(camera=cam,frame=frame,hr=quality(hr),center_hr=quality(center),lr=quality(lr))
        if cam in targets:
            t=torch.from_numpy(hr).permute(2,0,1).float()[None]/255
            low=F.interpolate(t,size=(252,336),mode='bicubic',align_corners=False,antialias=True).clamp(0,1)
            low=low[0].permute(1,2,0).mul(255).round().byte().numpy()
            row['lr_operator_max_uint8_error']=int(np.abs(low.astype(int)-lr.astype(int)).max())
            assert row['lr_operator_max_uint8_error']==0
            if cam in previous:
                row['adjacent_lr_mad']=float(np.abs(lr.astype(float)-previous[cam].astype(float)).mean()/255)
                row['duplicate_previous_lr']=bool(np.array_equal(lr,previous[cam]))
            previous[cam]=lr
        rows.append(row)
    # Re-decode the source video to distinguish prepared-image corruption from rendering failure.
    for cam in targets:
        cap=cv2.VideoCapture(m['cameras'][cam]['video']);assert cap.isOpened()
        checked=[]
        for frame in range(119):
            ok,bgr=cap.read();assert ok,(cam,frame)
            if frame not in m['frame_indices']:continue
            hr=cv2.resize(bgr,(1352,1014),interpolation=cv2.INTER_AREA)[3:1011,4:1348]
            expected=cv2.imread(str(root/f'hr/{cam}/{frame:04d}.png'))
            error=int(np.abs(hr.astype(int)-expected.astype(int)).max());assert error==0,(cam,frame,error)
            checked.append(frame)
        cap.release();raw_checks.append(dict(camera=cam,checked_frames=checked,max_uint8_error=0))
    summary={}
    for cam in m['cameras']:
        subset=[r for r in rows if r['camera']==cam]
        summary[cam]=dict(count=len(subset))
        for region in ['hr','center_hr','lr']:
            summary[cam][region]={k:float(np.mean([r[region][k] for r in subset])) for k in subset[0][region]}
        if cam in targets:
            summary[cam]['adjacent_lr_mad_mean']=float(np.mean([r['adjacent_lr_mad'] for r in subset if 'adjacent_lr_mad' in r]))
            summary[cam]['adjacent_duplicate_count']=sum(r.get('duplicate_previous_lr',False) for r in subset)
    geometry={};train=m['splits']['train']
    for cam in targets:
        c=np.asarray(m['cameras'][cam]['c2w']);neighbors=[]
        for other in train:
            v=np.asarray(m['cameras'][other]['c2w'])
            neighbors.append(dict(camera=other,center_distance=float(np.linalg.norm(c[:3,3]-v[:3,3])),
                optical_axis_angle_deg=float(np.degrees(np.arccos(np.clip(c[:3,2]@v[:3,2],-1,1))))))
        geometry[cam]=dict(center=c[:3,3].tolist(),nearest=sorted(neighbors,key=lambda x:x['center_distance'])[:4])
    for frame in frames:
        canvas=Image.new('RGB',(1344,1048),'white');draw=ImageDraw.Draw(canvas)
        for i,cam in enumerate(['cam00','cam01','cam02','cam11']):
            x=(i%2)*672;y=(i//2)*524
            draw.text((x+8,y+4),f'{cam} true HR frame {frame}; 50% display',fill='black')
            im=Image.open(root/f'hr/{cam}/{frame:04d}.png');canvas.paste(im.resize((672,504)),(x,y+20))
        canvas.save(a.out/f'inputs_frame{frame:04d}.png')
    result=dict(status='completed',audit='posthoc_camera_validity_diagnostic_only',manifest_sha256=sha(a.manifest),
        script_sha256=sha(__file__),rows=rows,by_camera=summary,raw_decode=raw_checks,camera_geometry=geometry,
        limitations=['Texture statistics depend on content, not a standalone blur test.',
            'Camera proximity and axis angle do not prove surface visibility.',
            'No target-camera observations become training or initialization inputs.',
            'No automatic camera exclusion or changed historical scores.'],parameter_updates=0)
    (a.out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status='completed',targets={c:summary[c] for c in targets},geometry=geometry),indent=2))


if __name__=='__main__':main()
