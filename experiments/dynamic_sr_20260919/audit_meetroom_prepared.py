"""CPU sanity checks for the prepared independent-domain pilot."""
from pathlib import Path
import json
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments/dynamic_sr_20260918'))
from n3dv_data import N3DVPreparedDataset, load_manifest, load_initial_points
sys.path.insert(0, str(Path(os.environ.get(
    'FOURDSR_UPSTREAM', '/home/cai_tianshun/Project/4dgs')).expanduser()))
from utils.graphics_utils import focal2fov, getProjectionMatrix

if __name__ == '__main__':
    path = ROOT / 'data/dynamic_sr/meetroom_prepared/discussion/manifest.json'
    m = load_manifest(path)
    pointcloud = load_initial_points(m)
    xyz = pointcloud['points'].astype(float)
    xh = np.c_[xyz, np.ones(len(xyz))]
    projection_rows = []
    for camera_id, camera in m['cameras'].items():
        w2c = np.asarray(camera['w2c'])
        xc = xh @ w2c.T
        for resolution in ('lr','hr'):
            width,height = m['resolutions'][resolution]
            k = np.asarray(camera['K_'+resolution])
            projected = xc[:,:3] @ k.T
            uv = projected[:,:2]/projected[:,2:]
            p = getProjectionMatrix(.01, 100., focal2fov(k[0,0],width),focal2fov(k[1,1],height)).numpy().T.astype(float)
            p[2,0]=(2*k[0,2]+1-width)/width
            p[2,1]=(2*k[1,2]+1-height)/height
            clip = xh @ w2c.T @ p
            ndc = clip[:,:2]/clip[:,3:]
            raster_uv = ((ndc+1)*np.array([width,height])-1)/2
            valid = (xc[:,2]>.01)&(xc[:,2]<100)&np.all(np.isfinite(uv),axis=1)
            visible = valid & (uv[:,0]>=0)&(uv[:,0]<width)&(uv[:,1]>=0)&(uv[:,1]<height)
            error = np.linalg.norm(uv-raster_uv,axis=1)[visible]
            assert len(error)>256 and np.max(error)<.001
            projection_rows.append({'camera_id':camera_id,'resolution':resolution,
                'visible_initial_points':int(visible.sum()),'max_projection_error_pixels':float(error.max())})
    degradation_rows=[]
    torch.set_num_threads(4)
    for camera in ('cam00','cam01','cam02','cam12'):
        for frame in (0,40,80,118):
            hr=np.array(Image.open(path.parent/f'hr/{camera}/{frame:04d}.png'))
            lr=np.array(Image.open(path.parent/f'lr/{camera}/{frame:04d}.png'))
            t=torch.from_numpy(hr.copy()).permute(2,0,1).float().div_(255).unsqueeze(0)
            expected=F.interpolate(t,size=(180,320),mode='bicubic',align_corners=False,antialias=True).clamp(0,1)
            expected=expected[0].permute(1,2,0).mul(255).round().byte().numpy()
            difference=int(np.abs(expected.astype(int)-lr.astype(int)).max())
            assert difference==0
            degradation_rows.append({'camera_id':camera,'frame':frame,'max_uint8_error':difference})
    datasets={split:N3DVPreparedDataset(m,split,'lr') for split in ('train','dev','test')}
    assert [len(datasets[s]) for s in ('train','dev','test')]==[660,60,60]
    for d in datasets.values():
        assert d[0]['image'].shape==(3,180,320)
    init=json.loads((path.parent/'initialization/report.json').read_text())
    assert not init['used_hr_images'] and not init['used_development_or_test_images']
    assert all(x['camera_id'] in m['splits']['train'] for x in init['source_images'])
    # Fixed full-image views, no hand-picked crop or rescaling of individual
    # regions. This is a dataset inspection artifact, never a training input.
    sheet=Image.new('RGB',(1280,4*204),(245,245,245))
    draw=ImageDraw.Draw(sheet)
    for index,camera in enumerate(sorted(m['cameras'])):
        x,y=(index%4)*320,(index//4)*204
        im=Image.open(path.parent/f'hr/{camera}/0040.png').convert('RGB')
        sheet.paste(im.resize((320,180),Image.Resampling.LANCZOS),(x,y+24))
        split='test' if camera=='cam00' else 'dev' if camera=='cam01' else 'train'
        draw.text((x+6,y+5),f'{camera} / {split} / frame 0040',fill=(0,0,0))
    sheet.save(path.parent/'contact_all_cameras_frame0040.png')
    temporal=Image.new('RGB',(1280,204),(245,245,245))
    draw=ImageDraw.Draw(temporal)
    for index,frame in enumerate((0,40,80,118)):
        x=index*320
        im=Image.open(path.parent/f'hr/cam00/{frame:04d}.png').convert('RGB')
        temporal.paste(im.resize((320,180),Image.Resampling.LANCZOS),(x,24))
        draw.text((x+6,5),f'cam00 / test / frame {frame:04d}',fill=(0,0,0))
    temporal.save(path.parent/'contact_test_fixed_times.png')
    result={'manifest':str(path),'split_counts':{s:len(d) for s,d in datasets.items()},
        'initialization_points':len(xyz),'initialization_world_bbox':[xyz.min(0).tolist(),xyz.max(0).tolist()],
        'initialization_median_reprojection_error_lr_px':init['reprojection_error_lr_px_median'],
        'projection_checks':projection_rows,'degradation_checks':degradation_rows,
        'projection_limitation':'CPU algebra agrees with exact supplied 4DGS projection convention; not an independent GPU raster smoke test.',
        'initialization_train_only_lr_verified':True,
        'fixed_prior_camera_ids':['cam02','cam04','cam08','cam12'],
        'training_command_note':'Use existing generic training adapter; pass --prior-cameras cam02,cam04,cam08,cam12 on SR branches.'}
    (path.parent/'prepared_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('projection_checks','degradation_checks')},indent=2))
