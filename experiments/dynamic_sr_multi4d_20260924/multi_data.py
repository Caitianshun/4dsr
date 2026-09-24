"""Manifest-only LR cameras and frame-zero LR initialization for Multi4D."""
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def camera(m, record, image):
    from scene.cameras import Camera
    from utils.graphics_utils import focal2fov
    c=m['cameras'][record['camera_id']];w,h=m['resolutions']['hr']
    k=np.asarray(c['K_hr']);rt=np.asarray(c['w2c'])
    cam=Camera(R=rt[:3,:3].T,T=rt[:3,3],FoVx=focal2fov(k[0,0],w),FoVy=focal2fov(k[1,1],h),
        image=image,image_name=f"{record['camera_id']}/{record['frame_index']:04d}",time=record['frame_index']/120.)
    cam.image_width,cam.image_height=w,h
    cam.projection_matrix[2,0]=(2*k[0,2]+1-w)/w
    cam.projection_matrix[2,1]=(2*k[1,2]+1-h)/h
    cam.full_proj_transform=cam.world_view_transform@cam.projection_matrix
    cam.record_key=(record['camera_id'],record['frame_index'])
    return cam


def projection_check(m, cameras):
    errors=[]
    for cam in cameras[::60]:
        k=np.asarray(m['cameras'][cam.record_key[0]]['K_hr']);rt=np.asarray(m['cameras'][cam.record_key[0]]['w2c'])
        xyz_cam=np.array([[-1.,-.5,5.],[0.,0.,10.],[2.,1.,15.]])
        xyz_world=np.c_[xyz_cam,np.ones(3)]@np.linalg.inv(rt).T
        clip=xyz_world@cam.full_proj_transform.double().numpy();ndc=clip[:,:2]/clip[:,3:]
        pix=((ndc+1)*np.array(m['resolutions']['hr'])-1)/2
        target=xyz_cam@k.T;target=target[:,:2]/target[:,2:]
        errors.append(float(np.abs(pix-target).max()))
    assert max(errors)<.001,max(errors)
    return dict(max_projection_error_hr_pixels=max(errors),coordinates='OpenCV pixel centers; transposed row-vector renderer',time='frame_index / 120',last_time=59/60)


class ManifestScene:
    def __init__(self, manifest, init, foreground, background, transient, out):
        from utils.graphics_utils import BasicPointCloud
        from utils.sh_utils import SH2RGB
        from scene.dataset_readers import getNerfppNorm
        self.manifest_path=Path(manifest);self.m=json.loads(self.manifest_path.read_text());root=self.manifest_path.parent
        m=self.m;assert m['scene']=='cook_spinach' and m['splits']['train']==[f'cam{i:02d}' for i in range(2,21)]
        self.records=[r for r in m['observations'] if r['split']=='train'];assert len(self.records)==1140
        self.train_camera=[];self.input_identity=[]
        for r in self.records:
            assert r['camera_id'] in m['splits']['train']
            path=root/r['lr_path'];assert sha(path)==r['lr_sha256']
            with Image.open(path) as im:
                assert list(im.size)==m['resolutions']['lr'];rgb=np.asarray(im.convert('RGB')).copy()
            image=torch.from_numpy(rgb).permute(2,0,1).float()/255.
            self.train_camera.append(camera(m,r,image))
            self.input_identity.append(dict(camera=r['camera_id'],frame=r['frame_index'],lr_path=str(path),sha256=sha(path)))
        self.projection=projection_check(m,self.train_camera)
        report=json.loads((Path(init).parent/'report.json').read_text())
        assert report['source_frames']==[0] and not report['used_hr_images'] and not report['used_development_or_test_images']
        for s in report['source_images']:
            assert s['frame_index']==0 and s['camera_id'] in m['splits']['train'] and sha(root/s['path'])==s['sha256']
        data=np.load(init);xyz=data['points'];colors=data['colors']
        assert np.all(data['source_frame_index']==0) and colors.min()>=0 and colors.max()<=1
        norm=getNerfppNorm(self.train_camera);self.cameras_extent=norm['radius']
        centers=np.hstack(norm['cam_centers']).T
        hi=np.vstack((centers.max(0)+5*self.cameras_extent,xyz.max(0))).min(0)
        lo=np.vstack((centers.min(0)-5*self.cameras_extent,xyz.min(0))).max(0)
        # Same official per-axis random draws, 10000 dynamic points and SH2RGB.
        dynamic=np.hstack([np.random.random((10000,1))*(hi[i]-lo[i])+lo[i] for i in range(3)])
        dynamic_colors=SH2RGB(np.random.random((10000,3)))
        pcd=BasicPointCloud(dynamic,dynamic_colors,np.zeros_like(dynamic))
        foreground._deformation.deformation_net.set_aabb(dynamic.max(0),dynamic.min(0))
        foreground.create_from_pcd(pcd,self.cameras_extent)
        background.create_from_pcd(BasicPointCloud(xyz,colors,np.zeros_like(xyz)),self.cameras_extent)
        tr_xyz=foreground.get_xyz.detach().cpu().numpy()[:1]
        tr_rgb=SH2RGB(foreground._features_dc.squeeze(1).detach().cpu()[:1]).numpy()
        transient.create_from_pcd(BasicPointCloud(tr_xyz,tr_rgb,np.zeros_like(tr_xyz)),self.cameras_extent)
        # Official TR seed has one point, but distCUDA2 averages THREE neighbours:
        # FLT_MAX + FLT_MAX + FLT_MAX overflows to inf, then backward becomes NaN.
        # Preserve that exact donor point/color/time; inherit its already defined
        # spatial scale from the full 10000-point FG initialization neighbourhood.
        assert len(tr_xyz)==1
        assert not bool(torch.isfinite(transient._scaling).all())
        transient._scaling=torch.nn.Parameter(foreground._scaling[:1].detach().clone())
        self.singleton_scale_fix='One TR seed inherits its identical FG donor spatial scale; native singleton three-neighbour KNN is undefined'
        self.gaussians,self.gaussians_second,self.gaussians_transient=foreground,background,transient
        self.model_path=str(out)

    def getTrainCameras(self):return self.train_camera
    def getTestCameras(self):return []
