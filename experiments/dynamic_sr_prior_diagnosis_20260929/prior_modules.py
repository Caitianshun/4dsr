"""Frozen legal-input texture module and the bounded 2x sampling adapter.

SplatSuRe weight_maps.py supplies the selection formula. Geometry is evaluated
at each single dynamic time, retaining this project's native rasterizer. No HR
diagnostic file is read by this module. Map generation is separate from training.
"""
from collections import OrderedDict
from dv_common import *
import torch
import torch.nn.functional as F


class TexturePrior:
    def __init__(self, enabled, specification, records, manifest, maximum=64):
        self.enabled = enabled
        self.specification = specification
        self.maximum = maximum
        self.cache = OrderedDict()
        self.prepare(dict(records=records, manifest=manifest))

    def prepare(self, legal_inputs):
        from training_support import TeacherCache
        records, m = legal_inputs['records'], legal_inputs['manifest']
        idx = read(bound(self.specification['teacher']))
        by = {(e['camera'], e['frame']): e for e in idx['entries']}
        self.keys = [(r['camera_id'], r['frame_index']) for r in records]
        assert len(self.keys) == 1140 and set(self.keys) == set(by)
        assert all(c not in ['cam00', 'cam01'] for c,f in self.keys)
        paths = {i:Path(m['_root'])/by[key]['relative_path'] for i,key in enumerate(self.keys)}
        w,h=m['resolutions']['hr']; self.shape=(3,h,w)
        self.teacher = TeacherCache(paths,self.shape,maximum=self.maximum)
        if self.enabled:
            idx = read(bound(self.specification['maps']))
            assert idx['information_boundary'] == 'legal_train_LR_and_U6000_only'
            self.maps = {(e['camera'],e['frame']):e for e in idx['entries']}
            assert set(self.maps)==set(self.keys)
            self.scale=float(self.specification.get('loss_scale',1.0))
            self.targets = None
            if self.specification.get('anchored_targets'):
                aidx=read(bound(self.specification['anchored_targets']))
                assert aidx['information_boundary']=='legal_train_LR_and_SwinIR_only'
                self.targets={(e['camera'],e['frame']):e for e in aidx['entries']}
                assert set(self.targets)==set(self.keys)

    def _load(self, kind, index):
        key=(kind,index)
        if key in self.cache:
            self.cache.move_to_end(key); return self.cache[key]
        e=(self.maps if kind=='weight' else self.targets)[self.keys[index]]
        import numpy as np
        a=np.load(bound(e),allow_pickle=False)
        value=torch.from_numpy(a.copy()).float().cuda()
        if kind=='weight':
            assert value.ndim==2 and torch.isfinite(value).all()
            assert value.min()>=0 and value.max()<=1
            value=F.interpolate(value[None,None],size=self.shape[1:],mode='nearest')[0]
        else:
            assert tuple(value.shape)==self.shape and torch.isfinite(value).all()
        assert not value.requires_grad
        self.cache[key]=value
        while len(self.cache)>self.maximum: self.cache.popitem(last=False)
        return value

    def loss(self, render_state, batch):
        prediction=render_state['render']; index=batch['sr_index']
        target=self._load('target',index) if self.enabled and self.targets is not None else self.teacher.get(index)
        error=(prediction-target).abs()
        if not self.enabled:return error.mean()
        return (error*self._load('weight',index)).mean()*self.scale

    def state_dict(self):
        return dict(enabled=self.enabled,specification=self.specification,
            source='SplatSuRe selection with same-time dynamic adaptation',
            normalization='fixed full image RGB mean; fixed global scale',
            gradient='joint original Gaussian and shared deformation; all cache tensors frozen')


def render_sampling(model, camera, enabled=False):
    from sr_attribute_router import render_model
    if not enabled:return render_model(model,camera)
    from common import resized_camera
    cam=resized_camera(camera,2*camera.image_height,2*camera.image_width)
    result=render_model(model,cam)
    result['render']=F.avg_pool2d(result['render'][None],2,2)[0]
    return result
