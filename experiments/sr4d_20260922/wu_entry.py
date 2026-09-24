"""Run the existing Wu/SwinIR controls on author-code float bilinear inputs.

Only the input loader and known observation operator change. Existing dynamics,
teacher loss, optimizer, capacity schedule and checkpoint helpers are reused.
"""
import argparse
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / 'dynamic_sr_20260918'
CONTROL = HERE.parent / 'dynamic_sr_20260919'
sys.path.insert(0, str(OLD))
import n3dv_data

original_getitem = n3dv_data.N3DVPreparedDataset.__getitem__


def float_getitem(self, index):
    if self.resolution != 'lr' or 'lr_float_path' not in self.observations[index]:
        return original_getitem(self,index)
    obs = self.observations[index]
    cal = self.cameras[obs['camera_id']]
    path = self.root / obs['lr_float_path']
    width,height = self.manifest['resolutions']['lr']
    if self._cache is not None and index in self._cache:
        im = self._cache[index]
    else:
        from prepare_author_lr import sha
        assert sha(path) == obs['lr_float_sha256'], path
        arr = np.load(path,allow_pickle=False)
        assert arr.dtype==np.float32 and arr.shape==(3,height,width)
        im = torch.from_numpy(arr.copy()).to(self.device)
        if self._cache is not None:
            self._cache[index]=im
    return dict(image=im,image_path=str(path),camera_id=obs['camera_id'],
                frame_index=obs['frame_index'],time=obs['time'],
                K=np.asarray(cal['K_lr'],dtype=np.float64),
                w2c=np.asarray(cal['w2c'],dtype=np.float64),
                c2w=np.asarray(cal['c2w'],dtype=np.float64),
                width=width,height=height,split=self.split)


def author_downsample(image,size):
    return F.interpolate(image[None],size=size,mode='bilinear',align_corners=False)[0].clamp(0,1)


def install():
    n3dv_data.N3DVPreparedDataset.__getitem__=float_getitem
    import common
    common.downsample=author_downsample


def main():
    p=argparse.ArgumentParser(add_help=False)
    p.add_argument('--entry',choices=('warmup','controlled'),required=True)
    a,rest=p.parse_known_args()
    install()
    import common
    if a.entry=='warmup':
        import run_experiment as target
        # Keep model selection and HR out of training; full fixed evaluation
        # is attached after successful process exit by the runner.
        target.quick_development=lambda *args,**kwargs: None
    else:
        sys.path.insert(0,str(CONTROL))
        import controlled_fit as target
        rest += ['--skip-fit']
    target.downsample=author_downsample
    # Persist the adaptation identity in addition to the original train snapshot.
    ix=rest.index('--out')
    out=Path(rest[ix+1])
    provenance=dict(wrapper=str(Path(__file__).resolve()),wrapper_sha256=common.sha256(Path(__file__)),
                    input_loader='lr_float_path CHW float32; hash checked',
                    observation_operator='bilinear x4 align_corners=False noAA, clamped rendered RGB',
                    baseline='same Wu4DGS/SwinIR; new data trained from initialization, no old weights')
    sys.argv=[str(Path(target.__file__)),*rest]
    target.main()
    common.write_json(out/'author_input_adaptation.json',provenance)


if __name__=='__main__':
    main()
