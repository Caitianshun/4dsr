"""Rebuild and verify the original B step-zero state; audit legal HR targets."""
import argparse
import json
from pathlib import Path
import random
import sys
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/dynamic_sr_detail_supervision_20260924'))
from training_support import initial_identity,digest_state
from motion_model import make_model,load_model,refinement_state
from common import load_checkpoint,downsample,image_tensor,sha256,write_json
from n3dv_data import load_manifest
sys.path.insert(0,str(ROOT/'experiments/dynamic_sr_20260920'))
from resume_control import restore_global_rng


def main():
    p=argparse.ArgumentParser()
    for k in ['manifest','parent','selection','legacy-config','out']:p.add_argument('--'+k,required=True,type=Path)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4)
    m=load_manifest(a.manifest);legacy=json.loads(a.legacy_config.read_text())
    assert sha256(a.parent)==legacy['parent_sha256'] and sha256(a.selection)==legacy['selection_sha256'] and sha256(a.manifest)==legacy['manifest_sha256']
    g,h,o,ck=load_checkpoint(a.parent);selection=torch.load(a.selection,map_location='cpu',weights_only=False)
    model=make_model(g,h,o,ck,m,'ordinary_split',selection)
    identity=initial_identity(model);assert identity==legacy['initial_identity'],(identity,legacy['initial_identity'])
    restore_global_rng(ck['rng']);lr_rng=random.Random(20260923+177);sr_rng=random.Random(20260923+211)
    rng=dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all(),numpy=__import__('numpy').random.get_state(),python=random.getstate())
    metadata=dict(ck['metadata'],stage='controlled_headroom_step0',intervention_step=0,method='common_B_before_SR',initial_identity=identity,
        parent_sha=sha256(a.parent),selection_sha256=sha256(a.selection),manifest_sha=sha256(a.manifest))
    payload=dict(model=g.capture(),hidden=vars(h),optim=vars(o),metadata=metadata,rng=rng,
        samplers=dict(lr=lr_rng.getstate(),sr=sr_rng.getstate(),source_prefix_exposure={},additional_exposure={}),motion_refinement=refinement_state(model))
    path=a.out/'common_start.pt';torch.save(payload,path)
    restored=load_model(path,m);assert initial_identity(restored)==identity
    write_json(a.out/'start_identity.json',dict(status='passed',identity=identity,path=str(path.resolve()),sha256=sha256(path),
        source_parent_sha256=sha256(a.parent),selection_sha256=sha256(a.selection),rng_digest=digest_state(rng),intervention_step=0,
        disclosure='B selection used LR+SR gradients; Z removes subsequent teacher loss, not a fully teacher-free pipeline'))
    rows=[];targets=[]
    with torch.no_grad():
        for r in m['observations']:
            if r['split']!='train':continue
            assert r['camera_id'] in [f'cam{i:02d}' for i in range(2,21)]
            hr=Path(m['_root'])/r['hr_path'];lr=Path(m['_root'])/r['lr_path']
            assert sha256(hr)==r['hr_sha256'] and sha256(lr)==r['lr_sha256']
            hrt=image_tensor(hr,device='cuda');lrt=image_tensor(lr,device='cuda');projected=downsample(hrt,lrt.shape[-2:]);delta=(projected-lrt).double()
            maximum=float(delta.abs().max());assert maximum<=.5/255+2e-6,(r['camera_id'],r['frame_index'],maximum)
            rows.append(dict(camera=r['camera_id'],frame=r['frame_index'],l1=float(delta.abs().mean()),mse=float(delta.square().mean()),max_abs=maximum))
            targets.append(dict(camera=r['camera_id'],frame=r['frame_index'],relative_path=r['hr_path'],sha256=r['hr_sha256'],lr_sha256=r['lr_sha256']))
    assert len(rows)==1140
    write_json(a.out/'privileged_train_hr.json',dict(status='completed_train_only_index',privileged_train_hr=True,manifest_sha256=sha256(a.manifest),entries=targets))
    write_json(a.out/'degradation_audit.json',dict(status='passed',observations=len(rows),tolerance=.5/255+2e-6,rows=rows,
        protocol='Existing common.downsample: bicubic antialias align_corners=False clamp; observed uint8 LR quantization'))


if __name__=='__main__':main()
