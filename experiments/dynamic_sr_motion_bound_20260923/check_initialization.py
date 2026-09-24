"""Small necessary checks: parent state, affine covariance, split perturbation."""
import argparse
from pathlib import Path
import sys
import torch
import motion_model as mm
from common import load_checkpoint,render_image,resized_camera,downsample,image_tensor,write_json,sha256
from n3dv_data import load_manifest
from run_experiment import load_training


def main():
    p=argparse.ArgumentParser()
    for key in ['manifest','checkpoint','selection','out']:p.add_argument('--'+key,required=True)
    a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4)
    m=load_manifest(a.manifest);selection=torch.load(a.selection,map_location='cpu',weights_only=False)
    models={}
    for branch in mm.BRANCHES:
        g,h,o,ck=load_checkpoint(a.checkpoint)
        models[branch]=mm.make_model(g,h,o,ck,m,branch,selection)
    b,c=models.values()
    for key,value in b.children.state_dict().items():assert torch.equal(value,c.children.state_dict()[key]),key
    # If the dynamic parent frame equals its canonical snapshot, binding must
    # recover arbitrary oriented anisotropic child covariance, not only axes.
    q=torch.tensor([[.8,.2,.3,.4]],device='cuda');ls=torch.tensor([[-2.,-1.,.2]],device='cuda')
    lc=mm.frame(ls,q);chart=c.children.chart[:1];ci=c.children.chart_inverse[:1]
    transformed=chart@ci@lc
    covariance_error=float((transformed@transformed.transpose(-1,-2)-lc@lc.transpose(-1,-2)).abs().max())
    assert covariance_error<1e-4,covariance_error
    g,h,o,ck=load_checkpoint(a.checkpoint)
    records,cameras=load_training(m);lookup={(r['camera_id'],int(r['frame_index'])):i for i,r in enumerate(records)}
    rows=[]
    for frame in [0,40,80]:
        i=lookup[('cam02',frame)];cam=resized_camera(cameras[i],720,1280)
        with torch.no_grad():
            original=render_image(g,cam)['render']
            t=torch.full((len(g._xyz),1),float(cam.time),device='cuda')
            d=g._deformation(g._xyz,g._scaling,g._rotation,g._opacity,g.get_features,t)
            fullcov=mm.rasterize(g,cam,d[0],mm.covariance(d[1],d[2]),d[3].sigmoid(),d[4])['render']
            parity=float((fullcov-original).abs().max())
            parity_mean=float((fullcov-original).abs().mean())
            above_uint8=float(((fullcov-original).abs()>1/255).float().mean())
            assert parity_mean<1e-5 and above_uint8<1e-4,(parity,parity_mean,above_uint8)
            images={name:mm.render_model(model,cam)['render'] for name,model in models.items()}
            lr=records[i]['image'].cuda()
            row=dict(frame=frame,time=float(cam.time),native_fullcov_max_abs=parity,
                     native_fullcov_mean_abs=parity_mean,native_fullcov_fraction_above_one_uint8=above_uint8,
                     bc_initial_mean_abs=float((images['ordinary_split']-images['bound_split']).abs().mean()),
                     parent_lr_l1=float((downsample(original,lr.shape[-2:])-lr).abs().mean()))
            for name,image in images.items():
                row[name]=dict(initial_hr_perturbation_l1=float((image-original).abs().mean()),
                               initial_lr_perturbation_l1=float((downsample(image,lr.shape[-2:])-downsample(original,lr.shape[-2:])).abs().mean()),
                               lr_l1=float((downsample(image,lr.shape[-2:])-lr).abs().mean()))
            rows.append(row)
    # One actual LR+SR backward on a training image in each arm.
    teacher=image_tensor(Path(m['_root'])/'sr_swinir_x4/cam02/0080.png')
    gradient={}
    for name,model in models.items():
        image=mm.render_model(model,cam)['render']
        loss=(downsample(image,lr.shape[-2:])-lr).abs().mean()+.1*(image-teacher).abs().mean()
        loss.backward();norms={}
        for key,param in model.children.named_parameters():
            assert param.grad is not None and torch.isfinite(param.grad).all(),(name,key)
            norms[key]=float(param.grad.norm())
        if name=='bound_split':
            ids=model.children.parent_ids.unique()
            for key in ['_opacity','_features_dc','_features_rest']:
                assert getattr(model.g,key).grad[ids].abs().max()==0,key
            norms['parent_xyz_selected']=float(model.g._xyz.grad[ids].norm())
            assert norms['parent_xyz_selected']>0
        gradient[name]=norms
    write_json(out/'complete.json',dict(status='completed_initialization_check',parameter_updates=0,
        selected_init_equal=True,covariance_identity_max_abs=covariance_error,rows=rows,gradient=gradient,
        capacity={name:mm.capacity_summary(model) for name,model in models.items()},
        source_sha256=sha256(__file__),model_sha256=sha256(mm.__file__),selection_sha256=sha256(a.selection)))
    print((out/'complete.json').read_text(),flush=True)


if __name__=='__main__':main()
