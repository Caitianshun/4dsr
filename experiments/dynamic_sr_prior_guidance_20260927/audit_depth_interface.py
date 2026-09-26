"""Synthetic numerical/derivative tests of the auxiliary depth-moment interface."""
import argparse
import copy
import math
import time
import sys
from shared import *
from moment_renderer import render_moments
from gradient_policy import effective_state


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4);ev=evaluator();m=load_manifest(a.manifest)
    obs=next(o for o in m['observations'] if o['camera_id']=='cam02' and o['frame_index']==40)
    original=ev.render_camera(m,obs,0);cam=copy.copy(original);cam.image_height=64;cam.image_width=64
    inv=torch.linalg.inv(cam.world_view_transform.cuda());screen=(32,32)
    def fixture(zs,opacities):
        xyz=(torch.tensor([[0.,0.,z,1.] for z in zs],device='cuda').reshape(-1,4)@inv)[:,:3]
        cov=torch.eye(3,device='cuda')[None].repeat(len(zs),1,1)*.04
        opacity=torch.tensor(opacities,device='cuda').reshape(-1,1)
        return xyz,cov,opacity
    tests={}
    for label,zs,ops in [('single',[2.],[.5]),('double',[2.,4.],[.5,.4]),('empty',[],[])]:
        args=fixture(zs,ops);r=render_moments(cam,*args);valid=r['alpha']>.1
        if label=='single':
            assert float((r['expected_z'][valid]-2).abs().max())<2e-5
            assert float(r['variance_z'][valid].max())<2e-5
        if label=='empty':assert float(r['moments'].abs().max())==0
        native_delta=float((r['native_depth'].squeeze()-r['moments'][1]).abs().max());assert native_delta<2e-5
        if label=='double':
            r0=render_moments(cam,*fixture([2.],[.5]));r1=render_moments(cam,*fixture([4.],[.4]))
            alpha0=r0['alpha'];alpha1=r1['alpha'];w1=(1-alpha0)*alpha1
            expected=(alpha0*2+w1*4)/(alpha0+w1).clamp_min(1e-6)
            delta=float((r['expected_z'][valid]-expected[valid]).abs().max());assert delta<2e-5
            tests['analytic_two_layers_maxabs']=delta
        tests[label]=dict(native_vs_M1_maxabs=native_delta,alpha_max=float(r['alpha'].max()),z_center=float(r['expected_z'][screen]),variance_center=float(r['variance_z'][screen]))
    # Smooth central pixels avoid bin/radius boundaries and alpha cutoffs.
    def loss(params):
        z,opacity,logscale=params.unbind();xyz=(torch.stack((z*0+.04,z*0+.025,z,z*0+1))[None]@inv)[:,:3]
        xyz2,cov2,op2=fixture([4.],[.4]);xyz=torch.cat((xyz,xyz2))
        cov=torch.cat((torch.eye(3,device='cuda')[None]*torch.exp(2*logscale),cov2))
        r=render_moments(cam,xyz,cov,torch.cat((opacity.reshape(1,1),op2)))
        # Includes normalization and moments without integer/image-range clamp.
        return (r['expected_z'][29:35,29:35]+.2*r['alpha'][29:35,29:35]+.05*r['variance_z'][29:35,29:35]).mean()
    params=torch.tensor([2.,.5,math.log(.2)],device='cuda',requires_grad=True);loss(params).backward();analytic=params.grad.clone();fd=[]
    for i in range(3):
        plus=params.detach().clone();minus=plus.clone();plus[i]+=.001;minus[i]-=.001
        fd.append(float((loss(plus)-loss(minus))/.002))
    comparison=[]
    for name,x,y in zip(['axial_center','opacity','logscale'],analytic.tolist(),fd):
        error=abs(x-y);assert error<=max(.003,.02*abs(y)),(name,x,y)
        comparison.append(dict(parameter=name,autograd=x,finite_difference=y,absolute_error=error))
    model=load_model(a.checkpoint,m)
    with torch.no_grad():
        state=effective_state(model,original.time)
        render_moments(original,state['xyz'],state['cov'],state['opacity'],(252,336));torch.cuda.synchronize();start=time.monotonic()
        for _ in range(10):r=render_moments(original,state['xyz'],state['cov'],state['opacity'],(252,336))
        torch.cuda.synchronize();ms=1000*(time.monotonic()-start)/10
    import diff_gaussian_rasterization as dr
    sources=[UPSTREAM/'submodules/depth-diff-gaussian-rasterization'/p for p in ['diff_gaussian_rasterization/__init__.py','cuda_rasterizer/forward.cu','cuda_rasterizer/backward.cu']]
    write_json(a.out/'complete.json',dict(status='passed',synthetic=tests,finite_difference=comparison,extra_render_ms=ms,
        native_definition='Sum transmittance*alpha*camera axial z, unnormalized M1; source has grad_depth backward',
        auxiliary_definition='M0,M1,M2 from precomputed float colors [1,z,z²]; black background, no clamp/8bit; bicubic AA reduction BEFORE normalization',
        reduction_caveat='D interpolation coefficients match RGB D, but its [0,1] clamp is inapplicable to moments. Bicubic overshoot can make tiny negative alpha; normalization uses epsilon and valid alpha mask.',
        sources=[dict(path=str(p),sha256=sha256(p)) for p in sources],extension=str(dr._C.__file__),extension_sha256=sha256(dr._C.__file__),gpu=torch.cuda.get_device_name(),checkpoint_sha256=sha256(a.checkpoint),used_for_training=False))


if __name__=='__main__':main()
