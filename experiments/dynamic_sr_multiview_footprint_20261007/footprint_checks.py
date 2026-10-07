"""Small necessary M/X operator fixtures; CPU work never validates CUDA moments.

Failures and elapsed time are recorded under the isolated experiment output.
These fixtures introduce no training updates and never read development images.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import traceback

import torch
import torch.nn.functional as F

from fp_common import OUT, entry, sha, write
from footprint import (project, projection_footprint, build_area_mipmap,
                       mip_sample, bilinear_support, degradation_support,
                       downsample, edge_prediction, _degradation_axis_support)


def camera(height=24, width=32, focal=16., tx=0., ty=0., tz=0., cx=None, cy=None):
    k = torch.tensor([[focal, 0., (width-1)/2 if cx is None else cx],
                      [0., focal, (height-1)/2 if cy is None else cy],
                      [0., 0., 1.]], dtype=torch.float64)
    c2w = torch.eye(4, dtype=torch.float64)
    c2w[:3, 3] = torch.tensor((tx, ty, tz), dtype=torch.float64)
    return dict(K=k, c2w=c2w, w2c=torch.linalg.inv(c2w))


def assert_close(actual, expected, tolerance=1e-9):
    error = float((actual-expected).abs().max())
    assert error <= tolerance, (error, tolerance)
    return error


def _finite_difference(fn, tensor, index, step=1e-5):
    high, low = tensor.detach().clone(), tensor.detach().clone()
    high[index] += step
    low[index] -= step
    return float((fn(high)-fn(low))/(2*step))


def cpu_checks():
    torch.set_num_threads(4)
    checks = {}
    dtype = torch.float64
    height, width = 24, 32
    yy, xx = torch.meshgrid(torch.arange(height, dtype=dtype),
                            torch.arange(width, dtype=dtype), indexing='ij')
    expected_xy = torch.stack((xx, yy), -1)
    # Explicitly noncentral half-pixel principal point exercises the project's K.
    same = camera(height, width, cx=11.25, cy=7.75)
    depth = torch.full((height, width), 4., dtype=dtype, requires_grad=True)
    image = torch.stack((xx/(2*width)+.2, yy/(2*height)+.2,
                         (xx+yy)/(2*(width+height))+.2)).requires_grad_()
    packet = project(same, same, depth)
    coordinate_error = assert_close(packet['xy'], expected_xy)
    assert bool(packet['valid'].all())
    warped = mip_sample(image, packet['xy'], projection_footprint(packet['xy']))
    error = assert_close(warped['rgb'], image)
    # Roundoff can conservatively exclude an outermost row/column; the interior
    # must remain fully supported and is the stable fixture for derivatives.
    assert bool(warped['valid'][1:-1, 1:-1].all())
    assert not warped['lod'].requires_grad and not warped['valid'].requires_grad
    checks['identity_noncentral_principal_point'] = dict(status='passed',
        coordinate_max_error=coordinate_error, rgb_max_error=error,
        supported_fraction=float(warped['valid'].float().mean()),
        boundary_roundoff_policy='strict complete nonzero support, no tolerance waiver')

    # A frontoparallel z=4 plane: source camera translation (+x) moves its
    # coordinates left by f*baseline/z, not by f*baseline/Euclidean-range.
    source = camera(height, width, tx=.375, ty=-.125, cx=10.25, cy=8.75)
    projected = project(source, same, depth)
    expected = expected_xy + expected_xy.new_tensor((-2.5, 1.5))
    plane_error = assert_close(projected['xy'], expected)
    assert_close(projected['source_z'], depth)
    grad = torch.autograd.grad(projected['xy'][9, 10, 0], depth)[0]
    assert_close(grad[9, 10], grad.new_tensor(16*.375/4**2))
    assert int(grad.count_nonzero()) == 1
    checks['known_plane_axial_depth_and_direction'] = dict(status='passed',
        coordinate_max_error=plane_error, shift_x=-2.5, shift_y=1.5,
        analytical_dudZ=.375, actual_dudZ=float(grad[9, 10]))

    # Scalar known world-z controls target depth, giving an explicit xyz->Z->u
    # chain. This checks projection autograd; CUDA moment detachment is separate.
    point_xyz = torch.tensor((.2, -.1, 4.), dtype=dtype, requires_grad=True)
    shifted = project(source, same, point_xyz[2].expand(height, width))
    xyz_grad = torch.autograd.grad(shifted['xy'][9, 10, 0], point_xyz)[0]
    assert_close(xyz_grad, xyz_grad.new_tensor((0., 0., .375)))
    checks['synthetic_xyz_depth_chain'] = dict(status='passed', gradient=xyz_grad.tolist(),
        scope='CPU analytic world-z fixture; not native rasterizer verification')

    # A fractional translation of a high-frequency image does not commute with
    # true bicubic-AA downsampling. Wrong LR point warp is explicitly contrasted.
    texture = ((xx.floor().long()+yy.floor().long()) % 2).to(dtype)
    texture = torch.stack((texture, texture*.6+.1, texture*.3+.2))
    xy = expected_xy + expected_xy.new_tensor((.375, .625))
    right = downsample(mip_sample(texture, xy, torch.ones_like(xx))['rgb'], (6, 8))[0]
    base_lr = downsample(texture, (6, 8))[0]
    ly, lx = torch.meshgrid(torch.arange(6, dtype=dtype), torch.arange(8, dtype=dtype), indexing='ij')
    wrong_xy = torch.stack((lx+.375/4, ly+.625/4), -1)
    wrong = mip_sample(base_lr, wrong_xy, torch.ones_like(lx))['rgb']
    order_difference = float((right[:, 1:-1, 1:-1]-wrong[:, 1:-1, 1:-1]).abs().max())
    assert order_difference > 1e-6, order_difference
    checks['warp_before_down_is_distinct'] = dict(status='passed',
        interior_max_difference=order_difference, order='HR warp then original clamped D')

    # Bilinear neighbours of strictly zero coefficient do not enter support;
    # a fractional neighbour or any padding contribution does enter it.
    small_valid = torch.ones((4, 5), dtype=torch.bool)
    small_valid[1, 2] = False
    coordinates = torch.tensor([[[1., 1.], [1.25, 1.], [-.25, 1.], [4., 3.]]], dtype=dtype)
    validity = bilinear_support(small_valid, coordinates)
    assert validity.tolist() == [[True, False, False, True]], validity
    checks['bilinear_complete_nonzero_support'] = dict(status='passed', result=validity.tolist())

    # Area levels require every source HR sample in the reduction bin, including
    # bins spanning odd sizes. Fractional LOD requires both participating levels.
    odd_image = torch.ones((3, 9, 13), dtype=dtype)
    odd_valid = torch.ones((9, 13), dtype=torch.bool)
    odd_valid[4, 6] = False
    pyramid = build_area_mipmap(odd_image, odd_valid)
    assert [tuple(v.shape[-2:]) for v in pyramid['levels']] == [(9,13),(5,7),(3,4),(2,2),(1,1)]
    for image_level, valid_level in zip(pyramid['levels'], pyramid['valid']):
        assert_close(image_level, torch.ones_like(image_level))
        expected_bad = F.adaptive_max_pool2d((~odd_valid)[None,None].float(), image_level.shape[-2:])[0,0] > 0
        assert torch.equal(~valid_level, expected_bad)
    centre = torch.tensor([[[6., 4.], [2., 2.]]], dtype=dtype)
    blend = mip_sample(odd_image, centre, torch.full((1,2), 2**1.5, dtype=dtype), pyramid=pyramid)
    assert not bool(blend['valid'][0,0])
    assert_close(blend['lod'], torch.full((1,2), 1.5, dtype=dtype))
    checks['odd_area_mipmap_and_fractional_LOD'] = dict(status='passed',
        level_sizes=[list(v.shape[-2:]) for v in pyramid['levels']],
        invalid_source_pixel=[4,6], fractional_lod=1.5)

    # Smooth plane minification has sigma=4 (not area determinant or mean scale).
    scaled = expected_xy * expected_xy.new_tensor((4., 2.))
    footprint = projection_footprint(scaled)
    assert_close(footprint, torch.full_like(xx, 4.))
    checks['Jacobian_largest_singular_value'] = dict(status='passed',
        expected_max_singular=4., min_lod=2., lod_detached=True)

    # Current RGB area levels retain their own gradients; two neighbouring LODs
    # both contribute. Hold detached LOD fixed for the finite-difference oracle.
    mip_rgb = image.detach().clone().requires_grad_()
    mip_xy = torch.tensor([[[13.2,10.4],[17.3,12.6]]],dtype=dtype,requires_grad=True)
    mip_scale = torch.full((1,2),2**1.5,dtype=dtype)
    def mip_objective(rgb,coords):
        return mip_sample(rgb,coords,mip_scale)['rgb'][0].sum()
    mip_value=mip_objective(mip_rgb,mip_xy)
    mip_grgb,mip_gxy=torch.autograd.grad(mip_value,(mip_rgb,mip_xy))
    mip_rgb_idx=(0,10,13);mip_xy_idx=(0,0,0)
    mip_rgb_fd=_finite_difference(lambda value:mip_objective(value,mip_xy.detach()),mip_rgb,mip_rgb_idx)
    mip_xy_fd=_finite_difference(lambda value:mip_objective(mip_rgb.detach(),value),mip_xy,mip_xy_idx)
    assert_close(mip_grgb[mip_rgb_idx],mip_grgb.new_tensor(mip_rgb_fd),1e-9)
    assert_close(mip_gxy[mip_xy_idx],mip_gxy.new_tensor(mip_xy_fd),1e-9)
    assert abs(mip_rgb_fd)>1e-8 and abs(mip_xy_fd)>1e-8
    # Strong source minification removes a checkerboard phase instead of point
    # aliasing it; this is the expected area-mipmap approximation, not exact SR.
    tiny_y,tiny_x=torch.meshgrid(torch.arange(3,dtype=dtype),torch.arange(5,dtype=dtype),indexing='ij')
    tiny_xy=torch.stack((tiny_x*4+5.25,tiny_y*4+5.25),-1)
    aa=mip_sample(texture,tiny_xy,torch.full_like(tiny_x,4.))
    assert bool(aa['valid'].all())
    assert_close(aa['rgb'][0],torch.full_like(tiny_x,.5),1e-12)
    reused=build_area_mipmap(mip_rgb)
    try:
        mip_sample(mip_rgb.clone(),mip_xy,mip_scale,pyramid=reused)
        raise AssertionError('A stale/different source pyramid was accepted')
    except ValueError:
        pass
    checks['area_mip_RGB_and_coordinate_finite_difference'] = dict(status='passed',
        fixed_detached_LOD=1.5,RGB_autograd=float(mip_grgb[mip_rgb_idx]),RGB_fd=mip_rgb_fd,
        coordinate_autograd=float(mip_gxy[mip_xy_idx]),coordinate_fd=mip_xy_fd,
        checkerboard_minified_mean=float(aa['rgb'][0].mean()),stale_pyramid_rejected=True)

    # Exact D0 support includes negative lobes. A pair of invalid positions can
    # cancel under signed bicubic mask averaging, but logical support rejects.
    valid_hr = torch.ones((24,32), dtype=torch.bool)
    valid_hr[8,13] = False
    exact = degradation_support(valid_hr, (6,8))
    yi, yu = _degradation_axis_support(24,6)
    xi, xu = _degradation_axis_support(32,8)
    expected_valid = torch.ones((6,8), dtype=torch.bool)
    for qy in range(6):
        for qx in range(8):
            expected_valid[qy,qx] = valid_hr[yi[qy,yu[qy]][:,None], xi[qx,xu[qx]][None,:]].all()
    assert torch.equal(exact, expected_valid)
    assert bool((~exact).any())
    # Any changed invalid HR input has *zero* effect on all retained LR pixels.
    first = torch.full((3,24,32), .4, dtype=dtype)
    second = first.clone(); second[:,8,13] = .95
    one, two = downsample(first,(6,8))[0], downsample(second,(6,8))[0]
    leak = assert_close(one[:,exact], two[:,exact], tolerance=0.)
    checks['D0_full_nonzero_kernel_no_invalid_leak'] = dict(status='passed',
        valid_lr_pixels=int(exact.sum()), rejected_lr_pixels=int((~exact).sum()),
        retained_max_leak=leak, signed_mask_average_used=False)

    # End-to-end stable support finite differences, detached level selection:
    # translation causes subpixel coordinates but not a changing LOD or mask.
    safe_source = camera(height,width,tx=.03125,ty=.015625,cx=11.25,cy=7.75)
    live_rgb = image.detach().clone().requires_grad_()
    live_depth = torch.full((height,width),4.,dtype=dtype,requires_grad=True)
    fixed = torch.zeros((height,width),dtype=torch.bool); fixed[4:-4,4:-4] = True
    edge = edge_prediction(live_rgb,live_depth,safe_source,same,(12,16),frozen_mask_hr=fixed)
    selection = edge['valid_lr']
    assert bool(selection.any())
    channel_weights = live_rgb.new_tensor((.7,.2,.1))[:,None,None]
    def objective(rgb, z):
        value = edge_prediction(rgb,z,safe_source,same,(12,16),frozen_mask_hr=fixed)
        assert torch.equal(value['valid_lr'],selection)
        return (value['prediction'][:,selection]*channel_weights[:,0,0,None]).mean()
    loss = objective(live_rgb,live_depth)
    grgb,gdepth = torch.autograd.grad(loss,(live_rgb,live_depth))
    rgb_index=(0,10,13); z_index=(10,13)
    rgb_fd=_finite_difference(lambda value:objective(value,live_depth.detach()),live_rgb,rgb_index)
    depth_fd=_finite_difference(lambda value:objective(live_rgb.detach(),value),live_depth,z_index)
    assert_close(grgb[rgb_index],grgb.new_tensor(rgb_fd),1e-9)
    assert_close(gdepth[z_index],gdepth.new_tensor(depth_fd),1e-9)
    assert abs(rgb_fd)>1e-8 and abs(depth_fd)>1e-10
    checks['RGB_and_depth_finite_difference'] = dict(status='passed',
        rgb_autograd=float(grgb[rgb_index]),rgb_finite_difference=rgb_fd,
        depth_autograd=float(gdepth[z_index]),depth_finite_difference=depth_fd,
        tolerance=1e-9, stable_support_pixels=int(selection.sum()))

    # Unsafe values are sanitized before projection, sample/area, or D, and
    # complete support rejects every LR footprint touching them. Never NaN*0.
    illegal_rgb = image.detach().clone(); illegal_rgb[0,8,13]=float('nan'); illegal_rgb[1,8,14]=float('inf')
    illegal_rgb.requires_grad_()
    illegal_depth=live_depth.detach().clone();illegal_depth[9,12]=float('nan');illegal_depth[9,14]=-1.;illegal_depth[9,16]=0.
    illegal_depth.requires_grad_()
    illegal=edge_prediction(illegal_rgb,illegal_depth,safe_source,same,(6,8))
    assert bool(torch.isfinite(illegal['prediction']).all())
    assert bool((~illegal['valid_lr']).any())
    illegal['prediction'][:,illegal['valid_lr']].sum().backward()
    assert bool(torch.isfinite(illegal_rgb.grad).all() & torch.isfinite(illegal_depth.grad).all())
    assert float(illegal_rgb.grad[0,8,13])==0 and float(illegal_depth.grad[9,12])==0
    checks['invalids_safe_before_all_arithmetic'] = dict(status='passed',
        prediction_finite=True,gradients_finite=True,rejected_lr_pixels=int((~illegal['valid_lr']).sum()),
        invalid_nan_input_gradient=0.)

    # Regression for the preserved project_screen_overflow_v1 incident. Finite
    # float32 XYZ can overflow at K multiplication; numerator must be finite
    # before divide, rather than relying on an xy mask after Inf has appeared.
    extreme_k=torch.tensor([[1000.,0.,0.],[0.,1000.,0.],[0.,0.,1.]],dtype=torch.float32)
    extreme_camera=dict(K=extreme_k,w2c=torch.eye(4))
    extreme_z=torch.full((2,8),1e38,dtype=torch.float32,requires_grad=True)
    extreme=project(extreme_camera,extreme_camera,extreme_z)
    extreme['xy'].sum().backward()
    assert bool(torch.isfinite(extreme['xy']).all() & torch.isfinite(extreme_z.grad).all())
    assert not bool(extreme['valid'][:,4:].any())
    assert float(extreme_z.grad[:,4:].abs().sum())==0.
    # A tiny but legal source depth and huge finite horizontal translation can
    # also overflow a quotient despite a finite screen. Reject before divide.
    quotient_target=dict(K=torch.eye(3),w2c=torch.eye(4))
    quotient_source=dict(K=torch.eye(3),w2c=torch.eye(4))
    quotient_source['w2c'][0,3]=1e38
    quotient_z=torch.full((2,2),1e-7,dtype=torch.float32,requires_grad=True)
    quotient=project(quotient_source,quotient_target,quotient_z)
    quotient['xy'].sum().backward()
    assert bool(torch.isfinite(quotient['xy']).all() & torch.isfinite(quotient_z.grad).all())
    assert not bool(quotient['valid'].any()) and float(quotient_z.grad.abs().sum())==0.
    checks['float32_screen_and_quotient_overflow_predivision_regression'] = dict(status='passed',
        screen_fixture_shape=[2,8],screen_fixture_depth=1e38,screen_fixture_focal=1000.,
        invalid_last_four_columns_gradient_zero=True,quotient_fixture_depth=1e-7,
        quotient_fixture_translation=1e38,xy_and_gradients_finite=True)

    # Clamp is exactly the mother D, and its diagnostics distinguish overshoot.
    over=torch.full((3,height,width),1.2,dtype=dtype,requires_grad=True)
    clamped,ratio=downsample(over,(6,8))
    assert_close(clamped,torch.ones_like(clamped),0.)
    assert float(ratio)==1.
    clamped.sum().backward();assert float(over.grad.abs().sum())==0.
    checks['mother_D_clamp_and_diagnostic'] = dict(status='passed',clamp_fraction=1.,outside_gradient=0.)
    return checks


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,default=OUT/'operator_checks'/'footprint_cpu')
    args=parser.parse_args()
    started=time.monotonic();stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    target=args.out/f'{stamp}_{time.time_ns()}.json'
    try:
        checks=cpu_checks()
        result=dict(status='passed_CPU_new_footprint_operator_checks',
            started_utc=stamp,seconds=time.monotonic()-started,device='cpu',dtype='float64',
            training_updates=0,RGB_renderer_forwards=0,moment_renderer_forwards=0,
            checks=checks,source={n:entry(Path(__file__).parent/n) for n in ('footprint.py','footprint_checks.py')},
            limits=['Native CUDA render_moments opacity/covariance detachment and effective_xyz gradient need the separately authorized GPU fixture.',
                    'These small synthetic fixtures do not establish method quality, physical calibration accuracy, or full-scene visibility correctness.'])
        write(target,result);write(args.out/'latest.json',dict(status=result['status'],receipt=entry(target)))
        print(json.dumps(dict(status=result['status'],receipt=str(target),seconds=result['seconds']),ensure_ascii=False),flush=True)
    except BaseException:
        result=dict(status='failed_CPU_new_footprint_operator_checks',started_utc=stamp,
            seconds=time.monotonic()-started,error=traceback.format_exc(),training_updates=0,
            source={n:entry(Path(__file__).parent/n) for n in ('footprint.py','footprint_checks.py')})
        write(target,result);print(json.dumps(result,ensure_ascii=False),flush=True);raise


if __name__=='__main__':
    main()
