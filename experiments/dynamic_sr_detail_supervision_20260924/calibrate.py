"""Match initial total teacher-render-output gradient norm, without HR reads."""
import argparse
from datetime import datetime, timezone
import math
from pathlib import Path
import sys
import time
import traceback
import torch
from training_support import ROOT, make_model, render_model, read, load_teacher_index, initial_identity
from detail_loss import OPERATOR, highpass, teacher_loss
from common import load_checkpoint, sha256, write_json, resized_camera, image_tensor
from n3dv_data import load_manifest
from run_experiment import load_training
sys.path.insert(0,str(ROOT/'experiments/dynamic_sr_20260920'))
from resume_control import restore_global_rng, assert_state_equal


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['manifest','checkpoint','selection','teacher-index','schedule','out']:
        p.add_argument('--'+k,required=True,type=Path)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);start=time.monotonic()
    try:
        m=load_manifest(a.manifest);g,h,o,ck=load_checkpoint(a.checkpoint)
        assert ck['metadata']['manifest_sha']==sha256(a.manifest)
        assert ck['metadata']['mode']=='lr_integrated' and ck['metadata']['step']==7200
        assert 'motion_refinement' not in ck
        assert_state_equal(ck['model'][12],g.optimizer.state_dict())
        selection=torch.load(a.selection,map_location='cpu',weights_only=False)
        assert selection['parent_sha256']==sha256(a.checkpoint) and selection['manifest_sha256']==sha256(a.manifest)
        model=make_model(g,h,o,ck,m,'ordinary_split',selection)
        records,cameras=load_training(m);w,hgt=m['resolutions']['hr']
        cameras=[resized_camera(c,hgt,w) for c in cameras]
        paths,teachers=load_teacher_index(a.teacher_index,m,records)
        schedule=read(a.schedule);assert schedule['manifest_sha256']==sha256(a.manifest)
        restore_global_rng(ck['rng']);initial=initial_identity(model)
        observations=[i for i,r in enumerate(records) if r['frame_index'] in [0,118]]
        assert len(observations)==2*len(schedule['train_cameras'])
        A=Q=c=0.;rows=[]
        for i in observations:
            with torch.no_grad(): prediction=render_model(model,cameras[i])['render']
            prediction=prediction.detach().requires_grad_(True)
            target=image_tensor(paths[i]);size=records[i]['image'].shape[-2:]
            rgb=(prediction-target).abs().mean()
            with torch.no_grad(): ht=highpass(target,size)
            detail=(highpass(prediction,size)-ht).abs().mean()
            gg=torch.autograd.grad(rgb,prediction,retain_graph=True)[0]
            hh=torch.autograd.grad(detail,prediction)[0]
            aa=.1*gg.double();hh=hh.double()
            ai=float(aa.square().sum());qi=float(hh.square().sum());ci=float((aa*hh).sum())
            assert all(math.isfinite(v) for v in [ai,qi,ci])
            A+=ai;Q+=qi;c+=ci
            rows.append(dict(camera=records[i]['camera_id'],frame=records[i]['frame_index'],A=ai,Q=qi,c=ci,
                             teacher_sha256=sha256(paths[i]),raw_render_sha256=None))
        statistics=dict(A=A,Q=Q,c=c,observations=rows)
        if not (A>0 and Q>0 and all(math.isfinite(v) for v in [A,Q,c])):
            write_json(a.out/'calibration.json',dict(status='degenerate',**statistics))
            raise ValueError('Nonfinite/degenerate gradient; F may not start')
        discriminant=math.sqrt(c*c+3*A*Q)
        alpha=3*A/(discriminant+c) if c>=0 else (-c+discriminant)/Q
        if not (math.isfinite(alpha) and 0<alpha<1000):
            write_json(a.out/'calibration.json',dict(status='anomalous_alpha',alpha=alpha,**statistics))
            raise ValueError('Anomalous coefficient; no clipping or F launch')
        actual_sq=0.;checks={};probe=None
        for i in observations:
            with torch.no_grad(): prediction=render_model(model,cameras[i])['render']
            prediction=prediction.detach().requires_grad_(True)
            target=image_tensor(paths[i]);size=records[i]['image'].shape[-2:]
            loss,_,_=teacher_loss(prediction,target,'F',alpha,size)
            actual=torch.autograd.grad(loss,prediction)[0]
            actual_sq+=float(actual.double().square().sum())
            if probe is None: probe=(prediction.detach(),target.detach(),tuple(size))
        target_norm=2*math.sqrt(A);actual_norm=math.sqrt(actual_sq)
        assert abs(actual_norm/target_norm-1)<1e-5, (actual_norm,target_norm)
        image,target,size=probe;image=image.requires_grad_(True)
        u,_,_=teacher_loss(image,target,'U',0,size);z,_,detail0=teacher_loss(image,target,'F',0,size)
        gu=torch.autograd.grad(u,image,retain_graph=True)[0];gz=torch.autograd.grad(z,image)[0]
        constant=torch.full_like(target,.5)
        constant_error=float(highpass(constant,size).abs().max())
        assert constant_error<1e-5 and torch.equal(gu,gz) and float(u)==float(z) and detail0 is None
        # Analytic-vector check: H backward differs from pretending lowpass is detached.
        test=image.detach().requires_grad_(True); weights=torch.linspace(-1,1,test.numel(),device=test.device).reshape_as(test)
        back=torch.autograd.grad((highpass(test,size)*weights).sum(),test)[0]
        lowpass_back=float((back-weights).abs().max())
        assert lowpass_back>1e-5 and bool(torch.isfinite(back).all())
        with torch.no_grad():
            ht1=highpass(target,size).cpu();ht2=highpass(target.clone(),size).cpu()
        assert torch.equal(ht1,ht2) and not target.requires_grad
        checks=dict(alpha_zero_U_equal=True,constant_response_max=constant_error,
                    lowpass_backward_nonzero_max=lowpass_back,teacher_has_gradient=False,
                    teacher_H_cache='not used; computed online',teacher_H_repeat_exact=True,
                    initial_identity_unchanged=initial==initial_identity(model))
        assert checks['initial_identity_unchanged']
        result=dict(status='calibrated',alpha=alpha,**statistics,target_W_gradient_norm=target_norm,
                    measured_F_gradient_norm=actual_norm,measured_ratio=actual_norm/target_norm,
                    U_gradient_norm=math.sqrt(A),operator=OPERATOR,checks=checks,initial_identity=initial,
                    manifest_sha256=sha256(a.manifest),parent_sha256=sha256(a.checkpoint),selection_sha256=sha256(a.selection),
                    teacher_index_sha256=sha256(a.teacher_index),schedule_sha256=sha256(a.schedule),
                    detail_loss_sha256=sha256(Path(__file__).with_name('detail_loss.py')),source_sha256=sha256(__file__),
                    gpu=torch.cuda.get_device_name(),torch=str(torch.__version__),seconds=time.monotonic()-start,
                    parameter_updates=0,HR_reads=0,sampler_draws=0,
                    interpretation='Initial fixed train observations, total teacher render-output gradients only; not LR/regularizer/parameter/Adam/direction/full-trajectory matching',
                    finished_utc=datetime.now(timezone.utc).isoformat())
        write_json(a.out/'calibration.json',result)
        write_json(a.out/'complete.json',dict(status='completed_calibration',calibration_sha256=sha256(a.out/'calibration.json'),parameter_updates=0))
        print({'alpha':alpha,'ratio':actual_norm/target_norm,'observations':len(rows),'seconds':result['seconds']})
    except BaseException:
        write_json(a.out/'failed.json',dict(status='failed_calibration',traceback=traceback.format_exc()))
        raise


if __name__=='__main__': main()
