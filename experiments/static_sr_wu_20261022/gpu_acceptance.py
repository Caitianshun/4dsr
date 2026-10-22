"""Bounded real-GPU wiring acceptance; never a parent, teacher, or benchmark.

Default is CPU --check-only semantics. Supply --run only inside the resource
owner's admitted single-GPU job. A fresh output directory is always required.
The fixture uses real training LR and full native image sizes, but fabricated
supervision to exercise loss/gradient plumbing. It does NOT validate teacher
networks, convergence, full topology schedules, or scientific SR performance.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import socket
import sys
import tempfile
import time
import traceback

from common import ROOT, METHODS, checked, identity, read, resolve, sha, training_manifest, write_new
from prepare_suite import SCENES, DEFAULT_PREPARED

FIXTURE_SCHEMA = 'static_sr_wu_gpu_acceptance_fixture_v1'
FIXTURE_STAGE = 'gpu_acceptance_fixture_never_formal_parent'
DEFAULT_SCENES = ('Cook07', 'Discussion07')
DEFAULT_FRAMES = (0, 149, 299)
RGB_GATE = {'maximum_absolute': 5e-4, 'mean_absolute': 5e-6}


def select_samples(manifest, rows, frames):
    """Same-time batches from at least three legal cameras; original t=f/300."""
    if len(frames) < 2 or len(set(frames)) != len(frames):
        raise ValueError('Acceptance needs at least two distinct original frames')
    if set(frames) - set(manifest['frame_indices']):
        raise ValueError('Requested acceptance frame is absent')
    train = set(manifest['splits']['train'])
    held = set(manifest['splits'].get('test', [])) | set(manifest['splits'].get('dev', []))
    groups = []
    for frame in frames:
        group = sorted((r for r in rows if r['frame_index'] == frame), key=lambda r:r['camera_id'])
        if len(group) < 3 or len({r['camera_id'] for r in group}) != len(group):
            raise ValueError('At least three distinct training cameras are required at every selected time')
        for r in group:
            if r['split'] != 'train' or r['camera_id'] not in train or r['camera_id'] in held or r['camera_id'] == 'cam00':
                raise ValueError('Acceptance may read only training cameras')
            if abs(r['time']-frame/300.) > 1e-8:
                raise ValueError('Acceptance keeps original frame_index / 300')
        groups.append(group)
    return groups


def make_plan(prepared, scenes, frames):
    result=[]
    if len(set(scenes)) != len(scenes) or set(scenes) - dict(SCENES).keys():
        raise ValueError('Use unique registered six-scene names')
    for short in scenes:
        configs={method:read(Path(prepared)/short/(method+'.json')) for method in METHODS}
        cfg=configs['iesrgs']
        manifest,rows=training_manifest(checked(cfg['manifest']), verify_pixels=False)
        if manifest['scene'] != dict(SCENES)[short]:
            raise ValueError('Scene label disagrees with manifest')
        for method,config in configs.items():
            if (config['method']!=method or config['manifest']!=cfg['manifest']
                    or config['scene']!=cfg['scene'] or config['wu_configuration']!=cfg['wu_configuration']):
                raise ValueError('Method configurations use different scenes or Wu representations')
            checked(config['source_configuration'])
        groups=select_samples(manifest,rows,frames)
        # Hash precisely the LR images read by the fixture, never held-out or HR.
        sampled=[]
        for group in groups:
            for row in group[:3]:
                path=Path(manifest['_root'])/row['lr_path']
                if sha(path)!=row['lr_sha256']:
                    raise ValueError(f'Observed training LR bytes changed: {path}')
                sampled.append(dict(camera=row['camera_id'],frame=row['frame_index'],time=row['time'],lr=identity(path)))
        result.append(dict(short=short,scene=cfg['scene'],manifest=cfg['manifest'],resolutions=manifest['resolutions'],
            configs={method:identity(Path(prepared)/short/(method+'.json')) for method in METHODS},
            initialization=identity(Path(manifest['_root'])/manifest['initialization']['npz_path']),
            original_training_camera_count=len(manifest['splits']['train']),samples=sampled,
            note='Only first three cameras load LR; all same-time training poses define Mip filters and Splat contribution statistics'))
    if not any(e['scene'].startswith('meetroom_') for e in result) or not any(not e['scene'].startswith('meetroom_') for e in result):
        raise ValueError('Acceptance must cover both N3DV and MeetRoom dimensions')
    return dict(schema='static_sr_wu_gpu_acceptance_v1',status='pending_GPU_execution',GPU_executed=False,
        scientific_result=False,formal_parent_allowed=False,formal_prior_allowed=False,
        scenes=result,frames=list(frames),methods=list(METHODS),source=identity(__file__),
        source_snapshot={str(p.relative_to(ROOT)):identity(p) for p in Path(__file__).parent.rglob('*.py')},
        rgb_equivalence_gate=RGB_GATE,gate_note='Preregistered engineering tolerance; not a published paper tolerance',
        exclusions=['official SR/depth/diffusion teacher-network inference','teacher quality','formal parent training',
                    'full densify/prune schedule','convergence','held-out evaluation','six-scene GPU completion'],
        training_HR_read=False,heldout_pixels_read=False,
        fixture_supervision='LR bicubic interpolation plus detached fresh-model render/depth perturbation; never formal priors')


def tensor_stats(value):
    import torch
    x=value.detach()
    if not bool(torch.isfinite(x).all()):
        raise FloatingPointError('Nonfinite GPU output')
    return dict(shape=list(x.shape),minimum=float(x.min()),maximum=float(x.max()),mean=float(x.float().mean()))


def compare_images(left,right,gate=RGB_GATE):
    import torch
    if left.shape!=right.shape:
        raise ValueError('Renderer shapes differ')
    difference=(left.detach()-right.detach()).abs().float()
    tensor_stats(difference)
    flat=difference.flatten()
    maximum,mean=float(flat.max()),float(flat.mean())
    record=dict(maximum_absolute=maximum,mean_absolute=mean,
        p95=float(torch.quantile(flat,.95)),p99=float(torch.quantile(flat,.99)),
        fraction_above_maximum_gate=float((flat>gate['maximum_absolute']).float().mean()),gate=dict(gate))
    record['passed']=maximum<=gate['maximum_absolute'] and mean<=gate['mean_absolute']
    return record


def tree_digest(value):
    """Device-independent exact identity of model AND Adam state, not object IDs."""
    import torch
    h=hashlib.sha256()
    def visit(x):
        if torch.is_tensor(x):
            a=x.detach().cpu().contiguous()
            h.update(str((str(a.dtype),tuple(a.shape))).encode());h.update(a.numpy().tobytes())
        elif isinstance(x,dict):
            h.update(b'dict')
            for key in sorted(x,key=lambda k:(type(k).__name__,str(k))):visit(key);visit(x[key])
        elif isinstance(x,(tuple,list)):
            h.update(type(x).__name__.encode())
            for item in x:visit(item)
        else:h.update((type(x).__name__+':'+repr(x)).encode())
    visit(value)
    return h.hexdigest()


def gradients(backend, objective, retain_graph=False):
    """Render-only autograd avoids regularization hiding a detached render path."""
    import torch
    groups=backend.g.optimizer.param_groups
    params=[p for group in groups for p in group['params'] if p.requires_grad]
    computed=torch.autograd.grad(objective,params,allow_unused=True,retain_graph=retain_graph)
    cursor=0; report={}
    for group in groups:
        values=[]
        for p in group['params']:
            if p.requires_grad:
                grad=computed[cursor];cursor+=1
                if grad is not None:
                    if not bool(torch.isfinite(grad).all()):raise FloatingPointError('Nonfinite render-only gradient')
                    values.append(float(grad.detach().abs().sum()))
        report[group['name']]=dict(absolute_sum=sum(values),connected_tensors=len(values))
    for name in ('xyz','deformation','grid'):
        if report.get(name,{}).get('absolute_sum',0)<=0:
            raise RuntimeError(f'Render path has no nonzero gradient to {name}: {report}')
    return report


def method_render(backend,obs,resolution='hr'):
    state=backend.effective_state(obs['time'])
    method=backend.config['method']
    if method in ('iesrgs','threedsr'):
        if method=='iesrgs':from methods.iesrgs_renderer import render_state
        else:from methods.threedsr_renderer import render_state
        refs=[backend.camera(row,resolution) for row in backend.observations if row['frame_index']==obs['frame_index']]
        return render_state(backend.camera(obs,resolution),state,backend.background,backend.g.active_sh_degree,refs)
    return backend.render(obs,state=state,resolution=resolution)


def save_array(out,name,value):
    import numpy as np
    path=out/(name+'.npy');path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as file:np.save(file,value.detach().float().cpu().numpy(),allow_pickle=False)
    return identity(path)


def splat_probe(backend,group,api):
    import torch
    from methods.splatsure import FidelityAccumulator, selection_weight_map
    with torch.no_grad():
        state=backend.effective_state(group[0]['time'])
        native=backend.render(group[0],state=state,resolution='lr')
        first=backend.render(group[0],state=state,resolution='lr',raster_api=api)
        raw=compare_images(native['render'],first['render'])
        clipped=compare_images(native['render'].clamp(0,1),first['render'].clamp(0,1))
        foreground=(native['render']-backend.background[:,None,None]).abs().amax(0)>1e-5
        region=compare_images(native['render'][:,foreground],first['render'][:,foreground]) if bool(foreground.any()) else None
        if not raw['passed']:
            raise RuntimeError(f'Splat/native raw RGB gate failed: {raw}')
        acc=FidelityAccumulator.create(len(state['xyz']),device=state['xyz'].device,dtype=state['xyz'].dtype)
        contribution=[]
        for index,obs in enumerate(group):
            packet=first if index==0 else backend.render(obs,state=state,resolution='lr',raster_api=api)
            for field in ('true_radii','is_contributing','contributions'):tensor_stats(packet[field])
            acc.update(index,packet['true_radii'],packet['is_contributing'])
            contribution.append(dict(camera=obs['camera_id'],contributing=int((packet['is_contributing']>0).sum()),
                visible=int(packet['visibility_filter'].sum())))
        if not any(r['contributing'] for r in contribution):raise RuntimeError('No real Splat contributors')
        scores=acc.score()
        a=backend.render(group[0],state=state,resolution='lr',raster_api=api,
            colors=scores[:,None].expand(-1,3).contiguous())['render']
        closest=(acc.max_camera_index==0).to(scores.dtype)
        b=backend.render(group[0],state=state,resolution='lr',raster_api=api,
            colors=closest[:,None].expand(-1,3).contiguous())['render']
        weight=selection_weight_map(a,b)
    return weight,dict(raw=raw,clamped=clipped,nonbackground=region,contributions=contribution,
        score=tensor_stats(scores),weight=tensor_stats(weight),
        fewer_than_three_views=int((acc.seen_count<3).sum()),fixture_parent=True)


def fixture_priors(backend,obs,out,weight=None):
    """Deliberately non-formal arrays; no prior-bundle schema is emitted."""
    import torch
    import torch.nn.functional as F
    with torch.no_grad():
        lr=backend.lr(obs)
        width,height=backend.manifest['resolutions']['hr']
        sr=F.interpolate(lr[None],size=(height,width),mode='bicubic',align_corners=False)[0].clamp(0,1)
        if backend.config['method']=='iesrgs':
            from methods.iesrgs_renderer import compute_3d_filter,render_state
            state=backend.effective_state(obs['time'])
            refs=[backend.camera(r,'lr') for r in backend.observations if r['frame_index']==obs['frame_index']]
            filt=compute_3d_filter(state['xyz'],refs)
            packet=render_state(backend.camera(obs,'hr'),state,backend.background,backend.g.active_sh_degree,refs,filter_3d=filt)
            depth=packet['median_depth']
            if float(depth.var())<=0:raise RuntimeError('Fixture median depth has no spatial variation')
            values={'internal_rgb':packet['render'].clamp_min(0)*.98,
                'internal_depth':depth*.97+.01,'external_rgb':sr,
                'external_inverse_depth':1/(depth.clamp_min(0)+.1)}
        else:
            values={'sr':sr}
            if backend.config['method']=='splatsure':
                if weight is None:raise ValueError('Real contribution-derived fixture weights required')
                values['weight']=weight
        return {name:save_array(out,f"{obs['camera_id']}_{obs['frame_index']:04d}_{name}",value) for name,value in values.items()}


def optimizer_step(backend,batch,stage,priors,iteration):
    import torch
    from train_stage import batch_loss
    backend.g.update_learning_rate(iteration)
    backend.g.optimizer.zero_grad(set_to_none=True)
    loss,packets=batch_loss(backend,batch,stage,priors,backend.config)
    if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite production objective')
    render_path=gradients(backend,packets[0]['render'].square().mean(),retain_graph=True)
    loss.backward()
    for packet in packets:
        grad=packet['viewspace_points'].grad
        if grad is None or not bool(torch.isfinite(grad).all()) or not bool(grad.abs().sum()>0):
            raise RuntimeError('Screen-space density gradient missing/nonfinite/zero')
    representatives={}
    for group in backend.g.optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                if not bool(torch.isfinite(param.grad).all()):raise FloatingPointError('Nonfinite optimizer gradient')
                if group['name'] not in representatives and bool(param.grad.abs().sum()>0):
                    representatives[group['name']]=(param,param.detach().clone())
    backend.g.optimizer.step()
    delta={name:float((param.detach()-before).abs().max()) for name,(param,before) in representatives.items()}
    for name in ('xyz','deformation','grid'):
        if delta.get(name,0)<=0:raise RuntimeError(f'Adam failed to change {name}')
    backend.g.optimizer.zero_grad(set_to_none=True)
    return dict(stage=stage,iteration=iteration,loss=float(loss.detach()),render_only_gradients=render_path,
        optimizer_parameter_max_change=delta,views=len(batch),screen_gradients='finite_nonzero',
        observation_keys=[f"{o['camera_id']}/{o['frame_index']:04d}" for o in batch],
        render_shapes=[list(p['render'].shape) for p in packets])


def seal_checkpoint(payload):
    """All saved deliverables use an unsupported formal-training schema."""
    result=dict(payload)
    result['schema']=FIXTURE_SCHEMA
    result['metadata']=dict(result.get('metadata',{}),stage=FIXTURE_STAGE,
        formal_parent_allowed=False,formal_prior_allowed=False,scientific_result=False)
    return result


def checkpoint_roundtrip(backend,obs,out,updates):
    import torch
    from backend import WuBackend
    from common import validate_lr_parent_payload
    before=tree_digest((backend.g.capture(),backend.g._deformation_accum))
    with torch.no_grad():reference=method_render(backend,obs)['render'].detach().clone()
    with tempfile.TemporaryDirectory(prefix='nonformal_roundtrip_',dir=out) as temporary:
        ck=backend.save(Path(temporary)/'temporary_checkpoint.pt',dict(stage=FIXTURE_STAGE,
            iterations=updates,optimizer_updates=updates,formal_parent_allowed=False,formal_prior_allowed=False))
        restored=WuBackend(backend.config,checkpoint=ck)
        after=tree_digest((restored.g.capture(),restored.g._deformation_accum))
        if before!=after:raise RuntimeError('Checkpoint roundtrip changed model/Adam/buffer state')
        with torch.no_grad():comparison=compare_images(reference,method_render(restored,obs)['render'],
            {'maximum_absolute':1e-6,'mean_absolute':1e-8})
        if not comparison['passed']:raise RuntimeError(f'Checkpoint roundtrip render differs: {comparison}')
        payload=torch.load(checked(ck),map_location='cpu',weights_only=False)
        for method in ('splatsure','threedsr'):
            try:validate_lr_parent_payload(payload,method)
            except ValueError:pass
            else:raise RuntimeError('Wiring fixture passed formal parent validation')
        if backend.config['method']=='iesrgs':
            from methods.iesrgs_priors import prepare_internal
            try:prepare_internal(restored,ck,out/'FORBIDDEN_FORMAL_PRIOR')
            except ValueError as exc:
                if 'internal-stage checkpoint' not in str(exc):raise
            else:raise RuntimeError('Wiring fixture passed IE internal teacher gate')
        sealed=out/'NONFORMAL_fixture_checkpoint.pt'
        torch.save(seal_checkpoint(payload),sealed)
        del restored,payload
    # The temporary loader-compatible file is removed before this receipt returns.
    gc.collect();torch.cuda.empty_cache()
    return dict(model_optimizer_buffers_sha256=before,restored_sha256=after,render=comparison,
        sealed_checkpoint=identity(sealed),schema=FIXTURE_SCHEMA,formal_parent_rejection_passed=True,
        temporary_loader_checkpoint_removed=True)


def extension_identities(modules):
    result={}
    for role,module in modules.items():
        path=Path(module.__file__).resolve()
        result[role]=dict(module_name=module.__name__,wrapper=identity(path),
            binaries=[identity(p) for p in sorted(path.parent.glob('_C*.so'))])
        if not result[role]['binaries']:raise RuntimeError(f'No extension binary identity: {role}')
    return result


def run_acceptance(plan,out,prepared,splat_source,splat_binary,ie_root,threedsr_binary):
    visible=os.environ.get('CUDA_VISIBLE_DEVICES','')
    if not visible or ',' in visible or visible=='-1':raise ValueError('Explicit one-GPU resource admission required')
    import numpy as np
    import torch
    from backend import WuBackend
    from methods.splatsure_prepare import load_official_rasterizer
    from methods import iesrgs_renderer,threedsr_renderer
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    if not torch.cuda.is_available():raise RuntimeError('Admitted GPU unavailable')
    if torch.cuda.device_count()!=1:raise RuntimeError('Exactly one visible CUDA device required')
    plan.update(status='running',GPU_executed=True,host=socket.gethostname(),device=visible,
        gpu_name=torch.cuda.get_device_name(),gpu_uuid=str(torch.cuda.get_device_properties(0).uuid),
        torch=torch.__version__,cuda=torch.version.cuda,started_utc=datetime.now(timezone.utc).isoformat(),results=[])
    begun=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    modules=None
    for scene in plan['scenes']:
        for method in METHODS:
            seed=read(checked(scene['configs'][method]))['seed']
            random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
            cfg=read(checked(scene['configs'][method]))
            backend=WuBackend(cfg,fresh=True)
            if modules is None:
                import diff_gaussian_rasterization as native_api
                splat_api=load_official_rasterizer(splat_source,splat_binary)
                ie_api=iesrgs_renderer.raster_api(ie_root)
                mip_api=threedsr_renderer.raster_api(binary_root=threedsr_binary)
                modules={'wu':native_api,'splatsure':splat_api,'iesrgs':ie_api,'threedsr':mip_api}
                plan['extensions']=extension_identities(modules)
                if len({id(m) for m in modules.values()})!=4:raise RuntimeError('Rasterizer modules alias each other')
            if sys.modules['diff_gaussian_rasterization'] is not modules['wu']:
                raise RuntimeError('An external rasterizer replaced the Wu native import')
            groups=select_samples(backend.manifest,backend.observations,plan['frames'])
            target=out/scene['short']/method;target.mkdir(parents=True,exist_ok=False)
            record=dict(scene=scene['short'],method=method,status='running',updates=[],depth_probes=[],splat_probes=[])
            plan['results'].append(record);save_receipt(out,plan)
            if method in ('splatsure','iesrgs','threedsr'):
                stage='internal' if method=='iesrgs' else 'lr'
                batch=groups[0][:3] if method=='iesrgs' else groups[0][:1]
                record['updates'].append(optimizer_step(backend,batch,stage,None,1))
            for group in groups:
                obs=group[0];weight=None
                if method=='splatsure':
                    weight,probe=splat_probe(backend,group,modules['splatsure'])
                    record['splat_probes'].append(dict(frame=obs['frame_index'],**probe))
                if method=='iesrgs':
                    packet=method_render(backend,obs)
                    stats=tensor_stats(packet['median_depth'])
                    if stats['maximum']<=0:raise RuntimeError('RADE median depth has no positive geometry')
                    grad=gradients(backend,packet['median_depth'].mean())
                    record['depth_probes'].append(dict(frame=obs['frame_index'],depth=stats,depth_only_gradients=grad,
                        filter=tensor_stats(packet['filter_3d']),depth_source='official_RADE_median_depth'))
                    del packet
                entries={(obs['camera_id'],obs['frame_index']):fixture_priors(backend,obs,target/'fixture_tensors',weight)}
                record['updates'].append(optimizer_step(backend,[obs],'sr',entries,len(record['updates'])+1))
                save_receipt(out,plan)
            record['roundtrip']=checkpoint_roundtrip(backend,groups[-1][0],target,len(record['updates']))
            record.update(status='passed',parameter_updates=len(record['updates']),formal_parent_allowed=False,
                formal_prior_allowed=False,gaussians=len(backend.g.get_xyz))
            del backend;gc.collect();torch.cuda.empty_cache();save_receipt(out,plan)
    torch.cuda.synchronize()
    if extension_identities(modules)!=plan['extensions']:raise RuntimeError('Extension identities changed during acceptance')
    if sys.modules['diff_gaussian_rasterization'] is not modules['wu']:raise RuntimeError('Wu extension import changed')
    plan.update(status='passed_bounded_GPU_wiring_only',elapsed_seconds=time.perf_counter()-begun,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),finished_utc=datetime.now(timezone.utc).isoformat(),
        all_extension_modules_coexisted=True,teacher_networks_validated=False,formal_training_ready_not_implied=True)
    return plan


def save_receipt(out,payload):
    temporary=out/'receipt.json.tmp'
    temporary.write_text(json.dumps(payload,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    os.replace(temporary,out/'receipt.json')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared',type=Path,default=DEFAULT_PREPARED)
    p.add_argument('--scenes',nargs='+',choices=[s for s,_ in SCENES],default=list(DEFAULT_SCENES))
    p.add_argument('--frames',nargs='+',type=int,default=list(DEFAULT_FRAMES))
    p.add_argument('--out',required=True,type=Path)
    mode=p.add_mutually_exclusive_group();mode.add_argument('--run',action='store_true');mode.add_argument('--check-only',action='store_true')
    p.add_argument('--splat-source',type=Path,default=ROOT/'tmp/static_sr_wu_20261022/upstream/splatsure-rasterizer')
    p.add_argument('--splat-binary',type=Path,default=ROOT/'output/static_sr_wu_20261022/runtime/splatsure')
    p.add_argument('--ie-root',type=Path,default=ROOT/'output/static_sr_wu_20261022/runtime/iesrgs')
    p.add_argument('--threedsr-binary',type=Path,default=ROOT/'output/static_sr_wu_20261022/runtime/threedsr/lib')
    a=p.parse_args()
    plan=make_plan(resolve(a.prepared),a.scenes,a.frames)
    out=resolve(a.out);out.mkdir(parents=True,exist_ok=False)
    plan['runtime_paths']={key:str(resolve(getattr(a,key))) for key in ('splat_source','splat_binary','ie_root','threedsr_binary')}
    save_receipt(out,plan)
    if a.run:
        try:
            run_acceptance(plan,out,resolve(a.prepared),resolve(a.splat_source),resolve(a.splat_binary),resolve(a.ie_root),resolve(a.threedsr_binary))
        except BaseException:
            plan.update(status='failed',error=traceback.format_exc());save_receipt(out,plan);raise
        save_receipt(out,plan)
    else:
        plan['torch_imported']='torch' in sys.modules
        save_receipt(out,plan)
    print(json.dumps({'status':plan['status'],'receipt':identity(out/'receipt.json')}))


if __name__=='__main__':main()
