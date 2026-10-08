"""Registered full-time native-parent/native-SR evaluation, with zero updates.

Register is metadata-only CPU work. Evaluate needs a root-allocated physical GPU.
The 300-frame project crop is a derived evaluation protocol, not the unchanged
Wu/N3DV paper benchmark. Test LR is diagnostic-only and opened after rendering.
"""
from __future__ import annotations
import argparse,ast,csv,fcntl,json,math,os,random,socket,sys,time,traceback
from pathlib import Path
from types import SimpleNamespace
from fp_common import ROOT,HERE,OUT,entry,bound,local,read,write,sha,module
import full_prefix_registered as prefix
import full_native_protocol as domain

SCHEMA='registered_full_native_evaluation_protocol_v1'
PARENT_SCHEMA='full_author_LR_prefix_checkpoint_v1'
SR_SCHEMA='registered_full_native_SR_refinement_v1'
REPRESENTATION='native_author_GaussianModel_no_children'
GROUPS=('xyz','deformation','grid','f_dc','f_rest','opacity','scaling','rotation')
LEGACY=ROOT/'experiments/dynamic_sr_20260918/evaluate.py'
CLOSURE=ROOT/'experiments/dynamic_sr_detail_supervision_20260924/evaluate.py'
FFT=ROOT/'experiments/dynamic_sr_same_observation_20260930/frequency.py'
ROI=ROOT/'experiments/dynamic_sr_prior_diagnosis_20260929/evaluation_adapter.py'
OLD_ROI=ROOT/'output/dynamic_sr_prior_diagnosis_20260929/spectrum/roi_protocol.json'


def extract(path,names,scope):
    tree=ast.parse(Path(path).read_text());nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    if set(n.name for n in nodes)!=set(names):raise ValueError('Frozen metric helper missing')
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),scope)
    return SimpleNamespace(**{name:scope[name] for name in names})


def metric_helpers():
    import numpy as np
    import cv2
    import torch
    from PIL import Image
    import torch.nn.functional as F
    scope=dict(np=np,cv2=cv2,torch=torch,Image=Image,Path=Path)
    legacy=extract(LEGACY,('read_rgb','image_array','ssim_map_rgb','masked_mean','metric_psnr','spatial_metrics','aggregate'),scope)
    closure=extract(CLOSURE,('lr_reprojection_metrics',),dict(np=np,cv2=cv2,F=F,legacy=legacy)).lr_reprojection_metrics
    roi=extract(ROI,('roi_metrics',),{}).roi_metrics
    fft=module('full_eval_frozen_radial_FFT',FFT)
    return legacy,closure,roi,fft


def sources():
    return {str(p.relative_to(ROOT)):sha(p) for p in (Path(__file__),HERE/'fp_common.py',HERE/'full_prefix.py',HERE/'full_prefix_registered.py',HERE/'full_native_protocol.py',HERE/'full_refine.py',HERE/'full_refine_registered_v2.py',HERE/'full_evaluate.py',prefix.LEGACY,LEGACY,CLOSURE,FFT,ROI,ROOT/'experiments/dynamic_sr_detail_supervision_20260924/detail_loss.py',ROOT/'experiments/dynamic_sr_20260918/common.py')}


def verify_sources(value):
    if value['sources']!=sources():raise ValueError('Frozen full evaluation source changed')


def validate_selection_gate(value,method=None):
    import full_refine_registered_v2 as native
    if value.get('status')!='registered_full_development_selection':raise ValueError('Confirmation requires an actual frozen development decision')
    methods=[value.get('baseline_method')]+value.get('selected_candidates',[])+value.get('necessary_ablation_methods',[])
    for candidate in ([method] if method else methods):native.validate_selection(value,candidate)
    return True


def validate_protocol(protocol,manifest,data,plan=None):
    if protocol.get('schema')!=SCHEMA or protocol.get('status')!='registered_full_native_evaluation_before_prediction_reads' or protocol['manifest']!=data['manifest']:raise ValueError('Exact registered full native evaluation protocol required; held confirmations require selection freeze')
    if protocol['seed']!=manifest['initialization']['seed'] or protocol['scene']!=manifest['scene'] or protocol.get('role')!=manifest.get('role'):raise ValueError('Protocol scene/seed/role differs')
    if protocol.get('data')!=data or protocol.get('resolutions')!=manifest['resolutions']:raise ValueError('Registered native data/grid identity differs')
    test=sorted((r for r in manifest['observations'] if r['split']=='test'),key=lambda r:(r['camera_id'],int(r['frame_index'])))
    if protocol.get('test_input_SHA256')!=prefix.digest([(r['camera_id'],r['frame_index'],r['hr_sha256'],r['lr_sha256']) for r in test]):raise ValueError('Frozen evaluation reference metadata differs')
    if protocol.get('measurement_definition_SHA256')!=prefix.digest(protocol['measurement_definitions']):raise ValueError('Evaluation measurement definition changed')
    if protocol['test_keys']!=[['cam00',f] for f in range(300)] or protocol['camera_interface_SHA256']!=prefix.digest(manifest['cameras']):raise ValueError('Registered full test/camera interface changed')
    if manifest.get('role') not in ('already_used_development','held_confirmation'):raise ValueError('Unregistered scene information boundary')
    if manifest['role']=='held_confirmation' and not protocol.get('selection'):raise ValueError('Held confirmation must wait for actual root-frozen selection')
    if protocol.get('selection'):validate_selection_gate(read(bound(protocol['selection'])),plan.get('method') if plan and plan['schema']==SR_SCHEMA else None)
    legal={(r['camera_id'],int(r['frame_index'])) for r in manifest['observations'] if r['split']=='train'}
    keys=[tuple(k) for k in protocol['train_teacher_diagnostics']['keys']]
    if len(keys)!=len(set(keys)) or not set(keys)<=legal or any(c=='cam00' for c,f in keys):raise ValueError('Train diagnostics cannot contain held-out observations')
    if bool(keys)!=bool(protocol['train_teacher_diagnostics']['teacher_index']):raise ValueError('Diagnostic teacher/keys mismatch')
    if plan and (plan['seed']!=protocol['seed'] or plan['scene']!=protocol['scene']):raise ValueError('Native checkpoint scene/seed differs from registered data')
    return True


def validate_full_manifest(path):
    path=local(path).resolve();m=read(path);seed=m['initialization']['seed'];_,identity,_=prefix.validate_manifest(path,seed)
    rows=sorted([o for o in m['observations'] if o['split']=='test'],key=lambda x:(x['camera_id'],int(x['frame_index'])))
    if len(rows)!=300 or {(r['camera_id'],int(r['frame_index'])) for r in rows}!={('cam00',f) for f in range(300)}:raise ValueError('Exact full cam00 x 300 test observations required')
    if any(abs(float(r['time'])-int(r['frame_index'])/300)>1e-7 for r in rows):raise ValueError('Full test time must be frame_index/300')
    for camera in m['cameras'].values():
        for grid in ('hr','lr'):
            if not all(math.isfinite(float(x)) for row in camera['K_'+grid] for x in row):raise ValueError('Nonfinite camera matrix')
    return path,m,identity,rows


def safe_asset(root,relative):
    path=Path(root)/relative
    if path.is_symlink() or not path.resolve().is_relative_to(Path(root).resolve()):raise ValueError('Pixel/float asset escaped its registered root')
    return path


def roi_bridge(manifest,manifest_path,roi_path):
    if roi_path is None:return dict(status='not_registered_no_region_invented',regions_by_camera={},frames=[])
    roi_path=local(roi_path);old=read(roi_path)
    if old['scene']!=manifest['scene']:raise ValueError('ROI scene differs; no region is invented for a new scene')
    rows={(o['camera_id'],int(o['frame_index'])):o for o in manifest['observations']};refs=[]
    for value in old['reference_images']:
        key=(value['camera'],int(value['frame']));row=rows[key]
        if row['hr_sha256']!=value['sha256']:raise ValueError('Old fixed ROI reference differs from full project crop')
        # Check bytes, but never decode HR or inspect prediction quality here.
        path=safe_asset(manifest_path.parent,row['hr_path'])
        if sha(path)!=value['sha256']:raise ValueError('ROI reference bytes changed')
        refs.append(dict(camera=key[0],frame=key[1],full_reference=dict(path=str(path.relative_to(ROOT)),sha256=value['sha256'])))
    regions={camera:items for camera,items in old['regions_by_camera_xyxy_exclusive'].items() if camera in manifest['cameras']}
    width,height=manifest['resolutions']['hr']
    for items in regions.values():
        for box in items.values():
            if len(box)!=4 or not all(type(x) is int for x in box) or not (0<=box[0]<box[2]<=width and 0<=box[1]<box[3]<=height):raise ValueError('Frozen ROI outside full crop')
    return dict(status='registered_old_fixed_ROI_same_crop_reference_SHA_bridge',original_protocol=entry(roi_path),regions_by_camera=regions,frames=old['fixed_still_frames'],reference_images=refs,interpretation='Exact old fixed xyxy rectangles at old frames40/80. Frame40 labels are reference locations, not semantic masks or tracked objects; old manifest identity is preserved.',new_HR_pixel_quality_selection=False)


def register(manifest,protocol,roi=None,teacher_index=None,train_keys=None,selection=None):
    started=time.monotonic();manifest,m,data,test=validate_full_manifest(manifest)
    if roi is None and m['scene']=='cook_spinach':roi=OLD_ROI
    keys=[(str(c),int(f)) for c,f in (train_keys or [])]
    legal={(r['camera_id'],int(r['frame_index'])) for r in m['observations'] if r['split']=='train'}
    if len(keys)!=len(set(keys)) or not set(keys)<=legal or any(c=='cam00' for c,f in keys):raise ValueError('Training diagnostics require distinct legal training keys; never test/dev')
    if bool(keys)!=bool(teacher_index):raise ValueError('Teacher diagnostics need explicit train keys and complete full teacher index together')
    teacher=entry(local(teacher_index)) if teacher_index else None
    if teacher:
        import full_refine_registered_v2 as native
        native.validate_teacher(read(bound(teacher)),m,data['manifest'],m['initialization']['seed'],manifest.parent)
    gate=entry(local(selection)) if selection else None
    if m.get('role') not in ('already_used_development','held_confirmation'):raise ValueError('Unregistered scene information boundary')
    held=m['role']=='held_confirmation'
    if gate:validate_selection_gate(read(bound(gate)))
    definitions=dict(quality='Same clamped float32 HWC RGB before PNG; legacy RGB no-shave PSNR/11x11 Gaussian SSIM sigma1.5 population covariance excluding image outer5; scalar AlexNet LPIPS v0.1.',ROI='Complete-image SSIM/spatial LPIPS maps then old fixed ROI masks; regional spatial LPIPS is not standard scalar LPIPS.',FFT='Same float32 residual -> float64 FFT; radial low<.125 mid[.125,.25) high>=.25, no window/DC removal, absolute band MSE and PSNR; shares from summed energies only.',LR_closure='Diagnostic after rendering: bicubic raw.float() antialias=True align_corners=False then clamp; compare actual quantized LR using float64 error. Test LR never enters model or training.',aggregation='Equal observation mean PSNR/SSIM/LPIPS; pooled-MSE PSNR separately named. Frame std describes correlated frames, not independent confidence.',scope=f"cam00 all300 frame_index/300. Registered {m['scene']} native HR{m['resolutions']['hr'][0]}x{m['resolutions']['hr'][1]} / LR{m['resolutions']['lr'][0]}x{m['resolutions']['lr'][1]}, corrected pixel-centre K. N3DV uses the project area/crop interface; MeetRoom uses original 1280x720 HR and area LR320x180 with the explicit Wu generic domain adaptation. Derived project full-time protocol, not unchanged author paper benchmark.")
    value=dict(schema=SCHEMA,status='registered_full_native_evaluation_before_prediction_reads' if not held or gate else 'pending_held_confirmation_selection_freeze',scene=m['scene'],seed=m['initialization']['seed'],manifest=data['manifest'],original_full_manifest=m['original_full_manifest'],data=data,test_keys=[['cam00',f] for f in range(300)],test_input_SHA256=prefix.digest([(r['camera_id'],r['frame_index'],r['hr_sha256'],r['lr_sha256']) for r in test]),camera_interface_SHA256=prefix.digest(m['cameras']),resolutions=m['resolutions'],ROI=roi_bridge(m,manifest,roi),train_teacher_diagnostics=dict(keys=[list(k) for k in keys],teacher_index=teacher,status='explicit_train_only_read_only_diagnostics' if keys else 'not_requested_basic_full_test_independent'),test_LR_closure='evaluation_only_after_render_not_method_supervision',selection=gate,role=m.get('role'),measurement_definitions=definitions,measurement_definition_SHA256=prefix.digest(definitions),sources=sources(),HR_quality_inspected_during_registration=False,new_GPU_forwards=0,Adam_calls=0,CPU_registration_wall_seconds=time.monotonic()-started)
    protocol=local(protocol)
    if protocol.exists():raise FileExistsError('Registered protocol preserved; choose a new protocol path')
    write(protocol,value);return value


def validate_checkpoint(checkpoint,manifest_entry):
    checkpoint=local(checkpoint);side=read(checkpoint.with_suffix('.json'))
    if side.get('schema') not in (PARENT_SCHEMA,SR_SCHEMA):raise ValueError('Short-window/two-Adam/custom checkpoint refused; no conversion')
    if {k:side[k] for k in ('path','sha256')}!=entry(checkpoint):raise ValueError('Native checkpoint SHA differs from committed sidecar')
    plan_path=checkpoint.parent/'config.json';plan=read(plan_path);meta=side['metadata']
    if plan['data']['manifest']!=manifest_entry or meta['plan_sha256']!=prefix.digest(plan) or plan['author_commit']!=prefix.PIN or meta['representation']!=REPRESENTATION:raise ValueError('Native checkpoint plan/manifest/representation differs')
    if [x['name'] for x in meta['audit']['optimizer_groups']]!=list(GROUPS):raise ValueError('Exactly one native eight-group Adam required')
    if side['schema']==PARENT_SCHEMA:
        import full_refine_registered_v2 as native
        parent_identity, validated_plan = native.full_parent(checkpoint, checkpoint.parent/'complete.json', manifest_entry, plan['seed'])
        if validated_plan != plan:
            raise ValueError('Native parent plan file differs from accepted completion plan')

    else:
        if plan['schema']!=SR_SCHEMA or plan['status']!='registered_ready_full_native_SR_refinement' or not 0<int(meta['cursor'])<=int(plan['schedule']['parameter_updates']) or meta.get('density_statistical_boundary_applied') is not True:raise ValueError('Unregistered native SR checkpoint')
        for item in plan['parent'].values():
            if isinstance(item,dict) and set(('path','sha256'))<=set(item):bound(item)
    return dict(checkpoint=entry(checkpoint),sidecar=entry(checkpoint.with_suffix('.json')),plan=entry(plan_path),complete=entry(checkpoint.parent/'complete.json') if side['schema']==PARENT_SCHEMA else None,schema=side['schema'],metadata=meta,model_kind='native_LR_parent_direct_HR_render' if side['schema']==PARENT_SCHEMA else 'native_full_SR_refinement'),plan


def validate_payload(payload,checkpoint_identity,plan):
    if payload['schema']!=checkpoint_identity['schema'] or payload['plan']!=plan or payload['metadata']!=checkpoint_identity['metadata']:raise ValueError('Payload and native committed metadata disagree')
    model=payload['model']
    if not isinstance(model,(tuple,list)) or len(model)!=14 or [g['name'] for g in model[12]['param_groups']]!=list(GROUPS):raise ValueError('Native14tuple/oneAdam8groups required; old custom model is forbidden')
    for item in ('rng','parameter_gradients','deformation_accum'): 
        if item not in payload:raise ValueError('Exact native model/RNG/gradient evidence missing')
    return True


def validate_author_runtime(plan,upstream):
    runtime=plan.get('runtime_source_sha256')
    if runtime is None:
        import full_refine_registered_v2 as native
        native.verify_sources(plan,upstream);runtime=plan['source_files']['upstream']
    if set(runtime)!=set(prefix.RUNTIME_FILES):raise ValueError('Incomplete author runtime identity')
    for relative,expected in runtime.items():
        if sha(Path(upstream)/relative)!=expected:raise ValueError('Native author runtime changed: '+relative)
    return runtime


def camera_item(manifest,observation,torch_module):
    import numpy as np
    camera=manifest['cameras'][observation['camera_id']];width,height=manifest['resolutions']['lr']
    return dict(image=torch_module.zeros((3,height,width),dtype=torch_module.float32,device='cpu'),camera_id=observation['camera_id'],frame_index=observation['frame_index'],time=observation['time'],width=width,height=height,K=np.asarray(camera['K_lr'],dtype=np.float64),w2c=np.asarray(camera['w2c'],dtype=np.float64))


def aggregate_flat(rows,fields):
    values={name:sum(float(r[name]) for r in rows)/len(rows) for name in fields if rows and all(r.get(name) is not None for r in rows)}
    return dict(observations=len(rows),equal_observation_means=values)


def spectrum_summary(rows):
    result=aggregate_flat(rows,('low_mse','mid_mse','high_mse','low_psnr','mid_psnr','high_psnr','mse','psnr'))
    result['maximum_parseval_gap']=max(r['parseval_gap'] for r in rows)
    total=sum(r['mse'] for r in rows)
    result['shares_of_total_MSE']={name:sum(r[name+'_mse'] for r in rows)/total if total>0 else None for name in ('low','mid','high')}
    return result


def metric_runtime(metric,lpips_module,torch_module,native):
    import importlib.metadata
    import cv2
    import numpy as np
    import torchvision
    learned=Path(lpips_module.__file__).resolve().parent/'weights/v0.1/alex.pth'
    alex=Path(torch_module.hub.get_dir())/'checkpoints/alexnet-owt-7be5be79.pth'
    if not learned.is_file() or not alex.is_file():raise ValueError('Exact cached LPIPS v0.1/AlexNet asset identities required')
    metric_files=(Path(lpips_module.__file__).resolve(),learned.parent.parent.parent/'pretrained_networks.py',Path(torchvision.__file__).resolve().parent/'models/alexnet.py')
    return dict(LPIPS=dict(network='alex',version='0.1',scalar_standard=True,spatial_ROI_extension=True,learned_weights=dict(path=str(learned),sha256=sha(learned)),AlexNet_pretrained_weights=dict(path=str(alex),sha256=sha(alex)),loaded_state_SHA256=native.tree_digest(metric.state_dict()),source_files={str(p):sha(p) for p in metric_files}),versions=dict(lpips=importlib.metadata.version('lpips'),numpy=str(np.__version__),opencv=str(cv2.__version__),torch=str(torch_module.__version__),torchvision=str(torchvision.__version__)))


def model_evidence(g,native,torch_module,numpy_module):
    return dict(native_capture=native.tree_digest(g.capture()),deformation_accum=native.tree_digest(g._deformation_accum),parameter_gradients=native.tree_digest({n:p.grad for n,p in prefix.parameter_map(g).items()}),RNG=native.tree_digest(prefix.rng_state(torch_module,numpy_module)),parameter_versions={name:int(p._version) for name,p in prefix.parameter_map(g).items()},topology_audit=prefix.topology_audit(g))


def csvwrite(path,rows):
    fields=list(dict.fromkeys(k for row in rows for k in row));path=Path(path)
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def evaluate(a):
    total_started=time.monotonic()
    checkpoint=local(a.checkpoint);manifest_path,m,data,test=validate_full_manifest(a.manifest);protocol_path=local(a.protocol);protocol=read(protocol_path)
    validate_protocol(protocol,m,data)
    verify_sources(protocol)
    identity,plan=validate_checkpoint(checkpoint,data['manifest']);upstream=Path(a.upstream or os.environ.get('FOURDSR_UPSTREAM','/home/cai_tianshun/Project/4dgs'))
    validate_protocol(protocol,m,data,plan)
    runtime=validate_author_runtime(plan,upstream)
    out=local(a.out).resolve();out.mkdir(parents=True,exist_ok=True)
    registration=dict(checkpoint=identity,manifest=data['manifest'],protocol=entry(protocol_path),label=a.label,sources=sources(),GPU=a.gpu_uuid,host=socket.gethostname())
    if any(out.iterdir()):raise FileExistsError('Evaluation outputs/partial results preserved; select a new directory')
    write(out/'registration.json',registration)
    if not a.operator_resource_resolved:raise ValueError('Root must resolve resource ownership before GPU dispatch')
    hardware=prefix.check_gpu(a.gpu_uuid)
    import numpy as np
    import cv2
    import torch
    import lpips
    import full_refine_registered_v2 as native
    torch.set_num_threads(a.cpu_threads);cv2.setNumThreads(a.cpu_threads);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    sys.path.insert(0,str(upstream));sys.path.insert(1,str(prefix.LEGACY.parent))
    from scene.gaussian_model import GaussianModel
    legacy,closure,roi_metric,fft=metric_helpers()
    cached_alex=Path(torch.hub.get_dir())/'checkpoints/alexnet-owt-7be5be79.pth'
    if not cached_alex.is_file():raise ValueError('Pretrained AlexNet must already be cached; network preparation is separately costed')
    metric=lpips.LPIPS(net='alex',version='0.1',spatial=False,pnet_rand=False,pretrained=True).cuda().eval().requires_grad_(False)
    metric_identity=metric_runtime(metric,lpips,torch,native);write(out/'metric_runtime.json',metric_identity)
    h=SimpleNamespace(**plan['configuration']['ModelHiddenParams']);opt=SimpleNamespace(**plan['configuration']['OptimizationParams']);pipe=SimpleNamespace(**plan['configuration']['PipelineParams'])
    g=GaussianModel(plan['configuration']['ModelParams']['sh_degree'],h);g._deformation=g._deformation.cuda()
    payload=native.checkpoint_payload(checkpoint,torch,plan,identity['schema'],'cuda');validate_payload(payload,identity,plan)
    restored=native.restore_native_model(g,payload,opt,torch,np,plan,identity['schema']);g._deformation.eval();del payload
    background=torch.tensor([1.,1.,1.] if plan.get('white_background',plan['configuration']['ModelParams']['_white_background']) else [0.,0.,0.],device='cuda')
    def forbidden(*args,**kwargs):raise RuntimeError('Evaluation cannot update optimizer or backpropagate')
    g.optimizer.step=forbidden;g.optimizer.zero_grad=forbidden
    before=model_evidence(g,native,torch,np);write(out/'model_before.json',before)
    started=time.monotonic();rows=[];freq_rows=[];region_rows=[];floats=[];diagnostics=[];cost=dict(RGB_forwards=0,deformation_forwards=0,LPIPS_forwards=0,Adam_calls=0,backward_calls=0,formal_updates=0,opened_test_HR=0,opened_test_LR_diagnostic=0,opened_train_LR_diagnostic=0,opened_train_HR_diagnostic=0,opened_train_teacher_diagnostic=0)
    torch.cuda.reset_peak_memory_stats()
    train_rows={(r['camera_id'],int(r['frame_index'])):r for r in m['observations'] if r['split']=='train'}
    teacher_spec=protocol['train_teacher_diagnostics'];teacher_rows={}
    if teacher_spec['keys']:
        index=read(bound(teacher_spec['teacher_index']));native.validate_teacher(index,m,data['manifest'],protocol['seed'],manifest_path.parent);teacher_rows={(r['camera'],int(r['frame'])):r for r in index['entries']}
    observed=test+[train_rows[tuple(k)] for k in teacher_spec['keys']]
    def pixels(obs,role,count):
        path=safe_asset(manifest_path.parent,obs[role+'_path'])
        if sha(path)!=obs[role+'_sha256']:raise ValueError('Reference/diagnostic image SHA changed')
        cost[count]+=1;return legacy.read_rgb(path)
    for number,obs in enumerate(observed):
        camera=obs['camera_id'];frame=int(obs['frame_index']);scope='full_test' if obs['split']=='test' else 'train_teacher_diagnostic'
        with torch.no_grad():
            cam=native.hr_camera(camera_item(m,obs,torch),number,m)
            state=native.native_effective_state(g,float(obs['time']),torch);cost['deformation_forwards']+=1
            raw=native.native_rgb(cam,g,state,pipe,background)['render'].detach();cost['RGB_forwards']+=1
            if raw.dtype!=torch.float32 or not bool(torch.isfinite(raw).all()) or tuple(raw.shape)!=(3,m['resolutions']['hr'][1],m['resolutions']['hr'][0]):raise ValueError('Invalid raw float32 native full render')
        stem=f'{camera}_{frame:04d}';path=out/'floats'/f'{stem}.npz';path.parent.mkdir(exist_ok=True)
        np.savez_compressed(path,rgb_raw=raw.cpu().numpy().astype(np.float32));float_entry=dict(camera=camera,frame=frame,scope=scope,path=str(path.relative_to(out)),sha256=sha(path),checkpoint_sha256=identity['checkpoint']['sha256'],raw_dtype='float32',training_cache=False)
        write(path.with_suffix('.json'),float_entry);floats.append(float_entry)
        # The native forward was completed without either test HR or test LR.
        hr=pixels(obs,'hr','opened_test_HR' if scope=='full_test' else 'opened_train_HR_diagnostic');lr=pixels(obs,'lr','opened_test_LR_diagnostic' if scope=='full_test' else 'opened_train_LR_diagnostic')
        pred=legacy.image_array(raw);empty=np.zeros(pred.shape[:2],bool)
        quality=legacy.spatial_metrics(pred,hr,empty,metric,'cuda')['full'];cost['LPIPS_forwards']+=2
        clos=closure(raw,lr,empty)['full'];row=dict(label=a.label,camera=camera,frame=frame,scope=scope,psnr=quality['psnr'],ssim=quality['ssim'],lpips=quality['lpips_alex'],mse=quality['mse'],LR_diagnostic_l1=clos['l1'],LR_diagnostic_mse=clos['mse'],LR_diagnostic_psnr=clos['psnr'],raw_below_zero_fraction=float((raw<0).float().mean()),raw_above_one_fraction=float((raw>1).float().mean()),hr_sha256=obs['hr_sha256'],lr_sha256=obs['lr_sha256'])
        if scope=='full_test':
            rows.append(row);freq_rows.append(dict(label=a.label,camera=camera,frame=frame,repeat=protocol['seed'],scope=scope,**fft.decompose(pred,hr)))
            if frame in protocol['ROI']['frames'] and camera in protocol['ROI']['regions_by_camera']:
                result=roi_metric(legacy,pred,hr,protocol['ROI']['regions_by_camera'][camera],metric,'cuda');cost['LPIPS_forwards']+=1
                for name,value in result.items():region_rows.append(dict(label=a.label,camera=camera,frame=frame,scope='fixed_test_ROI',region=name,**value))
        else:
            teacher=teacher_rows[camera,frame];teacher_path=safe_asset(manifest_path.parent,teacher['relative_path'])
            if teacher['lr_sha256']!=obs['lr_sha256'] or sha(teacher_path)!=teacher['sha256']:raise ValueError('Train teacher/input identity changed')
            tarray=legacy.read_rgb(teacher_path);cost['opened_train_teacher_diagnostic']+=1;t=torch.from_numpy(tarray.transpose(2,0,1).copy()).to(raw)
            # The same frozen H helper is extracted; no custom model imported.
            import torch.nn.functional as F
            down=extract(ROOT/'experiments/dynamic_sr_20260918/common.py',('downsample',),dict(F=F)).downsample
            highpass=extract(ROOT/'experiments/dynamic_sr_detail_supervision_20260924/detail_loss.py',('highpass',),dict(F=F,downsample=down)).highpass
            with torch.no_grad():
                row.update(teacher_RGB_l1=float((raw-t).abs().double().mean()),teacher_H_l1=float((highpass(raw,tuple(lr.shape[:2]))-highpass(t,tuple(lr.shape[:2]))).abs().double().mean()))
            tq=legacy.spatial_metrics(tarray,hr,empty,metric,'cuda')['full'];cost['LPIPS_forwards']+=2
            row.update(teacher_HR_psnr=tq['psnr'],teacher_HR_ssim=tq['ssim'],teacher_HR_lpips=tq['lpips_alex']);diagnostics.append(row)
        write(out/'progress.json',dict(status='running_full_native_zero_update_evaluation',completed_observations=number+1,cost=cost,seconds=time.monotonic()-started))
        if (number+1)%25==0:print(json.dumps(dict(stage='full_native_evaluation',label=a.label,observations=number+1,seconds=time.monotonic()-started)),flush=True)
    after=model_evidence(g,native,torch,np);write(out/'model_after.json',after)
    if before!=after:raise ValueError('Read-only native evaluation mutated model/Adam/buffers/gradients/RNG')
    verify_sources(protocol)
    if entry(checkpoint)!=identity['checkpoint'] or entry(manifest_path)!=data['manifest'] or entry(protocol_path)!=registration['protocol']:raise ValueError('Registered input bytes changed during evaluation')
    if len(rows)!=300 or {(r['camera'],r['frame']) for r in rows}!={('cam00',f) for f in range(300)}:raise ValueError('Full test results incomplete')
    csvwrite(out/'metrics_per_observation.csv',rows);csvwrite(out/'FFT_absolute_band_error.csv',freq_rows);csvwrite(out/'fixed_ROI_quality.csv',region_rows);csvwrite(out/'train_teacher_diagnostics.csv',diagnostics)
    write(out/'float_index.json',dict(status='completed_full_native_float_observations',identity=registration,entries=floats,test_observations=300,train_diagnostic_observations=len(diagnostics),training_cache=False))
    quality=aggregate_flat(rows,('psnr','ssim','lpips','mse'));quality['PSNR_of_pooled_MSE']=legacy.metric_psnr(quality['equal_observation_means']['mse'])
    spectrum=spectrum_summary(freq_rows)
    validate_protocol(protocol,m,data,plan)
    if native.tree_digest(metric.state_dict())!=metric_identity['LPIPS']['loaded_state_SHA256']:raise ValueError('Frozen metric parameters changed')
    summary=dict(status='completed_full_native_test300_zero_update_evaluation',identity=registration,quality=quality,FFT_absolute_error=spectrum,original_LR_closure_diagnostic=aggregate_flat(rows,('LR_diagnostic_l1','LR_diagnostic_mse','LR_diagnostic_psnr')),fixed_ROI_rows=len(region_rows),train_teacher_diagnostics=dict(status='completed_explicit_train_only_diagnostics' if diagnostics else 'not_requested_basic_full_test_independent',observations=len(diagnostics)),model_immutability=dict(before=entry(out/'model_before.json'),after=entry(out/'model_after.json'),equal=True),native_restore=restored,metric_runtime=entry(out/'metric_runtime.json'),cost=dict(cost,render_measurement_wall_seconds=time.monotonic()-started,total_wall_seconds=time.monotonic()-total_started,peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30),hardware=dict(physical_GPU=a.gpu_uuid,nvidia_metadata=hardware,GPU_name=torch.cuda.get_device_name(),torch=str(torch.__version__),cuda=torch.version.cuda,host=socket.gethostname()),scope=protocol['measurement_definitions']['scope'],method_selection_performed=False,not_unchanged_author_benchmark=True)
    write(out/'summary.json',summary);write(out/'complete.json',dict(status=summary['status'],identity=registration,summary=entry(out/'summary.json'),results={name:entry(out/name) for name in ('metrics_per_observation.csv','FFT_absolute_band_error.csv','fixed_ROI_quality.csv','train_teacher_diagnostics.csv','float_index.json','model_before.json','model_after.json','metric_runtime.json')},test_observations=300,train_diagnostic_observations=len(diagnostics),formal_updates=0,Adam_calls=0,backward_calls=0,model_RNG_unchanged=True));return summary


def parser():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=('register','evaluate'),required=True);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--protocol',type=Path,required=True);p.add_argument('--checkpoint',type=Path);p.add_argument('--out',type=Path);p.add_argument('--label');p.add_argument('--gpu-uuid');p.add_argument('--lock',type=Path);p.add_argument('--operator-resource-resolved',action='store_true');p.add_argument('--upstream',type=Path);p.add_argument('--cpu-threads',type=int,default=4);p.add_argument('--roi',type=Path);p.add_argument('--teacher-index',type=Path);p.add_argument('--train-keys',type=Path,help='JSON list of fixed legal [camera,frame] diagnostics');p.add_argument('--selection',type=Path);return p


def main():
    p=parser();a=p.parse_args()
    if a.mode=='register':return register(a.manifest,a.protocol,a.roi,a.teacher_index,read(a.train_keys) if a.train_keys else None,a.selection)
    if not all((a.checkpoint,a.out,a.label,a.gpu_uuid,a.operator_resource_resolved)):p.error('evaluate requires explicit checkpoint/out/label/GPU UUID/operator resource resolution')
    lockpath=local(a.lock or OUT/'locks'/('full_evaluation_'+a.gpu_uuid+'.lock'));lockpath.parent.mkdir(parents=True,exist_ok=True)
    with lockpath.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:return evaluate(a)
        except BaseException as e:
            target=OUT/'full_evaluation_failures';target.mkdir(parents=True,exist_ok=True);write(target/(str(time.time_ns())+'.json'),dict(status='failed_full_native_evaluation_saved_no_success_claim',attempted_out=str(local(a.out)),checkpoint=str(local(a.checkpoint)),protocol=str(local(a.protocol)),error=repr(e),traceback=traceback.format_exc(),formal_updates=0,Adam_calls=0));raise

if __name__=='__main__':main()
