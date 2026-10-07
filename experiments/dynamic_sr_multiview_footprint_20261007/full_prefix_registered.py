"""Registered full-time pure-LR native parent with explicit scene-domain protocols.

This entry does not import the short-window model, teacher, children or losses.
Plan/CPU checks need no CUDA. Training requires an explicitly allocated GPU UUID.
"""
from __future__ import annotations

import argparse
import ast
import copy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shlex
import socket
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

from fp_common import ROOT, HERE, OUT, read, write, sha, entry, local, bound
import full_native_protocol as domain

PIN = '843d5ac636c37e4b611242287754f3d4ed150144'
RNG_TRANSPORT_PREVIOUS_SOURCE = '269047a6e668d6ce298085c6378aaed05e454523328df7a0b7957392766f6855'
RNG_TRANSPORT_AMENDMENT = 'CUDA_RNG_transport_inputs_CPU_v1'
READINESS = OUT/'full_data_readiness/full_prefix_readiness.json'
LEGACY = ROOT/'experiments/dynamic_sr_20260918/n3dv_data.py'
RUNTIME_FILES = ('arguments/__init__.py', 'scene/gaussian_model.py', 'scene/deformation.py',
    'scene/hexplane.py', 'scene/grid.py', 'scene/regulation.py', 'scene/cameras.py',
    'gaussian_renderer/__init__.py', 'utils/general_utils.py', 'utils/graphics_utils.py',
    'utils/sh_utils.py', 'utils/system_utils.py', 'utils/loss_utils.py')
OPERATIONS = ('RGB', 'backward', 'Adam', 'densify', 'prune', 'grow', 'opacity_reset', 'iterations')
ADAPTATIONS = [
    'Pure-LR known-camera sparse points replace author supplied COLMAP points; each seed creates a fresh native GaussianModel and deformation network.',
    'Registered pixel-centre interfaces: N3DV HR1344x1008/LR336x252; MeetRoom native HR1280x720/LR320x180. Each immutable manifest selects its sole registered grid.',
    'MeetRoom discussion/VR use pinned Wu defaults plus generic dynerf/default.py batch4 as an explicit domain adaptation; no Cook batch2 override, StreamRF optimizer, or original Wu MeetRoom reproduction claim.',
    'Manifest train cameras include cam01; cam00 is held out and no test, dev, HR or teacher pixels are read by this entry.',
    'Explicit resumable torch shuffle draws replace 16-worker prefetch; no image augmentation exists. Torch DataLoader base-seed and RandomSampler-seed draws are preserved, including the author extra previous-batch reuse on epoch exhaustion.',
    'GUI, visualization and held-out quality reports are omitted. Nonfinite execution fails with costs preserved instead of author os.execv restarting.',
    'Atomic full optimizer/RNG/sampler/topology checkpoints and dispatched/completed operation journals supplement the author capture/restore interface.',
    'Native author Gaussian topology is used; this parent has no historical ordinary_split children or second child optimizer. A later SR suffix must explicitly adapt this parent.',
]


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()


def pure(node):
    if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='dict' and not node.args:
        return {k.arg:pure(k.value) for k in node.keywords}
    return ast.literal_eval(node)


def author_configuration(scene, readiness=READINESS):
    return domain.configuration(scene,readiness)


def validate_manifest(path, seed, readiness=READINESS):
    return domain.validate_manifest(path,seed,readiness)


def planned_counts(config):
    o=config['OptimizationParams'];coarse=o['coarse_iterations'];fine=o['iterations'];batch=o['batch_size']
    return dict(coarse_iterations=coarse,fine_iterations=fine,total_iterations=coarse+fine,
        coarse_Adam_calls=sum(i<fine for i in range(1,coarse+1)),fine_Adam_calls=fine-1,
        total_Adam_calls=coarse+fine-1,RGB_forwards=(coarse+fine)*batch,backward_calls=coarse+fine,
        batch_size=batch,stage_optimizer_resets=2,final_fine_Adam_skipped=True)


def build_plan(manifest,seed,readiness=READINESS,upstream=None):
    started=time.monotonic();m,identity,inputs=validate_manifest(manifest,seed,readiness)
    config,sources=author_configuration(m['scene'],readiness)
    upstream=Path(upstream or os.environ.get('FOURDSR_UPSTREAM','/home/cai_tianshun/Project/4dgs'))
    runtime={name:sha(upstream/name) for name in RUNTIME_FILES}
    # Core native representation/rendering must equal the recorded author code.
    for name in ('arguments/__init__.py','scene/gaussian_model.py','scene/deformation.py','scene/hexplane.py','gaussian_renderer/__init__.py'):
        if runtime[name]!=sha(sources[name]):raise ValueError('Core author runtime changed: '+name)
    return dict(status='registered_CPU_only_independent_full_LR_parent_pending_GPU_acceptance',seed=seed,scene=m['scene'],
        schema='registered_native_full_LR_prefix_v1',data=identity,author_commit=PIN,configuration=config,
        author_sources={name:entry(path) for name,path in sources.items()},runtime_source_sha256=runtime,
        project_sources={str(p.relative_to(ROOT)):sha(p) for p in (Path(__file__),HERE/'full_native_protocol.py',HERE/'full_prefix.py',HERE/'fp_common.py',LEGACY)},
        readiness=entry(local(readiness)),observation='native_lr',white_background=True,
        fresh_seed_model=True,representation='native_author_GaussianModel_no_children',
        author_adaptations=ADAPTATIONS,planned=planned_counts(config),teacher_used=False,HR_used=False,
        heldout_pixels_used=False,CUDA_initialized=False,GPU_forwards=0,Adam_calls=0,CPU_seconds=time.monotonic()-started),inputs


class AuthorShuffle:
    """Serializable equivalent of unaugmented shuffle DataLoader batch indices.

    Iterator base_seed and RandomSampler seed consume global CPU torch RNG.
    At exhaustion, only a new iterator base_seed is drawn; the previous batch
    is returned once. The next call draws its sampler seed and permutation.
    """
    def __init__(self,length,batch_size,torch_module,state=None):
        self.torch=torch_module;self.length=int(length);self.batch_size=int(batch_size)
        if state is None:
            self.order=None;self.cursor=0;self.previous=None;self.epoch=0;self.repeated=0
            self._iterator_seed()
        else:
            if (state['length'],state['batch_size'])!=(self.length,self.batch_size):raise ValueError('Sampler population changed')
            self.order=state['order'];self.cursor=state['cursor'];self.previous=state['previous'];self.epoch=state['epoch'];self.repeated=state['repeated']
            if self.order is not None and (len(self.order)!=self.length or set(self.order)!=set(range(self.length))):raise ValueError('Corrupt shuffle permutation')
            if not 0<=self.cursor<=self.length:raise ValueError('Corrupt shuffle cursor')

    def _iterator_seed(self):
        self.torch.empty((),dtype=self.torch.int64).random_().item()

    def next(self):
        if self.order is not None and self.cursor>=self.length:
            self._iterator_seed();self.order=None;self.cursor=0;self.epoch+=1;self.repeated+=1
            return list(self.previous)
        if self.order is None:
            generator=self.torch.Generator()
            generator.manual_seed(int(self.torch.empty((),dtype=self.torch.int64).random_().item()))
            self.order=self.torch.randperm(self.length,generator=generator).tolist()
        batch=self.order[self.cursor:self.cursor+self.batch_size];self.cursor+=len(batch);self.previous=list(batch)
        return batch

    def state_dict(self):
        return dict(length=self.length,batch_size=self.batch_size,order=self.order,cursor=self.cursor,
            previous=self.previous,epoch=self.epoch,repeated=self.repeated)


def parameter_map(g):
    values={name:getattr(g,name) for name in ('_xyz','_features_dc','_features_rest','_scaling','_rotation','_opacity')}
    values.update({'deformation.'+name:p for name,p in g._deformation.named_parameters()})
    return values


def rng_state(torch_module,numpy_module):
    return dict(python=random.getstate(),numpy=numpy_module.random.get_state(),torch=torch_module.get_rng_state(),
        cuda=torch_module.cuda.get_rng_state_all() if torch_module.cuda.is_initialized() else [])


def restore_rng(value,torch_module,numpy_module):
    random.setstate(value['python']);numpy_module.random.set_state(value['numpy']);torch_module.set_rng_state(value['torch'].cpu())
    # map_location='cuda' correctly restores model/Adam tensors on CUDA, but
    # generator.set_state requires its serialized ByteTensor input on CPU.
    if value['cuda']:torch_module.cuda.set_rng_state_all([state.cpu() for state in value['cuda']])


def rng_transport_source_contract(old_source):
    """Prove the RNG edit and unchanged author computation against pinned source."""
    old_source=bound(old_source) if isinstance(old_source,dict) else local(old_source)
    if sha(old_source)!=RNG_TRANSPORT_PREVIOUS_SOURCE:raise ValueError('RNG amendment requires the exact failed source version')
    old=ast.parse(old_source.read_text());new=ast.parse(Path(__file__).read_text())
    old_nodes={n.name:n for n in old.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    new_nodes={n.name:n for n in new.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    unchanged={name:ast.dump(node,include_attributes=False) for name,node in old_nodes.items()
               if name not in ('restore_rng','save_checkpoint','train','main')}
    for name,value in unchanged.items():
        if name not in new_nodes or ast.dump(new_nodes[name],include_attributes=False)!=value:
            raise ValueError('Author computation or restore protocol changed: '+name)
    expected=copy.deepcopy(old_nodes['restore_rng'])
    call=expected.body[-1].body[0].value
    if ast.unparse(call)!='torch_module.cuda.set_rng_state_all(value[\'cuda\'])':
        raise ValueError('Unexpected old RNG implementation')
    call.args=[ast.parse("[state.cpu() for state in value['cuda']]",mode='eval').body]
    if ast.dump(expected,include_attributes=False)!=ast.dump(new_nodes['restore_rng'],include_attributes=False):
        raise ValueError('RNG edit must only transport the CUDA state inputs to CPU')
    constants=lambda tree:{node.targets[0].id:ast.dump(node,include_attributes=False) for node in tree.body
        if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name)}
    old_constants,new_constants=constants(old),constants(new)
    if any(new_constants.get(name)!=value for name,value in old_constants.items()):
        raise ValueError('Original module algorithm/operation constants changed')
    # The complete author execution try/except, including initialization,
    # LR reads, sampling, topology, Adam and failures, is unchanged. Only the
    # final completion receipt adds the explicit execution-source lineage.
    old_try=next(node for node in old_nodes['train'].body if isinstance(node,ast.Try))
    new_try=copy.deepcopy(next(node for node in new_nodes['train'].body if isinstance(node,ast.Try)))
    for node in ast.walk(new_try):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='dict':
            node.keywords=[key for key in node.keywords if key.arg not in ('execution_source','resume_amendment')]
    if ast.dump(old_try,include_attributes=False)!=ast.dump(new_try,include_attributes=False):
        raise ValueError('Author initialization/training/control trajectory changed')
    saved=copy.deepcopy(new_nodes['save_checkpoint'])
    saved.body=[node for node in saved.body if not (isinstance(node,ast.If) and
        ast.unparse(node.test)=="journal.value.get('resume_amendment')")]
    if ast.dump(saved,include_attributes=False)!=ast.dump(old_nodes['save_checkpoint'],include_attributes=False):
        raise ValueError('Checkpoint capture changed beyond explicit lineage metadata')
    return dict(old_source_sha256=sha(old_source),unchanged_computation_functions_and_classes=sorted(unchanged),
        CUDA_RNG_only_change='set_rng_state_all([state.cpu() for state in saved_cuda_states])',
        model_Adam_map_location_unchanged=True,restore_checkpoint_and_latest_checkpoint_unchanged=True,
        original_module_constants_unchanged=True,whole_author_training_and_failure_body_unchanged=True,
        checkpoint_capture_unchanged_except_explicit_execution_lineage=True)


def validate_resume_amendment(path,registered,requested,out):
    """Accept one declared source leaf edit; never relax data/algorithm identity."""
    path=local(path);value=read(path)
    if value.get('status')!='registered_precise_RNG_CPU_transport_compatible_resume' or value.get('kind')!=RNG_TRANSPORT_AMENDMENT:
        raise ValueError('Not the explicit RNG transport resume amendment')
    if value['original_registered_plan']!=entry(Path(out)/'config.json') or read(bound(value['old_plan_source']))!=registered:
        raise ValueError('Original registration/plan source differs from this run')
    if value['execution_source']!=entry(Path(__file__)):raise ValueError('Amendment execution source changed')
    if digest(registered)!=value['registered_plan_sha256'] or digest(requested)!=value['requested_plan_sha256']:
        raise ValueError('Amendment plan identity changed')
    original=bound(value['parent_checkpoint']);sidecar=read(bound(value['parent_sidecar']))
    if entry(original)!={key:sidecar[key] for key in ('path','sha256')} or sidecar['metadata']['plan_sha256']!=digest(registered):
        raise ValueError('Original committed checkpoint identity differs')
    if sidecar['metadata']['state']!=value['parent_state']:raise ValueError('Original committed model/sampler position changed')
    expected=copy.deepcopy(registered);key=str(Path(__file__).relative_to(ROOT))
    if expected['project_sources'].get(key)!=RNG_TRANSPORT_PREVIOUS_SOURCE:raise ValueError('Unregistered previous source')
    expected['project_sources'][key]=sha(__file__)
    if expected!=requested:raise ValueError('Only the full_prefix.py RNG restore source leaf may change')
    contract=rng_transport_source_contract(value['old_source'])
    if contract!=value['source_contract']:raise ValueError('RNG-only source contract changed')
    for item in value['incident_evidence']:bound(item)
    for checkpoint in Path(out).glob('checkpoint_*.json'):
        metadata=read(checkpoint)['metadata']
        if metadata['completed_iterations']>sidecar['metadata']['completed_iterations'] and metadata.get('resume_amendment')!=entry(path):
            raise ValueError('Later checkpoint lacks this explicit compatible-resume lineage')
    return entry(path)


def create_resume_amendment(a,requested):
    """CPU-only enrollment; immutable old config/checkpoint stay byte-identical."""
    if not a.old_source or not a.old_plan or not a.incident_evidence or not a.resume or a.resume=='latest':
        raise ValueError('Amendment requires exact --old-source --old-plan --resume CHECKPOINT and incident evidence')
    out=a.out;registered=read(out/'config.json');requested={k:v for k,v in requested.items() if k!='CPU_seconds'}
    old_plan=local(a.old_plan)
    if read(old_plan)!=registered:raise ValueError('Archived old plan is not the original registered plan')
    parent=local(a.resume);sidecar=parent.with_suffix('.json');bound(read(sidecar))
    if parent.parent.resolve()!=out.resolve() or parent.name!='checkpoint_coarse_00100.pt':
        raise ValueError('This precise amendment binds the original 100-step acceptance checkpoint only')
    state=read(sidecar)['metadata']['state']
    if (state['stage'],state['coarse_iteration'],state['fine_iteration'],state['accepted_RGB'],state['accepted_backward'],state['accepted_Adam'])!=('coarse',100,0,200,100,100):
        raise ValueError('Not the original accepted coarse100 state')
    expected=copy.deepcopy(registered);key=str(Path(__file__).relative_to(ROOT))
    if expected['project_sources'].get(key)!=RNG_TRANSPORT_PREVIOUS_SOURCE:
        raise ValueError('Original registration does not bind the precise failed source')
    expected['project_sources'][key]=sha(__file__)
    if expected!=requested or read(sidecar)['metadata']['plan_sha256']!=digest(registered):
        raise ValueError('Cannot enroll an algorithm/data/other-source change as an RNG transport repair')
    value=dict(status='registered_precise_RNG_CPU_transport_compatible_resume',kind=RNG_TRANSPORT_AMENDMENT,
        original_registered_plan=entry(out/'config.json'),old_plan_source=entry(old_plan),
        registered_plan_sha256=digest(registered),requested_plan_sha256=digest(requested),
        old_source=entry(local(a.old_source)),execution_source=entry(Path(__file__)),
        parent_checkpoint=entry(parent),parent_sidecar=entry(sidecar),parent_state=state,
        source_contract=rng_transport_source_contract(a.old_source),
        incident_evidence=[entry(local(path)) for path in a.incident_evidence],
        registration_kept_byte_identical=True,original_checkpoint_kept_byte_identical=True,
        model_Adam_sampler_gradients_topology_and_rng_inherited=True,
        old_plan_is_original_registration_new_execution_source_is_explicit=True,
        learning_rule_change=False,cold_restart=False,formal_updates=0,GPU_calls=0)
    path=local(a.amendment_out or out/'resume_amendments'/f'RNG_CPU_transport_{sha(__file__)[:16]}.json')
    if path.exists() and read(path)!=value:raise ValueError('Existing amendment differs; preserve it')
    if not path.exists():write(path,value)
    validate_resume_amendment(path,registered,requested,out)
    return path


def combine_batch_statistics(packages,torch_module):
    radii=torch_module.stack([p['radii'] for p in packages]).max(dim=0).values
    visible=torch_module.stack([p['visibility_filter'] for p in packages]).any(dim=0)
    gradients=[p['viewspace_points'].grad for p in packages]
    if any(value is None for value in gradients):raise RuntimeError('Native viewspace gradient missing')
    summed=torch_module.stack(gradients).sum(dim=0)
    if not bool(torch_module.isfinite(summed).all()):raise FloatingPointError('Nonfinite summed viewspace gradient')
    return radii,visible,summed


def operation(ledger,name,call):
    if ledger:ledger.begin(name)
    result=call()
    if ledger:ledger.end(name)
    return result


def author_topology(g,opt,stage,iteration,extent,out,statistics,ledger=None):
    radii,visible,summed=statistics
    if iteration>=opt.densify_until_iter:return
    g.max_radii2D[visible]=g.max_radii2D[visible].maximum(radii[visible])
    g.add_densification_stats(summed,visible)
    if stage=='coarse':opacity_threshold=opt.opacity_threshold_coarse;threshold=opt.densify_grad_threshold_coarse
    else:
        opacity_threshold=opt.opacity_threshold_fine_init-iteration*(opt.opacity_threshold_fine_init-opt.opacity_threshold_fine_after)/opt.densify_until_iter
        threshold=opt.densify_grad_threshold_fine_init-iteration*(opt.densify_grad_threshold_fine_init-opt.densify_grad_threshold_after)/opt.densify_until_iter
    size=20 if iteration>opt.opacity_reset_interval else None
    if iteration>opt.densify_from_iter and iteration%opt.densification_interval==0 and g.get_xyz.shape[0]<360000:
        operation(ledger,'densify',lambda:g.densify(threshold,opacity_threshold,extent,size,5,5,str(out),iteration,stage))
    if iteration>opt.pruning_from_iter and iteration%opt.pruning_interval==0 and g.get_xyz.shape[0]>200000:
        operation(ledger,'prune',lambda:g.prune(threshold,opacity_threshold,extent,size))
    if iteration%opt.densification_interval==0 and g.get_xyz.shape[0]<360000 and opt.add_point:
        operation(ledger,'grow',lambda:g.grow(5,5,str(out),iteration,stage))
    if iteration%opt.opacity_reset_interval==0:operation(ledger,'opacity_reset',g.reset_opacity)


def author_iteration(g,h,opt,pipe,background,cameras,render,stage,iteration,extent,out,torch_module,ledger=None):
    """The author batch/loss/backward/topology/Adam order, with explicit costs."""
    g.update_learning_rate(iteration)
    if iteration%1000==0:g.oneupSHdegree()
    packages=[];images=[];targets=[]
    for camera in cameras:
        packet=operation(ledger,'RGB',lambda:render(camera,g,pipe,background,stage=stage,cam_type='dynerf'))
        packages.append(packet);images.append(packet['render'].unsqueeze(0));targets.append(camera.original_image.to(background.device).unsqueeze(0))
    prediction=torch_module.cat(images,dim=0);target=torch_module.cat(targets,dim=0)
    image_loss=(prediction-target[:,:3]).abs().mean();loss=image_loss
    if stage=='fine' and h.time_smoothness_weight!=0:
        loss=loss+g.compute_regulation(h.time_smoothness_weight,h.l1_time_planes,h.plane_tv_weight)
    if opt.lambda_dssim!=0:
        from utils.loss_utils import ssim
        loss=loss+opt.lambda_dssim*(1-ssim(prediction,target))
    if not bool(torch_module.isfinite(loss)):raise FloatingPointError('Nonfinite author batch loss')
    operation(ledger,'backward',loss.backward)
    finite=[torch_module.isfinite(p.grad).all() for p in parameter_map(g).values() if p.grad is not None]
    if finite and not bool(torch_module.stack(finite).all()):raise FloatingPointError('Nonfinite model gradient')
    statistics=combine_batch_statistics(packages,torch_module)
    with torch_module.no_grad():
        author_topology(g,opt,stage,iteration,extent,out,statistics,ledger)
        # This compares against opt.iterations even during coarse, as in 843d5.
        if iteration<opt.iterations:
            operation(ledger,'Adam',g.optimizer.step)
            g.optimizer.zero_grad(set_to_none=True)
    if ledger:ledger.completed['iterations']+=1;ledger.persist()
    return dict(loss=float(loss.detach()),L1=float(image_loss.detach()),stage=stage,iteration=iteration,
        Adam_applied=iteration<opt.iterations,points=int(g.get_xyz.shape[0]),batch_size=len(cameras))


def topology_audit(g):
    n=int(g.get_xyz.shape[0]);parameters=parameter_map(g);optimizer=g.optimizer
    ids=[id(p) for group in optimizer.param_groups for p in group['params']]
    if len(ids)!=len(set(ids)) or set(ids)!=set(id(p) for p in parameters.values()):raise ValueError('Optimizer/native parameter mapping differs')
    for name in ('_deformation_table','max_radii2D','xyz_gradient_accum','denom','_deformation_accum'):
        if getattr(g,name).shape[0]!=n:raise ValueError('Topology buffer count differs: '+name)
    for group in optimizer.param_groups:
        for p in group['params']:
            state=optimizer.state.get(p,{})
            for key in ('exp_avg','exp_avg_sq'):
                if key in state and state[key].shape!=p.shape:raise ValueError('Adam topology mapping differs')
    return dict(points=n,active_sh_degree=g.active_sh_degree,parameter_count=sum(p.numel() for p in parameters.values()),
        optimizer_groups=[dict(name=gr['name'],lr=float(gr['lr']),parameters=len(gr['params'])) for gr in optimizer.param_groups],
        buffers={name:list(getattr(g,name).shape) for name in ('_deformation_table','max_radii2D','xyz_gradient_accum','denom','_deformation_accum')})


class Journal:
    def __init__(self,out,identity,stage,iteration):
        self.path=Path(out)/'attempts'/f'{time.time_ns()}_{os.getpid()}.json'
        self.started=time.monotonic();self.completed={key:0 for key in OPERATIONS};self.attempted={key:0 for key in OPERATIONS}
        self.value=dict(status='running',identity=identity,host=socket.gethostname(),pid=os.getpid(),started_unix=time.time(),
            start_stage=stage,start_iteration=iteration,stage=stage,iteration=iteration,active_operation=None)
        self.persist()

    def persist(self):
        write(self.path,dict(self.value,completed=self.completed,attempted=self.attempted,seconds=time.monotonic()-self.started))

    def begin(self,name):
        self.attempted[name]+=1;self.value['active_operation']=name;self.persist()

    def end(self,name):
        self.completed[name]+=1;self.value['active_operation']=None;self.persist()

    def finish(self,status,**extra):
        self.value.update(status=status,**extra);self.persist()


def save_checkpoint(path,g,h,opt,state,plan,sampler,torch_module,numpy_module,journal):
    path=Path(path)
    if path.exists() or path.with_suffix('.json').exists():raise FileExistsError('Existing checkpoint preserved: '+str(path))
    path.parent.mkdir(parents=True,exist_ok=True)
    audit=topology_audit(g)
    metadata=dict(state=copy.deepcopy(state),plan_sha256=digest(plan),audit=audit,attempt=str(journal.path.relative_to(ROOT)),
        attempt_counters_at_save=dict(journal.completed),completed_iterations=state['coarse_iteration']+state['fine_iteration'],
        stage=state['stage'],iteration=state[state['stage']+'_iteration'],representation='native_author_GaussianModel_no_children')
    if journal.value.get('resume_amendment'):
        metadata.update(resume_amendment=journal.value['resume_amendment'],execution_source=journal.value['execution_source'])
    payload=dict(schema='full_author_LR_prefix_checkpoint_v1',model=g.capture(),hidden=vars(h),optim=vars(opt),
        metadata=metadata,sampler=sampler.state_dict(),rng=rng_state(torch_module,numpy_module),
        deformation_accum=g._deformation_accum,parameter_gradients={name:p.grad for name,p in parameter_map(g).items()},
        plan=plan)
    temporary=path.with_name(path.name+f'.{os.getpid()}.tmp')
    torch_module.save(payload,temporary)
    with temporary.open('rb') as handle:os.fsync(handle.fileno())
    temporary.replace(path)
    write(path.with_suffix('.json'),dict(**entry(path),schema=payload['schema'],metadata=metadata))
    return path


def restore_checkpoint(path,g,opt,plan,torch_module,numpy_module):
    path=Path(path);sidecar=read(path.with_suffix('.json'))
    if entry(path)!={key:sidecar[key] for key in ('path','sha256')}:raise ValueError('Checkpoint hash changed')
    payload=torch_module.load(path,map_location='cuda' if torch_module.cuda.is_initialized() else 'cpu',weights_only=False)
    if payload['schema']!='full_author_LR_prefix_checkpoint_v1' or payload['plan']!=plan or payload['metadata']!=sidecar['metadata']:
        raise ValueError('Checkpoint protocol/source metadata differs')
    g.restore(payload['model'],opt)
    g._deformation_accum=payload['deformation_accum']
    params=parameter_map(g)
    if set(params)!=set(payload['parameter_gradients']):raise ValueError('Checkpoint gradient coordinate mismatch')
    for name,p in params.items():p.grad=payload['parameter_gradients'][name]
    if topology_audit(g)!=payload['metadata']['audit']:raise ValueError('Restored Adam/LR/topology audit differs')
    sampler=AuthorShuffle(payload['sampler']['length'],payload['sampler']['batch_size'],torch_module,payload['sampler'])
    restore_rng(payload['rng'],torch_module,numpy_module)
    return payload['metadata']['state'],sampler,payload


def latest_checkpoint(out,torch_module,plan):
    """Recover only a complete atomic payload whose sidecar publication crashed."""
    candidates=[]
    for path in sorted(Path(out).glob('checkpoint_*.pt')):
        sidecar=path.with_suffix('.json')
        if not sidecar.exists():
            payload=torch_module.load(path,map_location='cpu',weights_only=False)
            if payload.get('schema')!='full_author_LR_prefix_checkpoint_v1' or payload['plan']!=plan:
                raise ValueError('Orphan checkpoint is not this complete prefix')
            write(sidecar,dict(**entry(path),schema=payload['schema'],metadata=payload['metadata']))
            write(Path(out)/'recovery'/f'orphan_sidecar_{time.time_ns()}.json',dict(status='recovered_complete_atomic_payload_sidecar',checkpoint=entry(path)))
        value=read(sidecar);bound(value)
        candidates.append((value['metadata']['completed_iterations'],value['metadata']['state']['optimizer_resets'],path))
    if not candidates:raise ValueError('No durable checkpoint; preserve failed initialization and use an explicit new run directory')
    return max(candidates)[2]


def historical_cost(out):
    attempts=[read(path) for path in sorted((Path(out)/'attempts').glob('*.json'))]
    return dict(attempts=len(attempts),completed={key:sum(a['completed'][key] for a in attempts) for key in OPERATIONS},
        dispatched={key:sum(a['attempted'][key] for a in attempts) for key in OPERATIONS},
        seconds=sum(a['seconds'] for a in attempts),unresolved_active_operations=[dict(pid=a['pid'],stage=a['stage'],iteration=a['iteration'],operation=a['active_operation'])
            for a in attempts if a.get('active_operation')],all_attempt_records=[entry(path) for path in sorted((Path(out)/'attempts').glob('*.json'))])


def check_gpu(uuid):
    if not uuid or not uuid.startswith('GPU-') or os.environ.get('CUDA_VISIBLE_DEVICES')!=uuid:
        raise ValueError('Root must explicitly bind CUDA_VISIBLE_DEVICES to the allocated physical GPU UUID')
    output=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv,noheader'],text=True)
    foreign=[r for r in output.splitlines() if r.startswith(uuid+',') and int(r.split(',')[1])!=os.getpid()]
    if foreign:raise RuntimeError('Allocated GPU occupied; existing jobs protected: '+'; '.join(foreign))
    inventory=subprocess.check_output(['nvidia-smi','-i',uuid,'--query-gpu=uuid,name,memory.total,memory.used,driver_version','--format=csv,noheader'],text=True).strip()
    if '5090' in inventory:raise ValueError('The project forbids training on RTX 5090')
    return inventory


def train(a,plan,inputs):
    plan={key:value for key,value in plan.items() if key!='CPU_seconds'}
    domain.validate_plan(plan);domain.forbid_completed_training(plan['scene'],plan['seed'])
    if (a.out/'complete.json').exists():raise ValueError('Closed native prefix preserved; do not rerun completed result')
    out=a.out;out.mkdir(parents=True,exist_ok=True)
    config_path=out/'config.json'
    amendment=None
    if config_path.exists():
        registered=read(config_path)
        if registered!=plan:raise ValueError('Registered domain/source/data differs; this independent revision cannot amend older prefix trajectories')
        if not a.resume:raise ValueError('Existing run requires explicit --resume latest or the latest checkpoint')
    else:
        if a.resume:raise ValueError('Resume requires an existing run registration')
        if any(path.name!='hardware.json' for path in out.iterdir()):raise ValueError('Unregistered existing output directory preserved')
        write(config_path,plan)
    import numpy as np
    import torch
    upstream=Path(a.upstream or os.environ.get('FOURDSR_UPSTREAM','/home/cai_tianshun/Project/4dgs'))
    sys.path.insert(0,str(upstream));sys.path.insert(1,str(LEGACY.parent))
    from scene.gaussian_model import GaussianModel
    from gaussian_renderer import render
    from utils.graphics_utils import BasicPointCloud
    from n3dv_data import load_manifest,load_initial_points,N3DVPreparedDataset,observation_to_4dgs_camera
    h=SimpleNamespace(**plan['configuration']['ModelHiddenParams']);opt=SimpleNamespace(**plan['configuration']['OptimizationParams'])
    pipe=SimpleNamespace(**plan['configuration']['PipelineParams'])
    manifest=load_manifest(a.manifest);dataset=N3DVPreparedDataset(manifest,'train','lr',device='cpu',cache=False)
    legal={(ROOT/i['path']).resolve():i['sha256'] for i in inputs}
    points_path=bound(plan['data']['initial_points']).resolve();reads=dict(LR=0,unique=set())
    def guard(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        path=Path(os.fsdecode(args[0]));suffix=path.suffix.lower()
        if suffix not in ('.png','.jpg','.jpeg','.npy','.npz'):return
        path=path.resolve()
        if path==points_path:return
        if path not in legal:raise AssertionError('Prefix image/array read outside legal LR: '+str(path))
        reads['LR']+=1;reads['unique'].add(path)
    sys.addaudithook(guard)
    # Author seed0 is deliberately replaced by the registered independent seed.
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed)
    torch.set_num_threads(a.cpu_threads)
    journal=Journal(out,digest(plan),'coarse',0)
    if amendment:journal.value.update(resume_amendment=amendment,execution_source=entry(Path(__file__)),original_registered_plan=entry(config_path));journal.persist()
    try:
        g=GaussianModel(plan['configuration']['ModelParams']['sh_degree'],h)
        if a.resume:
            path=latest_checkpoint(out,torch,plan)
            if a.resume!='latest' and local(a.resume).resolve()!=path.resolve():raise ValueError('Only latest checkpoint may resume; older rollbacks require explicit incident handling')
            saved=read(path.with_suffix('.json'))['metadata']['state']
            if a.stop_iteration is not None:
                stop_order=(0 if a.stop_stage=='coarse' else 1,a.stop_iteration)
                saved_order=(0 if saved['stage']=='coarse' else 1,saved[saved['stage']+'_iteration'])
                if stop_order<saved_order:raise ValueError('Requested stop precedes the latest committed checkpoint')
            g._deformation=g._deformation.cuda()
            state,sampler,payload=restore_checkpoint(path,g,opt,plan,torch,np)
            journal.value.update(start_stage=state['stage'],start_iteration=state[state['stage']+'_iteration'],resume=entry(path));journal.persist()
        else:
            initial=load_initial_points(manifest);xyz=initial['points'];colors=initial['colors']
            if xyz.shape[1]!=3 or not np.isfinite(xyz).all() or not np.isfinite(colors).all():raise ValueError('Invalid pure-LR initializer')
            # Match original Scene's point bounding box, without pilot's added margin.
            g._deformation.deformation_net.set_aabb(xyz.max(axis=0),xyz.min(axis=0))
            g.create_from_pcd(BasicPointCloud(xyz,colors,np.zeros_like(xyz)),plan['data']['cameras_extent'],300)
            g.training_setup(opt)
            sampler=AuthorShuffle(len(dataset),opt.batch_size,torch)
            state=dict(stage='coarse',coarse_iteration=0,fine_iteration=0,optimizer_resets=1,
                accepted_RGB=0,accepted_backward=0,accepted_Adam=0)
            save_checkpoint(out/'checkpoint_coarse_00000.pt',g,h,opt,state,plan,sampler,torch,np,journal)
        background=torch.ones(3,dtype=torch.float32,device='cuda')
        with (out/'training.jsonl').open('a',buffering=1) as log:
            for stage in ('coarse','fine'):
                if stage=='coarse' and state['stage']=='fine':continue
                if stage!=state['stage']:
                    g.training_setup(opt);sampler=AuthorShuffle(len(dataset),opt.batch_size,torch)
                    state.update(stage=stage,optimizer_resets=state['optimizer_resets']+1)
                    save_checkpoint(out/'checkpoint_fine_00000.pt',g,h,opt,state,plan,sampler,torch,np,journal)
                maximum=opt.coarse_iterations if stage=='coarse' else opt.iterations
                for iteration in range(state[stage+'_iteration']+1,maximum+1):
                    if a.stop_stage==stage and a.stop_iteration is not None and iteration>a.stop_iteration:break
                    journal.value.update(stage=stage,iteration=iteration);journal.attempted['iterations']+=1;journal.persist();batch=sampler.next();cameras=[]
                    for index in batch:
                        observation=dataset.observations[index];path=(Path(manifest['_root'])/observation['lr_path']).resolve()
                        if sha(path)!=legal[path]:raise ValueError('Registered LR source changed')
                        item=dataset[index];camera=observation_to_4dgs_camera(item,index)
                        if tuple(camera.original_image.shape)!=(3,*reversed(plan['data']['resolution_LR'])):raise ValueError('Native LR camera shape differs')
                        cameras.append(camera)
                    row=author_iteration(g,h,opt,pipe,background,cameras,render,stage,iteration,
                        plan['data']['cameras_extent'],out,torch,journal)
                    state[stage+'_iteration']=iteration;state['accepted_RGB']+=len(batch);state['accepted_backward']+=1
                    state['accepted_Adam']+=int(iteration<opt.iterations)
                    row.update(batch_indices=batch,sampler_epoch=sampler.epoch,sampler_previous_batch_repeats=sampler.repeated,
                        attempt=journal.path.name,cost_completed=dict(journal.completed));log.write(json.dumps(row,allow_nan=False)+'\n')
                    if iteration%a.checkpoint_interval==0 or iteration==maximum or (a.stop_stage==stage and iteration==a.stop_iteration):
                        save_checkpoint(out/f'checkpoint_{stage}_{iteration:05d}.pt',g,h,opt,state,plan,sampler,torch,np,journal)
                if a.stop_stage==stage and a.stop_iteration is not None and not (stage=='fine' and state['fine_iteration']==14000):
                    journal.finish('paused_at_registered_iteration',state=state)
                    write(out/'status.json',dict(status='paused_prefix_incomplete',state=state,cost=historical_cost(out),latest=entry(latest_checkpoint(out,torch,plan))))
                    return
        domain.validate_parent_state(state,plan['configuration'])
        torch.cuda.synchronize();journal.finish('completed_full_author_LR_prefix',state=state)
        path=latest_checkpoint(out,torch,plan)
        write(out/'complete.json',dict(status='completed_independent_full_author_LR_prefix',checkpoint=entry(path),plan=entry(config_path),
            execution_source=entry(Path(__file__)),resume_amendment=amendment,
            seed=a.seed,scene=plan['scene'],state=state,topology=topology_audit(g),cost_all_attempts=historical_cost(out),
            legal_LR_file_opens=reads['LR'],unique_LR_files_read_this_process=len(reads['unique']),HR_used=False,
            heldout_pixels_used=False,teacher_used=False,representation=plan['representation'],author_adaptations=ADAPTATIONS,
            hardware=read(out/'hardware.json'),torch=str(torch.__version__),cuda=torch.version.cuda,
            native_checkpoint_gradient_RNG_Adam_roundtrip_acceptance='CPU-controlled acceptance only; GPU fixture must be separately recorded by root',
            note='This is the native LR parent. Full SR teacher/support/suffix adapters and uniform evaluation remain separate tasks.'))
    except BaseException:
        journal.finish('failed_execution_preserved',error=traceback.format_exc())
        write(out/f'failure_{time.time_ns()}.json',dict(status='failed_prefix_preserved',attempt=entry(journal.path),
            cost_all_attempts=historical_cost(out),error=traceback.format_exc(),cold_restart=False))
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--mode',choices=('plan','train','launch-command'),default='plan');p.add_argument('--readiness',type=Path,default=READINESS)
    p.add_argument('--upstream',type=Path);p.add_argument('--out',type=Path);p.add_argument('--resume')
    p.add_argument('--gpu-uuid');p.add_argument('--lock',type=Path);p.add_argument('--operator-resource-resolved',action='store_true')
    p.add_argument('--checkpoint-interval',type=int,default=100);p.add_argument('--cpu-threads',type=int,default=4)
    p.add_argument('--stop-stage',choices=('coarse','fine'));p.add_argument('--stop-iteration',type=int)
    p.add_argument('--resume-amendment',type=Path,help='Explicit, SHA-bound RNG-only compatible resume; ordinary source/data guards remain strict')
    p.add_argument('--old-source',type=Path);p.add_argument('--old-plan',type=Path)
    p.add_argument('--incident-evidence',type=Path,action='append');p.add_argument('--amendment-out',type=Path)
    a=p.parse_args();a.manifest=local(a.manifest)
    registration_started=time.monotonic();plan,inputs=build_plan(a.manifest,a.seed,a.readiness,a.upstream)
    a.out=local(a.out or OUT/'full_LR_prefixes'/plan['scene']/f'seed_{a.seed}')
    if a.stop_iteration is not None:
        maximum=3000 if a.stop_stage=='coarse' else 14000
        if a.stop_stage is None or not 1<=a.stop_iteration<=maximum:p.error('Stop requires a stage and an iteration within its original schedule')
    elif a.stop_stage is not None:p.error('--stop-stage requires --stop-iteration')
    if a.checkpoint_interval<1 or a.cpu_threads<1:p.error('Positive checkpoint/thread settings required')
    if any((a.resume_amendment,a.old_source,a.old_plan,a.incident_evidence,a.amendment_out)):p.error('Old source-only RNG amendment is outside this registered domain revision; original accepted runs remain under full_prefix.py')
    if a.mode=='launch-command':
        args=[sys.executable,'-u',str(Path(__file__)), '--mode','train','--manifest',str(a.manifest),'--seed',str(a.seed),'--out',str(a.out),
              '--gpu-uuid',a.gpu_uuid or '<allocated-UUID>','--lock',str(a.lock or OUT/'locks/<allocated-UUID>.controller.lock'),'--operator-resource-resolved']
        if a.resume:args+=['--resume',a.resume]
        if a.resume_amendment:args+=['--resume-amendment',str(local(a.resume_amendment))]
        args+=['--checkpoint-interval',str(a.checkpoint_interval),'--cpu-threads',str(a.cpu_threads),
               '--readiness',str(a.readiness)]
        if a.upstream:args+=['--upstream',str(a.upstream)]
        if a.stop_iteration is not None:args+=['--stop-stage',a.stop_stage,'--stop-iteration',str(a.stop_iteration)]
        print('CUDA_VISIBLE_DEVICES='+shlex.quote(a.gpu_uuid or '<allocated-UUID>')+' '+shlex.join(args))
        print('Run this under an existing durable systemd/tmux service; this mode only prints the command and launches nothing.');return
    if a.mode=='plan':
        existing=domain.completed_prefix(plan['scene'],plan['seed'])
        if existing is not None:
            print(json.dumps(dict(status='existing_accepted_native_prefix_preserved_no_retraining',existing=existing,GPU_calls=0)));return
        path=OUT/'full_prefix_registered_plans'/plan['scene']/f"seed_{a.seed}"/'registration.json'
        plan={key:value for key,value in plan.items() if key!='CPU_seconds'}
        if path.exists() and read(path)!=plan:raise ValueError('Existing registered domain plan differs; preserve source revision')
        if path.exists():print(json.dumps(dict(status=plan['status'],plan=str(path),planned=plan['planned'],CPU_seconds=0.,registered_before=True)));return
        write(path,plan);print(json.dumps(dict(status=plan['status'],plan=str(path),planned=plan['planned'],CPU_seconds=time.monotonic()-registration_started)));return
    domain.forbid_completed_training(plan['scene'],plan['seed'])
    if not a.operator_resource_resolved or a.lock is None:p.error('Root resource resolution and a shared GPU --lock are required')
    a.lock.parent.mkdir(parents=True,exist_ok=True)
    with a.lock.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        hardware=check_gpu(a.gpu_uuid)
        a.out.mkdir(parents=True,exist_ok=True)
        hardware_path=a.out/'hardware.json'
        if hardware_path.exists():
            previous=read(hardware_path)
            if previous['host']!=socket.gethostname() or previous['gpu_uuid']!=a.gpu_uuid:raise ValueError('Independent prefix hardware changed')
        else:
            # train() permits this one controller-created receipt before config.
            write(hardware_path,dict(host=socket.gethostname(),gpu_uuid=a.gpu_uuid,inventory=hardware))
        train(a,plan,inputs)


if __name__=='__main__':main()
