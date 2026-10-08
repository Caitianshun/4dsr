"""Registered full-time SR adaptation of the pinned native Wu LR parent.

Version 2 reconciles the selector with the registered B0 baseline and Bsync candidate.

This is project SR refinement, not an author SR reproduction. No short-window
Gaussian children, second Adam, U6000 support, or held-out pixels are consumed.
CPU registration can be pending; training is impossible until every required
full-time dependency and an evidence-based candidate selection are frozen.
"""
from __future__ import annotations

import argparse
from collections import Counter, OrderedDict
import copy
import fcntl
import math
import os
from pathlib import Path
import socket
import sys
import time
import traceback
from types import SimpleNamespace

from fp_common import ROOT, HERE, OUT, local, read, write, sha, entry, bound, module
import full_prefix_registered as prefix
import full_native_protocol as domain
from config import ARMS, METHODS

STEPS = 6000
LR_OFFSET = 14000
SCHEMA = 'registered_full_native_SR_refinement_v1'
SCHEDULE_SCHEMA = 'full_native_SR_schedule_v1'
REPRESENTATION = 'native_author_GaussianModel_no_children'
OPERATIONS = prefix.OPERATIONS + ('moments', 'diagnostic_autograd')
ADAPTATIONS = [
    'Registered N3DV author scene or explicit MeetRoom generic domain configuration determines the parent grid and batch2/batch4 prefix RGB budget. Four accepted legacy Cook/Cut parent plans retain original bytes and accepted RNG amendment; no retraining or conversion.',
    'Independent native pure-LR coarse3000/fine14000 parent per seed; paired methods retain its native parameters, eight Adam groups, moments, topology buffers and global RNG.',
    'No conversion to ordinary_split, no children, no second Adam and no optimizer reset at the SR boundary. The unapplied final LR gradient is recorded then cleared before the new SR loss.',
    'Only max_radii2D, xyz_gradient_accum and denom are cleared once at the LR-to-HR SR boundary. Their LR pixel units/history cannot enter the new HR density statistics; model, Adam, global RNG, deformation accumulation and every other buffer are retained. Mid-SR resume does not repeat this boundary.',
    'Root-registered SR6000 topology clock restarts at 1 for every same-backbone arm. Original author fine thresholds, densify/prune/grow/opacity rules and topology-before-Adam order remain unchanged; this stage-clock restart is an explicit adaptation.',
    'Learning-rate and SH clock continue at fine14000+SRcursor. Every SR step applies one accumulated backward and one native Adam, including the final SR step.',
    'Three native HR RGB renders per step; same-time arms share one native deformation state. Max radii, any visibility and summed same-index RGB viewspace gradients feed the same author density rule in every arm.',
    'X adds three position-only raw HR moment renders at the shared effective state. RGB attributes remain trainable; moment covariance/opacity are detached. Six full-footprint edges use independently frozen full-parent support.',
    'The existing project L1 full-teacher supervision, true LR degradation and M/X/MX/E losses are adapted to full training cameras and all 0..299 frames. No claim of an unchanged author SR method or a completed benchmark is made.',
]


def safe_child(root, relative):
    root = Path(root).resolve(); path = (root/relative).resolve()
    if not path.is_relative_to(root): raise ValueError('Inventory path escaped its root')
    return path


def keys_of(manifest):
    return [(r['camera_id'], int(r['frame_index'])) for r in manifest['observations'] if r['split']=='train']


def validate_parent_plan(plan, complete=None):
    """Accept registered scene recipes or the exact four already accepted parents.

    A legacy plan keeps its original bytes and RNG compatibility amendment.
    A new plan binds the registered scene adapter and derives counts from its
    actual batch size. This function reads metadata and sources only.
    """
    descriptor = domain.descriptor(plan['scene'])
    configuration, sources = domain.configuration(plan['scene'])
    if plan['configuration'] != configuration or plan['author_commit'] != prefix.PIN or plan['representation'] != REPRESENTATION:
        raise ValueError('Native parent scene recipe/pin/representation differs')
    if plan['data']['resolution_LR'] != descriptor['native_LR']:
        raise ValueError('Native parent LR grid differs')
    if plan['planned'] != prefix.planned_counts(configuration):
        raise ValueError('Native parent actual-batch planned counts differ')
    if plan['schema'] == 'registered_native_full_LR_prefix_v1':
        domain.validate_plan(plan)
        if plan['data']['resolution_HR'] != descriptor['native_HR']:
            raise ValueError('Registered native parent HR grid differs')
        kind = 'registered_scene_native_parent'
        accepted = None
    elif plan['schema'] == 'full_author_LR_prefix_v1':
        if plan['scene'] not in ('cook_spinach', 'cut_roasted_beef'):
            raise ValueError('Only four already accepted Cook/Cut legacy parents may be reused')
        accepted = domain.completed_prefix(plan['scene'], plan['seed'])
        if accepted is None or complete is None:
            raise ValueError('Legacy parent requires its exact accepted completion/index')
        recorded = read(bound(accepted['entry']['complete']))
        if recorded != complete or complete['checkpoint'] != accepted['entry']['checkpoint']:
            raise ValueError('Legacy parent differs from the accepted native parent')
        # Cook seed1 registered the source before the documented RNG transport
        # fix. The accepted completion preserves that plan plus its amendment.
        actual = complete.get('execution_source')
        if not actual or actual['sha256'] != domain.LEGACY_PREFIX_SHA or bound(actual) != HERE/'full_prefix.py':
            raise ValueError('Accepted legacy execution source differs')
        for name, expected in plan['project_sources'].items():
            if name == str((HERE/'full_prefix.py').relative_to(ROOT)):
                if expected not in (domain.LEGACY_PREFIX_SHA, prefix.RNG_TRANSPORT_PREVIOUS_SOURCE):
                    raise ValueError('Unaccepted legacy prefix source')
                if expected != actual['sha256'] and not complete.get('resume_amendment'):
                    raise ValueError('Legacy source change lacks accepted RNG amendment')
            elif sha(ROOT/name) != expected:
                raise ValueError('Legacy parent dependency source changed: '+name)
        for name, item in plan['author_sources'].items():
            if name not in sources or entry(sources[name]) != item:
                raise ValueError('Legacy author source differs: '+name)
        kind = 'accepted_legacy_Cook_Cut_parent_original_plan_preserved'
    else:
        raise ValueError('Unregistered native parent plan schema')
    return dict(kind=kind, scene=plan['scene'], role=descriptor['role'],
                native_LR=descriptor['native_LR'], native_HR=descriptor['native_HR'],
                batch_size=configuration['OptimizationParams']['batch_size'],
                expected_state=domain.expected_parent_state(configuration),
                accepted_index_path=accepted['index']['path'] if accepted else None,
                accepted_entry=accepted['entry'] if accepted else None,
                accepted_exit=accepted['entry']['exit_acceptance'] if accepted else None)


def full_parent(parent, complete, manifest_entry, seed):
    """CPU sidecar validation without tensor load or rewriting an old plan."""
    parent = local(parent); complete = local(complete)
    done = read(complete); side = read(parent.with_suffix('.json'))
    if done['status'] != 'completed_independent_full_author_LR_prefix':
        raise ValueError('Full native LR parent incomplete')
    if done['checkpoint'] != entry(parent) or {k:side[k] for k in ('path','sha256')} != entry(parent):
        raise ValueError('Full parent SHA differs')
    if side.get('schema') != 'full_author_LR_prefix_checkpoint_v1':
        raise ValueError('Short-window or foreign parent refused')
    p = read(bound(done['plan'])); meta = side['metadata']; state = meta['state']
    contract = validate_parent_plan(p, done)
    if done.get('scene') != p['scene'] or done.get('representation') != REPRESENTATION:
        raise ValueError('Native completion scene/representation differs')
    if any(done.get(name) is not False for name in ('HR_used','heldout_pixels_used','teacher_used')):
        raise ValueError('Native LR parent must preserve the pure-LR information boundary')
    if p['schema'] == 'registered_native_full_LR_prefix_v1' and done.get('execution_source') != entry(HERE/'full_prefix_registered.py'):
        raise ValueError('Registered native parent execution source differs')
    if p['data']['manifest'] != manifest_entry or p['seed'] != seed or done['seed'] != seed:
        raise ValueError('Parent seed/manifest differs')
    if meta['representation'] != REPRESENTATION or state != contract['expected_state'] or done['state'] != state:
        raise ValueError('Complete parent actual batch/Adam/state differs')
    if meta['completed_iterations'] != state['accepted_backward'] or meta['plan_sha256'] != prefix.digest(p):
        raise ValueError('Complete native prefix plan/iteration identity differs')
    groups = ('xyz','deformation','grid','f_dc','f_rest','opacity','scaling','rotation')
    if tuple(x['name'] for x in meta['audit']['optimizer_groups']) != groups:
        raise ValueError('Exactly one native eight-group Adam required')
    if set(p['runtime_source_sha256']) != set(prefix.RUNTIME_FILES):
        raise ValueError('Incomplete native runtime identity')
    for name, value in p['runtime_source_sha256'].items():
        upstream = Path(os.environ.get('FOURDSR_UPSTREAM','/home/cai_tianshun/Project/4dgs'))
        if sha(upstream/name) != value: raise ValueError('Native parent runtime changed: '+name)
    return dict(checkpoint=entry(parent),complete=entry(complete),plan=done['plan'],sidecar=entry(parent.with_suffix('.json')),
                state=state,audit=meta['audit'],source_amendment=meta.get('resume_amendment'),
                original_registered_source=p['project_sources'],actual_execution_source=done.get('execution_source'),
                registered_domain_contract=contract), p


def validate_selection(value, method):
    if value.get('status')!='registered_full_development_selection': raise ValueError('Full candidates require an explicit registered short-window evidence decision')
    candidates = value['selected_candidates']; baseline = value['baseline_method']
    if not 1<=len(candidates)<=2 or len(candidates)!=len(set(candidates)) or not set(candidates)<=set(('Bsync','M','X','MX','E')): raise ValueError('At most two development candidates; no full six-arm grid')
    ablations = value.get('necessary_ablation_methods',[])
    if baseline not in ('B0','Bsync') or not set(ablations)<=set(('M','X')): raise ValueError('Invalid same-backbone baseline/necessary ablations')
    if baseline in candidates: raise ValueError('The baseline cannot also be listed as a selected candidate')
    if method not in [baseline]+candidates+ablations: raise ValueError('This method is outside the registered selected full development set')
    if value.get('uses_confirmation_for_selection') is not False or not value.get('reason') or not value.get('short_window_evidence'): raise ValueError('Development evidence/reason and unused confirmation boundary required')
    for item in value['short_window_evidence']: bound(item)


def validate_schedule(table, manifest, context, seed, updates=STEPS):
    keys = keys_of(manifest)
    if table.get('schema')!=SCHEDULE_SCHEMA or table.get('identity')!=context or table.get('seed')!=seed or table.get('updates')!=updates: raise ValueError('Frozen full native schedule identity differs')
    if table['record_keys']!=[list(key) for key in keys] or not table.get('audit',{}).get('passed'): raise ValueError('Full legal schedule order/audit differs')
    exposure = {}
    for name in ('rows','random_rows'):
        rows = table[name]
        if len(rows)!=updates: raise ValueError('Schedule does not contain all registered SR updates')
        for row in rows:
            if len(row)!=3 or any(type(i) is not int or not 0<=i<len(keys) for i in row): raise ValueError('Invalid full schedule row')
            if name=='rows' and (len({keys[i][0] for i in row})!=3 or len({keys[i][1] for i in row})!=1): raise ValueError('Same-time row must use three distinct training cameras')
        exposure[name] = [Counter(keys[row[col]] for row in rows) for col in range(3)]
    # Weighted LR/SR exposure must be matched exactly, not an unverified flag.
    reference = exposure['random_rows']
    if any(exposure['rows'][i]!=reference[i] for i in range(3)): raise ValueError('Per-column exposure differs between random and same-time schedules')
    if any(reference[i]!=reference[0] for i in (1,2)):raise ValueError('LR-weighted exposure differs between anchor-only and mean-three arms')
    if updates%len(keys)==0:
        if any(set(count)!=set(keys) or set(count.values())!={updates//len(keys)} for count in reference): raise ValueError('Complete block must expose every full observation once per column')
    triplets = table.get('calibration_triplets',[]); anchors = []
    for row in triplets:
        if len(row)!=3 or any(type(i) is not int or not 0<=i<len(keys) for i in row): raise ValueError('Invalid calibration triplet')
        if len({keys[i][0] for i in row})!=3 or len({keys[i][1] for i in row})!=1: raise ValueError('Calibration must use same-time training triplets')
        anchors.append(keys[row[0]])
    expected = {(c,f) for c in manifest['splits']['train'] for f in (0,100,200,299)}
    if len(anchors)!=len(expected) or set(anchors)!=expected: raise ValueError('Full calibration must balance every train camera at registered frames0/100/200/299')
    return exposure


def validate_teacher(index, manifest, manifest_entry, seed, dataset_root):
    expected = dict(zip(keys_of(manifest), [r for r in manifest['observations'] if r['split']=='train']))
    rows = {(e['camera'],int(e['frame'])):e for e in index['entries']}
    if index['status']!='completed_teacher_inventory' or index['manifest_sha256']!=manifest_entry['sha256'] or index['seed_consumer']!=seed: raise ValueError('Complete seed-consumer teacher inventory required')
    if len(rows)!=len(index['entries']) or set(rows)!=set(expected) or not index['seed_independent_frozen_prior']: raise ValueError('Teacher must exactly cover full legal training keys')
    teacher = module('full_refine_teacher_recipe', HERE/'full_teacher_prepare.py')
    if any(index['teacher_config'].get(k)!=v for k,v in teacher.FROZEN.items()): raise ValueError('Full teacher recipe differs from the frozen project mother teacher')
    files = []
    for key, e in rows.items():
        obs = expected[key]
        if e['lr_sha256']!=obs['lr_sha256'] or e['lr_relative_path']!=obs['lr_path']: raise ValueError('Teacher LR input differs from this complete manifest')
        path = safe_child(dataset_root,e['relative_path'])
        if not path.is_file(): raise FileNotFoundError(path)
        files.append(dict(path=str(path.relative_to(ROOT)),sha256=e['sha256'],camera=key[0],frame=key[1]))
    for name in ('producer_config','producer_summary','plan'): bound(index[name])
    return files


def required_edges(table, definition):
    if not definition['X']: return set()
    keys = table['record_keys']; edges=set()
    for row in table['rows']+table['calibration_triplets']:
        for i in range(3):
            for j in range(3):
                if i!=j: edges.add((keys[row[i]][0],keys[row[j]][0],int(keys[row[j]][1])))
    return edges


def validate_support(index, context, table, manifest):
    if index.get('status')!='completed_frozen_X_support' or not index.get('soft_weights_frozen'): raise ValueError('Full X support not complete/frozen')
    if index.get('full_native_context')!=context or index.get('scope')!='full_0_299_train_only': raise ValueError('Short U6000 support is not a full native parent cache')
    if index['identity'].get('source_sha256')!=sha(HERE/'support_cache.py') or index['identity'].get('footprint_source_sha256')!=sha(HERE/'footprint.py'): raise ValueError('Frozen support reader/operator source differs')
    if not index['identity'].get('producer_sources'):raise ValueError('Native full support producer provenance absent')
    for item in index['identity']['producer_sources'].values():bound(item)
    obs = {(e['camera'],int(e['frame'])) for e in index['observations']}
    if obs!=set(keys_of(manifest)) or len(obs)!=len(index['observations']): raise ValueError('Full support observations incomplete/illegal')
    edges = {(e['source'],e['target'],int(e['frame'])) for e in index['entries']}
    if len(edges)!=len(index['entries']) or not required_edges(table,ARMS['X'])<=edges: raise ValueError('Registered full X six-edge support missing')
    if not math.isfinite(index['tau_z']) or index['tau_z']<1e-6: raise ValueError('Invalid frozen full-parent dispersion scale')


def validate_calibration(value, context, support_identity, definition):
    if value.get('status')!='passed' or value.get('full_native_context')!=context: raise ValueError('Full native training-only calibration required')
    if value.get('HR_reference_images_read') is not False or value.get('development_images_read') is not False or value.get('Adam_calls')!=0 or value.get('parameter_updates')!=0 or not value.get('parent_state_unchanged'): raise ValueError('Calibration information/update boundary differs')
    for item in value.get('source_files',{}).values(): bound(item)
    if definition['E'] and (not math.isfinite(float(value.get('kappa',float('nan')))) or value['kappa']<=0): raise ValueError('Full E gradient calibration absent')
    if definition['X']:
        if value.get('support_index')!=support_identity or not value.get('source_RGB_and_target_moment_paths_included') or not value.get('xyz_gradients_summed_before_RMS'): raise ValueError('X calibration must bind complete native paths and full support')
        if not math.isfinite(float(value.get('lambda_X',float('nan')))) or value['lambda_X']<=0 or value.get('median_LR_to_X_RMS_ratio',float('inf'))>1e4: raise ValueError('No usable full X geometry signal; refuse compensating giant coefficient')


def calibration_coefficients(rows,require_X):
    """Training-gradient medians only; neither score thresholds nor a grid."""
    import statistics
    eps=1e-12;ratios=[];e_ratios=[]
    for row in rows:
        l1,l2=row['E_anchor_L1_RGB_RMS'],row['E_anchor_MSE_RGB_RMS']
        if not math.isfinite(l1+l2) or l2<=eps:raise ValueError('No finite E MSE gradient signal')
        e_ratios.append(l1/(l2+eps))
        if require_X and row['effective_edges']>0 and row['X_effective_xyz_RMS']>max(eps,1e-6*row['LR_effective_xyz_RMS']):
            ratios.append(row['LR_effective_xyz_RMS']/(row['X_effective_xyz_RMS']+eps))
    if not e_ratios:raise ValueError('No balanced full training calibration observations')
    median=statistics.median(ratios) if ratios else None
    if require_X and (median is None or not math.isfinite(median) or median<=0 or median>1e4):raise ValueError('Full X has no usable signal; do not manufacture a giant coefficient')
    return dict(kappa=statistics.median(e_ratios),lambda_X=.1*median if median is not None else None,
                median_LR_to_X_RMS_ratio=median,supported_X_batches=len(ratios),excluded_X_batches=len(rows)-len(ratios) if require_X else 0)


def source_files(upstream):
    paths = (Path(__file__),HERE/'full_refine.py',HERE/'full_prefix.py',HERE/'full_prefix_registered.py',HERE/'full_native_protocol.py',HERE/'full_teacher_prepare.py',HERE/'fp_common.py',HERE/'config.py',
             HERE/'losses.py',HERE/'footprint.py',HERE/'support_cache.py',prefix.LEGACY,
             ROOT/'experiments/dynamic_sr_confidence_geometry_20261006/depth_prior.py')
    return dict(project={str(p.relative_to(ROOT)):sha(p) for p in paths},
                upstream={name:sha(Path(upstream)/name) for name in prefix.RUNTIME_FILES})


def build_plan(a):
    """Register actual dependencies or machine-readable pending; no GPU probe."""
    manifest = local(a.manifest); m, data, lr_files = prefix.validate_manifest(manifest,a.seed)
    configuration,_ = prefix.author_configuration(m['scene'])
    definition = copy.deepcopy(ARMS[a.method]); missing=[]; dependencies={}; extra={}
    if a.steps!=STEPS or a.topology_clock!='refinement': raise ValueError('Register a separate audited adapter for any alternative to root-selected SR6000/restarted author topology clock')
    if configuration['PipelineParams']['convert_SHs_python'] or configuration['PipelineParams']['compute_cov3D_python'] or configuration['OptimizationParams']['lambda_dssim']!=0: raise ValueError('Only the pinned false/false raster and L1 author configuration is accepted')
    def available(name,path,required=True):
        if path is None or not local(path).is_file():
            if required: missing.append(dict(dependency=name,path=str(path) if path is not None else None))
            return None
        p=local(path); dependencies[name]=entry(p); return read(p)
    parent = None
    if not local(a.parent).is_file() or not local(a.parent_complete).is_file():
        missing.append(dict(dependency='complete_native_LR_parent',checkpoint=str(a.parent),complete=str(a.parent_complete)))
    else:
        parent, pp = full_parent(a.parent,a.parent_complete,data['manifest'],a.seed)
        if pp['configuration']!=configuration: raise ValueError('Full parent author options differ')
    selection = available('selection',a.selection)
    if selection is not None: validate_selection(selection,a.method)
    teacher = available('teacher',a.teacher)
    teacher_files = validate_teacher(teacher,m,data['manifest'],a.seed,manifest.parent) if teacher is not None else []
    table = available('schedule',a.schedule)
    base_context = dict(manifest=data['manifest'],parent=parent['checkpoint'] if parent else None)
    if table is not None:
        if parent is None: missing.append(dict(dependency='schedule_parent_binding_waits_for_native_parent'))
        else: validate_schedule(table,m,base_context,a.seed)
    context = dict(base_context,schedule=dependencies.get('schedule'))
    support = available('support_index',a.support_index,definition['X'])
    if support is not None and definition['X']:
        if parent is None or table is None: missing.append(dict(dependency='support_context_waits_for_native_parent_and_schedule'))
        else:
            validate_support(support,context,table,m)
            for row in support['observations']+support['entries']:
                path=safe_child(local(a.support_index).parent,row['path'])
                if not path.is_file():raise FileNotFoundError(path)
    cal = available('calibration',a.calibration,definition['X'] or definition['E'])
    if cal is not None and (definition['X'] or definition['E']):
        if parent is None or table is None: missing.append(dict(dependency='calibration_context_waits_for_native_parent_and_schedule'))
        else: validate_calibration(cal,context,dependencies.get('support_index'),definition)
    upstream = Path(a.upstream or os.environ.get('FOURDSR_UPSTREAM','/home/cai_tianshun/Project/4dgs'))
    plan = dict(schema=SCHEMA,status='pending_full_native_refinement_dependencies' if missing else 'registered_ready_full_native_SR_refinement',
        missing=missing,scene=m['scene'],seed=a.seed,method=a.method,definition=definition,data=data,parent=parent,
        dependencies=dependencies,full_native_context=context,configuration=configuration,source_files=source_files(upstream),
        representation=REPRESENTATION,author_commit=prefix.PIN,adaptations=ADAPTATIONS,training_files=dict(LR=lr_files,teacher=teacher_files),
        schedule=dict(steps=STEPS,LR_clock='fine14000+SRcursor',topology_clock='refinement',topology_steps=[1,STEPS],
          topology_rule_source=entry(HERE/'full_prefix_registered.py'),parameter_updates=STEPS,Adam_calls_per_update=1,optimizer_resets=0,
          SR_weights=[0.,.05,.05],X_ramp_steps=500,regularizer_once=True,SH_clock='fine14000+SRcursor',
          topology_buffers='clear only max_radii2D/xyz_gradient_accum/denom once at LR-to-HR boundary; inherit all others; never repeat on SR resume',
          viewspace_aggregation='same-index sum before norm; max radii; any visibility; all three RGB views for all arms'),
        planned_cost=dict(SR_updates=STEPS,RGB_forwards=3*STEPS,moment_forwards=3*STEPS if definition['X'] else 0,
          backward=STEPS,Adam=STEPS,prefix_per_seed=dict(iterations=domain.expected_parent_state(configuration)['accepted_backward'],RGB=domain.expected_parent_state(configuration)['accepted_RGB'],Adam=domain.expected_parent_state(configuration)['accepted_Adam'],batch_size=configuration['OptimizationParams']['batch_size']),
          prefix_shared_within_seed_only=True,teacher_support_calibration_diagnostics_evaluation_separate=True,
          unknown_density_changes='actual count/parameter bytes/checkpoint size logged; no fixed132972 assumption',
          equality='equal updates and shared topology rules; X has extra compute; not equal wallclock'),
        evaluation_contract=dict(heldout='cam00',frames=list(range(300)),floating_metrics=['PSNR','SSIM','LPIPS'],
          complete_frame=True,ROI_is_diagnostic=True,train_only_teacher_LR_closure=True,
          native_checkpoint_schema=SCHEMA,adapter_API=['restore_native_model','native_effective_state','native_rgb','hr_camera'],
          uniform_evaluator='separate native full evaluation required; short custom-model evaluator must not be used',
          evaluation_implemented_in_this_entry=False,benchmark_completed=False),
        CPU_only_registration=True,GPU_forwards=0,Adam_calls=0,HR_training_images_read=False,heldout_pixels_read=False)
    return plan


def verify_sources(plan, upstream):
    current=source_files(upstream)
    if current!=plan['source_files']: raise ValueError('Frozen refinement source tree changed')
    for item in plan['dependencies'].values(): bound(item)
    if plan['parent'] is not None:
        for key in ('checkpoint','complete','plan','sidecar'): bound(plan['parent'][key])


def native_effective_state(g, time_value, torch_module=None):
    """One author deformation call shared by native RGB and raw moments."""
    if torch_module is None: import torch as torch_module
    time_tensor=torch_module.tensor(time_value).to(g.get_xyz.device).repeat(g.get_xyz.shape[0],1)
    xyz,scaling,rotation,opacity,sh=g._deformation(g.get_xyz,g._scaling,g._rotation,g._opacity,g.get_features,time_tensor)
    return dict(xyz=xyz,scales=g.scaling_activation(scaling),rotation=g.rotation_activation(rotation),
                opacity=g.opacity_activation(opacity),sh=sh)


def state_covariance(state, torch_module=None):
    """Author quaternion/scale covariance; device-neutral implementation."""
    if torch_module is None: import torch as torch_module
    q=state['rotation']; q=q/torch_module.sqrt((q*q).sum(dim=1))[:,None]
    r,x,y,z=q.unbind(-1)
    rotation=torch_module.stack((1-2*(y*y+z*z),2*(x*y-r*z),2*(x*z+r*y),
        2*(x*y+r*z),1-2*(x*x+z*z),2*(y*z-r*x),2*(x*z-r*y),2*(y*z+r*x),1-2*(x*x+y*y)),dim=-1).reshape(-1,3,3)
    transform=rotation*state['scales'][:,None,:]
    return transform@transform.transpose(1,2)


def native_rgb(camera,g,state,pipe,background,raster_api=None):
    """Author raster settings/attributes with a precomputed shared fine state."""
    import torch
    if pipe.convert_SHs_python or pipe.compute_cov3D_python: raise ValueError('Only registered false/false author raster path')
    if raster_api is None:
        from diff_gaussian_rasterization import GaussianRasterizationSettings,GaussianRasterizer
    else: GaussianRasterizationSettings,GaussianRasterizer=raster_api
    xyz=state['xyz']; points=torch.zeros_like(g.get_xyz,dtype=g.get_xyz.dtype,device=xyz.device,requires_grad=True)+0
    # Author renderer tolerates rendering under no_grad; training retains the
    # screen-space tensor for its density statistics.
    if points.requires_grad:points.retain_grad()
    settings=GaussianRasterizationSettings(image_height=int(camera.image_height),image_width=int(camera.image_width),
        tanfovx=math.tan(camera.FoVx*.5),tanfovy=math.tan(camera.FoVy*.5),bg=background,scale_modifier=1.,
        viewmatrix=camera.world_view_transform.to(xyz.device),projmatrix=camera.full_proj_transform.to(xyz.device),
        sh_degree=g.active_sh_degree,campos=camera.camera_center.to(xyz.device),prefiltered=False,debug=pipe.debug)
    image,radii,depth=GaussianRasterizer(raster_settings=settings)(means3D=xyz,means2D=points,shs=state['sh'],
        colors_precomp=None,opacities=state['opacity'],scales=state['scales'],rotations=state['rotation'],cov3D_precomp=None)
    return dict(render=image,viewspace_points=points,visibility_filter=radii>0,radii=radii,depth=depth)


def hr_camera(observation,uid,manifest):
    """Use true LR image solely as loader payload; no HR reference is opened."""
    import numpy as np
    from n3dv_data import observation_to_4dgs_camera
    camera=observation_to_4dgs_camera(observation,uid)
    width,height=manifest['resolutions']['hr']; camera.image_width=width;camera.image_height=height
    camera.intrinsics=np.asarray(manifest['cameras'][observation['camera_id']]['K_hr'],dtype=np.float64)
    k=camera.intrinsics;camera.projection_matrix=camera.projection_matrix.clone()
    camera.projection_matrix[2,0]=(2*k[0,2]+1-width)/width;camera.projection_matrix[2,1]=(2*k[1,2]+1-height)/height
    camera.full_proj_transform=camera.world_view_transform@camera.projection_matrix
    return camera


def tree_digest(value):
    """Model, Adam, gradients and RNG digest used for exact restoration audits."""
    import hashlib
    import numpy as np
    import torch
    h=hashlib.sha256()
    def visit(x):
        h.update(type(x).__name__.encode()+b':')
        if torch.is_tensor(x):
            v=x.detach().cpu().contiguous();h.update(str((v.dtype,tuple(v.shape))).encode());h.update(v.numpy().tobytes())
        elif isinstance(x,np.ndarray):h.update(str((x.dtype,x.shape)).encode());h.update(x.tobytes())
        elif isinstance(x,dict):
            for key in sorted(x,key=str):visit(key);visit(x[key])
        elif isinstance(x,(list,tuple)):
            for v in x:visit(v)
        else:h.update(repr(x).encode())
    visit(value);return h.hexdigest()


def restore_native_model(g,payload,opt,torch_module,numpy_module,expected_plan,expected_schema):
    if payload.get('schema')!=expected_schema or payload.get('plan')!=expected_plan: raise ValueError('Foreign model/plan refused; native parent conversion is forbidden')
    g.restore(payload['model'],opt);g._deformation_accum=payload['deformation_accum']
    params=prefix.parameter_map(g)
    if set(params)!=set(payload['parameter_gradients']): raise ValueError('Native saved gradient coordinate mismatch')
    for name,p in params.items():p.grad=payload['parameter_gradients'][name]
    if prefix.topology_audit(g)!=payload['metadata']['audit']: raise ValueError('Native Adam/rates/topology restore differs')
    if tree_digest(g.capture())!=tree_digest(payload['model']): raise ValueError('Native parameter/Adam restore not exact')
    prefix.restore_rng(payload['rng'],torch_module,numpy_module)
    if tree_digest(prefix.rng_state(torch_module,numpy_module))!=tree_digest(payload['rng']): raise ValueError('Parent global RNG was not inherited exactly')
    if expected_schema==SCHEMA:
        if payload['metadata'].get('density_statistical_boundary_applied') is not True:raise ValueError('SR checkpoint lacks its committed one-time statistical boundary')
        g._full_SR_statistics_initialized=True
    return dict(one_Adam_exact=True,parameters_exact=True,topology_buffers_exact=True,
                saved_gradients_exact=True,global_RNG_exact=True,no_children=True,optimizer_conversion=False)


def initialize_SR_statistics(g,cursor,torch_module,numpy_module):
    """Explicit once-only LR-to-HR statistical boundary, not an Adam reset."""
    if cursor!=0 or getattr(g,'_full_SR_statistics_initialized',False):raise ValueError('LR-to-HR density boundary is only legal once at a fresh native parent fork')
    before=dict(parameters=tree_digest(prefix.parameter_map(g)),Adam=tree_digest(g.optimizer.state_dict()),
        RNG=tree_digest(prefix.rng_state(torch_module,numpy_module)),deformation_accum=tree_digest(g._deformation_accum),
        deformation_table=tree_digest(g._deformation_table))
    names=('max_radii2D','xyz_gradient_accum','denom')
    def summary():
        return {name:dict(sha256=tree_digest(getattr(g,name)),shape=list(getattr(g,name).shape),
            max_abs=float(getattr(g,name).abs().max()) if getattr(g,name).numel() else 0.,
            mean_abs=float(getattr(g,name).abs().mean()) if getattr(g,name).numel() else 0.) for name in names}
    old=summary()
    with torch_module.no_grad():
        for name in names:getattr(g,name).zero_()
    after=dict(parameters=tree_digest(prefix.parameter_map(g)),Adam=tree_digest(g.optimizer.state_dict()),
        RNG=tree_digest(prefix.rng_state(torch_module,numpy_module)),deformation_accum=tree_digest(g._deformation_accum),
        deformation_table=tree_digest(g._deformation_table))
    if before!=after:raise ValueError('LR-to-HR statistical boundary changed parameters/Adam/RNG/other native buffers')
    g._full_SR_statistics_initialized=True
    return dict(status='applied_once_at_fresh_parent_SR_boundary',cleared_buffers=list(names),before=old,after=summary(),
                unchanged=before,reason='LR density/radius units and past LR exposure excluded from new HR density statistics')


def gradient_diagnostics(terms,rgbs,states,torch_module,ledger=None):
    import losses
    vectors={};result={}
    for name,loss in terms.items():
        rms,xyz=prefix.operation(ledger,'diagnostic_autograd',lambda:losses.summed_xyz_gradient_rms(loss,[s['xyz'] for s in states]))
        rgb=prefix.operation(ledger,'diagnostic_autograd',lambda:torch_module.autograd.grad(loss,rgbs,retain_graph=True,allow_unused=True))
        rgb_vec=torch_module.stack([torch_module.zeros_like(im) if grad is None else grad for im,grad in zip(rgbs,rgb)])
        result[name]=dict(effective_xyz_RMS=rms,RGB_RMS=float(rgb_vec.detach().double().square().mean().sqrt()))
        vectors[name]=(xyz,rgb_vec)
    for left,right in (('LR','SR'),('LR','X'),('SR','X')):
        for idx,label in ((0,'xyz'),(1,'RGB')):
            a,b=vectors[left][idx].detach().double().flatten(),vectors[right][idx].detach().double().flatten()
            denom=a.norm()*b.norm();result[f'{left}_{right}_{label}_cosine']=float(a.dot(b)/denom) if float(denom)>0 else None
    return result


def refinement_iteration(g,h,opt,pipe,background,cameras,lrs,teachers,keys,definition,cursor,extent,out,torch_module,
                         cal=None,support=None,render=native_rgb,moment_render=None,ledger=None,diagnostics=False):
    """One shared-state accumulated update; author topology precedes one Adam."""
    import losses
    cal=cal or {};g.update_learning_rate(LR_OFFSET+cursor)
    if (LR_OFFSET+cursor)%1000==0:g.oneupSHdegree()
    # Final native fine gradient was not applied. It cannot be accumulated with
    # a different SR objective; parameters and all Adam moments are retained.
    g.optimizer.zero_grad(set_to_none=True)
    same_time=definition['schedule']=='rows'
    if same_time:
        if len({k[0] for k in keys})!=3 or len({int(k[1]) for k in keys})!=1:raise ValueError('Registered same-time batch differs')
        state=native_effective_state(g,cameras[0].time,torch_module);states=[state]*3
    else:states=[native_effective_state(g,c.time,torch_module) for c in cameras]
    packages=[prefix.operation(ledger,'RGB',lambda c=c,s=s:render(c,g,s,pipe,background)) for c,s in zip(cameras,states)]
    rgbs=[p['render'] for p in packages]
    lr_values=[losses.lr_loss(rgb,lr) for rgb,lr in zip(rgbs,lrs)]
    lr_term=sum(w*v for w,v in zip(definition['lr_weights'],lr_values))
    if definition['E']:lr_term=losses.e_loss(rgbs[0],lrs[0],cal['kappa'])
    sr_values=[losses.sr_loss(rgbs[i],teacher) for i,teacher in zip((1,2),teachers)]
    sr_term=.05*(sr_values[0]+sr_values[1])
    regularizer=g.compute_regulation(h.time_smoothness_weight,h.l1_time_planes,h.plane_tv_weight) if h.time_smoothness_weight!=0 else rgbs[0].sum()*0.
    x=rgbs[0].sum()*0.;xdiag=None;moments=[]
    if definition['X']:
        if support is None or moment_render is None:raise ValueError('Native X support/moment renderer missing')
        cov=state_covariance(states[0],torch_module)
        for camera in cameras:
            moments.append(prefix.operation(ledger,'moments',lambda c=camera:moment_render(c,states[0]['xyz'],cov,states[0]['opacity'])))
        x,xdiag=losses.x_loss(rgbs,moments,cameras,lrs,keys,support)
    coefficient=float(cal.get('lambda_X',0.))*min(1.,cursor/500) if definition['X'] else 0.
    loss=lr_term+sr_term+regularizer+coefficient*x
    if not bool(torch_module.isfinite(loss)):raise FloatingPointError('Nonfinite complete SR objective')
    gdiag=gradient_diagnostics(dict(LR=sum(lr_values)/3,SR=sr_term,X=x),rgbs,states,torch_module,ledger) if diagnostics else None
    # Diagnostic autograd may populate retained intermediate .grad. Native
    # viewspace statistics must contain the formal loss gradient only.
    for p in packages:p['viewspace_points'].grad=None
    prefix.operation(ledger,'backward',loss.backward)
    if any(not bool(torch_module.isfinite(p.grad).all()) for p in prefix.parameter_map(g).values() if p.grad is not None):raise FloatingPointError('Nonfinite native SR gradient')
    statistics=prefix.combine_batch_statistics(packages,torch_module)
    before=int(g.get_xyz.shape[0])
    xyz_gradient_RMS=float(statistics[2].detach().double().square().mean().sqrt())
    with torch_module.no_grad():
        prefix.author_topology(g,opt,'fine',cursor,extent,out,statistics,ledger)
        prefix.operation(ledger,'Adam',g.optimizer.step)
        g.optimizer.zero_grad(set_to_none=True)
    audit=prefix.topology_audit(g)
    if ledger:ledger.completed['iterations']+=1;ledger.persist()
    return dict(suffix_step=cursor,LR_clock=LR_OFFSET+cursor,topology_clock=cursor,loss=float(loss.detach()),
        LR_L1=[float(v.detach()) for v in lr_values],SR_L1=[float(v.detach()) for v in sr_values],
        LR_term=float(lr_term.detach()),SR_term=float(sr_term.detach()),regularizer=float(regularizer.detach()),
        X_loss=float(x.detach()),lambda_X=coefficient,X_diagnostic=xdiag,gdiag=gdiag,
        points_before=before,points_after=audit['points'],native_audit=audit,
        summed_viewspace_gradient_RMS=xyz_gradient_RMS,RGB=3,moments=len(moments),backward=1,Adam=1,
        density_aggregation='three RGB screen grads summed before norm; independent target-moment screen grads excluded by the author RGB statistic')


class Journal(prefix.Journal):
    def __init__(self,out,identity,cursor):
        self.path=Path(out)/'attempts'/f'{time.time_ns()}_{os.getpid()}.json';self.started=time.monotonic()
        self.completed={k:0 for k in OPERATIONS};self.attempted={k:0 for k in OPERATIONS}
        self.value=dict(status='running_native_SR_refinement',identity=identity,host=socket.gethostname(),pid=os.getpid(),
            started_unix=time.time(),suffix_start=cursor,suffix_cursor=cursor,active_operation=None)
        self.persist()


def save_checkpoint(path,g,h,opt,plan,cursor,parent_sampler,torch_module,numpy_module,journal,hardware):
    path=Path(path)
    if path.exists() or path.with_suffix('.json').exists():raise FileExistsError('Committed native SR checkpoint preserved')
    if not getattr(g,'_full_SR_statistics_initialized',False):raise ValueError('SR checkpoint requires explicit fresh-parent statistical boundary')
    metadata=dict(cursor=cursor,plan_sha256=prefix.digest(plan),audit=prefix.topology_audit(g),parent=plan['parent'],
        representation=REPRESENTATION,hardware=hardware,attempt=str(journal.path.relative_to(ROOT)),
        completed_operations=dict(journal.completed),terminal_zero_grad=True,density_statistical_boundary_applied=True)
    payload=dict(schema=SCHEMA,model=g.capture(),hidden=vars(h),optim=vars(opt),plan=plan,metadata=metadata,
        sampler=dict(kind='registered_full_native_SR_schedule',cursor=cursor,schedule=plan['dependencies']['schedule']),
        parent_LR_sampler=parent_sampler,rng=prefix.rng_state(torch_module,numpy_module),deformation_accum=g._deformation_accum,
        parameter_gradients={name:p.grad for name,p in prefix.parameter_map(g).items()})
    temp=path.with_name(path.name+f'.{os.getpid()}.tmp');torch_module.save(payload,temp)
    with temp.open('rb') as handle:os.fsync(handle.fileno())
    temp.replace(path);write(path.with_suffix('.json'),dict(**entry(path),schema=SCHEMA,metadata=metadata))
    return path


def checkpoint_payload(path,torch_module,expected_plan,expected_schema,device):
    path=Path(path);side=read(path.with_suffix('.json'))
    if {k:side[k] for k in ('path','sha256')}!=entry(path):raise ValueError('Native checkpoint bytes changed')
    payload=torch_module.load(path,map_location=device,weights_only=False)
    if payload['schema']!=expected_schema or payload['plan']!=expected_plan or payload['metadata']!=side['metadata']:raise ValueError('Native checkpoint source/identity changed')
    return payload


class TeacherCache:
    """Eight float32 CPU images at most, with lazy first-open SHA checking."""
    def __init__(self,paths,identities,device,expected_size,maximum=8):
        self.paths=paths;self.identities=identities;self.device=device;self.maximum=maximum;self.lru=OrderedDict();self.verified=set()
        self.expected_size=tuple(expected_size)
        if self.expected_size not in ((1344,1008),(1280,720)):raise ValueError('Unregistered full teacher grid')
    def get(self,key):
        import numpy as np
        import torch
        from PIL import Image
        if key not in self.lru:
            path=self.paths[key]
            if key not in self.verified:
                if sha(path)!=self.identities[key]:raise ValueError('Frozen full teacher image changed')
                self.verified.add(key)
            with Image.open(path) as image:
                if image.size!=self.expected_size:raise ValueError('Full teacher dimensions differ from the registered scene')
                self.lru[key]=torch.from_numpy(np.asarray(image.convert('RGB')).copy()).permute(2,0,1).float()/255
            if len(self.lru)>self.maximum:self.lru.popitem(last=False)
        value=self.lru.pop(key);self.lru[key]=value;return value.to(self.device)


def restore_training_start(a,plan,h,opt,torch,np,hardware):
    """Initialization is journalled by the caller even before any RGB call."""
    from scene.gaussian_model import GaussianModel
    g=GaussianModel(plan['configuration']['ModelParams']['sh_degree'],h);g._deformation=g._deformation.cuda()
    boundary=None
    if a.resume:
        candidates=[]
        for path in a.out.glob('checkpoint_*.pt'):
            if not path.with_suffix('.json').exists():raise ValueError('Orphan checkpoint preserved; explicit recovery required before further updates')
            candidates.append((read(path.with_suffix('.json'))['metadata']['cursor'],path))
        if not candidates:raise ValueError('No committed full SR checkpoint')
        cursor,path=max(candidates)
        if a.resume!='latest' and local(a.resume).resolve()!=path.resolve():raise ValueError('Native full refinement resume must use latest committed checkpoint')
        payload=checkpoint_payload(path,torch,plan,SCHEMA,'cuda')
        if payload['metadata']['hardware']!=hardware:raise ValueError('Resume hardware/environment differs; preserve this run')
        if payload['sampler']!=dict(kind='registered_full_native_SR_schedule',cursor=cursor,schedule=plan['dependencies']['schedule']):raise ValueError('Full SR schedule cursor differs')
        restore_native_model(g,payload,opt,torch,np,plan,SCHEMA);parent_sampler=payload['parent_LR_sampler']
    else:
        parent_plan=read(bound(plan['parent']['plan']));payload=checkpoint_payload(bound(plan['parent']['checkpoint']),torch,parent_plan,'full_author_LR_prefix_checkpoint_v1','cuda')
        cursor=0;parent_sampler=payload['sampler']
        audit=restore_native_model(g,payload,opt,torch,np,parent_plan,'full_author_LR_prefix_checkpoint_v1')
        boundary=initialize_SR_statistics(g,0,torch,np)
        write(a.out/'parent_reload_audit.json',dict(status='passed_native_parent_fork',parent=plan['parent'],audit=audit,density_statistical_boundary=boundary,
            final_unapplied_parent_gradient_digest=tree_digest(payload['parameter_gradients']),cleared_before_new_SR_backward=True,
            native_Adam_state_digest=tree_digest(g.optimizer.state_dict()),RNG_digest=tree_digest(payload['rng']),native_acceptance=entry(local(a.native_acceptance))))
    return g,cursor,parent_sampler,boundary


def train(a,plan):
    """Root dispatches this entry only after separate native CUDA acceptance."""
    if plan['status']!='registered_ready_full_native_SR_refinement':raise ValueError('Pending full dependencies refuse training: '+str(plan['missing']))
    upstream=Path(a.upstream or os.environ.get('FOURDSR_UPSTREAM','/home/cai_tianshun/Project/4dgs'));verify_sources(plan,upstream)
    if not a.native_acceptance:raise ValueError('Explicit native SR CUDA acceptance receipt required before formal dispatch')
    acceptance=read(local(a.native_acceptance))
    if acceptance.get('status')!='passed_native_full_refinement_CUDA_acceptance' or acceptance.get('source_sha256')!=sha(__file__) or acceptance.get('author_commit')!=prefix.PIN:raise ValueError('Native SR CUDA acceptance source/pin differs')
    prefix.check_gpu(a.gpu_uuid)
    if not a.out.resolve().is_relative_to((OUT/'full_SR_refinement').resolve()):raise ValueError('Separate full SR output tree required')
    a.out.mkdir(parents=True,exist_ok=True);config_path=a.out/'config.json'
    if config_path.exists():
        if not a.resume or read(config_path)!=plan:raise ValueError('Existing native run requires identical plan and explicit latest resume')
    else:
        if a.resume or any(a.out.iterdir()):raise ValueError('Existing unregistered directory preserved')
        write(config_path,plan)
    import numpy as np
    import torch
    sys.path.insert(0,str(upstream));sys.path.insert(1,str(prefix.LEGACY.parent))
    from n3dv_data import load_manifest,N3DVPreparedDataset
    manifest=load_manifest(bound(plan['data']['manifest']));dataset=N3DVPreparedDataset(manifest,'train','lr',device='cpu',cache=False)
    table=read(bound(plan['dependencies']['schedule']));definition=plan['definition'];rows=table[definition['schedule']]
    cal=read(bound(plan['dependencies']['calibration'])) if definition['X'] or definition['E'] else {}
    support=None;moment_render=None
    legal={str(local(e['path']).resolve()):e['sha256'] for files in plan['training_files'].values() for e in files}
    if definition['X']:
        from support_cache import SupportCache
        support=SupportCache(bound(plan['dependencies']['support_index']))
        legal.update({str(p.resolve()):row['sha256'] for p,row in zip(support.training_files(),support.index['observations']+support.index['entries'])})
        moment_render=module('full_refine_live_native_moments',ROOT/'experiments/dynamic_sr_confidence_geometry_20261006/depth_prior.py').render_moments
    opened=Counter()
    def guard(event,arg):
        if event!='open' or not isinstance(arg[0],(str,bytes,os.PathLike)):return
        path=Path(os.fsdecode(arg[0]))
        if path.suffix.lower() not in ('.png','.jpg','.jpeg','.npy','.npz'):return
        name=str(path.resolve())
        if name not in legal:raise AssertionError('Full refinement read outside legal LR/teacher/frozen support: '+name)
        opened[name]+=1
    sys.addaudithook(guard);torch.set_num_threads(a.cpu_threads)
    h=SimpleNamespace(**plan['configuration']['ModelHiddenParams']);opt=SimpleNamespace(**plan['configuration']['OptimizationParams']);pipe=SimpleNamespace(**plan['configuration']['PipelineParams'])
    journal=Journal(a.out,prefix.digest(plan),0)
    hardware=dict(host=socket.gethostname(),gpu_uuid=a.gpu_uuid,gpu=torch.cuda.get_device_name(),torch=str(torch.__version__),cuda=torch.version.cuda)
    try:
        g,cursor,parent_sampler,boundary=restore_training_start(a,plan,h,opt,torch,np,hardware)
        if not cursor<a.stop<=STEPS:raise ValueError('Stop must extend the accepted native SR cursor within6000')
    except BaseException as exc:
        journal.finish('failed_native_SR_initialization_preserved',error=repr(exc),traceback=traceback.format_exc(),
            RGB_forwards=0,moment_forwards=0,Adam_calls=0,CUDA_initialization_or_restore_time_included_in_elapsed=True)
        raise
    # A crash tail stays in its immutable attempt log; resumed logs append only
    # to new attempt files and never silently rewrite the prior trajectory.
    journal.value.update(suffix_start=cursor,suffix_cursor=cursor,hardware=hardware,density_statistical_boundary=boundary,
        boundary_reapplied_on_resume=False);journal.persist()
    teacher_index=read(bound(plan['dependencies']['teacher']));teacher_rows={(e['camera'],int(e['frame'])):e for e in teacher_index['entries']}
    paths={key:safe_child(Path(manifest['_root']),e['relative_path']) for key,e in teacher_rows.items()}
    teacher=TeacherCache(paths,{key:e['sha256'] for key,e in teacher_rows.items()},'cuda',manifest['resolutions']['hr'])
    background=torch.ones(3,device='cuda',dtype=torch.float32);torch.cuda.reset_peak_memory_stats();start=cursor
    trajectory=a.out/'trajectories'/f'{time.time_ns()}_{os.getpid()}.jsonl';trajectory.parent.mkdir(exist_ok=True)
    try:
        if cursor==0:
            g.optimizer.zero_grad(set_to_none=True)
            save_checkpoint(a.out/'checkpoint_00000.pt',g,h,opt,plan,0,parent_sampler,torch,np,journal,hardware)
        with trajectory.open('x',buffering=1) as log:
            for cursor in range(start+1,a.stop+1):
                tick=time.monotonic();journal.value['suffix_cursor']=cursor;journal.attempted['iterations']+=1;journal.persist()
                row=rows[cursor-1];observations=[]
                for index in row:
                    obs=dataset.observations[index];path=safe_child(Path(manifest['_root']),obs['lr_path'])
                    if sha(path)!=obs['lr_sha256']:raise ValueError('Actual full LR training bytes changed')
                    observations.append(dataset[index])
                cameras=[hr_camera(obs,i,manifest) for i,obs in zip(row,observations)]
                keys=[tuple(table['record_keys'][i]) for i in row];lrs=[obs['image'].cuda() for obs in observations]
                teachers=[teacher.get(keys[i]) for i in (1,2)]
                diagnostic=refinement_iteration(g,h,opt,pipe,background,cameras,lrs,teachers,keys,definition,cursor,
                    plan['data']['cameras_extent'],a.out,torch,cal,support,native_rgb,moment_render,journal,cursor in (1,100,500,3000,6000))
                diagnostic.update(keys=[list(k) for k in keys],seconds=time.monotonic()-tick,peak_GB=torch.cuda.max_memory_allocated()/1e9)
                log.write(__import__('json').dumps(diagnostic,ensure_ascii=False,allow_nan=False)+'\n')
                # All six-edge and native RGB graphs are scoped inside the
                # iteration function and released before the next HR batch.
                del observations,cameras,lrs,teachers
                if cursor%a.checkpoint_interval==0 or cursor==a.stop:save_checkpoint(a.out/f'checkpoint_{cursor:05d}.pt',g,h,opt,plan,cursor,parent_sampler,torch,np,journal,hardware)
        verify_sources(plan,upstream);journal.finish('completed_formal_SR_segment',cursor=cursor,trajectory=entry(trajectory))
        path=a.out/f'checkpoint_{cursor:05d}.pt'
        receipt=dict(status='completed_full_native_SR_training_requires_uniform_evaluation' if cursor==STEPS else 'paused_full_native_SR_incomplete',
            cursor=cursor,start=start,plan=entry(config_path),checkpoint=entry(path),trajectory=entry(trajectory),
            attempt=entry(journal.path),actual_cost=read(journal.path),all_prior_attempt_costs=[entry(p) for p in sorted((a.out/'attempts').glob('*.json'))],
            hardware=hardware,final_capacity=prefix.topology_audit(g),checkpoint_bytes=path.stat().st_size,teacher_unique_reads=len(teacher.verified),
            actual_training_reads=dict(opened),HR_training_images_read=False,heldout_training_reads=0,benchmark_completed=False)
        write(a.out/f'segment_{start:05d}_{cursor:05d}.json',receipt)
        if cursor==STEPS:write(a.out/'training_complete.json',receipt)
        return receipt
    except BaseException as exc:
        journal.finish('failed_full_native_SR_execution_preserved',error=repr(exc),traceback=traceback.format_exc(),uncommitted_active_operation_cost_unknown=True)
        raise


def calibrate(a,plan):
    """Native full-parent zero-update calibration; E avoids X cache/moments."""
    if a.method not in ('E','X','MX'):raise ValueError('M/baselines need no coefficient calibration')
    required=('selection','schedule')+ (('support_index',) if plan['definition']['X'] else ())
    if plan['parent'] is None or any(name not in plan['dependencies'] for name in required):raise ValueError('Native calibration parent/selection/schedule/support not ready')
    if not a.native_acceptance:raise ValueError('Native CUDA acceptance required before calibration')
    acceptance=read(local(a.native_acceptance))
    if acceptance.get('status')!='passed_native_full_refinement_CUDA_acceptance' or acceptance.get('source_sha256')!=sha(__file__):raise ValueError('Calibration native CUDA source acceptance differs')
    prefix.check_gpu(a.gpu_uuid)
    upstream=Path(a.upstream or os.environ.get('FOURDSR_UPSTREAM','/home/cai_tianshun/Project/4dgs'));verify_sources(plan,upstream)
    if a.out.exists():raise FileExistsError('Calibration histories immutable; use a distinct registered output revision')
    a.out.mkdir(parents=True);write(a.out/'registration.json',dict(status='registered_native_full_train_calibration_before_forward',
        full_native_context=plan['full_native_context'],source_files=plan['source_files'],method=a.method,
        calibration_keys='all actual train cameras x frames0/100/200/299; same-time triplets',
        coefficient_rule='kappa median RGB L1/MSE RMS; lambda_X .1 median summed-native-index LR/X xyz RMS',
        parent=plan['parent'],teacher_reads=0,HR_reads=0,heldout_reads=0,formal_updates=0,Adam_calls=0))
    import numpy as np
    import torch
    import losses
    sys.path.insert(0,str(upstream));sys.path.insert(1,str(prefix.LEGACY.parent))
    from scene.gaussian_model import GaussianModel
    from n3dv_data import load_manifest,N3DVPreparedDataset
    manifest=load_manifest(bound(plan['data']['manifest']));dataset=N3DVPreparedDataset(manifest,'train','lr',device='cpu',cache=False)
    table=read(bound(plan['dependencies']['schedule']));need_X=plan['definition']['X'];support=None;moment_render=None
    allowed={str(local(e['path']).resolve()) for e in plan['training_files']['LR']};opened=Counter()
    if need_X:
        from support_cache import SupportCache
        support=SupportCache(bound(plan['dependencies']['support_index']));allowed.update(str(p.resolve()) for p in support.training_files())
        moment_render=module('full_refine_calibration_native_moments',ROOT/'experiments/dynamic_sr_confidence_geometry_20261006/depth_prior.py').render_moments
    def guard(event,arg):
        if event!='open' or not isinstance(arg[0],(str,bytes,os.PathLike)):return
        p=Path(os.fsdecode(arg[0]))
        if p.suffix.lower() not in ('.png','.jpg','.jpeg','.npz','.npy'):return
        name=str(p.resolve())
        if name not in allowed:raise AssertionError('Calibration read outside full LR/frozen support: '+name)
        opened[name]+=1
    sys.addaudithook(guard);torch.set_num_threads(a.cpu_threads)
    h=SimpleNamespace(**plan['configuration']['ModelHiddenParams']);opt=SimpleNamespace(**plan['configuration']['OptimizationParams']);pipe=SimpleNamespace(**plan['configuration']['PipelineParams'])
    journal=Journal(a.out,prefix.digest(plan),0);rows=[]
    try:
        g=GaussianModel(plan['configuration']['ModelParams']['sh_degree'],h);g._deformation=g._deformation.cuda()
        parent_plan=read(bound(plan['parent']['plan']));payload=checkpoint_payload(bound(plan['parent']['checkpoint']),torch,parent_plan,'full_author_LR_prefix_checkpoint_v1','cuda')
        restore_native_model(g,payload,opt,torch,np,parent_plan,'full_author_LR_prefix_checkpoint_v1');del payload
        # The diagnostic parent retains its LR buffers and final gradient;
        # training's one-time LR-to-HR reset must never be invoked here.
        def immutable():return dict(model=tree_digest(g.capture()),deformation_accum=tree_digest(g._deformation_accum),
            gradients=tree_digest({name:p.grad for name,p in prefix.parameter_map(g).items()}),
            RNG=tree_digest(prefix.rng_state(torch,np)),versions={name:p._version for name,p in prefix.parameter_map(g).items()})
        before=immutable();background=torch.ones(3,device='cuda');torch.cuda.reset_peak_memory_stats()
        with (a.out/'gdiag.jsonl').open('x',buffering=1) as log:
            for number,row in enumerate(table['calibration_triplets']):
                journal.value['batch']=number;journal.persist();tick=time.monotonic();observations=[]
                for index in row:
                    obs=dataset.observations[index];path=safe_child(Path(manifest['_root']),obs['lr_path'])
                    if sha(path)!=obs['lr_sha256']:raise ValueError('Calibration actual LR bytes changed')
                    observations.append(dataset[index])
                keys=[tuple(table['record_keys'][i]) for i in row];cameras=[hr_camera(obs,i,manifest) for i,obs in zip(row,observations)]
                lrs=[obs['image'].cuda() for obs in observations];state=native_effective_state(g,cameras[0].time,torch)
                rgb=[prefix.operation(journal,'RGB',lambda c=c:native_rgb(c,g,state,pipe,background)['render']) for c in cameras]
                errors=[losses.degradation(im,lr.shape[-2:])-lr for im,lr in zip(rgb,lrs)]
                lr_term=sum(v.abs().mean() for v in errors)/3
                l1,l2=errors[0].abs().mean(),errors[0].square().mean()
                gl1,=prefix.operation(journal,'diagnostic_autograd',lambda:torch.autograd.grad(l1,rgb[0],retain_graph=True))
                gl2,=prefix.operation(journal,'diagnostic_autograd',lambda:torch.autograd.grad(l2,rgb[0],retain_graph=True))
                result=dict(batch=number,keys=[list(k) for k in keys],E_anchor_L1_RGB_RMS=float(gl1.detach().double().square().mean().sqrt()),
                    E_anchor_MSE_RGB_RMS=float(gl2.detach().double().square().mean().sqrt()),LR_loss=float(lr_term.detach()),
                    X_loss=None,LR_effective_xyz_RMS=None,X_effective_xyz_RMS=None,effective_edges=0,X_diagnostic=None,RGB=3,moments=0,Adam=0,formal_updates=0)
                if need_X:
                    cov=state_covariance(state);moments=[prefix.operation(journal,'moments',lambda c=c:moment_render(c,state['xyz'],cov,state['opacity'])) for c in cameras]
                    x,xdiag=losses.x_loss(rgb,moments,cameras,lrs,keys,support)
                    lr_rms,lr_gradient=prefix.operation(journal,'diagnostic_autograd',lambda:losses.summed_xyz_gradient_rms(lr_term,[state['xyz']]))
                    x_rms,x_gradient=prefix.operation(journal,'diagnostic_autograd',lambda:losses.summed_xyz_gradient_rms(x,[state['xyz']]))
                    rgb_gradient=prefix.operation(journal,'diagnostic_autograd',lambda:torch.autograd.grad(x,rgb,retain_graph=True,allow_unused=True))
                    result.update(X_loss=float(x.detach()),LR_effective_xyz_RMS=lr_rms,X_effective_xyz_RMS=x_rms,
                        effective_edges=xdiag['effective_edges'],X_diagnostic=xdiag,moments=3,
                        X_RGB_RMS=float(torch.stack([torch.zeros_like(im) if grad is None else grad for im,grad in zip(rgb,rgb_gradient)]).detach().double().square().mean().sqrt()))
                    del moments,cov,x,xdiag,lr_gradient,x_gradient,rgb_gradient
                result['seconds']=time.monotonic()-tick;rows.append(result);log.write(__import__('json').dumps(result,ensure_ascii=False,allow_nan=False)+'\n')
                # Release complete native RGB/moment/six-edge graphs every
                # batch. Only scalar result rows persist; no warmup or Adam.
                del observations,cameras,lrs,state,rgb,errors,lr_term,l1,l2,gl1,gl2
        coefficients=calibration_coefficients(rows,need_X)
        after=immutable()
        if before!=after:raise ValueError('Full calibration changed native model/Adam/buffers/gradients/versions/RNG')
        verify_sources(plan,upstream);journal.finish('completed_native_full_gradient_calibration_no_updates',batches=len(rows))
        report=dict(status='passed',full_native_context=plan['full_native_context'],support_index=plan['dependencies'].get('support_index') if need_X else None,
            calibration_scope='X_E' if need_X else 'E_only_no_X_cache',registered_batches=len(rows),rows=rows,**coefficients,
            parent_state_unchanged=True,immutable_before=before,immutable_after=after,source_RGB_and_target_moment_paths_included=need_X,
            xyz_gradients_summed_before_RMS=True,HR_reference_images_read=False,development_images_read=False,teacher_reads=0,
            Adam_calls=0,parameter_updates=0,source_files={rel:dict(path=rel,sha256=value) for rel,value in plan['source_files']['project'].items()},
            gdiag=entry(a.out/'gdiag.jsonl'),attempt=entry(journal.path),actual_cost=read(journal.path),actual_LR_and_support_reads=dict(opened),
            peak_GB=torch.cuda.max_memory_allocated()/1e9,gpu_uuid=a.gpu_uuid,gpu=torch.cuda.get_device_name(),
            tau_z=support.index['tau_z'] if need_X else None,no_coefficient_grid=True,formal_training_updates=0)
        write(a.out/'calibration.json',report);return report
    except BaseException as exc:
        journal.finish('failed_native_full_calibration_no_formal_updates',error=repr(exc),traceback=traceback.format_exc(),unreturned_operation_cost_unknown=True)
        raise


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('plan','train','calibrate'),default='plan');p.add_argument('--method',choices=METHODS,required=True)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--parent',type=Path,required=True);p.add_argument('--parent-complete',type=Path,required=True)
    p.add_argument('--teacher',type=Path,required=True);p.add_argument('--schedule',type=Path,required=True)
    p.add_argument('--selection',type=Path);p.add_argument('--support-index',type=Path);p.add_argument('--calibration',type=Path)
    p.add_argument('--steps',type=int,default=STEPS);p.add_argument('--topology-clock',choices=('refinement',),default='refinement')
    p.add_argument('--out',type=Path,required=True);p.add_argument('--upstream',type=Path);p.add_argument('--resume')
    p.add_argument('--stop',type=int,default=STEPS);p.add_argument('--checkpoint-interval',type=int,default=100)
    p.add_argument('--cpu-threads',type=int,default=4);p.add_argument('--gpu-uuid');p.add_argument('--native-acceptance',type=Path)
    return p


def main():
    a=parser().parse_args();a.out=local(a.out)
    if a.checkpoint_interval<1 or a.cpu_threads<1:raise ValueError('Positive interval/thread settings required')
    plan=build_plan(a)
    if a.mode=='plan':
        path=a.out/'registration.json'
        if path.exists() and read(path)!=plan:raise ValueError('Prior registration preserved; use a new revision directory')
        if not path.exists():write(path,plan)
        print(__import__('json').dumps(dict(status=plan['status'],missing=plan['missing'],registration=entry(path)),ensure_ascii=False));return
    if a.mode=='train' and plan['status']!='registered_ready_full_native_SR_refinement':raise ValueError('Pending full prerequisites; no CUDA initialized: '+str(plan['missing']))
    lock=OUT/'resource_locks'/f'{a.gpu_uuid}.lock';lock.parent.mkdir(parents=True,exist_ok=True)
    with lock.open('a+') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        print(__import__('json').dumps(calibrate(a,plan) if a.mode=='calibrate' else train(a,plan),ensure_ascii=False))


if __name__=='__main__':main()
