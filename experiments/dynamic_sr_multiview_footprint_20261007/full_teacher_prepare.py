"""CPU-only full-time frozen SwinIR planning and completed-cache acceptance.

No model is imported, no SR inference is started, and no GPU is selected here.
The exact historical generator remains a separate, root-scheduled prerequisite.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import struct
import time
from fp_common import ROOT, OUT, read, write, entry, sha, local, module

GENERATOR=ROOT/'output/dynamic_sr_20260918/cook_spinach_pilot_v1_lr_integrated/source_snapshot/06_generate_prior.py'
GENERATOR_SHA='d7be23d15473082c57e1e4091bfea9eaafad0de97ef7379db39308ae14c6cbcc'
NETWORK=ROOT.parent/'mml/scripts/network_swinir.py'
WEIGHT=ROOT.parent/'mml/outputs/week3_swinir/ckpt/001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth'
NETWORK_SHA='1d650d2c1c4519d95db771863691c6b239a0822222f287f6a8342a9e0eccbc6e'
WEIGHT_SHA='4e78e33f22c1aa8a773db0cf4a7381bae97c2362c717f155439ebc690cbd9215'
FROZEN=dict(generator_sha256=GENERATOR_SHA,network_sha256=NETWORK_SHA,checkpoint_sha256=WEIGHT_SHA,
 scale=4,window=8,tile=0,overlap=32,precision='float32',padding='official_mirror_concat_next_window')

def external_entry(path):
    path=Path(path)
    try:return entry(path)
    except ValueError:return dict(path=str(path),sha256=sha(path))

def check_sources():
    for p,expected in ((GENERATOR,GENERATOR_SHA),(NETWORK,NETWORK_SHA),(WEIGHT,WEIGHT_SHA)):
        if sha(p)!=expected:raise ValueError(f'Frozen teacher source/weight changed: {p}')
    return module('fp_full_teacher_original_cpu_inventory',GENERATOR)

def png_size(path):
    with Path(path).open('rb') as f:header=f.read(29)
    if header[:8]!=b'\x89PNG\r\n\x1a\n' or header[12:16]!=b'IHDR':raise ValueError('Teacher not PNG')
    w,h=struct.unpack('>II',header[16:24])
    if header[24]!=8 or header[25]!=2:raise ValueError('Teacher must be uint8 RGB PNG')
    return [w,h]

def old_cache_root(scene):
    if scene.startswith('meetroom_'):return ROOT/'data/dynamic_sr/meetroom_prepared'/scene.removeprefix('meetroom_')
    return ROOT/'data/dynamic_sr/n3dv_prepared'/scene

def equivalent_consumer(original,producer_manifest,seed):
    p=producer_manifest.parent/f'manifest_train_ready_seed{seed}.json'
    if not p.is_file():return dict(seed=seed,status='pending_LR_point_initialization',expected_manifest_path=str(p.relative_to(ROOT)))
    m=read(p)
    if m['original_full_manifest']!=entry(producer_manifest):raise ValueError('Derived consumer bound to a different full manifest')
    for key in ('scene','splits','frame_indices','resolutions','cameras','observations'):
        if m[key]!=original[key]:raise ValueError('Teacher inputs/cameras/split changed in seed consumer')
    if m['initialization']['seed']!=seed:raise ValueError('Seed consumer mismatch')
    return dict(seed=seed,status='completed_LR_point_manifest_teacher_generation_pending',manifest=entry(p),
      producer_equivalence='exact scene/splits/frame_indices/resolutions/cameras/observations; initialization-specific metadata differs',
      future_teacher_index=str((OUT/'full_teacher_prepare'/original['scene']/f'teacher_index_seed{seed}.json').relative_to(ROOT)))

def plan(manifest,seeds=(20261007,20261008)):
    started=time.monotonic();manifest=local(manifest);m=read(manifest)
    if not m.get('full_time_decode_verified') or m['frame_indices']!=list(range(300)):
        raise ValueError('Verified full300-frame original manifest required')
    if 'original_full_manifest' in m:raise ValueError('Plan producer uses original full manifest; seed consumers are separately bound')
    if set(m['splits']['train'])&set(m['splits'].get('dev',[])+m['splits'].get('test',[])) or 'cam00' in m['splits']['train']:raise ValueError('Held-out teacher input')
    original=check_sources();_,inventory=original.load_inventory(manifest,None)
    rows=sorted((r for r in m['observations'] if r['split']=='train'),key=lambda r:(r['camera_id'],int(r['frame_index'])))
    keys=[(r['camera_id'],int(r['frame_index'])) for r in rows]
    expected={(c,f) for c in m['splits']['train'] for f in range(300)}
    if len(rows)!=len(expected) or set(keys)!=expected or len(inventory)!=len(rows):raise ValueError('Incomplete full legal teacher inventory')
    lr=m['resolutions']['lr'];hr=m['resolutions']['hr']
    if hr!=[4*x for x in lr]:raise ValueError('Teacher requires exact registered x4 dimensions')
    old=old_cache_root(m['scene']);old_config=old/'sr_swinir_x4/prior_config.json';eligible=False;old_binding=None
    if old_config.is_file():
        cfg=read(old_config)
        if all(cfg.get(k)==v for k,v in FROZEN.items()) and (old/'manifest.json').is_file() and sha(old/'manifest.json')==cfg['manifest_sha256']:
            eligible=True;old_binding=dict(manifest=entry(old/'manifest.json'),prior_config=entry(old_config))
    entries=[];reusable=0;unavailable_old=0;changed_LR=0
    for r,(relative,source) in zip(rows,inventory):
        if relative!=f'{r["camera_id"]}/{int(r["frame_index"]):04d}.png':raise ValueError('Original generator row ordering differs')
        if sha(source)!=r['lr_sha256']:raise ValueError('Registered full LR input changed')
        target=manifest.parent/'sr_swinir_x4'/relative
        item=dict(camera=r['camera_id'],frame=int(r['frame_index']),relative_path='sr_swinir_x4/'+relative,
          target_path=str(target.relative_to(ROOT)),receipt_path=str(target.with_suffix('.json').relative_to(ROOT)),
          lr_relative_path=r['lr_path'],lr_path=str(source.relative_to(ROOT)),lr_sha256=r['lr_sha256'],existing_full_cache=target.exists(),reuse=None)
        if target.exists()!=target.with_suffix('.json').exists():raise ValueError('Preserve incomplete existing full target/receipt pair')
        previous=old/'sr_swinir_x4'/relative;previous_receipt=previous.with_suffix('.json')
        if eligible and previous.is_file() and previous_receipt.is_file():
            receipt=read(previous_receipt)
            if receipt['input_sha256']==r['lr_sha256']:
                if sha(previous)!=receipt['output_sha256'] or png_size(previous)!=hr or receipt['input_hw']!=list(reversed(lr)) or receipt['output_hw']!=list(reversed(hr)):raise ValueError('Historical teacher reuse candidate corrupted')
                item['reuse']=dict(status='eligible_exact_LR_and_frozen_recipe_but_not_copied',teacher=entry(previous),receipt=entry(previous_receipt),original_input_sha256=receipt['input_sha256']);reusable+=1
            else:changed_LR+=1
        else:unavailable_old+=1
        entries.append(item)
    directory=OUT/'full_teacher_prepare'/m['scene'];directory.mkdir(parents=True,exist_ok=True)
    frozen=directory/'source_snapshot/generate_prior_original.py';frozen.parent.mkdir(exist_ok=True)
    if frozen.exists() and sha(frozen)!=GENERATOR_SHA:raise ValueError('Isolated frozen generator changed')
    if not frozen.exists():frozen.write_bytes(GENERATOR.read_bytes())
    command=['<root-selected-python>',str(frozen.relative_to(ROOT)),'--manifest',str(manifest.relative_to(ROOT)),
      '--checkpoint','<host-path-with-pinned-weight-SHA>','--network','<host-path-with-pinned-network-SHA>',
      '--dependency-path','experiments/dynamic_sr_20260918/vendor','--device','cuda:0','--tile','0','--overlap','32','--log-every','60']
    value=dict(schema=1,status='CPU_plan_completed_GPU_teacher_generation_not_started',scene=m['scene'],source=entry(Path(__file__)),
      producer_manifest=entry(manifest),seed_consumers=[equivalent_consumer(m,manifest,int(s)) for s in seeds],
      teacher_generator=entry(frozen),original_teacher_generator=entry(GENERATOR),network=external_entry(NETWORK),checkpoint=external_entry(WEIGHT),frozen_recipe=FROZEN,
      training_camera_ids=m['splits']['train'],cam01_legal_included='cam01' in m['splits']['train'],frame_indices=list(range(300)),
      train_observation_count=len(rows),expected_LR_dimensions=lr,expected_SR_dimensions=hr,entries=entries,
      output_directory=str((manifest.parent/'sr_swinir_x4').relative_to(ROOT)),
      historical_reuse=dict(eligible_images=reusable,old_cache_missing_images=unavailable_old,old_LR_sha_differs_images=changed_LR,binding=old_binding,
        performed=False,new_GPU_images_if_all_eligible_are_verified_and_copied=len(rows)-reusable,
        copy_policy='Optional root-owned CPU copy into this new full directory only; preserve exact PNG/receipt bytes and old source identities, then original generator validates each skip. Do not copy old prior_config because producer manifest differs.'),
      GPU_command_template=command,required_environment={'CUDA_VISIBLE_DEVICES':'<root-assigned-free-GPU-identifier>','FOURDSR_ROOT':'host project root; translate command paths, preserve source/data identity'},
      no_explicit_camera_subset=True,batch_size=1,full_image_no_tiling=True,no_autocast=True,TF32_matmul_and_cudnn=False,cudnn_benchmark=False,
      target_representation='Frozen original SwinIR output clipped[0,1], multiplied255/roundeduint8 RGB PNG. TeacherCache loads PNG to float32/255; this preserves historical training targets rather than introducing a different float-output teacher.',
      input_policy='Exactly all manifest train LR; no cam00/dev/test/HR pixels read',external_pretraining='SwinIR-M x4 DF2K (DIV2K+Flickr2K), not target fine-tuned',
      seed_independent_prior='Generate once using original full manifest. Per-seed consumer teacher indices bind derived seed manifests and prove identical legal LR entries; independent LR model prefix seeds remain separate.',
      old_wrapper_not_full_compatible='prepare_teachers.py inventory hard-excludes cam01, requires60frames and fixesoldscene paths/GPU3090. Reuse exact frozen standalone generator, not that short-window dispatcher.',
      completion_requirement='Original full generator summary selection_complete=true and all train images/receipts SHA/dimensions verified; CPU accept writes per-seed completed teacher index. No completed index until actual GPU generation/cache completion.',
      future_accept_command=['<CPU-python>','experiments/dynamic_sr_multiview_footprint_20261007/full_teacher_prepare.py','--accept','--plan',str((directory/'plan.json').relative_to(ROOT))],
      GPU_inference_executed=False,CUDA_calls=0,HR_quality_evaluation=False,confirmation_method_selection=False)
    target=directory/'plan.json'
    if target.exists() and read(target)!=value:raise ValueError('CPU teacher plan identity changed; preserve earlier plan for explicit amendment')
    if not target.exists():write(target,value)
    write(directory/f'planning_receipt_{time.time_ns()}.json',dict(status=value['status'],plan=entry(target),CPU_worker_wall_seconds=time.monotonic()-started,
      CUDA_calls=0,HR_image_reads=0,GPU_inference_executed=False,eligible_reused_images=reusable,reuse_copy_executed=False))
    return value

def accept(plan_path):
    """Verify already generated complete cache on CPU; never start inference."""
    plan_path=local(plan_path);p=read(plan_path);manifest=local(p['producer_manifest']['path']);m=read(manifest)
    if sha(manifest)!=p['producer_manifest']['sha256'] or sha(Path(__file__))!=p['source']['sha256']:raise ValueError('Frozen plan/source/manifest changed')
    check_sources();cache=manifest.parent/'sr_swinir_x4';config=cache/'prior_config.json';cfg=read(config)
    if cfg['manifest_sha256']!=p['producer_manifest']['sha256'] or any(cfg.get(k)!=v for k,v in FROZEN.items()):raise ValueError('Producer config changed')
    complete=[(q,read(q)) for q in sorted(cache.glob('summary_*.json'))]
    complete=[(q,r) for q,r in complete if r.get('selection_complete') and r.get('selected')==len(p['entries']) and r.get('train_inventory_selected')==len(p['entries']) and r.get('generated',0)+r.get('skipped',0)==len(p['entries'])]
    if not complete:raise ValueError('No successful full inventory generator completion summary')
    rows=[]
    for e in p['entries']:
        source=local(e['lr_path']);target=local(e['target_path']);receipt=local(e['receipt_path']);r=read(receipt)
        if sha(source)!=e['lr_sha256'] or r['input_sha256']!=e['lr_sha256'] or sha(target)!=r['output_sha256']:raise ValueError('Complete teacher/input identity failed')
        if png_size(target)!=p['expected_SR_dimensions'] or r['input_hw']!=list(reversed(p['expected_LR_dimensions'])) or r['output_hw']!=list(reversed(p['expected_SR_dimensions'])):raise ValueError('Teacher dimensions failed')
        rows.append(dict(camera=e['camera'],frame=e['frame'],relative_path=e['relative_path'],sha256=sha(target),lr_sha256=e['lr_sha256'],
          lr_relative_path=e['lr_relative_path'],receipt_sha256=sha(receipt),original_reuse=e['reuse']))
    indexes=[]
    for consumer in p['seed_consumers']:
        if 'manifest' not in consumer:raise ValueError('Seed LR initialization was pending at plan registration; amend plan explicitly before acceptance')
        cp=local(consumer['manifest']['path'])
        if sha(cp)!=consumer['manifest']['sha256']:raise ValueError('Seed consumer manifest changed')
        equivalent_consumer(m,manifest,consumer['seed'])
        value=dict(schema=1,status='completed_teacher_inventory',scene=p['scene'],seed_consumer=consumer['seed'],manifest=str(cp),manifest_sha256=sha(cp),
          producer_manifest=p['producer_manifest'],producer_config=entry(config),teacher_config=cfg,teacher_config_sha256=sha(config),
          producer_summary=entry(complete[-1][0]),plan=entry(plan_path),entries=rows,train_cameras=p['training_camera_ids'],frame_indices=p['frame_indices'],
          input_policy=p['input_policy'],seed_independent_frozen_prior=True,CUDA_calls_in_acceptance=0,HR_quality_evaluation=False)
        target=local(consumer['future_teacher_index'])
        if target.exists() and read(target)!=value:raise ValueError('Completed consumer teacher inventory differs')
        if not target.exists():write(target,value)
        indexes.append(entry(target))
    write(plan_path.parent/'acceptance_complete.json',dict(status='completed_full_teacher_CPU_acceptance',producer_manifest=p['producer_manifest'],indexes=indexes,
      teacher_images=len(rows),source=entry(Path(__file__)),CUDA_calls=0,GPU_inference_executed_by_this_program=False,HR_image_reads=0))
    return indexes

def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--manifest',type=Path);a.add_argument('--seeds',type=int,nargs='+',default=[20261007,20261008]);a.add_argument('--accept',action='store_true');a.add_argument('--plan',type=Path);args=a.parse_args()
    if args.accept:
        if not args.plan or args.manifest:a.error('--accept requires --plan and no --manifest')
        print(json.dumps(dict(status='completed_CPU_teacher_acceptance',indices=accept(args.plan))))
    else:
        if not args.manifest or args.plan:a.error('planning requires --manifest and no --plan')
        v=plan(args.manifest,args.seeds);print(json.dumps(dict(status=v['status'],scene=v['scene'],train_images=v['train_observation_count'],eligible_reuse=v['historical_reuse']['eligible_images'],CUDA_calls=0)))

if __name__=='__main__':main()
