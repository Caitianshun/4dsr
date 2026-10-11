"""Frozen untiled SwinIR teacher: CPU plan, GPU three-image cost, run, CPU accept.

The pinned original generator supports verified resumption. Only a completed
three-image cost probe with identical frozen identity may precede full generation.
"""
from __future__ import annotations
import argparse, os, runpy, subprocess, sys, time, traceback
from pathlib import Path
from common import *
NATIVE=ROOT/'experiments/dynamic_sr_native_20261010/teacher_native.py'
old=module('highres_teacher_constants',NATIVE)
FROZEN=old.FROZEN
GENERATOR=old.DEFAULT_GENERATOR
NETWORK=old.DEFAULT_NETWORK
CHECKPOINT=ROOT/'third_party/swinir_frozen_20261010/001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth'
def inventory(path):
    path=local(path);m=read(path)
    if path!=DATA/'prepared'/m['scene']/'manifest.json' or m['scene'] not in CONDITIONS or not m.get('full_time_decode_verified'):raise ValueError('Original fresh completed highres manifest required')
    train=m['splits']['train'];rows=sorted([o for o in m['observations'] if o['split']=='train'],key=lambda r:(r['camera_id'],r['frame_index']))
    if train[0]!='cam01' or len(rows)!=len(train)*300 or 'cam00' in train:raise ValueError('Training boundary differs')
    return path,m,rows
def plan(path,seeds):
    path,m,rows=inventory(path);entries=[];consumers=[]
    for seed in seeds:
        p=path.parent/f'manifest_train_ready_seed{seed}.json';c=read(p)
        if c['original_full_manifest']!=entry(path):raise ValueError('Consumer differs')
        consumers.append(dict(seed=seed,manifest=entry(p)))
    for r in rows:
        source=path.parent/r['lr_path']
        if sha(source)!=r['lr_sha256']:raise ValueError('Training LR changed')
        entries.append(dict(camera=r['camera_id'],frame=r['frame_index'],lr_relative_path=r['lr_path'],lr_sha256=r['lr_sha256'],relative_path=f'sr_swinir_x4/{r["camera_id"]}/{r["frame_index"]:04d}.png'))
    value=dict(schema='highres_teacher_plan_v1',producer_manifest=entry(path),scene=m['scene'],entries=entries,seed_consumers=consumers,frozen_recipe=FROZEN,generator=entry(GENERATOR),network=entry(NETWORK),checkpoint=dict(path=str(CHECKPOINT.relative_to(ROOT)),sha256=FROZEN['checkpoint_sha256']),source=entry(__file__),train_cameras=m['splits']['train'],input_policy='all train LR only; test/dev/HR never opened',expected_LR_dimensions=m['resolutions']['lr'],expected_SR_dimensions=m['resolutions']['hr'])
    return write(REGISTRY/'teachers'/m['scene']/'plan.json',value)
def bound_plan(path):
    p=read(local(path));producer=local(p['producer_manifest']['path'])
    if sha(producer)!=p['producer_manifest']['sha256'] or p['frozen_recipe']!=FROZEN:raise ValueError('Frozen plan changed')
    for key,pin in [('generator','generator_sha256'),('network','network_sha256'),('checkpoint','checkpoint_sha256')]:
        if sha(local(p[key]['path']))!=FROZEN[pin]:raise ValueError('Pinned teacher artifact differs')
    return p,producer
def generate(planpath,cost=False):
    p,producer=bound_plan(planpath);disk_guard(producer.parent,35)
    cache=producer.parent/'sr_swinir_x4';costpath=local(planpath).parent/'teacher_cost.json'
    if not cost:
        if not costpath.exists():raise ValueError('Full teacher requires completed three-image cost probe before generation')
        receipt=read(costpath)
        if receipt.get('status')!='completed_three_image_untiled_teacher_cost' or receipt.get('plan')!=entry(planpath) or receipt.get('frozen_recipe')!=FROZEN or receipt.get('images')!=3 or receipt.get('tile')!=0:
            raise ValueError('Three-image cost probe is incomplete or belongs to a different frozen plan')
    if cost and cache.exists():raise FileExistsError('Cost probe requires absent teacher cache')
    argv=[str(local(p['generator']['path'])),'--manifest',str(producer),'--checkpoint',str(local(p['checkpoint']['path'])),'--network',str(local(p['network']['path'])),'--dependency-path',str(ROOT/'experiments/dynamic_sr_20260918/vendor'),'--device','cuda:0','--tile','0','--overlap','32','--log-every','60']
    if cost:argv+=['--limit','3']
    if cost:
        import torch
        started=time.monotonic();torch.cuda.reset_peak_memory_stats();sys.argv=argv;runpy.run_path(argv[0],run_name='__main__');torch.cuda.synchronize()
        write(costpath,dict(status='completed_three_image_untiled_teacher_cost',seconds=time.monotonic()-started,peak_allocated_GB=torch.cuda.max_memory_allocated()/1e9,peak_reserved_GB=torch.cuda.max_memory_reserved()/1e9,gpu=torch.cuda.get_device_name(),images=3,tile=0,plan=entry(planpath),frozen_recipe=FROZEN))
    else:
        subprocess.run([sys.executable,'-u',*argv],check=True,cwd=ROOT,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
    return p
def accept(planpath):
    p,producer=bound_plan(planpath);cache=producer.parent/'sr_swinir_x4';cfg=read(cache/'prior_config.json')
    if cfg['manifest_sha256']!=p['producer_manifest']['sha256'] or any(cfg.get(k)!=v for k,v in FROZEN.items()):raise ValueError('Producer config differs')
    good=[q for q in sorted(cache.glob('summary_*.json')) if read(q).get('selection_complete') is True and read(q).get('selected')==len(p['entries'])]
    if not good:raise ValueError('No full generation completion summary')
    rows=[]
    for e in p['entries']:
        target=producer.parent/e['relative_path'];receipt=target.with_suffix('.json');r=read(receipt)
        if sha(target)!=r['output_sha256'] or sha(producer.parent/e['lr_relative_path'])!=e['lr_sha256'] or r['input_sha256']!=e['lr_sha256']:raise ValueError('Teacher image identity differs')
        if old.png_size(target)!=p['expected_SR_dimensions'] or r['input_hw']!=p['expected_LR_dimensions'][::-1] or r['output_hw']!=p['expected_SR_dimensions'][::-1]:raise ValueError('Teacher dimensions differ')
        rows.append(dict(e,sha256=sha(target),receipt_sha256=sha(receipt),original_reuse=None))
    refs=[]
    for c in p['seed_consumers']:
        consumer=local(c['manifest']['path'])
        if sha(consumer)!=c['manifest']['sha256']:raise ValueError('Seed consumer changed')
        value=dict(schema=1,status='completed_teacher_inventory',scene=p['scene'],seed_consumer=c['seed'],manifest=str(consumer),manifest_sha256=sha(consumer),producer_manifest=p['producer_manifest'],producer_config=entry(cache/'prior_config.json'),teacher_config=cfg,teacher_config_sha256=sha(cache/'prior_config.json'),producer_summary=entry(good[-1]),plan=entry(planpath),entries=rows,train_cameras=p['train_cameras'],frame_indices=list(range(300)),input_policy=p['input_policy'],seed_independent_frozen_prior=True,CUDA_calls_in_acceptance=0,HR_quality_evaluation=False)
        refs.append(entry(write(REGISTRY/'teachers'/p['scene']/f'teacher_index_seed{c["seed"]}.json',value)))
    write(local(planpath).parent/'acceptance_complete.json',dict(status='completed_full_teacher_CPU_acceptance',producer_manifest=p['producer_manifest'],indexes=refs,teacher_images=len(rows),source=entry(__file__),CUDA_calls=0));return refs
def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True);pp=sub.add_parser('plan');pp.add_argument('--manifest',required=True,type=Path);pp.add_argument('--seeds',nargs='+',type=int,default=[20261007])
    for cmd in ['cost','generate','run','accept']:
        q=sub.add_parser(cmd);q.add_argument('--plan',required=True,type=Path)
    a=p.parse_args()
    try:
        if a.command=='plan':print(plan(a.manifest,a.seeds))
        elif a.command=='accept':print(accept(a.plan))
        else:
            generate(a.plan,a.command=='cost')
            if a.command=='run':print(accept(a.plan))
    except BaseException:
        if hasattr(a,'plan'):write(local(a.plan).parent/f'failure_{time.time_ns()}.json',dict(status='failed_teacher_preserved',error=traceback.format_exc(),source=entry(__file__)))
        raise
if __name__=='__main__':main()
