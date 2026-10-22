"""Prepare the six-method bicubic completion suite without submitting any work.

This entry point uses the standard library only. It inventories existing data,
freezes train/test image identities, and records missing work. It never imports
CUDA libraries, runs a training entry point, contacts hosts, or creates a queue.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SCENES = [('cook_spinach','Cook07','N3DV'), ('cut_roasted_beef','Cut07','N3DV'),
          ('coffee_martini','Coffee07','N3DV'), ('flame_steak','Flame07','N3DV'),
          ('meetroom_discussion','Discussion07','MeetRoom'), ('meetroom_vrheadset','VRHeadset07','MeetRoom')]
METHODS = ['wu','yang','4c4d','multi4d','localdygs','cem']
TRAIN_FRAMES = list(range(0,300,10))
TEST_FRAMES = list(range(300))
LPIPS_STATE_SHA256 = 'cd2943aa26b6c7e7d100ab7e09a6c0975653ff57be42499de2ab48991fff82de'
# One preserved A100 report used this explicit native-method display name.
# Keep the historical report immutable and recognize only this exact record.
METRIC_METHOD_NAMES = {('cem','cut_roasted_beef'):'CEM_native_original_LR'}

def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as s:
        for block in iter(lambda:s.read(2<<20),b''):h.update(block)
    return h.hexdigest()

def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(path)

def project_path(path) -> Path:
    p=Path(path);p=p if p.is_absolute() else ROOT/p
    if not p.resolve().is_relative_to(ROOT.resolve()):raise ValueError('Outside project: '+str(p))
    return p

def select_observations(manifest: dict, scope: str) -> list[dict]:
    if manifest.get('physical_lr_derivation'):raise ValueError('This suite is registered bicubic LR only')
    if manifest['frame_indices']!=TEST_FRAMES:raise ValueError('Expected complete frames 0..299')
    if manifest['splits']['test']!=['cam00'] or 'cam00' in manifest['splits']['train']:raise ValueError('Invalid held-out camera split')
    frames=set(TRAIN_FRAMES if scope=='train' else TEST_FRAMES)
    rows=sorted((o for o in manifest['observations'] if o['split']==scope and o['frame_index'] in frames),key=lambda o:(o['camera_id'],o['frame_index']))
    expected={(c,f) for c in manifest['splits'][scope] for f in frames}
    actual={(o['camera_id'],o['frame_index']) for o in rows}
    if actual!=expected or len(rows)!=len(expected):raise ValueError('Incomplete or duplicate observations')
    if any(abs(o['time']-o['frame_index']/300.)>1e-10 for o in rows):raise ValueError('Time must be original frame / 300')
    return rows

def freeze_scene(name, label, dataset, out: Path, verify: str) -> dict:
    mp=ROOT/'data/dynamic_sr/full_time_20261007/prepared'/name/'manifest_train_ready_seed20261007.json'
    m=json.loads(mp.read_text()); selected={scope:select_observations(m,scope) for scope in ['train','test']}
    train_all=[o for o in m['observations'] if o['split']=='train']
    if len(train_all)!=len(m['splits']['train'])*300:raise ValueError('Incomplete training pool')
    assets={};checked=0
    def asset(path, digest, role, sample=False):
        nonlocal checked
        p=project_path(path);rel=str(p.relative_to(ROOT))
        if not p.is_file():raise FileNotFoundError(p)
        if not isinstance(digest,str) or len(digest)!=64:raise ValueError('Missing registered SHA256: '+rel)
        if verify=='all' or (verify=='sample' and sample):
            if sha(p)!=digest:raise ValueError('SHA256 mismatch: '+rel)
            checked+=1
        prev=assets.get(rel)
        if prev and prev['sha256']!=digest:raise ValueError('Conflicting identity: '+rel)
        assets[rel]={'sha256':digest,'bytes':p.stat().st_size,'role':role,'content_verified':verify=='all' or (verify=='sample' and sample)}
    for o in train_all:
        asset(mp.parent/o['lr_path'],o['lr_sha256'],'training_LR_only',o['frame_index'] in [0,150,299])
    for scope, rows in selected.items():
        for o in rows:asset(mp.parent/o['hr_path'],o['hr_sha256'],scope+'_HR_scoring_only',o['frame_index'] in [0,150,290])
        view_rows=[{'camera_id':o['camera_id'],'frame':o['frame_index'],'time':o['time'],'reference_path':str((mp.parent/o['hr_path']).relative_to(ROOT)),
                    'reference_sha256':o['hr_sha256'],'render_path':f"{o['camera_id']}/{o['frame_index']:04d}.png"} for o in rows]
        dump(out/'scenes'/name/(scope+'_views.json'),{'schema':'dynamic_lr_expected_views_v1','status':'planned_not_rendered','scope':scope,
             'manifest':str(mp.relative_to(ROOT)),'manifest_sha256':sha(mp),'resolution':m['resolutions']['hr'],
             'cameras':m['splits'][scope],'frames':TRAIN_FRAMES if scope=='train' else TEST_FRAMES,'count':len(rows),
             'HR_use':'scoring_after_prediction_only','records':view_rows})
    init=m['initialization']
    if not init['train_only_lr']:raise ValueError('Initialization must be training-LR only')
    for key in ['npz','ply','report']:
        asset(mp.parent/init[key+'_path'],init[key+'_sha256'],'registered_LR_triangulation_optional_per_method',True)
    asset(mp,sha(mp),'metadata_manifest',True)
    dump(out/'scenes'/name/'assets.json',{'schema':'dynamic_lr_preparation_assets_v1','files':assets,'verified_mode':verify,
         'content_verified_count':checked,'all_paths_exist':True,'total_bytes':sum(v['bytes'] for v in assets.values()),
         'not_included':'held-out LR and training-HR pixels for training; author-native initialization may still need preparation'})
    dump(out/'scenes'/name/'training_inputs.json',{'schema':'dynamic_lr_training_allowlist_v1','manifest':str(mp.relative_to(ROOT)),
         'cameras':m['splits']['train'],'frames':TEST_FRAMES,'HR_pixel_inputs':False,'cam00_pixel_inputs':False,
         'files':{p:d for p,d in assets.items() if d['role']=='training_LR_only'}})
    return {'scene':name,'label':label,'dataset':dataset,'manifest':str(mp.relative_to(ROOT)),'manifest_sha256':sha(mp),
            'train_cameras':m['splits']['train'],'test_cameras':m['splits']['test'],'resolutions':m['resolutions'],
            'training_LR_images':len(train_all),'train_HR_metric_images':len(selected['train']),'test_HR_metric_images':len(selected['test']),
            'asset_count':len(assets),'sample_hashes_verified':checked,'inventory':str((out/'scenes'/name/'assets.json').relative_to(ROOT)),
            'initialization':init,'time_convention':m['time_convention'],'degradation':m['official_policy']['degradation']}

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,default=ROOT/'output/dynamic_lr_expansion_20261022/preparation')
    ap.add_argument('--verify',choices=['none','sample','all'],default='sample',help='all is mandatory on synchronized inputs before future execution')
    args=ap.parse_args();out=project_path(args.out);out.mkdir(parents=True,exist_ok=True)
    scenes=[freeze_scene(*s,out,args.verify) for s in SCENES]
    comparison=ROOT/'output/dynamic_lr_baselines_20261020/comparison/comparison.json'
    old=json.loads(comparison.read_text());reused={(r['method'],r['scene']):r for r in old['records'] if r['status']=='completed_verified_returned'}
    wi=ROOT/'output/dynamic_sr_multiview_footprint_20261007/full_LR_prefixes/checkpoint_index.json'
    wu={r['scene']:r for r in json.loads(wi.read_text())['entries'] if r['seed']==20261007 and r['status']=='completed_independent_full_author_LR_prefix'}
    tasks=[]
    for scene in scenes:
        for method in METHODS:
            name=scene['scene'];prior=reused.get((method,name));wr=wu.get(name) if method=='wu' else None
            checkpoints=prior['checkpoints'] if prior else [wr['checkpoint']] if wr else []
            for ck in checkpoints:
                if not project_path(ck['path']).is_file():raise FileNotFoundError(ck['path'])
            if prior:
                p=project_path(prior['metric_identity']['path'])
                if sha(p)!=prior['metric_identity']['sha256']:raise ValueError('Existing metric changed: '+str(p))
                score=json.loads(p.read_text())
                expected={'status':'completed','method':METRIC_METHOD_NAMES.get((method,name),method),'scene':name,'camera':'cam00','frames':300,
                    'resolution':scene['resolutions']['hr'],'manifest_sha256':scene['manifest_sha256'],
                    'helper_source_sha256':sha(ROOT/'experiments/dynamic_sr_20260918/evaluate.py'),
                    'lpips_state_dict_sha256':LPIPS_STATE_SHA256,
                    'input_representation':'8-bit rendered RGB PNG for every method'}
                if any(score.get(k)!=v for k,v in expected.items()):raise ValueError('Existing metric protocol mismatch: '+str(p))
            tasks.append({'method':method,'scene':name,'dataset':scene['dataset'],'input':'bicubic','status':'HELD_NO_EXECUTION',
                'execution_enabled':False,'training':'reuse_completed' if checkpoints else 'missing_author_training',
                'checkpoints':checkpoints,'test_metrics':'reuse_completed' if prior else 'missing_unified_test_metrics',
                'existing_test_metric':prior['metric_identity'] if prior else None,'train_metrics':'missing_fixed_30_frame_training_HR_metrics',
                'evaluation_scopes':{'train':{'view_list':str((out/'scenes'/name/'train_views.json').relative_to(ROOT)),'count':scene['train_HR_metric_images']},
                                     'test':{'view_list':str((out/'scenes'/name/'test_views.json').relative_to(ROOT)),'count':300}},
                'completion_order':['author_initialization_if_missing','author_training_if_missing','render_test_if_missing','score_test_if_missing','render_training_views','score_training_HR','verify_collect_and_update_tables'],
                'before_execution_required':['user_resumes_this_held_work','fresh_per_GPU_owner_lock_and_capacity_admission','per_host_full_file_SHA256_verification','author_runtime_camera_and_final_endpoint_acceptance'],
                'do_not_use':['test_LR_or_HR_for_training','training_HR_for_training','SR_teacher','w4','colour_affine','posthoc_visibility_cull']})
    summary={'schema':'dynamic_lr_six_scene_preparation_v1','created_utc':datetime.now(timezone.utc).isoformat(),
       'user_selection':'Continue existing bicubic; complete remaining four scenes; no training now because Claude owns GPU work.',
       'status':'HELD_PREPARATION_ONLY','execution_enabled':False,'watchers_created':False,'remote_jobs_submitted':0,'GPU_work_started':0,
       'scenes':scenes,'task_count':len(tasks),'reuse_training_count':sum(t['training']=='reuse_completed' for t in tasks),
       'missing_training_count':sum(t['training']=='missing_author_training' for t in tasks),
       'reuse_test_metrics_count':sum(t['test_metrics']=='reuse_completed' for t in tasks),
       'missing_test_metrics_count':sum(t['test_metrics']!='reuse_completed' for t in tasks),'missing_train_metrics_count':len(tasks),
       'protocol':{'input':'registered bicubic antialias LR','time':'frame_index / 300','train_metric_frames':TRAIN_FRAMES,'test_metric_frames':TEST_FRAMES,
         'train_metric_is_full_training_pool':False,'metric_representation':'uint8_RGB_PNG_for_all_methods','metric_names':['PSNR','SSIM','LPIPS'],
         'no_target_camera_input':True,'no_cross_scene_model_transfer':True,'no_automatic_start_after_GPU_becomes_idle':True,
         'metric_helper_sha256':sha(ROOT/'experiments/dynamic_sr_20260918/evaluate.py'),'lpips_state_dict_sha256':LPIPS_STATE_SHA256},
       'data_verification_mode':args.verify,'all_transfer_inputs_require_full_hash_verification_before_execution':True,
       'source_identities':{str(comparison.relative_to(ROOT)):sha(comparison),str(wi.relative_to(ROOT)):sha(wi)},'tasks':tasks}
    dump(out/'suite_plan.json',summary)
    dump(out/'HOLD.json',{'status':'HELD_BY_USER','execution_enabled':False,'user_resumption_required':True,
         'reason':'Current GPUs are executing Claude work. Preparation only. No watcher or autorun was installed.'})
    print(json.dumps({k:summary[k] for k in ['status','task_count','reuse_training_count','missing_training_count','reuse_test_metrics_count','missing_test_metrics_count','missing_train_metrics_count','GPU_work_started']}))

if __name__=='__main__':main()
