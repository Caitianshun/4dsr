"""Build a local CTS transfer/relocation inventory. Never contacts a host or runs CUDA.

Immutable dataset manifests and registered source configurations stay byte-identical.
Host-bound records/plans are explicitly regenerated on the destination instead of edited.
"""
from __future__ import annotations
import argparse, collections, hashlib, json, os, re, subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
REMOTE=Path('/home/cts/Project/4DSR/tasks/static_sr_wu_20261022')
SCENES=('Coffee07','Cook07','Cut07','Discussion07','Flame07','VRHeadset07')

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(8<<20),b''):h.update(part)
    return h.hexdigest()

def abs_paths(value,key='$'):
    if isinstance(value,dict):
        for k,v in value.items():yield from abs_paths(v,key+'.'+k)
    elif isinstance(value,list):
        for i,v in enumerate(value):yield from abs_paths(v,f'{key}[{i}]')
    elif isinstance(value,str) and value.startswith('/home/'):
        yield dict(field=key,value=value)

class Inventory:
    def __init__(self,remote,verify_pixels=False):
        self.remote=remote;self.verify_pixels=verify_pixels;self.files={};self.scenes=[];self.repos=[]
    def add(self,path,role,expected=None,hash_now=True):
        path=Path(path);path=path if path.is_absolute() else ROOT/path
        # Preserve path layout rather than following hardlinks/symlinks into a different source layout.
        rel=path.relative_to(ROOT)
        if not path.is_file():raise FileNotFoundError(path)
        if str(rel) in self.files:
            record=self.files[str(rel)]
            if role not in record['roles']:record['roles'].append(role)
            return record
        checksum=sha(path) if hash_now else expected
        if expected and checksum!=expected:raise ValueError(f'identity differs: {path}')
        record=dict(relative_path=str(rel),source=str(path),destination=str(self.remote/rel),bytes=path.stat().st_size,sha256=checksum,
                    hash_evidence='bytes_verified_now' if hash_now else 'immutable_manifest_expected_not_reread',roles=[role])
        if path.is_symlink():record['symlink_target']=str(path.resolve())
        self.files[str(rel)]=record;return record
    def tree(self,path,role,extensions=None,exclude=()):
        for p in sorted((ROOT/path).rglob('*')):
            if not p.is_file() or any(x in p.parts for x in ('__pycache__','.git','build','dist')):continue
            if any(x in p.parts or x in p.name for x in exclude):continue
            if extensions and p.suffix not in extensions:continue
            self.add(p,role)
    def scene(self,name):
        cfg_path=ROOT/'output/static_sr_wu_20261022/prepared_final'/name/'splatsure.json'
        cfg=json.loads(cfg_path.read_text());manifest=ROOT/cfg['manifest']['path'];m=json.loads(manifest.read_text())
        source=ROOT/cfg['source_configuration']['path'];source_data=json.loads(source.read_text())
        assert m['frame_indices']==list(range(300)) and m['splits']['test']==['cam00']
        assert source_data['data']['manifest']==cfg['manifest']
        mid=self.add(manifest,'immutable_manifest',cfg['manifest']['sha256'])
        sid=self.add(source,'immutable_source_configuration',cfg['source_configuration']['sha256'])
        self.add(manifest.parent/m['initialization']['npz_path'],'train_LR_initial_points',m['initialization']['npz_sha256'])
        rows=[o for o in m['observations'] if o['split']=='train'];evaluation=[o for o in m['observations'] if o['split']=='test']
        for row in rows:self.add(manifest.parent/row['lr_path'],'training_LR',row['lr_sha256'],self.verify_pixels)
        # Evaluation files are listed separately; no held-out image bytes are opened by this tool.
        for row in evaluation:self.add(manifest.parent/row['hr_path'],'heldout_HR_evaluation_only',row['hr_sha256'],False)
        for method in ('splatsure','iesrgs','sequence_matters','threedsr'):
            self.add(cfg_path.parent/(method+'.json'),'portable_registered_method_configuration')
        records=json.loads((cfg_path.parent/'training_records.json').read_text())
        self.scenes.append(dict(short=name,scene=m['scene'],train_count=len(rows),heldout_HR_count=len(evaluation),manifest=mid['relative_path'],source_configuration=sid['relative_path'],
            manifest_absolute_provenance=list(abs_paths(m)),source_configuration_absolute_fields=list(abs_paths(source_data)),
            records_regenerate_on_target=True,first_local_record=records['records'][0]['lr_path'],
            destination_record=str(self.remote/(manifest.parent/rows[0]['lr_path']).relative_to(ROOT))))
    def repo(self,name):
        path=ROOT/'tmp/static_sr_wu_20261022/upstream'/name
        head=subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip()
        url=subprocess.check_output(['git','-C',str(path),'remote','get-url','origin'],text=True).strip()
        dirty=subprocess.check_output(['git','-C',str(path),'status','--porcelain','--untracked-files=no'],text=True).strip()
        sub=subprocess.run(['git','-C',str(path),'submodule','status','--recursive'],text=True,capture_output=True)
        self.repos.append(dict(name=name,path=str(path.relative_to(ROOT)),destination=str(self.remote/path.relative_to(ROOT)),commit=head,url=url,
            tracked_status=dirty,submodule_status=sub.stdout.splitlines(),transport='preserve Git metadata and exact submodules or clone exact commits; git archive alone fails runtime commit checks'))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True);p.add_argument('--remote-root',default=str(REMOTE));p.add_argument('--verify-training-lr',action='store_true');p.add_argument('--stage',choices=['base','teacher','all'],default='all');p.add_argument('--existing-inventory');a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    if a.existing_inventory:
        origin=Path(a.existing_inventory);value=json.loads(origin.read_text())
        assert value['source_root']==str(ROOT) and value['schema']=='static_sr_cts_asset_inventory_v1'
        selected=[]
        for item in value['files']:
            if 'environment_rebuild_input_never_execute_local_pth_on_CTS' in item['roles']:continue
            teacher='generic_frozen_model' in item['roles']
            if (a.stage=='base' and teacher) or (a.stage=='teacher' and not teacher):continue
            selected.append(dict(item,path=item['relative_path'],destination=str(Path(a.remote_root)/item['relative_path'])))
        if a.stage in ('base','all'):
            for name in ['asset_inventory.py','sync_assets.py']:
                path=ROOT/'deployment/static_sr_cts_20261022'/name
                if path.is_file():
                    rel=str(path.relative_to(ROOT));selected.append(dict(path=rel,relative_path=rel,source=str(path),destination=str(Path(a.remote_root)/rel),bytes=path.stat().st_size,sha256=sha(path),hash_evidence='bytes_verified_now',roles=['deployment_tool']))
        value.update(files=selected,destination_root=a.remote_root,stage=a.stage,source_inventory={'path':str(origin.absolute()),'sha256':sha(origin)},total_listed_bytes=sum(i['bytes'] for i in selected))
        for repo in value['repositories']:repo['destination']=str(Path(a.remote_root)/repo['path'])
        value['required_environment']['FOURDSR_UPSTREAM']=str(Path(a.remote_root)/'third_party/4dgs_843d5ac_runtime')
        (out/'inventory.json').write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
        for name,select in [('training_and_prior_files.txt',lambda x:'heldout_HR_evaluation_only' not in x['roles']),('heldout_evaluation_files.txt',lambda x:'heldout_HR_evaluation_only' in x['roles'])]:
            (out/name).write_text(''.join(x['relative_path']+'\n' for x in selected if select(x)))
        print(json.dumps(dict(stage=a.stage,files=len(selected),bytes=value['total_listed_bytes'],out=str(out))))
        return
    if a.stage!='all':raise ValueError('Use --existing-inventory to create a phase from a verified full inventory')
    v=Inventory(Path(a.remote_root),a.verify_training_lr)
    for scene in SCENES:v.scene(scene)
    v.tree(Path('experiments/static_sr_wu_20261022'),'adaptation_source',extensions={'.py','.md','.txt','.json','.patch'})
    for name in ['rt.py','cullretrain.py']:v.add('experiments/dynamic_sr_cull_retrain_20261009/'+name,'Wu_backend_import_dependency')
    for name in ['fp_common.py','config.py','full_prefix_registered.py','full_native_protocol.py','full_refine_registered_v2.py','full_evaluate_registered_v2.py','losses.py']:
        v.add('experiments/dynamic_sr_multiview_footprint_20261007/'+name,'Wu_backend_import_dependency')
    v.add('experiments/dynamic_sr_20260918/n3dv_data.py','explicit_camera_LR_loader')
    v.tree(Path('experiments/dynamic_sr_20260918/vendor/timm'),'IE_SwinIR_explicit_timm_dependency',extensions={'.py','.json','.txt'})
    v.tree(Path('third_party/4dgs_843d5ac_runtime'),'pinned_Wu_runtime',extensions={'.py','.so','.json'},exclude=('hodome',))
    # Copy one canonical copy per generic model. HF/Torch/LPIPS aliases are recreated by path/hash on CTS.
    models=['weight/static_sr_wu_20261022/stablesr/stablesr_000117.ckpt','weight/static_sr_wu_20261022/stablesr/vqgan_cfw_00011.ckpt',
      'weight/static_sr_wu_20261022/stablesr/open_clip_pytorch_model.bin','weight/static_sr_wu_20261022/stablesr/runtime_cache/torch/hub/checkpoints/vgg16-397923af.pth',
      'weight/static_sr_wu_20261022/stablesr/runtime_cache/lpips/vgg.pth','weight/static_sr_wu_20261022/threedsr/stablesr_turbo.ckpt',
      'weight/static_sr_wu_20261022/sequence_matters/PSRT_Vimeo.pth','weight/static_sr_wu_20261022/sequence_matters/spynet_sintel_final-3d2a1287.pth',
      'weight/static_sr_wu_20261022/depth_anything_v2_large/depth_anything_v2_vitl.pth','third_party/swinir_frozen_20261010/001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth']
    for name in models:v.add(name,'generic_frozen_model')
    for name in ['splatsure','iesrgs','threedsr']:
        base=Path('output/static_sr_wu_20261022/runtime')/name
        for path in (ROOT/base).rglob('*'):
            if path.is_file() and path.suffix in ('.so','.py','.json') and 'source_cuda128' not in path.parts and not any(x in path.parts for x in ('build','__pycache__')):
                if path.suffix=='.py' and path.name!='__init__.py':continue
                v.add(path,'specialized_rasterizer_and_build_evidence')
    for name in ['stablesr','sequence']:
        base=Path('output/static_sr_wu_20261022/runtime')/(name+'_env')/'lib/python3.10/site-packages'
        for fname in ['sitecustomize.py','reuse_wu_torch.pth']:
            v.add(base/fname,'environment_rebuild_input_never_execute_local_pth_on_CTS')
        if name=='stablesr':v.add(base/'taming_pinned_namespace.pth','environment_rebuild_input_never_execute_local_pth_on_CTS')
        for fname in ['acceptance.json','pip_freeze.txt']:
            v.add(Path('output/static_sr_wu_20261022/runtime')/(name+'_env_audit')/fname,'local_environment_reference_only')
    for name in ['splatsure','splatsure-rasterizer','iesrgs','sequence_matters','threedsr','stablesr','taming-transformers','clip']:v.repo(name)
    files=sorted(v.files.values(),key=lambda x:x['relative_path']);counts=collections.Counter();sizes=collections.Counter()
    for item in files:
        role=item['roles'][0];counts[role]+=1;sizes[role]+=item['bytes']
    # Inspect CUDA fatbin targets without launching the CUDA runtime.
    architectures=[];cuobjdump=Path('/usr/local/cuda-12.8/bin/cuobjdump')
    if cuobjdump.is_file():
        for item in files:
            if item['relative_path'].endswith('.so'):
                r=subprocess.run([str(cuobjdump),'--list-elf',item['source']],capture_output=True,text=True)
                architectures.append(dict(path=item['relative_path'],returncode=r.returncode,architectures=sorted(set(re.findall(r'sm_\d+',r.stdout))),sm86_present='sm_86' in r.stdout))
    result=dict(schema='static_sr_cts_asset_inventory_v1',status='local_read_only_inventory_remote_unchecked',source_root=str(ROOT),destination_root=str(v.remote),
        immutable_manifests=True,training_LR_sha256_checked_now=a.verify_training_lr,heldout_pixels_opened=False,remote_commands_executed=0,GPU_calls=0,
        scenes=v.scenes,repositories=v.repos,files=files,counts=dict(counts),bytes_by_primary_role=dict(sizes),total_listed_bytes=sum(x['bytes'] for x in files),architectures=architectures,
        regenerate_on_CTS=['training_records.json and suite_index.json via prepare_suite register into a new CTS registration directory',
          'Sequence ALS receipts: local records include local absolute LR paths and their digest; rebuild from CTS records',
          'StableSR runtime assets/cache mappings and plan receipts: recreate paths, preserve original models/manifests and source audit',
          '3DSR runtime assets working directory/round records and all producer receipts',
          'venv pyvenv.cfg/bin launchers/.pth using admitted CTS base Python, never rsync local venv'],
        required_environment={'base':'Verify CTS Python3.10, torch2.7.1+cu128, torchvision0.22.1+cu128, NumPy1.26.4 before binary reuse','FOURDSR_UPSTREAM':str(v.remote/'third_party/4dgs_843d5ac_runtime'),'PYTHONDONTWRITEBYTECODE':'1','teacher_process_isolation':'StableSR and PSRT separate overlay envs and subprocesses; author ldm/basicsr package names collide'},
        excluded=['training-view HR images','held-out LR images','raw videos','existing project model checkpoints','scene-specific learned priors','old runtime_assets v1 and old plan v3/v4 execution on CTS','local venv trees','duplicate OpenCLIP aliases'],
        cautions=['pinned Wu runtime check rejects imports from CTS vendor/4dgs even if numerically identical; modules must resolve inside third_party/4dgs_843d5ac_runtime',
          'Wu runtime includes a historical metadata receipt and dormant dataset readers; only generic source/extension files listed, no HOI data or models',
          'Portable source_configuration means copy the single registered JSON and preserve hash, not its entire old experiment output directory',
          'Optional LPIPS AlexNet evaluation checkpoint is not downloaded by this tool; CTS evaluator cache must be checked before training completion callbacks',
          'All current .so contain sm86 according to local fatbin inspection where listed; this does not establish remote ABI or GPU correctness'])
    (out/'inventory.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    for name,select in [('training_and_prior_files.txt',lambda x:'heldout_HR_evaluation_only' not in x['roles']),('heldout_evaluation_files.txt',lambda x:'heldout_HR_evaluation_only' in x['roles'])]:
        (out/name).write_text(''.join(x['relative_path']+'\n' for x in files if select(x)))
    print(json.dumps(dict(status=result['status'],files=len(files),counts=dict(counts),bytes_by_primary_role=dict(sizes),total_bytes=result['total_listed_bytes']),indent=2))
if __name__=='__main__':main()
