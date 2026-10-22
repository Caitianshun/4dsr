"""StableSR main-family teacher bridge for the Wu + SplatSuRe adaptation.

Calls the pinned author's inference entry point unchanged. SplatSuRe identifies
StableSR but does not release its exact StableSR checkpoint/sampling flags;
this bridge explicitly selects StableSR's documented standard 512/DDPM200
recipe. It is not a claim of byte-identical SplatSuRe paper teacher outputs.
Planning/verification use the standard library, open only training LR bytes,
and do not download weights or import a GPU framework.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import time

ROOT = Path(__file__).resolve().parents[3]
STABLESR_COMMIT = '398ee9383777e255540ea027a704c8ce1f32145b'
DEFAULT_SOURCE = ROOT/'tmp/static_sr_wu_20261022/upstream/stablesr'
SCRIPT = 'scripts/sr_val_ddpm_text_T_vqganfin_oldcanvas_tile.py'
CONFIG = 'configs/stableSRNew/v2-finetune_text_T_512.yaml'
CFW_CONFIG = 'configs/autoencoder/autoencoder_kl_64x64x4_resi.yaml'
WEIGHTS = {
    'diffusion': ('stablesr_000117.ckpt', 'b8862bf3fd11c5b8fe82fb8a4618a1c74a29e0301a190bd6c2e84d68986ef9cb'),
    'cfw': ('vqgan_cfw_00011.ckpt', '5b20ed9a80e9bdbf9e76ff1642a6a5428ca3427b08b779248a88bfbba2e74e8e'),
}
RECIPE = 'SplatSuRe+StableSR_512_DDPM200_official_default_explicit_adaptation'
OPENCLIP_REVISION = '1c2b8495b28150b8a4922ee1c8edee224c284c0c'
OPENCLIP_CACHE_REPO = 'models--laion--CLIP-ViT-H-14-laion2B-s32B-b79K'
RUNTIME_SHA256 = {
    'openclip': '9a78ef8e8c73fd0df621682e7a8e8eb36c6916cb3c16b291a082ecd52ab79cc4',
    'lpips': 'a78928a0af1e5f0fcb1f3b9e8f8c3a2a5a3de244d830ad5c1feddc79b8432868',
    'vgg16': '397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0',
}
LPIPS_WORKDIR_PATH = 'taming/modules/autoencoder/lpips/vgg.pth'


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as file:
        for block in iter(lambda:file.read(1<<20),b''):digest.update(block)
    return digest.hexdigest()


def identity(path):return dict(path=str(Path(path).resolve()),sha256=sha(path))


def dump(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def validate_runtime_assets(assets):
    """Bind model bytes to the paths actually consulted by upstream loaders."""
    files=assets.get('files',[])
    models={item['role']:item for item in files}
    if len(models)!=len(files) or set(models)!=set(RUNTIME_SHA256):
        raise ValueError('runtime-assets needs exactly OpenCLIP, LPIPS and VGG16 model identities')
    for role,item in models.items():
        if item['sha256']!=RUNTIME_SHA256[role] or sha(item['path'])!=item['sha256']:
            raise ValueError('runtime cached model differs from the registered official bytes')
    env=assets.get('cache_environment',{})
    allowed={'TORCH_HOME','HF_HOME','XDG_CACHE_HOME','HUGGINGFACE_HUB_CACHE'}
    if set(env)-allowed or not {'TORCH_HOME','HUGGINGFACE_HUB_CACHE'}.issubset(env):
        raise ValueError('runtime cache environment must bind Torch and Hugging Face cache roots')
    if any(not Path(path).is_absolute() for path in env.values()):
        raise ValueError('runtime cache roots must be absolute')
    mappings=assets.get('cache_mappings',[])
    mapped={item['role']:item for item in mappings}
    if len(mapped)!=len(mappings) or set(mapped)!=set(models):
        raise ValueError('each runtime model requires one explicit cache mapping')
    hf_repo=Path(env['HUGGINGFACE_HUB_CACHE'])/OPENCLIP_CACHE_REPO
    expected={'openclip':hf_repo/'snapshots'/OPENCLIP_REVISION/'open_clip_pytorch_model.bin',
              'vgg16':Path(env['TORCH_HOME'])/'hub/checkpoints/vgg16-397923af.pth'}
    for role,item in mapped.items():
        if role in expected and Path(item['path']).absolute()!=expected[role].absolute():
            raise ValueError('cache mapping is not the upstream loader path')
        if item['sha256']!=models[role]['sha256'] or sha(item['path'])!=item['sha256']:
            raise ValueError('cache mapping hash differs from its registered model')
    ref=hf_repo/'refs/main'
    # Hugging Face's offline reader uses the ref bytes as the snapshot name.
    # A trailing newline is not stripped and would break the actual load.
    if ref.read_text()!=OPENCLIP_REVISION:
        raise ValueError('OpenCLIP offline revision ref changed')
    return mapped


def prepare_workdir(plan):
    """Stage only cache links/config references in a fresh task-local cwd."""
    mappings=validate_runtime_assets(plan['runtime_assets'])
    work=Path(plan['execution_workdir'])
    if work!=Path(plan['output_root'])/'execution_workdir':
        raise ValueError('execution working directory escaped the task output')
    work.mkdir(exist_ok=False)
    (work/'configs').symlink_to(Path(plan['source_root'])/'configs',target_is_directory=True)
    source=Path(mappings['lpips']['path'])
    target=work/LPIPS_WORKDIR_PATH
    target.parent.mkdir(parents=True)
    try:os.link(source,target)
    except OSError:shutil.copyfile(source,target)
    if sha(target)!=mappings['lpips']['sha256']:
        raise ValueError('staged LPIPS bytes changed')
    return work


def verify_workdir(plan):
    work=Path(plan['execution_workdir'])
    if work!=Path(plan['output_root'])/'execution_workdir':
        raise ValueError('execution working directory escaped the task output')
    if (work/'configs').resolve()!=(Path(plan['source_root'])/'configs').resolve():
        raise ValueError('working directory configs no longer refer to pinned source')
    if sha(work/LPIPS_WORKDIR_PATH)!=RUNTIME_SHA256['lpips']:
        raise ValueError('working directory LPIPS identity changed')


def source_check(source):
    source=Path(source).resolve()
    commit=subprocess.run(['git','-C',str(source),'rev-parse','HEAD'],check=True,capture_output=True,text=True).stdout.strip()
    if commit!=STABLESR_COMMIT:raise ValueError('StableSR source commit differs')
    if subprocess.run(['git','-C',str(source),'diff','--quiet','HEAD','--'],check=False).returncode:
        raise ValueError('StableSR tracked source changed')
    return source


def png_shape(path):
    with Path(path).open('rb') as file:header=file.read(29)
    if header[:8]!=b'\x89PNG\r\n\x1a\n' or header[12:16]!=b'IHDR' or header[24:26]!=bytes([8,2]):
        raise ValueError('expected an 8-bit RGB PNG')
    return list(struct.unpack('>II',header[16:24]))


def training_targets(manifest_path):
    manifest_path=Path(manifest_path).resolve()
    manifest=json.loads(manifest_path.read_text())
    if manifest.get('scale')!=4:raise ValueError('this teacher plan is x4 only')
    train=set(manifest['splits']['train'])
    held=set(manifest['splits'].get('test',[]))|set(manifest['splits'].get('dev',[]))
    if not train or train&held or 'cam00' in train:raise ValueError('invalid camera split')
    targets=[];seen=set()
    for row in manifest['observations']:
        if row['split']!='train':continue
        camera,frame=row['camera_id'],int(row['frame_index'])
        if camera not in train or (camera,frame) in seen:raise ValueError('foreign/duplicate training input')
        expected=Path('lr')/camera/f'{frame:04d}.png'
        if Path(row['lr_path'])!=expected:raise ValueError('unexpected training LR path')
        source=(manifest_path.parent/expected).resolve()
        if not source.is_relative_to(manifest_path.parent) or sha(source)!=row['lr_sha256']:
            raise ValueError('training LR identity changed')
        if png_shape(source)!=manifest['resolutions']['lr']:raise ValueError('training LR grid changed')
        seen.add((camera,frame))
        targets.append(dict(camera=camera,frame=frame,time=row['time'],split='train',lr_path=str(source),
                     lr_sha256=row['lr_sha256'],staged_name=f'{camera}_{frame:06d}.png',expected_sr_shape=manifest['resolutions']['hr']))
    if seen!={(c,f) for c in train for f in manifest['frame_indices']}:raise ValueError('incomplete training input set')
    return manifest,sorted(targets,key=lambda r:r['staged_name'])


def outer_tile_geometry(padded_shape_hw):
    """Select supported square autoencoder tiles without enlarging image content.

    StableSR README:179-180 permits tile >= 512 and stride < tile. Its
    ImageSpliterTh uses a square Gaussian weight even for a truncated patch.
    Cap only when the default chop branch would produce that invalid rectangle.
    The inner diffusion tile remains 512 pixels / 64 latent cells.
    """
    height,width=map(int,padded_shape_hw)
    if min(height,width)<512 or height%32 or width%32:
        raise ValueError('Expected preconditioned dimensions >=512 and divisible by32')
    tiled=max(height,width)>1280
    tile=min(1280,height,width) if tiled else 1280
    stride=min(1000,tile-32)
    return dict(schema='square_outer_tile_geometry_v1',padded_shape_hw=[height,width],
        tiled=tiled,requested_tile=1280,requested_stride=1000,tile=tile,stride=stride,
        policy='cap_square_tile_to_short_side_only_when_default_chop_required; stride=min(1000,tile-32)',
        adapted=tile!=1280,risk='Smaller autoencoder tiles may introduce border artifacts (StableSR README:180)')


def stablesr_geometry(input_shape_wh):
    """Predict the pinned author script's working grid, including its exact pad."""
    if len(input_shape_wh)!=2 or any(type(x) is not int or x<=0 for x in input_shape_wh):
        raise ValueError('Expected positive integer PNG width and height')
    width,height=input_shape_wh
    ratio=max(512/min(height,width),4)
    working=[int(height*ratio),int(width*ratio)]
    # The author pads BOTH axes to the next32 if either is unaligned. Preserve
    # this behavior: an already-aligned width can gain another32 in this branch.
    padded=[(n//32+1)*32 for n in working] if any(n%32 for n in working) else working[:]
    return dict(outer_tile_geometry(padded),input_shape_wh=list(input_shape_wh),
        working_shape_hw=working,target_shape_wh=[width*4,height*4],
        input_size=512,ddpm_steps=200,diffusion_tile_latent=64,diffusion_tile_overlap=32,
        padding_policy='unchanged_author_next32_both_axes_if_either_unaligned')


def command(plan):
    weights={item['role']:item['path'] for item in plan['model_identities']}
    geometry=stablesr_geometry(plan['input_shape_wh'])
    if 'tiling' in plan and plan['tiling']!=geometry:
        raise ValueError('Recorded tiling differs from the fixed input-geometry policy')
    return [plan['python'],str(Path(plan['source_root'])/SCRIPT),'--config',str(Path(plan['source_root'])/CONFIG),
        '--ckpt',weights['diffusion'],'--vqgan_ckpt',weights['cfw'],
        '--init-img',str(Path(plan['output_root'])/'inputs'),'--outdir',str(Path(plan['output_root'])/'generated'),
        '--ddpm_steps','200','--dec_w','0.5','--colorfix_type','adain','--upscale','4',
        '--seed',str(plan['seed']),'--n_samples','1','--input_size','512','--tile_overlap','32',
        '--vqgantile_size',str(geometry['tile']),'--vqgantile_stride',str(geometry['stride']),'--precision','autocast']


def plan(manifest_path,output_root,source=DEFAULT_SOURCE,weight_root=None,python=None,runtime_assets=None):
    source=source_check(source)
    manifest,targets=training_targets(manifest_path)
    out=Path(output_root).resolve();out.mkdir(parents=True,exist_ok=False)
    weight_root=Path(weight_root or ROOT/'weight/StableSR/official').resolve()
    # Preserve the venv launcher symlink; resolving it can bypass pyvenv.cfg.
    python=str(Path(python or ROOT/'output/static_sr_wu_20261022/runtime/stablesr_env/bin/python').absolute())
    models=[];missing=[]
    for role,(name,checksum) in WEIGHTS.items():
        path=weight_root/name
        item=dict(role=role,path=str(path),sha256=checksum,official_url=f'https://huggingface.co/Iceclear/StableSR/resolve/main/{name}')
        if not path.is_file():missing.append(str(path))
        elif sha(path)!=checksum:raise ValueError('StableSR model differs from official published SHA256')
        models.append(item)
    assets=None
    if runtime_assets:
        assets=json.loads(Path(runtime_assets).read_text())
        validate_runtime_assets(assets)
    else:missing.append('runtime-assets JSON: OpenCLIP ViT-H-14 laion2b_s32b_b79k, LPIPS/VGG16 assets and cache environment')
    if not Path(python).is_file():missing.append(python)
    value=dict(schema='wu_splatsure_stablesr_teacher_v1',status='planned_dependencies_missing' if missing else 'planned_runtime_unverified',
        method='splatsure',recipe=RECIPE,scene=manifest['scene'],manifest=identity(manifest_path),training_only=True,
        original_teacher_family='StableSR',exact_splatsure_teacher_configuration_published=False,
        recipe_source='StableSR README standard 512 DDPM200; upstream SplatSuRe gives no exact inference configuration',
        upstream_commit=STABLESR_COMMIT,source_root=str(source),output_root=str(out),
        execution_workdir=str(out/'execution_workdir'),python=python,seed=42,
        producer=[identity(__file__)]+[identity(source/p) for p in (SCRIPT,CONFIG,CFW_CONFIG,'environment.yaml','README.md')],
        model_identities=models,runtime_assets=assets,runtime_assets_identity=identity(runtime_assets) if runtime_assets else None,
        missing_dependencies=missing,parameter_updates=0,HR_images_read=0,heldout_images_read=0,targets=targets,
        input_shape_wh=manifest['resolutions']['lr'],tiling=stablesr_geometry(manifest['resolutions']['lr']))
    value['command']=command(value)
    dump(out/'plan.json',value)
    return value


def revalidate(plan):
    source_check(plan['source_root'])
    if sha(plan['manifest']['path'])!=plan['manifest']['sha256']:raise ValueError('manifest changed')
    if json.loads(Path(plan['manifest']['path']).read_text())['resolutions']['lr']!=plan['input_shape_wh']:
        raise ValueError('Planned tile input grid differs from the bound manifest')
    for item in plan['producer']+plan['model_identities']:
        if sha(item['path'])!=item['sha256']:raise ValueError('source or model identity changed')
    if plan['runtime_assets'] is None:raise ValueError('runtime cache identities missing')
    validate_runtime_assets(plan['runtime_assets'])
    source=plan.get('runtime_assets_identity')
    if source is None or sha(source['path'])!=source['sha256']:
        raise ValueError('runtime-assets registration changed')
    if json.loads(Path(source['path']).read_text())!=plan['runtime_assets']:
        raise ValueError('runtime-assets registration differs from the plan')
    for row in plan['targets']:
        if sha(row['lr_path'])!=row['lr_sha256']:raise ValueError('training LR changed')
    if command(plan)!=plan['command']:raise ValueError('inference command changed')


def accept(plan_path):
    plan_path=Path(plan_path);value=json.loads(plan_path.read_text());revalidate(value)
    verify_workdir(value)
    out=Path(value['output_root']);execution=json.loads((out/'execution.json').read_text())
    if execution['status']!='completed' or execution['returncode']!=0 or execution['plan']!=identity(plan_path):
        raise ValueError('no successful bound author-script execution')
    if (execution.get('execution_workdir')!=value['execution_workdir'] or
            execution.get('runtime_assets_identity')!=value['runtime_assets_identity']):
        raise ValueError('execution did not bind the isolated working directory and runtime caches')
    expected={r['staged_name'] for r in value['targets']}
    actual={p.name for p in (out/'generated').iterdir() if p.is_file()}
    if actual!=expected:raise ValueError('missing or foreign generated files')
    from PIL import Image
    for row in value['targets']:
        path=out/'generated'/row['staged_name']
        with Image.open(path) as image:
            image.verify()
        if png_shape(path)!=row['expected_sr_shape']:raise ValueError('generated image does not preserve registered HR grid')
        row['sr']=identity(path)
    value.update(status='completed',execution=identity(out/'execution.json'),plan=identity(plan_path))
    dump(out/'receipt.json',value)
    return value


def run(plan_path):
    plan_path=Path(plan_path).resolve();value=json.loads(plan_path.read_text());revalidate(value)
    visible=os.environ.get('CUDA_VISIBLE_DEVICES','')
    if not visible or visible=='-1' or ',' in visible:raise ValueError('requires one admitted visible GPU')
    workdir=prepare_workdir(value)
    verify_workdir(value)
    out=Path(value['output_root']);inputs=out/'inputs';inputs.mkdir(exist_ok=False)
    if (out/'generated').exists():raise FileExistsError('preserve previous outputs; use a fresh plan')
    for row in value['targets']:
        dst=inputs/row['staged_name']
        try:os.link(row['lr_path'],dst)
        except OSError:shutil.copyfile(row['lr_path'],dst)
        if sha(dst)!=row['lr_sha256']:raise ValueError('staged LR changed')
    env=os.environ.copy()
    for key,path in value['runtime_assets'].get('cache_environment',{}).items():
        if key not in ('TORCH_HOME','HF_HOME','XDG_CACHE_HOME','HUGGINGFACE_HUB_CACHE'):
            raise ValueError('unexpected runtime environment override')
        env[key]=path
    env.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    # PyTorch>=2.6 needs this to load the official, hash-verified Lightning ckpt.
    env['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD']='1'
    env['PYTHONPATH']=value['source_root']+os.pathsep+env.get('PYTHONPATH','')
    started=time.monotonic()
    with (out/'inference.log').open('x') as log:
        result=subprocess.run(value['command'],cwd=str(workdir),env=env,stdout=log,stderr=subprocess.STDOUT,check=False)
    execution=dict(status='completed' if result.returncode==0 else 'failed',returncode=result.returncode,
        plan=identity(plan_path),seconds=time.monotonic()-started,cuda_visible_devices=visible,
        command=value['command'],source_root=value['source_root'],execution_workdir=str(workdir),
        runtime_assets_identity=value['runtime_assets_identity'],extra_environment=dict(
            HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1'),parameter_updates=0)
    dump(out/'execution.json',execution)
    if result.returncode:raise RuntimeError('official StableSR inference failed; inspect inference.log')
    return accept(plan_path)


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='action',required=True)
    p=sub.add_parser('plan');p.add_argument('--manifest',required=True);p.add_argument('--out',required=True)
    p.add_argument('--source',default=DEFAULT_SOURCE);p.add_argument('--weight-root');p.add_argument('--python');p.add_argument('--runtime-assets')
    for action in ('run','accept'):sub.add_parser(action).add_argument('--plan',required=True)
    args=parser.parse_args()
    if args.action=='plan':result=plan(args.manifest,args.out,args.source,args.weight_root,args.python,args.runtime_assets)
    elif args.action=='run':result=run(args.plan)
    else:result=accept(args.plan)
    print(json.dumps(dict(status=result['status'],targets=len(result['targets']),missing_dependencies=result.get('missing_dependencies',[]))))

if __name__=='__main__':main()
