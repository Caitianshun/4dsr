"""Build relocated teacher overlays/cache registrations in an isolated project snapshot.

Controller executes this file on CTS after verified additive transfer. No SSH,
package downloads, model loading, inference, CUDA initialization or training.
Existing manifests, source configurations, upstream sources and records remain
unchanged; destination records and cache registrations are new derived files.
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, os, shutil, subprocess, sys, time, traceback
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
RUNTIME=ROOT/'output/static_sr_wu_20261022/runtime'
BASELINE=ROOT/'experiments/static_sr_wu_20261022'
CPU_ENV={'CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1','MPLBACKEND':'Agg',
         'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()

def identity(path):
    p=Path(path).absolute();return dict(path=str(p),sha256=sha(p))

def write_exact(path,text):
    """Allow identical retry; never overwrite conflicting deployed content."""
    path=Path(path)
    if path.exists():
        if path.read_text()!=text:raise FileExistsError(f'Conflicting existing file preserved: {path}')
        return False
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:f.write(text)
    return True

def write_json(path,value):
    return write_exact(path,json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')

def alias_file(source,target,expected):
    source,target=Path(source),Path(target)
    if sha(source)!=expected:raise ValueError(f'Source bytes differ: {source}')
    if target.exists():
        if sha(target)!=expected:raise FileExistsError(f'Conflicting cache alias: {target}')
        return
    target.parent.mkdir(parents=True,exist_ok=True)
    try:os.link(source,target)
    except OSError:shutil.copyfile(source,target)
    if sha(target)!=expected:raise ValueError(f'Cache alias changed: {target}')

def load_bridge():
    path=BASELINE/'methods/splatsure_stablesr.py'
    spec=importlib.util.spec_from_file_location('cts_splatsure_stablesr_bridge',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def cpu_subprocess_environment(base_python):
    """Match run_job's private library prefix without changing shared installs.

    Keep the venv launcher pathname: resolving its symlink can select /usr/bin
    and lose the project root. Only subprocess environments are modified.
    """
    launcher=Path(base_python).absolute()
    if launcher.parent.name!='bin':raise ValueError('Base Python must be a project venv bin/python launcher')
    project=launcher.parents[2];libraries=project/'tools/runtime-libs'
    required=libraries/'libBLT.2.5.so.8.6'
    if not required.is_file():raise FileNotFoundError(f'Private Tk/BLT runtime library missing: {required}')
    env=dict(os.environ,**CPU_ENV)
    env.setdefault('CUDA_HOME',str(project/'tools/cuda-12.8-minimal'))
    inherited=env.get('LD_LIBRARY_PATH','')
    env['LD_LIBRARY_PATH']=':'.join([str(libraries),env['CUDA_HOME']+'/lib64',inherited])
    record=dict(base_python_launcher=str(launcher),project_root=str(project),
        private_library_directory=str(libraries),required_libraries=[dict(identity(required),bytes=required.stat().st_size)],
        CUDA_HOME=env['CUDA_HOME'],LD_LIBRARY_PATH=env['LD_LIBRARY_PATH'],
        inherited_LD_LIBRARY_PATH=inherited,CPU_ENV=dict(CPU_ENV),shared_environment_modified=False)
    return env,record

def query_base(python,env=None):
    code="""import json,sys,sysconfig,torch,torchvision,numpy
assert not torch.cuda.is_initialized()
print(json.dumps(dict(python=sys.executable,prefix=sys.prefix,purelib=sysconfig.get_paths()['purelib'],python_version=list(sys.version_info[:3]),torch=torch.__version__,torchvision=torchvision.__version__,numpy=numpy.__version__,CUDA_initialized=torch.cuda.is_initialized())))"""
    env=dict(os.environ if env is None else env,**CPU_ENV)
    r=subprocess.run([str(python),'-c',code],env=env,text=True,capture_output=True,check=True)
    value=json.loads(r.stdout)
    if value['python_version'][:2]!=[3,10] or value['torch']!='2.7.1+cu128' or value['torchvision']!='0.22.1+cu128' or value['numpy']!='1.26.4':
        raise RuntimeError(f'Unaccepted base ABI/version: {value}')
    return value

def overlay(python,base,name,env=None):
    directory=RUNTIME/(name+'_env');site=directory/'lib/python3.10/site-packages'
    if not site.is_dir():raise FileNotFoundError(f'Transfer overlay site-packages first: {site}')
    if not (site/'sitecustomize.py').is_file():raise FileNotFoundError('Exact-function torchvision compatibility alias missing')
    if not (directory/'pyvenv.cfg').exists():
        subprocess.run([str(python),'-m','venv','--copies','--without-pip',str(directory)],env=dict(os.environ if env is None else env,**CPU_ENV),check=True)
    config=(directory/'pyvenv.cfg').read_text()
    if 'include-system-site-packages = false' not in config:raise RuntimeError('Unexpected venv system package access')
    write_exact(site/'reuse_wu_torch.pth',base['purelib']+'\n')
    # Ubuntu's stdlib sitecustomize.py can precede the task overlay on sys.path.
    # Install only the exact-function alias through a .pth startup hook instead
    # of relying on which unrelated sitecustomize module Python finds first.
    write_exact(site/'static_sr_torchvision_compat.pth',
        'import runpy; runpy.run_path('+repr(str(site/'sitecustomize.py'))+
        ", run_name='static_sr_torchvision_compat')\n")
    if name=='stablesr':write_exact(site/'taming_pinned_namespace.pth',str(ROOT/'tmp/static_sr_wu_20261022/upstream/taming-transformers')+'\n')
    return directory/'bin/python'

def runtime_assets(template_path,out):
    """Only operational paths are new. Original_source provenance is retained verbatim."""
    b=load_bridge();template=json.loads(Path(template_path).read_text());models={x['role']:x for x in template['files']}
    root=ROOT/'weight/static_sr_wu_20261022/stablesr';cache=root/'cts_runtime_cache';hf=cache/'hf/hub';th=cache/'torch'
    canonical={'openclip':root/'open_clip_pytorch_model.bin','vgg16':root/'runtime_cache/torch/hub/checkpoints/vgg16-397923af.pth','lpips':root/'runtime_cache/lpips/vgg.pth'}
    targets={'openclip':hf/b.OPENCLIP_CACHE_REPO/'snapshots'/b.OPENCLIP_REVISION/'open_clip_pytorch_model.bin','vgg16':th/'hub/checkpoints/vgg16-397923af.pth','lpips':cache/'lpips/vgg.pth'}
    mappings=[]
    for old in template['cache_mappings']:
        role=old['role'];expected=models[role]['sha256']
        if expected!=b.RUNTIME_SHA256[role]:raise ValueError('Unrecognized generic model identity')
        alias_file(canonical[role],targets[role],expected)
        item=dict(old);item.update(identity(targets[role]));item['original_source']=old.get('original_source',dict(models[role]));mappings.append(item)
    ref=hf/b.OPENCLIP_CACHE_REPO/'refs/main';write_exact(ref,b.OPENCLIP_REVISION)
    value=dict(template);value.update(status='CTS_cache_paths_and_bytes_verified_GPU_unverified',files=[dict(identity(canonical[r]),role=r) for r in ('openclip','vgg16','lpips')],cache_mappings=mappings,
        cache_metadata=[dict(identity(ref),role='openclip_offline_revision')],cache_environment={'HF_HOME':str(cache/'hf'),'HUGGINGFACE_HUB_CACHE':str(hf),'TORCH_HOME':str(th)},
        origin_registration=identity(template_path),relocation=dict(destination_root=str(ROOT),origin_paths_are_provenance=True),GPU_calls=0,teacher_images_generated=0)
    b.validate_runtime_assets(value)
    target=Path(out)/'runtime_assets_cts.json';write_json(target,value);return target

def check(kind,assets_path):
    """Subprocess-only check: guard checkpoint loading and CUDA initialization."""
    os.environ.update(CPU_ENV)
    assets=json.loads(Path(assets_path).read_text());os.environ.update(assets['cache_environment']);os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',FOURDSR_UPSTREAM=str(ROOT/'third_party/4dgs_843d5ac_runtime'))
    import torch,importlib
    if torch.cuda.is_initialized():raise RuntimeError('CUDA initialized before checks')
    forbidden=[]
    def reject(*args,**kwargs):
        forbidden.append('CUDA initialization/checkpoint load/download');raise RuntimeError('CPU acceptance prohibits CUDA initialization, checkpoint loading and downloads')
    torch.cuda.init=reject;torch.cuda._lazy_init=reject;torch._C._cuda_init=reject;torch.load=reject
    torch.hub.load_state_dict_from_url=reject;torch.hub.download_url_to_file=reject
    checked=[]
    if kind in ('stablesr','threedsr'):
        source=ROOT/'tmp/static_sr_wu_20261022/upstream'/kind
        sys.path[:0]=[str(source),str(source/('scripts' if kind=='stablesr' else 'third_parties'))]
        for name in ['ldm.util','ldm.models.diffusion.ddpm','ldm.models.autoencoder','ldm.modules.encoders.modules','ldm.modules.diffusionmodules.openaimodel','ldm.models.diffusion.ddim']:
            m=importlib.import_module(name);checked.append(dict(module=name,source=identity(m.__file__)))
        from omegaconf import OmegaConf
        def visit(cfg):
            if OmegaConf.is_dict(cfg):
                if 'target' in cfg:
                    module,symbol=str(cfg.target).rsplit('.',1);getattr(importlib.import_module(module),symbol)
                for key in cfg:visit(cfg[key])
            elif OmegaConf.is_list(cfg):
                for item in cfg:visit(item)
        for path in ['configs/stableSRNew/v2-finetune_text_T_512.yaml','configs/autoencoder/autoencoder_kl_64x64x4_resi.yaml']:visit(OmegaConf.load(source/path).model)
        from open_clip.pretrained import get_pretrained_cfg,download_pretrained
        actual=Path(download_pretrained(get_pretrained_cfg('ViT-H-14','laion2b_s32b_b79k')))
        expected=next(x['path'] for x in assets['cache_mappings'] if x['role']=='openclip')
        if not actual.samefile(expected):raise RuntimeError('OpenCLIP offline resolver found foreign cache')
        if kind=='stablesr':
            path=source/'scripts/sr_val_ddpm_text_T_vqganfin_oldcanvas_tile.py';spec=importlib.util.spec_from_file_location('author_script_import_only',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);checked.append(dict(script=identity(path)))
    elif kind=='sequence':
        source=ROOT/'tmp/static_sr_wu_20261022/upstream/sequence_matters';sys.path.insert(0,str(source/'vsr/psrt'))
        from archs.psrt_recurrent_arch import BasicRecurrentSwin
        import basicsr.utils.img_util
        checked=[dict(module='archs.psrt_recurrent_arch.BasicRecurrentSwin',source=identity(sys.modules['archs.psrt_recurrent_arch'].__file__)),dict(module='basicsr.utils.img_util',source=identity(basicsr.utils.img_util.__file__))]
    elif kind=='wu_extensions':
        sys.path.insert(0,str(ROOT/'experiments/dynamic_sr_cull_retrain_20261009'));import rt
        checked.append(rt.check_runtime())
        sys.path.insert(0,str(BASELINE))
        from methods.splatsure_prepare import load_official_rasterizer
        from methods.iesrgs_renderer import raster_api as ie
        from methods.threedsr_renderer import raster_api as td
        modules=[load_official_rasterizer(ROOT/'tmp/static_sr_wu_20261022/upstream/splatsure-rasterizer',RUNTIME/'splatsure'),ie(),td()]
        checked.extend(dict(module=m.__name__,source=identity(m.__file__)) for m in modules)
    if forbidden or torch.cuda.is_initialized():raise RuntimeError('Forbidden CPU acceptance operation')
    import importlib.metadata
    versions={name:importlib.metadata.version(name) for name in ['torch','torchvision','numpy']}
    alias=importlib.import_module('torchvision.transforms.functional_tensor') if kind!='wu_extensions' else None
    if alias:
        current=importlib.import_module('torchvision.transforms._functional_tensor');assert alias.rgb_to_grayscale is current.rgb_to_grayscale
    print(json.dumps(dict(status='CPU_imports_passed_GPU_unverified',kind=kind,root=str(ROOT),python=sys.executable,prefix=sys.prefix,versions=versions,checked=checked,CUDA_initialized=False,GPU_calls=0,checkpoint_loads=0,teacher_models_instantiated=0)))

def apply(args):
    out=Path(args.out).absolute();out.mkdir(parents=True,exist_ok=False);started=time.monotonic();record=dict(status='running',root=str(ROOT),script=identity(__file__),GPU_calls=0,checks={})
    try:
        env,record['subprocess_environment']=cpu_subprocess_environment(args.base_python)
        base=query_base(Path(args.base_python).absolute(),env=env);record['base_environment']=base
        envs={name:overlay(args.base_python,base,name,env=env) for name in ('stablesr','sequence')}
        assets=runtime_assets(args.runtime_assets_template,out);record['runtime_assets']=identity(assets)
        checks=record['checks']  # Keep completed and failing checks even if a later subprocess fails.
        for kind in ('stablesr','threedsr','sequence','wu_extensions'):
            python=envs['stablesr' if kind in ('stablesr','threedsr') else 'sequence'] if kind!='wu_extensions' else args.base_python
            log=out/(kind+'_cpu_check.log')
            with log.open('x') as stream:r=subprocess.run([str(python),str(Path(__file__).absolute()),'check','--kind',kind,'--runtime-assets',str(assets)],env=env,stdout=stream,stderr=subprocess.STDOUT)
            checks[kind]=dict(returncode=r.returncode,log=identity(log))
            if r.returncode:raise RuntimeError(f'CPU import check failed; inspect {log}')
        if args.register:
            command=[str(args.base_python),str(BASELINE/'prepare_suite.py'),'register','--prepared',str(ROOT/'output/static_sr_wu_20261022/cts_prepared'),'--plans',str(ROOT/'output/static_sr_wu_20261022/cts_cpu_prior_plans')]
            log=out/'registration.log'
            with log.open('x') as stream:r=subprocess.run(command,env=env,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
            record['registration']=dict(command=command,returncode=r.returncode,log=identity(log))
            if r.returncode:raise RuntimeError(f'CTS record registration failed: {log}')
        record.update(status='CPU_overlays_cache_imports_and_requested_registration_passed_GPU_unverified',environments={k:str(v) for k,v in envs.items()},teacher_images_generated=0,training_started=False)
    except Exception:
        record.update(status='failed_CPU_bootstrap_GPU_not_run',traceback=traceback.format_exc())
        raise
    finally:
        record['elapsed_seconds']=time.monotonic()-started;write_json(out/'receipt.json',record)
    print(json.dumps(record,indent=2))

def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='mode',required=True)
    a=sub.add_parser('apply');a.add_argument('--base-python',required=True);a.add_argument('--runtime-assets-template',type=Path,default=ROOT/'output/static_sr_wu_20261022/splatsure_stablesr_assets/runtime_assets_v2.json');a.add_argument('--out',type=Path,default=RUNTIME/'cts_bootstrap');a.add_argument('--register',action='store_true')
    c=sub.add_parser('check');c.add_argument('--kind',choices=['stablesr','threedsr','sequence','wu_extensions'],required=True);c.add_argument('--runtime-assets',required=True)
    args=p.parse_args()
    if args.mode=='check':check(args.kind,args.runtime_assets)
    else:apply(args)
if __name__=='__main__':main()
