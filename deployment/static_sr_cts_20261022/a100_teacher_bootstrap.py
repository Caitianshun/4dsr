"""Task-local StableSR teacher environment, verified without inference or CUDA.

``plan`` exports the exact, already reviewed CTS package/source subset. It does
not transfer anything. After a controller transfers those files and the pinned
weights, ``apply`` runs ONLY at the fixed A100 task root. It creates new venv
launchers and destination .pth files, registers private offline caches, and
imports the author's config targets and inference script in a guarded CPU
subprocess. No pip, downloads, SSH, datasets, Wu extensions, model construction,
checkpoint loading, inference, or training are performed. A CPU pass is not GPU
acceptance. Original CTS inventories remain unchanged provenance documents.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
A100_ROOT = Path('/home/ubuntu/3DGS/4dsr/tasks/static_sr_wu_20261022')
BASE_PYTHON = Path('/home/ubuntu/3DGS/4dsr/.venv/bin/python')
REL = Path('output/static_sr_wu_20261022')
ENV_REL = REL/'runtime/stablesr_env'
SITE_REL = ENV_REL/'lib/python3.10/site-packages'
UPSTREAM = Path('tmp/static_sr_wu_20261022/upstream')
WEIGHT_REL = Path('weight/static_sr_wu_20261022/stablesr')
CACHE_REL = WEIGHT_REL/'a100_runtime_cache'
MANIFESTS = {
    'overlays': (REL/'cts_snapshot_inventory_overlays/inventory.json',
                 '6ad1233ea23cad6e602591e256659fe6da7c36ab5291babc05b98cd9150a0125'),
    'upstream': (REL/'cts_snapshot_inventory_upstream_gittracked_v2/inventory.json',
                 '55f91a3109482b897ba240ca013b38b0096194fb88b3accfc4089ff16c073921'),
}
COMMITS = {'stablesr': '398ee9383777e255540ea027a704c8ce1f32145b',
           'taming-transformers': '3ba01b241669f5ade541ce990f7650a3b8f65318'}
SELECTED_PREFIXES = {'overlays': (SITE_REL,),
                     'upstream': tuple(UPSTREAM/name for name in COMMITS)}
EXPECTED_COUNTS = {'overlays': (5068, 156393834), 'upstream': (954, 315082920)}
OPENCLIP_REVISION = '1c2b8495b28150b8a4922ee1c8edee224c284c0c'
OPENCLIP_REPO = 'models--laion--CLIP-ViT-H-14-laion2B-s32B-b79K'
MODELS = {
    'diffusion': ('stablesr_000117.ckpt', 6481647231,
                  'b8862bf3fd11c5b8fe82fb8a4618a1c74a29e0301a190bd6c2e84d68986ef9cb'),
    'cfw': ('vqgan_cfw_00011.ckpt', 959719471,
            '5b20ed9a80e9bdbf9e76ff1642a6a5428ca3427b08b779248a88bfbba2e74e8e'),
    'openclip': ('open_clip_pytorch_model.bin', 3944692325,
                 '9a78ef8e8c73fd0df621682e7a8e8eb36c6916cb3c16b291a082ecd52ab79cc4'),
    'vgg16': ('vgg16-397923af.pth', 553433881,
              '397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0'),
    'lpips': ('vgg.pth', 7289,
              'a78928a0af1e5f0fcb1f3b9e8f8c3a2a5a3de244d830ad5c1feddc79b8432868'),
}
SHARED_CACHE = Path('/home/ubuntu/.cache/torch/hub/checkpoints')
BASE_VERSIONS = {'torch': '2.7.1+cu128', 'torchvision': '0.22.1+cu128', 'numpy': '1.26.4'}
OVERLAY_VERSIONS = {'omegaconf': '2.1.1', 'pytorch-lightning': '1.4.2',
                    'torchmetrics': '0.6.0', 'open-clip-torch': '2.0.2',
                    'transformers': '4.19.2', 'tokenizers': '0.12.1',
                    'huggingface-hub': '0.25.2', 'einops': '0.3.0', 'kornia': '0.6.12'}
CPU_ENV = {'CUDA_VISIBLE_DEVICES': '', 'PYTHONDONTWRITEBYTECODE': '1',
           'PYTHONNOUSERSITE': '1', 'MPLBACKEND': 'Agg', 'OMP_NUM_THREADS': '1',
           'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
           'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
           'HF_DATASETS_OFFLINE': '1', 'HF_HUB_DISABLE_TELEMETRY': '1',
           'GIT_OPTIONAL_LOCKS': '0'}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def regular(path):
    """No symlink at the leaf or any existing ancestor; no FIFOs/devices."""
    path = Path(path).absolute()
    for ancestor in (path, *path.parents):
        if ancestor.is_symlink():
            raise ValueError('Symlink is outside the closed-file contract: '+str(ancestor))
    if not stat.S_ISREG(path.stat().st_mode):
        raise ValueError('Expected a regular file: '+str(path))
    return path


def identity(path):
    path = regular(path)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path))


def inside(path, root=None):
    root = Path(ROOT if root is None else root).absolute()
    path = Path(path).absolute()
    if '..' in path.parts or not path.is_relative_to(root) or path == root:
        raise ValueError('Path escaped the task root: '+str(path))
    for ancestor in (path, *path.parents):
        if ancestor.is_symlink():
            raise ValueError('Symlink in task output path: '+str(ancestor))
    return path


def write_exact(path, value):
    path = inside(path)
    if path.exists():
        regular(path)
        if path.read_text() != value:
            raise FileExistsError('Conflicting file preserved: '+str(path))
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        stream.write(value)
    return True


def dump(path, value):
    return write_exact(path, json.dumps(value, indent=2, allow_nan=False)+'\n')


def strict_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('Duplicate JSON key: '+key)
            result[key] = value
        return result
    return json.loads(regular(path).read_text(), object_pairs_hook=pairs)


def relative(value):
    if not isinstance(value, str):
        raise ValueError('Noncanonical inventory path')
    path = PurePosixPath(value)
    if (not value or path.is_absolute()
            or '..' in path.parts or path.as_posix() != value or '\\' in value):
        raise ValueError('Noncanonical inventory path')
    return Path(value)


def selected_inventory(kind, path=None):
    """Only these two immutable manifests can authorize package/source bytes."""
    default, expected = MANIFESTS[kind]
    path = regular(path or ROOT/default)
    if sha(path) != expected:
        raise ValueError('Pinned '+kind+' inventory SHA differs')
    data = strict_json(path)
    if (data.get('schema') != 'static_sr_cts_transport_inventory_v1'
            or data.get('stage') != kind or data.get('symlinks')):
        raise ValueError('Unsupported transport inventory')
    rows, seen = [], set()
    for item in data['files']:
        rel = relative(item['path'])
        if str(rel) in seen:
            raise ValueError('Duplicate inventory member')
        seen.add(str(rel))
        if type(item['bytes']) is not int or item['bytes'] < 0:
            raise ValueError('Invalid file size')
        digest = item['sha256']
        if not isinstance(digest, str) or len(digest) != 64 or any(x not in '0123456789abcdef' for x in digest):
            raise ValueError('Invalid SHA256')
        if any(rel.is_relative_to(prefix) and rel != prefix for prefix in SELECTED_PREFIXES[kind]):
            rows.append(dict(path=str(rel), bytes=item['bytes'], sha256=digest))
    if (len(rows), sum(x['bytes'] for x in rows)) != EXPECTED_COUNTS[kind]:
        raise ValueError('Pinned teacher closure count/size differs')
    return sorted(rows, key=lambda x: x['path']), identity(path)


def plan(overlay_inventory=None, upstream_inventory=None):
    rows, sources = [], {}
    for kind, path in [('overlays', overlay_inventory), ('upstream', upstream_inventory)]:
        selected, sources[kind] = selected_inventory(kind, path)
        rows.extend(selected)
    return dict(schema='a100_stablesr_teacher_bootstrap_plan_v1',
        status='CPU_plan_only_no_transfer_or_environment_created', script=identity(__file__),
        destination_root=str(A100_ROOT), base_python=str(BASE_PYTHON),
        inventories=sources, files=sorted(rows, key=lambda x: x['path']),
        total_bytes=sum(x['bytes'] for x in rows), source_commits=COMMITS,
        required_task_weights=[dict(path=str(WEIGHT_REL/MODELS[r][0]), role=r,
            bytes=MODELS[r][1], sha256=MODELS[r][2]) for r in ('diffusion', 'cfw', 'openclip')],
        reuse_shared_models=[dict(path=str(SHARED_CACHE/MODELS[r][0]), role=r,
            bytes=MODELS[r][1], sha256=MODELS[r][2]) for r in ('vgg16', 'lpips')],
        generated_only=['venv bin/pyvenv.cfg', 'three task-local .pth files',
                        'private cache aliases and exact HF ref', 'CPU acceptance receipts'],
        copy_exclusions=['local bin', 'local pyvenv.cfg', 'local absolute .pth files',
                         'other method overlays/sources', 'datasets', 'CUDA environments'],
        GPU_calls=0, teacher_images_generated=0, training_started=False)


def verify_files(value):
    """Fresh transfer must exactly equal the closed subset, including Git files."""
    expected = {x['path']: x for x in value['files']}
    actual = set()
    for prefixes in SELECTED_PREFIXES.values():
        for prefix in prefixes:
            base = inside(ROOT/prefix)
            if not base.is_dir():
                raise FileNotFoundError('Missing transferred tree: '+str(base))
            for parent, dirs, names in os.walk(base, followlinks=False):
                for name in dirs:
                    inside(Path(parent)/name)
                for name in names:
                    path = regular(Path(parent)/name)
                    actual.add(str(path.relative_to(ROOT)))
    if actual != set(expected):
        raise ValueError('Teacher closure differs: missing='+repr(sorted(set(expected)-actual)[:8])
                         +' extra='+repr(sorted(actual-set(expected))[:8]))
    for rel, item in expected.items():
        path = regular(ROOT/rel)
        if path.stat().st_size != item['bytes'] or sha(path) != item['sha256']:
            raise ValueError('Transferred file identity differs: '+rel)
    # Venv launchers and absolute .pth must be made here, never transported.
    env_root = inside(ROOT/ENV_REL)
    for path in env_root.rglob('*'):
        if path.is_symlink() or (path.is_file() and str(path.relative_to(ROOT)) not in expected):
            raise ValueError('Unexpected preexisting venv content: '+str(path))
    return dict(files=len(expected), bytes=sum(x['bytes'] for x in expected.values()),
                all_sha256_verified=True, closed_subset=True)


def cpu_environment(cache=None):
    env = dict(os.environ, **CPU_ENV)
    for key in ('PYTHONPATH', 'PYTHONHOME', 'VIRTUAL_ENV'):
        env.pop(key, None)
    if cache:
        env.update(cache)
    return env


def validate_base(value):
    if value['python_version'][:2] != [3, 10] or value['versions'] != BASE_VERSIONS or value['CUDA_initialized']:
        raise RuntimeError('Unaccepted A100 base ABI/version: '+repr(value))
    expected = BASE_PYTHON.parent.parent/'lib/python3.10/site-packages'
    if Path(value['purelib']).absolute() != expected or Path(value['prefix']).absolute() != BASE_PYTHON.parent.parent:
        raise RuntimeError('Base Python did not use the registered project venv')
    if Path(value['python']).absolute() != BASE_PYTHON:
        raise RuntimeError('Base launcher pathname changed')
    return value


def query_base(env):
    code = """import json,sys,sysconfig,socket
def no_network(*a,**k): raise RuntimeError('CPU bootstrap forbids network')
socket.socket.connect=no_network;socket.socket.connect_ex=no_network;socket.create_connection=no_network
import torch,torchvision,numpy
assert not torch.cuda.is_initialized()
print(json.dumps(dict(python=sys.executable,prefix=sys.prefix,purelib=sysconfig.get_paths()['purelib'],python_version=list(sys.version_info[:3]),versions=dict(torch=torch.__version__,torchvision=torchvision.__version__,numpy=numpy.__version__),CUDA_initialized=torch.cuda.is_initialized())))
"""
    result = subprocess.run([str(BASE_PYTHON), '-c', code], env=env, text=True,
                            capture_output=True, check=True)
    return validate_base(json.loads(result.stdout))


def verify_sources(env):
    result = {}
    for name, expected in COMMITS.items():
        path = ROOT/UPSTREAM/name
        head = subprocess.run(['git', '-C', str(path), 'rev-parse', 'HEAD'], env=env,
                              text=True, capture_output=True, check=True).stdout.strip()
        clean = subprocess.run(['git', '-C', str(path), 'diff', '--quiet', 'HEAD', '--'], env=env)
        if head != expected or clean.returncode != 0:
            raise ValueError('Pinned source HEAD/content changed: '+name)
        result[name] = dict(path=str(path), commit=head, tracked_clean=True)
    return result


def overlay(base, env):
    directory, site = ROOT/ENV_REL, ROOT/SITE_REL
    if (directory/'pyvenv.cfg').exists() or (directory/'bin').exists():
        raise FileExistsError('Existing venv preserved; use the previous receipt or review failed setup')
    subprocess.run([str(BASE_PYTHON), '-m', 'venv', '--copies', '--without-pip', str(directory)],
                   env=env, check=True)
    if 'include-system-site-packages = false' not in (directory/'pyvenv.cfg').read_text():
        raise RuntimeError('Unexpected venv system package access')
    entries = {
        'reuse_wu_torch.pth': base['purelib']+'\n',
        'taming_pinned_namespace.pth': str(ROOT/UPSTREAM/'taming-transformers')+'\n',
        'static_sr_torchvision_compat.pth': 'import runpy; runpy.run_path('
            +repr(str(site/'sitecustomize.py'))+", run_name='static_sr_torchvision_compat')\n",
    }
    for name, content in entries.items():
        write_exact(site/name, content)
    return dict(python=str(directory/'bin/python'), pyvenv_config=identity(directory/'pyvenv.cfg'),
                generated_pth=[identity(site/name) for name in entries],
                base_purelib=base['purelib'], copied_local_environment=False)


def verified_models():
    result = {}
    for role, (name, size, expected) in MODELS.items():
        path = (SHARED_CACHE if role in ('vgg16', 'lpips') else ROOT/WEIGHT_REL)/name
        item = identity(path)
        if item['bytes'] != size or item['sha256'] != expected:
            raise ValueError('Generic model identity differs: '+role)
        result[role] = dict(item, role=role)
    return result


def alias_file(source, target, expected):
    source, target = regular(source), inside(target)
    if sha(source) != expected:
        raise ValueError('Cache source SHA differs')
    if target.exists():
        if sha(regular(target)) != expected:
            raise FileExistsError('Conflicting cache alias preserved: '+str(target))
        return 'existing_verified'
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
        storage = 'hardlink'
    except FileExistsError:
        raise
    except OSError:
        with source.open('rb') as src, target.open('xb') as dst:
            shutil.copyfileobj(src, dst, length=8 << 20)
        storage = 'copy'
    if sha(regular(target)) != expected:
        raise ValueError('Cache alias SHA differs after creation')
    return storage


def runtime_assets(models, out):
    cache = ROOT/CACHE_REL
    hf, th = cache/'hf/hub', cache/'torch'
    targets = {'openclip': hf/OPENCLIP_REPO/'snapshots'/OPENCLIP_REVISION/'open_clip_pytorch_model.bin',
               'vgg16': th/'hub/checkpoints/vgg16-397923af.pth', 'lpips': cache/'lpips/vgg.pth'}
    mappings = []
    for role, target in targets.items():
        source = models[role]
        storage = alias_file(source['path'], target, source['sha256'])
        mappings.append(dict(identity(target), role=role, storage=storage, original_source=source))
    ref = hf/OPENCLIP_REPO/'refs/main'
    write_exact(ref, OPENCLIP_REVISION)  # HF does not strip a trailing newline.
    value = dict(schema='a100_stablesr_runtime_assets_v1',
        status='exact_cache_bytes_registered_GPU_unverified',
        # Operational identities stay inside the task root for explicit
        # cross-host relocation. Shared A100 originals remain provenance only.
        files=[dict(identity(targets[x]), role=x) for x in ('openclip', 'vgg16', 'lpips')],
        cache_mappings=mappings,
        cache_metadata=[dict(identity(ref), role='openclip_offline_revision')],
        cache_environment={'HF_HOME': str(cache/'hf'), 'HUGGINGFACE_HUB_CACHE': str(hf), 'TORCH_HOME': str(th)},
        teacher_checkpoints=[models[x] for x in ('diffusion', 'cfw')],
        GPU_calls=0, teacher_images_generated=0, shared_cache_modified=False)
    path = out/'runtime_assets_a100.json'
    dump(path, value)
    return path


def validate_assets(value):
    """Validate exactly the paths consulted by the pinned offline loaders."""
    roles = ('openclip', 'vgg16', 'lpips')
    cache = ROOT/CACHE_REL
    expected_paths = {'openclip': cache/'hf/hub'/OPENCLIP_REPO/'snapshots'/OPENCLIP_REVISION/'open_clip_pytorch_model.bin',
                      'vgg16': cache/'torch/hub/checkpoints/vgg16-397923af.pth',
                      'lpips': cache/'lpips/vgg.pth'}
    for field in ('files', 'cache_mappings'):
        rows = value[field]
        if len(rows) != 3 or {x['role'] for x in rows} != set(roles):
            raise ValueError('Runtime asset role closure differs')
        for item in rows:
            name, size, digest = MODELS[item['role']]
            if Path(item['path']).absolute() != expected_paths[item['role']]:
                raise ValueError('Runtime model source escaped the registered task-local alias')
            actual = identity(item['path'])
            if actual['sha256'] != digest or actual['bytes'] != size or item['sha256'] != digest:
                raise ValueError('Runtime asset SHA/size differs')
    expected_env = {'HF_HOME': str(cache/'hf'), 'HUGGINGFACE_HUB_CACHE': str(cache/'hf/hub'),
                    'TORCH_HOME': str(cache/'torch')}
    if value['cache_environment'] != expected_env:
        raise ValueError('Runtime cache environment escaped task-local registration')
    for item in value['cache_mappings']:
        if Path(item['path']).absolute() != expected_paths[item['role']]:
            raise ValueError('Cache mapping differs from fixed loader path')
    if regular(cache/'hf/hub'/OPENCLIP_REPO/'refs/main').read_text() != OPENCLIP_REVISION:
        raise ValueError('Offline revision ref differs')
    return {x['role']: x for x in value['cache_mappings']}


def require_a100():
    if ROOT.absolute() != A100_ROOT:
        raise RuntimeError('Apply/check are restricted to the fixed A100 task root; use plan locally')
    inside(ROOT/REL)
    # Preserve the venv symlink launcher, unlike data and package files.
    if not BASE_PYTHON.is_file():
        raise FileNotFoundError('Registered A100 base launcher missing')


def check(assets_path, assets_sha, result_path):
    """Internal subprocess entry point; no package or model installations."""
    require_a100()
    assets_path, result_path = inside(assets_path), inside(result_path)
    record = dict(status='CPU_check_running', checked=[], config_targets=[], forbidden_attempts=[],
                  GPU_calls=0, checkpoint_loads=0, teacher_models_instantiated=0,
                  teacher_images_generated=0, CUDA_initialized=False)
    try:
        if sha(regular(assets_path)) != assets_sha:
            raise ValueError('Runtime asset registration SHA changed')
        assets = strict_json(assets_path)
        mappings = validate_assets(assets)
        os.environ.update(cpu_environment(assets['cache_environment']))
        import socket
        def reject(*args, **kwargs):
            record['forbidden_attempts'].append('network/CUDA/checkpoint operation')
            raise RuntimeError('Teacher CPU check prohibits network, CUDA and checkpoint loading')
        socket.socket.connect = reject
        socket.socket.connect_ex = reject
        socket.create_connection = reject
        import torch
        if torch.cuda.is_initialized():
            raise RuntimeError('CUDA initialized before CPU checks')
        torch.cuda.init = reject
        torch.cuda._lazy_init = reject
        torch._C._cuda_init = reject
        torch.cuda.is_available = lambda: False
        torch.cuda.device_count = lambda: 0
        torch.load = reject
        torch.hub.load_state_dict_from_url = reject
        torch.hub.download_url_to_file = reject
        import importlib
        import importlib.metadata
        versions = {name: importlib.metadata.version(name) for name in dict(BASE_VERSIONS, **OVERLAY_VERSIONS)}
        if versions != dict(BASE_VERSIONS, **OVERLAY_VERSIONS):
            raise RuntimeError('Overlay/base distribution versions differ: '+repr(versions))
        record['versions'] = versions
        source = ROOT/UPSTREAM/'stablesr'
        sys.path[:0] = [str(source), str(source/'scripts')]
        # Explicitly verify resolution to this overlay/base instead of silently
        # importing a coincidental package from the shared A100 environment.
        paths = {'torch': BASE_PYTHON.parent.parent/'lib/python3.10/site-packages',
                 'torchvision': BASE_PYTHON.parent.parent/'lib/python3.10/site-packages',
                 'numpy': BASE_PYTHON.parent.parent/'lib/python3.10/site-packages',
                 'taming': ROOT/UPSTREAM/'taming-transformers'}
        paths.update({name: ROOT/SITE_REL for name in ('omegaconf', 'pytorch_lightning',
                     'torchmetrics', 'open_clip', 'transformers', 'tokenizers',
                     'huggingface_hub', 'einops', 'kornia', 'clip')})
        for name, prefix in paths.items():
            module = importlib.import_module(name)
            if not Path(module.__file__).absolute().is_relative_to(prefix):
                raise RuntimeError('Module resolved outside registered package root: '+name)
            record['checked'].append(dict(module=name, source=identity(module.__file__)))
        for name in ('ldm.util', 'ldm.models.diffusion.ddpm', 'ldm.models.autoencoder',
                     'ldm.modules.encoders.modules', 'ldm.modules.diffusionmodules.openaimodel',
                     'ldm.models.diffusion.ddim'):
            module = importlib.import_module(name)
            if not Path(module.__file__).absolute().is_relative_to(source):
                raise RuntimeError('StableSR module shadowed: '+name)
            record['checked'].append(dict(module=name, source=identity(module.__file__)))
        from omegaconf import OmegaConf
        def visit(cfg):
            if OmegaConf.is_dict(cfg):
                if 'target' in cfg:
                    name, symbol = str(cfg.target).rsplit('.', 1)
                    getattr(importlib.import_module(name), symbol)
                    record['config_targets'].append(str(cfg.target))
                for key in cfg:
                    visit(cfg[key])
            elif OmegaConf.is_list(cfg):
                for item in cfg:
                    visit(item)
        for rel in ('configs/stableSRNew/v2-finetune_text_T_512.yaml',
                    'configs/autoencoder/autoencoder_kl_64x64x4_resi.yaml'):
            visit(OmegaConf.load(source/rel).model)
            record['checked'].append(dict(config=identity(source/rel)))
        from open_clip.pretrained import get_pretrained_cfg, download_pretrained
        actual = Path(download_pretrained(get_pretrained_cfg('ViT-H-14', 'laion2b_s32b_b79k')))
        if not actual.samefile(mappings['openclip']['path']):
            raise RuntimeError('OpenCLIP resolver returned a foreign cache')
        record['offline_openclip'] = identity(actual)
        author_path = source/'scripts/sr_val_ddpm_text_T_vqganfin_oldcanvas_tile.py'
        spec = importlib.util.spec_from_file_location('author_script_import_only', author_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # __main__ is deliberately not invoked.
        record['checked'].append(dict(author_script=identity(author_path)))
        old = importlib.import_module('torchvision.transforms.functional_tensor')
        new = importlib.import_module('torchvision.transforms._functional_tensor')
        if old.rgb_to_grayscale is not new.rgb_to_grayscale:
            raise RuntimeError('Torchvision compatibility alias changed the function')
        if record['forbidden_attempts'] or torch.cuda.is_initialized():
            raise RuntimeError('Forbidden operation during CPU acceptance')
        record.update(status='teacher_only_CPU_imports_config_and_offline_resolver_passed_GPU_unverified',
                      python=sys.executable, prefix=sys.prefix)
    except Exception:
        record.update(status='failed_CPU_check_GPU_not_run', traceback=traceback.format_exc())
        raise
    finally:
        dump(result_path, record)


def apply(args):
    require_a100()
    out = inside(args.out or ROOT/REL/'runtime/a100_teacher_bootstrap')
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    record = dict(schema='a100_stablesr_teacher_bootstrap_receipt_v1', status='running',
                  root=str(ROOT), script=identity(__file__), checks={}, GPU_calls=0,
                  teacher_images_generated=0, training_started=False)
    try:
        value = plan(args.overlay_inventory, args.upstream_inventory)
        dump(out/'closed_inventory_plan.json', value)
        record['plan'] = identity(out/'closed_inventory_plan.json')
        record['transfer_verification'] = verify_files(value)
        env = cpu_environment()
        record['subprocess_environment'] = dict(CPU_ENV, inherited_LD_LIBRARY_PATH=env.get('LD_LIBRARY_PATH', ''),
                                                dropped=['PYTHONPATH', 'PYTHONHOME', 'VIRTUAL_ENV'])
        record['source_verification'] = verify_sources(env)
        record['base_environment'] = query_base(env)
        models = verified_models()
        record['model_identities'] = list(models.values())
        record['overlay'] = overlay(record['base_environment'], env)
        assets = runtime_assets(models, out)
        record['runtime_assets'] = identity(assets)
        log, result = out/'stablesr_cpu_check.log', out/'stablesr_cpu_check.json'
        command = [record['overlay']['python'], str(Path(__file__).absolute()), 'check',
                   '--runtime-assets', str(assets), '--runtime-assets-sha256', sha(assets),
                   '--result', str(result)]
        with log.open('x') as stream:
            completed = subprocess.run(command, env=env, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        record['checks']['stablesr'] = dict(command=command, returncode=completed.returncode, log=identity(log))
        if result.exists():
            record['checks']['stablesr']['result'] = identity(result)
        if completed.returncode:
            raise RuntimeError('Teacher CPU check failed; inspect '+str(log))
        child = strict_json(result)
        if (child['status'] != 'teacher_only_CPU_imports_config_and_offline_resolver_passed_GPU_unverified'
                or child['forbidden_attempts'] or child['CUDA_initialized']
                or any(child[k] != 0 for k in ('GPU_calls', 'checkpoint_loads', 'teacher_models_instantiated', 'teacher_images_generated'))):
            raise RuntimeError('CPU acceptance receipt failed its contract')
        record.update(status='teacher_only_CPU_bootstrap_passed_GPU_unverified',
                      next_required='separate explicitly scheduled GPU teacher acceptance; no training authorization implied')
    except Exception:
        record.update(status='failed_CPU_bootstrap_GPU_not_run', traceback=traceback.format_exc())
        raise
    finally:
        record['elapsed_seconds'] = time.monotonic()-started
        dump(out/'receipt.json', record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    for name in ('plan', 'apply'):
        p = sub.add_parser(name)
        p.add_argument('--overlay-inventory', type=Path)
        p.add_argument('--upstream-inventory', type=Path)
        p.add_argument('--out', type=Path)
    p = sub.add_parser('check', help='internal guarded CPU subprocess')
    p.add_argument('--runtime-assets', type=Path, required=True)
    p.add_argument('--runtime-assets-sha256', required=True)
    p.add_argument('--result', type=Path, required=True)
    args = parser.parse_args()
    if args.mode == 'plan':
        value = plan(args.overlay_inventory, args.upstream_inventory)
        if args.out:
            dump(args.out, value)
        print(json.dumps({k: v for k, v in value.items() if k != 'files'}, indent=2))
    elif args.mode == 'apply':
        print(json.dumps(apply(args), indent=2))
    else:
        check(args.runtime_assets, args.runtime_assets_sha256, args.result)


if __name__ == '__main__':
    main()
