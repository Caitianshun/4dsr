"""Prepare exact Wu native extension sources, then compile on the target CPU.

The prepare command is local/read-only outside its fresh bundle. The build
command uses nvcc with CUDA_VISIBLE_DEVICES='' and an explicit sm_86 target;
it writes only a fresh build directory. Neither command installs packages,
replaces existing binaries, runs GPU kernels, or changes any original source.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import time
import traceback

from common import ROOT, identity, read, resolve, sha, write_new

WU = '843d5ac636c37e4b611242287754f3d4ed150144'
KNN = '44f764299fa305faf6ec5ebd99939e0508331503'
RASTER = 'e49506654e8e11ed8a62d22bcb693e943fdecacf'
GLM = '5c46b9c07008ae65cb81ab79cd677ecc1934b903'
DEFAULT_BUNDLE = ROOT / 'output/static_sr_wu_20261022/runtime/cts_native_build'


def command(args, cwd=None, env=None):
    return subprocess.check_output(list(map(str, args)), cwd=cwd, env=env, text=True).strip()


def version_requirements(path):
    text = command(['readelf', '--version-info', path])
    result = {}
    for family in ('GLIBCXX', 'GLIBC', 'CXXABI'):
        versions = sorted(set(re.findall(r'\b' + family + r'_([0-9.]+)', text)),
                          key=lambda value: tuple(map(int, value.split('.'))))
        result[family] = versions
        result['max_' + family] = versions[-1] if versions else None
    return result


def copy_file(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('xb') as stream:
        stream.write(source.read_bytes())
    if sha(source) != sha(destination):
        raise RuntimeError('Source copy differs: ' + str(source))


def copy_git_tree(source, destination, commit):
    if command(['git', '-C', source, 'rev-parse', 'HEAD']) != commit:
        raise ValueError('Unexpected source HEAD: ' + str(source))
    entries = []
    for row in command(['git', '-C', source, 'ls-tree', '-rz', '--full-tree', 'HEAD']).split('\0'):
        if not row:
            continue
        metadata, name = row.split('\t', 1)
        mode, kind, blob = metadata.split()
        if kind == 'commit':
            continue  # GLM is copied explicitly with its independently pinned commit.
        if kind != 'blob' or mode == '120000':
            raise ValueError('Unexpected nonregular source entry: ' + name)
        source_file, target = source / name, destination / name
        copy_file(source_file, target)
        entries.append(dict(relative_path=str(target.relative_to(destination)), source_path=str(source_file),
                            sha256=sha(target), bytes=target.stat().st_size, upstream_git_blob=blob))
    return entries


def prepare(wu_root, bundle):
    wu_root, bundle = resolve(wu_root), resolve(bundle)
    if bundle.exists():
        raise FileExistsError(bundle)
    runtime_receipt = ROOT / 'third_party/4dgs_843d5ac_runtime/RUNTIME_COPY.json'
    runtime = read(runtime_receipt)
    verification = read(wu_root / 'LOCAL_VERIFICATION.json')
    if (command(['git', '-C', wu_root, 'rev-parse', 'HEAD']) != WU
            or runtime['upstream_commit'] != WU or runtime['simple_knn_commit'] != KNN
            or runtime['rasterizer_commit'] != RASTER or verification['commit'] != WU):
        raise ValueError('Wu checkout and original runtime/build records do not agree')
    knn = wu_root / 'submodules/simple-knn'
    raster = wu_root / 'submodules/depth-diff-gaussian-rasterization'
    glm = raster / 'third_party/glm'
    knn_diff = command(['git', '-C', knn, 'diff', '--', 'simple_knn.cu'])
    raster_diff = command(['git', '-C', raster, 'diff'])
    knn_patch = (wu_root / 'patches/simple-knn-cuda12.patch').read_text()
    raster_patch = (wu_root / 'patches/rasterizer-cuda12.patch').read_text()
    if knn_diff != knn_patch.split('diff --git a/simple_knn/__init__.py')[0].strip():
        raise ValueError('KNN current patch differs from original local build patch')
    if raster_diff != runtime['rasterizer_local_diff'].strip() or raster_diff != raster_patch.strip():
        raise ValueError('Native depth rasterizer patch differs from original runtime/build records')
    if command(['git', '-C', knn, 'diff', '--name-only']) != 'simple_knn.cu':
        raise ValueError('Unexpected KNN tracked changes')
    if command(['git', '-C', glm, 'status', '--porcelain']):
        raise ValueError('GLM source must remain clean')
    originals = []
    for row in runtime['files']:
        if any('/' + name + '/' in row['copy'] for name in ('simple_knn', 'diff_gaussian_rasterization')):
            source, current = Path(row['source']), resolve(row['copy'])
            if sha(source) != row['sha256'] or sha(current) != row['sha256']:
                raise ValueError('Source/runtime wrapper or old binary differs from original copy receipt')
            originals.append(dict(**row, current_identity_verified=True))
    bundle.mkdir(parents=True)
    packages = []
    for name, source, commit, package in [('simple_knn', knn, KNN, 'simple_knn'),
                                         ('native_depth_rasterizer', raster, RASTER, 'diff_gaussian_rasterization')]:
        dest = bundle / 'sources' / name
        entries = copy_git_tree(source, dest, commit)
        if name == 'simple_knn':
            init = source / 'simple_knn/__init__.py'
            if init.read_text() != '"""CUDA nearest-neighbor extension used by 4DGaussians."""\n':
                raise ValueError('KNN package initializer differs from original compatibility patch')
            target = dest / 'simple_knn/__init__.py'
            copy_file(init, target)
            entries.append(dict(relative_path='simple_knn/__init__.py', source_path=str(init),
                sha256=sha(target), bytes=target.stat().st_size, upstream_git_blob=None,
                origin='existing local compatibility patch, byte-identical to runtime wrapper'))
        else:
            glm_entries = copy_git_tree(glm, dest / 'third_party/glm', GLM)
            for item in glm_entries:
                item['relative_path'] = 'third_party/glm/' + item['relative_path']
            entries.extend(glm_entries)
        packages.append(dict(name=name, package=package, source_relative=str(dest.relative_to(bundle)),
            upstream_commit=commit, glm_commit=GLM if name != 'simple_knn' else None,
            files=entries, source_status=command(['git', '-C', source, 'status', '--porcelain'])))
    for file in ['simple-knn-cuda12.patch', 'rasterizer-cuda12.patch']:
        copy_file(wu_root / 'patches' / file, bundle / 'patches' / file)
    evidence = []
    for source, name in [(runtime_receipt, 'RUNTIME_COPY.json'),
                         (wu_root / 'LOCAL_VERIFICATION.json', 'LOCAL_VERIFICATION.json'),
                         (wu_root / 'logs/local_setup/cuda_extensions_build_retry.log', 'original_successful_build.log')]:
        copy_file(source, bundle / 'evidence' / name)
        evidence.append(identity(bundle / 'evidence' / name))
    binaries = [resolve(r['copy']) for r in originals if r['copy'].endswith('.so')]
    binaries += [ROOT / 'output/static_sr_wu_20261022/runtime/splatsure/diff_gaussian_rasterization',
                 ROOT / 'output/static_sr_wu_20261022/runtime/iesrgs/lib/diff_gaussian_rasterization_rade',
                 ROOT / 'output/static_sr_wu_20261022/runtime/threedsr/lib/diff_gaussian_rasterization']
    audits = []
    for item in binaries:
        path = item if item.is_file() else next(item.glob('_C*.so'))
        audits.append(dict(binary=identity(path), **version_requirements(path),
            requires_target_rebuild_for_reported_CTS_libstdcpp=True))
    report = dict(schema='static_sr_wu_CTS_exact_native_source_bundle_v1', status='source_prepared_not_compiled_on_CTS',
        creator=identity(__file__), source_Wu_root=str(wu_root), source_Wu_commit=WU,
        source_runtime_receipt=identity(runtime_receipt), packages=packages, original_runtime_files=originals,
        evidence=evidence, patches=[dict(relative_path='patches/' + p.name, sha256=sha(p)) for p in (bundle / 'patches').iterdir()],
        existing_five_binary_ABI_audit=audits, GPU_executed=False, remote_execution=False,
        algorithm_changes=False, existing_sources_and_binaries_unchanged=True,
        lineage_scope='Source HEAD, tracked diffs, wrappers and existing binary hashes agree with original build/copy records; not a bit-reproducible rebuild claim.',
        build_recipe='build_ext --build-lib <fresh>/lib --build-temp <fresh>/temp; CUDA_VISIBLE_DEVICES empty; sm_86',
        native_runtime_integration='rt.check_runtime pins all native modules to rt.RUNTIME; do not assume PYTHONPATH alone overrides it. Root must select a new isolated runtime/deployment, preserving old task binaries.',
        created_utc=datetime.now(timezone.utc).isoformat())
    write_new(bundle / 'source_inventory.json', report)
    all_files = sorted(p for p in bundle.rglob('*') if p.is_file())
    write_new(bundle / 'transfer_inventory.json', dict(schema='closed_transfer_file_inventory_v1',
        files=[dict(path=str(p.relative_to(ROOT)), sha256=sha(p), bytes=p.stat().st_size) for p in all_files],
        total_bytes=sum(p.stat().st_size for p in all_files), GPU_executed=False))
    return bundle / 'source_inventory.json'


def verify_bundle(bundle):
    report = read(bundle / 'source_inventory.json')
    if report.get('schema') != 'static_sr_wu_CTS_exact_native_source_bundle_v1' or report['source_Wu_commit'] != WU:
        raise ValueError('Unexpected source bundle')
    for package in report['packages']:
        if package['upstream_commit'] != {'simple_knn': KNN, 'native_depth_rasterizer': RASTER}[package['name']]:
            raise ValueError('Source package commit differs')
        directory = bundle / package['source_relative']
        for item in package['files']:
            path = directory / item['relative_path']
            if not path.resolve().is_relative_to(directory.resolve()) or sha(path) != item['sha256']:
                raise ValueError('Bundled source changed: ' + str(path))
    for item in report['patches']:
        if sha(bundle / item['relative_path']) != item['sha256']:
            raise ValueError('Bundled patch changed')
    return report


def build(bundle, output, cuda_home, cc, cxx, jobs):
    bundle, output, cuda_home = map(resolve, (bundle, output, cuda_home))
    source = verify_bundle(bundle)
    if output.exists():
        raise FileExistsError(output)
    if not (cuda_home / 'bin/nvcc').is_file() or jobs not in (1, 2):
        raise ValueError('Provide installed CUDA compiler and one or two CPU build jobs')
    output.mkdir(parents=True)
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'CUDA_HOME': str(cuda_home),
           'TORCH_CUDA_ARCH_LIST': '8.6', 'MAX_JOBS': str(jobs), 'CC': cc, 'CXX': cxx,
           'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '1'}
    report = dict(schema='static_sr_wu_CTS_native_CPU_build_v1', status='running', source_inventory=identity(bundle / 'source_inventory.json'),
        script=identity(__file__), host=socket.gethostname(), Python=sys.executable,
        CPU_build_only=True, GPU_executed=False, existing_runtime_modified=False,
        environment={k:env[k] for k in ['CUDA_VISIBLE_DEVICES','CUDA_HOME','TORCH_CUDA_ARCH_LIST','MAX_JOBS','CC','CXX']},
        compiler=command([cxx, '--version']), nvcc=command([cuda_home / 'bin/nvcc', '--version']), packages=[])
    started = time.perf_counter()
    try:
        probe_code = "import json,torch;print(json.dumps(dict(torch=torch.__version__,cuda=torch.version.cuda,ABI=torch._C._GLIBCXX_USE_CXX11_ABI,CUDA_initialized=torch.cuda.is_initialized())))"
        report['torch'] = json.loads(command([sys.executable, '-c', probe_code], env=env))
        if report['torch']['torch'] != '2.7.1+cu128' or report['torch']['CUDA_initialized']:
            raise ValueError('Target build requires the verified torch2.7.1+cu128 CPU import')
        for package in source['packages']:
            work = output / 'sources' / package['name']
            shutil.copytree(bundle / package['source_relative'], work)
            library = output / 'lib'
            args = [sys.executable, 'setup.py', 'build_ext', '--build-lib', str(library),
                    '--build-temp', str(output / 'temp' / package['name'])]
            log = output / (package['name'] + '.build.log')
            with log.open('x') as stream:
                subprocess.run(args, cwd=work, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
            wrapper = work / package['package'] / '__init__.py'
            copy_file(wrapper, library / package['package'] / '__init__.py')
            binaries = list((library / package['package']).glob('_C*.so'))
            if len(binaries) != 1:
                raise ValueError('Expected exactly one freshly compiled extension')
            binary = binaries[0]
            report['packages'].append(dict(name=package['name'], binary=identity(binary), wrapper=identity(library / package['package'] / '__init__.py'),
                command=args, log=identity(log), **version_requirements(binary)))
        import_code = ("import json,sys,torch;sys.path.insert(0,sys.argv[1]);import simple_knn._C as k;"
            "import diff_gaussian_rasterization as r;print(json.dumps(dict(knn=k.__file__,raster=r.__file__,"
            "knn_symbols=[n for n in ['distCUDA2'] if hasattr(k,n)],raster_symbols=[n for n in "
            "['rasterize_gaussians','rasterize_gaussians_backward','mark_visible'] if hasattr(r._C,n)],"
            "CUDA_initialized=torch.cuda.is_initialized())))")
        report['CPU_import'] = json.loads(command([sys.executable, '-c', import_code, output / 'lib'], env=env))
        if report['CPU_import']['CUDA_initialized'] or report['CPU_import']['knn_symbols'] != ['distCUDA2'] or len(report['CPU_import']['raster_symbols']) != 3:
            raise ValueError('CPU extension import/symbol contract failed')
        for key in ('knn', 'raster'):
            if not Path(report['CPU_import'][key]).resolve().is_relative_to(output / 'lib'):
                raise ValueError('CPU import resolved outside fresh output')
        verify_bundle(bundle)
        report['status'] = 'target_CPU_compile_and_import_passed_GPU_pending'
    except BaseException:
        report.update(status='failed', error=traceback.format_exc())
        raise
    finally:
        report['elapsed_seconds'] = time.perf_counter()-started
        write_new(output / 'build_receipt.json', report)
    return output / 'build_receipt.json'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--wu-root', type=Path, required=True, help='Pinned Wu 4DGaussians checkout')
    p.add_argument('--bundle', type=Path, default=DEFAULT_BUNDLE)
    p = sub.add_parser('verify')
    p.add_argument('--bundle', type=Path, default=DEFAULT_BUNDLE)
    p = sub.add_parser('build')
    p.add_argument('--bundle', type=Path, default=DEFAULT_BUNDLE)
    p.add_argument('--out', type=Path, default=DEFAULT_BUNDLE / 'target_build')
    p.add_argument('--cuda-home', type=Path, required=True)
    p.add_argument('--cc', default='gcc'); p.add_argument('--cxx', default='g++')
    p.add_argument('--jobs', type=int, default=2)
    args = parser.parse_args()
    if args.mode == 'prepare':
        print(identity(prepare(args.wu_root, args.bundle)))
    elif args.mode == 'verify':
        report = verify_bundle(resolve(args.bundle))
        print(dict(status='CPU_source_hashes_passed', files=sum(len(p['files']) for p in report['packages'])))
    else:
        print(identity(build(args.bundle, args.out, args.cuda_home, args.cc, args.cxx, args.jobs)))


if __name__ == '__main__':
    main()
