"""One-shot CTS CPU build of the three pinned method-specific rasterizers.

Plan locally, transfer the plan with its already registered upstream sources,
then build on CTS with its compiler. No installation or old-binary replacement.
All builds hide GPUs, target sm_86, and use two CPU compiler jobs by default.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import difflib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
import traceback

from common import ROOT, checked, identity, read, resolve, sha, write_new
from cts_native_build import command, copy_file, version_requirements

METHODS = {
    'splatsure': dict(root='tmp/static_sr_wu_20261022/upstream/splatsure-rasterizer', subdir='.',
        commit='83814a5d942991a6a201dfd84f81606bc0382579', package='diff_gaussian_rasterization',
        wrapper='005147b8ae622a4d8327b7cc2c75bf9948b4905c8d4d325ea1cab52c24ff16e4',
        nvcc_prepend='--pre-include cstdint', output_lib='.', existing='runtime/splatsure/acceptance.json'),
    'iesrgs': dict(root='tmp/static_sr_wu_20261022/upstream/iesrgs', subdir='submodules/diff-gaussian-rasterization-rade',
        commit='74d60d5bb0f49974ac0efa3497dbbd9e246480f4', package='diff_gaussian_rasterization_rade',
        wrapper='c1d57f2c1f3c64a31d8d9f289c7e6e965cb4639dfec370474a7fb18d1996c8ec',
        nvcc_prepend='', output_lib='lib', existing='runtime/iesrgs/build_receipt.json'),
    'threedsr': dict(root='tmp/static_sr_wu_20261022/upstream/threedsr', subdir='submodules/diff-gaussian-rasterization',
        commit='8c70a9f64301b4de255672e4aa7da42acc0001a2', package='diff_gaussian_rasterization',
        wrapper='5ebad46b834c2c26bb36a88d0f8f310f03f6dab5e40ebcd87beed68e4ac749d2',
        nvcc_prepend='-include cstdint', output_lib='lib', existing='runtime/threedsr/build_receipt.json'),
}
DEFAULT_OUTPUT = ROOT / 'output/static_sr_wu_20261022/runtime/cts_method_build'
IE_PATCH = ROOT / 'experiments/static_sr_wu_20261022/methods/iesrgs_cuda128_cstdint.patch'
IE_PATCH_SHA = '51d0d1793ecccde66fa651170503e0a4b2e6c012d75268722489c76fb56f5e9d'


def source_files(source):
    return sorted(p for p in source.rglob('*') if p.is_file()
        and not any(part in ('build', '__pycache__', '.git') or part.endswith('.egg-info') for part in p.relative_to(source).parts)
        and p.suffix not in ('.so', '.o', '.pyc', '.pyo'))


def verify_repo(root, commit):
    if command(['git', '-C', root, 'rev-parse', 'HEAD']) != commit:
        raise ValueError('Upstream commit differs: ' + str(root))
    if command(['git', '-C', root, 'status', '--porcelain']):
        raise ValueError('Upstream checkout is not clean: ' + str(root))


def expected_ie_patch(source):
    before = (source / 'cuda_rasterizer/rasterizer_impl.h').read_text()
    after = before.replace('#pragma once\n', '#pragma once\n#include <cstdint>\n', 1)
    if before == after or '#include <cstdint>' in before:
        raise ValueError('IE header no longer matches the original CUDA compatibility patch')
    return ''.join(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
        fromfile='a/cuda_rasterizer/rasterizer_impl.h', tofile='b/cuda_rasterizer/rasterizer_impl.h'))


def create_plan(path):
    checked(dict(path=str(IE_PATCH), sha256=IE_PATCH_SHA))
    rows = []
    for method, spec in METHODS.items():
        root = resolve(spec['root']); source = root / spec['subdir']
        verify_repo(root, spec['commit'])
        if method == 'splatsure':
            glm = source / 'third_party/glm'
            verify_repo(glm, '5c46b9c07008ae65cb81ab79cd677ecc1934b903')
        wrapper = source / spec['package'] / '__init__.py'
        if sha(wrapper) != spec['wrapper']:
            raise ValueError('Method wrapper differs: ' + method)
        old = ROOT / 'output/static_sr_wu_20261022' / spec['existing']
        previous = read(old)
        if previous.get('source_commit', previous.get('upstream_commit')) != spec['commit']:
            raise ValueError('Existing successful build commit differs: ' + method)
        if method == 'iesrgs':
            if expected_ie_patch(source) != IE_PATCH.read_text() or previous['patch']['sha256'] != IE_PATCH_SHA:
                raise ValueError('IE patch differs from previously accepted build')
        elif previous['nvcc_prepend_flags'] != spec['nvcc_prepend']:
            raise ValueError('Compiler include workaround differs from existing build')
        rows.append(dict(method=method, **spec, previous_build_receipt=identity(old),
            files=[dict(relative_path=str(p.relative_to(source)), sha256=sha(p), bytes=p.stat().st_size)
                   for p in source_files(source)]))
    report = dict(schema='static_sr_wu_CTS_method_CPU_build_plan_v1', status='source_checked_target_compile_pending',
        source=identity(__file__), packages=rows, GPU_executed=False,
        IE_builder=identity(Path(__file__).with_name('methods') / 'iesrgs_build_runtime.py'), IE_patch=identity(IE_PATCH),
        CPU_build_only=True, architectures='8.6', max_jobs=2,
        code_changes='IE reuses the existing explicit cstdint patch; Splat/3DSR retain original compiler preinclude only',
        created_utc=datetime.now(timezone.utc).isoformat())
    write_new(path, report)
    return report


def verify_plan(path):
    plan = read(path)
    if plan.get('schema') != 'static_sr_wu_CTS_method_CPU_build_plan_v1' or [r['method'] for r in plan['packages']] != list(METHODS):
        raise ValueError('Wrong three-method build plan')
    checked(plan['IE_builder']); checked(plan['IE_patch'])
    if plan['IE_patch']['sha256'] != IE_PATCH_SHA:
        raise ValueError('IE compatibility patch is not pinned')
    for row in plan['packages']:
        if any(row[k] != v for k, v in METHODS[row['method']].items()):
            raise ValueError('Changed method build specification')
        root = resolve(row['root']); source = root / row['subdir']
        verify_repo(root, row['commit'])
        actual = {str(p.relative_to(source)) for p in source_files(source)}
        if actual != {f['relative_path'] for f in row['files']}:
            raise ValueError('Source file inventory changed')
        for file in row['files']:
            target = source / file['relative_path']
            if not target.resolve().is_relative_to(source.resolve()) or sha(target) != file['sha256']:
                raise ValueError('Method source hash changed: ' + str(target))
        if sha(source / row['package'] / '__init__.py') != row['wrapper']:
            raise ValueError('Method wrapper changed')
    return plan


def build(plan_path, output, cuda_home, cc, cxx, jobs):
    plan = verify_plan(plan_path)
    output, cuda_home = resolve(output), resolve(cuda_home)
    if output.exists():
        raise FileExistsError(output)
    if not (cuda_home / 'bin/nvcc').is_file() or jobs not in (1, 2):
        raise ValueError('Installed CUDA compiler and 1-2 CPU jobs required')
    output.mkdir(parents=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', CUDA_HOME=str(cuda_home), TORCH_CUDA_ARCH_LIST='8.6',
               MAX_JOBS=str(jobs), CC=cc, CXX=cxx, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1')
    env.pop('NVCC_APPEND_FLAGS', None)
    started = time.perf_counter()
    report = dict(schema='static_sr_wu_CTS_three_method_CPU_build_v1', status='running', plan=identity(plan_path),
        script=identity(__file__), host=socket.gethostname(), packages=[], GPU_executed=False,
        old_binaries_replaced=False, cuda_home=str(cuda_home), architectures='8.6', max_jobs=jobs,
        cc=command([cc, '--version']), cxx=command([cxx, '--version']),
        nvcc=command([cuda_home/'bin/nvcc', '--version']))
    try:
        code="import json,torch,sys;print(json.dumps(dict(torch=torch.__version__,python_abi=sys.implementation.cache_tag,CUDA_initialized=torch.cuda.is_initialized())))"
        info = json.loads(command([sys.executable, '-c', code], env=env))
        if info['torch'] != '2.7.1+cu128' or info['CUDA_initialized']:
            raise ValueError('Use verified torch2.7.1+cu128 CPU-only build environment')
        for row in plan['packages']:
            method = row['method']; out = output/method
            original = resolve(row['root'])/row['subdir']
            env['NVCC_PREPEND_FLAGS'] = row['nvcc_prepend']
            if method == 'iesrgs':
                args = [sys.executable, str(checked(plan['IE_builder'])), '--upstream', str(resolve(row['root'])),
                    '--out', str(out), '--cuda-home', str(cuda_home), '--architectures', '8.6', '--jobs', str(jobs)]
                log = output/'iesrgs.builder.log'
                with log.open('x') as stream:
                    subprocess.run(args, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
                if sha(out/'cuda128_cstdint.patch') != IE_PATCH_SHA:
                    raise ValueError('Existing IE builder generated a different patch')
            else:
                out.mkdir()
                work = out/'source_cuda128'
                for file in row['files']:
                    copy_file(original/file['relative_path'], work/file['relative_path'])
                library = out/row['output_lib']
                args = [sys.executable, 'setup.py', 'build_ext', '--build-lib', str(library),
                    '--build-temp', str(out/'build_cuda128')]
                log = out/'build_cuda128.log'
                with log.open('x') as stream:
                    subprocess.run(args, cwd=work, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
                copy_file(original/row['package']/'__init__.py', library/row['package']/'__init__.py')
            package = out/row['output_lib']/row['package']
            binaries = list(package.glob('_C*.so'))
            if len(binaries) != 1 or sha(package/'__init__.py') != row['wrapper']:
                raise ValueError('Fresh method binary/wrapper contract failed: ' + method)
            result = dict(method=method, source_commit=row['commit'], build_status='success',
                upstream_commit=row['commit'], torch_version=info['torch'], python_abi=info['python_abi'],
                gpu_executed=False, gpu_verified=False, cpu_import_verified=False,
                wrapper=dict(path=str(package/'__init__.py'), sha256=sha(package/'__init__.py')),
                binary=dict(name=binaries[0].name, sha256=sha(binaries[0])),
                binary_identities=[identity(binaries[0])], source_plan=identity(plan_path),
                nvcc_prepend_flags=row['nvcc_prepend'], source_patches=[plan['IE_patch']] if method=='iesrgs' else [],
                architectures=['sm_86'], cuda_home=str(cuda_home), max_jobs=jobs,
                log=identity(log), **version_requirements(binaries[0]))
            if method == 'iesrgs':
                receipt = read(out/'build_receipt.json')
                receipt.update(result)
                (out/'build_receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
            else:
                write_new(out/'build_receipt.json', result)
            report['packages'].append(result)
        # The actual existing loaders, simultaneously, in a fresh CPU process.
        check = ("import json,sys,torch;sys.path.insert(0,sys.argv[1]);from methods.splatsure_prepare import load_official_rasterizer;"
            "from methods.iesrgs_renderer import raster_api as ie;from methods.threedsr_renderer import raster_api as mip;"
            "from pathlib import Path;o=Path(sys.argv[2]);s=load_official_rasterizer(sys.argv[3],o/'splatsure');"
            "i=ie(o/'iesrgs');m=mip(upstream=sys.argv[4],binary_root=o/'threedsr/lib');"
            "print(json.dumps(dict(CUDA_initialized=torch.cuda.is_initialized(),modules={k:dict(path=v.__file__,"
            "settings=list(v.GaussianRasterizationSettings._fields)) for k,v in [('splatsure',s),('iesrgs',i),('threedsr',m)]})))")
        imported = json.loads(command([sys.executable, '-c', check, Path(__file__).parent, output,
            resolve(METHODS['splatsure']['root']), resolve(METHODS['threedsr']['root'])], env=env))
        if imported['CUDA_initialized'] or len({v['path'] for v in imported['modules'].values()}) != 3:
            raise ValueError('CPU coexistence check initialized CUDA or aliased method modules')
        for row in report['packages']:
            method = row['method']; path = output/method/'build_receipt.json'
            data = read(path)
            data.update(cpu_import_verified=True, cpu_import=imported['modules'][method],
                        status='target_CPU_compile_import_coexistence_passed_GPU_pending')
            path.write_text(json.dumps(data, indent=2)+'\n')
            if method == 'splatsure':
                write_new(output/method/'acceptance.json', data)
        verify_plan(plan_path)
        report.update(status='target_CPU_compile_import_coexistence_passed_GPU_pending', CPU_import=imported,
            final_build_receipts={method:identity(output/method/'build_receipt.json') for method in METHODS})
    except BaseException:
        report.update(status='failed', error=traceback.format_exc())
        raise
    finally:
        report['elapsed_seconds'] = time.perf_counter()-started
        write_new(output/'build_receipt.json', report)
    return output/'build_receipt.json'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    p = sub.add_parser('plan'); p.add_argument('--out', type=Path, required=True)
    p = sub.add_parser('verify'); p.add_argument('--plan', type=Path, required=True)
    p = sub.add_parser('build'); p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--out', type=Path, default=DEFAULT_OUTPUT)
    p.add_argument('--cuda-home', type=Path, required=True)
    p.add_argument('--cc', default='gcc'); p.add_argument('--cxx', default='g++')
    p.add_argument('--jobs', type=int, default=2)
    args = parser.parse_args()
    if args.mode == 'plan':
        create_plan(resolve(args.out)); print(identity(args.out))
    elif args.mode == 'verify':
        plan=verify_plan(args.plan); print(dict(status='CPU_source_hashes_passed', files=sum(len(r['files']) for r in plan['packages'])))
    else:
        print(identity(build(args.plan, args.out, args.cuda_home, args.cc, args.cxx, args.jobs)))


if __name__ == '__main__':
    main()
