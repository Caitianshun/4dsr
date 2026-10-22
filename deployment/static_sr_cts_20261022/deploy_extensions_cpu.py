"""Deploy freshly target-built extensions into this isolated task snapshot.

Requires completed CPU-build receipts and no active extension-consuming task.
Backs up each old destination, atomically replaces only the allowlisted files,
and verifies five-extension coexistence through the production CPU loaders.
On verification failure restores this invocation's old destinations. No GPU
kernels, broker operations, shared environment or shared runtime modifications.
"""
from __future__ import annotations
import argparse
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[2]
EXPECTED=Path('/home/cts/Project/4DSR/tasks/static_sr_wu_20261022')
RUNTIME=Path('output/static_sr_wu_20261022/runtime')
CONSUMERS={'gpu_acceptance.py','train_stage.py','run_threedsr.py','render_eval.py','prepare_priors.py'}
ALLOWED_GPU={'external_teacher_acceptance.py','metric_acceptance.py'}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def info(path):
    return dict(path=str(path.relative_to(ROOT)),sha256=sha(path),bytes=path.stat().st_size)


def read(path):return json.loads(Path(path).read_text())


def task_process_check():
    processes=[]
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
        try:argv=[v.decode(errors='replace') for v in (proc/'cmdline').read_bytes().split(b'\0') if v]
        except (OSError,PermissionError):continue
        if not any(str(ROOT) in v for v in argv):continue
        filenames={Path(v).name for v in argv}
        names=filenames & CONSUMERS
        if names:raise RuntimeError('Extension consumer is still running: pid='+proc.name+' scripts='+str(sorted(names)))
        if filenames & ALLOWED_GPU:processes.append(dict(pid=int(proc.name),scripts=sorted(filenames & ALLOWED_GPU),allowed='2D_teacher_or_metric_no_five_extension_dependency'))
    result=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader,nounits'],
                          capture_output=True,text=True,check=True)
    for line in result.stdout.splitlines():
        parts=[x.strip() for x in line.split(',')]
        if not parts or not parts[0].isdigit():continue
        proc=Path('/proc')/parts[0]
        try:argv=[v.decode(errors='replace') for v in (proc/'cmdline').read_bytes().split(b'\0') if v]
        except (OSError,PermissionError):continue
        if any(str(ROOT) in v for v in argv) and not {Path(v).name for v in argv} & ALLOWED_GPU:
            raise RuntimeError('Unknown task GPU process prevents extension deployment: '+parts[0])
    return dict(no_task_extension_consumer=True,allowed_teacher_metric_processes=processes,
                compute_process_listing=result.stdout.strip(),checked_unix=time.time())


def mappings(native, methods):
    nr=read(native/'build_receipt.json');mr=read(methods/'build_receipt.json')
    if nr['status']!='target_CPU_compile_and_import_passed_GPU_pending' or mr['status']!='target_CPU_compile_import_coexistence_passed_GPU_pending':
        raise ValueError('Both target CPU build/import receipts must pass')
    result=[]
    for package in ('simple_knn','diff_gaussian_rasterization'):
        srcdir=native/'lib'/package;destdir=ROOT/'third_party/4dgs_843d5ac_runtime'/package
        if sha(srcdir/'__init__.py')!=sha(destdir/'__init__.py'):raise ValueError('Native wrapper mismatch')
        files=list(srcdir.glob('_C*.so'))
        if len(files)!=1:raise ValueError('Expected one native extension')
        name='simple_knn' if package=='simple_knn' else 'native_depth_rasterizer'
        receipt=next(r for r in nr['packages'] if r['name']==name)
        if sha(files[0])!=receipt['binary']['sha256']:raise ValueError('Native compiled identity changed')
        result.append((files[0],destdir/files[0].name))
    for method,package,lib in [('splatsure','diff_gaussian_rasterization',''),('iesrgs','diff_gaussian_rasterization_rade','lib'),
                               ('threedsr','diff_gaussian_rasterization','lib')]:
        srcdir=methods/method/lib/package;destdir=ROOT/RUNTIME/method/lib/package
        if sha(srcdir/'__init__.py')!=sha(destdir/'__init__.py'):raise ValueError('Method wrapper mismatch')
        files=list(srcdir.glob('_C*.so'))
        if len(files)!=1:raise ValueError('Expected one method extension')
        receipt=read(methods/method/'build_receipt.json')
        if not receipt.get('cpu_import_verified') or sha(files[0])!=receipt['binary']['sha256']:
            raise ValueError('Method compiled identity/import receipt mismatch')
        result.extend([(files[0],destdir/files[0].name),
                       (methods/method/'build_receipt.json',ROOT/RUNTIME/method/'build_receipt.json')])
        if method=='splatsure':result.append((methods/method/'acceptance.json',ROOT/RUNTIME/method/'acceptance.json'))
    return result


def atomic_copy(source,destination):
    temporary=destination.with_name(destination.name+'.cts_cpu_deploy.tmp')
    if temporary.exists():raise FileExistsError(temporary)
    with temporary.open('xb') as f:
        with source.open('rb') as s:shutil.copyfileobj(s,f)
        f.flush();os.fsync(f.fileno())
    if sha(source)!=sha(temporary):raise RuntimeError('Atomic copy hash mismatch')
    os.replace(temporary,destination)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--native',type=Path,default=RUNTIME/'cts_native_build/target_build')
    p.add_argument('--methods',type=Path,default=RUNTIME/'cts_method_build')
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if ROOT!=EXPECTED or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Use the isolated CTS snapshot with GPUs hidden')
    native,methods,out=[(ROOT/v).resolve() for v in (a.native,a.methods,a.out)]
    if any(not x.is_relative_to(ROOT) for x in (native,methods,out)):raise ValueError('Paths escape task snapshot')
    out.mkdir(parents=True,exist_ok=False)
    record=dict(schema='static_sr_CTS_five_extension_deployment_v1',status='checking',GPU_kernels_executed=False,
        shared_runtime_modified=False,shared_environment_modified=False,script_sha256=sha(__file__),files=[],
        build_receipts=[info(native/'build_receipt.json'),info(methods/'build_receipt.json')])
    changed=[]
    lock=(ROOT/'.deployment.lock').open('a')
    try:
        # Keep the descriptor/lock alive through any rollback in the outer except.
        with contextlib.nullcontext(lock):
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            record['resource_check']=task_process_check()
            changes=mappings(native,methods)
            baseline=read(ROOT/RUNTIME/'cts_native_build/source_inventory.json')
            old_hashes={e['binary']['path']:e['binary']['sha256'] for e in baseline['existing_five_binary_ABI_audit']}
            for source,destination in changes:
                row=dict(source=info(source),destination=str(destination.relative_to(ROOT)))
                if destination.suffix=='.so' and sha(destination)!=old_hashes[row['destination']]:
                    raise ValueError('Old snapshot binary differs from preserved source inventory')
                if destination.exists():
                    backup=out/'old_files'/destination.relative_to(ROOT)
                    backup.parent.mkdir(parents=True,exist_ok=True)
                    with backup.open('xb') as f:
                        with destination.open('rb') as old:shutil.copyfileobj(old,f)
                    if sha(backup)!=sha(destination):raise RuntimeError('Old-file backup mismatch')
                    row['old']=info(backup)
                else:row['old']=None
                record['files'].append(row)
            record['precopy_resource_check']=task_process_check()
            for row,(source,destination) in zip(record['files'],changes):
                atomic_copy(source,destination);changed.append((row,destination))
                if sha(destination)!=row['source']['sha256']:raise RuntimeError('Deployed bytes differ')
            code=("import json,sys,torch;from pathlib import Path;r=Path(sys.argv[1]);"
                "sys.path.insert(0,str(r/'experiments/dynamic_sr_cull_retrain_20261009'));import rt;v=rt.check_runtime();"
                "sys.path.insert(0,str(r/'experiments/static_sr_wu_20261022'));"
                "from methods.splatsure_prepare import load_official_rasterizer;"
                "from methods.iesrgs_renderer import raster_api as ie;from methods.threedsr_renderer import raster_api as mip;"
                "s=load_official_rasterizer(r/'tmp/static_sr_wu_20261022/upstream/splatsure-rasterizer',r/'output/static_sr_wu_20261022/runtime/splatsure');"
                "i=ie();m=mip();import simple_knn._C as k;import diff_gaussian_rasterization as w;"
                "print(json.dumps(dict(runtime=v,CUDA_initialized=torch.cuda.is_initialized(),modules={n:x.__file__ for n,x in "
                "[('simple_knn',k),('wu_native',w),('splatsure',s),('iesrgs',i),('threedsr',m)]})))")
            child=subprocess.run([sys.executable,'-c',code,str(ROOT)],capture_output=True,text=True)
            (out/'CPU_import.stdout').write_text(child.stdout);(out/'CPU_import.stderr').write_text(child.stderr)
            if child.returncode:raise RuntimeError('Production CPU import failed: '+child.stderr[-2000:])
            imported=json.loads(child.stdout.strip().splitlines()[-1])
            if imported['CUDA_initialized'] or len(imported['modules'])!=5:raise RuntimeError('CPU import contract failed')
            record['CPU_import']=imported
            record['status']='five_extensions_deployed_SHA_CPU_coexistence_passed_GPU_pending'
    except BaseException:
        record.update(status='failed',error=traceback.format_exc())
        # Restore only changes performed by this invocation; original tasks stay intact.
        for row,destination in reversed(changed):
            if row['old'] is None:destination.unlink()
            else:atomic_copy(ROOT/row['old']['path'],destination)
        record['rolled_back_files']=len(changed)
        raise
    finally:
        (out/'receipt.json').write_text(json.dumps(record,indent=2)+'\n')
        lock.close()
    print(json.dumps(dict(status=record['status'],receipt=info(out/'receipt.json'))))


if __name__=='__main__':main()
