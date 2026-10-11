"""Owned independent50-step LR gradient/render smoke, never a formal parent.

The root's local event supervisor invokes this through a persistent remote
service after input synchronization and before long teacher/training work.
Import and --validate-only do not query GPUs or construct a model.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback
from common import DATA,HERE,OUT,REGISTRY,ROOT,entry,local,mkdir,read,sha,write,disk_guard
import persistent_runner as runner
import teacher_screened


def paths(condition,seed):
    return dict(manifest=DATA/'prepared'/condition/f'manifest_train_ready_seed{seed}.json',
                parent=REGISTRY/'full_LR_prefixes'/condition/f'seed_{seed}',
                teacher_plan=REGISTRY/'teachers'/condition/'plan.json')


def validate(a):
    if a.condition not in ('dgH_cube','l3H_long360') or a.seed!=20261007:
        raise ValueError('Only initial half-rung seed07 diagnostic smoke is authorized')
    expected_index=1 if a.condition=='l3H_long360' else 0
    if a.gpu!=expected_index:
        raise ValueError('Long360 startup uses A100 GPU1; DeskGames startup uses CTS GPU0')
    out=local(a.out)
    if not out.is_relative_to(OUT/'smoke') or out.exists():
        raise FileExistsError('Require absent fresh diagnostic-only smoke output')
    p=paths(a.condition,a.seed)
    m,points,original,extra=runner.prepared_inputs(a.condition,a.seed,p['manifest'])
    gate=teacher_screened.gate(p['teacher_plan'],a.z5_receipt,a.z5_samples)
    refs=[entry(q) for q in [p['manifest'],p['parent']/'config.json',p['teacher_plan'],points,original,*extra,local(a.z5_receipt),local(a.z5_samples),*sorted(HERE.glob('*.py'))]]
    plan=read(p['teacher_plan']);generator_ref=plan['generator'];generator=local(generator_ref['path'])
    if entry(generator)!=generator_ref or generator_ref['sha256']!=plan['frozen_recipe']['generator_sha256']:
        raise ValueError('Frozen original teacher generator changed before startup')
    refs.extend([entry(ROOT/rel) for rel in runner.METRIC_SOURCE_PATHS]);refs.append(entry(generator))
    parent=read(p['parent']/'config.json')
    for rel,digest in parent['runtime_source_sha256'].items():
        q=ROOT/'third_party/4dgs_843d5ac_runtime'/rel
        if sha(q)!=digest:raise ValueError('Pinned runtime source changed before startup')
        refs.append(entry(q))
    for rel,digest in parent['project_sources'].items():
        q=ROOT/rel
        if sha(q)!=digest:raise ValueError('Pinned project source changed before startup')
        refs.append(entry(q))
    return dict(paths=p,source_identity=refs,python_inventory=sorted(q.name for q in HERE.glob('*.py')),gate=gate)


def execute(a,accepted):
    out=local(a.out);mkdir(out.parent);out.mkdir(mode=0o2775);out.chmod(0o2775);mkdir(out/'logs');mkdir(out/'events')
    write(out/'creation.json',dict(PID=os.getpid(),status='startup_created_GPU_not_queried',source=entry(__file__)))
    gpu=runner.resolve_gpu(a.uuid)
    if gpu['uuid']!=a.uuid or int(gpu['index'])!=a.gpu:
        raise ValueError('Startup GPU index/UUID differs')
    if ('A100' not in gpu['name'] if a.condition=='l3H_long360' else '3090' not in gpu['name']) or '5090' in gpu['name']:
        raise ValueError('Wrong startup hardware')
    namespace=accepted['python_inventory'];sources=accepted['source_identity']
    record=dict(status='running_diagnostic50steps',diagnostic_only=True,eligible_for_formal_parent=False,
                host=socket.gethostname(),PID=os.getpid(),gpu=gpu,condition=a.condition,seed=a.seed,
                source_identity=sources,source=entry(__file__),gate=accepted['gate'],started_unix=time.time())
    write(out/'identity.json',record)
    def child(stage,argv):
        runner.check_frozen_identity(sources,namespace);storage=disk_guard(ROOT,15 if stage=='coarse50' else 3)
        idle=runner.check_gpu_idle(gpu,[])
        cmd=[sys.executable,'-u','-B',str(HERE/'entry.py'),*argv]
        with (out/'logs'/f'{stage}.log').open('xb') as log:
            proc=subprocess.Popen(cmd,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu['uuid'],PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4'),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            write(out/'events'/f'{stage}_started.json',dict(pid=proc.pid,argv=cmd,resource=idle,disk_guard=storage,started_unix=time.time()))
            begin=time.monotonic();code=proc.wait()
        write(out/'events'/f'{stage}_returned.json',dict(exit_code=code,seconds=time.monotonic()-begin,log=entry(out/'logs'/f'{stage}.log')))
        runner.check_frozen_identity(sources,namespace);disk_guard(ROOT,0)
        if code:raise RuntimeError(f'Startup {stage} exited{code}; no formal task may follow')
    try:
        with runner.gpu_claims([gpu],OUT/'claims',[]):
            coarse=out/'coarse50';evaluation=out/'eval3'
            parent=accepted['paths']['parent']
            child('coarse50',['lr','--parent',str(parent.relative_to(ROOT)),'--coarse-only','--smoke-coarse-steps','50','--arm','diagnostic_coarse50','--out',str(coarse.relative_to(ROOT))])
            cp=coarse/'checkpoint_fine_00000.pt';done=read(coarse/'complete.json');override=read(coarse/'smoke_override.json')
            if done.get('status')!='completed' or done['checkpoint']!=entry(cp) or done['state']['coarse_iteration']!=50 or done['state']['fine_iteration']!=0 or override.get('not_eligible_for_formal_parent') is not True:
                raise ValueError('Smoke checkpoint identity/budget/diagnostic marker differs')
            child('eval3',['eval','--checkpoint',str(cp.relative_to(ROOT)),'--cull','--frames','0','150','299','--out',str(evaluation.relative_to(ROOT)),'--label',a.condition+'_diagnostic50steps'])
            summary=read(evaluation/'summary.json')
            with (evaluation/'metrics_per_observation.csv').open() as f:rows=list(csv.DictReader(f))
            test=[r for r in rows if r['scope']=='test']
            if summary.get('status')!='completed' or summary.get('frames')!=3 or sorted((r['camera'],int(r['frame'])) for r in test)!=[('cam00',0),('cam00',150),('cam00',299)]:
                raise ValueError('Startup3-frame render evaluation incomplete')
            if not all(math.isfinite(float(r[k])) for r in test for k in ('psnr','ssim','lpips')):
                raise ValueError('Startup metrics contain nonfinite values')
            final=dict(record,status='completed_diagnostic50steps_and_eval3',seconds=time.time()-record['started_unix'],checkpoint=entry(cp),smoke_override=entry(coarse/'smoke_override.json'),evaluation=entry(evaluation/'summary.json'),metrics_csv=entry(evaluation/'metrics_per_observation.csv'),
                       diagnostic50steps=True,formal_author_schedule_unchanged=[3000,14000],meaning='new scene gradient/runtime/render path acceptance only; no quality or convergence claim')
            write(out/'complete.json',final);return final
    except BaseException:
        write(out/f'failure_{time.time_ns()}.json',dict(record,status='failed_startup_preserved_no_formal_training',error=traceback.format_exc(),automatic_retry=False));raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--condition',required=True,choices=['dgH_cube','l3H_long360']);p.add_argument('--seed',type=int,default=20261007)
    p.add_argument('--gpu',required=True,type=int);p.add_argument('--uuid',required=True);p.add_argument('--out',required=True,type=Path)
    p.add_argument('--z5-receipt',required=True,type=Path);p.add_argument('--z5-samples',required=True,type=Path);p.add_argument('--validate-only',action='store_true')
    a=p.parse_args()
    try:
        accepted=validate(a)
        if a.validate_only:
            print(json.dumps(dict(status='CPU_startup_prerequisites_validated_not_launched',condition=a.condition,source_count=len(accepted['source_identity']))));return
        print(json.dumps(execute(a,accepted),ensure_ascii=False))
    except BaseException:
        out=local(a.out)
        if not a.validate_only and out.is_relative_to(OUT/'smoke'):
            if not out.exists():
                mkdir(out);write(out/'preflight_failure.json',dict(status='failed_startup_preflight_no_formal_training',error=traceback.format_exc(),source=entry(__file__),automatic_retry=False))
            elif (out/'creation.json').is_file() and read(out/'creation.json')['PID']==os.getpid():
                write(out/f'startup_failure_{time.time_ns()}.json',dict(status='failed_startup_no_formal_training',error=traceback.format_exc(),source=entry(__file__),automatic_retry=False))
        raise
if __name__=='__main__':main()
