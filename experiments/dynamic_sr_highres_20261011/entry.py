"""Fresh runtime dispatch; unchanged cullretrain/modsr/refs process-local behavior."""
import json, os, runpy, sys, time, traceback
import importlib
from pathlib import Path
from common import *
def main():
    argv=sys.argv[1:]
    if not argv:raise ValueError('Expected lr/sr/eval/extras')
    kind=argv[0]
    smoke_coarse=None
    if '--smoke-coarse-steps' in argv:
        i=argv.index('--smoke-coarse-steps');smoke_coarse=int(argv[i+1]);del argv[i:i+2]
        if kind!='lr' or '--coarse-only' not in argv or not 1<=smoke_coarse<=50:raise ValueError('Smoke override applies only to LR coarse1..50')
    if '--out' not in argv:raise ValueError('Every task needs new --out')
    out=local(argv[argv.index('--out')+1])
    if not out.is_relative_to(OUT) or out.exists():raise ValueError(f'Output outside fresh namespace or already exists: {out}')
    if smoke_coarse is not None and 'smoke' not in out.parts:raise ValueError('Diagnostic smoke override requires a smoke output namespace')
    if kind not in ['lr','sr','eval','extras']:raise ValueError('Unsupported task kind')
    ref=any(f in argv for f in ['--image-source','--teacher-source','--hr-root'])
    script=ROOT/('experiments/dynamic_sr_modules_20261010/refs.py' if ref else 'experiments/dynamic_sr_modules_20261010/modsr.py' if kind=='extras' else 'experiments/dynamic_sr_cull_retrain_20261009/cullretrain.py')
    mkdir(out)
    write(out/'dispatch.json',dict(argv=argv,entry=entry(__file__),target_source=entry(script),source_adaptation='fresh data namespace; reference switches use unchanged refs.py',status='running',smoke_coarse_steps=smoke_coarse))
    os.chdir(ROOT);os.environ['PYTHONDONTWRITEBYTECODE']='1';sys.path.insert(0,str(script.parent));sys.argv=[str(script),*argv]
    try:
        if smoke_coarse is None:runpy.run_path(str(script),run_name='__main__')
        else:
            cull_dir=ROOT/'experiments/dynamic_sr_cull_retrain_20261009';sys.path.insert(0,str(cull_dir));cr=importlib.import_module('cullretrain');original=cr.configuration
            def configuration(plan):
                h,opt,pipe,sh=original(plan);opt.coarse_iterations=smoke_coarse;return h,opt,pipe,sh
            cr.configuration=configuration
            target=importlib.import_module('refs' if ref else 'cullretrain')
            target.main(argv)
            write(out/'smoke_override.json',dict(diagnostic_only=True,coarse_iterations=smoke_coarse,formal_config_unchanged=True,not_eligible_for_formal_parent=True))
        write(out/'dispatch_complete.json',dict(status='awaiting_review',argv=argv,completion_source='process returned successfully; native completion artifacts still require supervisor integrity acceptance',entry=entry(__file__),target_source=entry(script)))
    except BaseException as e:
        # SystemExit0 is success; every real exception gets a preserved failure.
        if isinstance(e,SystemExit) and e.code in [None,0]:return
        mkdir(out);write(out/f'dispatch_failure_{time.time_ns()}.json',dict(status='failed',argv=argv,entry=entry(__file__),target_source=entry(script),error=traceback.format_exc(),partial_outputs_preserved=True));raise
if __name__=='__main__':main()
