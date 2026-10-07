"""CPU fault injection for dispatch refusal, exit callbacks and bound recovery."""
from pathlib import Path
import argparse,json,sys,time,traceback
from types import SimpleNamespace
from unittest.mock import patch
from fp_common import ROOT,HERE,OUT,read,write,sha,entry,module

def main():
    started=time.monotonic();directory=OUT/'operator_checks/run_suite_CPU'/str(time.time_ns())
    directory.mkdir(parents=True)
    suite=module('fp_checked_durable_suite',HERE/'run_suite.py');checks={}
    # A newly launched foreign process must stop dispatch despite a good prior
    # sample. No CUDA child or source mutation occurs in this fixture.
    with patch.object(suite,'free_sample',side_effect=suite.ResourceBusy('foreign queue started')):
        with patch.object(suite.legacy,'call') as child:
            try:suite.run(['never-launch'],directory/'refused.log',{},'fixture_GPU')
            except suite.ResourceBusy:pass
            else:raise AssertionError('Busy GPU dispatched')
            assert child.call_count==0
    checks['immediate_prelaunch_foreign_job_refusal']='passed_no_child_started'
    marker=directory/'returned.json';callback=directory/'callback.json'
    command=[sys.executable,'-c','import json,time,sys; time.sleep(.05); open(sys.argv[1],"w").write(json.dumps({"event":"child completed"}))',str(marker)]
    suite.run(command,directory/'child.log',None)
    assert read(marker)['event']=='child completed'
    write(callback,dict(event='immediate callback after subprocess return',child=entry(marker)))
    checks['successful_exit_callback']='passed_process_return_no_polling'
    try:suite.run([sys.executable,'-c','raise SystemExit(9)'],directory/'failed_child.log',None)
    except __import__('subprocess').CalledProcessError as error:assert error.returncode==9
    else:raise AssertionError('Failed child accepted')
    checks['failed_child_no_completion_claim']='passed_exit9_propagated'
    run=directory/'runs/r1_B0';run.mkdir(parents=True)
    write(run/'config.json',dict(fixture=True))
    try:suite.continuation(run)
    except ValueError:pass
    else:raise AssertionError('No checkpoint silently cold restarted')
    for cursor in (100,3000):
        cp=run/f'checkpoint_{6000+cursor}.pt';cp.write_bytes(str(cursor).encode())
        write(cp.with_suffix('.json'),dict(path=str(cp.relative_to(ROOT)),sha256=sha(cp),metadata=dict(suffix_step=cursor)))
    assert suite.continuation(run).name=='checkpoint_9000.pt'
    checks['recovery_selects_highest_bound_checkpoint']='passed'
    (run/'checkpoint_9000.pt').write_bytes(b'corrupt')
    try:suite.continuation(run)
    except AssertionError:pass
    else:raise AssertionError('Corrupt checkpoint accepted')
    checks['corrupt_checkpoint_rejected']='passed'
    # Endpoint budgets are a separate requirement from file existence.
    fake_root=directory/'endpoint_root';target=fake_root/'runs/r1_B0';target.mkdir(parents=True)
    cp=target/'checkpoint_12000.pt';cp.write_bytes(b'CPU-only fixture, not a model')
    write(target/'segment_0000_6000.json',dict(start=0,suffix_endpoint=6000,checkpoint=entry(cp)))
    receipt=dict(status='completed_training',method='B0',updates=6000,training_rgb_forwards=18000,
        moment_forwards=0,adam_calls=12000,checkpoint=entry(cp),segments=[entry(target/'segment_0000_6000.json')])
    write(target/'complete.json',receipt)
    with patch.object(suite,'OUT',fake_root):
        assert suite.endpoint('r1_B0')==cp
        receipt['training_rgb_forwards']=17997;write(target/'complete.json',receipt)
        try:suite.endpoint('r1_B0')
        except AssertionError:pass
        else:raise AssertionError('Incomplete three-render budget accepted')
    checks['bound_endpoint_and_budget_rejection']='passed'
    write(directory/'receipt.json',dict(status='passed_durable_suite_CPU_fault_injection',checks=checks,
        source=entry(HERE/'run_suite.py'),check_source=entry(Path(__file__)),seconds=time.monotonic()-started,
        CUDA_operations=0,formal_updates=0,optimizer_calls=0,
        scope='Controller only; not native Gaussian training or method quality',process_exit_marker=entry(marker),callback=entry(callback)))
    print(json.dumps(dict(status='passed',checks=len(checks),receipt=str(directory/'receipt.json'))))

if __name__=='__main__':main()
