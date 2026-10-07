"""CPU fault-injection checks of the actual controller helper functions.

Functions/expressions are AST-extracted verbatim from the current train.py;
importing its GPU model stack is unnecessary. Tiny independent CPU Adam models
test saved state and stochastic continuation. They are fixtures, not experiment
training, and their optimizer calls are accounted for separately.
"""
from __future__ import annotations
import argparse
import ast
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import shutil
import time
import traceback
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from fp_common import ROOT,HERE,OUT,write,read,entry,sha,bound,local


def extracted_scope(source, fixture):
    tree=ast.parse(source)
    selected=[node for node in tree.body if isinstance(node,ast.FunctionDef) and
              node.name in ('rng','json_value','optimizer_audit','save_checkpoint','recover_log_tail','committed_segments')]
    assert len(selected)==6
    def refinement(model):
        return dict(children=dict(parameter=model.b.detach().clone()),child_optimizer=model.child_optimizer.state_dict())
    scope=dict(torch=torch,np=np,random=random,json=json,math=__import__('math'),os=__import__('os'),
               time=time,shutil=shutil,Path=Path,ROOT=ROOT,HERE=HERE,OUT=fixture,write=write,read=read,
               sha=sha,bound=bound,entry=entry,local=local,refinement_state=refinement)
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(HERE/'train.py'),'exec'),scope)
    return tree,scope


def equal(left,right):
    if isinstance(left,torch.Tensor):return isinstance(right,torch.Tensor) and torch.equal(left,right)
    if isinstance(left,np.ndarray):return isinstance(right,np.ndarray) and np.array_equal(left,right)
    if isinstance(left,dict):return left.keys()==right.keys() and all(equal(left[k],right[k]) for k in left)
    if isinstance(left,(list,tuple)):return len(left)==len(right) and all(equal(a,b) for a,b in zip(left,right))
    return left==right


def fake_model(a=None,b=None):
    a=torch.nn.Parameter(torch.tensor([.1,.2],dtype=torch.float64) if a is None else a.clone())
    b=torch.nn.Parameter(torch.tensor([-.1,.3],dtype=torch.float64) if b is None else b.clone())
    first=torch.optim.Adam([dict(params=[a],name='base',lr=.012)],betas=(.9,.999))
    second=torch.optim.Adam([dict(params=[b],name='children',lr=.023)],betas=(.9,.999))
    class Gaussian:
        optimizer=first
        def capture(self):
            data=[None]*13;data[0]=a.detach().clone();data[12]=first.state_dict();return tuple(data)
    return SimpleNamespace(a=a,b=b,g=Gaussian(),child_optimizer=second,h=SimpleNamespace(test='CPU'),
                           o=SimpleNamespace(test='CPU'),checkpoint=dict(metadata=dict(extent=1.)))


def fixture_update(model):
    draw=torch.randn(2,dtype=torch.float64)+float(np.random.normal())+random.random()
    for optimizer in (model.g.optimizer,model.child_optimizer):optimizer.zero_grad(set_to_none=True)
    loss=((model.a+2*model.b-draw).square()).mean()
    loss.backward()
    model.g.optimizer.step();model.child_optimizer.step()


def logrow(step,train_s,wall_s):
    return dict(suffix_step=step,rgb_forwards=3,moment_forwards=3,adam_calls=2,
                autograd_calls=1,train_s=train_s,wall_s=wall_s,peak_gb=.001)


def recovery_checks(scope,fixture):
    results={}
    for missing_log in (False,True):
        directory=fixture/('missing_log' if missing_log else 'no_log_tail');directory.mkdir()
        if not missing_log:(directory/'training.jsonl').write_text(json.dumps(logrow(100,10.,11.))+'\n')
        attempt=dict(suffix_step=101,rgb=2,moments=1,autograd_calls=0,adam_calls=0,
                     status='active_operation_cost_unknown_until_return',active_operation='moment_forward')
        write(directory/'last_attempt.json',attempt)
        scope['recover_log_tail'](directory,100)
        incidents=list((directory/'recovery_incidents').glob('*/receipt.json'))
        assert len(incidents)==1,'Unlogged returned operations were discarded'
        receipt=read(incidents[0]);assert receipt['incomplete_attempt']==attempt
        assert read(incidents[0].parent/'last_attempt.json')==attempt
        assert receipt['confirmed_completed_updates']==0
        retained=(directory/'training.jsonl').read_text().splitlines()
        assert len(retained)==(0 if missing_log else 1)
        results['absent_log' if missing_log else 'tail_empty']=dict(status='passed',
            preserved_incomplete_attempt_RGB=2,preserved_incomplete_attempt_moments=1,
            incomplete_active_moment_cost='unknown',confirmed_logged_updates=0)
    directory=fixture/'logged_tail';directory.mkdir()
    rows=[logrow(100,10.,11.),logrow(101,1.,1.1),logrow(102,2.,2.2)]
    (directory/'training.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows)+'{"suffix_step":103')
    attempt=dict(suffix_step=103,rgb=3,moments=2,autograd_calls=0,adam_calls=0,
                 active_operation='moment_forward')
    write(directory/'last_attempt.json',attempt)
    scope['recover_log_tail'](directory,100)
    receipt=read(next((directory/'recovery_incidents').glob('*/receipt.json')))
    assert receipt['confirmed_completed_updates']==2
    assert receipt['confirmed_RGB_forwards']==6 and receipt['confirmed_moment_forwards']==6
    assert receipt['confirmed_Adam_calls']==4 and receipt['incomplete_attempt']==attempt
    assert [json.loads(line)['suffix_step'] for line in (directory/'training.jsonl').read_text().splitlines()]==[100]
    results['logged_and_partial_tail']=dict(status='passed',completed_failed_updates=2,
        completed_failed_RGB=6,completed_failed_moments=6,completed_failed_Adam_calls=4,
        additional_incomplete_RGB=3,additional_incomplete_moments=2,active_operation_cost='unknown')
    return results


def segment_checks(scope,fixture):
    results={}
    for resumed in (False,True):
        directory=fixture/('segment_resumed_process' if resumed else 'segment_one_process');directory.mkdir()
        ck100=directory/'checkpoint_6100.pt';ck100.write_bytes(b'valid CPU fixture checkpoint marker 100')
        ck3000=directory/'checkpoint_9000.pt';ck3000.write_bytes(b'valid CPU fixture checkpoint marker 3000')
        write(directory/'segment_0000_0100.json',dict(start=0,suffix_endpoint=100,updates=100,
                                                     checkpoint=entry(ck100)))
        # The previous process's row100 has a different time origin if resumed.
        rows=[logrow(100,10.,11.)]+[logrow(i,(i-100)/100 if resumed else i/100,
                                          (i-100)/100+.5 if resumed else i/100+.5) for i in range(101,3001)]
        (directory/'training.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
        checkpoint=dict(metadata=dict(suffix_step=3000,method='MX',repeat='1',segment_start=100 if resumed else 0,
                                      gpu='CPU fixture',calibration_sha256='fake'),multiview_footprint=dict(sources={}))
        scope['committed_segments'](directory,3000,checkpoint)
        recovered=read(directory/'segment_0100_3000.json')
        assert recovered['training_rgb_forwards']==8700 and recovered['moment_forwards']==8700
        assert recovered['adam_calls']==5800 and recovered['updates']==2900
        expected_train=29. if resumed else 20.;expected_wall=29.5 if resumed else 19.5
        # One-process row100 wall11, row3000 wall30.5 =>19.5. Resumed row3000
        # wall29.5 begins at start100 and must never subtract the old wall11.
        assert abs(recovered['train_s']-expected_train)<1e-12,(resumed,recovered['train_s'],expected_train)
        assert abs(recovered['wall_s']-expected_wall)<1e-12,(resumed,recovered['wall_s'],expected_wall)
        results['resumed_process' if resumed else 'single_process']=dict(status='passed',updates=2900,
            RGB=8700,moments=8700,Adam_calls=5800,train_seconds=recovered['train_s'],wall_seconds=recovered['wall_s'])
        scope['committed_segments'](directory,3000,checkpoint)  # idempotent repair
        assert len(list(directory.glob('segment_*.json')))==2
    directory=fixture/'segment_overlap';directory.mkdir()
    ck=directory/'dummy.pt';ck.write_bytes(b'checkpoint')
    for start,stop in ((0,100),(90,200)):
        write(directory/f'segment_{start:04d}_{stop:04d}.json',dict(start=start,suffix_endpoint=stop,
            updates=stop-start,checkpoint=entry(ck)))
    try:
        scope['committed_segments'](directory,200,None)
        raise AssertionError('Overlapping segments accepted')
    except AssertionError as error:
        assert str(error)!='Overlapping segments accepted'
    results['overlap_rejected']=dict(status='passed')
    return results


def checkpoint_checks(scope,fixture):
    torch.manual_seed(111);np.random.seed(112);random.seed(113)
    model=fake_model();audit=scope['optimizer_audit'](model);assert audit['disjoint']
    overlap=SimpleNamespace(g=model.g,child_optimizer=torch.optim.Adam([model.a],lr=.1))
    try:
        scope['optimizer_audit'](overlap);raise AssertionError('Overlapping Adam accepted')
    except AssertionError as error:assert str(error)!='Overlapping Adam accepted'
    for _ in range(10):fixture_update(model)
    directory=fixture/'checkpoint_state';directory.mkdir()
    write(fixture/'protocol.json',dict(test='CPU fault-injection fixture, not registered experiment'))
    schedule=fixture/'schedule_1.json';write(schedule,dict(test='CPU'))
    p=dict(manifest=dict(sha256='fixture'),schedules={'1':entry(schedule)})
    a=SimpleNamespace(method='MX',repeat='1',out=directory)
    with patch.object(torch.cuda,'get_rng_state_all',return_value=[]),patch.object(torch.cuda,'get_device_name',return_value='CPU fixture'):
        checkpoint_path=scope['save_checkpoint'](model,a,p,None,100,{'test':'CPU'},0,10.,time.monotonic(),'fixture')
    ck=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
    assert ck['model'][12]['param_groups'][0]['lr']==.012
    assert ck['motion_refinement']['child_optimizer']['param_groups'][0]['lr']==.023
    assert ck['samplers']['cursor']==100 and ck['rng']['cuda']==[]
    for _ in range(10):fixture_update(model)
    expected=dict(a=model.a.detach().clone(),b=model.b.detach().clone(),
                  first=model.g.optimizer.state_dict(),second=model.child_optimizer.state_dict(),
                  torch_rng=torch.get_rng_state(),numpy_rng=np.random.get_state(),python_rng=random.getstate())
    resumed=fake_model(ck['model'][0],ck['motion_refinement']['children']['parameter'])
    resumed.g.optimizer.load_state_dict(ck['model'][12]);resumed.child_optimizer.load_state_dict(ck['motion_refinement']['child_optimizer'])
    torch.set_rng_state(ck['rng']['torch']);np.random.set_state(ck['rng']['numpy']);random.setstate(ck['rng']['python'])
    for _ in range(10):fixture_update(resumed)
    actual=dict(a=resumed.a.detach(),b=resumed.b.detach(),first=resumed.g.optimizer.state_dict(),
                second=resumed.child_optimizer.state_dict(),torch_rng=torch.get_rng_state(),
                numpy_rng=np.random.get_state(),python_rng=random.getstate())
    assert equal(expected,actual),'Saved native-format payload lost Adam/RNG/learning-rate state'
    return dict(status='passed',actual_save_checkpoint_function_used=True,two_real_CPU_Adam_states_exact=True,
        stochastic_continuation_exact=True,learning_rates=[.012,.023],saved_cursor=100,
        toy_CPU_optimizer_calls=60,formal_experiment_updates=0,CUDA_initialized=False)


def clamp_checks(tree):
    main=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='main')
    names=('degraded_raw','clamp_fractions','errors','lr_values','lr_term')
    nodes=[]
    loop=next(node for node in ast.walk(main) if isinstance(node,ast.For) and
              isinstance(node.target,ast.Name) and node.target.id=='cursor')
    for node in loop.body:
        if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name) and node.targets[0].id in names:
            nodes.append(node)
        if isinstance(node,ast.If) and isinstance(node.test,ast.Subscript) and ast.unparse(node.test)=="definition['E']":
            nodes.append(node)
    nodes.sort(key=lambda node:node.lineno)
    assert len(nodes)==6
    rgb=torch.zeros((3,64,64),dtype=torch.float64);rgb[:,:,32:]=1.
    inputs=[rgb.clone().requires_grad_() for _ in range(3)]
    scope=dict(torch=torch,rgbs=inputs,lrs=[torch.zeros((3,16,16),dtype=torch.float64) for _ in range(3)],
               definition=dict(lr_weights=[1.,0.,0.],E=True),cal=dict(kappa=1.25))
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(HERE/'train.py'),'exec'),scope)
    assert all(float(((v<0)|(v>1)).float().mean())==0 for v in inputs)
    assert all(float(v)>0 for v in scope['clamp_fractions']),'Logged fraction only measures RGB, not D0 overshoot'
    assert all(bool((error>=0).all() & (error<=1).all()) for error in scope['errors'])
    scope['lr_term'].backward()
    assert inputs[0].grad is not None and inputs[1].grad is None and inputs[2].grad is None
    return dict(status='passed',actual_train_expressions_used=True,raw_RGB_out_of_range_fraction=0.,
        actual_D0_clamp_fractions=[float(v) for v in scope['clamp_fractions']],
        E_uses_registered_anchor_degradation=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=OUT/'operator_checks'/'train_semantics')
    a=parser.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ');run=a.out/f'{stamp}_{time.time_ns()}'
    run.mkdir();fixture=run/'fixtures';fixture.mkdir();started=time.monotonic()
    source=(HERE/'train.py').read_text();(run/'train_source_snapshot.py').write_text(source)
    shutil.copyfile(Path(__file__).resolve(),run/'test_source_snapshot.py')
    checks={};stage='extract'
    try:
        tree,scope=extracted_scope(source,fixture)
        for stage,fn in [('checkpoint_state',lambda:checkpoint_checks(scope,fixture)),
                         ('recovery_tail',lambda:recovery_checks(scope,fixture)),
                         ('committed_segments',lambda:segment_checks(scope,fixture)),
                         ('actual_D_clamp',lambda:clamp_checks(tree))]:
            checks[stage]=fn()
        result=dict(status='passed_CPU_actual_training_controller_fault_injection_checks',checks=checks)
    except BaseException:
        result=dict(status='failed_CPU_actual_training_controller_fault_injection_checks',
                    stage=stage,checks=checks,error=traceback.format_exc())
    result.update(started_utc=stamp,seconds=time.monotonic()-started,device='CPU',
                  training_source_snapshot=entry(run/'train_source_snapshot.py'),
                  test_source=entry(run/'test_source_snapshot.py'),formal_updates=0,RGB_renderer_forwards=0,
                  moment_renderer_forwards=0,toy_CPU_optimizer_calls=60 if 'checkpoint_state' in checks else 'partial or zero',
                  limits=['AST extraction tests actual controller helpers and LR expressions; native renderer and real parent reload still need an authorized GPU fixture.'])
    write(run/'receipt.json',result)
    print(json.dumps(dict(status=result['status'],stage=stage,receipt=str(run/'receipt.json'),seconds=result['seconds'])),flush=True)
    if result['status'].startswith('failed'):raise RuntimeError(result['error'])
    write(a.out/'latest.json',dict(status=result['status'],receipt=entry(run/'receipt.json')))


if __name__=='__main__':main()
