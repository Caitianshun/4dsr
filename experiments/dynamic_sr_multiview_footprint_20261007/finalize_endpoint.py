"""Recover only missing endpoint receipts after an atomic final checkpoint.

CPU-only integrity work; no render, optimization, fresh state or warm start.
The committed training rows must cover every step exactly once and match the
saved protocol/source/calibration identity. Failed/uncommitted work is archived
by the real train helpers, which are AST-executed unchanged without CUDA imports.
"""
import argparse,ast,json,time,shutil,os
from collections import Counter
from fp_common import ROOT,HERE,OUT,read,write,entry,bound,sha,Path,source_identity
from config import ARMS

def main(task):
    import torch
    directory=OUT/'runs'/task
    config=read(directory/'config.json');method=config['method'];definition=ARMS[method]
    assert task==f"r{config['repeat']}_{method}"
    if (directory/'complete.json').exists():
        bound(read(directory/'complete.json')['checkpoint']);return
    sidecar=read(directory/'checkpoint_12000.json');path=bound(sidecar)
    checkpoint=torch.load(path,map_location='cpu',weights_only=False);metadata=checkpoint['metadata']
    assert metadata['suffix_step']==6000 and metadata['intervention_step']==12000 and metadata['points']==132972
    assert metadata['method']==method and metadata['repeat']==config['repeat']
    assert metadata['protocol_sha256']==config['protocol_sha256']==sha(OUT/'protocol.json')
    assert checkpoint['multiview_footprint']['sources']==config['source_identity']==source_identity(method)
    assert metadata['calibration_sha256']==config['calibration_sha256']
    if config['calibration_sha256']:assert sha(OUT/'calibration.json')==config['calibration_sha256']
    assert checkpoint['model'][12]['state'] and checkpoint['motion_refinement']['child_optimizer']['state']
    assert all(k in checkpoint['rng'] for k in ('torch','cuda','numpy','python'))
    schedule=OUT/'schedules'/f"schedule_{config['repeat']}.json"
    assert metadata['schedule_sha256']==sha(schedule)==config['schedule_sha256']
    for identity in config['training_files']:bound(identity)
    tree=ast.parse((HERE/'train.py').read_text())
    nodes=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in ('recover_log_tail','committed_segments')]
    assert len(nodes)==2
    namespace=dict(Path=Path,read=read,write=write,entry=entry,bound=bound,sha=sha,
                   json=json,time=time,shutil=shutil,os=os)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(HERE/'train.py'),'exec'),namespace)
    namespace['recover_log_tail'](directory,6000)
    namespace['committed_segments'](directory,6000,checkpoint)
    rows=[json.loads(line) for line in (directory/'training.jsonl').read_text().splitlines()]
    assert [row['suffix_step'] for row in rows]==list(range(1,6001))
    assert all(row['rgb_forwards']==3 and row['adam_calls']==2 and row['regularizer_calls']==1 for row in rows)
    assert all(row['moment_forwards']==(3 if definition['X'] else 0) for row in rows)
    required={1,100,500,3000,6000}
    assert all(next(row for row in rows if row['suffix_step']==step)['gdiag'] is not None for step in required)
    table=read(schedule);selected=table[definition['schedule']]
    assert [row['observations'] for row in rows]==[[table['record_keys'][i] for i in row] for row in selected]
    segments=[read(path) for path in sorted(directory.glob('segment_*.json'))]
    assert sum(segment['updates'] for segment in segments)==6000
    lr_exposure=Counter();sr_exposure=Counter()
    for row in selected:
        for index,w in zip(row,definition['lr_weights']):lr_exposure[str(table['record_keys'][index])]+=w
        for index,w in zip(row,definition['sr_weights']):sr_exposure[str(table['record_keys'][index])]+=w
    write(directory/'exposure_recovered_0000_6000.json',dict(start=0,stop=6000,schedule_sha256=sha(schedule),
        coefficient_weighted_LR=dict(lr_exposure),coefficient_weighted_SR=dict(sr_exposure),
        registered_balance_block=table['exposure_block_size'],no_100_step_balance_claim=True,definition=definition))
    # The audit hook raises before any illegal image can be returned. This
    # reconstructs the legal-boundary proof, not an unavailable actual read log.
    write(directory/'image_read_boundary_recovered.json',dict(status='verified_registered_audit_hook_and_committed_trajectory',
        all_registered_training_inputs_byte_verified=True,actual_open_event_counts='unknown after interrupted receipt writing',
        source_identity=config['source_identity'],original_HR_training=False))
    write(directory/'complete.json',dict(status='completed_training',method=method,repeat=config['repeat'],
        updates=6000,suffix_endpoint=6000,start=0,training_rgb_forwards=18000,
        moment_forwards=18000 if definition['X'] else 0,adam_calls=12000,
        train_s=sum(s['train_s'] for s in segments),wall_s=sum(s['wall_s'] for s in segments),
        peak_gb=max(s['peak_gb'] for s in segments),checkpoint=entry(path),gpu=config['gpu'],
        physical_gpu=config['physical_gpu'],topology_unchanged=True,source_identity=config['source_identity'],
        calibration_sha256=config['calibration_sha256'],segments=[entry(f) for f in sorted(directory.glob('segment_*.json'))],
        endpoint_receipt_recovered_from_atomic_checkpoint_without_training=True,
        finalizer_source=entry(Path(__file__)),recovery_seconds_no_GPU=True))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--task',required=True);a=p.parse_args();main(a.task)
