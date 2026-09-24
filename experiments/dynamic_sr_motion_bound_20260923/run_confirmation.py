#!/usr/bin/env python3
"""Persistent parent process for one new MeetRoom vrheadset confirmation.

Run under systemd-run or another persistent supervisor. Native-LR warmup and
LR-integrated adaptation run on train GPU while the existing official SwinIR
generator runs on eval GPU. Join both preparation jobs, then select once,
train this sequence's A, and run B/C plus all nine fixed evaluations. Children
are waited on directly; no polling loop discovers normal completion.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = Path(__file__).resolve().parent
OLD = ROOT / 'experiments/dynamic_sr_20260918'
SCENE = ROOT / 'experiments/dynamic_sr_scene_residual_20260923'
CAMERAS = 'cam02,cam04,cam08,cam12'
ALLOWED_DESKTOP = {'/opt/todesk/bin/ToDesk_Session', '/usr/libexec/gnome-remote-desktop-daemon'}


def stamp():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temp.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


def gpu_identity(selector):
    result = subprocess.check_output(['nvidia-smi', '-i', selector,
        '--query-gpu=uuid,name,memory.total', '--format=csv,noheader,nounits'], text=True)
    rows = list(csv.reader(result.strip().splitlines()))
    if len(rows) != 1 or len(rows[0]) != 3:
        raise ValueError(f'GPU selector must identify exactly one physical GPU: {selector}')
    uuid, name, total = [v.strip() for v in rows[0]]
    if '5090' in name:
        raise ValueError('5090 GPU is not authorized for this training workflow')
    return dict(selector=selector, uuid=uuid, name=name, memory_total_mib=total)


def gpu_snapshot(uuid):
    gpu = subprocess.check_output(['nvidia-smi', '-i', uuid,
        '--query-gpu=uuid,name,memory.used,memory.free,utilization.gpu',
        '--format=csv,noheader,nounits'], text=True)
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name,used_memory',
        '--format=csv,noheader'], text=True)
    occupied = []
    for row in csv.reader(apps.splitlines()):
        if len(row) >= 3 and row[0].strip() == uuid and row[2].strip() not in ALLOWED_DESKTOP:
            occupied.append([v.strip() for v in row])
    snapshot = dict(time_utc=stamp(), gpu=gpu, compute_apps=apps, unexpected_processes=occupied)
    if occupied:
        raise RuntimeError(f'Refusing to share occupied GPU {uuid}: {occupied}')
    return snapshot


class Runner:
    def __init__(self, out):
        self.out = out

    def child(self, label, command, gpu=None):
        tick = time.monotonic()
        event = self.out / 'events' / f'{label}.json'
        record = dict(status='checking', command=command, gpu=gpu, started_utc=stamp(),
                      host=socket.gethostname(), parent_pid=os.getpid())
        write(event, record)
        try:
            if gpu is not None:
                snapshots = [gpu_snapshot(gpu)]
                time.sleep(2)
                snapshots.append(gpu_snapshot(gpu))
                record['gpu_before'] = snapshots
            env = os.environ.copy()
            env.update(CUDA_DEVICE_ORDER='PCI_BUS_ID', OMP_NUM_THREADS='4', MPLBACKEND='Agg', PYTHONUNBUFFERED='1')
            if gpu is None:
                # The batch dispatcher itself chooses both UUIDs independently.
                env.pop('CUDA_VISIBLE_DEVICES', None)
            else:
                env['CUDA_VISIBLE_DEVICES'] = gpu
            with (self.out / f'{label}.log').open('w') as log:
                process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                record.update(status='running', pid=process.pid)
                write(event, record)
                returncode = process.wait()
            record.update(status='completed' if returncode == 0 else 'failed', returncode=returncode,
                          elapsed_s=time.monotonic()-tick, finished_utc=stamp())
            write(event, record)
            if returncode:
                raise RuntimeError(f'{label} exited with code {returncode}; see {label}.log')
            return record
        except BaseException:
            record.update(status='failed', elapsed_s=time.monotonic()-tick, finished_utc=stamp(),
                          traceback=traceback.format_exc())
            write(event, record)
            raise


def verify_prior(manifest_path, manifest):
    folder = manifest_path.parent / 'sr_swinir_x4'
    config = read(folder / 'prior_config.json')
    if config['manifest_sha256'] != sha(manifest_path) or config['tile'] != 0 or config['overlap'] != 32:
        raise ValueError('Unexpected teacher manifest/tile protocol')
    selected = [o for o in manifest['observations'] if o['split']=='train' and o['camera_id'] in CAMERAS.split(',')]
    if len(selected) != 240:
        raise ValueError('Expected exactly four teacher cameras by 60 training frames')
    for observation in selected:
        source = manifest_path.parent / observation['lr_path']
        target = folder / observation['camera_id'] / Path(observation['lr_path']).name
        receipt = read(target.with_suffix('.json'))
        if receipt['input_sha256'] != sha(source) or receipt['output_sha256'] != sha(target):
            raise ValueError(f'Teacher receipt mismatch: {target}')
    return dict(status='completed_and_verified', images=len(selected), directory=str(folder),
                prior_config_sha256=sha(folder / 'prior_config.json'))


def run(args, state):
    started = time.monotonic()
    manifest = read(args.manifest)
    if manifest['scene'] != 'meetroom_vrheadset':
        raise ValueError('This bounded confirmation is registered for meetroom_vrheadset only')
    if manifest['frame_indices'] != list(range(0,120,2)):
        raise ValueError('Confirmation requires the fixed 60-frame short window')
    train_gpu, eval_gpu = gpu_identity(args.train_gpu), gpu_identity(args.eval_gpu)
    if train_gpu['uuid'] == eval_gpu['uuid']:
        raise ValueError('Two different GPUs are required for independent preparation and evaluation')
    state.update(gpus={'train':train_gpu, 'eval':eval_gpu}, manifest_sha256=sha(args.manifest),
                 host=socket.gethostname(), python=sys.executable,
                 sources={str(p):sha(p) for p in [Path(__file__), SCRIPTS/'run_confirmation_batch.py',
                    SCRIPTS/'summarize_confirmation.py', SCRIPTS/'train.py', SCRIPTS/'motion_model.py',
                    SCRIPTS/'prepare_selection.py', SCRIPTS/'evaluate.py', OLD/'run_experiment.py',
                    OLD/'generate_prior.py', SCENE/'train.py', SCENE/'residual_model.py']})
    write(args.out/'status.json', state)
    runner = Runner(args.out)
    warm = args.out/'native_warmup'
    parent = args.out/'integrated_parent'
    def prepare_parent():
        runner.child('native_warmup', [sys.executable,str(OLD/'run_experiment.py'),
            '--manifest',str(args.manifest),'--out',str(warm),'--task','warmup',
            '--observation','native_lr','--coarse-steps','1000','--fine-steps','6000',
            '--max-points','120000','--seed','20260918'], train_gpu['uuid'])
        done = read(warm/'complete.json')
        if done['stage']!='fine' or done['step']!=6000:
            raise ValueError('Warmup completion does not match registered endpoint')
        runner.child('lr_integrated', [sys.executable,str(OLD/'run_experiment.py'),
            '--manifest',str(args.manifest),'--out',str(parent),'--task','branch',
            '--checkpoint',str(warm/'checkpoint_final.pt'),'--mode','lr_integrated',
            '--steps','1200','--seed','20260918','--prior-cameras',CAMERAS], train_gpu['uuid'])
        done = read(parent/'complete.json')
        if done['mode']!='lr_integrated' or done['step']!=7200:
            raise ValueError('Integrated parent completion does not match registered endpoint')
        return dict(status='completed', warmup_updates=7000, integrated_updates=1200,
                    checkpoint=str(parent/'checkpoint_final.pt'), checkpoint_sha256=sha(parent/'checkpoint_final.pt'),
                    note='Original entry retains built-in quick_development LR diagnostics; no extra parent evaluation job.')
    def prepare_teacher():
        runner.child('frozen_swinir', [sys.executable,str(OLD/'generate_prior.py'),
            '--manifest',str(args.manifest),'--cameras',CAMERAS,'--tile','0','--overlap','32'], eval_gpu['uuid'])
        return verify_prior(args.manifest,manifest)
    prep, errors = {}, []
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(prepare_parent):'parent',pool.submit(prepare_teacher):'teacher'}
        for future in as_completed(futures):
            name = futures[future]
            try:
                prep[name] = future.result()
            except BaseException:
                failure = dict(stage=f'prepare_{name}',traceback=traceback.format_exc(),time_utc=stamp())
                errors.append(failure)
                write(args.out/f'failed_prepare_{name}.json',failure)
            state.update(status='preparation_failed_waiting_other_owned_job' if errors else 'preparing',
                         preparation=prep, errors=errors)
            write(args.out/'status.json',state)
    if errors:
        raise RuntimeError(f'Preparation failed; both owned jobs joined: {errors}')
    state.update(status='selecting',preparation=prep)
    write(args.out/'status.json',state)
    selection = args.out/'selection_v1'
    runner.child('selection', [sys.executable,str(SCRIPTS/'prepare_selection.py'),
        '--manifest',str(args.manifest),'--checkpoint',str(parent/'checkpoint_final.pt'),
        '--out',str(selection)],train_gpu['uuid'])
    selection_receipt=read(selection/'complete.json')
    if selection_receipt['status']!='completed_selection' or selection_receipt['parameter_updates']!=0:
        raise ValueError('Selection did not complete without parameter updates')
    baseline=args.out/'joint_source'
    state.update(status='training_new_A')
    write(args.out/'status.json',state)
    runner.child('joint_A', [sys.executable,str(SCENE/'train.py'), '--manifest',str(args.manifest),
        '--checkpoint',str(parent/'checkpoint_final.pt'),'--out',str(baseline),'--branch','joint',
        '--steps','6000','--milestones','1200,6000','--seed','20260923','--prior-cameras',CAMERAS],train_gpu['uuid'])
    baseline_receipt=read(baseline/'complete.json')
    if baseline_receipt['status']!='completed' or baseline_receipt['parameter_updates']!=6000 or baseline_receipt['smoke']:
        raise ValueError('New sequence A training incomplete')
    state.update(status='training_B_C_and_evaluating_A_B_C',training_records={'joint':baseline_receipt})
    write(args.out/'status.json',state)
    batch=args.out/'batch_v1'
    runner.child('confirmation_batch', [sys.executable,str(SCRIPTS/'run_confirmation_batch.py'),
        '--manifest',str(args.manifest),'--checkpoint',str(parent/'checkpoint_final.pt'),
        '--selection',str(selection/'selection.pt'),'--baseline',str(baseline),
        '--out',str(batch),'--train-gpu',train_gpu['uuid'],'--eval-gpu',eval_gpu['uuid']],gpu=None)
    batch_receipt=read(batch/'complete.json')
    summary_receipt=read(batch/'summary/complete.json')
    if batch_receipt['status']!='completed_and_evaluated' or summary_receipt['status']!='completed_summary':
        raise ValueError('Batch/summary completion is missing')
    branches={b:read(batch/b/'complete.json') for b in ['joint','ordinary_split','bound_split']}
    eval_records=[dict(branch=b,split=split,step=step,path=str(batch/b/f'eval_{split}_{step}'/'complete.json'),
                      receipt=read(batch/b/f'eval_{split}_{step}'/'complete.json'))
                  for b in branches for split,step in [('dev',1200),('dev',6000),('test',6000)]]
    if any(r['receipt']['status']!='completed_evaluation' for r in eval_records):
        raise ValueError('Not all nine registered evaluations completed')
    state.update(status='completed_and_evaluated',training_completed=True,evaluation_completed=True,
                 summary_completed=True,preparation=prep,selection=selection_receipt,
                 training_records=branches,evaluation_records=eval_records,
                 batch=str(batch),summary=str(batch/'summary/summary.md'),
                 elapsed_s=time.monotonic()-started,finished_utc=stamp(),
                 note='One new sequence, A/B/C only, no D or candidate grid. New A trained once before batch reuse. No chat notification is implied.')
    write(args.out/'complete.json',state)
    write(args.out/'status.json',state)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--train-gpu',required=True,help='Physical GPU index or UUID')
    parser.add_argument('--eval-gpu',required=True,help='Different physical GPU index or UUID')
    args=parser.parse_args()
    args.manifest=args.manifest.resolve(); args.out=args.out.resolve()
    args.out.mkdir(parents=True,exist_ok=False)
    state=dict(status='starting',pid=os.getpid(),started_utc=stamp(),manifest=str(args.manifest),
               training_completed=False,evaluation_completed=False,summary_completed=False)
    write(args.out/'status.json',state)
    try:
        run(args,state)
    except BaseException:
        failure_traceback=traceback.format_exc()
        # Preserve completed scientific stages if only a later stage failed.
        # The batch writes its completion receipt before running CPU summary.
        batch=args.out/'batch_v1'
        completed_training, completed_evaluation = {}, []
        for branch in ['joint','ordinary_split','bound_split']:
            receipt_path=batch/branch/'complete.json'
            if receipt_path.is_file():
                receipt=read(receipt_path)
                if receipt.get('status')=='completed' and receipt.get('parameter_updates')==6000:
                    completed_training[branch]=receipt
            for split,step in [('dev',1200),('dev',6000),('test',6000)]:
                receipt_path=batch/branch/f'eval_{split}_{step}'/'complete.json'
                if receipt_path.is_file():
                    receipt=read(receipt_path)
                    if receipt.get('status')=='completed_evaluation':
                        completed_evaluation.append(dict(branch=branch,split=split,step=step,
                                                         path=str(receipt_path),receipt=receipt))
        if completed_training:
            state['training_records']=completed_training
        state.update(status='failed',finished_utc=stamp(),traceback=failure_traceback,
                     training_completed=len(completed_training)==3,
                     evaluation_completed=len(completed_evaluation)==9,
                     evaluation_records=completed_evaluation)
        write(args.out/'failed.json',state)
        write(args.out/'status.json',state)
        raise


if __name__=='__main__':
    main()
