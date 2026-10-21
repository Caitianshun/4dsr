"""Freeze CTS fixed-Wu inference or the closed28 common-image evaluation job.

CPU build paths never submit, SSH, or import Torch. The environment subcommand
is deliberately executable only inside an explicitly activated protected CTS
GPU job. Existing metric directories are verified and reused, never overwritten.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import shutil
import sys

from build_sr_job_cts_v2 import (ROOT, HERE, BATCH, OLD, PROFILE, CTS_ROOT, CTS_UUID, SCENES, EVALUATOR, LEGACY,
    sha, relative, read, rooted, identity, check_identity, manifest_path, bind, endpoint, asset_gate,
    shared_required, base_job, target, gate_job, evaluation_command, add_profile_args, write_job)


def environment(job_path):
    job = json.loads(job_path.read_text())
    profile, _ = gate_job(job, require_remote=True)
    assert sys.prefix == str(ROOT / '.venv'), 'Use the declared common metric leaf'
    assert Path(sys.executable).resolve() == (ROOT / profile['metric_python']).resolve()
    # Actual imports are deferred until this admitted runtime entry point.
    import cv2
    import lpips
    import numpy as np
    import torch
    torch.set_num_threads(4)
    os.environ.setdefault('TORCH_HOME', str(ROOT / '.cache/torch'))
    torch.hub.set_dir(str(ROOT / '.cache/torch/hub'))
    model = lpips.LPIPS(net='alex', version='0.1', spatial=False, pnet_rand=False, pretrained=True, verbose=False).eval().requires_grad_(False)
    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        digest.update(key.encode()); digest.update(value.detach().cpu().numpy().tobytes())
    context_path = Path(os.environ['CLAUDE_REMOTE_JOB_DIR']) / 'execution_context.json'
    context_copy = ROOT / job['out'] / 'execution_context.json'
    assert not context_copy.exists()
    shutil.copyfile(context_path, context_copy)
    assert sha(context_copy) == sha(context_path)
    value = dict(schema='CEM_area4_CTS_evaluation_environment_v1', status='verified',
        evaluation_profile=PROFILE, amendment=job['amendment'], activation=job['activation'],
        UTC=datetime.now(timezone.utc).isoformat(), hostname=socket.gethostname(), project_root=str(ROOT),
        python_executable=sys.executable, python_realpath=str(Path(sys.executable).resolve()), metric_python=profile['metric_python'],
        CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES'], selected_gpu_uuid=CTS_UUID,
        execution_context=identity(context_copy), original_remote_context_path=str(context_path),
        job=identity(job_path), torch=torch.__version__, numpy=np.__version__, opencv=cv2.__version__,
        gpu=torch.cuda.get_device_name(), evaluator_sha256=sha(ROOT / EVALUATOR), helper_source_sha256=sha(ROOT / LEGACY),
        lpips_state_dict_sha256=digest.hexdigest())
    assert value['hostname'] == 'cts'
    for key, expected in profile['expected_common_environment'].items():
        assert value[key] == expected, 'Current CTS environment differs: ' + key
    destination = ROOT / job['out'] / 'evaluation_environment.json'
    with destination.open('x') as stream:
        stream.write(json.dumps(value, indent=2) + '\n')
    print(json.dumps(dict(status='verified', path=str(destination), sha256=sha(destination))))


def add_asset_gate(args, scene, req):
    path = BATCH / 'assets' / f'cts_{scene}_SR_fallback_verified_v2.json'
    receipt, inventory, verified, table = asset_gate(scene, manifest_path(scene), path)
    req.add(receipt); req.add(inventory)
    return dict(path=str(receipt), inventory_sha256=verified['inventory_sha256']), table


def add_GT(req, scene, assets):
    manifest = manifest_path(scene)
    req.add(manifest)
    tests = [r for r in read(manifest)['observations'] if r['split'] == 'test']
    assert {(r['camera_id'], r['frame_index']) for r in tests} == {('cam00', i) for i in range(300)}
    for row in tests:
        path = manifest.parent / row['hr_path']
        assert assets['files'][str(path)]['sha256'] == row['hr_sha256']
        req.add(path, row['hr_sha256'])
    return manifest


def closed_transfer_gate(receipt_path, inventory_path, amendment):
    receipt, inventory = read(receipt_path), read(inventory_path)
    assert receipt["status"] == "verified" and receipt["all_listed_file_bytes_rehashed"] is True
    assert receipt["root"] == CTS_ROOT and receipt["failure_count"] == 0
    assert relative(receipt["inventory_path"]) == relative(inventory_path)
    assert receipt["inventory_sha256"] == sha(ROOT / inventory_path)
    assert inventory["target"] == "cts"
    assert inventory["execution_amendment"]["path"] == amendment["path"]
    assert inventory["execution_amendment"]["sha256"] == amendment["sha256"]
    assert receipt["file_count"] == receipt["verified_count"] == len(inventory["files"])
    assert set(receipt["files"]) == set(inventory["files"])
    for name, row in inventory["files"].items():
        check = receipt["files"][name]
        assert check["status"] == "verified" and check["stable_during_read"]
        assert (check["sha256"], check["bytes"]) == (row["sha256"], row["bytes"])
    return receipt, inventory


def wu_job(args):
    profile, plan, amendment, activation = bind(args, 'production')
    req = shared_required(profile, amendment, activation)
    receipt, assets = add_asset_gate(args, args.scene, req)
    manifest = add_GT(req, args.scene, assets)
    frozen = plan['frozen_Wu_inventory']
    req.frozen(frozen)
    inventory = read(frozen['path'])
    wu_receipt_path = BATCH / 'assets/cts_wu_reference_assets_verified_v2.json'
    wu_inventory_path = BATCH / 'assets/cts_wu_reference_assets_inventory_v2.json'
    wu_verified, wu_assets = closed_transfer_gate(wu_receipt_path, wu_inventory_path, amendment)
    assert wu_assets['source_inventory'] == dict(path=frozen['path'], sha256=frozen['sha256'])
    req.add(wu_receipt_path); req.add(wu_inventory_path)
    assert inventory['schema'] == 'Wu_area4_fixed_no_posthoc_reference_v1' and len(inventory['tasks']) == 12
    tasks = [r for r in inventory['tasks'] if r['scene'] == args.scene]
    assert [r['arm'] for r in tasks] == ['areaL0', 'naive', 'areaT', 'areaTm', 'w4', 'oracle']
    for item in [*inventory['common_required_files'], inventory['evidence']]:
        assert wu_assets['files'][item['path']]['sha256'] == item['sha256']
        req.frozen(item)
    commands, outputs, targets = [], [], []
    for task in tasks:
        canonical = {'w4':'areaTm_w4', 'oracle':'areaOracle'}.get(task['arm'], task['arm'])
        row = endpoint(plan, 'Wu', args.scene, canonical)
        assert row['frozen_task_id'] == task['id'] and row['checkpoint'] == task['checkpoint']
        assert not task['original_training_cull'] and task['original_training_culls'] == [] and not task['irreversible_posthoc_cull_detected']
        assert task['manifest']['sha256'] == sha(ROOT / manifest)
        for item in task['required_files']:
            assert wu_assets['files'][item['path']]['sha256'] == item['sha256']
            req.frozen(item)
        assert not (ROOT / row['run']).exists() and not (ROOT / row['metric_dir']).exists(), 'Never replay/overwrite a Wu task'
        assert task['render_command'] == ['.venv/bin/python', str(HERE / 'wu_reference.py'), 'render', '--inventory', frozen['path'], '--task', task['id']]
        commands.append(dict(name='Wu_' + task['arm'] + '_fixed_raw_render', argv=[rooted('.venv/bin/python'), rooted(HERE / 'wu_reference.py'), 'render', '--inventory', rooted(frozen['path']), '--task', task['id'], '--out', rooted(row['run'])]))
        commands.append(evaluation_command(row, manifest))
        outputs += [row['run'], row['metric_dir']]
        targets.append(target(row))
    out = BATCH / 'execution' / f'Wu_{args.scene}_CTS_v2'
    job = base_job(profile, amendment, activation, req, out, 'codex-cem-wuref-' + args.scene.split('_')[0] + '-ctsg0-20261020-v2', args.scene, 'wu_reference', 'Wu_fixed_reference_CTS_v2', commands, [receipt, dict(path=str(wu_receipt_path), inventory_sha256=wu_verified['inventory_sha256'])])
    job.update(return_paths=[str(out), *outputs], evaluation_targets=targets, reference_inventory=frozen,
        task_ids=[r['id'] for r in tasks], training_HR=False, parameter_updates=0, evaluation_only=True,
        inference_policy=inventory['inference_policy'], comparison_limit='Frozen selected Wu development endpoints; raw native rendering, no posthoc cull or new training.')
    gate_job(job)
    return job


def source_rendering(row, req):
    manifest = manifest_path(row['scene'])
    source_index = None
    if row['system'] == 'CEM':
        req.add(row['source_index'])
        source_index = read(row['source_index'])
        if row['arm'] in {'native_parent', 'HRref'}:
            assert source_index['status'] == 'native_endpoints_and_300_renders_verified'
            assert source_index['manifest_sha256'] == sha(ROOT / manifest)
            assert source_index['train_level'] == ('hr' if row['arm'] == 'HRref' else 'lr')
        else:
            assert source_index['status'] == 'SR_endpoint_and_303_renders_unified_metrics_verified'
            assert source_index['scene'] == row['scene'] and source_index['arm'] == row['arm']
            assert source_index['actual_parameter_updates'] == 6000 and not source_index['heldout_used_for_training']
            assert source_index['manifest']['sha256'] == sha(ROOT / manifest)
        path = Path(row['render_dir']) / 'render_receipt.json'
        value = read(path)
        assert value['frames'] == value['finite_before_clip_quantization_passed_frames'] == 300
        assert value['camera'] == 'cam00' and value['scope'] == 'heldout_300'
        assert value['manifest']['sha256'] == sha(ROOT / manifest) and not value['heldout_pixels_read']
        if row['arm'] in {'native_parent', 'HRref'}:
            assert source_index['renders_receipt_sha256'] == sha(ROOT / path)
        else:
            assert source_index['heldout_rendering']['receipt']['sha256'] == sha(ROOT / path)
            assert value['checkpoint']['sha256'] == source_index['checkpoint']['sha256']
        rows = value['records']
        assert [r['frame'] for r in rows] == list(range(300))
        for frame, record in enumerate(rows):
            assert record['path'] == f'{frame:04d}.png'
            req.add(Path(row['render_dir']) / record['path'], record['sha256'])
    else:
        path = Path(row['render_receipt'])
        value = read(path)
        assert value['schema'] == 'Wu_area4_fixed_raw_no_posthoc_render_v1' and value['frames'] == 300
        assert value['task']['id'] == row['frozen_task_id'] and value['task']['checkpoint'] == row['checkpoint']
        assert [r['frame'] for r in value['PNGs']] == list(range(300))
        for frame, record in enumerate(value['PNGs']):
            assert record['path'] == f'renders_hr/{frame:04d}.png'
            req.add(Path(row['run']) / record['path'], record['sha256'])
    assert value['status'] == 'completed'
    req.add(path)


def common_job(args):
    from index_sr_cts_v2 import metrics_contract
    profile, plan, amendment, activation = bind(args, 'production')
    req = shared_required(profile, amendment, activation)
    receipts, manifests = [], {}
    for scene in SCENES:
        receipt, assets = add_asset_gate(args, scene, req)
        receipts.append(receipt)
        manifests[scene] = add_GT(req, scene, assets)
    commands, targets, reused = [], [], []
    for row in plan['endpoints']:
        source_rendering(row, req)
        directory = ROOT / row['metric_dir']
        if directory.exists():
            # A partial/invalid directory is an error, never a reason to rerun.
            checked = metrics_contract(row, ROOT / manifests[row['scene']], profile=profile)
            req.frozen(checked['summary']); req.frozen(checked['per_frame'])
            reused.append(dict(**target(row), metrics=checked))
        else:
            commands.append(evaluation_command(row, manifests[row['scene']]))
            targets.append(target(row))
    assert commands, 'All28 already evaluated; no GPU job is necessary'
    out = BATCH / 'execution/common_CTS_28_v2'
    job = base_job(profile, amendment, activation, req, out, 'codex-cem-common28-ctsg0-20261020-v2', 'both', 'wu_reference', 'Common_CTS_existing_28_PNG_v2', commands, receipts)
    job.update(evaluation_targets=targets, reused_evaluation_targets=reused, evaluation_only=True, parameter_updates=0,
        training_HR=False, return_paths=[str(out), profile['common_metrics_root']], no_model_or_training_loaded=True)
    gate_job(job)
    return job


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    env = sub.add_parser('environment'); env.add_argument('--job', type=Path, required=True)
    for kind in ['wu', 'common']:
        p = sub.add_parser(kind); add_profile_args(p)
        p.add_argument('--output', type=Path, required=True)
        if kind == 'wu':
            p.add_argument('--scene', choices=SCENES, required=True)
    args = parser.parse_args()
    if args.command == 'environment':
        environment(args.job.resolve())
    else:
        write_job(args.output, wu_job(args) if args.command == 'wu' else common_job(args))


if __name__ == '__main__':
    main()
