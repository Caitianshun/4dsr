"""A retained full-time teacher batch on a root-assigned free GPU.

The exact frozen historical generator supports a resumable inventory limit.
Outputs count toward the full cache, never extra model-training updates. A
partial batch is not accepted as a complete teacher inventory.
"""
import argparse
import fcntl
import os
import time
import traceback
from fp_common import ROOT, OUT, HERE, read, write, bound, entry
import run_suite as suite
import full_teacher_prepare as teacher


def main(a):
    assert a.operator_resource_resolved and a.gpu.startswith('GPU-')
    plan = read(a.plan)
    bound(plan['source']); bound(plan['producer_manifest'])
    generator = bound(plan['teacher_generator'])
    teacher.check_sources()
    assert plan['frozen_recipe'] == teacher.FROZEN
    assert 0 < a.limit <= plan['train_observation_count']
    status = OUT / 'workers' / f'teacher_{plan["scene"]}.json'
    (OUT / 'locks').mkdir(parents=True, exist_ok=True)
    with (OUT / 'locks' / f'{a.gpu}.controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        resource = suite.check_resource(a.gpu, status)
        command = [a.python, '-u', str(generator), '--manifest',
                   str(bound(plan['producer_manifest'])), '--checkpoint',
                   str(teacher.WEIGHT), '--network', str(teacher.NETWORK),
                   '--dependency-path', str(ROOT / 'experiments/dynamic_sr_20260918/vendor'),
                   '--device', 'cuda:0', '--tile', '0', '--overlap', '32',
                   '--limit', str(a.limit), '--log-every', '20']
        started = time.monotonic()
        log = OUT / 'full_teacher_prepare' / plan['scene'] / f'batch_{a.limit}_{time.time_ns()}.log'
        write(status, dict(status='running_retained_full_teacher_batch',
                           plan=entry(a.plan), gpu=a.gpu, resource=resource,
                           limit=a.limit, command=command, log=str(log.relative_to(ROOT)),
                           formal_updates=0, Adam_calls=0))
        suite.run(command, log, suite.environment(a.gpu), a.gpu)
        cache = bound(plan['producer_manifest']).parent / 'sr_swinir_x4'
        summaries = [(p, read(p)) for p in cache.glob('summary_*.json')]
        summaries = [(p, v) for p, v in summaries
                     if v['selected'] == a.limit and v['generated'] + v['skipped'] == a.limit]
        assert summaries, 'No successful original generator batch receipt'
        p, result = max(summaries, key=lambda item: item[0].stat().st_mtime_ns)
        cfg = read(cache / 'prior_config.json')
        assert cfg['manifest_sha256'] == plan['producer_manifest']['sha256']
        assert all(cfg[k] == value for k, value in teacher.FROZEN.items())
        value = dict(status='completed_retained_full_teacher_batch', plan=entry(a.plan),
                     original_summary=entry(p), original_config=entry(cache / 'prior_config.json'),
                     generated=result['generated'], reused=result['skipped'], selected=a.limit,
                     full_inventory_completed=result['selection_complete'],
                     resource=resource, wall_seconds=time.monotonic() - started,
                     original_generation_seconds=result['seconds'],
                     GPU=result['gpu_name'], formal_updates=0, Adam_calls=0,
                     outputs_retained_for_full_inventory=True)
        target = OUT / 'full_teacher_prepare' / plan['scene'] / f'batch_{a.limit}_{time.time_ns()}.json'
        write(target, value)
        write(status, dict(status=value['status'], receipt=entry(target),
                           full_inventory_completed=value['full_inventory_completed'],
                           formal_updates=0, Adam_calls=0))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=teacher.Path, required=True)
    p.add_argument('--limit', type=int, required=True)
    p.add_argument('--gpu', required=True)
    p.add_argument('--python', default=suite.PYTHON)
    p.add_argument('--operator-resource-resolved', action='store_true')
    a = p.parse_args()
    try:
        main(a)
    except suite.ResourceBusy:
        write(OUT / 'workers' / f'teacher_busy_{time.time_ns()}.json',
              dict(status='resource_busy_no_work_launched', traceback=traceback.format_exc()))
        raise SystemExit(75)
    except BaseException:
        write(OUT / 'workers' / f'teacher_failure_{time.time_ns()}.json',
              dict(status='failed_preserved', traceback=traceback.format_exc(),
                   partial_outputs_preserved=True, unreturned_GPU_work_cost='unknown'))
        raise
