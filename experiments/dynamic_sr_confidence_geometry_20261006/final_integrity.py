"""Require complete fixed endpoints and byte-bound evidence; no quality threshold."""
import argparse, math, time
from cg_common import *

def verify_core():
    protocol = read(OUT/'protocol.json'); ph = sha(OUT/'protocol.json')
    cal = read(OUT/'calibration.json'); assert cal['status'] == 'passed'
    records, common_identity, exposure_by_repeat = [], None, {}
    for task in protocol['core_task_plan']:
        method, repeat = task['method'], str(task['repeat'])
        label = f'r{repeat}_{method}'; directory = OUT/'runs'/label
        complete = read(directory/'complete.json'); config = read(directory/'config.json')
        assert complete['status'] == 'completed_training' and complete['updates'] == 6000
        assert complete['topology_unchanged'] and complete['adam_calls'] == 12000
        rgb_per_round = 3 if method == 'B2perm' else 2 if method == 'Jperm' else 1
        assert complete['training_rgb_forwards'] == rgb_per_round * 6000
        assert complete['moment_forwards'] == (6000 if method in ['G','RG'] else 0)
        assert config['protocol_sha256'] == ph and config['parent'] == protocol['parent']
        assert config['original_HR_training'] is False
        assert config['schedule_sha256'] == protocol['schedules'][repeat]['sha256']
        if common_identity is None: common_identity = config['initial_identity']
        assert config['initial_identity'] == common_identity
        reload = read(directory/'reload_audit.json')
        assert reload['status'] == 'passed' and reload['global_rng_exact'] and reload['two_adam_exact']
        assert reload['fork_cursor'] == 0 and reload['identity'] == common_identity
        reads = read(directory/'image_reads.json'); assert reads['all_legal'] and not reads['HR_derived_training']
        for rel, digest in config['source_identity'].items():
            assert sha(directory/'sources'/rel) == digest, (label, rel)
        if method in ['R','G','RG']: assert config['calibration'] == cal
        checkpoint = bound(complete['checkpoint'])
        cp = read(checkpoint.with_suffix('.json'))
        assert cp['sha256'] == complete['checkpoint']['sha256']
        assert cp['metadata']['suffix_step'] == 6000 and cp['metadata']['intervention_step'] == 12000
        # Historical `step` counts fine iterations. The original 1000 coarse
        # iterations are separate, making total physical optimizer rounds 20200.
        assert cp['metadata']['step'] == 19200
        exposure = read(directory/'exposure.json')
        assert exposure['start'] == 0 and exposure['stop'] == 6000 and exposure['audit']['passed']
        assert sum(exposure['actual_LR'].values()) == 6000
        old = exposure_by_repeat.setdefault(repeat, exposure)
        assert exposure['actual_LR'] == old['actual_LR']
        assert set(exposure['coefficient_weighted_SR']) == set(old['coefficient_weighted_SR'])
        for key, value in exposure['coefficient_weighted_SR'].items():
            assert math.isclose(value, old['coefficient_weighted_SR'][key], abs_tol=1e-10)
        evaluation = OUT/'evaluation'/label
        receipt = read(evaluation/'complete.json')
        assert receipt['checkpoint']['sha256'] == complete['checkpoint']['sha256']
        splits = []
        for split in ['test','dev','train_fixed']:
            path = bound(receipt['splits'][split]); metric = read(path)
            assert metric['parameter_updates'] == 0 and metric['checkpoint_sha256'] == complete['checkpoint']['sha256']
            cameras = ['cam00'] if split == 'test' else ['cam01'] if split == 'dev' else protocol['evaluation']['train16_cameras']
            frames = protocol['evaluation']['train_frames'] if split == 'train_fixed' else protocol['evaluation']['dev_frames']
            expected = {(c,f) for c in cameras for f in frames}
            actual = {(r['camera_id'],r['frame_index']) for r in metric['rows']}
            assert actual == expected and len(metric['rows']) == len(expected)
            for row in metric['rows']:
                full = row['spatial']['full']
                assert all(math.isfinite(full[k]) for k in ['psnr','ssim','lpips_alex'])
            splits.append(entry(path))
        extra = read(evaluation/'extra/complete.json')
        assert extra['parameter_updates'] == 0
        index = read(evaluation/'extra/float_index.json')
        assert len(index['entries']) == 196 and index['parameter_updates'] == 0
        for item in index['entries']:
            assert item['checkpoint_sha256'] == complete['checkpoint']['sha256']
            assert sha(evaluation/'extra'/item['path']) == item['sha256']
        records.append(dict(task=label, checkpoint=entry(checkpoint), complete=entry(directory/'complete.json'),
                            splits=splits, extra=entry(evaluation/'extra/complete.json'), gpu=config['gpu']))
    assert len(records) == 12
    return dict(status='passed_core_endpoint_evidence', endpoints=records, protocol=entry(OUT/'protocol.json'),
                calibration=entry(OUT/'calibration.json'), formal_updates=72000, RGB=108000, moments=24000,
                same_parent_identity=common_identity, checkpoint_and_float_bytes_verified=True,
                source_snapshots_verified=True, legal_read_receipts_verified=True,
                quality_threshold_used=False, total_physical_rounds_per_endpoint=20200,
                historical_fine_step_per_endpoint=19200, short_window_not_standard_benchmark=True)

def main(a):
    started = time.monotonic(); result = verify_core()
    result.update(seconds=time.monotonic()-started, source=entry(HERE/'final_integrity.py'))
    write(a.out, result); print(json.dumps(dict(status=result['status'], endpoints=len(result['endpoints']))))

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--out', type=Path, default=OUT/'final_integrity.json')
    main(p.parse_args())
