"""Frozen camera cycle and exact within-window second-slot exposure control."""
import csv
import random
from collections import Counter
from copy import deepcopy


def generate(schedule, repeat):
    keys = schedule['record_keys']
    cameras = sorted({c for c, f in keys})
    assert cameras == [f'cam{i:02d}' for i in range(2, 21)]
    cycle = cameras[:]
    random.Random(2026092802).shuffle(cycle)
    mapping = dict(zip(cycle, cycle[1:] + cycle[:1]))
    lookup = {tuple(key): i for i, key in enumerate(keys)}
    sync = []
    for step, (li, si, _) in enumerate(schedule['rows'][6000:12000], 6001):
        ca, fa = keys[si]; cb = mapping[ca]
        sync.append(dict(step=step, lr_index=li, sr_a_index=si,
                         sr_b_index=lookup[cb, fa], camera_a=ca, camera_b=cb,
                         frame_a=fa, frame_b=fa))
    asynchronous = deepcopy(sync)
    seed = {'1': 2026092803, '2': 2026092804}[str(repeat)]
    rng = random.Random(seed)
    singletons = 0
    for begin in range(0, 6000, 100):
        for camera in cameras:
            indices = [i for i in range(begin, begin + 100) if sync[i]['camera_b'] == camera]
            rng.shuffle(indices)
            singletons += len(indices) == 1
            if not indices:
                continue
            for target, source in zip(indices, indices[1:] + indices[:1]):
                frame = sync[source]['frame_b']
                asynchronous[target].update(frame_b=frame, sr_b_index=lookup[camera, frame])
    result = dict(schema=1, repeat=str(repeat), camera_seed=2026092802, async_seed=seed,
                  camera_cycle=cycle, camera_mapping=mapping, window=100,
                  rows=dict(sync2=sync, async2=asynchronous), singleton_groups=singletons,
                  time_definition='frame_index / 300', coefficients=[0.05, 0.05])
    result['audit'] = audit(result, keys)
    return result


def counts(rows, slot):
    if slot == 'combined':
        return counts(rows, 'a') + counts(rows, 'b')
    return Counter((r['camera_' + slot], r['frame_' + slot]) for r in rows)


def audit(table, keys):
    sync, asynchronous = table['rows']['sync2'], table['rows']['async2']
    assert len(sync) == len(asynchronous) == 6000 and sync != asynchronous
    for i, (a, b) in enumerate(zip(sync, asynchronous), 6001):
        for k in ['step', 'lr_index', 'sr_a_index', 'camera_a', 'camera_b', 'frame_a']:
            assert a[k] == b[k]
        assert a['step'] == i and a['frame_a'] == a['frame_b']
        for r in [a, b]:
            assert r['camera_a'] != r['camera_b']
            assert list(keys[r['sr_a_index']]) == [r['camera_a'], r['frame_a']]
            assert list(keys[r['sr_b_index']]) == [r['camera_b'], r['frame_b']]
            assert 0 <= r['lr_index'] < len(keys)
    for start, stop in [(0, 6000), (0, 3000)] + [(n, n + 100) for n in range(0, 6000, 100)]:
        for slot in ['a', 'b', 'combined']:
            assert counts(sync[start:stop], slot) == counts(asynchronous[start:stop], slot)
    statistics = {}
    for mode, rows in table['rows'].items():
        gaps = Counter(abs(r['frame_b'] - r['frame_a']) for r in rows)
        statistics[mode] = dict(same_time_fraction=gaps[0] / 6000,
            absolute_frame_gap_counts=dict(sorted(gaps.items())),
            mean_absolute_time_gap=sum(k * v for k, v in gaps.items()) / (6000 * 300),
            time_gap_definition='abs(frame_b - frame_a) / 300')
    return dict(passed=True, windows=60, full_and_9000_exposure_exact=True,
                no_global_rng_draws=True, statistics=statistics)


def exposure_rows(table):
    for mode, rows in table['rows'].items():
        windows = [('full', 0, 6000), ('through_9000', 0, 3000)]
        windows += [(str(n // 100), n, n + 100) for n in range(0, 6000, 100)]
        for window, start, stop in windows:
            for slot in ['a', 'b', 'combined']:
                for (camera, frame), count in sorted(counts(rows[start:stop], slot).items()):
                    yield dict(repeat=table['repeat'], mode=mode, window=window,
                               first_step=6001 + start, last_step=6000 + stop,
                               slot=slot, camera=camera, frame=frame, count=count)
