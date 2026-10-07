"""Frame -> equal camera -> equal suffix aggregation with explicit pending state.

Historical endpoints are reused at their registered hashes, including the
completed Repeat7/Video7 branches. No absent result or partial suffix is called
a complete method mean. This CPU script does not select checkpoints or tune.
"""
from __future__ import annotations
import csv
from pathlib import Path
import statistics
from cg_common import ROOT, HERE, OUT, read, write, bound, sha, entry

METRICS = ('psnr', 'ssim', 'lpips')
CORE = ('C1', 'Jperm', 'B2perm', 'R', 'G', 'RG')
REFERENCES = ('U6000', 'J1', 'Async2', 'Sync2', 'Repeat7', 'Video7')


def csvread(path):
    with Path(path).open() as f:
        return list(csv.DictReader(f))


def csvwrite(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row)) or ['status']
    temporary = Path(path).with_suffix('.csv.tmp')
    with temporary.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    temporary.replace(path)


def mean_vector(rows):
    return {k: statistics.mean(float(row[k]) for row in rows) for k in METRICS}


def frame_vector(row):
    full = row['spatial']['full']
    result = dict(camera=row['camera_id'], frame=int(row['frame_index']), split='development',
                  psnr=full['psnr'], ssim=full['ssim'], lpips=full['lpips_alex'])
    for region in ('full', 'dynamic', 'static'):
        spatial = row.get('spatial', {}).get(region, {})
        temporal = row.get('temporal', {}).get(region, {})
        result[region+'_lpips_spatial'] = spatial.get('lpips_alex_spatial_mask')
        result[region+'_temporal_l1'] = temporal.get('gt_relative_warp_l1')
    closure = row.get('lr_reprojection', {}).get('full', {})
    result.update(lr_l1=closure.get('l1'), lr_mse=closure.get('mse'), lr_psnr=closure.get('psnr'))
    return result


def validate_development(rows):
    for camera in ('cam00', 'cam01'):
        frames = [int(row['frame']) for row in rows if row['camera'] == camera]
        assert sorted(frames) == list(range(0, 120, 2)), (camera, len(frames))
    assert len(rows) == 120


def historical():
    protocol_path = ROOT/'output/dynamic_sr_temporal_prior_20260928/protocol.json'
    protocol = read(protocol_path)
    registrations = {'U6000': protocol['baseline_reuse']}
    registrations.update({f'r{r}_J1': spec for r, spec in protocol['historical_J1'].items()})
    registrations.update(protocol['historical_multiview'])
    for repeat in ('1', '2'):
        for arm in ('Repeat7', 'Video7'):
            directory = ROOT/f'output/dynamic_sr_temporal_prior_20260928/evaluation/r{repeat}_{arm}'
            receipt = read(directory/'complete.json')
            assert receipt['status'] == 'completed_evaluation' and receipt['endpoint_sha256'] == sha(directory/'endpoint.json')
            registrations[f'r{repeat}_{arm}'] = dict(directory=str(directory.relative_to(ROOT)),
                endpoint=entry(directory/'endpoint.json'), receipt=entry(directory/'complete.json'))
    data, frames, provenance = {}, [], {}
    for label, registration in registrations.items():
        endpoint = read(bound(registration['endpoint']))
        directory = ROOT/registration['directory']
        raw = csvread(directory/'metrics_per_frame.csv')
        development = [dict(row, endpoint=label, split='development') for row in raw if row['camera'] in ('cam00', 'cam01')]
        validate_development(development)
        cameras = {c: mean_vector([r for r in development if r['camera'] == c]) for c in ('cam00', 'cam01')}
        # Reconstruct the mean from the same per-frame values, then check that
        # reused reports have not drifted to pooled-MSE PSNR or unequal views.
        for camera in cameras:
            for metric in METRICS:
                assert abs(cameras[camera][metric]-endpoint['cameras'][camera][metric]) < 1e-10
        data[label] = dict(cameras=cameras, equal_camera_mean=mean_vector(list(cameras.values())), historical=True)
        frames.extend(development)
        train = [dict(row, endpoint=label, split='train76') for row in raw if row['camera'] not in ('cam00', 'cam01')]
        frames.extend(train)
        provenance[label] = dict(endpoint=registration['endpoint'], per_frame=entry(directory/'metrics_per_frame.csv'),
                                 receipt=registration.get('receipt'), training_hardware='see historical complete/config records',
                                 inference='camera/time only', suffixes_share_parent=True)
    return data, frames, provenance


def summarize():
    protocol = read(OUT/'protocol.json'); data, frames, provenance = historical()
    primary_rows, all_frequency, all_regions, costs = [], [], [], []
    history_fft = ROOT/'output/dynamic_sr_same_observation_20260930/frequency/endpoints'
    for label in data:
        file = history_fft/('r1_U6000.json' if label == 'U6000' else label+'.json')
        if file.exists():
            value = read(file)
            registered = value['entry']['checkpoint']['sha256']
            expected = protocol['parent']['sha256'] if label == 'U6000' else read(bound(provenance[label]['receipt']))['checkpoint_sha256']
            assert registered == expected
            # Endpoint selection and renderer identities stay in the original
            # FFT receipt; retain that source instead of recomputing old models.
            for row in value['rows']:
                all_frequency.append(dict(row, endpoint=label, split='development', source='historical_registered_float_fft'))
            provenance[label]['frequency_source'] = entry(file)
    missing_primary, missing_extra = [], []
    available_labels = []
    plans = list(protocol['core_task_plan'])
    core_labels = {f"r{p['repeat']}_{p['method']}" for p in plans}
    for directory in sorted((OUT/'runs').glob('r*')):
        if directory.name not in core_labels and (directory/'complete.json').exists():
            receipt = read(directory/'complete.json')
            if receipt.get('updates') == 6000:
                plans.append(dict(method=receipt['method'], repeat=str(receipt['repeat']), updates=6000))
    for plan in plans:
        label = f"r{plan['repeat']}_{plan['method']}"; directory = OUT/'evaluation'/label
        receipt_path = directory/'complete.json'
        if not receipt_path.exists():
            if label in core_labels:
                missing_primary.append(label)
            continue
        receipt = read(receipt_path)
        assert receipt['status'] == 'completed_evaluation'
        checkpoint_path = bound(receipt['checkpoint'])
        dev_rows, temporal = [], {}
        for split, camera in [('test', 'cam00'), ('dev', 'cam01')]:
            metrics_path = directory/split/'metrics.json'
            assert receipt['splits'][split]['sha256'] == sha(metrics_path)
            metric = read(metrics_path)
            assert metric['lpips_status'] == 'alex_v0.1_standard_full_and_fixed_mask_spatial_map'
            assert metric['parameter_updates'] == 0
            converted = [dict(frame_vector(row), endpoint=label) for row in metric['rows']]
            assert {row['camera'] for row in converted} == {camera}
            dev_rows.extend(converted); temporal[camera] = metric['temporal_aggregate']
        validate_development(dev_rows); frames.extend(dev_rows)
        cameras = {c: mean_vector([r for r in dev_rows if r['camera'] == c]) for c in ('cam00', 'cam01')}
        data[label] = dict(cameras=cameras, equal_camera_mean=mean_vector(list(cameras.values())),
                           historical=False, temporal=temporal)
        available_labels.append(label)
        train_receipt_path = OUT/'runs'/label/'complete.json'
        train = read(train_receipt_path); assert train['checkpoint']['sha256'] == sha(checkpoint_path)
        hardware = train.get('gpu', 'unrecorded'); dispatch_path = OUT/f'dispatch_{label}.json'
        dispatch = read(dispatch_path) if dispatch_path.exists() else {}
        costs.append(dict(endpoint=label, method=plan['method'], repeat=str(plan['repeat']),
                          training_gpu=hardware, training_host=dispatch.get('host', 'record_unavailable'),
                          physical_gpu=train.get('physical_gpu'), updates=train['updates'],
                          training_rgb_forwards=train['training_rgb_forwards'], training_moment_forwards=train['moment_forwards'],
                          adam_calls=train['adam_calls'], training_seconds=train['train_s'], wall_seconds=train['wall_s'],
                          peak_gb=train['peak_gb'], checkpoint_sha256=train['checkpoint']['sha256']))
        provenance[label] = dict(checkpoint=train['checkpoint'], primary_receipt=entry(receipt_path),
                                 training_receipt=entry(train_receipt_path), training_gpu=hardware,
                                 training_host=dispatch.get('host'), common_parent=protocol['parent'],
                                 continuation_repeats=True, independent_from_scratch=False)
    for label in ['U6000', *available_labels]:
        extra = OUT/'evaluation'/label/'extra'
        if not (extra/'complete.json').exists():
            if label == 'U6000' or label in core_labels:
                missing_extra.append(label)
            continue
        complete = read(extra/'complete.json')
        assert complete['status'] == 'completed_extra_evaluation' and complete['observations'] == 196
        for value in complete['results'].values():
            bound(value)
        new_rows = csvread(extra/'metrics_per_frame.csv')
        train_rows = [dict(row, endpoint=label) for row in new_rows if row['split'] == 'train76']
        assert len(train_rows) == 76 and len({(r['camera'], int(r['frame'])) for r in train_rows}) == 76
        # Retain one train76 stream per endpoint; historical U6000 values are
        # replaced here by the same-float audit when available, with provenance.
        frames = [r for r in frames if not (r['endpoint'] == label and r['split'] == 'train76')]
        frames.extend(train_rows)
        all_frequency.extend(csvread(extra/'frequency_budget.csv'))
        all_regions.extend(csvread(extra/'regional_quality.csv'))
        data[label]['train76'] = mean_vector([mean_vector([r for r in train_rows if r['camera'] == c])
                                               for c in sorted({r['camera'] for r in train_rows})])
        data[label]['extra_receipt'] = entry(extra/'complete.json')
        cost = next((r for r in costs if r['endpoint'] == label), None)
        if cost is not None:
            cost.update(extra_rgb_forwards_new=complete['rgb_forwards_new'], extra_moment_forwards_new=complete['moment_forwards_new'],
                        extra_seconds=complete['seconds'], evaluation_gpu=complete['gpu'])
    for label, value in data.items():
        repeat = label[1] if label.startswith('r') else 'reference'
        arm = label.split('_', 1)[1] if label.startswith('r') else label
        for scope, vector in list(value['cameras'].items()) + [('equal_camera_mean', value['equal_camera_mean'])]:
            primary_rows.append(dict(endpoint=label, arm=arm, repeat=repeat, scope=scope, historical=value['historical'], **vector))
    means, mean_rows = {}, []
    arms = list(dict.fromkeys([*REFERENCES, *CORE, *[p['method'] for p in plans]]))
    for arm in arms:
        labels = ['U6000'] if arm == 'U6000' else [f'r{r}_{arm}' for r in ('1', '2') if f'r{r}_{arm}' in data]
        if not labels:
            continue
        expected = 1 if arm == 'U6000' else 2
        for scope in ('cam00', 'cam01', 'equal_camera_mean'):
            vectors = [data[label]['equal_camera_mean'] if scope == 'equal_camera_mean' else data[label]['cameras'][scope] for label in labels]
            vector = mean_vector(vectors)
            mean_rows.append(dict(arm=arm, scope=scope, repeat_count=len(labels), expected_repeats=expected,
                                  status='complete' if len(labels) == expected else 'partial_suffix_only', **vector))
            if scope == 'equal_camera_mean' and len(labels) == expected:
                means[arm] = vector
    gaps = []
    for label in available_labels:
        repeat = label[1]
        refs = ['U6000', f'r{repeat}_C1', *[f'r{repeat}_{arm}' for arm in REFERENCES if arm != 'U6000']]
        for reference in refs:
            if reference == label or reference not in data:
                continue
            for scope in ('cam00', 'cam01', 'equal_camera_mean'):
                source = data[label]['equal_camera_mean'] if scope == 'equal_camera_mean' else data[label]['cameras'][scope]
                target = data[reference]['equal_camera_mean'] if scope == 'equal_camera_mean' else data[reference]['cameras'][scope]
                gaps.append(dict(endpoint=label, reference=reference, scope=scope, repeat=repeat,
                                 definition='candidate minus reference; higher PSNR/SSIM and lower LPIPS are favourable',
                                 **{k+'_delta': source[k]-target[k] for k in METRICS}))
    paired = {}
    for candidate, reference in [('Jperm', 'C1'), ('B2perm', 'Jperm'), ('R', 'C1'), ('G', 'C1'), ('RG', 'C1'), ('RG', 'R'), ('RG', 'G')]:
        changes = []
        for repeat in ('1', '2'):
            a, b = f'r{repeat}_{candidate}', f'r{repeat}_{reference}'
            if a in data and b in data:
                changes.append(dict(repeat=repeat, **{k:data[a]['equal_camera_mean'][k]-data[b]['equal_camera_mean'][k] for k in METRICS}))
        paired[candidate+'-minus-'+reference] = dict(status='complete' if len(changes) == 2 else 'pending', repeats=len(changes), changes=changes,
            delta=mean_vector(changes) if len(changes) == 2 else None,
            direction_consistent={k:all(r[k] >= 0 for r in changes) or all(r[k] <= 0 for r in changes) for k in METRICS} if len(changes) == 2 else None)
    complete_core = not missing_primary and not missing_extra
    frequency_summary = []
    for endpoint in sorted({row['endpoint'] for row in all_frequency}):
        for camera in sorted({row['camera'] for row in all_frequency if row['endpoint'] == endpoint}):
            selected = [row for row in all_frequency if row['endpoint'] == endpoint and row['camera'] == camera]
            # U6000's newly rendered diagnostic stream supersedes its old cache
            # for comparison to current endpoints, never counted twice.
            current = [row for row in selected if row.get('source') != 'historical_registered_float_fft']
            if current:
                selected = current
            mse = statistics.mean(float(row['mse']) for row in selected)
            values = {band+'_mse':statistics.mean(float(row[band+'_mse']) for row in selected) for band in ('low', 'mid', 'high')}
            frequency_summary.append(dict(endpoint=endpoint, camera=camera, observations=len(selected), mse=mse, **values,
                **{band+'_share_percent':100*values[band+'_mse']/mse for band in ('low', 'mid', 'high')}))
    summary = dict(status='completed_core_evidence' if complete_core else 'pending',
                   execution=dict(expected_core_endpoints=12, completed_primary_endpoints=12-len(missing_primary),
                                  missing_primary=missing_primary, missing_extra=missing_extra, core_evidence_complete=complete_core),
                   primary_metrics=list(METRICS), primary_aggregation='equal frame within camera -> equal cam00/cam01 -> equal suffix; mean dB, not PSNR of pooled MSE',
                   means=means, paired=paired, data=data, provenance=provenance,
                   conclusions='Core evidence is available for review; no automatic combined score or coverage veto.' if complete_core else 'Core experiment or required extra diagnostics remain incomplete; scientific conclusions and conditional expansion decision are pending.',
                   protocol=entry(OUT/'protocol.json'), historical_protocol=entry(ROOT/'output/dynamic_sr_temporal_prior_20260928/protocol.json'),
                   repeat_interpretation='Two suffixes resume the identical U6000 state with preregistered alternative observation orders; they are not independent from-scratch runs.',
                   hardware_interpretation='Read training_gpu/host from actual completion and dispatch records. Shared evaluation on local GPU1 standardizes metric/render environment but does not remove training hardware optimization differences; tiny gains need same-hardware or paired confirmation.',
                   benchmark_scope='Fixed cook_spinach short window; cam00/cam01 are development views. Not a full independent-scene benchmark.',
                   sources={str(Path(__file__).relative_to(ROOT)):sha(__file__)})
    for name, rows in [('quality_per_endpoint.csv', primary_rows), ('quality_summary.csv', mean_rows), ('quality_gaps.csv', gaps),
                   ('metrics_per_frame.csv', frames), ('frequency_budget.csv', all_frequency), ('regional_quality.csv', all_regions), ('cost_ledger.csv', costs)]:
        csvwrite(OUT/name, rows)
    csvwrite(OUT/'frequency_summary.csv', frequency_summary)
    write(OUT/'quality_summary.json', summary)
    write(OUT/'summary_state.json', dict(status=summary['status'], execution=summary['execution']))
    return summary


if __name__ == '__main__':
    result = summarize()
    print(dict(status=result['status'], execution=result['execution'], available_means=list(result['means'])), flush=True)
