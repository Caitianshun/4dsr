"""Read-only, identity-bound six-arm short-window quality and cost summary.

Missing endpoints produce a pending report with no ranking or recommendation.
Three metric units are never combined into an invented scalar. Development
selection is at most two Pareto candidates; ambiguous fronts require an explicit
root research decision bound to the already completed endpoint evidence.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from statistics import mean
from fp_common import ROOT, HERE, OUT, read, write, sha, entry, bound, local

METHODS = ('B0', 'Bsync', 'M', 'X', 'MX', 'E')
METRICS = ('psnr', 'ssim', 'lpips')
BANDS = ('low', 'mid', 'high')
COMPARISONS = (('Bsync', 'B0'), ('M', 'Bsync'), ('X', 'Bsync'),
               ('MX', 'M'), ('MX', 'X'), ('E', 'B0'))
ROI_PATH = ROOT/'output/dynamic_sr_prior_diagnosis_20260929/spectrum/roi_protocol.json'


def numeric(value):
    if value in (None, ''): return None
    result = float(value)
    if not math.isfinite(result): raise ValueError(f'Nonfinite measured value: {value}')
    return result


def csv_rows(path):
    with Path(path).open(newline='') as source: return list(csv.DictReader(source))


def csv_write(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_suffix(path.suffix+'.tmp')
    with temporary.open('w', newline='') as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    temporary.replace(path)


def quality(rows, fields=METRICS):
    if not rows: raise ValueError('Empty quality observation set')
    return {field: mean(numeric(row[field]) for row in rows) for field in fields}


def fft_summary(rows):
    if not rows: raise ValueError('Empty FFT observation set')
    result = dict(observations=len(rows), mse=mean(numeric(row['mse']) for row in rows),
        mean_frame_psnr=mean(numeric(row['psnr']) for row in rows),
        maximum_parseval_gap=max(numeric(row['parseval_gap']) for row in rows))
    total = sum(numeric(row['mse']) for row in rows)
    for band in BANDS:
        energy = sum(numeric(row[band+'_mse']) for row in rows)
        result[band+'_mse'] = energy/len(rows)
        result[band+'_share_pct'] = 100*energy/total if total else None
    result['share_definition'] = 'sum band MSE / sum full MSE; both absolute error and pooled share retained'
    return result


def delta(current, baseline):
    raw = {metric: current[metric]-baseline[metric] for metric in METRICS}
    return dict(raw_delta=raw, benefit_delta=dict(psnr=raw['psnr'], ssim=raw['ssim'], negative_lpips=-raw['lpips']))


def interaction(mx, m, x, bsync):
    raw = {metric: (mx[metric]-m[metric])-(x[metric]-bsync[metric]) for metric in METRICS}
    return dict(raw_delta=raw, benefit_delta=dict(psnr=raw['psnr'], ssim=raw['ssim'], negative_lpips=-raw['lpips']),
        formula='(MX-M)-(X-Bsync); LPIPS benefit interaction is the negative of the raw LPIPS interaction')


def pareto(means):
    directions = {'psnr': 1, 'ssim': 1, 'lpips': -1}
    front, dominated = [], {}
    for method, point in means.items():
        dominators = []
        for other, rival in means.items():
            if other == method: continue
            differences = [directions[key]*(rival[key]-point[key]) for key in METRICS]
            if all(value >= 0 for value in differences) and any(value > 0 for value in differences):
                dominators.append(other)
        if dominators: dominated[method] = dominators
        else: front.append(method)
    return dict(front=front, dominated_by=dominated,
        definition='Weakly better in all three separate metric units and strictly better in at least one; no quality threshold.')


def _expected(manifest):
    return {(row['camera_id'], int(row['frame_index'])): row for row in manifest['observations']
            if row['camera_id'] in ('cam00', 'cam01') or
               (row['split'] == 'train' and int(row['frame_index']) in (0, 40, 80, 118))}


def load_endpoint(label, protocol, expected):
    """Bind completed receipts, CSV/index hashes and diagnostic immutability."""
    run = OUT/'runs'/label; evaluation = OUT/'evaluation'/label/'extra'
    diagnostic = OUT/'diagnostics'/label
    needed = [run/'complete.json', run/'config.json', evaluation/'adapter_complete.json',
              evaluation/'complete.json', evaluation/'float_index.json',
              diagnostic/'diagnostics.json']
    missing = [str(path.relative_to(ROOT)) for path in needed if not path.exists()]
    if missing: return None, dict(label=label, missing=missing)
    train = read(run/'complete.json'); config = read(run/'config.json')
    method = label.split('_', 1)[1]; repeat = label[1]
    if train['status'] != 'completed_training' or train['updates'] != 6000 or train['method'] != method:
        raise ValueError(f'{label}: training is not the registered 6000 endpoint')
    if train['training_rgb_forwards'] != 18000 or train['moment_forwards'] != (18000 if method in ('X', 'MX') else 0) or train['adam_calls'] != 12000:
        raise ValueError(f'{label}: registered forward/optimizer budget mismatch')
    checkpoint = bound(train['checkpoint'])
    if config['parent'] != protocol['parent'] or config['protocol_sha256'] != sha(OUT/'protocol.json'):
        raise ValueError(f'{label}: parent/protocol mismatch')
    position = 0
    for item in train['segments']:
        segment = read(bound(item))
        if segment['start'] != position: raise ValueError(f'{label}: segment overlap or gap')
        position = segment['suffix_endpoint']
    if position != 6000: raise ValueError(f'{label}: incomplete committed training trajectory')
    adapter = read(evaluation/'adapter_complete.json'); extra = read(evaluation/'complete.json')
    if adapter['status'] != 'completed_unchanged_uniform_float_evaluation' or adapter['observations'] != 196:
        raise ValueError(f'{label}: invalid uniform-evaluation adapter receipt')
    if adapter['checkpoint'] != train['checkpoint'] or adapter['protocol'] != entry(OUT/'protocol.json'):
        raise ValueError(f'{label}: evaluation checkpoint/protocol mismatch')
    bound(adapter['extra_complete'])
    if extra['status'] != 'completed_extra_evaluation' or extra['observations'] != 196 or extra['train76'] != 76:
        raise ValueError(f'{label}: incomplete extra evaluation')
    if extra['identity']['checkpoint_sha256'] != train['checkpoint']['sha256'] or extra['parameter_updates'] != 0:
        raise ValueError(f'{label}: evaluation identity or no-update violation')
    for item in extra['results'].values(): bound(item)
    floats = read(evaluation/'float_index.json')
    if floats['status'] != 'completed' or floats['identity'] != extra['identity'] or floats['parameter_updates'] != 0:
        raise ValueError(f'{label}: float index identity mismatch')
    float_keys = [(row['camera'], int(row['frame'])) for row in floats['entries']]
    if len(float_keys) != 196 or set(float_keys) != set(expected): raise ValueError(f'{label}: float observations mismatch')
    if any(row['checkpoint_sha256'] != train['checkpoint']['sha256'] for row in floats['entries']):
        raise ValueError(f'{label}: float files claim a different checkpoint')
    rows = csv_rows(evaluation/'metrics_per_frame.csv'); frequencies = csv_rows(evaluation/'frequency_budget.csv')
    regions = csv_rows(evaluation/'regional_quality.csv')
    for collection in (rows, frequencies):
        keys = [(row['camera'], int(row['frame'])) for row in collection]
        if len(keys) != 196 or set(keys) != set(expected): raise ValueError(f'{label}: CSV observations mismatch or duplicate')
        if any(row['endpoint'] != label for row in collection): raise ValueError(f'{label}: wrong CSV endpoint')
        for row in collection:
            key = (row['camera'], int(row['frame']))
            wanted = 'train76' if expected[key]['split'] == 'train' else 'development'
            if row['split'] != wanted: raise ValueError(f'{label}: CSV split mismatch')
    for row in rows:
        original = expected[(row['camera'], int(row['frame']))]
        if row['hr_sha256'] != original['hr_sha256'] or row['lr_sha256'] != original['lr_sha256']:
            raise ValueError(f'{label}: evaluation source image identity mismatch')
        for metric in METRICS: numeric(row[metric])
    for row in frequencies:
        energy = sum(numeric(row[band+'_mse']) for band in BANDS)
        if abs(energy-numeric(row['mse'])) > 1e-10: raise ValueError(f'{label}: Parseval band accounting mismatch')
    roi = read(ROI_PATH)
    wanted_regions = {(camera, frame, name) for camera, groups in roi['regions_by_camera_xyxy_exclusive'].items()
                      for frame in (40, 80) for name in list(groups)+['rest_complement']}
    actual_regions = {(row['camera'], int(row['frame']), row['region']) for row in regions}
    # Historical roi_metrics versions may use rest/rest_complement names or
    # include overlap-count rows. Fixed named ROI entries must always exist.
    fixed_regions = {(camera, frame, name) for camera, groups in roi['regions_by_camera_xyxy_exclusive'].items()
                     for frame in (40, 80) for name in groups}
    if not fixed_regions.issubset(actual_regions) or len(actual_regions) != len(regions):
        raise ValueError(f'{label}: fixed ROI missing or duplicated')
    if extra['roi_protocol_sha256'] != sha(ROI_PATH): raise ValueError(f'{label}: ROI protocol changed')
    diag = read(diagnostic/'diagnostics.json')
    if diag['status'] != 'passed_read_only_fixed_train76_diagnostics' or diag['checkpoint'] != train['checkpoint']:
        raise ValueError(f'{label}: fixed diagnostics are incomplete or inspect another checkpoint')
    if diag['cost']['formal_updates'] != 0 or diag['cost']['Adam_calls'] != 0:
        raise ValueError(f'{label}: diagnostic performed training updates')
    for field in ('parent_state_unchanged', 'two_Adam_states_unchanged', 'global_RNG_unchanged', 'tensor_mutation_versions_unchanged'):
        if not diag[field]: raise ValueError(f'{label}: failed diagnostic immutability: {field}')
    for field in ('immutability_audit', 'gdiag', 'calibration', 'schedule', 'support_index'): bound(diag[field])
    if len(diag['rows']) != 76 or diag['schedule'] != entry(OUT/'schedules/schedule_1.json'):
        raise ValueError(f'{label}: nonuniform fixed train76 diagnostic schedule')
    development = [row for row in rows if row['split'] == 'development']
    by_camera = {camera: quality([row for row in development if row['camera'] == camera]) for camera in ('cam00', 'cam01')}
    overall = {metric: mean(by_camera[camera][metric] for camera in ('cam00', 'cam01')) for metric in METRICS}
    per_region = {}
    for camera, region in sorted({(row['camera'], row['region']) for row in regions}):
        values = [row for row in regions if row['camera'] == camera and row['region'] == region]
        per_region[camera+'/'+region] = dict(camera=camera, region=region, observations=len(values),
            psnr=mean(numeric(row['psnr']) for row in values), ssim=mean(numeric(row['ssim']) for row in values),
            lpips=mean(numeric(row['lpips_alex_spatial_mask']) for row in values),
            LPIPS_role='regional full-context spatial map; not the full-image LPIPS scalar')
    training = [row for row in rows if row['split'] == 'train76']
    auxiliary = ('lr_l1', 'lr_mse', 'teacher_rgb_l1', 'teacher_H_l1', 'alpha_support_fraction',
                 'variance_supported_mean', 'expected_z_supported_mean')
    train_stats = {field: mean(numeric(row[field]) for row in training if numeric(row.get(field)) is not None)
                   for field in auxiliary if any(numeric(row.get(field)) is not None for row in training)}
    result = dict(label=label, method=method, repeat=repeat, overall=overall, by_camera=by_camera,
        ROI=per_region, FFT=fft_summary([row for row in frequencies if row['split'] == 'development']),
        FFT_by_camera={camera: fft_summary([row for row in frequencies if row['camera'] == camera]) for camera in ('cam00', 'cam01')},
        train76=train_stats, fixed_diagnostics=diag,
        training_hardware={key: config.get(key) for key in ('gpu', 'physical_gpu', 'torch', 'cuda')},
        evaluation_hardware={key: extra.get(key) for key in ('gpu', 'physical_gpu', 'torch', 'cuda', 'extension_sha256')},
        training_cost=train, evaluation_cost=extra,
        source_files={str(path.relative_to(ROOT)): entry(path) for path in needed+[evaluation/name for name in ('metrics_per_frame.csv', 'frequency_budget.csv', 'regional_quality.csv')]},
        float_index_bound=True, float_bytes_rehashed_by_summary=False,
        float_byte_verification_scope='The unchanged evaluator writes/reuses SHA-checked raw floats; summary binds its index and CSV/complete receipt without rerendering.',
        _rows=rows, _frequency_rows=frequencies, _region_rows=regions)
    return result, None


def _cost_record(path, category, data, cost=None, note=None):
    cost = data.get('cost', data.get('counters', data)) if cost is None else cost
    def choose(names):
        for key in names:
            if key in cost and isinstance(cost[key], (int, float)): return float(cost[key])
        return None
    return dict(receipt=entry(path), category=category, status=data.get('status'),
        rgb_forwards=choose(('training_rgb_forwards', 'rgb_forwards', 'rgb_forwards_new', 'RGB_forwards', 'RGB_renderer_forwards')),
        moment_forwards=choose(('moment_forwards', 'moment_forwards_new', 'moment_renderer_forwards')),
        updates=choose(('formal_updates', 'parameter_updates', 'formal_experiment_updates', 'updates')),
        Adam_calls=choose(('Adam_calls', 'adam_calls')),
        wall_seconds=choose(('wall_seconds', 'wall_s', 'seconds', 'CPU_seconds', 'preparation_seconds')),
        train_seconds=choose(('train_s',)), toy_CPU_Adam_calls=choose(('toy_CPU_optimizer_calls',)), note=note)


def cost_ledger(endpoints):
    """Charge leaf receipts once; aggregation wrappers and aliases are skipped."""
    entries, seen, unknown = [], set(), []
    def add(path, category, data=None, cost=None, note=None):
        path = Path(path)
        if not path.exists(): return
        identity = sha(path)
        if identity in seen: return
        seen.add(identity); data = read(path) if data is None else data
        entries.append(_cost_record(path, category, data, cost, note))
    for name in METHODS:
        for suffix in ('1', '2'):
            label = f'r{suffix}_{name}'; directory = OUT/'runs'/label
            if (directory/'complete.json').exists():
                add(directory/'complete.json', 'committed_core_training',
                    note='Cost is charged even while this endpoint still lacks its evaluation/diagnostics; scientific readiness is separate.')
            else:
                for path in sorted(directory.glob('segment_*.json')):
                    add(path, 'committed_partial_core_training')
            evaluation = OUT/'evaluation'/label/'extra'
            if (evaluation/'complete.json').exists():
                value = read(evaluation/'complete.json')
                # New-cohort raw floats are produced by one native RGB+moment
                # render each. A resumed final attempt can report fewer new
                # forwards, while its existing cached floats retain the
                # earlier successful asset-producing work. Charge that work
                # once rather than silently dropping it after a failed attempt.
                counter = dict(rgb_forwards=value['observations'], moment_forwards=value['observations'],
                    formal_updates=0, Adam_calls=0, seconds=value['seconds'])
                add(evaluation/'complete.json', 'uniform_evaluation', value, counter,
                    note=f"New-cohort asset-producing lower bound, including earlier cached work once; final attempt reports {value['rgb_forwards_new']} RGB and {value['moment_forwards_new']} moments. Failed renders before a durable float and prior failed-attempt elapsed time remain unknown.")
                if value.get('cached_observations', 0):
                    unknown.append(f'{label}: evaluation final timer does not include earlier failed cached-work wall; any unpersisted forwards remain unknown.')
            else:
                float_receipts = list((evaluation/'floats').glob('*.json'))
                if float_receipts:
                    values = [read(path) for path in float_receipts]
                    count = sum('checkpoint_sha256' in value and 'information' in value for value in values)
                    # One aggregate lower bound uses its existing config as
                    # an identity reference; these are not per-frame GPU costs.
                    config = evaluation/'config.json'
                    if config.exists():
                        add(config, 'partial_evaluation_durable_float_lower_bound', read(config),
                            dict(rgb_forwards=count, moment_forwards=count, formal_updates=0, Adam_calls=0),
                            note='Incomplete evaluation; durable new-cohort raw float receipts prove returned RGB+moment pairs. Attempt wall and unpersisted work are unknown.')
                    unknown.append(f'{label}: partial evaluation elapsed and unpersisted work unknown.')
            add(OUT/'diagnostics'/label/'diagnostics.json', 'fixed_train76_endpoint_diagnostic')
    for path in (OUT/'schedules/index.json', OUT/'support_physical_moment_fix_checks.json'):
        add(path, 'CPU_preparation_or_operator_check')
    for path in sorted((OUT/'operator_checks').rglob('*.json')):
        if 'fixtures' in path.parts or path.name in ('latest.json', 'cost_and_final_receipt.json'): continue
        data = read(path)
        if 'counters' in data:
            counter = dict(data['counters'])
            if 'wall_seconds' in data: counter['wall_seconds'] = data['wall_seconds']
            add(path, 'native_operator_check_success_or_failure', data, counter)
        elif 'seconds' in data and ('CPU_actual_training' in data.get('status', '') or 'CPU' in data.get('status', '')):
            add(path, 'CPU_operator_check_success_or_failure', data)
    parent_index = OUT/'support/parent_moments_hr/index.json'
    if parent_index.exists():
        value = read(parent_index)
        counters = dict(rgb_forwards=0, moment_forwards=value['total_moment_forwards'],
            parameter_updates=0, Adam_calls=0, seconds=value['per_entry_seconds_sum'])
        add(parent_index, 'frozen_HR_parent_export', value, counters,
            note='Sum the 1140 per-entry elapsed costs once, including cached entries from an interrupted export; startup overhead outside those timers is unknown.')
    else:
        for path in sorted((OUT/'support/parent_moments_hr').glob('cam*.json')):
            add(path, 'partial_frozen_HR_parent_export')
    support_index = OUT/'support/frozen/index.json'
    if support_index.exists():
        value = read(support_index)
        resumed = value.get('new_edges', len(value['entries'])) != len(value['entries'])
        if resumed:
            tau = read(bound(value['tau_registration']))
            counter = dict(rgb_forwards=0, moment_forwards=0, updates=0, Adam_calls=0,
                seconds=sum(row['seconds'] for row in value['entries'])+tau['seconds'])
            add(support_index, 'frozen_CPU_edge_support_preparation', value, counter,
                note='Resumed: sum unique asset-producing pair timers and the original tau timer once; this is a known nonoverlapping work lower bound, not a full attempt wall.')
            unknown.append('Frozen support resumed cached edge work. The final-attempt timer alone does not cover prior attempt wall; per-edge/tau receipts remain available.')
        else:
            add(support_index, 'frozen_CPU_edge_support_preparation', value,
                note='The all-new successful attempt wall includes tau preparation; do not additionally charge the nested tau timer.')
    if (OUT/'calibration.json').exists(): add(OUT/'calibration.json', 'single_train76_X_E_calibration')
    for path in sorted(OUT.rglob('*failure*.json')):
        if 'source_preparation_archive' in path.parts or 'operator_checks' in path.parts or 'fixtures' in path.parts: continue
        value = read(path)
        if value.get('last_attempt'):
            try:
                attempt = read(bound(value['last_attempt'])); counter = dict(attempt.get('confirmed_cost', {}))
                if 'seconds' in attempt: counter['seconds'] = attempt['seconds']
                add(path, 'failed_preparation_or_diagnostic_confirmed_cost', value, counter,
                    note='Returned-operation lower bound; any active operation at failure remains unknown.')
            except Exception as error:
                unknown.append(f'{path.relative_to(ROOT)}: cannot bind partial failed-operation cost: {error}')
        elif value.get('cost'): add(path, 'failed_preparation_or_diagnostic', value)
        elif 'failed' in value.get('status', ''):
            unknown.append(f'{path.relative_to(ROOT)}: failed operation has no instrumented elapsed/render count.')
    for path in sorted((OUT/'runs').glob('*/recovery_incidents/*/receipt.json')):
        value = read(path); incomplete = value.get('incomplete_attempt') or {}
        counter = dict(rgb_forwards=value.get('confirmed_RGB_forwards', 0)+incomplete.get('rgb', 0),
            moment_forwards=value.get('confirmed_moment_forwards', 0)+incomplete.get('moments', 0),
            updates=value.get('confirmed_completed_updates', 0),
            Adam_calls=value.get('confirmed_Adam_calls', 0)+incomplete.get('adam_calls', 0))
        add(path, 'failed_uncommitted_training_work', value, counter,
            note='Count actual discarded work separately from the 72000 committed endpoint updates; unreturned active operation and elapsed time can be unknown.')
        unknown.append(f'{path.relative_to(ROOT)}: uncommitted-tail wall is not reliably recoverable across process time origins.')
    data_root = OUT/'full_data_readiness'
    startup = data_root/'startup_incident_costs.json'
    if startup.exists():
        value = read(startup)
        for failed in value['failed_attempts']:
            path = bound(failed['state'])
            add(path, 'failed_full_data_preparation_or_retrieval', read(path),
                dict(rgb_forwards=0, moment_forwards=0, updates=0, Adam_calls=0, wall_seconds=failed['wall_seconds']))
        unknown.append('Full-data packet/range probe wall was not individually instrumented; it is unknown, not zero.')
    for path in sorted(data_root.glob('execution_state_*.json')):
        value = read(path)
        if value.get('finished_unix') is not None:
            seconds = value['finished_unix']-value['started_unix']
            add(path, 'full_data_CPU_worker', value,
                dict(rgb_forwards=0, moment_forwards=0, updates=0, Adam_calls=0, wall_seconds=seconds),
                note='Do not also add nested per-scene decode/acquisition timers covered by this worker wall.')
        else: unknown.append(f'{path.relative_to(ROOT)}: worker not terminal; final wall is pending.')
    # Old parent evaluation/calibration/source history is a reused reference,
    # never an actual new forward cost for this experiment.
    groups = {}
    for category in sorted({row['category'] for row in entries}):
        values = [row for row in entries if row['category'] == category]
        groups[category] = dict(receipts=len(values),
            known_sums={field: sum(row[field] for row in values if row[field] is not None)
                        for field in ('rgb_forwards', 'moment_forwards', 'updates', 'Adam_calls', 'wall_seconds', 'train_seconds', 'toy_CPU_Adam_calls')},
            unknown_fields={field: sum(row[field] is None for row in values)
                            for field in ('rgb_forwards', 'moment_forwards', 'wall_seconds')})
    return dict(entries=entries, groups=groups,
        known_total_sums={field: sum(row[field] for row in entries if row[field] is not None)
            for field in ('rgb_forwards', 'moment_forwards', 'updates', 'Adam_calls', 'wall_seconds', 'toy_CPU_Adam_calls')},
        unknown_or_pending=sorted(set(unknown))+['Uninstrumented process import/loading, editing, model analysis and some early CPU math checks are not assumed zero.'],
        wall_sum_is_parallel_makespan=False, train_seconds_are_subset_of_wall_not_added_twice=True,
        stages_overlap_across_workers=True, historical_reference_cost_charged_again=False,
        synthetic_fixture_counts_are_not_formal_training=True)


def selection(front, methods, decision=None):
    candidates = [method for method in front['front'] if method != 'B0']
    if decision is not None:
        choices = list(decision['selected_candidates'])
        if len(choices) > 2 or len(set(choices)) != len(choices): raise ValueError('Select at most two distinct full-development candidates')
        if any(choice not in candidates for choice in choices): raise ValueError('Research selection must disclose a development Pareto candidate')
        if not decision.get('rationale'): raise ValueError('Research selection needs an evidence-based rationale')
        return dict(status='root_research_selected_full_development_candidates', selected_candidates=choices,
            baseline='B0', rationale=decision['rationale'], confirmation_scores_used=False,
            maximum_candidates=2, Pareto_candidates=candidates)
    if len(candidates) <= 2:
        return dict(status='provisional_Pareto_development_selection' if candidates else 'B0_retained_no_quality_Pareto_alternative',
            selected_candidates=candidates, baseline='B0', maximum_candidates=2,
            rationale='Retain all non-B0 nondominated tradeoffs when there are at most two; disclose their exact metric/suffix/camera costs before full-time development.',
            confirmation_scores_used=False, Pareto_candidates=candidates,
            root_visual_and_magnitude_review_required_before_full_training=True)
    return dict(status='awaiting_root_quality_selection_from_complete_evidence', selected_candidates=[], baseline='B0',
        maximum_candidates=2, Pareto_candidates=candidates, confirmation_scores_used=False,
        rationale='More than two nondominated metric tradeoffs cannot be honestly collapsed into a scalar or arbitrary metric priority. Root must bind a choice of at most two to actual amplitudes, paired suffixes, camera distribution and fixed visuals.')


def build(decision_path=None):
    protocol = read(OUT/'protocol.json'); manifest = read(bound(protocol['manifest'])); expected = _expected(manifest)
    if len(expected) != 196: raise ValueError('Registered 120 development +76 train observations required')
    endpoints, missing, invalid = {}, [], []
    for repeat in ('1', '2'):
        for method in METHODS:
            label = f'r{repeat}_{method}'
            try:
                result, pending = load_endpoint(label, protocol, expected)
                if pending: missing.append(pending)
                else: endpoints[label] = result
            except Exception as error:
                invalid.append(dict(label=label, error=repr(error)))
    core_evidence = {label: dict(checkpoint=result['training_cost']['checkpoint'],
        training=entry(OUT/'runs'/label/'complete.json'),
        evaluation=entry(OUT/'evaluation'/label/'extra/complete.json'),
        diagnostics=entry(OUT/'diagnostics'/label/'diagnostics.json')) for label, result in endpoints.items()}
    identity = dict(protocol=entry(OUT/'protocol.json'), source=entry(Path(__file__)),
        schedules=entry(OUT/'schedules/index.json'), roi=entry(ROI_PATH), completed_endpoint_evidence=core_evidence)
    cost = cost_ledger(endpoints)
    data_inventory = OUT/'full_data_readiness/inventory.json'
    full_data = dict(inventory=entry(data_inventory), contents=read(data_inventory)) if data_inventory.exists() else dict(status='pending_data_inventory')
    base = dict(created_UTC=datetime.now(timezone.utc).isoformat(), identity=identity,
        registered_endpoints=12, completed_identity_checked_endpoints=len(endpoints), missing=missing, invalid=invalid,
        budget=protocol['budget'], cost=cost, full_data_readiness=full_data,
        recommendation=None, selection_uses_confirmation_HR=False,
        limitations=['Cook full-background 60-even-frame 0–118 short-window development; cam00/01 have already been used for development.',
            'Two continuations share U6000; they are not independent from-scratch seeds. Within-suffix hardware must match across all six arms.',
            'Arithmetic mean frame PSNR, then equal camera/suffix means; pooled-MSE PSNR is not substituted.',
            'ROI and FFT explain costs; they do not replace whole-frame quality. Lower low-frequency share alone is not reduced absolute low-band error.',
            'X residual, dispersion and cosine are mechanism clues, not true geometry accuracy. Video frames are correlated; no significance or independent-frame confidence claim.',
            'A positive MX mean does not prove synergy. Interaction is reported for each metric, with LPIPS benefit direction reversed.',
            'Full-time development and unused confirmation must follow a frozen recommendation; this summary does not claim an official complete benchmark.'])
    if missing or invalid:
        base.update(status='pending_complete_12_endpoint_evidence', metrics=None, paired=None,
            interaction=None, Pareto=None, recommendation=None,
            answers=dict(configuration='Pending actual 12 endpoint measurements; no quality recommendation.',
                M_X_effectiveness='Pending complete controlled contrasts.',
                low_frequency_and_dynamic_cost='Pending absolute FFT/whole-frame/hand ROI evidence.',
                full_scene_next_step='CPU full-data preparation continues; method refinement cannot be selected from absent results.'))
        return base, endpoints
    # Check pairing platform independently from quality values.
    platform_checks = {}
    for repeat in ('1', '2'):
        values = [endpoints[f'r{repeat}_{method}']['training_hardware'] for method in METHODS]
        platform = {(row['gpu'], row['torch'], row['cuda']) for row in values}
        if len(platform) != 1: raise ValueError(f'Suffix {repeat} methods were trained on different platforms')
        platform_checks[repeat] = values
    evaluation_platforms = {(row['evaluation_hardware']['gpu'], row['evaluation_hardware']['torch'],
                             row['evaluation_hardware']['cuda'], row['evaluation_hardware']['extension_sha256']) for row in endpoints.values()}
    if len(evaluation_platforms) != 1: raise ValueError('Uniform evaluation platform mismatch across core endpoints')
    methods = {}
    for method in METHODS:
        records = [endpoints[f'r{repeat}_{method}'] for repeat in ('1', '2')]
        camera_values = {camera: {metric: mean(row['by_camera'][camera][metric] for row in records) for metric in METRICS}
                         for camera in ('cam00', 'cam01')}
        methods[method] = dict(overall={metric: mean(row['overall'][metric] for row in records) for metric in METRICS},
            by_suffix={str(repeat): row['overall'] for repeat, row in zip((1, 2), records)}, by_camera=camera_values,
            ROI={key: {metric: mean(row['ROI'][key][metric] for row in records) for metric in METRICS} for key in records[0]['ROI']},
            FFT=fft_summary([value for row in records for value in row['_frequency_rows'] if value['split'] == 'development']),
            FFT_by_camera={camera: fft_summary([value for row in records for value in row['_frequency_rows'] if value['camera'] == camera])
                           for camera in ('cam00', 'cam01')},
            train76={field: mean(row['train76'][field] for row in records) for field in records[0]['train76']})
    paired = {}
    for current, baseline in COMPARISONS + tuple((method, 'B0') for method in ('M', 'X', 'MX')):
        name = current+'_minus_'+baseline
        camera_delta = {camera: delta(methods[current]['by_camera'][camera], methods[baseline]['by_camera'][camera])
                        for camera in ('cam00', 'cam01')}
        per_suffix = {repeat: dict(overall=delta(endpoints[f'r{repeat}_{current}']['overall'], endpoints[f'r{repeat}_{baseline}']['overall']),
            by_camera={camera: delta(endpoints[f'r{repeat}_{current}']['by_camera'][camera], endpoints[f'r{repeat}_{baseline}']['by_camera'][camera])
                       for camera in ('cam00', 'cam01')}) for repeat in ('1', '2')}
        paired[name] = dict(current=current, baseline=baseline,
            overall=delta(methods[current]['overall'], methods[baseline]['overall']), by_camera=camera_delta, by_suffix=per_suffix,
            ROI={key: delta(methods[current]['ROI'][key], methods[baseline]['ROI'][key]) for key in methods[current]['ROI']},
            FFT_by_camera={camera: {field: methods[current]['FFT_by_camera'][camera][field]-methods[baseline]['FFT_by_camera'][camera][field]
                for field in ('mse', 'low_mse', 'mid_mse', 'high_mse', 'low_share_pct', 'mid_share_pct', 'high_share_pct')}
                for camera in ('cam00', 'cam01')},
            per_observation=[dict(repeat=repeat, camera=key[0], frame=key[1],
                **delta(a[key], b[key])) for repeat in ('1', '2')
                for a, b in [({(r['camera'], int(r['frame'])): {metric: numeric(r[metric]) for metric in METRICS}
                              for r in endpoints[f'r{repeat}_{current}']['_rows'] if r['split'] == 'development'},
                             {(r['camera'], int(r['frame'])): {metric: numeric(r[metric]) for metric in METRICS}
                              for r in endpoints[f'r{repeat}_{baseline}']['_rows'] if r['split'] == 'development'})]
                for key in sorted(a)])
    interactions = dict(overall=interaction(*(methods[method]['overall'] for method in ('MX', 'M', 'X', 'Bsync'))),
        by_suffix={repeat: interaction(*(endpoints[f'r{repeat}_{method}']['overall'] for method in ('MX', 'M', 'X', 'Bsync'))) for repeat in ('1', '2')},
        by_camera={camera: interaction(*(methods[method]['by_camera'][camera] for method in ('MX', 'M', 'X', 'Bsync'))) for camera in ('cam00', 'cam01')})
    front = pareto({method: value['overall'] for method, value in methods.items()})
    decision = None
    if decision_path:
        decision = read(decision_path)
        if decision.get('status') != 'completed_root_quality_selection' or decision.get('completed_endpoint_evidence') != core_evidence:
            raise ValueError('Root decision is not bound to this actual complete endpoint evidence')
        if decision.get('confirmation_scores_used', True): raise ValueError('Confirmation quality cannot choose the configuration')
    chosen = selection(front, methods, decision)
    base.update(status='completed_core_quality_evidence', metrics=methods, paired=paired,
        interaction=interactions, Pareto=front, recommendation=chosen,
        root_decision=entry(decision_path) if decision_path else None,
        paired_training_platform_checks=platform_checks, uniform_evaluation_platform=list(evaluation_platforms)[0],
        endpoint_summaries={label: {key: value for key, value in row.items() if not key.startswith('_') and
            key not in ('fixed_diagnostics', 'training_cost', 'evaluation_cost')} for label, row in endpoints.items()},
        fixed_diagnostics={label: dict(gradient_RMS_distributions=row['fixed_diagnostics']['gradient_RMS_distributions'],
            gradient_cosine_distributions=row['fixed_diagnostics']['gradient_cosine_distributions'],
            balanced_anchor76_structure=row['fixed_diagnostics']['balanced_anchor76_structure'],
            triplet228_structure=row['fixed_diagnostics']['triplet228_structure'],
            X_diagnostic_distribution=row['fixed_diagnostics']['X_diagnostic_distribution']) for label, row in endpoints.items()})
    return base, endpoints


def readable(report):
    if report['status'].startswith('pending'):
        return ('# 多视图联合与跨视图足迹验证：执行状态\n\n'
            f"已取得并核验 {report['completed_identity_checked_endpoints']}/12 个训练、196观察评价及固定train76诊断端点。"
            '尚未取得完整结果，当前不推荐配置，也不判断 M/X 是否有效。\n\n'
            f"缺失端点：{', '.join(row['label'] for row in report['missing']) or '无'}。"
            f"无效证据端点：{', '.join(row['label'] for row in report['invalid']) or '无'}。\n\n"
            '正式预算为 72,000 更新、216,000 RGB、72,000 辅助矩；这些是登记预算，不是已完成数量。'
            '实际已知成本见 cost_ledger.csv 和 summary.json。完整数据准备与资源调度继续独立推进。\n')
    selected = report['recommendation']['selected_candidates']
    lines = ['# 多视图联合与跨视图足迹验证：短窗决策证据', '',
        ('完整时间开发候选：'+('、'.join(selected) if selected else '尚需根代理核对指标取舍；或保留 B0，见机器可读选择状态')+'。B0 保留为同骨干基线。'),
        '所有质量变化来自两个共享 U6000 的续训后缀，固定 6000 端点；它们不是独立从头训练种子。', '',
        '| 配置 | PSNR ↑ | SSIM ↑ | LPIPS ↓ |', '|---|---:|---:|---:|']
    for method in METHODS:
        q = report['metrics'][method]['overall']; lines.append(f"| {method} | {q['psnr']:.6f} | {q['ssim']:.8f} | {q['lpips']:.8f} |")
    lines += ['', '三指标分别保留，未相加为总分。PSNR 是逐帧 dB 的均值，再对两个开发相机及两个后缀等权平均。', '',
        '| 比较 | ΔPSNR | ΔSSIM | ΔLPIPS |', '|---|---:|---:|---:|']
    for name, row in report['paired'].items():
        q = row['overall']['raw_delta']; lines.append(f"| {name} | {q['psnr']:+.6f} | {q['ssim']:+.8f} | {q['lpips']:+.8f} |")
    lines += ['', '每项比较的两个后缀、cam00/cam01 分布、固定手部/墙角 ROI 和逐观察差值均在 CSV/JSON 中。ΔLPIPS 为负代表改善。', '',
        'M 的独立作用由 M−Bsync 判断；X 由 X−Bsync 判断。组合由 MX−M、MX−X 与交互差判断。',
        '交互差为 (MX−M)−(X−Bsync)。LPIPS 使用负 LPIPS 解释收益方向，不因组合均值较高就宣布互补。', '',
        '| 交互收益方向 | PSNR | SSIM | 负 LPIPS |', '|---|---:|---:|---:|']
    q = report['interaction']['overall']['benefit_delta']; lines.append(f"| I | {q['psnr']:+.6f} | {q['ssim']:+.8f} | {q['negative_lpips']:+.8f} |")
    lines += ['', '低频是否修复必须看绝对 low_mse 的减少，同时披露 cam00 动态纹理、cam01 背景以及高频误差。低频占比下降本身不证明修复。', '',
        '| 配置/相机 | 总 MSE | 低频 MSE | 低频占比 % | 高频 MSE |', '|---|---:|---:|---:|---:|']
    for method in METHODS:
        for camera in ('cam00', 'cam01'):
            q = report['metrics'][method]['FFT_by_camera'][camera]
            lines.append(f"| {method}/{camera} | {q['mse']:.8g} | {q['low_mse']:.8g} | {q['low_share_pct']:.4f} | {q['high_mse']:.8g} |")
    lines += ['', 'X 的支持面积、残差、梯度与矩厚度来自相同固定 train76，即使某方法训练时未使用 X，也检查同一个诊断目标。'
        '结构统计分别报告均衡76个 anchor 和含重复投影的228观察平均；矩分散不是真实几何精度。', '',
        '成本包含成功训练、评价、校准、固定诊断，以及已登记的失败准备/算子检查/丢弃训练工作。'
        '阶段耗时之和不是多卡总历时，train_s 是 wall 的子集；缺项写未知。', '',
        'Pareto 候选：'+', '.join(report['Pareto']['front'])+'。选择状态：'+report['recommendation']['status']+'。',
        report['recommendation']['rationale'], '',
        '本文件只描述 Cook 全画面短窗开发。正式完整场景推荐需结合固定图像复核，并推进从头独立前缀、完整时间开发及未用于选择的确认场景。'
        '没有完整数据/训练成果时，不能把数据清单或计划写成完整基准已完成。', '', '限制：']
    lines += ['- '+value for value in report['limitations']]
    return '\n'.join(lines)+'\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=OUT/'summary')
    parser.add_argument('--decision', type=Path)
    args = parser.parse_args(); report, endpoints = build(args.decision)
    args.out.mkdir(parents=True, exist_ok=True)
    write(args.out/'summary.json', report)
    (args.out/'summary.md').write_text(readable(report))
    csv_write(args.out/'cost_ledger.csv', report['cost']['entries'])
    if report['status'] == 'completed_core_quality_evidence':
        csv_write(args.out/'quality_by_suffix_camera.csv', [dict(method=row['method'], repeat=row['repeat'], camera=camera, **values)
            for row in endpoints.values() for camera, values in dict(overall=row['overall'], **row['by_camera']).items()])
        csv_write(args.out/'quality_method_means.csv', [dict(method=method, **row['overall']) for method, row in report['metrics'].items()])
        csv_write(args.out/'metrics_per_observation.csv', [value for row in endpoints.values() for value in row['_rows']])
        csv_write(args.out/'FFT_per_observation.csv', [value for row in endpoints.values() for value in row['_frequency_rows']])
        csv_write(args.out/'ROI_per_observation.csv', [value for row in endpoints.values() for value in row['_region_rows']])
        csv_write(args.out/'paired_per_observation.csv', [dict(comparison=name, repeat=row['repeat'], camera=row['camera'], frame=row['frame'],
            **{key+'_delta': value for key, value in row['raw_delta'].items()},
            negative_lpips_benefit=row['benefit_delta']['negative_lpips'])
            for name, contrast in report['paired'].items() for row in contrast['per_observation']])
        csv_write(args.out/'module_interaction.csv', [dict(scope='overall', **report['interaction']['overall']['raw_delta'],
            negative_lpips_benefit=report['interaction']['overall']['benefit_delta']['negative_lpips'])]+
            [dict(scope='suffix_'+repeat, **row['raw_delta'], negative_lpips_benefit=row['benefit_delta']['negative_lpips']) for repeat, row in report['interaction']['by_suffix'].items()]+
            [dict(scope=camera, **row['raw_delta'], negative_lpips_benefit=row['benefit_delta']['negative_lpips']) for camera, row in report['interaction']['by_camera'].items()])
    write(args.out/'complete.json', dict(status=report['status'], summary=entry(args.out/'summary.json'),
        readable=entry(args.out/'summary.md'), cost_ledger=entry(args.out/'cost_ledger.csv'),
        actual_identity_checked_endpoints=report['completed_identity_checked_endpoints'],
        source=entry(Path(__file__)), formal_updates=0, GPU_calls=0,
        no_quality_recommendation_while_pending=report['recommendation'] is None))
    print(json.dumps(dict(status=report['status'], endpoints=report['completed_identity_checked_endpoints'],
        recommendation=report['recommendation'], output=str(args.out)), ensure_ascii=False), flush=True)


if __name__ == '__main__': main()
