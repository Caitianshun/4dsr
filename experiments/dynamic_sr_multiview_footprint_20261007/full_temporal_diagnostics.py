"""CPU residual-change diagnostics of an already completed native full evaluation.

No render, optimizer, flow, ROI creation, or method selection occurs here.  This
entry deliberately refuses partial evaluations and pre-freeze confirmations.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import time

from fp_common import ROOT, HERE, entry, local, sha, write

SCHEMA = 'registered_full_native_temporal_residual_diagnostic_v1'
EVAL_STATUS = 'completed_full_native_test300_zero_update_evaluation'
SR_SCHEMA = 'registered_full_native_SR_refinement_v1'
PARENT_SCHEMA = 'full_author_LR_prefix_checkpoint_v1'
PROTOCOL_SCHEMA = 'registered_full_native_evaluation_protocol_v1'
MAIN_FREEZE_STATUS = 'root_frozen_single_main_configuration_before_confirmation_HR'
DEVELOPMENT = ('cook_spinach', 'cut_roasted_beef', 'meetroom_discussion', 'meetroom_vrheadset')
CONFIRMATION = ('coffee_martini', 'flame_steak')
SEEDS = (20261007, 20261008)
GROUPS = ('xyz', 'deformation', 'grid', 'f_dc', 'f_rest', 'opacity', 'scaling', 'rotation')
DEFINITIONS = dict(
    primary='e_t = float32(clamp(raw_render_t,0,1)-HR_GT_t); mean((float64(e_t)-float64(e_t_minus_1))**2) over all RGB pixels, then equal-pair mean over299 consecutive pairs.',
    secondary='Unclamped raw_render residual change and per-frame clamped/raw spatial residual MSE are separately named. They do not replace the official clamped diagnostic.',
    precision='Raw NPZ float32 CHW -> HWC; GT RGB uint8 converted float32/255 exactly as the frozen evaluator. Residuals are float32; squaring and averaging use float64.',
    interpretation='Measures changes in reconstruction error at the same image pixels, including flicker and bias in real scene changes. Without flow/correspondence it is not a geometric, perceptual, or motion-compensated temporal stability proof.',
    statistics='Frames and adjacent pairs are correlated. Their standard deviation is descriptive, not an independent-sample confidence interval. No ROI, masks or flow are created.',
)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def safe_path(root, name):
    root = Path(root).resolve()
    path = Path(name)
    path = path if path.is_absolute() else root / path
    require(not path.is_symlink() and path.resolve().is_relative_to(root), 'Bound path escaped its root or is a symlink')
    require(path.is_file(), 'Missing bound file: ' + str(path))
    return path.resolve()


def read_bound(root, reference, json_value=True):
    require(isinstance(reference, dict) and isinstance(reference.get('path'), str) and isinstance(reference.get('sha256'), str), 'Exact path/SHA reference required')
    path = safe_path(root, reference['path'])
    require(sha(path) == reference['sha256'], 'Bound input SHA changed: ' + str(path))
    return read(path) if json_value else path


def closed_evaluation(value):
    require(value.get('status') == EVAL_STATUS and value.get('test_observations') == 300, 'Completed native test300 evaluation required')
    require(all(value.get(k) == 0 for k in ('formal_updates', 'Adam_calls', 'backward_calls')) and value.get('model_RNG_unchanged') is True, 'Evaluation update/RNG boundary differs')


def final_freeze_gate(freeze, selection, selection_ref, plan, development):
    """Same final-main API as the independent full-stage materializer.

    Files behind these metadata objects are bound by load_inputs.  No external
    service state or a two-candidate development decision substitutes for this.
    """
    require(freeze is not None and freeze.get('status') == MAIN_FREEZE_STATUS, 'Held confirmation needs a final single-main freeze')
    require(selection.get('status') == 'registered_full_development_selection' and selection.get('uses_confirmation_for_selection') is False, 'Legal frozen development selection required')
    candidates = selection.get('selected_candidates', [])
    require(1 <= len(candidates) <= 2 and len(set(candidates)) == len(candidates) and set(candidates) <= {'M', 'X', 'MX', 'E'}, 'Invalid development candidate set')
    require(freeze.get('selection') == selection_ref and freeze.get('main_method') in selection.get('selected_candidates', []), 'Frozen main method/selection identity differs')
    require(freeze.get('main_method') in ('M', 'X', 'MX', 'E') and selection.get('baseline_method') in ('B0', 'Bsync'), 'Invalid frozen main/baseline')
    require(freeze.get('configuration_frozen_before_confirmation_HR') is True and freeze.get('confirmation_HR_used_for_selection') is False and freeze.get('frozen_configuration'), 'Confirmation configuration was not frozen before HR use')
    require(freeze.get('development_results_complete') is True and freeze.get('confirmation_scenes') == list(CONFIRMATION), 'Full development/registered confirmation boundary differs')
    expected = {s + '/' + str(z) for s in DEVELOPMENT for z in SEEDS}
    require(set(freeze.get('full_development_evidence', {})) == expected and set(development) == expected, 'All four development scenes/two seeds required before confirmation')
    platforms = []
    for key, bundle in development.items():
        closed, dep_plan, summary = bundle
        closed_evaluation(closed)
        scene, seed = key.split('/')
        cp = closed['identity']['checkpoint']
        require(cp['schema'] == SR_SCHEMA and cp['metadata']['cursor'] == 6000 and dep_plan['scene'] == scene and dep_plan['seed'] == int(seed) and dep_plan['method'] == freeze['main_method'], 'Development evidence is not the final main-method endpoint')
        require(summary.get('identity') == closed['identity'] and summary.get('status') == EVAL_STATUS and summary.get('model_immutability', {}).get('equal') is True, 'Development summary identity/immutability differs')
        h = summary['hardware']
        platforms.append((h['physical_GPU'], h['host'], h['torch'], h['cuda'], json.dumps(closed['identity']['sources'], sort_keys=True)))
    require(len(set(platforms)) == 1, 'Full development evidence is not from a uniform evaluator/source platform')
    require(plan.get('method') in (freeze['main_method'], selection['baseline_method']), 'Confirmation checkpoint is outside the frozen main/baseline set')
    require(plan.get('schema') == SR_SCHEMA, 'Held diagnostic supports only the frozen SR main/baseline endpoints')


def confirmation_authorization_gate(auth, launch, auth_ref, freeze_ref, complete_ref, identity):
    """Bind a before-spawn authorization to the actual returned evaluation.

    The native evaluator itself creates registration.json in an empty output.
    The operator therefore records its exact expected dict before launch, then
    binds the actual registration after its child exits. These are operational
    evidence contracts, not claimed production evidence from fake CPU checks.
    """
    require(auth is not None and launch is not None and auth.get('status') == 'root_authorized_frozen_main_before_confirmation_evaluation', 'Held diagnostic needs actual pre-evaluation authorization and launch proof')
    require(auth.get('final_main_freeze') == freeze_ref and auth.get('expected_evaluation_registration') == identity and auth.get('protocol') == identity['protocol'] and auth.get('checkpoint') == identity['checkpoint']['checkpoint'], 'Confirmation authorization identity differs')
    require(auth.get('HR_reads_before_authorization') == 0 and auth.get('authorized_before_prediction_and_HR') is True, 'Confirmation preauthorization read boundary differs')
    require(launch.get('status') == 'completed_authorized_confirmation_evaluation_child_Exit0' and launch.get('preauthorization') == auth_ref and launch.get('final_main_freeze') == freeze_ref and launch.get('evaluation_complete') == complete_ref, 'Launch proof is not this authorized completed evaluation')
    for key in ('backend_source', 'manager_spec', 'manager_identity'):
        require(auth.get(key) and launch.get(key) == auth[key], 'Confirmation manager/source/spec identity differs: ' + key)
    manager = auth['manager_identity']
    require(set(('unit', 'invocation_id', 'pid', 'start_ticks', 'start_monotonic', 'host', 'boot_id')) <= set(manager) and str(manager['unit']).endswith('.service') and len(str(manager['invocation_id'])) == 32 and all(int(manager[k]) > 0 for k in ('pid', 'start_ticks', 'start_monotonic')) and manager['host'] and manager['boot_id'], 'Exact owned persistent manager identity required')
    require(manager['host'] == identity['host'] and launch.get('child_host') == manager['host'] and launch.get('child_boot_id') == manager['boot_id'], 'Monotonic launch proof must be from the actual evaluator host and same boot')
    times = [auth.get('authorization_written_monotonic_ns'), launch.get('child_spawn_monotonic_ns'), launch.get('child_exit_monotonic_ns')]
    require(all(type(x) is int and x > 0 for x in times) and times[0] < times[1] < times[2], 'Actual same-manager write/spawn/wait order not proved')
    require(launch.get('authorization_written_monotonic_ns') == times[0] and launch.get('wait_authority') == 'own_subprocess_wait_return' and launch.get('child_exit_code') == 0, 'Actual owned child Exit0 required')
    require(int(launch.get('child_pid', 0)) > 0 and int(launch.get('child_start_ticks', 0)) > 0 and isinstance(auth.get('evaluation_command'), list) and len(auth['evaluation_command']) > 3 and launch.get('child_argv') == auth['evaluation_command'], 'Native child PID/start/exact argv differs')
    source_key = str((HERE/'full_evaluate_registered.py').relative_to(ROOT))
    expected_sha = identity['sources'].get(source_key)
    require(isinstance(expected_sha, str) and len(expected_sha) == 64 and auth.get('evaluator_source') == dict(path=source_key, sha256=expected_sha), 'Confirmation evaluator source differs')


def validate_metadata(complete, summary, index, manifest, protocol, plan, sidecar,
                      freeze=None, selection=None, development=None):
    """Pure metadata validation, also used by the small fake CPU contracts."""
    closed_evaluation(complete)
    identity = complete['identity']
    cp = identity['checkpoint']
    require(summary.get('identity') == identity and index.get('identity') == identity, 'Evaluation/summary/float identity differs')
    require(summary.get('status') == EVAL_STATUS and summary.get('model_immutability', {}).get('equal') is True, 'Summary is not a read-only completed evaluation')
    require(index.get('status') == 'completed_full_native_float_observations' and index.get('test_observations') == 300 and index.get('training_cache') is False, 'Full float index incomplete or training cache')
    require(protocol.get('schema') == PROTOCOL_SCHEMA and protocol.get('status') == 'registered_full_native_evaluation_before_prediction_reads', 'Registered full evaluation protocol required')
    scene = manifest['scene']; role = manifest.get('role')
    expected_role = 'already_used_development' if scene in DEVELOPMENT else 'held_confirmation' if scene in CONFIRMATION else None
    require(expected_role is not None and role == expected_role and protocol.get('role') == role, 'Unknown scene or wrong development/confirmation role')
    require(manifest.get('official_split') is True and manifest.get('additional_cam01_dev_holdout') is False and manifest.get('full_time_decode_verified') is True, 'Full official-derived split/decode metadata required')
    require(manifest['frame_indices'] == list(range(300)) and manifest['splits']['test'] == ['cam00'] and not manifest['splits']['dev'] and 'cam01' in manifest['splits']['train'], 'Exact full300 cam00 test split required; cam01 is training')
    require(plan['scene'] == protocol['scene'] == scene and plan['seed'] == protocol['seed'] == manifest['initialization']['seed'] and plan['seed'] in SEEDS, 'Scene/seed identity differs')
    require(plan['data']['manifest'] == protocol['manifest'] == identity['manifest'], 'Manifest identity differs')
    require(protocol['data'] == plan['data'] and protocol['sources'] == identity['sources'], 'Registered data/source identity differs')
    require(protocol['camera_interface_SHA256'] == digest(manifest['cameras']), 'Camera metadata identity differs')
    require(summary['hardware']['physical_GPU'] == identity['GPU'] and summary['hardware']['host'] == identity['host'], 'Evaluation hardware identity differs')
    require(all(summary['cost'].get(k) == 0 for k in ('Adam_calls', 'backward_calls', 'formal_updates')), 'Summary update cost differs')
    require(all(summary['native_restore'].get(k) is True for k in ('one_Adam_exact', 'parameters_exact', 'topology_buffers_exact', 'saved_gradients_exact', 'global_RNG_exact', 'no_children')) and summary['native_restore'].get('optimizer_conversion') is False, 'Native restore evidence incomplete')
    require(sidecar.get('schema') == cp['schema'] and sidecar.get('metadata') == cp['metadata'] and {k: sidecar[k] for k in ('path', 'sha256')} == cp['checkpoint'], 'Checkpoint sidecar identity differs')
    require(cp['schema'] in (SR_SCHEMA, PARENT_SCHEMA) and tuple(x['name'] for x in cp['metadata']['audit']['optimizer_groups']) == GROUPS, 'Native14/oneAdam8 metadata required')
    require(cp['metadata']['plan_sha256'] == digest(plan), 'Checkpoint plan digest differs')
    if cp['schema'] == SR_SCHEMA:
        require(plan['schema'] == SR_SCHEMA and plan['status'] == 'registered_ready_full_native_SR_refinement' and cp['metadata'].get('cursor') == 6000 and cp['metadata'].get('density_statistical_boundary_applied') is True and cp['metadata'].get('terminal_zero_grad') is True, 'Only final native SR6000 accepted')
    else:
        state = cp['metadata']['state']
        batch = plan['configuration']['OptimizationParams']['batch_size']
        require(batch in (2, 4) and state['coarse_iteration'] == 3000 and state['fine_iteration'] == 14000 and state['accepted_backward'] == 17000 and state['accepted_Adam'] == 16999 and state['accepted_RGB'] == 17000*batch, 'Only complete native LR parent allowed')
    require(protocol['test_keys'] == [['cam00', f] for f in range(300)] and protocol['measurement_definition_SHA256'] == digest(protocol['measurement_definitions']), 'Test keys/measurement definition differs')
    rows = sorted((r for r in manifest['observations'] if r['split'] == 'test'), key=lambda r: int(r['frame_index']))
    require(len(rows) == 300 and [(r['camera_id'], r['frame_index']) for r in rows] == [('cam00', f) for f in range(300)], 'Missing/duplicate/wrong test frames')
    require(all(abs(float(r['time']) - r['frame_index'] / 300) < 1e-7 for r in rows), 'Wrong full frame time')
    require(protocol['test_input_SHA256'] == digest([(r['camera_id'], r['frame_index'], r['hr_sha256'], r['lr_sha256']) for r in rows]), 'Test reference identity differs')
    floats = sorted((r for r in index['entries'] if r['scope'] == 'full_test'), key=lambda r: int(r['frame']))
    require(len(floats) == 300 and [(r['camera'], r['frame']) for r in floats] == [('cam00', f) for f in range(300)], 'Missing/duplicate/wrong float frames')
    require(all(r.get('raw_dtype') == 'float32' and r.get('training_cache') is False and r.get('checkpoint_sha256') == cp['checkpoint']['sha256'] for r in floats), 'Wrong raw dtype/cache/checkpoint identity')
    require(len({r['path'] for r in floats}) == 300 and all(r['path'] == f"floats/cam00_{r['frame']:04d}.npz" for r in floats), 'Wrong/duplicate float path')
    require(all(r['hr_path'] == f"hr/cam00/{r['frame_index']:04d}.png" for r in rows), 'Unregistered HR reference path')
    expected_hr = [1280, 720] if scene.startswith('meetroom_') else [1344, 1008]
    require(manifest['resolutions']['hr'] == protocol['resolutions']['hr'] == expected_hr and manifest['resolutions']['lr'] == [x // 4 for x in expected_hr], 'Wrong native grid')
    if role == 'held_confirmation':
        final_freeze_gate(freeze, selection or {}, protocol.get('selection'), plan, development or {})
    return rows, floats


def load_inputs(complete_path, manifest_path, final_freeze_path=None,
                authorization_path=None, launch_proof_path=None, root=ROOT):
    """Validate closed metadata and every selected original float SHA before GT."""
    complete_path = safe_path(root, complete_path); manifest_path = safe_path(root, manifest_path)
    complete = read(complete_path); closed_evaluation(complete); identity = complete['identity']
    summary = read_bound(root, complete['summary'])
    required = ('metrics_per_observation.csv', 'FFT_absolute_band_error.csv', 'fixed_ROI_quality.csv', 'train_teacher_diagnostics.csv', 'float_index.json', 'model_before.json', 'model_after.json', 'metric_runtime.json')
    require(set(complete['results']) == set(required), 'Completed evaluation result references incomplete')
    for name in required:
        read_bound(root, complete['results'][name], json_value=False)
    require(read_bound(root, complete['results']['model_before.json']) == read_bound(root, complete['results']['model_after.json']), 'Native model/Adam/gradients/buffers/RNG changed')
    require(summary['model_immutability']['before'] == complete['results']['model_before.json'] and summary['model_immutability']['after'] == complete['results']['model_after.json'] and summary['metric_runtime'] == complete['results']['metric_runtime.json'], 'Summary evidence references differ')
    with read_bound(root, complete['results']['metrics_per_observation.csv'], json_value=False).open(newline='') as f:
        metrics = list(csv.DictReader(f))
    require(len(metrics) == 300 and [(r['camera'], int(r['frame'])) for r in metrics] == [('cam00', f) for f in range(300)], 'Completed metric CSV is not full ordered test300')
    index = read_bound(root, complete['results']['float_index.json'])
    require(entry(manifest_path) == identity['manifest'], 'Explicit manifest differs from evaluation')
    manifest = read(manifest_path); protocol = read_bound(root, identity['protocol']); cp = identity['checkpoint']
    plan = read_bound(root, cp['plan']); sidecar = read_bound(root, cp['sidecar'])
    require(identity['sources'].get(str((HERE/'full_evaluate_registered.py').relative_to(ROOT))) == sha(HERE/'full_evaluate_registered.py'), 'Evaluation source differs from the frozen registered evaluator')
    require(protocol['sources'] == identity['sources'], 'Evaluation protocol source identity differs')
    for name, expected in identity['sources'].items():
        require(sha(safe_path(root, name)) == expected, 'Frozen evaluation dependency source changed')
    freeze = selection = None; development = {}; freeze_ref = None
    if final_freeze_path is not None:
        path = safe_path(root, final_freeze_path); freeze = read(path); freeze_ref = entry(path)
        selection = read_bound(root, freeze['selection']); read_bound(root, freeze['frozen_configuration'], json_value=False)
        for key, reference in freeze.get('full_development_evidence', {}).items():
            closed = read_bound(root, reference)
            development[key] = (closed, read_bound(root, closed['identity']['checkpoint']['plan']), read_bound(root, closed['summary']))
    rows, floats = validate_metadata(complete, summary, index, manifest, protocol, plan, sidecar, freeze, selection, development)
    auth_ref = launch_ref = None
    if manifest['role'] == 'held_confirmation':
        require(authorization_path is not None and launch_proof_path is not None, 'Confirmation may not add a final freeze after quality was evaluated')
        path = safe_path(root, authorization_path); auth = read(path); auth_ref = entry(path)
        path = safe_path(root, launch_proof_path); launch = read(path); launch_ref = entry(path)
        confirmation_authorization_gate(auth, launch, auth_ref, freeze_ref, entry(complete_path), identity)
        require(read_bound(root, launch['actual_evaluation_registration']) == identity, 'Actual evaluator registration differs from its authorized expected dict')
        for key in ('backend_source', 'manager_spec', 'evaluator_source'):
            read_bound(root, auth[key], json_value=False)
        spec = read_bound(root, auth['manager_spec'])
        require(all(spec.get(key) == auth[key] for key in ('backend_source', 'evaluation_command', 'expected_evaluation_registration', 'final_main_freeze')), 'Actual manager spec does not bind the authorized evaluation')
    float_paths = []
    for item in floats:
        path = safe_path(complete_path.parent, item['path'])
        require(sha(path) == item['sha256'], 'Original full-test float SHA changed')
        require(read(path.with_suffix('.json')) == item, 'Original float sidecar differs')
        float_paths.append(path)
    return dict(complete=complete, complete_ref=entry(complete_path), manifest=manifest,
                manifest_ref=entry(manifest_path), manifest_root=manifest_path.parent,
                summary_ref=complete['summary'], float_index_ref=complete['results']['float_index.json'],
                protocol_ref=identity['protocol'], freeze_ref=freeze_ref, authorization_ref=auth_ref,
                launch_proof_ref=launch_ref, rows=rows,
                floats=floats, float_paths=float_paths, all_selected_float_SHA_verified=300)


def validate_arrays(raw, gt):
    import numpy as np
    require(isinstance(raw, np.ndarray) and isinstance(gt, np.ndarray) and raw.dtype == gt.dtype == np.float32, 'float32 RGB arrays required')
    require(raw.ndim == 3 and raw.shape == gt.shape and raw.shape[-1] == 3 and all(x > 0 for x in raw.shape), 'Finite same-grid HWC RGB shape required')
    require(bool(np.isfinite(raw).all()) and bool(np.isfinite(gt).all()) and bool(((gt >= 0) & (gt <= 1)).all()), 'Nonfinite render or invalid normalized GT')


def residuals(raw, gt):
    import numpy as np
    validate_arrays(raw, gt)
    return np.clip(raw, 0, 1) - gt, raw - gt


def frame_errors(raw, gt):
    import numpy as np
    clamped, unclamped = residuals(raw, gt)
    return dict(clamped_spatial_residual_mse=float(np.square(clamped.astype(np.float64)).mean()),
                raw_spatial_residual_mse=float(np.square(unclamped.astype(np.float64)).mean()),
                raw_below_zero_fraction=float((raw < 0).mean()), raw_above_one_fraction=float((raw > 1).mean()))


def adjacent_errors(previous_raw, current_raw, previous_gt, current_gt):
    import numpy as np
    previous = residuals(previous_raw, previous_gt); current = residuals(current_raw, current_gt)
    require(previous_raw.shape == current_raw.shape, 'Adjacent frame shape changed')
    return dict(clamped_temporal_residual_change_mse=float(np.square(current[0].astype(np.float64) - previous[0].astype(np.float64)).mean()),
                raw_temporal_residual_change_mse=float(np.square(current[1].astype(np.float64) - previous[1].astype(np.float64)).mean()))


def read_frame(bundle, number):
    import numpy as np
    from PIL import Image
    item = bundle['floats'][number]; blob = bundle['float_paths'][number].read_bytes()
    require(hashlib.sha256(blob).hexdigest() == item['sha256'], 'Float changed after initial validation')
    with np.load(io.BytesIO(blob), allow_pickle=False) as archive:
        require(archive.files == ['rgb_raw'], 'Unknown float archive members')
        raw = archive['rgb_raw']
    width, height = bundle['manifest']['resolutions']['hr']
    require(raw.dtype == np.float32 and raw.shape == (3, height, width), 'Float payload dtype/grid differs')
    require(bool(np.isfinite(raw).all()), 'Nonfinite float render refused before GT decode')
    row = bundle['rows'][number]; path = safe_path(bundle['manifest_root'], row['hr_path']); blob = path.read_bytes()
    require(hashlib.sha256(blob).hexdigest() == row['hr_sha256'], 'GT reference bytes changed')
    with Image.open(io.BytesIO(blob)) as image:
        require(image.size == (width, height), 'GT reference grid differs')
        gt = np.asarray(image.convert('RGB')).astype(np.float32) / 255.0
    raw = raw.transpose(1, 2, 0); validate_arrays(raw, gt)
    return raw, gt


def csvwrite(path, rows):
    with Path(path).open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def diagnose(a):
    import socket
    started = time.monotonic(); out = local(a.out)
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Run this CPU-only postprocess with CUDA_VISIBLE_DEVICES empty')
    require(out.resolve().is_relative_to((ROOT/'output'/HERE.name/'full_temporal_diagnostics').resolve()), 'Separate full_temporal_diagnostics output subtree required')
    require(not out.exists(), 'Immutable output/partial history exists; choose a new output directory')
    def inputs():
        return load_inputs(local(a.evaluation_complete), local(a.manifest), local(a.final_main_freeze) if a.final_main_freeze else None,
                           local(a.confirmation_authorization) if a.confirmation_authorization else None,
                           local(a.confirmation_launch_proof) if a.confirmation_launch_proof else None)
    bundle = inputs()
    validation_seconds = time.monotonic() - started
    out.mkdir(parents=True, exist_ok=False)
    source = entry(Path(__file__)); common_source = entry(HERE/'fp_common.py')
    identity = dict(evaluation_complete=bundle['complete_ref'], manifest=bundle['manifest_ref'], summary=bundle['summary_ref'], float_index=bundle['float_index_ref'], protocol=bundle['protocol_ref'], final_main_freeze=bundle['freeze_ref'], confirmation_authorization=bundle['authorization_ref'], confirmation_launch_proof=bundle['launch_proof_ref'], source=source, common_source=common_source)
    registration = dict(schema=SCHEMA, status='registered_CPU_diagnostic_after_full_native_evaluation', identity=identity, measurement_definitions=DEFINITIONS, measurement_definition_sha256=digest(DEFINITIONS), original_float_SHA_verified=300, selected_observations=[['cam00', f] for f in range(300)], training_reads=0, LR_reads=0, method_selection_performed=False)
    write(out/'registration.json', registration)
    counts = dict(float_payload_reads=0, GT_reads=0, diagnostic_pairs=0, formal_updates=0, Adam_calls=0, model_forwards=0, GPU_calls=0)
    try:
        import numpy as np
        from PIL import __version__ as pillow_version
        pairs = []; spatial = []; previous = None
        for number in range(300):
            raw, gt = read_frame(bundle, number); counts['float_payload_reads'] += 1; counts['GT_reads'] += 1
            spatial.append(dict(camera='cam00', frame=number, **frame_errors(raw, gt)))
            if previous is not None:
                pairs.append(dict(camera='cam00', previous_frame=number-1, frame=number, **adjacent_errors(previous[0], raw, previous[1], gt))); counts['diagnostic_pairs'] += 1
            previous = (raw, gt)
        require(sha(Path(__file__)) == source['sha256'] and sha(HERE/'fp_common.py') == common_source['sha256'], 'Diagnostic source changed')
        # Revalidate frozen small identities and every float; GT also remains
        # SHA-checked when consumed. No checkpoint tensor or training image opens.
        after = inputs()
        require(all(after[k] == bundle[k] for k in ('complete_ref', 'manifest_ref', 'freeze_ref', 'authorization_ref', 'launch_proof_ref')), 'Diagnostic bound identities changed')
        csvwrite(out/'per_adjacent_frame.csv', pairs); csvwrite(out/'spatial_frame_mse.csv', spatial)
        def aggregate(rows, key):
            values = np.asarray([r[key] for r in rows], dtype=np.float64)
            return dict(equal_observation_mean=float(values.mean()), descriptive_population_std=float(values.std()), minimum=float(values.min()), maximum=float(values.max()))
        summary = dict(schema=SCHEMA, status='completed_registered_full_native_temporal_residual_diagnostic', identity=identity, scene=bundle['manifest']['scene'], seed=bundle['manifest']['initialization']['seed'], role=bundle['manifest']['role'], camera='cam00', observations=300, adjacent_pairs=299,
                       primary_clamped_temporal_residual_change_mse=aggregate(pairs,'clamped_temporal_residual_change_mse'), secondary_raw_temporal_residual_change_mse=aggregate(pairs,'raw_temporal_residual_change_mse'), clamped_spatial_residual_mse=aggregate(spatial,'clamped_spatial_residual_mse'), raw_spatial_residual_mse=aggregate(spatial,'raw_spatial_residual_mse'), measurement_definitions=DEFINITIONS,
                       input_closure='Original completed evaluation references and all300 selected float SHA verified before and after processing; eachGT SHA verified when consumed.', no_geometric_or_perceptual_temporal_stability_claim=True, correlated_frames_not_independent_samples=True, method_selection_performed=False, cost=dict(counts, original_float_SHA_validation_passes=2, original_float_bytes_per_pass=sum(p.stat().st_size for p in bundle['float_paths']), CPU_validation_seconds=validation_seconds, total_wall_seconds=time.monotonic()-started, host=socket.gethostname(), numpy=np.__version__, pillow=pillow_version))
        write(out/'summary.json',summary)
        write(out/'complete.json',dict(schema=SCHEMA,status=summary['status'],identity=identity,observations=300,adjacent_pairs=299,formal_updates=0,Adam_calls=0,model_forwards=0,GPU_calls=0,summary=entry(out/'summary.json'),results={n:entry(out/n) for n in ('registration.json','per_adjacent_frame.csv','spatial_frame_mse.csv')}))
        return summary
    except BaseException as exc:
        write(out/'failure.json',dict(schema=SCHEMA,status='failed_CPU_temporal_diagnostic_partial_preserved',identity=identity,error=repr(exc),cost=dict(counts,total_wall_seconds=time.monotonic()-started),active_read_or_CPU_operation_cost_unknown=True))
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evaluation-complete',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--final-main-freeze',type=Path,help='Required for the two held confirmation scenes; development<=2 candidate selection alone is insufficient')
    p.add_argument('--confirmation-authorization',type=Path,help='Held only: root authorization written by the actual operator before evaluator launch')
    p.add_argument('--confirmation-launch-proof',type=Path,help='Held only: same-manager monotonic launch/wait and actual-registration proof')
    a=p.parse_args();print(json.dumps(diagnose(a),ensure_ascii=False,allow_nan=False))


if __name__ == '__main__':
    main()
