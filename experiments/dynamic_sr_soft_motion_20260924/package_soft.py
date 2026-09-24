#!/usr/bin/env python3
"""Package declared completed soft-motion scenes on CPU, without loading models.

Each ROOT/scene_methods/*.json declares absolute manifest, methods_file,
summary_dir and views_dir paths. No model tensors are deserialized. Weights,
motion/selection caches, and complete HR/LR/prediction sequences are excluded.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import zipfile

PROJECT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RECORDS = {'.json', '.jsonl', '.log', '.txt', '.md', '.csv', '.py', '.sh'}
IMAGES = {'.png', '.jpg', '.jpeg', '.svg'}
VIDEOS = {'.mp4', '.webm', '.mov', '.avi'}
WEIGHTS = {'.pt', '.pth', '.ckpt', '.safetensors', '.npy', '.npz'}
REQUIRED_EVALUATIONS = ('train_fixed_6000', 'dev_6000', 'test_6000')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def encoded(value):
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode()


def read(path):
    return json.loads(Path(path).read_text())


def local_path(value, base):
    path = Path(value).expanduser()
    return path.absolute() if path.is_absolute() else (base / path).absolute()


class Hashes:
    def __init__(self):
        self.cache = {}

    def sha(self, path):
        real = Path(path).resolve(strict=True)
        before = real.stat()
        key = (str(real), before.st_size, before.st_mtime_ns)
        if key not in self.cache:
            h = hashlib.sha256()
            with real.open('rb') as stream:
                for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
                    h.update(block)
            after = real.stat()
            require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
                    f'Source changed while hashing: {path}')
            self.cache[key] = h.hexdigest()
        return self.cache[key]


class Package:
    def __init__(self, root, report, media_limit):
        self.root, self.report = root, report
        self.hashes = Hashes()
        self.files, self.checkpoints, self.caches = {}, {}, {}
        self.parents_seen = set()
        self.scenes, self.media_exclusions, self.supplementary_checks = [], [], []
        self.media_candidates = {}
        self.media_bytes, self.media_limit = 0, media_limit
        self.external_references = []
        self.identity_candidates = {}
        self.path_resolutions, self.source_snapshots = [], []
        # Only identity records are inspected here; no tensors or quality
        # metrics are loaded to locate returned remote artifacts.
        inventory = root / 'asset_inventory.json'
        if inventory.is_file():
            self.register_identities(read(inventory), inventory.parent)
        for directory in sorted(root.glob('*reference*')):
            if directory.is_dir():
                for name in ('complete.json', 'reference_metadata.json', 'metadata.json', 'calibration.json'):
                    receipt = directory / name
                    if receipt.is_file():
                        self.register_identities(read(receipt), directory)
        for receipt in sorted(root.glob('reference*.json')):
            self.register_identities(read(receipt), receipt.parent)

    def register_identities(self, value, base):
        """Register possible local copies; their bytes are checked on use."""
        if isinstance(value, list):
            for child in value:
                self.register_identities(child, base)
        elif isinstance(value, dict):
            pairs = []
            for key in ('path', 'local_path', 'resolved_path', 'snapshot'):
                pairs.append((value.get(key), value.get('sha256')))
            for key, digest in value.items():
                if key.endswith('_sha256') and isinstance(digest, str):
                    stem = key[:-7]
                    for path_key in (stem, stem + '_path'):
                        pairs.append((value.get(path_key), digest))
                    # Reference receipts also name their adjacent artifacts
                    # implicitly. Never infer byte identity from the filename.
                    if stem in ('reference', 'selection', 'calibration'):
                        suffix = '.json' if stem == 'calibration' else '.pt'
                        pairs.append((str(base / (stem + suffix)), digest))
            for raw, digest in pairs:
                if isinstance(raw, str) and isinstance(digest, str) and re.fullmatch(r'[0-9a-f]{64}', digest):
                    path = local_path(raw, base)
                    if path.is_file():
                        self.identity_candidates.setdefault(digest, set()).add(path)
            for child in value.values():
                if isinstance(child, (dict, list)):
                    self.register_identities(child, base)

    def resolve_source(self, recorded, expected=None, hints=(), kind='source'):
        """Resolve provenance to real local bytes, never merely rename a path.

        A missing remote artifact is nonfatal for packaging local results.
        Relocation always requires a recorded SHA256; unknowns are explicit.
        """
        recorded = str(recorded)
        original = local_path(recorded, PROJECT)
        candidates = [original]
        candidates.extend(Path(p).absolute() for p in hints)
        if expected:
            parts = original.parts
            for marker in ('data', 'output', 'experiments'):
                if marker in parts:
                    candidates.append(PROJECT.joinpath(*parts[parts.index(marker):]))
            candidates.extend(sorted(self.identity_candidates.get(expected, ())))
        seen, mismatches = set(), []
        for path in candidates:
            if str(path) in seen or not path.is_file():
                continue
            seen.add(str(path))
            if path != original and not expected:
                continue
            digest = self.hashes.sha(path)
            if expected and digest != expected:
                mismatches.append(str(path))
                continue
            row = {'kind': kind, 'recorded_path': recorded, 'local_path': str(path),
                   'expected_sha256': expected, 'sha256': digest, 'bytes': path.stat().st_size,
                   'status': 'resolved_sha256_verified' if expected else 'present_local_hashed_no_expected_identity',
                   'mismatching_candidates': mismatches}
            self.path_resolutions.append(row)
            self.identity_candidates.setdefault(digest, set()).add(path)
            return path
        self.path_resolutions.append({'kind': kind, 'recorded_path': recorded, 'local_path': None,
                                      'expected_sha256': expected, 'sha256': None, 'bytes': None,
                                      'status': 'unresolved', 'mismatching_candidates': mismatches,
                                      'reason': 'No local file with matching recorded SHA256' if expected else
                                                'Recorded file absent; relocation requires a known SHA256'})
        return None

    def name(self, path):
        path = Path(path).absolute()
        try:
            return path.relative_to(PROJECT).as_posix()
        except ValueError:
            # Preserve siblings' relative layout for source snapshots outside
            # this repository, without embedding an absolute archive path.
            key = hashlib.sha256(str(path.parent).encode()).hexdigest()[:12]
            return f'external/{key}/{path.name}'

    def add(self, path, category, required=True):
        path = Path(path).absolute()
        if not path.is_file():
            require(not required, f'Missing package source: {path}')
            return
        require(path.suffix.lower() not in WEIGHTS, f'Binary weight/cache copying prohibited: {path}')
        name = self.name(path)
        if name not in self.files:
            self.files[name] = {'path': path, 'source_sha256': self.hashes.sha(path), 'categories': []}
        if category not in self.files[name]['categories']:
            self.files[name]['categories'].append(category)

    def records(self, folder, category, recursive=False):
        folder = Path(folder)
        if not folder.is_dir():
            return
        for path in sorted(folder.rglob('*') if recursive else folder.iterdir()):
            if path.is_file() and path.suffix.lower() in RECORDS and '__pycache__' not in path.parts:
                self.add(path, category)

    def cache(self, recorded, expected=None, hints=(), kind='cache'):
        path = self.resolve_source(recorded, expected, hints, kind)
        if path is None:
            return None
        digest = self.hashes.sha(path)
        self.caches[(str(recorded), str(path))] = {'recorded_path': str(recorded), 'local_path': str(path),
                                               'path': str(path), 'sha256': digest, 'bytes': path.stat().st_size,
                                               'recorded_sha256': expected, 'status': 'local_bytes_verified'}
        return path

    def checkpoint(self, path, owner, role, config=None, done=None, expected=None, recorded=None):
        path = Path(path).absolute()
        require(path.is_file(), f'Checkpoint missing: {path}')
        config, done = config or {}, done or {}
        digest = self.hashes.sha(path)
        require(expected is None or digest == expected, f'Checkpoint identity changed: {path}')
        numeric = re.fullmatch(r'checkpoint_(\d+)\.pt', path.name)
        added = int(numeric[1]) if numeric and role == 'formal_branch' else None
        if role == 'formal_branch' and path.name == 'checkpoint_final.pt':
            added = done.get('intervention_step', done.get('parameter_updates'))
        if added is not None and 'scheduler_offset' in config:
            step, semantics = config['scheduler_offset'] + added, 'scheduler_offset + added updates'
        elif numeric:
            step, semantics = int(numeric[1]), 'filename stage-local step; no total-step inference'
        else:
            step, semantics = done.get('step'), 'owning completion receipt; unknown if null'
        item = self.checkpoints.setdefault(str(path), {
            'path': str(path), 'local_path': str(path), 'recorded_path': str(recorded or path),
            'recorded_paths': [], 'resolved_path': str(path.resolve()), 'sha256': digest,
            'bytes': path.stat().st_size, 'symlink': path.is_symlink(),
            'kind': 'final' if path.name == 'checkpoint_final.pt' else 'milestone_or_referenced_checkpoint',
            'step': step, 'added_updates': added, 'step_semantics': semantics,
            'status': done.get('status', 'unknown'),
            'status_scope': 'Owning training receipt; existence and byte hash checked separately',
            'training_gpu': config.get('gpu', config.get('device', done.get('gpu'))),
            'training_gpu_uuid': config.get('visible_cuda', done.get('visible_cuda')),
            'training_host': config.get('host', config.get('hostname', done.get('host'))),
            'hardware_note': 'Saved training provenance only; null means unavailable, never replaced by evaluation hardware',
            'owners': []})
        if str(recorded or path) not in item['recorded_paths']:
            item['recorded_paths'].append(str(recorded or path))
        ownership = {'name': owner, 'role': role}
        if ownership not in item['owners']:
            item['owners'].append(ownership)

    def parent(self, recorded, expected=None):
        path = self.resolve_source(recorded, expected, kind='referenced_parent')
        if path is None:
            return
        folder = path.parent
        config = read(folder / 'config.json') if (folder / 'config.json').is_file() else {}
        done = read(folder / 'complete.json') if (folder / 'complete.json').is_file() else {}
        self.checkpoint(path, folder.name, 'referenced_parent', config, done, expected, recorded=recorded)
        if str(path) in self.parents_seen:
            return
        self.parents_seen.add(str(path))
        self.records(folder, 'referenced_parent_records')
        self.register_identities(config, folder)
        ancestor = config.get('checkpoint') or done.get('args', {}).get('checkpoint')
        if ancestor:
            self.parent(local_path(ancestor, PROJECT), config.get('parent_sha256') or done.get('parent_sha'))

    def assets(self, config):
        extra = config.get('extra_child_regularization', config.get('soft_motion', {}))
        for key in ('selection', 'reference'):
            if config.get(key):
                expected = config.get(key + '_sha256') or extra.get(key + '_sha256')
                path = self.cache(config[key], expected, kind=key)
                if path is None:
                    continue
                self.records(path.parent, key + '_receipts')
                if key == 'reference':
                    receipt_path = path.parent / 'complete.json'
                    receipt = read(receipt_path) if receipt_path.is_file() else {}
                    if receipt.get('status') == 'completed_reference_and_calibration':
                        require(receipt['reference_sha256'] == self.hashes.sha(path), 'Frozen reference hash mismatch')
                    else:
                        self.path_resolutions.append({'kind': 'reference_completion_receipt',
                            'recorded_path': str(Path(config[key]).parent / 'complete.json'), 'local_path': None,
                            'status': 'unresolved', 'reason': 'Local reference bytes verified; owning reference receipt unavailable'})
        if config.get('calibration'):
            path = self.resolve_source(config['calibration'], config.get('calibration_sha256') or
                                       extra.get('calibration_sha256'), kind='calibration')
            if path is not None:
                require(read(path).get('status') == 'calibrated', f'Incomplete calibration: {path}')
                self.add(path, 'calibration')
        for teacher in config.get('teacher_inputs', []):
            self.cache(teacher['path'], teacher['sha256'], kind='frozen_teacher')

    def snapshots(self, folder, config):
        for source in config.get('sources', []):
            recorded = source.get('snapshot')
            if not recorded:
                continue
            path = self.resolve_source(recorded, source.get('sha256'),
                                       hints=[folder / 'sources' / Path(recorded).name], kind='training_source_snapshot')
            self.source_snapshots.append({'recorded_source_path': source.get('path'),
                'recorded_snapshot_path': recorded, 'local_snapshot_path': str(path) if path else None,
                'recorded_sha256': source.get('sha256'), 'sha256': self.hashes.sha(path) if path else None,
                'status': 'resolved_sha256_verified' if path else 'unresolved'})
            if path is not None:
                self.add(path, 'training_source_snapshots')

    def method(self, scene, label, item, methods_file, manifest_hash):
        checkpoint = local_path(item['checkpoint'], methods_file.parent)
        folder = local_path(item.get('train_dir', str(checkpoint.parent)), methods_file.parent)
        config, done = read(folder / 'config.json'), read(folder / 'complete.json')
        require(done.get('status') == 'completed' and done.get('parameter_updates') == 6000
                and not done.get('smoke', False), f'Incomplete formal 6000-step training: {folder}')
        require(config['manifest_sha256'] == manifest_hash and config['seed'] == 20260923
                and config['steps'] == 6000 and config['sr_weight'] == .1, f'Wrong registered control: {folder}')
        self.register_identities(config, folder)
        self.records(folder, 'formal_training_records')
        self.records(folder / 'sources', 'training_source_snapshots', recursive=True)
        self.snapshots(folder, config)
        for path in sorted(folder.glob('checkpoint*.pt')):
            self.checkpoint(path, f'{scene}/{label}', 'formal_branch', config, done,
                            recorded=Path(config.get('out', folder)) / path.name)
        self.checkpoint(checkpoint, f'{scene}/{label}', 'formal_branch', config, done,
                        recorded=Path(config.get('out', folder)) / checkpoint.name)
        self.parent(local_path(config['checkpoint'], PROJECT), config['parent_sha256'])
        self.assets(config)
        evaluations = item.get('evaluations', {})
        if not evaluations:
            evaldir = local_path(item.get('evaldir', str(folder)), methods_file.parent)
            evaluations = {key: str(evaldir / ('eval_' + key)) for key in (*REQUIRED_EVALUATIONS, 'dev_1200')
                           if (evaldir / ('eval_' + key)).is_dir()}
        require(set(REQUIRED_EVALUATIONS) <= set(evaluations), f'{scene}/{label}: missing required evaluation')
        job_dirs = set()
        for endpoint, value in evaluations.items():
            location = local_path(value, methods_file.parent)
            location = location.parent if location.name == 'metrics.json' else location
            receipt = read(location / 'complete.json')
            require(receipt.get('status') == 'completed_evaluation' and receipt.get('parameter_updates') == 0,
                    f'Incomplete evaluation: {location}')
            require(receipt['metrics_sha256'] == self.hashes.sha(location / 'metrics.json'), f'Metrics changed: {location}')
            step = int(endpoint.rsplit('_', 1)[1])
            actual = checkpoint if step == 6000 else folder / f'checkpoint_{step}.pt'
            require(receipt['checkpoint_sha256'] == self.hashes.sha(actual), f'Evaluated checkpoint changed: {location}')
            for name in ('metrics.json', 'complete.json'):
                self.add(location / name, 'evaluation_records')
            if (location.parent / 'train_receipt.json').is_file():
                job_dirs.add(location.parent)
        if item.get('job_dir'):
            job_dirs.add(local_path(item['job_dir'], methods_file.parent))
        for job in job_dirs:
            require(read(job / 'complete.json').get('status') == 'completed_and_evaluated', f'Incomplete declared job: {job}')
            self.records(job, 'orchestration_logs_and_receipts')
            self.records(job / 'events', 'orchestration_events', recursive=True)
        cost = item.get('cost')
        if isinstance(cost, str):
            self.add(local_path(cost, methods_file.parent), 'preparation_and_run_cost')

    def scene(self, spec_file):
        spec = read(spec_file)
        paths = {}
        for key in ('manifest', 'methods_file', 'summary_dir', 'views_dir'):
            require(isinstance(spec.get(key), str) and Path(spec[key]).is_absolute(), f'{spec_file}: {key} must be absolute')
            paths[key] = Path(spec[key]).absolute()
        manifest, methods_file, summary, views = [paths[k] for k in ('manifest', 'methods_file', 'summary_dir', 'views_dir')]
        scene = read(manifest)['scene']
        summary_done, view_done = read(summary / 'complete.json'), read(views / 'complete.json')
        require(summary_done.get('status') == 'completed_summary', f'Incomplete summary: {summary}')
        require(summary_done['summary_sha256'] == self.hashes.sha(summary / 'summary.json'), 'Summary hash mismatch')
        require(view_done.get('status') == 'completed_fixed_views', f'Incomplete fixed views: {views}')
        require(view_done['views_sha256'] == self.hashes.sha(views / 'views.json'), 'Views hash mismatch')
        summary_info, views_info = read(summary / 'summary.json'), read(views / 'views.json')
        manifest_hash, methods_hash = self.hashes.sha(manifest), self.hashes.sha(methods_file)
        require(summary_info['manifest_sha256'] == manifest_hash and summary_info['methods_mapping_sha256'] == methods_hash,
                f'Summary describes different scene/method files: {scene}')
        require(views_info['methods_mapping_sha256'] == methods_hash, f'Views describe different methods: {scene}')
        require(summary_info['scene'] == views_info['scene'] == scene, 'Scene identity mismatch')
        raw = read(methods_file)
        methods = raw.get('methods', raw)
        require(isinstance(methods, dict) and methods, 'Empty method mapping')
        for name, item in methods.items():
            self.method(scene, name, item, methods_file, manifest_hash)
        for path, category in ((spec_file, 'scene_registry'), (manifest, 'data_manifest_without_images'), (methods_file, 'methods_mapping')):
            self.add(path, category)
        self.records(summary, 'completed_scene_summary', recursive=True)
        self.records(views, 'fixed_view_records', recursive=True)
        roi = local_path(views_info['roi_protocol'], PROJECT)
        require(self.hashes.sha(roi) == views_info['roi_protocol_sha256'], f'ROI registry changed: {roi}')
        self.add(roi, 'fixed_roi_registry')
        candidates = [p for p in views.rglob('*') if p.is_file() and p.suffix.lower() in IMAGES | VIDEOS
                      and 'video_frames' not in p.relative_to(views).parts]
        for path in sorted(candidates, key=lambda p: (2 if p.name.startswith('full_') else 1 if p.suffix.lower() in VIDEOS else 0, str(p))):
            self.media_candidate(path, 'fixed_views')
        for path in sorted(summary.rglob('*')):
            if path.is_file() and path.suffix.lower() in IMAGES | VIDEOS:
                self.media_candidate(path, 'summary_visualization')
        self.scenes.append({'scene': scene, 'registry': str(spec_file), **{k: str(v) for k, v in paths.items()},
                            'methods': list(methods), 'completion_checked': True})

    def media_candidate(self, path, category):
        self.media_candidates.setdefault(str(path), {'path': path, 'category': category})

    def select_media(self):
        # Allocate globally across declared scenes: short videos and native ROI
        # panels precede redundant individual full-image stills. No score or
        # visual method differences are used to choose any file.
        priority = lambda r: (2 if r['path'].name.startswith('full_') else
                              0 if r['path'].suffix.lower() in VIDEOS else 1, str(r['path']))
        for record in sorted(self.media_candidates.values(), key=priority):
            self.add_budgeted_media(record['path'], record['category'])

    def add_budgeted_media(self, path, category):
        if self.name(path) in self.files:
            return
        if self.media_bytes + path.stat().st_size > self.media_limit:
            self.media_exclusions.append({'path': str(path), 'bytes': path.stat().st_size,
                                          'reason': 'Fixed shared media-byte limit; no recompression or content-dependent selection'})
        else:
            self.media_bytes += path.stat().st_size
            self.add(path, category)

    def supplementary(self):
        # Names determine inclusion, not observed quality values. Historical
        # failed checks retain their status and are not represented as passes.
        for name in ('engineering_decision.json', 'review_cook.md', 'review_discussion.md'):
            self.add(self.root / name, 'engineering_decision_or_completed_review', required=False)
        self.add(PROJECT / 'AGENTS.md', 'current_project_execution_rules')
        # Deliberately bounded deployment record lists: no SSH configuration,
        # identities, arbitrary home-directory files, or binary assets.
        cts = self.root / 'cts_deployment_v1'
        for name in ('README.md', 'expected_files.json', 'remote_wait.log', 'resource_launch_record.json',
                     'resource_completion_record.json', 'return_job.py', 'return_rsync.log',
                     'return_service.log', 'return_status.json', 'returned_file_index.json',
                     'rsync_files.txt', 'run_remote.py', 'run_remote.sh', 'verify.py'):
            self.add(cts / name, 'cts_deployment_and_verified_return', required=False)
        for name in ('launch_provenance.json', 'expected_files.json', 'verification.json',
                     'verify.py', 'service.log', 'run_remote.sh', 'run_remote.py'):
            self.add(cts / 'remote_records' / name, 'original_cts_execution_records', required=False)
        a100 = PROJECT / 'deployment/soft_motion_a100_20260924'
        for name in ('deployment_receipt.json', 'jobs.json', 'expected_base_assets.json',
                     'expected_isolated_files.json', 'cook_s_remote.sh', 'discussion_s_remote.sh'):
            self.add(a100 / name, 'a100_attempt_original_status_not_training_success', required=False)
        for path in sorted(self.root.iterdir()):
            if not any(word in path.name.lower() for word in ('parity', 'repeat', 'roi')):
                continue
            if path.is_file() and path.suffix in RECORDS:
                self.add(path, 'supplementary_check_or_roi')
                self.supplementary_checks.append({'path': str(path), 'scope': 'Original record preserved; no inferred pass'})
            elif path.is_dir():
                self.records(path, 'parity_repeat_or_roi_records', recursive=True)
                for name in ('complete.json', 'failed.json', 'status.json'):
                    if (path / name).is_file():
                        self.supplementary_checks.append({'path': str(path / name), 'status': read(path / name).get('status')})

    def report_and_sources(self):
        companion = self.report.with_name(self.report.stem + '.codex.md')
        math_dir = self.report.parent / 'assets/codex_math' / self.report.stem
        math_info = read(math_dir / 'manifest.json')
        require(math_info['source_sha256'] == self.hashes.sha(self.report),
                'Codex math companion is stale; regenerate it before final packaging')
        for formula in math_info.get('formulas', []):
            if formula.get('mode') == 'image':
                for field in ('png', 'image', 'source'):
                    require(Path(formula[field]).is_file(), f'Missing generated formula resource: {formula[field]}')
                require(self.hashes.sha(formula['png']) == formula['png_sha256'], 'Formula PNG identity mismatch')
        for path in (self.report, companion):
            self.add(path, 'main_report')
        for path in math_dir.iterdir():
            if path.is_file():
                self.add(path, 'report_math_resource')
        self.add(self.root / 'asset_inventory.json', 'asset_inventory')
        for path in HERE.glob('*.py'):
            self.add(path, 'soft_experiment_source')
        dependencies = ['experiments/dynamic_sr_motion_bound_20260923/train.py',
                        'experiments/dynamic_sr_motion_bound_20260923/motion_model.py',
                        'experiments/dynamic_sr_motion_bound_20260923/prepare_selection.py',
                        'experiments/dynamic_sr_scene_residual_20260923/train.py',
                        'experiments/dynamic_sr_scene_residual_20260923/residual_model.py',
                        'experiments/dynamic_sr_20260918/common.py', 'experiments/dynamic_sr_20260918/n3dv_data.py',
                        'experiments/dynamic_sr_20260918/run_experiment.py', 'experiments/dynamic_sr_20260918/evaluate.py',
                        'experiments/dynamic_sr_20260920/resume_control.py', 'scripts/render_codex_math.mjs']
        for path in dependencies:
            self.add(PROJECT / path, 'direct_source_dependency')

    def rewrite_markdown(self, name, path, data, lookup):
        rewritten = 0
        def replace(match):
            nonlocal rewritten
            image, label, raw_target = match.groups()
            target = raw_target.strip('<>')
            if re.match(r'[A-Za-z][A-Za-z0-9+.-]*:', target) or target.startswith('#'):
                return match.group(0)
            target, separator, fragment = target.partition('#')
            absolute = local_path(target, path.parent)
            included = lookup.get(str(absolute)) or lookup.get(str(absolute.resolve()))
            rewritten += 1
            if included:
                relative = os.path.relpath(included, str(Path(name).parent))
                if separator:
                    relative += '#' + fragment
                return f'{image}[{label}](<{relative}>)'
            self.external_references.append({'source_document': str(path), 'target': str(absolute),
                                             'reason': 'Source retained by path only; outside declared compact package or media budget'})
            relative = os.path.relpath('external_references.json', str(Path(name).parent))
            return f'[{label}（外部来源，见索引）]({relative})'
        text = re.sub(r'(!?)\[([^\]\n]*)\]\((<?[^)\n]+>?)\)', replace, data.decode())
        return text.encode(), rewritten

    def build(self, out):
        created = datetime.now(timezone.utc).isoformat()
        checkpoints = {'schema': 2, 'created_utc': created, 'weights_included': False,
                       'model_deserialization': False, 'checkpoints': list(self.checkpoints.values())}
        caches = {'schema': 2, 'payloads_included': False, 'files': list(self.caches.values())}
        portable = [{'scene': s['scene'], **{k: self.name(s[k]) for k in ('manifest', 'methods_file')},
                     'summary': self.name(Path(s['summary_dir']) / 'summary.md'),
                     'views': self.name(Path(s['views_dir']) / 'README.md')} for s in self.scenes]
        readme = ('# 动态 SR 软运动实验交付包\n\n'
                  f'[主报告]({self.name(self.report)}) · [Codex 阅读版]({self.name(self.report.with_name(self.report.stem + ".codex.md"))})\n\n'
                  '只收录登记且已完成的场景、固定端点评价、图像导出和原始执行记录。训练固定16张与留出60帧比较保持分开；当前为单种子短窗开发证据，不作完整基准、统计显著性或真实高频恢复保证。\n\n'
                  + ''.join(f'- {s["scene"]}：[汇总]({p["summary"]}) · [固定图像/视频]({p["views"]})\n' for s,p in zip(self.scenes,portable))
                  + '\ncheckpoint_index.json 登记正式里程碑/final及引用父模型的真实路径、SHA256、字节数与已记录训练硬件；未知硬件保留null，不用评价卡代替。cache_index.json只登记缓存路径、SHA256和字节数。所有大文件均流式哈希，从未反序列化。\n\n'
                  '远端来源路径保留为recorded_path，本地实际文件列为local_path；异地副本仅在实际SHA256一致时解析。path_resolution_index.json保留未解析来源及原因，未解析文件不冒充本机检查点或缓存；source_snapshot_index.json保留原源码、原快照与回传快照路径。已完成本机检查点的收录不以远端原路径仍可访问为条件。\n\n'
                  f'包不含权重、reference/selection张量、完整HR/LR/预测图像全集，也不复制60张视频中间帧。固定媒体按文件类别及路径排序收录、总量最多{self.media_limit / 1024**2:g}MiB，不会按结果好坏筛选；省略项见media_exclusions.json。旧vrheadset本轮未新增S训练，不重复收录其大图。\n\n'
                  '原JSON/config路径保持来源原貌；包内Markdown链接改为相对路径。未收录的本机目标改链external_references.json，原文件不变。source_hash_manifest.json同时记录来源哈希与实际包内字节哈希；SHA256SUMS供离线核验。parity/repeat保留原始状态，历史失败不冒充通过。\n')
        generated = {'README.md': readme.encode(), 'checkpoint_index.json': encoded(checkpoints),
                     'cache_index.json': encoded(caches), 'portable_scene_index.json': encoded(portable),
                     'path_resolution_index.json': encoded({'schema': 1, 'resolutions': self.path_resolutions}),
                     'source_snapshot_index.json': encoded({'schema': 1, 'sources': self.source_snapshots}),
                     'media_exclusions.json': encoded(self.media_exclusions),
                     'supplementary_checks_index.json': encoded(self.supplementary_checks)}
        lookup = {}
        for name, record in self.files.items():
            lookup[str(record['path'])] = name
            lookup.setdefault(str(record['path'].resolve()), name)
        rows, sums = [], []
        archive = out / 'soft_motion_results.zip'
        partial = out / 'soft_motion_results.zip.part'
        with zipfile.ZipFile(partial, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            for name, record in sorted(self.files.items()):
                path = record['path']
                require(self.hashes.sha(path) == record['source_sha256'], f'Source changed after completion checks: {path}')
                rewritten = 0
                if path.suffix == '.md':
                    source = path.read_bytes()
                    require(hashlib.sha256(source).hexdigest() == record['source_sha256'], f'Markdown changed: {path}')
                    data, rewritten = self.rewrite_markdown(name, path, source, lookup)
                    digest, count = hashlib.sha256(data).hexdigest(), len(data)
                    zf.writestr(name, data)
                else:
                    h, count = hashlib.sha256(), 0
                    with path.open('rb') as source, zf.open(name, 'w') as target:
                        for block in iter(lambda: source.read(4 * 1024 * 1024), b''):
                            target.write(block); h.update(block); count += len(block)
                    digest = h.hexdigest()
                    require(digest == record['source_sha256'], f'Source changed during copy: {path}')
                rows.append({'archive_path': name, 'source_path': str(path), 'resolved_source_path': str(path.resolve()),
                             'source_sha256': record['source_sha256'], 'archive_sha256': digest, 'archive_bytes': count,
                             'markdown_links_rewritten': rewritten, 'categories': record['categories']})
                sums.append(f'{digest}  {name}\n')
            generated['external_references.json'] = encoded(self.external_references)
            generated['source_hash_manifest.json'] = encoded({'schema': 1, 'created_utc': created, 'sources': rows,
                'scenes': self.scenes, 'media_bytes': self.media_bytes, 'media_limit_bytes': self.media_limit,
                'exclusions': ['weights', 'reference_and_selection_tensors', 'complete_HR_LR_prediction_sequences', 'video_frames']})
            for name, data in generated.items():
                zf.writestr(name, data)
                (out / name).write_bytes(data)
                sums.append(f'{hashlib.sha256(data).hexdigest()}  {name}\n')
            zf.writestr('SHA256SUMS', ''.join(sums))
            (out / 'SHA256SUMS').write_text(''.join(sums))
        with zipfile.ZipFile(partial) as zf:
            require(zf.testzip() is None, 'ZIP CRC verification failed')
            require(not any(Path(n).suffix.lower() in WEIGHTS or 'video_frames' in Path(n).parts for n in zf.namelist()),
                    'Prohibited binary payload in package')
        partial.replace(archive)
        complete = {'status': 'completed_cpu_package', 'archive': str(archive),
                    'archive_sha256': self.hashes.sha(archive), 'archive_bytes': archive.stat().st_size,
                    'scenes': [s['scene'] for s in self.scenes], 'source_files': len(rows),
                    'checkpoint_records': len(self.checkpoints), 'cache_records': len(self.caches),
                    'unresolved_provenance_records': sum(r['status'] == 'unresolved' for r in self.path_resolutions),
                    'media_bytes': self.media_bytes, 'gpu_used': False, 'model_deserialization': False,
                    'finished_utc': datetime.now(timezone.utc).isoformat()}
        (out / 'complete.json').write_bytes(encoded(complete))
        print(json.dumps(complete, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True, help='New package directory; never overwrite')
    parser.add_argument('--max-media-mib', type=float, default=30, help='Total fixed image/video payload limit; no recompression')
    args = parser.parse_args()
    root, report, out = (p.resolve() for p in (args.root, args.report, args.out))
    require(not out.exists(), f'Output already exists: {out}')
    require(args.max_media_mib > 0, 'Media budget must be positive')
    specs = sorted((root / 'scene_methods').glob('*.json'))
    require(specs, 'No declared scene_methods/*.json; refusing an unfinished/empty package')
    if (root / 'complete.json').is_file():
        status = read(root / 'complete.json').get('status', '')
        require(status.startswith('completed'), f'Root completion receipt is not completed: {status}')
    package = Package(root, report, int(args.max_media_mib * 1024 ** 2))
    # All completion and identity gates run before any output directory exists.
    for spec in specs:
        package.scene(spec)
    package.select_media()
    package.supplementary()
    package.report_and_sources()
    for name in ('complete.json', 'README.md'):
        package.add(root / name, 'root_delivery_record', required=False)
    out.mkdir(parents=True, exist_ok=False)
    try:
        package.build(out)
    except BaseException as error:
        (out / 'failed.json').write_bytes(encoded({'status': 'failed_cpu_package', 'error': repr(error),
                                                  'complete_package': False, 'gpu_used': False}))
        raise


if __name__ == '__main__':
    main()
