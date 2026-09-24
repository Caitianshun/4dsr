#!/usr/bin/env python3
"""Restore missing prepared HR/LR from official raw videos without changing metadata.

--verify-only reads all selected HR/LR files and prints its report without writing.
Writing requires --allow-regenerate, intended for this task's new remote partial
deployment. Existing files are never overwritten; mismatching files stop the run.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile


SCENES = {
    'cook_spinach': ('n3dv_prepared/cook_spinach', 'n3dv_raw/cook_spinach',
                     'dynamic_sr_20260918/prepare_n3dv.py'),
    'cut_roasted_beef': ('n3dv_prepared/cut_roasted_beef', 'n3dv_raw/cut_roasted_beef',
                         'dynamic_sr_20260918/prepare_n3dv.py'),
    'discussion': ('meetroom_prepared/discussion', 'meetroom_raw/discussion',
                   'dynamic_sr_20260919/prepare_meetroom.py'),
}


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def image_path(root, relative):
    path = Path(relative)
    if path.is_absolute() or '..' in path.parts or path.parts[0] not in ('hr', 'lr'):
        raise ValueError(f'Unexpected image path: {relative}')
    target = root / path
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError(f'Image path escapes scene: {relative}')
    return target


def check_file(path, expected):
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f'SHA256 mismatch: {path}; expected={expected}; actual={actual}')


def check_observations(root, observations, allow_missing=False):
    checked, missing = 0, []
    for observation in observations:
        for resolution in ('hr', 'lr'):
            path = image_path(root, observation[resolution + '_path'])
            if allow_missing and not os.path.lexists(path):
                missing.append((observation, resolution))
                continue
            check_file(path, observation[resolution + '_sha256'])
            checked += 1
    return checked, missing


def extract_and_check(module, camera, frames, split, scene, expected):
    # Use the original extractor only inside a new disposable directory. It
    # cannot overwrite prepared files, manifests, initialization, or SR priors.
    with tempfile.TemporaryDirectory(prefix='.restore_images_', dir=scene) as temporary:
        staging = Path(temporary)
        extracted = module.extract_camera(camera, staging, frames, split)
        expected_keys = {(camera['camera_id'], frame) for frame in frames}
        actual_keys = {(item['camera_id'], item['frame_index']) for item in extracted}
        if actual_keys != expected_keys or len(extracted) != len(expected_keys):
            raise ValueError('Extractor returned an unexpected frame inventory')
        references = [expected[key] for key in sorted(expected_keys)]
        check_observations(staging, references)
        yield staging, references


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--scene', choices=tuple(SCENES), action='append',
                        help='Repeat to select scenes; default is all three')
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--verify-only', action='store_true')
    mode.add_argument('--allow-regenerate', action='store_true',
                      help='Explicitly authorize missing-image creation in a new partial deployment')
    parser.add_argument('--report', type=Path, help='New report path; writing mode only')
    args = parser.parse_args(argv)
    if args.verify_only and args.report is not None:
        parser.error('--verify-only is read-only; its report is printed to stdout')
    root = args.project_root.expanduser().resolve()
    selected = list(dict.fromkeys(args.scene or SCENES))
    report_path = None
    if args.allow_regenerate:
        report_path = (args.report.expanduser().absolute() if args.report else
                       root / 'deployment/a100_20260920' /
                       ('restore_images_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') +
                        f'_{os.getpid()}.json'))
        if os.path.lexists(report_path):
            parser.error(f'Refusing to overwrite report: {report_path}')
        if report_path.resolve().is_relative_to(root / 'data'):
            parser.error('Report must be outside data; data metadata must remain unchanged')
    report = dict(status='running', project_root=str(root), verify_only=args.verify_only,
                  started_at=datetime.now(timezone.utc).isoformat(), scenes={},
                  method='Original CPU extract_camera; pinned versions; original manifest SHA256 verification')
    exit_code = 1
    try:
        # Also avoid creating Python bytecode during the read-only verification.
        sys.dont_write_bytecode = True
        os.environ['OMP_NUM_THREADS'] = '4'
        os.environ['OPENBLAS_NUM_THREADS'] = '4'
        import cv2
        import numpy as np
        import torch
        cv2.setNumThreads(4)
        cv2.setRNGSeed(20260918)
        torch.set_num_threads(4)
        versions = {'opencv': cv2.__version__, 'numpy': np.__version__, 'torch': torch.__version__}
        report.update(runtime_versions=versions, threads=4,
                      opencv_build=cv2.getBuildInformation())
        contexts = []
        sys.path.insert(0, str(root / 'experiments/dynamic_sr_20260918'))
        for name in selected:
            prepared, raw, script = SCENES[name]
            scene = root / 'data/dynamic_sr' / prepared
            raw = root / 'data/dynamic_sr' / raw
            manifest_path = scene / 'manifest.json'
            manifest_sha = sha256(manifest_path)
            manifest = json.loads(manifest_path.read_text())
            entry = report['scenes'][name] = dict(
                manifest=str(manifest_path), manifest_sha256=manifest_sha,
                original_versions=manifest['versions'], raw_directory=str(raw),
                restored_files=0, completed_cameras=[])
            if versions != manifest['versions']:
                raise ValueError(f'{name}: runtime versions differ from original: {versions} != {manifest["versions"]}')
            source = root / 'experiments' / script
            for filename, expected_sha in manifest['script_sha256'].items():
                check_file(source.with_name(filename), expected_sha)
            observations = manifest['observations']
            expected = {(item['camera_id'], item['frame_index']): item for item in observations}
            if len(expected) != len(observations):
                raise ValueError(f'{name}: duplicate manifest observations')
            checked, missing = check_observations(scene, observations, allow_missing=args.allow_regenerate)
            entry.update(initial_verified_files=checked, initial_missing_files=len(missing))
            context = dict(name=name, scene=scene, raw=raw, manifest=manifest,
                           expected=expected, entry=entry, source=source, missing=missing)
            contexts.append(context)
        if args.allow_regenerate:
            # Validate frame 0 for one camera in every selected scene before
            # publishing any regenerated images for any scene.
            for context in contexts:
                manifest, entry = context['manifest'], context['entry']
                check_file(context['raw'] / 'poses_bounds.npy', manifest['poses_bounds_sha256'])
                spec = importlib.util.spec_from_file_location('restore_' + context['name'], context['source'])
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                context['module'] = module
                first = min(manifest['cameras'])
                if (first, 0) not in context['expected']:
                    raise ValueError(f'{context["name"]}: required frame-0 probe missing')
                camera = dict(manifest['cameras'][first])
                camera['video'] = str(context['raw'] / Path(camera['video']).name)
                split = context['expected'][(first, 0)]['split']
                for _ in extract_and_check(module, camera, [0], split, context['scene'], context['expected']):
                    pass
                entry['frame0_probe'] = {'camera_id': first, 'hr_sha256_matches': True, 'lr_sha256_matches': True}
                print(f'PROBE_OK {context["name"]} {first}/0000', flush=True)
            for context in contexts:
                manifest, scene, entry = context['manifest'], context['scene'], context['entry']
                for camera_id in sorted(manifest['cameras']):
                    observations = [item for item in manifest['observations'] if item['camera_id'] == camera_id]
                    _, missing = check_observations(scene, observations, allow_missing=True)
                    frames = sorted({item['frame_index'] for item, _ in missing})
                    if frames:
                        camera = dict(manifest['cameras'][camera_id])
                        camera['video'] = str(context['raw'] / Path(camera['video']).name)
                        for staging, references in extract_and_check(
                                context['module'], camera, frames, observations[0]['split'], scene, context['expected']):
                            for item in references:
                                for resolution in ('hr', 'lr'):
                                    relative = item[resolution + '_path']
                                    target = image_path(scene, relative)
                                    if os.path.lexists(target):
                                        check_file(target, item[resolution + '_sha256'])
                                        continue
                                    target.parent.mkdir(parents=True, exist_ok=True)
                                    # Exclusive atomic publication on the same
                                    # filesystem; an existing file is never replaced.
                                    os.link(staging / relative, target)
                                    entry['restored_files'] += 1
                    entry['completed_cameras'].append(camera_id)
                    print(f'CAMERA_OK {context["name"]} {camera_id}', flush=True)
        for context in contexts:
            checked, _ = check_observations(context['scene'], context['manifest']['observations'])
            check_file(context['scene'] / 'manifest.json', context['entry']['manifest_sha256'])
            context['entry'].update(final_verified_files=checked, all_image_hashes_match=True,
                                    manifest_unchanged=True)
        report['status'] = 'complete'
        exit_code = 0
    except (Exception, KeyboardInterrupt) as error:
        report.update(status='failed', error=repr(error))
    report['finished_at'] = datetime.now(timezone.utc).isoformat()
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open('x') as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False)
            handle.write('\n')
        print(f'REPORT {report_path}', flush=True)
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
