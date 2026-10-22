"""CPU-only storage experiment on completed, non-formal IE GPU fixtures.

No producer, reader, tensor precision, or scientific protocol is changed. The
uncompressed six-scene payload is authoritative; extrapolated compression from
a few fixtures is only a planning estimate, never a disk-capacity guarantee.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import math
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import time
import traceback

from common import ROOT, checked, identity, read, resolve, sha, write_new
from prepare_suite import SCENES

CHANNELS = {'internal_rgb': 3, 'internal_depth': 1,
            'external_rgb': 3, 'external_inverse_depth': 1}
DEFAULT_SUITE = ROOT / 'output/static_sr_wu_20261022/prepared_final/suite_index.json'
INTERNAL_LIMIT = ('Perturbed fresh-model wiring fixtures, not a trained 30k internal '
                  'teacher; their compression is not representative evidence for formal internal caches.')


def suite_sizes(path):
    """Read and hash metadata only; never open train/held-out/HR pixels."""
    suite = read(path)
    if suite.get('schema') != 'static_sr_wu_six_scene_suite_v1':
        raise ValueError('Require the registered six-scene suite')
    rows = suite['scenes']
    if [(r['short'], r['scene']) for r in rows] != list(SCENES):
        raise ValueError('Suite must contain the exact six current scenes in registered order')
    result = []
    for row in rows:
        manifest = read(checked(row['manifest']))
        train = manifest['splits']['train']
        observations = [r for r in manifest['observations'] if r['split'] == 'train']
        keys = {(r['camera_id'], r['frame_index']) for r in observations}
        if (manifest['scene'] != row['scene'] or row['frames'] != 300
                or manifest['frame_indices'] != list(range(300)) or 'cam00' in train
                or row['train_camera_count'] != len(train)
                or len(keys) != len(observations)
                or keys != {(c, f) for c in train for f in range(300)}
                or row['lr_count'] != len(observations)):
            raise ValueError('Scene/frame/training-camera inventory differs: ' + row['short'])
        width, height = manifest['resolutions']['hr']
        pixels = width * height * len(observations)
        fields = {name: pixels * count * 4 for name, count in CHANNELS.items()}
        result.append(dict(short=row['short'], scene=row['scene'],
            domain='MeetRoom' if row['scene'].startswith('meetroom_') else 'N3DV',
            manifest=row['manifest'], hr_wh=[width, height], observations=len(observations),
            frames=300, train_cameras=len(train), pixels=pixels, field_payload_bytes=fields,
            float32_payload_bytes=sum(fields.values()),
            uint8_inverse_depth_only_payload_bytes=sum(fields.values()) - pixels * 3))
    return result


def collect_samples(external_receipts, wiring_receipts, scenes):
    """Only finished external inference or completed IE wiring records qualify."""
    inventory = {r['short']: r for r in scenes}
    samples, receipts = [], []
    for file in external_receipts:
        receipt = read(file)
        if (receipt.get('schema') != 'iesrgs_external_teacher_GPU_acceptance_v1'
                or receipt.get('GPU_executed') is not True
                or receipt.get('status') != 'passed_external_teacher_GPU_inference_only'):
            raise ValueError('External receipt must certify completed real GPU inference')
        receipts.append(identity(file))
        for row in receipt['results']:
            scene = inventory[row['short']]
            if row['status'] != 'passed' or row.get('formal_prior_bundle') is not False:
                raise ValueError('Only completed non-formal external fixtures are accepted')
            for field in ('external_rgb', 'external_inverse_depth'):
                path = checked(row['outputs'][field])
                samples.append(dict(scene=scene['short'], domain=scene['domain'], field=field,
                    camera=row['camera'], frame=row['frame'], hr_wh=scene['hr_wh'],
                    source=identity(path), receipt=identity(file),
                    origin='real_pretrained_external_teacher_inference',
                    limitation='One or a few real train images; not a full-scene compression survey.'))
    for file in wiring_receipts:
        receipt = read(file)
        if (receipt.get('GPU_executed') is not True
                or receipt.get('status') not in ('passed_bounded_GPU_wiring_only', 'failed')):
            raise ValueError('Wiring receipt must be finished; never probe a running output')
        receipts.append(identity(file))
        selected = [r for r in receipt.get('results', []) if r['method'] == 'iesrgs' and r['status'] == 'passed']
        if not selected:
            raise ValueError('No completed IE wiring record in receipt')
        for row in selected:
            scene = inventory[row['scene']]
            directory = resolve(file).parent / scene['short'] / 'iesrgs/fixture_tensors'
            frames = {r['frame'] for r in row['depth_probes']}
            for field in ('internal_rgb', 'internal_depth'):
                files = sorted(directory.glob('*_' + field + '.npy'))
                parsed = [re.fullmatch(r'(cam\d+)_(\d+)_' + field + r'\.npy', f.name) for f in files]
                if not files or any(p is None for p in parsed) or {int(p[2]) for p in parsed} != frames:
                    raise ValueError('Wiring internal-array frame coverage differs')
                for path, match in zip(files, parsed):
                    samples.append(dict(scene=scene['short'], domain=scene['domain'], field=field,
                        camera=match[1], frame=int(match[2]), hr_wh=scene['hr_wh'],
                        source=identity(path), receipt=identity(file),
                        origin='perturbed_fresh_Wu_wiring_fixture', limitation=INTERNAL_LIMIT))
    paths = [s['source']['path'] for s in samples]
    if len(paths) != len(set(paths)):
        raise ValueError('Duplicate sample paths would bias the storage summary')
    for sample in samples:
        if sample['camera'] == 'cam00' or sample['frame'] not in range(300):
            raise ValueError('Only actual training-view fixture keys are accepted')
    return samples, receipts


def assert_exact(original, recovered):
    import numpy as np
    if original.dtype != recovered.dtype or original.shape != recovered.shape:
        raise ValueError('Lossless roundtrip changed dtype or shape')
    if not np.array_equal(original, recovered):
        raise ValueError('Lossless roundtrip changed array values')
    if original.tobytes(order='C') != recovered.tobytes(order='C'):
        raise ValueError('Lossless roundtrip changed floating-point bit patterns')
    return dict(array_equal=True, dtype_equal=True, shape_equal=True, bitwise_equal=True)


def inverse_depth_uint8(array):
    """Exact source encoding test, not a precision-reduction proposal."""
    import numpy as np
    if array.dtype != np.float32 or not np.isfinite(array).all():
        raise ValueError('Source inverse depth must be finite float32')
    scaled = array * np.float32(65536.)
    if scaled.min() < 0 or scaled.max() > 255 or not np.array_equal(scaled, np.rint(scaled)):
        raise ValueError('Depth is not exactly the source uint8 / 65536 representation')
    encoded = scaled.astype(np.uint8)
    decoded = encoded.astype(np.float32) / np.float32(65536.)
    assert_exact(array, decoded)
    return encoded


def benchmark_codec(path, array, output, codec, zstd=None):
    import numpy as np
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    started = time.perf_counter()
    if codec == 'npz_deflate':
        with output.open('xb') as stream:
            np.savez_compressed(stream, array=array)
    elif codec == 'zstd_level3':
        if zstd is None:
            raise ValueError('zstd executable unavailable')
        subprocess.run([zstd, '-q', '-3', '-T1', str(path), '-o', str(output)], check=True,
                       capture_output=True, text=True)
    else:
        raise ValueError('Unknown codec')
    compression = time.perf_counter() - started
    started = time.perf_counter()
    if codec == 'npz_deflate':
        with np.load(output, allow_pickle=False) as archive:
            recovered = archive['array']
    else:
        temporary = output.with_name(output.name + '.roundtrip.npy')
        if temporary.exists():
            raise FileExistsError(temporary)
        subprocess.run([zstd, '-q', '-d', str(output), '-o', str(temporary)], check=True,
                       capture_output=True, text=True)
        recovered = np.load(temporary, allow_pickle=False)
    decompression = time.perf_counter() - started
    exact = assert_exact(array, recovered)
    if codec == 'zstd_level3':
        temporary.unlink()  # Only this invocation's verified reconstruction.
    original_size, compressed_size = Path(path).stat().st_size, output.stat().st_size
    return dict(codec=codec, source_npy_bytes=original_size, array_payload_bytes=array.nbytes,
        compressed_bytes=compressed_size, stored_fraction_of_npy=compressed_size / original_size,
        stored_fraction_of_payload=compressed_size / array.nbytes,
        compression_factor=original_size / compressed_size, compression_seconds=compression,
        decompression_seconds=decompression, validation_seconds_excluded=True,
        roundtrip=exact, compressed_file=identity(output))


def extrapolate(scenes, results, safety_factor=1.10):
    """Worst observed ratio per domain/field, with explicit missing-data fallback."""
    if not math.isfinite(safety_factor) or safety_factor < 1:
        raise ValueError('Sample safety factor must be finite and >= 1')
    codecs = sorted({b['codec'] for r in results for b in r['benchmarks']})
    estimates = []
    for codec in codecs:
        entries = []
        for scene in scenes:
            for field, payload in scene['field_payload_bytes'].items():
                observed = [b['stored_fraction_of_payload'] for r in results
                    if r['domain'] == scene['domain'] and r['field'] == field
                    for b in r['benchmarks'] if b['codec'] == codec]
                # Missing fields receive the full payload and 4 KiB/file overhead.
                estimated = (math.ceil(payload * max(observed) * safety_factor) if observed
                             else payload + scene['observations'] * 4096)
                entries.append(dict(scene=scene['short'], field=field,
                    sample_count=len(observed), worst_observed_stored_fraction=max(observed) if observed else None,
                    estimated_bytes=estimated,
                    basis='worst_sample_times_safety_factor' if observed else 'uncompressed_plus_4KiB_per_file'))
        estimates.append(dict(codec=codec, estimated_six_scene_bytes=sum(r['estimated_bytes'] for r in entries),
            all_domain_fields_sampled=all(r['sample_count'] for r in entries), components=entries,
            capacity_guaranteed=False, safety_factor=safety_factor,
            limitation='Sample-based planning estimate; fresh internal fixtures may compress very differently after training.'))
    return estimates


def probe(samples, out, zstd):
    import numpy as np
    results = []
    for index, sample in enumerate(samples):
        path = checked(sample['source'])
        array = np.load(path, allow_pickle=False)
        width, height = sample['hr_wh']
        if (array.dtype != np.float32 or array.shape != (CHANNELS[sample['field']], height, width)
                or not np.isfinite(array).all()):
            raise ValueError('Production float32 calibrated array contract differs: ' + str(path))
        directory = out / ('%03d_%s_%s_%04d_%s' %
            (index, sample['scene'], sample['camera'], sample['frame'], sample['field']))
        directory.mkdir()
        row = dict(**sample, dtype=str(array.dtype), shape=list(array.shape),
                   array_payload_bytes=array.nbytes, source_npy_bytes=path.stat().st_size, benchmarks=[])
        row['benchmarks'].append(benchmark_codec(path, array, directory / 'array.npz', 'npz_deflate'))
        if zstd:
            row['benchmarks'].append(benchmark_codec(path, array, directory / 'array.npy.zst', 'zstd_level3', zstd))
        if sample['field'] == 'external_inverse_depth':
            started = time.perf_counter()
            encoded = inverse_depth_uint8(array)
            encode_seconds = time.perf_counter() - started
            encoded_path = directory / 'SOURCE_UINT8_ENCODING_ONLY.npy'
            with encoded_path.open('xb') as stream:
                np.save(stream, encoded, allow_pickle=False)
            started = time.perf_counter()
            decoded = np.load(encoded_path, allow_pickle=False).astype(np.float32) / np.float32(65536.)
            decode_seconds = time.perf_counter() - started
            row['exact_source_uint8_encoding'] = dict(file=identity(encoded_path),
                bytes=encoded_path.stat().st_size, roundtrip=assert_exact(array, decoded),
                encode_and_check_seconds=encode_seconds, load_and_decode_seconds=decode_seconds,
                decode='uint8.astype(float32) / float32(65536)', producer_or_reader_changed=False,
                per_pixel_saved_bytes=3, full_IE_payload_saving_fraction=3/32)
        checked(sample['source'])  # Refuse changed/live inputs even if compression passed.
        results.append(row)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--external-receipt', type=Path, action='append', default=[])
    parser.add_argument('--wiring-receipt', type=Path, action='append', default=[])
    parser.add_argument('--suite-index', type=Path, default=DEFAULT_SUITE)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--plan-only', action='store_true')
    parser.add_argument('--safety-factor', type=float, default=1.10)
    args = parser.parse_args()
    out = resolve(args.out)
    scenes = suite_sizes(args.suite_index)
    samples, receipts = collect_samples(args.external_receipt, args.wiring_receipt, scenes)
    if not samples and not args.plan_only:
        raise ValueError('Provide completed real GPU receipts, or use --plan-only for pending sizing')
    if not math.isfinite(args.safety_factor) or args.safety_factor < 1:
        raise ValueError('Sample safety factor must be finite and >= 1')
    out.mkdir(parents=True, exist_ok=False)
    zstd = shutil.which('zstd')
    payload = sum(r['float32_payload_bytes'] for r in scenes)
    uint8_payload = sum(r['uint8_inverse_depth_only_payload_bytes'] for r in scenes)
    report = dict(schema='static_sr_wu_lossless_cache_CPU_probe_v1',
        status='pending_real_sample_compression', source=identity(__file__), host=socket.gethostname(),
        suite_index=identity(args.suite_index), receipts=receipts, scenes=scenes,
        started_utc=datetime.now(timezone.utc).isoformat(), GPU_executed=False,
        torch_imported='torch' in sys.modules, producer_or_reader_changed=False, scientific_protocol_changed=False,
        formal_prior_bundle=False, training_HR_read=False, heldout_pixels_read=False,
        float32_six_scene_payload_bytes=payload, raw_capacity_basis='32 bytes per HR pixel; excludes NPY headers, indexes, models and temporary files',
        exact_uint8_inverse_depth_only=dict(hypothesis='Exact source uint8/65536 representation must pass per-array roundtrip',
            full_six_scene_payload_bytes=uint8_payload, saved_payload_bytes=payload-uint8_payload,
            full_IE_payload_saving_fraction=3/32, automatically_applied=False),
        codec_environment=dict(zstd_executable=zstd,
            zstd_version=subprocess.run([zstd, '--version'], capture_output=True, text=True, check=True).stdout.strip() if zstd else None,
            zstd_level=3, zstd_threads=1, python=sys.version, cpu_count=os.cpu_count()),
        internal_sample_limitation=INTERNAL_LIMIT, samples=samples, results=[])
    started = time.perf_counter()
    try:
        if not args.plan_only:
            report['results'] = probe(samples, out, zstd)
            report['estimates'] = extrapolate(scenes, report['results'], args.safety_factor)
            report['status'] = 'passed_exact_CPU_sample_roundtrips_capacity_not_guaranteed'
    except BaseException:
        report.update(status='failed', error=traceback.format_exc())
        raise
    finally:
        report.update(elapsed_seconds=time.perf_counter()-started, torch_imported='torch' in sys.modules,
                      finished_utc=datetime.now(timezone.utc).isoformat())
        write_new(out / 'receipt.json', report)
    print({'status': report['status'], 'receipt': identity(out / 'receipt.json'),
           'float32_six_scene_payload_bytes': payload})


if __name__ == '__main__':
    main()
