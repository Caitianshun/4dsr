"""CPU-only exact container/production-loss acceptance on returned real fixtures.

Internal arrays are perturbed fresh Wu wiring outputs, never a 30k teacher.
External arrays are real pretrained teacher outputs. This check produces no
formal prior bundle, changes no source array, and performs no model update.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import os
import sys
import time
import traceback

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from common import checked, identity, load_prior_array, read, resolve, write_new
from lossless_cache_probe import DEFAULT_SUITE, INTERNAL_LIMIT, assert_exact, collect_samples, suite_sizes

# The source that produced the original full-size NPY baseline used exactly
# the rendered/depth construction below. Reuse cannot silently change it.
ORIGINAL_BASELINE_SCRIPT_SHA256 = 'd0b7f01b3251643f6949f9d330a41d602f867099231b5db326f5661abf725289'


def baseline_receipt(path, expected_sha, receipts):
    if not expected_sha:
        raise ValueError('Reusing a baseline requires its pre-existing --baseline-sha256')
    item = {'path': str(path), 'sha256': expected_sha}
    value = read(checked(item))
    if (value.get('schema') != 'iesrgs_lossless_container_CPU_acceptance_v1'
            or value.get('status') != 'passed_real_full_resolution_CPU_container_loss_gradient_bitwise_parity'
            or value.get('GPU_executed') is not False or value.get('CUDA_initialized') is not False
            or value.get('parameter_updates') != 0 or value.get('receipts') != receipts):
        raise ValueError('Baseline is incomplete, foreign, or derived from different fixtures')
    source = {Path(v['path']).name: v for v in value['source']}
    if source['iesrgs_cache_real_acceptance.py']['sha256'] != ORIGINAL_BASELINE_SCRIPT_SHA256:
        raise ValueError('Unknown baseline prediction fixture recipe')
    if source['iesrgs.py'] != identity(BASE / 'methods/iesrgs.py'):
        raise ValueError('The scientific IE objective changed since the baseline')
    rows = {r['scene']: r for r in value['results']}
    if set(rows) != {'Cook07', 'Discussion07'} or len(rows) != len(value['results']):
        raise ValueError('Baseline scene coverage differs')
    return rows, identity(checked(item))


def exact_result(torch, rendered, depth, priors):
    from methods.iesrgs import fusion_loss
    rgb, median = rendered.clone().requires_grad_(), depth.clone().requires_grad_()
    terms = fusion_loss(rgb, median, priors)
    terms['total'].backward()
    tensors = {**terms, 'render_rgb_gradient': rgb.grad, 'median_depth_gradient': median.grad}
    result, bits = {}, {}
    for name, tensor in tensors.items():
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError('Nonfinite complete IE objective/gradient: ' + name)
        if name.endswith('_gradient') and not bool(tensor.abs().sum() > 0):
            raise ValueError('Gradient check has no signal: ' + name)
        bits[name] = tensor.detach().numpy().tobytes(order='C')
        result[name] = dict(shape=list(tensor.shape), dtype=str(tensor.dtype),
            sha256=hashlib.sha256(bits[name]).hexdigest(),
            value=float(tensor.detach()) if tensor.numel() == 1 else None,
            absolute_sum=float(tensor.detach().abs().sum()))
    return result, bits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--external-receipt', required=True)
    parser.add_argument('--wiring-receipt', required=True)
    parser.add_argument('--suite-index', default=str(DEFAULT_SUITE))
    parser.add_argument('--out', required=True)
    parser.add_argument('--cache-format', choices=('npy', 'npz_deflate', 'npz_byte_shuffle'), action='append',
                        help='Formats to validate; default preserves the original NPY/plain-NPZ matrix')
    parser.add_argument('--baseline-receipt')
    parser.add_argument('--baseline-sha256')
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Run CPU-only with CUDA_VISIBLE_DEVICES explicitly empty')
    import torch
    from methods.iesrgs import IESRGSPriors
    from methods.iesrgs_priors import _save_array, load_priors
    from train_stage import load_tensor
    torch.set_num_threads(1)
    scenes = suite_sizes(args.suite_index)
    samples, receipts = collect_samples([args.external_receipt], [args.wiring_receipt], scenes)
    formats = args.cache_format or ['npy', 'npz_deflate']
    if len(formats) != len(set(formats)):
        raise ValueError('Do not repeat a cache format')
    previous, previous_identity = baseline_receipt(args.baseline_receipt, args.baseline_sha256, receipts) if args.baseline_receipt else ({}, None)
    source = [identity(path) for path in (Path(__file__), BASE / 'common.py', BASE / 'train_stage.py',
              BASE / 'prepare_priors.py', BASE / 'methods/iesrgs_priors.py', BASE / 'methods/iesrgs.py')]
    out = resolve(args.out)
    out.mkdir(parents=True, exist_ok=False)
    report = dict(schema='iesrgs_lossless_container_CPU_acceptance_v1', status='running',
        source=source, receipts=receipts, started_utc=datetime.now(timezone.utc).isoformat(),
        GPU_executed=False, CUDA_initialized=False, parameter_updates=0, formal_prior_bundle=False,
        formal_parent_allowed=False, formal_prior_allowed=False,
        training_HR_read=False, heldout_pixels_read=False, tensor_precision='float32 unchanged',
        internal_limit=INTERNAL_LIMIT, cache_formats=formats, reused_baseline=previous_identity,
        baseline_comparison='raw tensor SHA256/shape/dtype' if previous else 'in-memory exact bytes',
        tests=[], results=[])
    started = time.perf_counter()
    try:
        for scene in ('Cook07', 'Discussion07'):
            external = [r for r in samples if r['scene'] == scene and r['field'] == 'external_rgb']
            if len(external) != 1:
                raise ValueError('Expected one exact external fixture per domain')
            key = (external[0]['camera'], external[0]['frame'])
            selected = [r for r in samples if r['scene'] == scene and (r['camera'], r['frame']) == key]
            entries = {r['field']: r['source'] for r in selected}
            fields = ('internal_rgb', 'internal_depth', 'external_rgb', 'external_inverse_depth')
            if set(entries) != set(fields) or len(selected) != 4:
                raise ValueError('Internal/external sample keys do not match exactly')
            arrays = {name: load_prior_array(item, require_float32=True) for name, item in entries.items()}
            shape = arrays['external_rgb'].shape
            if shape != (3, *arrays['external_inverse_depth'].shape[-2:]):
                raise ValueError('Real external sample shape differs')
            variants = {} if previous else {'legacy_npy': entries}
            row = dict(scene=scene, camera=key[0], frame=key[1], shape=list(shape),
                       source_arrays=selected, containers=[], loss_and_gradient=[])
            for fmt in formats:
                variants[fmt] = {}
                for name, array in arrays.items():
                    t = time.perf_counter()
                    item = _save_array(out / scene / fmt, name + '.npy', torch.from_numpy(array), fmt)
                    save_seconds = time.perf_counter() - t
                    t = time.perf_counter()
                    recovered = load_prior_array(item, require_float32=True)
                    load_seconds = time.perf_counter() - t
                    exact = assert_exact(array, recovered)
                    variants[fmt][name] = item
                    row['containers'].append(dict(field=name, format=fmt, file=item,
                        payload_bytes=array.nbytes, stored_bytes=checked(item).stat().st_size,
                        write_seconds=save_seconds, verified_read_seconds=load_seconds, roundtrip=exact))
            external_tensor = torch.from_numpy(arrays['external_rgb'])
            internal_tensor = torch.from_numpy(arrays['internal_rgb'])
            rendered = external_tensor * .7 + internal_tensor * .3 + .01
            depth = torch.from_numpy(arrays['internal_depth']) * 1.05 + .002
            baseline = None
            reference = None
            if previous:
                old = previous[scene]
                if old['source_arrays'] != selected or old['shape'] != list(shape) or (old['camera'], old['frame']) != key:
                    raise ValueError('Baseline source arrays or observation identity changed')
                choices = [r for r in old['loss_and_gradient'] if r['format'] == 'legacy_npy' and r['reader'] == 'train_stage.load_tensor']
                if len(choices) != 1:
                    raise ValueError('Exact original NPY reader baseline missing')
                reference = choices[0]['tensors']
            for fmt, entry in variants.items():
                for reader in ('train_stage.load_tensor', 'iesrgs_priors.load_priors'):
                    t = time.perf_counter()
                    if reader == 'train_stage.load_tensor':
                        priors = IESRGSPriors(**{name: load_tensor(item, 'cpu') for name, item in entry.items()})
                    else:
                        priors = load_priors(entry, 'cpu')
                    for name, tensor in vars(priors).items():
                        assert_exact(arrays[name], tensor.numpy())
                    result, bits = exact_result(torch, rendered, depth, priors)
                    if reference is not None:
                        if set(result) != set(reference) or any(
                                result[name][field] != reference[name][field]
                                for name in result for field in ('shape', 'dtype', 'sha256')):
                            raise ValueError(f'IE loss/gradient differs from pinned NPY baseline: {scene}/{fmt}/{reader}')
                    elif baseline is None:
                        baseline = bits
                    elif bits != baseline:
                        raise ValueError(f'Complete IE loss or gradient bits differ: {scene}/{fmt}/{reader}')
                    row['loss_and_gradient'].append(dict(format=fmt, reader=reader,
                        seconds=time.perf_counter()-t, bitwise_equal_to_legacy=True, tensors=result))
            report['results'].append(row)
        for item in source + receipts + ([previous_identity] if previous_identity else []) + [s['source'] for s in samples]:
            checked(item)
        if torch.cuda.is_initialized():
            raise RuntimeError('CPU acceptance unexpectedly initialized CUDA')
        report['status'] = 'passed_real_full_resolution_CPU_container_loss_gradient_bitwise_parity'
        report['tests'] = ['legacy NPY compatible', 'float32 shape and every bit preserved',
            'selected production lossless containers: ' + ', '.join(formats), 'both production readers',
            'all complete IE loss terms and nonzero RGB/depth gradients bitwise identical',
            'two real domains and full original output sizes', 'source and fixture SHA unchanged']
    except BaseException:
        report.update(status='failed', error=traceback.format_exc())
        raise
    finally:
        report.update(elapsed_seconds=time.perf_counter()-started,
                      CUDA_initialized=torch.cuda.is_initialized(), finished_utc=datetime.now(timezone.utc).isoformat())
        write_new(out / 'receipt.json', report)
    print({'status': report['status'], 'receipt': identity(out / 'receipt.json'),
           'elapsed_seconds': report['elapsed_seconds']})


if __name__ == '__main__':
    main()
