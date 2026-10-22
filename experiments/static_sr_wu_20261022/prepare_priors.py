"""CLI for isolated source-specific cache producers. No automatic GPU selection."""
from __future__ import annotations
import argparse
import os
from common import ROOT, checked, identity, read, training_manifest, validate_lr_parent_payload


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--kind', required=True, choices=['splatsure_weights','iesrgs_internal','iesrgs_external','iesrgs_merge'])
    p.add_argument('--out', required=True)
    p.add_argument('--checkpoint')
    p.add_argument('--upstream')
    p.add_argument('--rasterizer-source')
    p.add_argument('--rasterizer-binary-root', default=str(ROOT / 'output/static_sr_wu_20261022/runtime/splatsure'))
    p.add_argument('--swinir-checkpoint')
    p.add_argument('--depth-checkpoint')
    p.add_argument('--internal-index')
    p.add_argument('--external-index')
    p.add_argument('--cache-format', choices=('npy', 'npz_deflate', 'npz_byte_shuffle'), default='npy',
                   help='IE array container only; both preserve the exact float32 values')
    a = p.parse_args()
    if a.cache_format != 'npy' and a.kind not in ('iesrgs_external', 'iesrgs_internal'):
        p.error('--cache-format applies only to IE external/internal production; merge infers both formats')
    cfg = read(a.config)
    training_manifest(checked(cfg['manifest']), verify_pixels=True)
    if a.kind != 'iesrgs_merge':
        visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        if not visible or ',' in visible or visible == '-1':
            raise ValueError('Run through resource admission with exactly one GPU')
    if a.kind == 'iesrgs_merge':
        from methods.iesrgs_priors import merge_priors
        merge_priors(a.external_index, a.internal_index, a.out)
    elif a.kind == 'iesrgs_external':
        from methods.iesrgs_priors import prepare_external
        prepare_external(cfg, a.upstream, identity(a.swinir_checkpoint), identity(a.depth_checkpoint), a.out,
                         device='cuda', cache_format=a.cache_format)
    else:
        from backend import WuBackend
        ck = identity(a.checkpoint)
        backend = WuBackend(cfg, checkpoint=ck)
        if a.kind == 'iesrgs_internal':
            from methods.iesrgs_priors import prepare_internal
            prepare_internal(backend, ck, a.out, cache_format=a.cache_format)
        else:
            from methods.splatsure_prepare import load_official_rasterizer, prepare_splatsure_cache
            payload = backend.torch.load(checked(ck), map_location='cpu', weights_only=False)
            validate_lr_parent_payload(payload,'splatsure')
            del payload
            api = load_official_rasterizer(a.rasterizer_source, a.rasterizer_binary_root)
            prepare_splatsure_cache(backend, checked(cfg['manifest']), a.out, api,
                dict(parent_checkpoint_sha256=ck['sha256'], parent_checkpoint=ck,
                     source_config=identity(a.config), backend=identity(__file__)))

if __name__ == '__main__':
    main()
