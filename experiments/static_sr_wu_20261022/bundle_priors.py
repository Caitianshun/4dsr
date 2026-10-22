"""Bind real method producer receipts into a complete per-scene train-only bundle."""
from __future__ import annotations
import argparse
from pathlib import Path
from common import checked, identity, read, resolve, training_manifest, write_new, verify_prior_index


def same_file(left, right):
    """Accept absolute producer paths and project-relative registrations equally."""
    return checked(left) == checked(right) and left['sha256'] == right['sha256']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--sr-receipt', required=True)
    p.add_argument('--weights-receipt', help='SplatSuRe frozen LR-map receipt')
    p.add_argument('--model', action='append', required=True, help='Actual frozen weight/checkpoint file; repeat')
    p.add_argument('--output', required=True)
    a = p.parse_args()
    cfg = read(a.config)
    m, rows = training_manifest(checked(cfg['manifest']))
    sr = read(a.sr_receipt)
    if sr.get('status') not in ('teacher_complete', 'completed'):
        raise ValueError('Only successful actual teacher production can be bundled')
    if cfg['method'] == 'splatsure':
        variant = cfg.get('teacher_variant','stablesr')
        expected_schema = 'wu_splatsure_stablesr_teacher_v1' if variant == 'stablesr' else 'wu_splatsure_swinir_teacher_v1'
        if sr.get('schema') != expected_schema or not same_file(sr['manifest'], cfg['manifest']):
            raise ValueError('Wrong SplatSuRe teacher recipe or input manifest')
        if variant == 'swinir' and sr.get('recipe') != 'SplatSuRe+SwinIR-M_DF2K_x4_generic':
            raise ValueError('Do not replace source generic SwinIR with a project teacher')
        if variant == 'stablesr' and sr.get('recipe') != 'SplatSuRe+StableSR_512_DDPM200_official_default_explicit_adaptation':
            raise ValueError('StableSR teacher recipe differs from the registered adaptation')
        expected_models = {str(checked(v)): v['sha256'] for v in sr['model_identities']}
        actual_models = {str(resolve(p)): identity(p)['sha256'] for p in a.model}
        if not expected_models or actual_models != expected_models:
            raise ValueError('Registered weights must exactly match the actual SplatSuRe teacher receipt')
        targets = {f"{e['camera']}/{e['frame']:04d}": str(checked(e['sr'])) for e in sr['targets']}
        hashes = {f"{e['camera']}/{e['frame']:04d}": e['lr_sha256'] for e in sr['targets']}
    else:
        targets = sr['targets']
        hashes = sr.get('lr_sha256', {})
        expected_models = {str(checked({'path':p,'sha256':s})):s for p,s in sr.get('weights',{}).items()}
        actual_models = {str(resolve(p)): identity(p)['sha256'] for p in a.model}
        if not expected_models or actual_models != expected_models:
            raise ValueError('Registered weights must exactly match the actual PSRT/SpyNet teacher receipt')
        if set(sr.get('target_sha256',{})) != set(targets):
            raise ValueError('Teacher receipt must bind generated target bytes')
        for key,path in targets.items():
            checked({'path':path,'sha256':sr['target_sha256'][key]})
    expected = {f"{o['camera_id']}/{o['frame_index']:04d}":o for o in rows}
    if set(targets) != set(expected):
        raise ValueError('Teacher keys do not cover exactly the legal training inputs')
    # Check source LR hashes, rather than accepting similarly named caches.
    if not hashes:
        for group in sr.get('groups', []):
            hashes.update(group.get('lr_sha256', {}))
    if hashes != {k:o['lr_sha256'] for k,o in expected.items()}:
        raise ValueError('Teacher receipt does not bind every exact LR input')
    entries = [dict(camera=o['camera_id'], frame=o['frame_index'], time=o['time'],
                    lr_sha256=o['lr_sha256'], sr=identity(targets[k])) for k,o in expected.items()]
    if cfg['method'] == 'splatsure':
        if not a.weights_receipt:
            raise ValueError('SplatSuRe cannot train without actual contribution-based maps')
        weights = read(a.weights_receipt)
        if weights.get('status') != 'prepared' or weights.get('manifest_sha256') != cfg['manifest']['sha256']:
            raise ValueError('SplatSuRe map receipt mismatch')
        maps = {(o['camera_id'],o['frame_index']):o for o in weights['records']}
        if set(maps) != {(o['camera_id'],o['frame_index']) for o in rows}:
            raise ValueError('Incomplete weight maps')
        for entry in entries:
            row = maps[(entry['camera'],entry['frame'])]
            item = identity(resolve(a.weights_receipt).parent / row['path'])
            if item['sha256'] != row['sha256']:
                raise ValueError('Weight map bytes changed')
            entry['weight'] = item
    elif cfg['method'] != 'sequence_matters':
        raise ValueError('IE-SRGS has merge_priors; 3DSR targets must stay inside round driver')
    result = dict(schema='static_sr_wu_prior_bundle_v1', method=cfg['method'], manifest=cfg['manifest'],
                  teacher_variant=cfg.get('teacher_variant',cfg['method']), teacher_recipe=sr.get('recipe'),
                  scene=m['scene'], training_only=True, producer=[identity(__file__), identity(a.sr_receipt)],
                  model_identities=[identity(p) for p in a.model], entries=entries)
    if a.weights_receipt:
        result['producer'].append(identity(a.weights_receipt))
        parent = weights['provenance']['parent_checkpoint']
        if parent['sha256'] != weights['provenance']['parent_checkpoint_sha256']:
            raise ValueError('Map parent identity differs')
        checked(parent)
        result['model_identities'].append(parent)
        result['weight_producer_receipt'] = identity(a.weights_receipt)
    result['teacher_producer_receipt'] = identity(a.sr_receipt)
    if cfg['method'] == 'splatsure':
        for model in sr['model_identities']:
            checked(model)
        result['model_identities'] += sr['model_identities']
    write_new(a.output, result)
    verify_prior_index(a.output,cfg['manifest'],m,rows,('sr','weight') if cfg['method']=='splatsure' else ('sr',))

if __name__ == '__main__':
    main()
