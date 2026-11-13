"""SplatSuRe on our base configuration, SR stage (Claude, 2026-11-12; follow-up to the attribution review).

Outermost layer around geoprobe_20261017/modkit6.py, the training entry of run_modkit7_remote.py for every areaTm row, so
the run keeps the base model's whole chain: area operator, srfix --mode fix, --prune-gate none, --affine, D-SSIM 0.2,
6000 steps, the same LR parent, teacher index and same-time three-view schedule. Only the objective of cullretrain.run_sr
changes (two exact source replacements, compiled in the cullretrain namespace):
  - every view of a row gets SplatSuRe's loss (static_sr_wu_20261022/methods/splatsure.py, frozen): (1 - gamma) LR term +
    gamma SR term, each 0.8 L1 + 0.2 D-SSIM, the SR term weighted per pixel by the selection map (no normalisation);
    gamma 0.4; the row loss is the mean over its three views (as in static_sr_v2_20261026/sv2.py), plus the backbone's
    time-plane regulariser;
  - the LR term uses our area operator (losses.degradation as patched by modkit), and the rendered RGB is clamped to [0, 1]
    before the per-camera colour affine (SplatSuRe's renderer clamps its output);
  - teachers and selection maps are read for all three views.
The maps come from sps_weights.py (our LR parent). sps.json in --out records the switches, hashes and replaced source.
For Sear Steak / Flame Salmon, --newscene-index maps the teacher paths exactly as newscenes_reg_20261103/teacher_paths_new.py
does for its direct entries (same resolver, inventory checks and torch-hub directory); nothing else changes.
  sps_layer.py --weights RECEIPT [--gamma 0.4] [--newscene-index INDEX] -- <modkit6.py argv>"""
import argparse
import hashlib
import inspect
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
MOD = HERE.parent
ROOT = MOD.parents[1]
KIT = MOD / 'geoprobe_20261017'
CULL = ROOT / 'experiments/dynamic_sr_cull_retrain_20261009'
WU = ROOT / 'experiments/static_sr_wu_20261022'
CULL_SHA = '83f0b4d983ab2c4944e801f186bc6db45490fd909916fa260e1d8dc34b287ee8'  # cullretrain.py of the base-model runs

OLD_FETCH = "        teachers = [read_teacher(teacher_paths[keys[j]]) for j in (1, 2)]\n"
NEW_FETCH = "        teachers = [read_teacher(teacher_paths[keys[j]]) for j in (0, 1, 2)] + [SPS['weight'](keys[j]) for j in (0, 1, 2)]\n"
OLD_LOSS = """            rgbs = [p['render'] for p in packages]
            if affine is not None:
                rgbs = [affine.apply(rgb, obs['camera_id']) for rgb, obs in zip(rgbs, items)]
            lr_values = [losses.lr_loss(rgb, lr) for rgb, lr in zip(rgbs, lrs)]
            lr_term = sum(w * v for w, v in zip(definition['lr_weights'], lr_values))
            sr_values = [losses.sr_loss(rgbs[i], t) for i, t in zip((1, 2), teachers)]
            if a.dssim:
                down = losses.degradation(rgbs[0], lrs[0].shape[-2:])
                lr_term = (1 - a.dssim) * lr_term + a.dssim * (1 - ssim(down[None], lrs[0][None]))
                sr_mix = [(1 - a.dssim) * v + a.dssim * (1 - ssim(rgbs[i][None], t[None])) for v, i, t in zip(sr_values, (1, 2), teachers)]
                sr_term = a.lambda_sr * (sr_mix[0] + sr_mix[1])
            else:
                sr_term = a.lambda_sr * (sr_values[0] + sr_values[1])
"""
NEW_LOSS = """            rgbs = [p['render'].clamp(0, 1) for p in packages]
            if affine is not None:
                rgbs = [affine.apply(rgb, obs['camera_id']) for rgb, obs in zip(rgbs, items)]
            sps = [SPS['loss'](rgb, lr, t, w, SPS['config'], downsample=losses.degradation)
                   for rgb, lr, t, w in zip(rgbs, lrs, teachers[:3], teachers[3:])]
            lr_values = [v['lr_l1'] for v in sps]
            sr_values = [v['sr_l1'] for v in sps]
            lr_term = sum((1 - SPS['config'].gamma) * v['lr_loss'] for v in sps) / len(sps)
            sr_term = sum(SPS['config'].gamma * v['sr_loss'] for v in sps) / len(sps)
"""
OLD_RECORD = "    record = dict(kind='sr', arm=a.arm,"
NEW_RECORD = "    record = dict(kind='sr', objective=SPS['describe'], arm=a.arm,"


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    argv = sys.argv[1:]
    if '--' not in argv:
        raise SystemExit(__doc__)
    k = argv.index('--')
    p = argparse.ArgumentParser()
    p.add_argument('--weights', required=True)
    p.add_argument('--gamma', type=float, default=0.4)
    p.add_argument('--newscene-index')
    a = p.parse_args(argv[:k])
    rest = argv[k + 1:]
    if '--' not in rest or 'sr' not in rest:
        raise SystemExit('expected <modkit6 options> -- --mode fix cullretrain sr ...')
    sr_args = rest[rest.index('sr'):]
    out = Path(sr_args[sr_args.index('--out') + 1])
    out = out if out.is_absolute() else ROOT / out
    if (out / 'complete.json').exists() and (out / 'sps.json').exists():
        print(json.dumps(dict(sps='already complete', out=str(out.relative_to(ROOT)))), flush=True)
        return
    if sha(CULL / 'cullretrain.py') != CULL_SHA:
        raise SystemExit('cullretrain.py differs from the base-model runs (83f0b4d9)')
    for q in (KIT, MOD, CULL, WU):
        if str(q) not in sys.path:
            sys.path.insert(0, str(q))
    import cullretrain as cr
    import torch
    from methods.splatsure import SplatSuReConfig, splatsure_loss

    receipt_path = Path(a.weights) if Path(a.weights).is_absolute() else ROOT / a.weights
    receipt = json.loads(receipt_path.read_text())
    if receipt.get('status') != 'prepared' or len(receipt['records']) != 6000:
        raise SystemExit('selection maps incomplete')
    parent = sr_args[sr_args.index('--parent-checkpoint') + 1]
    if receipt['provenance']['parent_checkpoint_sha256'] != sha(ROOT / parent):
        raise SystemExit('selection maps belong to a different LR parent')
    root = receipt_path.parent
    maps = {(r['camera_id'], int(r['frame_index'])): (root / r['path'], r['sha256']) for r in receipt['records']}

    def check(item):
        path, digest = item
        if sha(path) != digest:
            raise SystemExit(f'selection map bytes differ: {path}')
    with ThreadPoolExecutor(8) as pool:
        list(pool.map(check, maps.values()))

    config = SplatSuReConfig(gamma=a.gamma, lr_operator='area')
    describe = dict(name='splatsure_on_base_config', gamma=config.gamma, lambda_dssim=config.lambda_dssim,
                    lr_term='our area operator (losses.degradation)', rgb_clamp='before colour affine', views='all three of each row',
                    weights=dict(path=str(receipt_path.relative_to(ROOT)), sha256=sha(receipt_path)))
    cr.SPS = dict(config=config, loss=splatsure_loss, describe=describe,
                  weight=lambda key: torch.load(maps[(key[0], int(key[1]))][0], map_location='cpu', weights_only=True))
    src = inspect.getsource(cr.run_sr)
    for old, new in ((OLD_FETCH, NEW_FETCH), (OLD_LOSS, NEW_LOSS), (OLD_RECORD, NEW_RECORD)):
        if src.count(old) != 1:
            raise SystemExit('run_sr source differs from the expected block: ' + old[:60])
        src = src.replace(old, new)
    exec(compile(src, '<sps_layer run_sr>', 'exec'), cr.__dict__)

    if a.newscene_index:
        sys.path.insert(0, str(MOD / 'newscenes_reg_20261103'))
        import common_new as cn
        import rt
        import teacher_paths_new as tpn
        index = cn.local(a.newscene_index)
        inventory = cn.read(index)
        scene = inventory['scene']
        cn.require(scene in cn.NEW, 'Only new scenes use the teacher-path map')
        manifest_path = cn.PREP / f'{scene}_area4/manifest_train_ready_seed{cn.SEED}.json'
        files = tpn.teacher_files(index, cn.read(manifest_path), manifest_path)
        mapping = {row['relative_path']: cn.local(f['path']) for row, f in zip(inventory['entries'], files)}
        cr.native.safe_child = tpn.resolver(cr.native.safe_child, manifest_path.parent, mapping)
        original_check = rt.check_runtime

        def check_runtime():
            value = original_check()
            torch.hub.set_dir(str(cn.OUT / 'model_cache/torch_home'))
            return value
        rt.check_runtime = check_runtime
        describe['newscene_teacher_map'] = dict(index=str(index.relative_to(ROOT)), files=len(mapping),
                                                adapter=str((MOD / 'newscenes_reg_20261103/teacher_paths_new.py').relative_to(ROOT)))

    import modkit6
    sys.argv = [str(KIT / 'modkit6.py')] + rest
    modkit6.main()
    record = dict(describe, argv=rest, run_sr_sha256=hashlib.sha256(src.encode()).hexdigest(), maps=len(maps),
                  mean_map_weight=sum(r['mean'] for r in receipt['records']) / len(receipt['records']),
                  sources={str(f.relative_to(ROOT)): sha(f) for f in (Path(__file__).resolve(), CULL / 'cullretrain.py',
                                                                      WU / 'methods/splatsure.py', KIT / 'modkit6.py')})
    (out / 'sps.json').write_text(json.dumps(record, indent=1))
    (out / 'sps_run_sr.py').write_text(src)
    print(json.dumps(dict(sps='done', out=str(out.relative_to(ROOT)))), flush=True)


if __name__ == '__main__':
    main()
