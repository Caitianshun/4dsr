"""Spec of the x4 models to time and render, from inventory.json (Claude 4dsr-ac, 2026-11-02).

A cull variant of a static-SR baseline is rendered only where the labelled main-table cull row differs from the uncut row
(|dPSNR| > 0.005 dB or |dLPIPS| > 0.0005); elsewhere the cull leaves the cam00 metrics unchanged (x4 table, 10-30).
  /usr/bin/python3 -I make_spec.py --inventory tmp/claude_qualfig_fps_20261102/inventory.json --out experiments/.../spec_x4.json
"""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
V2 = 'output/dynamic_sr_modules_20261010/static_sr_v2_20261026'
S4 = 'output/dynamic_sr_modules_20261010/sr4d_area4_20261028'


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--inventory', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    inv = json.loads((ROOT / a.inventory).read_text())
    spec = {}
    for short, v in inv.items():
        full, m = v['scene'], v['methods']
        manifest = f'data/dynamic_sr/full_time_20261007/prepared/{full}_area4/manifest_train_ready_seed20261007.json'
        oi = json.loads((ROOT / m['ours']['renders_cam00'] / 'render_index.json').read_text())
        if sha(ROOT / manifest) != oi['manifest_sha256']:
            raise SystemExit('manifest differs from the main-table renders: ' + manifest)
        s = dict(scene=full, manifest=manifest,
                 ours=dict(checkpoint=m['ours']['checkpoint'], residuals=m['ours']['residuals'], renders=m['ours']['renders_cam00']),
                 wu=dict(checkpoint=m['wu']['checkpoint'], renders=m['wu']['renders_cam00']))
        for method in ('splatsure', 'iesrgs', 'sequence_matters', 'threedsr'):
            r = m[method]
            tag = r['tag']
            nat, cull = r['main_table_test'], r.get('cull_test')
            need = bool(cull) and (abs(cull['psnr'] - nat['psnr']) > 0.005 or abs(cull['lpips'] - nat['lpips']) > 0.0005)
            s[method] = dict(config=f'experiments/dynamic_sr_modules_20261010/static_sr_v2_20261026/configs/{short}/{method}.json',
                             checkpoint=r['checkpoint'], checkpoint_sha256=r['checkpoint_sha256'], tag=tag,
                             main_table_render_index=f'{V2}/eval/{short}/{tag}_render_test/render_index.json',
                             main_table_cull_render_index=f'{V2}/eval/{short}/{tag}_cull_render_test/render_index.json' if cull else None,
                             main_table_test=nat, cull_test=cull, render_cull=need)
        r = m['sr4d']
        s['sr4d'] = dict(run=r['run_dir'], tag=r['tag'], main_table_test=r['main_table_test'],
                         main_table_render_index=f'{S4}/eval/{short}/{r["tag"]}_render_test/render_index.json')
        spec[short] = s
    (ROOT / a.out).write_text(json.dumps(spec, indent=1) + '\n')
    print(json.dumps({k: [mm for mm in ('splatsure', 'iesrgs', 'sequence_matters', 'threedsr') if v[mm]['render_cull']] for k, v in spec.items()}))


if __name__ == '__main__':
    main()
