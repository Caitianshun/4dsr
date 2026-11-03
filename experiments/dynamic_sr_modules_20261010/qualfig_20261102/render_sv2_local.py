"""cam00 PNG renders (all 300 frames) of a static-SR v2 baseline on the local GPUs (Claude 4dsr-ac, 2026-11-02).

Same code path as the main-table renders made on the A100 / CTS (static_sr_v2_20261026): sv2d.render_views for the uncut
model, sv2_cullrender.render_culled for the labelled post-hoc-cull variant; the task root is the project root, so the
frozen migration modules and the local rasterizer builds (output/static_sr_wu_20261022/runtime, sm_86 + sm_120) are used.
Only rendering: no scoring (patch_maps.py scores later), nothing is deleted. Afterwards the PNG sha256 of every frame is
compared with the main-table render_index of the same checkpoint (renders from another GPU need not be byte-identical).
  render_sv2_local.py --spec SPEC.json --scene S --method M [--cull] --out RENDER_DIR
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from argparse import Namespace
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
V2 = ROOT / 'experiments/dynamic_sr_modules_20261010/static_sr_v2_20261026'
sys.path.insert(0, str(V2))


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def git_safe_directory():
    """The 3DSR renderer checks its upstream clone with `git rev-parse HEAD` / `git diff --quiet HEAD`; under the training service
    the clone belongs to another user, so git refuses it (dubious ownership). Mark exactly that clone as safe for this process."""
    import os
    os.environ['GIT_CONFIG_COUNT'] = '1'
    os.environ['GIT_CONFIG_KEY_0'] = 'safe.directory'
    os.environ['GIT_CONFIG_VALUE_0'] = str(ROOT / 'tmp/static_sr_wu_20261022/upstream/threedsr')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--spec', required=True)
    p.add_argument('--scene', required=True)
    p.add_argument('--method', required=True, choices=('splatsure', 'iesrgs', 'sequence_matters', 'threedsr'))
    p.add_argument('--cull', action='store_true')
    p.add_argument('--out', required=True)
    a = p.parse_args()
    e = json.loads((ROOT / a.spec).read_text())[a.scene][a.method]
    if sha(ROOT / e['checkpoint']) != e['checkpoint_sha256']:
        raise SystemExit('checkpoint differs from the main-table render: ' + e['checkpoint'])
    git_safe_directory()
    import sv2d as sv2
    sv2.setup(ROOT)
    out = ROOT / a.out
    if a.cull:
        import sv2_cullrender
        sv2_cullrender.render_culled(Namespace(scope='test'), e['config'], e['checkpoint'], a.out)
        ref_index = e.get('main_table_cull_render_index')
    else:
        sv2.render_views(Namespace(task_root=str(ROOT), config=str(ROOT / e['config']), checkpoint=e['checkpoint'], scope='test', out=str(out)))
        ref_index = e.get('main_table_render_index')
    idx = json.loads((out / 'render_index.json').read_text())
    same = None
    if ref_index and (ROOT / ref_index).is_file():
        ref = {(r['camera_id'], int(r['frame'])): r['render_sha256'] for r in json.loads((ROOT / ref_index).read_text())['records']}
        same = sum(int(ref.get((r['camera_id'], int(r['frame']))) == r['render_sha256']) for r in idx['records'])
    import torch
    note = dict(scene=a.scene, method=a.method, cull=a.cull, frames=len(idx['records']), main_table_render_index=ref_index,
                pngs_identical_to_main_table=same, gpu=torch.cuda.get_device_name(), checkpoint=e['checkpoint'],
                checkpoint_sha256=e['checkpoint_sha256'], script=dict(path=str(Path(__file__).relative_to(ROOT)), sha256=sha(__file__)))
    (out / 'local_render.json').write_text(json.dumps(note, indent=1) + '\n')
    print(json.dumps(note), flush=True)


if __name__ == '__main__':
    main()
