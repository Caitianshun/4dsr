"""SplatSuRe selection maps computed from OUR LR parent (Claude, 2026-11-12; SplatSuRe on our base configuration).

The frozen producer static_sr_wu_20261022/methods/splatsure_prepare.py and the pinned official rasterizer (83814a5) are used
unchanged, with the same calls as static_sr_v2_20261026/sv2.py weights. Only the backend differs: it loads the cullretrain
LR parent that the base model starts from. WuBackend refuses colour-corrected parents; the maps depend only on geometry
(true radii, contributions, fidelity/closest renders on a black background), so the LR-stage affine is not used here.
  sps_weights.py --parent CK --out DIR"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
WU = ROOT / 'experiments/static_sr_wu_20261022'
CULL = ROOT / 'experiments/dynamic_sr_cull_retrain_20261009'


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--parent', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    parent = Path(a.parent) if Path(a.parent).is_absolute() else ROOT / a.parent
    out = Path(a.out) if Path(a.out).is_absolute() else ROOT / a.out
    done = out / 'receipt.json'
    if done.is_file() and json.loads(done.read_text()).get('status') == 'prepared':
        if json.loads(done.read_text())['provenance']['parent_checkpoint_sha256'] != sha(parent):
            raise SystemExit('existing maps belong to a different LR parent')
        print(json.dumps(dict(status='prepared', reused=str(done.relative_to(ROOT)))), flush=True)
        return
    for q in (CULL, WU):
        sys.path.insert(0, str(q))
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    import common
    import backend
    common.ROOT = backend.ROOT = ROOT
    import cullretrain as cr
    from scene.gaussian_model import GaussianModel
    from n3dv_data import N3DVPreparedDataset
    from methods.splatsure_prepare import load_official_rasterizer, prepare_splatsure_cache

    class OurParent(backend.WuBackend):
        def __init__(self, path):
            self.torch = torch
            self.cr, self.native, self.prefix = cr, cr.native, cr.prefix
            payload = torch.load(path, map_location='cuda', weights_only=False)
            plan = payload['plan']
            self.h, self.opt, self.pipe, sh = cr.configuration(plan)
            self.manifest_path, self.manifest = cr.plan_scene(plan)
            self.dataset = N3DVPreparedDataset(self.manifest, 'train', 'lr', device='cpu', cache=False)
            self.observations = self.dataset.observations
            self.indices = {(o['camera_id'], o['frame_index']): i for i, o in enumerate(self.dataset.observations)}
            self.g = GaussianModel(sh, self.h)
            self.g._deformation = self.g._deformation.cuda()
            self.g.restore(payload['model'], self.opt)
            self.g._deformation_accum = payload['deformation_accum']
            self.g.optimizer.zero_grad(set_to_none=True)
            self.background = torch.ones(3, device='cuda') if cr.white_background(plan) else torch.zeros(3, device='cuda')
            self.points = int(self.g.get_xyz.shape[0])

    b = OurParent(parent)
    source = ROOT / 'tmp/static_sr_wu_20261022/upstream/splatsure-rasterizer'
    # the training account does not own the checkout; trust only this directory so the producer's pin checks can run
    os.environ.update(GIT_CONFIG_COUNT='1', GIT_CONFIG_KEY_0='safe.directory', GIT_CONFIG_VALUE_0=str(source))
    api = load_official_rasterizer(source, ROOT / 'output/static_sr_wu_20261022/runtime/splatsure')
    digest = sha(parent)
    receipt = prepare_splatsure_cache(b, b.manifest_path, out, api,
                                      dict(parent_checkpoint_sha256=digest,
                                           parent_checkpoint=dict(path=str(parent.relative_to(ROOT)), sha256=digest),
                                           backend=dict(path=str(Path(__file__).resolve().relative_to(ROOT)), sha256=sha(__file__)),
                                           backend_kind='cullretrain LR parent (our base configuration)', parent_points=b.points))
    diag = receipt['frame_diagnostics']
    print(json.dumps(dict(status=receipt['status'], maps=len(receipt['records']), seconds=round(receipt['seconds']),
                          mean_weight=sum(r['mean'] for r in receipt['records']) / len(receipt['records']),
                          fewer_than_three=sum(d['fewer_than_three_views_fraction'] for d in diag) / len(diag))), flush=True)


if __name__ == '__main__':
    main()
