"""Reference arms on the unchanged LR/SR recipe (modsr.py and cullretrain.py are imported, never modified).

  lr --image-source hr        the LR-stage recipe (author coarse + fine, same initialisation, sampler, schedule and
                              switches) trained on the HR training pixels with K_hr at HR size: HR-input reference of
                              the same backbone. Reads HR training pixels (upper-bound diagnostic, never a method).
  lr --image-source teacher   the same recipe on the frozen SwinIR x4 output of every training observation
                              (SR-then-reconstruct baseline). Needs --image-baseline-sr (a registered Bsync endpoint,
                              for its teacher index). Reads no HR pixels.
  sr --teacher-source hr      the SR stage with each SwinIR teacher replaced by the HR image of the same training
                              observation, SHA256-checked against the manifest (oracle-teacher diagnostic). Reads HR
                              training pixels.
  sr --densify-grad-threshold G   constant fine-stage densification threshold G in the SR stage (author 2e-4):
                              capacity probe, legal (no extra pixels).
  --hr-root DIR               with --image-source hr or --teacher-source hr: read training HR from DIR/<hr_path>
                              and check it against DIR/hr_full_storage.json (native track, where the prepared
                              directory keeps training HR absent by contract).
All other arguments, outputs and subcommands are modsr's; refs.json in --out records the reference switch.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import modsr  # noqa: E402  (imports cullretrain, never modifies it)

cr, native, rt = modsr.cr, modsr.native, modsr.rt
np, torch = modsr.np, modsr.torch


def teacher_entries(baseline_sr):
    base = cr.read(Path(baseline_sr) / 'config.json')
    index = cr.read(cr.ROOT / base['dependencies']['teacher']['path'])
    return base, index, {(e['camera'], int(e['frame'])): e for e in index['entries']}


def hr_index(hr_root):
    """(camera, frame) -> sha256 from a native hr_full_storage.json; None when no root is given."""
    if hr_root is None:
        return None
    index = cr.read(Path(hr_root) / 'hr_full_storage.json')
    return {(c, int(f)): digest for c, v in index['cameras'].items() for f, digest in v['hr_sha256'].items()}


def read_hr(path, digest):
    from PIL import Image
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != digest:
        raise ValueError(f'HR image hash differs from its index: {path}')
    with Image.open(io.BytesIO(data)) as image:
        return torch.from_numpy(np.asarray(image.convert('RGB')).copy()).permute(2, 0, 1).float() / 255


def patch_image_source(source, manifest_path, baseline_sr, hr_root=None):
    """Replace the LR training dataset of cullretrain.run_lr by HR pixels or by the frozen teachers (K_hr, HR size)."""
    base_cls = cr.N3DVPreparedDataset
    teacher = None
    digests = hr_index(hr_root) if source == 'hr' else None
    if source == 'teacher':
        _, index, teacher = teacher_entries(baseline_sr)
        if index['manifest_sha256'] != rt.sha(manifest_path):
            raise ValueError('teacher index belongs to another manifest')

    class Source(base_cls):
        def __init__(self, manifest, split='train', resolution='lr', device='cpu', cache=False):
            if split != 'train' or resolution != 'lr':
                raise ValueError(f'reference source replaces the train/lr dataset only, got {split}/{resolution}')
            super().__init__(manifest, 'train', 'hr', device=device, cache=cache)
            if teacher is not None:
                missing = [k for k in ((o['camera_id'], int(o['frame_index'])) for o in self.observations) if k not in teacher]
                if missing:
                    raise ValueError(f'{len(missing)} training observations without a teacher, e.g. {missing[:3]}')

        def __getitem__(self, index):
            if teacher is None and digests is None:
                return super().__getitem__(index)  # HR pixels, K_hr, HR size
            if teacher is None:  # HR from the separate native root, SHA-checked
                obs = self.observations[index]
                camera = self.cameras[obs['camera_id']]
                path = native.safe_child(Path(hr_root), obs['hr_path'])
                width, height = self.manifest['resolutions']['hr']
                image = read_hr(path, digests[(obs['camera_id'], int(obs['frame_index']))])
                if tuple(image.shape) != (3, height, width):
                    raise ValueError(f'HR size {tuple(image.shape)} differs from {(3, height, width)}: {path}')
                return {'image': image.to(self.device), 'image_path': str(path), 'camera_id': obs['camera_id'],
                        'frame_index': obs['frame_index'], 'time': obs['time'],
                        'K': np.asarray(camera['K_hr'], dtype=np.float64),
                        'w2c': np.asarray(camera['w2c'], dtype=np.float64),
                        'c2w': np.asarray(camera['c2w'], dtype=np.float64),
                        'width': width, 'height': height, 'split': self.split}
            obs = self.observations[index]  # teacher pixels only; the HR file is never opened
            camera = self.cameras[obs['camera_id']]
            path = native.safe_child(self.root, teacher[(obs['camera_id'], int(obs['frame_index']))]['relative_path'])
            width, height = self.manifest['resolutions']['hr']
            image = cr.read_teacher(path)
            if tuple(image.shape) != (3, height, width):
                raise ValueError(f'teacher size {tuple(image.shape)} differs from HR {(3, height, width)}: {path}')
            return {'image': image.to(self.device), 'image_path': str(path), 'camera_id': obs['camera_id'],
                    'frame_index': obs['frame_index'], 'time': obs['time'],
                    'K': np.asarray(camera['K_hr'], dtype=np.float64),
                    'w2c': np.asarray(camera['w2c'], dtype=np.float64),
                    'c2w': np.asarray(camera['c2w'], dtype=np.float64),
                    'width': width, 'height': height, 'split': self.split}

    cr.N3DVPreparedDataset = Source


def patch_teacher_source(baseline_sr, hr_root=None):
    """Serve the HR image of the same training observation wherever run_sr reads a SwinIR teacher."""
    base, index, teacher = teacher_entries(baseline_sr)
    manifest_path = cr.ROOT / base['data']['manifest']['path']
    if index['manifest_sha256'] != rt.sha(manifest_path):
        raise ValueError('teacher index belongs to another manifest')
    manifest = cr.load_manifest(manifest_path)
    root = Path(manifest['_root'])
    train = {(o['camera_id'], int(o['frame_index'])): o for o in manifest['observations'] if o['split'] == 'train'}
    digests = hr_index(hr_root)
    hr_base = Path(hr_root) if hr_root is not None else root
    redirect = {str(native.safe_child(root, e['relative_path'])): (
        native.safe_child(hr_base, train[key]['hr_path']), digests[key] if digests is not None else train[key]['hr_sha256'])
        for key, e in teacher.items()}
    if any(d is None for _, d in redirect.values()):
        raise ValueError('training HR without a SHA256 (native track: pass --hr-root)')

    def read_teacher(path):
        target, digest = redirect[str(path)]
        return read_hr(target, digest)

    cr.read_teacher = read_teacher
    return len(redirect)


def patch_densify_threshold(value):
    """cullretrain.topology interpolates the fine threshold from *_fine_init to *_after; set both to the probe value."""
    original = cr.configuration

    def configuration(plan):
        h, opt, pipe, sh = original(plan)
        opt.densify_grad_threshold_fine_init = opt.densify_grad_threshold_after = value
        return h, opt, pipe, sh

    cr.configuration = configuration


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    pre = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    pre.add_argument('--image-source', choices=('lr', 'hr', 'teacher'), default='lr')
    pre.add_argument('--image-baseline-sr')
    pre.add_argument('--teacher-source', choices=('swinir', 'hr'), default='swinir')
    pre.add_argument('--densify-grad-threshold', type=float)
    pre.add_argument('--hr-root')
    known, rest = pre.parse_known_args(argv)
    cmd = rest[0] if rest else None
    if known.image_source != 'lr' and cmd != 'lr':
        raise SystemExit('--image-source applies to the lr subcommand only')
    if (known.teacher_source != 'swinir' or known.densify_grad_threshold is not None) and cmd != 'sr':
        raise SystemExit('--teacher-source and --densify-grad-threshold apply to the sr subcommand only')
    if known.image_source == 'teacher' and not known.image_baseline_sr:
        raise SystemExit('--image-source teacher needs --image-baseline-sr')
    record = dict(kind='refs', image_source=known.image_source, image_baseline_sr=known.image_baseline_sr,
                  teacher_source=known.teacher_source, densify_grad_threshold=known.densify_grad_threshold,
                  hr_root=known.hr_root, argv=rest,
                  HR_training_pixels_read=known.image_source == 'hr' or known.teacher_source == 'hr',
                  sources={cr.rel(p): rt.sha(p) for p in (Path(__file__), Path(modsr.__file__), Path(cr.__file__))})
    if cmd == 'lr' and known.image_source != 'lr':
        manifest_path, _ = cr.plan_scene(cr.read(Path(rest[rest.index('--parent') + 1]) / 'config.json'))
        patch_image_source(known.image_source, manifest_path, known.image_baseline_sr, known.hr_root)
    if cmd == 'sr' and known.teacher_source != 'swinir':
        record['redirected_teachers'] = patch_teacher_source(rest[rest.index('--baseline-sr') + 1], known.hr_root)
    if cmd == 'sr' and known.densify_grad_threshold is not None:
        patch_densify_threshold(known.densify_grad_threshold)
    if '--out' in rest:
        out = Path(rest[rest.index('--out') + 1])
        out.mkdir(parents=True, exist_ok=True)
        rt.write_json(out / 'refs.json', record)
    modsr.main(rest)


if __name__ == '__main__':
    main()
