"""Fixed evaluation galleries. Display quantization never enters metric computation."""
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from cg_common import *

METHODS = ['C1', 'Jperm', 'B2perm', 'R', 'G', 'RG']
ROI = ROOT / 'output/dynamic_sr_prior_diagnosis_20260929/spectrum/roi_protocol.json'
DATA = ROOT / 'data/dynamic_sr/n3dv_prepared/cook_spinach'

def image_rgb(path):
    with Image.open(path) as im:
        return np.asarray(im.convert('RGB'), dtype=np.float32) / 255.

def displayed_raw(path):
    receipt = read(path.with_suffix('.json'))
    assert receipt['sha256'] == sha(path)
    with np.load(path, allow_pickle=False) as z:
        return np.clip(z['rgb_raw'].transpose(1, 2, 0), 0, 1)

def canvas(panels, columns, cell, destination, title):
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 15)
    width, height = cell
    rows = (len(panels) + columns - 1) // columns
    out = Image.new('RGB', (columns * (width + 8) + 8, rows * (height + 32) + 40), 'white')
    draw = ImageDraw.Draw(out); draw.text((8, 8), title, fill='black', font=font)
    for i, (label, rgb) in enumerate(panels):
        x = 8 + (i % columns) * (width + 8)
        y = 36 + (i // columns) * (height + 32)
        img = Image.fromarray(np.rint(np.clip(rgb, 0, 1) * 255).astype(np.uint8))
        img.thumbnail((width, height), Image.Resampling.BOX)
        draw.text((x, y), label, fill='black', font=font)
        out.paste(img, (x, y + 23))
    destination.parent.mkdir(parents=True, exist_ok=True); out.save(destination)

def main(a):
    roi = read(ROI); protocol = read(OUT / 'protocol.json')
    evidence, products = {}, []
    for camera in ['cam00', 'cam01', 'cam02']:
        for frame in roi['fixed_still_frames']:
            stem = f'{camera}_{frame:04d}'
            hr = DATA / 'hr' / camera / f'{frame:04d}.png'
            lr = DATA / 'lr' / camera / f'{frame:04d}.png'
            reference = image_rgb(hr)
            observed = image_rgb(lr)
            # The figure reference has actual target LR: diagnostic, not a novel-view method.
            enlarged = np.asarray(Image.fromarray(np.rint(observed * 255).astype(np.uint8)).resize(
                (1344, 1008), Image.Resampling.BICUBIC), dtype=np.float32) / 255.
            evidence[str(hr.relative_to(ROOT))] = sha(hr); evidence[str(lr.relative_to(ROOT))] = sha(lr)
            refs = [('HR reference', reference), ('Observed LR bicubic (diagnostic)', enlarged)]
            teacher = DATA / 'sr_swinir_x4' / camera / f'{frame:04d}.png'
            if camera == 'cam02':
                assert teacher.exists(); refs.append(('Frozen SR teacher (train view)', image_rgb(teacher)))
                evidence[str(teacher.relative_to(ROOT))] = sha(teacher)
            parent = OUT / 'evaluation/U6000/extra/floats' / f'{stem}.npz'
            assert parent.exists(), parent
            refs.append(('U6000 parent', displayed_raw(parent))); evidence[str(parent.relative_to(ROOT))] = sha(parent)
            for repeat in ['1', '2']:
                panels = list(refs)
                for method in METHODS:
                    label = f'r{repeat}_{method}'
                    raw = OUT / 'evaluation' / label / 'extra/floats' / f'{stem}.npz'
                    assert raw.exists(), raw
                    receipt = read(OUT / 'runs' / label / 'complete.json')
                    assert read(raw.with_suffix('.json'))['checkpoint_sha256'] == receipt['checkpoint']['sha256']
                    panels.append((f'{method} suffix {repeat}', displayed_raw(raw)))
                    evidence[str(raw.relative_to(ROOT))] = sha(raw)
                full = a.out / f'{stem}_suffix{repeat}_full.png'
                canvas(panels, 3, (448, 336), full, f'{camera} frame {frame} | fixed full scene, display reduced')
                products.append(entry(full))
                for region, (x0, y0, x1, y1) in roi['regions_by_camera_xyxy_exclusive'][camera].items():
                    cropped = [(label, rgb[y0:y1, x0:x1]) for label, rgb in panels]
                    target = a.out / f'{stem}_suffix{repeat}_{region}.png'
                    # Preserve native crop pixels; use only a larger label area for short crops.
                    canvas(cropped, 3, (max(x1-x0, 320), y1-y0), target,
                           f'{camera} frame {frame} | {region} | native crop')
                    products.append(entry(target))
    write(a.out / 'index.json', dict(status='generated_pending_visual_review',
          protocol_sha256=sha(OUT/'protocol.json'), roi=entry(ROI), source=entry(HERE/'figures.py'),
          fixed_cameras=['cam00','cam01','cam02'], fixed_frames=roi['fixed_still_frames'],
          prediction_quality_selection=False, same_coordinates_for_all_methods=True,
          parameter_updates=0, gpu_forwards=0, images=products, source_hashes=evidence,
          interpretation='cam00/01 novel views; observed target LR is diagnostic only. cam02 is a training view. '
            'References and crops are pre-existing registered choices. PNG quantization and display resizing '
            'are confined to galleries; all quality/FFT calculations retain their original floating values.'))

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--out', type=Path, default=OUT/'figures')
    main(p.parse_args())
