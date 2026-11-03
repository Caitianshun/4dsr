"""Qualitative comparison figure from selected crops (Claude 4dsr-ac, 2026-11-02). CPU only (Pillow).

Layout per selected frame: left, the HR frame with the zoom boxes drawn in; right, one row of crops per box, one column per
method (HR first, Ours second-to-last or as ordered by --methods), each crop with a border in its box colour and, optionally, the
window PSNR / LPIPS under it. Crops are pasted at native resolution (no resampling), so the figure shows the 8-bit renders as
scored; the overview is the only resampled image (Lanczos down-scale, boxes drawn after scaling). With --tile T, every crop is
enlarged by the integer factor T / size with nearest neighbour (no interpolation), so rows with 128 px and 192 px boxes align;
labels are computed on the native crops. A row may carry its own "variants" (method -> render variant) instead of the scene's.
Input: a figure JSON {"rows": [{"scene": "Cook07", "frame": 150, "boxes": [[x, y, size], ...]}, ...],
                      "methods": ["hr", "splatsure", ...], "variants": {"Cook07": {"sequence_matters": "sequence_matters_cull"}}}
Render directories come from the patch_maps metadata (<patch>/<Scene>_area4.json); HR from the manifest.
  /usr/bin/python3 -I make_figure.py --figure FIG.json --patch PATCH_DIR --out OUT.png [--metrics]
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[3]
LABEL = dict(hr='HR (GT)', ours='Ours', splatsure='SplatSuRe', iesrgs='IE-SRGS', sequence_matters='Sequence Matters', threedsr='3DSR',
             sr4d='SR4D', wu='Wu 4DGS', yang='Yang 4DGS', multi4d='Multi4D', localdygs='LocalDyGS', cem='CEM-4DGS', **{'4c4d': '4C4D'})
COLORS = [(230, 25, 75), (0, 130, 200), (60, 180, 75), (245, 130, 48)]
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
BOLD = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'


def rgb(path):
    with Image.open(path) as im:
        return im.convert('RGB')


def window_metrics(a, b):
    a = np.asarray(a, dtype=np.float64) / 255.0
    b = np.asarray(b, dtype=np.float64) / 255.0
    return -10 * math.log10(max(((a - b) ** 2).mean(), 1e-12))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--figure', required=True)
    p.add_argument('--patch', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--metrics', action='store_true')
    p.add_argument('--candidates', nargs='*', default=[], help='candidates.json files (window LPIPS for the labels)')
    p.add_argument('--tile', type=int, default=0, help='common displayed crop size (integer multiple of every box size)')
    a = p.parse_args()
    fig = json.loads((ROOT / a.figure).read_text())
    methods = fig['methods']
    cand = {}
    for c in a.candidates:
        for scene, rows in json.loads((ROOT / c).read_text()).items():
            cand.setdefault(scene, []).extend(rows)
    font, bold, small = ImageFont.truetype(FONT, 22), ImageFont.truetype(BOLD, 24), ImageFont.truetype(FONT, 19)
    pad, border, head, foot = 10, 4, 36, (26 if a.metrics else 0)
    blocks = []
    for row in fig['rows']:
        scene, f, boxes = row['scene'], row['frame'], row['boxes']
        meta = json.loads((ROOT / a.patch / f'{scene}_area4.json').read_text())
        mp = ROOT / meta['manifest']
        man = json.loads(mp.read_text())
        hr_path = next(mp.parent / o['hr_path'] for o in man['observations'] if o['camera_id'] == 'cam00' and int(o['frame_index']) == f)
        hr = rgb(hr_path)
        size = boxes[0][2]
        disp = a.tile or size
        if disp % size or any(b[2] != size for b in boxes):
            raise SystemExit(f'--tile {disp} is not an integer multiple of the box size {size} (or mixed box sizes in a row)')
        tile = disp + 2 * border
        rows_h = len(boxes) * (tile + foot) + (len(boxes) - 1) * pad
        scale = rows_h / hr.height
        ov = hr.resize((round(hr.width * scale), rows_h), Image.LANCZOS)
        d = ImageDraw.Draw(ov)
        for k, (x, y, s) in enumerate(boxes):
            d.rectangle([x * scale, y * scale, (x + s) * scale, (y + s) * scale], outline=COLORS[k], width=4)
        variants = row.get('variants', fig.get('variants', {}).get(scene, {}))
        crops = []
        for k, (x, y, s) in enumerate(boxes):
            line = []
            ref = hr.crop((x, y, x + s, y + s))
            for m in methods:
                if m == 'hr':
                    im = ref
                    lab = ''
                else:
                    v = variants.get(m, m)
                    im = rgb(ROOT / meta['methods'][v] / f'cam00/{f:04d}.png').crop((x, y, x + s, y + s))
                    lp = None
                    for r in cand.get(scene, []):
                        if r['frame'] == f:
                            for w in r['windows']:
                                if (w['x'], w['y'], w['size']) == (x, y, s) and v in w['metrics']:
                                    lp = w['metrics'][v]['lpips']
                    lab = f'{window_metrics(im, ref):.2f} dB' + (f' / {lp:.3f}' if lp is not None else '')
                t = Image.new('RGB', (tile, tile), COLORS[k])
                t.paste(im.resize((disp, disp), Image.NEAREST), (border, border))
                line.append((t, lab))
            crops.append(line)
        blocks.append((ov, crops, scene, f))
    ncol = len(methods)
    tile_w = blocks[0][1][0][0][0].width
    ov_w = max(b[0].width for b in blocks)
    W = pad + ov_w + pad + ncol * (tile_w + pad)
    H = pad + head + sum(b[0].height + pad for b in blocks)
    canvas = Image.new('RGB', (W, H), 'white')
    d = ImageDraw.Draw(canvas)
    x0 = pad + ov_w + pad
    for j, m in enumerate(methods):
        lab = LABEL.get(m, m)
        tw = d.textlength(lab, font=bold if m == 'ours' else font)
        d.text((x0 + j * (tile_w + pad) + (tile_w - tw) / 2, pad), lab, fill='black', font=bold if m == 'ours' else font)
    y = pad + head
    for ov, crops, scene, f in blocks:
        canvas.paste(ov, (pad + (ov_w - ov.width) // 2, y))
        for k, line in enumerate(crops):
            yy = y + k * (tile_w + foot + pad)
            for j, (t, lab) in enumerate(line):
                xx = x0 + j * (tile_w + pad)
                canvas.paste(t, (xx, yy))
                if a.metrics and lab:
                    tw = d.textlength(lab, font=small)
                    d.text((xx + (tile_w - tw) / 2, yy + tile_w + 2), lab, fill='black', font=small)
        y += ov.height + pad
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    canvas.save(out.with_suffix('.pdf'), resolution=300)
    print(json.dumps(dict(out=str(out.relative_to(ROOT)), size=list(canvas.size), rows=[(s, f) for _, _, s, f in blocks])))


if __name__ == '__main__':
    main()
