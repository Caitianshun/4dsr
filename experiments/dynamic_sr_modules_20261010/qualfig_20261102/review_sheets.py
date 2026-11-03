"""Grid-layout review sheets for the crop candidates of select_crops.py (Claude 4dsr-ac, 2026-11-02). CPU only (NumPy + Pillow).

select_crops.py writes one long strip per candidate (about 5000 px wide), which a vision reviewer sees heavily downscaled. This
script re-lays the same tiles as a 4-column grid with every tile at about 384 px (nearest-neighbour), keeping the letter
shuffle of <sel>/blind/key.json unchanged. Selection itself is untouched.

Outputs:
  <blind_out>/bNN.png        blind sheets with neutral names (HR first as reference, then letters A, B, ...)
  <sel>/blind_grid_map.json  bNN -> candidate id (kept outside the reviewer's read path)
  <sel>/sheets_grid/<id>.png labelled sheets (method, window PSNR / LPIPS) for Claude
  /usr/bin/python3 -I review_sheets.py --sel output/.../selection_w192 --patch output/.../patch --blind-out tmp/.../blind_w192
"""
import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[3]
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
LABEL = dict(ours='Ours', splatsure='SplatSuRe', iesrgs='IE-SRGS', sequence_matters='Sequence Matters', threedsr='3DSR', sr4d='SR4D',
             wu='Wu 4DGS', yang='Yang 4DGS', **{'4c4d': '4C4D'}, multi4d='Multi4D', localdygs='LocalDyGS', cem='CEM-4DGS')
COLS, TILE, PAD, LH = 4, 384, 10, 40


def grid(tiles, labels, path):
    font = ImageFont.truetype(FONT, 26)
    rows = (len(tiles) + COLS - 1) // COLS
    im = Image.new('RGB', (COLS * (TILE + PAD) + PAD, rows * (TILE + LH + PAD) + PAD), 'white')
    d = ImageDraw.Draw(im)
    for i, (t, lab) in enumerate(zip(tiles, labels)):
        x, y = PAD + (i % COLS) * (TILE + PAD), PAD + (i // COLS) * (TILE + LH + PAD)
        d.text((x + 2, y + 4), lab, fill='black', font=font)
        im.paste(t.resize((TILE, TILE), Image.NEAREST), (x, y + LH))
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--sel', required=True)
    p.add_argument('--patch', required=True)
    p.add_argument('--blind-out', required=True)
    a = p.parse_args()
    sel = ROOT / a.sel
    cands = json.loads((sel / 'candidates.json').read_text())
    key = json.loads((sel / 'blind' / 'key.json').read_text())
    mapping, n = {}, 0
    for scene, rows in cands.items():
        meta = json.loads((ROOT / a.patch / f'{scene}_area4.json').read_text())
        manifest_path = ROOT / meta['manifest']
        manifest = json.loads(manifest_path.read_text())
        hr = {int(o['frame_index']): manifest_path.parent / o['hr_path'] for o in manifest['observations'] if o['camera_id'] == 'cam00'}
        for r in rows:
            f = r['frame']
            for k, win in enumerate(r['windows']):
                x, y, w = win['x'], win['y'], win['size']
                bid = f'{scene}_f{f:03d}_w{k}'

                def tile(path):
                    with Image.open(path) as im:
                        return im.convert('RGB').crop((x, y, x + w, y + w))

                letters = sorted(key[bid])
                methods = [key[bid][c] for c in letters]
                tiles = [tile(hr[f])] + [tile(ROOT / meta['methods'][m] / f'cam00/{f:04d}.png') for m in methods]
                n += 1
                name = f'b{n:02d}'
                mapping[name] = bid
                grid(tiles, ['Reference (HR)'] + letters, ROOT / a.blind_out / f'{name}.png')
                per = win['metrics']
                order = ['ours'] + [m for m in methods if m != 'ours']
                tiles = [tiles[0]] + [tiles[1 + methods.index(m)] for m in order]
                labs = ['HR'] + [f"{LABEL[m.replace('_cull', '')]}{'*' if m.endswith('_cull') else ''} "
                                 f"{per[m]['psnr']:.1f}/{per[m]['lpips']:.3f}" for m in order]
                grid(tiles, labs, sel / 'sheets_grid' / f'{bid}.png')
    (sel / 'blind_grid_map.json').write_text(json.dumps(mapping, indent=1) + '\n')
    print(len(mapping), 'sheets')


if __name__ == '__main__':
    main()
