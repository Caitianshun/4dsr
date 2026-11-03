"""Round-2 crop candidates restricted to motion regions (Claude 4dsr-ac, 2026-11-02). CPU only (NumPy + Pillow).

Round 1 (select_crops.py, 192 px) showed that windows on static seams win by metrics but look the same, while the visible wins
were in moving hands / tools. Same pass rule and score as select_crops.py (Ours >= best competitor + 1.0 dB PSNR, <= - 0.02
LPIPS, SSIM >= every competitor, HR gradient >= 60th percentile), plus dynamic share >= --min-dyn. Per scene the frames are ranked
by their best such window; the top --frames frames (>= 20 frames apart) give one window each.
Outputs: <out>/candidates.json, <out>/blind/key.json, blind grid sheets <blind_out>/mNN.png, <out>/blind_grid_map.json and
labelled grid sheets <out>/sheets_grid/<id>.png.
  /usr/bin/python3 -I select_motion.py --patch output/.../patch --out output/.../selection_m128 --blind-out tmp/.../blind_m128
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import review_sheets as RS  # noqa: E402
import select_crops as SC  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--patch', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--blind-out', required=True)
    p.add_argument('--win', type=int, default=128)
    p.add_argument('--min-dyn', type=float, default=0.4)
    p.add_argument('--frames', type=int, default=4)
    a = p.parse_args()
    out, w = SC.ROOT / a.out, a.win
    rng = random.Random(20261103)
    cands, key, mapping, n = {}, {}, {}, 0
    for jf in sorted((SC.ROOT / a.patch).glob('*_area4.json')):
        meta = json.loads(jf.read_text())
        scene = meta['scene']
        z = np.load(jf.with_suffix('.npz'))
        names = list(meta['methods'])
        P0, S0, L0 = SC.metric(z, 'ours', w)
        comp = SC.competitors(z, names, w)
        Pa, Sa, La, _ = SC.best(comp, SC.SR + SC.DYN)
        grad = z[f'hr__grad_{w}']
        dyn = z[f'hr__dyn_{w}'].astype(np.float32)
        dP, dL = P0 - Pa, La - L0
        ok = (dP >= SC.DP) & (dL >= SC.DL) & (S0 >= Sa) & (grad >= np.percentile(grad, SC.GQ)) & (dyn >= a.min_dyn)
        score = np.where(ok, np.minimum(dP / SC.DP, dL / SC.DL), -np.inf)
        flat = score.reshape(score.shape[0], -1)
        best_idx = flat.argmax(1)
        best_val = flat[np.arange(flat.shape[0]), best_idx]
        chosen = []
        for f in np.argsort(best_val)[::-1]:
            if not np.isfinite(best_val[f]):
                break
            if all(abs(int(f) - g) >= 20 for g, _ in chosen):
                chosen.append((int(f), int(best_idx[f])))
            if len(chosen) == a.frames:
                break
        xs, ys = meta['origins'][str(w)]['x'], meta['origins'][str(w)]['y']
        mp = SC.ROOT / meta['manifest']
        man = json.loads(mp.read_text())
        hr = {int(o['frame_index']): mp.parent / o['hr_path'] for o in man['observations'] if o['camera_id'] == 'cam00'}
        rows = []
        for f, idx in chosen:
            gy, gx = np.unravel_index(idx, score.shape[1:])
            x, y = xs[gx], ys[gy]
            per = {'ours': dict(psnr=float(P0[f, gy, gx]), ssim=float(S0[f, gy, gx]), lpips=float(L0[f, gy, gx]))}
            show = ['ours']
            for c in SC.SR + SC.DYN:
                if c not in comp:
                    continue
                var = c
                if c + '_cull' in names and SC.metric(z, c + '_cull', w)[0][f, gy, gx] > SC.metric(z, c, w)[0][f, gy, gx]:
                    var = c + '_cull'
                pv, sv, lv = SC.metric(z, var, w)
                per[var] = dict(psnr=float(pv[f, gy, gx]), ssim=float(sv[f, gy, gx]), lpips=float(lv[f, gy, gx]))
                show.append(var)
            bid = f'{scene}_f{f:03d}_m'
            tiles = {m: SC.crop(SC.ROOT / meta['methods'][m] / f'cam00/{f:04d}.png', x, y, w) for m in show}
            hr_tile = SC.crop(hr[f], x, y, w)
            order = list(range(len(show)))
            rng.shuffle(order)
            letters = [chr(65 + i) for i in range(len(show))]
            key[bid] = {letters[i]: show[order[i]] for i in range(len(show))}
            n += 1
            mapping[f'm{n:02d}'] = bid
            RS.grid([hr_tile] + [tiles[key[bid][c]] for c in letters], ['Reference (HR)'] + letters, SC.ROOT / a.blind_out / f'm{n:02d}.png')
            RS.grid([hr_tile] + [tiles[m] for m in show],
                    ['HR'] + [f"{RS.LABEL[m.replace('_cull', '')]}{'*' if m.endswith('_cull') else ''} {per[m]['psnr']:.1f}/{per[m]['lpips']:.3f}"
                              for m in show], out / 'sheets_grid' / f'{bid}.png')
            rows.append(dict(frame=f, total_score=float(score[f, gy, gx]),
                             windows=[dict(x=int(x), y=int(y), size=w, score=float(score[f, gy, gx]), dynamic_share=float(dyn[f, gy, gx]), metrics=per)]))
        cands[scene] = rows
        print(scene, [(r['frame'], r['windows'][0]['x'], r['windows'][0]['y'], round(r['total_score'], 2), round(r['windows'][0]['dynamic_share'], 2)) for r in rows], flush=True)
    (out / 'blind').mkdir(parents=True, exist_ok=True)
    (out / 'candidates.json').write_text(json.dumps(cands, indent=1) + '\n')
    (out / 'blind' / 'key.json').write_text(json.dumps(key, indent=1) + '\n')
    (out / 'blind_grid_map.json').write_text(json.dumps(mapping, indent=1) + '\n')


if __name__ == '__main__':
    main()
