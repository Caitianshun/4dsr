"""Crop candidates for the qualitative comparison figure, by the pre-registered rule (Claude 4dsr-ac, 2026-11-02;
tmp/claude_qualfig_fps_20261102/prereg_crops.md). CPU only (NumPy + Pillow).

Inputs: patch_maps.py outputs (<patch>/<Scene>_area4.{npz,json}).
Competitors per window: SR methods = SplatSuRe, IE-SRGS, Sequence Matters, 3DSR (each the better of its uncut and post-hoc-cull
render where both exist, per metric) and SR4D; dynamic reconstruction = Wu plus any re-rendered dynamic baseline present.
Pass (rule 4 + 5): Ours PSNR >= best competitor + 1.0 dB, Ours LPIPS <= best competitor - 0.02, Ours SSIM >= every competitor,
HR gradient >= the scene's 60th percentile. Score = min(dPSNR / 1.0, dLPIPS / 0.02). Per scene, frames are ranked by the summed
score of their two best non-overlapping passing windows (192 px); the top 3 frames are candidates.
Representativeness (rule 8): among detail windows of all frames, the share where Ours beats every SR competitor on both PSNR and
LPIPS, and the share that passes the threshold rule.
Outputs: <out>/candidates.json, <out>/stats.json, contact sheets <out>/sheets/<Scene>_f<frame>_w<k>.png (labelled, for Claude)
and <out>/blind/<id>.png with <out>/blind/key.json (shuffled letters, HR labelled as reference; for the blind check).
  /usr/bin/python3 -I select_crops.py --patch output/.../patch --out output/.../selection [--win 192]
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[3]
SR = ['splatsure', 'iesrgs', 'sequence_matters', 'threedsr', 'sr4d']
DYN = ['wu', 'yang', '4c4d', 'multi4d', 'localdygs', 'cem']
LABEL = dict(ours='Ours', splatsure='SplatSuRe', iesrgs='IE-SRGS', sequence_matters='Sequence Matters', threedsr='3DSR', sr4d='SR4D',
             wu='Wu 4DGS', yang='Yang 4DGS', **{'4c4d': '4C4D'}, multi4d='Multi4D', localdygs='LocalDyGS', cem='CEM-4DGS', hr='HR')
DP, DL, GQ = 1.0, 0.02, 60
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'


def metric(z, m, w):
    mse = z[f'{m}__mse_{w}'].astype(np.float64)
    return -10 * np.log10(np.maximum(mse, 1e-12)), z[f'{m}__ssim_{w}'].astype(np.float32), z[f'{m}__lpips_{w}'].astype(np.float32)


def competitors(z, names, w):
    """Per competitor (P, S, L) arrays; methods with a cull variant take the better value per metric."""
    out = {}
    for c in SR + DYN:
        if c not in names:
            continue
        p, s, l = metric(z, c, w)
        if c + '_cull' in names:
            p2, s2, l2 = metric(z, c + '_cull', w)
            p, s, l = np.maximum(p, p2), np.maximum(s, s2), np.minimum(l, l2)
        out[c] = (p, s, l)
    return out


def best(comp, group):
    ks = [c for c in group if c in comp]
    P = np.max([comp[c][0] for c in ks], axis=0)
    S = np.max([comp[c][1] for c in ks], axis=0)
    L = np.min([comp[c][2] for c in ks], axis=0)
    return P, S, L, ks


def crop(path, x, y, w):
    with Image.open(path) as im:
        return im.convert('RGB').crop((x, y, x + w, y + w))


def sheet(tiles, labels, scale, path, title=None):
    font = ImageFont.truetype(FONT, 16)
    tw = tiles[0].width * scale
    pad, lh = 6, 40
    cols = len(tiles)
    W = cols * (tw + pad) + pad
    H = tw + lh + 2 * pad + (24 if title else 0)
    im = Image.new('RGB', (W, H), 'white')
    d = ImageDraw.Draw(im)
    top = pad + (24 if title else 0)
    if title:
        d.text((pad, 4), title, fill='black', font=font)
    for i, (t, lab) in enumerate(zip(tiles, labels)):
        x = pad + i * (tw + pad)
        im.paste(t.resize((tw, tw), Image.NEAREST), (x, top))
        for j, line in enumerate(lab.split('\n')[:2]):
            d.text((x + 2, top + tw + 2 + 18 * j), line, fill='black', font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--patch', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--win', type=int, default=192)
    p.add_argument('--frames-per-scene', type=int, default=3)
    a = p.parse_args()
    out = ROOT / a.out
    cands, stats, blind_key = {}, {}, {}
    rng = random.Random(20261102)
    for jf in sorted((ROOT / a.patch).glob('*_area4.json')):
        meta = json.loads(jf.read_text())
        scene = meta['scene']
        z = np.load(jf.with_suffix('.npz'))
        names = list(meta['methods'])
        w = a.win
        P0, S0, L0 = metric(z, 'ours', w)
        comp = competitors(z, names, w)
        Ps, Ss, Ls, sr_used = best(comp, SR)
        Pa, Sa, La, all_used = best(comp, SR + DYN)
        grad = z[f'hr__grad_{w}']
        dyn = z[f'hr__dyn_{w}'].astype(np.float32)
        detail = grad >= np.percentile(grad, GQ)
        dP, dL, dS = P0 - Pa, La - L0, S0 - Sa
        ok = (dP >= DP) & (dL >= DL) & (dS >= 0) & detail
        score = np.where(ok, np.minimum(dP / DP, dL / DL), -np.inf)
        # representativeness over detail windows of every frame
        win_sr = (P0 > Ps) & (L0 < Ls)
        st = dict(windows=int(detail.sum()), sr_competitors=sr_used, all_competitors=all_used,
                  share_beats_all_sr_psnr_and_lpips=float(win_sr[detail].mean()),
                  share_beats_all_competitors_psnr_and_lpips=float(((P0 > Pa) & (L0 < La))[detail].mean()),
                  share_passes_threshold=float(ok[detail].mean()),
                  share_beats_each={c: float(((P0 > comp[c][0]) & (L0 < comp[c][2]))[detail].mean()) for c in comp})
        stats[scene] = st
        xs, ys = meta['origins'][str(w)]['x'], meta['origins'][str(w)]['y']
        frame_rank = []
        for f in range(score.shape[0]):
            sf = score[f]
            order = np.argsort(sf, axis=None)[::-1]
            picked = []
            for idx in order[:200]:
                if not np.isfinite(sf.flat[idx]):
                    break
                gy, gx = np.unravel_index(idx, sf.shape)
                x, y = xs[gx], ys[gy]
                if all(abs(x - px) >= w or abs(y - py) >= w for px, py, _, _ in picked):
                    picked.append((x, y, float(sf.flat[idx]), (int(gy), int(gx))))
                if len(picked) == 2:
                    break
            if picked:
                frame_rank.append((sum(s for _, _, s, _ in picked) if len(picked) == 2 else picked[0][2] / 2, f, picked))
        frame_rank.sort(reverse=True)
        chosen, used = [], []
        for tot, f, picked in frame_rank:
            if any(abs(f - g) < 20 for g in used):  # spread the candidate frames over time
                continue
            used.append(f)
            chosen.append((tot, f, picked))
            if len(chosen) == a.frames_per_scene:
                break
        hr_root = ROOT / meta['manifest']
        manifest = json.loads(hr_root.read_text())
        hr = {int(o['frame_index']): hr_root.parent / o['hr_path'] for o in manifest['observations'] if o['camera_id'] == 'cam00'}
        rows = []
        for tot, f, picked in chosen:
            wins = []
            for k, (x, y, s, (gy, gx)) in enumerate(picked):
                per = {'ours': dict(psnr=float(P0[f, gy, gx]), ssim=float(S0[f, gy, gx]), lpips=float(L0[f, gy, gx]))}
                show = ['ours']
                for c in SR + DYN:
                    if c not in comp:
                        continue
                    var = c
                    if c + '_cull' in names:
                        pn = metric(z, c, w)[0][f, gy, gx]
                        pc = metric(z, c + '_cull', w)[0][f, gy, gx]
                        var = c + '_cull' if pc > pn else c
                    pv, sv, lv = metric(z, var, w)
                    per[var] = dict(psnr=float(pv[f, gy, gx]), ssim=float(sv[f, gy, gx]), lpips=float(lv[f, gy, gx]))
                    show.append(var)
                wins.append(dict(x=int(x), y=int(y), size=w, score=s, dynamic_share=float(dyn[f, gy, gx]), metrics=per))
                tiles = [crop(hr[f], x, y, w)] + [crop(ROOT / meta['methods'][m] / f'cam00/{f:04d}.png', x, y, w) for m in show]
                labs = ['HR'] + [f"{LABEL[m.replace('_cull', '')]}{' (cull)' if m.endswith('_cull') else ''}\n"
                                 f"{per[m]['psnr']:.2f} / {per[m]['lpips']:.3f}" for m in show]
                sheet(tiles, labs, 2, out / 'sheets' / f'{scene}_f{f:03d}_w{k}.png', f'{scene} frame {f} window ({x},{y}) {w}px  [PSNR / LPIPS]')
                bid = f'{scene}_f{f:03d}_w{k}'
                order = list(range(len(show)))
                rng.shuffle(order)
                letters = [chr(65 + i) for i in range(len(show))]
                blind_key[bid] = {letters[i]: show[order[i]] for i in range(len(show))}
                btiles = [tiles[0]] + [tiles[1 + order[i]] for i in range(len(show))]
                sheet(btiles, ['Reference (HR)'] + letters, 2, out / 'blind' / f'{bid}.png')
            rows.append(dict(frame=f, total_score=tot, windows=wins))
        cands[scene] = rows
        print(json.dumps(dict(scene=scene, detail_windows=st['windows'], beats_all_sr=round(st['share_beats_all_sr_psnr_and_lpips'], 3),
                              passes=round(st['share_passes_threshold'], 4), frames=[r['frame'] for r in rows])), flush=True)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'candidates.json').write_text(json.dumps(cands, indent=1) + '\n')
    (out / 'stats.json').write_text(json.dumps(dict(rule=dict(dpsnr=DP, dlpips=DL, grad_percentile=GQ, window=a.win), scenes=stats), indent=1) + '\n')
    (out / 'blind' / 'key.json').write_text(json.dumps(blind_key, indent=1) + '\n')


if __name__ == '__main__':
    main()
