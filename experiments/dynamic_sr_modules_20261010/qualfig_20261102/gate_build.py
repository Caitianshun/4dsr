"""AGENTS.md 0.3 acceptability gate, redone for the final method (Claude 4dsr-ac, 2026-11-02).

Same windows and layout as the 10-18 review (experiments/dynamic_sr_physlr_20261016/br_build.py, cmd_build): the registered cam00
windows of positions_v1.json, positions_v1_disc.json and positions_v1_holdout.json (fixed from clean HR frames on 10-18, before any
crop existed), 128x128 HR px at 3x nearest-neighbour zoom, HR on top, shuffled candidates A-E below, no names, order
numpy default_rng(seed_base + i).permutation(n). The training-camera (teacher) windows are not repeated: the matched teacher
of the final method is the 10-18 teacher (swinir_area_ft_v1), already reviewed.
Candidates per cam00 window: LR bicubic x4 (anchor), matched teacher on the cam00 LR (single-image SR reference, 10-18 PNGs),
the final method (main-table 8-bit render), SplatSuRe and IE-SRGS (main-table models re-rendered on the 3090; the labelled
post-hoc-cull render where it exists, i.e. each baseline's stronger version).
  gate_build.py --review-dir DIR --key KEY.json   (run with mml_bw on the CPU: CUDA_VISIBLE_DEVICES='')
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BR = ROOT / 'experiments/dynamic_sr_physlr_20261016'
sys.path.insert(0, str(BR))
import br_build as B  # noqa: E402  (windows, obs_of, rgb, bicubic_up, sha, PREP, SHORT, FONT, Z)

POS = [ROOT / 'output/dynamic_sr_physlr_20261016/blind_review' / n for n in ('positions_v1.json', 'positions_v1_disc.json', 'positions_v1_holdout.json')]
SISR = ROOT / 'output/dynamic_sr_physlr_20261016/blind_review/sisr'
OURS = ROOT / 'output/dynamic_lr_area4_20261024/ours'
QF = ROOT / 'output/dynamic_sr_modules_20261010/qualfig_20261102/renders'


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--review-dir', required=True)
    p.add_argument('--key', required=True)
    p.add_argument('--seed-base', type=int, default=20261102)
    a = p.parse_args()
    rev = ROOT / a.review_dir
    rev.mkdir(parents=True, exist_ok=False)
    font = ImageFont.truetype(B.FONT, 26)
    key = dict(positions=[dict(path=str(q.relative_to(ROOT)), sha256=B.sha(q)) for q in POS], seed_base=a.seed_base,
               code=dict(path=str(Path(__file__).relative_to(ROOT)), sha256=B.sha(__file__)), windows=[])
    obs_cache, n_out = {}, 0
    for i, scene, cam, f, name, (x0, y0, x1, y1) in B.windows([str(q) for q in POS]):
        if cam != 'cam00':
            continue
        short = B.SHORT[scene]
        obs = obs_cache.setdefault(scene, B.obs_of(scene))
        o = obs[(cam, f)]
        root = B.PREP / f'{scene}_area4'
        hp, lp = root / o['hr_path'], root / o['lr_path']
        if B.sha(hp) != o['hr_sha256'] or B.sha(lp) != o['lr_sha256']:
            raise SystemExit(f'HR/LR changed: {scene} {cam} {f}')
        hr = B.rgb(hp)
        cands = [('bicubic', B.bicubic_up(B.rgb(lp)), str(lp.relative_to(ROOT))),
                 ('sisr_matched_on_cam00_LR', B.rgb(SISR / f'{scene}_cam00_{f:04d}.png'), str((SISR / f'{scene}_cam00_{f:04d}.png').relative_to(ROOT))),
                 ('ours_final', B.rgb(OURS / f'{short}_area4_ours/renders_test/cam00/{f:04d}.png'),
                  str((OURS / f'{short}_area4_ours/renders_test/cam00/{f:04d}.png').relative_to(ROOT)))]
        for m in ('splatsure', 'iesrgs'):
            d = QF / f'{short}_area4' / (m + '_cull')
            d = d if (d / 'render_index.json').exists() else QF / f'{short}_area4' / m
            cands.append((d.name, B.rgb(d / f'cam00/{f:04d}.png'), str((d / f'cam00/{f:04d}.png').relative_to(ROOT))))
        for _, img, src in cands:
            if img.shape != hr.shape:
                raise SystemExit(f'size mismatch {src}: {img.shape} vs {hr.shape}')
        n_out += 1
        perm = np.random.default_rng(a.seed_base + n_out).permutation(len(cands))
        cw, gap, lab = 128 * B.Z, 16, 40
        n = len(cands)
        W = n * cw + (n + 1) * gap
        H = lab + cw + gap + lab + cw + gap
        sheet = Image.new('RGB', (W, H), (255, 255, 255))
        d = ImageDraw.Draw(sheet)
        crop = lambda img: Image.fromarray(img[y0:y1, x0:x1]).resize((cw, cw), Image.NEAREST)  # noqa: E731
        hx = (W - cw) // 2
        d.text((hx, 4), 'HR', fill=(0, 0, 0), font=font)
        sheet.paste(crop(hr), (hx, lab))
        rows = []
        for j, k in enumerate(perm):
            letter = chr(ord('A') + j)
            x = gap + j * (cw + gap)
            d.text((x, lab + cw + gap + 4), letter, fill=(0, 0, 0), font=font)
            sheet.paste(crop(cands[k][1]), (x, lab + cw + gap + lab))
            rows.append(dict(label=letter, candidate=cands[k][0], source=cands[k][2]))
        out = rev / f'w{n_out:02d}.png'
        sheet.save(out)
        key['windows'].append(dict(file=out.name, sha256=B.sha(out), scene=scene, camera=cam, frame=f, window=name,
                                   box=[x0, y0, x1, y1], kind='novel_view', labels=rows))
        print(out.name, scene, f, name, flush=True)
    kp = ROOT / a.key
    if kp.exists():
        raise SystemExit(f'key exists: {kp}')
    kp.write_text(json.dumps(key, indent=1, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
