"""Eight-scene block for the last two rows of Tab. 3 (Claude, 2026-11-12): 'Consensus from one source' (k1) and 'DropGaussian for
consensus' (drop), test view cam00, 300 frames, main-table evaluator. Cook07 = Tab. 3 (ablation_rescore.json); other scenes =
t3fill_20261112/rows/<S>/<row>/score_test (CTS scores are downloaded to the same paths). Missing cells are listed, never filled.
  /usr/bin/python3 -I build_t3fill.py"""
import json
from pathlib import Path

R = Path('/home/cai_tianshun/Project/4dsr')
OF = R / 'output/dynamic_sr_modules_20261010/t3fill_20261112'
ABL = json.loads((R / 'tmp/claude_paper_draft_20261103/ablation_rescore.json').read_text())['rows']
AT = json.loads((R / 'output/dynamic_sr_modules_20261010/attrib_20261112/attrib_table.json').read_text())
SCENES = ['Coffee07', 'Cook07', 'Cut07', 'Salmon07', 'Flame07', 'Sear07', 'Discussion07', 'VRHeadset07']
M = ('psnr', 'ssim', 'lpips')
vals, missing, src = {}, [], {}
for s in SCENES:
    for row in ('k1', 'drop'):
        if s == 'Cook07':
            vals[f'{s}/{row}'] = {m: float(ABL[row]['test'][m]) for m in M}
            src[f'{s}/{row}'] = 'Tab. 3'
            continue
        p = OF / f'rows/{s}/{row}/score_test/summary.json'
        if p.is_file():
            d = json.loads(p.read_text())
            assert d['status'] == 'completed' and d['frames'] == 300 and d['cameras'] == ['cam00'], p
            vals[f'{s}/{row}'] = {m: float(d['test'][m]) for m in M}
            src[f'{s}/{row}'] = str(p.relative_to(R))
        else:
            missing.append(f'{s}/{row}')
full = {s: AT['values'][f'{s}/full'] for s in SCENES}
full['Cook07'] = dict(full['Cook07'], lpips=0.0997)
out = dict(values=vals, sources=src, missing=missing)
for row in ('k1', 'drop'):
    have = [s for s in SCENES if f'{s}/{row}' in vals]
    if len(have) == len(SCENES):
        out[f'mean_{row}'] = {m: sum(vals[f'{s}/{row}'][m] for s in SCENES) / len(SCENES) for m in M}
    out[f'pinsr_minus_{row}'] = {s: {m: full[s][m] - vals[f'{s}/{row}'][m] for m in M} for s in have}
out['mean_pinsr_tab3'] = {m: sum(full[s][m] for s in SCENES) / len(SCENES) for m in M}
(OF / 't3fill_table.json').write_text(json.dumps(out, indent=1))
L = ['| Scene | k1 PSNR/SSIM/LPIPS | drop PSNR/SSIM/LPIPS | PinSR − k1 | PinSR − drop |', '|---|---|---|---|---|']
for s in SCENES:
    f = lambda r: '{psnr:.3f}/{ssim:.4f}/{lpips:.4f}'.format(**vals[f'{s}/{r}']) if f'{s}/{r}' in vals else '—'
    g = lambda r: f"{out[f'pinsr_minus_{r}'][s]['psnr']:+.3f}" if s in out[f'pinsr_minus_{r}'] else '—'
    L.append(f'| {s} | {f("k1")} | {f("drop")} | {g("k1")} | {g("drop")} |')
for r in ('k1', 'drop'):
    if f'mean_{r}' in out:
        L.append(f'mean {r}: ' + '{psnr:.3f} / {ssim:.4f} / {lpips:.4f}'.format(**out[f'mean_{r}']))
L.append('mean PinSR (Tab. 3 block): ' + '{psnr:.3f} / {ssim:.4f} / {lpips:.4f}'.format(**out['mean_pinsr_tab3']))
(OF / 't3fill_table.md').write_text('\n'.join(L) + '\n')
print('\n'.join(L))
print('missing:', missing)
