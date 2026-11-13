"""Collects the 8-scene attribution rows (base, + field, + residuals, PinSR) into attrib_table.json / .md (Claude, 2026-11-12).
Test view cam00, 300 frames, main-table evaluator. Cook07 rows = Tab. 3 (ablation_rescore.json); PinSR rows = Tab. 1 sources.
Rows of job 2 (attr-p6-02) are read from rows2/. Missing rows are listed, never filled.
  /usr/bin/python3 -I build_attrib_table.py"""
import json
from pathlib import Path

R = Path('/home/cai_tianshun/Project/4dsr')
A = R / 'output/dynamic_sr_modules_20261010/attrib_20261112'
SCENES = ['Coffee07', 'Cook07', 'Cut07', 'Salmon07', 'Flame07', 'Sear07', 'Discussion07', 'VRHeadset07']
NAMES = dict(Coffee07='Coffee Martini', Cook07='Cook Spinach', Cut07='Cut Beef', Salmon07='Flame Salmon', Flame07='Flame Steak',
             Sear07='Sear Steak', Discussion07='Discussion', VRHeadset07='VR Headset')
ROWS = ['base', 'field', 'pc', 'full', 'nores']  # nores = PinSR student without frame residuals (not part of the cumulative chain)
M = ('psnr', 'ssim', 'lpips')
X4 = json.loads((R / 'output/dynamic_sr_modules_20261010/static_sr_v2_20261026/summary/x4_table_20261030.json').read_text())['test']
NS = json.loads((R / 'tmp/claude_cvpr_gap_20261102/newscene_table.json').read_text())
ABL = json.loads((R / 'tmp/claude_paper_draft_20261103/ablation_rescore.json').read_text())['rows']

vals, src, missing = {}, {}, []
for s in SCENES:
    for row in ROWS:
        v = None
        if row == 'full':
            v = NS['Ours'][s]['test'] if s in ('Sear07', 'Salmon07') else X4['Ours'][s]
            src[(s, row)] = 'Tab. 1'
        elif s == 'Cook07':
            v = ABL[row]['test']
            src[(s, row)] = 'Tab. 3'
        else:
            for sub in ('rows2', 'rows'):
                p = A / sub / s / row / 'score_test/summary.json'
                if p.is_file():
                    d = json.loads(p.read_text())
                    assert d['status'] == 'completed' and d['frames'] == 300 and d['cameras'] == ['cam00'], p
                    v = d['test']
                    src[(s, row)] = str(p.relative_to(R))
                    break
        if v is None:
            missing.append(f'{s}/{row}')
        else:
            vals[(s, row)] = {m: float(v[m]) for m in M}

out = dict(scenes=SCENES, rows=ROWS, missing=missing,
           values={f'{s}/{r}': vals[(s, r)] for s in SCENES for r in ROWS if (s, r) in vals},
           sources={f'{s}/{r}': src[(s, r)] for (s, r) in src})
complete = [s for s in SCENES if all((s, r) in vals for r in ROWS)]
out['complete_scenes'] = complete
if complete:
    mean = {r: {m: sum(vals[(s, r)][m] for s in complete) / len(complete) for m in M} for r in ROWS}
    out['mean_over'] = len(complete)
    out['mean'] = mean
    out['steps'] = {k: {m: mean[b][m] - mean[a][m] for m in M} for k, (a, b) in
                    dict(field=('base', 'field'), residuals=('field', 'pc'), consensus=('pc', 'full'), total=('base', 'full')).items()}
    out['per_scene_total_psnr'] = {s: vals[(s, 'full')]['psnr'] - vals[(s, 'base')]['psnr'] for s in complete}
(A / 'attrib_table.json').write_text(json.dumps(out, indent=1))

L = ['| Scene | ' + ' | '.join(f'{r} PSNR/SSIM/LPIPS' for r in ROWS) + ' |', '|' + '---|' * (len(ROWS) + 1)]
for s in SCENES:
    L.append(f'| {NAMES[s]} | ' + ' | '.join(('{psnr:.2f} / {ssim:.3f} / {lpips:.3f}'.format(**vals[(s, r)]) if (s, r) in vals else '—')
                                           for r in ROWS) + ' |')
if complete:
    L.append(f'| Mean ({len(complete)}) | ' + ' | '.join('{psnr:.2f} / {ssim:.3f} / {lpips:.3f}'.format(**out['mean'][r]) for r in ROWS) + ' |')
L.append('')
L.append('missing: ' + (', '.join(missing) or 'none'))
(A / 'attrib_table.md').write_text('\n'.join(L) + '\n')
print('\n'.join(L))
if complete:
    print(json.dumps(out['steps'], indent=1))
