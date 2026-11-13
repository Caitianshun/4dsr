"""Summary of the SplatSuRe-on-our-base-configuration control (Claude, 2026-11-12). Test view cam00, 300 frames, main-table
evaluator. Rows per scene: SplatSuRe as in Tab. 1, SplatSuRe with our culling (x4 table), SplatSuRe on our base configuration
(this job), our base model (attribution rows / Tab. 3), the same-card base model of this job, PinSR (Tab. 1).
Missing rows are listed, never filled.  /usr/bin/python3 -I build_sps_table.py"""
import json
from pathlib import Path

R = Path('/home/cai_tianshun/Project/4dsr')
O = R / 'output/dynamic_sr_modules_20261010/sps_base_20261112'
A = R / 'output/dynamic_sr_modules_20261010/attrib_20261112'
X4 = json.loads((R / 'output/dynamic_sr_modules_20261010/static_sr_v2_20261026/summary/x4_table_20261030.json').read_text())['test']
ABL = json.loads((R / 'tmp/claude_paper_draft_20261103/ablation_rescore.json').read_text())['rows']
M = ('psnr', 'ssim', 'lpips')
SCENES = ['Flame07', 'Cook07']
ROWS = ['splatsure_tab1', 'splatsure_cull', 'splatsure_ourbase', 'base', 'base_p6', 'pinsr']


def score(p):
    if not p.is_file():
        return None
    d = json.loads(p.read_text())
    assert d['status'] == 'completed' and d['frames'] == 300 and d['cameras'] == ['cam00'], p
    return {m: float(d['test'][m]) for m in M}


vals, missing, runs = {}, [], {}
for s in SCENES:
    got = dict(splatsure_tab1=X4['SplatSuRe'][s], splatsure_cull=X4['SplatSuRe + 同款事后剔除'][s], pinsr=X4['Ours'][s],
               splatsure_ourbase=score(O / f'rows/{s}/sps/score_test/summary.json'),
               base=ABL['base']['test'] if s == 'Cook07' else score(A / f'rows/{s}/base/score_test/summary.json'),
               base_p6=score(O / f'rows/{s}/basep6/score_test/summary.json'))
    for r in ROWS:
        if got[r] is None:
            missing.append(f'{s}/{r}')
        else:
            vals[f'{s}/{r}'] = {m: float(got[r][m]) for m in M}
    for arm in ('sps', 'basep6'):
        c = O / f'sr/{s}_area4_{arm}/complete.json'
        if c.is_file():
            d = json.loads(c.read_text())
            runs[f'{s}/{arm}'] = dict(seconds=round(d['seconds']), points=d['points'], peak_GB=round(d['peak_GB'], 2), gpu=d['gpu'])
    w = O / f'weights/{s}/receipt.json'
    if w.is_file():
        d = json.loads(w.read_text())
        runs[f'{s}/weights'] = dict(status=d['status'], seconds=round(d.get('seconds', 0)), maps=len(d['records']),
                                    mean_weight=sum(r['mean'] for r in d['records']) / max(1, len(d['records'])))
gaps = {}
for s in SCENES:
    v = lambda r: vals.get(f'{s}/{r}')
    for a, b in (('splatsure_ourbase', 'base'), ('splatsure_ourbase', 'base_p6'), ('splatsure_ourbase', 'splatsure_tab1'),
                 ('splatsure_ourbase', 'splatsure_cull'), ('pinsr', 'splatsure_ourbase')):
        if v(a) and v(b):
            gaps[f'{s}: {a} - {b}'] = {m: v(a)[m] - v(b)[m] for m in M}
out = dict(scenes=SCENES, rows=ROWS, missing=missing, values=vals, gaps=gaps, runs=runs)
(O / 'sps_table.json').write_text(json.dumps(out, indent=1))
L = ['| Row | ' + ' | '.join(f'{s} PSNR / SSIM / LPIPS' for s in SCENES) + ' |', '|---|' + '---|' * len(SCENES)]
for r in ROWS:
    L.append(f'| {r} | ' + ' | '.join('{psnr:.3f} / {ssim:.4f} / {lpips:.4f}'.format(**vals[f'{s}/{r}']) if f'{s}/{r}' in vals else '—'
                                    for s in SCENES) + ' |')
L += ['', '| Gap | dPSNR | dSSIM | dLPIPS |', '|---|---|---|---|']
L += [f'| {k} | {g["psnr"]:+.3f} | {g["ssim"]:+.4f} | {g["lpips"]:+.4f} |' for k, g in gaps.items()]
(O / 'sps_table.md').write_text('\n'.join(L) + '\n')
print('\n'.join(L))
print(json.dumps(dict(missing=missing, runs=runs), indent=1))
