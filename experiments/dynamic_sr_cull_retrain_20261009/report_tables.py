"""Markdown tables (Chinese headers) for the culling-retrain report, generated from result files only."""
import csv
import json
import statistics as st
from pathlib import Path

import rt
from tasks import registry, OUT

W = rt.ROOT / OUT / 'wave1'
SEEDS = ['Cook07', 'Cook08', 'Cut07', 'Cut08', 'Discussion07', 'Discussion08', 'VRHeadset07', 'VRHeadset08']


def official(e):
    rows = [r for r in csv.DictReader(open(rt.ROOT / e['eval'] / 'metrics_per_observation.csv')) if r.get('scope', 'full_test') == 'full_test']
    out = {k: st.fmean(float(r[k]) for r in rows) for k in ('psnr', 'ssim', 'lpips', 'LR_diagnostic_psnr')}
    f = list(csv.DictReader(open(rt.ROOT / e['eval'] / 'FFT_absolute_band_error.csv')))
    out.update({k: st.fmean(float(r[k]) for r in f) for k in ('low_psnr', 'mid_psnr', 'high_psnr')})
    return out


def mine(path):
    p = path / 'summary.json'
    if not p.exists():
        return None
    s = json.loads(p.read_text())
    return dict(s['test'], train=s['train_diagnostic'], points=s['points_evaluated'], cull=s.get('cull'))


def lr(seed, arm):
    p = W / 'lr' / f'{seed}_{arm}' / 'complete.json'
    return json.loads(p.read_text()) if p.exists() else None


def table(header, rows):
    out = ['| ' + ' | '.join(header) + ' |', '| ' + ' | '.join(['---'] + ['---:'] * (len(header) - 1)) + ' |']
    out += ['| ' + ' | '.join(str(c) for c in r) + ' |' for r in rows]
    return '\n'.join(out)


def f3(x):
    return '—' if x is None else f'{x:.3f}'


def f4(x):
    return '—' if x is None else f'{x:.4f}'


def main():
    reg = registry()
    parts = []
    # 1 post-hoc
    rows = []
    for k in sorted(reg, key=lambda k: (SEEDS.index(k.rsplit('_', 1)[0]), k)):
        o, c = official(reg[k]), mine(W / 'posthoc' / f'{k}_cull')
        rows.append([k.replace('_', ' '), f"{o['psnr']:.3f} → {c['psnr']:.3f}", f"{o['ssim']:.4f} → {c['ssim']:.4f}",
                     f"{o['lpips']:.4f} → {c['lpips']:.4f}", f"{o['LR_diagnostic_psnr']:.2f} → {c['LR_diagnostic_psnr']:.2f}", f"{c['cull']['culled']:,}"])
    parts.append(('posthoc', table(['端点', 'PSNR', 'SSIM', 'LPIPS', '测试 LR 闭环', '剔除数'], rows)))
    # 2 group means for official/post-hoc
    rows = []
    for arm in ('B0', 'Bsync'):
        for name, getter in (('官方', lambda k: official(reg[k])), ('事后剔除', lambda k: mine(W / 'posthoc' / f'{k}_cull'))):
            for grp, members in (('N3DV', SEEDS[:4]), ('MeetRoom', SEEDS[4:]), ('全部', SEEDS)):
                vals = [getter(f'{s}_{arm}') for s in members]
                rows.append([f'{arm} {name}', grp, f3(st.fmean(v['psnr'] for v in vals)), f4(st.fmean(v['ssim'] for v in vals)), f4(st.fmean(v['lpips'] for v in vals))])
    parts.append(('posthoc_means', table(['变体', '组', 'PSNR', 'SSIM', 'LPIPS'], rows)))
    # 3 LR culls
    rows = []
    for s in SEEDS:
        for arm in ('cull', 'cullaff'):
            c = lr(s, arm)
            if not c:
                continue
            ev = c['culls']
            rows.append([s, '剔除' if arm == 'cull' else '剔除 + 仿射', f"{ev[0]['culled']:,}", f"{sum(x['culled'] for x in ev[1:]):,}",
                         f"{sum(x['culled_opacity_gt_0p1'] for x in ev):,}", f"{c['counts']['pruned_points']:,}", f"{c['points']:,}"])
    parts.append(('lr_culls', table(['种子', '臂', '第 1000 步剔除', '之后各次剔除合计', '其中不透明度 >0.1（全部）', '低不透明度剪枝', 'LR 段结束点数'], rows)))
    # 4 wave 1 per seed
    rows = []
    for s in SEEDS:
        o = official(reg[f'{s}_Bsync'])
        p = mine(W / 'posthoc' / f'{s}_Bsync_cull')
        a = mine(W / 'eval' / f'{s}_cull')
        b = mine(W / 'eval' / f'{s}_cullaff')
        fmt = lambda v: '—' if v is None else f"{v['psnr']:.3f} / {v['ssim']:.4f} / {v['lpips']:.4f}"
        rows.append([s, fmt(o), fmt(p), fmt(a), fmt(b)])
    parts.append(('wave1', table(['种子', '官方 Bsync', '官方 + 事后剔除', '训练中剔除', '训练中剔除 + 仿射'], rows)))
    rows = []
    for s in SEEDS:
        p, a, b = mine(W / 'posthoc' / f'{s}_Bsync_cull'), mine(W / 'eval' / f'{s}_cull'), mine(W / 'eval' / f'{s}_cullaff')
        if None in (p, a, b):
            continue
        rows.append([s] + [f"{v['low_psnr']:.2f} / {v['mid_psnr']:.2f} / {v['high_psnr']:.2f}" for v in (p, a, b)] +
                    [f"{v['LR_diagnostic_psnr']:.2f}" for v in (p, a, b)])
    parts.append(('wave1_bands', table(['种子', '事后剔除 低/中/高频', '训练中剔除 低/中/高频', '剔除 + 仿射 低/中/高频', 'LR 闭环 事后', 'LR 闭环 剔除', 'LR 闭环 剔除+仿射'], rows)))
    # 5 sweep
    rows = []
    for arm, label in (('cull_lam0p1', 'λ_SR = 0.1'), ('cull_lam0p2', 'λ_SR = 0.2'), ('cull_lam0p5', 'λ_SR = 0.5'), ('cull_dssim0p2', 'D-SSIM 0.2（λ_SR = 0.05）')):
        for grp, members in (('N3DV', SEEDS[:4]), ('MeetRoom', SEEDS[4:]), ('全部', SEEDS)):
            d = [(mine(W / 'eval' / f'{s}_{arm}'), mine(W / 'eval' / f'{s}_cull')) for s in members]
            d = [(x, y) for x, y in d if x and y]
            if not d:
                continue
            rows.append([label, grp, len(d), f"{st.fmean(x['psnr'] - y['psnr'] for x, y in d):+.3f}", f"{st.fmean(x['ssim'] - y['ssim'] for x, y in d):+.4f}",
                         f"{st.fmean(x['lpips'] - y['lpips'] for x, y in d):+.4f}",
                         f"{sum(x['psnr'] > y['psnr'] for x, y in d)}/{sum(x['ssim'] > y['ssim'] for x, y in d)}/{sum(x['lpips'] < y['lpips'] for x, y in d)}"])
    parts.append(('sweep', table(['变体（相对 λ_SR = 0.05）', '组', '种子数', 'ΔPSNR', 'ΔSSIM', 'ΔLPIPS', '改善种子数 PSNR/SSIM/LPIPS'], rows)))
    # 6 repeats
    rows, d = [], []
    for s in SEEDS:
        a, b = mine(W / 'eval' / f'{s}_cull'), mine(W / 'eval_repeat' / f'{s}_cull_r2')
        if a and b:
            d.append((b['psnr'] - a['psnr'], b['ssim'] - a['ssim'], b['lpips'] - a['lpips']))
            rows.append([s, f"{a['psnr']:.3f} / {b['psnr']:.3f}", f'{d[-1][0]:+.3f}', f'{d[-1][1]:+.4f}', f'{d[-1][2]:+.4f}', f"{b['low_psnr'] - a['low_psnr']:+.2f}", f"{b['high_psnr'] - a['high_psnr']:+.2f}"])
    if len(d) > 1:
        rows.append(['配对差标准差', '', f'{st.stdev(x[0] for x in d):.3f}', f'{st.stdev(x[1] for x in d):.4f}', f'{st.stdev(x[2] for x in d):.4f}', '', ''])
        rows.append(['配对差绝对值中位数', '', f'{st.median(abs(x[0]) for x in d):.3f}', f'{st.median(abs(x[1]) for x in d):.4f}', f'{st.median(abs(x[2]) for x in d):.4f}', '', ''])
    parts.append(('repeats', table(['种子', 'PSNR 第 1 次 / 第 2 次', 'ΔPSNR', 'ΔSSIM', 'ΔLPIPS', 'Δ低频 PSNR', 'Δ高频 PSNR'], rows)))
    # 7 full comparison (means by group; two-run means where a repeat exists)
    def run(path, repeat=None):
        a = mine(W / path)
        if a is None:
            return None
        b = mine(W / repeat) if repeat else None
        if b is None:
            return a
        return {k: (a[k] + b[k]) / 2 for k in ('psnr', 'ssim', 'lpips', 'LR_diagnostic_psnr', 'low_psnr', 'high_psnr')}
    variants = [
        ('官方 Bsync（作者配置，混合硬件）', lambda s: official(reg[f'{s}_Bsync'])),
        ('官方 + 事后剔除', lambda s: mine(W / 'posthoc' / f'{s}_Bsync_cull')),
        ('同硬件不剔除', lambda s: run(f'eval/{s}_nocull')),
        ('同硬件不剔除 + 事后剔除（基线）', lambda s: run(f'eval/{s}_nocull_posthoc_cull')),
        ('去门槛 + 事后剔除', lambda s: run(f'eval/{s}_gate_posthoc_cull')),
        ('训练中剔除（两次均值）', lambda s: run(f'eval/{s}_cull', f'eval_repeat/{s}_cull_r2')),
        ('去门槛 + 仿射（不剔除）', lambda s: run(f'eval/{s}_gateaff')),
        ('去门槛 + 仿射 + 事后剔除', lambda s: run(f'eval/{s}_gateaff_posthoc_cull')),
        ('训练中剔除 + 仿射（两次均值）', lambda s: run(f'eval/{s}_cullaff', f'eval_repeat/{s}_cullaff_r2')),
        ('不剔除 + D-SSIM + 事后剔除', lambda s: run(f'eval/{s}_nocull_dssim0p2_posthoc_cull')),
        ('训练中剔除 + D-SSIM', lambda s: run(f'eval/{s}_cull_dssim0p2')),
        ('去门槛 + 仿射 + D-SSIM + 事后剔除', lambda s: run(f'eval/{s}_gateaff_dssim0p2_posthoc_cull')),
        ('训练中剔除 + 仿射 + D-SSIM', lambda s: run(f'eval/{s}_cullaff_dssim0p2')),
        ('训练中剔除 + 仿射 + λ_SR 0.2 + D-SSIM', lambda s: run(f'eval/{s}_cullaff_lam0p2_dssim0p2')),
    ]
    base = {s: run(f'eval/{s}_nocull_posthoc_cull') for s in SEEDS}
    rows = []
    for name, get in variants:
        vals = {s: get(s) for s in SEEDS}
        cells = [name]
        for members in (SEEDS[:4], SEEDS[4:]):
            v = [vals[s] for s in members]
            if any(x is None for x in v):
                cells += ['—', '—', '—']
                continue
            cells += [f3(st.fmean(x['psnr'] for x in v)), f4(st.fmean(x['ssim'] for x in v)), f4(st.fmean(x['lpips'] for x in v))]
        ok = [s for s in SEEDS if vals[s] and base[s]]
        cells.append(f"{sum(vals[s]['psnr'] > base[s]['psnr'] for s in ok)}/{sum(vals[s]['lpips'] < base[s]['lpips'] for s in ok)}（{len(ok)}）" if ok else '—')
        rows.append(cells)
    parts.append(('full', table(['变体', 'N3DV PSNR', 'N3DV SSIM', 'N3DV LPIPS', 'MeetRoom PSNR', 'MeetRoom SSIM', 'MeetRoom LPIPS',
                                 '优于基线的种子数 PSNR/LPIPS（可比种子）'], rows)))
    # 8 cullaff repeats
    rows, d = [], []
    for s in SEEDS:
        a, b = mine(W / 'eval' / f'{s}_cullaff'), mine(W / 'eval_repeat' / f'{s}_cullaff_r2')
        if a and b:
            d.append((b['psnr'] - a['psnr'], b['ssim'] - a['ssim'], b['lpips'] - a['lpips']))
            rows.append([s, f"{a['psnr']:.3f} / {b['psnr']:.3f}", f'{d[-1][0]:+.3f}', f'{d[-1][1]:+.4f}', f'{d[-1][2]:+.4f}'])
    if len(d) > 1:
        rows.append(['配对差标准差', '', f'{st.stdev(x[0] for x in d):.3f}', f'{st.stdev(x[1] for x in d):.4f}', f'{st.stdev(x[2] for x in d):.4f}'])
    parts.append(('repeats_cullaff', table(['种子', 'PSNR 第 1 次 / 第 2 次', 'ΔPSNR', 'ΔSSIM', 'ΔLPIPS'], rows)))
    # 9 SH-degree truncation (cam00 PSNR deg0..3, artifact area deg3)
    rows = []
    for s in SEEDS:
        for tag, label in (('official_posthoc', '官方 + 事后剔除'), ('cull', '训练中剔除'), ('cull_r2', '训练中剔除（重复）'), ('cullaff', '剔除 + 仿射')):
            p = W / 'shcheck' / f'{s}_{tag}' / 'summary.json'
            if not p.exists():
                continue
            dg = json.loads(p.read_text())['degrees']
            rows.append([s, label] + [f"{dg[k]['cam00']['psnr']:.2f}" for k in ('0', '1', '2', '3')] +
                        [f"{dg['3']['cam00']['artifact_area'] * 100:.2f}% → {dg['0']['cam00']['artifact_area'] * 100:.2f}%"])
    parts.append(('shcheck', table(['种子', '模型', 'SH 0 阶', 'SH 1 阶', 'SH 2 阶', 'SH 3 阶（训练值）', '伪影面积 3 阶 → 0 阶'], rows)))
    # 10 affine-corrected cam01 diagnostics
    rows = []
    for s in SEEDS:
        for arm in ('cullaff', 'gateaff'):
            p = W / 'traindiag_affine' / f'{s}_{arm}' / 'summary.json'
            c = mine(W / 'eval' / f'{s}_cull')
            if not p.exists() or c is None:
                continue
            dd = json.loads(p.read_text())
            rows.append([s, '剔除 + 仿射' if arm == 'cullaff' else '去门槛 + 仿射', f"{dd['raw']['psnr']:.2f}", f"{dd['affine']['psnr']:.2f}",
                         f"{dd['raw']['LR_psnr']:.2f} → {dd['affine']['LR_psnr']:.2f}", f"{c['train']['psnr']:.2f}"])
    parts.append(('traindiag', table(['种子', '臂', 'cam01 PSNR 未校色', 'cam01 PSNR 加仿射', 'cam01 LR 闭环 未校色 → 加仿射', '只剔除臂 cam01 PSNR'], rows)))
    # 11 wave-5 hardware check and per-seed D-SSIM comparison
    rows = []
    for s_ in SEEDS:
        a = mine(W / 'eval' / f'{s_}_gateaff_dssim0p2')
        b = mine(W / 'eval' / f'{s_}_gateaff_dssim0p2_posthoc_cull')
        c = mine(W / 'eval' / f'{s_}_cullaff_dssim0p2')
        n = mine(W / 'eval' / f'{s_}_nocull_dssim0p2_posthoc_cull')
        h = mine(W / 'eval_hwcheck' / f'{s_}_gateaff_dssim0p2_pro6000_posthoc_cull')
        fmt = lambda v: '—' if v is None else f"{v['psnr']:.3f} / {v['ssim']:.4f} / {v['lpips']:.4f}"
        rows.append([s_, fmt(n), fmt(a), fmt(b), fmt(h), fmt(c)])
    parts.append(('dssim_arms', table(['种子', '不剔除 + D-SSIM + 事后剔除', '仿射 + D-SSIM（不剔除）', '仿射 + D-SSIM + 事后剔除（3090）',
                                      '同上，PRO 6000 复跑', '训练中剔除 + 仿射 + D-SSIM'], rows)))
    out = rt.ROOT / OUT / 'summary' / 'report_tables.md'
    out.write_text('\n\n'.join(f'<!-- {name} -->\n{t}' for name, t in parts) + '\n')
    print(out.read_text())


if __name__ == '__main__':
    main()
