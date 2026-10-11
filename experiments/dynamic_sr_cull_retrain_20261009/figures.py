"""Report figures for the culling retrain (static PNG; colours fixed per entity, validated all-pairs)."""
import csv
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

import rt
from tasks import registry, OUT

FONT = Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
if FONT.exists():
    font_manager.fontManager.addfont(str(FONT))
    plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(FONT)).get_name()
plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': '#8a8984',
                     'axes.labelcolor': '#52514e', 'xtick.color': '#52514e', 'ytick.color': '#52514e',
                     'axes.grid': True, 'grid.color': '#e6e5e0', 'grid.linewidth': 0.8, 'figure.dpi': 150,
                     'savefig.facecolor': '#fcfcfb', 'axes.facecolor': '#fcfcfb', 'font.size': 9})
INK = '#0b0b0b'
COLOR = dict(official='#2a78d6', posthoc='#eb6834', cull='#1baf7a', cullaff='#4a3aa7')
LABEL = dict(official='官方端点（作者拓扑）', posthoc='官方端点 + 事后剔除', cull='训练中剔除', cullaff='训练中剔除 + 每相机颜色仿射')
MARK = dict(official='o', posthoc='D', cull='s', cullaff='^')
W = rt.ROOT / OUT / 'wave1'
FIG = rt.ROOT / 'docs/assets/dynamic_sr_cull_retrain_2026-10-09'


def summary(path):
    p = path / 'summary.json'
    if not p.exists():
        return None
    s = json.loads(p.read_text())
    return s if s.get('frames') == 300 else None


def official_psnr(e):
    rows = list(csv.DictReader(open(rt.ROOT / e['eval'] / 'metrics_per_observation.csv')))
    rows = [r for r in rows if r.get('scope', 'full_test') == 'full_test']
    return statistics.fmean(float(r['psnr']) for r in rows), statistics.fmean(float(r['lpips']) for r in rows)


def fig_posthoc(reg):
    keys = sorted(reg, key=lambda k: (k.split('_')[0][:-2], k.split('_')[0][-2:], k.split('_')[1]))
    rows = []
    for k in keys:
        s = summary(W / 'posthoc' / f'{k}_cull')
        if s:
            rows.append((k, official_psnr(reg[k])[0], s['test']['psnr']))
    if not rows:
        return None
    fig, ax = plt.subplots(figsize=(6.4, 0.32 * len(rows) + 1.2))
    for i, (k, a, b) in enumerate(rows):
        ax.plot([a, b], [i, i], color='#c3c2b7', linewidth=2, zorder=1)
        ax.scatter([a], [i], s=36, color=COLOR['official'], marker=MARK['official'], zorder=2, edgecolor='#fcfcfb', linewidth=1)
        ax.scatter([b], [i], s=36, color=COLOR['posthoc'], marker=MARK['posthoc'], zorder=3, edgecolor='#fcfcfb', linewidth=1)
        if b - a > 1:
            ax.text(b + 0.25, i, f'+{b - a:.1f} dB', va='center', fontsize=8, color='#52514e')
    ax.set_yticks(range(len(rows)), [r[0].replace('_', ' ') for r in rows])
    ax.invert_yaxis()
    ax.set_xlabel('cam00 PSNR（dB，300 帧均值）')
    ax.grid(axis='y', visible=False)
    ax.legend(handles=[plt.Line2D([], [], marker=MARK[n], color=COLOR[n], linestyle='', label=LABEL[n]) for n in ('official', 'posthoc')],
              loc='lower left', bbox_to_anchor=(0, 1.0), ncol=2, frameon=False)
    fig.tight_layout()
    path = FIG / 'fig1_posthoc_cull_300frames.png'
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_cull_trace(seeds):
    fig, axes = plt.subplots(2, 4, figsize=(9.6, 4.4), sharex=True, sharey=True)
    any_data = False
    for ax, seed in zip(axes.flat, seeds):
        for arm in ('cull', 'cullaff'):
            p = W / 'lr' / f'{seed}_{arm}' / 'complete.json'
            if not p.exists():
                continue
            c = json.loads(p.read_text())
            xs = [x['iteration'] for x in c['culls']]
            ys = [x['culled'] for x in c['culls']]
            ax.plot(xs, ys, color=COLOR[arm], marker=MARK[arm], markersize=4, linewidth=2, label=LABEL[arm])
            any_data = True
        ax.set_title(seed, fontsize=9, color=INK)
        ax.set_yscale('symlog', linthresh=1)
        ax.set_ylim(-0.2, 3000)
    if not any_data:
        plt.close(fig)
        return None
    for ax in axes[1]:
        ax.set_xlabel('LR 段 fine 步数')
    for ax in axes[:, 0]:
        ax.set_ylabel('本次剔除的高斯数')
    handles = [plt.Line2D([], [], color=COLOR[a], marker=MARK[a], linewidth=2, label=LABEL[a]) for a in ('cull', 'cullaff')]
    fig.legend(handles=handles, loc='upper center', ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = FIG / 'fig2_in_training_cull_counts.png'
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_arms(reg, seeds):
    rows = []
    for seed in seeds:
        values = dict(official=official_psnr(reg[f'{seed}_Bsync'])[0])
        s = summary(W / 'posthoc' / f'{seed}_Bsync_cull')
        values['posthoc'] = s['test']['psnr'] if s else None
        for arm in ('cull', 'cullaff'):
            s = summary(W / 'eval' / f'{seed}_{arm}')
            values[arm] = s['test']['psnr'] if s else None
        rows.append((seed, values))
    if not any(v['cull'] is not None for _, v in rows):
        return None
    fig, ax = plt.subplots(figsize=(6.4, 0.42 * len(rows) + 1.3))
    lo = 25.0
    for i, (seed, v) in enumerate(rows):
        for j, arm in enumerate(('official', 'posthoc', 'cull', 'cullaff')):
            if v[arm] is None:
                continue
            y = i + (j - 1.5) * 0.12
            if v[arm] < lo:
                ax.scatter([lo + 0.05], [y], s=34, color=COLOR[arm], marker='<', edgecolor='#fcfcfb', linewidth=1, zorder=3)
                ax.text(lo + 0.25, y, f'{v[arm]:.1f} dB（轴外）', va='center', fontsize=7.5, color='#52514e')
            else:
                ax.scatter([v[arm]], [y], s=34, color=COLOR[arm], marker=MARK[arm], edgecolor='#fcfcfb', linewidth=1, zorder=3)
    ax.set_xlim(lo, None)
    ax.set_yticks(range(len(rows)), [s for s, _ in rows])
    ax.invert_yaxis()
    ax.grid(axis='y', visible=False)
    ax.set_xlabel('cam00 PSNR（dB，300 帧均值，Bsync）')
    ax.legend(handles=[plt.Line2D([], [], marker=MARK[n], color=COLOR[n], linestyle='', label=LABEL[n]) for n in ('official', 'posthoc', 'cull', 'cullaff')],
              loc='lower left', bbox_to_anchor=(0, 1.0), ncol=2, frameon=False)
    fig.tight_layout()
    path = FIG / 'fig3_arms_per_seed.png'
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_lambda(seeds):
    lams = [('cull', 0.05), ('cull_lam0p1', 0.1), ('cull_lam0p2', 0.2), ('cull_lam0p5', 0.5)]
    data = {}
    for seed in seeds:
        vals = []
        for arm, lam in lams:
            s = summary(W / 'eval' / f'{seed}_{arm}')
            vals.append(None if s is None else (s['test']['psnr'], s['test']['ssim'], s['test']['lpips']))
        if all(v is not None for v in vals):
            data[seed] = vals
    if not data:
        return None
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.0))
    x = [lam for _, lam in lams]
    for k, (ax, name, scale) in enumerate(zip(axes, ('ΔPSNR（dB）', 'ΔSSIM（$\\times10^{-3}$）', 'ΔLPIPS（$\\times10^{-3}$）'), (1, 1e3, 1e3))):
        curves = []
        for seed, vals in data.items():
            y = [(v[k] - vals[0][k]) * scale for v in vals]
            curves.append(y)
            ax.plot(x, y, color='#c3c2b7', linewidth=1, zorder=1)
        mean = np.mean(curves, axis=0)
        ax.plot(x, mean, color=COLOR['cull'], linewidth=2, marker=MARK['cull'], markersize=5, zorder=2, label='开发种子均值')
        ax.axhline(0, color='#8a8984', linewidth=0.8)
        ax.set_xscale('log')
        ax.set_xticks(x, [str(v) for v in x])
        ax.minorticks_off()
        ax.set_xlabel('λ_SR')
        ax.set_title(name + '，相对 λ_SR=0.05', fontsize=9, color=INK)
    axes[0].legend(handles=[plt.Line2D([], [], color=COLOR['cull'], marker=MARK['cull'], linewidth=2, label=f'{len(data)} 个种子均值'),
                            plt.Line2D([], [], color='#c3c2b7', linewidth=1, label='单个种子')], frameon=False, loc='best')
    fig.tight_layout()
    path = FIG / 'fig4_lambda_sr_sweep.png'
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_crops(reg, seed='Cook08', frame=150):
    from PIL import Image
    e = reg[f'{seed}_Bsync']
    m = json.loads((rt.ROOT / json.loads((rt.ROOT / e['sr'] / 'config.json').read_text())['data']['manifest']['path']).read_text())
    obs = next(o for o in m['observations'] if o['camera_id'] == 'cam00' and int(o['frame_index']) == frame)
    root = rt.ROOT / json.loads((rt.ROOT / e['sr'] / 'config.json').read_text())['data']['manifest']['path']
    hr = np.asarray(Image.open(root.parent / obs['hr_path']).convert('RGB')).astype(np.float32) / 255
    panels = [('HR（真值）', hr)]
    off = rt.ROOT / e['eval'] / 'floats' / f'cam00_{frame:04d}.npz'
    if off.exists():
        panels.append((LABEL['official'], np.load(off)['rgb_raw'].transpose(1, 2, 0)))
    for name, path in (('posthoc', W / 'posthoc' / f'{seed}_Bsync_cull'), ('cull', W / 'eval' / f'{seed}_cull'), ('cullaff', W / 'eval' / f'{seed}_cullaff')):
        p = path / 'previews' / f'cam00_{frame:04d}.npz'
        if p.exists():
            panels.append((LABEL[name], np.load(p)['rgb_raw'].transpose(1, 2, 0)))
    if len(panels) < 3:
        return None
    fig, axes = plt.subplots(1, len(panels), figsize=(2.6 * len(panels), 2.2))
    for ax, (name, img) in zip(axes, panels):
        ax.imshow(np.clip(img, 0, 1))
        ax.set_title(name, fontsize=8, color=INK)
        ax.axis('off')
    fig.tight_layout()
    path = FIG / f'fig5_{seed}_cam00_{frame:04d}.png'
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_repeats(seeds):
    fig, ax = plt.subplots(figsize=(6.4, 0.42 * len(seeds) + 1.3))
    any_data = False
    for i, seed in enumerate(seeds):
        for j, arm in enumerate(('cull', 'cullaff')):
            a, b = summary(W / 'eval' / f'{seed}_{arm}'), summary(W / 'eval_repeat' / f'{seed}_{arm}_r2')
            if not (a and b):
                continue
            any_data = True
            y = i + (j - 0.5) * 0.25
            ax.plot([a['test']['psnr'], b['test']['psnr']], [y, y], color='#c3c2b7', linewidth=2, zorder=1)
            ax.scatter([a['test']['psnr'], b['test']['psnr']], [y, y], s=30, color=COLOR[arm], marker=MARK[arm], edgecolor='#fcfcfb', linewidth=1, zorder=2)
    if not any_data:
        plt.close(fig)
        return None
    ax.set_yticks(range(len(seeds)), seeds)
    ax.invert_yaxis()
    ax.grid(axis='y', visible=False)
    ax.set_xlabel('cam00 PSNR（dB，300 帧均值）：同一父模型 SR 段两次运行，连线为两次之差')
    ax.legend(handles=[plt.Line2D([], [], marker=MARK[n], color=COLOR[n], linestyle='', label=LABEL[n]) for n in ('cull', 'cullaff')],
              loc='lower left', bbox_to_anchor=(0, 1.0), ncol=2, frameon=False)
    fig.tight_layout()
    path = FIG / 'fig7_repeat_runs.png'
    fig.savefig(path)
    plt.close(fig)
    return path


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    reg = registry()
    seeds = sorted({k.rsplit('_', 1)[0] for k in reg}, key=lambda s: (('Cook', 'Cut', 'Discussion', 'VRHeadset').index(s[:-2]), s[-2:]))
    for f in (lambda: fig_posthoc(reg), lambda: fig_cull_trace(seeds), lambda: fig_arms(reg, seeds), lambda: fig_lambda(seeds), lambda: fig_crops(reg), lambda: fig_repeats(seeds)):
        print(f())


if __name__ == '__main__':
    main()
