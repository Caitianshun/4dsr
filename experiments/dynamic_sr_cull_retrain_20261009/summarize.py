"""Aggregate official, post-hoc culled and retrained cam00 all-300-frame results into JSON + Markdown tables."""
import argparse
import csv
import json
import statistics
from pathlib import Path

import rt
from tasks import registry, OUT

SCENES = ('Cook', 'Cut', 'Discussion', 'VRHeadset')
N3DV = ('Cook', 'Cut')


def official(e):
    rows = [r for r in csv.DictReader(open(rt.ROOT / e['eval'] / 'metrics_per_observation.csv')) if r.get('scope', 'full_test') == 'full_test']
    if len(rows) != 300:
        raise ValueError(f"official {e['eval']} has {len(rows)} test rows")
    m = lambda k: statistics.fmean(float(r[k]) for r in rows)
    out = dict(psnr=m('psnr'), ssim=m('ssim'), lpips=m('lpips'), LR_diagnostic_psnr=m('LR_diagnostic_psnr'))
    fft = rt.ROOT / e['eval'] / 'FFT_absolute_band_error.csv'
    if fft.exists():
        f = list(csv.DictReader(open(fft)))
        for k in ('low_psnr', 'mid_psnr', 'high_psnr'):
            out[k] = statistics.fmean(float(r[k]) for r in f)
    return out


def mine(path):
    p = rt.ROOT / path / 'summary.json'
    if not p.exists():
        return None
    s = json.loads(p.read_text())
    if s.get('frames') != 300:
        return None
    t = s['test']
    return dict(psnr=t['psnr'], ssim=t['ssim'], lpips=t['lpips'], LR_diagnostic_psnr=t['LR_diagnostic_psnr'],
                low_psnr=t['low_psnr'], mid_psnr=t['mid_psnr'], high_psnr=t['high_psnr'],
                train_psnr=s['train_diagnostic']['psnr'], points=s['points_evaluated'],
                culled=(s.get('cull') or {}).get('culled'), near=s.get('cam00_near_count_loaded'))


def training_cost(path):
    p = rt.ROOT / path / 'complete.json'
    if not p.exists():
        return None
    c = json.loads(p.read_text())
    return dict(seconds=c['seconds'], points=c['points'], culled=sum(x['culled'] for x in c.get('culls', [])),
                culled_opaque=sum(x['culled_opacity_gt_0p1'] for x in c.get('culls', [])),
                pruned=c['counts'].get('pruned_points'), peak_GB=c['peak_GB'])


def fmt(v, k):
    if v is None:
        return '—'
    return f'{v:.4f}' if k in ('ssim', 'lpips') else f'{v:.3f}'


def failures(values):
    """Endpoints whose cam00 PSNR is > 3 dB below their scene median (design-doc failure definition)."""
    by_scene = {}
    for key, v in values.items():
        by_scene.setdefault(key.split('_')[0][:-2], []).append((key, v))
    out = []
    for scene, items in by_scene.items():
        med = statistics.median(v for _, v in items)
        out += [k for k, v in items if v < med - 3]
    return sorted(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', default=f'{OUT}/summary')
    a = p.parse_args()
    reg = registry()
    W = f'{OUT}/wave1'
    table = {}
    for key, e in reg.items():
        seed_key, arm = key.rsplit('_', 1)
        table[key] = dict(official=official(e), posthoc=mine(f'{W}/posthoc/{key}_cull'))
    arms = {}
    for seed_key in sorted({k.rsplit('_', 1)[0] for k in reg}):
        for arm in ('cull', 'cullaff', 'cull_lam0p1', 'cull_lam0p2', 'cull_lam0p5', 'cull_dssim0p2'):
            run = f'{seed_key}_{arm}'
            lr = training_cost(f'{W}/lr/{seed_key}_{arm.split("_")[0]}')
            arms[run] = dict(eval=mine(f'{W}/eval/{run}'), sr=training_cost(f'{W}/sr/{run}'), lr=lr)
    result = dict(official_and_posthoc=table, retrained=arms)
    out = rt.ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    rt.write_json(out / 'summary.json', result)

    lines = ['# cam00 all-300-frame results', '']
    lines += ['## Official endpoints and post-hoc culling', '',
              '| Endpoint | PSNR official | PSNR post-hoc cull | SSIM official | SSIM cull | LPIPS official | LPIPS cull | LR closure official | LR closure cull | culled |',
              '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for key in sorted(table):
        o, c = table[key]['official'], table[key]['posthoc'] or {}
        lines.append(f"| {key} | {fmt(o['psnr'],'psnr')} | {fmt(c.get('psnr'),'psnr')} | {fmt(o['ssim'],'ssim')} | {fmt(c.get('ssim'),'ssim')} | "
                     f"{fmt(o['lpips'],'lpips')} | {fmt(c.get('lpips'),'lpips')} | {fmt(o['LR_diagnostic_psnr'],'p')} | {fmt(c.get('LR_diagnostic_psnr'),'p')} | {c.get('culled', '—')} |")
    lines += ['', '## Retrained arms (Bsync SR stage)', '',
              '| Seed | Arm | PSNR | SSIM | LPIPS | LR closure | high-band PSNR | cam01 PSNR | LR culled (opaque) | SR culled | final points | LR s | SR s |',
              '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for run in sorted(arms):
        v = arms[run]
        if v['eval'] is None and v['sr'] is None:
            continue
        e, s, l = v['eval'] or {}, v['sr'] or {}, v['lr'] or {}
        seed_key, arm = run.split('_', 1)
        lines.append(f"| {seed_key} | {arm} | {fmt(e.get('psnr'),'psnr')} | {fmt(e.get('ssim'),'ssim')} | {fmt(e.get('lpips'),'lpips')} | "
                     f"{fmt(e.get('LR_diagnostic_psnr'),'p')} | {fmt(e.get('high_psnr'),'p')} | {fmt(e.get('train_psnr'),'p')} | "
                     f"{l.get('culled','—')} ({l.get('culled_opaque','—')}) | {s.get('culled','—')} | {e.get('points', s.get('points','—'))} | "
                     f"{round(l['seconds']) if l else '—'} | {round(s['seconds']) if s else '—'} |")
    # group means over seeds that have every compared variant
    lines += ['', '## Means over complete seeds', '']
    variants = dict(official_Bsync=lambda sk: table[f'{sk}_Bsync']['official'], posthoc_Bsync=lambda sk: table[f'{sk}_Bsync']['posthoc'],
                    official_B0=lambda sk: table[f'{sk}_B0']['official'], posthoc_B0=lambda sk: table[f'{sk}_B0']['posthoc'])
    for arm in ('cull', 'cullaff', 'cull_lam0p1', 'cull_lam0p2', 'cull_lam0p5', 'cull_dssim0p2'):
        variants[arm] = (lambda arm: lambda sk: arms[f'{sk}_{arm}']['eval'])(arm)
    seeds = sorted({k.rsplit('_', 1)[0] for k in reg})
    lines += ['| Variant | group | seeds | PSNR | SSIM | LPIPS | LR closure | failures (>3 dB below scene median) |', '| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |']
    means = {}
    for name, get in variants.items():
        for group, members in (('N3DV', [s for s in seeds if s.startswith(N3DV)]), ('MeetRoom', [s for s in seeds if not s.startswith(N3DV)]), ('all', seeds)):
            vals = {s: get(s) for s in members}
            if any(v is None for v in vals.values()):
                continue
            m = {k: statistics.fmean(v[k] for v in vals.values()) for k in ('psnr', 'ssim', 'lpips', 'LR_diagnostic_psnr')}
            fails = failures({f'{s}_x': v['psnr'] for s, v in vals.items()})
            means[f'{name}/{group}'] = dict(m, seeds=len(vals), failures=fails)
            lines.append(f"| {name} | {group} | {len(vals)} | {m['psnr']:.3f} | {m['ssim']:.4f} | {m['lpips']:.4f} | {m['LR_diagnostic_psnr']:.3f} | {', '.join(f.rsplit('_',1)[0] for f in fails) or 'none'} |")
    rt.write_json(out / 'means.json', means)
    (out / 'tables.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
