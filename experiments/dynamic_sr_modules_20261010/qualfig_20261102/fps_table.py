"""Render-speed table from the fps_bench.py / fps_sr4d.py records (Claude 4dsr-ac, 2026-11-02). CPU only.
FPS = 1000 / mean per-frame latency (synchronised, cam00, 300 frames x 2 passes, after 30 warm-up frames); the back-to-back
throughput FPS is listed in the JSON. Means over N3DV (Cook07, Cut07, Flame07, Coffee07) and MeetRoom (Discussion07, VRHeadset07).
  /usr/bin/python3 -I fps_table.py --fps output/.../fps/rtx3090 --out output/.../fps/rtx3090_table
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCENES = ['Cook07', 'Cut07', 'Flame07', 'Coffee07', 'Discussion07', 'VRHeadset07']
ORDER = [('wu', 'Wu 4DGS（主表基线）'), ('splatsure', 'SplatSuRe'), ('iesrgs', 'IE-SRGS'), ('sequence_matters', 'Sequence Matters'),
         ('threedsr', '3DSR'), ('sr4d', 'SR4D（只渲染 HR）'), ('sr4d_author_call', 'SR4D（作者渲染调用，含 LR 光栅化）'),
         ('ours_noresid', 'Ours 去掉逐帧残差'), ('ours', 'Ours（完整最终模型）')]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--fps', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    d = ROOT / a.fps
    recs = {}
    for f in d.glob('*_area4/*.json'):
        r = json.loads(f.read_text())
        if r.get('status') == 'ok':
            recs[(r['scene'], r['method'])] = r
    gpu = next(iter(recs.values()))['environment']['gpu'] if recs else '?'
    table, lines = {}, []
    lines += [f'# 渲染速度（{gpu}，cam00 测试视角，N3DV 1344×1008，MeetRoom 1280×720）', '',
              '每格：FPS（每帧平均耗时 ms，点数 k）。计时含逐帧形变查询、残差（仅 Ours）与光栅化，不含加载、写图与评分。', '',
              '| 方法 | ' + ' | '.join(SCENES) + ' | N3DV 均值 FPS | MeetRoom 均值 FPS |', '|' + '---|' * (len(SCENES) + 3)]
    for m, label in ORDER:
        cells, n3, mr = [], [], []
        for s in SCENES:
            r = recs.get((s, m))
            if r is None:
                cells.append('—')
                continue
            fps = r['fps_from_mean_latency']
            cells.append(f"{fps:.0f}（{r['latency_ms']['mean']:.1f} ms，{r['info']['points'] / 1000:.0f}k）")
            (n3 if s in SCENES[:4] else mr).append(fps)
            table.setdefault(m, {})[s] = dict(fps=fps, fps_throughput=r['fps_throughput'], ms_mean=r['latency_ms']['mean'],
                                              ms_median=r['latency_ms']['median'], points=r['info']['points'],
                                              stage_ms={k: v['mean'] for k, v in r.get('stage_ms', {}).items()},
                                              render_extra_memory_mib=r.get('render_extra_memory_mib'), resolution=r.get('resolution_hr'))
        mean = lambda v: f'{sum(v) / len(v):.0f}' + ('' if len(v) in (0, 2, 4) else f' ({len(v)})') if v else '—'
        lines.append(f'| {label} | ' + ' | '.join(cells) + f' | {mean(n3)} | {mean(mr)} |')
        if m in table:
            table[m]['N3DV_mean_fps'] = sum(n3) / len(n3) if n3 else None
            table[m]['MeetRoom_mean_fps'] = sum(mr) / len(mr) if mr else None
    lines += ['', 'Ours 分阶段平均耗时（ms，形变 / 残差 / 光栅化）：']
    for s in SCENES:
        r = recs.get((s, 'ours'))
        if r:
            st = r['stage_ms']
            lines.append(f"- {s}：{st['deform']['mean']:.2f} / {st['extra']['mean']:.2f} / {st['raster']['mean']:.2f}；渲染额外显存 {r['render_extra_memory_mib']:.0f} MiB")
    out = ROOT / a.out
    Path(str(out) + '.md').write_text('\n'.join(lines) + '\n')
    Path(str(out) + '.json').write_text(json.dumps(dict(gpu=gpu, table=table), indent=1) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
