"""Decode Codex blind crop rankings with the hidden key (Claude 4dsr-ac, 2026-11-02). CPU only.
  /usr/bin/python3 -I decode_blind.py --sel output/.../selection_w192 --result ATTEMPT/result.json [...] --out <sel>/blind_decoded.json
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--sel', required=True)
    p.add_argument('--result', nargs='+', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    sel = ROOT / a.sel
    mapping = json.loads((sel / 'blind_grid_map.json').read_text())
    key = json.loads((sel / 'blind' / 'key.json').read_text())
    rows = []
    for rf in a.result:
        r = json.loads(Path(rf).read_text())
        for w in r['windows']:
            name = Path(w['file']).stem
            bid = mapping[name]
            lab = key[bid]
            dec = lambda s: lab.get(s.strip(), s)  # noqa: E731
            ranking = [dec(s) for s in w['ranking']]
            rows.append(dict(file=name, id=bid, ours_rank=ranking.index('ours') + 1 if 'ours' in ranking else None,
                             best=dec(w['best']), second=dec(w['second']), margin=w['margin'], ranking=ranking,
                             complete=sorted(ranking) == sorted(lab.values()), best_advantages=w['best_advantages'],
                             best_defects=w['best_defects'], worst=dec(w['worst']), note=w['note']))
    rows.sort(key=lambda x: x['file'])
    (ROOT / a.out).write_text(json.dumps(rows, indent=1, ensure_ascii=False) + '\n')
    for x in rows:
        print(x['file'], x['id'], 'ours_rank', x['ours_rank'], 'best', x['best'], 'second', x['second'], x['margin'],
              '' if x['complete'] else 'INCOMPLETE', '|', x['best_advantages'][:70])


if __name__ == '__main__':
    main()
