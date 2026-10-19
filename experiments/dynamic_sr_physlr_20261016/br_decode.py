"""Decode a blind-review result with the separately stored key (Claude 4dsr-7c, 2026-10-18).

  br_decode.py --result RESULT.json --manifest IMAGES_MANIFEST.json --key KEY.json --out DECODED.json

The reviewer saw the frozen copies 000.png ... in attachment order; the executor's images/manifest.json maps each frozen copy to its
source composite (w01.png ...) with the sha256, which must equal the key's sha256 of that composite. Each window record of the
result is matched to its composite by the file name the reviewer wrote, else by position, and the check is reported. Output: per
window the decoded candidates, plus totals per candidate kind (legibility counts, acceptable counts, closest / worst counts) split by
window kind (novel view / teacher) and content (face / text).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--result", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--key", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    res = json.loads(Path(a.result).read_text())
    man = json.loads(Path(a.manifest).read_text())
    key = json.loads(Path(a.key).read_text())
    by_file = {w["file"]: w for w in key["windows"]}
    order = [Path(m["source_path"]).name for m in man]
    for m in man:
        k = by_file[Path(m["source_path"]).name]
        if m["sha256"] != k["sha256"]:
            raise SystemExit(f"frozen image differs from the keyed composite: {m['source_path']}")
    rows, checks = [], []
    for pos, w in enumerate(res["windows"]):
        m = re.search(r"w(\d\d)", w["file"])
        name = f"w{m.group(1)}.png" if m else order[pos]
        checks.append(dict(position=pos, reviewer_file=w["file"], used=name, matches_position=name == order[pos]))
        k = by_file[name]
        lab = {x["label"]: x for x in k["labels"]}
        cands = []
        for c in w["candidates"]:
            L = c["label"].strip().upper()[:1]
            if L not in lab:
                raise SystemExit(f"{name}: unknown label {c['label']}")
            cands.append(dict(label=L, candidate=lab[L]["candidate"], legibility=c["legibility"], acceptable=c["acceptable"],
                              text_read=c["text_read"], defects=c["defects"]))
        decode = lambda s: lab[s.strip().upper()[:1]]["candidate"] if s.strip().upper()[:1] in lab else s  # noqa: E731
        content = "text" if k["window"].startswith("text") else "face"
        rows.append(dict(file=name, scene=k["scene"], camera=k["camera"], frame=k["frame"], window=k["window"], kind=k["kind"],
                         content=content, reviewer_content=w["content"], hr_legibility=w["hr_legibility"],
                         hr_text_read=w["hr_text_read"], closest=decode(w["closest"]), worst=decode(w["worst"]),
                         note=w["note"], candidates=cands))
    tot = defaultdict(lambda: dict(legibility=Counter(), acceptable=0, n=0, closest=0, worst=0))
    for r in rows:
        for c in r["candidates"]:
            t = tot[(r["kind"], r["content"], c["candidate"])]
            t["legibility"][c["legibility"]] += 1
            t["acceptable"] += bool(c["acceptable"])
            t["n"] += 1
            t["closest"] += r["closest"] == c["candidate"]
            t["worst"] += r["worst"] == c["candidate"]
    totals = [dict(kind=k[0], content=k[1], candidate=k[2], n=v["n"], legibility=dict(v["legibility"]), acceptable=v["acceptable"],
                   closest=v["closest"], worst=v["worst"]) for k, v in sorted(tot.items())]
    hr = Counter((r["content"], r["hr_legibility"]) for r in rows)
    out = dict(result=dict(path=a.result, sha256=hashlib.sha256(Path(a.result).read_bytes()).hexdigest()),
               key=dict(path=a.key, sha256=hashlib.sha256(Path(a.key).read_bytes()).hexdigest()),
               order_checks=checks, hr_legibility={f"{k[0]}:{k[1]}": v for k, v in hr.items()}, totals=totals, windows=rows,
               reviewer_summary=res.get("summary"))
    Path(a.out).write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    print("order matches position:", all(c["matches_position"] for c in checks))
    print("HR legibility:", dict(hr))
    for t in totals:
        print(t["kind"], t["content"], t["candidate"], "n", t["n"], t["legibility"], "acc", t["acceptable"], "closest", t["closest"],
              "worst", t["worst"])


if __name__ == "__main__":
    main()
